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

"""Tests for JAX bindings of batched cell list neighbor construction methods."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list

from .conftest import requires_gpu

pytestmark = requires_gpu


class TestBatchCellList:
    """Test batch_cell_list function."""

    def test_two_systems_with_pbc(self):
        """Test batch_cell_list with two systems."""
        positions1 = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        positions2 = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )

        positions = jnp.vstack([positions1, positions2])

        cells = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ]
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])

        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        cutoff = 2.0

        neighbor_matrix, num_neighbors, shifts = batch_cell_list(
            positions,
            cutoff,
            cells,
            pbcs,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
        )

        assert neighbor_matrix.shape[0] == 4
        assert num_neighbors.shape == (4,)
        assert shifts.shape[0] == 4
        assert shifts.shape[2] == 3

    def test_topology_only_grad_pbc_is_zero(self):
        """Topology-only batch cell-list outputs do not differentiate Warp FFI."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        cells = jnp.array(
            [
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            ],
            dtype=jnp.float32,
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)

        def loss(pos):
            neighbor_matrix, num_neighbors, shifts = batch_cell_list(
                pos,
                2.0,
                cell=cells,
                pbc=pbcs,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=8,
                strategy="auto",
            )
            return (
                neighbor_matrix.astype(pos.dtype).sum()
                + num_neighbors.astype(pos.dtype).sum()
                + shifts.astype(pos.dtype).sum()
            )

        grad = jax.grad(loss)(positions)
        assert jnp.isfinite(grad).all().item()
        np.testing.assert_allclose(np.asarray(grad), 0.0)


class TestBatchCellListEdgeCases:
    """Edge case tests for batch_cell_list."""

    def test_two_systems_different_sizes(self):
        """Batch cell list with systems of different sizes."""
        pos1 = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        pos2 = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        positions = jnp.vstack([pos1, pos2])
        cells = jnp.array(
            [
                [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
                [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            ],
            dtype=jnp.float32,
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])
        batch_idx = jnp.array([0, 0, 0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 4, 6], dtype=jnp.int32)
        cutoff = 1.5

        nm, nn, shifts = batch_cell_list(
            positions,
            cutoff,
            cells,
            pbcs,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
        )
        assert nm.shape[0] == 6
        assert nn.shape == (6,)
        assert shifts.shape[0] == 6

    def test_batch_no_pbc_zero_shifts(self):
        """Batch cell list with no PBC should have all zero shifts."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        cells = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ],
            dtype=jnp.float32,
        )
        pbcs = jnp.array([[False, False, False], [False, False, False]])
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)

        nm, nn, shifts = batch_cell_list(
            positions,
            cutoff=1.0,
            cell=cells,
            pbc=pbcs,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
        )
        if int(jnp.sum(nn)) > 0:
            assert jnp.all(shifts == 0)


class TestBatchCellListJIT:
    """Smoke tests for batch_cell_list compatibility with jax.jit."""

    def test_jit_with_pbc_requires_precomputed_sizing(self):
        """The traced sizing path should fail before allocating JAX buffers."""
        positions = jnp.vstack(
            [
                jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32),
                jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32),
            ]
        )
        cells = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ]
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)

        @jax.jit
        def jitted_batch_cell_list(positions, cells, pbcs, batch_idx, batch_ptr):
            return batch_cell_list(
                positions,
                cutoff=2.0,
                cell=cells,
                pbc=pbcs,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
            )

        with pytest.raises(
            jax.errors.TracerBoolConversionError,
            match="Attempted boolean conversion",
        ):
            jitted_batch_cell_list(positions, cells, pbcs, batch_idx, batch_ptr)

    def test_jit_with_pbc_precomputed_sizing(self):
        """Batched PBC cell list should work under JIT with concrete sizing."""
        positions = jnp.vstack(
            [
                jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32),
                jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32),
            ]
        )
        cells = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ]
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)

        @jax.jit
        def jitted_batch_cell_list(positions, cells, pbcs, batch_idx, batch_ptr):
            return batch_cell_list(
                positions,
                cutoff=2.0,
                cell=cells,
                pbc=pbcs,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_total_cells=16,
            )

        nm, nn, shifts = jitted_batch_cell_list(
            positions, cells, pbcs, batch_idx, batch_ptr
        )

        assert nm.shape[0] == 4
        assert nn.shape == (4,)
        assert shifts.shape[0] == 4
        assert shifts.shape[2] == 3

    def test_jit_auto_falls_back_when_pair_centric_sizing_is_traced(self):
        """``strategy='auto'`` must not expose pair-centric host reads to JIT."""
        atoms_per_system = 200
        total_atoms = atoms_per_system * 2
        max_neighbors = 128
        box_size = 15.0
        positions = jnp.vstack(
            [
                jax.random.uniform(
                    jax.random.PRNGKey(1),
                    (atoms_per_system, 3),
                    dtype=jnp.float32,
                )
                * box_size,
                jax.random.uniform(
                    jax.random.PRNGKey(2),
                    (atoms_per_system, 3),
                    dtype=jnp.float32,
                )
                * box_size,
            ]
        )
        cells = jnp.stack(
            [
                jnp.eye(3, dtype=jnp.float32) * box_size,
                jnp.eye(3, dtype=jnp.float32) * box_size,
            ]
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])
        batch_idx = jnp.concatenate(
            [
                jnp.zeros(atoms_per_system, dtype=jnp.int32),
                jnp.ones(atoms_per_system, dtype=jnp.int32),
            ]
        )
        batch_ptr = jnp.array([0, atoms_per_system, total_atoms], dtype=jnp.int32)

        @jax.jit
        def jitted_batch_cell_list(positions, cells, pbcs, batch_idx, batch_ptr):
            return batch_cell_list(
                positions,
                cutoff=6.0,
                cell=cells * 1.5,
                pbc=pbcs,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=max_neighbors,
                max_total_cells=32,
            )

        nm, nn, shifts = jitted_batch_cell_list(
            positions, cells, pbcs, batch_idx, batch_ptr
        )

        assert nm.shape == (total_atoms, max_neighbors)
        assert nm.dtype == jnp.int32
        assert nn.shape == (total_atoms,)
        assert shifts.shape == (total_atoms, max_neighbors, 3)

    def test_jit_explicit_pair_centric_still_requires_concrete_sizing(self):
        """Explicit pair-centric keeps the concrete launch-sizing contract."""
        atoms_per_system = 200
        total_atoms = atoms_per_system * 2
        box_size = 15.0
        positions = jnp.vstack(
            [
                jax.random.uniform(
                    jax.random.PRNGKey(3),
                    (atoms_per_system, 3),
                    dtype=jnp.float32,
                )
                * box_size,
                jax.random.uniform(
                    jax.random.PRNGKey(4),
                    (atoms_per_system, 3),
                    dtype=jnp.float32,
                )
                * box_size,
            ]
        )
        cells = jnp.stack(
            [
                jnp.eye(3, dtype=jnp.float32) * box_size,
                jnp.eye(3, dtype=jnp.float32) * box_size,
            ]
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])
        batch_idx = jnp.concatenate(
            [
                jnp.zeros(atoms_per_system, dtype=jnp.int32),
                jnp.ones(atoms_per_system, dtype=jnp.int32),
            ]
        )
        batch_ptr = jnp.array([0, atoms_per_system, total_atoms], dtype=jnp.int32)

        @jax.jit
        def jitted_batch_cell_list(positions, cells, pbcs, batch_idx, batch_ptr):
            return batch_cell_list(
                positions,
                cutoff=6.0,
                cell=cells * 1.5,
                pbc=pbcs,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=128,
                max_total_cells=32,
                strategy="pair_centric",
            )

        with pytest.raises(ValueError, match="needs a concrete neighbor_search_radius"):
            jitted_batch_cell_list(positions, cells, pbcs, batch_idx, batch_ptr)


class TestBatchCellListReturnNeighborList:
    """Regression tests for batch_cell_list with return_neighbor_list=True.

    These tests ensure that when return_neighbor_list=True, the shifts are
    returned in list format (num_pairs, 3) rather than matrix format
    (total_atoms, max_neighbors, 3).
    """

    def test_return_neighbor_list_shapes(self):
        """Test that return_neighbor_list=True returns correct shapes.

        This is the core regression test ensuring shifts are in list format
        (num_pairs, 3) rather than matrix format (total_atoms, max_neighbors, 3).
        """
        positions = jnp.vstack(
            [
                jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32),
                jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32),
            ]
        )
        cells = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ]
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        cutoff = 2.0

        neighbor_list, neighbor_ptr, shifts = batch_cell_list(
            positions,
            cutoff,
            cells,
            pbcs,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
        )

        # neighbor_list is COO format: (2, num_pairs)
        assert neighbor_list.shape[0] == 2
        # neighbor_ptr has shape (total_atoms + 1,) = (4 + 1,)
        assert neighbor_ptr.shape == (5,)
        # KEY REGRESSION CHECK: shifts must be 2D (num_pairs, 3), not 3D
        assert shifts.ndim == 2, f"shifts should be 2D, got {shifts.ndim}D"
        assert shifts.shape[1] == 3
        # num_pairs consistency
        assert shifts.shape[0] == neighbor_list.shape[1]

    def test_return_neighbor_list_shifts_dtype(self):
        """Test that shifts have int32 dtype when return_neighbor_list=True."""
        positions = jnp.vstack(
            [
                jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32),
                jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32),
            ]
        )
        cells = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ]
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        cutoff = 2.0

        _, _, shifts = batch_cell_list(
            positions,
            cutoff,
            cells,
            pbcs,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
        )

        assert shifts.dtype == jnp.int32

    def test_return_neighbor_list_consistency_with_matrix_mode(self):
        """Test that list mode and matrix mode produce consistent results.

        Verifies that the set of (i, j, shift_x, shift_y, shift_z) tuples
        are identical between list mode and matrix mode.
        """
        positions = jnp.vstack(
            [
                jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32),
                jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32),
            ]
        )
        cells = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ]
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        cutoff = 2.0

        # Matrix mode
        neighbor_matrix, num_neighbors, shifts_matrix = batch_cell_list(
            positions,
            cutoff,
            cells,
            pbcs,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            return_neighbor_list=False,
        )

        # List mode
        neighbor_list, neighbor_ptr, shifts_list = batch_cell_list(
            positions,
            cutoff,
            cells,
            pbcs,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
        )

        # Verify pair counts match
        num_pairs = int(jnp.sum(num_neighbors))
        assert neighbor_list.shape[1] == num_pairs
        assert shifts_list.shape[0] == num_pairs

        # Extract tuples from matrix mode
        fill_value = positions.shape[0]
        matrix_tuples = []
        neighbor_matrix_np = np.asarray(neighbor_matrix)
        shifts_matrix_np = np.asarray(shifts_matrix)
        for i in range(neighbor_matrix_np.shape[0]):
            for k in range(neighbor_matrix_np.shape[1]):
                j = neighbor_matrix_np[i, k]
                if j != fill_value:
                    shift = shifts_matrix_np[i, k, :]
                    matrix_tuples.append((i, j, shift[0], shift[1], shift[2]))

        # Extract tuples from list mode
        neighbor_list_np = np.asarray(neighbor_list)
        shifts_list_np = np.asarray(shifts_list)
        list_tuples = []
        for p in range(neighbor_list_np.shape[1]):
            i = neighbor_list_np[0, p]
            j = neighbor_list_np[1, p]
            shift = shifts_list_np[p, :]
            list_tuples.append((i, j, shift[0], shift[1], shift[2]))

        # Sort and compare
        matrix_tuples_sorted = sorted(matrix_tuples)
        list_tuples_sorted = sorted(list_tuples)
        np.testing.assert_array_equal(
            matrix_tuples_sorted,
            list_tuples_sorted,
            err_msg="List mode and matrix mode produce different neighbor pairs",
        )

    def test_return_neighbor_list_no_pbc_shifts_zero(self):
        """Test that shifts are zero when PBC is disabled.

        With no periodic boundary conditions, all shifts should be zero
        since atoms cannot interact across periodic images.
        """
        positions = jnp.vstack(
            [
                jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32),
                jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32),
            ]
        )
        cells = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ]
        )
        pbcs = jnp.array([[False, False, False], [False, False, False]])
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        cutoff = 2.0

        neighbor_list, _, shifts = batch_cell_list(
            positions,
            cutoff,
            cells,
            pbcs,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
        )

        if neighbor_list.shape[1] > 0:
            assert jnp.all(shifts == 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
class TestBatchCellListSelectiveRebuildFlags:
    """Test selective rebuild (rebuild_flags) for JAX batch cell list."""

    def test_no_rebuild_preserves_data(self, dtype):
        """All flags False: neighbor data should remain unchanged for all systems."""
        from nvalchemiops.jax.neighbors.batch_cell_list import (
            batch_build_cell_list,
            batch_query_cell_list,
        )

        positions = jnp.vstack(
            [
                jnp.array(
                    [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
                    dtype=dtype,
                ),
                jnp.array(
                    [[10.0, 0.0, 0.0], [10.5, 0.0, 0.0], [10.0, 0.5, 0.0]],
                    dtype=dtype,
                ),
            ]
        )
        cells = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ],
            dtype=dtype,
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])
        batch_idx = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 3, 6], dtype=jnp.int32)
        cutoff = 1.0
        max_neighbors = 10

        # Build cell list
        cell_cache = batch_build_cell_list(
            positions,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cutoff=cutoff,
            cell=cells,
            pbc=pbcs,
        )
        (
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_search_radius,
            _cell_origin,
        ) = cell_cache

        # Initial query
        nm, nn, nm_shifts = batch_query_cell_list(
            positions,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cutoff=cutoff,
            cell=cells,
            pbc=pbcs,
            cells_per_dimension=cells_per_dimension,
            atom_periodic_shifts=atom_periodic_shifts,
            atom_to_cell_mapping=atom_to_cell_mapping,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            neighbor_search_radius=neighbor_search_radius,
            max_neighbors=max_neighbors,
        )

        saved_nn = jnp.array(nn)

        # Selective rebuild with all flags=False
        rebuild_flags = jnp.zeros(2, dtype=jnp.bool_)
        nm2, nn2, _ = batch_query_cell_list(
            positions,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cutoff=cutoff,
            cell=cells,
            pbc=pbcs,
            cells_per_dimension=cells_per_dimension,
            atom_periodic_shifts=atom_periodic_shifts,
            atom_to_cell_mapping=atom_to_cell_mapping,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            neighbor_search_radius=neighbor_search_radius,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm,
            num_neighbors=nn,
            rebuild_flags=rebuild_flags,
        )

        assert jnp.all(nn2 == saved_nn), (
            "num_neighbors must be unchanged when all rebuild_flags are False"
        )

    def test_rebuild_updates_data(self, dtype):
        """True flags: rebuilt system data should match a fresh full rebuild."""
        from nvalchemiops.jax.neighbors.batch_cell_list import (
            batch_build_cell_list,
            batch_query_cell_list,
        )

        positions = jnp.vstack(
            [
                jnp.array(
                    [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
                    dtype=dtype,
                ),
                jnp.array(
                    [[10.0, 0.0, 0.0], [10.5, 0.0, 0.0], [10.0, 0.5, 0.0]],
                    dtype=dtype,
                ),
            ]
        )
        cells = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ],
            dtype=dtype,
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])
        batch_idx = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 3, 6], dtype=jnp.int32)
        cutoff = 1.0
        max_neighbors = 10

        cell_cache = batch_build_cell_list(
            positions,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cutoff=cutoff,
            cell=cells,
            pbc=pbcs,
        )
        (
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_search_radius,
            _cell_origin,
        ) = cell_cache

        # Reference: full query
        _, nn_ref, _ = batch_query_cell_list(
            positions,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cutoff=cutoff,
            cell=cells,
            pbc=pbcs,
            cells_per_dimension=cells_per_dimension,
            atom_periodic_shifts=atom_periodic_shifts,
            atom_to_cell_mapping=atom_to_cell_mapping,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            neighbor_search_radius=neighbor_search_radius,
            max_neighbors=max_neighbors,
        )

        # Selective rebuild with all flags=True
        nm_stale = jnp.full((positions.shape[0], max_neighbors), 99, dtype=jnp.int32)
        nn_stale = jnp.full((positions.shape[0],), 99, dtype=jnp.int32)

        rebuild_flags = jnp.ones(2, dtype=jnp.bool_)
        _, nn2, _ = batch_query_cell_list(
            positions,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cutoff=cutoff,
            cell=cells,
            pbc=pbcs,
            cells_per_dimension=cells_per_dimension,
            atom_periodic_shifts=atom_periodic_shifts,
            atom_to_cell_mapping=atom_to_cell_mapping,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            neighbor_search_radius=neighbor_search_radius,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm_stale,
            num_neighbors=nn_stale,
            rebuild_flags=rebuild_flags,
        )

        assert jnp.all(nn2 == nn_ref), (
            "num_neighbors should match full rebuild when all flags=True"
        )


class TestJaxBatchCellListAutograd:
    """Differentiable per-pair distances/vectors via ``return_distances`` /
    ``return_vectors`` flags.  Exercises the autograd primitive in
    :mod:`nvalchemiops.jax.neighbors._autograd`.
    """

    def _make_two_systems(self, dtype=jnp.float64, n_per=6, box=5.0, scale=0.6):
        key = jax.random.key(0)
        pos = jax.random.normal(key, (2 * n_per, 3), dtype=dtype) * scale
        batch_idx = jnp.concatenate(
            [jnp.zeros(n_per, dtype=jnp.int32), jnp.ones(n_per, dtype=jnp.int32)]
        )
        cell = jnp.tile(jnp.eye(3, dtype=dtype)[None] * box, (2, 1, 1))
        pbc = jnp.ones((2, 3), dtype=jnp.bool_)
        return pos, cell, pbc, batch_idx

    def test_forward_returns_distances_and_vectors(self):
        pos, cell, pbc, batch_idx = self._make_two_systems()
        out = batch_cell_list(
            pos,
            1.5,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            return_distances=True,
            return_vectors=True,
        )
        assert len(out) == 5  # nm, nn, shifts, d, v
        nm, nn, shifts, d, v = out
        assert d.shape == (pos.shape[0], nm.shape[1])
        assert v.shape == (pos.shape[0], nm.shape[1], 3)
        assert d.dtype == pos.dtype
        assert v.dtype == pos.dtype

    def test_return_tuple_shape_extends_with_flags(self):
        pos, cell, pbc, batch_idx = self._make_two_systems()
        base = batch_cell_list(
            pos,
            1.5,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
        )
        assert len(base) == 3
        plus_d = batch_cell_list(
            pos,
            1.5,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            return_distances=True,
        )
        assert len(plus_d) == 4
        plus_v = batch_cell_list(
            pos,
            1.5,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            return_vectors=True,
        )
        assert len(plus_v) == 4
        plus_both = batch_cell_list(
            pos,
            1.5,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            return_distances=True,
            return_vectors=True,
        )
        assert len(plus_both) == 5

    def test_grad_positions_finite(self):
        pos, cell, pbc, batch_idx = self._make_two_systems()

        def loss(p):
            *_, d, _ = batch_cell_list(
                p,
                1.5,
                cell=cell,
                pbc=pbc,
                batch_idx=batch_idx,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        g = jax.grad(loss)(pos)
        assert g.shape == pos.shape
        assert jnp.isfinite(g).all().item()

    def test_grad_cell_finite(self):
        pos, cell, pbc, batch_idx = self._make_two_systems()

        def loss(c):
            *_, d, _ = batch_cell_list(
                pos,
                1.5,
                cell=c,
                pbc=pbc,
                batch_idx=batch_idx,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        g = jax.grad(loss)(cell)
        assert g.shape == cell.shape
        assert jnp.isfinite(g).all().item()

    def test_check_grads_against_finite_differences(self):
        from jax.test_util import check_grads

        pos, cell, pbc, batch_idx = self._make_two_systems()

        def loss(p):
            *_, d, _ = batch_cell_list(
                p,
                1.5,
                cell=cell,
                pbc=pbc,
                batch_idx=batch_idx,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        check_grads(loss, (pos,), order=1, atol=1e-4, rtol=1e-4, modes=["rev"])

    def test_hessian_vector_product_smoke(self):
        """Second-order HVP smoke — see TestJaxNaiveAutograd."""
        pos, cell, pbc, batch_idx = self._make_two_systems()
        v = jax.random.normal(jax.random.key(1), pos.shape, dtype=pos.dtype)

        def loss(p):
            *_, d, _ = batch_cell_list(
                p,
                1.5,
                cell=cell,
                pbc=pbc,
                batch_idx=batch_idx,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        hvp = jax.grad(lambda p: jnp.vdot(jax.grad(loss)(p), v))(pos)
        assert jnp.isfinite(hvp).all().item()
        assert hvp.shape == pos.shape

    def test_target_indices_partial_matches_full_restricted(self):
        """``target_indices`` (partial neighbor lists) is wired (task 5).

        The compact output has ``num_targets`` rows (row ``r`` -> atom
        ``target_indices[r]``); each row's neighbor set must equal the full
        matrix restricted to that atom.  Targets span both systems, exercising
        the per-target ``batch_idx`` lookup.  COO source index ``nl[0]`` is the
        compact row in ``[0, num_targets)`` (matches the torch contract)."""
        pos, cell, pbc, batch_idx = self._make_two_systems(n_per=6)
        n = pos.shape[0]
        # Targets in both systems (0..5 -> system 0, 6..11 -> system 1).
        targets = jnp.array([0, 2, 7, 9], dtype=jnp.int32)
        nt = int(targets.shape[0])
        mn = 24

        pnm, pnn, _ = batch_cell_list(
            pos,
            1.5,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            max_neighbors=mn,
            target_indices=targets,
            fill_value=n,
        )
        assert pnm.shape == (nt, mn) and pnn.shape == (nt,)

        fnm, fnn, _ = batch_cell_list(
            pos,
            1.5,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            max_neighbors=mn,
            fill_value=n,
        )
        pnm, pnn, fnm, fnn, tg = (np.asarray(x) for x in (pnm, pnn, fnm, fnn, targets))

        def row_set(nm, count):
            return {int(nm[k]) for k in range(int(count))}

        for r in range(nt):
            assert row_set(pnm[r], pnn[r]) == row_set(fnm[int(tg[r])], fnn[int(tg[r])])

        nl, _nptr, _nls = batch_cell_list(
            pos,
            1.5,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            max_neighbors=mn,
            target_indices=targets,
            return_neighbor_list=True,
        )
        nl = np.asarray(nl)
        if nl.shape[1] > 0:
            assert int(nl[0].max()) < nt
