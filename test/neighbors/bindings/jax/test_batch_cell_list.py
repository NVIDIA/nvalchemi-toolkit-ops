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

    @pytest.mark.xfail(
        reason="estimate_batch_cell_list_sizes derives array shapes from traced input "
        "data (cell geometry), which is incompatible with jax.jit. Provide "
        "max_total_cells explicitly to bypass. See TODO in "
        "estimate_batch_cell_list_sizes.",
        raises=TypeError,
        strict=True,
    )
    def test_jit_with_pbc(self):
        """Test batched cell list with PBC works with jax.jit."""
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

        nm, nn, shifts = jitted_batch_cell_list(
            positions, cells, pbcs, batch_idx, batch_ptr
        )

        assert nm.shape[0] == 4
        assert nn.shape == (4,)
        assert shifts.shape[0] == 4
        assert shifts.shape[2] == 3


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
