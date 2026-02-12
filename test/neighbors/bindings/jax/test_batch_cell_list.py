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

"""Tests for JAX bindings of batched cell list neighbor construction methods."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nvalchemiops.jax.neighbors.batched import batch_cell_list

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
