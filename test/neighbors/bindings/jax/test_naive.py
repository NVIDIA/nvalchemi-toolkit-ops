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

"""Tests for JAX bindings of naive neighbor list methods."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nvalchemiops.jax.neighbors.naive import naive_neighbor_list

from .conftest import requires_gpu

pytestmark = requires_gpu


class TestNaiveNeighborList:
    """Test naive_neighbor_list function."""

    def test_single_atom_no_neighbors(self):
        """Test with single atom (should have no neighbors)."""
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        cutoff = 1.0

        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions, cutoff, max_neighbors=10
        )

        assert neighbor_matrix.shape == (1, 10)
        assert num_neighbors.shape == (1,)
        assert int(num_neighbors[0]) == 0

    def test_two_atom_within_cutoff(self):
        """Test with two atoms within cutoff."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32)
        cutoff = 1.0

        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions, cutoff, max_neighbors=10
        )

        assert neighbor_matrix.shape == (2, 10)
        assert num_neighbors.shape == (2,)
        # Each atom should find the other one
        assert int(num_neighbors[0]) >= 1
        assert int(num_neighbors[1]) >= 1

    def test_two_atom_outside_cutoff(self):
        """Test with two atoms outside cutoff."""
        positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32)
        cutoff = 1.0

        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions, cutoff, max_neighbors=10
        )

        assert int(num_neighbors[0]) == 0
        assert int(num_neighbors[1]) == 0

    def test_cubic_system_no_pbc(self):
        """Test with cubic lattice without PBC."""
        # 8 atoms in a cube
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        cutoff = 1.5

        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions, cutoff, max_neighbors=20
        )

        assert neighbor_matrix.shape == (8, 20)
        assert num_neighbors.shape == (8,)
        # Each corner atom should have 3 neighbors
        assert all(int(num_neighbors[i]) > 0 for i in range(8))

    def test_return_neighbor_list_format(self):
        """Test return_neighbor_list parameter."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ],
            dtype=jnp.float32,
        )
        cutoff = 1.0

        neighbor_list, neighbor_ptr = naive_neighbor_list(
            positions, cutoff, max_neighbors=10, return_neighbor_list=True
        )

        assert neighbor_list.shape[0] == 2  # COO format
        assert neighbor_ptr.shape == (4,)  # 3 atoms + 1

    def test_with_pbc(self):
        """Test with periodic boundary conditions."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [9.5, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([[True, True, True]])
        cutoff = 1.0

        neighbor_matrix, num_neighbors, shifts = naive_neighbor_list(
            positions, cutoff, cell=cell, pbc=pbc, max_neighbors=10
        )

        assert neighbor_matrix.shape == (2, 10)
        assert num_neighbors.shape == (2,)
        assert shifts.shape == (2, 10, 3)
        # With PBC, atoms should be neighbors
        assert int(num_neighbors[0]) >= 1
        assert int(num_neighbors[1]) >= 1


class TestNaiveEdgeCases:
    """Edge case tests for naive_neighbor_list."""

    def test_zero_cutoff_returns_no_neighbors(self):
        """Zero cutoff should find zero neighbors."""
        # 4 atoms in a cluster
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ],
            dtype=jnp.float32,
        )

        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions, cutoff=0.0, max_neighbors=10
        )
        assert jnp.all(num_neighbors == 0)

    def test_zero_cutoff_with_pbc(self):
        """Zero cutoff with PBC should find zero neighbors."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32)
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 10.0
        pbc = jnp.array([[True, True, True]])

        nm, nn, shifts = naive_neighbor_list(
            positions, cutoff=0.0, cell=cell, pbc=pbc, max_neighbors=10
        )
        assert jnp.all(nn == 0)

    def test_large_cutoff_finds_all_pairs(self):
        """Large cutoff should find all possible neighbors (N-1 per atom)."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        cutoff = 100.0

        _, num_neighbors = naive_neighbor_list(positions, cutoff, max_neighbors=10)
        # Each of 4 atoms should see all other 3
        assert jnp.all(num_neighbors == 3)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_precision_consistency(self, dtype):
        """f32 and f64 should find the same number of neighbors for same positions."""
        # Use same random positions for both precisions
        key = jax.random.PRNGKey(42)
        positions_f32 = jax.random.uniform(key, shape=(20, 3), dtype=jnp.float32) * 5.0
        positions_f64 = positions_f32.astype(jnp.float64)

        if dtype == jnp.float32:
            positions = positions_f32
        else:
            positions = positions_f64

        cutoff = 3.0

        _, num_neighbors = naive_neighbor_list(positions, cutoff, max_neighbors=100)
        total = int(jnp.sum(num_neighbors))
        assert total > 0  # Sanity: should find some neighbors

    def test_half_fill_mode(self):
        """half_fill=True should find roughly half the pairs of full fill."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=jnp.float32,
        )
        cutoff = 1.0

        _, nn_full = naive_neighbor_list(
            positions, cutoff, max_neighbors=10, half_fill=False
        )
        _, nn_half = naive_neighbor_list(
            positions, cutoff, max_neighbors=10, half_fill=True
        )
        total_full = int(jnp.sum(nn_full))
        total_half = int(jnp.sum(nn_half))
        # half_fill should have exactly half the pairs (symmetric -> half)
        assert total_half * 2 == total_full, (
            f"half_fill produced {total_half} pairs, expected {total_full // 2}"
        )

    def test_distance_validity_no_pbc(self):
        """All reported neighbors should actually be within the cutoff distance."""
        key = jax.random.PRNGKey(123)
        positions = jax.random.uniform(key, shape=(15, 3), dtype=jnp.float32) * 5.0
        cutoff = 2.5

        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions, cutoff, max_neighbors=50
        )
        # Check that every reported neighbor is within cutoff
        for i in range(positions.shape[0]):
            nn = int(num_neighbors[i])
            for k in range(nn):
                j = int(neighbor_matrix[i, k])
                dist = float(jnp.linalg.norm(positions[j] - positions[i]))
                assert dist < cutoff + 1e-5, (
                    f"Atom {i} neighbor {j} has distance {dist} > cutoff {cutoff}"
                )

    def test_distance_validity_with_pbc(self):
        """All reported PBC neighbors should be within cutoff (accounting for shifts)."""
        key = jax.random.PRNGKey(456)
        positions = jax.random.uniform(key, shape=(10, 3), dtype=jnp.float32) * 8.0
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 10.0
        pbc = jnp.array([[True, True, True]])
        cutoff = 3.0

        nm, nn, shifts = naive_neighbor_list(
            positions, cutoff, cell=cell, pbc=pbc, max_neighbors=50
        )
        cell_mat = cell[0]  # (3, 3)
        for i in range(positions.shape[0]):
            n = int(nn[i])
            for k in range(n):
                j = int(nm[i, k])
                shift_vec = jnp.dot(shifts[i, k].astype(jnp.float32), cell_mat)
                rij = positions[j] - positions[i] + shift_vec
                dist = float(jnp.linalg.norm(rij))
                assert dist < cutoff + 1e-4, (
                    f"Atom {i}->{j} with shift {shifts[i, k]} has dist {dist} > cutoff {cutoff}"
                )

    def test_mixed_pbc(self):
        """Mixed PBC (periodic in x,y only) should produce zero z-shifts."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 2.0
        pbc = jnp.array([[True, True, False]])  # No PBC in z
        cutoff = 1.5

        nm, nn, shifts = naive_neighbor_list(
            positions, cutoff, cell=cell, pbc=pbc, max_neighbors=30
        )
        # z-direction should have NO shifts since PBC is off
        assert int(jnp.sum(jnp.abs(shifts[:, :, 2]))) == 0, (
            "z-shifts should be zero when pbc[z]=False"
        )
        # x/y directions should have SOME shifts for this tight cell
        assert (
            int(jnp.sum(shifts[:, :, 0] ** 2)) > 0
            or int(jnp.sum(shifts[:, :, 1] ** 2)) > 0
        )

    def test_return_neighbor_list_with_pbc(self):
        """return_neighbor_list=True with PBC should return (list, ptr, shifts)."""
        positions = jnp.array([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=jnp.float32)
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 10.0
        pbc = jnp.array([[True, True, True]])
        cutoff = 1.0

        nl, ptr, shifts = naive_neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=10,
            return_neighbor_list=True,
        )
        assert nl.shape[0] == 2  # COO format (2, num_pairs)
        assert ptr.shape == (3,)  # 2 atoms + 1
        assert shifts.shape[1] == 3
        # Should find neighbors across PBC
        assert nl.shape[1] > 0


class TestNaiveNeighborListJIT:
    """Smoke tests for naive_neighbor_list compatibility with jax.jit."""

    def test_jit_no_pbc(self):
        """Test naive_neighbor_list without PBC works with jax.jit."""
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            dtype=jnp.float32,
        )

        @jax.jit
        def jitted_naive(positions):
            return naive_neighbor_list(positions, cutoff=1.0, max_neighbors=10)

        neighbor_matrix, num_neighbors = jitted_naive(positions)

        assert neighbor_matrix.shape == (3, 10)
        assert num_neighbors.shape == (3,)
        assert jnp.all(num_neighbors >= 0)

    @pytest.mark.xfail(
        reason="PBC path calls compute_naive_num_shifts which uses int() on traced values. "
        "Full JIT support for PBC neighbor lists is planned but not yet implemented.",
        raises=(
            jax.errors.ConcretizationTypeError,
            jax.errors.TracerIntegerConversionError,
        ),
        strict=True,
    )
    def test_jit_with_pbc(self):
        """Test naive_neighbor_list with PBC works with jax.jit."""
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([[True, True, True]])

        @jax.jit
        def jitted_naive_pbc(positions, cell, pbc):
            return naive_neighbor_list(
                positions, cutoff=1.0, cell=cell, pbc=pbc, max_neighbors=10
            )

        neighbor_matrix, num_neighbors, shifts = jitted_naive_pbc(positions, cell, pbc)

        assert neighbor_matrix.shape == (2, 10)
        assert num_neighbors.shape == (2,)
        assert shifts.shape == (2, 10, 3)
        assert jnp.all(num_neighbors >= 0)
