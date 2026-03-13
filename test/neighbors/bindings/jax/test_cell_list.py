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

"""Tests for JAX bindings of cell list neighbor construction methods."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors.cell_list import cell_list, estimate_cell_list_sizes

from .conftest import requires_gpu, requires_vesin

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


class TestCellListJIT:
    """Smoke tests for cell_list compatibility with jax.jit."""

    @pytest.mark.xfail(
        reason="estimate_cell_list_sizes derives array shapes from traced input data "
        "(cell geometry), which is incompatible with jax.jit. Provide max_total_cells "
        "explicitly to bypass. See TODO in estimate_cell_list_sizes.",
        raises=TypeError,
        strict=True,
    )
    def test_jit_with_pbc(self):
        """Test cell_list with PBC works with jax.jit."""
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([[True, True, True]])

        @jax.jit
        def jitted_cell_list(positions, cell, pbc):
            return cell_list(positions, cutoff=1.0, cell=cell, pbc=pbc)

        neighbor_matrix, num_neighbors, shifts = jitted_cell_list(positions, cell, pbc)

        assert neighbor_matrix.shape[0] == 2
        assert num_neighbors.shape == (2,)
        assert shifts.shape[0] == 2
        assert shifts.shape[2] == 3


class TestEstimateCellListSizes:
    """Tests for estimate_cell_list_sizes neighbor_search_radius output."""

    def test_search_radius_small_cell(self):
        """When cutoff exceeds cell size, search radius must be > 1."""
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 2.0
        pbc = jnp.array([[True, True, True]])

        _, _, neighbor_search_radius = estimate_cell_list_sizes(
            positions, cell, cutoff=5.0, pbc=pbc
        )

        # ceil(5.0 / 2.0) = 3 in each dimension
        assert neighbor_search_radius.shape == (3,)
        for i in range(3):
            assert int(neighbor_search_radius[i]) >= 3, (
                f"dim {i}: expected search radius >= 3, got {int(neighbor_search_radius[i])}"
            )

    def test_search_radius_large_cell(self):
        """When cell size exceeds cutoff, search radius should be 1."""
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 10.0
        pbc = jnp.array([[True, True, True]])

        _, _, neighbor_search_radius = estimate_cell_list_sizes(
            positions, cell, cutoff=2.0, pbc=pbc
        )

        assert neighbor_search_radius.shape == (3,)
        for i in range(3):
            assert int(neighbor_search_radius[i]) == 1, (
                f"dim {i}: expected search radius == 1, got {int(neighbor_search_radius[i])}"
            )

    def test_search_radius_no_pbc(self):
        """With no PBC and a single cell, search radius should be 0."""
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 5.0
        pbc = jnp.array([[False, False, False]])

        _, _, neighbor_search_radius = estimate_cell_list_sizes(
            positions, cell, cutoff=10.0, pbc=pbc
        )

        assert neighbor_search_radius.shape == (3,)
        for i in range(3):
            assert int(neighbor_search_radius[i]) == 0, (
                f"dim {i}: expected search radius == 0 with no PBC, "
                f"got {int(neighbor_search_radius[i])}"
            )

    def test_search_radius_d3_scenario(self):
        """D3 benchmark scenario: cell 12.42 A, cutoff 21.2 A -> radius 2."""
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 12.42
        pbc = jnp.array([[True, True, True]])

        _, _, neighbor_search_radius = estimate_cell_list_sizes(
            positions, cell, cutoff=21.2, pbc=pbc
        )

        assert neighbor_search_radius.shape == (3,)
        for i in range(3):
            assert int(neighbor_search_radius[i]) == 2, (
                f"dim {i}: expected search radius == 2, got {int(neighbor_search_radius[i])}"
            )


def _vesin_brute_force(positions_np, cell_np, pbc_np, cutoff):
    """Compute reference neighbor list using vesin.

    Returns numpy arrays (i, j, shifts) for a full (bidirectional) neighbor list.
    """
    from vesin import NeighborList

    calculator = NeighborList(cutoff=cutoff, full_list=True, sorted=True)
    i, j, shifts = calculator.compute(
        points=positions_np, box=cell_np, periodic=pbc_np, quantities="ijS"
    )
    return i.astype(np.int32), j.astype(np.int32), shifts.astype(np.int32)


class TestCellListCorrectnessVesin:
    """Correctness tests comparing JAX cell_list against vesin reference.

    These tests specifically exercise scenarios where cutoff > cell size,
    requiring neighbor_search_radius > 1, to prevent regressions of the
    hardcoded search radius bug.
    """

    @requires_vesin
    @pytest.mark.parametrize(
        "cell_size, cutoff",
        [
            (2.0, 5.0),
            (4.0, 5.0),
            (10.0, 5.0),
            (12.42, 21.2),
        ],
        ids=[
            "cutoff_2.5x_cell",
            "cutoff_1.25x_cell",
            "cutoff_lt_cell",
            "d3_benchmark",
        ],
    )
    def test_neighbor_count_matches_vesin(self, cell_size, cutoff):
        """Total neighbor count from cell_list must match vesin reference."""
        num_atoms = 8
        n_side = 2
        spacing = cell_size / n_side
        coords = [
            [ix * spacing, iy * spacing, iz * spacing]
            for ix in range(n_side)
            for iy in range(n_side)
            for iz in range(n_side)
        ]
        positions_np = np.array(coords[:num_atoms], dtype=np.float64)
        cell_np = np.eye(3, dtype=np.float64) * cell_size
        pbc_np = np.array([True, True, True])

        ref_i, ref_j, _ = _vesin_brute_force(positions_np, cell_np, pbc_np, cutoff)
        ref_total = len(ref_i)

        positions_jax = jnp.array(positions_np, dtype=jnp.float32)
        cell_jax = jnp.array(cell_np, dtype=jnp.float32).reshape(1, 3, 3)
        pbc_jax = jnp.array([[True, True, True]])

        _, num_neighbors, _ = cell_list(
            positions_jax, cutoff, cell_jax, pbc_jax, max_neighbors=2000
        )
        jax_total = int(jnp.sum(num_neighbors))

        assert jax_total == ref_total, (
            f"Neighbor count mismatch: JAX cell_list={jax_total}, "
            f"vesin reference={ref_total} "
            f"(cell_size={cell_size}, cutoff={cutoff})"
        )

    @requires_vesin
    @pytest.mark.parametrize(
        "cell_size, cutoff",
        [
            (2.0, 5.0),
            (4.0, 5.0),
            (10.0, 5.0),
            (12.42, 21.2),
        ],
        ids=[
            "cutoff_2.5x_cell",
            "cutoff_1.25x_cell",
            "cutoff_lt_cell",
            "d3_benchmark",
        ],
    )
    def test_neighbor_pairs_match_vesin(self, cell_size, cutoff):
        """Sorted (i, j, shift) triples from cell_list must match vesin."""
        num_atoms = 8
        n_side = 2
        spacing = cell_size / n_side
        coords = [
            [ix * spacing, iy * spacing, iz * spacing]
            for ix in range(n_side)
            for iy in range(n_side)
            for iz in range(n_side)
        ]
        positions_np = np.array(coords[:num_atoms], dtype=np.float64)
        cell_np = np.eye(3, dtype=np.float64) * cell_size
        pbc_np = np.array([True, True, True])

        ref_i, ref_j, ref_shifts = _vesin_brute_force(
            positions_np, cell_np, pbc_np, cutoff
        )

        positions_jax = jnp.array(positions_np, dtype=jnp.float32)
        cell_jax = jnp.array(cell_np, dtype=jnp.float32).reshape(1, 3, 3)
        pbc_jax = jnp.array([[True, True, True]])

        nl, ptr, shifts = cell_list(
            positions_jax,
            cutoff,
            cell_jax,
            pbc_jax,
            max_neighbors=2000,
            return_neighbor_list=True,
        )

        jax_i = np.asarray(nl[0])
        jax_j = np.asarray(nl[1])
        jax_shifts = np.asarray(shifts)

        assert len(jax_i) == len(ref_i), (
            f"Pair count mismatch: JAX={len(jax_i)}, vesin={len(ref_i)}"
        )

        def _sort_key(i_arr, j_arr, s_arr):
            return np.lexsort([s_arr[:, 2], s_arr[:, 1], s_arr[:, 0], j_arr, i_arr])

        jax_order = _sort_key(jax_i, jax_j, jax_shifts)
        ref_order = _sort_key(ref_i, ref_j, ref_shifts)

        np.testing.assert_array_equal(jax_i[jax_order], ref_i[ref_order])
        np.testing.assert_array_equal(jax_j[jax_order], ref_j[ref_order])
        np.testing.assert_array_equal(jax_shifts[jax_order], ref_shifts[ref_order])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
class TestCellListSelectiveRebuildFlags:
    """Test selective rebuild (rebuild_flags) for JAX cell list."""

    def test_no_rebuild_preserves_data(self, dtype):
        """Flag=False: neighbor data should remain unchanged."""
        from nvalchemiops.jax.neighbors.cell_list import (
            build_cell_list,
            query_cell_list,
        )

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
            dtype=dtype,
        )
        cell = jnp.eye(3, dtype=dtype).reshape(1, 3, 3) * 4.0
        pbc = jnp.array([[True, True, True]])
        cutoff = 2.5
        max_neighbors = 20

        # Build cell list
        (
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_search_radius,
        ) = build_cell_list(positions, cutoff, cell, pbc)

        # Initial query
        nm, nn, nm_shifts = query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_search_radius,
            max_neighbors=max_neighbors,
        )

        saved_nn = jnp.array(nn)

        # Selective rebuild with flag=False
        rebuild_flags = jnp.zeros(1, dtype=jnp.bool_)
        nm2, nn2, nm_shifts2 = query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_search_radius,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm,
            num_neighbors=nn,
            rebuild_flags=rebuild_flags,
        )

        assert jnp.all(nn2 == saved_nn), (
            "num_neighbors must be unchanged when rebuild_flags is False"
        )

    def test_rebuild_updates_data(self, dtype):
        """Flag=True: result should match a fresh full rebuild."""
        from nvalchemiops.jax.neighbors.cell_list import (
            build_cell_list,
            query_cell_list,
        )

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
            dtype=dtype,
        )
        cell = jnp.eye(3, dtype=dtype).reshape(1, 3, 3) * 4.0
        pbc = jnp.array([[True, True, True]])
        cutoff = 2.5
        max_neighbors = 20

        # Build cell list
        (
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_search_radius,
        ) = build_cell_list(positions, cutoff, cell, pbc)

        # Reference: full query
        _, nn_ref, _ = query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_search_radius,
            max_neighbors=max_neighbors,
        )

        # Selective rebuild with flag=True
        nm_stale = jnp.full((positions.shape[0], max_neighbors), 99, dtype=jnp.int32)
        nn_stale = jnp.full((positions.shape[0],), 99, dtype=jnp.int32)

        rebuild_flags = jnp.ones(1, dtype=jnp.bool_)
        _, nn2, _ = query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_search_radius,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm_stale,
            num_neighbors=nn_stale,
            rebuild_flags=rebuild_flags,
        )

        assert jnp.all(nn2 == nn_ref), (
            "num_neighbors should match full rebuild when flag=True"
        )
