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

from nvalchemiops.jax.neighbors.cell_list import (
    build_cell_list,
    cell_list,
    estimate_cell_list_sizes,
    query_cell_list,
)

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

    def test_topology_only_grad_pbc_is_zero(self):
        """Topology-only cell-list outputs do not differentiate Warp FFI."""
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        cell = jnp.eye(3, dtype=jnp.float32) * 5.0
        pbc = jnp.array([True, True, True])

        def loss(pos):
            neighbor_matrix, num_neighbors, shifts = cell_list(
                pos,
                2.0,
                cell=cell,
                pbc=pbc,
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

    def test_jit_with_pbc_requires_precomputed_sizing(self):
        """The traced sizing path should fail with a clear JAX shape error."""
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([[True, True, True]])

        @jax.jit
        def jitted_cell_list(positions, cell, pbc):
            return cell_list(positions, cutoff=1.0, cell=cell, pbc=pbc)

        with pytest.raises(TypeError, match="Shapes must be .* concrete values"):
            jitted_cell_list(positions, cell, pbc)

    def test_jit_with_pbc_precomputed_sizing(self):
        """PBC cell list should work under JIT when sizing is concrete."""
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([[True, True, True]])
        neighbor_search_radius = jnp.ones(3, dtype=jnp.int32)

        @jax.jit
        def jitted_cell_list(positions, cell, pbc):
            return cell_list(
                positions,
                cutoff=1.0,
                cell=cell,
                pbc=pbc,
                max_total_cells=8,
                neighbor_search_radius=neighbor_search_radius,
            )

        neighbor_matrix, num_neighbors, shifts = jitted_cell_list(positions, cell, pbc)

        assert neighbor_matrix.shape[0] == 2
        assert num_neighbors.shape == (2,)
        assert shifts.shape[0] == 2
        assert shifts.shape[2] == 3

    def test_jit_auto_falls_back_when_pair_centric_sizing_is_traced(self):
        """``strategy='auto'`` must not expose pair-centric host reads to JIT."""
        num_atoms = 200
        max_neighbors = 128
        box_size = 15.0
        key = jax.random.PRNGKey(42)
        positions = (
            jax.random.uniform(key, (num_atoms, 3), dtype=jnp.float32) * box_size
        )
        cell = jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, :, :] * box_size
        pbc = jnp.array([[True, True, True]])

        @jax.jit
        def jitted_cell_list(positions, cell, pbc):
            return cell_list(
                positions,
                cutoff=6.0,
                cell=cell * 1.5,
                pbc=pbc,
                max_neighbors=max_neighbors,
                max_total_cells=16,
            )

        neighbor_matrix, num_neighbors, shifts = jitted_cell_list(positions, cell, pbc)

        assert neighbor_matrix.shape == (num_atoms, max_neighbors)
        assert neighbor_matrix.dtype == jnp.int32
        assert num_neighbors.shape == (num_atoms,)
        assert shifts.shape == (num_atoms, max_neighbors, 3)

    def test_jit_explicit_pair_centric_still_requires_concrete_sizing(self):
        """Explicit pair-centric keeps the concrete launch-sizing contract."""
        num_atoms = 200
        box_size = 15.0
        key = jax.random.PRNGKey(43)
        positions = (
            jax.random.uniform(key, (num_atoms, 3), dtype=jnp.float32) * box_size
        )
        cell = jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, :, :] * box_size
        pbc = jnp.array([[True, True, True]])

        @jax.jit
        def jitted_cell_list(positions, cell, pbc):
            return cell_list(
                positions,
                cutoff=6.0,
                cell=cell * 1.5,
                pbc=pbc,
                max_neighbors=128,
                max_total_cells=16,
                strategy="pair_centric",
            )

        with pytest.raises(ValueError, match="needs a concrete neighbor_search_radius"):
            jitted_cell_list(positions, cell, pbc)


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
        """D3 benchmark scenario: cell 12.42 A, cutoff 21.2 A.

        With the warp kernel's adaptive promotion (each PBC axis is bumped to
        ``ADAPTIVE_MIN_CELLS=4`` and then halved to fit ``max_total_cells``),
        a tiny ``max_total_cells=8`` budget lands at ``cells_per_dim=[2, 2, 2]``
        so the search radius is ``ceil(21.2 * 2 / 12.42) = 4`` per axis.
        """
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 12.42
        pbc = jnp.array([[True, True, True]])

        _, _, neighbor_search_radius = estimate_cell_list_sizes(
            positions, cell, cutoff=21.2, pbc=pbc
        )

        assert neighbor_search_radius.shape == (3,)
        for i in range(3):
            assert int(neighbor_search_radius[i]) == 4, (
                f"dim {i}: expected search radius == 4, "
                f"got {int(neighbor_search_radius[i])}"
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
            pytest.param(2.0, 5.0, marks=pytest.mark.slow),
            (4.0, 5.0),
            (10.0, 5.0),
            pytest.param(12.42, 21.2, marks=pytest.mark.slow),
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
            pytest.param(2.0, 5.0, marks=pytest.mark.slow),
            (4.0, 5.0),
            (10.0, 5.0),
            pytest.param(12.42, 21.2, marks=pytest.mark.slow),
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


def _assert_arrays_equal(lhs, rhs) -> None:
    """Assert two tuples of JAX arrays are exactly equal."""
    assert len(lhs) == len(rhs)
    for left, right in zip(lhs, rhs, strict=True):
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        assert jnp.array_equal(left, right)


def _make_cell_list_inputs(dtype):
    """Create a small periodic system for cell-list graph-mode tests."""
    positions = jnp.array(
        [
            [0.1, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [0.0, 1.7, 0.0],
            [1.7, 1.7, 0.0],
            [0.0, 0.0, 1.8],
        ],
        dtype=dtype,
    )
    cell = jnp.eye(3, dtype=dtype).reshape(1, 3, 3) * 4.0
    pbc = jnp.array([[True, True, True]])
    cutoff = 2.2
    max_neighbors = 16
    max_total_cells, _, neighbor_search_radius = estimate_cell_list_sizes(
        positions,
        cell,
        cutoff,
        pbc,
    )
    return (
        positions,
        cell,
        pbc,
        cutoff,
        max_neighbors,
        int(max_total_cells),
        neighbor_search_radius,
    )


class TestCellListAtomCentricDirect:
    """``atom_centric_path="direct"`` skips the sorted gather and reads positions
    directly; it must produce the same neighbor sets as the sorted kernel."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_direct_matches_sorted(self, dtype):
        if dtype == jnp.float64:
            jax.config.update("jax_enable_x64", True)
        positions, cell, pbc, cutoff, max_neighbors, max_total_cells, nsr = (
            _make_cell_list_inputs(dtype)
        )

        def neighbor_sets(acp):
            nm, nn, _sh = cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                atom_centric_path=acp,
                max_neighbors=max_neighbors,
                max_total_cells=max_total_cells,
                neighbor_search_radius=nsr,
            )
            fill = positions.shape[0]
            sets = [frozenset(j for j in row if j != fill) for row in nm.tolist()]
            return sets, nn.tolist()

        sorted_sets, sorted_counts = neighbor_sets("sorted")
        direct_sets, direct_counts = neighbor_sets("direct")
        assert direct_sets == sorted_sets
        assert direct_counts == sorted_counts


class TestCellListGraphMode:
    """Graph-mode coverage for JAX cell-list bindings."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_build_matches_default(self, dtype):
        """Cell-list build graph mode should match the default path."""
        (
            positions,
            cell,
            pbc,
            cutoff,
            _max_neighbors,
            max_total_cells,
            neighbor_search_radius,
        ) = _make_cell_list_inputs(dtype)

        common = {
            "cells_per_dimension": jnp.full((3,), -1, dtype=jnp.int32),
            "neighbor_search_radius": neighbor_search_radius,
            "atom_periodic_shifts": jnp.full(
                (positions.shape[0], 3), -1, dtype=jnp.int32
            ),
            "atom_to_cell_mapping": jnp.full(
                (positions.shape[0], 3), -1, dtype=jnp.int32
            ),
            "atoms_per_cell_count": jnp.full((max_total_cells,), 9, dtype=jnp.int32),
            "cell_atom_start_indices": jnp.full(
                (max_total_cells,), -1, dtype=jnp.int32
            ),
            "cell_atom_list": jnp.full((positions.shape[0],), -1, dtype=jnp.int32),
            "max_total_cells": max_total_cells,
        }

        none_result = build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            graph_mode="none",
            **common,
        )
        warp_result = build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            graph_mode="warp",
            **common,
        )

        _assert_arrays_equal(none_result, warp_result)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("selective_flag", [None, False, True])
    def test_query_matches_default(self, dtype, selective_flag):
        """Cell-list query graph mode should match the default path."""
        (
            positions,
            cell,
            pbc,
            cutoff,
            max_neighbors,
            max_total_cells,
            neighbor_search_radius,
        ) = _make_cell_list_inputs(dtype)

        build_result = build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            max_total_cells=max_total_cells,
            neighbor_search_radius=neighbor_search_radius,
            graph_mode="none",
        )
        rebuild_flags = (
            None
            if selective_flag is None
            else jnp.array([selective_flag], dtype=jnp.bool_)
        )

        common = {
            "positions": positions,
            "cutoff": cutoff,
            "cell": cell,
            "pbc": pbc,
            "cells_per_dimension": build_result[0],
            "atom_periodic_shifts": build_result[1],
            "atom_to_cell_mapping": build_result[2],
            "atoms_per_cell_count": build_result[3],
            "cell_atom_start_indices": build_result[4],
            "cell_atom_list": build_result[5],
            "neighbor_search_radius": build_result[6],
            "max_neighbors": max_neighbors,
            "neighbor_matrix": jnp.full(
                (positions.shape[0], max_neighbors),
                42,
                dtype=jnp.int32,
            ),
            "neighbor_matrix_shifts": jnp.full(
                (positions.shape[0], max_neighbors, 3),
                -4,
                dtype=jnp.int32,
            ),
            "num_neighbors": jnp.full((positions.shape[0],), 42, dtype=jnp.int32),
            "rebuild_flags": rebuild_flags,
        }

        none_result = query_cell_list(graph_mode="none", **common)
        warp_result = query_cell_list(graph_mode="warp", **common)
        _assert_arrays_equal(none_result, warp_result)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("return_neighbor_list", [False, True])
    def test_top_level_matches_default(self, dtype, return_neighbor_list: bool):
        """Top-level cell_list graph mode should match the default path."""
        (
            positions,
            cell,
            pbc,
            cutoff,
            max_neighbors,
            max_total_cells,
            neighbor_search_radius,
        ) = _make_cell_list_inputs(dtype)

        common = {
            "positions": positions,
            "cutoff": cutoff,
            "cell": cell,
            "pbc": pbc,
            "max_neighbors": max_neighbors,
            "max_total_cells": max_total_cells,
            "return_neighbor_list": return_neighbor_list,
            "cells_per_dimension": jnp.full((3,), -1, dtype=jnp.int32),
            "neighbor_search_radius": neighbor_search_radius,
            "atom_periodic_shifts": jnp.full(
                (positions.shape[0], 3), -1, dtype=jnp.int32
            ),
            "atom_to_cell_mapping": jnp.full(
                (positions.shape[0], 3), -1, dtype=jnp.int32
            ),
            "atoms_per_cell_count": jnp.full((max_total_cells,), 7, dtype=jnp.int32),
            "cell_atom_start_indices": jnp.full(
                (max_total_cells,), -1, dtype=jnp.int32
            ),
            "cell_atom_list": jnp.full((positions.shape[0],), -1, dtype=jnp.int32),
            "neighbor_matrix": jnp.full(
                (positions.shape[0], max_neighbors),
                88,
                dtype=jnp.int32,
            ),
            "neighbor_matrix_shifts": jnp.full(
                (positions.shape[0], max_neighbors, 3),
                -8,
                dtype=jnp.int32,
            ),
            "num_neighbors": jnp.full((positions.shape[0],), 88, dtype=jnp.int32),
        }

        none_result = cell_list(graph_mode="none", **common)
        warp_result = cell_list(graph_mode="warp", **common)
        _assert_arrays_equal(none_result, warp_result)

    def test_invalid_value(self):
        """Invalid graph_mode values should raise for cell-list APIs."""
        positions = jnp.zeros((2, 3), dtype=jnp.float32)
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3)
        pbc = jnp.ones((1, 3), dtype=jnp.bool_)
        with pytest.raises(ValueError, match="graph_mode"):
            build_cell_list(
                positions,
                1.0,
                cell,
                pbc,
                max_total_cells=8,
                graph_mode="bad",
            )


class TestCellListPreallocatedBufferReuse:
    """Preallocated-buffer in-place reset branches in ``query_cell_list``.

    Covers the ``elif rebuild_flags is None and graph_mode == "none":`` paths
    in nvalchemiops/jax/neighbors/cell_list.py for ``neighbor_matrix``,
    ``num_neighbors``, and ``neighbor_matrix_shifts``.
    """

    def test_preallocated_buffers_reset_in_place(self):
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
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 4.0
        pbc = jnp.array([[True, True, True]])
        cutoff = 2.5
        max_neighbors = 20
        N = positions.shape[0]

        (
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_search_radius,
        ) = build_cell_list(positions, cutoff, cell, pbc)

        # Pre-allocate buffers seeded with sentinel garbage to verify the
        # in-place .at[:].set() reset paths actually overwrite.
        nm_buf = jnp.full((N, max_neighbors), -9, dtype=jnp.int32)
        nn_buf = jnp.full((N,), 99, dtype=jnp.int32)
        nms_buf = jnp.full((N, max_neighbors, 3), 7, dtype=jnp.int32)

        # rebuild_flags=None (default) + graph_mode='none' (default) +
        # all three buffers preallocated → hits all three elif branches.
        nm, nn, nms = query_cell_list(
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
            neighbor_matrix=nm_buf,
            num_neighbors=nn_buf,
            neighbor_matrix_shifts=nms_buf,
        )
        # Reference (no preallocation) must match.
        nm_ref, nn_ref, nms_ref = query_cell_list(
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
        assert jnp.array_equal(nn, nn_ref)
        # active-range pair sets per row must match (tail may diverge under
        # the always-write-shifts contract; both runs see the same prefix)
        for i in range(N):
            k = int(nn_ref[i])
            assert jnp.array_equal(jnp.sort(nm[i, :k]), jnp.sort(nm_ref[i, :k]))


# ---------------------------------------------------------------------------
# JAX autograd path — exercises ``_route_pair_outputs`` /
# ``_reconstruct_pair_geometry`` in nvalchemiops.jax.neighbors._autograd via a
# pure-JAX forward closure.  Validates the autograd math and JIT compatibility
# without the warp-side ``jax_callable`` plumbing.  The real per-family wrappers
# (e.g. ``TestJaxCellListAutograd`` below) route through the same
# ``_route_pair_outputs``.
# ---------------------------------------------------------------------------


class TestJaxAutograd:
    """End-to-end checks for the JAX autograd primitive.

    Forward is a pure-JAX builder (no Warp) so the test runs on any device
    where JAX is available.  The same primitive is what the per-family
    wrappers will route through once wired.
    """

    @pytest.fixture(autouse=True)
    def _enable_x64(self):
        jax.config.update("jax_enable_x64", True)

    @staticmethod
    def _pure_jax_pair_forward(positions, cell, *, cutoff=1.5, M=4):
        """Per-pair (d, r) plus integer indices, computed entirely in JAX.

        Provides the same NamedTuple contract as the wrapper forward closure
        will, once landed.  Distances/vectors are functions of positions and
        cell; the integer indices/shifts are constants from the autograd
        perspective.
        """
        from nvalchemiops.jax.neighbors._autograd import (
            _build_index_residuals,
            _NeighborForwardOutput,
        )

        N = positions.shape[0]
        diffs = positions[:, None, :] - positions[None, :, :]
        dist = jnp.linalg.norm(diffs, axis=-1)
        within = (dist < cutoff) & (dist > 0)
        sort_keys = -within.astype(jnp.float32) * (1e8 - dist)
        sorted_j = jnp.argsort(sort_keys, axis=1)[:, :M].astype(jnp.int32)
        nn = within.sum(axis=1).astype(jnp.int32).clip(max=M)
        col_idx = jnp.arange(M, dtype=jnp.int32)
        active_mask = col_idx[None, :] < nn[:, None]
        nm = jnp.where(active_mask, sorted_j, jnp.int32(N))
        shifts = jnp.zeros((N, M, 3), dtype=jnp.int32)
        pos_padded = jnp.concatenate(
            [positions, jnp.zeros((1, 3), dtype=positions.dtype)],
            axis=0,
        )
        j_safe = jnp.where(active_mask, nm, jnp.int32(N))
        r_full = pos_padded[j_safe] - positions[:, None, :]
        d_full = jnp.linalg.norm(r_full, axis=-1)
        r_full = jnp.where(active_mask[:, :, None], r_full, 0.0)
        d_full = jnp.where(active_mask, d_full, 0.0)
        i_idx, j_idx, shifts_ret, batch_idx, mask_ = _build_index_residuals(
            nm,
            nn,
            shifts,
        )
        return _NeighborForwardOutput(
            distances=d_full,
            vectors=r_full,
            extra_outputs=(nm, nn, shifts),
            i_idx=i_idx,
            j_idx=j_idx,
            shifts=shifts_ret,
            batch_idx=batch_idx,
            active_mask=mask_,
            matrix_shape=(N, M),
        )

    def _make_system(self):
        key = jax.random.key(0)
        positions = jax.random.normal(key, (6, 3), dtype=jnp.float64) * 0.3
        cell = (jnp.eye(3, dtype=jnp.float64) * 4.0)[None]
        return positions, cell

    def test_forward_returns_differentiable_outputs(self):
        from nvalchemiops.jax.neighbors._autograd import _route_pair_outputs

        positions, cell = self._make_system()
        d, v, nm, nn, shifts = _route_pair_outputs(
            positions,
            cell,
            self._pure_jax_pair_forward,
            {"cutoff": 1.5, "M": 4},
        )
        assert d.shape == (6, 4)
        assert v.shape == (6, 4, 3)
        assert jnp.isfinite(d).all() and jnp.isfinite(v).all()

    def test_check_grads_first_order_positions(self):
        from jax.test_util import check_grads

        from nvalchemiops.jax.neighbors._autograd import _route_pair_outputs

        positions, cell = self._make_system()

        def loss(p):
            d, *_ = _route_pair_outputs(
                p,
                cell,
                self._pure_jax_pair_forward,
                {"cutoff": 1.5, "M": 4},
            )
            return d.sum()

        check_grads(loss, (positions,), order=1, atol=1e-5, rtol=1e-5, modes=["rev"])

    def test_check_grads_first_order_cell(self):
        from jax.test_util import check_grads

        from nvalchemiops.jax.neighbors._autograd import _route_pair_outputs

        positions, cell = self._make_system()

        def loss(c):
            d, *_ = _route_pair_outputs(
                positions,
                c,
                self._pure_jax_pair_forward,
                {"cutoff": 1.5, "M": 4},
            )
            return d.sum()

        check_grads(loss, (cell,), order=1, atol=1e-5, rtol=1e-5, modes=["rev"])

    def test_check_grads_second_order(self):
        """Reverse-mode second-order autograd via reconstruction in bwd."""
        from jax.test_util import check_grads

        from nvalchemiops.jax.neighbors._autograd import _route_pair_outputs

        positions, cell = self._make_system()

        def loss(p):
            d, *_ = _route_pair_outputs(
                p,
                cell,
                self._pure_jax_pair_forward,
                {"cutoff": 1.5, "M": 4},
            )
            # Non-linear in d so the second derivative is non-trivial.
            return (d**2).sum()

        check_grads(loss, (positions,), order=2, atol=1e-3, rtol=1e-3, modes=["rev"])

    def test_vectors_gradient(self):
        from jax.test_util import check_grads

        from nvalchemiops.jax.neighbors._autograd import _route_pair_outputs

        positions, cell = self._make_system()

        def loss(p):
            _, v, *_ = _route_pair_outputs(
                p,
                cell,
                self._pure_jax_pair_forward,
                {"cutoff": 1.5, "M": 4},
            )
            return v.sum()

        check_grads(loss, (positions,), order=1, atol=1e-5, rtol=1e-5, modes=["rev"])

    def test_jit_grad_compatibility(self):
        """jax.jit(jax.grad(loss)) must compile and return finite grads."""
        from nvalchemiops.jax.neighbors._autograd import _route_pair_outputs

        positions, cell = self._make_system()

        def loss(p):
            d, *_ = _route_pair_outputs(
                p,
                cell,
                self._pure_jax_pair_forward,
                {"cutoff": 1.5, "M": 4},
            )
            return d.sum()

        grad_jit = jax.jit(jax.grad(loss))(positions)
        assert grad_jit.shape == positions.shape
        assert jnp.isfinite(grad_jit).all()

    def test_no_grad_path_unchanged(self):
        """Calling the route helper without jax.grad returns plain arrays
        numerically equal to running the forward closure directly."""
        from nvalchemiops.jax.neighbors._autograd import _route_pair_outputs

        positions, cell = self._make_system()
        d_direct = self._pure_jax_pair_forward(
            positions,
            cell,
            cutoff=1.5,
            M=4,
        ).distances
        d_routed, *_ = _route_pair_outputs(
            positions,
            cell,
            self._pure_jax_pair_forward,
            {"cutoff": 1.5, "M": 4},
        )
        assert jnp.allclose(d_direct, d_routed)


class TestJaxCellListAutograd:
    """End-to-end autograd through the real JAX ``cell_list`` wrapper.

    These tests exercise the full pair-output path: the wrapper runs the
    warp ``build_cell_list`` and pair-output query kernels (which determine the
    neighbour topology) and routes through ``_route_pair_outputs``, which
    reconstructs the differentiable geometry in pure JAX via
    ``_reconstruct_pair_geometry``.
    """

    @pytest.fixture(autouse=True)
    def _enable_x64(self):
        jax.config.update("jax_enable_x64", True)

    def _make_system(self):
        key = jax.random.key(0)
        positions = jax.random.normal(key, (6, 3), dtype=jnp.float64) * 0.3
        cell = (jnp.eye(3, dtype=jnp.float64) * 4.0)[None]
        pbc = jnp.array([[True, True, True]])
        return positions, cell, pbc

    def test_forward_returns_distances_and_vectors(self):
        positions, cell, pbc = self._make_system()
        out = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            return_distances=True,
            return_vectors=True,
        )
        assert len(out) == 5
        nm, nn, shifts, d, v = out
        assert d.shape == (6, nm.shape[1])
        assert v.shape == (6, nm.shape[1], 3)
        assert jnp.isfinite(d).all() and jnp.isfinite(v).all()

    def test_coo_distances_vectors_aligned(self):
        """``return_neighbor_list=True`` repacks per-pair geometry into COO
        order that index-aligns with the neighbor list."""
        positions, cell, pbc = self._make_system()
        nl, _nptr, nl_shifts, d, v = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            return_neighbor_list=True,
            return_distances=True,
            return_vectors=True,
        )
        num_pairs = nl.shape[1]
        assert d.shape == (num_pairs,)
        assert v.shape == (num_pairs, 3)
        i_idx, j_idx = nl[0], nl[1]
        rij = positions[j_idx] - positions[i_idx] + nl_shifts @ cell[0]
        assert jnp.allclose(rij, v, atol=1e-6)
        assert jnp.allclose(jnp.linalg.norm(rij, axis=1), d, atol=1e-6)

    def test_hvp_nonlinear_loss_matches_analytic(self):
        """Regression: the HVP of a loss *nonlinear in distance* matches the exact
        analytic Hessian through the real ``cell_list`` wrapper.

        ``test_check_grads_second_order`` only exercised the pure-JAX forward (which
        always reconstructed live), so it never caught the detached-distance
        higher-order bug on the real binding; this does.
        """
        from .conftest import analytic_distance_sq_hvp

        positions, cell, pbc = self._make_system()
        v = jax.random.normal(jax.random.key(1), positions.shape, dtype=positions.dtype)

        def loss(p):
            *_, d, _ = cell_list(
                p, 1.5, cell, pbc, return_distances=True, return_vectors=True
            )
            return (d**2).sum()

        hvp = np.asarray(jax.grad(lambda p: jnp.vdot(jax.grad(loss)(p), v))(positions))
        nl, *_ = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            return_neighbor_list=True,
            return_distances=True,
            return_vectors=True,
        )
        hvp_true = analytic_distance_sq_hvp(nl, v, positions.shape[0])
        assert nl.shape[1] > 0
        assert np.allclose(hvp, hvp_true, atol=1e-9, rtol=1e-9)

    def test_return_tuple_shape_extends_with_flags(self):
        """0.3.1-compat: tuple stays at 3 elements when flags off."""
        positions, cell, pbc = self._make_system()
        out_default = cell_list(positions, 1.5, cell, pbc)
        assert len(out_default) == 3

        out_d = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            return_distances=True,
        )
        assert len(out_d) == 4

        out_v = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            return_vectors=True,
        )
        assert len(out_v) == 4

        out_dv = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            return_distances=True,
            return_vectors=True,
        )
        assert len(out_dv) == 5

    def test_grad_positions_finite(self):
        positions, cell, pbc = self._make_system()

        def loss(p):
            *_, d, _ = cell_list(
                p,
                1.5,
                cell,
                pbc,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        grad_pos = jax.grad(loss)(positions)
        assert grad_pos.shape == positions.shape
        assert jnp.isfinite(grad_pos).all()

    def test_grad_cell_finite(self):
        positions, cell, pbc = self._make_system()

        def loss(c):
            *_, d, _ = cell_list(
                positions,
                1.5,
                c,
                pbc,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        grad_cell = jax.grad(loss)(cell)
        assert grad_cell.shape == cell.shape
        assert jnp.isfinite(grad_cell).all()

    def test_check_grads_against_finite_differences(self):
        """jax.test_util.check_grads compares analytical vs FD gradient."""
        from jax.test_util import check_grads

        positions, cell, pbc = self._make_system()

        def loss(p):
            *_, d, _ = cell_list(
                p,
                1.5,
                cell,
                pbc,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        check_grads(loss, (positions,), order=1, atol=1e-4, rtol=1e-4, modes=["rev"])

    def test_hessian_vector_product_smoke(self):
        """Second-order autograd: HVP runs and stays finite.

        See TestJaxNaiveAutograd.test_hessian_vector_product_smoke for
        why we prefer HVP over ``check_grads(order=2)``.
        """
        positions, cell, pbc = self._make_system()
        v = jax.random.normal(jax.random.key(1), positions.shape, dtype=positions.dtype)

        def loss(p):
            *_, d, _ = cell_list(
                p,
                1.5,
                cell,
                pbc,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        hvp = jax.grad(lambda p: jnp.vdot(jax.grad(loss)(p), v))(positions)
        assert jnp.isfinite(hvp).all().item()
        assert hvp.shape == positions.shape

    def test_target_indices_partial_matches_full_restricted(self):
        """``target_indices`` (partial neighbor lists) is wired (task 5).

        The compact output has ``num_targets`` rows (row ``r`` -> atom
        ``target_indices[r]``); each row's neighbor set must equal the full
        matrix restricted to that atom.  COO source index ``nl[0]`` is the
        compact row in ``[0, num_targets)`` (matches the torch contract)."""
        positions, cell, pbc = self._make_system()
        n = positions.shape[0]
        targets = jnp.array([0, 2, 4], dtype=jnp.int32)
        nt = int(targets.shape[0])
        mn = 32

        # Partial matrix.
        pnm, pnn, _pnms = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            max_neighbors=mn,
            target_indices=targets,
            fill_value=n,
        )
        assert pnm.shape == (nt, mn) and pnn.shape == (nt,)

        # Full matrix reference, restricted to the target rows.
        fnm, fnn, _fnms = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            max_neighbors=mn,
            fill_value=n,
        )
        pnm, pnn, fnm, fnn, tg = (np.asarray(x) for x in (pnm, pnn, fnm, fnn, targets))

        def row_set(nm, count):
            return {int(nm[k]) for k in range(int(count))}

        for r in range(nt):
            assert row_set(pnm[r], pnn[r]) == row_set(fnm[int(tg[r])], fnn[int(tg[r])])

        # COO: compact-row source index, matching torch.
        nl, _nptr, _nls = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            max_neighbors=mn,
            target_indices=targets,
            return_neighbor_list=True,
        )
        nl = np.asarray(nl)
        if nl.shape[1] > 0:
            assert int(nl[0].max()) < nt

    def test_graph_mode_warp_rejected_with_pair_outputs(self):
        """graph_mode='warp' with pair outputs raises (CUDA-graph follow-up)."""
        positions, cell, pbc = self._make_system()
        with pytest.raises(NotImplementedError, match="graph_mode='none'"):
            cell_list(
                positions,
                1.5,
                cell,
                pbc,
                return_distances=True,
                graph_mode="warp",
            )
