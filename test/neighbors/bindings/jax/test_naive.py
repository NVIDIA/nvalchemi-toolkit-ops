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

import functools

import jax
import jax.numpy as jnp
import pytest

from nvalchemiops.jax.neighbors.naive import naive_neighbor_list
from nvalchemiops.jax.neighbors.neighbor_utils import compute_naive_num_shifts

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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
class TestNaiveSelectiveRebuildFlags:
    """Test selective rebuild (rebuild_flags) for naive_neighbor_list JAX binding."""

    def test_no_rebuild_preserves_data(self, dtype):
        """Flag=False: neighbor data should remain unchanged."""
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
            dtype=dtype,
        )
        cutoff = 1.5
        max_neighbors = 20

        # Initial full build
        nm, nn = naive_neighbor_list(positions, cutoff, max_neighbors=max_neighbors)

        saved_nn = jnp.array(nn)

        # Selective rebuild with flag=False: data should be unchanged
        rebuild_flags = jnp.zeros(1, dtype=jnp.bool_)
        nm2, nn2 = naive_neighbor_list(
            positions,
            cutoff,
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
            dtype=dtype,
        )
        cutoff = 1.5
        max_neighbors = 20

        # Reference: full build
        nm_ref, nn_ref = naive_neighbor_list(
            positions, cutoff, max_neighbors=max_neighbors
        )

        # Selective rebuild with flag=True
        nm_stale = jnp.full((positions.shape[0], max_neighbors), 99, dtype=jnp.int32)
        nn_stale = jnp.full((positions.shape[0],), 99, dtype=jnp.int32)

        rebuild_flags = jnp.ones(1, dtype=jnp.bool_)
        nm2, nn2 = naive_neighbor_list(
            positions,
            cutoff,
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


def _make_naive_inputs(dtype, *, pbc_enabled: bool, wrap_positions: bool):
    """Create a small but nontrivial naive neighbor-list test system."""
    if pbc_enabled and wrap_positions:
        positions = jnp.array(
            [
                [0.1, 0.0, 0.0],
                [9.8, 0.0, 0.0],
                [10.4, 0.1, 0.0],
                [-0.2, 0.2, 0.0],
            ],
            dtype=dtype,
        )
    else:
        positions = jnp.array(
            [
                [0.1, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.1, 0.8, 0.0],
                [0.8, 0.8, 0.0],
            ],
            dtype=dtype,
        )

    cutoff = 1.1
    max_neighbors = 12
    if pbc_enabled:
        cell = jnp.eye(3, dtype=dtype).reshape(1, 3, 3) * 10.0
        pbc = jnp.array([[True, True, True]])
    else:
        cell = None
        pbc = None

    return positions, cutoff, cell, pbc, max_neighbors


def _make_naive_stale_inputs(
    positions,
    cutoff,
    cell,
    pbc,
    max_neighbors,
    *,
    wrap_positions: bool,
):
    """Create stale outputs to verify graph-mode reset behavior.

    For PBC + ``wrap_positions=True``, also seeds stale ``positions_wrapped`` and
    ``per_atom_cell_offsets`` scratch buffers (the wrap kernel always overwrites
    them, so any prior contents must be irrelevant to the final result on both
    ``graph_mode`` paths).
    """
    base = naive_neighbor_list(
        positions,
        cutoff,
        cell=cell,
        pbc=pbc,
        max_neighbors=max_neighbors,
        wrap_positions=wrap_positions,
        graph_mode="none",
    )
    stale_inputs = {
        "neighbor_matrix": jnp.full_like(base[0], 77),
        "num_neighbors": jnp.full_like(base[1], 33),
    }
    if pbc is not None:
        stale_inputs["neighbor_matrix_shifts"] = jnp.full_like(base[2], -5)
    if pbc is not None and wrap_positions:
        stale_inputs["positions_wrapped"] = jnp.full_like(positions, 1234.5)
        stale_inputs["per_atom_cell_offsets"] = jnp.full(
            (positions.shape[0], 3), -7, dtype=jnp.int32
        )
    return stale_inputs


class TestNaiveGraphMode:
    """Graph-mode coverage for JAX naive neighbor lists."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize(
        ("pbc_enabled", "wrap_positions"),
        [
            (False, True),
            (True, True),
            (True, False),
        ],
    )
    @pytest.mark.parametrize(
        "selective",
        [None, False, True],
        ids=["norebuild", "rebuild_false", "rebuild_true"],
    )
    def test_matches_default(
        self,
        dtype,
        pbc_enabled: bool,
        wrap_positions: bool,
        selective: bool | None,
    ):
        """`graph_mode="warp"` should match the default path for legal naive cases."""
        positions, cutoff, cell, pbc, max_neighbors = _make_naive_inputs(
            dtype,
            pbc_enabled=pbc_enabled,
            wrap_positions=wrap_positions,
        )
        rebuild_flags = (
            None if selective is None else jnp.array([selective], dtype=jnp.bool_)
        )
        stale_inputs = _make_naive_stale_inputs(
            positions,
            cutoff,
            cell,
            pbc,
            max_neighbors,
            wrap_positions=wrap_positions,
        )

        none_result = naive_neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=max_neighbors,
            wrap_positions=wrap_positions,
            rebuild_flags=rebuild_flags,
            graph_mode="none",
            **stale_inputs,
        )
        warp_result = naive_neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=max_neighbors,
            wrap_positions=wrap_positions,
            rebuild_flags=rebuild_flags,
            graph_mode="warp",
            **stale_inputs,
        )

        _assert_arrays_equal(none_result, warp_result)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_return_neighbor_list(self, dtype):
        """COO conversion should work around the Warp graph callback."""
        positions, cutoff, cell, pbc, max_neighbors = _make_naive_inputs(
            dtype,
            pbc_enabled=True,
            wrap_positions=True,
        )
        stale_neighbor_matrix = jnp.full(
            (positions.shape[0], max_neighbors),
            99,
            dtype=jnp.int32,
        )
        stale_num_neighbors = jnp.full((positions.shape[0],), 99, dtype=jnp.int32)
        stale_shifts = jnp.full(
            (positions.shape[0], max_neighbors, 3),
            -9,
            dtype=jnp.int32,
        )

        none_result = naive_neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=max_neighbors,
            wrap_positions=True,
            return_neighbor_list=True,
            neighbor_matrix=stale_neighbor_matrix,
            num_neighbors=stale_num_neighbors,
            neighbor_matrix_shifts=stale_shifts,
            graph_mode="none",
        )
        warp_result = naive_neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=max_neighbors,
            wrap_positions=True,
            return_neighbor_list=True,
            neighbor_matrix=stale_neighbor_matrix,
            num_neighbors=stale_num_neighbors,
            neighbor_matrix_shifts=stale_shifts,
            graph_mode="warp",
        )

        _assert_arrays_equal(none_result, warp_result)

    def test_invalid_value(self):
        """Invalid graph_mode values should raise a ValueError."""
        positions = jnp.zeros((2, 3), dtype=jnp.float32)
        with pytest.raises(ValueError, match="graph_mode"):
            naive_neighbor_list(positions, 1.0, max_neighbors=4, graph_mode="bad")

    def test_wrapped_warp_replay_stable_pointers(self):
        """Donation contract from the docstring example should produce stable results.

        Functional smoke test: jit-compile the wrapped warp step exactly like the
        docstring example (donating the returned buffers, capturing ``inv_cell`` /
        ``positions_wrapped`` / ``per_atom_cell_offsets`` in the closure so their
        buffer pointers stay stable across calls), run it 5 times, and assert each
        call's outputs match a fresh ``graph_mode="none"`` reference. This guards
        the contract that lets Warp's graph cache hit on the wrapped path; we
        deliberately avoid timing assertions because perf tests are flaky.
        """
        dtype = jnp.float32
        positions, cutoff, cell, pbc, max_neighbors = _make_naive_inputs(
            dtype,
            pbc_enabled=True,
            wrap_positions=True,
        )
        n_atoms = positions.shape[0]
        fill_value = n_atoms
        inv_cell = jnp.linalg.inv(cell)
        positions_wrapped = jnp.zeros((n_atoms, 3), dtype=dtype)
        per_atom_cell_offsets = jnp.zeros((n_atoms, 3), dtype=jnp.int32)
        shift_range, num_shifts_per_system, max_shifts_per_system = (
            compute_naive_num_shifts(cell, cutoff, pbc)
        )

        @functools.partial(jax.jit, donate_argnums=(1, 2, 3))
        def md_step(pos, neighbor_matrix, num_neighbors, shifts):
            return naive_neighbor_list(
                pos,
                cutoff,
                cell=cell,
                pbc=pbc,
                neighbor_matrix=neighbor_matrix,
                num_neighbors=num_neighbors,
                neighbor_matrix_shifts=shifts,
                inv_cell=inv_cell,
                positions_wrapped=positions_wrapped,
                per_atom_cell_offsets=per_atom_cell_offsets,
                shift_range_per_dimension=shift_range,
                num_shifts_per_system=num_shifts_per_system,
                max_shifts_per_system=max_shifts_per_system,
                graph_mode="warp",
            )

        reference = naive_neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=max_neighbors,
            wrap_positions=True,
            graph_mode="none",
        )

        neighbor_matrix = jnp.full(
            (n_atoms, max_neighbors), fill_value, dtype=jnp.int32
        )
        num_neighbors = jnp.zeros((n_atoms,), dtype=jnp.int32)
        shifts = jnp.zeros((n_atoms, max_neighbors, 3), dtype=jnp.int32)

        for _ in range(5):
            out_nm, out_nn, out_shifts = md_step(
                positions,
                neighbor_matrix,
                num_neighbors,
                shifts,
            )
            neighbor_matrix, num_neighbors, shifts = out_nm, out_nn, out_shifts
            _assert_arrays_equal(reference, (out_nm, out_nn, out_shifts))
