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

"""API tests for the generic JAX neighbor_list wrapper function."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nvalchemiops.jax.neighbors as neighbor_module
from nvalchemiops.jax.neighbors import (
    batch_naive_neighbor_list_dual_cutoff,
    cell_list,
    estimate_neighbor_list_costs,
    naive_neighbor_list,
    naive_neighbor_list_dual_cutoff,
    neighbor_list,
    suggest_neighbor_list_method,
)
from nvalchemiops.neighbors.base_dispatch import neighbor_list_strategy_run_args

from .conftest import create_batch_idx_and_ptr_jax, requires_gpu

pytestmark = requires_gpu

# ==============================================================================
# Helpers
# ==============================================================================


def create_random_system_jax(
    num_atoms: int,
    cell_size: float,
    dtype=jnp.float32,
    seed: int = 42,
):
    """Create a random system with JAX arrays (positions inside a box)."""
    key = jax.random.PRNGKey(seed)
    positions = jax.random.uniform(key, (num_atoms, 3), dtype=dtype) * cell_size
    cell = (jnp.eye(3, dtype=dtype) * cell_size).reshape(1, 3, 3)
    pbc = jnp.array([[True, True, True]])
    return (
        positions,
        cell,
        pbc,
    )


def assert_neighbor_matrix_equal_jax(result1, result2):
    """Assert that two JAX neighbor matrix results are equivalent.

    Compares num_neighbors exactly and neighbor_matrix rows after sorting.
    """
    if len(result1) == 2:
        nm1, nn1 = result1
        nm2, nn2 = result2
        shifts1 = shifts2 = None
    elif len(result1) == 3:
        nm1, nn1, shifts1 = result1
        nm2, nn2, shifts2 = result2
    else:
        raise ValueError(f"Unexpected result length: {len(result1)}")

    # num_neighbors must match exactly
    np.testing.assert_array_equal(np.asarray(nn1), np.asarray(nn2))

    # Compare rows of neighbor_matrix after sorting
    nm1_np = np.asarray(nm1)
    nm2_np = np.asarray(nm2)
    assert nm1_np.shape == nm2_np.shape, (
        f"Neighbor matrix shapes differ: {nm1_np.shape} vs {nm2_np.shape}"
    )
    for i in range(nm1_np.shape[0]):
        np.testing.assert_array_equal(np.sort(nm1_np[i]), np.sort(nm2_np[i]))

    if shifts1 is not None and shifts2 is not None:
        s1_np = np.asarray(shifts1)
        s2_np = np.asarray(shifts2)
        assert s1_np.shape == s2_np.shape, (
            f"Shifts shapes differ: {s1_np.shape} vs {s2_np.shape}"
        )


# ==============================================================================
# Tests: Auto-Selection
# ==============================================================================


class TestNeighborListAutoSelection:
    """Test automatic method selection based on system size."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_auto_select_cell_list_sparse_no_cell(self, dtype, device):
        """Cell-less auto dispatch is correct at method-dependent COO arity.

        With no input cell the cost model picks naive (2-tuple, non-periodic) or
        cell_list (synthesizes a non-PBC cell, so a 3-tuple with zeroed shifts).
        Whichever it picks, the pair count must match the naive reference
        (method choice must not change results).
        """
        target_density = 0.25
        num_atoms = 100
        volume = num_atoms / target_density
        box_size = volume ** (1 / 3)

        key = jax.random.PRNGKey(42)
        positions = jax.random.uniform(key, (num_atoms, 3), dtype=dtype) * box_size
        cutoff = 2.0

        result = neighbor_list(positions, cutoff, return_neighbor_list=True)

        # Cell-less COO arity is method-dependent (naive -> 2-tuple, cell_list ->
        # 3-tuple). The COO list/ptr always live at [0]/[1]; shifts appear only
        # in the 3-tuple cell_list case.
        assert len(result) in (2, 3)
        neighbor_list_coo, neighbor_ptr = result[0], result[1]
        assert neighbor_list_coo.shape[0] == 2  # COO format
        assert neighbor_ptr.shape[0] == num_atoms + 1
        assert int(neighbor_ptr[0]) == 0
        if len(result) == 3:
            assert result[2].shape[1] == 3  # shifts present only for cell_list

        # Method choice must not change results: same pair count as naive.
        naive_ptr = neighbor_list(
            positions, cutoff, return_neighbor_list=True, method="naive"
        )[1]
        assert int(neighbor_ptr[-1]) == int(naive_ptr[-1])

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_auto_select_cell_list_sparse_with_pbc(self, dtype, device):
        """Auto-select for small sparse systems with PBC → 3-tuple."""
        positions, cell, pbc = create_random_system_jax(100, 10.0, dtype=dtype)
        cutoff = 2.0

        result = neighbor_list(
            positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=True
        )

        # With PBC → 3-tuple (neighbor_list_coo, neighbor_ptr, shifts)
        assert len(result) == 3
        neighbor_list_coo, neighbor_ptr, shifts = result
        assert neighbor_list_coo.shape[0] == 2
        assert neighbor_ptr.shape[0] == 101
        assert int(neighbor_ptr[0]) == 0
        assert shifts.shape[1] == 3

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_auto_select_cell_list_large_sparse_system(
        self, dtype, device, monkeypatch
    ):
        """Auto-select cell_list for large sparse systems without an input cell.

        With no input cell the wrapper synthesizes a non-PBC cell, so the COO
        result includes shifts (3-tuple).
        """

        def fake_cell_list(positions, cutoff, *args, **kwargs):
            del cutoff, args, kwargs
            return (
                jnp.empty((2, 0), dtype=jnp.int32),
                jnp.zeros((positions.shape[0] + 1,), dtype=jnp.int32),
                jnp.empty((0, 3), dtype=jnp.int32),
            )

        monkeypatch.setattr(neighbor_module, "cell_list", fake_cell_list)
        key = jax.random.PRNGKey(0)
        positions = jax.random.normal(key, (2000, 3), dtype=dtype) * 50.0
        cutoff = 2.0

        result = neighbor_list(positions, cutoff, return_neighbor_list=True)

        # Sparse geometry auto-selects cell_list, which synthesizes a non-PBC
        # cell and therefore returns shifts.
        assert len(result) == 3
        neighbor_list_coo, neighbor_ptr, shifts = result
        assert neighbor_list_coo.shape[0] == 2
        assert neighbor_ptr.shape[0] == 2001
        assert int(neighbor_ptr[0]) == 0
        assert shifts.shape[1] == 3

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_auto_select_naive_dual_cutoff(self, dtype, device):
        """Auto-select naive_dual_cutoff when cutoff2 is provided → 6-tuple with PBC."""
        positions, cell, pbc = create_random_system_jax(100, 10.0, dtype=dtype)
        cutoff1 = 2.5
        cutoff2 = 3.5

        result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            cutoff2=cutoff2,
            max_neighbors1=50,
            max_neighbors2=50,
            return_neighbor_list=True,
        )

        assert len(result) == 6
        nlist1, ptr1, shifts1, nlist2, ptr2, shifts2 = result
        assert nlist1.shape[0] == 2
        assert nlist2.shape[0] == 2
        assert ptr1.shape[0] == 101
        assert ptr2.shape[0] == 101
        assert shifts1.shape[1] == 3
        assert shifts2.shape[1] == 3

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_auto_select_batch_naive(self, dtype, device):
        """Auto-select batch_naive when batch_idx is provided for small system."""
        positions1, cell1, pbc1 = create_random_system_jax(
            50, 10.0, dtype=dtype, seed=42
        )
        positions2, cell2, pbc2 = create_random_system_jax(
            30, 10.0, dtype=dtype, seed=43
        )
        cutoff = 2.0

        positions = jnp.concatenate([positions1, positions2], axis=0)
        cell = jnp.stack([cell1.squeeze(0), cell2.squeeze(0)], axis=0)
        pbc = jnp.stack([pbc1.squeeze(0), pbc2.squeeze(0)], axis=0)

        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax([50, 30])

        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
        )

        # batch_naive with PBC → 3-tuple
        assert len(result) == 3
        nlist, neighbor_ptr, _ = result
        assert nlist.shape[0] == 2
        assert neighbor_ptr.shape[0] == 81  # 50 + 30 + 1
        assert int(neighbor_ptr[0]) == 0

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_auto_select_batch_naive_dual_cutoff(self, dtype, device):
        """Auto-select batch_naive_dual_cutoff when both cutoff2 and batch_idx are provided."""
        positions1, cell1, pbc1 = create_random_system_jax(
            50, 10.0, dtype=dtype, seed=42
        )
        positions2, cell2, pbc2 = create_random_system_jax(
            30, 10.0, dtype=dtype, seed=43
        )

        positions = jnp.concatenate([positions1, positions2], axis=0)
        cell = jnp.stack([cell1.squeeze(0), cell2.squeeze(0)], axis=0)
        pbc = jnp.stack([pbc1.squeeze(0), pbc2.squeeze(0)], axis=0)

        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax([50, 30])

        cutoff1 = 2.5
        cutoff2 = 3.5

        result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cutoff2=cutoff2,
            max_neighbors1=50,
            max_neighbors2=50,
            return_neighbor_list=True,
        )

        assert len(result) == 6
        nlist1, ptr1, shifts1, nlist2, ptr2, shifts2 = result
        assert nlist1.shape[0] == 2
        assert nlist2.shape[0] == 2
        assert ptr1.shape[0] == 81
        assert ptr2.shape[0] == 81
        assert shifts1.shape[1] == 3
        assert shifts2.shape[1] == 3

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_auto_select_batch_ptr_only_no_cell(self, dtype, device):
        """method=None + batch_ptr-only (no batch_idx, no cell) hits the
        ``elif batch_ptr is not None`` branch in jax __init__.py dispatch."""
        key = jax.random.PRNGKey(42)
        positions = jax.random.normal(key, (80, 3), dtype=dtype) * 5.0
        batch_ptr = jnp.array([0, 50, 80], dtype=jnp.int32)
        result = neighbor_list(
            positions,
            cutoff=2.0,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
        )
        # Cell-less batch COO arity is method-dependent (batch_naive -> 2-tuple,
        # batch_cell_list -> 3-tuple). The COO list/ptr always live at [0]/[1].
        assert len(result) in (2, 3)
        nlist, neighbor_ptr = result[0], result[1]
        assert nlist.shape[0] == 2
        assert neighbor_ptr.shape[0] == 81

        # Method choice must not change results: same pair count as batch_naive
        # (guards the synthesized per-system cell on this previously-faulting path).
        naive_ptr = neighbor_list(
            positions,
            cutoff=2.0,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
            method="batch_naive",
        )[1]
        assert int(neighbor_ptr[-1]) == int(naive_ptr[-1])

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_auto_select_batch_idx_only_no_cell(self, dtype, device):
        """method=None + batch_idx-only (no batch_ptr, no cell) hits the
        ``elif batch_idx is not None`` branch in jax __init__.py dispatch."""
        key = jax.random.PRNGKey(43)
        positions = jax.random.normal(key, (80, 3), dtype=dtype) * 5.0
        batch_idx = jnp.concatenate(
            [
                jnp.zeros(50, dtype=jnp.int32),
                jnp.ones(30, dtype=jnp.int32),
            ]
        )
        result = neighbor_list(
            positions,
            cutoff=2.0,
            batch_idx=batch_idx,
            return_neighbor_list=True,
        )
        # Cell-less batch COO arity is method-dependent (batch_naive -> 2-tuple,
        # batch_cell_list -> 3-tuple). The COO list/ptr always live at [0]/[1].
        assert len(result) in (2, 3)
        nlist, neighbor_ptr = result[0], result[1]
        assert nlist.shape[0] == 2
        assert neighbor_ptr.shape[0] == 81

        # Method choice must not change results: same pair count as batch_naive
        # (guards the synthesized per-system cell on this previously-faulting path).
        naive_ptr = neighbor_list(
            positions,
            cutoff=2.0,
            batch_idx=batch_idx,
            return_neighbor_list=True,
            method="batch_naive",
        )[1]
        assert int(neighbor_ptr[-1]) == int(naive_ptr[-1])

    @pytest.mark.parametrize("dtype", [jnp.float32])
    def test_auto_select_batch_cell_list_large(self, dtype, device, monkeypatch):
        """method=None + large avg_atoms + batched input hits the
        ``method = 'batch_cell_list'`` dispatch branch."""

        def fake_batch_cell_list(positions, cutoff, *args, **kwargs):
            del cutoff, args, kwargs
            return (
                jnp.empty((2, 0), dtype=jnp.int32),
                jnp.zeros((positions.shape[0] + 1,), dtype=jnp.int32),
                jnp.empty((0, 3), dtype=jnp.int32),
            )

        monkeypatch.setattr(neighbor_module, "batch_cell_list", fake_batch_cell_list)
        key = jax.random.PRNGKey(44)
        positions = jax.random.normal(key, (5000, 3), dtype=dtype) * 50.0
        batch_ptr = jnp.array([0, 2500, 5000], dtype=jnp.int32)
        cell = jnp.eye(3, dtype=dtype).reshape(1, 3, 3).repeat(2, axis=0) * 60.0
        pbc = jnp.array([[True, True, True], [True, True, True]])
        result = neighbor_list(
            positions,
            cutoff=2.0,
            cell=cell,
            pbc=pbc,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
        )
        # batch_cell_list with PBC → 3-tuple
        assert len(result) == 3
        nlist, neighbor_ptr, _ = result
        assert nlist.shape[0] == 2
        assert neighbor_ptr.shape[0] == 5001


# ==============================================================================
# Tests: Explicit Method
# ==============================================================================


class TestNeighborListExplicitMethod:
    """Test explicit method selection matches direct calls."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_explicit_naive(self, dtype, device):
        """Test explicit naive method matches direct naive_neighbor_list call."""
        positions, cell, pbc = create_random_system_jax(100, 10.0, dtype=dtype)
        cutoff = 2.0

        wrapper_result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            return_neighbor_list=False,
        )

        direct_result = naive_neighbor_list(
            positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=False
        )

        assert len(wrapper_result) == len(direct_result)
        assert_neighbor_matrix_equal_jax(wrapper_result, direct_result)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_explicit_cell_list(self, dtype, device):
        """Test explicit cell_list method matches direct cell_list call."""
        positions, cell, pbc = create_random_system_jax(500, 20.0, dtype=dtype)
        cutoff = 2.0

        wrapper_result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="cell_list",
            return_neighbor_list=False,
        )

        direct_result = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=False
        )

        assert len(wrapper_result) == len(direct_result)
        assert_neighbor_matrix_equal_jax(wrapper_result, direct_result)

    @pytest.mark.parametrize(
        ("method", "expected_route", "expected_options"),
        [
            ("naive", "batch_naive", {"strategy": "auto"}),
            ("cell_list", "batch_cell_list", {"strategy": "auto"}),
            ("cluster_tile", "batch_cluster_tile", {}),
            ("naive_dual_cutoff", "batch_naive_dual_cutoff", {}),
            ("naive_scalar", "batch_naive", {"strategy": "scalar"}),
            ("naive_tile", "batch_naive", {"strategy": "tile"}),
            (
                "cell_list_atom_centric",
                "batch_cell_list",
                {"strategy": "atom_centric", "atom_centric_path": "direct"},
            ),
            (
                "cell_list_pair_centric",
                "batch_cell_list",
                {"strategy": "pair_centric", "atom_centric_path": "sorted"},
            ),
        ],
    )
    @pytest.mark.parametrize("batch_arg", ["batch_idx", "batch_ptr", "both"])
    def test_explicit_unbatched_method_promotes_with_batch_metadata(
        self, method, expected_route, expected_options, batch_arg, monkeypatch
    ):
        """Explicit single-system methods route to their batch equivalents."""
        positions = jnp.zeros((6, 3), dtype=jnp.float32)
        cell = jnp.repeat(jnp.eye(3, dtype=jnp.float32)[None, :, :] * 8.0, 2, axis=0)
        pbc = jnp.ones((2, 3), dtype=bool)
        kwargs = {
            "batch_idx": jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32),
            "batch_ptr": jnp.array([0, 3, 6], dtype=jnp.int32),
        }
        if batch_arg == "batch_idx":
            kwargs.pop("batch_ptr")
        elif batch_arg == "batch_ptr":
            kwargs.pop("batch_idx")

        seen = {}

        def fake_batch_naive(*args, **call_kwargs):
            seen["route"] = "batch_naive"
            seen["kwargs"] = call_kwargs
            return "batch_naive"

        def fake_batch_cell_list(*args, **call_kwargs):
            seen["route"] = "batch_cell_list"
            seen["args"] = args
            seen["kwargs"] = call_kwargs
            return "batch_cell_list"

        def fake_batch_cluster_tile(*args, **call_kwargs):
            seen["route"] = "batch_cluster_tile"
            seen["args"] = args
            seen["kwargs"] = call_kwargs
            return "batch_cluster_tile"

        def fake_batch_naive_dual_cutoff(*args, **call_kwargs):
            seen["route"] = "batch_naive_dual_cutoff"
            seen["kwargs"] = call_kwargs
            return "batch_naive_dual_cutoff"

        monkeypatch.setattr(
            neighbor_module, "batch_naive_neighbor_list", fake_batch_naive
        )
        monkeypatch.setattr(neighbor_module, "batch_cell_list", fake_batch_cell_list)
        monkeypatch.setattr(
            neighbor_module,
            "batch_cluster_tile_neighbor_list",
            fake_batch_cluster_tile,
        )
        monkeypatch.setattr(
            neighbor_module,
            "batch_naive_neighbor_list_dual_cutoff",
            fake_batch_naive_dual_cutoff,
        )

        result = neighbor_list(
            positions,
            2.0,
            cutoff2=3.0 if method == "naive_dual_cutoff" else None,
            cell=cell,
            pbc=pbc,
            method=method,
            **kwargs,
        )

        assert result == expected_route
        assert seen["route"] == expected_route
        for key, expected in expected_options.items():
            assert seen["kwargs"][key] == expected

    @pytest.mark.parametrize("method", ["naive", "cell_list"])
    def test_explicit_unbatched_method_without_batch_metadata_stays_unbatched(
        self, method, monkeypatch
    ):
        """A 3D cell alone is not batch metadata for explicit methods."""
        positions = jnp.zeros((6, 3), dtype=jnp.float32)
        cell = jnp.repeat(jnp.eye(3, dtype=jnp.float32)[None, :, :] * 8.0, 2, axis=0)
        pbc = jnp.zeros((2, 3), dtype=bool)

        def fake_unbatched(*args, **kwargs):
            return "unbatched"

        def fail_batched(*args, **kwargs):
            raise AssertionError("batch method should not be selected")

        monkeypatch.setattr(neighbor_module, "batch_naive_neighbor_list", fail_batched)
        monkeypatch.setattr(neighbor_module, "batch_cell_list", fail_batched)
        monkeypatch.setattr(neighbor_module, "naive_neighbor_list", fake_unbatched)
        monkeypatch.setattr(neighbor_module, "cell_list", fake_unbatched)

        assert (
            neighbor_list(positions, 2.0, cell=cell, pbc=pbc, method=method)
            == "unbatched"
        )

    @pytest.mark.parametrize(
        ("method", "expected_route"),
        [("naive", "batch_naive"), ("cell_list", "batch_cell_list")],
    )
    @pytest.mark.parametrize("batch_arg", ["batch_idx", "batch_ptr", "both"])
    def test_explicit_unbatched_method_promotes_single_system_batch_metadata(
        self, method, expected_route, batch_arg, monkeypatch
    ):
        """Even one-system batch metadata is still explicit batch metadata."""
        positions = jnp.zeros((6, 3), dtype=jnp.float32)
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 8.0
        pbc = jnp.zeros((1, 3), dtype=bool)
        kwargs = {
            "batch_idx": jnp.zeros(6, dtype=jnp.int32),
            "batch_ptr": jnp.array([0, 6], dtype=jnp.int32),
        }
        if batch_arg == "batch_idx":
            kwargs.pop("batch_ptr")
        elif batch_arg == "batch_ptr":
            kwargs.pop("batch_idx")

        def fake_batch_naive(*args, **call_kwargs):
            return "batch_naive"

        def fake_batch_cell_list(*args, **call_kwargs):
            return "batch_cell_list"

        monkeypatch.setattr(
            neighbor_module, "batch_naive_neighbor_list", fake_batch_naive
        )
        monkeypatch.setattr(neighbor_module, "batch_cell_list", fake_batch_cell_list)

        assert (
            neighbor_list(positions, 2.0, cell=cell, pbc=pbc, method=method, **kwargs)
            == expected_route
        )

    def test_invalid_method_with_batch_metadata_stays_invalid(self):
        """Unknown method names are not promoted into generated batch names."""
        positions = jnp.zeros((6, 3), dtype=jnp.float32)
        batch_idx = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)

        with pytest.raises(ValueError, match="Invalid method"):
            neighbor_list(positions, 2.0, method="not_a_method", batch_idx=batch_idx)

    def test_promoted_naive_does_not_cross_batch_boundaries(self):
        """Promoted naive honors batch boundaries for overlapping coordinates."""
        molecule = jnp.array(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            dtype=jnp.float32,
        )
        positions = jnp.concatenate([molecule, molecule], axis=0)
        batch_idx = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 3, 6], dtype=jnp.int32)

        pairs, _ptr = neighbor_list(
            positions,
            1.1,
            method="naive",
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
            max_neighbors=4,
        )

        assert pairs.shape[1] == 8
        np.testing.assert_array_equal(
            np.asarray(batch_idx[pairs[0]]), np.asarray(batch_idx[pairs[1]])
        )

    def test_method_none_and_batch_method_accept_batch_metadata(self):
        """Auto and explicit batch methods accept batch metadata."""
        positions = jnp.zeros((6, 3), dtype=jnp.float32)
        cell = jnp.repeat(jnp.eye(3, dtype=jnp.float32)[None, :, :], 2, axis=0)
        pbc = jnp.zeros((2, 3), dtype=bool)
        batch_idx = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 3, 6], dtype=jnp.int32)

        auto_result = neighbor_list(
            positions,
            2.0,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=8,
        )
        batch_result = neighbor_list(
            positions,
            2.0,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_naive",
            max_neighbors=8,
        )

        assert auto_result[0].shape == (6, 8)
        assert batch_result[0].shape == (6, 8)


# ==============================================================================
# Tests: Fine-Grained Method Equivalence
# ==============================================================================


def _canonical_pairs(coo_result):
    """Return the undirected ``(i, j, shift)`` set from a COO neighbor-list result.

    Takes the full ``(neighbor_list, neighbor_ptr, shifts)`` tuple and keys each
    pair on the lower-index endpoint, negating the periodic shift when the
    endpoints are swapped — so the set is invariant to which atom "owns" the
    pair (half-fill owner conventions differ between kernels) while still
    distinguishing different periodic images of the same atom pair.
    """
    coo = np.asarray(coo_result[0])
    src, dst = coo[0], coo[1]
    shifts = np.asarray(coo_result[2])
    out = set()
    for i, j, s in zip(src, dst, shifts):
        if int(i) <= int(j):
            out.add((int(i), int(j), int(s[0]), int(s[1]), int(s[2])))
        else:
            out.add((int(j), int(i), -int(s[0]), -int(s[1]), -int(s[2])))
    return out


class TestNeighborListFineGrainedMethodEquivalence:
    """Fine-grained ``method=`` names run the requested kernel and match base.

    JAX now honors fine-grained sub-options: ``cell_list_pair_centric`` runs
    the pair-centric kernel and ``naive_tile`` runs the tiled kernel (via
    ``jax_callable``).  These are performance variants with identical pair
    sets, so the fine-grained name must produce the same pair SET as the
    corresponding base method.
    """

    def _periodic_system(self, dtype):
        key = jax.random.PRNGKey(42)
        positions = jax.random.uniform(key, (64, 3), dtype=dtype) * 20.0
        cell = (jnp.eye(3, dtype=dtype) * 20.0).reshape(1, 3, 3)
        pbc = jnp.array([[True, True, True]])
        return positions, cell, pbc

    @pytest.mark.parametrize("dtype", [jnp.float32])
    @pytest.mark.parametrize(
        "method, base",
        [
            ("naive_scalar", "naive"),
            ("naive_tile", "naive"),
            ("cell_list_atom_centric", "cell_list"),
            ("cell_list_pair_centric", "cell_list"),
        ],
    )
    def test_fine_grained_name_matches_base(self, method, base, dtype):
        """Fine-grained name runs the kernel and matches the base (i, j, shift) set."""
        positions, cell, pbc = self._periodic_system(dtype)
        cutoff = 5.0

        base_res = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method=base,
            return_neighbor_list=True,
        )
        fine_res = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method=method,
            return_neighbor_list=True,
        )
        assert _canonical_pairs(fine_res) == _canonical_pairs(base_res)

    def test_suggest_result_roundtrips_through_method(self):
        """A name from ``suggest_neighbor_list_method`` runs as ``method=`` and
        matches the base method's (i, j, shift) set (claim-1 honesty)."""
        positions, cell, pbc = self._periodic_system(jnp.float32)
        cutoff = 5.0
        batch_ptr = jnp.array([0, positions.shape[0]], dtype=jnp.int32)

        name = suggest_neighbor_list_method(batch_ptr, cell, pbc, cutoff)
        base = neighbor_list_strategy_run_args(name)[0]
        suggested_res = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method=name,
            return_neighbor_list=True,
        )
        base_res = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method=base,
            return_neighbor_list=True,
        )
        assert _canonical_pairs(suggested_res) == _canonical_pairs(base_res)


class TestNeighborListCellListHalfFillFillValue:
    """JAX cell_list now honors ``half_fill`` and ``fill_value`` (parity)."""

    def _periodic_float32_system(self):
        key = jax.random.PRNGKey(3)
        positions = jax.random.uniform(key, (32, 3), dtype=jnp.float32) * 18.0
        cell = (jnp.eye(3, dtype=jnp.float32) * 18.0).reshape(1, 3, 3)
        pbc = jnp.array([[True, True, True]])
        return positions, cell, pbc

    def test_explicit_cell_list_half_fill_matches_naive_half(self):
        """method='cell_list', half_fill=True: half pair set == naive half."""
        positions, cell, pbc = self._periodic_float32_system()
        cutoff = 5.0

        cl_full = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="cell_list",
            return_neighbor_list=True,
        )[1]
        cl_half = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="cell_list",
            half_fill=True,
            return_neighbor_list=True,
        )
        nv_half = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            half_fill=True,
            return_neighbor_list=True,
        )
        assert int(cl_half[1][-1]) * 2 == int(cl_full[-1])
        assert _canonical_pairs(cl_half) == _canonical_pairs(nv_half)

    def test_auto_dispatch_half_fill_returns_half(self):
        """method=None with half_fill must not raise and must return the half list."""
        positions, cell, pbc = self._periodic_float32_system()
        cutoff = 5.0

        full = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            return_neighbor_list=True,
        )[1]
        half = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            half_fill=True,
            return_neighbor_list=True,
        )[1]
        assert int(half[-1]) * 2 == int(full[-1])

    def test_explicit_cell_list_fill_value_remaps_matrix_tail(self):
        """method='cell_list', matrix format: a custom fill_value fills the tail."""
        positions, cell, pbc = self._periodic_float32_system()
        cutoff = 5.0

        matrix = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="cell_list",
            fill_value=-1,
        )[0]
        nm = np.asarray(matrix)
        assert (nm == -1).any()
        assert not (nm == positions.shape[0]).any()


class TestNeighborListPairOutputAndExplicitStrategy:
    """Pair-output (return_distances/vectors) half_fill/fill_value contracts and
    the explicit-vs-auto pair_centric + half_fill behavior."""

    def _periodic_float32_system(self):
        key = jax.random.PRNGKey(5)
        positions = jax.random.uniform(key, (32, 3), dtype=jnp.float32) * 18.0
        cell = (jnp.eye(3, dtype=jnp.float32) * 18.0).reshape(1, 3, 3)
        pbc = jnp.array([[True, True, True]])
        return positions, cell, pbc

    def test_half_fill_with_pair_outputs(self):
        """half_fill now combines with the JAX cell-list pair-output path; each
        emitted pair is self-consistent (``|vec| == dist``)."""
        positions, cell, pbc = self._periodic_float32_system()
        nm, _nn, _sh, dist, vec = neighbor_list(
            positions,
            5.0,
            cell=cell,
            pbc=pbc,
            method="cell_list",
            half_fill=True,
            return_distances=True,
            return_vectors=True,
        )
        active = np.asarray(nm) != positions.shape[0]
        assert int(active.sum()) > 0
        d = np.asarray(dist)[active]
        v = np.asarray(vec)[active]
        assert np.all(d <= 5.0 + 1e-4)
        np.testing.assert_allclose(d, np.linalg.norm(v, axis=-1), atol=1e-5, rtol=1e-5)

    def test_fill_value_remapped_with_pair_outputs(self):
        """A custom fill_value fills the matrix tail even on the pair-output path."""
        positions, cell, pbc = self._periodic_float32_system()
        out = neighbor_list(
            positions,
            5.0,
            cell=cell,
            pbc=pbc,
            method="cell_list",
            fill_value=-1,
            return_distances=True,
        )
        nm = np.asarray(out[0])
        assert (nm == -1).any()
        assert not (nm == positions.shape[0]).any()

    def test_explicit_pair_centric_half_fill_raises(self):
        """Explicit cell_list_pair_centric + half_fill raises (full-fill only)."""
        positions, cell, pbc = self._periodic_float32_system()
        with pytest.raises(NotImplementedError, match="pair.?centric|half_fill"):
            neighbor_list(
                positions,
                5.0,
                cell=cell,
                pbc=pbc,
                method="cell_list_pair_centric",
                half_fill=True,
            )


class TestNeighborListBatchStrategyParity:
    """Batched fine-grained strategies run the requested kernel and match base
    (perf-only variants -> identical pair sets).  GPU-gated module-wide."""

    def _batched_periodic(self, dtype):
        key = jax.random.PRNGKey(11)
        positions = jax.random.uniform(key, (64, 3), dtype=dtype) * 16.0
        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax([28, 36])
        cell = jnp.broadcast_to(jnp.eye(3, dtype=dtype) * 16.0, (2, 3, 3))
        pbc = jnp.ones((2, 3), dtype=jnp.bool_)
        return positions, cell, pbc, batch_idx, batch_ptr

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_batch_naive_tile_matches_batch_naive(self, dtype):
        positions, cell, pbc, batch_idx, batch_ptr = self._batched_periodic(dtype)
        base = neighbor_list(
            positions,
            5.0,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_naive",
            return_neighbor_list=True,
        )
        tile = neighbor_list(
            positions,
            5.0,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_naive_tile",
            return_neighbor_list=True,
        )
        assert _canonical_pairs(tile) == _canonical_pairs(base)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_batch_cell_list_pair_centric_matches_atom_centric(self, dtype):
        positions, cell, pbc, batch_idx, batch_ptr = self._batched_periodic(dtype)
        base = neighbor_list(
            positions,
            5.0,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_cell_list_atom_centric",
            return_neighbor_list=True,
        )
        pc = neighbor_list(
            positions,
            5.0,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_cell_list_pair_centric",
            return_neighbor_list=True,
        )
        assert _canonical_pairs(pc) == _canonical_pairs(base)

    def test_batch_cell_list_half_fill_with_pair_outputs(self):
        """C (batch): half_fill now combines with the pair-output path."""
        positions, cell, pbc, batch_idx, batch_ptr = self._batched_periodic(jnp.float32)
        nm, _nn, _sh, dist, vec = neighbor_list(
            positions,
            5.0,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_cell_list",
            half_fill=True,
            return_distances=True,
            return_vectors=True,
        )
        active = np.asarray(nm) != positions.shape[0]
        assert int(active.sum()) > 0
        d = np.asarray(dist)[active]
        v = np.asarray(vec)[active]
        assert np.all(d <= 5.0 + 1e-4)
        np.testing.assert_allclose(d, np.linalg.norm(v, axis=-1), atol=1e-5, rtol=1e-5)

    def test_batch_cell_list_fill_value_remapped_with_pair_outputs(self):
        """D (batch): a custom fill_value fills the matrix tail on the pair-output path."""
        positions, cell, pbc, batch_idx, batch_ptr = self._batched_periodic(jnp.float32)
        out = neighbor_list(
            positions,
            5.0,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_cell_list",
            fill_value=-1,
            return_distances=True,
        )
        nm = np.asarray(out[0])
        assert (nm == -1).any()
        assert not (nm == positions.shape[0]).any()


# ==============================================================================
# Tests: B1/B2 Regressions
# ==============================================================================


class TestNeighborListNoCellRegressions:
    """B1/B2 regression guards for ``cell=None`` dispatch."""

    def test_explicit_cell_list_no_cell_nonperiodic_matches_naive(self):
        """B1: explicit ``cell_list`` with ``cell=None``/``pbc=None`` on a
        spread-out non-periodic system yields the same pair count as ``naive``
        (no spurious periodic wrap pairs from a synthesized box)."""
        key = jax.random.PRNGKey(7)
        positions = jax.random.normal(key, (32, 3), dtype=jnp.float32) * 15.0
        cutoff = 3.0

        cell_ptr = neighbor_list(
            positions, cutoff, method="cell_list", return_neighbor_list=True
        )[1]
        naive_ptr = neighbor_list(
            positions, cutoff, method="naive", return_neighbor_list=True
        )[1]
        assert int(cell_ptr[-1]) == int(naive_ptr[-1])

    def test_empty_positions_no_cell_returns_empty(self):
        """B2: ``(0, 3)`` positions with ``cell=None`` returns empty outputs."""
        positions = jnp.zeros((0, 3), dtype=jnp.float32)
        cutoff = 2.0

        result = neighbor_list(positions, cutoff, return_neighbor_list=True)

        neighbor_list_coo, neighbor_ptr = result[0], result[1]
        assert neighbor_list_coo.shape[1] == 0
        assert neighbor_ptr.shape[0] == 1
        assert int(neighbor_ptr[0]) == 0


# ==============================================================================
# Tests: Dual Cutoff
# ==============================================================================


class TestNeighborListDualCutoff:
    """Test dual cutoff functionality."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_naive_dual_cutoff(self, dtype, device):
        """Test explicit naive dual cutoff matches direct call."""
        positions, cell, pbc = create_random_system_jax(100, 10.0, dtype=dtype)
        cutoff1 = 2.5
        cutoff2 = 3.5

        wrapper_result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            cutoff2=cutoff2,
            method="naive_dual_cutoff",
            return_neighbor_list=False,
        )

        direct_result = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            cell=cell,
            pbc=pbc,
            return_neighbor_list=False,
        )

        assert len(wrapper_result) == 6
        assert len(direct_result) == 6

        # Compare first cutoff results
        assert_neighbor_matrix_equal_jax(wrapper_result[:3], direct_result[:3])
        # Compare second cutoff results
        assert_neighbor_matrix_equal_jax(wrapper_result[3:], direct_result[3:])

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_batch_naive_dual_cutoff(self, dtype, device):
        """Test batch naive dual cutoff matches direct call."""
        positions1, cell1, pbc1 = create_random_system_jax(
            50, 10.0, dtype=dtype, seed=42
        )
        positions2, cell2, pbc2 = create_random_system_jax(
            30, 10.0, dtype=dtype, seed=43
        )

        positions = jnp.concatenate([positions1, positions2], axis=0)
        cell = jnp.stack([cell1.squeeze(0), cell2.squeeze(0)], axis=0)
        pbc = jnp.stack([pbc1.squeeze(0), pbc2.squeeze(0)], axis=0)

        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax([50, 30])

        cutoff1 = 2.5
        cutoff2 = 3.5

        wrapper_result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cutoff2=cutoff2,
            method="batch_naive_dual_cutoff",
            max_neighbors1=50,
            max_neighbors2=50,
            return_neighbor_list=False,
        )

        direct_result = batch_naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=50,
            max_neighbors2=50,
            return_neighbor_list=False,
        )

        assert len(wrapper_result) == 6
        assert len(direct_result) == 6

        assert_neighbor_matrix_equal_jax(wrapper_result[:3], direct_result[:3])
        assert_neighbor_matrix_equal_jax(wrapper_result[3:], direct_result[3:])


# ==============================================================================
# Tests: Return Formats
# ==============================================================================


class TestNeighborListReturnFormats:
    """Test different return formats (matrix vs COO list)."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_return_neighbor_matrix(self, dtype, device):
        """Test returning neighbor matrix (default) has correct shapes."""
        positions, cell, pbc = create_random_system_jax(100, 10.0, dtype=dtype)
        cutoff = 5.0

        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            return_neighbor_list=False,
        )

        neighbor_matrix, num_neighbors, shifts = result

        assert neighbor_matrix.ndim == 2
        assert neighbor_matrix.shape[0] == 100
        assert num_neighbors.shape[0] == 100
        assert shifts.ndim == 3
        assert shifts.shape[0] == 100

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_return_neighbor_list_coo(self, dtype, device):
        """Test returning neighbor list in COO format has correct shapes."""
        positions, cell, pbc = create_random_system_jax(100, 10.0, dtype=dtype)
        cutoff = 5.0

        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            return_neighbor_list=True,
        )

        neighbor_list_coo, neighbor_ptr, shifts = result

        assert neighbor_list_coo.shape[0] == 2  # [sources, targets]
        assert neighbor_ptr.shape[0] == 101  # total_atoms + 1
        assert shifts.ndim == 2
        assert shifts.shape[1] == 3


# ==============================================================================
# Tests: Half Fill
# ==============================================================================


class TestNeighborListHalfFill:
    """Test half_fill parameter forwarding to naive method."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("half_fill", [False, True])
    def test_half_fill_parameter(self, dtype, device, half_fill):
        """Test that half_fill parameter is forwarded correctly."""
        positions, cell, pbc = create_random_system_jax(50, 10.0, dtype=dtype)
        cutoff = 5.0

        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            half_fill=half_fill,
            return_neighbor_list=True,
        )

        neighbor_list_coo, _, _ = result

        if half_fill:
            # Each pair should appear only once: no (i,j) and (j,i)
            sources = np.asarray(neighbor_list_coo[0])
            targets = np.asarray(neighbor_list_coo[1])
            pairs = set(zip(sources, targets))
            reverse_pairs = set(zip(targets, sources))
            overlap = pairs.intersection(reverse_pairs)
            assert len(overlap) == 0, "Half-fill should not have reciprocal pairs"


class TestNeighborListClusterTileAutoGuards:
    """``method=None`` selects cluster_tile only when selector guards allow it."""

    def _cluster_tile_eligible_metadata(self):
        batch_ptr = jnp.array([0, 2048], dtype=jnp.int32)
        cell = (jnp.eye(3, dtype=jnp.float32) * 7.0).reshape(1, 3, 3)
        pbc = jnp.array([[True, True, True]])
        return batch_ptr, cell, pbc

    def test_auto_dispatch_cluster_tile_eligible_metadata_selects_cluster_tile(self):
        """Dense periodic float32 metadata crosses the cluster-tile selector gate."""
        batch_ptr, cell, pbc = self._cluster_tile_eligible_metadata()

        assert suggest_neighbor_list_method(batch_ptr, cell, pbc, 3.0) == "cluster_tile"

    def test_auto_dispatch_half_fill_excludes_cluster_tile(self):
        """Half-fill excludes an otherwise cluster-tile-eligible selector input."""
        batch_ptr, cell, pbc = self._cluster_tile_eligible_metadata()

        base_report = estimate_neighbor_list_costs(batch_ptr, cell, pbc, 3.0)
        half_report = estimate_neighbor_list_costs(
            batch_ptr,
            cell,
            pbc,
            3.0,
            optional_outputs=["half_fill"],
        )
        assert "cluster_tile" in [name for name, _ in base_report]
        assert "cluster_tile" not in [name for name, _ in half_report]
        assert (
            suggest_neighbor_list_method(
                batch_ptr,
                cell,
                pbc,
                3.0,
                optional_outputs=["half_fill"],
            )
            != "cluster_tile"
        )


# ==============================================================================
# Tests: No PBC
# ==============================================================================


class TestNeighborListNoPBC:
    """Test neighbor list without periodic boundary conditions."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_no_pbc_naive(self, dtype, device):
        """Test naive without PBC returns 2-tuple (no shifts)."""
        key = jax.random.PRNGKey(42)
        positions = jax.random.normal(key, (100, 3), dtype=dtype) * 5.0
        cutoff = 3.0

        result = neighbor_list(
            positions, cutoff, method="naive", return_neighbor_list=True
        )

        assert len(result) == 2
        neighbor_list_coo, neighbor_ptr = result
        assert neighbor_list_coo.shape[0] == 2
        assert neighbor_ptr.shape[0] == 101


# ==============================================================================
# Tests: Invalid Method
# ==============================================================================


class TestNeighborListInvalidMethod:
    """Test error handling for invalid methods."""

    def test_invalid_method_name(self):
        """Test that invalid method name raises ValueError."""
        positions = jnp.ones((10, 3), dtype=jnp.float32)
        cutoff = 2.0

        with pytest.raises(ValueError, match="Invalid method"):
            neighbor_list(positions, cutoff, method="invalid_method")

    def test_dual_cutoff_without_cutoff2(self):
        """Test that naive_dual_cutoff without cutoff2 raises ValueError."""
        positions = jnp.ones((10, 3), dtype=jnp.float32)
        cutoff = 2.0

        with pytest.raises(ValueError, match="cutoff2 must be provided"):
            neighbor_list(positions, cutoff, method="naive_dual_cutoff")

    def test_batch_dual_cutoff_without_cutoff2(self):
        """Test that batch_naive_dual_cutoff without cutoff2 raises ValueError."""
        positions = jnp.ones((10, 3), dtype=jnp.float32)
        cutoff = 2.0

        with pytest.raises(ValueError, match="cutoff2 must be provided"):
            neighbor_list(positions, cutoff, method="batch_naive_dual_cutoff")


# ==============================================================================
# Tests: Kwargs Forwarding
# ==============================================================================


class TestNeighborListKwargs:
    """Test kwargs passing to underlying methods."""

    def test_kwargs_max_neighbors_naive(self):
        """Test passing max_neighbors kwarg to naive method shapes the matrix."""
        positions, cell, pbc = create_random_system_jax(50, 10.0, dtype=jnp.float32)
        cutoff = 5.0

        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            max_neighbors=20,
            return_neighbor_list=False,
        )

        neighbor_matrix, _, _ = result
        assert neighbor_matrix.shape[1] == 20

    def test_kwargs_max_neighbors_dual_cutoff(self):
        """Test passing max_neighbors1 and max_neighbors2 to dual cutoff."""
        positions, cell, pbc = create_random_system_jax(50, 10.0, dtype=jnp.float32)
        cutoff1 = 2.5
        cutoff2 = 3.5

        result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            cutoff2=cutoff2,
            method="naive_dual_cutoff",
            max_neighbors1=15,
            max_neighbors2=25,
            return_neighbor_list=False,
        )

        nm1, _, _, nm2, _, _ = result
        assert nm1.shape[1] == 15
        assert nm2.shape[1] == 25

    def test_kwargs_forwarded_with_auto_selection(self):
        """Test that kwargs are forwarded correctly with auto method selection."""
        positions, cell, pbc = create_random_system_jax(50, 10.0, dtype=jnp.float32)
        cutoff = 5.0

        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=25,
            return_neighbor_list=False,
        )

        neighbor_matrix, _, _ = result
        assert neighbor_matrix.shape[1] == 25


# ==============================================================================
# Tests: Edge Cases
# ==============================================================================


class TestNeighborListEdgeCases:
    """Test edge cases."""

    def test_single_atom(self):
        """Test with single atom system (no neighbors expected)."""
        positions = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
        cutoff = 2.0

        result = neighbor_list(
            positions, cutoff, method="naive", return_neighbor_list=True
        )

        assert len(result) == 2
        neighbor_list_coo, neighbor_ptr = result
        assert neighbor_list_coo.shape[1] == 0  # No pairs
        assert neighbor_ptr.shape[0] == 2  # 1 atom + 1
        assert int(neighbor_ptr[0]) == 0

    def test_neighbor_list_rejects_short_batch_ptr_length(self):
        """Top-level neighbor_list rejects one-entry batch_ptr."""
        positions = jnp.zeros((0, 3), dtype=jnp.float32)
        batch_ptr = jnp.array([0], dtype=jnp.int32)
        with pytest.raises(ValueError, match="batch_ptr.*length at least 2"):
            neighbor_list(positions, 3.0, batch_ptr=batch_ptr)
