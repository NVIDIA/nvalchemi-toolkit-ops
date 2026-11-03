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

"""Focused tests for the immutable JAX prepared neighbor-list contract."""

from __future__ import annotations

import inspect
from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from nvalchemiops.jax.neighbors import neighbor_list
from nvalchemiops.jax.neighbors.batch_cell_list import (
    batch_cell_list,
    estimate_batch_cell_list_sizes,
)
from nvalchemiops.jax.neighbors.batch_cluster_tile import (
    batch_cluster_tile_neighbor_list,
)
from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list
from nvalchemiops.jax.neighbors.batch_naive_dual_cutoff import (
    batch_naive_neighbor_list_dual_cutoff,
)
from nvalchemiops.jax.neighbors.cell_list import cell_list, estimate_cell_list_sizes
from nvalchemiops.jax.neighbors.cluster_tile import cluster_tile_neighbor_list
from nvalchemiops.jax.neighbors.naive import naive_neighbor_list
from nvalchemiops.jax.neighbors.naive_dual_cutoff import (
    naive_neighbor_list_dual_cutoff,
)
from nvalchemiops.jax.neighbors.neighbor_utils import (
    NeighborOverflowError,
    TileBufferOverflow,
)
from nvalchemiops.jax.neighbors.prepared_neighbor_list import (
    _FIXED_CELL_GEOMETRY_LEAF,
    NeighborListState,
    check_neighbor_list_state,
    prepare_neighbor_list,
)
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

from .conftest import requires_gpu
from .test_cluster_tile import _brute_force_pairs_full, _matrix_to_pair_set_full

pytestmark = requires_gpu


@wp.func
def _prepared_pair_fn(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    """Return simple pair outputs for prepared cluster-route coverage."""
    return pair_params[i, 0] + pair_params[j, 0] + distance, -r_ij


def _cell_inputs(num_atoms: int = 4) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return a small fully periodic single-system geometry."""
    positions = jnp.arange(num_atoms * 3, dtype=jnp.float32).reshape(num_atoms, 3)
    cell = jnp.eye(3, dtype=jnp.float32) * 32.0
    pbc = jnp.ones(3, dtype=jnp.bool_)
    return positions, cell, pbc


def _assert_active_cell_results_equal(
    actual: tuple[jax.Array, ...],
    expected: tuple[jax.Array, ...],
    rows: set[int] | None = None,
) -> None:
    """Compare active cell-list pairs, shifts, counts, and pair geometry."""
    actual_counts, expected_counts = np.asarray(actual[1]), np.asarray(expected[1])
    if rows is None:
        rows = set(range(len(actual_counts)))
    row_indices = sorted(rows)
    np.testing.assert_array_equal(
        actual_counts[row_indices], expected_counts[row_indices]
    )
    actual_pairs = {
        key: value
        for key, value in _matrix_geometry_by_pair(actual).items()
        if key[0] in rows
    }
    expected_pairs = {
        key: value
        for key, value in _matrix_geometry_by_pair(expected).items()
        if key[0] in rows
    }
    assert actual_pairs.keys() == expected_pairs.keys()
    for key in actual_pairs:
        actual_vectors, actual_distances = actual_pairs[key]
        expected_vectors, expected_distances = expected_pairs[key]
        np.testing.assert_allclose(actual_vectors, expected_vectors, atol=3e-6)
        np.testing.assert_allclose(actual_distances, expected_distances, atol=3e-6)


def _assert_active_cell_topology_equal(
    actual: tuple[jax.Array, ...],
    expected: tuple[jax.Array, ...],
    rows: set[int] | None = None,
) -> None:
    """Compare active cell-list pairs, shifts, and counts without pair outputs."""
    actual_matrix, actual_counts = np.asarray(actual[0]), np.asarray(actual[1])
    expected_matrix, expected_counts = np.asarray(expected[0]), np.asarray(expected[1])
    if rows is None:
        rows = set(range(len(actual_counts)))
    row_indices = sorted(rows)
    np.testing.assert_array_equal(
        actual_counts[row_indices], expected_counts[row_indices]
    )

    def pair_keys(
        matrix: np.ndarray, counts: np.ndarray, result: tuple[jax.Array, ...]
    ):
        shifts = np.asarray(result[2]) if len(result) > 2 else None
        keys = set()
        for row in row_indices:
            for slot in range(int(counts[row])):
                shift = (
                    tuple(int(value) for value in shifts[row, slot])
                    if shifts is not None
                    else (0, 0, 0)
                )
                keys.add((row, int(matrix[row, slot]), *shift))
        return keys

    assert pair_keys(actual_matrix, actual_counts, actual) == pair_keys(
        expected_matrix, expected_counts, expected
    )


def test_prepared_state_is_immutable_single_leaf_pytree() -> None:
    """Prepared state has static auxiliary configuration and unique leaves."""
    positions, cell, pbc = _cell_inputs()
    state = prepare_neighbor_list(
        positions,
        2.0,
        cell=cell,
        pbc=pbc,
        method="naive",
        max_neighbors=32,
        selective=True,
    )
    leaves, treedef = jax.tree_util.tree_flatten(state)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, NeighborListState)
    assert len(leaves) == len({id(leaf) for leaf in leaves})
    assert state.selective is True
    with pytest.raises(AttributeError, match="read-only"):
        state.cutoff = 3.0


@pytest.mark.parametrize("fixed_cell", [False, True])
@pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
@pytest.mark.parametrize("strategy", ["atom_centric", "pair_centric"])
def test_prepared_cell_list_grid_policy_dispatch(
    fixed_cell: bool, batched: bool, strategy: str
) -> None:
    """Prepared cell-list states use configured sizing for both geometry modes."""
    positions = jnp.array(
        [[0.1, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    pbc = jnp.zeros((3,), dtype=jnp.bool_)
    method = "cell_list"
    prepare_options: dict[str, Any] = {}
    if batched:
        positions = jnp.concatenate((positions, positions), axis=0)
        prepare_options["batch_ptr"] = jnp.array([0, 3, 6], dtype=jnp.int32)
        cell = jnp.broadcast_to(cell, (2, 3, 3))
        pbc = jnp.broadcast_to(pbc, (2, 3))
        method = "batch_cell_list"
        max_total_cells = estimate_batch_cell_list_sizes(
            positions,
            batch_ptr=prepare_options["batch_ptr"],
            cell=cell,
            pbc=pbc,
            cutoff=0.75,
        )[0]
    else:
        max_total_cells = estimate_cell_list_sizes(positions, cell, 0.75, pbc)[0]
    state = prepare_neighbor_list(
        positions,
        0.75,
        cell=cell,
        pbc=pbc,
        method=method,
        strategy=strategy,
        max_neighbors=4,
        max_total_cells=max_total_cells,
        fixed_cell=fixed_cell,
        **prepare_options,
    )

    default_result, default_successor = neighbor_list(positions, state=state)
    configured_result, configured_successor = neighbor_list(
        positions, state=state, grid_policy="configured"
    )
    for expected, actual in zip(default_result, configured_result, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert default_successor.fixed_cell is fixed_cell
    assert configured_successor.fixed_cell is fixed_cell
    assert bool(jnp.all(default_successor.valid))
    assert bool(jnp.all(configured_successor.valid))
    check_neighbor_list_state(default_successor)
    check_neighbor_list_state(configured_successor)

    with pytest.raises(
        ValueError,
        match="grid_policy='adaptive' is not supported with NeighborListState",
    ):
        neighbor_list(positions, state=state, grid_policy="adaptive")
    with pytest.raises(ValueError, match="grid_policy"):
        neighbor_list(positions, state=state, grid_policy="unknown")


@pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.float64], ids=["float32", "float64"]
)
def test_prepared_fixed_cell_list_coo_matches_direct(
    batched: bool, compiled: bool, dtype: Any
) -> None:
    """Fixed cell-list states preserve fixed-capacity COO outputs."""
    positions = jnp.array(
        [
            [0.1, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.4, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    one_cell = jnp.eye(3, dtype=dtype) * 8.0
    one_pbc = jnp.array([True, False, False], dtype=jnp.bool_)
    batch_ptr = jnp.array([0, 2, 2, 5], dtype=jnp.int32)
    options: dict[str, Any] = {
        "cell": (
            jnp.stack((one_cell, one_cell * 1.5, one_cell * 2.0))
            if batched
            else one_cell
        ),
        "pbc": (
            jnp.array(
                [[True, False, False], [False, False, False], [True, True, False]],
                dtype=jnp.bool_,
            )
            if batched
            else one_pbc
        ),
        "method": "batch_cell_list" if batched else "cell_list",
        "strategy": "atom_centric",
        "max_neighbors": 4,
        "max_total_cells": 256 if batched else 64,
        "return_neighbor_list": True,
        "coo_capacity": 16,
    }
    if batched:
        options["batch_ptr"] = batch_ptr
    state = prepare_neighbor_list(positions, 0.75, fixed_cell=True, **options)
    direct_options = {key: value for key, value in options.items() if key != "method"}
    direct_builder = batch_cell_list if batched else cell_list
    direct = direct_builder(positions, 0.75, **direct_options)

    def execute(values: jax.Array, prepared: NeighborListState):
        return neighbor_list(values, state=prepared)

    compiled_execute = jax.jit(execute) if compiled else execute
    actual, successor = compiled_execute(positions, state)

    assert len(actual) == len(direct) == 5
    np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(direct[1]))
    np.testing.assert_array_equal(np.asarray(actual[3]), np.asarray(direct[3]))
    np.testing.assert_array_equal(np.asarray(actual[4]), np.asarray(direct[4]))

    def coo_records(result: tuple[jax.Array, ...]) -> list[tuple[int, ...]]:
        pairs, shifts = np.asarray(result[0]), np.asarray(result[2])
        return sorted(
            (int(pairs[0, slot]), int(pairs[1, slot]), *map(int, shifts[slot]))
            for slot in range(pairs.shape[1])
            if int(pairs[0, slot]) != positions.shape[0]
        )

    assert coo_records(actual) == coo_records(direct)
    assert successor.fixed_cell
    assert bool(jnp.all(successor.valid))
    check_neighbor_list_state(successor)

    moved_positions = positions.at[1, 0].set(7.8).at[4, 0].set(5.0)
    moved_actual, moved_successor = compiled_execute(moved_positions, successor)
    moved_direct = direct_builder(moved_positions, 0.75, **direct_options)
    np.testing.assert_array_equal(
        np.asarray(moved_actual[1]), np.asarray(moved_direct[1])
    )
    np.testing.assert_array_equal(
        np.asarray(moved_actual[3]), np.asarray(moved_direct[3])
    )
    np.testing.assert_array_equal(
        np.asarray(moved_actual[4]), np.asarray(moved_direct[4])
    )
    assert coo_records(moved_actual) == coo_records(moved_direct)
    assert coo_records(moved_actual) != coo_records(actual)
    assert moved_successor.fixed_cell
    assert bool(jnp.all(moved_successor.valid))
    check_neighbor_list_state(moved_successor)


def test_preparation_rejects_stored_runtime_inputs() -> None:
    """Runtime rebuild flags and pair parameters are not captured at prepare time."""
    positions, cell, pbc = _cell_inputs()
    with pytest.raises(TypeError, match="unexpected prepared"):
        prepare_neighbor_list(
            positions,
            2.0,
            cell=cell,
            pbc=pbc,
            method="naive",
            rebuild_flags=jnp.ones((1,), dtype=jnp.bool_),
        )
    with pytest.raises(ValueError, match="pair_params is a runtime input"):
        prepare_neighbor_list(
            positions,
            2.0,
            cell=cell,
            pbc=pbc,
            method="naive",
            pair_params=jnp.ones((positions.shape[0], 1), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="format contradicts return_neighbor_list"):
        prepare_neighbor_list(
            positions,
            2.0,
            cell=cell,
            pbc=pbc,
            method="naive",
            format="coo",
        )
    with pytest.raises(TypeError, match="unexpected prepared"):
        prepare_neighbor_list(
            positions,
            2.0,
            cell=cell,
            pbc=pbc,
            method="naive",
            max_pairs=16,
        )


def test_prepared_jit_threads_successor_and_tree_round_trip() -> None:
    """A prepared successor can be passed through a jitted public call."""
    positions, cell, pbc = _cell_inputs()
    state = prepare_neighbor_list(
        positions,
        2.0,
        cell=cell,
        pbc=pbc,
        method="naive",
        max_neighbors=32,
    )
    leaves, treedef = jax.tree_util.tree_flatten(state)
    state = jax.tree_util.tree_unflatten(treedef, leaves)

    signature = inspect.signature(neighbor_list)
    assert signature.parameters["cutoff"].default is None
    assert signature.parameters["state"].kind is inspect.Parameter.KEYWORD_ONLY

    def execute(
        current_positions: jax.Array, current_state: NeighborListState
    ) -> tuple[tuple[Any, ...], NeighborListState]:
        return neighbor_list(
            current_positions,
            cell=cell,
            pbc=jnp.zeros_like(pbc),
            method="cluster_tile",
            return_neighbor_list=True,
            state=current_state,
        )

    compiled = jax.jit(execute)
    first, state1 = compiled(positions, state)
    second, state2 = compiled(positions, state1)
    np.testing.assert_array_equal(np.asarray(first[1]), np.asarray(second[1]))
    np.testing.assert_array_equal(np.asarray(state1.valid), np.asarray(state2.valid))
    assert bool(jnp.all(state1.initialized))


def test_prepared_selective_fixed_coo_fit_then_atomic_rollback() -> None:
    """Selective fixed-COO rollback preserves inactive topology and state."""
    close = jnp.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    far = jnp.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    initial = jnp.concatenate((far, close), axis=0)
    changed = jnp.concatenate((close, close), axis=0)
    batch_ptr = jnp.array([0, 4, 8], dtype=jnp.int32)
    batch_idx = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 4)
    cell = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32) * 64.0, (2, 3, 3))
    pbc = jnp.ones((2, 3), dtype=jnp.bool_)
    state = prepare_neighbor_list(
        initial,
        1.0,
        cell=cell,
        pbc=pbc,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        method="batch_naive",
        return_neighbor_list=True,
        coo_capacity=16,
        max_neighbors=8,
        selective=True,
        fixed_cell=True,
    )
    previous, state1 = neighbor_list(
        initial,
        1.0,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        state=state,
        rebuild_flags=jnp.ones((2,), dtype=jnp.bool_),
    )
    assert bool(jnp.all(state1.valid))
    rollback, state2 = neighbor_list(
        changed,
        1.0,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        state=state1,
        rebuild_flags=jnp.array([True, False], dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(np.asarray(previous[0]), np.asarray(rollback[0]))
    np.testing.assert_array_equal(np.asarray(previous[1]), np.asarray(rollback[1]))
    np.testing.assert_array_equal(np.asarray(state2.valid), np.array([False, True]))
    with pytest.raises(NeighborOverflowError, match=r"systems \[0\]"):
        check_neighbor_list_state(state2)
    preserved, state3 = neighbor_list(
        changed,
        state=state2,
        rebuild_flags=jnp.zeros((2,), dtype=jnp.bool_),
    )
    for expected, actual in zip(rollback, preserved, strict=True):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
    np.testing.assert_array_equal(
        np.asarray(state3.initialized), np.array([True, True])
    )
    np.testing.assert_array_equal(np.asarray(state3.valid), np.array([False, True]))
    with pytest.raises(NeighborOverflowError, match=r"systems \[0\]"):
        check_neighbor_list_state(state3)


@pytest.mark.parametrize(
    ("method", "dtype", "offset", "span_margin"),
    [
        ("cell_list", jnp.float32, 100.0, 0.0),
        ("batch_cell_list", jnp.float32, -100.0, 0.2),
        ("cell_list", jnp.float64, -100.0, 0.0),
        ("batch_cell_list", jnp.float64, 100.0, 0.2),
    ],
)
def test_prepared_synthesized_decimal_translation_matches_fresh_route(
    method: str, dtype: Any, offset: float, span_margin: float
) -> None:
    """Translated decimal coordinates match a fresh synthesized cell-list call."""
    base_positions = jnp.array(
        [[0.1, -0.3, 0.15], [1.3, 0.8, 0.47], [0.7, 0.2, -0.25]], dtype=dtype
    )
    positions = (
        jnp.concatenate((base_positions, base_positions + 5.0), axis=0)
        if method == "batch_cell_list"
        else base_positions
    )
    translation = jnp.asarray([offset, -offset, offset], dtype=dtype)
    translated = positions + translation
    options = {
        "method": method,
        "strategy": "atom_centric",
        "max_neighbors": 8,
        "max_total_cells": 128,
        "return_vectors": True,
        "return_distances": True,
    }
    if method == "batch_cell_list":
        options["batch_idx"] = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 3)
        options["batch_ptr"] = jnp.array([0, 3, 6], dtype=jnp.int32)
    state = prepare_neighbor_list(positions, 2.0, span_margin=span_margin, **options)
    execute = jax.jit(lambda values, prepared: neighbor_list(values, state=prepared))
    actual, successor = execute(translated, state)
    assert bool(jnp.all(successor.valid))
    reused, successor2 = execute(translated, successor)
    assert bool(jnp.all(successor2.valid))
    direct_options = {key: value for key, value in options.items() if key != "strategy"}
    expected = neighbor_list(translated, 2.0, **direct_options)
    assert int(jnp.sum(actual[1])) > 0
    _assert_active_cell_results_equal(actual, expected)
    _assert_active_cell_results_equal(reused, expected)


@pytest.mark.parametrize(
    ("method", "dtype", "offset", "span_margin"),
    [
        ("cell_list", jnp.float32, 100.0, 0.0),
        ("batch_cell_list", jnp.float32, -100.0, 0.2),
        ("cell_list", jnp.float64, -100.0, 0.0),
        ("batch_cell_list", jnp.float64, 100.0, 0.2),
    ],
)
def test_prepared_synthesized_span_accepts_tiny_real_expansion(
    method: str, dtype: Any, offset: float, span_margin: float
) -> None:
    """A one-step represented expansion beyond capacity stays within tolerance."""
    base_positions = jnp.array(
        [[0.1, -0.3, 0.15], [1.3, 0.8, 0.47], [0.7, 0.2, -0.25]], dtype=dtype
    )
    positions = (
        jnp.concatenate((base_positions, base_positions + 5.0), axis=0)
        if method == "batch_cell_list"
        else base_positions
    )
    translation = jnp.asarray([offset, -offset, offset], dtype=dtype)
    translated = positions + translation
    exemplar_span = jnp.max(base_positions, axis=0) - jnp.min(base_positions, axis=0)
    span_capacity = exemplar_span + jnp.asarray(span_margin, dtype=dtype)
    translated_minimum = jnp.min(translated[:3, 0])
    boundary = translated_minimum + span_capacity[0]
    expanded_maximum = jnp.nextafter(boundary, jnp.asarray(jnp.inf, dtype=dtype))
    expanded = translated.at[1, 0].set(expanded_maximum)
    current_span = jnp.max(expanded[:3, 0]) - jnp.min(expanded[:3, 0])
    cell_length = span_capacity[0] + jnp.asarray(0.2, dtype=dtype)
    assert bool(current_span > span_capacity[0])
    assert bool(current_span <= cell_length)

    options = {
        "method": method,
        "strategy": "atom_centric",
        "max_neighbors": 8,
        "max_total_cells": 128,
        "return_vectors": True,
        "return_distances": True,
    }
    if method == "batch_cell_list":
        options["batch_idx"] = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 3)
        options["batch_ptr"] = jnp.array([0, 3, 6], dtype=jnp.int32)
    state = prepare_neighbor_list(positions, 2.0, span_margin=span_margin, **options)
    execute = jax.jit(lambda values, prepared: neighbor_list(values, state=prepared))
    actual, successor = execute(expanded, state)
    assert bool(jnp.all(successor.valid))
    direct_options = {key: value for key, value in options.items() if key != "strategy"}
    expected = neighbor_list(expanded, 2.0, **direct_options)
    assert int(jnp.sum(actual[1])) > 0
    _assert_active_cell_results_equal(actual, expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_prepared_synthesized_span_skips_unselected_growth_and_latches_selected(
    dtype: Any,
) -> None:
    """Selective span checks ignore an unselected grown system and stay sticky."""
    positions = jnp.array(
        [
            [0.1, -0.3, 0.15],
            [1.3, 0.8, 0.47],
            [5.1, 0.2, -0.25],
            [6.4, 0.7, 0.3],
        ],
        dtype=dtype,
    )
    batch_ptr = jnp.array([0, 2, 2, 4], dtype=jnp.int32)
    batch_idx = jnp.array([0, 0, 2, 2], dtype=jnp.int32)
    options = {
        "batch_idx": batch_idx,
        "batch_ptr": batch_ptr,
        "method": "batch_cell_list",
        "strategy": "atom_centric",
        "max_neighbors": 8,
        "max_total_cells": 128,
        "selective": True,
    }
    state = prepare_neighbor_list(positions, 2.0, **options)
    execute = jax.jit(
        lambda values, prepared, flags: neighbor_list(
            values, state=prepared, rebuild_flags=flags
        )
    )
    all_systems = jnp.ones((3,), dtype=jnp.bool_)
    initial_result, initialized = execute(positions, state, all_systems)
    translated_selected = positions.at[:2, :].add(
        jnp.asarray([100.0, -100.0, 100.0], dtype=dtype)
    )
    grown_unselected = translated_selected.at[3, 0].add(0.5)
    selected_systems = jnp.array([True, True, False], dtype=jnp.bool_)
    actual, successor = execute(grown_unselected, initialized, selected_systems)
    assert successor.valid.tolist() == [True, True, True]
    repeated, successor2 = execute(grown_unselected, successor, selected_systems)
    assert successor2.valid.tolist() == [True, True, True]
    direct_options = {
        key: value
        for key, value in options.items()
        if key not in {"selective", "strategy"}
    }
    expected = neighbor_list(grown_unselected, 2.0, **direct_options)
    assert int(jnp.sum(actual[1])) > 0
    _assert_active_cell_topology_equal(actual, expected, rows={0, 1})
    _assert_active_cell_topology_equal(actual, initial_result, rows={2, 3})
    _assert_active_cell_topology_equal(repeated, expected, rows={0, 1})
    _assert_active_cell_topology_equal(repeated, initial_result, rows={2, 3})

    failed, failed_state = execute(
        grown_unselected,
        successor2,
        jnp.array([False, False, True], dtype=jnp.bool_),
    )
    del failed
    assert failed_state.valid.tolist() == [True, True, False]
    with pytest.raises(
        RuntimeError, match="prepared nonperiodic span exceeded.*system 2"
    ):
        check_neighbor_list_state(failed_state)
    _restored, sticky = execute(positions, failed_state, all_systems)
    assert sticky.valid.tolist() == [True, True, False]
    with pytest.raises(
        RuntimeError, match="prepared nonperiodic span exceeded.*system 2"
    ):
        check_neighbor_list_state(sticky)


def test_prepared_synthesized_span_rejects_over_tolerance_and_cell_bounds() -> None:
    """Real growth beyond tolerance or the represented cell bound stays sticky."""
    positions = jnp.array(
        [[0.1, -0.3, 0.15], [1.3, 0.8, 0.47], [0.7, 0.2, -0.25]],
        dtype=jnp.float32,
    )
    translated = positions + jnp.array([100.0, -100.0, 100.0], dtype=jnp.float32)
    exemplar_span = jnp.max(positions[:, 0]) - jnp.min(positions[:, 0])
    span_capacity = exemplar_span + jnp.asarray(0.2, dtype=jnp.float32)
    cell_length = span_capacity + jnp.asarray(0.075, dtype=jnp.float32)
    translated_minimum = jnp.min(translated[:, 0])
    options = {
        "method": "cell_list",
        "strategy": "atom_centric",
        "max_neighbors": 8,
        "max_total_cells": 128,
    }
    state = prepare_neighbor_list(positions, 0.75, span_margin=0.2, **options)
    execute = jax.jit(lambda values, prepared: neighbor_list(values, state=prepared))

    for increment, crosses_cell_bound in ((0.04, False), (0.08, True)):
        expanded_maximum = (
            translated_minimum
            + span_capacity
            + jnp.asarray(increment, dtype=jnp.float32)
        )
        expanded = translated.at[1, 0].set(expanded_maximum)
        current_span = jnp.max(expanded[:, 0]) - jnp.min(expanded[:, 0])
        assert bool(current_span > cell_length) is crosses_cell_bound
        _result, failed = execute(expanded, state)
        assert failed.valid.tolist() == [False]
        with pytest.raises(RuntimeError, match="prepared nonperiodic span exceeded"):
            check_neighbor_list_state(failed)
        _restored, sticky = execute(positions, failed)
        assert sticky.valid.tolist() == [False]
        with pytest.raises(RuntimeError, match="prepared nonperiodic span exceeded"):
            check_neighbor_list_state(sticky)


def test_prepared_synthesized_span_is_translation_invariant_and_selected() -> None:
    """Only selected systems are span-checked, and translation is free."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    batch_idx = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 2)
    state = prepare_neighbor_list(
        positions,
        1.5,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        method="batch_cell_list",
        max_neighbors=8,
        span_margin=0.5,
        selective=True,
    )
    _result, state = neighbor_list(
        positions,
        1.5,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        state=state,
        rebuild_flags=jnp.ones((2,), dtype=jnp.bool_),
    )
    translated = positions + jnp.array([100.0, -20.0, 3.0], dtype=jnp.float32)
    _result, translated_state = neighbor_list(
        translated,
        1.5,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        state=state,
        rebuild_flags=jnp.array([True, False], dtype=jnp.bool_),
    )
    assert bool(jnp.all(translated_state.valid))
    expanded = translated.at[1, 0].add(1.0)
    _result, failed_state = neighbor_list(
        expanded,
        1.5,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        state=translated_state,
        rebuild_flags=jnp.array([True, False], dtype=jnp.bool_),
    )
    assert bool(failed_state.valid[0]) is False
    assert bool(failed_state.valid[1]) is True


@pytest.mark.parametrize("return_neighbor_list", [False, True])
def test_prepared_synthesized_cell_maps_shift_results(return_neighbor_list) -> None:
    """Synthesized cell routes map every returned shift array to state."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [1.2, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    options = {
        "method": "cell_list",
        "max_neighbors": 8,
        "max_total_cells": 8,
        "return_neighbor_list": return_neighbor_list,
    }
    if return_neighbor_list:
        options["coo_capacity"] = 16
    state = prepare_neighbor_list(positions, 0.75, **options)
    result, successor = jax.jit(
        lambda values, prepared: neighbor_list(values, state=prepared)
    )(positions, state)

    shifts = (
        successor.neighbor_list_shifts
        if return_neighbor_list
        else successor.neighbor_matrix_shifts
    )
    shift_index = 2
    assert shifts is not None
    np.testing.assert_array_equal(np.asarray(shifts), np.asarray(result[shift_index]))


def test_prepared_runtime_cell_does_not_replace_fixed_cell() -> None:
    """A runtime cell applies to one call without replacing prepared geometry."""
    positions = jnp.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=jnp.float32)
    prepared_cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    runtime_cell = jnp.eye(3, dtype=jnp.float32) * 2.5
    pbc = jnp.ones(3, dtype=jnp.bool_)
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=prepared_cell,
        pbc=pbc,
        method="cell_list",
        strategy="atom_centric",
        max_neighbors=128,
        max_total_cells=512,
    )

    compiled = jax.jit(
        lambda values, current_cell, prepared: neighbor_list(
            values, cell=current_cell, state=prepared
        )
    )
    runtime_result, successor = compiled(positions, runtime_cell, state)
    runtime_expected = cell_list(
        positions,
        1.0,
        runtime_cell,
        pbc,
        max_neighbors=128,
        max_total_cells=512,
        strategy="atom_centric",
    )
    for actual, expected in zip(runtime_result, runtime_expected, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    fixed_result, _ = neighbor_list(positions, state=successor)
    fixed_expected = cell_list(
        positions,
        1.0,
        prepared_cell,
        pbc,
        max_neighbors=128,
        max_total_cells=512,
        strategy="atom_centric",
    )
    for actual, expected in zip(fixed_result, fixed_expected, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_prepared_dual_fixed_coo_pbc_named_properties_match_tuple_slots() -> None:
    """Dual fixed-COO PBC outputs keep distinct primary and secondary names."""
    positions = jnp.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    pbc = jnp.ones(3, dtype=jnp.bool_)
    state = prepare_neighbor_list(
        positions,
        0.75,
        cutoff2=1.5,
        cell=cell,
        pbc=pbc,
        method="naive",
        return_neighbor_list=True,
        coo_capacity=(8, 16),
        max_neighbors1=4,
        max_neighbors2=4,
    )
    result, successor = neighbor_list(positions, state=state)
    names = (
        "neighbor_list1",
        "neighbor_ptr1",
        "neighbor_list_shifts1",
        "num_neighbors1",
        "metadata_valid1",
        "neighbor_list2",
        "neighbor_ptr2",
        "neighbor_list_shifts2",
        "num_neighbors2",
        "metadata_valid2",
    )
    assert len(result) == len(names)
    for index, name in enumerate(names):
        value = getattr(successor, name)
        assert value is not None
        np.testing.assert_array_equal(np.asarray(value), np.asarray(result[index]))


def test_prepared_sticky_failure_survives_later_success() -> None:
    """A failed initialized system remains invalid after a later fit rebuild."""
    far = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [110.0, 0.0, 0.0],
            [120.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    close_first = far.at[:3].set(
        jnp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]],
            dtype=jnp.float32,
        )
    )
    batch_ptr = jnp.array([0, 3, 6], dtype=jnp.int32)
    batch_idx = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 3)
    state = prepare_neighbor_list(
        far,
        1.0,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        method="batch_naive",
        max_neighbors=1,
        selective=True,
    )
    initialized_result, initialized = neighbor_list(
        far,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        state=state,
        rebuild_flags=jnp.ones((2,), dtype=jnp.bool_),
    )
    _, failed = neighbor_list(
        close_first,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        state=initialized,
        rebuild_flags=jnp.array([True, False], dtype=jnp.bool_),
    )
    assert failed.initialized.tolist() == [True, True]
    assert failed.valid.tolist() == [False, True]
    _, later = neighbor_list(
        far,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        state=failed,
        rebuild_flags=jnp.array([True, False], dtype=jnp.bool_),
    )
    assert later.valid.tolist() == [False, True]
    with pytest.raises(NeighborOverflowError, match=r"systems \[0\].*segment 0"):
        check_neighbor_list_state(later)


def test_prepared_simultaneous_failures_report_sorted_indices_and_lowest_detail() -> (
    None
):
    """The checker lists every failed system and details the lowest index."""
    close = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.2, 0.0, 0.0],
            [10.4, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    batch_ptr = jnp.array([0, 3, 6], dtype=jnp.int32)
    batch_idx = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 3)
    state = prepare_neighbor_list(
        close,
        1.0,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        method="batch_naive",
        max_neighbors=1,
        selective=True,
    )
    _, failed = neighbor_list(
        close,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        state=state,
        rebuild_flags=jnp.ones((2,), dtype=jnp.bool_),
    )
    assert failed.valid.tolist() == [False, False]
    with pytest.raises(
        NeighborOverflowError,
        match=r"systems \[0, 1\].*segment 0",
    ):
        check_neighbor_list_state(failed)


def test_prepared_dual_cutoff_precedence_secondary_row_beats_primary_coo() -> None:
    """Cross-cutoff status chooses a secondary row failure over primary COO."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    state = prepare_neighbor_list(
        positions,
        0.5,
        cutoff2=1.0,
        method="naive",
        return_neighbor_list=True,
        coo_capacity=(1, 16),
        max_neighbors1=2,
        max_neighbors2=1,
    )
    _, failed = neighbor_list(positions, state=state)
    assert failed.valid.tolist() == [False]
    with pytest.raises(NeighborOverflowError, match=r"2 > 1"):
        check_neighbor_list_state(failed)


def test_cell_metadata_status_is_independent_of_row_and_coo_failures() -> None:
    """Stale metadata remains visible when row and COO capacities also fail."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    cell = jnp.eye(3, dtype=jnp.float32) * 10.0
    diagnostic = cell_list(
        positions,
        1.0,
        cell,
        jnp.zeros(3, dtype=jnp.bool_),
        max_neighbors=1,
        max_total_cells=8,
        strategy="pair_centric",
        pair_centric_n_outer=0,
        return_neighbor_list=True,
        coo_capacity=1,
        _return_status=True,
    )
    tail = diagnostic[-11:]
    assert int(tail[0][0]) == 3
    assert bool(tail[3][0])
    assert bool(tail[1][0])
    assert bool(tail[2][0])


def test_prepared_cell_capacity_keeps_structured_checker_error() -> None:
    """Cell-storage failure remains distinct from periodic-image coverage."""
    positions = jnp.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32) * 10.0
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=jnp.zeros(3, dtype=jnp.bool_),
        method="cell_list",
        strategy="atom_centric",
        max_neighbors=8,
        max_total_cells=1,
    )
    _, failed = neighbor_list(positions, state=state)
    assert failed.valid.tolist() == [False]
    with pytest.raises(NeighborOverflowError, match=r"systems \[0\].*> 1"):
        check_neighbor_list_state(failed)
    _, later = neighbor_list(
        positions, cell=jnp.eye(3, dtype=jnp.float32), state=failed
    )
    assert later.valid.tolist() == [False]
    with pytest.raises(NeighborOverflowError, match=r"systems \[0\].*> 1"):
        check_neighbor_list_state(later)


@pytest.mark.parametrize("method", ["naive", "cell_list"])
def test_prepared_state_donation_allows_successor_reuse_without_aliasing(
    method,
) -> None:
    """A donated state successor can drive the next compiled execution."""
    positions, cell, pbc = _cell_inputs()
    options = (
        {"strategy": "atom_centric", "max_total_cells": 512}
        if method == "cell_list"
        else {}
    )
    state = prepare_neighbor_list(
        positions,
        2.0,
        cell=cell,
        pbc=pbc,
        method=method,
        max_neighbors=32,
        **options,
    )

    def execute(
        current_positions: jax.Array, current_state: NeighborListState
    ) -> tuple[tuple[Any, ...], NeighborListState]:
        return neighbor_list(
            current_positions,
            state=current_state,
        )

    compiled = jax.jit(execute, donate_argnums=(1,))
    first, successor = compiled(positions, state)
    first_counts = np.asarray(first[1])
    second, successor2 = compiled(positions, successor)
    np.testing.assert_array_equal(first_counts, np.asarray(second[1]))
    assert jax.tree.structure(successor) == jax.tree.structure(successor2)
    assert successor2.initialized.tolist() == [True]


def test_prepared_span_boundary_and_selected_overflow() -> None:
    """Exact synthesized span capacity succeeds while selected overflow latches."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 0.0, 0.0], [6.5, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    batch_idx = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 2)
    state = prepare_neighbor_list(
        positions,
        1.0,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        method="batch_cell_list",
        max_neighbors=4,
        span_margin=0.0,
        selective=True,
    )
    _, equal = neighbor_list(
        positions,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        state=state,
        rebuild_flags=jnp.ones((2,), dtype=jnp.bool_),
    )
    assert equal.valid.tolist() == [True, True]
    translated = positions + jnp.array([100.0, -20.0, 3.0], dtype=jnp.float32)
    _, translated_state = neighbor_list(
        translated,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        state=equal,
        rebuild_flags=jnp.array([True, False], dtype=jnp.bool_),
    )
    assert translated_state.valid.tolist() == [True, True]
    overflow = translated.at[1, 0].add(0.1)
    _, failed = neighbor_list(
        overflow,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        state=translated_state,
        rebuild_flags=jnp.array([True, False], dtype=jnp.bool_),
    )
    assert failed.valid.tolist() == [False, True]


@pytest.mark.parametrize("method", ["naive", "batch_naive"])
@pytest.mark.parametrize("dual", [False, True])
def test_prepared_naive_periodic_contraction_latches_coverage_failure(
    method: str, dual: bool
) -> None:
    """A contracted runtime cell cannot exceed prepared periodic-image coverage."""
    positions = jnp.array([[0.02, 0.0, 0.0], [0.12, 0.0, 0.0]], dtype=jnp.float32)
    prepared_cell = jnp.eye(3, dtype=jnp.float32) * 16.0
    runtime_cell = jnp.eye(3, dtype=jnp.float32) * 0.1
    pbc = jnp.ones(3, dtype=jnp.bool_)
    kwargs: dict[str, Any] = {
        "cell": prepared_cell,
        "pbc": pbc,
        "method": method,
        "max_neighbors": 128,
    }
    direct_kwargs: dict[str, Any] = {
        "cell": runtime_cell,
        "pbc": pbc,
        "method": method,
        "max_neighbors": 128,
    }
    if method == "batch_naive":
        batch_ptr = jnp.array([0, positions.shape[0]], dtype=jnp.int32)
        kwargs["batch_ptr"] = batch_ptr
        direct_kwargs["batch_ptr"] = batch_ptr
    if dual:
        kwargs.pop("max_neighbors")
        direct_kwargs.pop("max_neighbors")
        kwargs.update(cutoff2=0.3, max_neighbors1=128, max_neighbors2=128)
        direct_kwargs.update(
            method=f"{method}_dual_cutoff",
            cutoff2=0.3,
            max_neighbors1=128,
            max_neighbors2=128,
        )
    state = prepare_neighbor_list(positions, 0.15, **kwargs)
    direct = neighbor_list(positions, 0.15, **direct_kwargs)

    execute = jax.jit(lambda x, c, s: neighbor_list(x, cell=c, state=s))
    prepared, failed = execute(positions, runtime_cell, state)

    count_index = 4 if dual else 1
    assert int(jnp.sum(direct[count_index])) > 0
    assert prepared[count_index].shape == direct[count_index].shape
    assert not bool(jnp.all(failed.valid))
    with pytest.raises(RuntimeError, match="periodic-image coverage.*cached.*runtime"):
        check_neighbor_list_state(failed)
    _result, sticky = neighbor_list(positions, cell=prepared_cell, state=failed)
    assert not bool(jnp.all(sticky.valid))


@pytest.mark.parametrize("method", ["cluster_tile", "batch_cluster_tile"])
def test_prepared_cluster_rejects_nonzero_span_margin(method: str) -> None:
    """Cluster routes reject span margins while synthesized cell lists accept them."""
    positions, cell, pbc = _cell_inputs(2)
    kwargs: dict[str, Any] = {
        "cell": cell,
        "pbc": pbc,
        "method": method,
        "max_neighbors": 8,
        "max_tiles_per_group": 1,
    }
    if method == "batch_cluster_tile":
        kwargs["cell"] = cell[None]
        kwargs["pbc"] = pbc[None]
        kwargs["batch_ptr"] = jnp.array([0, positions.shape[0]], dtype=jnp.int32)
    prepare_neighbor_list(positions, 1.0, span_margin=0.0, **kwargs)
    with pytest.raises(ValueError, match="span_margin.*cell=None"):
        prepare_neighbor_list(positions, 1.0, span_margin=0.1, **kwargs)


@pytest.mark.parametrize("method", ["cell_list", "batch_cell_list"])
def test_prepared_synthesized_cell_list_accepts_positive_span_margin(
    method: str,
) -> None:
    """Positive span margins remain supported only for synthesized cell lists."""
    positions, _cell, _pbc = _cell_inputs(2)
    kwargs: dict[str, Any] = {"method": method, "max_neighbors": 8}
    if method == "batch_cell_list":
        kwargs["batch_ptr"] = jnp.array([0, positions.shape[0]], dtype=jnp.int32)
    state = prepare_neighbor_list(positions, 1.0, span_margin=0.1, **kwargs)
    assert isinstance(state, NeighborListState)


def test_prepared_cluster_matrix_and_cell_fixed_coo_parse_private_suffixes() -> None:
    """Prepared cluster and cell routes expose only their public prefixes."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]],
        dtype=jnp.float32,
    )
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    pbc = jnp.ones(3, dtype=jnp.bool_)
    cluster = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        method="cluster_tile",
        max_neighbors=8,
        max_tiles_per_group=1,
    )
    cluster_result, cluster_next = neighbor_list(positions, state=cluster)
    assert len(cluster_result) == 3
    assert cluster_next.neighbor_matrix.shape == (4, 8)

    cell_state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=jnp.zeros(3, dtype=jnp.bool_),
        method="cell_list",
        return_neighbor_list=True,
        coo_capacity=16,
        max_neighbors=8,
        max_total_cells=216,
        strategy="atom_centric",
    )
    cell_result, cell_next = neighbor_list(positions, state=cell_state)
    assert len(cell_result) == 5
    assert cell_next.neighbor_list.shape == (2, 16)
    assert bool(cell_next.metadata_valid)


def test_prepared_cluster_tile_output_requires_cluster_and_is_public() -> None:
    """Prepared cluster routes accept tile output without exposing status tails."""
    positions, cell, pbc = _cell_inputs(4)
    state = prepare_neighbor_list(
        positions,
        2.0,
        cell=cell,
        pbc=pbc,
        method="cluster_tile",
        format="tile",
        max_tiles_per_group=1,
    )
    result, successor = neighbor_list(positions, state=state)
    assert len(result) == 7
    assert successor.num_tiles is not None
    assert successor.neighbor_matrix is None

    with pytest.raises(ValueError, match="format contradicts return_neighbor_list"):
        prepare_neighbor_list(
            positions,
            2.0,
            cell=cell,
            pbc=pbc,
            method="cluster_tile",
            format="tile",
            return_neighbor_list=True,
            coo_capacity=16,
        )
    with pytest.raises(ValueError, match="format contradicts return_neighbor_list"):
        prepare_neighbor_list(
            positions,
            2.0,
            method="naive",
            format="tile",
        )


def test_prepared_cluster_tile_capacity_defaults_and_validation() -> None:
    """Cluster tile output uses the default group bound and rejects invalid bounds."""
    positions = jnp.zeros((40, 3), dtype=jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32) * 20.0
    pbc = jnp.ones((3,), dtype=jnp.bool_)

    for max_tiles_per_group in (0, -1):
        with pytest.raises(ValueError, match="max_tiles_per_group must be positive"):
            prepare_neighbor_list(
                positions,
                1.0,
                cell=cell,
                pbc=pbc,
                method="cluster_tile",
                format="tile",
                max_tiles_per_group=max_tiles_per_group,
            )

    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        method="cluster_tile",
        format="tile",
    )
    result, successor = neighbor_list(positions, state=state)
    assert len(result) == 7
    # Forty atoms require two groups. The omitted per-group bound defaults to
    # two, so the tile arrays can hold all four group-pair combinations.
    assert result[1].shape == (4,)
    assert result[2].shape == (4,)
    assert bool(jnp.all(successor.valid))


def test_prepared_cluster_coo_omitted_capacity_preserves_active_geometry() -> None:
    """Omitted cluster COO capacity defaults to N times row width."""
    positions_np, cell_np, cutoff = _fixed_cluster_geometry_case(
        "orthogonal_nonzero_image"
    )
    positions = jnp.asarray(positions_np)
    cell = jnp.asarray(cell_np)
    state = prepare_neighbor_list(
        positions,
        cutoff,
        cell=cell,
        pbc=jnp.ones((3,), dtype=jnp.bool_),
        method="cluster_tile",
        return_neighbor_list=True,
        max_neighbors=1,
        max_tiles_per_group=1,
        return_distances=True,
        return_vectors=True,
    )

    expected_capacity = len(positions_np)
    assert state.coo_layout == "compact"
    result, successor = neighbor_list(positions, state=state)
    assert len(result) == 5
    assert bool(jnp.all(successor.valid))

    active_pairs = int(np.asarray(result[1])[-1])
    pairs, shifts = (
        np.asarray(result[0])[:, :active_pairs],
        np.asarray(result[2])[:active_pairs],
    )
    expected = _fixed_cluster_expected_geometry(positions_np, cell_np, cutoff)
    assert active_pairs == len(expected) == expected_capacity
    np.testing.assert_array_equal(
        np.diff(np.asarray(result[1])),
        np.bincount(pairs[0], minlength=len(positions_np)),
    )
    geometry = {
        (int(pair[0]), int(pair[1]), *(int(value) for value in shift)): (
            np.asarray(result[4])[index].copy(),
            float(np.asarray(result[3])[index]),
        )
        for index, (pair, shift) in enumerate(zip(pairs.T, shifts, strict=True))
    }
    assert geometry.keys() == expected.keys()
    for key, (expected_vector, expected_distance) in expected.items():
        actual_vector, actual_distance = geometry[key]
        np.testing.assert_allclose(actual_vector, expected_vector, atol=3e-5)
        np.testing.assert_allclose(actual_distance, expected_distance, atol=3e-5)


def test_prepared_atomic_density_none_preserves_atom_count_default() -> None:
    """Omitting or explicitly clearing density keeps the atom-count width."""
    positions, _cell, _pbc = _cell_inputs(4)
    default = prepare_neighbor_list(positions, 1.0, method="naive")
    explicit_none = prepare_neighbor_list(
        positions, 1.0, method="naive", atomic_density=None
    )
    assert default.neighbor_matrix.shape == (4, 4)
    assert explicit_none.neighbor_matrix.shape == default.neighbor_matrix.shape


def test_prepared_atomic_density_none_keeps_cluster_default_below_floor() -> None:
    """Without a density hint, cluster widths keep the existing atom-count default."""
    positions, cell, pbc = _cell_inputs(4)
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        method="cluster_tile",
        max_tiles_per_group=1,
    )
    assert state.neighbor_matrix.shape == (4, 4)


@pytest.mark.parametrize(
    ("method", "batched"),
    [
        ("naive", False),
        ("cell_list", False),
        ("batch_naive", True),
        ("batch_cell_list", True),
    ],
    ids=["naive", "cell-list", "batch-naive", "batch-cell-list"],
)
@pytest.mark.parametrize("density", [1, 100.0], ids=["python-int", "python-float"])
def test_prepared_atomic_density_estimates_single_width(
    method: str, batched: bool, density: int | float
) -> None:
    """Each single-cutoff route estimates omitted width and preserves active topology."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [5.0, 0.0, 0.0], [5.4, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32) if batched else None
    batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32) if batched else None
    expected_width = estimate_max_neighbors(1.2, atomic_density=density)
    prepared_options: dict[str, Any] = {"method": method}
    direct_options: dict[str, Any] = {
        "method": method,
        "max_neighbors": expected_width,
    }
    if batched:
        prepared_options.update(batch_idx=batch_idx, batch_ptr=batch_ptr)
        direct_options.update(batch_idx=batch_idx, batch_ptr=batch_ptr)
    if method.endswith("cell_list"):
        prepared_options.update(strategy="atom_centric", max_total_cells=128)
        direct_options["max_total_cells"] = 128

    state = prepare_neighbor_list(
        positions,
        1.2,
        atomic_density=density,
        **prepared_options,
    )
    assert state.neighbor_matrix.shape == (4, expected_width)
    actual, successor = neighbor_list(positions, state=state)
    assert bool(jnp.all(successor.valid))
    assert int(jnp.sum(actual[1])) > 0

    expected = neighbor_list(positions, 1.2, **direct_options)
    _assert_active_cell_topology_equal(actual, expected)


def test_prepared_atomic_density_preserves_explicit_single_width() -> None:
    """The density hint does not replace an explicit single-cutoff width."""
    positions, _cell, _pbc = _cell_inputs(4)
    state = prepare_neighbor_list(
        positions,
        1.2,
        method="naive",
        max_neighbors=7,
        atomic_density=100.0,
    )
    assert state.neighbor_matrix.shape == (4, 7)


@pytest.mark.parametrize("batched", [False, True], ids=["single", "two-system-batch"])
def test_prepared_atomic_density_estimates_dual_widths_per_cutoff(
    batched: bool,
) -> None:
    """Dual primary and secondary widths use their own cutoff estimates."""
    positions, _cell, _pbc = _cell_inputs(4)
    batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32) if batched else None
    state = prepare_neighbor_list(
        positions,
        0.5,
        cutoff2=1.5,
        method="naive_dual_cutoff",
        batch_ptr=batch_ptr,
        atomic_density=4.0,
    )
    primary = estimate_max_neighbors(0.5, atomic_density=4.0)
    secondary = estimate_max_neighbors(1.5, atomic_density=4.0)
    assert state.neighbor_matrix1.shape == (4, primary)
    assert state.neighbor_matrix2.shape == (4, secondary)
    assert state.is_batched is batched
    assert state.num_systems == (2 if batched else 1)


def test_prepared_atomic_density_keeps_dual_cutoff_order_validation() -> None:
    """Density estimation does not relax the existing dual cutoff order."""
    positions, _cell, _pbc = _cell_inputs(4)
    with pytest.raises(ValueError, match="cutoff must not exceed cutoff2"):
        prepare_neighbor_list(
            positions,
            1.5,
            cutoff2=0.5,
            method="naive_dual_cutoff",
            atomic_density=4.0,
        )


@pytest.mark.parametrize(
    ("explicit", "expected"),
    [
        ({"max_neighbors1": 7, "max_neighbors2": 9}, (7, 9)),
        ({"max_neighbors1": 7}, (7, 64)),
        ({"max_neighbors2": 9}, (16, 9)),
    ],
    ids=["both-explicit", "secondary-estimated", "primary-estimated"],
)
def test_prepared_atomic_density_preserves_explicit_dual_widths(
    explicit: dict[str, int], expected: tuple[int, int]
) -> None:
    """Explicit dual widths take precedence while omitted widths are estimated."""
    positions, _cell, _pbc = _cell_inputs(4)
    state = prepare_neighbor_list(
        positions,
        0.5,
        cutoff2=1.5,
        method="naive_dual_cutoff",
        atomic_density=4.0,
        **explicit,
    )
    assert state.neighbor_matrix1.shape == (4, expected[0])
    assert state.neighbor_matrix2.shape == (4, expected[1])


@pytest.mark.parametrize("density", [0.01, 100.0], ids=["floor", "outer-cutoff"])
def test_prepared_atomic_density_cluster_uses_outer_cutoff_and_floor(
    density: float,
) -> None:
    """Dual cluster widths use the outer cutoff and a 32-neighbor estimate floor."""
    positions, cell, pbc = _cell_inputs(4)
    expected = estimate_max_neighbors(
        1.5,
        atomic_density=density,
        max_neighbors_lower_bound=32,
    )
    state = prepare_neighbor_list(
        positions,
        0.5,
        cutoff2=1.5,
        cell=cell,
        pbc=pbc,
        method="cluster_tile",
        atomic_density=density,
        max_tiles_per_group=1,
    )
    assert state.neighbor_matrix1.shape == (4, expected)
    assert state.neighbor_matrix2.shape == (4, expected)


def test_prepared_atomic_density_cluster_keeps_explicit_widths() -> None:
    """The cluster floor and density estimate do not replace explicit widths."""
    positions, cell, pbc = _cell_inputs(4)
    state = prepare_neighbor_list(
        positions,
        0.5,
        cutoff2=1.5,
        cell=cell,
        pbc=pbc,
        method="cluster_tile",
        max_neighbors1=7,
        max_neighbors2=9,
        atomic_density=1e308,
        max_tiles_per_group=1,
    )
    assert state.neighbor_matrix1.shape == (4, 7)
    assert state.neighbor_matrix2.shape == (4, 9)


@pytest.mark.parametrize(
    "density",
    [
        True,
        False,
        0,
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        "0.5",
        1 + 2j,
        np.float64(0.5),
        np.array(0.5),
        jnp.asarray(0.5),
        10**1000,
    ],
    ids=[
        "true",
        "false",
        "zero",
        "negative",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "string",
        "complex",
        "numpy-scalar",
        "numpy-array",
        "jax-array",
        "integer-conversion-overflow",
    ],
)
def test_prepared_atomic_density_rejects_invalid_values_even_with_explicit_width(
    density: Any,
) -> None:
    """Invalid density hints are rejected even when no width needs estimating."""
    positions, _cell, _pbc = _cell_inputs(4)
    with pytest.raises(ValueError, match="atomic_density"):
        prepare_neighbor_list(
            positions,
            1.0,
            method="naive",
            max_neighbors=8,
            atomic_density=density,
        )


def test_prepared_atomic_density_is_not_a_runtime_option() -> None:
    """Density is consumed during preparation and rejected during execution."""
    positions, _cell, _pbc = _cell_inputs(4)
    state = prepare_neighbor_list(positions, 1.0, method="naive")
    with pytest.raises(TypeError, match="unexpected prepared neighbor-list option"):
        neighbor_list(positions, state=state, atomic_density=1.0)


def test_prepared_atomic_density_state_threads_jitted_successors() -> None:
    """A density-sized state executes and threads normally through JIT."""
    positions, _cell, _pbc = _cell_inputs(4)
    state = prepare_neighbor_list(
        positions,
        1.0,
        method="naive",
        atomic_density=1.0,
    )
    compiled = jax.jit(lambda values, prepared: neighbor_list(values, state=prepared))
    first, state1 = compiled(positions, state)
    second, state2 = compiled(positions, state1)
    np.testing.assert_array_equal(np.asarray(first[1]), np.asarray(second[1]))
    assert bool(jnp.all(state1.valid))
    assert bool(jnp.all(state2.valid))


def test_prepared_atomic_density_underestimate_remains_sticky() -> None:
    """An intentionally undersized density estimate keeps sticky overflow status."""
    positions = jnp.arange(18, dtype=jnp.float32)[:, None] * 0.01
    positions = jnp.concatenate(
        (positions, jnp.zeros((18, 2), dtype=jnp.float32)), axis=1
    )
    state = prepare_neighbor_list(
        positions,
        1.0,
        method="naive",
        atomic_density=0.001,
    )
    assert state.neighbor_matrix.shape == (18, 16)
    _, failed = neighbor_list(positions, state=state)
    assert failed.valid.tolist() == [False]
    separated = positions.at[:, 0].set(jnp.arange(18, dtype=jnp.float32) * 2.0)
    _, succeeded = neighbor_list(separated, state=failed)
    assert succeeded.valid.tolist() == [False]
    with pytest.raises(NeighborOverflowError):
        check_neighbor_list_state(succeeded)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("naive", "batch_naive"),
        ("naive_dual_cutoff", "batch_naive_dual_cutoff"),
        ("cell_list", "batch_cell_list"),
        ("cluster_tile", "batch_cluster_tile"),
    ],
)
def test_prepared_explicit_single_method_aliases_batch(method, expected) -> None:
    """Explicit single-system names normalize to one batched binding."""
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.4, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    batch_idx = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 2)
    cell = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32) * 8.0, (2, 3, 3))
    pbc = jnp.ones((2, 3), dtype=jnp.bool_)
    options = {
        "batch_idx": batch_idx,
        "batch_ptr": batch_ptr,
        "method": method,
        "max_neighbors": 8,
    }
    if method == "naive_dual_cutoff":
        options.update(cutoff2=1.5, max_neighbors1=8, max_neighbors2=8)
    elif method == "cell_list":
        options.update(cell=cell, pbc=jnp.zeros((2, 3), dtype=jnp.bool_))
    elif method == "cluster_tile":
        options.update(
            cell=cell,
            pbc=pbc,
            max_tiles_per_group=1,
        )
    state = prepare_neighbor_list(positions, 1.0, **options)
    assert state.method == expected
    result, successor = neighbor_list(positions, state=state)
    assert result
    assert successor.is_batched


def test_prepared_batch_cluster_compact_coo_reports_shared_capacity_failure() -> None:
    """Aggregate compact COO overflow reports total required/global capacity."""
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.4, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    cell = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32) * 8.0, (2, 3, 3))
    pbc = jnp.ones((2, 3), dtype=jnp.bool_)
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_cluster_tile",
        return_neighbor_list=True,
        coo_capacity=2,
        max_neighbors=4,
        max_tiles_per_group=1,
    )
    result, failed = neighbor_list(positions, state=state)
    assert len(result) == 3
    assert not bool(jnp.all(failed.valid))
    with pytest.raises(NeighborOverflowError, match=r"4 > 2"):
        check_neighbor_list_state(failed)


def test_prepared_batch_cluster_compact_coo_names_its_pointer_layout() -> None:
    """Nonselective batch cluster COO uses its public compact pointer layout."""
    positions = jnp.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=jnp.float32)
    batch_ptr = jnp.array([0, positions.shape[0]], dtype=jnp.int32)
    cell = jnp.eye(3, dtype=jnp.float32)[None] * 8.0
    pbc = jnp.ones((1, 3), dtype=jnp.bool_)
    common = {
        "cell": cell,
        "pbc": pbc,
        "batch_ptr": batch_ptr,
        "method": "batch_cluster_tile",
        "return_neighbor_list": True,
        "coo_capacity": 16,
        "max_neighbors": 8,
        "max_tiles_per_group": 1,
    }
    compact = prepare_neighbor_list(positions, 1.0, **common)
    result, compact_successor = neighbor_list(positions, state=compact)
    assert compact_successor.coo_layout == "compact"
    np.testing.assert_array_equal(
        np.asarray(compact_successor.neighbor_ptr), np.asarray(result[1])
    )
    segmented = prepare_neighbor_list(positions, 1.0, selective=True, **common)
    assert segmented.coo_layout == "segmented"
    assert segmented.neighbor_ptr is None


@pytest.mark.parametrize("format", ["matrix", "coo", "tile"])
def test_prepared_batch_cluster_cell_shapes_match_with_live_positions(
    format: str,
) -> None:
    """Shared and per-system cells agree across prepared batch-cluster reuse."""
    positions = jnp.array(
        [
            [0.1, 1.0, 1.0],
            [11.9, 1.0, 1.0],
            [3.0, 3.0, 3.0],
            [3.2, 3.0, 3.0],
            [0.2, 1.0, 1.0],
            [11.8, 1.0, 1.0],
            [7.0, 3.0, 3.0],
            [7.2, 3.0, 3.0],
        ],
        dtype=jnp.float32,
    )
    moved = positions.at[1].set(jnp.array([5.0, 5.0, 5.0], dtype=jnp.float32))
    moved = moved.at[5].set(jnp.array([10.0, 5.0, 5.0], dtype=jnp.float32))
    batch_ptr = jnp.array([0, 4, 8], dtype=jnp.int32)
    shared_cell = jnp.eye(3, dtype=jnp.float32) * 12.0
    per_system_cell = jnp.broadcast_to(shared_cell, (2, 3, 3))
    shared_pbc = jnp.ones((3,), dtype=jnp.bool_)
    per_system_pbc = jnp.ones((2, 3), dtype=jnp.bool_)
    common = {
        "method": "batch_cluster_tile",
        "format": format,
        "return_neighbor_list": format == "coo",
        "coo_capacity": 64 if format == "coo" else None,
        "max_neighbors": 8,
        "max_tiles_per_group": 1,
        "fixed_cell": True,
    }
    shared_state = prepare_neighbor_list(
        positions,
        0.5,
        cell=shared_cell,
        pbc=shared_pbc,
        batch_ptr=batch_ptr,
        **common,
    )
    per_system_state = prepare_neighbor_list(
        positions,
        0.5,
        cell=per_system_cell,
        pbc=per_system_pbc,
        batch_ptr=batch_ptr,
        **common,
    )

    def signature(result: tuple[jax.Array, ...]) -> tuple[Any, ...]:
        """Build a format-specific signature without assuming tile order."""
        if format == "matrix":
            matrix, counts, shifts = map(np.asarray, result[:3])
            pairs = {
                (row, int(matrix[row, slot]), *map(int, shifts[row, slot]))
                for row in range(len(counts))
                for slot in range(int(counts[row]))
            }
            return tuple(map(int, counts)), pairs
        if format == "coo":
            neighbors, pointer, shifts = map(np.asarray, result[:3])
            pairs = {
                (
                    int(neighbors[0, index]),
                    int(neighbors[1, index]),
                    *map(int, shifts[index]),
                )
                for index in range(int(pointer[-1]))
            }
            return tuple(map(int, np.diff(pointer))), pairs
        active_tiles = int(np.asarray(result[0])[0])
        tile_row_group = np.asarray(result[1])[:active_tiles]
        tile_col_group = np.asarray(result[2])[:active_tiles]
        tile_system = np.asarray(result[3])[:active_tiles]
        active_records = tuple(
            sorted(
                (int(system), int(row_group), int(col_group))
                for system, row_group, col_group in zip(
                    tile_system, tile_row_group, tile_col_group, strict=True
                )
            )
        )
        return (
            active_tiles,
            active_records,
            *(np.asarray(value) for value in result[4:]),
        )

    def assert_periodic_images(result: tuple[jax.Array, ...]) -> None:
        if format == "matrix":
            matrix, counts, shifts = map(np.asarray, result[:3])
            for system_start in (0, 4):
                assert any(
                    np.any(shifts[row, : int(counts[row])] != 0)
                    for row in range(system_start, system_start + 4)
                )
        elif format == "coo":
            neighbors, pointer, shifts = map(np.asarray, result[:3])
            active = int(pointer[-1])
            sources = neighbors[0, :active]
            for system_start in (0, 4):
                system_pairs = (sources >= system_start) & (sources < system_start + 4)
                assert np.any(system_pairs)
                assert np.any(np.any(shifts[:active][system_pairs] != 0, axis=1))

    def assert_direct(result: tuple[jax.Array, ...], values: jax.Array) -> None:
        direct = batch_cluster_tile_neighbor_list(
            values,
            0.5,
            per_system_cell,
            batch_ptr,
            max_neighbors=8,
            format=format,
            max_pairs=64,
            max_tiles_per_group=1,
        )
        actual_signature = signature(result)
        direct_signature = signature(direct)
        if format == "tile":
            assert actual_signature[:2] == direct_signature[:2]
            for actual, expected in zip(
                actual_signature[2:], direct_signature[2:], strict=True
            ):
                np.testing.assert_array_equal(actual, expected)
        else:
            assert actual_signature == direct_signature

    def assert_equivalent(
        actual: tuple[jax.Array, ...], expected: tuple[jax.Array, ...]
    ) -> None:
        actual_signature = signature(actual)
        expected_signature = signature(expected)
        if format == "tile":
            assert actual_signature[:2] == expected_signature[:2]
            for actual, expected in zip(
                actual_signature[2:], expected_signature[2:], strict=True
            ):
                np.testing.assert_array_equal(actual, expected)
        else:
            assert actual_signature == expected_signature

    current_frames = (positions, moved)
    for frame_index, current in enumerate(current_frames):
        shared_result, shared_state = neighbor_list(current, state=shared_state)
        per_system_result, per_system_state = neighbor_list(
            current, state=per_system_state
        )
        assert_equivalent(shared_result, per_system_result)
        assert_direct(per_system_result, current)
        assert bool(jnp.all(shared_state.valid))
        assert bool(jnp.all(per_system_state.valid))
        if frame_index == 0 and format in {"matrix", "coo"}:
            assert_periodic_images(shared_result)


@pytest.mark.parametrize("method", ["cluster_tile", "batch_cluster_tile"])
@pytest.mark.parametrize("format", ["matrix", "coo"])
@pytest.mark.parametrize("fixed_cell", [False, True])
def test_prepared_cluster_geometry_and_callback_outputs_are_stateful(
    method: str, format: str, fixed_cell: bool
) -> None:
    """Prepared cluster geometry/callback tails survive successor threading."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [4.0, 0.0, 0.0], [4.4, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    pbc = jnp.ones((3,), dtype=jnp.bool_)
    kwargs: dict[str, Any] = {
        "cell": cell if method == "cluster_tile" else jnp.broadcast_to(cell, (1, 3, 3)),
        "pbc": pbc if method == "cluster_tile" else jnp.broadcast_to(pbc, (1, 3)),
        "method": method,
        "max_neighbors": 8,
        "max_tiles_per_group": 1,
        "return_distances": True,
        "return_vectors": True,
        "pair_fn": _prepared_pair_fn,
        "fixed_cell": fixed_cell,
    }
    if method == "batch_cluster_tile":
        kwargs["batch_ptr"] = jnp.array([0, positions.shape[0]], dtype=jnp.int32)
    if format == "coo":
        kwargs["coo_capacity"] = 32
    state = prepare_neighbor_list(
        positions, 2.0, return_neighbor_list=format == "coo", **kwargs
    )
    params = jnp.ones((positions.shape[0], 1), dtype=jnp.float32)
    result, successor = neighbor_list(positions, state=state, pair_params=params)
    assert len(result) == 7
    assert successor.neighbor_distances is not None
    assert successor.neighbor_vectors is not None
    assert successor.pair_energies is not None
    assert successor.pair_forces is not None

    direct_kwargs: dict[str, Any] = {
        "max_neighbors": 8,
        "max_tiles_per_group": 1,
        "return_distances": True,
        "return_vectors": True,
        "pair_fn": _prepared_pair_fn,
        "pair_params": params,
        "format": "matrix" if format == "matrix" else "coo",
    }
    if format == "coo":
        direct_kwargs["max_pairs"] = 32
    if method == "batch_cluster_tile":
        direct = batch_cluster_tile_neighbor_list(
            positions,
            2.0,
            jnp.broadcast_to(cell, (1, 3, 3)),
            jnp.array([0, positions.shape[0]], dtype=jnp.int32),
            **direct_kwargs,
        )
    else:
        direct = cluster_tile_neighbor_list(positions, 2.0, cell, **direct_kwargs)

    if format == "matrix":
        assert _matrix_to_pair_set_full(*result[:3], len(positions)) == (
            _matrix_to_pair_set_full(*direct[:3], len(positions))
        )
        counts = np.asarray(result[1])
        np.testing.assert_array_equal(counts, np.asarray(direct[1]))
        for row, count in enumerate(counts):
            active = int(count)
            for index in (3, 4, 5, 6):
                np.testing.assert_allclose(
                    np.asarray(result[index])[row, :active],
                    np.asarray(direct[index])[row, :active],
                    atol=3e-5,
                )
    else:
        np.testing.assert_array_equal(np.asarray(result[1]), np.asarray(direct[1]))
        active_pairs = int(np.asarray(result[1])[-1])
        np.testing.assert_array_equal(
            np.asarray(result[0])[:, :active_pairs],
            np.asarray(direct[0])[:, :active_pairs],
        )
        np.testing.assert_array_equal(
            np.asarray(result[2])[:active_pairs], np.asarray(direct[2])[:active_pairs]
        )
        for index in (3, 4, 5, 6):
            np.testing.assert_allclose(
                np.asarray(result[index])[:active_pairs],
                np.asarray(direct[index])[:active_pairs],
                atol=3e-5,
            )


@pytest.mark.parametrize("method", ["cluster_tile", "batch_cluster_tile"])
@pytest.mark.parametrize("format", ["matrix", "coo"])
def test_prepared_empty_cluster_routes_keep_public_prefix_and_status(
    method: str, format: str
) -> None:
    """Empty prepared cluster routes consume the same six-field status tail."""
    positions = jnp.empty((0, 3), dtype=jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    kwargs: dict[str, Any] = {
        "cell": cell if method == "cluster_tile" else cell[jnp.newaxis],
        "pbc": jnp.ones(3, dtype=jnp.bool_)
        if method == "cluster_tile"
        else jnp.ones((1, 3), dtype=jnp.bool_),
        "method": method,
        "max_neighbors": 8,
        "max_tiles_per_group": 1,
    }
    if method == "batch_cluster_tile":
        kwargs["batch_ptr"] = jnp.array([0, 0], dtype=jnp.int32)
    if format == "coo":
        kwargs["coo_capacity"] = 8
    state = prepare_neighbor_list(
        positions, 2.0, return_neighbor_list=format == "coo", **kwargs
    )
    result, successor = neighbor_list(positions, state=state)
    assert len(result) == 3
    assert bool(jnp.all(successor.valid))


@pytest.mark.parametrize("method", ["cluster_tile", "batch_cluster_tile"])
@pytest.mark.parametrize("fixed_cell", [False, True])
def test_prepared_pair_output_cluster_tile_overflow_is_sticky(
    method: str, fixed_cell: bool
) -> None:
    """Pair-output cluster routes latch bounded tile overflow in their successor."""
    positions = jnp.zeros((64, 3), dtype=jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    kwargs: dict[str, Any] = {
        "cell": cell if method == "cluster_tile" else cell[jnp.newaxis],
        "pbc": jnp.ones(3, dtype=jnp.bool_)
        if method == "cluster_tile"
        else jnp.ones((1, 3), dtype=jnp.bool_),
        "method": method,
        "max_neighbors": 64,
        "max_tiles_per_group": 1,
        "pair_fn": _prepared_pair_fn,
        "fixed_cell": fixed_cell,
    }
    if method == "batch_cluster_tile":
        kwargs["batch_ptr"] = jnp.array([0, 64], dtype=jnp.int32)
    state = prepare_neighbor_list(positions, 2.0, **kwargs)
    _, failed = neighbor_list(
        positions,
        state=state,
        pair_params=jnp.ones((64, 1), dtype=jnp.float32),
    )
    assert not bool(jnp.all(failed.valid))
    with pytest.raises(TileBufferOverflow):
        check_neighbor_list_state(failed)


@pytest.mark.parametrize("batched", [False, True])
def test_cluster_pair_output_status_preserves_direct_prefix(batched: bool) -> None:
    """Private pair-output status collection leaves the direct tuple untouched."""
    positions, cell, _pbc = _cell_inputs(4)
    params = jnp.ones((4, 1), dtype=jnp.float32)
    call = batch_cluster_tile_neighbor_list if batched else cluster_tile_neighbor_list
    args = (
        (positions, 2.0, cell[jnp.newaxis], jnp.array([0, 4], dtype=jnp.int32))
        if batched
        else (positions, 2.0, cell)
    )
    kwargs = dict(
        max_neighbors=8,
        max_tiles_per_group=1,
        pair_fn=_prepared_pair_fn,
        pair_params=params,
    )
    direct = call(*args, **kwargs)
    status = call(*args, _return_status=True, **kwargs)
    assert len(status) == len(direct) + 6
    for expected, actual in zip(direct, status[: len(direct)], strict=True):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))


def test_prepared_selective_fixed_coo_initialized_false_fast_path() -> None:
    """Initialized all-false selective fixed COO preserves published state under jit."""
    positions = jnp.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=jnp.ones((3,), dtype=jnp.bool_),
        method="naive",
        selective=True,
        return_neighbor_list=True,
        coo_capacity=8,
        max_neighbors=8,
        fixed_cell=True,
    )
    false = jnp.zeros((1,), dtype=jnp.bool_)
    _, fresh_preserve = neighbor_list(positions, state=state, rebuild_flags=false)
    assert not bool(jnp.all(fresh_preserve.valid))
    with pytest.raises(RuntimeError, match="uninitialized"):
        check_neighbor_list_state(fresh_preserve)
    initialized_result, initialized = neighbor_list(
        positions, state=state, rebuild_flags=jnp.ones((1,), dtype=jnp.bool_)
    )
    run = jax.jit(
        lambda p, s, flags: neighbor_list(p, state=s, rebuild_flags=flags),
        donate_argnums=(1,),
    )
    published = tuple(np.asarray(value) for value in initialized_result)
    expected, preserved = run(positions, initialized, false)
    for before, after in zip(
        published,
        expected,
        strict=True,
    ):
        np.testing.assert_array_equal(before, np.asarray(after))
    assert bool(jnp.all(preserved.initialized))
    assert bool(jnp.all(preserved.valid))


def test_prepared_selective_fixed_coo_compiled_branch_transitions() -> None:
    """A donated batched dual state crosses rebuild and preserve branches."""
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.4, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    moved = positions.at[1].set(jnp.array([2.0, 0.0, 0.0], dtype=jnp.float32))
    batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    batch_idx = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 2)
    cell = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32) * 8.0, (2, 3, 3))
    pbc = jnp.ones((2, 3), dtype=jnp.bool_)
    state = prepare_neighbor_list(
        positions,
        0.75,
        cutoff2=1.0,
        cell=cell,
        pbc=pbc,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        method="batch_naive_dual_cutoff",
        return_neighbor_list=True,
        coo_capacity=(16, 16),
        max_neighbors1=4,
        max_neighbors2=4,
        selective=True,
    )

    def execute(current_positions, current_state, flags):
        return neighbor_list(
            current_positions,
            state=current_state,
            rebuild_flags=flags,
        )

    compiled = jax.jit(execute, donate_argnums=(1,))
    rebuild_all = jnp.ones((2,), dtype=jnp.bool_)
    preserve_all = jnp.zeros((2,), dtype=jnp.bool_)
    rebuild_first = jnp.array([True, False], dtype=jnp.bool_)

    initial_result, state1 = compiled(positions, state, rebuild_all)
    initial_values = tuple(np.asarray(value) for value in initial_result)
    initial_counts1 = np.asarray(state1.num_neighbors1)
    initial_counts2 = np.asarray(state1.num_neighbors2)
    with pytest.raises(ValueError, match="runtime positions shape"):
        neighbor_list(positions[:3], state=state1, rebuild_flags=preserve_all)
    with pytest.raises(ValueError, match="rebuild_flags must"):
        neighbor_list(
            positions,
            state=state1,
            rebuild_flags=jnp.zeros((1,), dtype=jnp.bool_),
        )
    with pytest.raises(ValueError, match="runtime cell shape or dtype"):
        neighbor_list(
            positions,
            cell=jnp.eye(3, dtype=jnp.float32),
            state=state1,
            rebuild_flags=preserve_all,
        )

    preserved_result, state2 = compiled(positions, state1, preserve_all)
    for expected, actual in zip(initial_values, preserved_result, strict=True):
        np.testing.assert_array_equal(expected, np.asarray(actual))

    rebuilt_result, state3 = compiled(moved, state2, rebuild_first)
    rebuilt_values = tuple(np.asarray(value) for value in rebuilt_result)
    rebuilt_counts1 = np.asarray(state3.num_neighbors1)
    rebuilt_counts2 = np.asarray(state3.num_neighbors2)
    assert not np.array_equal(initial_counts1[:2], rebuilt_counts1[:2])
    assert not np.array_equal(initial_counts2[:2], rebuilt_counts2[:2])
    np.testing.assert_array_equal(initial_counts1[2:], rebuilt_counts1[2:])
    np.testing.assert_array_equal(initial_counts2[2:], rebuilt_counts2[2:])

    final_result, state4 = compiled(moved, state3, preserve_all)
    for expected, actual in zip(rebuilt_values, final_result, strict=True):
        np.testing.assert_array_equal(expected, np.asarray(actual))
    np.testing.assert_array_equal(np.asarray(state4.num_neighbors1), rebuilt_counts1)
    np.testing.assert_array_equal(np.asarray(state4.num_neighbors2), rebuilt_counts2)
    assert state4.initialized.tolist() == [True, True]
    assert state4.valid.tolist() == [True, True]


@pytest.mark.parametrize(
    "fixed_cell", [False, True], ids=["default-cell", "fixed-cell"]
)
@pytest.mark.parametrize("batched", [False, True], ids=["single", "batched"])
def test_prepared_selective_cell_list_jit_preserves_successor(
    fixed_cell: bool, batched: bool
) -> None:
    """Jitted selective cell-list execution preserves a successor when skipped."""
    if batched:
        positions = jnp.array(
            [
                [0.2, 0.1, 0.2],
                [0.4, 0.1, 0.2],
                [3.0, 0.1, 0.2],
                [3.2, 0.1, 0.2],
            ],
            dtype=jnp.float32,
        )
        batch_idx = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 2)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        cell = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32) * 8.0, (2, 3, 3))
        pbc = jnp.broadcast_to(jnp.array([True, False, True], dtype=jnp.bool_), (2, 3))
        options = {
            "cell": cell,
            "pbc": pbc,
            "batch_idx": batch_idx,
            "batch_ptr": batch_ptr,
            "method": "batch_cell_list",
            "strategy": "atom_centric",
            "max_neighbors": 8,
            "max_total_cells": 512,
            "selective": True,
            "fixed_cell": fixed_cell,
        }
    else:
        positions = jnp.array(
            [
                [0.2, 0.1, 0.2],
                [7.8, 0.1, 0.2],
                [3.0, 1.0, 1.0],
                [3.4, 1.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        batch_idx = None
        batch_ptr = None
        cell = jnp.eye(3, dtype=jnp.float32) * 8.0
        pbc = jnp.array([True, False, True], dtype=jnp.bool_)
        options = {
            "cell": cell,
            "pbc": pbc,
            "method": "cell_list",
            "strategy": "atom_centric",
            "max_neighbors": 8,
            "max_total_cells": 512,
            "selective": True,
            "fixed_cell": fixed_cell,
        }
    state = prepare_neighbor_list(positions, 0.7, **options)
    assert state.supports_compilation
    assert state.fixed_cell is fixed_cell
    compiled = jax.jit(
        lambda current_positions, current_state, flags: neighbor_list(
            current_positions,
            state=current_state,
            rebuild_flags=flags,
        )
    )

    if batched:

        def direct(values: jax.Array) -> tuple[jax.Array, ...]:
            return batch_cell_list(
                values,
                0.7,
                cell,
                pbc,
                batch_idx,
                batch_ptr,
                max_neighbors=8,
                max_total_cells=512,
                strategy="atom_centric",
            )

        all_systems = jnp.ones((2,), dtype=jnp.bool_)
        preserve_all = jnp.zeros((2,), dtype=jnp.bool_)
        initial_result, initialized = compiled(positions, state, all_systems)
        _assert_active_cell_topology_equal(initial_result, direct(positions))
        assert initialized.initialized.tolist() == [True, True]
        assert initialized.valid.tolist() == [True, True]

        moved = positions.at[1].set(jnp.array([1.2, 0.1, 0.2], dtype=jnp.float32))
        moved = moved.at[3].set(jnp.array([4.2, 0.1, 0.2], dtype=jnp.float32))
        fresh_moved = direct(moved)
        assert int(jnp.sum(fresh_moved[1])) < int(jnp.sum(initial_result[1]))

        preserved_result, preserved = compiled(moved, initialized, preserve_all)
        _assert_active_cell_topology_equal(preserved_result, initial_result)
        assert preserved.initialized.tolist() == [True, True]
        assert preserved.valid.tolist() == [True, True]

        rebuild_first = jnp.array([True, False], dtype=jnp.bool_)
        mixed_result, mixed = compiled(moved, preserved, rebuild_first)
        hybrid_positions = moved.at[2:].set(positions[2:])
        _assert_active_cell_topology_equal(
            mixed_result, direct(hybrid_positions), rows={0, 1}
        )
        _assert_active_cell_topology_equal(mixed_result, initial_result, rows={2, 3})
        assert int(jnp.sum(mixed_result[1][:2])) < int(jnp.sum(initial_result[1][:2]))
        assert int(jnp.sum(mixed_result[1][2:])) == int(jnp.sum(initial_result[1][2:]))
        assert mixed.initialized.tolist() == [True, True]
        assert mixed.valid.tolist() == [True, True]

        reused_result, reused = compiled(moved, mixed, preserve_all)
        _assert_active_cell_topology_equal(reused_result, mixed_result)
        assert reused.initialized.tolist() == [True, True]
        assert reused.valid.tolist() == [True, True]

        rebuilt_result, rebuilt = compiled(moved, reused, all_systems)
        _assert_active_cell_topology_equal(rebuilt_result, fresh_moved)
        assert int(jnp.sum(rebuilt_result[1])) < int(jnp.sum(mixed_result[1]))
        assert rebuilt.initialized.tolist() == [True, True]
        assert rebuilt.valid.tolist() == [True, True]
        assert rebuilt.fixed_cell is fixed_cell
        return

    moved = positions.at[1].set(jnp.array([6.8, 0.1, 0.2], dtype=jnp.float32))
    initial_result, initialized = compiled(
        positions, state, jnp.ones((1,), dtype=jnp.bool_)
    )
    direct = cell_list(
        positions,
        0.7,
        cell,
        pbc,
        max_neighbors=8,
        max_total_cells=512,
        strategy="atom_centric",
    )
    _assert_active_cell_topology_equal(initial_result, direct)
    assert bool(jnp.all(initialized.initialized))
    assert bool(jnp.all(initialized.valid))

    preserved_result, successor = compiled(
        moved, initialized, jnp.zeros((1,), dtype=jnp.bool_)
    )
    for expected, actual in zip(initial_result, preserved_result, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert bool(jnp.all(successor.initialized))
    assert bool(jnp.all(successor.valid))
    assert successor.fixed_cell is fixed_cell

    rebuilt_result, rebuilt = compiled(
        moved, successor, jnp.ones((1,), dtype=jnp.bool_)
    )
    direct_moved = cell_list(
        moved,
        0.7,
        cell,
        pbc,
        max_neighbors=8,
        max_total_cells=512,
        strategy="atom_centric",
    )
    _assert_active_cell_topology_equal(rebuilt_result, direct_moved)
    assert int(jnp.sum(rebuilt_result[1])) != int(jnp.sum(initial_result[1]))
    assert bool(jnp.all(rebuilt.initialized))
    assert bool(jnp.all(rebuilt.valid))
    assert rebuilt.fixed_cell is fixed_cell


def test_prepared_fixed_cell_api_cache_leaf_and_tree_round_trip() -> None:
    """Fixed geometry is a read-only state option and a dynamic cache child."""
    positions, cell, pbc = _cell_inputs(4)
    with pytest.raises(ValueError, match="fixed_cell=True requires an explicit cell"):
        prepare_neighbor_list(positions, 2.0, fixed_cell=True, method="naive")
    for value in (1, np.bool_(True), "yes"):
        with pytest.raises(ValueError, match="fixed_cell must be a Python bool"):
            prepare_neighbor_list(
                positions,
                2.0,
                cell=cell,
                pbc=pbc,
                fixed_cell=value,
                method="naive",
            )

    dynamic = prepare_neighbor_list(
        positions, 2.0, cell=cell, pbc=pbc, method="naive", max_neighbors=8
    )
    assert dynamic.fixed_cell is False
    state = prepare_neighbor_list(
        positions,
        2.0,
        cell=cell,
        pbc=pbc,
        method="naive",
        max_neighbors=8,
        fixed_cell=True,
    )
    assert state.fixed_cell is True
    with pytest.raises(AttributeError, match="read-only"):
        state.fixed_cell = False
    with pytest.raises(TypeError, match="unexpected prepared neighbor-list option"):
        neighbor_list(positions, state=state, fixed_cell=True)

    leaves, treedef = jax.tree_util.tree_flatten(state)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.fixed_cell is True
    original_result, _ = neighbor_list(positions, state=state)
    round_trip_result, _ = neighbor_list(positions, state=rebuilt)
    replacement_result, replacement_state = neighbor_list(
        positions, cell=jnp.array(cell), state=state
    )
    original_pairs = _matrix_to_pair_set_full(*original_result[:3], len(positions))
    assert original_pairs == _matrix_to_pair_set_full(
        *round_trip_result[:3], len(positions)
    )
    np.testing.assert_array_equal(original_result[1], round_trip_result[1])
    assert original_pairs == _matrix_to_pair_set_full(
        *replacement_result[:3], len(positions)
    )
    np.testing.assert_array_equal(original_result[1], replacement_result[1])
    assert replacement_state.fixed_cell

    alternate_cell = cell * 0.5
    alternate_state = prepare_neighbor_list(
        jnp.array([[0.1, 0.0, 0.0], [7.9, 0.0, 0.0]], dtype=jnp.float32),
        0.5,
        cell=alternate_cell,
        pbc=pbc,
        method="naive",
        max_neighbors=8,
        fixed_cell=True,
    )
    same_config_positions = jnp.array(
        [[0.1, 0.0, 0.0], [7.9, 0.0, 0.0]], dtype=jnp.float32
    )
    first_cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    first_state = prepare_neighbor_list(
        same_config_positions,
        0.5,
        cell=first_cell,
        pbc=pbc,
        method="naive",
        max_neighbors=8,
        fixed_cell=True,
    )
    compiled = jax.jit(lambda values, prepared: neighbor_list(values, state=prepared))
    first_result, _ = compiled(same_config_positions, first_state)
    alternate_result, _ = compiled(same_config_positions, alternate_state)
    assert _matrix_to_pair_set_full(*first_result[:3], 2) != _matrix_to_pair_set_full(
        *alternate_result[:3], 2
    )


@pytest.mark.parametrize(
    "method",
    [
        "naive",
        "batch_naive",
        "naive_dual_cutoff",
        "batch_naive_dual_cutoff",
    ],
)
def test_prepared_fixed_naive_nonperiodic_explicit_cell_executes(method: str) -> None:
    """An explicit fixed cell is retained while nonperiodic naive routes omit it."""
    batched = method.startswith("batch_")
    dual = method.endswith("dual_cutoff")
    positions = jnp.array([[0.1, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    if batched:
        cell = cell[None]
    batch_ptr = jnp.array([0, 2], dtype=jnp.int32)
    options: dict[str, Any] = {
        "cell": cell,
        "pbc": None,
        "method": method,
    }
    direct_options: dict[str, Any] = {}
    if dual:
        options.update(max_neighbors1=4, max_neighbors2=4)
        direct_options.update(max_neighbors1=4, max_neighbors2=4)
    else:
        options["max_neighbors"] = 4
        direct_options["max_neighbors"] = 4
    if batched:
        options["batch_ptr"] = batch_ptr
        direct_options["batch_ptr"] = batch_ptr

    state = prepare_neighbor_list(
        positions,
        0.5,
        cutoff2=1.0 if dual else None,
        fixed_cell=True,
        **options,
    )
    actual, successor = neighbor_list(positions, state=state)
    assert successor.fixed_cell
    if method == "naive":
        direct = naive_neighbor_list(positions, 0.5, **direct_options)
    elif method == "batch_naive":
        direct = batch_naive_neighbor_list(positions, 0.5, **direct_options)
    elif method == "naive_dual_cutoff":
        direct = naive_neighbor_list_dual_cutoff(positions, 0.5, 1.0, **direct_options)
    else:
        direct = batch_naive_neighbor_list_dual_cutoff(
            positions, 0.5, 1.0, **direct_options
        )

    for offset in (0, 2) if dual else (0,):
        np.testing.assert_array_equal(
            np.asarray(actual[offset + 1]), np.asarray(direct[offset + 1])
        )
        for row, count in enumerate(np.asarray(actual[offset + 1])):
            active = int(count)
            np.testing.assert_array_equal(
                np.sort(np.asarray(actual[offset])[row, :active]),
                np.sort(np.asarray(direct[offset])[row, :active]),
            )


@pytest.mark.parametrize(
    ("method", "strategy", "wrap_positions"),
    [
        ("naive", "scalar", True),
        ("naive", "tile", False),
        ("naive", "tile", True),
        ("naive_dual_cutoff", "scalar", True),
        ("batch_naive", "scalar", True),
        ("batch_naive", "tile", True),
        ("batch_naive_dual_cutoff", "scalar", True),
    ],
)
def test_prepared_fixed_naive_routes_match_direct_partial_pbc(
    method: str, strategy: str, wrap_positions: bool
) -> None:
    """Fixed naive scalar, tiled, dual, and batched routes preserve PBC results."""
    batched = method.startswith("batch_")
    positions = jnp.array(
        [
            [0.2, 0.0, 0.0],
            [7.8, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    one_cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    one_pbc = jnp.array([True, False, True], dtype=jnp.bool_)
    batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    call_cell = jnp.broadcast_to(one_cell, (2, 3, 3)) if batched else one_cell
    call_pbc = (
        jnp.array([[True, False, True], [False, False, False]], dtype=jnp.bool_)
        if batched
        else one_pbc
    )
    kwargs: dict[str, Any] = {
        "cell": call_cell,
        "pbc": call_pbc,
        "method": method,
        "strategy": strategy,
        "wrap_positions": wrap_positions,
        "max_neighbors": 16,
    }
    if batched:
        kwargs["batch_ptr"] = batch_ptr
    if method.endswith("dual_cutoff"):
        kwargs.update(cutoff2=1.5, max_neighbors1=16, max_neighbors2=16)

    state = prepare_neighbor_list(positions, 0.75, fixed_cell=True, **kwargs)
    if method == "naive":
        expected = naive_neighbor_list(
            positions,
            0.75,
            cell=call_cell,
            pbc=call_pbc,
            max_neighbors=16,
            strategy=strategy,
            wrap_positions=wrap_positions,
        )
    elif method == "batch_naive":
        expected = batch_naive_neighbor_list(
            positions,
            0.75,
            cell=call_cell,
            pbc=call_pbc,
            batch_ptr=batch_ptr,
            max_neighbors=16,
            strategy=strategy,
            wrap_positions=wrap_positions,
        )
    elif method == "naive_dual_cutoff":
        expected = naive_neighbor_list_dual_cutoff(
            positions,
            0.75,
            1.5,
            cell=call_cell,
            pbc=call_pbc,
            max_neighbors1=16,
            max_neighbors2=16,
            wrap_positions=wrap_positions,
        )
    else:
        expected = batch_naive_neighbor_list_dual_cutoff(
            positions,
            0.75,
            1.5,
            cell=call_cell,
            pbc=call_pbc,
            batch_ptr=batch_ptr,
            max_neighbors1=16,
            max_neighbors2=16,
            wrap_positions=wrap_positions,
        )
    actual, successor = neighbor_list(positions, state=state)
    assert successor.fixed_cell
    if method.endswith("dual_cutoff"):
        for offset in (0, 3):
            assert _matrix_to_pair_set_full(
                *actual[offset : offset + 3], len(positions)
            ) == _matrix_to_pair_set_full(
                *expected[offset : offset + 3], len(positions)
            )
            np.testing.assert_array_equal(
                np.asarray(actual[offset + 1]), np.asarray(expected[offset + 1])
            )
    else:
        assert _matrix_to_pair_set_full(*actual[:3], len(positions)) == (
            _matrix_to_pair_set_full(*expected[:3], len(positions))
        )
        np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(expected[1]))


@pytest.mark.parametrize("method", ["naive", "batch_naive"])
@pytest.mark.parametrize("fixed_cell", [False, True])
def test_prepared_tiled_wrapped_pbc_default_and_fixed_modes(
    method: str, fixed_cell: bool
) -> None:
    """Wrapped tile callbacks preserve default scratch and fixed-cache modes."""
    positions_np = np.array([[60.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    positions = jnp.asarray(positions_np)
    cell_np = np.eye(3, dtype=np.float32) * 5.0
    cell = jnp.asarray(cell_np)
    pbc = jnp.array([True, False, False], dtype=jnp.bool_)
    kwargs: dict[str, Any] = {
        "cell": cell,
        "pbc": pbc,
        "method": method,
        "strategy": "tile",
        "wrap_positions": True,
        "max_neighbors": 4,
    }
    direct_kwargs: dict[str, Any] = {
        "cell": cell,
        "pbc": pbc,
        "strategy": "tile",
        "wrap_positions": True,
        "max_neighbors": 4,
    }
    if method == "batch_naive":
        batch_ptr = jnp.array([0, 2], dtype=jnp.int32)
        kwargs["batch_ptr"] = batch_ptr
        direct_kwargs["batch_ptr"] = batch_ptr
    cutoff = 0.75
    state = prepare_neighbor_list(positions, cutoff, fixed_cell=fixed_cell, **kwargs)
    assert state.fixed_cell is fixed_cell
    cached_inverse_before = (
        np.asarray(state._leaves[_FIXED_CELL_GEOMETRY_LEAF][0]).copy()
        if fixed_cell
        else None
    )

    def _raw_image_oracle(values: np.ndarray, box: np.ndarray) -> set[tuple[int, ...]]:
        pairs: set[tuple[int, ...]] = set()
        for source in range(len(values)):
            for neighbor in range(len(values)):
                for shift_x in range(-20, 21):
                    if source == neighbor and shift_x == 0:
                        continue
                    displacement = (
                        values[neighbor]
                        - values[source]
                        + np.array([shift_x, 0, 0], dtype=values.dtype) @ box
                    )
                    if float(np.linalg.norm(displacement)) < cutoff:
                        pairs.add((source, neighbor, shift_x, 0, 0))
        return pairs

    updated_cell_np = cell_np if fixed_cell else np.eye(3, dtype=np.float32) * 6.0
    if not fixed_cell:
        assert _raw_image_oracle(positions_np, cell_np) != _raw_image_oracle(
            positions_np, updated_cell_np
        )
    current_state = state
    for step in range(2):
        if fixed_cell:
            current_positions_np = positions_np.copy()
            if step == 1:
                current_positions_np[0, 0] = 65.1
            runtime_cell_np = cell_np
        else:
            current_positions_np = positions_np
            runtime_cell_np = cell_np if step == 0 else updated_cell_np
        current_positions = jnp.asarray(current_positions_np)
        runtime_cell = jnp.asarray(runtime_cell_np)
        result, current_state = neighbor_list(
            current_positions, cell=runtime_cell, state=current_state
        )
        direct_call = (
            batch_naive_neighbor_list
            if method == "batch_naive"
            else naive_neighbor_list
        )
        direct = direct_call(
            current_positions,
            cutoff,
            **{**direct_kwargs, "cell": runtime_cell},
        )
        expected = _raw_image_oracle(current_positions_np, runtime_cell_np)
        assert _matrix_to_pair_set_full(*result[:3], len(positions_np)) == expected
        assert _matrix_to_pair_set_full(*direct[:3], len(positions_np)) == expected
        if fixed_cell:
            np.testing.assert_array_equal(
                np.asarray(current_state._leaves[_FIXED_CELL_GEOMETRY_LEAF][0]),
                cached_inverse_before,
            )
    assert current_state.fixed_cell is fixed_cell


def _matrix_geometry_by_pair(
    result: tuple[jax.Array, ...],
) -> dict[tuple[int, int, int, int, int], tuple[np.ndarray, float]]:
    """Map active matrix entries to their output vector and distance."""
    matrix, counts, shifts, distances, vectors = map(np.asarray, result[:5])
    pairs = {}
    for i in range(len(counts)):
        for slot in range(int(counts[i])):
            shift = tuple(int(value) for value in shifts[i, slot])
            pairs[(i, int(matrix[i, slot]), *shift)] = (
                vectors[i, slot].copy(),
                float(distances[i, slot]),
            )
    return pairs


@pytest.mark.parametrize(
    ("method", "strategy"),
    [
        ("naive", "scalar"),
        ("batch_naive", "scalar"),
        ("cell_list", "atom_centric"),
        ("cell_list", "pair_centric"),
        ("batch_cell_list", "atom_centric"),
    ],
)
def test_prepared_fixed_pair_callbacks_match_direct(method: str, strategy: str) -> None:
    """Fixed single-cutoff supported naive/cell routes forward pair callbacks."""
    batched = method.startswith("batch_")
    positions = jnp.array(
        [
            [0.2, 1.0, 1.0],
            [7.8, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [3.4, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    one_cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    one_pbc = jnp.ones((3,), dtype=jnp.bool_)
    batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    cell = jnp.broadcast_to(one_cell, (2, 3, 3)) if batched else one_cell
    pbc = jnp.broadcast_to(one_pbc, (2, 3)) if batched else one_pbc
    params = jnp.array([[0.1], [0.2], [0.3], [0.4]], dtype=jnp.float32)
    options: dict[str, Any] = {
        "cell": cell,
        "pbc": pbc,
        "method": method,
        "strategy": strategy,
        "max_neighbors": 8,
        "return_distances": True,
        "return_vectors": True,
        "pair_fn": _prepared_pair_fn,
    }
    direct_options: dict[str, Any] = {
        "max_neighbors": 8,
        "return_distances": True,
        "return_vectors": True,
        "pair_fn": _prepared_pair_fn,
        "pair_params": params,
        "strategy": strategy,
    }
    if method in {"cell_list", "batch_cell_list"}:
        options["max_total_cells"] = 512
        direct_options["max_total_cells"] = 512
    if batched:
        options["batch_ptr"] = batch_ptr
        direct_options["batch_ptr"] = batch_ptr

    state = prepare_neighbor_list(positions, 0.7, fixed_cell=True, **options)
    actual, successor = neighbor_list(positions, state=state, pair_params=params)
    assert successor.fixed_cell
    if method == "batch_naive":
        direct = batch_naive_neighbor_list(
            positions, 0.7, cell=cell, pbc=pbc, **direct_options
        )
    elif method == "batch_cell_list":
        direct = batch_cell_list(positions, 0.7, cell=cell, pbc=pbc, **direct_options)
    elif method == "naive":
        direct = naive_neighbor_list(
            positions, 0.7, cell=cell, pbc=pbc, **direct_options
        )
    else:
        direct = cell_list(positions, 0.7, cell=cell, pbc=pbc, **direct_options)

    actual_counts = np.asarray(actual[1])
    direct_counts = np.asarray(direct[1])
    np.testing.assert_array_equal(actual_counts, direct_counts)
    assert _matrix_to_pair_set_full(*actual[:3], len(positions)) == (
        _matrix_to_pair_set_full(*direct[:3], len(positions))
    )
    for row, count in enumerate(actual_counts):
        active = int(count)
        for output_index in (3, 4, 5, 6):
            np.testing.assert_allclose(
                np.asarray(actual[output_index])[row, :active],
                np.asarray(direct[output_index])[row, :active],
            )


def test_prepared_fixed_naive_wrapped_pairs_track_moving_positions_and_gradients() -> (
    None
):
    """Wrapped fixed-naive pairs retain image offsets and live geometry gradients."""
    positions = jnp.array([[0.2, 1.0, 1.0], [7.8, 1.0, 1.0]], dtype=jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    pbc = jnp.array([True, False, False], dtype=jnp.bool_)
    options = {
        "cell": cell,
        "pbc": pbc,
        "method": "naive",
        "strategy": "scalar",
        "max_neighbors": 4,
        "return_vectors": True,
        "return_distances": True,
    }
    state = prepare_neighbor_list(positions, 0.6, fixed_cell=True, **options)
    for current_positions in (
        positions,
        positions.at[0, 0].set(8.1),
    ):
        actual, state = neighbor_list(current_positions, state=state)
        direct = naive_neighbor_list(
            current_positions,
            0.6,
            cell=cell,
            pbc=pbc,
            max_neighbors=4,
            strategy="scalar",
            return_vectors=True,
            return_distances=True,
        )
        actual_geometry = _matrix_geometry_by_pair(actual)
        direct_geometry = _matrix_geometry_by_pair(direct)
        assert set(actual_geometry) == set(direct_geometry)
        expected_keys = _brute_force_pairs_full(
            np.asarray(current_positions), np.asarray(cell), 0.6, pbc=True
        )
        assert set(actual_geometry) == expected_keys
        for key in actual_geometry:
            np.testing.assert_allclose(actual_geometry[key][0], direct_geometry[key][0])
            np.testing.assert_allclose(actual_geometry[key][1], direct_geometry[key][1])

    def first_vector_x(current_positions: jax.Array, current_cell: jax.Array):
        result, _ = neighbor_list(
            current_positions,
            cell=current_cell,
            state=state,
        )
        return result[4][0, 0, 0]

    vector = first_vector_x(positions, cell)
    assert float(vector) == pytest.approx(-0.4, abs=2e-5)
    position_grad = jax.grad(lambda current: first_vector_x(current, cell))(positions)
    cell_grad = jax.grad(lambda current: first_vector_x(positions, current))(cell)
    np.testing.assert_allclose(np.asarray(position_grad[:, 0]), [-1.0, 1.0])
    np.testing.assert_allclose(np.asarray(cell_grad[:, 0]), [-1.0, 0.0, 0.0])


def test_prepared_fixed_naive_partial_rows_match_direct() -> None:
    """Fixed-cell inverse reuse preserves compact partial source rows."""
    positions = jnp.array(
        [
            [0.2, 1.0, 1.0],
            [7.8, 1.0, 1.0],
            [4.0, 1.0, 1.0],
            [4.3, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    pbc = jnp.array([True, False, False], dtype=jnp.bool_)
    targets = jnp.array([3, 0], dtype=jnp.int32)
    options = {
        "cell": cell,
        "pbc": pbc,
        "method": "naive",
        "strategy": "scalar",
        "target_indices": targets,
        "max_neighbors": 4,
    }
    state = prepare_neighbor_list(positions, 0.6, fixed_cell=True, **options)
    actual, _ = neighbor_list(positions, state=state)
    direct = naive_neighbor_list(
        positions,
        0.6,
        cell=cell,
        pbc=pbc,
        target_indices=targets,
        max_neighbors=4,
        strategy="scalar",
    )

    def compact_pairs(result):
        matrix, counts, shifts = map(np.asarray, result[:3])
        pairs = set()
        for row, source in enumerate(np.asarray(targets)):
            for slot in range(int(counts[row])):
                sx, sy, sz = (int(value) for value in shifts[row, slot])
                pairs.add((int(source), int(matrix[row, slot]), sx, sy, sz))
        return pairs

    assert actual[0].shape == (2, 4)
    assert compact_pairs(actual) == compact_pairs(direct)
    np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(direct[1]))


def _partial_image_rows(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    targets: np.ndarray,
    cutoff: float,
    *,
    half_fill: bool = False,
) -> list[set[tuple[int, int, int, int]]]:
    """Enumerate compact-row image neighbors independently in NumPy."""
    result: list[set[tuple[int, int, int, int]]] = []
    shifts = [range(-2, 3) if periodic else (0,) for periodic in pbc]
    for source in targets:
        row: set[tuple[int, int, int, int]] = set()
        for neighbor in range(len(positions)):
            for shift in product(*shifts):
                if half_fill and (
                    (shift == (0, 0, 0) and int(source) >= neighbor)
                    or (shift != (0, 0, 0) and shift <= (0, 0, 0))
                ):
                    continue
                if neighbor == int(source) and shift == (0, 0, 0):
                    continue
                displacement = (
                    positions[neighbor]
                    - positions[int(source)]
                    + (np.asarray(shift) @ cell)
                )
                if float(np.dot(displacement, displacement)) < cutoff * cutoff:
                    row.add((neighbor, *shift))
        result.append(row)
    return result


def _assert_partial_matrix_matches_image_rows(
    result: tuple[jax.Array, ...], expected_rows: list[set[tuple[int, int, int, int]]]
) -> None:
    """Compare active compact matrix rows with the independent image oracle."""
    matrix, counts, shifts = map(np.asarray, result[:3])
    assert matrix.shape[0] == len(expected_rows)
    for row, expected in enumerate(expected_rows):
        assert int(counts[row]) == len(expected)
        actual = {
            (int(matrix[row, slot]), *map(int, shifts[row, slot]))
            for slot in range(int(counts[row]))
        }
        assert actual == expected


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_prepared_partial_tile_matrix_jit_matches_image_oracle(dtype: Any) -> None:
    """Prepared partial tile reuses fixed geometry and follows live positions."""
    numpy_dtype = np.float64 if dtype == jnp.float64 else np.float32
    cell_np = np.array(
        [[8.0, 0.0, 0.0], [1.0, 7.0, 0.0], [0.5, 0.75, 6.0]],
        dtype=numpy_dtype,
    )
    fractional = np.array(
        [[0.03, 0.2, 0.05], [0.97, 0.2, 0.05], [0.4, 0.4, 0.98], [0.4, 0.4, 0.02]],
        dtype=numpy_dtype,
    )
    positions_np = fractional @ cell_np
    targets_np = np.array([3, 0, 3], dtype=np.int32)
    cell = jnp.asarray(cell_np)
    pbc = jnp.array([True, False, True], dtype=jnp.bool_)
    targets = jnp.asarray(targets_np)
    positions = jnp.asarray(positions_np)
    state = prepare_neighbor_list(
        positions,
        0.7,
        cell=cell,
        pbc=pbc,
        method="naive",
        strategy="tile",
        target_indices=targets,
        max_neighbors=8,
        fixed_cell=True,
    )
    assert state.method == "naive" and state.strategy == "tile"

    compiled = jax.jit(
        lambda values, prepared: neighbor_list(values, state=prepared),
        donate_argnums=(1,),
    )
    current = positions
    for step in range(2):
        result, state = compiled(current, state)
        current_np = np.asarray(current)
        expected = _partial_image_rows(
            current_np, cell_np, np.array([True, False, True]), targets_np, 0.7
        )
        _assert_partial_matrix_matches_image_rows(result, expected)
        assert result[0].shape == (3, 8)
        if step == 0:
            moved_np = current_np.copy()
            moved_np[3] = np.array([0.4, 0.4, 0.5], dtype=numpy_dtype) @ cell_np
            current = jnp.asarray(moved_np)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_prepared_batched_partial_tile_coo_prepared_cell_metadata(dtype: Any) -> None:
    """Batched prewrapped tile keeps compact COO rows and prepared shift bounds."""
    numpy_dtype = np.float64 if dtype == jnp.float64 else np.float32
    cell_np = np.array(
        [[8.0, 0.0, 0.0], [1.0, 7.0, 0.0], [0.5, 0.75, 6.0]],
        dtype=numpy_dtype,
    )
    fractions = np.array(
        [[0.97, 0.2, 0.05], [0.03, 0.2, 0.05], [0.03, 0.4, 0.5], [0.97, 0.4, 0.5]],
        dtype=numpy_dtype,
    )
    positions_np = fractions @ cell_np
    targets_np = np.array([3, 0, 3], dtype=np.int32)
    cells_np = np.stack((cell_np, cell_np * 1.01))
    pbc_np = np.array([[True, False, True], [True, False, False]])
    batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    targets = jnp.asarray(targets_np)
    state = prepare_neighbor_list(
        jnp.asarray(positions_np),
        0.7,
        cell=jnp.asarray(cells_np),
        pbc=jnp.asarray(pbc_np),
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        method="batch_naive",
        strategy="tile",
        target_indices=targets,
        half_fill=True,
        return_neighbor_list=True,
        coo_capacity=24,
        max_neighbors=8,
        wrap_positions=False,
    )
    assert state.strategy == "tile"

    def execute(values, runtime_cells, prepared):
        return neighbor_list(values, cell=runtime_cells, state=prepared)

    compiled = jax.jit(execute)
    current_positions_np = positions_np.copy()
    first_expected_counts = None
    for step, scale in enumerate((1.0, 1.01)):
        runtime_cells_np = cells_np.copy()
        runtime_cells_np[0, 0, 0] *= scale
        runtime_cells_np[1, 0, 0] *= scale
        if step == 1:
            current_positions_np[2, 0] = 4.0
        runtime_cells = jnp.asarray(runtime_cells_np)
        result, state = compiled(
            jnp.asarray(current_positions_np, dtype=dtype), runtime_cells, state
        )
        expected_rows: list[set[tuple[int, int, int, int]]] = []
        for source in targets_np:
            system = 0 if source < 2 else 1
            start, stop = (0, 2) if system == 0 else (2, 4)
            local_source = int(source) - start
            local_rows = _partial_image_rows(
                current_positions_np[start:stop],
                runtime_cells_np[system],
                pbc_np[system],
                np.array([local_source], dtype=np.int32),
                0.7,
                half_fill=True,
            )
            expected_rows.append(
                {
                    (neighbor + start, sx, sy, sz)
                    for neighbor, sx, sy, sz in local_rows[0]
                }
            )
        expected_counts = list(map(len, expected_rows))
        if step == 0:
            assert expected_counts == [1, 1, 1]
            first_expected_counts = expected_counts
        else:
            assert first_expected_counts == [1, 1, 1]
            assert expected_counts == [0, 1, 0]
        neighbors, pointer, shifts = map(np.asarray, result[:3])
        assert pointer.shape == (len(targets_np) + 1,)
        assert int(pointer[-1]) == sum(expected_counts)
        np.testing.assert_array_equal(np.diff(pointer), expected_counts)
        np.testing.assert_array_equal(np.asarray(state.num_neighbors), expected_counts)
        np.testing.assert_array_equal(np.asarray(result[3]), expected_counts)
        for row, expected in enumerate(expected_rows):
            start, stop = int(pointer[row]), int(pointer[row + 1])
            actual = {
                (int(neighbors[1, index]), *map(int, shifts[index]))
                for index in range(start, stop)
            }
            np.testing.assert_array_equal(
                neighbors[0, start:stop], np.full(stop - start, row, dtype=np.int32)
            )
            assert actual == expected


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_prepared_batched_partial_tile_wrapped_fixed_cell_cache(dtype: Any) -> None:
    """Wrapped batched compact rows reuse fixed geometry with live positions."""
    cells_np = np.stack(
        [np.eye(3, dtype=np.float64 if dtype == jnp.float64 else np.float32) * 8.0] * 2
    )
    positions_np = np.array(
        [[8.1, 1.0, 1.0], [7.9, 1.0, 1.0], [16.1, 1.0, 1.0], [15.9, 1.0, 1.0]],
        dtype=np.float64 if dtype == jnp.float64 else np.float32,
    )
    pbc_np = np.array([[True, False, False], [True, False, False]])
    targets_np = np.array([3, 0, 3], dtype=np.int32)
    batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    cells = jnp.asarray(cells_np)
    positions = jnp.asarray(positions_np)
    pbc = jnp.asarray(pbc_np)
    targets = jnp.asarray(targets_np)
    state = prepare_neighbor_list(
        positions,
        0.5,
        cell=cells,
        pbc=pbc,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        method="batch_naive",
        strategy="tile",
        target_indices=targets,
        max_neighbors=4,
        fixed_cell=True,
        wrap_positions=True,
    )
    execute = jax.jit(
        lambda values, runtime_cells, prepared: neighbor_list(
            values, cell=runtime_cells, state=prepared
        )
    )

    current_np = positions_np.copy()
    for step in range(2):
        result, state = execute(jnp.asarray(current_np), cells, state)
        expected_rows: list[set[tuple[int, int, int, int]]] = []
        for source in targets_np:
            system = 0 if source < 2 else 1
            batch_start = 2 * system
            local_rows = _partial_image_rows(
                current_np[batch_start : batch_start + 2],
                cells_np[system],
                pbc_np[system],
                np.array([int(source) - batch_start], dtype=np.int32),
                0.5,
            )
            expected_rows.append(
                {
                    (neighbor + batch_start, sx, sy, sz)
                    for neighbor, sx, sy, sz in local_rows[0]
                }
            )
        _assert_partial_matrix_matches_image_rows(result, expected_rows)
        if step == 0:
            current_np[2, 0] = 18.0
            current_np[3, 0] = 22.0


def test_prepared_method_none_selects_target_aware_naive_tile() -> None:
    """Automatic preparation accounts for compact work and pins its route."""
    num_atoms = 8192
    positions = jax.random.uniform(
        jax.random.key(19), (num_atoms, 3), dtype=jnp.float32, maxval=100.0
    )
    cell = jnp.eye(3, dtype=jnp.float32) * 100.0
    pbc = jnp.ones(3, dtype=jnp.bool_)
    targets = jnp.array([num_atoms - 1, 0], dtype=jnp.int32)
    full = prepare_neighbor_list(positions, 2.0, cell=cell, pbc=pbc, max_neighbors=8)
    partial = prepare_neighbor_list(
        positions,
        2.0,
        cell=cell,
        pbc=pbc,
        target_indices=targets,
        max_neighbors=8,
    )
    pinned_auto = prepare_neighbor_list(
        positions,
        2.0,
        cell=cell,
        pbc=pbc,
        method="naive",
        strategy="auto",
        target_indices=targets,
        max_neighbors=8,
    )
    assert full.method == "cell_list"
    assert partial.method == "naive" and partial.strategy == "tile"
    assert pinned_auto.method == "naive" and pinned_auto.strategy == "scalar"


@pytest.mark.parametrize("return_neighbor_list", [False, True])
def test_prepared_partial_tile_accepts_empty_targets(
    return_neighbor_list: bool,
) -> None:
    """Empty compact rows retain valid matrix and COO pointer shapes."""
    positions = jnp.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=jnp.float32)
    targets = jnp.empty((0,), dtype=jnp.int32)
    state = prepare_neighbor_list(
        positions,
        0.5,
        method="naive",
        strategy="tile",
        target_indices=targets,
        max_neighbors=4,
        return_neighbor_list=return_neighbor_list,
        coo_capacity=8 if return_neighbor_list else None,
    )
    result, successor = neighbor_list(positions, state=state)
    assert bool(jnp.all(successor.valid))
    if return_neighbor_list:
        assert result[0].shape == (2, 8)
        np.testing.assert_array_equal(np.asarray(result[1]), [0])
        np.testing.assert_array_equal(np.asarray(successor.num_neighbors), [])
    else:
        assert result[0].shape == (0, 4)
        assert result[1].shape == (0,)


@pytest.mark.parametrize(
    ("max_neighbors", "failure_code"), [(1, 1), (8, 2)], ids=["row-and-coo", "coo"]
)
def test_prepared_partial_tile_coo_overflow_is_attributed_and_sticky(
    max_neighbors: int, failure_code: int
) -> None:
    """Compact COO overflows identify target owners and retain first failure."""
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.2, 0.0, 0.0],
            [20.4, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    batch_ptr = jnp.array([0, 3, 6], dtype=jnp.int32)
    batch_idx = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 3)
    targets = jnp.array([5, 0, 5], dtype=jnp.int32)
    state = prepare_neighbor_list(
        positions,
        0.5,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        method="batch_naive",
        strategy="tile",
        target_indices=targets,
        return_neighbor_list=True,
        coo_capacity=1,
        max_neighbors=max_neighbors,
    )
    result, failed = neighbor_list(positions, state=state)
    assert failed.valid.tolist() == [True, False]
    np.testing.assert_array_equal(np.asarray(result[2]), [2, 0, 2])
    with pytest.raises(NeighborOverflowError, match=r"systems \[1\]") as error:
        check_neighbor_list_state(failed)
    assert error.value.system_index == 1
    assert error.value.max_neighbors == (max_neighbors if failure_code == 1 else 1)
    assert error.value.num_neighbors == (2 if failure_code == 1 else 4)

    far = positions.at[3:, 0].set(jnp.array([20.0, 30.0, 40.0]))
    _, later = neighbor_list(far, state=failed)
    assert later.valid.tolist() == [True, False]
    with pytest.raises(NeighborOverflowError, match=r"systems \[1\]") as sticky_error:
        check_neighbor_list_state(later)
    assert sticky_error.value.system_index == 1
    assert sticky_error.value.max_neighbors == error.value.max_neighbors
    assert sticky_error.value.num_neighbors == error.value.num_neighbors


def test_prepared_partial_tile_dynamic_cell_contraction_is_sticky() -> None:
    """Insufficient cached image bounds fail compact tiled reuse permanently."""
    positions = jnp.array([[0.02, 0.0, 0.0], [0.12, 0.0, 0.0]], dtype=jnp.float32)
    prepared_cell = jnp.eye(3, dtype=jnp.float32) * 16.0
    runtime_cell = jnp.eye(3, dtype=jnp.float32) * 16.0
    runtime_cell = runtime_cell.at[0, 0].set(0.1)
    pbc = jnp.array([True, False, False], dtype=jnp.bool_)
    targets = jnp.array([1, 0], dtype=jnp.int32)
    state = prepare_neighbor_list(
        positions,
        0.15,
        cell=prepared_cell,
        pbc=pbc,
        method="naive",
        strategy="tile",
        target_indices=targets,
        max_neighbors=32,
    )
    execute = jax.jit(
        lambda values, box, prepared: neighbor_list(values, cell=box, state=prepared)
    )
    _, failed = execute(positions, runtime_cell, state)
    assert failed.valid.tolist() == [False]
    with pytest.raises(RuntimeError, match="periodic-image coverage.*cached.*runtime"):
        check_neighbor_list_state(failed)
    _, sticky = execute(positions, prepared_cell, failed)
    assert sticky.valid.tolist() == [False]
    with pytest.raises(RuntimeError, match="periodic-image coverage.*cached.*runtime"):
        check_neighbor_list_state(sticky)


def test_prepared_fixed_cell_list_pair_geometry_tracks_live_gradients() -> None:
    """Cached cell-list inverses do not detach live pair vectors from inputs."""
    positions = jnp.array([[0.2, 1.0, 1.0], [7.8, 1.0, 1.0]], dtype=jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    pbc = jnp.array([True, False, False], dtype=jnp.bool_)
    state = prepare_neighbor_list(
        positions,
        0.6,
        cell=cell,
        pbc=pbc,
        method="cell_list",
        strategy="atom_centric",
        max_neighbors=4,
        max_total_cells=512,
        return_vectors=True,
        return_distances=True,
        fixed_cell=True,
    )

    def first_vector_x(current_positions: jax.Array, current_cell: jax.Array):
        result, _ = neighbor_list(
            current_positions,
            cell=current_cell,
            state=state,
        )
        return result[4][0, 0, 0]

    np.testing.assert_allclose(float(first_vector_x(positions, cell)), -0.4, atol=2e-5)
    position_grad = jax.grad(lambda current: first_vector_x(current, cell))(positions)
    cell_grad = jax.grad(lambda current: first_vector_x(positions, current))(cell)
    np.testing.assert_allclose(np.asarray(position_grad[:, 0]), [-1.0, 1.0])
    np.testing.assert_allclose(np.asarray(cell_grad[:, 0]), [-1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("method", "strategy", "dtype"),
    [
        ("cell_list", "atom_centric", jnp.float32),
        ("cell_list", "pair_centric", jnp.float64),
        ("batch_cell_list", "atom_centric", jnp.float64),
        ("batch_cell_list", "pair_centric", jnp.float32),
    ],
)
def test_prepared_fixed_cell_list_routes_match_direct(
    method: str, strategy: str, dtype: Any
) -> None:
    """Fixed cell-list 32/64-bit atom/pair routes reuse the prepared grid."""
    batched = method == "batch_cell_list"
    positions = jnp.array(
        [
            [0.2, 0.0, 0.0],
            [7.8, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    one_cell = jnp.eye(3, dtype=dtype) * 8.0
    one_pbc = jnp.array([True, False, True], dtype=jnp.bool_)
    batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    kwargs: dict[str, Any] = {
        "cell": jnp.broadcast_to(one_cell, (2, 3, 3)) if batched else one_cell,
        "pbc": (
            jnp.array([[True, False, True], [False, False, False]], dtype=jnp.bool_)
            if batched
            else one_pbc
        ),
        "method": method,
        "strategy": strategy,
        "max_neighbors": 16,
        "max_total_cells": 512,
    }
    if batched:
        kwargs["batch_ptr"] = batch_ptr
    state = prepare_neighbor_list(positions, 1.0, fixed_cell=True, **kwargs)
    if batched:
        expected = batch_cell_list(
            positions,
            1.0,
            kwargs["cell"],
            kwargs["pbc"],
            batch_ptr=batch_ptr,
            max_neighbors=16,
            max_total_cells=512,
            strategy=strategy,
        )
    else:
        expected = cell_list(
            positions,
            1.0,
            kwargs["cell"],
            kwargs["pbc"],
            max_neighbors=16,
            max_total_cells=512,
            strategy=strategy,
        )
    actual, successor = neighbor_list(positions, state=state)
    assert successor.fixed_cell
    if method == "batch_cell_list":
        assert _matrix_to_pair_set_full(*actual[:3], len(positions)) == (
            _matrix_to_pair_set_full(*expected[:3], len(positions))
        )
        np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(expected[1]))
    else:
        assert _matrix_to_pair_set_full(*actual[:3], len(positions)) == (
            _matrix_to_pair_set_full(*expected[:3], len(positions))
        )
        np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(expected[1]))


def test_prepared_pair_centric_cell_list_jit_threads_live_positions() -> None:
    """Compilation-eligible pair-centric state threads positions and shifts."""
    positions = jnp.array(
        [
            [0.2, 0.1, 0.2],
            [7.8, 0.1, 0.2],
            [3.0, 1.0, 1.0],
            [3.4, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    moved = positions.at[1].set(jnp.array([6.8, 0.1, 0.2], dtype=jnp.float32))
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    pbc = jnp.array([True, False, True], dtype=jnp.bool_)
    state = prepare_neighbor_list(
        positions,
        0.7,
        cell=cell,
        pbc=pbc,
        method="cell_list",
        strategy="pair_centric",
        max_neighbors=8,
        max_total_cells=512,
        fixed_cell=True,
    )
    assert state.supports_compilation
    assert state.compilation_blocker is None
    compiled = jax.jit(
        lambda current_positions, current_state: neighbor_list(
            current_positions, state=current_state
        )
    )

    first, state1 = compiled(positions, state)
    direct_first = cell_list(
        positions,
        0.7,
        cell,
        pbc,
        max_neighbors=8,
        max_total_cells=512,
        strategy="pair_centric",
    )
    _assert_active_cell_topology_equal(first, direct_first)
    second, state2 = compiled(moved, state1)
    direct_second = cell_list(
        moved,
        0.7,
        cell,
        pbc,
        max_neighbors=8,
        max_total_cells=512,
        strategy="pair_centric",
    )
    _assert_active_cell_topology_equal(second, direct_second)
    assert bool(jnp.all(state2.valid))
    assert state2.fixed_cell


def test_prepared_batched_pair_centric_cell_list_jit_rebuilds_live_positions() -> None:
    """Batched pair-centric JIT refreshes occupancy with default geometry."""
    positions = jnp.array(
        [
            [0.1, 0.1, 0.1],
            [0.5, 0.1, 0.1],
            [4.1, 4.1, 4.1],
            [4.5, 4.1, 4.1],
        ],
        dtype=jnp.float32,
    )
    moved = positions.at[1].set(jnp.array([2.0, 1.0, 1.0], dtype=jnp.float32))
    moved = moved.at[3].set(jnp.array([6.0, 4.1, 4.1], dtype=jnp.float32))
    cell = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32) * 8.0, (2, 3, 3))
    pbc = jnp.zeros((2, 3), dtype=jnp.bool_)
    batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_cell_list",
        strategy="pair_centric",
        max_neighbors=8,
        max_total_cells=1024,
        fixed_cell=False,
    )
    assert state.supports_compilation

    compiled = jax.jit(
        lambda values, runtime_cell, prepared: neighbor_list(
            values, cell=runtime_cell, state=prepared
        )
    )
    first, state1 = compiled(positions, cell, state)
    second, state2 = compiled(moved, cell, state1)
    expected = batch_cell_list(
        moved,
        1.0,
        cell,
        pbc,
        batch_ptr=batch_ptr,
        max_neighbors=8,
        max_total_cells=1024,
        strategy="pair_centric",
    )
    _assert_active_cell_topology_equal(second, expected)
    assert not np.array_equal(np.asarray(first[1]), np.asarray(second[1]))
    assert bool(jnp.all(state2.valid))
    assert not state2.fixed_cell


def test_prepared_fixed_cell_list_rebuilds_occupancy_under_jit_and_default_mode() -> (
    None
):
    """Cached cell geometry still bins moved atoms and both cell modes jit."""
    positions = jnp.array([[0.1, 0.1, 0.1], [4.1, 4.1, 4.1]], dtype=jnp.float32)
    moved = positions.at[1].set(jnp.array([0.3, 0.1, 0.1], dtype=jnp.float32))
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    pbc = jnp.ones((3,), dtype=jnp.bool_)
    options = {
        "cell": cell,
        "pbc": pbc,
        "method": "cell_list",
        "strategy": "atom_centric",
        "max_neighbors": 4,
        "max_total_cells": 512,
    }
    fixed = prepare_neighbor_list(positions, 1.0, fixed_cell=True, **options)
    dynamic = prepare_neighbor_list(positions, 1.0, **options)
    compiled = jax.jit(lambda values, prepared: neighbor_list(values, state=prepared))

    fixed_initial, fixed = compiled(positions, fixed)
    dynamic_initial, dynamic = compiled(positions, dynamic)
    assert int(np.asarray(fixed_initial[1]).sum()) == 0
    assert int(np.asarray(dynamic_initial[1]).sum()) == 0

    fixed_result, fixed = compiled(moved, fixed)
    dynamic_result, dynamic = compiled(moved, dynamic)
    direct = cell_list(
        moved,
        1.0,
        cell,
        pbc,
        max_neighbors=4,
        max_total_cells=512,
        strategy="atom_centric",
    )
    assert int(np.asarray(fixed_result[1]).sum()) == 2
    assert _matrix_to_pair_set_full(*fixed_result[:3], len(positions)) == (
        _matrix_to_pair_set_full(*direct[:3], len(positions))
    )
    assert _matrix_to_pair_set_full(*dynamic_result[:3], len(positions)) == (
        _matrix_to_pair_set_full(*direct[:3], len(positions))
    )
    assert fixed.fixed_cell is True
    assert dynamic.fixed_cell is False


@pytest.mark.parametrize("method", ["cluster_tile", "batch_cluster_tile"])
def test_prepared_fixed_cluster_tile_format_matches_direct(method: str) -> None:
    """Fixed cluster tile output keeps the public single and batch tile layouts."""
    positions_np, cell_np, cutoff = _fixed_cluster_geometry_case("strong_skew")
    positions = jnp.asarray(positions_np)
    cell = jnp.asarray(cell_np)
    pbc = jnp.ones((3,), dtype=jnp.bool_)
    if method == "batch_cluster_tile":
        batch_ptr = jnp.array([0, len(positions_np)], dtype=jnp.int32)
        kwargs = {
            "cell": cell[None],
            "pbc": pbc[None],
            "batch_ptr": batch_ptr,
            "method": method,
        }
        direct = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell[None],
            batch_ptr,
            max_tiles_per_group=1,
            format="tile",
        )
    else:
        kwargs = {"cell": cell, "pbc": pbc, "method": method}
        direct = cluster_tile_neighbor_list(
            positions, cutoff, cell, max_tiles_per_group=1, format="tile"
        )
    state = prepare_neighbor_list(
        positions,
        cutoff,
        fixed_cell=True,
        format="tile",
        max_tiles_per_group=1,
        **kwargs,
    )
    actual, successor = neighbor_list(positions, state=state)
    assert successor.fixed_cell
    assert len(actual) == len(direct)
    active_tiles = int(np.asarray(actual[0])[0])
    for index in (1, 2):
        np.testing.assert_array_equal(
            np.asarray(actual[index])[:active_tiles],
            np.asarray(direct[index])[:active_tiles],
        )
    for index in range(3, len(actual)):
        np.testing.assert_array_equal(
            np.asarray(actual[index]), np.asarray(direct[index])
        )


def _fixed_cluster_expected_geometry(
    positions: np.ndarray, cell: np.ndarray, cutoff: float
) -> dict[tuple[int, int, int, int, int], tuple[np.ndarray, float]]:
    """Attach expected vector geometry to the independent cluster pair oracle."""
    pairs = _brute_force_pairs_full(positions, cell, cutoff, pbc=True)
    result = {}
    for i, j, sx, sy, sz in pairs:
        shift = np.array([sx, sy, sz], dtype=np.int32)
        vector = positions[j] - positions[i] + shift @ cell
        result[(i, j, sx, sy, sz)] = (vector, float(np.linalg.norm(vector)))
    return result


def _fixed_cluster_result_geometry(
    result: tuple[jax.Array, ...], num_atoms: int
) -> dict[tuple[int, int, int, int, int], tuple[np.ndarray, float]]:
    """Map prepared matrix output to directed pair shifts and live geometry."""
    matrix, counts, shifts, distances, vectors = map(np.asarray, result[:5])
    found = {}
    for i in range(num_atoms):
        for slot in range(int(counts[i])):
            shift = tuple(int(value) for value in shifts[i, slot])
            key = (i, int(matrix[i, slot]), *shift)
            found[key] = (vectors[i, slot].copy(), float(distances[i, slot]))
    return found


def _fixed_cluster_geometry_case(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Build discriminating skew, QR-height, translated, and orthogonal cases."""
    if name == "strong_skew":
        cell = np.array(
            [[10.0, 0.0, 0.0], [4.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            dtype=np.float32,
        )
        positions = np.array([[4.0, 5.0, 0.0], [8.3, 3.0, 0.0]], dtype=np.float32)
        return positions, cell, 4.8
    if name == "uncertified_qr_height":
        cell = np.array(
            [[10.0, 0.0, 0.0], [4.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            dtype=np.float32,
        )
        positions = np.array([[0.0, 0.0, 0.0], [4.0, 4.9, 0.0]], dtype=np.float32)
        return positions, cell, 5.11
    if name in {"translated_skew", "rotated_translated_skew"}:
        cell = np.array(
            [[20.0, 0.0, 0.0], [8.0, 20.0, 0.0], [2.0, 4.0, 20.0]],
            dtype=np.float32,
        )
        p0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        delta = cell[0] - cell[1] + np.array([4.5, 0.0, 0.0], dtype=np.float32)
        positions = np.stack((p0, p0 + delta))
        if name == "rotated_translated_skew":
            rotation = np.array(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            cell = cell @ rotation.T
            positions = positions @ rotation.T
        return positions, cell, 4.8
    if name == "orthogonal_nonzero_image":
        cell = np.eye(3, dtype=np.float32) * 10.0
        positions = np.array([[0.1, 1.0, 1.0], [9.9, 1.0, 1.0]], dtype=np.float32)
        return positions, cell, 0.5
    raise AssertionError(f"unexpected cluster fixture {name!r}")


@pytest.mark.parametrize(
    "case",
    [
        "strong_skew",
        "uncertified_qr_height",
        "translated_skew",
        "rotated_translated_skew",
        "orthogonal_nonzero_image",
    ],
)
def test_prepared_fixed_cluster_matches_lattice_oracle(case: str) -> None:
    """Cached QR geometry preserves exact pairs, images, vectors, and distances."""
    positions_np, cell_np, cutoff = _fixed_cluster_geometry_case(case)
    positions, cell = jnp.asarray(positions_np), jnp.asarray(cell_np)
    state = prepare_neighbor_list(
        positions,
        cutoff,
        cell=cell,
        pbc=jnp.ones((3,), dtype=jnp.bool_),
        method="cluster_tile",
        max_neighbors=8,
        max_tiles_per_group=1,
        return_vectors=True,
        return_distances=True,
        fixed_cell=True,
    )
    expected = _fixed_cluster_expected_geometry(positions_np, cell_np, cutoff)
    result, successor = neighbor_list(
        positions, cell=cell + jnp.zeros_like(cell), state=state
    )
    actual = _fixed_cluster_result_geometry(result, len(positions_np))
    assert actual.keys() == expected.keys()
    for key, (expected_vector, expected_distance) in expected.items():
        actual_vector, actual_distance = actual[key]
        np.testing.assert_allclose(actual_vector, expected_vector, atol=3e-5)
        np.testing.assert_allclose(actual_distance, expected_distance, atol=3e-5)
    assert successor.fixed_cell
    repeated, repeated_state = neighbor_list(positions, state=successor)
    assert repeated_state.fixed_cell
    assert _fixed_cluster_result_geometry(repeated, len(positions_np)).keys() == (
        expected.keys()
    )


def test_prepared_fixed_cluster_outer_cutoff_and_live_geometry_gradients() -> None:
    """Dual cutoff caches the outer geometry and pair gradients use the live cell."""
    positions_np, cell_np, outer_cutoff = _fixed_cluster_geometry_case("strong_skew")
    positions, cell = jnp.asarray(positions_np), jnp.asarray(cell_np)
    for primary_cutoff in (2.0, outer_cutoff):
        dual = prepare_neighbor_list(
            positions,
            primary_cutoff,
            cutoff2=outer_cutoff,
            cell=cell,
            pbc=jnp.ones((3,), dtype=jnp.bool_),
            method="cluster_tile",
            max_neighbors=8,
            max_tiles_per_group=1,
            fixed_cell=True,
        )
        dual_result, _ = neighbor_list(positions, state=dual)
        assert _matrix_to_pair_set_full(*dual_result[:3], len(positions_np)) == (
            _brute_force_pairs_full(positions_np, cell_np, primary_cutoff, pbc=True)
        )
        assert _matrix_to_pair_set_full(*dual_result[3:6], len(positions_np)) == (
            _brute_force_pairs_full(positions_np, cell_np, outer_cutoff, pbc=True)
        )

    positions_np, cell_np, cutoff = _fixed_cluster_geometry_case("translated_skew")
    positions, cell = jnp.asarray(positions_np), jnp.asarray(cell_np)
    pbc = jnp.ones((3,), dtype=jnp.bool_)
    state = prepare_neighbor_list(
        positions,
        cutoff,
        cell=cell,
        pbc=pbc,
        method="cluster_tile",
        max_neighbors=4,
        max_tiles_per_group=1,
        return_vectors=True,
        return_distances=True,
        fixed_cell=True,
    )

    def first_vector_x(current_cell: jax.Array) -> jax.Array:
        current, _ = neighbor_list(positions, cell=current_cell, state=state)
        return current[4][0, 0, 0]

    def rebuilt_first_vector_x(current_cell: jax.Array) -> jax.Array:
        perturbed_state = prepare_neighbor_list(
            positions,
            cutoff,
            cell=current_cell,
            pbc=pbc,
            method="cluster_tile",
            max_neighbors=4,
            max_tiles_per_group=1,
            return_vectors=True,
            return_distances=True,
            fixed_cell=True,
        )
        current, _ = neighbor_list(positions, cell=current_cell, state=perturbed_state)
        return current[4][0, 0, 0]

    vector_cell_grad = jax.grad(first_vector_x)(cell)
    vector_pos_grad = jax.grad(
        lambda current_positions: neighbor_list(
            current_positions, cell=cell, state=state
        )[0][4][0, 0, 0]
    )(positions)
    distance_cell_grad = jax.grad(
        lambda current_cell: neighbor_list(positions, cell=current_cell, state=state)[
            0
        ][3][0, 0]
    )(cell)
    np.testing.assert_allclose(np.asarray(vector_cell_grad[:, 0]), [-1.0, 1.0, 0.0])
    np.testing.assert_allclose(np.asarray(vector_pos_grad[:, 0]), [-1.0, 1.0])
    np.testing.assert_allclose(np.asarray(distance_cell_grad[:, 0]), [-1.0, 1.0, 0.0])

    epsilon = jnp.float32(1e-2)
    for row, expected in ((0, -1.0), (1, 1.0)):
        plus = rebuilt_first_vector_x(cell.at[row, 0].add(epsilon))
        minus = rebuilt_first_vector_x(cell.at[row, 0].add(-epsilon))
        finite_difference = (plus - minus) / (2.0 * epsilon)
        np.testing.assert_allclose(float(finite_difference), expected, atol=2e-4)


def test_prepared_fixed_cluster_jit_donation_successor_and_callback() -> None:
    """Fixed cluster modules work in JIT and retain caches through donated reuse."""
    positions_np, cell_np, cutoff = _fixed_cluster_geometry_case(
        "uncertified_qr_height"
    )
    positions, cell = jnp.asarray(positions_np), jnp.asarray(cell_np)
    state = prepare_neighbor_list(
        positions,
        cutoff,
        cell=cell,
        pbc=jnp.ones((3,), dtype=jnp.bool_),
        method="cluster_tile",
        max_neighbors=8,
        max_tiles_per_group=1,
        fixed_cell=True,
    )
    expected = _brute_force_pairs_full(positions_np, cell_np, cutoff, pbc=True)
    compiled = jax.jit(
        lambda current_positions, prepared: neighbor_list(
            current_positions, state=prepared
        ),
        donate_argnums=(1,),
    )
    first, state1 = compiled(positions, state)
    first_expected = _matrix_to_pair_set_full(*first[:3], len(positions_np))
    assert first_expected == expected
    second, state2 = compiled(positions, state1)
    assert _matrix_to_pair_set_full(*second[:3], len(positions_np)) == expected
    assert state2.fixed_cell

    callback_cell = jnp.asarray(np.array(cell_np, copy=True))
    callback_state = prepare_neighbor_list(
        positions,
        cutoff,
        cell=callback_cell,
        pbc=jnp.ones((3,), dtype=jnp.bool_),
        method="cluster_tile",
        max_neighbors=8,
        max_tiles_per_group=1,
        pair_fn=_prepared_pair_fn,
        fixed_cell=True,
    )
    pair_params = jnp.arange(len(positions_np), dtype=jnp.float32)[:, None]
    callback_result, _ = neighbor_list(
        positions, state=callback_state, pair_params=pair_params
    )
    direct_result = cluster_tile_neighbor_list(
        positions,
        cutoff,
        callback_cell,
        max_neighbors=8,
        max_tiles_per_group=1,
        pair_fn=_prepared_pair_fn,
        pair_params=pair_params,
    )
    assert _matrix_to_pair_set_full(
        *direct_result[:3], len(positions_np)
    ) == _matrix_to_pair_set_full(*callback_result[:3], len(positions_np))
    np.testing.assert_array_equal(callback_result[1], direct_result[1])
    for row, count in enumerate(np.asarray(callback_result[1])):
        active = int(count)
        for output_index in (3, 4):
            np.testing.assert_allclose(
                np.asarray(callback_result[output_index])[row, :active],
                np.asarray(direct_result[output_index])[row, :active],
                atol=3e-5,
            )


def test_prepared_fixed_cell_overflow_remains_sticky_after_reuse() -> None:
    """Fixed geometry retains capacity failures across a later successful call."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=jnp.float32
    )
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    state = prepare_neighbor_list(
        positions,
        0.75,
        cell=cell,
        pbc=jnp.ones((3,), dtype=jnp.bool_),
        method="naive",
        max_neighbors=1,
        fixed_cell=True,
    )
    _overflow_result, failed = neighbor_list(positions, state=state)
    assert not bool(jnp.all(failed.valid))
    with pytest.raises(NeighborOverflowError):
        check_neighbor_list_state(failed)
    separated = jnp.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=jnp.float32
    )
    _successful_result, still_failed = neighbor_list(separated, state=failed)
    assert not bool(jnp.all(still_failed.valid))
    with pytest.raises(NeighborOverflowError):
        check_neighbor_list_state(still_failed)


def test_prepared_fixed_cell_selective_all_false_survives_donated_jit() -> None:
    """An all-false selective call carries fixed caches through donation."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [3.0, 0.0, 0.0], [3.4, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=jnp.ones((3,), dtype=jnp.bool_),
        method="naive",
        selective=True,
        return_neighbor_list=True,
        coo_capacity=16,
        max_neighbors=8,
        fixed_cell=True,
    )
    initial_result, initialized = neighbor_list(
        positions,
        state=state,
        rebuild_flags=jnp.ones((1,), dtype=jnp.bool_),
    )
    assert initialized.fixed_cell
    preserve = jnp.zeros((1,), dtype=jnp.bool_)
    execute = jax.jit(
        lambda values, prepared, flags: neighbor_list(
            values, state=prepared, rebuild_flags=flags
        ),
        donate_argnums=(1,),
    )
    preserved_result, successor = execute(positions, initialized, preserve)
    for expected, actual in zip(initial_result, preserved_result, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert successor.fixed_cell
    assert bool(jnp.all(successor.initialized))
    assert bool(jnp.all(successor.valid))
