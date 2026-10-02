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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from nvalchemiops.jax.neighbors import neighbor_list
from nvalchemiops.jax.neighbors.batch_cluster_tile import (
    batch_cluster_tile_neighbor_list,
)
from nvalchemiops.jax.neighbors.cell_list import cell_list
from nvalchemiops.jax.neighbors.cluster_tile import cluster_tile_neighbor_list
from nvalchemiops.jax.neighbors.neighbor_utils import (
    NeighborOverflowError,
    TileBufferOverflow,
)
from nvalchemiops.jax.neighbors.prepared_neighbor_list import (
    NeighborListState,
    check_neighbor_list_state,
    prepare_neighbor_list,
)

from .conftest import requires_gpu

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
    state = prepare_neighbor_list(
        initial,
        1.0,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        method="batch_naive",
        return_neighbor_list=True,
        coo_capacity=16,
        max_neighbors=8,
        selective=True,
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


@pytest.mark.parametrize("method", ["cluster_tile", "batch_cluster_tile"])
@pytest.mark.parametrize("format", ["matrix", "coo"])
def test_prepared_cluster_geometry_and_callback_outputs_are_stateful(
    method: str, format: str
) -> None:
    """Prepared cluster geometry/callback tails survive successor threading."""
    positions, cell, pbc = _cell_inputs(4)
    kwargs: dict[str, Any] = {
        "cell": cell if method == "cluster_tile" else jnp.broadcast_to(cell, (1, 3, 3)),
        "pbc": pbc if method == "cluster_tile" else jnp.broadcast_to(pbc, (1, 3)),
        "method": method,
        "max_neighbors": 8,
        "max_tiles_per_group": 1,
        "return_distances": True,
        "return_vectors": True,
        "pair_fn": _prepared_pair_fn,
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
def test_prepared_pair_output_cluster_tile_overflow_is_sticky(method: str) -> None:
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
    state = prepare_neighbor_list(
        positions,
        1.0,
        method="naive",
        selective=True,
        return_neighbor_list=True,
        coo_capacity=8,
        max_neighbors=8,
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
