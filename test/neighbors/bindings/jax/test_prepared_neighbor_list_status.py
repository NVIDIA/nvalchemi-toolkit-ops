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

"""Status-tail tests for JAX prepared neighbor-list routes."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nvalchemiops.jax.neighbors.cluster_tile as _cluster_tile
from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list
from nvalchemiops.jax.neighbors.cell_list import cell_list
from nvalchemiops.jax.neighbors.cluster_tile import (
    build_cluster_tile_list,
    cluster_tile_neighbor_list,
    query_cluster_tile_coo,
)
from nvalchemiops.neighbors.neighbor_utils import NeighborOverflowError

from .conftest import requires_gpu

pytestmark = requires_gpu


def _cell_inputs(num_atoms: int = 4) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return a small fully periodic single-system geometry."""
    positions = jnp.arange(num_atoms * 3, dtype=jnp.float32).reshape(num_atoms, 3)
    cell = jnp.eye(3, dtype=jnp.float32) * 32.0
    pbc = jnp.ones(3, dtype=jnp.bool_)
    return positions, cell, pbc


def test_cell_status_suffix_preserves_public_prefix_under_jit() -> None:
    """The diagnostic suffix does not alter the public cell-list tuple."""
    positions, cell, pbc = _cell_inputs()

    def call(with_status: bool) -> tuple[jax.Array, ...]:
        return cell_list(
            positions,
            2.0,
            cell,
            pbc,
            max_neighbors=32,
            max_total_cells=216,
            _return_status=with_status,
        )

    public = jax.jit(lambda: call(False))()
    diagnostic = jax.jit(lambda: call(True))()
    assert len(public) == 3
    assert len(diagnostic) == 14
    for expected, actual in zip(public, diagnostic[:3], strict=True):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
    for value in diagnostic[3:]:
        assert value.shape == (1,)


def test_batch_cell_status_suffix_has_one_value_per_system() -> None:
    """Batched cell diagnostics use exact per-system tail shapes."""
    positions, cell, pbc = _cell_inputs(6)
    batch_ptr = jnp.array([0, 3, 6], dtype=jnp.int32)
    batch_idx = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 3)
    out = jax.jit(
        lambda: batch_cell_list(
            positions,
            2.0,
            jnp.broadcast_to(cell, (2, 3, 3)),
            jnp.broadcast_to(pbc, (2, 3)),
            batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=32,
            max_total_cells=432,
            _return_status=True,
        )
    )()
    assert len(out) == 14
    for value in out[3:]:
        assert value.shape == (2,)


def test_cell_status_reports_real_row_and_coo_failures() -> None:
    """Cell diagnostics distinguish row width from fixed-COO capacity."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    cell = jnp.eye(3, dtype=jnp.float32) * 10.0
    pbc = jnp.zeros(3, dtype=jnp.bool_)
    row = cell_list(
        positions,
        1.0,
        cell,
        pbc,
        max_neighbors=1,
        max_total_cells=8,
        _return_status=True,
    )
    row_tail = row[-11:]
    assert bool(jnp.all(row_tail[1]))
    assert not bool(jnp.any(row_tail[2]))
    assert int(row_tail[5][0]) > int(row_tail[6][0])

    coo = cell_list(
        positions,
        1.0,
        cell,
        pbc,
        max_neighbors=8,
        max_total_cells=8,
        return_neighbor_list=True,
        coo_capacity=1,
        _return_status=True,
    )
    coo_tail = coo[-11:]
    assert not bool(jnp.any(coo_tail[1]))
    assert bool(jnp.all(coo_tail[2]))
    assert int(coo_tail[7][0]) > int(coo_tail[8][0])


def test_cell_status_reports_stale_metadata_independently() -> None:
    """Stale pair-centric metadata is visible even when rows fit."""
    positions, cell, pbc = _cell_inputs()
    out = jax.jit(
        lambda: cell_list(
            positions,
            2.0,
            cell,
            pbc,
            max_neighbors=32,
            max_total_cells=216,
            strategy="pair_centric",
            pair_centric_n_outer=0,
            return_neighbor_list=True,
            coo_capacity=64,
            _return_status=True,
        )
    )()
    tail = out[-11:]
    assert bool(jnp.all(tail[3]))
    assert int(tail[0][0]) == 3


def test_cluster_status_reports_tile_and_coo_values() -> None:
    """Cluster diagnostics retain real tile/COO required and capacity values."""
    positions, cell, _pbc = _cell_inputs(128)
    tile = cluster_tile_neighbor_list(
        positions,
        2.0,
        cell,
        max_neighbors=256,
        max_tiles_per_group=1,
        _return_status=True,
    )
    tile_tail = tile[-6:]
    assert bool(jnp.all(tile_tail[0]))
    assert int(tile_tail[2][0]) > int(tile_tail[3][0])

    coo = cluster_tile_neighbor_list(
        positions,
        2.0,
        cell,
        max_neighbors=256,
        max_pairs=1,
        max_tiles_per_group=256,
        format="coo",
        _return_status=True,
    )
    coo_tail = coo[-6:]
    assert not bool(jnp.any(coo_tail[0]))
    assert bool(jnp.all(coo_tail[1]))
    assert int(coo_tail[4][0]) > int(coo_tail[5][0])


@pytest.mark.parametrize("max_pairs", [0, 1])
def test_cluster_status_coo_uses_one_bounded_counter_launch(
    monkeypatch, max_pairs
) -> None:
    """Compact COO status derives its raw count from the bounded launch."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    cell = jnp.eye(3, dtype=jnp.float32) * 10.0
    full = cluster_tile_neighbor_list(
        positions,
        2.0,
        cell,
        max_neighbors=8,
        max_pairs=64,
        max_tiles_per_group=256,
        format="coo",
    )
    # Compact COO keeps its direct eager overflow exception.  The private
    # status route is the bounded alternative used by prepared execution.
    with pytest.raises(NeighborOverflowError):
        cluster_tile_neighbor_list(
            positions,
            2.0,
            cell,
            max_neighbors=8,
            max_pairs=max_pairs,
            max_tiles_per_group=256,
            format="coo",
        )
    capacities = []
    original_query = _cluster_tile.query_cluster_tile_coo

    def spy_query(*args: Any, **kwargs: Any) -> Any:
        capacities.append(args[10])
        return original_query(*args, **kwargs)

    monkeypatch.setattr(_cluster_tile, "query_cluster_tile_coo", spy_query)
    status = cluster_tile_neighbor_list(
        positions,
        2.0,
        cell,
        max_neighbors=8,
        max_pairs=max_pairs,
        max_tiles_per_group=256,
        format="coo",
        _return_status=True,
    )
    assert capacities == [max_pairs]
    assert int(status[-2][0]) == full[0].shape[1]
    assert int(status[-1][0]) == max_pairs


def test_cluster_status_suffix_and_private_pair_counter() -> None:
    """Cluster diagnostics are six arrays and pair counters stay private."""
    positions, cell, pbc = _cell_inputs()
    del pbc
    tile = build_cluster_tile_list(
        positions,
        2.0,
        cell,
        max_tiles_per_group=256,
    )
    compact = query_cluster_tile_coo(
        tile[0],
        tile[2],
        tile[3],
        tile[4],
        tile[11],
        tile[12],
        tile[13],
        cell,
        2.0,
        positions.shape[0],
        64,
    )
    with_counter = query_cluster_tile_coo(
        tile[0],
        tile[2],
        tile[3],
        tile[4],
        tile[11],
        tile[12],
        tile[13],
        cell,
        2.0,
        positions.shape[0],
        64,
        _return_pair_counter=True,
    )
    assert len(compact) == 3
    assert len(with_counter) == 4
    assert with_counter[-1].shape == (1,)

    status = cluster_tile_neighbor_list(
        positions,
        2.0,
        cell,
        max_neighbors=32,
        max_tiles_per_group=256,
        _return_status=True,
    )
    assert len(status) == 9
    for value in status[-6:]:
        assert value.shape == (1,)


def test_cluster_status_preserves_default_matrix_prefix() -> None:
    """Private status collection leaves the public direct matrix prefix intact."""
    positions, cell, _pbc = _cell_inputs()
    direct = cluster_tile_neighbor_list(
        positions, 2.0, cell, max_neighbors=32, max_tiles_per_group=256
    )
    status = cluster_tile_neighbor_list(
        positions,
        2.0,
        cell,
        max_neighbors=32,
        max_tiles_per_group=256,
        _return_status=True,
    )
    assert len(direct) == 3
    assert len(status) == 9
    for expected, actual in zip(direct, status[:3], strict=True):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
