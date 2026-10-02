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

"""Exercise eager JAX grid selection through the public neighbor bindings."""

import cProfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors import neighbor_list
from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list
from nvalchemiops.jax.neighbors.cell_list import cell_list

from .conftest import requires_gpu

pytestmark = requires_gpu


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_single_grid_rejects_more_neighbor_work(dtype):
    """A shorter stencil preserves the configured grid when neighbor work grows."""
    from nvalchemiops.jax.neighbors._cell_grid import _select_pair_grid

    cell = jnp.eye(3, dtype=dtype)[None] * 12
    pbc = jnp.ones((1, 3), dtype=jnp.bool_)
    ptr = jnp.asarray([0, 512], dtype=jnp.int32)
    grid, _, _ = _select_pair_grid(cell, pbc, ptr, 6.0, 64, single_system=True)
    np.testing.assert_array_equal(grid, [[4, 4, 4]])


def _pairs(result, compact):
    """Return sorted source, target, and image rows independent of grid order."""
    matrix, counts, shifts = map(np.asarray, result[:3])
    if compact:
        length = int(counts[-1])
        rows = np.column_stack((matrix[:, :length].T, shifts[:length]))
    else:
        assert np.all(counts <= matrix.shape[1])
        source, slot = np.nonzero(np.arange(matrix.shape[1])[None, :] < counts[:, None])
        rows = np.column_stack((source, matrix[source, slot], shifts[source, slot]))
    return rows[np.lexsort(rows.T[::-1])]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("strategy", ["pair_centric", "auto"])
def test_public_eager_selector_matches_pairs(batched, compact, dtype, skew, strategy):
    """Eager calls select current grids and preserve pairs as geometry changes."""
    atoms = 512
    systems = 2 if batched else 1
    cell = np.eye(3) * 17.445
    if skew:
        cell[1, 0] = cell[0, 0] * 0.2
    fractions = np.random.default_rng(29).random((systems, atoms, 3))
    base = (fractions @ cell).reshape(-1, 3)
    pbc = np.ones((systems, 3), dtype=bool)
    if skew:
        pbc[0, 1] = False
    call = batch_cell_list if batched else cell_list
    common = dict(
        cutoff=6.0,
        max_neighbors=atoms,
        max_total_cells=64 * systems,
        pbc=jnp.asarray(pbc),
        return_neighbor_list=compact,
    )
    if batched:
        common.update(
            batch_ptr=jnp.arange(systems + 1, dtype=jnp.int32) * atoms,
            batch_idx=jnp.repeat(jnp.arange(systems, dtype=jnp.int32), atoms),
        )
    for scale in (1.0, 0.9, 1.1):
        kwargs = dict(
            common,
            positions=jnp.asarray(base * scale, dtype=dtype),
            cell=jnp.asarray(np.tile(cell[None], (systems, 1, 1)) * scale, dtype=dtype),
        )
        reference = call(**kwargs, strategy="atom_centric")
        profiler = cProfile.Profile()
        with profiler:
            actual = call(**kwargs, strategy=strategy)
            jax.block_until_ready(actual)
        selections = sum(
            e.callcount
            for e in profiler.getstats()
            if getattr(e.code, "co_name", "") == "_select_pair_grid"
        )
        assert selections == 1, "Public eager pair call must use the grid selector"
        np.testing.assert_array_equal(
            _pairs(actual, compact), _pairs(reference, compact)
        )


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_static_launch_metadata_keeps_existing_grid(batched, compiled):
    """Supplied launch metadata remains valid in eager and traced public calls."""
    from nvalchemiops.jax.neighbors.batch_cell_list import batch_build_cell_list
    from nvalchemiops.jax.neighbors.cell_list import build_cell_list
    from nvalchemiops.neighbors.cell_list import compute_batch_pair_centric_n_outer

    atoms = 64
    systems = 2 if batched else 1
    cell = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None] * 12, (systems, 1, 1))
    pbc = jnp.ones((systems, 3), dtype=jnp.bool_)
    positions = jnp.asarray(
        np.random.default_rng(41).random((atoms * systems, 3)) * 12, dtype=jnp.float32
    )
    common = dict(cell=cell, pbc=pbc, cutoff=3.0, max_total_cells=64 * systems)
    if batched:
        common.update(
            batch_ptr=jnp.arange(systems + 1, dtype=jnp.int32) * atoms,
            batch_idx=jnp.repeat(jnp.arange(systems, dtype=jnp.int32), atoms),
        )
    build = batch_build_cell_list if batched else build_cell_list
    built = build(positions, **common)
    radius = tuple(np.asarray(built[6]).reshape(-1, 3).max(axis=0).tolist())
    call = batch_cell_list if batched else cell_list
    common.update(
        max_neighbors=atoms,
        strategy="pair_centric",
        return_neighbor_list=False,
        pair_centric_n_outer=compute_batch_pair_centric_n_outer(radius, False),
    )
    if batched:
        common.update(
            pair_centric_total_cells=int(np.prod(np.asarray(built[0]), axis=1).sum()),
            pair_centric_r_max=radius,
        )
    reference = call(positions, **common)

    def query(pos):
        return call(pos, **common)

    query_fn = jax.jit(query) if compiled else query
    profiler = cProfile.Profile()
    with profiler:
        actual = query_fn(positions)
        jax.block_until_ready(actual)
    assert not any(
        getattr(e.code, "co_name", "") == "_select_pair_grid"
        for e in profiler.getstats()
    )
    np.testing.assert_array_equal(_pairs(actual, False), _pairs(reference, False))


def test_selector_reuses_executable_with_current_boxes():
    """Changing geometry updates grids while the warmed executable is reused."""
    from nvalchemiops.jax.neighbors._cell_grid import _select_pair_grid

    cell = jnp.eye(3, dtype=jnp.float32)[None] * 17.445
    pbc = jnp.ones((1, 3), dtype=jnp.bool_)
    boundaries = jnp.asarray([0, 512], dtype=jnp.int32)
    for scale in (1.0, 1.1):
        jax.block_until_ready(_select_pair_grid(cell * scale, pbc, boundaries, 6.0, 64))
    profiler = cProfile.Profile()
    with profiler:
        first = _select_pair_grid(cell, pbc, boundaries, 6.0, 64)
        second = _select_pair_grid(cell * 1.1, pbc, boundaries, 6.0, 64)
        jax.block_until_ready((first, second))
    assert not any(
        getattr(e.code, "co_name", "") == "backend_compile_and_load"
        for e in profiler.getstats()
    )
    np.testing.assert_array_equal(first[0], [[2, 2, 2]])
    np.testing.assert_array_equal(second[0], [[3, 3, 3]])
    constrained = _select_pair_grid(cell, pbc, boundaries, 6.0, 7)
    assert int(constrained[2][0]) <= 7
    with pytest.raises(ValueError, match="at least one cell per system"):
        _select_pair_grid(cell, pbc, boundaries, 6.0, 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize(
    "atoms,side,cutoff",
    [(512, 17.445, 15.0), (686, 28.833, 25.0), (4394, 53.547, 25.0)],
)
def test_wide_search_retains_configured_grid(dtype, atoms, side, cutoff):
    """Wide searches retain the configured grid instead of crowded cells."""
    from nvalchemiops.jax.neighbors._cell_grid import _select_pair_grid

    cell = jnp.eye(3, dtype=dtype)[None] * side
    pbc = jnp.ones((1, 3), dtype=jnp.bool_)
    boundaries = jnp.asarray([0, atoms], dtype=jnp.int32)
    grids, _, counts = _select_pair_grid(
        cell, pbc, boundaries, cutoff, 64, single_system=True
    )
    np.testing.assert_array_equal(grids, [[4, 4, 4]])
    np.testing.assert_array_equal(counts, [64])


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("public", [False, True])
def test_automatic_capacity_matches_atom_centric(batched, public):
    """Estimator-produced scalar capacities work through public eager calls."""
    atoms = 64
    systems = 2 if batched else 1
    positions = jnp.asarray(
        np.random.default_rng(37).random((atoms * systems, 3)) * 20,
        dtype=jnp.float32,
    )
    kwargs = dict(
        cell=jnp.tile(jnp.eye(3, dtype=jnp.float32)[None] * 20, (systems, 1, 1)),
        pbc=jnp.ones((systems, 3), dtype=jnp.bool_),
        cutoff=5.0,
        max_neighbors=atoms,
        return_neighbor_list=True,
    )
    if batched:
        kwargs.update(batch_ptr=jnp.arange(systems + 1, dtype=jnp.int32) * atoms)
    call = batch_cell_list if batched else cell_list
    expected = call(positions, **kwargs, strategy="atom_centric")
    if public:
        kwargs["method"] = "batch_cell_list" if batched else "cell_list"
        actual = neighbor_list(positions, **kwargs)
    else:
        actual = call(positions, **kwargs, strategy="pair_centric")
    np.testing.assert_array_equal(_pairs(actual, True), _pairs(expected, True))
