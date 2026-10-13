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
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from nvalchemiops.jax.neighbors import neighbor_list
from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list
from nvalchemiops.jax.neighbors.cell_list import cell_list

from .conftest import requires_gpu
from .test_pair_fn import _PAIR_FN, _pair_params

pytestmark = requires_gpu


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize(
    "return_vectors,return_distances", [(True, False), (False, True), (True, True)]
)
def test_pair_output_grid_policy_gpu_execution(
    batched, public, return_vectors, return_distances
):
    """Auto pair outputs retain configured GPU work; explicit pairs opt in."""
    systems = 2 if batched else 1
    atoms_per_system = 32
    atoms = systems * atoms_per_system
    positions = jnp.asarray(
        np.random.default_rng(89).random((atoms, 3)) * 12, dtype=jnp.float32
    )
    common = dict(
        cutoff=6.0,
        cell=jnp.tile(jnp.eye(3, dtype=jnp.float32)[None] * 12, (systems, 1, 1)),
        pbc=jnp.ones((systems, 3), dtype=jnp.bool_),
        max_neighbors=64,
        max_total_cells=64 * systems,
        return_vectors=return_vectors,
        return_distances=return_distances,
    )
    if batched:
        common.update(
            batch_idx=jnp.repeat(
                jnp.arange(systems, dtype=jnp.int32), atoms_per_system
            ),
            batch_ptr=jnp.arange(systems + 1, dtype=jnp.int32) * atoms_per_system,
        )
    call = neighbor_list if public else (batch_cell_list if batched else cell_list)

    def record(policy, strategy):
        """Record warmed Warp GPU launches with all returned work completed."""
        options = dict(common, grid_policy=policy)
        if public:
            method = "batch_cell_list" if batched else "cell_list"
            options["method"] = (
                f"{method}_pair_centric" if strategy == "pair_centric" else method
            )
        elif strategy is not None:
            options["strategy"] = strategy
        jax.block_until_ready(call(positions, **options))
        with wp.ScopedTimer(
            "pair-output grid policy", cuda_filter=wp.TIMING_KERNEL, print=False
        ) as timer:
            jax.block_until_ready(call(positions, **options))
        launches = Counter(event.name for event in timer.timing_results)
        assert launches, "The GPU recorder must observe warmed Warp launches"
        return launches

    # Compare observed operations between policies, without hard-coding kernel
    # names or inspecting Python frames. The explicit-pair control must expose
    # adaptive-only GPU work, so a missing selector observation cannot pass.
    configured_pair = record("configured", "pair_centric")
    adaptive_pair = record("adaptive", "pair_centric")
    adaptive_only = adaptive_pair - configured_pair
    assert adaptive_only, "Explicit pair-centric calls must execute adaptive GPU work"
    for strategy in (None,) if public else (None, "auto"):
        configured = record("configured", strategy)
        adaptive = record("adaptive", strategy)
        assert adaptive == configured, (
            "Auto pair-output calls must retain configured GPU work"
        )
        assert not (adaptive & adaptive_only)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize(
    "public,strategy",
    [
        pytest.param(False, None, id="direct-default"),
        pytest.param(False, "auto", id="direct-auto"),
        pytest.param(False, "atom_centric", id="direct-atom"),
        pytest.param(False, "pair_centric", id="direct-pair"),
        pytest.param(True, "auto", id="dispatcher-auto"),
        pytest.param(True, "atom_centric", id="dispatcher-atom"),
        pytest.param(True, "pair_centric", id="dispatcher-pair"),
    ],
)
@pytest.mark.parametrize(
    "layout,outputs",
    [
        ("matrix", "vectors"),
        ("matrix", "distances"),
        ("matrix", "both"),
        ("matrix", "buffers"),
        ("matrix", "pair_fn"),
        ("matrix", "pair_fn_geometry"),
        ("coo", "vectors"),
        ("coo", "distances"),
        ("coo", "both"),
        ("coo", "pair_fn"),
        ("coo", "pair_fn_geometry"),
        ("fixed_coo", "both"),
        ("fixed_coo", "pair_fn"),
        ("fixed_coo", "pair_fn_geometry"),
    ],
)
def test_pair_output_grid_policy_matches_query(
    batched, dtype, public, strategy, layout, outputs
):
    """Grid preparation matches the executed query across interacting options."""
    populations = [17, 0, 32] if batched else [32]
    atoms = sum(populations)
    systems = len(populations)
    positions = jnp.asarray(np.random.default_rng(73).random((atoms, 3)) * 12, dtype)
    cell = jnp.tile(jnp.eye(3, dtype=dtype)[None] * 12, (systems, 1, 1))
    batch_indices = np.repeat(np.arange(systems), populations)
    common = dict(
        positions=positions,
        cutoff=6.0,
        cell=cell,
        pbc=jnp.ones((systems, 3), dtype=jnp.bool_),
        max_neighbors=64,
        max_total_cells=64 * systems,
        return_vectors=outputs in {"vectors", "both", "pair_fn_geometry"},
        return_distances=outputs in {"distances", "both", "pair_fn_geometry"},
    )
    if batched:
        common.update(
            batch_idx=jnp.asarray(batch_indices, dtype=jnp.int32),
            batch_ptr=jnp.asarray(np.cumsum([0, *populations]), dtype=jnp.int32),
        )
    if outputs == "buffers":
        common["neighbor_vectors"] = jnp.zeros((atoms, 64, 3), dtype=dtype)
        common["neighbor_distances"] = jnp.zeros((atoms, 64), dtype=dtype)
    fused = outputs in {"pair_fn", "pair_fn_geometry"}
    if fused:
        common.update(pair_fn=_PAIR_FN[dtype], pair_params=_pair_params(atoms, dtype))
    direct = batch_cell_list if batched else cell_list
    expected = direct(**common, strategy="atom_centric", grid_policy="configured")
    compact = layout != "matrix"
    common["return_neighbor_list"] = compact
    if layout == "fixed_coo":
        common["coo_capacity"] = atoms * 64
    call = neighbor_list if public else direct
    options = {}
    if public:
        method = "batch_cell_list" if batched else "cell_list"
        options["method"] = method if strategy == "auto" else f"{method}_{strategy}"
    elif strategy is not None:
        options["strategy"] = strategy
    for policy in (None, "configured", "adaptive"):
        policy_options = {} if policy is None else {"grid_policy": policy}
        actual = call(**common, **options, **policy_options)
        jax.block_until_ready(actual)
        counts = np.diff(np.asarray(actual[1])) if compact else actual[1]
        np.testing.assert_array_equal(counts, expected[1])
        np.testing.assert_array_equal(_pairs(actual, compact), _pairs(expected, False))
        matrix, counts, shifts = map(np.asarray, actual[:3])
        if compact:
            length = int(counts[-1])
            rows, targets = matrix[:, :length]
            valid = slice(0, length)
            images = shifts[valid]
            capacity = common.get("coo_capacity", length)
            assert matrix.shape == (2, capacity)
            assert shifts.shape == (capacity, 3)
            pair_shape = (capacity,)
        else:
            rows, slots = np.nonzero(
                np.arange(matrix.shape[1])[None, :] < counts[:, None]
            )
            valid = (rows, slots)
            targets = matrix[valid]
            images = shifts[valid]
            pair_shape = (atoms, 64)
        assert len(rows) > 0
        vectors = np.asarray(positions)[targets] - np.asarray(positions)[rows]
        vectors += np.einsum(
            "ni,nij->nj", images, np.asarray(cell)[batch_indices[rows]]
        )
        distances = np.linalg.norm(vectors, axis=1)
        output_index = 5 if layout == "fixed_coo" else 3
        if layout == "fixed_coo":
            np.testing.assert_array_equal(actual[3], expected[1])
            assert bool(actual[4])
        if common["return_distances"]:
            assert actual[output_index].shape == pair_shape
            np.testing.assert_allclose(
                np.asarray(actual[output_index])[valid],
                distances,
                rtol=1e-5,
                atol=1e-5,
            )
            output_index += 1
        if common["return_vectors"]:
            assert actual[output_index].shape == (*pair_shape, 3)
            np.testing.assert_allclose(
                np.asarray(actual[output_index])[valid],
                vectors,
                rtol=1e-5,
                atol=1e-5,
            )
            output_index += 1
        if fused:
            params = np.asarray(common["pair_params"])
            assert actual[output_index].shape == pair_shape
            assert actual[output_index + 1].shape == (*pair_shape, 3)
            np.testing.assert_allclose(
                np.asarray(actual[output_index])[valid],
                params[rows, 0] + params[targets, 0] + distances,
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.asarray(actual[output_index + 1])[valid],
                -vectors,
                rtol=1e-5,
                atol=1e-5,
            )
            output_index += 2
        assert len(actual) == output_index


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
        reference = call(grid_policy="adaptive", **kwargs, strategy="atom_centric")
        profiler = cProfile.Profile()
        with profiler:
            actual = call(grid_policy="adaptive", **kwargs, strategy=strategy)
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
    reference = call(positions, grid_policy="adaptive", **common)

    def query(pos):
        return call(pos, grid_policy="adaptive", **common)

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
    expected = call(
        positions, grid_policy="adaptive", **kwargs, strategy="atom_centric"
    )
    if public:
        kwargs["method"] = "batch_cell_list" if batched else "cell_list"
        actual = neighbor_list(positions, grid_policy="adaptive", **kwargs)
    else:
        actual = call(
            positions, grid_policy="adaptive", **kwargs, strategy="pair_centric"
        )
    np.testing.assert_array_equal(_pairs(actual, True), _pairs(expected, True))


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_grid_policy_default_and_opt_in(batched, public, compiled):
    """Both policies and the default preserve neighbor records under eager/JIT."""
    systems = 2 if batched else 1
    positions = jnp.asarray(
        np.random.default_rng(71).random((32 * systems, 3)) * 12, dtype=jnp.float32
    )
    cell = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None] * 12, (systems, 1, 1))
    common = dict(
        cutoff=6.0,
        cell=cell,
        pbc=jnp.ones((systems, 3), dtype=jnp.bool_),
        max_neighbors=64,
        max_total_cells=64 * systems,
    )
    if batched:
        common.update(
            batch_idx=jnp.repeat(jnp.arange(systems, dtype=jnp.int32), 32),
            batch_ptr=jnp.arange(systems + 1, dtype=jnp.int32) * 32,
        )
    call = neighbor_list if public else (batch_cell_list if batched else cell_list)
    if public:
        common["method"] = (
            "batch_cell_list_pair_centric" if batched else "cell_list_pair_centric"
        )
    else:
        common["strategy"] = "pair_centric"
    if compiled:
        if batched:
            common.update(
                pair_centric_total_cells=64 * systems,
                pair_centric_n_outer=124,
                pair_centric_r_max=(2, 2, 2),
            )
        else:
            common["pair_centric_n_outer"] = 124
    outputs = []
    for policy in (None, "configured", "adaptive"):
        policy_kwargs = {} if policy is None else {"grid_policy": policy}

        def execute(pos):
            return call(pos, **common, **policy_kwargs)

        result = (jax.jit(execute) if compiled else execute)(positions)
        outputs.append(_pairs(result, False))
    np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(outputs[0], outputs[2])
    with pytest.raises(ValueError, match="grid_policy"):
        call(positions, **common, grid_policy="unknown")
