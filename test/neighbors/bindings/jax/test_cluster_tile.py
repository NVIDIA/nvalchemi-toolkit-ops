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

"""Tests for JAX bindings of the single-system cluster-pair tile neighbor list."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors import _cluster_tile_preload
from nvalchemiops.jax.neighbors.cluster_tile import (
    _CLUSTER_TILE_QUERIES,
    TILE_GROUP_SIZE,
    build_cluster_tile_list,
    cluster_tile_neighbor_list,
    estimate_cluster_tile_list_sizes,
    query_cluster_tile_coo,
)

from .conftest import requires_gpu

pytestmark = requires_gpu


def _orthorhombic_cell(cell_size: float, dtype=jnp.float32) -> jax.Array:
    return jnp.eye(3, dtype=dtype) * cell_size


def _traced_preload_device_count() -> int:
    """Return how many local devices a traced graph callback must preload."""
    local_devices = tuple(jax.local_devices())
    accelerators = tuple(
        device for device in local_devices if device.platform in {"gpu", "cuda", "rocm"}
    )
    return len(accelerators or local_devices)


def _row_neighbor_set(row: jax.Array, n: int) -> set[int]:
    return {int(x) for x in row[:n]}


def _brute_force_pairs_full(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    pbc: bool = True,
) -> set[tuple[int, int, int, int, int]]:
    """numpy reference: ALL directed ``(i, j, sx, sy, sz)`` tuples within
    ``cutoff`` (full-fill: both ``(i, j, s)`` and ``(j, i, -s)``).

    Brute-force: iterate atom pairs across the minimal PBC shift box that
    can fit any pair within ``cutoff`` (works for any orthorhombic /
    triclinic cell where the cell extent is at least the cutoff in each
    direction — which is the standard cluster-tile precondition).
    """
    natom = positions.shape[0]
    cell = np.asarray(cell).reshape(3, 3)
    # Choose a shift range that covers any cutoff <= cell extent in each
    # cell direction.  Conservative: 1 in each direction.
    shift_range = 1 if pbc else 0
    pairs: set[tuple[int, int, int, int, int]] = set()
    for i in range(natom):
        for j in range(natom):
            for sx in range(-shift_range, shift_range + 1):
                for sy in range(-shift_range, shift_range + 1):
                    for sz in range(-shift_range, shift_range + 1):
                        if i == j and (sx, sy, sz) == (0, 0, 0):
                            continue
                        shift_vec = np.array([sx, sy, sz]) @ cell
                        d = positions[j] - positions[i] + shift_vec
                        if float(np.linalg.norm(d)) < cutoff:
                            pairs.add((i, j, sx, sy, sz))
    return pairs


def _matrix_to_pair_set_full(
    nm: jax.Array,
    nn: jax.Array,
    shifts: jax.Array,
    natom: int,
) -> set[tuple[int, int, int, int, int]]:
    """ALL directed ``(i, j, sx, sy, sz)`` tuples from the cluster-tile matrix.

    cluster-tile is full-fill: every atom's row lists all its neighbors, so
    each unordered pair appears in both rows (with negated shifts).
    """
    nm_np = np.asarray(nm)
    nn_np = np.asarray(nn)
    sh_np = np.asarray(shifts)
    pairs: set[tuple[int, int, int, int, int]] = set()
    for i in range(natom):
        ni = int(nn_np[i])
        for k in range(ni):
            j = int(nm_np[i, k])
            if not (0 <= j < natom):
                continue
            sx, sy, sz = (int(x) for x in sh_np[i, k])
            pairs.add((i, j, sx, sy, sz))
    return pairs


class TestTileNeighborListCorrectness:
    """Smoke + small-system correctness tests."""

    def test_single_atom_no_neighbors(self):
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        cell = _orthorhombic_cell(2.0)
        nm, nn, _ = cluster_tile_neighbor_list(positions, 3.0, cell, max_neighbors=8)
        assert int(nn.sum()) == 0
        assert nm.shape == (1, 8)

    def test_topology_only_grad_matrix_is_zero(self):
        """Matrix topology from cluster-tile is nondifferentiable."""
        positions = (
            jax.random.uniform(jax.random.key(0), (64, 3), dtype=jnp.float32) * 10.0
        )
        cell = _orthorhombic_cell(10.0)

        def loss(pos):
            neighbor_matrix, num_neighbors, shifts = cluster_tile_neighbor_list(
                pos,
                2.0,
                cell,
                max_neighbors=32,
            )
            return (
                neighbor_matrix.astype(pos.dtype).sum()
                + num_neighbors.astype(pos.dtype).sum()
                + shifts.astype(pos.dtype).sum()
            )

        grad = jax.grad(loss)(positions)
        assert jnp.isfinite(grad).all().item()
        np.testing.assert_allclose(np.asarray(grad), 0.0)

    def test_two_atom_pair(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32)
        cell = _orthorhombic_cell(2.0)
        nm, nn, _ = cluster_tile_neighbor_list(positions, 1.0, cell, max_neighbors=8)
        # Full-fill: each atom lists the other (matches cell_list half_fill=False).
        assert int(nn.sum()) == 2
        assert int(nn[0]) == 1 and int(nn[1]) == 1
        assert int(nm[0, 0]) == 1 and int(nm[1, 0]) == 0

    def test_cubic_lattice(self):
        # 4x4x4 = 64 atoms on a simple cubic lattice with spacing 1.0.
        n_side = 4
        coords = [
            [float(ix), float(iy), float(iz)]
            for ix in range(n_side)
            for iy in range(n_side)
            for iz in range(n_side)
        ]
        positions = jnp.array(coords, dtype=jnp.float32)
        cell = _orthorhombic_cell(float(n_side))
        cutoff = 1.1  # picks up just the 6 nearest neighbors
        nm, nn, _ = cluster_tile_neighbor_list(
            positions, cutoff, cell, max_neighbors=64
        )
        # Full-fill: each atom has 6 nearest neighbors, all stored per row.
        assert int(nn.sum()) == 6 * positions.shape[0]

    def test_non_aligned_N_accepts_padding(self):
        # N not divisible by TILE_GROUP_SIZE — the JAX wrapper pads
        # internally; the kernel uses sentinel Morton codes for padding.
        positions = jnp.array(
            np.random.RandomState(0).uniform(0, 10, size=(33, 3)).astype(np.float32),
        )
        cell = _orthorhombic_cell(10.0)
        nm, nn, _ = cluster_tile_neighbor_list(positions, 2.5, cell, max_neighbors=32)
        # Shape sanity; no negative counts.
        assert nm.shape == (33, 32)
        assert bool(jnp.all(nn >= 0))


class TestClusterTileGraphPreload:
    """Exercise the public WARP graph callback after its kernel preload."""

    @pytest.fixture(autouse=True)
    def _isolate_preload_caches(self, monkeypatch) -> None:
        """Keep graph-preload cache entries local to each regression test."""
        for cache_name in (
            "_preload_cluster_tile_query_kernel_cached",
            "_preload_cluster_tile_coo_kernel_cached",
        ):
            original = getattr(_cluster_tile_preload, cache_name)
            monkeypatch.setattr(
                _cluster_tile_preload,
                cache_name,
                functools.cache(original.__wrapped__),
            )

    def test_jitted_matrix_query_preloads_and_executes_warp_graph_callback(self):
        """A public matrix query preloads and executes under JAX graph capture."""
        _cluster_tile_preload._preload_cluster_tile_query_kernel_cached.cache_clear()
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]] * 16,
            dtype=jnp.float32,
        )
        cell = _orthorhombic_cell(4.0)

        @jax.jit
        def query(positions):
            return cluster_tile_neighbor_list(
                positions,
                1.0,
                cell,
                max_neighbors=32,
            )

        neighbor_matrix, num_neighbors, _shifts = query(positions)
        neighbor_matrix.block_until_ready()

        assert (
            _cluster_tile_preload._preload_cluster_tile_query_kernel_cached.cache_info().currsize
            == _traced_preload_device_count()
        )
        assert int(num_neighbors.sum()) > 0

    def test_jitted_matrix_query_reuses_preload_cache(self):
        """Repeated jitted matrix queries keep one preload cache entry."""
        _cluster_tile_preload._preload_cluster_tile_query_kernel_cached.cache_clear()
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]] * 16,
            dtype=jnp.float32,
        )
        cell = _orthorhombic_cell(4.0)

        @jax.jit
        def query(positions):
            return cluster_tile_neighbor_list(
                positions,
                1.0,
                cell,
                max_neighbors=32,
            )

        first = query(positions)
        first[0].block_until_ready()
        second = query(positions)
        second[0].block_until_ready()

        assert (
            _cluster_tile_preload._preload_cluster_tile_query_kernel_cached.cache_info().currsize
            == _traced_preload_device_count()
        )
        assert int(second[1].sum()) > 0

    def test_jitted_compact_coo_registration_preloads_and_executes(self):
        """Compact COO graph callback executes under JIT with fixed buffers."""
        _cluster_tile_preload._preload_cluster_tile_coo_kernel_cached.cache_clear()
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]] * 16,
            dtype=jnp.float32,
        )
        cell = _orthorhombic_cell(4.0)
        (
            sorted_atom_index,
            _morton,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            _group_ctr_x,
            _group_ctr_y,
            _group_ctr_z,
            _group_ext_x,
            _group_ext_y,
            _group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
        ) = build_cluster_tile_list(positions, 1.0, cell)
        max_pairs = 256
        pair_counter = jnp.zeros(1, dtype=jnp.int32)
        coo_list = jnp.zeros((max_pairs, 2), dtype=jnp.int32)
        coo_shifts = jnp.zeros((max_pairs, 3), dtype=jnp.int32)
        registration = _CLUSTER_TILE_QUERIES["coo"]

        cell_n = cell[jnp.newaxis, :, :]
        inv_cell_n = jnp.linalg.inv(cell_n[0])[jnp.newaxis, :, :]

        @jax.jit
        def query(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            pair_counter,
            coo_list,
            coo_shifts,
        ):
            registration.preload()
            return registration.callable(
                sorted_atom_index,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                num_tiles,
                tile_row_group,
                tile_col_group,
                cell_n,
                inv_cell_n,
                pair_counter,
                coo_list,
                coo_shifts,
                1.0,
                positions.shape[0],
                max_pairs,
            )

        pair_counter, coo_list, coo_shifts = query(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            pair_counter,
            coo_list,
            coo_shifts,
        )
        coo_list.block_until_ready()
        assert (
            _cluster_tile_preload._preload_cluster_tile_coo_kernel_cached.cache_info().currsize
            == 1
        )
        assert int(pair_counter[0]) > 0

        pair_counter2, coo_list2, coo_shifts2 = query(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            pair_counter,
            coo_list,
            coo_shifts,
        )
        coo_list2.block_until_ready()
        assert (
            _cluster_tile_preload._preload_cluster_tile_coo_kernel_cached.cache_info().currsize
            == 1
        )
        assert int(pair_counter2[0]) > 0


class TestTileNeighborListFormats:
    """Tests for the three output formats."""

    def test_matrix_vs_coo_pair_count_match(self):
        rng = np.random.RandomState(1)
        positions = jnp.array(rng.uniform(0, 10, size=(64, 3)).astype(np.float32))
        cell = _orthorhombic_cell(10.0)
        cutoff = 2.5

        nm, nn, _ = cluster_tile_neighbor_list(
            positions, cutoff, cell, max_neighbors=64
        )
        nl, ptr, _ = cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=64,
            format="coo",
        )
        assert int(nn.sum()) == int(nl.shape[1])
        assert int(ptr[-1]) == int(nl.shape[1])

    def test_tile_format_returns_state(self):
        positions = jnp.array(
            np.random.RandomState(2).uniform(0, 10, size=(32, 3)).astype(np.float32),
        )
        cell = _orthorhombic_cell(10.0)
        out = cluster_tile_neighbor_list(positions, 2.5, cell, format="tile")
        assert len(out) == 7
        num_tiles, tile_row_group, tile_col_group, *_ = out
        n_tiles = int(num_tiles[0])
        assert n_tiles > 0
        # Tile indices fall within [0, ngroup).
        ngroup = positions.shape[0] // TILE_GROUP_SIZE
        assert int(tile_row_group[:n_tiles].max()) < ngroup
        assert int(tile_col_group[:n_tiles].max()) < ngroup


class TestTileNeighborListErrors:
    """Error path tests."""

    def test_wrong_dtype_raises(self):
        positions = jnp.zeros((32, 3), dtype=jnp.float64)
        cell = _orthorhombic_cell(4.0, dtype=jnp.float64)
        with pytest.raises(TypeError):
            cluster_tile_neighbor_list(positions, 1.0, cell, max_neighbors=8)

    def test_invalid_format_raises(self):
        positions = jnp.zeros((32, 3), dtype=jnp.float32)
        cell = _orthorhombic_cell(4.0)
        with pytest.raises(ValueError, match="format"):
            cluster_tile_neighbor_list(positions, 1.0, cell, format="bogus")


class TestEstimateSizes:
    """Pure-Python sizing helper tests."""

    def test_aligned_sizes(self):
        n_padded, ngroup, ngroup_padded, max_tiles = estimate_cluster_tile_list_sizes(
            512,
        )
        assert n_padded == 512
        assert ngroup == 512 // TILE_GROUP_SIZE
        assert ngroup_padded % TILE_GROUP_SIZE == 0 and ngroup_padded > ngroup
        assert max_tiles >= ngroup

    def test_non_aligned_sizes(self):
        n_padded, ngroup, _, _ = estimate_cluster_tile_list_sizes(33)
        assert n_padded == 64  # ceil(33 / 32) * 32
        assert ngroup == 2

    def test_zero_atoms(self):
        n_padded, ngroup, _, _ = estimate_cluster_tile_list_sizes(0)
        # Must always reserve at least one tile group.
        assert n_padded == TILE_GROUP_SIZE


class TestJaxClusterTileAutograd:
    """Differentiable per-pair distances/vectors for cluster_tile_neighbor_list."""

    def _make_system(self, n=32, box=5.0, scale=1.0):
        key = jax.random.key(0)
        pos = jax.random.normal(key, (n, 3), dtype=jnp.float32) * scale
        cell = jnp.eye(3, dtype=jnp.float32) * box
        return pos, cell

    def test_forward_returns_distances_and_vectors(self):
        pos, cell = self._make_system()
        out = cluster_tile_neighbor_list(
            pos,
            1.5,
            cell,
            return_distances=True,
            return_vectors=True,
        )
        assert len(out) == 5
        nm, nn, shifts, d, v = out
        assert d.shape == nm.shape
        assert v.shape == nm.shape + (3,)
        assert d.dtype == jnp.float32
        assert v.dtype == jnp.float32

    def test_return_tuple_shape_extends_with_flags(self):
        pos, cell = self._make_system()
        base = cluster_tile_neighbor_list(pos, 1.5, cell)
        assert len(base) == 3
        plus_d = cluster_tile_neighbor_list(
            pos,
            1.5,
            cell,
            return_distances=True,
        )
        assert len(plus_d) == 4
        plus_v = cluster_tile_neighbor_list(
            pos,
            1.5,
            cell,
            return_vectors=True,
        )
        assert len(plus_v) == 4

    def test_grad_positions_finite(self):
        pos, cell = self._make_system()

        def loss(p):
            *_, d, _ = cluster_tile_neighbor_list(
                p,
                1.5,
                cell,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        g = jax.grad(loss)(pos)
        assert g.shape == pos.shape
        assert jnp.isfinite(g).all().item()

    def test_check_grads_against_finite_differences(self):
        from jax.test_util import check_grads

        # Cluster_tile is fp32-only; push the neighbor-set discontinuity
        # well out of FD reach by clustering positions tightly inside a
        # large box and using a wide cutoff.  Use a larger FD step (1e-3)
        # to survive fp32 catastrophic cancellation.
        key = jax.random.key(0)
        pos = jax.random.normal(key, (32, 3), dtype=jnp.float32) * 0.15
        cell = jnp.eye(3, dtype=jnp.float32) * 20.0

        # Drain asynchronous input construction before Warp starts CUDA graph capture.
        jax.block_until_ready((pos, cell))

        def loss(p):
            *_, d, _ = cluster_tile_neighbor_list(
                p,
                5.0,
                cell,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        check_grads(
            loss, (pos,), order=1, atol=1e-4, rtol=1e-4, modes=["rev"], eps=1e-3
        )

    def test_pair_fn_supported(self):
        """pair_fn is now wired through the JAX cluster_tile binding (matrix and COO;
        returns per-pair pe/pf).  See test_pair_fn.py for full coverage."""
        from .test_pair_fn import _sum_pair_fn_f32

        pos, cell = self._make_system()
        pp = ((jnp.arange(pos.shape[0], dtype=jnp.float32) + 1.0) * 0.5).reshape(-1, 1)
        out = cluster_tile_neighbor_list(
            pos,
            1.5,
            cell,
            max_neighbors=64,
            return_distances=True,
            return_vectors=True,
            pair_fn=_sum_pair_fn_f32,
            pair_params=pp,
        )
        # nm, nn, shifts, distances, vectors, pe, pf
        assert len(out) == 7
        assert out[5].shape == (pos.shape[0], out[0].shape[1])
        assert out[6].shape == (pos.shape[0], out[0].shape[1], 3)

    def test_format_tile_with_pair_outputs_rejected(self):
        """Pair outputs work with 'matrix' and 'coo'; only 'tile' rejects them."""
        pos, cell = self._make_system()
        with pytest.raises(NotImplementedError, match="format='tile'"):
            cluster_tile_neighbor_list(
                pos,
                1.5,
                cell,
                format="tile",
                return_distances=True,
            )

    def test_hessian_vector_product_smoke(self):
        """fp32 second-order: HVP runs and returns finite values.

        Order-2 ``check_grads`` is too tight for fp32; we settle for a
        Hessian-vector product smoke test which still exercises the
        backward of the backward.
        """
        pos, cell = self._make_system()
        v = jax.random.normal(jax.random.key(1), pos.shape, dtype=pos.dtype)

        def loss(p):
            *_, d, _ = cluster_tile_neighbor_list(
                p,
                1.5,
                cell,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        hvp = jax.grad(lambda p: jnp.vdot(jax.grad(loss)(p), v))(pos)
        assert jnp.isfinite(hvp).all().item()
        assert hvp.shape == pos.shape

    def test_hvp_nonlinear_loss_matches_analytic(self):
        """Regression: nonlinear-loss HVP matches the exact analytic Hessian on the
        cluster_tile path (fp32).  The existing HVP smoke only used a *linear* loss
        (``d.sum()``); this guards the nonlinear 2nd-order through the shared
        live-reconstruction autograd (the old detached backward got it wrong)."""
        from .conftest import analytic_distance_sq_hvp

        pos, cell = self._make_system()
        v = jax.random.normal(jax.random.key(1), pos.shape, dtype=pos.dtype)
        cutoff = 1.5

        def loss(p):
            *_, d, _ = cluster_tile_neighbor_list(
                p,
                cutoff,
                cell,
                max_neighbors=64,
                return_distances=True,
                return_vectors=True,
            )
            return (d**2).sum()

        hvp = np.asarray(jax.grad(lambda p: jnp.vdot(jax.grad(loss)(p), v))(pos))

        # cluster_tile rejects COO + pair outputs, so derive the directed (i, j)
        # pairs from the neighbour matrix the loss sums over.
        out = cluster_tile_neighbor_list(
            pos,
            cutoff,
            cell,
            max_neighbors=64,
            return_distances=True,
            return_vectors=True,
        )
        nm_np, nn_np = np.asarray(out[0]), np.asarray(out[1])
        width = nm_np.shape[1]
        i_list, j_list = [], []
        for i in range(nm_np.shape[0]):
            for s in range(min(int(nn_np[i]), width)):
                i_list.append(i)
                j_list.append(int(nm_np[i, s]))
        nl = np.array([i_list, j_list])
        assert nl.shape[1] > 0
        hvp_true = analytic_distance_sq_hvp(nl, v, pos.shape[0])
        assert np.allclose(hvp, hvp_true, atol=1e-2, rtol=1e-2)

    def test_no_grad_path_unchanged(self):
        pos, cell = self._make_system()
        nm_a, nn_a, sh_a = cluster_tile_neighbor_list(pos, 1.5, cell)
        nm_b, nn_b, sh_b, d_b, v_b = cluster_tile_neighbor_list(
            pos,
            1.5,
            cell,
            return_distances=True,
            return_vectors=True,
        )
        assert jnp.all(nn_a == nn_b)
        # cluster_tile may emit pairs in different order between the two
        # kernel variants; compare as sets per row.
        for i in range(nm_a.shape[0]):
            n = int(nn_a[i])
            row_a = sorted(int(x) for x in nm_a[i, :n])
            row_b = sorted(int(x) for x in nm_b[i, :n])
            assert row_a == row_b
        assert jnp.isfinite(d_b).all().item()
        assert jnp.isfinite(v_b).all().item()


class TestJaxClusterTileBruteForce:
    """Pair-set + shift identity checks against a numpy brute-force
    reference.  These guard against silently-wrong outputs that the
    existing shape/count assertions would not catch.
    """

    def test_random_small_pbc_matches_brute_force(self):
        rng = np.random.default_rng(0)
        positions_np = rng.uniform(0, 4.0, size=(12, 3)).astype(np.float32)
        cell_np = np.eye(3, dtype=np.float32) * 4.0
        positions = jnp.asarray(positions_np)
        cell = jnp.asarray(cell_np)
        cutoff = 1.5

        nm, nn, shifts = cluster_tile_neighbor_list(
            positions, cutoff, cell, max_neighbors=32
        )
        got = _matrix_to_pair_set_full(nm, nn, shifts, positions_np.shape[0])
        ref = _brute_force_pairs_full(positions_np, cell_np, cutoff, pbc=True)
        assert got == ref, (
            f"cluster_tile output disagrees with brute-force:\n"
            f"  missing: {ref - got}\n"
            f"  extra:   {got - ref}"
        )

    def test_dense_small_pbc_matches_brute_force(self):
        # Smaller, denser system: more pairs, exercises the half-fill
        # canonicalization more thoroughly.
        rng = np.random.default_rng(1)
        positions_np = rng.uniform(0, 2.5, size=(8, 3)).astype(np.float32)
        cell_np = np.eye(3, dtype=np.float32) * 2.5
        positions = jnp.asarray(positions_np)
        cell = jnp.asarray(cell_np)
        cutoff = 1.2

        nm, nn, shifts = cluster_tile_neighbor_list(
            positions, cutoff, cell, max_neighbors=32
        )
        got = _matrix_to_pair_set_full(nm, nn, shifts, positions_np.shape[0])
        ref = _brute_force_pairs_full(positions_np, cell_np, cutoff, pbc=True)
        assert got == ref

    def test_coo_format_matches_brute_force(self):
        rng = np.random.default_rng(2)
        positions_np = rng.uniform(0, 4.0, size=(10, 3)).astype(np.float32)
        cell_np = np.eye(3, dtype=np.float32) * 4.0
        positions = jnp.asarray(positions_np)
        cell = jnp.asarray(cell_np)
        cutoff = 1.5

        nl, _nptr, nl_shifts = cluster_tile_neighbor_list(
            positions, cutoff, cell, max_neighbors=32, format="coo"
        )
        # nl shape: (2, npairs); canonicalize each pair to (i, j, shift)
        # with i <= j to match the brute-force reference.
        # COO is full-fill: collect all directed (i, j, shift) triples.
        nl_np = np.asarray(nl)
        sh_np = np.asarray(nl_shifts)
        got: set[tuple[int, int, int, int, int]] = set()
        for k in range(nl_np.shape[1]):
            i, j = int(nl_np[0, k]), int(nl_np[1, k])
            sx, sy, sz = (int(x) for x in sh_np[k])
            got.add((i, j, sx, sy, sz))
        ref = _brute_force_pairs_full(positions_np, cell_np, cutoff, pbc=True)
        assert got == ref


class TestJaxClusterTileCutoff2Selective:
    """Matrix-only cutoff2 and selective rebuild coverage."""

    def test_cutoff2_matrix_returns_two_cutoff_groups(self):
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.4, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        cell = _orthorhombic_cell(5.0)
        out = cluster_tile_neighbor_list(
            positions,
            1.0,
            cell,
            max_neighbors=8,
            cutoff2=1.6,
        )
        assert len(out) == 6
        _nm1, nn1, _sh1, _nm2, nn2, _sh2 = out
        assert int(nn2.sum()) >= int(nn1.sum())
        assert int(nn2.sum()) > 0

    def test_rebuild_flags_false_preserves_previous_single_system_outputs(self):
        rng = np.random.RandomState(23)
        positions = jnp.array(rng.uniform(0, 6, size=(64, 3)).astype(np.float32))
        cell = _orthorhombic_cell(6.0)
        cutoff = 2.0
        nm, nn, shifts = cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=64,
        )
        num_tiles, tile_row_group, tile_col_group, *_ = cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            format="tile",
        )

        moved = positions.at[0, 0].add(0.25)
        out = cluster_tile_neighbor_list(
            moved,
            cutoff,
            cell,
            max_neighbors=64,
            rebuild_flags=jnp.array([False], dtype=jnp.bool_),
            previous_num_tiles=num_tiles,
            previous_tile_row_group=tile_row_group,
            previous_tile_col_group=tile_col_group,
            previous_neighbor_matrix=nm,
            previous_num_neighbors=nn,
            previous_neighbor_matrix_shifts=shifts,
        )
        nm2, nn2, shifts2, num_tiles2, row2, col2 = out
        np.testing.assert_array_equal(np.asarray(nm2), np.asarray(nm))
        np.testing.assert_array_equal(np.asarray(nn2), np.asarray(nn))
        np.testing.assert_array_equal(np.asarray(shifts2), np.asarray(shifts))
        np.testing.assert_array_equal(np.asarray(num_tiles2), np.asarray(num_tiles))
        np.testing.assert_array_equal(np.asarray(row2), np.asarray(tile_row_group))
        np.testing.assert_array_equal(np.asarray(col2), np.asarray(tile_col_group))

    def test_rebuild_flags_require_previous_state(self):
        positions = jnp.zeros((32, 3), dtype=jnp.float32)
        cell = _orthorhombic_cell(4.0)
        with pytest.raises(ValueError, match="previous cluster_tile state"):
            cluster_tile_neighbor_list(
                positions,
                1.0,
                cell,
                max_neighbors=8,
                rebuild_flags=jnp.array([True], dtype=jnp.bool_),
            )

    def test_rebuild_flags_coo_false_preserves_fixed_buffers(self):
        rng = np.random.RandomState(35)
        positions = jnp.array(rng.uniform(0, 6, size=(64, 3)).astype(np.float32))
        cell = _orthorhombic_cell(6.0)
        cutoff = 2.0
        num_tiles, tile_row_group, tile_col_group, *_ = cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            format="tile",
        )
        pair_offsets = jnp.array([0, 128], dtype=jnp.int32)
        pair_counts = jnp.array([7], dtype=jnp.int32)
        neighbor_list = jnp.arange(256, dtype=jnp.int32).reshape(2, 128)
        neighbor_shifts = jnp.arange(384, dtype=jnp.int32).reshape(128, 3)

        out = cluster_tile_neighbor_list(
            positions.at[0, 0].add(0.25),
            cutoff,
            cell,
            max_neighbors=64,
            format="coo",
            rebuild_flags=jnp.array([False], dtype=jnp.bool_),
            previous_num_tiles=num_tiles,
            previous_tile_row_group=tile_row_group,
            previous_tile_col_group=tile_col_group,
            pair_offsets=pair_offsets,
            previous_pair_counts=pair_counts,
            previous_neighbor_list=neighbor_list,
            previous_neighbor_list_shifts=neighbor_shifts,
        )
        nl2, offsets2, counts2, shifts2, nt2, row2, col2 = out
        np.testing.assert_array_equal(np.asarray(nl2), np.asarray(neighbor_list))
        np.testing.assert_array_equal(np.asarray(offsets2), np.asarray(pair_offsets))
        np.testing.assert_array_equal(np.asarray(counts2), np.asarray(pair_counts))
        np.testing.assert_array_equal(np.asarray(shifts2), np.asarray(neighbor_shifts))
        np.testing.assert_array_equal(np.asarray(nt2), np.asarray(num_tiles))
        np.testing.assert_array_equal(np.asarray(row2), np.asarray(tile_row_group))
        np.testing.assert_array_equal(np.asarray(col2), np.asarray(tile_col_group))

    def test_rebuild_flags_coo_true_writes_segment_counts(self):
        rng = np.random.RandomState(36)
        positions = jnp.array(rng.uniform(0, 6, size=(64, 3)).astype(np.float32))
        cell = _orthorhombic_cell(6.0)
        cutoff = 2.0
        tile_state = cluster_tile_neighbor_list(positions, cutoff, cell, format="tile")
        num_tiles, tile_row_group, tile_col_group, *_ = tile_state
        pair_offsets = jnp.array([0, 4096], dtype=jnp.int32)
        previous_neighbor_list = jnp.zeros((2, 4096), dtype=jnp.int32)
        previous_neighbor_shifts = jnp.zeros((4096, 3), dtype=jnp.int32)

        out = cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=64,
            format="coo",
            rebuild_flags=jnp.array([True], dtype=jnp.bool_),
            previous_num_tiles=num_tiles,
            previous_tile_row_group=tile_row_group,
            previous_tile_col_group=tile_col_group,
            pair_offsets=pair_offsets,
            previous_pair_counts=jnp.zeros(1, dtype=jnp.int32),
            previous_neighbor_list=previous_neighbor_list,
            previous_neighbor_list_shifts=previous_neighbor_shifts,
        )
        _nl, _offsets, pair_counts, _shifts, nt2, _row2, _col2 = out
        assert int(pair_counts[0]) > 0
        assert int(pair_counts[0]) <= int(pair_offsets[1] - pair_offsets[0])
        assert int(nt2[0]) > 0

    def test_rebuild_flags_coo_clamps_valid_overflow_to_physical_capacity(self):
        """Valid fixed metadata cannot report more pairs than it can store."""
        positions = jnp.zeros((64, 3), dtype=jnp.float32)
        cell = _orthorhombic_cell(6.0)
        num_tiles, tile_row_group, tile_col_group, *_ = cluster_tile_neighbor_list(
            positions,
            2.0,
            cell,
            format="tile",
        )
        capacity = 64
        out = cluster_tile_neighbor_list(
            positions,
            2.0,
            cell,
            max_neighbors=64,
            format="coo",
            rebuild_flags=jnp.array([True], dtype=jnp.bool_),
            previous_num_tiles=num_tiles,
            previous_tile_row_group=tile_row_group,
            previous_tile_col_group=tile_col_group,
            pair_offsets=jnp.array([0, capacity], dtype=jnp.int32),
            previous_pair_counts=jnp.zeros(1, dtype=jnp.int32),
            previous_neighbor_list=jnp.full((2, capacity), -77, dtype=jnp.int32),
            previous_neighbor_list_shifts=jnp.full(
                (capacity, 3),
                -77,
                dtype=jnp.int32,
            ),
        )

        np.testing.assert_array_equal(
            np.asarray(out[2]),
            np.array([capacity], dtype=np.int32),
        )
        assert np.all(np.asarray(out[0]) != -77)
        assert np.all(np.asarray(out[3]) != -77)

    def test_rebuild_flags_coo_requires_segment_buffers(self):
        positions = jnp.zeros((32, 3), dtype=jnp.float32)
        cell = _orthorhombic_cell(4.0)
        num_tiles, tile_row_group, tile_col_group, *_ = cluster_tile_neighbor_list(
            positions,
            1.0,
            cell,
            format="tile",
        )
        with pytest.raises(ValueError, match="previous cluster_tile state"):
            cluster_tile_neighbor_list(
                positions,
                1.0,
                cell,
                max_neighbors=8,
                format="coo",
                rebuild_flags=jnp.array([False], dtype=jnp.bool_),
                previous_num_tiles=num_tiles,
                previous_tile_row_group=tile_row_group,
                previous_tile_col_group=tile_col_group,
            )

    def test_rebuild_flags_coo_requires_one_single_system_segment(self):
        """Single-system segmented COO rejects unused extra metadata segments."""
        positions = jnp.zeros((32, 3), dtype=jnp.float32)
        cell = _orthorhombic_cell(6.0)
        num_tiles, tile_row_group, tile_col_group, *_ = cluster_tile_neighbor_list(
            positions,
            2.0,
            cell,
            format="tile",
        )

        with pytest.raises(ValueError, match=r"pair_offsets must have shape \(2,\)"):
            cluster_tile_neighbor_list(
                positions,
                2.0,
                cell,
                max_neighbors=64,
                format="coo",
                rebuild_flags=jnp.array([False], dtype=jnp.bool_),
                previous_num_tiles=num_tiles,
                previous_tile_row_group=tile_row_group,
                previous_tile_col_group=tile_col_group,
                pair_offsets=jnp.array([0, 4, 8], dtype=jnp.int32),
                previous_pair_counts=jnp.array([0, 0], dtype=jnp.int32),
                previous_neighbor_list=jnp.full((2, 8), -77, dtype=jnp.int32),
                previous_neighbor_list_shifts=jnp.full((8, 3), -77, dtype=jnp.int32),
            )

    @pytest.mark.parametrize(
        ("pair_offsets_values", "rebuild_flag"),
        [
            ((64, 0), True),
            ((0, 32), True),
            ((1, 64), True),
            ((64, 0), False),
            ((0, 32), False),
            ((1, 64), False),
        ],
        ids=[
            "reversed-true",
            "undersized-true",
            "nonzero-start-true",
            "reversed-false",
            "undersized-false",
            "nonzero-start-false",
        ],
    )
    def test_jit_selective_coo_invalid_offsets_fail_closed(
        self,
        pair_offsets_values,
        rebuild_flag,
    ):
        """JIT returns no active entries for malformed single-system metadata."""
        positions = jnp.zeros((32, 3), dtype=jnp.float32)
        cell = _orthorhombic_cell(6.0)
        num_tiles, tile_row_group, tile_col_group, *_ = cluster_tile_neighbor_list(
            positions,
            2.0,
            cell,
            format="tile",
        )
        neighbor_list = jnp.full((2, 64), -77, dtype=jnp.int32)
        neighbor_list_shifts = jnp.full((64, 3), -77, dtype=jnp.int32)

        @jax.jit
        def run(pair_offsets, pair_counts, runtime_rebuild_flags):
            return cluster_tile_neighbor_list(
                positions,
                2.0,
                cell,
                max_neighbors=64,
                format="coo",
                rebuild_flags=runtime_rebuild_flags,
                previous_num_tiles=num_tiles,
                previous_tile_row_group=tile_row_group,
                previous_tile_col_group=tile_col_group,
                pair_offsets=pair_offsets,
                previous_pair_counts=pair_counts,
                previous_neighbor_list=neighbor_list,
                previous_neighbor_list_shifts=neighbor_list_shifts,
            )

        out = run(
            jnp.array(pair_offsets_values, dtype=jnp.int32),
            jnp.array([0 if rebuild_flag else 7], dtype=jnp.int32),
            jnp.array([rebuild_flag], dtype=jnp.bool_),
        )

        np.testing.assert_array_equal(np.asarray(out[2]), np.zeros(1, dtype=np.int32))
        np.testing.assert_array_equal(np.asarray(out[0]), np.asarray(neighbor_list))
        np.testing.assert_array_equal(
            np.asarray(out[3]),
            np.asarray(neighbor_list_shifts),
        )

    def test_jit_query_coo_uses_fixed_buffer_capacity(self):
        """Fixed buffers keep a traced max_pairs argument out of sizing logic."""
        positions = jnp.zeros((32, 3), dtype=jnp.float32)
        cell = _orthorhombic_cell(6.0)
        state = build_cluster_tile_list(positions, 2.0, cell)
        neighbor_list = jnp.full((2, 64), -77, dtype=jnp.int32)
        neighbor_list_shifts = jnp.full((64, 3), -77, dtype=jnp.int32)

        @jax.jit
        def run(runtime_max_pairs):
            return query_cluster_tile_coo(
                state[0],
                state[2],
                state[3],
                state[4],
                state[11],
                state[12],
                state[13],
                cell,
                2.0,
                32,
                runtime_max_pairs,
                rebuild_flags=jnp.array([False], dtype=jnp.bool_),
                pair_offsets=jnp.array([0, 64], dtype=jnp.int32),
                pair_counts=jnp.array([7], dtype=jnp.int32),
                neighbor_list=neighbor_list,
                neighbor_list_shifts=neighbor_list_shifts,
            )

        out = run(jnp.array(1, dtype=jnp.int32))

        np.testing.assert_array_equal(np.asarray(out[2]), np.array([7], dtype=np.int32))
        np.testing.assert_array_equal(np.asarray(out[0]), np.asarray(neighbor_list))
        np.testing.assert_array_equal(
            np.asarray(out[3]),
            np.asarray(neighbor_list_shifts),
        )


class TestJaxClusterTileEmptySelective:
    """Selective zero-atom returns keep their fixed tuple layout."""

    @staticmethod
    def _tile_state():
        """Return distinctive single-system tile state buffers."""
        return {
            "previous_num_tiles": jnp.array([3], dtype=jnp.int32),
            "previous_tile_row_group": jnp.arange(4, dtype=jnp.int32),
            "previous_tile_col_group": jnp.arange(4, dtype=jnp.int32),
        }

    @pytest.mark.parametrize("rebuild_flag", [False, True])
    def test_empty_selective_coo_returns_segmented_state(self, rebuild_flag):
        """Zero atoms retain the seven-array single-system COO contract."""
        pair_offsets = jnp.array([0, 5], dtype=jnp.int32)
        previous_pair_counts = jnp.array([4], dtype=jnp.int32)
        previous_neighbor_list = jnp.full((2, 5), 7, dtype=jnp.int32)
        previous_neighbor_list_shifts = jnp.full((5, 3), 7, dtype=jnp.int32)
        tile_state = self._tile_state()

        out = cluster_tile_neighbor_list(
            jnp.empty((0, 3), dtype=jnp.float32),
            1.0,
            _orthorhombic_cell(4.0),
            max_neighbors=8,
            format="coo",
            rebuild_flags=jnp.array([rebuild_flag], dtype=jnp.bool_),
            pair_offsets=pair_offsets,
            previous_pair_counts=previous_pair_counts,
            previous_neighbor_list=previous_neighbor_list,
            previous_neighbor_list_shifts=previous_neighbor_list_shifts,
            **tile_state,
        )

        neighbor_list, offsets, pair_counts, shifts, num_tiles, row, col = out
        assert neighbor_list.shape == (2, 5)
        assert shifts.shape == (5, 3)
        np.testing.assert_array_equal(np.asarray(offsets), np.asarray(pair_offsets))
        np.testing.assert_array_equal(
            np.asarray(pair_counts),
            np.array([0 if rebuild_flag else 4], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(neighbor_list), np.asarray(previous_neighbor_list)
        )
        np.testing.assert_array_equal(
            np.asarray(shifts),
            np.asarray(previous_neighbor_list_shifts),
        )
        np.testing.assert_array_equal(
            np.asarray(num_tiles),
            np.array([0 if rebuild_flag else 3], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(row), np.asarray(tile_state["previous_tile_row_group"])
        )
        np.testing.assert_array_equal(
            np.asarray(col), np.asarray(tile_state["previous_tile_col_group"])
        )

    def test_empty_selective_coo_invalid_offsets_fail_closed(self):
        """Malformed metadata zeros an inactive empty-system count."""
        pair_offsets = jnp.array([0, 3], dtype=jnp.int32)
        previous_pair_counts = jnp.array([4], dtype=jnp.int32)
        previous_neighbor_list = jnp.full((2, 5), 7, dtype=jnp.int32)
        previous_neighbor_list_shifts = jnp.full((5, 3), 7, dtype=jnp.int32)

        out = cluster_tile_neighbor_list(
            jnp.empty((0, 3), dtype=jnp.float32),
            1.0,
            _orthorhombic_cell(4.0),
            max_neighbors=8,
            format="coo",
            rebuild_flags=jnp.array([False], dtype=jnp.bool_),
            pair_offsets=pair_offsets,
            previous_pair_counts=previous_pair_counts,
            previous_neighbor_list=previous_neighbor_list,
            previous_neighbor_list_shifts=previous_neighbor_list_shifts,
            **self._tile_state(),
        )

        np.testing.assert_array_equal(np.asarray(out[2]), np.zeros(1, dtype=np.int32))
        np.testing.assert_array_equal(
            np.asarray(out[0]), np.asarray(previous_neighbor_list)
        )
        np.testing.assert_array_equal(
            np.asarray(out[3]),
            np.asarray(previous_neighbor_list_shifts),
        )

    def test_empty_selective_coo_rejects_mismatched_fixed_buffer_shapes(self):
        """Fixed COO and shift buffers must describe one shared capacity."""
        with pytest.raises(ValueError, match="neighbor_list_shifts"):
            cluster_tile_neighbor_list(
                jnp.empty((0, 3), dtype=jnp.float32),
                1.0,
                _orthorhombic_cell(4.0),
                max_neighbors=8,
                format="coo",
                rebuild_flags=jnp.array([False], dtype=jnp.bool_),
                pair_offsets=jnp.array([0, 5], dtype=jnp.int32),
                previous_pair_counts=jnp.array([4], dtype=jnp.int32),
                previous_neighbor_list=jnp.full((2, 5), 7, dtype=jnp.int32),
                previous_neighbor_list_shifts=jnp.full((4, 3), 7, dtype=jnp.int32),
                **self._tile_state(),
            )

    @pytest.mark.parametrize("rebuild_flag", [False, True])
    def test_empty_selective_matrix_returns_state(self, rebuild_flag):
        """Zero atoms retain the six-array selective matrix contract."""
        tile_state = self._tile_state()
        out = cluster_tile_neighbor_list(
            jnp.empty((0, 3), dtype=jnp.float32),
            1.0,
            _orthorhombic_cell(4.0),
            max_neighbors=8,
            rebuild_flags=jnp.array([rebuild_flag], dtype=jnp.bool_),
            previous_neighbor_matrix=jnp.empty((0, 8), dtype=jnp.int32),
            previous_num_neighbors=jnp.empty(0, dtype=jnp.int32),
            previous_neighbor_matrix_shifts=jnp.empty((0, 8, 3), dtype=jnp.int32),
            **tile_state,
        )

        assert len(out) == 6
        assert out[0].shape == (0, 8)
        assert out[1].shape == (0,)
        assert out[2].shape == (0, 8, 3)
        for returned, name in zip(
            out[3:],
            (
                "previous_num_tiles",
                "previous_tile_row_group",
                "previous_tile_col_group",
            ),
        ):
            expected = (
                np.zeros(1, dtype=np.int32)
                if rebuild_flag and name == "previous_num_tiles"
                else np.asarray(tile_state[name])
            )
            np.testing.assert_array_equal(
                np.asarray(returned),
                expected,
            )

    def test_empty_selective_dual_cutoff_matrix_returns_both_outputs(self):
        """Zero atoms retain both matrix triples plus selective tile state."""
        tile_state = self._tile_state()
        matrix = jnp.empty((0, 8), dtype=jnp.int32)
        counts = jnp.empty(0, dtype=jnp.int32)
        shifts = jnp.empty((0, 8, 3), dtype=jnp.int32)

        out = cluster_tile_neighbor_list(
            jnp.empty((0, 3), dtype=jnp.float32),
            1.0,
            _orthorhombic_cell(4.0),
            cutoff2=2.0,
            max_neighbors=8,
            rebuild_flags=jnp.array([True], dtype=jnp.bool_),
            previous_neighbor_matrix=matrix,
            previous_num_neighbors=counts,
            previous_neighbor_matrix_shifts=shifts,
            previous_neighbor_matrix2=matrix,
            previous_num_neighbors2=counts,
            previous_neighbor_matrix_shifts2=shifts,
            **tile_state,
        )

        assert len(out) == 9
        assert tuple(value.shape for value in out[:6]) == (
            (0, 8),
            (0,),
            (0, 8, 3),
            (0, 8),
            (0,),
            (0, 8, 3),
        )
        np.testing.assert_array_equal(
            np.asarray(out[6]),
            np.zeros(1, dtype=np.int32),
        )

    @pytest.mark.parametrize("rebuild_flag, expected_count", [(False, 4), (True, 0)])
    def test_jit_empty_selective_coo_keeps_fixed_arity(
        self,
        rebuild_flag,
        expected_count,
    ):
        """JIT preserves the seven-array zero-atom selective COO contract."""
        tile_state = self._tile_state()

        @jax.jit
        def build(positions, rebuild_flags):
            return cluster_tile_neighbor_list(
                positions,
                1.0,
                _orthorhombic_cell(4.0),
                max_neighbors=8,
                format="coo",
                rebuild_flags=rebuild_flags,
                pair_offsets=jnp.array([0, 5], dtype=jnp.int32),
                previous_pair_counts=jnp.array([4], dtype=jnp.int32),
                previous_neighbor_list=jnp.full((2, 5), 7, dtype=jnp.int32),
                previous_neighbor_list_shifts=jnp.full((5, 3), 7, dtype=jnp.int32),
                **tile_state,
            )

        out = build(
            jnp.empty((0, 3), dtype=jnp.float32),
            jnp.array([rebuild_flag], dtype=jnp.bool_),
        )

        assert len(out) == 7
        assert out[0].shape == (2, 5)
        assert out[2].shape == (1,)
        assert int(out[2][0]) == expected_count


class TestJaxClusterTileNeighborListDispatcher:
    """Unified JAX neighbor_list routes cluster-tile kwargs explicitly."""

    def test_explicit_method_cluster_tile_accepts_cutoff2(self):
        from nvalchemiops.jax.neighbors import neighbor_list

        positions = jnp.array(
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.4, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        cell = _orthorhombic_cell(5.0)
        out = neighbor_list(
            positions,
            1.0,
            cell=cell,
            pbc=jnp.ones(3, dtype=jnp.bool_),
            method="cluster_tile",
            cutoff2=1.6,
            max_neighbors=8,
        )
        assert len(out) == 6

    def test_explicit_method_cluster_tile_rebuild_flags_raise_clear_state_error(self):
        from nvalchemiops.jax.neighbors import neighbor_list

        positions = jnp.zeros((32, 3), dtype=jnp.float32)
        cell = _orthorhombic_cell(4.0)
        with pytest.raises(ValueError, match="previous cluster_tile state"):
            neighbor_list(
                positions,
                1.0,
                cell=cell,
                pbc=jnp.ones(3, dtype=jnp.bool_),
                method="cluster_tile",
                max_neighbors=8,
                rebuild_flags=jnp.array([True], dtype=jnp.bool_),
            )


class TestJaxClusterTileTileSizing:
    """The JAX tile buffer must geometry-size (concrete) / fall back (traced).

    Guards the fix that stopped the JAX default/auto-select path from silently
    undercounting dense high-cutoff systems where a fixed
    ``max_tiles_per_group=256`` cannot cover all neighboring row groups.
    """

    def test_geometry_sizing_scales_with_cutoff_when_concrete(self):
        from nvalchemiops.jax.neighbors.cluster_tile import (
            _tile_buffer_max_tiles_per_group,
        )

        n = 32768  # ngroup = 1024
        pos = jnp.zeros((n, 3), dtype=jnp.float32)
        cell = _orthorhombic_cell(69.8)
        # Low cutoff: floor (256).  High cutoff: must scale up so the tile
        # buffer (ngroup * min(ngroup, mtpg)) covers the dense tile count.
        low = _tile_buffer_max_tiles_per_group(pos, n, 6.0, cell)
        high = _tile_buffer_max_tiles_per_group(pos, n, 25.0, cell)
        assert low >= 256
        assert high > low
        # 25 A on this cell needs ~ngroup neighbour groups per row (dense).
        assert high >= 1024

    def test_traced_inputs_fall_back_to_ngroup(self):
        from nvalchemiops.jax.neighbors.cluster_tile import (
            _tile_buffer_max_tiles_per_group,
        )

        n = 2048  # ngroup = 64
        cell = _orthorhombic_cell(20.0)

        # Tracing positions (e.g. grad/jit) -> trace-safe ngroup fallback.
        def f(p):
            return _tile_buffer_max_tiles_per_group(p, n, 5.0, cell)

        captured = {}

        def grab(p):
            captured["mtpg"] = f(p)
            return p.sum()

        jax.grad(grab)(jnp.zeros((n, 3), dtype=jnp.float32))
        assert captured["mtpg"] == 64  # ngroup, capacity ngroup**2 (no overflow)
