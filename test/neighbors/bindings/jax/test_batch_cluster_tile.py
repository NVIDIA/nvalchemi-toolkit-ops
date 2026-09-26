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

"""Tests for JAX bindings of the batched cluster-pair tile neighbor list."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors import _cluster_tile_preload
from nvalchemiops.jax.neighbors.batch_cluster_tile import (
    _BATCH_CLUSTER_TILE_QUERIES,
    TILE_GROUP_SIZE,
    allocate_batch_cluster_tile_list,
    batch_build_cluster_tile_list,
    batch_cluster_tile_neighbor_list,
    batch_query_cluster_tile,
    estimate_batch_cluster_tile_list_sizes,
    estimate_batch_cluster_tile_segments,
    estimate_batch_max_tiles_per_group,
)
from nvalchemiops.neighbors.cluster_tile import estimate_max_tiles_per_group
from nvalchemiops.neighbors.neighbor_utils import (
    NeighborOverflowError,
    TileBufferOverflow,
)

from .conftest import requires_gpu

pytestmark = requires_gpu


def _make_batch(
    sys_sizes: list[int],
    cell_sizes: list[float],
    seed: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    rng = np.random.RandomState(seed)
    pos_chunks, cells = [], []
    for sz, L in zip(sys_sizes, cell_sizes):
        pos_chunks.append(rng.uniform(0, L, size=(sz, 3)).astype(np.float32))
        cells.append(np.eye(3, dtype=np.float32) * L)
    positions = jnp.array(np.concatenate(pos_chunks, axis=0))
    cell_batch = jnp.array(np.stack(cells, axis=0))
    bp = [0]
    for sz in sys_sizes:
        bp.append(bp[-1] + sz)
    batch_ptr = jnp.array(bp, dtype=jnp.int32)
    return positions, cell_batch, batch_ptr


def _traced_preload_device_count() -> int:
    """Return how many local devices a traced graph callback must preload."""
    local_devices = tuple(jax.local_devices())
    accelerators = tuple(
        device for device in local_devices if device.platform in {"gpu", "cuda", "rocm"}
    )
    return len(accelerators or local_devices)


class TestBatchClusterTileDualCutoffValidation:
    """Exercise the batched public matrix dual-cutoff boundaries."""

    def test_batch_query_rejects_reversed_dual_cutoffs_and_accepts_equal(self):
        """The direct batched matrix query validates cutoff ordering."""
        positions, cell_batch, batch_ptr = _make_batch([32], [4.0])
        positions = positions.astype(jnp.float32)
        tile_state = batch_cluster_tile_neighbor_list(
            positions,
            1.0,
            cell_batch,
            batch_ptr,
            format="tile",
        )
        query_args = (
            tile_state[4],
            tile_state[5],
            tile_state[6],
            tile_state[7],
            cell_batch,
            tile_state[0],
            tile_state[1],
            tile_state[2],
            tile_state[3],
            1.0,
            positions.shape[0],
            32,
        )

        with pytest.raises(
            ValueError,
            match="cutoff2 must be greater than or equal to cutoff",
        ):
            batch_query_cluster_tile(*query_args, cutoff2=0.5)

        result = batch_query_cluster_tile(*query_args, cutoff2=1.0)
        assert len(result) == 6

    def test_batch_wrapper_rejects_reversed_dual_cutoffs_and_accepts_equal(self):
        """The one-shot batch wrapper applies the same ordering contract."""
        positions, cell_batch, batch_ptr = _make_batch([32], [4.0])
        positions = positions.astype(jnp.float32)

        with pytest.raises(
            ValueError,
            match="cutoff2 must be greater than or equal to cutoff",
        ):
            batch_cluster_tile_neighbor_list(
                positions,
                1.0,
                cell_batch,
                batch_ptr,
                cutoff2=0.5,
            )

        result = batch_cluster_tile_neighbor_list(
            positions,
            1.0,
            cell_batch,
            batch_ptr,
            cutoff2=1.0,
        )
        assert len(result) == 6


class TestJaxBatchClusterTileValidation:
    """Validate public option combinations rejected before kernel launch."""

    def test_pair_outputs_reject_tile_format(self):
        """Pair-output requests are not supported with tile-format output."""
        positions, cell_batch, batch_ptr = _make_batch([32], [8.0])

        with pytest.raises(NotImplementedError, match="format='tile'"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                format="tile",
                return_distances=True,
            )

    def test_cutoff2_and_selective_reject_unsupported_formats(self):
        """Dual cutoff and selective rebuild are restricted output modes."""
        positions, cell_batch, batch_ptr = _make_batch([32], [8.0])

        with pytest.raises(NotImplementedError, match="format='matrix'"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                format="coo",
                cutoff2=3.0,
            )

        with pytest.raises(NotImplementedError, match="format='matrix' or segmented"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                format="tile",
                rebuild_flags=jnp.ones(1, dtype=jnp.bool_),
            )

    def test_selective_requires_previous_state(self):
        """Selective rebuild reports every required previous-state buffer."""
        positions, cell_batch, batch_ptr = _make_batch([32], [8.0])

        with pytest.raises(ValueError, match="previous batch_cluster_tile state"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                rebuild_flags=jnp.ones(1, dtype=jnp.bool_),
            )


class TestBatchTileNeighborListCorrectness:
    """Smoke + multi-system tests."""

    def test_single_system_batch(self):
        positions, cell_batch, batch_ptr = _make_batch([64], [10.0], seed=5)
        cutoff = 2.5
        nm, nn, _ = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
        )
        assert nm.shape == (64, 64)
        assert bool(jnp.all(nn >= 0))

    def test_topology_only_grad_matrix_is_zero(self):
        """Matrix topology from batched cluster-tile is nondifferentiable."""
        positions, cell_batch, batch_ptr = _make_batch([32, 32], [10.0, 10.0], seed=8)

        def loss(pos):
            neighbor_matrix, num_neighbors, shifts = batch_cluster_tile_neighbor_list(
                pos,
                2.0,
                cell_batch,
                batch_ptr,
                max_neighbors=32,
                max_tiles_per_group=1,
            )
            return (
                neighbor_matrix.astype(pos.dtype).sum()
                + num_neighbors.astype(pos.dtype).sum()
                + shifts.astype(pos.dtype).sum()
            )

        grad = jax.grad(loss)(positions)
        assert jnp.isfinite(grad).all().item()
        np.testing.assert_allclose(np.asarray(grad), 0.0)

    def test_two_systems_different_sizes(self):
        positions, cell_batch, batch_ptr = _make_batch([64, 96], [10.0, 8.0], seed=6)
        cutoff = 2.5
        nm, nn, _ = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
        )
        assert nm.shape == (160, 64)
        # Per-system slices have independent neighbor counts.
        nn_np = np.asarray(nn)
        assert nn_np[:64].sum() > 0
        assert nn_np[64:].sum() > 0

    def test_neighbors_stay_within_system(self):
        """Every emitted neighbor must share a system with its source atom."""
        positions, cell_batch, batch_ptr = _make_batch([48, 80], [9.0, 7.0], seed=7)
        cutoff = 2.5
        nm, nn, _ = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=32,
        )
        N = int(batch_ptr[-1])
        # Per-atom system index.
        sys_sizes = [int(batch_ptr[i + 1] - batch_ptr[i]) for i in range(2)]
        batch_idx = np.concatenate(
            [
                np.zeros(sys_sizes[0], dtype=np.int32),
                np.ones(sys_sizes[1], dtype=np.int32),
            ]
        )
        nm_np = np.asarray(nm)
        nn_np = np.asarray(nn)
        for i in range(N):
            for k in range(int(nn_np[i])):
                j = int(nm_np[i, k])
                if j == N:  # sentinel padding
                    continue
                assert batch_idx[i] == batch_idx[j], (
                    f"Cross-system pair ({i}, {j}) emitted"
                )


class TestBatchTileNeighborListFormats:
    """Tests for the three output formats."""

    def test_matrix_vs_coo_pair_count_match(self):
        positions, cell_batch, batch_ptr = _make_batch([48, 80], [9.0, 7.0], seed=8)
        cutoff = 2.5
        nm, nn, _ = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=32,
        )
        nl, ptr, _ = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=32,
            format="coo",
        )
        assert int(nn.sum()) == int(nl.shape[1])
        assert int(ptr[-1]) == int(nl.shape[1])

    def test_compact_coo_one_short_capacity_raises(self):
        """Eager batched compact COO reports the full required pair count."""
        positions = jnp.zeros((8, 3), dtype=jnp.float32)
        cell_batch = jnp.repeat(jnp.eye(3, dtype=jnp.float32)[None], 2, axis=0) * 8.0
        batch_ptr = jnp.array([0, 4, 8], dtype=jnp.int32)
        kwargs = {
            "max_neighbors": 8,
            "format": "coo",
        }
        adequate = batch_cluster_tile_neighbor_list(
            positions,
            2.0,
            cell_batch,
            batch_ptr,
            max_pairs=64,
            **kwargs,
        )
        required_pairs = int(adequate[0].shape[1])
        assert required_pairs > 0

        exact = batch_cluster_tile_neighbor_list(
            positions,
            2.0,
            cell_batch,
            batch_ptr,
            max_pairs=required_pairs,
            **kwargs,
        )
        assert exact[0].shape == (2, required_pairs)
        assert exact[2].shape == (required_pairs, 3)

        with pytest.raises(NeighborOverflowError) as caught:
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                max_pairs=required_pairs - 1,
                **kwargs,
            )

        assert caught.value.max_neighbors == required_pairs - 1
        assert caught.value.num_neighbors == required_pairs
        assert caught.value.system_index is None

    def test_tile_format_returns_state(self):
        positions, cell_batch, batch_ptr = _make_batch([32, 64], [8.0, 8.0], seed=9)
        out = batch_cluster_tile_neighbor_list(
            positions, 2.5, cell_batch, batch_ptr, format="tile"
        )
        # 11-tuple matching the torch sibling:
        # (num_tiles, tile_row_group, tile_col_group, tile_system,
        # sorted_atom_index, sorted_pos_x, sorted_pos_y, sorted_pos_z,
        # batch_idx_sorted, batch_ptr_padded, group_ptr).
        assert len(out) == 11
        num_tiles = out[0]
        assert int(num_tiles[0]) > 0


class TestBatchClusterTileGraphPreload:
    """Exercise batched WARP graph callbacks after kernel preload."""

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
        """A public batched matrix query preloads and executes under JIT."""
        _cluster_tile_preload._preload_cluster_tile_query_kernel_cached.cache_clear()
        positions, cell_batch, batch_ptr = _make_batch([32], [4.0], seed=5)

        @jax.jit
        def query(positions):
            return batch_cluster_tile_neighbor_list(
                positions,
                1.0,
                cell_batch,
                batch_ptr,
                max_neighbors=32,
                max_tiles_per_group=1,
            )

        neighbor_matrix, num_neighbors, _shifts = query(positions)
        neighbor_matrix.block_until_ready()
        assert (
            _cluster_tile_preload._preload_cluster_tile_query_kernel_cached.cache_info().currsize
            == _traced_preload_device_count()
        )
        assert int(num_neighbors.sum()) > 0

        second = query(positions)
        second[0].block_until_ready()
        assert (
            _cluster_tile_preload._preload_cluster_tile_query_kernel_cached.cache_info().currsize
            == _traced_preload_device_count()
        )

    def test_jitted_segmented_coo_registration_preloads_and_executes(self):
        """Segmented batched COO graph callback executes under JIT with fixed buffers."""
        _cluster_tile_preload._preload_cluster_tile_coo_kernel_cached.cache_clear()
        positions, cell_batch, batch_ptr = _make_batch([32, 64], [6.0, 6.0], seed=36)
        cutoff = 2.0
        max_neighbors = 64
        tile_state = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            format="tile",
        )
        (
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            _batch_idx_sorted,
            _batch_ptr_padded,
            _group_ptr,
        ) = tile_state
        _tile_caps, tile_offsets, _pair_caps, pair_offsets = (
            estimate_batch_cluster_tile_segments(batch_ptr, max_neighbors=max_neighbors)
        )
        max_pairs = int(pair_offsets[-1])
        rebuild_flags = jnp.array([True, True], dtype=jnp.bool_)
        tile_counts = jnp.zeros(2, dtype=jnp.int32)
        (
            sorted_atom_index,
            _sort_inv,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            _batch_idx_sorted,
            _batch_ptr_padded,
            _group_system,
            _group_ptr,
            _group_ctr_x,
            _group_ctr_y,
            _group_ctr_z,
            _group_ext_x,
            _group_ext_y,
            _group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            tile_counts,
        ) = batch_build_cluster_tile_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            rebuild_flags=rebuild_flags,
            tile_offsets=tile_offsets,
            tile_counts=tile_counts,
            num_tiles=num_tiles,
            tile_row_group=tile_row_group,
            tile_col_group=tile_col_group,
            tile_system=tile_system,
        )
        pair_counts = jnp.zeros(2, dtype=jnp.int32)
        pair_counter = jnp.zeros(1, dtype=jnp.int32)
        neighbor_list = jnp.zeros((2, max_pairs), dtype=jnp.int32)
        coo_list = neighbor_list.T.copy()
        coo_shifts = jnp.zeros((max_pairs, 3), dtype=jnp.int32)
        inv_cell_batch = jnp.linalg.inv(cell_batch)
        registration = _BATCH_CLUSTER_TILE_QUERIES["coo_segmented"]

        @jax.jit
        def query(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            pair_counter,
            pair_offsets,
            pair_counts,
            coo_list,
            coo_shifts,
        ):
            registration.preload()
            return registration.callable(
                sorted_atom_index,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                cell_batch,
                inv_cell_batch,
                num_tiles,
                tile_offsets,
                tile_counts,
                rebuild_flags,
                tile_row_group,
                tile_col_group,
                tile_system,
                pair_counter,
                pair_offsets,
                pair_counts,
                coo_list,
                coo_shifts,
                cutoff,
                positions.shape[0],
                max_pairs,
            )

        pair_counter, pair_counts, coo_list, coo_shifts = query(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            pair_counter,
            pair_offsets,
            pair_counts,
            coo_list,
            coo_shifts,
        )
        pair_counts.block_until_ready()
        assert (
            _cluster_tile_preload._preload_cluster_tile_coo_kernel_cached.cache_info().currsize
            == 1
        )
        assert pair_counts.shape == (2,)
        assert int(pair_counts.sum()) > 0

        pair_counter2, pair_counts2, coo_list2, coo_shifts2 = query(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            pair_counter,
            pair_offsets,
            pair_counts,
            coo_list,
            coo_shifts,
        )
        pair_counts2.block_until_ready()
        assert (
            _cluster_tile_preload._preload_cluster_tile_coo_kernel_cached.cache_info().currsize
            == 1
        )


class TestBatchTileNeighborListErrors:
    """Error path tests."""

    def test_wrong_dtype_raises(self):
        positions = jnp.zeros((32, 3), dtype=jnp.float64)
        cell_batch = jnp.eye(3, dtype=jnp.float64)[jnp.newaxis, :, :]
        batch_ptr = jnp.array([0, 32], dtype=jnp.int32)
        with pytest.raises(TypeError):
            batch_cluster_tile_neighbor_list(positions, 1.0, cell_batch, batch_ptr)

    def test_bad_cell_shape_raises(self):
        positions = jnp.zeros((32, 3), dtype=jnp.float32)
        cell_batch = jnp.eye(3, dtype=jnp.float32)  # missing system axis
        batch_ptr = jnp.array([0, 32], dtype=jnp.int32)
        with pytest.raises(ValueError, match="cell_batch"):
            batch_cluster_tile_neighbor_list(positions, 1.0, cell_batch, batch_ptr)

    def test_tile_buffer_overflow_raises(self):
        """Compact and segmented eager paths report tile-buffer overflow."""
        positions = jnp.zeros((128, 3), dtype=jnp.float32)
        cell_batch = jnp.eye(3, dtype=jnp.float32)[None] * 12.0
        batch_ptr = jnp.array([0, 128], dtype=jnp.int32)
        for return_distances in (False, True):
            with pytest.raises(TileBufferOverflow) as caught:
                batch_cluster_tile_neighbor_list(
                    positions,
                    5.0,
                    cell_batch,
                    batch_ptr,
                    max_neighbors=256,
                    max_tiles_per_group=1,
                    return_distances=return_distances,
                )
            assert caught.value.num_tiles > caught.value.max_tiles
            assert caught.value.system_index is None

        segmented_positions = jnp.zeros((160, 3), dtype=jnp.float32)
        segmented_cells = jnp.tile(
            jnp.eye(3, dtype=jnp.float32)[None] * 12.0, (2, 1, 1)
        )
        segmented_ptr = jnp.array([0, 32, 160], dtype=jnp.int32)
        with pytest.raises(TileBufferOverflow) as segmented:
            batch_cluster_tile_neighbor_list(
                segmented_positions,
                5.0,
                segmented_cells,
                segmented_ptr,
                max_neighbors=256,
                rebuild_flags=jnp.ones(2, dtype=jnp.bool_),
                tile_offsets=jnp.array([0, 1, 2], dtype=jnp.int32),
                previous_tile_counts=jnp.zeros(2, dtype=jnp.int32),
                previous_num_tiles=jnp.zeros(1, dtype=jnp.int32),
                previous_tile_row_group=jnp.zeros(2, dtype=jnp.int32),
                previous_tile_col_group=jnp.zeros(2, dtype=jnp.int32),
                previous_tile_system=jnp.zeros(2, dtype=jnp.int32),
                previous_neighbor_matrix=jnp.empty((160, 256), dtype=jnp.int32),
                previous_num_neighbors=jnp.zeros(160, dtype=jnp.int32),
                previous_neighbor_matrix_shifts=jnp.empty(
                    (160, 256, 3), dtype=jnp.int32
                ),
                max_tiles_per_group=1,
            )
        assert segmented.value.system_index == 1
        assert segmented.value.max_tiles == 1
        assert segmented.value.num_tiles > segmented.value.max_tiles


class TestBatchClusterTileBuildCapacity:
    """Direct batched builders report compact and segmented tile overflow."""

    def test_full_build_overflow_and_adequate_retry(self):
        """Compact overflow reports the global required count and retry works."""
        positions = jnp.zeros((64, 3), dtype=jnp.float32)
        cell_batch = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None] * 12.0, (2, 1, 1))
        batch_ptr = jnp.array([0, 32, 64], dtype=jnp.int32)

        undersized = allocate_batch_cluster_tile_list(
            batch_ptr, 64, max_tiles_per_group=1
        )
        with pytest.raises(TileBufferOverflow) as caught:
            batch_build_cluster_tile_list(
                positions,
                5.0,
                cell_batch,
                batch_ptr,
                max_tiles_per_group=256,
                tile_offsets=jnp.array([0, 1000, 2000], dtype=jnp.int32),
                tile_counts=jnp.zeros(2, dtype=jnp.int32),
                num_tiles=undersized[0],
                tile_row_group=undersized[1][:1],
                tile_col_group=undersized[2][:1],
                tile_system=undersized[3][:1],
            )
        assert caught.value.max_tiles == 1
        required = caught.value.num_tiles

        adequate = allocate_batch_cluster_tile_list(
            batch_ptr, 64, max_tiles_per_group=1
        )
        state = batch_build_cluster_tile_list(
            positions,
            5.0,
            cell_batch,
            batch_ptr,
            max_tiles_per_group=256,
            num_tiles=adequate[0],
            tile_row_group=adequate[1],
            tile_col_group=adequate[2],
            tile_system=adequate[3],
        )
        assert int(state[15][0]) == required
        neighbor_matrix, num_neighbors, shifts = batch_query_cluster_tile(
            state[0],
            state[2],
            state[3],
            state[4],
            cell_batch,
            state[15],
            state[16],
            state[17],
            state[18],
            5.0,
            positions.shape[0],
            32,
        )
        assert np.all(np.asarray(num_neighbors) == 31)
        matrix = np.asarray(neighbor_matrix)
        for atom in range(64):
            system_start = 0 if atom < 32 else 32
            expected = set(range(system_start, system_start + 32))
            expected.remove(atom)
            assert set(matrix[atom, :31]) == expected
        assert shifts.shape == (64, 32, 3)

    def test_selective_build_reports_overflowing_system(self):
        """Segmented selective overflow identifies the system with the short segment."""
        positions = jnp.zeros((96, 3), dtype=jnp.float32)
        cell_batch = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None] * 12.0, (2, 1, 1))
        batch_ptr = jnp.array([0, 32, 96], dtype=jnp.int32)
        with pytest.raises(TileBufferOverflow) as caught:
            batch_build_cluster_tile_list(
                positions,
                5.0,
                cell_batch,
                batch_ptr,
                max_tiles_per_group=256,
                rebuild_flags=jnp.ones(2, dtype=jnp.bool_),
                tile_offsets=jnp.array([0, 1, 2], dtype=jnp.int32),
                tile_counts=jnp.zeros(2, dtype=jnp.int32),
                num_tiles=jnp.zeros(1, dtype=jnp.int32),
                tile_row_group=jnp.zeros(2, dtype=jnp.int32),
                tile_col_group=jnp.zeros(2, dtype=jnp.int32),
                tile_system=jnp.zeros(2, dtype=jnp.int32),
            )
        assert caught.value.system_index == 1
        assert caught.value.max_tiles == 1
        assert caught.value.num_tiles > caught.value.max_tiles

    def test_jit_complete_supplied_storage_omits_capacity_factor(self):
        """Complete batched tile arrays remove only the allocation-time factor."""
        positions = jnp.zeros((64, 3), dtype=jnp.float32)
        cell_batch = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None] * 12.0, (2, 1, 1))
        batch_ptr = jnp.array([0, 32, 64], dtype=jnp.int32)

        @jax.jit
        def build(positions, tile_row_group, tile_col_group, tile_system):
            return batch_build_cluster_tile_list(
                positions,
                5.0,
                cell_batch,
                batch_ptr,
                tile_row_group=tile_row_group,
                tile_col_group=tile_col_group,
                tile_system=tile_system,
            )

        state = build(
            positions,
            jnp.zeros(2, dtype=jnp.int32),
            jnp.zeros(2, dtype=jnp.int32),
            jnp.zeros(2, dtype=jnp.int32),
        )
        assert int(state[15][0]) == 2
        assert state[16].shape == state[17].shape == state[18].shape == (2,)

    def test_jit_partial_supplied_storage_still_requires_capacity_factor(self):
        """A missing batched tile-index array still requires static allocation."""
        positions = jnp.zeros((64, 3), dtype=jnp.float32)
        cell_batch = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None] * 12.0, (2, 1, 1))
        batch_ptr = jnp.array([0, 32, 64], dtype=jnp.int32)

        @jax.jit
        def build(positions, tile_row_group, tile_col_group):
            return batch_build_cluster_tile_list(
                positions,
                5.0,
                cell_batch,
                batch_ptr,
                tile_row_group=tile_row_group,
                tile_col_group=tile_col_group,
            )

        with pytest.raises(ValueError, match="static Python integer"):
            build(
                positions,
                jnp.zeros(2, dtype=jnp.int32),
                jnp.zeros(2, dtype=jnp.int32),
            )

    def test_jit_complete_storage_still_requires_static_batch_ptr(self):
        """Caller-owned tile storage does not make batch segmentation dynamic."""
        positions = jnp.zeros((64, 3), dtype=jnp.float32)
        cell_batch = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None] * 12.0, (2, 1, 1))

        @jax.jit
        def build(positions, batch_ptr, tile_row_group, tile_col_group, tile_system):
            return batch_build_cluster_tile_list(
                positions,
                5.0,
                cell_batch,
                batch_ptr,
                tile_row_group=tile_row_group,
                tile_col_group=tile_col_group,
                tile_system=tile_system,
            )

        with pytest.raises(ValueError, match="batch_ptr.*concrete"):
            build(
                positions,
                jnp.array([0, 32, 64], dtype=jnp.int32),
                jnp.zeros(2, dtype=jnp.int32),
                jnp.zeros(2, dtype=jnp.int32),
                jnp.zeros(2, dtype=jnp.int32),
            )

    @pytest.mark.parametrize("invalid_factor", [0, -1, True, 1.5])
    def test_complete_supplied_storage_rejects_invalid_factor(self, invalid_factor):
        """Complete batched buffers do not excuse an invalid explicit factor."""
        positions = jnp.zeros((32, 3), dtype=jnp.float32)
        cell_batch = jnp.eye(3, dtype=jnp.float32)[None] * 4.0
        batch_ptr = jnp.array([0, 32], dtype=jnp.int32)
        with pytest.raises(ValueError, match="positive integer"):
            batch_build_cluster_tile_list(
                positions,
                1.0,
                cell_batch,
                batch_ptr,
                max_tiles_per_group=invalid_factor,
                tile_row_group=jnp.zeros(1, dtype=jnp.int32),
                tile_col_group=jnp.zeros(1, dtype=jnp.int32),
                tile_system=jnp.zeros(1, dtype=jnp.int32),
            )

    def test_segmented_overflow_retries_reinitialize_resized_state(self):
        """Successive first-system failures lead to a full-state adequate retry."""
        positions = jnp.zeros((160, 3), dtype=jnp.float32)
        cell_batch = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None] * 12.0, (2, 1, 1))
        batch_ptr = jnp.array([0, 64, 160], dtype=jnp.int32)

        def build_with_offsets(tile_offsets):
            capacity = int(tile_offsets[-1])
            return batch_build_cluster_tile_list(
                positions,
                5.0,
                cell_batch,
                batch_ptr,
                rebuild_flags=jnp.ones(2, dtype=jnp.bool_),
                tile_offsets=tile_offsets,
                tile_counts=jnp.zeros(2, dtype=jnp.int32),
                num_tiles=jnp.zeros(1, dtype=jnp.int32),
                tile_row_group=jnp.zeros(capacity, dtype=jnp.int32),
                tile_col_group=jnp.zeros(capacity, dtype=jnp.int32),
                tile_system=jnp.zeros(capacity, dtype=jnp.int32),
            )

        with pytest.raises(TileBufferOverflow) as first:
            build_with_offsets(jnp.array([0, 1, 2], dtype=jnp.int32))
        assert first.value.system_index == 0
        assert first.value.num_tiles == 3

        with pytest.raises(TileBufferOverflow) as second:
            build_with_offsets(jnp.array([0, 3, 4], dtype=jnp.int32))
        assert second.value.system_index == 1
        assert second.value.num_tiles == 6

        state = build_with_offsets(jnp.array([0, 3, 9], dtype=jnp.int32))
        np.testing.assert_array_equal(np.asarray(state[-1]), np.array([3, 6]))
        assert state[16].shape == state[17].shape == state[18].shape == (9,)


class TestEstimateBatchSizes:
    """Pure-Python sizing helper tests."""

    def test_batch_max_tiles_per_group_matches_scalar_max(self):
        """Batched max-tile sizing should match the scalar estimator."""
        batch_ptr = jnp.array([0, 32, 32800], dtype=jnp.int32)
        cell_batch = jnp.asarray(
            np.stack(
                [
                    np.eye(3, dtype=np.float32) * 10.0,
                    np.eye(3, dtype=np.float32) * 10.0,
                ],
            ),
        )
        cutoff = 20.0
        expected = 256
        for start, stop, cell in zip(
            np.asarray(batch_ptr[:-1]),
            np.asarray(batch_ptr[1:]),
            np.asarray(cell_batch),
        ):
            expected = max(
                expected,
                estimate_max_tiles_per_group(
                    int(stop - start),
                    cutoff,
                    float(abs(np.linalg.det(cell))),
                ),
            )

        got = estimate_batch_max_tiles_per_group(batch_ptr, cutoff, cell_batch)

        assert got == expected
        assert got > 256

    def test_batch_max_tiles_per_group_rejects_short_batch_ptr_length(self):
        """Direct cluster-tile sizing rejects one-entry batch_ptr."""
        batch_ptr = jnp.array([0], dtype=jnp.int32)
        cell_batch = jnp.zeros((0, 3, 3), dtype=jnp.float32)
        with pytest.raises(ValueError, match="batch_ptr.*length at least 2"):
            estimate_batch_max_tiles_per_group(batch_ptr, 3.0, cell_batch)

    def test_batch_max_tiles_per_group_rejects_bad_cell_shape(self):
        """Public batch max-tile sizing rejects malformed concrete cells."""
        batch_ptr = jnp.array([0, 32, 64], dtype=jnp.int32)
        cell_batch = jnp.arange(18, dtype=jnp.float32)

        with pytest.raises(ValueError, match="cell_batch.*shape"):
            estimate_batch_max_tiles_per_group(batch_ptr, 2.0, cell_batch)

    def test_batch_max_tiles_per_group_rejects_cell_count_mismatch(self):
        """Public sizing requires one cell per batch segment."""
        batch_ptr = jnp.array([0, 32], dtype=jnp.int32)
        cell_batch = jnp.zeros((0, 3, 3), dtype=jnp.float32)

        with pytest.raises(ValueError, match="cell_volumes"):
            estimate_batch_max_tiles_per_group(batch_ptr, 2.0, cell_batch)

    def test_batch_tile_buffer_max_tiles_per_group_rejects_cell_batch_mismatch(self):
        """Non-empty batch pointers still validate against cell_batch length."""
        positions = jnp.zeros((32, 3), dtype=jnp.float32)
        batch_ptr = jnp.array([0, 32], dtype=jnp.int32)
        cell_batch = jnp.zeros((0, 3, 3), dtype=jnp.float32)

        with pytest.raises(ValueError, match="cell_volumes"):
            from nvalchemiops.jax.neighbors.batch_cluster_tile import (
                _batch_tile_buffer_max_tiles_per_group,
            )

            _batch_tile_buffer_max_tiles_per_group(
                positions,
                batch_ptr,
                2.0,
                cell_batch,
            )

    def test_batch_max_tiles_per_group_rejects_traced_cell_batch(self):
        """Public batch max-tile sizing requires concrete cell_batch."""
        batch_ptr = jnp.array([0, 32], dtype=jnp.int32)

        @jax.jit
        def call_with_traced_cell(cell_batch):
            return estimate_batch_max_tiles_per_group(batch_ptr, 2.0, cell_batch)

        with pytest.raises(ValueError, match="cell_batch.*concrete"):
            call_with_traced_cell(jnp.eye(3, dtype=jnp.float32)[None])

    def test_batch_max_tiles_per_group_rejects_traced_batch_ptr(self):
        """Public batch max-tile sizing requires concrete batch_ptr."""
        cell_batch = jnp.eye(3, dtype=jnp.float32)[None]

        @jax.jit
        def call_with_traced_batch_ptr(batch_ptr):
            return estimate_batch_max_tiles_per_group(batch_ptr, 2.0, cell_batch)

        with pytest.raises(ValueError, match="batch_ptr.*concrete"):
            call_with_traced_batch_ptr(jnp.array([0, 32], dtype=jnp.int32))

    def test_jit_requires_static_batch_ptr_for_tile_allocation(self):
        """A dynamic batch pointer cannot determine tile buffer capacity."""
        positions, cell_batch, batch_ptr = _make_batch([32], [4.0])

        @jax.jit
        def build(positions, batch_ptr):
            return batch_cluster_tile_neighbor_list(
                positions,
                1.0,
                cell_batch,
                batch_ptr,
                max_neighbors=32,
                max_tiles_per_group=1,
            )

        with pytest.raises(ValueError, match="close over batch_ptr"):
            build(positions, batch_ptr)

    def test_jit_requires_static_cutoff_for_tile_allocation(self):
        """A dynamic cutoff cannot determine tile buffer capacity."""
        positions, cell_batch, batch_ptr = _make_batch([32], [4.0])

        @jax.jit
        def build(positions, cutoff):
            return batch_cluster_tile_neighbor_list(
                positions,
                cutoff,
                cell_batch,
                batch_ptr,
                max_neighbors=32,
                max_tiles_per_group=1,
            )

        with pytest.raises(ValueError, match="close over cutoff before tracing"):
            build(positions, jnp.asarray(1.0, dtype=jnp.float32))

    def test_jit_dense_tile_output_with_explicit_capacity(self):
        """An explicit compiled capacity contains every dense tile pair."""
        num_groups = 512
        num_atoms = num_groups * TILE_GROUP_SIZE
        positions = jnp.zeros((num_atoms, 3), dtype=jnp.float32)
        cell_batch = (jnp.eye(3, dtype=jnp.float32) * 64.0)[None]
        batch_ptr = jnp.array([0, num_atoms], dtype=jnp.int32)

        @jax.jit
        def build(positions):
            return batch_cluster_tile_neighbor_list(
                positions,
                1.0,
                cell_batch,
                batch_ptr,
                format="tile",
                max_tiles_per_group=(num_groups + 2) // 2,
            )

        num_tiles, tile_row_group, tile_col_group, tile_system, *_ = build(positions)
        tile_count = int(num_tiles[0])
        expected_tiles = num_groups * (num_groups + 1) // 2

        assert tile_count == expected_tiles
        assert tile_count <= tile_row_group.shape[0]
        assert tile_row_group.shape == tile_col_group.shape == tile_system.shape

    def test_aligned_two_systems(self):
        batch_ptr = jnp.array([0, 64, 192], dtype=jnp.int32)
        n_padded, ngroup, _, _, num_systems = estimate_batch_cluster_tile_list_sizes(
            batch_ptr,
        )
        # Both systems already 32-aligned; n_padded sums to 64 + 128 = 192.
        assert n_padded == 192
        assert ngroup == 192 // TILE_GROUP_SIZE
        assert num_systems == 2

    def test_non_aligned_padding(self):
        # 33 atoms pad to 64; 80 atoms pad to 96. Total padded = 160.
        batch_ptr = jnp.array([0, 33, 113], dtype=jnp.int32)
        n_padded, _, _, _, num_systems = estimate_batch_cluster_tile_list_sizes(
            batch_ptr,
        )
        assert n_padded == 64 + 96
        assert num_systems == 2


class TestJaxBatchClusterTileAutograd:
    """Differentiable per-pair distances/vectors for the batched binding."""

    def _make_batch(self, n_per=32, box=5.0, scale=1.0):
        key = jax.random.key(0)
        pos = jax.random.normal(key, (2 * n_per, 3), dtype=jnp.float32) * scale
        batch_ptr = jnp.array([0, n_per, 2 * n_per], dtype=jnp.int32)
        cell_batch = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None] * box, (2, 1, 1))
        return pos, cell_batch, batch_ptr

    def test_forward_returns_distances_and_vectors(self):
        pos, cell_batch, batch_ptr = self._make_batch()
        out = batch_cluster_tile_neighbor_list(
            pos,
            1.5,
            cell_batch,
            batch_ptr,
            return_distances=True,
            return_vectors=True,
        )
        assert len(out) == 5
        nm, nn, shifts, d, v = out
        assert d.shape == nm.shape
        assert v.shape == nm.shape + (3,)

    def test_grad_positions_finite(self):
        pos, cell_batch, batch_ptr = self._make_batch()

        def loss(p):
            *_, d, _ = batch_cluster_tile_neighbor_list(
                p,
                1.5,
                cell_batch,
                batch_ptr,
                max_tiles_per_group=2,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        g = jax.grad(loss)(pos)
        assert g.shape == pos.shape
        assert jnp.isfinite(g).all().item()

    def test_grad_cell_finite(self):
        pos, cell_batch, batch_ptr = self._make_batch()

        def loss(c):
            *_, d, _ = batch_cluster_tile_neighbor_list(
                pos,
                1.5,
                c,
                batch_ptr,
                max_tiles_per_group=2,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        g = jax.grad(loss)(cell_batch)
        assert g.shape == cell_batch.shape
        assert jnp.isfinite(g).all().item()

    def test_check_grads_against_finite_differences(self):
        from jax.test_util import check_grads

        # Cluster_tile is fp32-only; use a tight cluster + large box +
        # wide cutoff to push the neighbor-set discontinuity out of FD
        # reach, with a larger FD step for fp32 numerical headroom.
        key = jax.random.key(0)
        n_per = 32
        pos = jax.random.normal(key, (2 * n_per, 3), dtype=jnp.float32) * 0.15
        batch_ptr = jnp.array([0, n_per, 2 * n_per], dtype=jnp.int32)
        cell_batch = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None] * 20.0, (2, 1, 1))

        # Drain asynchronous input construction before Warp starts CUDA graph capture.
        jax.block_until_ready((pos, cell_batch, batch_ptr))

        def loss(p):
            *_, d, _ = batch_cluster_tile_neighbor_list(
                p,
                5.0,
                cell_batch,
                batch_ptr,
                max_tiles_per_group=2,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        check_grads(
            loss, (pos,), order=1, atol=1e-4, rtol=1e-4, modes=["rev"], eps=1e-3
        )

    def test_pair_fn_supported(self):
        """pair_fn is now wired through the JAX batch_cluster_tile binding
        (matrix and COO; returns per-pair pe/pf).  See test_pair_fn.py for coverage."""
        from .test_pair_fn import _sum_pair_fn_f32

        pos, cell_batch, batch_ptr = self._make_batch()
        pp = ((jnp.arange(pos.shape[0], dtype=jnp.float32) + 1.0) * 0.5).reshape(-1, 1)
        out = batch_cluster_tile_neighbor_list(
            pos,
            1.5,
            cell_batch,
            batch_ptr,
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

    def test_hessian_vector_product_smoke(self):
        """fp32 second-order HVP smoke — see TestJaxClusterTileAutograd."""
        pos, cell_batch, batch_ptr = self._make_batch()
        v = jax.random.normal(jax.random.key(1), pos.shape, dtype=pos.dtype)

        def loss(p):
            *_, d, _ = batch_cluster_tile_neighbor_list(
                p,
                1.5,
                cell_batch,
                batch_ptr,
                max_tiles_per_group=2,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        hvp = jax.grad(lambda p: jnp.vdot(jax.grad(loss)(p), v))(pos)
        assert jnp.isfinite(hvp).all().item()
        assert hvp.shape == pos.shape

    def test_no_grad_path_unchanged(self):
        pos, cell_batch, batch_ptr = self._make_batch()
        nm_a, nn_a, sh_a = batch_cluster_tile_neighbor_list(
            pos,
            1.5,
            cell_batch,
            batch_ptr,
        )
        nm_b, nn_b, sh_b, d_b, v_b = batch_cluster_tile_neighbor_list(
            pos,
            1.5,
            cell_batch,
            batch_ptr,
            return_distances=True,
            return_vectors=True,
        )
        assert jnp.all(nn_a == nn_b)
        for i in range(nm_a.shape[0]):
            n = int(nn_a[i])
            row_a = sorted(int(x) for x in nm_a[i, :n])
            row_b = sorted(int(x) for x in nm_b[i, :n])
            assert row_a == row_b
        assert jnp.isfinite(d_b).all().item()
        assert jnp.isfinite(v_b).all().item()


# Reuse the single-system brute-force helpers from the sibling test file.
# They operate on numpy arrays and one cell at a time, so per-system
# slicing is enough to extend to the batched case.
from test.neighbors.bindings.jax.test_cluster_tile import (  # noqa: E402
    _brute_force_pairs_full,
    _matrix_to_pair_set_full,
)


class TestJaxBatchClusterTileBruteForce:
    """Pair-set + shift identity checks for the batched binding.

    Mirrors the single-system :class:`TestJaxClusterTileBruteForce` —
    runs the batched cluster_tile build then compares each system's
    output against a per-system numpy brute-force reference.
    """

    def test_skewed_cell_exact_image_all_formats(self):
        """Batched JAX matrix, COO, and tile paths use exact cell images."""
        positions = jnp.array(
            [[4.0, 5.0, 0.0], [8.3, 3.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        cell_batch = jnp.array(
            [
                [[10.0, 0.0, 0.0], [4.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ],
            dtype=jnp.float32,
        )
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        cutoff = 4.8
        expected = {
            (0, 1, 0, 0, 0),
            (1, 0, 0, 0, 0),
            (2, 3, 0, 0, 0),
            (3, 2, 0, 0, 0),
        }

        matrix, counts, shifts, distances, vectors = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=8,
            return_distances=True,
            return_vectors=True,
        )
        assert _matrix_to_pair_set_full(matrix, counts, shifts, 4) == expected
        np.testing.assert_allclose(
            np.asarray(vectors)[:, 0],
            [[4.3, -2.0, 0.0], [-4.3, 2.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        )
        np.testing.assert_allclose(
            np.asarray(distances)[:, 0], [np.hypot(4.3, 2.0)] * 2 + [1.0] * 2
        )

        pairs, pointer, coo_shifts, coo_distances, coo_vectors = (
            batch_cluster_tile_neighbor_list(
                positions,
                cutoff,
                cell_batch,
                batch_ptr,
                max_neighbors=8,
                format="coo",
                max_pairs=8,
                return_distances=True,
                return_vectors=True,
            )
        )
        pair_array = np.asarray(pairs)
        shift_array = np.asarray(coo_shifts)
        got_coo = {
            (
                int(pair_array[0, slot]),
                int(pair_array[1, slot]),
                *(int(value) for value in shift_array[slot]),
            )
            for slot in range(pair_array.shape[1])
        }
        assert got_coo == expected
        np.testing.assert_array_equal(np.asarray(pointer), [0, 1, 2, 3, 4])
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(coo_vectors), axis=1),
            np.asarray(coo_distances),
        )

        tile = batch_cluster_tile_neighbor_list(
            positions, cutoff, cell_batch, batch_ptr, format="tile"
        )
        num_tiles, tile_rows, tile_cols, tile_system = tile[:4]
        tile_count = int(np.asarray(num_tiles)[0])
        assert tile_count == 2
        assert set(np.asarray(tile_system)[:tile_count].tolist()) == {0, 1}
        assert np.all(
            np.asarray(tile_rows)[:tile_count] <= np.asarray(tile_cols)[:tile_count]
        )

    def test_two_systems_random_pbc_matches_brute_force(self):
        sys_sizes = [10, 12]
        cell_sizes = [4.0, 3.5]
        positions, cell_batch, batch_ptr = _make_batch(sys_sizes, cell_sizes, seed=3)
        cutoff = 1.4

        nm, nn, shifts = batch_cluster_tile_neighbor_list(
            positions, cutoff, cell_batch, batch_ptr, max_neighbors=32
        )
        nm_np = np.asarray(nm)
        nn_np = np.asarray(nn)
        sh_np = np.asarray(shifts)
        pos_np = np.asarray(positions)
        cb_np = np.asarray(cell_batch)
        bp_np = np.asarray(batch_ptr)

        for sys_idx in range(len(sys_sizes)):
            start, end = int(bp_np[sys_idx]), int(bp_np[sys_idx + 1])
            local_pos = pos_np[start:end]
            local_cell = cb_np[sys_idx]
            # Extract the per-system local pair set from the batched output.
            got: set[tuple[int, int, int, int, int]] = set()
            for i in range(start, end):
                ni = int(nn_np[i])
                for k in range(ni):
                    j = int(nm_np[i, k])
                    if not (start <= j < end):
                        # Cross-system neighbor would be a bug — but on
                        # the contiguous-batch_idx contract, the kernel
                        # only emits within-system pairs.
                        continue
                    sx, sy, sz = (int(x) for x in sh_np[i, k])
                    i_loc, j_loc = i - start, j - start
                    # Full-fill: collect all directed (i, j, shift) triples.
                    got.add((i_loc, j_loc, sx, sy, sz))
            ref = _brute_force_pairs_full(local_pos, local_cell, cutoff, pbc=True)
            assert got == ref, (
                f"system {sys_idx}: cluster_tile output disagrees with "
                f"brute-force\n  missing: {ref - got}\n  extra: {got - ref}"
            )


class TestJaxBatchClusterTileCutoff2Selective:
    """Matrix-only cutoff2 and selective rebuild coverage for batches."""

    def test_segment_sizing_helper_is_exported(self):
        batch_ptr = jnp.array([0, 32, 96], dtype=jnp.int32)
        tile_caps, tile_offsets, pair_caps, pair_offsets = (
            estimate_batch_cluster_tile_segments(batch_ptr, max_neighbors=16)
        )
        assert tile_caps.shape == (2,)
        assert tile_offsets.shape == (3,)
        assert pair_caps.tolist() == [32 * 16, 64 * 16]
        assert int(pair_offsets[-1]) == sum(pair_caps.tolist())

    def test_cutoff2_matrix_returns_two_cutoff_groups(self):
        positions, cell_batch, batch_ptr = _make_batch([32, 32], [6.0, 6.0], seed=31)
        out = batch_cluster_tile_neighbor_list(
            positions,
            1.0,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            cutoff2=2.0,
        )
        assert len(out) == 6
        _nm1, nn1, _sh1, _nm2, nn2, _sh2 = out
        assert int(nn2.sum()) >= int(nn1.sum())

    @pytest.mark.parametrize("cutoff, cutoff2", [(0.91, 4.0), (4.0, 4.0)])
    def test_default_capacity_uses_larger_dual_cutoff(self, cutoff, cutoff2):
        """Ordered and equal dual cutoffs preserve each input-order matrix."""
        positions = jnp.stack(
            (
                jnp.arange(40, dtype=jnp.float32) * jnp.float32(0.05),
                jnp.zeros(40, dtype=jnp.float32),
                jnp.zeros(40, dtype=jnp.float32),
            ),
            axis=1,
        )
        cell_batch = jnp.eye(3, dtype=jnp.float32)[None] * 10.0
        batch_ptr = jnp.array([0, 40], dtype=jnp.int32)
        out = batch_cluster_tile_neighbor_list(
            positions, cutoff, cell_batch, batch_ptr, cutoff2=cutoff2
        )
        for offset, reference_cutoff in ((0, cutoff), (3, cutoff2)):
            got = _matrix_to_pair_set_full(
                *out[offset : offset + 3], positions.shape[0]
            )
            reference = _brute_force_pairs_full(
                np.asarray(positions),
                np.asarray(cell_batch[0]),
                reference_cutoff,
                pbc=True,
            )
            assert got == reference

    def test_rebuild_flags_false_preserves_previous_batch_outputs(self):
        positions, cell_batch, batch_ptr = _make_batch([32, 64], [6.0, 6.0], seed=32)
        cutoff = 2.0
        nm, nn, shifts = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
        )
        tile_state = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            format="tile",
        )
        num_tiles, tile_row_group, tile_col_group, tile_system, *_ = tile_state
        _tile_caps, tile_offsets, _pair_caps, _pair_offsets = (
            estimate_batch_cluster_tile_segments(batch_ptr, max_neighbors=64)
        )
        tile_counts = jnp.zeros((2,), dtype=jnp.int32)

        moved = positions.at[0, 0].add(0.25)
        out = batch_cluster_tile_neighbor_list(
            moved,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            rebuild_flags=jnp.array([False, False], dtype=jnp.bool_),
            tile_offsets=tile_offsets,
            previous_tile_counts=tile_counts,
            previous_num_tiles=num_tiles,
            previous_tile_row_group=tile_row_group,
            previous_tile_col_group=tile_col_group,
            previous_tile_system=tile_system,
            previous_neighbor_matrix=nm,
            previous_num_neighbors=nn,
            previous_neighbor_matrix_shifts=shifts,
        )
        nm2, nn2, shifts2, *_state = out
        np.testing.assert_array_equal(np.asarray(nm2), np.asarray(nm))
        np.testing.assert_array_equal(np.asarray(nn2), np.asarray(nn))
        np.testing.assert_array_equal(np.asarray(shifts2), np.asarray(shifts))

    def test_mixed_rebuild_flags_preserve_unflagged_system(self):
        """Rebuilding one system leaves the other system's topology intact."""
        positions, cell_batch, batch_ptr = _make_batch([32, 64], [6.0, 6.0], seed=34)
        cutoff = 2.0
        max_neighbors = 64
        n_atoms = int(batch_ptr[-1])
        (
            empty_num_tiles,
            empty_tile_row_group,
            empty_tile_col_group,
            empty_tile_system,
            empty_tile_counts,
            tile_offsets,
        ) = allocate_batch_cluster_tile_list(batch_ptr, max_neighbors)
        initial = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=max_neighbors,
            rebuild_flags=jnp.ones(2, dtype=jnp.bool_),
            tile_offsets=tile_offsets,
            previous_tile_counts=empty_tile_counts,
            previous_num_tiles=empty_num_tiles,
            previous_tile_row_group=empty_tile_row_group,
            previous_tile_col_group=empty_tile_col_group,
            previous_tile_system=empty_tile_system,
            previous_neighbor_matrix=jnp.full(
                (n_atoms, max_neighbors), n_atoms, dtype=jnp.int32
            ),
            previous_num_neighbors=jnp.zeros(n_atoms, dtype=jnp.int32),
            previous_neighbor_matrix_shifts=jnp.zeros(
                (n_atoms, max_neighbors, 3), dtype=jnp.int32
            ),
        )
        (
            initial_matrix,
            initial_counts,
            initial_shifts,
            _initial_offsets,
            initial_tile_counts,
            initial_num_tiles,
            initial_tile_row_group,
            initial_tile_col_group,
            initial_tile_system,
        ) = initial

        # Change both systems to a dense geometry. Only the first is rebuilt;
        # the second must retain its previous topology even though rebuilding it
        # would produce a detectably different result.
        moved = jnp.zeros_like(positions)
        mixed = batch_cluster_tile_neighbor_list(
            moved,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=max_neighbors,
            rebuild_flags=jnp.array([True, False], dtype=jnp.bool_),
            tile_offsets=tile_offsets,
            previous_tile_counts=initial_tile_counts,
            previous_num_tiles=initial_num_tiles,
            previous_tile_row_group=initial_tile_row_group,
            previous_tile_col_group=initial_tile_col_group,
            previous_tile_system=initial_tile_system,
            previous_neighbor_matrix=initial_matrix,
            previous_num_neighbors=initial_counts,
            previous_neighbor_matrix_shifts=initial_shifts,
        )
        mixed_matrix, mixed_counts, mixed_shifts, _, mixed_tile_counts, *_ = mixed

        fresh_matrix, fresh_counts, fresh_shifts = batch_cluster_tile_neighbor_list(
            moved,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=max_neighbors,
        )

        np.testing.assert_array_equal(
            np.asarray(mixed_counts[32:]), np.asarray(initial_counts[32:])
        )
        assert np.any(np.asarray(fresh_counts[32:]) != np.asarray(initial_counts[32:]))
        assert int(mixed_tile_counts[1]) == int(initial_tile_counts[1])
        initial_matrix_np = np.asarray(initial_matrix)
        initial_shifts_np = np.asarray(initial_shifts)
        mixed_matrix_np = np.asarray(mixed_matrix)
        mixed_shifts_np = np.asarray(mixed_shifts)
        fresh_matrix_np = np.asarray(fresh_matrix)
        fresh_shifts_np = np.asarray(fresh_shifts)

        for atom in range(32):
            mixed_pairs = {
                (
                    int(mixed_matrix_np[atom, index]),
                    *map(int, mixed_shifts_np[atom, index]),
                )
                for index in range(int(mixed_counts[atom]))
            }
            fresh_pairs = {
                (
                    int(fresh_matrix_np[atom, index]),
                    *map(int, fresh_shifts_np[atom, index]),
                )
                for index in range(int(fresh_counts[atom]))
            }
            assert mixed_pairs == fresh_pairs

        for atom in range(32, 96):
            count = int(initial_counts[atom])
            initial_pairs = {
                (
                    int(initial_matrix_np[atom, index]),
                    *map(int, initial_shifts_np[atom, index]),
                )
                for index in range(count)
            }
            mixed_pairs = {
                (
                    int(mixed_matrix_np[atom, index]),
                    *map(int, mixed_shifts_np[atom, index]),
                )
                for index in range(int(mixed_counts[atom]))
            }
            assert mixed_pairs == initial_pairs

    def test_rebuild_flags_true_from_empty_segmented_state(self):
        positions, cell_batch, batch_ptr = _make_batch([32, 64], [6.0, 6.0], seed=33)
        cutoff = 2.0
        max_neighbors = 64
        _tile_caps, tile_offsets, _pair_caps, _pair_offsets = (
            estimate_batch_cluster_tile_segments(batch_ptr, max_neighbors=max_neighbors)
        )
        max_tiles = int(tile_offsets[-1])
        n_atoms = int(batch_ptr[-1])
        out = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=max_neighbors,
            rebuild_flags=jnp.array([True, True], dtype=jnp.bool_),
            tile_offsets=tile_offsets,
            previous_tile_counts=jnp.zeros((2,), dtype=jnp.int32),
            previous_num_tiles=jnp.zeros((1,), dtype=jnp.int32),
            previous_tile_row_group=jnp.zeros(max_tiles, dtype=jnp.int32),
            previous_tile_col_group=jnp.zeros(max_tiles, dtype=jnp.int32),
            previous_tile_system=jnp.zeros(max_tiles, dtype=jnp.int32),
            previous_neighbor_matrix=jnp.full(
                (n_atoms, max_neighbors), n_atoms, dtype=jnp.int32
            ),
            previous_num_neighbors=jnp.zeros(n_atoms, dtype=jnp.int32),
            previous_neighbor_matrix_shifts=jnp.zeros(
                (n_atoms, max_neighbors, 3), dtype=jnp.int32
            ),
        )
        _nm, nn, _shifts, _tile_offsets, tile_counts, *_state = out
        assert int(nn.sum()) > 0
        assert int(tile_counts.sum()) > 0

    def test_allocate_batch_cluster_tile_list_zeros_and_runs(self):
        """``allocate_batch_cluster_tile_list`` yields zeroed, usable buffers.

        Regression for the segmented+batched query reading ``tile_system``
        before bounds-guarding: the allocator must zero ``tile_system`` (and
        siblings) so the selective path is safe, and its outputs must drive a
        correct rebuild matching the manual allocation.
        """
        positions, cell_batch, batch_ptr = _make_batch([32, 64], [6.0, 6.0], seed=33)
        cutoff = 2.0
        max_neighbors = 64
        n_atoms = int(batch_ptr[-1])

        (
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            tile_counts,
            tile_offsets,
        ) = allocate_batch_cluster_tile_list(batch_ptr, max_neighbors)

        # The allocator must hand back zeroed buffers (tile_system especially).
        for buf in (
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            tile_counts,
        ):
            assert int(jnp.count_nonzero(buf)) == 0

        out = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=max_neighbors,
            rebuild_flags=jnp.array([True, True], dtype=jnp.bool_),
            tile_offsets=tile_offsets,
            previous_tile_counts=tile_counts,
            previous_num_tiles=num_tiles,
            previous_tile_row_group=tile_row_group,
            previous_tile_col_group=tile_col_group,
            previous_tile_system=tile_system,
            previous_neighbor_matrix=jnp.full(
                (n_atoms, max_neighbors), n_atoms, dtype=jnp.int32
            ),
            previous_num_neighbors=jnp.zeros(n_atoms, dtype=jnp.int32),
            previous_neighbor_matrix_shifts=jnp.zeros(
                (n_atoms, max_neighbors, 3), dtype=jnp.int32
            ),
        )
        nm, nn, _shifts, _tile_offsets, tile_counts_out, *_state = out
        assert int(nn.sum()) > 0
        assert int(tile_counts_out.sum()) > 0

        # Equivalent to a full non-selective build over the same positions.
        ref_nm, ref_nn, *_ = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=max_neighbors,
        )
        np.testing.assert_array_equal(
            np.sort(np.asarray(nm), axis=1), np.sort(np.asarray(ref_nm), axis=1)
        )
        np.testing.assert_array_equal(np.asarray(nn), np.asarray(ref_nn))

    def test_rebuild_flags_coo_false_preserves_segmented_buffers(self):
        positions, cell_batch, batch_ptr = _make_batch([32, 64], [6.0, 6.0], seed=35)
        cutoff = 2.0
        max_neighbors = 64
        tile_state = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            format="tile",
        )
        num_tiles, tile_row_group, tile_col_group, tile_system, *_ = tile_state
        _tile_caps, tile_offsets, _pair_caps, pair_offsets = (
            estimate_batch_cluster_tile_segments(batch_ptr, max_neighbors=max_neighbors)
        )
        total_pairs = int(pair_offsets[-1])
        pair_counts = jnp.array([3, 5], dtype=jnp.int32)
        tile_counts = jnp.array([1, 2], dtype=jnp.int32)
        neighbor_list = jnp.arange(2 * total_pairs, dtype=jnp.int32).reshape(
            2, total_pairs
        )
        neighbor_shifts = jnp.arange(3 * total_pairs, dtype=jnp.int32).reshape(
            total_pairs, 3
        )

        out = batch_cluster_tile_neighbor_list(
            positions.at[0, 0].add(0.25),
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=max_neighbors,
            format="coo",
            rebuild_flags=jnp.array([False, False], dtype=jnp.bool_),
            tile_offsets=tile_offsets,
            previous_tile_counts=tile_counts,
            previous_num_tiles=num_tiles,
            previous_tile_row_group=tile_row_group,
            previous_tile_col_group=tile_col_group,
            previous_tile_system=tile_system,
            pair_offsets=pair_offsets,
            previous_pair_counts=pair_counts,
            previous_neighbor_list=neighbor_list,
            previous_neighbor_list_shifts=neighbor_shifts,
        )
        (
            nl2,
            offsets2,
            counts2,
            shifts2,
            tile_offsets2,
            tile_counts2,
            nt2,
            row2,
            col2,
            system2,
        ) = out
        np.testing.assert_array_equal(np.asarray(nl2), np.asarray(neighbor_list))
        np.testing.assert_array_equal(np.asarray(offsets2), np.asarray(pair_offsets))
        np.testing.assert_array_equal(np.asarray(counts2), np.asarray(pair_counts))
        np.testing.assert_array_equal(np.asarray(shifts2), np.asarray(neighbor_shifts))
        np.testing.assert_array_equal(
            np.asarray(tile_offsets2), np.asarray(tile_offsets)
        )
        np.testing.assert_array_equal(np.asarray(tile_counts2), np.asarray(tile_counts))
        np.testing.assert_array_equal(np.asarray(nt2), np.asarray(num_tiles))
        np.testing.assert_array_equal(np.asarray(row2), np.asarray(tile_row_group))
        np.testing.assert_array_equal(np.asarray(col2), np.asarray(tile_col_group))
        np.testing.assert_array_equal(np.asarray(system2), np.asarray(tile_system))

    def test_rebuild_flags_coo_true_writes_segment_counts(self):
        positions, cell_batch, batch_ptr = _make_batch([32, 64], [6.0, 6.0], seed=36)
        cutoff = 2.0
        max_neighbors = 64
        tile_state = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            format="tile",
        )
        num_tiles, tile_row_group, tile_col_group, tile_system, *_ = tile_state
        _tile_caps, tile_offsets, _pair_caps, pair_offsets = (
            estimate_batch_cluster_tile_segments(batch_ptr, max_neighbors=max_neighbors)
        )
        total_pairs = int(pair_offsets[-1])

        out = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=max_neighbors,
            format="coo",
            rebuild_flags=jnp.array([True, True], dtype=jnp.bool_),
            tile_offsets=tile_offsets,
            previous_tile_counts=jnp.zeros(2, dtype=jnp.int32),
            previous_num_tiles=jnp.zeros_like(num_tiles),
            previous_tile_row_group=tile_row_group,
            previous_tile_col_group=tile_col_group,
            previous_tile_system=tile_system,
            pair_offsets=pair_offsets,
            previous_pair_counts=jnp.zeros(2, dtype=jnp.int32),
            previous_neighbor_list=jnp.zeros((2, total_pairs), dtype=jnp.int32),
            previous_neighbor_list_shifts=jnp.zeros((total_pairs, 3), dtype=jnp.int32),
        )
        (
            _nl,
            _offsets,
            pair_counts,
            _shifts,
            _tile_offsets,
            tile_counts,
            _num_tiles,
            *_state,
        ) = out
        assert int(pair_counts.sum()) > 0
        assert bool(jnp.all(pair_counts <= (pair_offsets[1:] - pair_offsets[:-1])))
        assert int(tile_counts.sum()) > 0

    def test_rebuild_flags_require_previous_state(self):
        positions, cell_batch, batch_ptr = _make_batch([32], [6.0], seed=34)
        with pytest.raises(ValueError, match="previous batch_cluster_tile state"):
            batch_cluster_tile_neighbor_list(
                positions,
                1.0,
                cell_batch,
                batch_ptr,
                max_neighbors=16,
                rebuild_flags=jnp.array([True], dtype=jnp.bool_),
            )


class TestJaxBatchClusterTileEmptySelective:
    """Selective zero-atom returns keep their fixed tuple layout."""

    @staticmethod
    def _tile_state():
        """Return distinctive batched tile state buffers."""
        return {
            "tile_offsets": jnp.array([0, 4], dtype=jnp.int32),
            "previous_tile_counts": jnp.array([3], dtype=jnp.int32),
            "previous_num_tiles": jnp.array([3], dtype=jnp.int32),
            "previous_tile_row_group": jnp.arange(4, dtype=jnp.int32),
            "previous_tile_col_group": jnp.arange(4, dtype=jnp.int32),
            "previous_tile_system": jnp.zeros(4, dtype=jnp.int32),
        }

    @pytest.mark.parametrize("rebuild_flag", [False, True])
    def test_empty_selective_coo_returns_segmented_state(self, rebuild_flag):
        """Zero atoms retain the ten-array batched COO contract."""
        batch_ptr = jnp.array([0, 0], dtype=jnp.int32)
        pair_offsets = jnp.array([0, 5], dtype=jnp.int32)
        previous_pair_counts = jnp.array([4], dtype=jnp.int32)
        previous_neighbor_list = jnp.full((2, 5), 7, dtype=jnp.int32)
        previous_neighbor_list_shifts = jnp.full((5, 3), 7, dtype=jnp.int32)
        tile_state = self._tile_state()

        out = batch_cluster_tile_neighbor_list(
            jnp.empty((0, 3), dtype=jnp.float32),
            1.0,
            jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, :, :],
            batch_ptr,
            max_neighbors=8,
            format="coo",
            rebuild_flags=jnp.array([rebuild_flag], dtype=jnp.bool_),
            pair_offsets=pair_offsets,
            previous_pair_counts=previous_pair_counts,
            previous_neighbor_list=previous_neighbor_list,
            previous_neighbor_list_shifts=previous_neighbor_list_shifts,
            **tile_state,
        )

        (
            neighbor_list,
            offsets,
            pair_counts,
            shifts,
            returned_tile_offsets,
            tile_counts,
            num_tiles,
            row,
            col,
            system,
        ) = out
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
            np.asarray(returned_tile_offsets), np.asarray(tile_state["tile_offsets"])
        )
        np.testing.assert_array_equal(
            np.asarray(tile_counts),
            np.array([0 if rebuild_flag else 3], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(num_tiles), np.asarray(tile_state["previous_num_tiles"])
        )
        np.testing.assert_array_equal(
            np.asarray(row), np.asarray(tile_state["previous_tile_row_group"])
        )
        np.testing.assert_array_equal(
            np.asarray(col), np.asarray(tile_state["previous_tile_col_group"])
        )
        np.testing.assert_array_equal(
            np.asarray(system), np.asarray(tile_state["previous_tile_system"])
        )

    @pytest.mark.parametrize("rebuild_flag", [False, True])
    def test_empty_selective_matrix_returns_state(self, rebuild_flag):
        """Zero atoms retain the nine-array selective matrix contract."""
        tile_state = self._tile_state()
        out = batch_cluster_tile_neighbor_list(
            jnp.empty((0, 3), dtype=jnp.float32),
            1.0,
            jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, :, :],
            jnp.array([0, 0], dtype=jnp.int32),
            max_neighbors=8,
            rebuild_flags=jnp.array([rebuild_flag], dtype=jnp.bool_),
            previous_neighbor_matrix=jnp.empty((0, 8), dtype=jnp.int32),
            previous_num_neighbors=jnp.empty(0, dtype=jnp.int32),
            previous_neighbor_matrix_shifts=jnp.empty((0, 8, 3), dtype=jnp.int32),
            **tile_state,
        )

        assert len(out) == 9
        assert tuple(value.shape for value in out[:3]) == (
            (0, 8),
            (0,),
            (0, 8, 3),
        )
        for returned, name in zip(out[3:], tile_state):
            expected = (
                np.zeros(1, dtype=np.int32)
                if rebuild_flag and name == "previous_tile_counts"
                else np.asarray(tile_state[name])
            )
            np.testing.assert_array_equal(
                np.asarray(returned),
                expected,
            )

    def test_empty_two_system_selective_matrix_returns_state(self):
        """Two empty systems retain per-system offsets, counts, and flags."""
        tile_state = {
            "tile_offsets": jnp.array([0, 2, 4], dtype=jnp.int32),
            "previous_tile_counts": jnp.array([1, 2], dtype=jnp.int32),
            "previous_num_tiles": jnp.array([3], dtype=jnp.int32),
            "previous_tile_row_group": jnp.arange(4, dtype=jnp.int32),
            "previous_tile_col_group": jnp.arange(4, dtype=jnp.int32),
            "previous_tile_system": jnp.array([0, 0, 1, 1], dtype=jnp.int32),
        }
        out = batch_cluster_tile_neighbor_list(
            jnp.empty((0, 3), dtype=jnp.float32),
            1.0,
            jnp.repeat(jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, :, :], 2, axis=0),
            jnp.array([0, 0, 0], dtype=jnp.int32),
            max_neighbors=8,
            rebuild_flags=jnp.array([False, True], dtype=jnp.bool_),
            previous_neighbor_matrix=jnp.empty((0, 8), dtype=jnp.int32),
            previous_num_neighbors=jnp.empty(0, dtype=jnp.int32),
            previous_neighbor_matrix_shifts=jnp.empty((0, 8, 3), dtype=jnp.int32),
            **tile_state,
        )

        assert len(out) == 9
        expected_state = (
            tile_state["tile_offsets"],
            jnp.array([1, 0], dtype=jnp.int32),
            tile_state["previous_num_tiles"],
            tile_state["previous_tile_row_group"],
            tile_state["previous_tile_col_group"],
            tile_state["previous_tile_system"],
        )
        for returned, expected in zip(out[3:], expected_state):
            np.testing.assert_array_equal(
                np.asarray(returned),
                np.asarray(expected),
            )

    def test_empty_selective_dual_cutoff_matrix_returns_both_outputs(self):
        """Zero atoms retain both matrix triples plus batched selective state."""
        tile_state = self._tile_state()
        matrix = jnp.empty((0, 8), dtype=jnp.int32)
        counts = jnp.empty(0, dtype=jnp.int32)
        shifts = jnp.empty((0, 8, 3), dtype=jnp.int32)

        out = batch_cluster_tile_neighbor_list(
            jnp.empty((0, 3), dtype=jnp.float32),
            1.0,
            jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, :, :],
            jnp.array([0, 0], dtype=jnp.int32),
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

        assert len(out) == 12
        assert tuple(value.shape for value in out[:6]) == (
            (0, 8),
            (0,),
            (0, 8, 3),
            (0, 8),
            (0,),
            (0, 8, 3),
        )
        np.testing.assert_array_equal(
            np.asarray(out[6]), np.asarray(tile_state["tile_offsets"])
        )

    @pytest.mark.parametrize("rebuild_flag, expected_count", [(False, 4), (True, 0)])
    def test_jit_empty_selective_coo_keeps_fixed_arity(
        self,
        rebuild_flag,
        expected_count,
    ):
        """JIT preserves the ten-array zero-atom selective COO contract."""
        tile_state = self._tile_state()

        @jax.jit
        def build(positions, rebuild_flags):
            return batch_cluster_tile_neighbor_list(
                positions,
                1.0,
                jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, :, :],
                jnp.array([0, 0], dtype=jnp.int32),
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

        assert len(out) == 10
        assert out[0].shape == (2, 5)
        assert out[2].shape == (1,)
        assert int(out[2][0]) == expected_count
