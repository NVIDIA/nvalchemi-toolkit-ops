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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors.batch_cluster_tile import (
    TILE_GROUP_SIZE,
    allocate_batch_cluster_tile_list,
    batch_cluster_tile_neighbor_list,
    estimate_batch_cluster_tile_list_sizes,
    estimate_batch_cluster_tile_segments,
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


class TestEstimateBatchSizes:
    """Pure-Python sizing helper tests."""

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

        def loss(p):
            *_, d, _ = batch_cluster_tile_neighbor_list(
                p,
                5.0,
                cell_batch,
                batch_ptr,
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
)


class TestJaxBatchClusterTileBruteForce:
    """Pair-set + shift identity checks for the batched binding.

    Mirrors the single-system :class:`TestJaxClusterTileBruteForce` —
    runs the batched cluster_tile build then compares each system's
    output against a per-system numpy brute-force reference.
    """

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
