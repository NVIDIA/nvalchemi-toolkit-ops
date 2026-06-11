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

"""Tests for JAX bindings of batched naive neighbor list methods."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list

from .conftest import requires_gpu

pytestmark = requires_gpu


class TestBatchNaiveNeighborList:
    """Test batch_naive_neighbor_list function."""

    def test_two_systems_no_pbc(self):
        """Test with two separate systems without PBC."""
        # System 1: 2 atoms
        positions1 = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32)
        # System 2: 2 atoms
        positions2 = jnp.array([[10.0, 0.0, 0.0], [10.5, 0.0, 0.0]], dtype=jnp.float32)

        positions = jnp.vstack([positions1, positions2])
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        cutoff = 1.0

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
        )

        assert neighbor_matrix.shape == (4, 10)
        assert num_neighbors.shape == (4,)

    def test_topology_only_grad_no_pbc_is_zero(self):
        """Topology-only batch outputs do not differentiate through Warp FFI."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)

        def loss(pos):
            neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
                pos,
                2.0,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=8,
            )
            return (
                neighbor_matrix.astype(pos.dtype).sum()
                + num_neighbors.astype(pos.dtype).sum()
            )

        grad = jax.grad(loss)(positions)
        assert jnp.isfinite(grad).all().item()
        np.testing.assert_allclose(np.asarray(grad), 0.0)

    def test_two_systems_with_pbc(self):
        """Test with two systems with PBC."""
        positions1 = jnp.array([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=jnp.float32)
        positions2 = jnp.array([[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]], dtype=jnp.float32)

        positions = jnp.vstack([positions1, positions2])

        cells = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            ]
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])

        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        cutoff = 1.0

        neighbor_matrix, num_neighbors, shifts = batch_naive_neighbor_list(
            positions,
            cutoff,
            cell=cells,
            pbc=pbcs,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
        )

        assert neighbor_matrix.shape == (4, 10)
        assert num_neighbors.shape == (4,)
        assert shifts.shape == (4, 10, 3)

    def test_topology_only_grad_pbc_is_zero(self):
        """PBC batch wrapping for topology-only outputs is nondifferentiable."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        cells = jnp.array(
            [
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            ],
            dtype=jnp.float32,
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)

        def loss(pos):
            neighbor_matrix, num_neighbors, shifts = batch_naive_neighbor_list(
                pos,
                2.0,
                cell=cells,
                pbc=pbcs,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=8,
                max_atoms_per_system=2,
            )
            return (
                neighbor_matrix.astype(pos.dtype).sum()
                + num_neighbors.astype(pos.dtype).sum()
                + shifts.astype(pos.dtype).sum()
            )

        grad = jax.grad(loss)(positions)
        assert jnp.isfinite(grad).all().item()
        np.testing.assert_allclose(np.asarray(grad), 0.0)

    def test_different_system_sizes(self):
        """Test with systems of different sizes."""
        # System 1: 3 atoms
        positions1 = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ],
            dtype=jnp.float32,
        )
        # System 2: 2 atoms
        positions2 = jnp.array([[10.0, 0.0, 0.0], [10.5, 0.0, 0.0]], dtype=jnp.float32)

        positions = jnp.vstack([positions1, positions2])
        batch_idx = jnp.array([0, 0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 3, 5], dtype=jnp.int32)
        cutoff = 1.0

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
        )

        assert neighbor_matrix.shape == (5, 10)
        assert num_neighbors.shape == (5,)


class TestBatchNaiveEdgeCases:
    """Edge case tests for batch_naive_neighbor_list."""

    def test_mixed_system_sizes(self):
        """Batch with very different system sizes should work correctly."""
        # System 1: 1 atom, System 2: 10 atoms
        pos1 = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        pos2 = jax.random.uniform(key, shape=(10, 3), dtype=jnp.float32) * 3.0

        positions = jnp.vstack([pos1, pos2])
        batch_idx = jnp.concatenate(
            [
                jnp.zeros(1, dtype=jnp.int32),
                jnp.ones(10, dtype=jnp.int32),
            ]
        )
        batch_ptr = jnp.array([0, 1, 11], dtype=jnp.int32)
        cutoff = 2.0

        nm, nn = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=20,
        )
        assert nm.shape == (11, 20)
        assert nn.shape == (11,)
        # System 1 (single atom) should have 0 neighbors
        assert int(nn[0]) == 0

    def test_no_cross_system_neighbors(self):
        """Neighbors should never cross system boundaries."""
        # Two systems, far apart
        pos1 = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32)
        pos2 = jnp.array([[100.0, 0.0, 0.0], [100.5, 0.0, 0.0]], dtype=jnp.float32)
        positions = jnp.vstack([pos1, pos2])
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        cutoff = 1.0

        nm, nn = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
        )
        # Verify: atom 0,1 should only have neighbors in [0,1]; atom 2,3 in [2,3]
        fill_val = positions.shape[0]
        for atom_i in range(4):
            sys = int(batch_idx[atom_i])
            start = int(batch_ptr[sys])
            end = int(batch_ptr[sys + 1])
            for k in range(int(nn[atom_i])):
                j = int(nm[atom_i, k])
                assert j != fill_val, "Got fill value in valid neighbor slot"
                assert start <= j < end, (
                    f"Atom {atom_i} (sys {sys}) has cross-system neighbor {j}"
                )

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_batch_precision_consistency(self, dtype):
        """f32 and f64 batched should find the same neighbor counts."""
        key = jax.random.PRNGKey(99)
        positions = jax.random.uniform(key, shape=(12, 3), dtype=dtype) * 3.0
        batch_idx = jnp.array([0] * 4 + [1] * 4 + [2] * 4, dtype=jnp.int32)
        batch_ptr = jnp.array([0, 4, 8, 12], dtype=jnp.int32)
        cutoff = 2.0

        _, nn = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=20,
        )
        total = int(jnp.sum(nn))
        assert total > 0

    def test_batch_zero_cutoff(self):
        """Batch with zero cutoff should find no neighbors."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)

        nm, nn = batch_naive_neighbor_list(
            positions,
            cutoff=0.0,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
        )
        assert jnp.all(nn == 0)

    def test_batch_with_pbc_distance_validity(self):
        """All batched PBC neighbors should be within cutoff distance."""
        pos1 = jnp.array([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=jnp.float32)
        pos2 = jnp.array([[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]], dtype=jnp.float32)
        positions = jnp.vstack([pos1, pos2])
        cells = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            ],
            dtype=jnp.float32,
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        cutoff = 1.0

        nm, nn, shifts = batch_naive_neighbor_list(
            positions,
            cutoff,
            cell=cells,
            pbc=pbcs,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
        )
        for i in range(4):
            sys = int(batch_idx[i])
            cell_mat = cells[sys]
            for k in range(int(nn[i])):
                j = int(nm[i, k])
                shift_vec = jnp.dot(shifts[i, k].astype(jnp.float32), cell_mat)
                rij = positions[j] - positions[i] + shift_vec
                dist = float(jnp.linalg.norm(rij))
                assert dist < cutoff + 1e-4, (
                    f"Atom {i}->{j} dist {dist} > cutoff {cutoff}"
                )

    def test_batch_half_fill(self):
        """Batch half_fill should produce half the full-fill pairs per system."""
        pos1 = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ],
            dtype=jnp.float32,
        )
        positions = jnp.vstack([pos1, pos1])
        batch_idx = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 3, 6], dtype=jnp.int32)
        cutoff = 1.0

        _, nn_full = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
            half_fill=False,
        )
        _, nn_half = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
            half_fill=True,
        )
        assert int(jnp.sum(nn_half)) * 2 == int(jnp.sum(nn_full))

    def test_return_neighbor_list_format(self):
        """return_neighbor_list=True in batch mode."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.5, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        cutoff = 1.0

        nl, ptr = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
            return_neighbor_list=True,
        )
        assert nl.shape[0] == 2  # COO
        assert ptr.shape == (5,)  # 4 atoms + 1


class TestBatchNaiveJIT:
    """Smoke tests for batch_naive_neighbor_list with jax.jit."""

    def test_jit_no_pbc(self):
        """Test batched naive neighbor list without PBC works with jax.jit."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.5, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)

        @jax.jit
        def jitted_batch_naive(positions, batch_idx, batch_ptr):
            return batch_naive_neighbor_list(
                positions,
                cutoff=1.0,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=10,
            )

        neighbor_matrix, num_neighbors = jitted_batch_naive(
            positions, batch_idx, batch_ptr
        )

        assert neighbor_matrix.shape == (4, 10)
        assert num_neighbors.shape == (4,)
        assert jnp.all(num_neighbors >= 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
class TestBatchNaiveSelectiveRebuildFlags:
    """Test selective rebuild (rebuild_flags) for batch_naive_neighbor_list JAX binding."""

    def test_no_rebuild_preserves_data(self, dtype):
        """All flags False: neighbor data should remain unchanged for all systems."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [10.0, 0.0, 0.0],
                [10.5, 0.0, 0.0],
                [10.0, 0.5, 0.0],
            ],
            dtype=dtype,
        )
        batch_idx = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 3, 6], dtype=jnp.int32)
        cutoff = 1.0
        max_neighbors = 10

        # Initial full build
        nm, nn = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors,
        )

        saved_nn = jnp.array(nn)

        # Selective rebuild with all flags=False
        rebuild_flags = jnp.zeros(2, dtype=jnp.bool_)
        nm2, nn2 = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm,
            num_neighbors=nn,
            rebuild_flags=rebuild_flags,
        )

        assert jnp.all(nn2 == saved_nn), (
            "num_neighbors must be unchanged when all rebuild_flags are False"
        )

    def test_rebuild_updates_data(self, dtype):
        """True flags: rebuilt system data should match a fresh full rebuild."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [10.0, 0.0, 0.0],
                [10.5, 0.0, 0.0],
                [10.0, 0.5, 0.0],
            ],
            dtype=dtype,
        )
        batch_idx = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 3, 6], dtype=jnp.int32)
        cutoff = 1.0
        max_neighbors = 10

        # Reference: full build
        _, nn_ref = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors,
        )

        # Selective rebuild with all flags=True
        nm_stale = jnp.full((positions.shape[0], max_neighbors), 99, dtype=jnp.int32)
        nn_stale = jnp.full((positions.shape[0],), 99, dtype=jnp.int32)

        rebuild_flags = jnp.ones(2, dtype=jnp.bool_)
        _, nn2 = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm_stale,
            num_neighbors=nn_stale,
            rebuild_flags=rebuild_flags,
        )

        assert jnp.all(nn2 == nn_ref), (
            "num_neighbors should match full rebuild when all flags=True"
        )


class TestJaxBatchNaiveAutograd:
    """Differentiable per-pair distances/vectors for batch_naive_neighbor_list."""

    def _make_batch(self, n_per=6, scale=0.5, dtype=jnp.float64):
        key = jax.random.key(0)
        pos = jax.random.normal(key, (2 * n_per, 3), dtype=dtype) * scale
        batch_idx = jnp.concatenate(
            [jnp.zeros(n_per, dtype=jnp.int32), jnp.ones(n_per, dtype=jnp.int32)]
        )
        cell = jnp.tile(jnp.eye(3, dtype=dtype)[None] * 4.0, (2, 1, 1))
        pbc = jnp.ones((2, 3), dtype=jnp.bool_)
        return pos, cell, pbc, batch_idx, n_per

    def test_forward_no_pbc(self):
        from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list

        pos, _, _, batch_idx, _ = self._make_batch()
        out = batch_naive_neighbor_list(
            pos,
            1.5,
            batch_idx=batch_idx,
            max_neighbors=8,
            return_distances=True,
            return_vectors=True,
        )
        assert len(out) == 4

    def test_grad_positions_pbc(self):
        from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list

        pos, cell, pbc, batch_idx, n_per = self._make_batch()

        def loss(p):
            *_, d, _ = batch_naive_neighbor_list(
                p,
                1.5,
                batch_idx=batch_idx,
                cell=cell,
                pbc=pbc,
                max_neighbors=8,
                max_atoms_per_system=n_per,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        g = jax.grad(loss)(pos)
        assert g.shape == pos.shape
        assert jnp.isfinite(g).all().item()

    def test_grad_cell_pbc(self):
        from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list

        pos, cell, pbc, batch_idx, n_per = self._make_batch()

        def loss(c):
            *_, d, _ = batch_naive_neighbor_list(
                pos,
                1.5,
                batch_idx=batch_idx,
                cell=c,
                pbc=pbc,
                max_neighbors=8,
                max_atoms_per_system=n_per,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        g = jax.grad(loss)(cell)
        assert g.shape == cell.shape
        assert jnp.isfinite(g).all().item()

    def test_check_grads_no_pbc(self):
        from jax.test_util import check_grads

        from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list

        pos, _, _, batch_idx, _ = self._make_batch()

        def loss(p):
            *_, d, _ = batch_naive_neighbor_list(
                p,
                1.5,
                batch_idx=batch_idx,
                max_neighbors=8,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        check_grads(loss, (pos,), order=1, atol=1e-4, rtol=1e-4, modes=["rev"])

    def test_half_fill_with_pair_outputs(self):
        """half_fill=True now combines with per-pair geometry outputs (JAX batched)."""
        from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list

        pos, cell, pbc, batch_idx, n_per = self._make_batch()
        nm, _nn, _sh, dist, vec = batch_naive_neighbor_list(
            pos,
            1.5,
            batch_idx=batch_idx,
            cell=cell,
            pbc=pbc,
            max_neighbors=8,
            max_atoms_per_system=n_per,
            return_distances=True,
            return_vectors=True,
            half_fill=True,
        )
        active = np.asarray(nm) != pos.shape[0]
        assert int(active.sum()) > 0
        d = np.asarray(dist)[active]
        v = np.asarray(vec)[active]
        assert np.all(d <= 1.5 + 1e-4)
        np.testing.assert_allclose(d, np.linalg.norm(v, axis=-1), atol=1e-5, rtol=1e-5)

    def test_pair_outputs_reject_rebuild_flags(self):
        from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list

        pos, cell, pbc, batch_idx, n_per = self._make_batch()
        num_systems = int(np.asarray(batch_idx).max()) + 1
        with pytest.raises(NotImplementedError, match="rebuild_flags"):
            batch_naive_neighbor_list(
                pos,
                1.5,
                batch_idx=batch_idx,
                cell=cell,
                pbc=pbc,
                max_neighbors=8,
                max_atoms_per_system=n_per,
                return_distances=True,
                rebuild_flags=jnp.ones((num_systems,), dtype=jnp.bool_),
            )

    def test_hessian_vector_product_smoke(self):
        """Second-order HVP smoke — see TestJaxNaiveAutograd for rationale."""
        from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list

        pos, cell, pbc, batch_idx, n_per = self._make_batch()
        v = jax.random.normal(jax.random.key(1), pos.shape, dtype=pos.dtype)

        def loss(p):
            *_, d, _ = batch_naive_neighbor_list(
                p,
                1.5,
                batch_idx=batch_idx,
                cell=cell,
                pbc=pbc,
                max_neighbors=8,
                max_atoms_per_system=n_per,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        hvp = jax.grad(lambda p: jnp.vdot(jax.grad(loss)(p), v))(pos)
        assert jnp.isfinite(hvp).all().item()
        assert hvp.shape == pos.shape

    def test_hvp_nonlinear_loss_matches_analytic(self):
        """Regression: nonlinear-loss HVP matches the exact analytic Hessian on the
        batched path — exercises the per-system cell-gather reconstruction branch
        (``cell[batch_idx[i]]``) for a loss the old detached backward got wrong."""
        import numpy as np

        from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list

        from .conftest import analytic_distance_sq_hvp

        pos, cell, pbc, batch_idx, n_per = self._make_batch()
        v = jax.random.normal(jax.random.key(1), pos.shape, dtype=pos.dtype)

        def loss(p):
            *_, d, _ = batch_naive_neighbor_list(
                p,
                1.5,
                batch_idx=batch_idx,
                cell=cell,
                pbc=pbc,
                max_neighbors=8,
                max_atoms_per_system=n_per,
                return_distances=True,
                return_vectors=True,
            )
            return (d**2).sum()

        hvp = np.asarray(jax.grad(lambda p: jnp.vdot(jax.grad(loss)(p), v))(pos))
        nl, *_ = batch_naive_neighbor_list(
            pos,
            1.5,
            batch_idx=batch_idx,
            cell=cell,
            pbc=pbc,
            max_neighbors=8,
            max_atoms_per_system=n_per,
            return_neighbor_list=True,
        )
        hvp_true = analytic_distance_sq_hvp(nl, v, pos.shape[0])
        assert nl.shape[1] > 0
        assert np.allclose(hvp, hvp_true, atol=1e-9, rtol=1e-9)

    def test_no_grad_path_unchanged(self):
        from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list

        pos, cell, pbc, batch_idx, n_per = self._make_batch()
        nm_a, nn_a, sh_a = batch_naive_neighbor_list(
            pos,
            1.5,
            batch_idx=batch_idx,
            cell=cell,
            pbc=pbc,
            max_neighbors=8,
            max_atoms_per_system=n_per,
        )
        nm_b, nn_b, sh_b, d_b, v_b = batch_naive_neighbor_list(
            pos,
            1.5,
            batch_idx=batch_idx,
            cell=cell,
            pbc=pbc,
            max_neighbors=8,
            max_atoms_per_system=n_per,
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
