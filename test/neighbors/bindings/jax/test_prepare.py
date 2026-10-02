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


"""Tests for the JAX neighbor-list preallocation helper."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors import (
    neighbor_list,
    prepare_neighbor_list_method,
)

from .conftest import requires_gpu

pytestmark = requires_gpu

# Every naive and cell-list strategy, coarse and fine-grained.
SINGLE_METHODS = (
    "naive",
    "naive_scalar",
    "naive_tile",
    "cell_list",
    "cell_list_atom_centric",
    "cell_list_pair_centric",
)
BATCH_METHODS = tuple(f"batch_{name}" for name in SINGLE_METHODS)
# cluster_tile is guarded: CUDA, float32 and fully periodic only, so it is
# listed apart from the strategies those guards do not constrain.
CLUSTER_TILE_METHODS = ("cluster_tile",)
BATCH_CLUSTER_TILE_METHODS = ("batch_cluster_tile",)
DUAL_METHODS = ("naive_dual_cutoff",)
BATCH_DUAL_METHODS = ("batch_naive_dual_cutoff",)

_FREE_PBC = jnp.zeros((3,), dtype=bool)


def periodic_system(num_atoms=60, cell_size=6.0, dtype=jnp.float64, seed=0):
    """A periodic box with enough neighbours to exercise capacity handling."""
    key = jax.random.PRNGKey(seed)
    positions = jax.random.uniform(key, (num_atoms, 3), dtype=dtype) * cell_size
    cell = (jnp.eye(3, dtype=dtype) * cell_size).reshape(1, 3, 3)
    pbc = jnp.ones((1, 3), dtype=bool)
    return positions, cell, pbc


def batch_system(per_system=40, num_systems=3, cell_size=6.0, dtype=jnp.float64):
    """Equal-sized systems laid out contiguously, with CSR metadata."""
    total = per_system * num_systems
    key = jax.random.PRNGKey(1)
    positions = jax.random.uniform(key, (total, 3), dtype=dtype) * cell_size
    cell = jnp.broadcast_to(
        (jnp.eye(3, dtype=dtype) * cell_size).reshape(1, 3, 3), (num_systems, 3, 3)
    )
    pbc = jnp.ones((num_systems, 3), dtype=bool)
    batch_ptr = jnp.arange(num_systems + 1, dtype=jnp.int32) * per_system
    batch_idx = jnp.repeat(jnp.arange(num_systems, dtype=jnp.int32), per_system)
    return positions, cell, pbc, batch_idx, batch_ptr


def _free_cell(size=60.0, dtype=jnp.float64):
    """A box large enough that no periodic image is within any test cutoff."""
    return jnp.eye(3, dtype=dtype) * size


def reference_pairs(positions, cutoff):
    """Brute-force pair count, free boundary, to anchor the prepared path."""
    pos = np.asarray(positions)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return int((d < cutoff).sum())


def pair_total(result):
    """Total neighbours written, from a matrix-format result."""
    return int(jnp.sum(result[1]))


class TestPrepareNeighborListMethod:
    """Behaviour of the JAX preallocation helper."""

    def test_matches_brute_force_free_boundary(self):
        """The prepared path reproduces a brute-force count."""
        positions, _, _ = periodic_system(num_atoms=80, cell_size=20.0)
        kwargs = prepare_neighbor_list_method(positions, 4.0)
        result = neighbor_list(positions, 4.0, **kwargs)
        assert pair_total(result) == reference_pairs(positions, 4.0)

    @pytest.mark.parametrize("method", SINGLE_METHODS)
    def test_every_single_system_method(self, method):
        """Every naive and cell-list strategy agrees on the pair count."""
        positions, cell, pbc = periodic_system()
        baseline = neighbor_list(positions, 2.5, cell=cell, pbc=pbc, method="naive")
        kwargs = prepare_neighbor_list_method(
            positions, 2.5, cell=cell, pbc=pbc, method=method
        )
        assert kwargs["method"] == method
        assert pair_total(neighbor_list(positions, 2.5, **kwargs)) == pair_total(
            baseline
        )

    @pytest.mark.parametrize("method", BATCH_METHODS)
    def test_every_batched_method(self, method):
        """Every batched strategy agrees with the batched naive baseline."""
        positions, cell, pbc, batch_idx, batch_ptr = batch_system()
        baseline = neighbor_list(
            positions,
            2.5,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_naive",
        )
        kwargs = prepare_neighbor_list_method(
            positions,
            2.5,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method=method,
        )
        assert pair_total(neighbor_list(positions, 2.5, **kwargs)) == pair_total(
            baseline
        )

    @pytest.mark.parametrize("method", DUAL_METHODS)
    def test_every_dual_cutoff_method(self, method):
        """Dual-cutoff strategies get two independently sized output sets."""
        positions, cell, pbc = periodic_system()
        kwargs = prepare_neighbor_list_method(
            positions, 2.0, cell=cell, pbc=pbc, cutoff2=3.0, method=method
        )
        assert "max_neighbors1" in kwargs and "max_neighbors2" in kwargs
        assert kwargs["max_neighbors2"] >= kwargs["max_neighbors1"]
        result = neighbor_list(positions, 2.0, **kwargs)
        assert result is not None

    @pytest.mark.parametrize("method", BATCH_DUAL_METHODS)
    def test_every_batched_dual_cutoff_method(self, method):
        """Batched dual-cutoff strategies prepare both output sets."""
        positions, cell, pbc, batch_idx, batch_ptr = batch_system()
        kwargs = prepare_neighbor_list_method(
            positions,
            2.0,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cutoff2=3.0,
            method=method,
        )
        assert "max_neighbors1" in kwargs and "max_neighbors2" in kwargs
        assert neighbor_list(positions, 2.0, **kwargs) is not None

    @pytest.mark.parametrize("method", CLUSTER_TILE_METHODS)
    def test_cluster_tile_single_system(self, method):
        """cluster_tile needs float32 and a fully periodic cell."""
        positions, cell, pbc = periodic_system(
            num_atoms=256, cell_size=8.0, dtype=jnp.float32
        )
        baseline = neighbor_list(positions, 2.5, cell=cell, pbc=pbc, method="naive")
        kwargs = prepare_neighbor_list_method(
            positions, 2.5, cell=cell, pbc=pbc, method=method
        )
        assert pair_total(neighbor_list(positions, 2.5, **kwargs)) == pair_total(
            baseline
        )

    @pytest.mark.parametrize("method", BATCH_CLUSTER_TILE_METHODS)
    def test_cluster_tile_batched(self, method):
        """Batched cluster_tile prepares from the same metadata."""
        positions, cell, pbc, batch_idx, batch_ptr = batch_system(
            per_system=128, num_systems=2, cell_size=8.0, dtype=jnp.float32
        )
        kwargs = prepare_neighbor_list_method(
            positions,
            2.5,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method=method,
        )
        assert neighbor_list(positions, 2.5, **kwargs) is not None

    def test_free_boundary_omits_cell_and_pbc(self):
        """A free-boundary call carries no cell or pbc into the result."""
        positions, _, _ = periodic_system(cell_size=20.0)
        kwargs = prepare_neighbor_list_method(positions, 3.0)
        assert "cell" not in kwargs
        assert "pbc" not in kwargs

    def test_all_false_pbc_is_dropped(self):
        """An all-False pbc must not select the periodic kernels."""
        positions, _, _ = periodic_system(cell_size=20.0)
        kwargs = prepare_neighbor_list_method(
            positions,
            3.0,
            cell=_free_cell().reshape(1, 3, 3),
            pbc=_FREE_PBC.reshape(1, 3),
        )
        assert "cell" not in kwargs
        assert "pbc" not in kwargs

    def test_mismatched_cell_and_pbc_raises(self):
        """cell and pbc must be given together."""
        positions, cell, _ = periodic_system()
        with pytest.raises(ValueError, match="together"):
            prepare_neighbor_list_method(positions, 2.5, cell=cell)

    def test_explicit_max_neighbors_is_honored(self):
        """A caller-supplied capacity is used verbatim."""
        positions, cell, pbc = periodic_system()
        kwargs = prepare_neighbor_list_method(
            positions, 2.5, cell=cell, pbc=pbc, max_neighbors=37
        )
        assert kwargs["max_neighbors"] == 37
        assert kwargs["neighbor_matrix"].shape[1] == 37

    def test_estimated_capacity_holds_for_this_geometry(self):
        """The estimate covers the busiest row it was sized for."""
        positions, cell, pbc = periodic_system()
        kwargs = prepare_neighbor_list_method(positions, 2.5, cell=cell, pbc=pbc)
        result = neighbor_list(positions, 2.5, **kwargs)
        assert int(jnp.max(result[1])) <= kwargs["max_neighbors"]

    def test_cell_list_state_is_allocated(self):
        """Cell-list strategies carry their grid scratch."""
        positions, cell, pbc = periodic_system()
        kwargs = prepare_neighbor_list_method(
            positions, 2.5, cell=cell, pbc=pbc, method="cell_list_atom_centric"
        )
        for key in ("cells_per_dimension", "atom_to_cell_mapping", "cell_atom_list"):
            assert key in kwargs
        # Query-sort scratch is single-system only, and JAX names the shift
        # buffer differently from Torch.
        assert "sorted_positions" in kwargs
        assert "sorted_atom_periodic_shifts" in kwargs

    def test_naive_periodic_state_is_allocated(self):
        """Periodic naive carries its shift range and wrap buffers."""
        positions, cell, pbc = periodic_system()
        kwargs = prepare_neighbor_list_method(
            positions, 2.5, cell=cell, pbc=pbc, method="naive_tile"
        )
        for key in (
            "shift_range_per_dimension",
            "num_shifts_per_system",
            "max_shifts_per_system",
            "positions_wrapped_buffer",
            "inv_cell_buffer",
        ):
            assert key in kwargs

    def test_free_boundary_naive_has_no_shift_buffers(self):
        """Aperiodic naive needs no image scratch at all."""
        positions, _, _ = periodic_system(cell_size=20.0)
        kwargs = prepare_neighbor_list_method(positions, 3.0, method="naive_tile")
        assert "shift_range_per_dimension" not in kwargs
        assert "neighbor_matrix_shifts" not in kwargs

    @pytest.mark.parametrize(
        ("method", "prefix"),
        [
            ("cell_list_atom_centric", ""),
            ("batch_cell_list", ""),
            ("cluster_tile", ""),
        ],
    )
    def test_output_buffers_are_emitted(self, method, prefix):
        """Every strategy gets real output buffers, under its own naming.

        ``batch_cell_list`` only grew its ``neighbor_matrix`` and
        ``num_neighbors`` parameters alongside this helper -- before that it
        allocated them internally and could not be preallocated at all -- and
        the cluster-tile entry points only dropped their ``previous_`` prefix
        at the same time.
        """
        positions, cell, pbc = periodic_system(
            num_atoms=256, cell_size=8.0, dtype=jnp.float32
        )
        extra = {}
        if method.startswith("batch_"):
            extra = {
                "batch_idx": jnp.zeros(256, dtype=jnp.int32),
                "batch_ptr": jnp.array([0, 256], dtype=jnp.int32),
            }
        kwargs = prepare_neighbor_list_method(
            positions, 2.5, cell=cell, pbc=pbc, method=method, **extra
        )
        for name in ("neighbor_matrix", "num_neighbors"):
            assert f"{prefix}{name}" in kwargs, f"{method} missing {prefix}{name}"
        assert neighbor_list(positions, 2.5, **kwargs) is not None

    def test_buffers_are_reused_across_calls(self):
        """The same mapping drives repeated calls without re-preparing."""
        positions, cell, pbc = periodic_system()
        kwargs = prepare_neighbor_list_method(positions, 2.5, cell=cell, pbc=pbc)
        first = pair_total(neighbor_list(positions, 2.5, **kwargs))
        second = pair_total(neighbor_list(positions, 2.5, **kwargs))
        assert first == second
