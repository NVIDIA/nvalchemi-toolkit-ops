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

"""Tests for JAX bindings of batched naive dual cutoff neighbor list methods."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nvalchemiops.jax.neighbors import batch_naive_neighbor_list_dual_cutoff

from .conftest import (
    create_batch_idx_and_ptr_jax,
    create_simple_cubic_system_jax,
    requires_gpu,
)

pytestmark = requires_gpu


class TestBatchedDualCutoffListFormat:
    """Test batched dual cutoff neighbor list in COO list format."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_batched_list_format_no_pbc(self, dtype):
        """Test batched dual cutoff in list format without PBC."""
        positions1, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=dtype
        )
        positions2, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.5, dtype=dtype
        )
        positions = jnp.concatenate([positions1, positions2], axis=0)

        atoms_per_system = [8, 8]
        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax(atoms_per_system)

        cutoff1 = 1.0
        cutoff2 = 1.5

        neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1,
                cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=15,
                max_neighbors2=25,
                return_neighbor_list=True,
            )
        )

        # Verify COO format shapes
        assert neighbor_list1.shape[0] == 2
        assert neighbor_list2.shape[0] == 2
        assert neighbor_ptr1.shape == (17,)  # 16 atoms + 1
        assert neighbor_ptr2.shape == (17,)
        # Larger cutoff should find at least as many pairs
        assert neighbor_list2.shape[1] >= neighbor_list1.shape[1]

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_batched_list_format_with_pbc(self, dtype):
        """Test batched dual cutoff in list format with PBC."""
        positions1, cell1, pbc1 = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=dtype
        )
        positions2, cell2, pbc2 = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.5, dtype=dtype
        )
        positions = jnp.concatenate([positions1, positions2], axis=0)
        cell = jnp.concatenate([cell1, cell2], axis=0)
        pbc = jnp.concatenate([pbc1, pbc2], axis=0)

        atoms_per_system = [8, 8]
        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax(atoms_per_system)

        cutoff1 = 1.0
        cutoff2 = 1.5

        (
            neighbor_list1,
            neighbor_ptr1,
            unit_shifts1,
            neighbor_list2,
            neighbor_ptr2,
            unit_shifts2,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cell=cell,
            pbc=pbc,
            max_neighbors1=15,
            max_neighbors2=25,
            return_neighbor_list=True,
        )

        # Verify COO format shapes
        assert neighbor_list1.shape[0] == 2
        assert neighbor_list2.shape[0] == 2
        assert neighbor_ptr1.shape == (17,)
        assert neighbor_ptr2.shape == (17,)
        assert unit_shifts1.shape[0] == neighbor_list1.shape[1]
        assert unit_shifts2.shape[0] == neighbor_list2.shape[1]


class TestBatchNaiveDualCutoffJIT:
    """Smoke tests for batch_naive_neighbor_list_dual_cutoff with jax.jit."""

    def test_jit_no_pbc(self):
        """Test batched dual cutoff without PBC works with jax.jit."""
        positions1, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=jnp.float32
        )
        positions2, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.5, dtype=jnp.float32
        )
        positions = jnp.concatenate([positions1, positions2], axis=0)
        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax([8, 8])

        @jax.jit
        def jitted_batch_dual(positions, batch_idx, batch_ptr):
            return batch_naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1=1.0,
                cutoff2=1.5,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=15,
                max_neighbors2=25,
            )

        nm1, nn1, nm2, nn2 = jitted_batch_dual(positions, batch_idx, batch_ptr)

        assert nm1.shape == (16, 15)
        assert nm2.shape == (16, 25)
        assert nn1.shape == (16,)
        assert nn2.shape == (16,)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
class TestBatchNaiveDualCutoffSelectiveRebuildFlags:
    """Test selective rebuild (rebuild_flags) for batch_naive_neighbor_list_dual_cutoff JAX."""

    def test_no_rebuild_preserves_data(self, dtype):
        """All flags False: neighbor data should remain unchanged for all systems."""
        positions1, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=dtype
        )
        positions2, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.5, dtype=dtype
        )
        positions = jnp.concatenate([positions1, positions2], axis=0)
        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax([8, 8])

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        # Initial full build
        nm1, nn1, nm2, nn2 = batch_naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
        )

        saved_nn1 = jnp.array(nn1)
        saved_nn2 = jnp.array(nn2)

        # Selective rebuild with all flags=False
        rebuild_flags = jnp.zeros(2, dtype=jnp.bool_)
        nm1b, nn1b, nm2b, nn2b = batch_naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            neighbor_matrix1=nm1,
            neighbor_matrix2=nm2,
            num_neighbors1=nn1,
            num_neighbors2=nn2,
            rebuild_flags=rebuild_flags,
        )

        assert jnp.all(nn1b == saved_nn1), "nn1 must be unchanged when flags are False"
        assert jnp.all(nn2b == saved_nn2), "nn2 must be unchanged when flags are False"

    def test_rebuild_updates_data(self, dtype):
        """True flags: rebuilt system data should match a fresh full rebuild."""
        positions1, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=dtype
        )
        positions2, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.5, dtype=dtype
        )
        positions = jnp.concatenate([positions1, positions2], axis=0)
        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax([8, 8])

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        # Reference: full build
        _, nn1_ref, _, nn2_ref = batch_naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
        )

        # Selective rebuild with all flags=True
        nm1_stale = jnp.full((16, max_neighbors1), 99, dtype=jnp.int32)
        nm2_stale = jnp.full((16, max_neighbors2), 99, dtype=jnp.int32)
        nn1_stale = jnp.full((16,), 99, dtype=jnp.int32)
        nn2_stale = jnp.full((16,), 99, dtype=jnp.int32)

        rebuild_flags = jnp.ones(2, dtype=jnp.bool_)
        _, nn1b, _, nn2b = batch_naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            neighbor_matrix1=nm1_stale,
            neighbor_matrix2=nm2_stale,
            num_neighbors1=nn1_stale,
            num_neighbors2=nn2_stale,
            rebuild_flags=rebuild_flags,
        )

        assert jnp.all(nn1b == nn1_ref), (
            "nn1 should match full rebuild when all flags=True"
        )
        assert jnp.all(nn2b == nn2_ref), (
            "nn2 should match full rebuild when all flags=True"
        )
