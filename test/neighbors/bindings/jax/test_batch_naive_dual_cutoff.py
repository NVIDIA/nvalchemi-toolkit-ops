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
