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

"""API tests for the generic JAX neighbor_list wrapper function."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors import (
    batch_naive_neighbor_list_dual_cutoff,
    cell_list,
    naive_neighbor_list,
    naive_neighbor_list_dual_cutoff,
    neighbor_list,
)

from .conftest import create_batch_idx_and_ptr_jax, requires_gpu

pytestmark = requires_gpu

# ==============================================================================
# Helpers
# ==============================================================================


def create_random_system_jax(
    num_atoms: int,
    cell_size: float,
    dtype=jnp.float32,
    seed: int = 42,
):
    """Create a random system with JAX arrays (positions inside a box)."""
    key = jax.random.PRNGKey(seed)
    positions = jax.random.uniform(key, (num_atoms, 3), dtype=dtype) * cell_size
    cell = (jnp.eye(3, dtype=dtype) * cell_size).reshape(1, 3, 3)
    pbc = jnp.array([[True, True, True]])
    return (
        positions,
        cell,
        pbc,
    )


def assert_neighbor_matrix_equal_jax(result1, result2):
    """Assert that two JAX neighbor matrix results are equivalent.

    Compares num_neighbors exactly and neighbor_matrix rows after sorting.
    """
    if len(result1) == 2:
        nm1, nn1 = result1
        nm2, nn2 = result2
        shifts1 = shifts2 = None
    elif len(result1) == 3:
        nm1, nn1, shifts1 = result1
        nm2, nn2, shifts2 = result2
    else:
        raise ValueError(f"Unexpected result length: {len(result1)}")

    # num_neighbors must match exactly
    np.testing.assert_array_equal(np.asarray(nn1), np.asarray(nn2))

    # Compare rows of neighbor_matrix after sorting
    nm1_np = np.asarray(nm1)
    nm2_np = np.asarray(nm2)
    assert nm1_np.shape == nm2_np.shape, (
        f"Neighbor matrix shapes differ: {nm1_np.shape} vs {nm2_np.shape}"
    )
    for i in range(nm1_np.shape[0]):
        np.testing.assert_array_equal(np.sort(nm1_np[i]), np.sort(nm2_np[i]))

    if shifts1 is not None and shifts2 is not None:
        s1_np = np.asarray(shifts1)
        s2_np = np.asarray(shifts2)
        assert s1_np.shape == s2_np.shape, (
            f"Shifts shapes differ: {s1_np.shape} vs {s2_np.shape}"
        )


# ==============================================================================
# Tests: Auto-Selection
# ==============================================================================


class TestNeighborListAutoSelection:
    """Test automatic method selection based on system size."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_auto_select_naive_small_system(self, dtype, device):
        """Auto-select naive for small systems (< 5000 atoms), no PBC → 2-tuple."""
        target_density = 0.25
        num_atoms = 100
        volume = num_atoms / target_density
        box_size = volume ** (1 / 3)

        key = jax.random.PRNGKey(42)
        positions = jax.random.uniform(key, (num_atoms, 3), dtype=dtype) * box_size
        cutoff = 2.0

        result = neighbor_list(positions, cutoff, return_neighbor_list=True)

        # No PBC → 2-tuple (neighbor_list_coo, neighbor_ptr)
        assert len(result) == 2
        neighbor_list_coo, neighbor_ptr = result
        assert neighbor_list_coo.shape[0] == 2  # COO format
        assert neighbor_ptr.shape[0] == num_atoms + 1
        assert int(neighbor_ptr[0]) == 0

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_auto_select_naive_with_pbc(self, dtype, device):
        """Auto-select naive for small systems with PBC → 3-tuple."""
        positions, cell, pbc = create_random_system_jax(100, 10.0, dtype=dtype)
        cutoff = 2.0

        result = neighbor_list(
            positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=True
        )

        # With PBC → 3-tuple (neighbor_list_coo, neighbor_ptr, shifts)
        assert len(result) == 3
        neighbor_list_coo, neighbor_ptr, shifts = result
        assert neighbor_list_coo.shape[0] == 2
        assert neighbor_ptr.shape[0] == 101
        assert int(neighbor_ptr[0]) == 0
        assert shifts.shape[1] == 3

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_auto_select_cell_list_large_system(self, dtype, device):
        """Auto-select cell_list for large systems (>= 5000 atoms).

        When no cell/pbc is provided, the wrapper auto-creates identity cell
        with pbc=False.  We explicitly provide cell/pbc placed on the correct
        device to avoid device-mismatch issues with the auto-created arrays.
        """
        key = jax.random.PRNGKey(0)
        positions = jax.random.normal(key, (5000, 3), dtype=dtype) * 50.0

        # Provide cell/pbc
        cell = jnp.eye(3, dtype=dtype).reshape(1, 3, 3)
        pbc = jnp.array([[False, False, False]])
        cutoff = 2.0

        result = neighbor_list(
            positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=True
        )

        # cell_list always returns 3-tuple with shifts
        assert len(result) == 3
        neighbor_list_coo, neighbor_ptr, shifts = result
        assert neighbor_list_coo.shape[0] == 2
        assert neighbor_ptr.shape[0] == 5001
        assert int(neighbor_ptr[0]) == 0
        assert shifts.shape[1] == 3

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_auto_select_naive_dual_cutoff(self, dtype, device):
        """Auto-select naive_dual_cutoff when cutoff2 is provided → 6-tuple with PBC."""
        positions, cell, pbc = create_random_system_jax(100, 10.0, dtype=dtype)
        cutoff1 = 2.5
        cutoff2 = 3.5

        result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            cutoff2=cutoff2,
            max_neighbors1=50,
            max_neighbors2=50,
            return_neighbor_list=True,
        )

        assert len(result) == 6
        nlist1, ptr1, shifts1, nlist2, ptr2, shifts2 = result
        assert nlist1.shape[0] == 2
        assert nlist2.shape[0] == 2
        assert ptr1.shape[0] == 101
        assert ptr2.shape[0] == 101
        assert shifts1.shape[1] == 3
        assert shifts2.shape[1] == 3

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_auto_select_batch_naive(self, dtype, device):
        """Auto-select batch_naive when batch_idx is provided for small system."""
        positions1, cell1, pbc1 = create_random_system_jax(
            50, 10.0, dtype=dtype, seed=42
        )
        positions2, cell2, pbc2 = create_random_system_jax(
            30, 10.0, dtype=dtype, seed=43
        )
        cutoff = 2.0

        positions = jnp.concatenate([positions1, positions2], axis=0)
        cell = jnp.stack([cell1.squeeze(0), cell2.squeeze(0)], axis=0)
        pbc = jnp.stack([pbc1.squeeze(0), pbc2.squeeze(0)], axis=0)

        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax([50, 30])

        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
        )

        # batch_naive with PBC → 3-tuple
        assert len(result) == 3
        nlist, neighbor_ptr, _ = result
        assert nlist.shape[0] == 2
        assert neighbor_ptr.shape[0] == 81  # 50 + 30 + 1
        assert int(neighbor_ptr[0]) == 0

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_auto_select_batch_naive_dual_cutoff(self, dtype, device):
        """Auto-select batch_naive_dual_cutoff when both cutoff2 and batch_idx are provided."""
        positions1, cell1, pbc1 = create_random_system_jax(
            50, 10.0, dtype=dtype, seed=42
        )
        positions2, cell2, pbc2 = create_random_system_jax(
            30, 10.0, dtype=dtype, seed=43
        )

        positions = jnp.concatenate([positions1, positions2], axis=0)
        cell = jnp.stack([cell1.squeeze(0), cell2.squeeze(0)], axis=0)
        pbc = jnp.stack([pbc1.squeeze(0), pbc2.squeeze(0)], axis=0)

        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax([50, 30])

        cutoff1 = 2.5
        cutoff2 = 3.5

        result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cutoff2=cutoff2,
            max_neighbors1=50,
            max_neighbors2=50,
            return_neighbor_list=True,
        )

        assert len(result) == 6
        nlist1, ptr1, shifts1, nlist2, ptr2, shifts2 = result
        assert nlist1.shape[0] == 2
        assert nlist2.shape[0] == 2
        assert ptr1.shape[0] == 81
        assert ptr2.shape[0] == 81
        assert shifts1.shape[1] == 3
        assert shifts2.shape[1] == 3


# ==============================================================================
# Tests: Explicit Method
# ==============================================================================


class TestNeighborListExplicitMethod:
    """Test explicit method selection matches direct calls."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_explicit_naive(self, dtype, device):
        """Test explicit naive method matches direct naive_neighbor_list call."""
        positions, cell, pbc = create_random_system_jax(100, 10.0, dtype=dtype)
        cutoff = 2.0

        wrapper_result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            return_neighbor_list=False,
        )

        direct_result = naive_neighbor_list(
            positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=False
        )

        assert len(wrapper_result) == len(direct_result)
        assert_neighbor_matrix_equal_jax(wrapper_result, direct_result)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_explicit_cell_list(self, dtype, device):
        """Test explicit cell_list method matches direct cell_list call."""
        positions, cell, pbc = create_random_system_jax(500, 20.0, dtype=dtype)
        cutoff = 2.0

        wrapper_result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="cell_list",
            return_neighbor_list=False,
        )

        direct_result = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=False
        )

        assert len(wrapper_result) == len(direct_result)
        assert_neighbor_matrix_equal_jax(wrapper_result, direct_result)


# ==============================================================================
# Tests: Dual Cutoff
# ==============================================================================


class TestNeighborListDualCutoff:
    """Test dual cutoff functionality."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_naive_dual_cutoff(self, dtype, device):
        """Test explicit naive dual cutoff matches direct call."""
        positions, cell, pbc = create_random_system_jax(100, 10.0, dtype=dtype)
        cutoff1 = 2.5
        cutoff2 = 3.5

        wrapper_result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            cutoff2=cutoff2,
            method="naive_dual_cutoff",
            return_neighbor_list=False,
        )

        direct_result = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            cell=cell,
            pbc=pbc,
            return_neighbor_list=False,
        )

        assert len(wrapper_result) == 6
        assert len(direct_result) == 6

        # Compare first cutoff results
        assert_neighbor_matrix_equal_jax(wrapper_result[:3], direct_result[:3])
        # Compare second cutoff results
        assert_neighbor_matrix_equal_jax(wrapper_result[3:], direct_result[3:])

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_batch_naive_dual_cutoff(self, dtype, device):
        """Test batch naive dual cutoff matches direct call."""
        positions1, cell1, pbc1 = create_random_system_jax(
            50, 10.0, dtype=dtype, seed=42
        )
        positions2, cell2, pbc2 = create_random_system_jax(
            30, 10.0, dtype=dtype, seed=43
        )

        positions = jnp.concatenate([positions1, positions2], axis=0)
        cell = jnp.stack([cell1.squeeze(0), cell2.squeeze(0)], axis=0)
        pbc = jnp.stack([pbc1.squeeze(0), pbc2.squeeze(0)], axis=0)

        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax([50, 30])

        cutoff1 = 2.5
        cutoff2 = 3.5

        wrapper_result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cutoff2=cutoff2,
            method="batch_naive_dual_cutoff",
            max_neighbors1=50,
            max_neighbors2=50,
            return_neighbor_list=False,
        )

        direct_result = batch_naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=50,
            max_neighbors2=50,
            return_neighbor_list=False,
        )

        assert len(wrapper_result) == 6
        assert len(direct_result) == 6

        assert_neighbor_matrix_equal_jax(wrapper_result[:3], direct_result[:3])
        assert_neighbor_matrix_equal_jax(wrapper_result[3:], direct_result[3:])


# ==============================================================================
# Tests: Return Formats
# ==============================================================================


class TestNeighborListReturnFormats:
    """Test different return formats (matrix vs COO list)."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_return_neighbor_matrix(self, dtype, device):
        """Test returning neighbor matrix (default) has correct shapes."""
        positions, cell, pbc = create_random_system_jax(100, 10.0, dtype=dtype)
        cutoff = 5.0

        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            return_neighbor_list=False,
        )

        neighbor_matrix, num_neighbors, shifts = result

        assert neighbor_matrix.ndim == 2
        assert neighbor_matrix.shape[0] == 100
        assert num_neighbors.shape[0] == 100
        assert shifts.ndim == 3
        assert shifts.shape[0] == 100

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_return_neighbor_list_coo(self, dtype, device):
        """Test returning neighbor list in COO format has correct shapes."""
        positions, cell, pbc = create_random_system_jax(100, 10.0, dtype=dtype)
        cutoff = 5.0

        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            return_neighbor_list=True,
        )

        neighbor_list_coo, neighbor_ptr, shifts = result

        assert neighbor_list_coo.shape[0] == 2  # [sources, targets]
        assert neighbor_ptr.shape[0] == 101  # total_atoms + 1
        assert shifts.ndim == 2
        assert shifts.shape[1] == 3


# ==============================================================================
# Tests: Half Fill
# ==============================================================================


class TestNeighborListHalfFill:
    """Test half_fill parameter forwarding to naive method."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("half_fill", [False, True])
    def test_half_fill_parameter(self, dtype, device, half_fill):
        """Test that half_fill parameter is forwarded correctly."""
        positions, cell, pbc = create_random_system_jax(50, 10.0, dtype=dtype)
        cutoff = 5.0

        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            half_fill=half_fill,
            return_neighbor_list=True,
        )

        neighbor_list_coo, _, _ = result

        if half_fill:
            # Each pair should appear only once: no (i,j) and (j,i)
            sources = np.asarray(neighbor_list_coo[0])
            targets = np.asarray(neighbor_list_coo[1])
            pairs = set(zip(sources, targets))
            reverse_pairs = set(zip(targets, sources))
            overlap = pairs.intersection(reverse_pairs)
            assert len(overlap) == 0, "Half-fill should not have reciprocal pairs"


# ==============================================================================
# Tests: No PBC
# ==============================================================================


class TestNeighborListNoPBC:
    """Test neighbor list without periodic boundary conditions."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_no_pbc_naive(self, dtype, device):
        """Test naive without PBC returns 2-tuple (no shifts)."""
        key = jax.random.PRNGKey(42)
        positions = jax.random.normal(key, (100, 3), dtype=dtype) * 5.0
        cutoff = 3.0

        result = neighbor_list(
            positions, cutoff, method="naive", return_neighbor_list=True
        )

        assert len(result) == 2
        neighbor_list_coo, neighbor_ptr = result
        assert neighbor_list_coo.shape[0] == 2
        assert neighbor_ptr.shape[0] == 101


# ==============================================================================
# Tests: Invalid Method
# ==============================================================================


class TestNeighborListInvalidMethod:
    """Test error handling for invalid methods."""

    def test_invalid_method_name(self):
        """Test that invalid method name raises ValueError."""
        positions = jnp.ones((10, 3), dtype=jnp.float32)
        cutoff = 2.0

        with pytest.raises(ValueError, match="Invalid method"):
            neighbor_list(positions, cutoff, method="invalid_method")

    def test_dual_cutoff_without_cutoff2(self):
        """Test that naive_dual_cutoff without cutoff2 raises ValueError."""
        positions = jnp.ones((10, 3), dtype=jnp.float32)
        cutoff = 2.0

        with pytest.raises(ValueError, match="cutoff2 must be provided"):
            neighbor_list(positions, cutoff, method="naive_dual_cutoff")

    def test_batch_dual_cutoff_without_cutoff2(self):
        """Test that batch_naive_dual_cutoff without cutoff2 raises ValueError."""
        positions = jnp.ones((10, 3), dtype=jnp.float32)
        cutoff = 2.0

        with pytest.raises(ValueError, match="cutoff2 must be provided"):
            neighbor_list(positions, cutoff, method="batch_naive_dual_cutoff")


# ==============================================================================
# Tests: Kwargs Forwarding
# ==============================================================================


class TestNeighborListKwargs:
    """Test kwargs passing to underlying methods."""

    def test_kwargs_max_neighbors_naive(self):
        """Test passing max_neighbors kwarg to naive method shapes the matrix."""
        positions, cell, pbc = create_random_system_jax(50, 10.0, dtype=jnp.float32)
        cutoff = 5.0

        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            max_neighbors=20,
            return_neighbor_list=False,
        )

        neighbor_matrix, _, _ = result
        assert neighbor_matrix.shape[1] == 20

    def test_kwargs_max_neighbors_dual_cutoff(self):
        """Test passing max_neighbors1 and max_neighbors2 to dual cutoff."""
        positions, cell, pbc = create_random_system_jax(50, 10.0, dtype=jnp.float32)
        cutoff1 = 2.5
        cutoff2 = 3.5

        result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            cutoff2=cutoff2,
            method="naive_dual_cutoff",
            max_neighbors1=15,
            max_neighbors2=25,
            return_neighbor_list=False,
        )

        nm1, _, _, nm2, _, _ = result
        assert nm1.shape[1] == 15
        assert nm2.shape[1] == 25

    def test_kwargs_forwarded_with_auto_selection(self):
        """Test that kwargs are forwarded correctly with auto method selection."""
        positions, cell, pbc = create_random_system_jax(50, 10.0, dtype=jnp.float32)
        cutoff = 5.0

        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=25,
            return_neighbor_list=False,
        )

        neighbor_matrix, _, _ = result
        assert neighbor_matrix.shape[1] == 25


# ==============================================================================
# Tests: Edge Cases
# ==============================================================================


class TestNeighborListEdgeCases:
    """Test edge cases."""

    def test_single_atom(self):
        """Test with single atom system (no neighbors expected)."""
        positions = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
        cutoff = 2.0

        result = neighbor_list(
            positions, cutoff, method="naive", return_neighbor_list=True
        )

        assert len(result) == 2
        neighbor_list_coo, neighbor_ptr = result
        assert neighbor_list_coo.shape[1] == 0  # No pairs
        assert neighbor_ptr.shape[0] == 2  # 1 atom + 1
        assert int(neighbor_ptr[0]) == 0
