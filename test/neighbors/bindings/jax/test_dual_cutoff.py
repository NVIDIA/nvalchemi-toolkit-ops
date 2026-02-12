# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for JAX bindings of naive dual cutoff neighbor list methods."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors import (
    batch_naive_neighbor_list_dual_cutoff,
    naive_neighbor_list_dual_cutoff,
)

# ==============================================================================
# Device Utilities
# ==============================================================================


def get_available_devices() -> list[str]:
    """Get available JAX devices (CPU and GPU if available)."""
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("gpu")
    except RuntimeError:
        pass
    return devices


def place_on_device(arr: jax.Array, device: str) -> jax.Array:
    """Place a JAX array on the specified device type."""
    if device == "cpu":
        jax_device = jax.devices("cpu")[0]
    else:
        jax_device = jax.devices("gpu")[0]
    return jax.device_put(arr, jax_device)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture(params=get_available_devices())
def device(request):
    """Parametrized fixture for testing on CPU and GPU."""
    if request.param == "gpu" and len(jax.devices("gpu")) == 0:
        pytest.skip("No CUDA device available.")
    return request.param


# ==============================================================================
# Utility Functions
# ==============================================================================


def create_simple_cubic_system_jax(
    num_atoms: int = 8,
    cell_size: float = 2.0,
    dtype=jnp.float32,
    device: str = "cpu",
):
    """Create a simple cubic system with JAX arrays.

    Parameters
    ----------
    num_atoms : int
        Number of atoms (will be rounded to nearest perfect cube)
    cell_size : float
        Size of the cubic cell
    dtype : jnp dtype
        Data type for arrays
    device : str
        Device to place arrays on

    Returns
    -------
    positions : jax.Array, shape (num_atoms, 3)
        Atomic coordinates
    cell : jax.Array, shape (1, 3, 3)
        Cell matrix
    pbc : jax.Array, shape (1, 3)
        Periodic boundary condition flags
    """
    n_side = int(round(num_atoms ** (1 / 3)))
    if n_side**3 != num_atoms:
        n_side = int(np.ceil(num_atoms ** (1 / 3)))

    coords = []
    spacing = cell_size / n_side
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                if len(coords) < num_atoms:
                    coords.append([i * spacing, j * spacing, k * spacing])

    positions = jnp.array(coords[:num_atoms], dtype=dtype)
    cell = (jnp.eye(3, dtype=dtype) * cell_size).reshape(1, 3, 3)
    pbc = jnp.array([[True, True, True]])

    return (
        place_on_device(positions, device),
        place_on_device(cell, device),
        place_on_device(pbc, device),
    )


def create_batch_idx_and_ptr_jax(atoms_per_system: list[int], device: str = "cpu"):
    """Create batch_idx and batch_ptr arrays for JAX.

    Parameters
    ----------
    atoms_per_system : list[int]
        Number of atoms in each system
    device : str
        Device to place arrays on

    Returns
    -------
    batch_idx : jax.Array, shape (total_atoms,)
        System index for each atom
    batch_ptr : jax.Array, shape (num_systems + 1,)
        Cumulative atom counts
    """
    total_atoms = sum(atoms_per_system)
    batch_idx = jnp.zeros(total_atoms, dtype=jnp.int32)
    batch_ptr_list = [0]

    start = 0
    for i, n in enumerate(atoms_per_system):
        batch_idx = batch_idx.at[start : start + n].set(i)
        start += n
        batch_ptr_list.append(start)

    batch_ptr = jnp.array(batch_ptr_list, dtype=jnp.int32)

    return place_on_device(batch_idx, device), place_on_device(batch_ptr, device)


# ==============================================================================
# Tests: naive_neighbor_list_dual_cutoff
# ==============================================================================


class TestNaiveDualCutoffCorrectness:
    """Test correctness of naive dual cutoff neighbor list."""

    def test_matrix_format_no_pbc(self, device):
        """Test dual cutoff neighbor list in matrix format without PBC."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=jnp.float32, device=device
        )

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1,
                cutoff2,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
            )
        )

        # Verify output shapes and types
        assert neighbor_matrix1.shape == (8, max_neighbors1)
        assert neighbor_matrix2.shape == (8, max_neighbors2)
        assert num_neighbors1.shape == (8,)
        assert num_neighbors2.shape == (8,)
        assert neighbor_matrix1.dtype == jnp.int32
        assert neighbor_matrix2.dtype == jnp.int32
        assert num_neighbors1.dtype == jnp.int32
        assert num_neighbors2.dtype == jnp.int32

        # Verify neighbor counts are reasonable
        assert jnp.all(num_neighbors1 >= 0)
        assert jnp.all(num_neighbors2 >= 0)
        assert jnp.all(num_neighbors1 <= max_neighbors1)
        assert jnp.all(num_neighbors2 <= max_neighbors2)
        # Larger cutoff should find at least as many neighbors
        assert jnp.all(num_neighbors2 >= num_neighbors1)

    def test_matrix_format_with_pbc(self, device):
        """Test dual cutoff neighbor list in matrix format with PBC."""
        positions, cell, pbc = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=jnp.float32, device=device
        )

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            pbc=pbc,
            cell=cell,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
        )

        # Verify output shapes and types
        assert neighbor_matrix1.shape == (8, max_neighbors1)
        assert neighbor_matrix2.shape == (8, max_neighbors2)
        assert neighbor_matrix_shifts1.shape == (8, max_neighbors1, 3)
        assert neighbor_matrix_shifts2.shape == (8, max_neighbors2, 3)
        assert num_neighbors1.shape == (8,)
        assert num_neighbors2.shape == (8,)

        # Verify dtypes
        assert neighbor_matrix1.dtype == jnp.int32
        assert neighbor_matrix2.dtype == jnp.int32
        assert num_neighbors1.dtype == jnp.int32
        assert num_neighbors2.dtype == jnp.int32
        assert neighbor_matrix_shifts1.dtype == jnp.int32
        assert neighbor_matrix_shifts2.dtype == jnp.int32

        # Verify neighbor counts
        assert jnp.all(num_neighbors1 >= 0)
        assert jnp.all(num_neighbors2 >= 0)
        assert jnp.all(num_neighbors2 >= num_neighbors1)


class TestNaiveDualCutoffEdgeCases:
    """Test edge cases for naive dual cutoff neighbor list."""

    def test_single_atom(self, device):
        """Test with single atom (should have no neighbors)."""
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        positions = place_on_device(positions, device)

        cutoff1 = 1.0
        cutoff2 = 1.5

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1,
                cutoff2,
                max_neighbors1=10,
                max_neighbors2=10,
            )
        )

        assert int(num_neighbors1[0]) == 0
        assert int(num_neighbors2[0]) == 0

    def test_identical_cutoffs(self, device):
        """Test with identical cutoffs (both lists should match)."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=jnp.float32, device=device
        )

        cutoff = 1.2
        max_neighbors = 20

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff,
                cutoff,
                max_neighbors1=max_neighbors,
                max_neighbors2=max_neighbors,
            )
        )

        # Neighbor counts should be identical
        assert jnp.all(num_neighbors1 == num_neighbors2)


# ==============================================================================
# Tests: batch_naive_neighbor_list_dual_cutoff
# ==============================================================================


class TestBatchNaiveDualCutoffCorrectness:
    """Test correctness of batch naive dual cutoff neighbor list."""

    def test_matrix_format_no_pbc(self, device):
        """Test dual cutoff batch neighbor list in matrix format without PBC."""
        # Create two simple systems
        positions1, _, _ = create_simple_cubic_system_jax(
            num_atoms=6, cell_size=2.0, dtype=jnp.float32, device=device
        )
        positions2, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.5, dtype=jnp.float32, device=device
        )

        # Concatenate positions
        positions_batch = jnp.concatenate([positions1, positions2], axis=0)

        atoms_per_system = [6, 8]
        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 20
        max_neighbors2 = 30

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
            )
        )

        # Check output types and shapes
        expected_rows = positions_batch.shape[0]
        assert neighbor_matrix1.dtype == jnp.int32
        assert neighbor_matrix2.dtype == jnp.int32
        assert num_neighbors1.dtype == jnp.int32
        assert num_neighbors2.dtype == jnp.int32
        assert neighbor_matrix1.shape == (expected_rows, max_neighbors1)
        assert neighbor_matrix2.shape == (expected_rows, max_neighbors2)
        assert num_neighbors1.shape == (expected_rows,)
        assert num_neighbors2.shape == (expected_rows,)

        # Verify neighbor counts
        assert jnp.all(num_neighbors1 >= 0)
        assert jnp.all(num_neighbors2 >= 0)
        assert jnp.all(num_neighbors2 >= num_neighbors1)

    def test_matrix_format_with_pbc(self, device):
        """Test dual cutoff batch neighbor list in matrix format with PBC."""
        # Create two simple systems
        positions1, cell1, pbc1 = create_simple_cubic_system_jax(
            num_atoms=6, cell_size=2.0, dtype=jnp.float32, device=device
        )
        positions2, cell2, pbc2 = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.5, dtype=jnp.float32, device=device
        )

        # Concatenate
        positions_batch = jnp.concatenate([positions1, positions2], axis=0)
        cell_batch = jnp.concatenate([cell1, cell2], axis=0)
        pbc_batch = jnp.concatenate([pbc1, pbc2], axis=0)

        atoms_per_system = [6, 8]
        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 20
        max_neighbors2 = 30

        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=pbc_batch,
            cell=cell_batch,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
        )

        # Check output types and shapes
        expected_rows = positions_batch.shape[0]
        assert neighbor_matrix1.shape == (expected_rows, max_neighbors1)
        assert neighbor_matrix2.shape == (expected_rows, max_neighbors2)
        assert neighbor_matrix_shifts1.shape == (expected_rows, max_neighbors1, 3)
        assert neighbor_matrix_shifts2.shape == (expected_rows, max_neighbors2, 3)
        assert num_neighbors1.shape == (expected_rows,)
        assert num_neighbors2.shape == (expected_rows,)

        # Verify dtypes
        assert neighbor_matrix1.dtype == jnp.int32
        assert neighbor_matrix2.dtype == jnp.int32
        assert num_neighbors1.dtype == jnp.int32
        assert num_neighbors2.dtype == jnp.int32

        # Verify neighbor counts
        assert jnp.all(num_neighbors1 >= 0)
        assert jnp.all(num_neighbors2 >= 0)
        assert jnp.all(num_neighbors2 >= num_neighbors1)

    def test_larger_cutoff_finds_more_neighbors(self, device):
        """Verify that larger cutoff finds at least as many neighbors."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=jnp.float32, device=device
        )

        atoms_per_system = [8]
        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 2.0
        max_neighbors1 = 20
        max_neighbors2 = 40

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
            )
        )

        # Every atom should have more (or equal) neighbors with larger cutoff
        assert jnp.all(num_neighbors2 >= num_neighbors1)
        # At least some atoms should have strictly more neighbors
        assert jnp.any(num_neighbors2 > num_neighbors1)


class TestBatchNaiveDualCutoffEdgeCases:
    """Test edge cases for batch naive dual cutoff neighbor list."""

    def test_single_atom_system(self, device):
        """Test with single atom system (should have no neighbors)."""
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        positions = place_on_device(positions, device)

        atoms_per_system = [1]
        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1,
                cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=10,
                max_neighbors2=10,
            )
        )

        assert int(num_neighbors1[0]) == 0
        assert int(num_neighbors2[0]) == 0

    def test_identical_cutoffs(self, device):
        """Test with identical cutoffs (both lists should match)."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=jnp.float32, device=device
        )

        atoms_per_system = [8]
        batch_idx, batch_ptr = create_batch_idx_and_ptr_jax(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 20

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions,
                cutoff,
                cutoff,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=max_neighbors,
                max_neighbors2=max_neighbors,
            )
        )

        # Neighbor counts should be identical
        assert jnp.all(num_neighbors1 == num_neighbors2)
