# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for JAX unbatched neighbor list functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nvalchemiops.jax.neighbors.unbatched import cell_list, naive_neighbor_list

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
# Fixtures
# ==============================================================================


@pytest.fixture(params=get_available_devices())
def device(request):
    """Parametrized fixture for testing on CPU and GPU."""
    if request.param == "gpu" and len(jax.devices("gpu")) == 0:
        pytest.skip("No CUDA device available.")
    return request.param


# ==============================================================================
# Tests: naive_neighbor_list
# ==============================================================================


class TestNaiveNeighborList:
    """Test naive_neighbor_list function."""

    def test_single_atom_no_neighbors(self, device):
        """Test with single atom (should have no neighbors)."""
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        cutoff = 1.0

        positions = place_on_device(positions, device)

        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions, cutoff, max_neighbors=10
        )

        assert neighbor_matrix.shape == (1, 10)
        assert num_neighbors.shape == (1,)
        assert int(num_neighbors[0]) == 0

    def test_two_atom_within_cutoff(self, device):
        """Test with two atoms within cutoff."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32)
        cutoff = 1.0

        positions = place_on_device(positions, device)

        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions, cutoff, max_neighbors=10
        )

        assert neighbor_matrix.shape == (2, 10)
        assert num_neighbors.shape == (2,)
        # Each atom should find the other one
        assert int(num_neighbors[0]) >= 1
        assert int(num_neighbors[1]) >= 1

    def test_two_atom_outside_cutoff(self, device):
        """Test with two atoms outside cutoff."""
        positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32)
        cutoff = 1.0

        positions = place_on_device(positions, device)

        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions, cutoff, max_neighbors=10
        )

        assert int(num_neighbors[0]) == 0
        assert int(num_neighbors[1]) == 0

    def test_cubic_system_no_pbc(self, device):
        """Test with cubic lattice without PBC."""
        # 8 atoms in a cube
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        cutoff = 1.5

        positions = place_on_device(positions, device)

        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions, cutoff, max_neighbors=20
        )

        assert neighbor_matrix.shape == (8, 20)
        assert num_neighbors.shape == (8,)
        # Each corner atom should have 3 neighbors
        assert all(int(num_neighbors[i]) > 0 for i in range(8))

    def test_return_neighbor_list_format(self, device):
        """Test return_neighbor_list parameter."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ],
            dtype=jnp.float32,
        )
        cutoff = 1.0

        positions = place_on_device(positions, device)

        neighbor_list, neighbor_ptr = naive_neighbor_list(
            positions, cutoff, max_neighbors=10, return_neighbor_list=True
        )

        assert neighbor_list.shape[0] == 2  # COO format
        assert neighbor_ptr.shape == (4,)  # 3 atoms + 1

    def test_with_pbc(self, device):
        """Test with periodic boundary conditions."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [9.5, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([[True, True, True]])
        cutoff = 1.0

        positions = place_on_device(positions, device)
        cell = place_on_device(cell, device)
        pbc = place_on_device(pbc, device)

        neighbor_matrix, num_neighbors, shifts = naive_neighbor_list(
            positions, cutoff, cell=cell, pbc=pbc, max_neighbors=10
        )

        assert neighbor_matrix.shape == (2, 10)
        assert num_neighbors.shape == (2,)
        assert shifts.shape == (2, 10, 3)
        # With PBC, atoms should be neighbors
        assert int(num_neighbors[0]) >= 1
        assert int(num_neighbors[1]) >= 1

    def test_output_device_matches_input(self, device):
        """Test that output device matches input device."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32)
        cutoff = 1.0

        positions = place_on_device(positions, device)

        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions, cutoff, max_neighbors=10
        )

        # Check device matches
        expected_device = positions.devices().pop().platform
        assert neighbor_matrix.devices().pop().platform == expected_device
        assert num_neighbors.devices().pop().platform == expected_device


# ==============================================================================
# Tests: cell_list
# ==============================================================================


class TestCellList:
    """Test cell_list function."""

    def test_single_atom_no_neighbors(self, device):
        """Test cell_list with single atom."""
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([[True, True, True]])
        cutoff = 1.0

        positions = place_on_device(positions, device)
        cell = place_on_device(cell, device)
        pbc = place_on_device(pbc, device)

        neighbor_matrix, num_neighbors, shifts = cell_list(positions, cutoff, cell, pbc)

        assert num_neighbors.shape == (1,)
        assert int(num_neighbors[0]) == 0

    def test_cubic_system_with_pbc(self, device):
        """Test cell_list with cubic system."""
        # Simple cubic: 8 atoms at corners
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [2.0, 0.0, 2.0],
                [0.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=jnp.float32,
        )
        cell = jnp.array([[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]])
        pbc = jnp.array([[True, True, True]])
        cutoff = 2.5  # Include nearest neighbors at distance 2.0

        positions = place_on_device(positions, device)
        cell = place_on_device(cell, device)
        pbc = place_on_device(pbc, device)

        neighbor_matrix, num_neighbors, shifts = cell_list(positions, cutoff, cell, pbc)

        assert neighbor_matrix.shape[0] == 8
        assert num_neighbors.shape == (8,)
        assert shifts.shape[0] == 8
        assert shifts.shape[2] == 3
        # Each atom should have at least some neighbors
        assert jnp.sum(num_neighbors) > 0

    def test_return_neighbor_list_format(self, device):
        """Test cell_list with return_neighbor_list=True."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([[True, True, True]])
        cutoff = 1.0

        positions = place_on_device(positions, device)
        cell = place_on_device(cell, device)
        pbc = place_on_device(pbc, device)

        neighbor_list, neighbor_ptr, shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        assert neighbor_list.shape[0] == 2  # COO format
        assert neighbor_ptr.shape == (3,)  # 2 atoms + 1
        assert shifts.shape[1] == 3

    def test_no_pbc(self, device):
        """Test cell_list without PBC."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([[False, False, False]])
        cutoff = 1.0

        positions = place_on_device(positions, device)
        cell = place_on_device(cell, device)
        pbc = place_on_device(pbc, device)

        neighbor_matrix, num_neighbors, shifts = cell_list(positions, cutoff, cell, pbc)

        # With no PBC, all shifts should be zero
        if int(jnp.sum(num_neighbors)) > 0:
            assert jnp.all(shifts == 0)

    def test_output_device_matches_input(self, device):
        """Test that output device matches input device."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([[True, True, True]])
        cutoff = 1.0

        positions = place_on_device(positions, device)
        cell = place_on_device(cell, device)
        pbc = place_on_device(pbc, device)

        neighbor_matrix, num_neighbors, shifts = cell_list(positions, cutoff, cell, pbc)

        # Check device matches
        expected_device = positions.devices().pop().platform
        assert neighbor_matrix.devices().pop().platform == expected_device
        assert num_neighbors.devices().pop().platform == expected_device
        assert shifts.devices().pop().platform == expected_device

    def test_different_dtypes(self, device):
        """Test cell_list with different dtypes."""
        for dtype in [jnp.float32, jnp.float64]:
            positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype)
            cell = jnp.array(
                [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]], dtype=dtype
            )
            pbc = jnp.array([[True, True, True]])
            cutoff = 1.0

            positions = place_on_device(positions, device)
            cell = place_on_device(cell, device)
            pbc = place_on_device(pbc, device)

            neighbor_matrix, num_neighbors, shifts = cell_list(
                positions, cutoff, cell, pbc
            )

            assert neighbor_matrix.dtype == jnp.int32
            assert num_neighbors.dtype == jnp.int32
            assert shifts.dtype == jnp.int32
