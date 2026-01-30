# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for JAX batched neighbor list functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nvalchemiops.jax.neighbors.batched import (
    batch_cell_list,
    batch_naive_neighbor_list,
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
# Fixtures
# ==============================================================================


@pytest.fixture(params=get_available_devices())
def device(request):
    """Parametrized fixture for testing on CPU and GPU."""
    if request.param == "gpu" and len(jax.devices("gpu")) == 0:
        pytest.skip("No CUDA device available.")
    return request.param


# ==============================================================================
# Tests: batch_naive_neighbor_list
# ==============================================================================


class TestBatchNaiveNeighborList:
    """Test batch_naive_neighbor_list function."""

    def test_two_systems_no_pbc(self, device):
        """Test with two separate systems without PBC."""
        # System 1: 2 atoms
        positions1 = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32)
        # System 2: 2 atoms
        positions2 = jnp.array([[10.0, 0.0, 0.0], [10.5, 0.0, 0.0]], dtype=jnp.float32)

        positions = jnp.vstack([positions1, positions2])
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        cutoff = 1.0

        positions = place_on_device(positions, device)
        batch_idx = place_on_device(batch_idx, device)
        batch_ptr = place_on_device(batch_ptr, device)

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
        )

        assert neighbor_matrix.shape == (4, 10)
        assert num_neighbors.shape == (4,)

    def test_two_systems_with_pbc(self, device):
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

        positions = place_on_device(positions, device)
        cells = place_on_device(cells, device)
        pbcs = place_on_device(pbcs, device)
        batch_idx = place_on_device(batch_idx, device)
        batch_ptr = place_on_device(batch_ptr, device)

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

    def test_different_system_sizes(self, device):
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

        positions = place_on_device(positions, device)
        batch_idx = place_on_device(batch_idx, device)
        batch_ptr = place_on_device(batch_ptr, device)

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
        )

        assert neighbor_matrix.shape == (5, 10)
        assert num_neighbors.shape == (5,)

    def test_output_device_matches_input(self, device):
        """Test that output device matches input device."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32)
        batch_idx = jnp.array([0, 0], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2], dtype=jnp.int32)
        cutoff = 1.0

        positions = place_on_device(positions, device)
        batch_idx = place_on_device(batch_idx, device)
        batch_ptr = place_on_device(batch_ptr, device)

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
        )

        expected_device = positions.devices().pop().platform
        assert neighbor_matrix.devices().pop().platform == expected_device
        assert num_neighbors.devices().pop().platform == expected_device


# ==============================================================================
# Tests: batch_cell_list
# ==============================================================================


class TestBatchCellList:
    """Test batch_cell_list function."""

    def test_two_systems_with_pbc(self, device):
        """Test batch_cell_list with two systems."""
        positions1 = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        positions2 = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )

        positions = jnp.vstack([positions1, positions2])

        cells = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ]
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])

        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        cutoff = 2.0

        positions = place_on_device(positions, device)
        cells = place_on_device(cells, device)
        pbcs = place_on_device(pbcs, device)
        batch_idx = place_on_device(batch_idx, device)
        batch_ptr = place_on_device(batch_ptr, device)

        neighbor_matrix, num_neighbors, shifts = batch_cell_list(
            positions,
            cutoff,
            cells,
            pbcs,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
        )

        assert neighbor_matrix.shape[0] == 4
        assert num_neighbors.shape == (4,)
        assert shifts.shape[0] == 4
        assert shifts.shape[2] == 3

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
        batch_idx = jnp.array([0, 0], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2], dtype=jnp.int32)
        cutoff = 1.0

        positions = place_on_device(positions, device)
        cell = place_on_device(cell, device)
        pbc = place_on_device(pbc, device)
        batch_idx = place_on_device(batch_idx, device)
        batch_ptr = place_on_device(batch_ptr, device)

        neighbor_matrix, num_neighbors, shifts = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
        )

        expected_device = positions.devices().pop().platform
        assert neighbor_matrix.devices().pop().platform == expected_device
        assert num_neighbors.devices().pop().platform == expected_device
        assert shifts.devices().pop().platform == expected_device
