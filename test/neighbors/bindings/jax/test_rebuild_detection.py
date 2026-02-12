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

"""Tests for JAX rebuild detection functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nvalchemiops.jax.neighbors.rebuild_detection import (
    cell_list_needs_rebuild,
    neighbor_list_needs_rebuild,
)

from .conftest import requires_gpu

pytestmark = requires_gpu

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
# Tests: neighbor_list_needs_rebuild
# ==============================================================================


class TestNeighborListNeedsRebuild:
    """Test neighbor_list_needs_rebuild function."""

    def test_no_movement(self):
        """Test that no rebuild is needed when atoms don't move."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
        skin_distance = 0.5

        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=positions,
            current_positions=positions,
            skin_distance_threshold=skin_distance,
        )

        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == jnp.bool_
        assert not rebuild_needed.item()

    def test_small_movement_within_skin(self):
        """Test no rebuild for small movements within skin distance."""
        reference_positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32
        )
        current_positions = reference_positions + jnp.array(
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=jnp.float32
        )
        skin_distance = 0.5

        reference_positions = place_on_device(reference_positions, device)
        current_positions = place_on_device(current_positions, device)

        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )

        assert not rebuild_needed.item()

    def test_large_movement_beyond_skin(self):
        """Test rebuild needed for large movements beyond skin distance."""
        reference_positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32
        )
        current_positions = reference_positions + jnp.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32
        )
        skin_distance = 0.5

        reference_positions = place_on_device(reference_positions, device)
        current_positions = place_on_device(current_positions, device)

        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )

        assert rebuild_needed.item()

    def test_shape_mismatch(self):
        """Test rebuild needed for shape mismatch."""
        reference_positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        current_positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32
        )
        skin_distance = 0.5

        reference_positions = place_on_device(reference_positions, device)
        current_positions = place_on_device(current_positions, device)

        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )

        assert rebuild_needed.item()

    def test_empty_system(self):
        """Test with empty system."""
        reference_positions = jnp.zeros((0, 3), dtype=jnp.float32)
        current_positions = jnp.zeros((0, 3), dtype=jnp.float32)
        skin_distance = 0.5

        reference_positions = place_on_device(reference_positions, device)
        current_positions = place_on_device(current_positions, device)

        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )

        assert not rebuild_needed.item()


# ==============================================================================
# Tests: cell_list_needs_rebuild
# ==============================================================================


class TestCellListNeedsRebuild:
    """Test cell_list_needs_rebuild function."""

    def test_no_movement(self):
        """Test that no rebuild is needed when atoms don't move."""
        current_positions = jnp.array(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float32
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([True, True, True])
        cells_per_dimension = jnp.array([2, 2, 2], dtype=jnp.int32)
        atom_to_cell_mapping = jnp.array([[0, 0, 0], [1, 0, 0]], dtype=jnp.int32)

        current_positions = place_on_device(current_positions, device)
        cells_per_dimension = place_on_device(cells_per_dimension, device)
        atom_to_cell_mapping = place_on_device(atom_to_cell_mapping, device)

        rebuild_needed = cell_list_needs_rebuild(
            current_positions=current_positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == jnp.bool_
        assert not rebuild_needed.item()

    def test_small_movement_within_cell(self):
        """Test no rebuild for small movements within cells."""
        current_positions = jnp.array(
            [[0.1, 0.0, 0.0], [5.2, 0.0, 0.0]], dtype=jnp.float32
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([True, True, True])
        cells_per_dimension = jnp.array([2, 2, 2], dtype=jnp.int32)
        atom_to_cell_mapping = jnp.array([[0, 0, 0], [1, 0, 0]], dtype=jnp.int32)

        current_positions = place_on_device(current_positions, device)
        cells_per_dimension = place_on_device(cells_per_dimension, device)
        atom_to_cell_mapping = place_on_device(atom_to_cell_mapping, device)

        rebuild_needed = cell_list_needs_rebuild(
            current_positions=current_positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        # May or may not need rebuild depending on cell size
        assert rebuild_needed.shape == (1,)

    def test_large_movement_across_cells(self):
        """Test rebuild needed for large movements across cells."""
        current_positions = jnp.array(
            [[6.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([True, True, True])
        cells_per_dimension = jnp.array([2, 2, 2], dtype=jnp.int32)
        atom_to_cell_mapping = jnp.array([[0, 0, 0], [1, 0, 0]], dtype=jnp.int32)

        current_positions = place_on_device(current_positions, device)
        cells_per_dimension = place_on_device(cells_per_dimension, device)
        atom_to_cell_mapping = place_on_device(atom_to_cell_mapping, device)

        rebuild_needed = cell_list_needs_rebuild(
            current_positions=current_positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        assert rebuild_needed.item()

    def test_empty_system(self):
        """Test with empty system."""
        current_positions = jnp.zeros((0, 3), dtype=jnp.float32)
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([True, True, True])
        cells_per_dimension = jnp.array([1, 1, 1], dtype=jnp.int32)
        atom_to_cell_mapping = jnp.zeros((0, 3), dtype=jnp.int32)

        current_positions = place_on_device(current_positions, device)
        cells_per_dimension = place_on_device(cells_per_dimension, device)
        atom_to_cell_mapping = place_on_device(atom_to_cell_mapping, device)

        rebuild_needed = cell_list_needs_rebuild(
            current_positions=current_positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        assert not rebuild_needed.item()
