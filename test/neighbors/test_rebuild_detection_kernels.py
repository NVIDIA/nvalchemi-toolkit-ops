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

"""Tests for rebuild detection warp launchers."""

import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.cell_list import build_cell_list
from nvalchemiops.neighbors.rebuild_detection import (
    check_cell_list_rebuild,
    check_neighbor_list_rebuild,
)
from nvalchemiops.torch.neighbors.cell_list import estimate_cell_list_sizes
from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

from .test_utils import create_simple_cubic_system

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda:0")
dtypes = [torch.float32, torch.float64]


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", dtypes)
class TestRebuildDetectionWpLaunchers:
    """Test the public launcher API for rebuild detection."""

    def test_check_cell_list_rebuild_no_movement(self, device, dtype):
        """Test check_cell_list_rebuild launcher with no atomic movement."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        pbc = pbc.squeeze(0)
        cutoff = 1.0

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(wp.device_from_torch(positions.device))

        # Build cell list using warp launcher
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
        cell_list_cache = allocate_cell_list(
            positions.shape[0], max_cells, neighbor_search_radius, device
        )
        (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = cell_list_cache

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool)
        wp_cells_per_dimension = wp.from_torch(cells_per_dimension, dtype=wp.int32)
        wp_atom_periodic_shifts = wp.from_torch(atom_periodic_shifts, dtype=wp.vec3i)
        wp_atom_to_cell_mapping = wp.from_torch(atom_to_cell_mapping, dtype=wp.vec3i)
        wp_atoms_per_cell_count = wp.from_torch(atoms_per_cell_count, dtype=wp.int32)
        wp_cell_atom_start_indices = wp.from_torch(
            cell_atom_start_indices, dtype=wp.int32
        )
        wp_cell_atom_list = wp.from_torch(cell_atom_list, dtype=wp.int32)

        build_cell_list(
            wp_positions,
            wp_cell,
            wp_pbc,
            cutoff,
            wp_cells_per_dimension,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
            wp_dtype,
            wp_device,
        )

        # Prepare rebuild flag output
        rebuild_needed = torch.zeros(1, dtype=torch.bool, device=device)
        wp_rebuild_needed = wp.from_torch(rebuild_needed, dtype=wp.bool)

        # Call launcher
        check_cell_list_rebuild(
            wp_positions,
            wp_atom_to_cell_mapping,
            wp_cells_per_dimension,
            wp_cell,
            wp_pbc,
            wp_rebuild_needed,
            wp_dtype,
            wp_device,
        )

        # Should not need rebuild
        assert not rebuild_needed.item(), "Should not need rebuild with no movement"

    def test_check_cell_list_rebuild_large_movement(self, device, dtype):
        """Test check_cell_list_rebuild launcher with large atomic movement."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        pbc = pbc.squeeze(0)
        cutoff = 1.0

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(wp.device_from_torch(positions.device))

        # Build cell list using warp launcher
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
        cell_list_cache = allocate_cell_list(
            positions.shape[0], max_cells, neighbor_search_radius, device
        )
        (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = cell_list_cache

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool)
        wp_cells_per_dimension = wp.from_torch(cells_per_dimension, dtype=wp.int32)
        wp_atom_periodic_shifts = wp.from_torch(atom_periodic_shifts, dtype=wp.vec3i)
        wp_atom_to_cell_mapping = wp.from_torch(atom_to_cell_mapping, dtype=wp.vec3i)
        wp_atoms_per_cell_count = wp.from_torch(atoms_per_cell_count, dtype=wp.int32)
        wp_cell_atom_start_indices = wp.from_torch(
            cell_atom_start_indices, dtype=wp.int32
        )
        wp_cell_atom_list = wp.from_torch(cell_atom_list, dtype=wp.int32)

        build_cell_list(
            wp_positions,
            wp_cell,
            wp_pbc,
            cutoff,
            wp_cells_per_dimension,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
            wp_dtype,
            wp_device,
        )

        # Move first atom significantly
        new_positions = positions.clone()
        new_positions[0] += 1.0
        wp_new_positions = wp.from_torch(new_positions, dtype=wp_vec_dtype)

        # Prepare rebuild flag output
        rebuild_needed = torch.zeros(1, dtype=torch.bool, device=device)
        wp_rebuild_needed = wp.from_torch(rebuild_needed, dtype=wp.bool)

        # Call launcher
        check_cell_list_rebuild(
            wp_new_positions,
            wp_atom_to_cell_mapping,
            wp_cells_per_dimension,
            wp_cell,
            wp_pbc,
            wp_rebuild_needed,
            wp_dtype,
            wp_device,
        )

        # Should need rebuild
        assert rebuild_needed.item(), "Should need rebuild with large movement"

    def test_check_neighbor_list_rebuild_no_movement(self, device, dtype):
        """Test check_neighbor_list_rebuild launcher with no movement."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        skin_distance = 0.5

        reference_positions = positions.clone()
        current_positions = positions.clone()

        # Prepare rebuild flag output
        rebuild_needed = torch.zeros(1, dtype=torch.bool, device=device)

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_device = str(wp.device_from_torch(positions.device))

        wp_reference = wp.from_torch(reference_positions, dtype=wp_vec_dtype)
        wp_current = wp.from_torch(current_positions, dtype=wp_vec_dtype)
        wp_rebuild_needed = wp.from_torch(rebuild_needed, dtype=wp.bool)

        # Call launcher
        check_neighbor_list_rebuild(
            wp_reference,
            wp_current,
            (skin_distance * skin_distance),
            wp_rebuild_needed,
            wp_dtype,
            wp_device,
        )

        # Should not need rebuild
        assert not rebuild_needed.item(), "Should not need rebuild with no movement"

    def test_check_neighbor_list_rebuild_large_movement(self, device, dtype):
        """Test check_neighbor_list_rebuild launcher with large movement."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        skin_distance = 0.5

        reference_positions = positions.clone()
        current_positions = positions.clone()
        current_positions[0] += 1.0  # Move beyond skin distance

        # Prepare rebuild flag output
        rebuild_needed = torch.zeros(1, dtype=torch.bool, device=device)

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_device = str(wp.device_from_torch(positions.device))

        wp_reference = wp.from_torch(reference_positions, dtype=wp_vec_dtype)
        wp_current = wp.from_torch(current_positions, dtype=wp_vec_dtype)
        wp_rebuild_needed = wp.from_torch(rebuild_needed, dtype=wp.bool)

        # Call launcher
        check_neighbor_list_rebuild(
            wp_reference,
            wp_current,
            (skin_distance * skin_distance),
            wp_rebuild_needed,
            wp_dtype,
            wp_device,
        )

        # Should need rebuild
        assert rebuild_needed.item(), "Should need rebuild with large movement"

    def test_check_neighbor_list_rebuild_empty_system(self, device, dtype):
        """Test check_neighbor_list_rebuild launcher with empty system."""
        reference_positions = torch.empty((0, 3), dtype=dtype, device=device)
        current_positions = torch.empty((0, 3), dtype=dtype, device=device)
        skin_distance = 0.5

        # Prepare rebuild flag output
        rebuild_needed = torch.zeros(1, dtype=torch.bool, device=device)

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_device = str(wp.device_from_torch(reference_positions.device))

        wp_reference = wp.from_torch(reference_positions, dtype=wp_vec_dtype)
        wp_current = wp.from_torch(current_positions, dtype=wp_vec_dtype)
        wp_rebuild_needed = wp.from_torch(rebuild_needed, dtype=wp.bool)

        # Call launcher
        check_neighbor_list_rebuild(
            wp_reference,
            wp_current,
            (skin_distance * skin_distance),
            wp_rebuild_needed,
            wp_dtype,
            wp_device,
        )

        # Should not need rebuild
        assert not rebuild_needed.item(), "Empty system should not need rebuild"
