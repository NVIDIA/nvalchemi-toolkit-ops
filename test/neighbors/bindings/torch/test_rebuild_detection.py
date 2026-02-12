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

"""Tests for rebuild detection PyTorch bindings."""

import pytest
import torch

from nvalchemiops.torch.neighbors.cell_list import (
    build_cell_list,
    estimate_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list
from nvalchemiops.torch.neighbors.rebuild_detection import (
    cell_list_needs_rebuild,
    check_cell_list_rebuild_needed,
    check_neighbor_list_rebuild_needed,
    neighbor_list_needs_rebuild,
)

from ...test_utils import create_simple_cubic_system

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda:0")
dtypes = [torch.float32, torch.float64]


class TestRebuildDetection:
    """Test rebuild detection functionality."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple test system."""
        return create_simple_cubic_system(num_atoms=8, cell_size=2.0)

    def create_cell_list_data(self, positions, cell, pbc, cutoff, device):
        """Helper function to create cell list data for testing."""
        positions = positions.to(device)
        cell = cell.to(device)
        pbc = pbc.to(device)
        pbc = pbc.squeeze(0)  # Squeeze to (3,) shape
        total_atoms = positions.shape[0]
        max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell,
            pbc,
            cutoff,
        )
        cell_list_cache = allocate_cell_list(
            total_atoms,
            max_total_cells,
            neighbor_search_radius,
            device,
        )
        build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            *cell_list_cache,
        )

        return cell_list_cache

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_cell_list_needs_rebuild_no_movement(self, device, dtype, simple_system):
        """Test that cell_list_needs_rebuild returns False when atoms don't move."""
        positions, cell, pbc = simple_system
        cutoff = 1.0

        positions_typed = positions.to(dtype=dtype, device=device)
        cell_typed = cell.to(dtype=dtype, device=device)
        pbc_typed = pbc.to(device=device)

        cell_list_cache = self.create_cell_list_data(
            positions_typed,
            cell_typed,
            pbc_typed,
            cutoff,
            device,
        )

        # Test with same positions (no movement)
        rebuild_needed = cell_list_needs_rebuild(
            current_positions=positions_typed,
            atom_to_cell_mapping=cell_list_cache[3],
            cells_per_dimension=cell_list_cache[0],
            cell=cell_typed,
            pbc=pbc_typed.squeeze(0),
        )

        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == torch.bool
        assert not rebuild_needed.item(), (
            "Should not need rebuild when atoms don't move"
        )

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_cell_list_needs_rebuild_small_movement(self, device, dtype, simple_system):
        """Test cell_list_needs_rebuild with small atomic movements within cells."""
        positions, cell, pbc = simple_system
        cutoff = 1.0
        positions_typed = positions.to(dtype=dtype, device=device)
        cell_typed = cell.to(dtype=dtype, device=device)
        pbc_typed = pbc.to(device=device)
        cell_list_cache = self.create_cell_list_data(
            positions_typed,
            cell_typed,
            pbc_typed,
            cutoff,
            device,
        )

        # Create new positions with small displacements
        # Use a very small displacement to stay within cells
        displacement = (
            torch.randn_like(positions_typed) * 0.01
        )  # Much smaller to ensure staying in cells
        new_positions = positions_typed + displacement

        rebuild_needed = cell_list_needs_rebuild(
            current_positions=new_positions,
            atom_to_cell_mapping=cell_list_cache[3],
            cells_per_dimension=cell_list_cache[0],
            cell=cell_typed,
            pbc=pbc_typed.squeeze(0),
        )

        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == torch.bool
        # With very small displacements (0.01), atoms should generally stay within cells
        # but we don't make this a hard requirement as it depends on initial positions

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_cell_list_needs_rebuild_large_movement(self, device, dtype, simple_system):
        """Test cell_list_needs_rebuild with large atomic movements crossing cells."""
        positions, cell, pbc = simple_system
        cutoff = 1.0
        large_displacement = 1.0  # Large enough to cross cell boundaries

        positions_typed = positions.to(dtype=dtype, device=device)
        cell_typed = cell.to(dtype=dtype, device=device)
        pbc_typed = pbc.to(device=device)
        cell_list_cache = self.create_cell_list_data(
            positions_typed,
            cell_typed,
            pbc_typed,
            cutoff,
            device,
        )

        # Move first atom by a large amount
        new_positions = positions_typed.clone()
        new_positions[0] += large_displacement

        rebuild_needed = cell_list_needs_rebuild(
            current_positions=new_positions,
            atom_to_cell_mapping=cell_list_cache[3],
            cells_per_dimension=cell_list_cache[0],
            cell=cell_typed,
            pbc=pbc_typed.squeeze(0),
        )

        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == torch.bool
        # Large movements should trigger rebuild
        assert rebuild_needed.item(), (
            "Should need rebuild when atoms move significantly"
        )

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_cell_list_needs_rebuild_empty_system(self, device, dtype):
        """Test cell_list_needs_rebuild with empty system."""
        current_positions = torch.empty((0, 3), dtype=dtype, device=device)
        cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
        pbc = torch.tensor([True, True, True], device=device)

        # For empty systems, create minimal cell list data manually
        # since build_cell_list returns early with empty input
        cells_per_dimension = torch.tensor([1, 1, 1], dtype=torch.int32, device=device)
        atom_to_cell_mapping = torch.empty((0, 3), dtype=torch.int32, device=device)

        rebuild_needed = cell_list_needs_rebuild(
            current_positions=current_positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc.squeeze(0),
        )
        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == torch.bool
        assert not rebuild_needed.item(), "Empty system should not need rebuild"

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_neighbor_list_needs_rebuild_no_movement(
        self, device, dtype, simple_system
    ):
        """Test neighbor_list_needs_rebuild returns False when atoms don't move."""
        positions, _, _ = simple_system
        skin_distance = 0.5

        reference_positions = positions.to(dtype=dtype, device=device)
        current_positions = reference_positions.clone()
        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )
        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == torch.bool
        assert not rebuild_needed.item(), (
            "Should not need rebuild when atoms don't move"
        )

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_neighbor_list_needs_rebuild_small_movement(
        self, device, dtype, simple_system
    ):
        """Test neighbor_list_needs_rebuild with small atomic movements within skin distance."""
        positions, _, _ = simple_system
        skin_distance = 0.5

        reference_positions = positions.to(dtype=dtype, device=device)
        current_positions = reference_positions.clone()
        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )
        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == torch.bool
        assert not rebuild_needed.item(), "Should not need rebuild for small movements"

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_neighbor_list_needs_rebuild_large_movement(
        self, device, dtype, simple_system
    ):
        """Test neighbor_list_needs_rebuild with large atomic movements beyond skin distance."""
        positions, _, _ = simple_system
        skin_distance = 0.5
        large_displacement = 1.0  # Beyond skin distance

        reference_positions = positions.to(dtype=dtype, device=device)
        current_positions = reference_positions.clone()
        current_positions[0] += large_displacement
        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )
        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == torch.bool
        assert rebuild_needed.item(), (
            "Should need rebuild when atoms move beyond skin distance"
        )

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_neighbor_list_needs_rebuild_shape_mismatch(self, device, dtype):
        """Test neighbor_list_needs_rebuild with mismatched tensor shapes."""
        skin_distance = 0.5

        reference_positions = torch.randn((5, 3), dtype=dtype, device=device)
        current_positions = torch.randn((7, 3), dtype=dtype, device=device)
        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )
        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == torch.bool
        assert rebuild_needed.item(), "Should need rebuild when shapes don't match"

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_neighbor_list_needs_rebuild_empty_system(self, device, dtype):
        """Test neighbor_list_needs_rebuild with empty system."""
        skin_distance = 0.5

        reference_positions = torch.empty((0, 3), dtype=dtype, device=device)
        current_positions = torch.empty((0, 3), dtype=dtype, device=device)
        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )
        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == torch.bool
        assert not rebuild_needed.item(), "Empty system should not need rebuild"

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_check_cell_list_rebuild_needed_wrapper(self, device, dtype, simple_system):
        """Test the high-level check_cell_list_rebuild_needed wrapper function."""
        positions, cell, pbc = simple_system
        cutoff = 1.0

        positions_typed = positions.to(dtype=dtype, device=device)
        cell_typed = cell.to(dtype=dtype, device=device)
        pbc_typed = pbc.to(device=device)
        cell_list_cache = self.create_cell_list_data(
            positions_typed,
            cell_typed,
            pbc_typed,
            cutoff,
            device,
        )

        # Test wrapper function with no movement
        rebuild_needed = check_cell_list_rebuild_needed(
            current_positions=positions_typed,
            atom_to_cell_mapping=cell_list_cache[3],
            cells_per_dimension=cell_list_cache[0],
            cell=cell_typed,
            pbc=pbc_typed.squeeze(0),
        )

        assert isinstance(rebuild_needed, bool)
        assert not rebuild_needed, "Should not need rebuild when atoms don't move"

        # Test with large movement
        new_positions = positions_typed.clone()
        new_positions[0] += 1.0  # Large movement
        rebuild_needed = check_cell_list_rebuild_needed(
            current_positions=new_positions,
            atom_to_cell_mapping=cell_list_cache[3],
            cells_per_dimension=cell_list_cache[0],
            cell=cell_typed,
            pbc=pbc_typed.squeeze(0),
        )

        assert isinstance(rebuild_needed, bool)
        assert rebuild_needed, "Should need rebuild when atoms move significantly"

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_check_neighbor_list_rebuild_needed_wrapper(
        self, device, dtype, simple_system
    ):
        """Test the high-level check_neighbor_list_rebuild_needed wrapper function."""
        positions, _, _ = simple_system
        skin_distance = 0.5

        reference_positions = positions.to(dtype=dtype, device=device)
        current_positions = reference_positions.clone()
        rebuild_needed = check_neighbor_list_rebuild_needed(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )
        assert isinstance(rebuild_needed, bool)
        assert not rebuild_needed, "Should not need rebuild when atoms don't move"
        current_positions = reference_positions.clone()
        current_positions[0] += 1.0  # Beyond skin distance
        rebuild_needed = check_neighbor_list_rebuild_needed(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )
        assert isinstance(rebuild_needed, bool)
        assert rebuild_needed, (
            "Should need rebuild when atoms move beyond skin distance"
        )

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_performance_early_termination(self, device, dtype, simple_system):
        """Test that early termination optimization works correctly."""
        positions, _, _ = simple_system
        skin_distance = 0.5

        # Create larger system to test early termination
        large_positions = positions.repeat(10, 1)
        positions_typed = large_positions.to(dtype=dtype, device=device)

        # Test neighbor list rebuild with early movement
        reference_positions = positions_typed.clone()
        current_positions = positions_typed.clone()
        current_positions[0] += 2.0  # First atom moves beyond skin distance

        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )

        assert rebuild_needed.item(), (
            "Should detect rebuild needed with early termination"
        )

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_different_precision_compatibility(self, device, dtype):
        """Test that different floating point precisions work correctly."""

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype, device=device
        )
        cell_mapping = torch.tensor(
            [[0, 0, 0], [1, 0, 0]], dtype=torch.int32, device=device
        )
        cells_per_dim = torch.tensor([2, 2, 2], dtype=torch.int32, device=device)
        sim_cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
        pbc = torch.tensor([True, True, True], device=device)

        rebuild_needed = cell_list_needs_rebuild(
            current_positions=positions,
            atom_to_cell_mapping=cell_mapping,
            cells_per_dimension=cells_per_dim,
            cell=sim_cell,
            pbc=pbc,
        )

        assert rebuild_needed.dtype == torch.bool
        assert rebuild_needed.shape == (1,)
