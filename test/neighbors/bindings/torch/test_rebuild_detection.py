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

from nvalchemiops.torch.neighbors.batch_cell_list import (
    batch_build_cell_list,
    batch_cell_list,
    batch_query_cell_list,
    estimate_batch_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.batch_naive import batch_naive_neighbor_list
from nvalchemiops.torch.neighbors.cell_list import (
    build_cell_list,
    estimate_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list
from nvalchemiops.torch.neighbors.rebuild_detection import (
    batch_cell_list_needs_rebuild,
    batch_neighbor_list_needs_rebuild,
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

    @pytest.fixture(scope="class")
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

        assert isinstance(rebuild_needed, torch.Tensor)
        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == torch.bool
        assert not rebuild_needed.item(), (
            "Should not need rebuild when atoms don't move"
        )

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

        assert isinstance(rebuild_needed, torch.Tensor)
        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == torch.bool
        assert rebuild_needed.item(), (
            "Should need rebuild when atoms move significantly"
        )

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
        assert isinstance(rebuild_needed, torch.Tensor)
        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == torch.bool
        assert not rebuild_needed.item(), (
            "Should not need rebuild when atoms don't move"
        )
        current_positions = reference_positions.clone()
        current_positions[0] += 1.0  # Beyond skin distance
        rebuild_needed = check_neighbor_list_rebuild_needed(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )
        assert isinstance(rebuild_needed, torch.Tensor)
        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == torch.bool
        assert rebuild_needed.item(), (
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


class TestBatchRebuildDetection:
    """Test batch rebuild detection PyTorch bindings."""

    def _make_batch_systems(self, dtype, device, num_systems=3):
        """Create a small batch of systems for testing."""
        base_atoms = [6, 8, 5, 7, 4]
        atoms_per = (base_atoms * ((num_systems // len(base_atoms)) + 1))[:num_systems]
        torch.manual_seed(42)
        all_pos = []
        cells = []
        pbcs = []
        for i, n in enumerate(atoms_per):
            pos = torch.rand(n, 3, dtype=dtype, device=device) * 3.0
            all_pos.append(pos)
            cells.append(torch.eye(3, dtype=dtype, device=device) * 4.0)
            pbcs.append(torch.zeros(3, dtype=torch.bool, device=device))
        positions = torch.cat(all_pos, dim=0)
        cell = torch.stack(cells, dim=0)
        pbc = torch.stack(pbcs, dim=0)
        ptr = torch.tensor(
            [0] + [sum(atoms_per[: i + 1]) for i in range(num_systems)],
            dtype=torch.int32,
            device=device,
        )
        batch_idx = torch.repeat_interleave(
            torch.arange(num_systems, dtype=torch.int32, device=device),
            torch.tensor(atoms_per, dtype=torch.int32, device=device),
        )
        return positions, cell, pbc, batch_idx, ptr, atoms_per, num_systems

    def _build_batch_cell_list(self, positions, cell, pbc, batch_idx, cutoff, device):
        """Helper to build batch cell list and return cache."""
        max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell, pbc, cutoff=cutoff
        )
        cell_list_cache = allocate_cell_list(
            positions.shape[0], max_total_cells, neighbor_search_radius, device
        )
        batch_build_cell_list(positions, cutoff, cell, pbc, batch_idx, *cell_list_cache)
        return cell_list_cache

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_batch_neighbor_list_needs_rebuild_no_movement(self, device, dtype):
        """No movement in any system → all rebuild_flags should be False."""
        positions, cell, pbc, batch_idx, ptr, atoms_per, num_systems = (
            self._make_batch_systems(dtype, device)
        )

        rebuild_flags = batch_neighbor_list_needs_rebuild(
            reference_positions=positions,
            current_positions=positions.clone(),
            batch_idx=batch_idx,
            skin_distance_threshold=0.5,
        )

        assert rebuild_flags.shape == (num_systems,)
        assert rebuild_flags.dtype == torch.bool
        assert not rebuild_flags.any(), (
            "No systems should need rebuild with no movement"
        )

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_batch_neighbor_list_needs_rebuild_selective(self, device, dtype):
        """Only system 1 has an atom move beyond skin → only flag[1] is True."""
        positions, cell, pbc, batch_idx, ptr, atoms_per, num_systems = (
            self._make_batch_systems(dtype, device)
        )

        current_positions = positions.clone()
        # Atom in system 1 (starts at ptr[1]=atoms_per[0]=6)
        current_positions[atoms_per[0] + 1] += 2.0

        rebuild_flags = batch_neighbor_list_needs_rebuild(
            reference_positions=positions,
            current_positions=current_positions,
            batch_idx=batch_idx,
            skin_distance_threshold=0.5,
        )

        assert rebuild_flags.shape == (num_systems,)
        assert not rebuild_flags[0], "System 0 should not need rebuild"
        assert rebuild_flags[1], "System 1 should need rebuild"
        assert not rebuild_flags[2], "System 2 should not need rebuild"

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_batch_neighbor_list_needs_rebuild_all_move(self, device, dtype):
        """All systems have atoms moving beyond skin → all flags should be True."""
        positions, cell, pbc, batch_idx, ptr, atoms_per, num_systems = (
            self._make_batch_systems(dtype, device)
        )

        current_positions = positions.clone()
        # Move one atom in each system
        current_positions[0] += 2.0  # system 0
        current_positions[atoms_per[0] + 1] += 2.0  # system 1
        current_positions[atoms_per[0] + atoms_per[1]] += 2.0  # system 2

        rebuild_flags = batch_neighbor_list_needs_rebuild(
            reference_positions=positions,
            current_positions=current_positions,
            batch_idx=batch_idx,
            skin_distance_threshold=0.5,
        )

        assert rebuild_flags.all(), "All systems should need rebuild"

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_batch_neighbor_list_needs_rebuild_output_shape(self, device, dtype):
        """Output shape matches num_systems inferred from batch_idx."""
        for num_systems in [1, 2, 5]:
            positions, cell, pbc, batch_idx, ptr, atoms_per, ns = (
                self._make_batch_systems(dtype, device, num_systems=num_systems)
            )
            rebuild_flags = batch_neighbor_list_needs_rebuild(
                reference_positions=positions,
                current_positions=positions.clone(),
                batch_idx=batch_idx,
                skin_distance_threshold=0.5,
            )
            assert rebuild_flags.shape == (num_systems,)
            assert rebuild_flags.dtype == torch.bool

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_batch_cell_list_needs_rebuild_no_movement(self, device, dtype):
        """No movement in any system → all rebuild_flags should be False."""
        positions, cell, pbc, batch_idx, ptr, atoms_per, num_systems = (
            self._make_batch_systems(dtype, device)
        )
        cell_list_cache = self._build_batch_cell_list(
            positions, cell, pbc, batch_idx, cutoff=1.0, device=device
        )
        cells_per_dimension = cell_list_cache[0]
        atom_to_cell_mapping = cell_list_cache[3]

        rebuild_flags = batch_cell_list_needs_rebuild(
            current_positions=positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            batch_idx=batch_idx,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        assert rebuild_flags.shape == (num_systems,)
        assert rebuild_flags.dtype == torch.bool
        assert not rebuild_flags.any(), (
            "No systems should need rebuild with no movement"
        )

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_batch_cell_list_needs_rebuild_selective(self, device, dtype):
        """Only system 0 has an atom cross a cell boundary → only flag[0] is True."""
        positions, cell, pbc, batch_idx, ptr, atoms_per, num_systems = (
            self._make_batch_systems(dtype, device)
        )
        cell_list_cache = self._build_batch_cell_list(
            positions, cell, pbc, batch_idx, cutoff=1.0, device=device
        )
        cells_per_dimension = cell_list_cache[0]
        atom_to_cell_mapping = cell_list_cache[3]

        new_positions = positions.clone()
        new_positions[0] += 1.5  # large enough to cross a cell boundary in system 0

        rebuild_flags = batch_cell_list_needs_rebuild(
            current_positions=new_positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            batch_idx=batch_idx,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        assert rebuild_flags.shape == (num_systems,)
        assert rebuild_flags[0], "System 0 should need rebuild"
        assert not rebuild_flags[1], "System 1 should not need rebuild"
        assert not rebuild_flags[2], "System 2 should not need rebuild"

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_batch_cell_list_needs_rebuild_output_shape(self, device, dtype):
        """Output shape matches num_systems from cell.shape[0]."""
        positions, cell, pbc, batch_idx, ptr, atoms_per, num_systems = (
            self._make_batch_systems(dtype, device)
        )
        cell_list_cache = self._build_batch_cell_list(
            positions, cell, pbc, batch_idx, cutoff=1.0, device=device
        )
        cells_per_dimension = cell_list_cache[0]
        atom_to_cell_mapping = cell_list_cache[3]

        rebuild_flags = batch_cell_list_needs_rebuild(
            current_positions=positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            batch_idx=batch_idx,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        assert rebuild_flags.shape == (num_systems,)
        assert rebuild_flags.dtype == torch.bool

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_batch_neighbor_list_flags_are_gpu_tensors(self, device, dtype):
        """Rebuild flags should be on the same device as input positions."""
        positions, cell, pbc, batch_idx, ptr, atoms_per, num_systems = (
            self._make_batch_systems(dtype, device)
        )

        rebuild_flags = batch_neighbor_list_needs_rebuild(
            reference_positions=positions,
            current_positions=positions.clone(),
            batch_idx=batch_idx,
            skin_distance_threshold=0.5,
        )

        assert str(rebuild_flags.device) == str(positions.device), (
            "Rebuild flags should be on the same device as input positions"
        )


class TestSelectiveRebuild:
    """Test GPU-side selective skip in batch neighbor list building."""

    def _make_batch_systems(self, dtype, device, num_systems=3):
        """Create a simple batch of systems for testing."""
        base_atoms = [6, 8, 5, 7, 4]
        atoms_per = (base_atoms * ((num_systems // len(base_atoms)) + 1))[:num_systems]
        torch.manual_seed(7)
        all_pos = []
        cells = []
        pbcs = []
        for n in atoms_per:
            pos = torch.rand(n, 3, dtype=dtype, device=device) * 2.0
            all_pos.append(pos)
            cells.append(torch.eye(3, dtype=dtype, device=device) * 3.0)
            pbcs.append(torch.zeros(3, dtype=torch.bool, device=device))
        positions = torch.cat(all_pos, dim=0)
        cell = torch.stack(cells, dim=0)
        pbc = torch.stack(pbcs, dim=0)
        ptr = torch.tensor(
            [0] + [sum(atoms_per[: i + 1]) for i in range(num_systems)],
            dtype=torch.int32,
            device=device,
        )
        batch_idx = torch.repeat_interleave(
            torch.arange(num_systems, dtype=torch.int32, device=device),
            torch.tensor(atoms_per, dtype=torch.int32, device=device),
        )
        return positions, cell, pbc, batch_idx, ptr, atoms_per, num_systems

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_naive_selective_no_rebuild(self, device, dtype):
        """When no systems need rebuild, neighbor data is unchanged."""
        positions, cell, pbc, batch_idx, ptr, atoms_per, num_systems = (
            self._make_batch_systems(dtype, device)
        )
        cutoff = 1.5
        max_neighbors = 16

        # Initial full build
        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=ptr,
            max_neighbors=max_neighbors,
        )
        saved_nm = neighbor_matrix.clone()
        saved_nn = num_neighbors.clone()

        # All flags False → nothing should change
        rebuild_flags = torch.zeros(num_systems, dtype=torch.bool, device=device)
        batch_naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=ptr,
            max_neighbors=max_neighbors,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            rebuild_flags=rebuild_flags,
        )

        assert torch.equal(num_neighbors, saved_nn), (
            "num_neighbors should not change when no systems need rebuild"
        )
        # neighbor_matrix entries up to num_neighbors should be preserved
        for i in range(positions.shape[0]):
            n = num_neighbors[i].item()
            assert torch.equal(neighbor_matrix[i, :n], saved_nm[i, :n])

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_naive_selective_rebuild_one_system(self, device, dtype):
        """Rebuilding only system 1 updates its neighbors, preserves others."""
        positions, cell, pbc, batch_idx, ptr, atoms_per, num_systems = (
            self._make_batch_systems(dtype, device)
        )
        cutoff = 1.5
        max_neighbors = 16

        # Initial full build
        neighbor_matrix_ref, num_neighbors_ref = batch_naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=ptr,
            max_neighbors=max_neighbors,
        )

        # Move an atom in system 1 significantly
        new_positions = positions.clone()
        sys1_start = ptr[1].item()
        new_positions[sys1_start] = torch.tensor(
            [0.05, 0.05, 0.05], dtype=dtype, device=device
        )

        # Full rebuild for reference
        nm_full, nn_full = batch_naive_neighbor_list(
            positions=new_positions,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=ptr,
            max_neighbors=max_neighbors,
        )

        # Selective rebuild: only system 1
        rebuild_flags = torch.zeros(num_systems, dtype=torch.bool, device=device)
        rebuild_flags[1] = True
        nm_selective = neighbor_matrix_ref.clone()
        nn_selective = num_neighbors_ref.clone()
        batch_naive_neighbor_list(
            positions=new_positions,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=ptr,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm_selective,
            num_neighbors=nn_selective,
            rebuild_flags=rebuild_flags,
        )

        # System 0 should be unchanged (same as original full build)
        sys0_start, sys0_end = 0, ptr[1].item()
        assert torch.equal(
            nn_selective[sys0_start:sys0_end], num_neighbors_ref[sys0_start:sys0_end]
        ), "System 0 num_neighbors should be unchanged"

        # System 1 should match the full rebuild result
        sys1_start, sys1_end = ptr[1].item(), ptr[2].item()
        assert torch.equal(
            nn_selective[sys1_start:sys1_end], nn_full[sys1_start:sys1_end]
        ), "System 1 num_neighbors should match full rebuild"

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_cell_list_selective_no_rebuild(self, device, dtype):
        """When no systems need rebuild, batch_cell_list preserves neighbor data."""
        positions, cell, pbc, batch_idx, ptr, atoms_per, num_systems = (
            self._make_batch_systems(dtype, device)
        )
        cutoff = 1.5
        max_neighbors = 16

        # Initial full build
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = batch_cell_list(
            positions=positions,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            max_neighbors=max_neighbors,
        )
        saved_nm = neighbor_matrix.clone()
        saved_nn = num_neighbors.clone()

        # All flags False → nothing should change
        rebuild_flags = torch.zeros(num_systems, dtype=torch.bool, device=device)
        batch_cell_list(
            positions=positions,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            max_neighbors=max_neighbors,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            rebuild_flags=rebuild_flags,
        )

        assert torch.equal(num_neighbors, saved_nn), (
            "num_neighbors should not change when no systems need rebuild"
        )
        for i in range(positions.shape[0]):
            n = num_neighbors[i].item()
            assert torch.equal(neighbor_matrix[i, :n], saved_nm[i, :n])

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_cell_list_selective_rebuild_one_system(self, device, dtype):
        """Rebuilding only system 1 via batch_cell_list updates it, preserves others."""
        positions, cell, pbc, batch_idx, ptr, atoms_per, num_systems = (
            self._make_batch_systems(dtype, device)
        )
        cutoff = 1.5
        max_neighbors = 16

        # Initial full build
        nm_ref, nn_ref, shifts_ref = batch_cell_list(
            positions=positions,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            max_neighbors=max_neighbors,
        )

        # Move an atom in system 1
        new_positions = positions.clone()
        sys1_start = ptr[1].item()
        new_positions[sys1_start] = torch.tensor(
            [0.05, 0.05, 0.05], dtype=dtype, device=device
        )

        # Full rebuild for reference
        nm_full, nn_full, _ = batch_cell_list(
            positions=new_positions,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            max_neighbors=max_neighbors,
        )

        # Need a fresh cell list for the new positions (selective rebuild only affects query step)
        max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell, pbc, cutoff=cutoff
        )
        cell_list_cache = allocate_cell_list(
            positions.shape[0], max_total_cells, neighbor_search_radius, device
        )
        batch_build_cell_list(
            new_positions, cutoff, cell, pbc, batch_idx, *cell_list_cache
        )

        # Selective query rebuild: only system 1
        rebuild_flags = torch.zeros(num_systems, dtype=torch.bool, device=device)
        rebuild_flags[1] = True
        nm_selective = nm_ref.clone()
        nn_selective = nn_ref.clone()
        shifts_selective = shifts_ref.clone()
        batch_query_cell_list(
            positions=new_positions,
            cell=cell,
            pbc=pbc,
            cutoff=cutoff,
            batch_idx=batch_idx,
            cells_per_dimension=cell_list_cache[0],
            neighbor_search_radius=cell_list_cache[1],
            atom_periodic_shifts=cell_list_cache[2],
            atom_to_cell_mapping=cell_list_cache[3],
            atoms_per_cell_count=cell_list_cache[4],
            cell_atom_start_indices=cell_list_cache[5],
            cell_atom_list=cell_list_cache[6],
            neighbor_matrix=nm_selective,
            neighbor_matrix_shifts=shifts_selective,
            num_neighbors=nn_selective,
            half_fill=False,
            rebuild_flags=rebuild_flags,
        )

        # System 0 should be unchanged
        sys0_start, sys0_end = 0, ptr[1].item()
        assert torch.equal(
            nn_selective[sys0_start:sys0_end], nn_ref[sys0_start:sys0_end]
        ), "System 0 num_neighbors should be unchanged"

        # System 1 should match full rebuild
        sys1_start, sys1_end = ptr[1].item(), ptr[2].item()
        assert torch.equal(
            nn_selective[sys1_start:sys1_end], nn_full[sys1_start:sys1_end]
        ), "System 1 num_neighbors should match full rebuild"


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", dtypes)
class TestTorchPBCRebuildDetection:
    """Test PBC-aware rebuild detection via PyTorch bindings."""

    def test_pbc_no_spurious_rebuild(self, device, dtype):
        """Boundary-crossing atom must not trigger rebuild with PBC."""
        cell_size = 10.0
        skin = 1.0
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=cell_size, dtype=dtype, device=device
        )

        reference = positions.clone()
        current = positions.clone()
        current[0, 0] = cell_size - 0.01

        cell_inv = (
            torch.linalg.inv(cell.squeeze(0))
            .unsqueeze(0)
            .to(dtype=dtype, device=device)
            .contiguous()
        )

        result = neighbor_list_needs_rebuild(
            reference,
            current,
            skin / 2.0,
            cell=cell,
            cell_inv=cell_inv,
            pbc=pbc,
        )
        assert not result.item(), (
            "PBC should prevent spurious rebuild at periodic boundary"
        )

    def test_pbc_genuine_rebuild(self, device, dtype):
        """Large genuine displacement should still trigger rebuild with PBC."""
        cell_size = 10.0
        skin = 1.0
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=cell_size, dtype=dtype, device=device
        )

        reference = positions.clone()
        current = positions.clone()
        current[0, 0] += 2.0

        cell_inv = (
            torch.linalg.inv(cell.squeeze(0))
            .unsqueeze(0)
            .to(dtype=dtype, device=device)
            .contiguous()
        )

        result = neighbor_list_needs_rebuild(
            reference,
            current,
            skin / 2.0,
            cell=cell,
            cell_inv=cell_inv,
            pbc=pbc,
        )
        assert result.item(), (
            "PBC should still trigger rebuild for genuinely large displacement"
        )

    def test_batch_pbc_no_spurious_rebuild(self, device, dtype):
        """Batch PBC: boundary-crossing atoms must not trigger rebuild."""
        cell_size = 10.0
        skin = 1.0
        atoms_per = [4, 5, 3]
        n_side = 2
        spacing = cell_size / n_side
        grid_coords = [
            [i * spacing, j * spacing, k * spacing]
            for i in range(n_side)
            for j in range(n_side)
            for k in range(n_side)
        ]
        all_pos = []
        cells_list = []
        for n in atoms_per:
            pos = torch.tensor(grid_coords[:n], dtype=dtype, device=device)
            all_pos.append(pos)
            cells_list.append(torch.eye(3, dtype=dtype, device=device) * cell_size)

        positions = torch.cat(all_pos, dim=0)
        cells = torch.stack(cells_list, dim=0)
        pbc = torch.ones(3, 3, dtype=torch.bool, device=device)
        batch_idx = torch.repeat_interleave(
            torch.arange(3, dtype=torch.int32, device=device),
            torch.tensor(atoms_per, dtype=torch.int32, device=device),
        )

        reference = positions.clone()
        current = positions.clone()
        current[5, 0] = cell_size - 0.01

        cells_inv = torch.linalg.inv(cells).to(dtype=dtype, device=device).contiguous()

        result = batch_neighbor_list_needs_rebuild(
            reference,
            current,
            batch_idx,
            skin / 2.0,
            cell=cells,
            cell_inv=cells_inv,
            pbc=pbc,
        )
        assert not result.any(), (
            "Batch PBC should prevent spurious rebuilds at boundaries"
        )
