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
    check_batch_cell_list_rebuild,
    check_batch_neighbor_list_rebuild,
    check_cell_list_rebuild,
    check_neighbor_list_rebuild,
)
from nvalchemiops.torch.neighbors.cell_list import estimate_cell_list_sizes
from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

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
            str(device),
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
            str(device),
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
            str(device),
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
            str(device),
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
            str(device),
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
            str(device),
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
            str(device),
        )

        # Should not need rebuild
        assert not rebuild_needed.item(), "Empty system should not need rebuild"


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", dtypes)
class TestBatchRebuildDetectionWpLaunchers:
    """Test batch warp launchers for rebuild detection."""

    def _make_batch_positions(self, device, dtype):
        """Create three small systems concatenated into a batch."""
        # System 0: 4 atoms, system 1: 5 atoms, system 2: 3 atoms
        atoms_per = [4, 5, 3]
        torch.manual_seed(0)
        all_pos = []
        cells = []
        for n in atoms_per:
            pos = torch.rand(n, 3, dtype=dtype, device=device) * 3.0
            all_pos.append(pos)
            cells.append(torch.eye(3, dtype=dtype, device=device) * 4.0)
        positions = torch.cat(all_pos, dim=0)
        cell = torch.stack(cells, dim=0)  # (3, 3, 3)
        pbc = torch.zeros(3, 3, dtype=torch.bool, device=device)
        ptr = torch.tensor([0, 4, 9, 12], dtype=torch.int32, device=device)
        # build batch_idx from ptr
        batch_idx = torch.repeat_interleave(
            torch.arange(3, dtype=torch.int32, device=device),
            torch.tensor(atoms_per, dtype=torch.int32, device=device),
        )
        return positions, cell, pbc, batch_idx, ptr, atoms_per

    def test_check_batch_neighbor_list_rebuild_no_movement(self, device, dtype):
        """All systems: no movement → all rebuild_flags should be False."""
        positions, _, _, batch_idx, _, _ = self._make_batch_positions(device, dtype)

        reference_positions = positions.clone()
        current_positions = positions.clone()

        rebuild_flags = torch.zeros(3, dtype=torch.bool, device=device)

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_reference = wp.from_torch(
            reference_positions, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_current = wp.from_torch(
            current_positions, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)
        wp_flags = wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True)

        check_batch_neighbor_list_rebuild(
            reference_positions=wp_reference,
            current_positions=wp_current,
            batch_idx=wp_batch_idx,
            skin_distance_threshold=0.5,
            rebuild_flags=wp_flags,
            wp_dtype=wp_dtype,
            device=str(device),
        )

        assert rebuild_flags.shape == (3,)
        assert not rebuild_flags.any(), (
            "No systems should need rebuild with no movement"
        )

    def test_check_batch_neighbor_list_rebuild_one_system(self, device, dtype):
        """Only system 1 moves beyond skin distance → only flag[1] should be True."""
        positions, _, _, batch_idx, ptr, _ = self._make_batch_positions(device, dtype)

        reference_positions = positions.clone()
        current_positions = positions.clone()
        # Move an atom in system 1 (atoms 4..8)
        current_positions[5] += 2.0

        rebuild_flags = torch.zeros(3, dtype=torch.bool, device=device)

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_reference = wp.from_torch(
            reference_positions, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_current = wp.from_torch(
            current_positions, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)
        wp_flags = wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True)

        check_batch_neighbor_list_rebuild(
            reference_positions=wp_reference,
            current_positions=wp_current,
            batch_idx=wp_batch_idx,
            skin_distance_threshold=0.5,
            rebuild_flags=wp_flags,
            wp_dtype=wp_dtype,
            device=str(device),
        )

        assert not rebuild_flags[0], "System 0 should not need rebuild"
        assert rebuild_flags[1], "System 1 should need rebuild"
        assert not rebuild_flags[2], "System 2 should not need rebuild"

    def test_check_batch_neighbor_list_rebuild_all_systems(self, device, dtype):
        """All systems have atoms moving beyond skin → all flags should be True."""
        positions, _, _, batch_idx, ptr, _ = self._make_batch_positions(device, dtype)

        reference_positions = positions.clone()
        current_positions = positions.clone()
        # Move one atom per system
        current_positions[0] += 2.0  # system 0
        current_positions[5] += 2.0  # system 1
        current_positions[10] += 2.0  # system 2

        rebuild_flags = torch.zeros(3, dtype=torch.bool, device=device)

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_reference = wp.from_torch(
            reference_positions, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_current = wp.from_torch(
            current_positions, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)
        wp_flags = wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True)

        check_batch_neighbor_list_rebuild(
            reference_positions=wp_reference,
            current_positions=wp_current,
            batch_idx=wp_batch_idx,
            skin_distance_threshold=0.5,
            rebuild_flags=wp_flags,
            wp_dtype=wp_dtype,
            device=str(device),
        )

        assert rebuild_flags.all(), "All systems should need rebuild"

    def test_check_batch_neighbor_list_rebuild_empty(self, device, dtype):
        """Empty batch: all rebuild_flags remain False."""
        reference_positions = torch.empty((0, 3), dtype=dtype, device=device)
        current_positions = torch.empty((0, 3), dtype=dtype, device=device)
        batch_idx = torch.empty(0, dtype=torch.int32, device=device)
        rebuild_flags = torch.zeros(2, dtype=torch.bool, device=device)

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_reference = wp.from_torch(
            reference_positions, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_current = wp.from_torch(
            current_positions, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)
        wp_flags = wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True)

        check_batch_neighbor_list_rebuild(
            reference_positions=wp_reference,
            current_positions=wp_current,
            batch_idx=wp_batch_idx,
            skin_distance_threshold=0.5,
            rebuild_flags=wp_flags,
            wp_dtype=wp_dtype,
            device=str(device),
        )

        assert not rebuild_flags.any(), "Empty batch should not set any rebuild flags"

    def test_check_batch_cell_list_rebuild_no_movement(self, device, dtype):
        """All systems: no movement → all rebuild_flags should be False."""
        from nvalchemiops.torch.neighbors.batch_cell_list import (
            batch_build_cell_list,
            estimate_batch_cell_list_sizes,
        )
        from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list

        positions, cell, pbc, batch_idx, ptr, _ = self._make_batch_positions(
            device, dtype
        )

        num_systems = 3
        max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell, pbc, cutoff=1.0
        )
        (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = allocate_cell_list(
            positions.shape[0], max_total_cells, neighbor_search_radius, device
        )

        batch_build_cell_list(
            positions,
            1.0,
            cell,
            pbc,
            batch_idx,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )

        rebuild_flags = torch.zeros(num_systems, dtype=torch.bool, device=device)

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
        wp_atom_to_cell = wp.from_torch(
            atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
        )
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)
        wp_cpd = wp.from_torch(cells_per_dimension, dtype=wp.vec3i, return_ctype=True)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
        wp_flags = wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True)

        check_batch_cell_list_rebuild(
            current_positions=wp_positions,
            atom_to_cell_mapping=wp_atom_to_cell,
            batch_idx=wp_batch_idx,
            cells_per_dimension=wp_cpd,
            cell=wp_cell,
            pbc=wp_pbc,
            rebuild_flags=wp_flags,
            wp_dtype=wp_dtype,
            device=str(device),
        )

        assert not rebuild_flags.any(), (
            "No systems should need rebuild with no movement"
        )

    def test_check_batch_cell_list_rebuild_one_system_moves(self, device, dtype):
        """Only system 0 has an atom cross a cell boundary → only flag[0] should be True."""
        from nvalchemiops.torch.neighbors.batch_cell_list import (
            batch_build_cell_list,
            estimate_batch_cell_list_sizes,
        )
        from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list

        positions, cell, pbc, batch_idx, ptr, _ = self._make_batch_positions(
            device, dtype
        )
        num_systems = 3
        max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell, pbc, cutoff=1.0
        )
        (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = allocate_cell_list(
            positions.shape[0], max_total_cells, neighbor_search_radius, device
        )

        batch_build_cell_list(
            positions,
            1.0,
            cell,
            pbc,
            batch_idx,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )

        # Move atom 0 (in system 0) across a full cell
        new_positions = positions.clone()
        new_positions[0] += 1.5

        rebuild_flags = torch.zeros(num_systems, dtype=torch.bool, device=device)

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        wp_new_positions = wp.from_torch(
            new_positions, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
        wp_atom_to_cell = wp.from_torch(
            atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
        )
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)
        wp_cpd = wp.from_torch(cells_per_dimension, dtype=wp.vec3i, return_ctype=True)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
        wp_flags = wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True)

        check_batch_cell_list_rebuild(
            current_positions=wp_new_positions,
            atom_to_cell_mapping=wp_atom_to_cell,
            batch_idx=wp_batch_idx,
            cells_per_dimension=wp_cpd,
            cell=wp_cell,
            pbc=wp_pbc,
            rebuild_flags=wp_flags,
            wp_dtype=wp_dtype,
            device=str(device),
        )

        assert rebuild_flags[0], (
            "System 0 should need rebuild (atom crossed cell boundary)"
        )
        assert not rebuild_flags[1], "System 1 should not need rebuild"
        assert not rebuild_flags[2], "System 2 should not need rebuild"


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", dtypes)
class TestPBCRebuildDetection:
    """Tests for PBC-aware rebuild detection kernels."""

    def test_pbc_no_spurious_rebuild_at_boundary(self, device, dtype):
        """Atom wrapped from x=0 to x=L-eps must NOT trigger rebuild with PBC."""
        cell_size = 10.0
        skin = 1.0
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=cell_size, dtype=dtype, device=device
        )
        pbc = pbc.squeeze(0)

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        reference_positions = positions.clone()
        current_positions = positions.clone()
        # Simulate a boundary crossing: atom was near x=0, now wrapped to x=L-0.01
        current_positions[0, 0] = cell_size - 0.01

        wp_reference = wp.from_torch(reference_positions, dtype=wp_vec_dtype)
        wp_current = wp.from_torch(current_positions, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)

        cell_inv = (
            torch.linalg.inv(cell.squeeze(0))
            .unsqueeze(0)
            .to(dtype=dtype, device=device)
            .contiguous()
        )
        wp_cell_inv = wp.from_torch(cell_inv, dtype=wp_mat_dtype)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool)

        rebuild_needed = torch.zeros(1, dtype=torch.bool, device=device)
        wp_rebuild_flag = wp.from_torch(rebuild_needed, dtype=wp.bool)

        check_neighbor_list_rebuild(
            reference_positions=wp_reference,
            current_positions=wp_current,
            skin_distance_threshold=skin / 2.0,
            rebuild_flag=wp_rebuild_flag,
            wp_dtype=wp_dtype,
            device=str(device),
            cell=wp_cell,
            cell_inv=wp_cell_inv,
            pbc=wp_pbc,
        )

        # PBC displacement is ~0.01, well below skin/2=0.5 → no rebuild
        assert not rebuild_needed.item(), (
            "PBC should prevent spurious rebuild at periodic boundary"
        )

    def test_pbc_genuine_large_displacement_triggers_rebuild(self, device, dtype):
        """Atom truly moving beyond skin should still trigger rebuild with PBC."""
        cell_size = 10.0
        skin = 1.0
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=cell_size, dtype=dtype, device=device
        )
        pbc = pbc.squeeze(0)

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        reference_positions = positions.clone()
        current_positions = positions.clone()
        current_positions[0, 0] += 2.0  # Move 2 Å — well beyond skin/2

        wp_reference = wp.from_torch(reference_positions, dtype=wp_vec_dtype)
        wp_current = wp.from_torch(current_positions, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)

        cell_inv = (
            torch.linalg.inv(cell.squeeze(0))
            .unsqueeze(0)
            .to(dtype=dtype, device=device)
            .contiguous()
        )
        wp_cell_inv = wp.from_torch(cell_inv, dtype=wp_mat_dtype)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool)

        rebuild_needed = torch.zeros(1, dtype=torch.bool, device=device)
        wp_rebuild_flag = wp.from_torch(rebuild_needed, dtype=wp.bool)

        check_neighbor_list_rebuild(
            reference_positions=wp_reference,
            current_positions=wp_current,
            skin_distance_threshold=skin / 2.0,
            rebuild_flag=wp_rebuild_flag,
            wp_dtype=wp_dtype,
            device=str(device),
            cell=wp_cell,
            cell_inv=wp_cell_inv,
            pbc=wp_pbc,
        )

        assert rebuild_needed.item(), (
            "PBC should still trigger rebuild for genuinely large displacement"
        )

    def test_pbc_non_periodic_dim_uses_raw_displacement(self, device, dtype):
        """Non-periodic dimensions should use raw displacement, not PBC wrapping."""
        cell_size = 10.0
        skin = 1.0
        positions, cell, _ = create_simple_cubic_system(
            num_atoms=8, cell_size=cell_size, dtype=dtype, device=device
        )
        # Only x is periodic, y and z are not
        pbc = torch.tensor([True, False, False], dtype=torch.bool, device=device)

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        reference_positions = positions.clone()
        current_positions = positions.clone()
        # Large jump in non-periodic y dimension
        current_positions[0, 1] = cell_size - 0.01

        wp_reference = wp.from_torch(reference_positions, dtype=wp_vec_dtype)
        wp_current = wp.from_torch(current_positions, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)

        cell_inv = (
            torch.linalg.inv(cell.squeeze(0))
            .unsqueeze(0)
            .to(dtype=dtype, device=device)
            .contiguous()
        )
        wp_cell_inv = wp.from_torch(cell_inv, dtype=wp_mat_dtype)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool)

        rebuild_needed = torch.zeros(1, dtype=torch.bool, device=device)
        wp_rebuild_flag = wp.from_torch(rebuild_needed, dtype=wp.bool)

        check_neighbor_list_rebuild(
            reference_positions=wp_reference,
            current_positions=wp_current,
            skin_distance_threshold=skin / 2.0,
            rebuild_flag=wp_rebuild_flag,
            wp_dtype=wp_dtype,
            device=str(device),
            cell=wp_cell,
            cell_inv=wp_cell_inv,
            pbc=wp_pbc,
        )

        # Non-periodic y dimension: raw displacement is ~10 Å, triggers rebuild
        assert rebuild_needed.item(), (
            "Non-periodic dimension should use raw displacement and trigger rebuild"
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

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        reference_positions = positions.clone()
        current_positions = positions.clone()
        # Simulate boundary wrapping in system 1 (atom index 5)
        current_positions[5, 0] = cell_size - 0.01

        cells_inv = torch.linalg.inv(cells).to(dtype=dtype, device=device).contiguous()

        rebuild_flags = torch.zeros(3, dtype=torch.bool, device=device)

        wp_reference = wp.from_torch(
            reference_positions, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_current = wp.from_torch(
            current_positions, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)
        wp_cells = wp.from_torch(cells, dtype=wp_mat_dtype, return_ctype=True)
        wp_cells_inv = wp.from_torch(cells_inv, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
        wp_flags = wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True)

        check_batch_neighbor_list_rebuild(
            reference_positions=wp_reference,
            current_positions=wp_current,
            batch_idx=wp_batch_idx,
            skin_distance_threshold=skin / 2.0,
            rebuild_flags=wp_flags,
            wp_dtype=wp_dtype,
            device=str(device),
            cell=wp_cells,
            cell_inv=wp_cells_inv,
            pbc=wp_pbc,
        )

        # PBC displacement for atom 5 is small → no rebuild for any system
        assert not rebuild_flags.any(), (
            "Batch PBC should prevent spurious rebuilds at boundaries"
        )
