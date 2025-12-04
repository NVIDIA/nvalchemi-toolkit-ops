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

"""Tests for batch cell list kernel functions."""

import pytest
import torch
import warp as wp

from nvalchemiops.neighborlist.batch_cell_list import (
    _batch_cell_list_bin_atoms_overload,
    _batch_cell_list_build_neighbor_matrix_overload,
    _batch_cell_list_construct_bin_size_overload,
    _batch_cell_list_count_atoms_per_bin_overload,
    batch_build_cell_list,
    estimate_batch_cell_list_sizes,
)
from nvalchemiops.neighborlist.neighbor_utils import allocate_cell_list
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

from .test_utils import create_batch_systems

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda:0")
dtypes = [torch.float32, torch.float64]


class TestBatchCellListKernels:
    """Test individual batch cell list kernel functions."""

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_batch_construct_bin_size(self, device, dtype):
        """Test _batch_cell_list_construct_bin_size kernel."""
        # Create batch of systems
        _, cell, pbc, _ = create_batch_systems(
            num_systems=3,
            atoms_per_system=[5, 8, 6],
            cell_sizes=[2.0, 3.0, 2.5],
            dtype=dtype,
            device=device,
        )

        cutoff = 1.0
        max_nbins = 1000000
        num_systems = 3

        # Convert to warp types
        wp_dtype = get_wp_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)

        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(
            pbc.to(dtype=torch.bool), dtype=wp.bool, return_ctype=True
        )
        wp_cutoff = wp_dtype(cutoff)

        # Output arrays
        cells_per_dimension = torch.zeros(
            num_systems, 3, dtype=torch.int32, device=device
        )
        wp_cells_per_dimension = wp.from_torch(
            cells_per_dimension, dtype=wp.vec3i, return_ctype=True
        )

        # Launch kernel
        wp.launch(
            _batch_cell_list_construct_bin_size_overload[wp_dtype],
            dim=num_systems,
            device=wp_device,
            inputs=(
                wp_cell,
                wp_pbc,
                wp_cells_per_dimension,
                wp_cutoff,
                max_nbins,
            ),
        )

        # Check results for each system
        for sys_idx in range(num_systems):
            sys_cell_counts = cells_per_dimension[sys_idx]

            assert torch.all(sys_cell_counts > 0), (
                f"System {sys_idx}: All cell counts should be positive"
            )

            # Total cells should not exceed max_nbins
            total_cells = sys_cell_counts.prod().item()
            assert total_cells <= max_nbins, (
                f"System {sys_idx}: Total cells {total_cells} exceeds max_nbins {max_nbins}"
            )

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_batch_cell_list_count_atoms_per_bin(self, device, dtype):
        """Test _batch_cell_list_count_atoms_per_bin kernel."""
        # Create small batch for easier testing
        positions, cell, pbc, ptr = create_batch_systems(
            num_systems=2,
            atoms_per_system=[4, 3],
            cell_sizes=[2.0, 2.5],
            dtype=dtype,
            device=device,
        )
        idx = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device)
        num_systems = 2
        total_atoms = positions.shape[0]

        # Estimate cell list sizes
        cells_per_dimension = torch.tensor(
            [[2, 2, 2], [2, 2, 2]], dtype=torch.int32, device=device
        )

        # Cell offsets for each system
        cells_per_system = cells_per_dimension.prod(dim=1)
        cell_offsets = torch.zeros(num_systems + 1, dtype=torch.int32, device=device)
        torch.cumsum(cells_per_system, dim=0, out=cell_offsets[1:])
        total_cells = cell_offsets[-1].item()

        # Convert to warp types
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)

        wp_dtype = get_wp_dtype(dtype)
        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(
            pbc.to(dtype=torch.bool), dtype=wp.bool, return_ctype=True
        )
        wp_idx = wp.from_torch(
            idx.to(dtype=torch.int32), dtype=wp.int32, return_ctype=True
        )
        wp_cells_per_dimension = wp.from_torch(
            cells_per_dimension, dtype=wp.vec3i, return_ctype=True
        )
        wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32, return_ctype=True)

        # Output arrays
        atoms_per_cell_count = torch.zeros(
            total_cells, dtype=torch.int32, device=device
        )
        atom_periodic_shifts = torch.zeros(
            total_atoms, 3, dtype=torch.int32, device=device
        )
        wp_atoms_per_cell_count = wp.from_torch(
            atoms_per_cell_count, dtype=wp.int32, return_ctype=True
        )
        wp_atom_periodic_shifts = wp.from_torch(
            atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
        )

        # Launch kernel
        wp.launch(
            _batch_cell_list_count_atoms_per_bin_overload[wp_dtype],
            dim=total_atoms,
            device=wp_device,
            inputs=(
                wp_positions,
                wp_cell,
                wp_pbc,
                wp_idx,
                wp_cells_per_dimension,
                wp_cell_offsets,
                wp_atoms_per_cell_count,
                wp_atom_periodic_shifts,
            ),
        )

        # Check results
        total_atoms_counted = atoms_per_cell_count.sum().item()
        assert total_atoms_counted == total_atoms, (
            f"Expected {total_atoms} atoms counted, got {total_atoms_counted}"
        )

        # Check that atoms are binned correctly per system
        for sys_idx in range(num_systems):
            sys_start = cell_offsets[sys_idx].item()
            sys_end = cell_offsets[sys_idx + 1].item()
            sys_atom_count = atoms_per_cell_count[sys_start:sys_end].sum().item()
            expected_atoms = ptr[sys_idx + 1].item() - ptr[sys_idx].item()
            assert sys_atom_count == expected_atoms, (
                f"System {sys_idx}: expected {expected_atoms} atoms, got {sys_atom_count}"
            )

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_batch_bin_atoms(self, device, dtype):
        """Test _batch_cell_list_bin_atoms kernel."""
        # Create batch system
        positions, cell, pbc, ptr = create_batch_systems(
            num_systems=2,
            atoms_per_system=[3, 4],
            cell_sizes=[2.0, 2.5],
            dtype=dtype,
            device=device,
        )
        idx = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device)

        num_systems = ptr.shape[0] - 1
        total_atoms = positions.shape[0]

        # Setup cell structure
        cells_per_dimension = torch.tensor(
            [[2, 2, 2], [2, 2, 2]], dtype=torch.int32, device=device
        )

        cells_per_system = cells_per_dimension.prod(dim=1)
        cell_offsets = torch.zeros(num_systems + 1, dtype=torch.int32, device=device)
        torch.cumsum(cells_per_system, dim=0, out=cell_offsets[1:])
        total_cells = cell_offsets[-1].item()

        # First count atoms
        atoms_per_cell_count = torch.zeros(
            total_cells, dtype=torch.int32, device=device
        )
        atom_to_cell_mapping = torch.zeros(
            total_atoms, 3, dtype=torch.int32, device=device
        )
        atom_periodic_shifts = torch.zeros(
            total_atoms, 3, dtype=torch.int32, device=device
        )
        cell_atom_start_indices = torch.zeros(
            total_cells, dtype=torch.int32, device=device
        )
        cell_atom_list = torch.zeros(total_atoms, dtype=torch.int32, device=device)

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(
            pbc.to(dtype=torch.bool), dtype=wp.bool, return_ctype=True
        )
        wp_idx = wp.from_torch(
            idx.to(dtype=torch.int32), dtype=wp.int32, return_ctype=True
        )
        wp_cells_per_dimension = wp.from_torch(
            cells_per_dimension, dtype=wp.vec3i, return_ctype=True
        )
        wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32, return_ctype=True)
        wp_atoms_per_cell_count = wp.from_torch(
            atoms_per_cell_count, dtype=wp.int32, return_ctype=True
        )
        wp_atom_periodic_shifts = wp.from_torch(
            atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
        )

        wp.launch(
            _batch_cell_list_count_atoms_per_bin_overload[wp_dtype],
            dim=total_atoms,
            device=wp_device,
            inputs=(
                wp_positions,
                wp_cell,
                wp_pbc,
                wp_idx,
                wp_cells_per_dimension,
                wp_cell_offsets,
                wp_atoms_per_cell_count,
                wp_atom_periodic_shifts,
            ),
        )

        # Compute cell offsets for atom storage
        atom_cell_offsets = torch.zeros(
            total_cells + 1, dtype=torch.int32, device=device
        )
        torch.cumsum(atoms_per_cell_count, dim=0, out=atom_cell_offsets[1:])

        # Allocate atom indices storage
        wp_atom_to_cell_mapping = wp.from_torch(
            atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
        )
        wp_cell_atom_start_indices = wp.from_torch(
            cell_atom_start_indices, dtype=wp.int32, return_ctype=True
        )
        wp_cell_atom_list = wp.from_torch(
            cell_atom_list, dtype=wp.int32, return_ctype=True
        )

        # Reset counts for binning
        atoms_per_cell_count.zero_()

        # Launch bin_atoms kernel
        wp.launch(
            _batch_cell_list_bin_atoms_overload[wp_dtype],
            dim=total_atoms,
            device=wp_device,
            inputs=(
                wp_positions,
                wp_cell,
                wp_pbc,
                wp_idx,
                wp_cells_per_dimension,
                wp_cell_offsets,
                wp_atom_to_cell_mapping,
                wp_atoms_per_cell_count,
                wp_cell_atom_start_indices,
                wp_cell_atom_list,
            ),
        )

        # Check that all atoms are binned
        total_binned = atoms_per_cell_count.sum().item()
        assert total_binned == total_atoms, (
            f"Expected {total_atoms} atoms binned, got {total_binned}"
        )

        # Check atom indices are valid
        valid_indices = (cell_atom_list >= 0) & (cell_atom_list < total_atoms)
        assert torch.all(valid_indices[:total_binned]), (
            "All atom indices should be valid"
        )

        # Check that each atom is assigned to a valid cell
        for atom_idx in range(total_atoms):
            cell_idx = atom_to_cell_mapping[atom_idx]
            assert torch.all(cell_idx >= 0), (
                f"Atom {atom_idx}: cell indices should be non-negative"
            )

            # Find which system this atom belongs to
            sys_idx = torch.searchsorted(ptr[1:], atom_idx, right=False).item()
            sys_cell_counts = cells_per_dimension[sys_idx]
            assert torch.all(cell_idx < sys_cell_counts), (
                f"Atom {atom_idx}: cell indices should be within system bounds"
            )

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_batch_build_neighbor_matrix(self, device, dtype):
        """Test _batch_cell_list_build_neighbor_matrix kernel."""
        # Create batch system
        positions, cell, pbc, ptr = create_batch_systems(
            num_systems=2,
            atoms_per_system=[3, 4],
            cell_sizes=[2.0, 2.5],
            dtype=dtype,
            device=device,
        )
        idx = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device)

        num_systems = ptr.shape[0] - 1
        total_atoms = positions.shape[0]
        cutoff = 1.0

        max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell, pbc, cutoff
        )
        cell_list_cache = allocate_cell_list(
            total_atoms,
            max_total_cells,
            neighbor_search_radius,
            device,
        )

        # Build cell list
        batch_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            idx,
            *cell_list_cache,
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

        # Output arrays - neighbor matrix format
        max_neighbors = 100
        neighbor_matrix = torch.full(
            (total_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (total_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(total_atoms, dtype=torch.int32, device=device)

        # Get warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)
        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
        wp_cutoff = wp_dtype(cutoff)
        wp_idx = wp.from_torch(
            idx.to(dtype=torch.int32), dtype=wp.int32, return_ctype=True
        )

        wp_cells_per_dimension = wp.from_torch(
            cells_per_dimension, dtype=wp.vec3i, return_ctype=True
        )
        wp_neighbor_search_radius = wp.from_torch(
            neighbor_search_radius, dtype=wp.vec3i, return_ctype=True
        )

        wp_atom_periodic_shifts = wp.from_torch(
            atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
        )
        wp_atom_to_cell_mapping = wp.from_torch(
            atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
        )
        wp_atoms_per_cell_count = wp.from_torch(
            atoms_per_cell_count, dtype=wp.int32, return_ctype=True
        )
        cell_offsets = torch.zeros(num_systems + 1, dtype=torch.int32, device=device)
        torch.cumsum(cells_per_dimension.prod(dim=1), dim=0, out=cell_offsets[1:])
        wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32, return_ctype=True)
        wp_cell_atom_start_indices = wp.from_torch(
            cell_atom_start_indices, dtype=wp.int32, return_ctype=True
        )
        wp_cell_atom_list = wp.from_torch(
            cell_atom_list, dtype=wp.int32, return_ctype=True
        )

        wp_neighbor_matrix = wp.from_torch(
            neighbor_matrix, dtype=wp.int32, return_ctype=True
        )
        wp_neighbor_matrix_shifts = wp.from_torch(
            neighbor_matrix_shifts, dtype=wp.vec3i, return_ctype=True
        )
        wp_num_neighbors = wp.from_torch(
            num_neighbors, dtype=wp.int32, return_ctype=True
        )

        # Build neighbor matrix
        wp.launch(
            _batch_cell_list_build_neighbor_matrix_overload[wp_dtype],
            dim=total_atoms,
            inputs=(
                wp_positions,
                wp_cell,
                wp_pbc,
                wp_idx,
                wp_cutoff,
                wp_cells_per_dimension,
                wp_neighbor_search_radius,
                wp_atom_periodic_shifts,
                wp_atom_to_cell_mapping,
                wp_atoms_per_cell_count,
                wp_cell_atom_start_indices,
                wp_cell_atom_list,
                wp_cell_offsets,
                wp_neighbor_matrix,
                wp_neighbor_matrix_shifts,
                wp_num_neighbors,
                False,
            ),
            device=wp_device,
        )

        # Check that neighbor counts are reasonable
        assert torch.all(num_neighbors >= 0), (
            "All neighbor counts should be non-negative"
        )

        # Check that pairs are from the same system
        for atom_idx in range(total_atoms):
            n_neigh = num_neighbors[atom_idx].item()
            sys_i = idx[atom_idx].item()
            for neigh_idx in range(min(n_neigh, max_neighbors)):
                atom_j = neighbor_matrix[atom_idx, neigh_idx].item()
                if atom_j == -1:
                    break

                sys_j = idx[atom_j].item()
                assert sys_i == sys_j, (
                    f"Atoms {atom_idx} and {atom_j} should be from same system"
                )
