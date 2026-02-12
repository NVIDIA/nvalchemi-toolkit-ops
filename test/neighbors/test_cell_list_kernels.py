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

"""Tests for individual cell list kernel functions."""

import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.cell_list import (
    _cell_list_bin_atoms_overload,
    _cell_list_build_neighbor_matrix_overload,
    _cell_list_compute_cell_offsets,
    _cell_list_construct_bin_size_overload,
    _cell_list_count_atoms_per_bin_overload,
    build_cell_list,
    query_cell_list,
)
from nvalchemiops.torch.neighbors.cell_list import (
    build_cell_list as torch_build_cell_list,
)
from nvalchemiops.torch.neighbors.cell_list import (
    estimate_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

from .test_utils import create_random_system, create_simple_cubic_system

dtypes = [torch.float32, torch.float64]


@pytest.mark.parametrize("dtype", dtypes)
class TestCellListKernels:
    """Test individual cell list kernel functions."""

    def test_construct_bin_size(self, device, dtype):
        """Test _cell_list_construct_bin_size kernel."""
        # Create test system
        _, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=4.0, dtype=dtype, device=device
        )
        cutoff = 1.5
        max_nbins = 1000000
        pbc = pbc.squeeze(0)
        # Convert to warp types
        wp_dtype = get_wp_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)

        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)

        # Output arrays
        cell_counts = torch.zeros(3, dtype=torch.int32, device=device)
        wp_cell_counts = wp.from_torch(cell_counts, dtype=wp.vec3i, return_ctype=True)

        # Launch kernel
        wp.launch(
            _cell_list_construct_bin_size_overload[wp_dtype],
            dim=1,
            device=wp_device,
            inputs=(
                wp_cell,
                wp_pbc,
                wp_cell_counts,
                wp_dtype(cutoff),
                max_nbins,
            ),
        )

        # Check results
        assert torch.all(cell_counts > 0), "All cell counts should be positive"

        # Total cells should not exceed max_nbins
        total_cells = cell_counts.prod().item()
        assert total_cells <= max_nbins, (
            f"Total cells {total_cells} exceeds max_nbins {max_nbins}"
        )

    def test_count_atoms_per_bin(self, device, dtype):
        """Test _cell_list_count_atoms_per_bin kernel."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=3.0, dtype=dtype, device=device
        )

        # Manual cell counts for testing (simple 2x2x2 grid)
        cell_counts = torch.tensor([2, 2, 2], dtype=torch.int32, device=device)
        total_cells = cell_counts.prod().item()
        pbc = pbc.squeeze(0)

        # Convert to warp types
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
        wp_cell_counts = wp.from_torch(cell_counts, dtype=wp.int32, return_ctype=True)

        # Output arrays
        cell_atom_counts = torch.zeros(total_cells, dtype=torch.int32, device=device)
        cell_shifts = torch.zeros(
            positions.shape[0], 3, dtype=torch.int32, device=device
        )
        wp_cell_atom_counts = wp.from_torch(
            cell_atom_counts, dtype=wp.int32, return_ctype=True
        )
        wp_cell_shifts = wp.from_torch(cell_shifts, dtype=wp.vec3i, return_ctype=True)

        # Launch kernel
        wp.launch(
            _cell_list_count_atoms_per_bin_overload[wp_dtype],
            dim=positions.shape[0],
            device=wp_device,
            inputs=(
                wp_positions,
                wp_cell,
                wp_pbc,
                wp_cell_counts,
                wp_cell_atom_counts,
                wp_cell_shifts,
            ),
        )

        # Check results
        total_atoms_counted = cell_atom_counts.sum().item()
        assert total_atoms_counted == positions.shape[0], (
            f"Expected {positions.shape[0]} atoms, counted {total_atoms_counted}"
        )

        # All counts should be non-negative
        assert torch.all(cell_atom_counts >= 0), (
            "All cell atom counts should be non-negative"
        )

    def test_compute_cell_offsets(self, device, dtype):
        """Test _cell_list_compute_cell_offsets kernel."""
        wp_device = str(device)

        # Test data: cell atom counts
        cell_atom_counts = torch.tensor(
            [3, 0, 2, 1, 4], dtype=torch.int32, device=device
        )
        num_cells = len(cell_atom_counts)

        # Output array
        cell_offsets = torch.zeros(num_cells, dtype=torch.int32, device=device)
        wp_cell_atom_counts = wp.from_torch(
            cell_atom_counts, dtype=wp.int32, return_ctype=True
        )
        wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32, return_ctype=True)

        # Launch kernel
        wp.launch(
            _cell_list_compute_cell_offsets,
            dim=num_cells,
            device=wp_device,
            inputs=(wp_cell_atom_counts, wp_cell_offsets, num_cells),
        )

        # Expected prefix sum: [0, 3, 3, 5, 6]
        expected = torch.tensor([0, 3, 3, 5, 6], dtype=torch.int32, device=device)
        torch.testing.assert_close(cell_offsets, expected)

    def test_bin_atoms(self, device, dtype):
        """Test _cell_list_bin_atoms kernel."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        pbc = pbc.squeeze(0)
        # Setup for 2x2x2 grid
        cell_counts = torch.tensor([2, 2, 2], dtype=torch.int32, device=device)
        total_cells = cell_counts.prod().item()

        # Convert to warp types
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
        wp_cell_counts = wp.from_torch(cell_counts, dtype=wp.int32, return_ctype=True)

        # First, count atoms per bin
        cell_atom_counts = torch.zeros(total_cells, dtype=torch.int32, device=device)
        wp_cell_atom_counts = wp.from_torch(
            cell_atom_counts, dtype=wp.int32, return_ctype=True
        )
        cell_shifts = torch.zeros(
            positions.shape[0], 3, dtype=torch.int32, device=device
        )
        wp_cell_shifts = wp.from_torch(cell_shifts, dtype=wp.vec3i, return_ctype=True)

        wp.launch(
            _cell_list_count_atoms_per_bin_overload[wp_dtype],
            dim=positions.shape[0],
            device=wp_device,
            inputs=(
                wp_positions,
                wp_cell,
                wp_pbc,
                wp_cell_counts,
                wp_cell_atom_counts,
                wp_cell_shifts,
            ),
        )

        # Compute offsets
        cell_offsets = torch.zeros(total_cells, dtype=torch.int32, device=device)
        wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32, return_ctype=True)

        wp.launch(
            _cell_list_compute_cell_offsets,
            dim=total_cells,
            device=wp_device,
            inputs=(wp_cell_atom_counts, wp_cell_offsets, total_cells),
        )

        # Allocate atom indices array
        total_cells = cell_offsets[-1].item() + cell_atom_counts[-1].item()
        cell_atom_indices = torch.zeros(total_cells, dtype=torch.int32, device=device)
        wp_cell_atom_indices = wp.from_torch(
            cell_atom_indices, dtype=wp.int32, return_ctype=True
        )

        # Arrays for bin_atoms kernel
        atom_cell_indices = torch.zeros(
            positions.shape[0], 3, dtype=torch.int32, device=device
        )
        wp_atom_cell_indices = wp.from_torch(
            atom_cell_indices, dtype=wp.vec3i, return_ctype=True
        )

        # Reset counts for binning
        cell_atom_counts.zero_()

        # Launch bin_atoms kernel
        wp.launch(
            _cell_list_bin_atoms_overload[wp_dtype],
            dim=positions.shape[0],
            device=wp_device,
            inputs=(
                wp_positions,
                wp_cell,
                wp_pbc,
                wp_cell_counts,
                wp_atom_cell_indices,
                wp_cell_atom_counts,
                wp_cell_offsets,
                wp_cell_atom_indices,
            ),
        )

        # Check that all atoms are binned
        total_binned = cell_atom_counts.sum().item()
        assert total_binned == positions.shape[0], (
            f"Expected {positions.shape[0]} atoms binned, got {total_binned}"
        )

        # Check atom indices are valid
        valid_indices = (cell_atom_indices >= 0) & (
            cell_atom_indices < positions.shape[0]
        )
        assert torch.all(valid_indices[:total_binned]), (
            "All atom indices should be valid"
        )

    def test_build_neighbor_matrix(self, device, dtype):
        """Test _cell_list_build_neighbor_matrix kernel."""
        # Simple two-atom system
        positions, cell, pbc = create_random_system(
            num_atoms=16, cell_size=1.0, dtype=dtype, device=device, pbc_flag=False
        )
        pbc = pbc.squeeze(0)
        cell = cell.reshape(1, 3, 3)
        cutoff = 0.5

        # Estimate cell list sizes
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell,
            pbc,
            cutoff,
        )
        # Simplified setup - put both atoms in same cell
        (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius,
            device,
        )
        torch_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )

        # Output arrays - neighbor matrix format
        max_neighbors = 10  # Generous allocation
        num_atoms = positions.shape[0]
        neighbor_matrix = torch.full(
            (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (num_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(num_atoms, dtype=torch.int32, device=device)

        # Convert to warp types
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
        wp_cells_per_dimension = wp.from_torch(
            cells_per_dimension, dtype=wp.int32, return_ctype=True
        )
        wp_atom_periodic_shifts = wp.from_torch(
            atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
        )
        wp_neighbor_search_radius = wp.from_torch(
            neighbor_search_radius, dtype=wp.vec3i, return_ctype=True
        )
        wp_atom_to_cell_mapping = wp.from_torch(
            atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
        )
        wp_atoms_per_cell_count = wp.from_torch(
            atoms_per_cell_count, dtype=wp.int32, return_ctype=True
        )
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

        # Launch kernel
        wp.launch(
            _cell_list_build_neighbor_matrix_overload[wp_dtype],
            dim=positions.shape[0],
            device=wp_device,
            inputs=(
                wp_positions,
                wp_cell,
                wp_pbc,
                wp_dtype(cutoff),
                wp_cells_per_dimension,
                wp_atom_periodic_shifts,
                wp_neighbor_search_radius,
                wp_atom_to_cell_mapping,
                wp_atoms_per_cell_count,
                wp_cell_atom_start_indices,
                wp_cell_atom_list,
                wp_neighbor_matrix,
                wp_neighbor_matrix_shifts,
                wp_num_neighbors,
                True,
            ),
        )

        # Check that we found some neighbors
        assert torch.all(num_neighbors >= 0), "Neighbor counts should be non-negative"

        # Each atom should have the other as a neighbor
        for atom_idx in range(num_atoms):
            n_neigh = num_neighbors[atom_idx].item()
            if n_neigh > 0:
                # Check that distances are within cutoff
                for neigh_idx in range(min(n_neigh, max_neighbors)):
                    atom_j = neighbor_matrix[atom_idx, neigh_idx].item()
                    if atom_j == -1:
                        break

                    # Verify distance is within cutoff (shifts are in fractional coords)
                    shift_frac = neighbor_matrix_shifts[atom_idx, neigh_idx]
                    shift_cart = cell.reshape(3, 3).T @ shift_frac.to(dtype)

                    rij = positions[atom_j] - positions[atom_idx] + shift_cart
                    dist = torch.norm(rij).item()
                    assert dist < cutoff + 1e-5, (
                        f"Distance {dist} should be within cutoff {cutoff}"
                    )


@pytest.mark.parametrize("dtype", dtypes)
class TestCellListWpLaunchers:
    """Test the public launcher API for cell lists."""

    def test_build_cell_list(self, device, dtype):
        """Test build_cell_list launcher."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        pbc = pbc.squeeze(0)
        cutoff = 1.1

        # Get size estimates
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)

        # Allocate cell list
        (
            cells_per_dimension,
            _,  # neighbor_search_radius not needed for wp launcher
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = allocate_cell_list(
            positions.shape[0], max_cells, neighbor_search_radius, device
        )

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
        wp_cells_per_dimension = wp.from_torch(
            cells_per_dimension, dtype=wp.int32, return_ctype=True
        )
        wp_atom_periodic_shifts = wp.from_torch(
            atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
        )
        wp_atom_to_cell_mapping = wp.from_torch(
            atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
        )
        wp_atoms_per_cell_count = wp.from_torch(atoms_per_cell_count, dtype=wp.int32)
        wp_cell_atom_start_indices = wp.from_torch(
            cell_atom_start_indices, dtype=wp.int32
        )
        wp_cell_atom_list = wp.from_torch(
            cell_atom_list, dtype=wp.int32, return_ctype=True
        )

        # Call build_cell_list launcher
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

        # Verify results
        assert torch.all(cells_per_dimension > 0), "Cell dimensions should be positive"
        total_binned = atoms_per_cell_count.sum().item()
        assert total_binned == positions.shape[0], (
            f"Expected {positions.shape[0]} atoms binned, got {total_binned}"
        )

    def test_query_cell_list(self, device, dtype):
        """Test query_cell_list launcher."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        pbc = pbc.squeeze(0)
        cutoff = 1.1

        # Build cell list first using warp launcher directly
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
        cell_list_cache = allocate_cell_list(
            positions.shape[0], max_cells, neighbor_search_radius, device
        )

        # Convert to warp and call wp_build_cell_list
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
        wp_cells_per_dimension = wp.from_torch(
            cell_list_cache[0], dtype=wp.int32, return_ctype=True
        )
        wp_atom_periodic_shifts = wp.from_torch(
            cell_list_cache[2], dtype=wp.vec3i, return_ctype=True
        )
        wp_atom_to_cell_mapping = wp.from_torch(
            cell_list_cache[3], dtype=wp.vec3i, return_ctype=True
        )
        wp_atoms_per_cell_count = wp.from_torch(cell_list_cache[4], dtype=wp.int32)
        wp_cell_atom_start_indices = wp.from_torch(cell_list_cache[5], dtype=wp.int32)
        wp_cell_atom_list = wp.from_torch(
            cell_list_cache[6], dtype=wp.int32, return_ctype=True
        )

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

        # Prepare neighbor matrix
        max_neighbors = 10
        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            -1,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts = torch.zeros(
            (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(
            (positions.shape[0],), dtype=torch.int32, device=device
        )

        # Re-convert to warp arrays (already converted above for build)
        wp_neighbor_search_radius = wp.from_torch(
            cell_list_cache[1], dtype=wp.int32, return_ctype=True
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

        # Call query_cell_list launcher
        query_cell_list(
            wp_positions,
            wp_cell,
            wp_pbc,
            cutoff,
            wp_cells_per_dimension,
            wp_neighbor_search_radius,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
            wp_neighbor_matrix,
            wp_neighbor_matrix_shifts,
            wp_num_neighbors,
            wp_dtype,
            str(device),
            True,
        )

        # Verify we found some neighbors
        assert torch.all(num_neighbors >= 0), "Neighbor counts should be non-negative"
        assert num_neighbors.sum() > 0, "Should find some neighbors"
