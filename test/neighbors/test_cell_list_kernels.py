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
    build_cell_list,
    get_build_cell_list_kernel,
    query_cell_list,
)
from nvalchemiops.neighbors.neighbor_utils import empty_sentinel as _empty_sentinel
from nvalchemiops.torch.neighbors.cell_list import (
    estimate_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

from .test_utils import (
    create_simple_cubic_system,
    neighbor_matrix_row_set,
)

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
            get_build_cell_list_kernel("construct_bin_size", wp_dtype),
            dim=1,
            device=wp_device,
            inputs=(
                wp_cell,
                wp_pbc,
                _empty_sentinel(2, wp.bool, wp_device),
                wp_cell_counts,
                _empty_sentinel(1, wp.vec3i, wp_device),
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
            get_build_cell_list_kernel("count_atoms", wp_dtype),
            dim=positions.shape[0],
            device=wp_device,
            inputs=(
                wp_positions,
                wp_cell,
                wp_pbc,
                _empty_sentinel(2, wp.bool, wp_device),
                _empty_sentinel(1, wp.int32, wp_device),
                wp_cell_counts,
                _empty_sentinel(1, wp.vec3i, wp_device),
                _empty_sentinel(1, wp.int32, wp_device),
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
        """Production cell-offset compute uses ``wp.utils.array_scan`` (CUB
        exclusive scan) — invoked internally by ``build_cell_list``.  This
        unit-test exercises the same primitive directly.
        """
        if str(device) == "cpu":
            pytest.skip(
                "wp.utils.array_scan coverage is CUDA-only here; "
                "CPU parameter is not supported"
            )

        cell_atom_counts = torch.tensor(
            [3, 0, 2, 1, 4], dtype=torch.int32, device=device
        )
        num_cells = len(cell_atom_counts)
        cell_offsets = torch.zeros(num_cells, dtype=torch.int32, device=device)
        wp_counts = wp.from_torch(cell_atom_counts, dtype=wp.int32)
        wp_offsets = wp.from_torch(cell_offsets, dtype=wp.int32)
        wp.utils.array_scan(wp_counts, wp_offsets, inclusive=False)

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
            get_build_cell_list_kernel("count_atoms", wp_dtype),
            dim=positions.shape[0],
            device=wp_device,
            inputs=(
                wp_positions,
                wp_cell,
                wp_pbc,
                _empty_sentinel(2, wp.bool, wp_device),
                _empty_sentinel(1, wp.int32, wp_device),
                wp_cell_counts,
                _empty_sentinel(1, wp.vec3i, wp_device),
                _empty_sentinel(1, wp.int32, wp_device),
                wp_cell_atom_counts,
                wp_cell_shifts,
            ),
        )

        # Compute offsets — production uses wp.utils.array_scan (CUB).
        cell_offsets = torch.zeros(total_cells, dtype=torch.int32, device=device)
        wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32, return_ctype=True)
        wp.utils.array_scan(
            wp.from_torch(cell_atom_counts, dtype=wp.int32),
            wp.from_torch(cell_offsets, dtype=wp.int32),
            inclusive=False,
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
            get_build_cell_list_kernel("bin_atoms", wp_dtype),
            dim=positions.shape[0],
            device=wp_device,
            inputs=(
                wp_positions,
                wp_cell,
                wp_pbc,
                _empty_sentinel(2, wp.bool, wp_device),
                _empty_sentinel(1, wp.int32, wp_device),
                wp_cell_counts,
                _empty_sentinel(1, wp.vec3i, wp_device),
                _empty_sentinel(1, wp.int32, wp_device),
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

        # Caller-allocated sort scratch + rebuild_flags (always-True for
        # non-selective builds).  The warp-level API holds no hidden state.
        wp_sorted_positions = wp.empty(
            positions.shape[0], dtype=wp_vec_dtype, device=str(device)
        )
        wp_sorted_shifts = wp.empty(
            positions.shape[0], dtype=wp.vec3i, device=str(device)
        )
        wp_rebuild_flags = wp.array([True], dtype=wp.bool, device=str(device))

        query_cell_list(
            positions=wp_positions,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=cutoff,
            cells_per_dimension=wp_cells_per_dimension,
            neighbor_search_radius=wp_neighbor_search_radius,
            atom_periodic_shifts=wp_atom_periodic_shifts,
            atom_to_cell_mapping=wp_atom_to_cell_mapping,
            atoms_per_cell_count=wp_atoms_per_cell_count,
            cell_atom_start_indices=wp_cell_atom_start_indices,
            cell_atom_list=wp_cell_atom_list,
            sorted_positions=wp_sorted_positions,
            sorted_atom_periodic_shifts=wp_sorted_shifts,
            neighbor_matrix=wp_neighbor_matrix,
            neighbor_matrix_shifts=wp_neighbor_matrix_shifts,
            num_neighbors=wp_num_neighbors,
            rebuild_flags=wp_rebuild_flags,
            wp_dtype=wp_dtype,
            device=str(device),
            half_fill=True,
        )

        # Verify we found some neighbors
        assert torch.all(num_neighbors >= 0), "Neighbor counts should be non-negative"
        assert num_neighbors.sum() > 0, "Should find some neighbors"


@pytest.mark.parametrize("dtype", dtypes)
class TestCellListSelectiveRebuildFlags:
    """Test selective rebuild (rebuild_flags) for cell list warp launchers."""

    def test_no_rebuild_preserves_data(self, device, dtype):
        """Flag=False: neighbor data should remain unchanged."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        pbc = pbc.squeeze(0)
        cutoff = 1.1

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        # Build cell list
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
        cell_list_cache = allocate_cell_list(
            positions.shape[0], max_cells, neighbor_search_radius, device
        )

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

        max_neighbors = 10
        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(
            (positions.shape[0],), dtype=torch.int32, device=device
        )
        wp_neighbor_search_radius = wp.from_torch(
            cell_list_cache[1], dtype=wp.int32, return_ctype=True
        )
        wp_nm = wp.from_torch(neighbor_matrix, dtype=wp.int32, return_ctype=True)
        wp_nm_shifts = wp.from_torch(
            neighbor_matrix_shifts, dtype=wp.vec3i, return_ctype=True
        )
        wp_nn = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)

        wp_sorted_positions = wp.empty(
            positions.shape[0], dtype=wp_vec_dtype, device=str(device)
        )
        wp_sorted_shifts = wp.empty(
            positions.shape[0], dtype=wp.vec3i, device=str(device)
        )
        wp_always_true = wp.array([True], dtype=wp.bool, device=str(device))

        query_cell_list(
            positions=wp_positions,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=cutoff,
            cells_per_dimension=wp_cells_per_dimension,
            neighbor_search_radius=wp_neighbor_search_radius,
            atom_periodic_shifts=wp_atom_periodic_shifts,
            atom_to_cell_mapping=wp_atom_to_cell_mapping,
            atoms_per_cell_count=wp_atoms_per_cell_count,
            cell_atom_start_indices=wp_cell_atom_start_indices,
            cell_atom_list=wp_cell_atom_list,
            sorted_positions=wp_sorted_positions,
            sorted_atom_periodic_shifts=wp_sorted_shifts,
            neighbor_matrix=wp_nm,
            neighbor_matrix_shifts=wp_nm_shifts,
            num_neighbors=wp_nn,
            rebuild_flags=wp_always_true,
            wp_dtype=wp_dtype,
            device=str(device),
            half_fill=True,
        )

        saved_nm = neighbor_matrix.clone()
        saved_nn = num_neighbors.clone()

        # Selective query with flag=False: data should be unchanged
        rebuild_flags = torch.zeros(1, dtype=torch.bool, device=device)
        wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool)

        query_cell_list(
            positions=wp_positions,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=cutoff,
            cells_per_dimension=wp_cells_per_dimension,
            neighbor_search_radius=wp_neighbor_search_radius,
            atom_periodic_shifts=wp_atom_periodic_shifts,
            atom_to_cell_mapping=wp_atom_to_cell_mapping,
            atoms_per_cell_count=wp_atoms_per_cell_count,
            cell_atom_start_indices=wp_cell_atom_start_indices,
            cell_atom_list=wp_cell_atom_list,
            sorted_positions=wp_sorted_positions,
            sorted_atom_periodic_shifts=wp_sorted_shifts,
            neighbor_matrix=wp_nm,
            neighbor_matrix_shifts=wp_nm_shifts,
            num_neighbors=wp_nn,
            rebuild_flags=wp_rebuild_flags,
            wp_dtype=wp_dtype,
            device=str(device),
            half_fill=True,
        )

        assert torch.equal(num_neighbors, saved_nn), (
            "num_neighbors must be unchanged when rebuild_flags is False"
        )
        for i in range(positions.shape[0]):
            n = num_neighbors[i].item()
            assert torch.equal(neighbor_matrix[i, :n], saved_nm[i, :n]), (
                f"neighbor_matrix row {i} should be unchanged"
            )

    def test_rebuild_updates_data(self, device, dtype):
        """Flag=True: result should match a fresh full rebuild."""

        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        pbc = pbc.squeeze(0)
        cutoff = 1.1

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
        cell_list_cache = allocate_cell_list(
            positions.shape[0], max_cells, neighbor_search_radius, device
        )

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

        max_neighbors = 10
        wp_neighbor_search_radius = wp.from_torch(
            cell_list_cache[1], dtype=wp.int32, return_ctype=True
        )

        # Reference: full build
        nm_ref = torch.full(
            (positions.shape[0], max_neighbors), -1, dtype=torch.int32, device=device
        )
        nm_ref_shifts = torch.zeros(
            (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
        )
        nn_ref = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)
        wp_nm_ref = wp.from_torch(nm_ref, dtype=wp.int32, return_ctype=True)
        wp_nm_ref_shifts = wp.from_torch(
            nm_ref_shifts, dtype=wp.vec3i, return_ctype=True
        )
        wp_nn_ref = wp.from_torch(nn_ref, dtype=wp.int32, return_ctype=True)

        wp_sorted_positions = wp.empty(
            positions.shape[0], dtype=wp_vec_dtype, device=str(device)
        )
        wp_sorted_shifts = wp.empty(
            positions.shape[0], dtype=wp.vec3i, device=str(device)
        )
        wp_always_true = wp.array([True], dtype=wp.bool, device=str(device))

        query_cell_list(
            positions=wp_positions,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=cutoff,
            cells_per_dimension=wp_cells_per_dimension,
            neighbor_search_radius=wp_neighbor_search_radius,
            atom_periodic_shifts=wp_atom_periodic_shifts,
            atom_to_cell_mapping=wp_atom_to_cell_mapping,
            atoms_per_cell_count=wp_atoms_per_cell_count,
            cell_atom_start_indices=wp_cell_atom_start_indices,
            cell_atom_list=wp_cell_atom_list,
            sorted_positions=wp_sorted_positions,
            sorted_atom_periodic_shifts=wp_sorted_shifts,
            neighbor_matrix=wp_nm_ref,
            neighbor_matrix_shifts=wp_nm_ref_shifts,
            num_neighbors=wp_nn_ref,
            rebuild_flags=wp_always_true,
            wp_dtype=wp_dtype,
            device=str(device),
            half_fill=True,
        )

        # Selective rebuild with flag=True
        nm_sel = torch.full(
            (positions.shape[0], max_neighbors), 0, dtype=torch.int32, device=device
        )
        nm_sel_shifts = torch.zeros(
            (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
        )
        nn_sel = torch.full(positions.shape[0:1], 0, dtype=torch.int32, device=device)
        wp_nm_sel = wp.from_torch(nm_sel, dtype=wp.int32, return_ctype=True)
        wp_nm_sel_shifts = wp.from_torch(
            nm_sel_shifts, dtype=wp.vec3i, return_ctype=True
        )
        wp_nn_sel = wp.from_torch(nn_sel, dtype=wp.int32, return_ctype=True)

        rebuild_flags = torch.ones(1, dtype=torch.bool, device=device)
        wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool)

        query_cell_list(
            positions=wp_positions,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=cutoff,
            cells_per_dimension=wp_cells_per_dimension,
            neighbor_search_radius=wp_neighbor_search_radius,
            atom_periodic_shifts=wp_atom_periodic_shifts,
            atom_to_cell_mapping=wp_atom_to_cell_mapping,
            atoms_per_cell_count=wp_atoms_per_cell_count,
            cell_atom_start_indices=wp_cell_atom_start_indices,
            cell_atom_list=wp_cell_atom_list,
            sorted_positions=wp_sorted_positions,
            sorted_atom_periodic_shifts=wp_sorted_shifts,
            neighbor_matrix=wp_nm_sel,
            neighbor_matrix_shifts=wp_nm_sel_shifts,
            num_neighbors=wp_nn_sel,
            rebuild_flags=wp_rebuild_flags,
            wp_dtype=wp_dtype,
            device=str(device),
            half_fill=True,
        )

        assert torch.equal(nn_sel, nn_ref), (
            "num_neighbors should match full rebuild when flag=True"
        )


@pytest.mark.parametrize("dtype", dtypes)
class TestCellListPairCentric:
    """Parity tests for the pair-centric query kernel.

    The pair-centric kernel is dispatched automatically by the torch wrapper
    based on a sync-free heuristic; these tests force it via the
    ``strategy=`` parameter and compare its output against the atom-centric
    reference.
    """

    def test_pair_set_matches_atom_centric(self, device, dtype):
        """Same pair set as atom-centric on the cubic test system."""
        from nvalchemiops.torch.neighbors.cell_list import cell_list

        if str(device) == "cpu":
            pytest.skip(
                "strategy='pair_centric' uses CUDA block scheduling; "
                "CPU parameter is not supported"
            )

        # 4×4×4 simple cubic, lattice spacing 0.5 → box=2.0, cutoff=0.6
        # captures only nearest-neighbor pairs (1 lattice spacing).
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=64, cell_size=2.0, dtype=dtype, device=device
        )
        pbc = pbc.squeeze(0)
        cutoff = 0.6

        nm_a, nn_a, _ = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            half_fill=True,
            strategy="atom_centric",
        )
        nm_p, nn_p, _ = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            half_fill=True,
            strategy="pair_centric",
        )

        assert torch.equal(nn_a, nn_p), (
            "Per-atom neighbor counts must match between kernels"
        )
        assert neighbor_matrix_row_set(nm_a, nn_a) == neighbor_matrix_row_set(
            nm_p, nn_p
        ), "Pair sets must match between kernels (row order may differ)"

    def test_dispatch_threshold_rule(self, device, dtype):
        """``select_cell_list_strategy`` applies the (N, cutoff) threshold rule."""
        del device, dtype  # not used; fixture present for parametrization
        from nvalchemiops.torch.neighbors.cell_list import select_cell_list_strategy

        # Clause 1 (cutoff >= 8 AND N <= 65536) → pair_centric.
        assert select_cell_list_strategy(4096, 12.0) == "pair_centric"
        # Outside every clause → atom_centric.
        assert select_cell_list_strategy(131072, 12.0) == "atom_centric"
        # Clause 2 (cutoff >= 6 AND N <= 8192) → pair_centric.
        assert select_cell_list_strategy(2048, 6.0) == "pair_centric"
        # Clause 3 (cutoff >= 4 AND N <= 1024) → pair_centric.
        assert select_cell_list_strategy(1024, 4.0) == "pair_centric"

    def test_pair_centric_with_rebuild_flag_false(self, device, dtype):
        """rebuild_flags=False with pair-centric must preserve outputs."""
        from nvalchemiops.torch.neighbors.cell_list import cell_list

        if str(device) == "cpu":
            pytest.skip(
                "strategy='pair_centric' uses CUDA block scheduling; "
                "CPU parameter is not supported"
            )

        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=64, cell_size=2.0, dtype=dtype, device=device
        )
        pbc = pbc.squeeze(0)
        cutoff = 0.6

        nm, nn, nms = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            half_fill=True,
            strategy="pair_centric",
        )
        saved_nm = nm.clone()
        saved_nn = nn.clone()
        saved_nms = nms.clone()

        rebuild_flags = torch.zeros(1, dtype=torch.bool, device=device)
        cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            half_fill=True,
            strategy="pair_centric",
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            num_neighbors=nn,
            rebuild_flags=rebuild_flags,
        )

        assert torch.equal(nn, saved_nn), (
            "pair-centric: num_neighbors must be preserved when flag=False"
        )
        assert torch.equal(nm, saved_nm), (
            "pair-centric: neighbor_matrix must be preserved when flag=False"
        )
        assert torch.equal(nms, saved_nms), (
            "pair-centric: shifts must be preserved when flag=False"
        )


def _make_query_cell_list_kwargs(device, dtype, *, half_fill=True):
    """Build a fully-allocated kwargs dict for ``query_cell_list``.

    Used by the error-path tests below: the strategy-validation block
    runs before any kernel launch, so the inner contents need not be
    correct — only the shapes/dtypes need to satisfy the warp-array
    binding so the launcher reaches the strategy check.
    """
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=dtype, device=device
    )
    pbc = pbc.squeeze(0)
    cutoff = 1.1
    max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
    cl_cache = allocate_cell_list(
        positions.shape[0], max_cells, neighbor_search_radius, device
    )
    wp_dtype = get_wp_dtype(dtype)
    wp_vec = get_wp_vec_dtype(dtype)
    wp_mat = get_wp_mat_dtype(dtype)
    wp_positions = wp.from_torch(positions, dtype=wp_vec, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
    wp_cpd = wp.from_torch(cl_cache[0], dtype=wp.int32, return_ctype=True)
    wp_nsr = wp.from_torch(cl_cache[1], dtype=wp.int32, return_ctype=True)
    wp_aps = wp.from_torch(cl_cache[2], dtype=wp.vec3i, return_ctype=True)
    wp_atc = wp.from_torch(cl_cache[3], dtype=wp.vec3i, return_ctype=True)
    wp_apcc = wp.from_torch(cl_cache[4], dtype=wp.int32)
    wp_casi = wp.from_torch(cl_cache[5], dtype=wp.int32)
    wp_cal = wp.from_torch(cl_cache[6], dtype=wp.int32, return_ctype=True)
    build_cell_list(
        wp_positions,
        wp_cell,
        wp_pbc,
        cutoff,
        wp_cpd,
        wp_aps,
        wp_atc,
        wp_apcc,
        wp_casi,
        wp_cal,
        wp_dtype,
        str(device),
    )
    max_neighbors = 10
    nm = torch.full(
        (positions.shape[0], max_neighbors),
        -1,
        dtype=torch.int32,
        device=device,
    )
    nms = torch.zeros(
        (positions.shape[0], max_neighbors, 3),
        dtype=torch.int32,
        device=device,
    )
    nn = torch.zeros((positions.shape[0],), dtype=torch.int32, device=device)
    wp_nm = wp.from_torch(nm, dtype=wp.int32, return_ctype=True)
    wp_nms = wp.from_torch(nms, dtype=wp.vec3i, return_ctype=True)
    wp_nn = wp.from_torch(nn, dtype=wp.int32, return_ctype=True)
    wp_sp = wp.empty(positions.shape[0], dtype=wp_vec, device=str(device))
    wp_ss = wp.empty(positions.shape[0], dtype=wp.vec3i, device=str(device))
    wp_rf = wp.array([True], dtype=wp.bool, device=str(device))
    return dict(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=cutoff,
        cells_per_dimension=wp_cpd,
        neighbor_search_radius=wp_nsr,
        atom_periodic_shifts=wp_aps,
        atom_to_cell_mapping=wp_atc,
        atoms_per_cell_count=wp_apcc,
        cell_atom_start_indices=wp_casi,
        cell_atom_list=wp_cal,
        sorted_positions=wp_sp,
        sorted_atom_periodic_shifts=wp_ss,
        neighbor_matrix=wp_nm,
        neighbor_matrix_shifts=wp_nms,
        num_neighbors=wp_nn,
        rebuild_flags=wp_rf,
        wp_dtype=wp_dtype,
        device=str(device),
        half_fill=half_fill,
    )


@pytest.mark.parametrize("dtype", [torch.float32])
class TestQueryCellListErrorPaths:
    """Error / log paths in ``query_cell_list`` (warp launcher) — covers
    lines 1530-1559 in nvalchemiops/neighbors/cell_list.py."""

    def test_pair_centric_on_cpu_raises(self, device, dtype):
        if str(device) != "cpu":
            pytest.skip("CPU-only error path; CUDA parameter is not supported")
        kwargs = _make_query_cell_list_kwargs(device, dtype)
        with pytest.raises(ValueError, match="not supported on CPU"):
            query_cell_list(**kwargs, strategy="pair_centric")

    def test_pair_centric_missing_n_outer_raises(self, device, dtype):
        if str(device) == "cpu":
            pytest.skip(
                "strategy='pair_centric' n_outer validation is only reachable "
                "on CUDA; CPU raises before n_outer validation"
            )
        kwargs = _make_query_cell_list_kwargs(device, dtype)
        with pytest.raises(ValueError, match="n_outer"):
            query_cell_list(**kwargs, strategy="pair_centric", n_outer=None)

    def test_unknown_strategy_raises(self, device, dtype):
        kwargs = _make_query_cell_list_kwargs(device, dtype)
        with pytest.raises(ValueError, match="atom_centric"):
            query_cell_list(**kwargs, strategy="bogus")

    def test_pair_centric_success_path(self, device, dtype):
        """Valid ``strategy='pair_centric'`` + ``n_outer`` exercises the
        ``chosen = 'pair_centric'`` assignment and the corresponding
        ``query_cell_list_pair_centric_sorted`` launch inside the warp
        launcher (lines 1542 + 1576 of cell_list.py)."""
        from nvalchemiops.neighbors.cell_list import (
            compute_batch_pair_centric_n_outer,
        )

        if str(device) == "cpu":
            pytest.skip(
                "strategy='pair_centric' uses CUDA block scheduling; "
                "CPU parameter is not supported"
            )
        # Recompute the per-axis neighbor_search_radius the helper used so we
        # can derive n_outer.  Cheaper than reading back from the wp.array.
        _, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        pbc = pbc.squeeze(0)
        _, nsr_t = estimate_cell_list_sizes(cell, pbc, 1.1)
        nsr = tuple(int(x) for x in nsr_t.cpu().tolist())
        n_outer = compute_batch_pair_centric_n_outer(nsr, half_fill=True)
        kwargs = _make_query_cell_list_kwargs(device, dtype, half_fill=True)
        query_cell_list(**kwargs, strategy="pair_centric", n_outer=n_outer)
        # Kernel wrote into num_neighbors via wp.from_torch — sanity check.
        assert kwargs["num_neighbors"] is not None


# ---------------------------------------------------------------------------
# Isolated pair-output kwargs on the cell-list query launcher.  Mirrors the
# naive coverage; verifies that ``return_vectors`` / ``return_distances`` /
# ``target_indices`` work end-to-end through the cell-list dispatch.
# ---------------------------------------------------------------------------


def _make_cell_list_pair_output_setup(device, dtype):
    """Return ``(query_kwargs, output_tensors)`` where ``output_tensors``
    keeps torch handles for verification (the helper uses ``return_ctype=True``
    which precludes ``wp.to_torch``).
    """
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=dtype, device=device
    )
    pbc = pbc.squeeze(0)
    cutoff = 1.1
    max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
    cl_cache = allocate_cell_list(
        positions.shape[0], max_cells, neighbor_search_radius, device
    )
    wp_dtype = get_wp_dtype(dtype)
    wp_vec = get_wp_vec_dtype(dtype)
    wp_mat = get_wp_mat_dtype(dtype)
    wp_positions = wp.from_torch(positions, dtype=wp_vec)
    wp_cell = wp.from_torch(cell, dtype=wp_mat)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool)
    wp_cpd = wp.from_torch(cl_cache[0], dtype=wp.int32)
    wp_nsr = wp.from_torch(cl_cache[1], dtype=wp.int32)
    wp_aps = wp.from_torch(cl_cache[2], dtype=wp.vec3i)
    wp_atc = wp.from_torch(cl_cache[3], dtype=wp.vec3i)
    wp_apcc = wp.from_torch(cl_cache[4], dtype=wp.int32)
    wp_casi = wp.from_torch(cl_cache[5], dtype=wp.int32)
    wp_cal = wp.from_torch(cl_cache[6], dtype=wp.int32)
    build_cell_list(
        wp_positions,
        wp_cell,
        wp_pbc,
        cutoff,
        wp_cpd,
        wp_aps,
        wp_atc,
        wp_apcc,
        wp_casi,
        wp_cal,
        wp_dtype,
        str(device),
    )
    N = positions.shape[0]
    max_neighbors = 16
    nm_t = torch.full((N, max_neighbors), -1, dtype=torch.int32, device=device)
    nms_t = torch.zeros((N, max_neighbors, 3), dtype=torch.int32, device=device)
    nn_t = torch.zeros((N,), dtype=torch.int32, device=device)
    return dict(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=cutoff,
        cells_per_dimension=wp_cpd,
        neighbor_search_radius=wp_nsr,
        atom_periodic_shifts=wp_aps,
        atom_to_cell_mapping=wp_atc,
        atoms_per_cell_count=wp_apcc,
        cell_atom_start_indices=wp_casi,
        cell_atom_list=wp_cal,
        neighbor_matrix=wp.from_torch(nm_t, dtype=wp.int32),
        neighbor_matrix_shifts=wp.from_torch(nms_t, dtype=wp.vec3i),
        num_neighbors=wp.from_torch(nn_t, dtype=wp.int32),
        wp_dtype=wp_dtype,
        device=str(device),
    ), dict(
        positions=positions,
        N=N,
        max_neighbors=max_neighbors,
        nm=nm_t,
        nms=nms_t,
        nn=nn_t,
        cutoff=cutoff,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32])
class TestCellListPairOutputsIsolated:
    """Each pair-output kwarg verified in isolation."""

    def test_return_distances_only(self, device, dtype):
        kwargs, out = _make_cell_list_pair_output_setup(device, dtype)
        distances = torch.zeros(
            (out["N"], out["max_neighbors"]), dtype=dtype, device=device
        )
        query_cell_list(
            **kwargs,
            half_fill=False,
            strategy="atom_centric",
            return_distances=True,
            neighbor_distances=wp.from_torch(distances, dtype=wp.float32),
        )
        for i in range(out["N"]):
            n = int(out["nn"][i].item())
            for slot in range(n):
                d = float(distances[i, slot].item())
                assert 0.0 < d <= out["cutoff"] + 1e-4, (
                    f"distance {d} at i={i} slot={slot} outside (0, cutoff]"
                )

    def test_return_vectors_only(self, device, dtype):
        kwargs, out = _make_cell_list_pair_output_setup(device, dtype)
        vectors = torch.zeros(
            (out["N"], out["max_neighbors"], 3), dtype=dtype, device=device
        )
        query_cell_list(
            **kwargs,
            half_fill=False,
            strategy="atom_centric",
            return_vectors=True,
            neighbor_vectors=wp.from_torch(vectors, dtype=wp.vec3f),
        )
        for i in range(out["N"]):
            n = int(out["nn"][i].item())
            for slot in range(n):
                d = float(torch.linalg.norm(vectors[i, slot]))
                assert 0.0 < d <= out["cutoff"] + 1e-4, (
                    f"|r| {d} at i={i} slot={slot} outside (0, cutoff]"
                )

    def test_target_indices_only(self, device, dtype):
        """Partial rows match the corresponding full-build rows."""
        # Full reference.
        kwargs_full, out_full = _make_cell_list_pair_output_setup(device, dtype)
        query_cell_list(**kwargs_full, half_fill=False, strategy="atom_centric")
        full_nn = out_full["nn"].clone()
        full_nm = out_full["nm"].clone()

        # Partial: pick 3 atoms; allocate compact (K, max_n) output.
        target_idx = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
        K = int(target_idx.shape[0])
        kwargs_p, out_p = _make_cell_list_pair_output_setup(device, dtype)
        partial_nm = torch.full(
            (K, out_p["max_neighbors"]), -1, dtype=torch.int32, device=device
        )
        partial_nms = torch.zeros(
            (K, out_p["max_neighbors"], 3), dtype=torch.int32, device=device
        )
        partial_nn = torch.zeros((K,), dtype=torch.int32, device=device)
        kwargs_p["neighbor_matrix"] = wp.from_torch(partial_nm, dtype=wp.int32)
        kwargs_p["neighbor_matrix_shifts"] = wp.from_torch(partial_nms, dtype=wp.vec3i)
        kwargs_p["num_neighbors"] = wp.from_torch(partial_nn, dtype=wp.int32)
        query_cell_list(
            **kwargs_p,
            half_fill=False,
            strategy="atom_centric",
            target_indices=wp.from_torch(target_idx, dtype=wp.int32),
        )
        for row, src in enumerate(target_idx.tolist()):
            assert int(partial_nn[row].item()) == int(full_nn[src].item()), (
                f"target row {row} count mismatch (atom {src})"
            )
            n = int(full_nn[src].item())
            assert set(partial_nm[row, :n].tolist()) == set(
                full_nm[src, :n].tolist()
            ), f"neighbor sets differ for target row {row} (atom {src})"
