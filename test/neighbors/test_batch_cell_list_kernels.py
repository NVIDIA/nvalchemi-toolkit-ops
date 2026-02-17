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

"""Tests for batch cell list kernel functions.

This module tests the warp kernels and launchers directly without
PyTorch bindings. Test data is created via test_utils (which uses PyTorch)
but is immediately converted to warp arrays for kernel testing.
"""

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.batch_cell_list import (
    _batch_cell_list_bin_atoms_overload,
    _batch_cell_list_build_neighbor_matrix_overload,
    _batch_cell_list_construct_bin_size_overload,
    _batch_cell_list_count_atoms_per_bin_overload,
    _batch_estimate_cell_list_sizes_overload,
    batch_build_cell_list,
    batch_query_cell_list,
)
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

from .test_utils import create_batch_systems, create_random_system

# Map torch dtypes to warp dtypes for parametrization
TORCH_TO_WP_DTYPE = {
    torch.float32: wp.float32,
    torch.float64: wp.float64,
}

TORCH_TO_WP_VEC_DTYPE = {
    torch.float32: wp.vec3f,
    torch.float64: wp.vec3d,
}

TORCH_TO_WP_MAT_DTYPE = {
    torch.float32: wp.mat33f,
    torch.float64: wp.mat33d,
}

dtypes = [torch.float32, torch.float64]


def estimate_batch_cell_list_sizes_wp(
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    wp_dtype: type,
    device: str,
    max_nbins: int = 1000,
) -> tuple[int, wp.array]:
    """Estimate cell list sizes using warp kernel directly.

    Parameters
    ----------
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrices for each system.
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool
        Periodic boundary conditions.
    cutoff : float
        Neighbor search cutoff distance.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str
        Warp device string.
    max_nbins : int
        Maximum number of bins per system.

    Returns
    -------
    max_total_cells : int
        Maximum total cells across all systems.
    neighbor_search_radius : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Neighbor search radius for each system.
    """
    num_systems = cell.shape[0]

    number_of_cells = wp.zeros(num_systems, dtype=wp.int32, device=device)
    neighbor_search_radius = wp.zeros(num_systems, dtype=wp.vec3i, device=device)

    wp.launch(
        _batch_estimate_cell_list_sizes_overload[wp_dtype],
        dim=num_systems,
        device=device,
        inputs=(
            cell,
            pbc,
            wp_dtype(cutoff),
            max_nbins,
            number_of_cells,
            neighbor_search_radius,
        ),
    )

    # Sum the number of cells across all systems
    number_of_cells_np = number_of_cells.numpy()
    max_total_cells = int(np.sum(number_of_cells_np))

    return max_total_cells, neighbor_search_radius


def allocate_cell_list_wp(
    total_atoms: int,
    max_total_cells: int,
    num_systems: int,
    device: str,
) -> tuple[wp.array, wp.array, wp.array, wp.array, wp.array, wp.array]:
    """Allocate warp arrays for cell list data structures.

    Parameters
    ----------
    total_atoms : int
        Total number of atoms across all systems.
    max_total_cells : int
        Maximum total cells to allocate.
    num_systems : int
        Number of systems in the batch.
    device : str
        Warp device string.

    Returns
    -------
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
    atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
    atom_to_cell_mapping : wp.array, shape (total_atoms,), dtype=wp.vec3i
    atoms_per_cell_count : wp.array, shape (max_total_cells,), dtype=wp.int32
    cell_atom_start_indices : wp.array, shape (max_total_cells,), dtype=wp.int32
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
    """
    cells_per_dimension = wp.zeros(num_systems, dtype=wp.vec3i, device=device)
    atom_periodic_shifts = wp.zeros(total_atoms, dtype=wp.vec3i, device=device)
    atom_to_cell_mapping = wp.zeros(total_atoms, dtype=wp.vec3i, device=device)
    atoms_per_cell_count = wp.zeros(max_total_cells, dtype=wp.int32, device=device)
    cell_atom_start_indices = wp.zeros(max_total_cells, dtype=wp.int32, device=device)
    cell_atom_list = wp.zeros(total_atoms, dtype=wp.int32, device=device)

    return (
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    )


@pytest.mark.parametrize("dtype", dtypes)
class TestBatchCellListKernels:
    """Test individual batch cell list kernel functions."""

    def test_batch_construct_bin_size(self, device, dtype):
        """Test _batch_cell_list_construct_bin_size kernel."""
        # Create batch of systems (using torch via test_utils)
        _, cell_torch, pbc_torch, _ = create_batch_systems(
            num_systems=3,
            atoms_per_system=[5, 8, 6],
            cell_sizes=[2.0, 3.0, 2.5],
            dtype=dtype,
            device=device,
        )

        cutoff = 1.0
        max_nbins = 1000000
        num_systems = 3

        # Get warp dtypes
        wp_dtype = TORCH_TO_WP_DTYPE[dtype]
        wp_mat_dtype = TORCH_TO_WP_MAT_DTYPE[dtype]
        wp_device = str(device)

        # Convert torch tensors to warp arrays
        wp_cell = wp.from_torch(cell_torch, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(
            pbc_torch.to(dtype=torch.bool), dtype=wp.bool, return_ctype=True
        )

        # Output arrays - keep torch tensor reference for assertions (memory is shared)
        cells_per_dimension_torch = torch.zeros(
            num_systems, 3, dtype=torch.int32, device=device
        )
        wp_cells_per_dimension = wp.from_torch(
            cells_per_dimension_torch,
            dtype=wp.vec3i,
            return_ctype=True,
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
                wp_dtype(cutoff),
                max_nbins,
            ),
        )

        # Check results - use original torch tensor (memory is shared with warp array)
        cells_per_dimension_np = cells_per_dimension_torch.cpu().numpy()

        for sys_idx in range(num_systems):
            sys_cell_counts = cells_per_dimension_np[sys_idx]

            assert np.all(sys_cell_counts > 0), (
                f"System {sys_idx}: All cell counts should be positive"
            )

            # Total cells should not exceed max_nbins
            total_cells = np.prod(sys_cell_counts)
            assert total_cells <= max_nbins, (
                f"System {sys_idx}: Total cells {total_cells} exceeds max_nbins {max_nbins}"
            )

    def test_batch_cell_list_count_atoms_per_bin(self, device, dtype):
        """Test _batch_cell_list_count_atoms_per_bin kernel."""
        # Create small batch for easier testing
        positions_torch, cell_torch, pbc_torch, ptr_torch = create_batch_systems(
            num_systems=2,
            atoms_per_system=[4, 3],
            cell_sizes=[2.0, 2.5],
            dtype=dtype,
            device=device,
        )
        idx_torch = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device
        )
        num_systems = 2
        total_atoms = positions_torch.shape[0]

        # Get warp dtypes
        wp_dtype = TORCH_TO_WP_DTYPE[dtype]
        wp_vec_dtype = TORCH_TO_WP_VEC_DTYPE[dtype]
        wp_mat_dtype = TORCH_TO_WP_MAT_DTYPE[dtype]
        wp_device = str(device)

        # Estimate cell list sizes
        cells_per_dimension_torch = torch.tensor(
            [[2, 2, 2], [2, 2, 2]], dtype=torch.int32, device=device
        )

        # Cell offsets for each system
        cells_per_system = cells_per_dimension_torch.prod(dim=1)
        cell_offsets_torch = (cells_per_system.cumsum(0) - cells_per_system).to(
            torch.int32
        )
        total_cells = cells_per_system.sum().item()

        # Convert to warp arrays
        wp_positions = wp.from_torch(
            positions_torch, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_cell = wp.from_torch(cell_torch, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(
            pbc_torch.to(dtype=torch.bool), dtype=wp.bool, return_ctype=True
        )
        wp_idx = wp.from_torch(idx_torch, dtype=wp.int32, return_ctype=True)
        wp_cells_per_dimension = wp.from_torch(
            cells_per_dimension_torch, dtype=wp.vec3i, return_ctype=True
        )
        wp_cell_offsets = wp.from_torch(
            cell_offsets_torch, dtype=wp.int32, return_ctype=True
        )

        # Output arrays - keep torch references for assertions (memory is shared)
        atoms_per_cell_count_torch = torch.zeros(
            total_cells, dtype=torch.int32, device=device
        )
        atom_periodic_shifts_torch = torch.zeros(
            total_atoms, 3, dtype=torch.int32, device=device
        )
        wp_atoms_per_cell_count = wp.from_torch(
            atoms_per_cell_count_torch,
            dtype=wp.int32,
            return_ctype=True,
        )
        wp_atom_periodic_shifts = wp.from_torch(
            atom_periodic_shifts_torch,
            dtype=wp.vec3i,
            return_ctype=True,
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

        # Check results - use original torch tensor (memory is shared)
        atoms_per_cell_count_np = atoms_per_cell_count_torch.cpu().numpy()
        total_atoms_counted = np.sum(atoms_per_cell_count_np)
        assert total_atoms_counted == total_atoms, (
            f"Expected {total_atoms} atoms counted, got {total_atoms_counted}"
        )

        # Check that atoms are binned correctly per system
        cell_offsets_np = cell_offsets_torch.cpu().numpy()
        cells_per_system_np = cells_per_system.cpu().numpy()
        ptr_np = ptr_torch.cpu().numpy()
        for sys_idx in range(num_systems):
            sys_start = cell_offsets_np[sys_idx]
            sys_end = sys_start + cells_per_system_np[sys_idx]
            sys_atom_count = np.sum(atoms_per_cell_count_np[sys_start:sys_end])
            expected_atoms = ptr_np[sys_idx + 1] - ptr_np[sys_idx]
            assert sys_atom_count == expected_atoms, (
                f"System {sys_idx}: expected {expected_atoms} atoms, got {sys_atom_count}"
            )

    def test_batch_bin_atoms(self, device, dtype):
        """Test _batch_cell_list_bin_atoms kernel."""
        # Create batch system
        positions_torch, cell_torch, pbc_torch, ptr_torch = create_batch_systems(
            num_systems=2,
            atoms_per_system=[3, 4],
            cell_sizes=[2.0, 2.5],
            dtype=dtype,
            device=device,
        )
        idx_torch = torch.tensor(
            [0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device
        )

        total_atoms = positions_torch.shape[0]

        # Get warp dtypes
        wp_dtype = TORCH_TO_WP_DTYPE[dtype]
        wp_vec_dtype = TORCH_TO_WP_VEC_DTYPE[dtype]
        wp_mat_dtype = TORCH_TO_WP_MAT_DTYPE[dtype]
        wp_device = str(device)

        # Setup cell structure
        cells_per_dimension_torch = torch.tensor(
            [[2, 2, 2], [2, 2, 2]], dtype=torch.int32, device=device
        )

        cells_per_system = cells_per_dimension_torch.prod(dim=1)
        cell_offsets_torch = (cells_per_system.cumsum(0) - cells_per_system).to(
            torch.int32
        )
        total_cells = cells_per_system.sum().item()

        # Convert to warp arrays
        wp_positions = wp.from_torch(
            positions_torch, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_cell = wp.from_torch(cell_torch, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(
            pbc_torch.to(dtype=torch.bool), dtype=wp.bool, return_ctype=True
        )
        wp_idx = wp.from_torch(idx_torch, dtype=wp.int32, return_ctype=True)
        wp_cells_per_dimension = wp.from_torch(
            cells_per_dimension_torch, dtype=wp.vec3i, return_ctype=True
        )
        wp_cell_offsets = wp.from_torch(
            cell_offsets_torch, dtype=wp.int32, return_ctype=True
        )

        # Allocate output arrays
        atoms_per_cell_count_torch = torch.zeros(
            total_cells, dtype=torch.int32, device=device
        )
        atom_to_cell_mapping_torch = torch.zeros(
            total_atoms, 3, dtype=torch.int32, device=device
        )
        atom_periodic_shifts_torch = torch.zeros(
            total_atoms, 3, dtype=torch.int32, device=device
        )
        cell_atom_start_indices_torch = torch.zeros(
            total_cells, dtype=torch.int32, device=device
        )
        cell_atom_list_torch = torch.zeros(
            total_atoms, dtype=torch.int32, device=device
        )

        wp_atoms_per_cell_count = wp.from_torch(
            atoms_per_cell_count_torch, dtype=wp.int32, return_ctype=True
        )
        wp_atom_periodic_shifts = wp.from_torch(
            atom_periodic_shifts_torch, dtype=wp.vec3i, return_ctype=True
        )

        # First count atoms per bin
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

        # Compute cell offsets for atom storage using torch cumsum
        # (we're testing the bin_atoms kernel, not the scan)
        torch.cumsum(
            atoms_per_cell_count_torch, dim=0, out=cell_atom_start_indices_torch
        )
        # Shift to get exclusive scan (start indices)
        cell_atom_start_indices_torch = torch.roll(cell_atom_start_indices_torch, 1)
        cell_atom_start_indices_torch[0] = 0

        wp_atom_to_cell_mapping = wp.from_torch(
            atom_to_cell_mapping_torch, dtype=wp.vec3i, return_ctype=True
        )
        wp_cell_atom_start_indices = wp.from_torch(
            cell_atom_start_indices_torch, dtype=wp.int32, return_ctype=True
        )
        wp_cell_atom_list = wp.from_torch(
            cell_atom_list_torch, dtype=wp.int32, return_ctype=True
        )

        # Reset counts for binning
        atoms_per_cell_count_torch.zero_()

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
        atoms_per_cell_count_np = atoms_per_cell_count_torch.cpu().numpy()
        total_binned = np.sum(atoms_per_cell_count_np)
        assert total_binned == total_atoms, (
            f"Expected {total_atoms} atoms binned, got {total_binned}"
        )

        # Check atom indices are valid
        cell_atom_list_np = cell_atom_list_torch.cpu().numpy()
        valid_indices = (cell_atom_list_np >= 0) & (cell_atom_list_np < total_atoms)
        assert np.all(valid_indices[:total_binned]), "All atom indices should be valid"

        # Check that each atom is assigned to a valid cell
        atom_to_cell_mapping_np = atom_to_cell_mapping_torch.cpu().numpy()
        cells_per_dimension_np = cells_per_dimension_torch.cpu().numpy()
        ptr_np = ptr_torch.cpu().numpy()
        for atom_idx in range(total_atoms):
            cell_idx = atom_to_cell_mapping_np[atom_idx]
            assert np.all(cell_idx >= 0), (
                f"Atom {atom_idx}: cell indices should be non-negative"
            )

            # Find which system this atom belongs to
            sys_idx = np.searchsorted(ptr_np[1:], atom_idx, side="right")
            sys_cell_counts = cells_per_dimension_np[sys_idx]
            assert np.all(cell_idx < sys_cell_counts), (
                f"Atom {atom_idx}: cell indices should be within system bounds"
            )

    def test_batch_build_neighbor_matrix(self, device, dtype):
        """Test _batch_cell_list_build_neighbor_matrix kernel."""
        # Create batch system
        positions_torch, cell_torch, pbc_torch, ptr_torch = create_batch_systems(
            num_systems=2,
            atoms_per_system=[3, 4],
            cell_sizes=[2.0, 2.5],
            dtype=dtype,
            device=device,
        )
        idx_torch = torch.tensor(
            [0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device
        )

        num_systems = ptr_torch.shape[0] - 1
        total_atoms = positions_torch.shape[0]
        cutoff = 1.0

        # Get warp dtypes
        wp_dtype = TORCH_TO_WP_DTYPE[dtype]
        wp_vec_dtype = TORCH_TO_WP_VEC_DTYPE[dtype]
        wp_mat_dtype = TORCH_TO_WP_MAT_DTYPE[dtype]
        wp_device = str(device)

        # Convert input tensors to warp arrays
        wp_positions = wp.from_torch(
            positions_torch, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_cell = wp.from_torch(cell_torch, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(pbc_torch, dtype=wp.bool, return_ctype=True)
        wp_idx = wp.from_torch(idx_torch, dtype=wp.int32, return_ctype=True)

        # Estimate cell list sizes using warp
        max_total_cells, wp_neighbor_search_radius = estimate_batch_cell_list_sizes_wp(
            wp_cell, wp_pbc, cutoff, wp_dtype, wp_device
        )

        # Allocate cell list arrays
        (
            wp_cells_per_dimension,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
        ) = allocate_cell_list_wp(total_atoms, max_total_cells, num_systems, wp_device)

        # Allocate cell offsets and cells_per_system scratch buffer
        wp_cell_offsets = wp.zeros(num_systems, dtype=wp.int32, device=wp_device)
        wp_cells_per_system = wp.zeros(num_systems, dtype=wp.int32, device=wp_device)

        # Build cell list using warp launcher
        batch_build_cell_list(
            wp_positions,
            wp_cell,
            wp_pbc,
            cutoff,
            wp_idx,
            wp_cells_per_dimension,
            wp_cell_offsets,
            wp_cells_per_system,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
            wp_dtype,
            wp_device,
        )

        # Output arrays - neighbor matrix format
        max_neighbors = 100
        neighbor_matrix_torch = torch.full(
            (total_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts_torch = torch.zeros(
            (total_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors_torch = torch.zeros(total_atoms, dtype=torch.int32, device=device)

        wp_neighbor_matrix = wp.from_torch(
            neighbor_matrix_torch, dtype=wp.int32, return_ctype=True
        )
        wp_neighbor_matrix_shifts = wp.from_torch(
            neighbor_matrix_shifts_torch, dtype=wp.vec3i, return_ctype=True
        )
        wp_num_neighbors = wp.from_torch(
            num_neighbors_torch, dtype=wp.int32, return_ctype=True
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
                wp_dtype(cutoff),
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
        num_neighbors_np = num_neighbors_torch.cpu().numpy()
        assert np.all(num_neighbors_np >= 0), (
            "All neighbor counts should be non-negative"
        )

        # Check that pairs are from the same system
        neighbor_matrix_np = neighbor_matrix_torch.cpu().numpy()
        idx_np = idx_torch.cpu().numpy()
        for atom_idx in range(total_atoms):
            n_neigh = num_neighbors_np[atom_idx]
            sys_i = idx_np[atom_idx]
            for neigh_idx in range(min(n_neigh, max_neighbors)):
                atom_j = neighbor_matrix_np[atom_idx, neigh_idx]
                if atom_j == -1:
                    break

                sys_j = idx_np[atom_j]
                assert sys_i == sys_j, (
                    f"Atoms {atom_idx} and {atom_j} should be from same system"
                )


@pytest.mark.parametrize("dtype", dtypes)
class TestBatchCellListWpLaunchers:
    """Test the public launcher API for batch cell lists."""

    def test_batch_build_cell_list(self, device, dtype):
        """Test batch_build_cell_list launcher."""
        positions_torch, cell_torch, pbc_torch, ptr_torch = create_batch_systems(
            num_systems=2,
            atoms_per_system=[4, 6],
            cell_sizes=[2.0, 2.5],
            dtype=dtype,
            device=device,
        )
        idx_torch = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.int32, device=device
        )
        cutoff = 1.0
        num_systems = 2
        total_atoms = positions_torch.shape[0]

        # Get warp dtypes
        wp_dtype = TORCH_TO_WP_DTYPE[dtype]
        wp_vec_dtype = TORCH_TO_WP_VEC_DTYPE[dtype]
        wp_mat_dtype = TORCH_TO_WP_MAT_DTYPE[dtype]
        wp_device = str(device)

        # Convert to warp arrays
        wp_positions = wp.from_torch(
            positions_torch, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_cell = wp.from_torch(cell_torch, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(pbc_torch, dtype=wp.bool, return_ctype=True)
        wp_idx = wp.from_torch(idx_torch, dtype=wp.int32, return_ctype=True)

        # Get size estimates using warp
        max_cells, wp_neighbor_search_radius = estimate_batch_cell_list_sizes_wp(
            wp_cell, wp_pbc, cutoff, wp_dtype, wp_device
        )

        # Allocate cell list arrays using warp
        (
            wp_cells_per_dimension,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
        ) = allocate_cell_list_wp(total_atoms, max_cells, num_systems, wp_device)

        # Allocate cell offsets and cells_per_system scratch buffer
        wp_cell_offsets = wp.zeros(num_systems, dtype=wp.int32, device=wp_device)
        wp_cells_per_system = wp.zeros(num_systems, dtype=wp.int32, device=wp_device)

        # Build cell list using warp launcher
        batch_build_cell_list(
            wp_positions,
            wp_cell,
            wp_pbc,
            cutoff,
            wp_idx,
            wp_cells_per_dimension,
            wp_cell_offsets,
            wp_cells_per_system,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
            wp_dtype,
            wp_device,
        )

        # Verify results - convert to numpy for assertions
        cells_per_dimension_np = wp_cells_per_dimension.numpy()
        atoms_per_cell_count_np = wp_atoms_per_cell_count.numpy()

        assert np.all(cells_per_dimension_np > 0), "Cell dimensions should be positive"
        total_binned = np.sum(atoms_per_cell_count_np)
        assert total_binned == total_atoms, (
            f"Expected {total_atoms} atoms binned, got {total_binned}"
        )

    def test_batch_query_cell_list(self, device, dtype):
        """Test batch_query_cell_list launcher."""
        positions_torch, cell_torch, pbc_torch, ptr_torch = create_batch_systems(
            num_systems=2,
            atoms_per_system=[4, 6],
            cell_sizes=[2.0, 2.5],
            dtype=dtype,
            device=device,
        )
        idx_torch = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.int32, device=device
        )
        cutoff = 1.0
        num_systems = 2
        total_atoms = positions_torch.shape[0]

        # Get warp dtypes
        wp_dtype = TORCH_TO_WP_DTYPE[dtype]
        wp_vec_dtype = TORCH_TO_WP_VEC_DTYPE[dtype]
        wp_mat_dtype = TORCH_TO_WP_MAT_DTYPE[dtype]
        wp_device = str(device)

        # Convert to warp arrays
        wp_positions = wp.from_torch(
            positions_torch, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_cell = wp.from_torch(cell_torch, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(pbc_torch, dtype=wp.bool, return_ctype=True)
        wp_idx = wp.from_torch(idx_torch, dtype=wp.int32, return_ctype=True)

        # Get size estimates and build cell list
        max_cells, wp_neighbor_search_radius = estimate_batch_cell_list_sizes_wp(
            wp_cell, wp_pbc, cutoff, wp_dtype, wp_device
        )

        (
            wp_cells_per_dimension,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
        ) = allocate_cell_list_wp(total_atoms, max_cells, num_systems, wp_device)

        wp_cell_offsets = wp.zeros(num_systems, dtype=wp.int32, device=wp_device)
        wp_cells_per_system = wp.zeros(num_systems, dtype=wp.int32, device=wp_device)

        batch_build_cell_list(
            wp_positions,
            wp_cell,
            wp_pbc,
            cutoff,
            wp_idx,
            wp_cells_per_dimension,
            wp_cell_offsets,
            wp_cells_per_system,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
            wp_dtype,
            wp_device,
        )

        # Prepare neighbor matrix using torch (for convenience in assertions)
        max_neighbors = 20
        neighbor_matrix_torch = torch.full(
            (total_atoms, max_neighbors),
            -1,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts_torch = torch.zeros(
            (total_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors_torch = torch.zeros(total_atoms, dtype=torch.int32, device=device)

        # Convert to warp arrays
        wp_neighbor_matrix = wp.from_torch(
            neighbor_matrix_torch, dtype=wp.int32, return_ctype=True
        )
        wp_neighbor_matrix_shifts = wp.from_torch(
            neighbor_matrix_shifts_torch, dtype=wp.vec3i
        )
        wp_num_neighbors = wp.from_torch(
            num_neighbors_torch, dtype=wp.int32, return_ctype=True
        )

        # Call batch_query_cell_list launcher
        batch_query_cell_list(
            wp_positions,
            wp_cell,
            wp_pbc,
            cutoff,
            wp_idx,
            wp_cells_per_dimension,
            wp_neighbor_search_radius,
            wp_cell_offsets,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
            wp_neighbor_matrix,
            wp_neighbor_matrix_shifts,
            wp_num_neighbors,
            wp_dtype,
            wp_device,
            False,
        )

        # Verify we found neighbors
        num_neighbors_np = num_neighbors_torch.cpu().numpy()
        assert np.all(num_neighbors_np >= 0), "Neighbor counts should be non-negative"
        assert np.sum(num_neighbors_np) > 0, "Should find some neighbors"

        # Verify neighbors are within same system
        neighbor_matrix_np = neighbor_matrix_torch.cpu().numpy()
        idx_np = idx_torch.cpu().numpy()
        for atom_idx in range(total_atoms):
            sys_i = idx_np[atom_idx]
            for neigh_idx in range(num_neighbors_np[atom_idx]):
                atom_j = neighbor_matrix_np[atom_idx, neigh_idx]
                if atom_j == -1:
                    break
                sys_j = idx_np[atom_j]
                assert sys_i == sys_j, "Neighbors should be from same system"


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize(
    "pbc_flags",
    [
        [[True, True, True], [True, True, True]],
        [[False, False, False], [False, False, False]],
        [[True, False, True], [False, True, False]],
    ],
)
@pytest.mark.parametrize("num_atoms", [10, 20])
@pytest.mark.parametrize("cutoff", [1.0, 3.0])
class TestBatchCellListScalingPureWarp:
    """Test batch cell list scaling with pure warp (no PyTorch bindings)."""

    def test_batch_scaling_correctness_warp(
        self, device, dtype, pbc_flags, num_atoms, cutoff
    ):
        """Test batch with various sizes and configurations using pure warp."""
        # Get warp dtypes
        wp_dtype = TORCH_TO_WP_DTYPE[dtype]
        wp_vec_dtype = TORCH_TO_WP_VEC_DTYPE[dtype]
        wp_mat_dtype = TORCH_TO_WP_MAT_DTYPE[dtype]
        wp_device = str(device)

        # Create batch systems using torch (for convenience)
        positions_list = []
        cells_list = []
        pbcs_list = []
        batch_idx_list = []

        for sys_idx, pbc_flag in enumerate(pbc_flags):
            pos, cell, pbc = create_random_system(
                num_atoms=num_atoms,
                cell_size=3.0,
                dtype=dtype,
                device=device,
                seed=42 + sys_idx,
                pbc_flag=pbc_flag,
            )
            positions_list.append(pos)
            cells_list.append(cell)
            pbcs_list.append(pbc)
            batch_idx_list.append(
                torch.full((num_atoms,), sys_idx, dtype=torch.int32, device=device)
            )

        positions_torch = torch.cat(positions_list, dim=0)
        cell_torch = torch.cat(cells_list, dim=0)
        pbc_torch = torch.cat(pbcs_list, dim=0)
        batch_idx_torch = torch.cat(batch_idx_list, dim=0)

        num_systems = cell_torch.shape[0]
        total_atoms = positions_torch.shape[0]

        # Convert to warp arrays
        wp_positions = wp.from_torch(
            positions_torch, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_cell = wp.from_torch(cell_torch, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(pbc_torch, dtype=wp.bool, return_ctype=True)
        wp_idx = wp.from_torch(batch_idx_torch, dtype=wp.int32, return_ctype=True)

        # Estimate cell list sizes using warp
        max_total_cells, wp_neighbor_search_radius = estimate_batch_cell_list_sizes_wp(
            wp_cell, wp_pbc, cutoff, wp_dtype, wp_device
        )

        print(
            f"\nTest params: num_atoms={num_atoms}, cutoff={cutoff}, pbc_flags={pbc_flags}"
        )
        print(f"  max_total_cells estimated: {max_total_cells}")
        print(f"  per_system_limit: {max_total_cells // num_systems}")

        # Allocate cell list arrays using warp
        (
            wp_cells_per_dimension,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
        ) = allocate_cell_list_wp(total_atoms, max_total_cells, num_systems, wp_device)

        # Allocate cell offsets and cells_per_system scratch buffer
        wp_cell_offsets = wp.zeros(num_systems, dtype=wp.int32, device=wp_device)
        wp_cells_per_system = wp.zeros(num_systems, dtype=wp.int32, device=wp_device)

        # Build cell list using warp launcher
        batch_build_cell_list(
            wp_positions,
            wp_cell,
            wp_pbc,
            cutoff,
            wp_idx,
            wp_cells_per_dimension,
            wp_cell_offsets,
            wp_cells_per_system,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
            wp_dtype,
            wp_device,
        )

        # Check that actual cells don't exceed allocated
        cells_per_dimension_np = wp_cells_per_dimension.numpy()
        actual_total_cells = 0
        for sys_idx in range(num_systems):
            sys_cells = int(np.prod(cells_per_dimension_np[sys_idx]))
            actual_total_cells += sys_cells
            print(
                f"  System {sys_idx}: {cells_per_dimension_np[sys_idx]} = {sys_cells} cells"
            )

        print(f"  actual_total_cells: {actual_total_cells}")
        assert actual_total_cells <= max_total_cells, (
            f"Actual cells {actual_total_cells} exceeds allocated {max_total_cells}"
        )

        # Query using the cell list
        estimated_density = num_atoms / cell_torch[0].det().abs().item()
        max_neighbors = estimate_max_neighbors(
            cutoff, atomic_density=estimated_density, safety_factor=5.0
        )

        neighbor_matrix_torch = torch.full(
            (total_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts_torch = torch.zeros(
            (total_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors_torch = torch.zeros(total_atoms, dtype=torch.int32, device=device)

        wp_neighbor_matrix = wp.from_torch(
            neighbor_matrix_torch, dtype=wp.int32, return_ctype=True
        )
        wp_neighbor_matrix_shifts = wp.from_torch(
            neighbor_matrix_shifts_torch, dtype=wp.vec3i, return_ctype=True
        )
        wp_num_neighbors = wp.from_torch(
            num_neighbors_torch, dtype=wp.int32, return_ctype=True
        )

        # Query cell list
        batch_query_cell_list(
            wp_positions,
            wp_cell,
            wp_pbc,
            cutoff,
            wp_idx,
            wp_cells_per_dimension,
            wp_neighbor_search_radius,
            wp_cell_offsets,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
            wp_neighbor_matrix,
            wp_neighbor_matrix_shifts,
            wp_num_neighbors,
            wp_dtype,
            wp_device,
            False,
        )

        # Verify we found neighbors
        num_neighbors_np = num_neighbors_torch.cpu().numpy()
        assert np.all(num_neighbors_np >= 0), "Neighbor counts should be non-negative"

        total_neighbors = np.sum(num_neighbors_np)
        print(f"  Found {total_neighbors} total neighbors")

        # Verify neighbors are within same system
        neighbor_matrix_np = neighbor_matrix_torch.cpu().numpy()
        idx_np = batch_idx_torch.cpu().numpy()
        for atom_idx in range(min(10, total_atoms)):
            sys_i = idx_np[atom_idx]
            for neigh_idx in range(min(5, num_neighbors_np[atom_idx])):
                atom_j = neighbor_matrix_np[atom_idx, neigh_idx]
                if atom_j == -1:
                    break
                sys_j = idx_np[atom_j]
                assert sys_i == sys_j, (
                    f"Atoms {atom_idx} and {atom_j} should be from same system"
                )

        # Check distances for a subset of neighbors
        neighbor_matrix_shifts_np = neighbor_matrix_shifts_torch.cpu().numpy()
        positions_np = positions_torch.cpu().numpy()
        cell_np = cell_torch.cpu().numpy()

        for atom_idx in range(min(5, total_atoms)):
            sys_idx = idx_np[atom_idx]
            for neigh_idx in range(min(5, num_neighbors_np[atom_idx])):
                atom_j = neighbor_matrix_np[atom_idx, neigh_idx]
                if atom_j == -1:
                    break

                shift = neighbor_matrix_shifts_np[atom_idx, neigh_idx]
                cartesian_shift = shift @ cell_np[sys_idx]
                rij = positions_np[atom_j] - positions_np[atom_idx] + cartesian_shift
                dist = np.linalg.norm(rij)
                assert dist < cutoff + 1e-5, (
                    f"Distance {dist} exceeds cutoff {cutoff} for atoms {atom_idx}->{atom_j}"
                )
