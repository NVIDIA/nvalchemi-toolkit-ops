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

from nvalchemiops.neighbors.cell_list import (
    batch_build_cell_list,
    batch_query_cell_list,
    get_build_cell_list_kernel,
)
from nvalchemiops.neighbors.neighbor_utils import empty_sentinel as _empty_sentinel
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

from .test_utils import (
    create_batch_systems,
    create_random_system,
    neighbor_matrix_row_set,
)

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


def _neighbor_shift_entries(
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
) -> set[tuple[int, int, tuple[int, int, int]]]:
    """Return order-independent matrix entries including shift vectors."""
    matrix_cpu = neighbor_matrix.detach().cpu()
    counts_cpu = num_neighbors.detach().cpu()
    shifts_cpu = neighbor_matrix_shifts.detach().cpu()
    entries: set[tuple[int, int, tuple[int, int, int]]] = set()
    for atom_idx in range(matrix_cpu.shape[0]):
        for slot in range(int(counts_cpu[atom_idx].item())):
            entries.add(
                (
                    atom_idx,
                    int(matrix_cpu[atom_idx, slot].item()),
                    tuple(int(x) for x in shifts_cpu[atom_idx, slot].tolist()),
                )
            )
    return entries


def _reciprocal_full_fill_entries(
    half_entries: set[tuple[int, int, tuple[int, int, int]]],
) -> set[tuple[int, int, tuple[int, int, int]]]:
    """Expand half-fill entries to the expected full-fill pair set."""
    full_entries = set(half_entries)
    for atom_i, atom_j, shift in half_entries:
        full_entries.add((atom_j, atom_i, tuple(-x for x in shift)))
    return full_entries


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
        get_build_cell_list_kernel("estimate_sizes", wp_dtype, batched=True),
        dim=num_systems,
        device=device,
        inputs=(
            cell,
            _empty_sentinel(1, wp.bool, device),
            pbc,
            wp_dtype(cutoff),
            max_nbins,
            number_of_cells,
            _empty_sentinel(1, wp.int32, device),
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
            get_build_cell_list_kernel("construct_bin_size", wp_dtype, batched=True),
            dim=num_systems,
            device=wp_device,
            inputs=(
                wp_cell,
                _empty_sentinel(1, wp.bool, wp_device),
                wp_pbc,
                _empty_sentinel(1, wp.int32, wp_device),
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
            get_build_cell_list_kernel("count_atoms", wp_dtype, batched=True),
            dim=total_atoms,
            device=wp_device,
            inputs=(
                wp_positions,
                wp_cell,
                _empty_sentinel(1, wp.bool, wp_device),
                wp_pbc,
                wp_idx,
                _empty_sentinel(1, wp.int32, wp_device),
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
            get_build_cell_list_kernel("count_atoms", wp_dtype, batched=True),
            dim=total_atoms,
            device=wp_device,
            inputs=(
                wp_positions,
                wp_cell,
                _empty_sentinel(1, wp.bool, wp_device),
                wp_pbc,
                wp_idx,
                _empty_sentinel(1, wp.int32, wp_device),
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
            get_build_cell_list_kernel("bin_atoms", wp_dtype, batched=True),
            dim=total_atoms,
            device=wp_device,
            inputs=(
                wp_positions,
                wp_cell,
                _empty_sentinel(1, wp.bool, wp_device),
                wp_pbc,
                wp_idx,
                _empty_sentinel(1, wp.int32, wp_device),
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

        # Sorted scratch + always-True rebuild flags (sorted-reads kernel
        # requires both).
        wp_sorted_pos = wp.zeros(total_atoms, dtype=wp_vec_dtype, device=wp_device)
        wp_sorted_shifts = wp.zeros(total_atoms, dtype=wp.vec3i, device=wp_device)
        wp_rebuild_flags = wp.full(
            (num_systems,), True, dtype=wp.bool, device=wp_device
        )

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
            sorted_positions=wp_sorted_pos,
            sorted_atom_periodic_shifts=wp_sorted_shifts,
            rebuild_flags=wp_rebuild_flags,
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


@pytest.mark.slow
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

        assert actual_total_cells <= max_total_cells, (
            f"Actual cells {actual_total_cells} exceeds allocated {max_total_cells}"
        )

        # Query using the cell list
        estimated_density = num_atoms / cell_torch[0].det().abs().item()
        max_neighbors = estimate_max_neighbors(
            cutoff, atomic_density=estimated_density * 5.0
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

        # Sorted scratch + always-True rebuild flags.
        wp_sorted_pos = wp.zeros(total_atoms, dtype=wp_vec_dtype, device=wp_device)
        wp_sorted_shifts = wp.zeros(total_atoms, dtype=wp.vec3i, device=wp_device)
        wp_rebuild_flags = wp.full(
            (num_systems,), True, dtype=wp.bool, device=wp_device
        )

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
            sorted_positions=wp_sorted_pos,
            sorted_atom_periodic_shifts=wp_sorted_shifts,
            rebuild_flags=wp_rebuild_flags,
        )

        # Verify we found neighbors
        num_neighbors_np = num_neighbors_torch.cpu().numpy()
        assert np.all(num_neighbors_np >= 0), "Neighbor counts should be non-negative"

        total_neighbors = np.sum(num_neighbors_np)
        assert total_neighbors > 0, "expected non-empty neighbor list"

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


@pytest.mark.parametrize("dtype", dtypes)
class TestBatchCellListSelectiveRebuildFlags:
    """Test selective rebuild (rebuild_flags) for batch cell list warp launchers."""

    def test_no_rebuild_preserves_data(self, device, dtype):
        """All flags False: neighbor data should remain unchanged for all systems."""
        positions_torch, cell_torch, pbc_torch, _ = create_batch_systems(
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

        wp_dtype = TORCH_TO_WP_DTYPE[dtype]
        wp_vec_dtype = TORCH_TO_WP_VEC_DTYPE[dtype]
        wp_mat_dtype = TORCH_TO_WP_MAT_DTYPE[dtype]
        wp_device = str(device)

        wp_positions = wp.from_torch(
            positions_torch, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_cell = wp.from_torch(cell_torch, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(pbc_torch, dtype=wp.bool, return_ctype=True)
        wp_idx = wp.from_torch(idx_torch, dtype=wp.int32, return_ctype=True)

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

        max_neighbors = 20
        nm_torch = torch.full(
            (total_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        nm_shifts_torch = torch.zeros(
            (total_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        nn_torch = torch.zeros(total_atoms, dtype=torch.int32, device=device)
        wp_nm = wp.from_torch(nm_torch, dtype=wp.int32, return_ctype=True)
        wp_nm_shifts = wp.from_torch(nm_shifts_torch, dtype=wp.vec3i)
        wp_nn = wp.from_torch(nn_torch, dtype=wp.int32, return_ctype=True)

        wp_sorted_pos = wp.zeros(total_atoms, dtype=wp_vec_dtype, device=wp_device)
        wp_sorted_shifts = wp.zeros(total_atoms, dtype=wp.vec3i, device=wp_device)
        wp_rebuild_flags_full = wp.full(
            (num_systems,), True, dtype=wp.bool, device=wp_device
        )

        batch_query_cell_list(
            positions=wp_positions,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=cutoff,
            batch_idx=wp_idx,
            cells_per_dimension=wp_cells_per_dimension,
            neighbor_search_radius=wp_neighbor_search_radius,
            cell_offsets=wp_cell_offsets,
            atom_periodic_shifts=wp_atom_periodic_shifts,
            atom_to_cell_mapping=wp_atom_to_cell_mapping,
            atoms_per_cell_count=wp_atoms_per_cell_count,
            cell_atom_start_indices=wp_cell_atom_start_indices,
            cell_atom_list=wp_cell_atom_list,
            sorted_positions=wp_sorted_pos,
            sorted_atom_periodic_shifts=wp_sorted_shifts,
            neighbor_matrix=wp_nm,
            neighbor_matrix_shifts=wp_nm_shifts,
            num_neighbors=wp_nn,
            rebuild_flags=wp_rebuild_flags_full,
            wp_dtype=wp_dtype,
            device=wp_device,
            half_fill=False,
        )

        saved_nm = nm_torch.clone()
        saved_nn = nn_torch.clone()

        # Selective rebuild with all flags=False: data should be unchanged
        rebuild_flags = torch.zeros(num_systems, dtype=torch.bool, device=device)
        wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool)

        batch_query_cell_list(
            positions=wp_positions,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=cutoff,
            batch_idx=wp_idx,
            cells_per_dimension=wp_cells_per_dimension,
            neighbor_search_radius=wp_neighbor_search_radius,
            cell_offsets=wp_cell_offsets,
            atom_periodic_shifts=wp_atom_periodic_shifts,
            atom_to_cell_mapping=wp_atom_to_cell_mapping,
            atoms_per_cell_count=wp_atoms_per_cell_count,
            cell_atom_start_indices=wp_cell_atom_start_indices,
            cell_atom_list=wp_cell_atom_list,
            sorted_positions=wp_sorted_pos,
            sorted_atom_periodic_shifts=wp_sorted_shifts,
            neighbor_matrix=wp_nm,
            neighbor_matrix_shifts=wp_nm_shifts,
            num_neighbors=wp_nn,
            rebuild_flags=wp_rebuild_flags,
            wp_dtype=wp_dtype,
            device=wp_device,
            half_fill=False,
        )

        assert torch.equal(nn_torch, saved_nn), (
            "num_neighbors must be unchanged when all rebuild_flags are False"
        )
        for i in range(total_atoms):
            n = nn_torch[i].item()
            assert torch.equal(nm_torch[i, :n], saved_nm[i, :n]), (
                f"neighbor_matrix row {i} should be unchanged"
            )

    def test_rebuild_updates_data(self, device, dtype):
        """True flags: rebuilt system data should match a fresh full rebuild."""
        positions_torch, cell_torch, pbc_torch, _ = create_batch_systems(
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

        wp_dtype = TORCH_TO_WP_DTYPE[dtype]
        wp_vec_dtype = TORCH_TO_WP_VEC_DTYPE[dtype]
        wp_mat_dtype = TORCH_TO_WP_MAT_DTYPE[dtype]
        wp_device = str(device)

        wp_positions = wp.from_torch(
            positions_torch, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_cell = wp.from_torch(cell_torch, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(pbc_torch, dtype=wp.bool, return_ctype=True)
        wp_idx = wp.from_torch(idx_torch, dtype=wp.int32, return_ctype=True)

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

        max_neighbors = 20

        # Reference: full build
        nm_ref = torch.full(
            (total_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        nm_ref_shifts = torch.zeros(
            (total_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        nn_ref = torch.zeros(total_atoms, dtype=torch.int32, device=device)
        wp_nm_ref = wp.from_torch(nm_ref, dtype=wp.int32, return_ctype=True)
        wp_nm_ref_shifts = wp.from_torch(nm_ref_shifts, dtype=wp.vec3i)
        wp_nn_ref = wp.from_torch(nn_ref, dtype=wp.int32, return_ctype=True)

        wp_sorted_pos = wp.zeros(total_atoms, dtype=wp_vec_dtype, device=wp_device)
        wp_sorted_shifts = wp.zeros(total_atoms, dtype=wp.vec3i, device=wp_device)
        wp_rebuild_flags_full = wp.full(
            (num_systems,), True, dtype=wp.bool, device=wp_device
        )

        batch_query_cell_list(
            positions=wp_positions,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=cutoff,
            batch_idx=wp_idx,
            cells_per_dimension=wp_cells_per_dimension,
            neighbor_search_radius=wp_neighbor_search_radius,
            cell_offsets=wp_cell_offsets,
            atom_periodic_shifts=wp_atom_periodic_shifts,
            atom_to_cell_mapping=wp_atom_to_cell_mapping,
            atoms_per_cell_count=wp_atoms_per_cell_count,
            cell_atom_start_indices=wp_cell_atom_start_indices,
            cell_atom_list=wp_cell_atom_list,
            sorted_positions=wp_sorted_pos,
            sorted_atom_periodic_shifts=wp_sorted_shifts,
            neighbor_matrix=wp_nm_ref,
            neighbor_matrix_shifts=wp_nm_ref_shifts,
            num_neighbors=wp_nn_ref,
            rebuild_flags=wp_rebuild_flags_full,
            wp_dtype=wp_dtype,
            device=wp_device,
            half_fill=False,
        )

        # Selective rebuild with all flags=True
        nm_sel = torch.full(
            (total_atoms, max_neighbors), 99, dtype=torch.int32, device=device
        )
        nm_sel_shifts = torch.zeros(
            (total_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        nn_sel = torch.full((total_atoms,), 99, dtype=torch.int32, device=device)
        wp_nm_sel = wp.from_torch(nm_sel, dtype=wp.int32, return_ctype=True)
        wp_nm_sel_shifts = wp.from_torch(nm_sel_shifts, dtype=wp.vec3i)
        wp_nn_sel = wp.from_torch(nn_sel, dtype=wp.int32, return_ctype=True)

        rebuild_flags = torch.ones(num_systems, dtype=torch.bool, device=device)
        wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool)

        batch_query_cell_list(
            positions=wp_positions,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=cutoff,
            batch_idx=wp_idx,
            cells_per_dimension=wp_cells_per_dimension,
            neighbor_search_radius=wp_neighbor_search_radius,
            cell_offsets=wp_cell_offsets,
            atom_periodic_shifts=wp_atom_periodic_shifts,
            atom_to_cell_mapping=wp_atom_to_cell_mapping,
            atoms_per_cell_count=wp_atoms_per_cell_count,
            cell_atom_start_indices=wp_cell_atom_start_indices,
            cell_atom_list=wp_cell_atom_list,
            sorted_positions=wp_sorted_pos,
            sorted_atom_periodic_shifts=wp_sorted_shifts,
            neighbor_matrix=wp_nm_sel,
            neighbor_matrix_shifts=wp_nm_sel_shifts,
            num_neighbors=wp_nn_sel,
            rebuild_flags=wp_rebuild_flags,
            wp_dtype=wp_dtype,
            device=wp_device,
            half_fill=False,
        )

        assert torch.equal(nn_sel, nn_ref), (
            "num_neighbors should match full rebuild when all flags=True"
        )


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("pbc_flag", [False, True])
class TestBatchCellListFullFillDirect:
    """Regression coverage for direct atom-centric full-fill output."""

    def test_full_fill_matches_half_fill_reciprocal_expansion(
        self, device, dtype, pbc_flag
    ):
        """Full-fill direct output should equal half-fill plus reciprocal rows."""
        from nvalchemiops.torch.neighbors.batch_cell_list import batch_cell_list

        atoms_per_system = [17, 19]
        positions, cell, pbc, _ptr = create_batch_systems(
            num_systems=2,
            atoms_per_system=atoms_per_system,
            cell_sizes=[8.0, 8.0],
            dtype=dtype,
            device=device,
            seed=7,
            pbc_flag=pbc_flag,
        )
        batch_idx = torch.repeat_interleave(
            torch.arange(2, dtype=torch.int32, device=device),
            torch.tensor(atoms_per_system, dtype=torch.int32, device=device),
        )
        kwargs = {
            "cell": cell,
            "pbc": pbc,
            "batch_idx": batch_idx,
            "max_neighbors": 256,
            "fill_value": positions.shape[0],
            "return_neighbor_list": False,
            "strategy": "atom_centric",
        }

        full_matrix, full_counts, full_shifts = batch_cell_list(
            positions,
            3.2,
            half_fill=False,
            **kwargs,
        )
        half_matrix, half_counts, half_shifts = batch_cell_list(
            positions,
            3.2,
            half_fill=True,
            **kwargs,
        )

        full_entries = _neighbor_shift_entries(full_matrix, full_counts, full_shifts)
        expected_entries = _reciprocal_full_fill_entries(
            _neighbor_shift_entries(half_matrix, half_counts, half_shifts)
        )

        assert full_entries == expected_entries


@pytest.mark.parametrize("dtype", dtypes)
class TestBatchCellListPairCentric:
    """Parity tests for the batch pair-centric query kernel.

    Forces the pair-centric path via the
    ``NVALCHEMI_NEIGHLIST_BATCH_PAIR_*`` env vars and compares its
    output against the atom-centric reference.
    """

    def test_dispatch_threshold_env(self, device, dtype, monkeypatch):
        """Env-var overrides change the strategy decision."""
        del device, dtype  # not used; fixture present for parametrization
        from nvalchemiops.torch.neighbors.batch_cell_list import (
            select_batch_cell_list_strategy,
        )

        # High cutoff still picks pair-centric when each system is large enough.
        assert (
            select_batch_cell_list_strategy(
                total_atoms=1_000_000, num_systems=64, cutoff=12.0
            )
            == "pair_centric"
        )

        # High-cutoff many-small-system batches stay atom-centric.
        assert (
            select_batch_cell_list_strategy(
                total_atoms=1001, num_systems=60, cutoff=15.0
            )
            == "atom_centric"
        )
        assert (
            select_batch_cell_list_strategy(
                total_atoms=100_001, num_systems=5365, cutoff=15.0
            )
            == "atom_centric"
        )

        # cutoff=6, large total, many systems → atom-centric.
        assert (
            select_batch_cell_list_strategy(
                total_atoms=1_000_000, num_systems=64, cutoff=6.0
            )
            == "atom_centric"
        )

        # cutoff=6, few large systems (total > 65k cap) → pair (clause 3).
        assert (
            select_batch_cell_list_strategy(
                total_atoms=1_000_000, num_systems=8, cutoff=6.0
            )
            == "pair_centric"
        )

        # cutoff=6, few small systems (total ≤ 65k cap, avg_aps < 4096
        # floor) → atom: clause 3 is gated by total_atoms > total_cap to
        # avoid setup overhead.
        assert (
            select_batch_cell_list_strategy(
                total_atoms=4_096, num_systems=4, cutoff=6.0
            )
            == "atom_centric"
        )

        # cutoff=6, total below cap AND avg_aps reasonable → pair (clause 2).
        assert (
            select_batch_cell_list_strategy(
                total_atoms=32_768, num_systems=8, cutoff=6.0
            )
            == "pair_centric"
        )

        # cutoff=6, total below cap but avg_aps too small → atom (clause 2 fails).
        assert (
            select_batch_cell_list_strategy(
                total_atoms=32_768, num_systems=128, cutoff=6.0
            )
            == "atom_centric"
        )

        # Lifting all clauses disables pair-centric entirely.
        monkeypatch.setenv("NVALCHEMI_NEIGHLIST_BATCH_PAIR_CUTOFF_FLOOR", "20.0")
        monkeypatch.setenv("NVALCHEMI_NEIGHLIST_BATCH_PAIR_TOTAL_CAP", "0")
        monkeypatch.setenv("NVALCHEMI_NEIGHLIST_BATCH_PAIR_NSYS_CAP", "0")
        assert (
            select_batch_cell_list_strategy(
                total_atoms=1_000_000, num_systems=64, cutoff=12.0
            )
            == "atom_centric"
        )

    def test_pair_set_matches_atom_centric(self, device, dtype, monkeypatch):
        """Same pair set as atom-centric for a small batch with PBC."""
        from nvalchemiops.torch.neighbors.batch_cell_list import batch_cell_list

        if str(device) == "cpu":
            pytest.skip(
                "strategy='pair_centric' uses CUDA block scheduling; "
                "CPU parameter is not supported"
            )

        positions, cell, pbc, _ptr = create_batch_systems(
            num_systems=4,
            atoms_per_system=[64, 64, 64, 64],
            cell_sizes=[2.0, 2.0, 2.0, 2.0],
            dtype=dtype,
            device=device,
            pbc_flag=True,
        )
        cutoff = 0.6
        batch_idx = torch.repeat_interleave(
            torch.arange(4, dtype=torch.int32, device=device),
            torch.tensor([64] * 4, dtype=torch.int32, device=device),
        )

        # Force atom-centric.
        monkeypatch.setenv("NVALCHEMI_NEIGHLIST_BATCH_PAIR_CUTOFF_FLOOR", "999")
        monkeypatch.setenv("NVALCHEMI_NEIGHLIST_BATCH_PAIR_TOTAL_CAP", "0")
        monkeypatch.setenv("NVALCHEMI_NEIGHLIST_BATCH_PAIR_NSYS_CAP", "0")
        nm_a, nn_a, _ = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            half_fill=True,
        )

        # Force pair-centric.
        monkeypatch.setenv("NVALCHEMI_NEIGHLIST_BATCH_PAIR_CUTOFF_FLOOR", "0")
        monkeypatch.setenv("NVALCHEMI_NEIGHLIST_BATCH_PAIR_TOTAL_CAP", "999999999")
        monkeypatch.setenv("NVALCHEMI_NEIGHLIST_BATCH_PAIR_NSYS_CAP", "999999999")
        nm_p, nn_p, _ = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            half_fill=True,
        )

        assert torch.equal(nn_a, nn_p), (
            "Per-atom neighbor counts must match between batch kernels"
        )
        assert neighbor_matrix_row_set(nm_a, nn_a) == neighbor_matrix_row_set(
            nm_p, nn_p
        ), "Pair sets must match between batch kernels (row order may differ)"
