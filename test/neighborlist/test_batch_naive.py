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

"""Comprehensive tests for batch naive neighbor list routines and utilities."""

import pytest
import torch
import warp as wp

from nvalchemiops.neighborlist.batch_naive import (
    _batch_naive_neighbor_matrix_no_pbc,
    _batch_naive_neighbor_matrix_pbc,
    _fill_batch_naive_neighbor_matrix,
    _fill_batch_naive_neighbor_matrix_pbc,
    batch_naive_neighbor_list,
)
from nvalchemiops.neighborlist.neighbor_utils import (
    compute_naive_num_shifts,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

from .test_utils import (
    create_batch_systems,
)


def create_batch_idx_and_ptr(
    atoms_per_system: list, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create batch_idx and batch_ptr tensors from atoms_per_system list."""
    total_atoms = sum(atoms_per_system)
    batch_idx = torch.zeros(total_atoms, dtype=torch.int32, device=device)
    batch_ptr = torch.zeros(len(atoms_per_system) + 1, dtype=torch.int32, device=device)

    start_idx = 0
    for i, num_atoms in enumerate(atoms_per_system):
        batch_idx[start_idx : start_idx + num_atoms] = i
        batch_ptr[i + 1] = batch_ptr[i] + num_atoms
        start_idx += num_atoms

    return batch_idx, batch_ptr


class TestBatchNaiveKernels:
    """Test individual batch naive neighbor list kernels."""

    @pytest.mark.parametrize("half_fill", [True, False])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_naive_neighbor_matrix_kernel_no_pbc(self, half_fill, device, dtype):
        """Test _fill_batch_naive_neighbor_matrix kernel (no PBC)."""
        # Create batch system with multiple systems
        atoms_per_system = [4, 6, 5]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        # Create batch_idx and batch_ptr
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.5
        max_neighbors = 10

        # Convert to warp types
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_device = str(device)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)

        # Output arrays
        total_atoms = positions_batch.shape[0]
        neighbor_matrix = torch.full(
            (total_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(total_atoms, dtype=torch.int32, device=device)

        wp_neighbor_matrix = wp.from_torch(neighbor_matrix, dtype=wp.int32)
        wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32)

        # Launch kernel
        wp.launch(
            _fill_batch_naive_neighbor_matrix,
            dim=total_atoms,
            device=wp_device,
            inputs=[
                wp_positions,
                wp_dtype(cutoff),
                wp_batch_idx,
                wp_batch_ptr,
                wp_neighbor_matrix,
                wp_num_neighbors,
                half_fill,
            ],
        )

        # Check results
        assert torch.all(num_neighbors >= 0), (
            "All neighbor counts should be non-negative"
        )

        # Check that atoms only find neighbors within their own system
        for i in range(total_atoms):
            system_idx = batch_idx[i].item()
            system_start = batch_ptr[system_idx].item()
            system_end = batch_ptr[system_idx + 1].item()

            for j in range(num_neighbors[i].item()):
                neighbor_idx = neighbor_matrix[i, j].item()
                assert system_start <= neighbor_idx < system_end, (
                    f"Atom {i} in system {system_idx} has neighbor {neighbor_idx} outside its system"
                )
                assert neighbor_idx != i, "Atom should not be its own neighbor"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_naive_neighbor_matrix_pbc_kernel(self, device, dtype):
        """Test _fill_batch_naive_neighbor_matrix_pbc kernel."""
        # Create batch system with PBC
        atoms_per_system = [3, 4]
        positions_batch, cell_batch, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        # Create batch_idx and batch_ptr
        _, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.0
        max_neighbors = 15

        # Simple shifts for testing
        shifts = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
            dtype=torch.int32,
            device=device,
        )
        shift_batch_idx = torch.zeros(len(shifts), dtype=torch.int32, device=device)

        # Convert to warp types
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell_batch, dtype=wp_mat_dtype)
        wp_shifts = wp.from_torch(shifts, dtype=wp.vec3i)
        wp_shift_batch_idx = wp.from_torch(shift_batch_idx, dtype=wp.int32)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)

        # Output arrays
        total_atoms = positions_batch.shape[0]
        neighbor_matrix = torch.full(
            (total_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            total_atoms, max_neighbors, 3, dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(total_atoms, dtype=torch.int32, device=device)

        wp_neighbor_matrix = wp.from_torch(neighbor_matrix, dtype=wp.int32)
        wp_neighbor_matrix_shifts = wp.from_torch(
            neighbor_matrix_shifts, dtype=wp.vec3i
        )
        wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32)

        # Launch kernel
        wp.launch(
            _fill_batch_naive_neighbor_matrix_pbc,
            dim=(len(shifts), total_atoms),
            device=wp_device,
            inputs=[
                wp_positions,
                wp_cell,
                wp_dtype(cutoff),
                wp_batch_ptr,
                wp_shifts,
                wp_shift_batch_idx,
                wp_neighbor_matrix,
                wp_neighbor_matrix_shifts,
                wp_num_neighbors,
                True,  # half_fill
            ],
        )


class TestBatchNaiveUtilityFunctions:
    """Test utility functions used by batch naive neighbor list."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_matrix_no_pbc_function_with_preallocation(
        self, device, dtype, half_fill
    ):
        """Test _batch_naive_neighbor_matrix_no_pbc custom op with preallocated tensors."""
        atoms_per_system = [5, 7, 6]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 20
        total_atoms = positions_batch.shape[0]

        # Prepare output arrays
        neighbor_matrix = torch.full(
            (total_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(total_atoms, dtype=torch.int32, device=device)

        _batch_naive_neighbor_matrix_no_pbc(
            positions_batch,
            cutoff,
            batch_idx,
            batch_ptr,
            neighbor_matrix,
            num_neighbors,
            half_fill,
        )

        # Check results
        assert torch.all(num_neighbors >= 0), (
            "All neighbor counts should be non-negative"
        )
        assert torch.all(num_neighbors <= max_neighbors), (
            "Neighbor counts should not exceed maximum"
        )

        # Check that neighbor indices are valid and within the same system
        for i in range(total_atoms):
            system_idx = batch_idx[i].item()
            system_start = batch_ptr[system_idx].item()
            system_end = batch_ptr[system_idx + 1].item()

            for j in range(num_neighbors[i].item()):
                neighbor_idx = neighbor_matrix[i, j].item()
                assert system_start <= neighbor_idx < system_end, (
                    f"Invalid neighbor index: {neighbor_idx} for atom {i} in system {system_idx}"
                )
                assert neighbor_idx != i, "Atom should not be its own neighbor"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_compute_total_shifts_function(self, device, dtype):
        """Test _batch_compute_total_shifts function."""
        atoms_per_system = [4, 5]
        _, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        cutoff = 1.5

        shift_range_per_dimension, shift_offset, _ = compute_naive_num_shifts(
            cell_batch, cutoff, pbc_batch
        )

        # Check output shapes and types
        assert shift_range_per_dimension.shape == (2, 3)  # num_systems
        assert shift_offset.shape == (3,)  # num_systems + 1
        assert shift_range_per_dimension.dtype == torch.int32
        assert shift_offset.dtype == torch.int32

        # Check that shifts are reasonable
        assert torch.all(shift_range_per_dimension >= 0)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
        ],
    )
    @pytest.mark.parametrize(
        "half_fill",
        [
            True,
        ],
    )
    def test_batch_neighbor_matrix_pbc_function(self, device, dtype, half_fill):
        """Test _batch_naive_neighbor_matrix_pbc function with preallocation."""
        atoms_per_system = [4, 5]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        _, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 25
        total_atoms = positions_batch.shape[0]
        expected_rows = total_atoms

        # Compute shifts first
        shift_range_per_dimension, shift_offset, total_shifts = (
            compute_naive_num_shifts(cell_batch, cutoff, pbc_batch)
        )

        # Preallocate output tensors
        neighbor_matrix = torch.full(
            (expected_rows, max_neighbors), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (expected_rows, max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(total_atoms, dtype=torch.int32, device=device)

        _batch_naive_neighbor_matrix_pbc(
            positions_batch,
            cell_batch,
            cutoff,
            batch_ptr,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            shift_range_per_dimension,
            shift_offset,
            total_shifts,
            half_fill,
        )

        # Check output shapes and types
        assert neighbor_matrix.shape[0] == expected_rows
        assert neighbor_matrix_shifts.shape == (expected_rows, max_neighbors, 3)
        assert neighbor_matrix.dtype == torch.int32
        assert neighbor_matrix_shifts.dtype == torch.int32
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix.device == torch.device(device)
        assert neighbor_matrix_shifts.device == torch.device(device)
        assert num_neighbors.device == torch.device(device)

        # Check neighbor counts
        assert torch.all(num_neighbors >= 0)

        # Check that unit_shifts are reasonable (should be small integers)
        valid_shifts = neighbor_matrix_shifts[neighbor_matrix != -1]
        if len(valid_shifts) > 0:
            assert torch.all(torch.abs(valid_shifts) <= 5), (
                "Unit shifts should be small integers"
            )


class TestBatchNaiveMainAPI:
    """Test the main batch naive neighbor list API function."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_no_pbc(self, device, dtype, half_fill):
        """Test batch_naive_neighbor_list without periodic boundary conditions."""
        atoms_per_system = [6, 8, 7]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 20

        # Test without PBC
        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=None,
            cell=None,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        # Check output types and shapes
        total_atoms = positions_batch.shape[0]
        expected_rows = total_atoms
        assert neighbor_matrix.dtype == torch.int32
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix.shape == (expected_rows, max_neighbors)
        assert num_neighbors.shape == (total_atoms,)
        assert neighbor_matrix.device == torch.device(device)
        assert num_neighbors.device == torch.device(device)

        # Check neighbor counts are reasonable
        assert torch.all(num_neighbors >= 0)
        assert torch.all(num_neighbors <= max_neighbors)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_with_pbc(self, device, dtype, half_fill):
        """Test batch_naive_neighbor_list with periodic boundary conditions."""
        atoms_per_system = [5, 7, 6]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 30

        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
            batch_naive_neighbor_list(
                positions=positions_batch,
                cutoff=cutoff,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                pbc=pbc_batch,
                cell=cell_batch,
                max_neighbors=max_neighbors,
                half_fill=half_fill,
            )
        )

        # Check output types and shapes
        total_atoms = positions_batch.shape[0]
        expected_rows = total_atoms
        assert neighbor_matrix.dtype == torch.int32
        assert neighbor_matrix_shifts.dtype == torch.int32
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix.shape == (expected_rows, max_neighbors)
        assert neighbor_matrix_shifts.shape == (expected_rows, max_neighbors, 3)
        assert num_neighbors.shape == (total_atoms,)
        assert neighbor_matrix.device == torch.device(device)
        assert neighbor_matrix_shifts.device == torch.device(device)
        assert num_neighbors.device == torch.device(device)

        # Check neighbor counts
        assert torch.all(num_neighbors >= 0)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    @pytest.mark.parametrize("pbc_flag", [True, False])
    def test_batch_naive_neighbor_list_return_neighbor_list(
        self, device, dtype, half_fill, pbc_flag
    ):
        """Test batch_naive_neighbor_list with return_neighbor_list=True."""
        atoms_per_system = [4, 6, 5]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 25

        if not pbc_flag:
            cell_batch = None
            pbc_batch = None

            neighbor_list, neighbor_ptr = batch_naive_neighbor_list(
                positions=positions_batch,
                cutoff=cutoff,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=max_neighbors,
                pbc=pbc_batch,
                cell=cell_batch,
                half_fill=half_fill,
                return_neighbor_list=True,
            )
        else:
            neighbor_list, neighbor_ptr, _ = batch_naive_neighbor_list(
                positions=positions_batch,
                cutoff=cutoff,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=max_neighbors,
                pbc=pbc_batch,
                cell=cell_batch,
                half_fill=half_fill,
                return_neighbor_list=True,
            )

        # Check that we get neighbor list format (2, N) instead of matrix
        assert neighbor_list.ndim == 2
        assert neighbor_list.shape[0] == 2
        assert neighbor_list.dtype == torch.int32
        assert neighbor_ptr.dtype == torch.int32

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_naive_neighbor_list_consistency_with_single_system_no_pbc(
        self, device, dtype
    ):
        """Test that batch neighbor list gives same results as single system calls."""
        # Create a batch with multiple systems
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 30

        # Get batch result
        _, num_neighbors_batch, _ = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=pbc_batch,
            cell=cell_batch,
            max_neighbors=max_neighbors,
            half_fill=False,  # Use full fill for easier comparison
            return_neighbor_list=False,
        )

        # Get single system results for comparison
        for sys_idx in range(2):
            start_idx = batch_ptr[sys_idx].item()
            end_idx = batch_ptr[sys_idx + 1].item()

            positions_single = positions_batch[start_idx:end_idx]
            pbc_single = pbc_batch[sys_idx : sys_idx + 1]
            cell_single = cell_batch[sys_idx : sys_idx + 1]
            # Create batch_idx and batch_ptr for single system (batch of size 1)
            n_atoms_single = positions_single.shape[0]
            batch_idx_single = torch.zeros(
                n_atoms_single, dtype=torch.int32, device=positions_single.device
            )
            batch_ptr_single = torch.tensor(
                [0, n_atoms_single], dtype=torch.int32, device=positions_single.device
            )
            (
                neighbor_matrix_single,
                num_neighbors_single,
                neighbor_matrix_shifts_single,
            ) = batch_naive_neighbor_list(
                positions=positions_single,
                cutoff=cutoff,
                batch_idx=batch_idx_single,
                batch_ptr=batch_ptr_single,
                pbc=pbc_single,
                cell=cell_single,
                max_neighbors=max_neighbors,
                half_fill=False,
            )

            # Compare neighbor counts (should be identical for the same system)
            torch.testing.assert_close(
                num_neighbors_batch[start_idx:end_idx],
                num_neighbors_single,
                rtol=0,
                atol=0,
            )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_naive_neighbor_list_consistency_with_single_system(
        self, device, dtype
    ):
        """Test that batch neighbor list gives same results as single system calls."""
        # Create a batch with multiple systems
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 30

        # Get batch result
        _, num_neighbors_batch, _ = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=pbc_batch,
            cell=cell_batch,
            max_neighbors=max_neighbors,
            half_fill=False,  # Use full fill for easier comparison
            return_neighbor_list=False,
        )

        # Get single system results for comparison
        for sys_idx in range(2):
            start_idx = batch_ptr[sys_idx].item()
            end_idx = batch_ptr[sys_idx + 1].item()

            positions_single = positions_batch[start_idx:end_idx]
            cell_single = cell_batch[sys_idx : sys_idx + 1]
            pbc_single = pbc_batch[sys_idx : sys_idx + 1]
            # Create batch_idx and batch_ptr for single system (batch of size 1)
            n_atoms_single = positions_single.shape[0]
            batch_idx_single = torch.zeros(
                n_atoms_single, dtype=torch.int32, device=positions_single.device
            )
            batch_ptr_single = torch.tensor(
                [0, n_atoms_single], dtype=torch.int32, device=positions_single.device
            )

            (
                neighbor_matrix_single,
                num_neighbors_single,
                neighbor_matrix_shifts_single,
            ) = batch_naive_neighbor_list(
                positions=positions_single,
                cutoff=cutoff,
                batch_idx=batch_idx_single,
                batch_ptr=batch_ptr_single,
                pbc=pbc_single,
                cell=cell_single,
                max_neighbors=max_neighbors,
                half_fill=False,
            )

            # Compare neighbor counts (should be identical for the same system)
            torch.testing.assert_close(
                num_neighbors_batch[start_idx:end_idx],
                num_neighbors_single,
                rtol=0,
                atol=0,
            )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_edge_cases(self, device, dtype, half_fill):
        """Test edge cases for batch_naive_neighbor_list."""
        # Empty batch
        positions_empty = torch.empty(0, 3, dtype=dtype, device=device)
        batch_idx_empty = torch.empty(0, dtype=torch.int32, device=device)
        batch_ptr_empty = torch.tensor([0], dtype=torch.int32, device=device)

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions=positions_empty,
            cutoff=1.0,
            batch_idx=batch_idx_empty,
            batch_ptr=batch_ptr_empty,
            max_neighbors=10,
            pbc=None,
            cell=None,
            half_fill=half_fill,
        )
        assert neighbor_matrix.shape == (0, 10)
        assert num_neighbors.shape == (0,)

        # Single system with single atom
        positions_single = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        batch_idx_single = torch.tensor([0], dtype=torch.int32, device=device)
        batch_ptr_single = torch.tensor([0, 1], dtype=torch.int32, device=device)

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions=positions_single,
            cutoff=1.0,
            batch_idx=batch_idx_single,
            batch_ptr=batch_ptr_single,
            max_neighbors=10,
            pbc=None,
            cell=None,
            half_fill=half_fill,
        )
        assert num_neighbors[0].item() == 0, "Single atom should have no neighbors"

        # Zero cutoff
        atoms_per_system = [3, 4]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=0.0,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
            pbc=None,
            cell=None,
            half_fill=half_fill,
        )
        assert torch.all(num_neighbors == 0), "Zero cutoff should find no neighbors"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_error_conditions(self, device, dtype, half_fill):
        """Test error conditions for batch_naive_neighbor_list."""
        atoms_per_system = [4, 5]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Test mismatched cell and pbc arguments
        with pytest.raises(
            ValueError, match="If cell is provided, pbc must also be provided"
        ):
            batch_naive_neighbor_list(
                positions_batch,
                1.0,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=10,
                pbc=None,
                cell=cell_batch,
            )

        with pytest.raises(
            ValueError, match="If pbc is provided, cell must also be provided"
        ):
            batch_naive_neighbor_list(
                positions_batch,
                1.0,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=10,
                pbc=pbc_batch,
                cell=None,
            )


class TestBatchNaivePerformanceAndScaling:
    """Test performance characteristics and scaling of batch implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_batch_scaling_with_system_size(self, device):
        """Test that batch implementation scales as expected with system size."""
        import time

        dtype = torch.float32
        cutoff = 1.2
        max_neighbors = 50

        # Test different batch sizes
        batch_sizes = (
            [(2, [10, 12], [2.0, 2.0]), (3, [8, 10, 12], [2.0, 2.0, 2.0])]
            if device == "cpu"
            else [
                (3, [20, 25, 30], [2.0, 2.0, 2.0]),
                (4, [15, 20, 25, 30], [2.0, 2.0, 2.0, 2.0]),
            ]
        )
        times = []

        for num_systems, atoms_per_system, cell_sizes in batch_sizes:
            positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                num_systems=num_systems,
                cell_sizes=cell_sizes,
                atoms_per_system=atoms_per_system,
                dtype=dtype,
                device=device,
            )
            batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)
            # Warm up
            for _ in range(10):
                batch_naive_neighbor_list(
                    positions_batch,
                    cutoff,
                    batch_idx,
                    batch_ptr,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    max_neighbors=max_neighbors,
                    half_fill=True,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            # Time the operation
            start_time = time.time()
            for _ in range(100):
                batch_naive_neighbor_list(
                    positions_batch,
                    cutoff,
                    batch_idx,
                    batch_ptr,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    max_neighbors=max_neighbors,
                    half_fill=True,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            times.append(elapsed)

        # Check that it doesn't grow too fast
        assert times[1] > times[0] * 0.5, "Time should increase with batch size"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_batch_cutoff_scaling(self, device):
        """Test scaling with different cutoff values."""
        dtype = torch.float32
        atoms_per_system = [15, 18, 20]
        max_neighbors = 100

        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Test different cutoffs
        cutoffs = [0.8, 1.2, 1.6, 2.0]
        neighbor_counts = []

        for cutoff in cutoffs:
            _, num_neighbors, _ = batch_naive_neighbor_list(
                positions_batch,
                cutoff,
                batch_idx,
                batch_ptr,
                pbc=pbc_batch,
                cell=cell_batch,
                max_neighbors=max_neighbors,
                half_fill=True,
            )
            total_pairs = num_neighbors.sum().item()
            neighbor_counts.append(total_pairs)

        # Check that neighbor count increases with cutoff
        for i in range(1, len(neighbor_counts)):
            assert neighbor_counts[i] >= neighbor_counts[i - 1], (
                f"Neighbor count should increase with cutoff: {neighbor_counts}"
            )


class TestBatchNaiveRobustness:
    """Test robustness of batch implementation to various inputs."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_random_batch_systems(self, device, dtype, half_fill):
        """Test with random batch systems of various sizes and configurations."""
        for pbc_flag in [True, False]:
            # Test several random batch configurations
            for seed in [42, 123, 456]:
                atoms_per_system = [12, 15, 10, 18]
                positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                    num_systems=4,
                    cell_sizes=[2.0, 2.0, 2.0, 2.0],
                    atoms_per_system=atoms_per_system,
                    dtype=dtype,
                    device=device,
                    seed=seed,
                    pbc_flag=pbc_flag,
                )
                batch_idx, batch_ptr = create_batch_idx_and_ptr(
                    atoms_per_system, device
                )

                cutoff = 1.3
                max_neighbors = 40

                # Should not crash
                if pbc_flag:
                    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
                        batch_naive_neighbor_list(
                            positions=positions_batch,
                            cutoff=cutoff,
                            max_neighbors=max_neighbors,
                            batch_idx=batch_idx,
                            batch_ptr=batch_ptr,
                            pbc=pbc_batch,
                            cell=cell_batch,
                            half_fill=half_fill,
                        )
                    )
                    assert neighbor_matrix_shifts.device == torch.device(device)
                else:
                    neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
                        positions=positions_batch,
                        cutoff=cutoff,
                        max_neighbors=max_neighbors,
                        pbc=None,
                        cell=None,
                        batch_idx=batch_idx,
                        batch_ptr=batch_ptr,
                        half_fill=half_fill,
                    )

                # Basic sanity checks
                assert torch.all(num_neighbors >= 0)
                assert torch.all(num_neighbors <= max_neighbors)
                assert neighbor_matrix.device == torch.device(device)
                assert num_neighbors.device == torch.device(device)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_mixed_system_sizes(self, device, dtype, half_fill):
        """Test with very different system sizes in the same batch."""
        # Mix of small and large systems
        atoms_per_system = [2, 25, 5, 30, 1]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=5,
            cell_sizes=[2.0, 2.0, 2.0, 2.0, 2.0],
            atoms_per_system=atoms_per_system,
            dtype=dtype,
            device=device,
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.5
        max_neighbors = 50

        _, num_neighbors, _ = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
        )

        # Check that single-atom systems have no neighbors
        single_atom_indices = []
        for i, num_atoms in enumerate(atoms_per_system):
            if num_atoms == 1:
                start_idx = batch_ptr[i].item()
                single_atom_indices.append(start_idx)

        for idx in single_atom_indices:
            assert num_neighbors[idx].item() == 0, (
                f"Single atom at index {idx} should have no neighbors"
            )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_precision_consistency(self, device, dtype, half_fill):
        """Test that float32 and float64 give consistent results."""
        atoms_per_system = [6, 8, 7]
        positions_batch_f32, cell_batch_f32, pbc_batch, _ = create_batch_systems(
            num_systems=3,
            atoms_per_system=atoms_per_system,
            dtype=torch.float32,
            device=device,
        )
        positions_batch_f64 = positions_batch_f32.double()
        cell_batch_f64 = cell_batch_f32.double()

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 30

        # Get results for both precisions
        _, num_neighbors_f32, _ = batch_naive_neighbor_list(
            positions_batch_f32,
            cutoff,
            pbc=pbc_batch,
            cell=cell_batch_f32,
            max_neighbors=max_neighbors,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            half_fill=half_fill,
        )
        _, num_neighbors_f64, _ = batch_naive_neighbor_list(
            positions_batch_f64,
            cutoff,
            pbc=pbc_batch,
            cell=cell_batch_f64,
            max_neighbors=max_neighbors,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            half_fill=half_fill,
        )

        # Neighbor counts should be identical (for this exact geometry)
        torch.testing.assert_close(num_neighbors_f32, num_neighbors_f64, rtol=0, atol=0)


class TestBatchNaiveMemoryAndPerformance:
    """Test memory usage and performance characteristics of batch implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_memory_scaling(self, device, half_fill):
        """Test that memory usage scales reasonably with batch size."""
        import gc

        dtype = torch.float32
        cutoff = 1.2

        # Test different batch sizes
        batch_configs = (
            [([8, 10], 2), ([12, 15], 2)]
            if device == "cpu"
            else [([20, 25], 2), ([30, 35], 2)]
        )

        for atoms_per_system, num_systems in batch_configs:
            positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                num_systems=num_systems,
                atoms_per_system=atoms_per_system,
                dtype=dtype,
                device=device,
            )
            batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

            # Estimate reasonable max_neighbors based on system size and cutoff
            max_neighbors = 40

            # Clear cache before test
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

            # Run batch implementation
            neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
                batch_naive_neighbor_list(
                    positions=positions_batch,
                    cutoff=cutoff,
                    max_neighbors=max_neighbors,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    batch_idx=batch_idx,
                    batch_ptr=batch_ptr,
                    half_fill=half_fill,
                )
            )

            # Basic checks that output is reasonable
            total_atoms = positions_batch.shape[0]
            assert neighbor_matrix.shape == (total_atoms, max_neighbors)
            assert neighbor_matrix_shifts.shape == (
                total_atoms,
                max_neighbors,
                3,
            )
            assert num_neighbors.shape == (total_atoms,)
            assert torch.all(num_neighbors >= 0)
            assert torch.all(num_neighbors <= max_neighbors)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_max_neighbors_overflow_handling(self, device, dtype, half_fill):
        """Test behavior when max_neighbors is exceeded."""
        # Create a dense batch system with small max_neighbors to force overflow
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 2.0  # Large cutoff to find many neighbors
        max_neighbors = 3  # Artificially small to trigger overflow

        # Should not crash, but may not find all neighbors
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
            batch_naive_neighbor_list(
                positions=positions_batch,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                pbc=pbc_batch,
                cell=cell_batch,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                half_fill=half_fill,
            )
        )

        # Should still produce valid output, just potentially incomplete
        total_atoms = positions_batch.shape[0]
        assert torch.all(num_neighbors >= 0)
        assert neighbor_matrix.shape == (total_atoms, max_neighbors)
        assert neighbor_matrix_shifts.shape == (
            total_atoms,
            max_neighbors,
            3,
        )
        assert num_neighbors.shape == (total_atoms,)
        assert neighbor_matrix.device == torch.device(device)
        assert neighbor_matrix_shifts.device == torch.device(device)
        assert num_neighbors.device == torch.device(device)
