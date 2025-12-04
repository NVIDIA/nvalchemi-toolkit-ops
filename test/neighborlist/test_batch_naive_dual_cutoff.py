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

"""Comprehensive tests for batch naive dual cutoff neighbor list routines and utilities."""

import pytest
import torch
import warp as wp

from nvalchemiops.neighborlist.batch_naive import (
    batch_naive_neighbor_list,
)
from nvalchemiops.neighborlist.batch_naive_dual_cutoff import (
    _batch_naive_neighbor_matrix_no_pbc_dual_cutoff,
    _batch_naive_neighbor_matrix_pbc_dual_cutoff,
    _fill_batch_naive_neighbor_matrix_dual_cutoff,
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff,
    batch_naive_neighbor_list_dual_cutoff,
)
from nvalchemiops.neighborlist.neighbor_utils import (
    compute_naive_num_shifts,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

from .test_utils import (
    create_batch_systems,
)


def create_batch_idx_and_ptr(atoms_per_system, device):
    """Helper function to create batch_idx and batch_ptr tensors."""
    batch_idx = []
    batch_ptr = [0]

    for sys_idx, num_atoms in enumerate(atoms_per_system):
        batch_idx.extend([sys_idx] * num_atoms)
        batch_ptr.append(batch_ptr[-1] + num_atoms)

    batch_idx = torch.tensor(batch_idx, dtype=torch.int32, device=device)
    batch_ptr = torch.tensor(batch_ptr, dtype=torch.int32, device=device)

    return batch_idx, batch_ptr


class TestBatchNaiveDualCutoffKernels:
    """Test individual batch naive dual cutoff neighbor list kernels."""

    @pytest.mark.parametrize("half_fill", [True, False])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_naive_dual_cutoff_kernel_no_pbc(self, half_fill, device, dtype):
        """Test _fill_batch_naive_neighbor_matrix_dual_cutoff kernel (no PBC)."""
        # Create batch system
        atoms_per_system = [4, 6]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 10
        max_neighbors2 = 15

        # Convert to warp types
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_device = str(device)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)

        # Output arrays
        neighbor_matrix1 = torch.full(
            (positions_batch.shape[0], max_neighbors1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix2 = torch.full(
            (positions_batch.shape[0], max_neighbors2),
            -1,
            dtype=torch.int32,
            device=device,
        )
        num_neighbors1 = torch.zeros(
            positions_batch.shape[0], dtype=torch.int32, device=device
        )
        num_neighbors2 = torch.zeros(
            positions_batch.shape[0], dtype=torch.int32, device=device
        )

        wp_neighbor_matrix1 = wp.from_torch(neighbor_matrix1, dtype=wp.int32)
        wp_neighbor_matrix2 = wp.from_torch(neighbor_matrix2, dtype=wp.int32)
        wp_num_neighbors1 = wp.from_torch(num_neighbors1, dtype=wp.int32)
        wp_num_neighbors2 = wp.from_torch(num_neighbors2, dtype=wp.int32)

        # Launch kernel
        wp.launch(
            _fill_batch_naive_neighbor_matrix_dual_cutoff,
            dim=positions_batch.shape[0],
            device=wp_device,
            inputs=[
                wp_positions,
                wp_dtype(cutoff1),
                wp_dtype(cutoff2),
                wp_batch_idx,
                wp_batch_ptr,
                wp_neighbor_matrix1,
                wp_num_neighbors1,
                wp_neighbor_matrix2,
                wp_num_neighbors2,
                half_fill,
            ],
        )

        # Check results
        # num_neighbors2 should be >= num_neighbors1 (larger cutoff finds more neighbors)
        assert torch.all(num_neighbors2 >= num_neighbors1), (
            "Larger cutoff should find at least as many neighbors"
        )

        # Check that neighbor indices are valid
        for i in range(positions_batch.shape[0]):
            for j in range(num_neighbors1[i].item()):
                neighbor_idx = neighbor_matrix1[i, j].item()
                assert 0 <= neighbor_idx < positions_batch.shape[0], (
                    f"Invalid neighbor index in matrix1: {neighbor_idx}"
                )
                assert neighbor_idx != i, "Atom should not be its own neighbor"

            for j in range(num_neighbors2[i].item()):
                neighbor_idx = neighbor_matrix2[i, j].item()
                assert 0 <= neighbor_idx < positions_batch.shape[0], (
                    f"Invalid neighbor index in matrix2: {neighbor_idx}"
                )
                assert neighbor_idx != i, "Atom should not be its own neighbor"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_naive_dual_cutoff_pbc_kernel(self, device, dtype):
        """Test _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff kernel."""
        # Create batch system with PBC
        atoms_per_system = [4, 4]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        _, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 20

        # Compute shifts for both cutoffs
        _, shift_offset, total_shifts = compute_naive_num_shifts(
            cell_batch, cutoff2, pbc_batch
        )
        total_shifts = shift_offset[-1].item()

        # Convert to warp types
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell_batch, dtype=wp_mat_dtype)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)

        # Create shift arrays (simplified for testing)
        shifts = torch.zeros(total_shifts, 3, dtype=torch.int32, device=device)
        shift_system_idx = torch.zeros(total_shifts, dtype=torch.int32, device=device)

        wp_shifts = wp.from_torch(shifts, dtype=wp.vec3i)
        wp_shift_system_idx = wp.from_torch(shift_system_idx, dtype=wp.int32)

        # Output arrays
        neighbor_matrix1 = torch.full(
            (positions_batch.shape[0], max_neighbors1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix2 = torch.full(
            (positions_batch.shape[0], max_neighbors2),
            -1,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts1 = torch.zeros(
            positions_batch.shape[0],
            max_neighbors1,
            3,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts2 = torch.zeros(
            positions_batch.shape[0],
            max_neighbors2,
            3,
            dtype=torch.int32,
            device=device,
        )
        num_neighbors1 = torch.zeros(
            positions_batch.shape[0], dtype=torch.int32, device=device
        )
        num_neighbors2 = torch.zeros(
            positions_batch.shape[0], dtype=torch.int32, device=device
        )

        wp_neighbor_matrix1 = wp.from_torch(neighbor_matrix1, dtype=wp.int32)
        wp_neighbor_matrix2 = wp.from_torch(neighbor_matrix2, dtype=wp.int32)
        wp_neighbor_matrix_shifts1 = wp.from_torch(
            neighbor_matrix_shifts1, dtype=wp.vec3i
        )
        wp_neighbor_matrix_shifts2 = wp.from_torch(
            neighbor_matrix_shifts2, dtype=wp.vec3i
        )
        wp_num_neighbors1 = wp.from_torch(num_neighbors1, dtype=wp.int32)
        wp_num_neighbors2 = wp.from_torch(num_neighbors2, dtype=wp.int32)

        # Launch kernel (use smaller shift array for testing)
        wp.launch(
            _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff,
            dim=(total_shifts, positions_batch.shape[0]),
            device=wp_device,
            inputs=[
                wp_positions,
                wp_cell,
                wp_dtype(cutoff1),
                wp_dtype(cutoff2),
                wp_batch_ptr,
                wp_shifts,
                wp_shift_system_idx,
                wp_neighbor_matrix1,
                wp_neighbor_matrix2,
                wp_neighbor_matrix_shifts1,
                wp_neighbor_matrix_shifts2,
                wp_num_neighbors1,
                wp_num_neighbors2,
                True,  # half_fill
            ],
        )

        # Check that we have reasonable results
        assert torch.all(num_neighbors1 >= 0), (
            "All neighbor counts should be non-negative"
        )
        assert torch.all(num_neighbors2 >= 0), (
            "All neighbor counts should be non-negative"
        )
        assert torch.all(num_neighbors2 >= num_neighbors1), (
            "Larger cutoff should find at least as many neighbors"
        )


class TestBatchNaiveDualCutoffUtilityFunctions:
    """Test utility functions used by batch naive dual cutoff neighbor list."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_matrix_no_pbc_dual_cutoff_function_with_preallocation(
        self, device, dtype, half_fill
    ):
        """Test _batch_naive_neighbor_matrix_no_pbc_dual_cutoff custom op with preallocated tensors."""
        atoms_per_system = [6, 8]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 20
        max_neighbors2 = 30

        # Prepare output arrays
        neighbor_matrix1 = torch.full(
            (positions_batch.shape[0], max_neighbors1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix2 = torch.full(
            (positions_batch.shape[0], max_neighbors2),
            -1,
            dtype=torch.int32,
            device=device,
        )
        num_neighbors1 = torch.zeros(
            positions_batch.shape[0], dtype=torch.int32, device=device
        )
        num_neighbors2 = torch.zeros(
            positions_batch.shape[0], dtype=torch.int32, device=device
        )

        _batch_naive_neighbor_matrix_no_pbc_dual_cutoff(
            positions_batch,
            cutoff1,
            cutoff2,
            batch_idx,
            batch_ptr,
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix2,
            num_neighbors2,
            half_fill,
        )

        # Check results
        assert torch.all(num_neighbors1 >= 0), (
            "All neighbor counts should be non-negative"
        )
        assert torch.all(num_neighbors2 >= 0), (
            "All neighbor counts should be non-negative"
        )
        assert torch.all(num_neighbors1 <= max_neighbors1), (
            "Neighbor counts should not exceed maximum"
        )
        assert torch.all(num_neighbors2 <= max_neighbors2), (
            "Neighbor counts should not exceed maximum"
        )
        assert torch.all(num_neighbors2 >= num_neighbors1), (
            "Larger cutoff should find at least as many neighbors"
        )

        # Check that neighbor indices are valid
        for i in range(positions_batch.shape[0]):
            for j in range(num_neighbors1[i].item()):
                neighbor_idx = neighbor_matrix1[i, j].item()
                assert 0 <= neighbor_idx < positions_batch.shape[0], (
                    f"Invalid neighbor index: {neighbor_idx}"
                )
                assert neighbor_idx != i, "Atom should not be its own neighbor"

            for j in range(num_neighbors2[i].item()):
                neighbor_idx = neighbor_matrix2[i, j].item()
                assert 0 <= neighbor_idx < positions_batch.shape[0], (
                    f"Invalid neighbor index: {neighbor_idx}"
                )
                assert neighbor_idx != i, "Atom should not be its own neighbor"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_neighbor_matrix_pbc_dual_cutoff_function(
        self, device, dtype, half_fill
    ):
        """Test _batch_naive_neighbor_matrix_pbc_dual_cutoff function with preallocation."""
        atoms_per_system = [4, 6]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        _, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 30
        max_neighbors2 = 40
        total_atoms = positions_batch.shape[0]
        expected_rows = total_atoms

        # Compute shifts for both cutoffs
        shift_range_per_dimension, shift_offset, total_shifts = (
            compute_naive_num_shifts(cell_batch, cutoff2, pbc_batch)
        )
        # Preallocate output tensors
        neighbor_matrix1 = torch.full(
            (expected_rows, max_neighbors1), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix2 = torch.full(
            (expected_rows, max_neighbors2), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts1 = torch.zeros(
            (expected_rows, max_neighbors1, 3), dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts2 = torch.zeros(
            (expected_rows, max_neighbors2, 3), dtype=torch.int32, device=device
        )
        num_neighbors1 = torch.zeros(total_atoms, dtype=torch.int32, device=device)
        num_neighbors2 = torch.zeros(total_atoms, dtype=torch.int32, device=device)
        _batch_naive_neighbor_matrix_pbc_dual_cutoff(
            positions_batch,
            cell_batch,
            cutoff1,
            cutoff2,
            batch_ptr,
            neighbor_matrix1,
            neighbor_matrix2,
            neighbor_matrix_shifts1,
            neighbor_matrix_shifts2,
            num_neighbors1,
            num_neighbors2,
            shift_range_per_dimension,
            shift_offset,
            total_shifts,
            half_fill,
        )
        # Check output shapes and types
        assert neighbor_matrix1.shape == (expected_rows, max_neighbors1)
        assert neighbor_matrix2.shape == (expected_rows, max_neighbors2)
        assert neighbor_matrix_shifts1.shape == (expected_rows, max_neighbors1, 3)
        assert neighbor_matrix_shifts2.shape == (expected_rows, max_neighbors2, 3)
        assert num_neighbors1.shape == (total_atoms,)
        assert num_neighbors2.shape == (total_atoms,)

        assert neighbor_matrix1.dtype == torch.int32
        assert neighbor_matrix2.dtype == torch.int32
        assert neighbor_matrix_shifts1.dtype == torch.int32
        assert neighbor_matrix_shifts2.dtype == torch.int32
        assert num_neighbors1.dtype == torch.int32
        assert num_neighbors2.dtype == torch.int32

        assert neighbor_matrix1.device == torch.device(device)
        assert neighbor_matrix2.device == torch.device(device)
        assert neighbor_matrix_shifts1.device == torch.device(device)
        assert neighbor_matrix_shifts2.device == torch.device(device)
        assert num_neighbors1.device == torch.device(device)
        assert num_neighbors2.device == torch.device(device)

        # Check neighbor counts
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors2 >= num_neighbors1)

        # Check that unit_shifts are reasonable (should be small integers)
        valid_shifts1 = neighbor_matrix_shifts1[neighbor_matrix1 != -1]
        valid_shifts2 = neighbor_matrix_shifts2[neighbor_matrix2 != -1]
        if len(valid_shifts1) > 0:
            assert torch.all(torch.abs(valid_shifts1) <= 5), (
                "Unit shifts should be small integers"
            )
        if len(valid_shifts2) > 0:
            assert torch.all(torch.abs(valid_shifts2) <= 5), (
                "Unit shifts should be small integers"
            )


class TestBatchNaiveDualCutoffMainAPI:
    """Test the main batch naive dual cutoff neighbor list API function."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_dual_cutoff_no_pbc(
        self, device, dtype, half_fill
    ):
        """Test batch_naive_neighbor_list_dual_cutoff without periodic boundary conditions."""
        atoms_per_system = [6, 8]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 20
        max_neighbors2 = 30

        # Test without PBC
        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                pbc=None,
                cell=None,
                half_fill=half_fill,
            )
        )

        # Check output types and shapes
        expected_rows = positions_batch.shape[0]
        assert neighbor_matrix1.dtype == torch.int32
        assert neighbor_matrix2.dtype == torch.int32
        assert num_neighbors1.dtype == torch.int32
        assert num_neighbors2.dtype == torch.int32
        assert neighbor_matrix1.shape == (expected_rows, max_neighbors1)
        assert neighbor_matrix2.shape == (expected_rows, max_neighbors2)
        assert num_neighbors1.shape == (positions_batch.shape[0],)
        assert num_neighbors2.shape == (positions_batch.shape[0],)
        assert neighbor_matrix1.device == torch.device(device)
        assert neighbor_matrix2.device == torch.device(device)
        assert num_neighbors1.device == torch.device(device)
        assert num_neighbors2.device == torch.device(device)

        # Check neighbor counts are reasonable
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors1 <= max_neighbors1)
        assert torch.all(num_neighbors2 <= max_neighbors2)
        assert torch.all(num_neighbors2 >= num_neighbors1)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_dual_cutoff_with_pbc(
        self, device, dtype, half_fill
    ):
        """Test batch_naive_neighbor_list_dual_cutoff with periodic boundary conditions."""
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 30
        max_neighbors2 = 50

        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
        )

        # Check output types and shapes
        expected_rows = positions_batch.shape[0]
        assert neighbor_matrix1.dtype == torch.int32
        assert neighbor_matrix2.dtype == torch.int32
        assert neighbor_matrix_shifts1.dtype == torch.int32
        assert neighbor_matrix_shifts2.dtype == torch.int32
        assert num_neighbors1.dtype == torch.int32
        assert num_neighbors2.dtype == torch.int32
        assert neighbor_matrix1.shape == (expected_rows, max_neighbors1)
        assert neighbor_matrix2.shape == (expected_rows, max_neighbors2)
        assert neighbor_matrix_shifts1.shape == (expected_rows, max_neighbors1, 3)
        assert neighbor_matrix_shifts2.shape == (expected_rows, max_neighbors2, 3)
        assert num_neighbors1.shape == (positions_batch.shape[0],)
        assert num_neighbors2.shape == (positions_batch.shape[0],)
        assert neighbor_matrix1.device == torch.device(device)
        assert neighbor_matrix2.device == torch.device(device)
        assert neighbor_matrix_shifts1.device == torch.device(device)
        assert neighbor_matrix_shifts2.device == torch.device(device)
        assert num_neighbors1.device == torch.device(device)
        assert num_neighbors2.device == torch.device(device)

        # Check neighbor counts
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors2 >= num_neighbors1)

        # With PBC, should generally have more neighbors than without
        assert num_neighbors1.sum() >= 0, "Should find some neighbors with PBC"
        assert num_neighbors2.sum() >= 0, "Should find some neighbors with PBC"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    @pytest.mark.parametrize("pbc_flag", [True, False])
    def test_batch_naive_neighbor_list_dual_cutoff_return_neighbor_list(
        self, device, dtype, half_fill, pbc_flag
    ):
        """Test batch_naive_neighbor_list_dual_cutoff with return_neighbor_list=True."""
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 30
        max_neighbors2 = 50

        if not pbc_flag:
            cell_batch = None
            pbc_batch = None

            neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2 = (
                batch_naive_neighbor_list_dual_cutoff(
                    positions=positions_batch,
                    cutoff1=cutoff1,
                    cutoff2=cutoff2,
                    batch_idx=batch_idx,
                    batch_ptr=batch_ptr,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    half_fill=half_fill,
                    return_neighbor_list=True,
                )
            )

        else:
            (
                neighbor_list1,
                neighbor_ptr1,
                unit_shifts1,
                neighbor_list2,
                neighbor_ptr2,
                unit_shifts2,
            ) = batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                pbc=pbc_batch,
                cell=cell_batch,
                half_fill=half_fill,
                return_neighbor_list=True,
            )

        # Check that we get neighbor list format (2, N) instead of matrix
        assert neighbor_list1.ndim == 2
        assert neighbor_list2.ndim == 2
        assert neighbor_list1.shape[0] == 2
        assert neighbor_list2.shape[0] == 2
        assert neighbor_list1.dtype == torch.int32
        assert neighbor_list2.dtype == torch.int32
        assert neighbor_ptr1.dtype == torch.int32
        assert neighbor_ptr2.dtype == torch.int32

        # Check that neighbor_list2 has at least as many pairs as neighbor_list1
        assert neighbor_list2.shape[1] >= neighbor_list1.shape[1], (
            "Larger cutoff should find at least as many neighbor pairs"
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_naive_neighbor_list_dual_cutoff_consistency_with_single_cutoff(
        self, device, dtype
    ):
        """Test that dual cutoff results are consistent with two single cutoff calls."""
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 30
        max_neighbors2 = 50

        # Get dual cutoff result
        (
            neighbor_matrix1_dual,
            num_neighbors1_dual,
            neighbor_matrix_shifts1_dual,
            neighbor_matrix2_dual,
            num_neighbors2_dual,
            neighbor_matrix_shifts2_dual,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=False,  # Use full fill for easier comparison
        )

        # Get single cutoff results
        (
            neighbor_matrix1_single,
            num_neighbors1_single,
            neighbor_matrix_shifts1_single,
        ) = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff1,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=False,
        )

        (
            neighbor_matrix2_single,
            num_neighbors2_single,
            neighbor_matrix_shifts2_single,
        ) = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=False,
        )

        # Compare neighbor counts (should be identical)
        torch.testing.assert_close(
            num_neighbors1_dual, num_neighbors1_single, rtol=0, atol=0
        )
        torch.testing.assert_close(
            num_neighbors2_dual, num_neighbors2_single, rtol=0, atol=0
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_dual_cutoff_edge_cases(
        self, device, dtype, half_fill
    ):
        """Test edge cases for batch_naive_neighbor_list_dual_cutoff."""
        # Empty system
        positions_empty = torch.empty(0, 3, dtype=dtype, device=device)
        batch_idx_empty = torch.empty(0, dtype=torch.int32, device=device)
        batch_ptr_empty = torch.tensor([0], dtype=torch.int32, device=device)

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_empty,
                cutoff1=1.0,
                cutoff2=1.5,
                batch_idx=batch_idx_empty,
                batch_ptr=batch_ptr_empty,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
                half_fill=half_fill,
            )
        )
        assert neighbor_matrix1.shape == (0, 10)
        assert neighbor_matrix2.shape == (0, 15)
        assert num_neighbors1.shape == (0,)
        assert num_neighbors2.shape == (0,)

        # Single atom system
        positions_single = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        batch_idx_single = torch.tensor([0], dtype=torch.int32, device=device)
        batch_ptr_single = torch.tensor([0, 1], dtype=torch.int32, device=device)

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_single,
                cutoff1=1.0,
                cutoff2=1.5,
                batch_idx=batch_idx_single,
                batch_ptr=batch_ptr_single,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
                half_fill=half_fill,
            )
        )
        assert num_neighbors1[0].item() == 0, "Single atom should have no neighbors"
        assert num_neighbors2[0].item() == 0, "Single atom should have no neighbors"

        # Zero cutoffs
        atoms_per_system = [4, 4]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=0.0,
                cutoff2=0.0,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
                half_fill=half_fill,
            )
        )
        assert torch.all(num_neighbors1 == 0), "Zero cutoffs should find no neighbors"
        assert torch.all(num_neighbors2 == 0), "Zero cutoffs should find no neighbors"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_dual_cutoff_error_conditions(
        self, device, dtype, half_fill
    ):
        """Test error conditions for batch_naive_neighbor_list_dual_cutoff."""
        atoms_per_system = [4, 6]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Test mismatched cell and pbc arguments
        with pytest.raises(
            ValueError, match="If cell is provided, pbc must also be provided"
        ):
            batch_naive_neighbor_list_dual_cutoff(
                positions_batch,
                1.0,
                1.5,
                batch_idx,
                batch_ptr,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=cell_batch,
            )

        with pytest.raises(
            ValueError, match="If pbc is provided, cell must also be provided"
        ):
            batch_naive_neighbor_list_dual_cutoff(
                positions_batch,
                1.0,
                1.5,
                batch_idx,
                batch_ptr,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=pbc_batch,
                cell=None,
            )

    def test_max_neighbors_same_value(self):
        """Test that both neighbor matrices have correct shape when given same max_neighbors."""
        atoms_per_system = [4, 6]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2,
            atoms_per_system=atoms_per_system,
            dtype=torch.float32,
            device="cpu",
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, "cpu")

        # Test with same max_neighbors for both cutoffs
        neighbor_matrix1, _, neighbor_matrix2, _ = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=1.0,
                cutoff2=1.5,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=10,
                max_neighbors2=10,  # Same as max_neighbors1
                pbc=None,
                cell=None,
            )
        )

        # Both matrices should have same number of columns
        assert neighbor_matrix1.shape[1] == neighbor_matrix2.shape[1] == 10


class TestBatchNaiveDualCutoffPerformanceAndScaling:
    """Test performance characteristics and scaling of batch naive dual cutoff implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_batch_dual_cutoff_scaling_with_system_size(self, device):
        """Test that batch dual cutoff implementation scales as expected with system size."""
        import time

        dtype = torch.float32
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 50
        max_neighbors2 = 80

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
                batch_naive_neighbor_list_dual_cutoff(
                    positions_batch,
                    cutoff1,
                    cutoff2,
                    batch_idx,
                    batch_ptr,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    half_fill=True,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            # Time the operation
            start_time = time.time()
            for _ in range(100):
                batch_naive_neighbor_list_dual_cutoff(
                    positions_batch,
                    cutoff1,
                    cutoff2,
                    batch_idx,
                    batch_ptr,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    half_fill=True,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            times.append(elapsed)

        # Check that it doesn't grow too fast (should be roughly O(N^2))
        # This is a loose check since we can't expect perfect scaling
        assert times[1] > times[0] * 0.5, "Time should increase with system size"
        if len(times) > 2:
            # Very loose scaling check
            scaling_factor = times[-1] / times[0]
            total_atoms_ratio = sum(batch_sizes[-1][1]) / sum(batch_sizes[0][1])
            size_factor = total_atoms_ratio**2
            assert scaling_factor < size_factor * 5, (
                "Scaling should not be much worse than O(N^2)"
            )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_batch_dual_cutoff_cutoff_scaling(self, device):
        """Test scaling with different cutoff values."""
        dtype = torch.float32
        atoms_per_system = [15, 20]
        max_neighbors1 = 100
        max_neighbors2 = 150

        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Test different cutoff pairs
        cutoff_pairs = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5)]
        neighbor_counts1 = []
        neighbor_counts2 = []

        for cutoff1, cutoff2 in cutoff_pairs:
            (_, num_neighbors1, _, _, num_neighbors2, _) = (
                batch_naive_neighbor_list_dual_cutoff(
                    positions_batch,
                    cutoff1,
                    cutoff2,
                    batch_idx,
                    batch_ptr,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    half_fill=True,
                )
            )
            total_pairs1 = num_neighbors1.sum().item()
            total_pairs2 = num_neighbors2.sum().item()
            neighbor_counts1.append(total_pairs1)
            neighbor_counts2.append(total_pairs2)

        # Check that neighbor count increases with cutoff
        for i in range(1, len(neighbor_counts1)):
            assert neighbor_counts1[i] >= neighbor_counts1[i - 1], (
                f"Neighbor count should increase with cutoff1: {neighbor_counts1}"
            )
            assert neighbor_counts2[i] >= neighbor_counts2[i - 1], (
                f"Neighbor count should increase with cutoff2: {neighbor_counts2}"
            )

        # Check that cutoff2 always finds at least as many neighbors as cutoff1
        for count1, count2 in zip(neighbor_counts1, neighbor_counts2):
            assert count2 >= count1, (
                f"Larger cutoff should find at least as many neighbors: {count1} vs {count2}"
            )


class TestBatchNaiveDualCutoffRobustness:
    """Test robustness of batch naive dual cutoff implementation to various inputs."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_random_systems(self, device, dtype, half_fill):
        """Test with random systems of various sizes and configurations."""
        for pbc_flag in [True, False]:
            # Test several random systems
            for seed in [42, 123, 456]:
                atoms_per_system = [15, 20, 18]
                positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                    num_systems=3,
                    atoms_per_system=atoms_per_system,
                    dtype=dtype,
                    device=device,
                    seed=seed,
                    pbc_flag=pbc_flag,
                )
                batch_idx, batch_ptr = create_batch_idx_and_ptr(
                    atoms_per_system, device
                )

                cutoff1 = 1.0
                cutoff2 = 1.5
                max_neighbors1 = 50
                max_neighbors2 = 80

                if not pbc_flag:
                    cell_batch = None
                    pbc_batch = None

                # Should not crash
                result = batch_naive_neighbor_list_dual_cutoff(
                    positions=positions_batch,
                    cutoff1=cutoff1,
                    cutoff2=cutoff2,
                    batch_idx=batch_idx,
                    batch_ptr=batch_ptr,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    half_fill=half_fill,
                )

                if pbc_flag:
                    (
                        neighbor_matrix1,
                        num_neighbors1,
                        neighbor_matrix_shifts1,
                        neighbor_matrix2,
                        num_neighbors2,
                        neighbor_matrix_shifts2,
                    ) = result
                    # Basic sanity checks
                    assert torch.all(num_neighbors1 >= 0)
                    assert torch.all(num_neighbors2 >= 0)
                    assert torch.all(num_neighbors1 <= max_neighbors1)
                    assert torch.all(num_neighbors2 <= max_neighbors2)
                    assert torch.all(num_neighbors2 >= num_neighbors1)
                    assert neighbor_matrix1.device == torch.device(device)
                    assert neighbor_matrix_shifts1.device == torch.device(device)
                    assert neighbor_matrix2.device == torch.device(device)
                    assert neighbor_matrix_shifts2.device == torch.device(device)
                    assert num_neighbors1.device == torch.device(device)
                    assert num_neighbors2.device == torch.device(device)
                else:
                    (
                        neighbor_matrix1,
                        num_neighbors1,
                        neighbor_matrix2,
                        num_neighbors2,
                    ) = result
                    # Basic sanity checks
                    assert torch.all(num_neighbors1 >= 0)
                    assert torch.all(num_neighbors2 >= 0)
                    assert torch.all(num_neighbors1 <= max_neighbors1)
                    assert torch.all(num_neighbors2 <= max_neighbors2)
                    assert torch.all(num_neighbors2 >= num_neighbors1)
                    assert neighbor_matrix1.device == torch.device(device)
                    assert neighbor_matrix2.device == torch.device(device)
                    assert num_neighbors1.device == torch.device(device)
                    assert num_neighbors2.device == torch.device(device)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_extreme_geometries(self, device, dtype, half_fill):
        """Test with extreme cell geometries."""
        # Very elongated cells
        atoms_per_system = [8, 10]
        positions_batch = torch.rand(18, 3, dtype=dtype, device=device)
        cell_batch = torch.tensor(
            [
                [[10.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
                [[0.1, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.1]],
            ],
            dtype=dtype,
            device=device,
        )
        pbc_batch = torch.tensor(
            [[True, True, True], [True, True, True]], device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Scale positions to fit in cells
        positions_batch[:8] = positions_batch[:8] * torch.tensor(
            [10.0, 0.1, 0.1], device=device
        )
        positions_batch[8:] = positions_batch[8:] * torch.tensor(
            [0.1, 10.0, 0.1], device=device
        )

        cutoff1 = 0.15
        cutoff2 = 0.25
        max_neighbors1 = 20
        max_neighbors2 = 30

        # Should handle extreme aspect ratios
        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
        )

        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors2 >= num_neighbors1)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_large_cutoffs(self, device, dtype, half_fill):
        """Test with very large cutoffs."""
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Cutoffs larger than cell size
        large_cutoff1 = 4.0
        large_cutoff2 = 6.0
        max_neighbors1 = 100
        max_neighbors2 = 150

        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=large_cutoff1,
            cutoff2=large_cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
        )

        # Should find many neighbors
        assert num_neighbors1.sum() > 0
        assert num_neighbors2.sum() > 0
        assert torch.all(num_neighbors2 >= num_neighbors1)
        # Each atom should have multiple neighbors (including periodic images)
        assert torch.all(
            num_neighbors1 >= 0
        )  # Some atoms might have no neighbors in half_fill mode
        assert torch.all(num_neighbors2 >= 0)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_precision_consistency(self, device, dtype, half_fill):
        """Test that float32 and float64 give consistent results."""
        atoms_per_system = [6, 8]
        positions_batch_f32, cell_batch_f32, pbc_batch, _ = create_batch_systems(
            num_systems=2,
            atoms_per_system=atoms_per_system,
            dtype=torch.float32,
            device=device,
        )
        positions_batch_f64 = positions_batch_f32.double()
        cell_batch_f64 = cell_batch_f32.double()
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 50
        max_neighbors2 = 80

        # Get results for both precisions
        (_, num_neighbors1_f32, _, _, num_neighbors2_f32, _) = (
            batch_naive_neighbor_list_dual_cutoff(
                positions_batch_f32,
                cutoff1,
                cutoff2,
                batch_idx,
                batch_ptr,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                pbc=pbc_batch,
                cell=cell_batch_f32,
                half_fill=half_fill,
            )
        )
        (_, num_neighbors1_f64, _, _, num_neighbors2_f64, _) = (
            batch_naive_neighbor_list_dual_cutoff(
                positions_batch_f64,
                cutoff1,
                cutoff2,
                batch_idx,
                batch_ptr,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                pbc=pbc_batch,
                cell=cell_batch_f64,
                half_fill=half_fill,
            )
        )

        # Neighbor counts should be identical (for this exact geometry)
        torch.testing.assert_close(
            num_neighbors1_f32, num_neighbors1_f64, rtol=0, atol=0
        )
        torch.testing.assert_close(
            num_neighbors2_f32, num_neighbors2_f64, rtol=0, atol=0
        )


class TestBatchNaiveDualCutoffMemoryAndPerformance:
    """Test memory usage and performance characteristics of batch naive dual cutoff implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_memory_scaling(self, device, half_fill):
        """Test that memory usage scales reasonably with system size."""
        import gc

        dtype = torch.float32
        cutoff1 = 1.0
        cutoff2 = 1.5

        # Test different system sizes
        sizes = (
            [(2, [8, 10], [2.0, 2.0]), (3, [6, 8, 10], [2.0, 2.0, 2.0])]
            if device == "cpu"
            else [
                (3, [15, 20, 25], [2.0, 2.0, 2.0]),
                (4, [12, 15, 18, 20], [2.0, 2.0, 2.0, 2.0]),
            ]
        )

        for num_systems, atoms_per_system, cell_sizes in sizes:
            positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                num_systems=num_systems,
                atoms_per_system=atoms_per_system,
                dtype=dtype,
                device=device,
                cell_sizes=cell_sizes,
            )
            batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

            # Estimate reasonable max_neighbors based on system size and cutoff
            max_neighbors1 = 50
            max_neighbors2 = 80

            # Clear cache before test
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

            # Run batch dual cutoff implementation
            (
                neighbor_matrix1,
                num_neighbors1,
                neighbor_matrix_shifts1,
                neighbor_matrix2,
                num_neighbors2,
                neighbor_matrix_shifts2,
            ) = batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                pbc=pbc_batch,
                cell=cell_batch,
                half_fill=half_fill,
            )

            # Basic checks that output is reasonable
            total_atoms = positions_batch.shape[0]
            assert neighbor_matrix1.shape == (total_atoms, max_neighbors1)
            assert neighbor_matrix2.shape == (total_atoms, max_neighbors2)
            assert neighbor_matrix_shifts1.shape == (total_atoms, max_neighbors1, 3)
            assert neighbor_matrix_shifts2.shape == (total_atoms, max_neighbors2, 3)
            assert num_neighbors1.shape == (total_atoms,)
            assert num_neighbors2.shape == (total_atoms,)
            assert torch.all(num_neighbors1 >= 0), (
                "All neighbor counts should be non-negative"
            )
            assert torch.all(num_neighbors2 >= 0), (
                "All neighbor counts should be non-negative"
            )
            assert torch.all(num_neighbors1 <= max_neighbors1), (
                "Neighbor counts should not exceed maximum"
            )
            assert torch.all(num_neighbors2 <= max_neighbors2), (
                "Neighbor counts should not exceed maximum"
            )
            assert torch.all(num_neighbors2 >= num_neighbors1), (
                "Larger cutoff should find at least as many neighbors"
            )

            # Clean up
            del (
                neighbor_matrix1,
                neighbor_matrix2,
                neighbor_matrix_shifts1,
                neighbor_matrix_shifts2,
                num_neighbors1,
                num_neighbors2,
                positions_batch,
                cell_batch,
                pbc_batch,
                batch_idx,
                batch_ptr,
            )
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_max_neighbors_overflow_handling(self, device, dtype, half_fill):
        """Test behavior when max_neighbors is exceeded."""

        # Create a dense system with small max_neighbors to force overflow
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 2.0  # Large cutoffs to find many neighbors
        cutoff2 = 3.0
        max_neighbors1 = 3  # Artificially small to trigger overflow
        max_neighbors2 = 5

        # Should not crash, but may not find all neighbors
        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
        )

        # Should still produce valid output, just potentially incomplete
        total_atoms = sum(atoms_per_system)
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert neighbor_matrix1.shape == (total_atoms, max_neighbors1)
        assert neighbor_matrix2.shape == (total_atoms, max_neighbors2)
        assert neighbor_matrix_shifts1.shape == (
            total_atoms,
            max_neighbors1,
            3,
        )
        assert neighbor_matrix_shifts2.shape == (
            total_atoms,
            max_neighbors2,
            3,
        )
        assert num_neighbors1.shape == (total_atoms,)
        assert num_neighbors2.shape == (total_atoms,)
        assert neighbor_matrix1.device == torch.device(device)
        assert neighbor_matrix2.device == torch.device(device)
        assert neighbor_matrix_shifts1.device == torch.device(device)
        assert neighbor_matrix_shifts2.device == torch.device(device)
        assert num_neighbors1.device == torch.device(device)
        assert num_neighbors2.device == torch.device(device)
