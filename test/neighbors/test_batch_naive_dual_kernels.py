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

"""Tests for batch naive dual cutoff neighbor list kernels and launchers."""

import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.batch_naive_dual_cutoff import (
    _fill_batch_naive_neighbor_matrix_dual_cutoff,
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff,
    batch_naive_neighbor_matrix_dual_cutoff,
    batch_naive_neighbor_matrix_pbc_dual_cutoff,
)
from nvalchemiops.neighbors.neighbor_utils import _expand_naive_shifts
from nvalchemiops.torch.neighbors.neighbor_utils import compute_naive_num_shifts
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

from .test_utils import create_batch_systems


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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
class TestBatchNaiveDualCutoffKernels:
    """Test individual batch naive dual cutoff neighbor list kernels."""

    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_dual_cutoff_kernel_no_pbc(self, device, dtype, half_fill):
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
        assert torch.all(num_neighbors2 >= num_neighbors1), (
            "Larger cutoff should find at least as many neighbors"
        )

        # Check that neighbor indices are valid
        for i in range(positions_batch.shape[0]):
            for j in range(num_neighbors1[i].item()):
                neighbor_idx = neighbor_matrix1[i, j].item()
                assert 0 <= neighbor_idx < positions_batch.shape[0]
                assert neighbor_idx != i

            for j in range(num_neighbors2[i].item()):
                neighbor_idx = neighbor_matrix2[i, j].item()
                assert 0 <= neighbor_idx < positions_batch.shape[0]
                assert neighbor_idx != i

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

        # Launch kernel
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
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors2 >= num_neighbors1)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("half_fill", [True, False])
class TestBatchNaiveDualCutoffWpLaunchers:
    """Test the public launcher API for batch naive dual cutoff neighbor lists."""

    def test_batch_naive_neighbor_matrix_dual_cutoff(self, device, dtype, half_fill):
        """Test batch_naive_neighbor_matrix_dual_cutoff launcher (no PBC)."""
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

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)
        wp_neighbor_matrix1 = wp.from_torch(neighbor_matrix1, dtype=wp.int32)
        wp_neighbor_matrix2 = wp.from_torch(neighbor_matrix2, dtype=wp.int32)
        wp_num_neighbors1 = wp.from_torch(num_neighbors1, dtype=wp.int32)
        wp_num_neighbors2 = wp.from_torch(num_neighbors2, dtype=wp.int32)

        # Call launcher
        batch_naive_neighbor_matrix_dual_cutoff(
            wp_positions,
            cutoff1,
            cutoff2,
            wp_batch_idx,
            wp_batch_ptr,
            wp_neighbor_matrix1,
            wp_num_neighbors1,
            wp_neighbor_matrix2,
            wp_num_neighbors2,
            wp_dtype,
            device,
            half_fill,
        )

        # Verify results
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors2 >= num_neighbors1)

    def test_batch_naive_neighbor_matrix_pbc_dual_cutoff(
        self, device, dtype, half_fill
    ):
        """Test batch_naive_neighbor_matrix_pbc_dual_cutoff launcher (with PBC)."""
        atoms_per_system = [4, 6]
        num_systems = 2
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=num_systems,
            atoms_per_system=atoms_per_system,
            dtype=dtype,
            device=device,
        )
        _, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 30
        max_neighbors2 = 40
        max_atoms_per_system = max(atoms_per_system)

        # Compute shift ranges
        shift_range_per_dimension, shift_offset, total_shifts = (
            compute_naive_num_shifts(cell_batch, cutoff2, pbc_batch)
        )

        # Expand shift ranges into actual shift vectors
        shifts = torch.zeros(total_shifts, 3, dtype=torch.int32, device=device)
        shift_system_idx = torch.zeros(total_shifts, dtype=torch.int32, device=device)

        wp_shift_range_per_dimension = wp.from_torch(
            shift_range_per_dimension, dtype=wp.vec3i
        )
        wp_shift_offset = wp.from_torch(shift_offset, dtype=wp.int32)
        wp_shifts = wp.from_torch(shifts, dtype=wp.vec3i)
        wp_shift_system_idx = wp.from_torch(shift_system_idx, dtype=wp.int32)

        wp.launch(
            _expand_naive_shifts,
            dim=num_systems,
            device=device,
            inputs=[
                wp_shift_range_per_dimension,
                wp_shift_offset,
                wp_shifts,
                wp_shift_system_idx,
            ],
        )

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
        neighbor_matrix_shifts1 = torch.zeros(
            (positions_batch.shape[0], max_neighbors1, 3),
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts2 = torch.zeros(
            (positions_batch.shape[0], max_neighbors2, 3),
            dtype=torch.int32,
            device=device,
        )
        num_neighbors1 = torch.zeros(
            positions_batch.shape[0], dtype=torch.int32, device=device
        )
        num_neighbors2 = torch.zeros(
            positions_batch.shape[0], dtype=torch.int32, device=device
        )

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell_batch, dtype=wp_mat_dtype)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)
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

        # Call launcher
        batch_naive_neighbor_matrix_pbc_dual_cutoff(
            wp_positions,
            wp_cell,
            cutoff1,
            cutoff2,
            wp_batch_ptr,
            wp_shifts,
            wp_shift_system_idx,
            wp_neighbor_matrix1,
            wp_neighbor_matrix2,
            wp_neighbor_matrix_shifts1,
            wp_neighbor_matrix_shifts2,
            wp_num_neighbors1,
            wp_num_neighbors2,
            wp_dtype,
            device,
            max_atoms_per_system,
            half_fill,
        )

        # Verify results
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors2 >= num_neighbors1)

        # Check that unit_shifts are reasonable
        valid_shifts1 = neighbor_matrix_shifts1[neighbor_matrix1 != -1]
        valid_shifts2 = neighbor_matrix_shifts2[neighbor_matrix2 != -1]
        if len(valid_shifts1) > 0:
            assert torch.all(torch.abs(valid_shifts1) <= 5)
        if len(valid_shifts2) > 0:
            assert torch.all(torch.abs(valid_shifts2) <= 5)
