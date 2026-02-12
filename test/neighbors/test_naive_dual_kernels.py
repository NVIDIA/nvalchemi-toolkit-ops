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

"""Tests for naive dual cutoff neighbor list kernels and launchers."""

import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.naive_dual_cutoff import (
    _fill_naive_neighbor_matrix_dual_cutoff,
    _fill_naive_neighbor_matrix_pbc_dual_cutoff,
    naive_neighbor_matrix_dual_cutoff,
    naive_neighbor_matrix_pbc_dual_cutoff,
)
from nvalchemiops.neighbors.neighbor_utils import _expand_naive_shifts
from nvalchemiops.torch.neighbors.neighbor_utils import compute_naive_num_shifts
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

from .test_utils import create_simple_cubic_system


class TestNaiveDualCutoffKernels:
    """Test individual naive dual cutoff neighbor list kernels."""

    @pytest.mark.parametrize("half_fill", [True, False])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_naive_dual_cutoff_kernel_no_pbc(self, half_fill, device, dtype):
        """Test _fill_naive_neighbor_matrix_dual_cutoff kernel (no PBC)."""
        # Create simple system
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],  # Within both cutoffs
                [1.0, 0.0, 0.0],  # Within cutoff2 only
                [1.5, 0.0, 0.0],  # Outside both cutoffs
            ],
            dtype=dtype,
            device=device,
        )
        cutoff1 = 0.7  # Should find first neighbor only
        cutoff2 = 1.2  # Should find first two neighbors
        max_neighbors1 = 10
        max_neighbors2 = 10

        # Convert to warp types
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_device = str(device)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)

        # Output arrays
        neighbor_matrix1 = torch.full(
            (positions.shape[0], max_neighbors1), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix2 = torch.full(
            (positions.shape[0], max_neighbors2), -1, dtype=torch.int32, device=device
        )
        num_neighbors1 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )
        num_neighbors2 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        wp_neighbor_matrix1 = wp.from_torch(neighbor_matrix1, dtype=wp.int32)
        wp_neighbor_matrix2 = wp.from_torch(neighbor_matrix2, dtype=wp.int32)
        wp_num_neighbors1 = wp.from_torch(num_neighbors1, dtype=wp.int32)
        wp_num_neighbors2 = wp.from_torch(num_neighbors2, dtype=wp.int32)

        # Launch kernel
        wp.launch(
            _fill_naive_neighbor_matrix_dual_cutoff,
            dim=positions.shape[0],
            device=wp_device,
            inputs=[
                wp_positions,
                wp_dtype(cutoff1 * cutoff1),
                wp_dtype(cutoff2 * cutoff2),
                wp_neighbor_matrix1,
                wp_num_neighbors1,
                wp_neighbor_matrix2,
                wp_num_neighbors2,
                half_fill,
            ],
        )

        # Check results
        total_neighbors1 = num_neighbors1.sum().item()
        total_neighbors2 = num_neighbors2.sum().item()
        assert total_neighbors2 >= total_neighbors1, (
            f"Larger cutoff should find at least as many neighbors: {total_neighbors2} >= {total_neighbors1}"
        )

        # Atom 0 should have neighbors within respective cutoffs
        assert num_neighbors1[0].item() == 1, (
            f"Atom 0 should have 1 neighbor in cutoff1, got {num_neighbors1[0].item()}"
        )
        assert num_neighbors2[0].item() == 2, (
            f"Atom 0 should have 2 neighbors in cutoff2, got {num_neighbors2[0].item()}"
        )
        if half_fill:
            assert num_neighbors1[3].item() == 0
            assert num_neighbors2[3].item() == 0
        else:
            assert num_neighbors1[3].item() == 1
            assert num_neighbors2[3].item() == 2

        # Check neighbor indices are valid
        for i in range(positions.shape[0]):
            for j in range(num_neighbors1[i].item()):
                neighbor_idx = neighbor_matrix1[i, j].item()
                assert 0 <= neighbor_idx < positions.shape[0]
                assert neighbor_idx != i
            for j in range(num_neighbors2[i].item()):
                neighbor_idx = neighbor_matrix2[i, j].item()
                assert 0 <= neighbor_idx < positions.shape[0]
                assert neighbor_idx != i

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_naive_dual_cutoff_pbc_kernel(self, device, dtype):
        """Test _fill_naive_neighbor_matrix_pbc_dual_cutoff kernel."""
        # Simple system with PBC
        positions = torch.tensor(
            [[0.1, 0.1, 0.1], [1.9, 0.1, 0.1]],
            dtype=dtype,
            device=device,
        )
        cell = torch.eye(3, dtype=dtype, device=device) * 2.0
        cutoff1 = 0.3
        cutoff2 = 0.5
        max_neighbors1 = 5
        max_neighbors2 = 10

        # Simple shifts for testing
        shifts = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=torch.int32, device=device
        )

        # Convert to warp types
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell.unsqueeze(0), dtype=wp_mat_dtype)
        wp_shifts = wp.from_torch(shifts, dtype=wp.vec3i)

        # Output arrays
        neighbor_matrix1 = torch.full(
            (positions.shape[0], max_neighbors1), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix2 = torch.full(
            (positions.shape[0], max_neighbors2), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts1 = torch.zeros(
            positions.shape[0], max_neighbors1, 3, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts2 = torch.zeros(
            positions.shape[0], max_neighbors2, 3, dtype=torch.int32, device=device
        )
        num_neighbors1 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )
        num_neighbors2 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
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
            _fill_naive_neighbor_matrix_pbc_dual_cutoff,
            dim=(len(shifts), positions.shape[0]),
            device=wp_device,
            inputs=[
                wp_positions,
                wp_dtype(cutoff1 * cutoff1),
                wp_dtype(cutoff2 * cutoff2),
                wp_cell,
                wp_shifts,
                wp_neighbor_matrix1,
                wp_neighbor_matrix2,
                wp_neighbor_matrix_shifts1,
                wp_neighbor_matrix_shifts2,
                wp_num_neighbors1,
                wp_num_neighbors2,
                True,  # half_fill
            ],
        )

        # Check that we found some neighbors via PBC
        total_neighbors1 = num_neighbors1.sum().item()
        total_neighbors2 = num_neighbors2.sum().item()
        assert total_neighbors2 >= total_neighbors1, (
            "Larger cutoff should find at least as many neighbors"
        )


class TestNaiveDualCutoffWpLaunchers:
    """Test the public launcher API for naive dual cutoff neighbor lists."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_naive_neighbor_matrix_dual_cutoff(self, device, dtype, half_fill):
        """Test naive_neighbor_matrix_dual_cutoff launcher (no PBC)."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        # Prepare output arrays
        neighbor_matrix1 = torch.full(
            (positions.shape[0], max_neighbors1), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix2 = torch.full(
            (positions.shape[0], max_neighbors2), -1, dtype=torch.int32, device=device
        )
        num_neighbors1 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )
        num_neighbors2 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)
        wp_neighbor_matrix1 = wp.from_torch(neighbor_matrix1, dtype=wp.int32)
        wp_neighbor_matrix2 = wp.from_torch(neighbor_matrix2, dtype=wp.int32)
        wp_num_neighbors1 = wp.from_torch(num_neighbors1, dtype=wp.int32)
        wp_num_neighbors2 = wp.from_torch(num_neighbors2, dtype=wp.int32)

        # Call launcher
        naive_neighbor_matrix_dual_cutoff(
            wp_positions,
            cutoff1,
            cutoff2,
            wp_neighbor_matrix1,
            wp_num_neighbors1,
            wp_neighbor_matrix2,
            wp_num_neighbors2,
            wp_dtype,
            str(device),
            half_fill,
        )

        # Verify results
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors2 >= num_neighbors1)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_naive_neighbor_matrix_pbc_dual_cutoff(self, device, dtype, half_fill):
        """Test naive_neighbor_matrix_pbc_dual_cutoff launcher (with PBC)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff1 = 1.0
        cutoff2 = 2.5
        max_neighbors1 = 20
        max_neighbors2 = 35

        # Compute shift ranges
        shift_range_per_dimension, shift_offset, total_shifts = (
            compute_naive_num_shifts(cell.reshape(1, 3, 3), cutoff2, pbc.reshape(1, 3))
        )

        # Expand shifts
        shifts = torch.zeros((total_shifts, 3), dtype=torch.int32, device=device)
        shift_system_idx = torch.zeros(total_shifts, dtype=torch.int32, device=device)
        wp_shift_range = wp.from_torch(shift_range_per_dimension, dtype=wp.vec3i)
        wp_shift_offset = wp.from_torch(shift_offset, dtype=wp.int32)
        wp_shifts = wp.from_torch(shifts, dtype=wp.vec3i)
        wp_shift_system_idx = wp.from_torch(shift_system_idx, dtype=wp.int32)

        wp.launch(
            _expand_naive_shifts,
            dim=cell.reshape(1, 3, 3).shape[0],  # num_systems
            device=str(device),
            inputs=[
                wp_shift_range,
                wp_shift_offset,
                wp_shifts,
                wp_shift_system_idx,
            ],
        )

        # Prepare output arrays
        neighbor_matrix1 = torch.full(
            (positions.shape[0], max_neighbors1), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix2 = torch.full(
            (positions.shape[0], max_neighbors2), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts1 = torch.zeros(
            positions.shape[0], max_neighbors1, 3, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts2 = torch.zeros(
            positions.shape[0], max_neighbors2, 3, dtype=torch.int32, device=device
        )
        num_neighbors1 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )
        num_neighbors2 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell.reshape(1, 3, 3), dtype=wp_mat_dtype)
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
        naive_neighbor_matrix_pbc_dual_cutoff(
            wp_positions,
            cutoff1,
            cutoff2,
            wp_cell,
            wp_shifts,
            wp_neighbor_matrix1,
            wp_neighbor_matrix2,
            wp_neighbor_matrix_shifts1,
            wp_neighbor_matrix_shifts2,
            wp_num_neighbors1,
            wp_num_neighbors2,
            wp_dtype,
            str(device),
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
