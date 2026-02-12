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

"""Tests for batch naive neighbor list kernels and launchers."""

import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.batch_naive import (
    _fill_batch_naive_neighbor_matrix,
    _fill_batch_naive_neighbor_matrix_pbc,
    batch_naive_neighbor_matrix,
    batch_naive_neighbor_matrix_pbc,
)
from nvalchemiops.neighbors.neighbor_utils import _expand_naive_shifts
from nvalchemiops.torch.neighbors.neighbor_utils import compute_naive_num_shifts
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

from .test_utils import create_batch_systems


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
        assert num_neighbors.sum().item() > 0, "Should find some neighbors"

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


class TestBatchNaiveWpLaunchers:
    """Test the public launcher API for batch naive neighbor lists."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_matrix(self, device, dtype, half_fill):
        """Test batch_naive_neighbor_matrix launcher (no PBC)."""
        atoms_per_system = [5, 7]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)
        cutoff = 1.2
        max_neighbors = 20

        # Prepare output arrays
        total_atoms = positions_batch.shape[0]
        neighbor_matrix = torch.full(
            (total_atoms, max_neighbors), total_atoms, dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(total_atoms, dtype=torch.int32, device=device)

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)
        wp_neighbor_matrix = wp.from_torch(neighbor_matrix, dtype=wp.int32)
        wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32)

        # Call launcher
        batch_naive_neighbor_matrix(
            wp_positions,
            cutoff,
            wp_batch_idx,
            wp_batch_ptr,
            wp_neighbor_matrix,
            wp_num_neighbors,
            wp_dtype,
            device,
            half_fill,
        )

        # Verify results
        assert torch.all(num_neighbors >= 0), "Neighbor counts should be non-negative"
        assert num_neighbors.sum().item() > 0, "Should find some neighbors"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_matrix_pbc(self, device, dtype, half_fill):
        """Test batch_naive_neighbor_matrix_pbc launcher (with PBC)."""
        atoms_per_system = [4, 5]
        num_systems = 2
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=num_systems,
            atoms_per_system=atoms_per_system,
            dtype=dtype,
            device=device,
        )

        _, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)
        cutoff = 1.2
        max_neighbors = 25
        max_atoms_per_system = max(atoms_per_system)

        # Compute shift ranges
        shift_range_per_dimension, shift_offset, total_shifts = (
            compute_naive_num_shifts(cell_batch, cutoff, pbc_batch)
        )

        # Expand shift ranges into actual shift vectors
        shifts = torch.zeros(total_shifts, 3, dtype=torch.int32, device=device)
        shift_system_idx = torch.zeros(total_shifts, dtype=torch.int32, device=device)

        wp.launch(
            _expand_naive_shifts,
            dim=num_systems,
            device=device,
            inputs=[
                shift_range_per_dimension,
                shift_offset,
                shifts,
                shift_system_idx,
            ],
        )

        # Prepare output arrays
        total_atoms = positions_batch.shape[0]
        neighbor_matrix = torch.full(
            (total_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (total_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(total_atoms, dtype=torch.int32, device=device)

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell_batch, dtype=wp_mat_dtype)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)
        wp_shifts = wp.from_torch(shifts, dtype=wp.vec3i)
        wp_shift_system_idx = wp.from_torch(shift_system_idx, dtype=wp.int32)
        wp_neighbor_matrix = wp.from_torch(neighbor_matrix, dtype=wp.int32)
        wp_neighbor_matrix_shifts = wp.from_torch(
            neighbor_matrix_shifts, dtype=wp.vec3i
        )
        wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32)

        # Call launcher
        batch_naive_neighbor_matrix_pbc(
            wp_positions,
            wp_cell,
            cutoff,
            wp_batch_ptr,
            wp_shifts,
            wp_shift_system_idx,
            wp_neighbor_matrix,
            wp_neighbor_matrix_shifts,
            wp_num_neighbors,
            wp_dtype,
            device,
            max_atoms_per_system,
            half_fill,
        )

        # Verify results
        assert torch.all(num_neighbors >= 0), "Neighbor counts should be non-negative"
        assert num_neighbors.sum().item() > 0, "Should find some neighbors"

        # Check that unit shifts are reasonable
        valid_shifts = neighbor_matrix_shifts[neighbor_matrix != -1]
        if len(valid_shifts) > 0:
            assert torch.all(torch.abs(valid_shifts) <= 5), (
                "Unit shifts should be small integers"
            )
