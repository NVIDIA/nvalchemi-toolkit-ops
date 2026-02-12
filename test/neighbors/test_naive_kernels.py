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

"""Comprehensive tests for naive neighbor list routines and utilities."""

from importlib import import_module

import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.naive import (
    _fill_naive_neighbor_matrix,
    _fill_naive_neighbor_matrix_pbc,
    naive_neighbor_matrix,
    naive_neighbor_matrix_pbc,
)
from nvalchemiops.neighbors.neighbor_utils import (
    _compute_naive_num_shifts,
    _expand_naive_shifts,
    _update_neighbor_matrix,
    _update_neighbor_matrix_pbc,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

from .test_utils import (
    create_simple_cubic_system,
)

try:
    _ = import_module("vesin")
    run_vesin_checks = True
except ModuleNotFoundError:
    run_vesin_checks = False


@wp.kernel
def _update_neighbor_matrix_kernel(
    i: int,
    j: int,
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    max_neighbors: int,
    half_fill: bool,
):
    _update_neighbor_matrix(
        i, j, neighbor_matrix, num_neighbors, max_neighbors, half_fill
    )


@wp.kernel
def _update_neighbor_matrix_pbc_kernel(
    i: int,
    j: int,
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    unit_shift: wp.vec3i,
    max_neighbors: int,
    half_fill: bool,
):
    _update_neighbor_matrix_pbc(
        i,
        j,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        unit_shift,
        max_neighbors,
        half_fill,
    )


@pytest.mark.parametrize("half_fill", [True, False])
class TestNaiveKernels:
    """Test individual naive neighbor list kernels."""

    def test_update_neighbor_matrix_kernel(self, half_fill):
        """Test _update_neighbor_matrix kernel."""
        wp_device = "cuda:0"
        neighbor_matrix = torch.full((10, 10), -1, dtype=torch.int32, device=wp_device)
        wp_neighbor_matrix = wp.from_torch(neighbor_matrix, dtype=wp.int32)
        num_neighbors = torch.zeros(10, dtype=torch.int32, device=wp_device)
        wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32)

        ij = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        for i, j in ij:
            wp.launch(
                _update_neighbor_matrix_kernel,
                dim=1,
                device=wp_device,
                inputs=[i, j, wp_neighbor_matrix, wp_num_neighbors, 10, half_fill],
            )

        assert num_neighbors[0].item() == 2
        assert num_neighbors[1].item() == 2 if half_fill else 3
        assert num_neighbors[2].item() == 1 if half_fill else 3
        assert num_neighbors[3].item() == 0 if half_fill else 2

        assert neighbor_matrix[0, 0].item() == 1
        assert neighbor_matrix[0, 1].item() == 2
        assert neighbor_matrix[0, 2].item() == -1
        assert neighbor_matrix[0, 3].item() == -1

        if half_fill:
            assert neighbor_matrix[1, 0].item() == 2
            assert neighbor_matrix[1, 1].item() == 3
            assert neighbor_matrix[1, 2].item() == neighbor_matrix[1, 3] == -1

            assert neighbor_matrix[2, 0].item() == 3
            assert neighbor_matrix[2, 1].item() == -1
            assert neighbor_matrix[2, 2].item() == -1
            assert neighbor_matrix[2, 3].item() == -1

            assert neighbor_matrix[3, 0].item() == -1
            assert neighbor_matrix[3, 1].item() == -1
            assert neighbor_matrix[3, 2].item() == -1
            assert neighbor_matrix[3, 3].item() == -1
        else:
            assert neighbor_matrix[1, 0].item() == 0
            assert neighbor_matrix[1, 1].item() == 2
            assert neighbor_matrix[1, 2].item() == 3
            assert neighbor_matrix[1, 3].item() == -1

            assert neighbor_matrix[2, 0].item() == 0
            assert neighbor_matrix[2, 1].item() == 1
            assert neighbor_matrix[2, 2].item() == 3
            assert neighbor_matrix[2, 3].item() == -1

            assert neighbor_matrix[3, 0].item() == 1
            assert neighbor_matrix[3, 1].item() == 2
            assert neighbor_matrix[3, 2].item() == -1
            assert neighbor_matrix[3, 3].item() == -1

    def test_update_neighbor_matrix_pbc_kernel(self, half_fill):
        """Test _update_neighbor_matrix kernel."""
        wp_device = "cuda:0"
        neighbor_matrix = torch.full((10, 10), -1, dtype=torch.int32, device=wp_device)
        neighbor_matrix_shifts = torch.zeros(
            (10, 10, 3), dtype=torch.int32, device=wp_device
        )
        wp_neighbor_matrix = wp.from_torch(neighbor_matrix, dtype=wp.int32)
        wp_neighbor_matrix_shifts = wp.from_torch(
            neighbor_matrix_shifts, dtype=wp.vec3i
        )
        num_neighbors = torch.zeros(10, dtype=torch.int32, device=wp_device)
        wp_unit_shift = wp.vec3i(1, 0, 0)
        wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32)

        ij = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        for i, j in ij:
            wp.launch(
                _update_neighbor_matrix_pbc_kernel,
                dim=1,
                device=wp_device,
                inputs=[
                    i,
                    j,
                    wp_neighbor_matrix,
                    wp_neighbor_matrix_shifts,
                    wp_num_neighbors,
                    wp_unit_shift,
                    10,
                    half_fill,
                ],
            )

        assert num_neighbors[0].item() == 2
        assert num_neighbors[1].item() == 2 if half_fill else 3
        assert num_neighbors[2].item() == 1 if half_fill else 3
        assert num_neighbors[3].item() == 0 if half_fill else 2

        assert neighbor_matrix[0, 0].item() == 1
        assert neighbor_matrix[0, 1].item() == 2
        assert neighbor_matrix[0, 2].item() == -1
        assert neighbor_matrix[0, 3].item() == -1

        if half_fill:
            assert neighbor_matrix[1, 0].item() == 2
            assert neighbor_matrix[1, 1].item() == 3
            assert neighbor_matrix[1, 2].item() == neighbor_matrix[1, 3] == -1

            assert neighbor_matrix[2, 0].item() == 3
            assert neighbor_matrix[2, 1].item() == -1
            assert neighbor_matrix[2, 2].item() == -1
            assert neighbor_matrix[2, 3].item() == -1

            assert neighbor_matrix[3, 0].item() == -1
            assert neighbor_matrix[3, 1].item() == -1
            assert neighbor_matrix[3, 2].item() == -1
            assert neighbor_matrix[3, 3].item() == -1

            assert neighbor_matrix_shifts[1, 0, 0].item() == 1
            assert neighbor_matrix_shifts[1, 1, 0].item() == 1
            assert neighbor_matrix_shifts[1, 2, 0].item() == 0
            assert neighbor_matrix_shifts[1, 3, 0].item() == 0

            assert neighbor_matrix_shifts[2, 0, 0].item() == 1
            assert neighbor_matrix_shifts[2, 1, 0].item() == 0
            assert neighbor_matrix_shifts[2, 2, 0].item() == 0
            assert neighbor_matrix_shifts[2, 3, 0].item() == 0

            assert neighbor_matrix_shifts[3, 0, 0].item() == 0
            assert neighbor_matrix_shifts[3, 1, 0].item() == 0
            assert neighbor_matrix_shifts[3, 2, 0].item() == 0
            assert neighbor_matrix_shifts[3, 3, 0].item() == 0
        else:
            assert neighbor_matrix[1, 0].item() == 0
            assert neighbor_matrix[1, 1].item() == 2
            assert neighbor_matrix[1, 2].item() == 3
            assert neighbor_matrix[1, 3].item() == -1

            assert neighbor_matrix[2, 0].item() == 0
            assert neighbor_matrix[2, 1].item() == 1
            assert neighbor_matrix[2, 2].item() == 3
            assert neighbor_matrix[2, 3].item() == -1

            assert neighbor_matrix[3, 0].item() == 1
            assert neighbor_matrix[3, 1].item() == 2
            assert neighbor_matrix[3, 2].item() == -1
            assert neighbor_matrix[3, 3].item() == -1

            assert neighbor_matrix_shifts[1, 0, 0].item() == -1
            assert neighbor_matrix_shifts[1, 1, 0].item() == 1
            assert neighbor_matrix_shifts[1, 2, 0].item() == 1
            assert neighbor_matrix_shifts[1, 3, 0].item() == 0

            assert neighbor_matrix_shifts[2, 0, 0].item() == -1
            assert neighbor_matrix_shifts[2, 1, 0].item() == -1
            assert neighbor_matrix_shifts[2, 2, 0].item() == 1
            assert neighbor_matrix_shifts[2, 3, 0].item() == 0

            assert neighbor_matrix_shifts[3, 0, 0].item() == -1
            assert neighbor_matrix_shifts[3, 1, 0].item() == -1
            assert neighbor_matrix_shifts[3, 2, 0].item() == 0
            assert neighbor_matrix_shifts[3, 3, 0].item() == 0

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_naive_neighbor_matrix_kernel(self, half_fill, device, dtype):
        """Test _naive_neighbor_matrix kernel (no PBC)."""
        # Create simple system
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],  # Within cutoff
                [1.5, 0.0, 0.0],  # Outside cutoff
                [0.0, 0.7, 0.0],  # Within cutoff
            ],
            dtype=dtype,
            device=device,
        )
        cutoff = 1.0
        max_neighbors = 10

        # Convert to warp types
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_device = str(device)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)

        # Output arrays
        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors), -1, dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        wp_neighbor_matrix = wp.from_torch(neighbor_matrix, dtype=wp.int32)
        wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32)

        neighbor_matrix.fill_(-1)
        num_neighbors.zero_()

        # Launch kernel
        wp.launch(
            _fill_naive_neighbor_matrix,
            dim=positions.shape[0],
            device=wp_device,
            inputs=[
                wp_positions,
                wp_dtype(cutoff * cutoff),
                wp_neighbor_matrix,
                wp_num_neighbors,
                half_fill,
            ],
        )

        # Check results
        # Atom 0 should have atoms 1 and 3 as neighbors
        assert num_neighbors[0].item() == 2, (
            f"Atom 0 should have 2 neighbors, got {num_neighbors[0].item()}"
        )

        if half_fill:
            # In half_fill mode, only i < j relationships are stored
            assert num_neighbors[1].item() == 1, (
                "Atom 1 should have 1 neighbors in half_fill mode"
            )
            assert num_neighbors[3].item() == 0, (
                "Atom 3 should have 0 neighbors in half_fill mode"
            )
        else:
            # In full mode, symmetric relationships are stored
            assert num_neighbors[1].item() == 2, "Atom 1 should have 1 neighbor"
            assert num_neighbors[3].item() == 2, "Atom 3 should have 1 neighbor"

        # Atom 2 should have no neighbors (too far)
        assert num_neighbors[2].item() == 0, "Atom 2 should have no neighbors"

        # Check neighbor indices are valid
        for i in range(positions.shape[0]):
            for j in range(num_neighbors[i].item()):
                neighbor_idx = neighbor_matrix[i, j].item()
                assert 0 <= neighbor_idx < positions.shape[0], (
                    f"Invalid neighbor index: {neighbor_idx}"
                )
                assert neighbor_idx != i, "Atom should not be its own neighbor"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_naive_num_shifts_kernel(self, half_fill, device, dtype):
        """Test _naive_num_shifts kernel."""
        # Create test cell and PBC
        cell = torch.eye(3, dtype=dtype, device=device) * 3.0
        pbc = torch.tensor([True, True, True], device=device)
        cutoff = 1.5

        # Convert to warp types
        wp_dtype = get_wp_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)

        wp_cell = wp.from_torch(cell.unsqueeze(0), dtype=wp_mat_dtype)
        wp_pbc = wp.from_torch(pbc.unsqueeze(0), dtype=wp.bool)

        # Output arrays
        num_shifts = torch.zeros(1, dtype=torch.int32, device=device)
        shift_range = torch.zeros(1, 3, dtype=torch.int32, device=device)

        wp_num_shifts = wp.from_torch(num_shifts, dtype=wp.int32)
        wp_shift_range = wp.from_torch(shift_range, dtype=wp.vec3i)

        # Launch kernel
        wp.launch(
            _compute_naive_num_shifts,
            dim=1,
            device=wp_device,
            inputs=[
                wp_cell,
                wp_dtype(cutoff),
                wp_pbc,
                wp_num_shifts,
                wp_shift_range,
            ],
        )

        # Check results
        assert num_shifts[0].item() > 0, "Should need some periodic shifts"
        assert torch.all(shift_range >= 0), "Shift ranges should be non-negative"

        # Test with mixed PBC
        pbc_mixed = torch.tensor([True, False, True], device=device)
        wp_pbc_mixed = wp.from_torch(pbc_mixed.unsqueeze(0), dtype=wp.bool)

        num_shifts.zero_()
        shift_range.zero_()

        wp.launch(
            _compute_naive_num_shifts,
            dim=1,
            device=wp_device,
            inputs=[
                wp_cell,
                wp_dtype(cutoff),
                wp_pbc_mixed,
                wp_num_shifts,
                wp_shift_range,
            ],
        )

        # Y-direction should have zero shifts (non-periodic)
        assert shift_range[0, 1].item() == 0, "Y-direction should have zero shifts"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_expand_naive_shifts_kernel(self, half_fill, device):
        """Test _naive_expand_shifts kernel."""
        wp_device = str(device)

        # Simple test data
        cell = (torch.eye(3, dtype=torch.float32, device=device) * 3.0).reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device).reshape(
            1, 3
        )
        cutoff = 1.5
        shift_range_per_dimension, shift_offset, total_shifts = (
            compute_naive_num_shifts(cell, cutoff, pbc)
        )

        # Output arrays
        shifts = torch.zeros(total_shifts, 3, dtype=torch.int32, device=device)
        shift_system_idx = torch.zeros(total_shifts, dtype=torch.int32, device=device)

        # Launch kernel
        wp.launch(
            _expand_naive_shifts,
            dim=1,  # One system
            device=wp_device,
            inputs=[
                shift_range_per_dimension,
                shift_offset,
                shifts,
                shift_system_idx,
            ],
        )
        # Check results
        assert torch.all(shift_system_idx == 0), "All shifts should belong to system 0"

        # Check that we have the expected shifts (avoiding double counting)
        expected_shifts = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, -1],
            [0, 1, 0],
            [0, 1, 1],
            [1, -1, -1],
            [1, -1, 0],
            [1, -1, 1],
            [1, 0, -1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, -1],
            [1, 1, 0],
            [1, 1, 1],
        ]

        assert len(shifts) == len(expected_shifts), (
            f"Expected {len(expected_shifts)} shifts, got {len(shifts)}"
        )

        # Check that (0,0,0) shift is included
        zero_shift_found = False
        for i in range(len(shifts)):
            if torch.all(shifts[i] == 0):
                zero_shift_found = True
                break
        assert zero_shift_found, "Should include (0,0,0) shift"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_naive_neighbor_matrix_pbc_with_shifts_kernel(
        self, half_fill, device, dtype
    ):
        """Test _naive_neighbor_matrix_pbc kernel."""
        # Simple system with PBC
        positions = torch.tensor(
            [[0.1, 0.1, 0.1], [1.9, 0.1, 0.1]],  # Should be neighbors via PBC
            dtype=dtype,
            device=device,
        )
        cell = torch.eye(3, dtype=dtype, device=device) * 2.0
        cutoff = 0.5
        max_neighbors = 5

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
        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            positions.shape[0], max_neighbors, 3, dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        wp_neighbor_matrix = wp.from_torch(neighbor_matrix, dtype=wp.int32)
        wp_neighbor_matrix_shifts = wp.from_torch(
            neighbor_matrix_shifts, dtype=wp.vec3i
        )
        wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32)

        # Launch kernel
        wp.launch(
            _fill_naive_neighbor_matrix_pbc,
            dim=(len(shifts), positions.shape[0]),
            device=wp_device,
            inputs=[
                wp_positions,
                wp_dtype(cutoff * cutoff),
                wp_cell,
                wp_shifts,
                wp_neighbor_matrix,
                wp_neighbor_matrix_shifts,
                wp_num_neighbors,
                True,  # half_fill
            ],
        )

        # Check that we found some neighbors via PBC
        total_neighbors = num_neighbors.sum().item()
        assert total_neighbors > 0, "Should find neighbors via PBC"


@pytest.mark.parametrize("half_fill", [True, False])
class TestNaiveWpLaunchers:
    """Test the public launcher API for naive neighbor lists."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_naive_neighbor_matrix(self, device, dtype, half_fill):
        """Test naive_neighbor_matrix launcher (no PBC)."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20

        # Prepare output arrays
        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            positions.shape[0],
            dtype=torch.int32,
            device=device,
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)
        wp_neighbor_matrix = wp.from_torch(neighbor_matrix, dtype=wp.int32)
        wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32)

        # Call launcher
        naive_neighbor_matrix(
            wp_positions,
            cutoff,
            wp_neighbor_matrix,
            wp_num_neighbors,
            wp_dtype,
            str(device),
            half_fill,
        )

        # Verify results
        assert torch.all(num_neighbors >= 0), "Neighbor counts should be non-negative"
        assert num_neighbors.sum() > 0, "Should find some neighbors"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_naive_neighbor_matrix_pbc(self, device, dtype, half_fill):
        """Test naive_neighbor_matrix_pbc launcher (with PBC)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1

        max_neighbors = 30
        # Compute shift ranges
        shift_range_per_dimension, shift_offset, total_shifts = (
            compute_naive_num_shifts(cell, cutoff, pbc)
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
            dim=cell.shape[0],  # num_systems
            device=str(device),
            inputs=[
                wp_shift_range,
                wp_shift_offset,
                wp_shifts,
                wp_shift_system_idx,
            ],
        )

        # Prepare output arrays
        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            positions.shape[0],
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts = torch.zeros(
            (positions.shape[0], max_neighbors, 3),
            dtype=torch.int32,
            device=device,
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)
        wp_neighbor_matrix = wp.from_torch(neighbor_matrix, dtype=wp.int32)
        wp_neighbor_matrix_shifts = wp.from_torch(
            neighbor_matrix_shifts, dtype=wp.vec3i
        )
        wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32)

        # Call launcher
        naive_neighbor_matrix_pbc(
            wp_positions,
            cutoff,
            wp_cell,
            wp_shifts,
            wp_neighbor_matrix,
            wp_neighbor_matrix_shifts,
            wp_num_neighbors,
            wp_dtype,
            str(device),
            half_fill,
        )

        # Verify results
        assert torch.all(num_neighbors >= 0), "Neighbor counts should be non-negative"
        assert num_neighbors.sum() > 0, "Should find some neighbors"

        # Check that unit_shifts are reasonable
        valid_shifts = neighbor_matrix_shifts[neighbor_matrix != -1]
        if len(valid_shifts) > 0:
            assert torch.all(torch.abs(valid_shifts) <= 5), (
                "Unit shifts should be small integers"
            )
