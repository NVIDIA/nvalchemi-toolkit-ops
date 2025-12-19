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

"""Comprehensive tests for naive neighbor list routines and utilities."""

from importlib import import_module

import pytest
import torch
import warp as wp

from nvalchemiops.neighborlist.naive import (
    _fill_naive_neighbor_matrix,
    _fill_naive_neighbor_matrix_pbc,
    _naive_neighbor_matrix_no_pbc,
    _naive_neighbor_matrix_pbc,
    naive_neighbor_list,
)
from nvalchemiops.neighborlist.neighbor_utils import (
    NeighborOverflowError,
    _compute_naive_num_shifts,
    _expand_naive_shifts,
    _update_neighbor_matrix,
    _update_neighbor_matrix_pbc,
    compute_naive_num_shifts,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

from .test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
    create_random_system,
    create_simple_cubic_system,
    create_structure_HoTlPd,
    create_structure_SiCu,
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


class TestNaiveKernels:
    """Test individual naive neighbor list kernels."""

    @pytest.mark.parametrize("half_fill", [True, False])
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

    @pytest.mark.parametrize("half_fill", [True, False])
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

    @pytest.mark.parametrize("half_fill", [True, False])
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
    def test_naive_num_shifts_kernel(self, device, dtype):
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
    def test_expand_naive_shifts_kernel(self, device):
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
    def test_naive_neighbor_matrix_pbc_with_shifts_kernel(self, device, dtype):
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


class TestNaiveUtilityFunctions:
    """Test utility functions used by naive neighbor list."""

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_naive_neighbor_matrix_no_pbc_function(self, device, dtype, half_fill):
        """Test _naive_neighbor_matrix_no_pbc function."""
        """Test _fill_neighbor_matrix custom op."""
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

        _naive_neighbor_matrix_no_pbc(
            positions,
            cutoff,
            neighbor_matrix,
            num_neighbors,
            half_fill,
        )
        # Check neighbor counts are reasonable
        assert torch.all(num_neighbors >= 0)

        # Compare with brute force reference (for small systems)
        if positions.shape[0] <= 27:
            cell = (
                torch.eye(3, dtype=dtype, device=device) * 10.0
            )  # Large cell (no PBC effect)
            pbc = torch.tensor([False, False, False], device=device)
            i_ref, _, _, _ = brute_force_neighbors(positions, cell, pbc, cutoff)

            # Convert neighbor matrix to neighbor list format for comparison
            neighbor_list, _ = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                fill_value=positions.shape[0],
            )
            if half_fill:
                # For half_fill, we expect roughly half the pairs
                assert len(neighbor_list[0]) <= len(i_ref)
            else:
                # For full fill, we expect symmetric pairs
                assert len(neighbor_list[0]) == len(i_ref)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_neighbor_matrix_pbc_function(self, device, dtype, half_fill):
        """Test _naive_neighbor_matrix_pbc function."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1

        max_neighbors = 30
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
        shift_range_per_dimension, shift_offset, total_shifts = (
            compute_naive_num_shifts(cell.reshape(1, 3, 3), cutoff, pbc.reshape(1, 3))
        )
        neighbor_matrix.fill_(positions.shape[0])
        num_neighbors.zero_()

        _naive_neighbor_matrix_pbc(
            positions,
            cutoff,
            cell.reshape(1, 3, 3),
            pbc.reshape(1, 3),
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            shift_range_per_dimension,
            shift_offset,
            total_shifts,
            half_fill,
        )

        # Check output shapes and types
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

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_get_neighbor_list_from_neighbor_matrix(self, device):
        """Test get_neighbor_list_from_neighbor_matrix utility function."""
        # Create simple neighbor matrix
        neighbor_matrix = torch.tensor(
            [
                [1, 2, -1, -1],
                [0, 2, -1, -1],
                [0, 1, -1, -1],
                [-1, -1, -1, -1],
            ],
            dtype=torch.int32,
            device=device,
        )

        num_neighbors = torch.tensor([2, 2, 2, 0], dtype=torch.int32, device=device)
        # Test without shift matrix
        neighbor_list, _ = get_neighbor_list_from_neighbor_matrix(
            neighbor_matrix, num_neighbors=num_neighbors, fill_value=-1
        )

        assert neighbor_list.shape[0] == 2, "Should return stacked (i, j) tensor"
        i = neighbor_list[0]
        j = neighbor_list[1]
        assert i.dtype == torch.int32
        assert j.dtype == torch.int32
        assert len(i) == len(j)
        assert len(i) == 6, "Should have 6 neighbor pairs"

        # Test with shift matrix
        shift_matrix = torch.zeros_like(neighbor_matrix).unsqueeze(-1).expand(-1, -1, 3)
        neighbor_list, neighbor_ptr, shifts = get_neighbor_list_from_neighbor_matrix(
            neighbor_matrix,
            num_neighbors=num_neighbors,
            neighbor_shift_matrix=shift_matrix,
            fill_value=-1,
        )

        assert neighbor_list.shape[0] == 2, "Should return stacked (i, j) tensor"
        assert shifts.shape == (6, 3), "Should return shift vectors"

        # Test error conditions
        with pytest.raises(NeighborOverflowError):
            num_neighbors = torch.tensor([8, 8, 8, 8], dtype=torch.int32, device=device)
            get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                neighbor_shift_matrix=shift_matrix,
                fill_value=-1,
            )


class TestNaiveMainAPI:
    """Test the main naive neighbor list API function."""

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", ["cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("half_fill", [False])
    @pytest.mark.parametrize("pbc_flag", [True])
    @pytest.mark.parametrize("preallocate", [True])
    @pytest.mark.parametrize("return_neighbor_list", [True])
    @pytest.mark.parametrize("fill_value", [-1])
    def test_naive_neighbor_matrix_function(
        self,
        device,
        dtype,
        half_fill,
        pbc_flag,
        preallocate,
        return_neighbor_list,
        fill_value,
    ):
        """Test _naive_neighbor_matrix_no_pbc function."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20

        if not pbc_flag:
            cell = None
            pbc = None
        if preallocate:
            neighbor_matrix = torch.full(
                (positions.shape[0], max_neighbors),
                fill_value,
                dtype=torch.int32,
                device=device,
            )
            num_neighbors = torch.zeros(
                positions.shape[0], dtype=torch.int32, device=device
            )
            args = (positions, cutoff)
            kwargs = {
                "fill_value": fill_value,
                "half_fill": half_fill,
                "neighbor_matrix": neighbor_matrix,
                "num_neighbors": num_neighbors,
                "return_neighbor_list": return_neighbor_list,
            }
            if pbc_flag:
                shift_range_per_dimension, shift_offset, total_shifts = (
                    compute_naive_num_shifts(cell, cutoff, pbc)
                )
                kwargs["cell"] = cell
                kwargs["pbc"] = pbc
                kwargs["shift_range_per_dimension"] = shift_range_per_dimension
                kwargs["shift_offset"] = shift_offset
                kwargs["total_shifts"] = total_shifts
                neighbor_matrix_shifts = torch.zeros(
                    (positions.shape[0], max_neighbors, 3),
                    dtype=torch.int32,
                    device=device,
                )
                kwargs["neighbor_matrix_shifts"] = neighbor_matrix_shifts
            results = naive_neighbor_list(*args, **kwargs)
            if return_neighbor_list:
                if pbc_flag:
                    neighbor_list, neighbor_ptr, neighbor_shifts = results
                    idx_i = neighbor_list[0]
                    idx_j = neighbor_list[1]
                    u = neighbor_shifts
                    num_neighbors = neighbor_ptr[1:] - neighbor_ptr[:-1]
                else:
                    neighbor_list, neighbor_ptr = results
                    idx_i = neighbor_list[0]
                    idx_j = neighbor_list[1]
                    u = torch.zeros(
                        (idx_i.shape[0], 3), dtype=torch.int32, device=device
                    )
                    num_neighbors = neighbor_ptr[1:] - neighbor_ptr[:-1]
        else:
            args = (positions, cutoff)
            kwargs = {
                "max_neighbors": max_neighbors,
                "fill_value": fill_value,
                "half_fill": half_fill,
                "return_neighbor_list": return_neighbor_list,
            }
            if pbc_flag:
                kwargs["cell"] = cell
                kwargs["pbc"] = pbc
            results = naive_neighbor_list(*args, **kwargs)
            if pbc_flag:
                if return_neighbor_list:
                    neighbor_list, neighbor_ptr, neighbor_shifts = results
                    idx_i = neighbor_list[0]
                    idx_j = neighbor_list[1]
                    u = neighbor_shifts
                    num_neighbors = neighbor_ptr[1:] - neighbor_ptr[:-1]
                else:
                    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = results
            else:
                if return_neighbor_list:
                    neighbor_list, neighbor_ptr = results
                    idx_i = neighbor_list[0]
                    idx_j = neighbor_list[1]
                    u = torch.zeros(
                        (idx_i.shape[0], 3), dtype=torch.int32, device=device
                    )
                    num_neighbors = neighbor_ptr[1:] - neighbor_ptr[:-1]
                else:
                    neighbor_matrix, num_neighbors = results

        # Check output shapes and types
        assert num_neighbors.dtype == torch.int32
        assert num_neighbors.shape == (positions.shape[0],)
        assert num_neighbors.device == torch.device(device)
        if return_neighbor_list:
            assert neighbor_list.dtype == torch.int32
            assert neighbor_list.shape == (2, num_neighbors.sum())
            assert neighbor_list.device == torch.device(device)

            if pbc_flag:
                assert u.dtype == torch.int32
                assert u.shape == (num_neighbors.sum(), 3)
                assert u.device == torch.device(device)

        else:
            assert neighbor_matrix.dtype == torch.int32
            assert neighbor_matrix.shape == (
                positions.shape[0],
                max_neighbors,
            )
            assert neighbor_matrix.device == torch.device(device)

            if pbc_flag:
                assert neighbor_matrix_shifts.dtype == torch.int32
                assert neighbor_matrix_shifts.shape == (
                    positions.shape[0],
                    max_neighbors,
                    3,
                )
                assert neighbor_matrix_shifts.device == torch.device(device)

        # Get reference result
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)

        if return_neighbor_list and not half_fill:
            assert_neighbor_lists_equal((idx_i, idx_j, u), (i_ref, j_ref, u_ref))

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_naive_neighbor_list_edge_cases(self, device, dtype, half_fill):
        """Test edge cases for naive_neighbor_list."""
        # Empty system
        positions_empty = torch.empty(0, 3, dtype=dtype, device=device)
        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions=positions_empty,
            cutoff=1.0,
            pbc=None,
            cell=None,
            max_neighbors=10,
            half_fill=half_fill,
        )
        assert neighbor_matrix.shape == (0, 10)
        assert num_neighbors.shape == (0,)

        # Single atom
        positions_single = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions=positions_single,
            cutoff=1.0,
            pbc=None,
            cell=None,
            max_neighbors=10,
            half_fill=half_fill,
        )
        assert num_neighbors[0].item() == 0, "Single atom should have no neighbors"

        # Zero cutoff
        positions, _, _ = create_simple_cubic_system(
            num_atoms=4, dtype=dtype, device=device
        )
        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions=positions,
            cutoff=0.0,
            pbc=None,
            cell=None,
            max_neighbors=10,
            half_fill=half_fill,
        )
        assert torch.all(num_neighbors == 0), "Zero cutoff should find no neighbors"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_naive_neighbor_list_error_conditions(self, device, dtype, half_fill):
        """Test error conditions for naive_neighbor_list."""
        positions, cell, pbc = create_simple_cubic_system(dtype=dtype, device=device)

        # Test mismatched cell and pbc arguments
        with pytest.raises(
            ValueError, match="If cell is provided, pbc must also be provided"
        ):
            naive_neighbor_list(
                positions,
                1.0,
                pbc=None,
                cell=cell,
                max_neighbors=10,
            )

        with pytest.raises(
            ValueError, match="If pbc is provided, cell must also be provided"
        ):
            naive_neighbor_list(
                positions,
                1.0,
                pbc=pbc,
                cell=None,
                max_neighbors=10,
            )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_num_neighbors_HoTlPd(self, device, dtype):
        positions, cell, pbc = create_structure_HoTlPd(dtype, device)
        reference = [
            torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
            torch.tensor([[13, 13, 13, 14, 14, 14, 11, 11, 11]]),
            torch.tensor([[42, 42, 42, 36, 36, 36, 41, 41, 44]]),
        ]
        for i, cutoff in enumerate((1.0, 4.0, 6.0)):
            _, num_neighbors, _ = naive_neighbor_list(
                positions=positions, cutoff=cutoff, pbc=pbc, cell=cell
            )
            assert (num_neighbors.cpu() == reference[i]).all()

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_num_neighbors_SiCu(self, device, dtype):
        positions, cell, pbc = create_structure_SiCu(dtype, device)
        reference = [
            torch.tensor([[0, 0]]),
            torch.tensor([[6, 6]]),
            torch.tensor([[26, 26]]),
        ]
        for i, cutoff in enumerate((1.0, 4.0, 6.0)):
            _, num_neighbors, _ = naive_neighbor_list(
                positions=positions, cutoff=cutoff, pbc=pbc, cell=cell
            )
            assert (num_neighbors.cpu() == reference[i]).all()


class TestNaivePerformanceAndScaling:
    """Test performance characteristics and scaling of naive implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_naive_scaling_with_system_size(self, device):
        """Test that naive implementation scales as expected with system size."""
        import time

        dtype = torch.float32
        cutoff = 1.1
        max_neighbors = 100

        # Test different system sizes
        sizes = [10, 50, 100] if device == "cpu" else [50, 100, 200]
        times = []

        for num_atoms in sizes:
            positions, cell, pbc = create_simple_cubic_system(
                num_atoms=num_atoms, dtype=dtype, device=device
            )

            # Warm up
            for _ in range(10):
                naive_neighbor_list(
                    positions,
                    cutoff,
                    pbc=pbc,
                    cell=cell,
                    max_neighbors=max_neighbors,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            # Time the operation
            start_time = time.time()
            for _ in range(100):
                naive_neighbor_list(
                    positions,
                    cutoff,
                    pbc=pbc,
                    cell=cell,
                    max_neighbors=max_neighbors,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            times.append(elapsed)

        # Check that it doesn't grow too fast (should be roughly O(N^2))
        # This is a loose check since we can't expect perfect scaling
        assert times[1] > times[0] * 0.8, "Time should increase with system size"
        if len(times) > 2:
            # Very loose scaling check
            scaling_factor = times[-1] / times[0]
            size_factor = (sizes[-1] / sizes[0]) ** 2
            assert scaling_factor < size_factor * 5, (
                "Scaling should not be much worse than O(N^2)"
            )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_naive_cutoff_scaling(self, device):
        """Test scaling with different cutoff values."""
        dtype = torch.float32
        num_atoms = 50
        max_neighbors = 200

        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=num_atoms, dtype=dtype, device=device
        )

        # Test different cutoffs
        cutoffs = [0.5, 1.0, 1.5, 2.0]
        neighbor_counts = []

        for cutoff in cutoffs:
            _, num_neighbors, _ = naive_neighbor_list(
                positions,
                cutoff,
                pbc=pbc,
                cell=cell,
                max_neighbors=max_neighbors,
            )
            total_pairs = num_neighbors.sum().item()
            neighbor_counts.append(total_pairs)

        # Check that neighbor count increases with cutoff
        for i in range(1, len(neighbor_counts)):
            assert neighbor_counts[i] >= neighbor_counts[i - 1], (
                f"Neighbor count should increase with cutoff: {neighbor_counts}"
            )


class TestNaiveRobustness:
    """Test robustness of naive implementation to various inputs."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_random_systems(self, device, dtype, half_fill):
        """Test with random systems of various sizes and configurations."""
        for pbc_flag in [True, False]:
            # Test several random systems
            for seed in [42, 123, 456]:
                positions, cell, pbc = create_random_system(
                    num_atoms=20,
                    cell_size=3.0,
                    dtype=dtype,
                    device=device,
                    seed=seed,
                    pbc_flag=pbc_flag,
                )
                cutoff = 1.2
                max_neighbors = 50

                # Should not crash
                neighbor_matrix, num_neighbors, unit_shifts = naive_neighbor_list(
                    positions=positions,
                    cutoff=cutoff,
                    pbc=pbc,
                    cell=cell,
                    max_neighbors=max_neighbors,
                    half_fill=half_fill,
                )

                # Basic sanity checks
                assert torch.all(num_neighbors >= 0)
                assert torch.all(num_neighbors <= max_neighbors)
                assert neighbor_matrix.device == torch.device(device)
                assert unit_shifts.device == torch.device(device)
                assert num_neighbors.device == torch.device(device)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_extreme_geometries(self, device, dtype, half_fill):
        """Test with extreme cell geometries."""
        # Very elongated cell
        positions = torch.rand(10, 3, dtype=dtype, device=device)
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]],
            dtype=dtype,
            device=device,
        ).reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], device=device).reshape(1, 3)
        cutoff = 0.2
        max_neighbors = 20

        # Should handle extreme aspect ratios
        _, num_neighbors, _ = naive_neighbor_list(
            positions=positions * torch.tensor([10.0, 0.1, 0.1], device=device),
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        assert torch.all(num_neighbors >= 0)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_large_cutoffs(self, device, dtype, half_fill):
        """Test with very large cutoffs."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )

        # Cutoff larger than cell size
        large_cutoff = 5.0
        max_neighbors = 200

        _, num_neighbors, _ = naive_neighbor_list(
            positions=positions,
            cutoff=large_cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        # Should find many neighbors
        assert num_neighbors.sum() > 0
        # Each atom should have multiple neighbors (including periodic images)
        assert torch.all(num_neighbors > 0)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_precision_consistency(self, device, dtype, half_fill):
        """Test that float32 and float64 give consistent results."""
        positions_f32, cell_f32, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=torch.float32, device=device
        )
        positions_f64 = positions_f32.double()
        cell_f64 = cell_f32.double()

        cutoff = 1.1
        max_neighbors = 50

        # Get results for both precisions
        _, num_neighbors_f32, _ = naive_neighbor_list(
            positions_f32,
            cutoff,
            pbc=pbc,
            cell=cell_f32,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )
        _, num_neighbors_f64, _ = naive_neighbor_list(
            positions_f64,
            cutoff,
            pbc=pbc,
            cell=cell_f64,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        # Neighbor counts should be identical (for this exact geometry)
        torch.testing.assert_close(num_neighbors_f32, num_neighbors_f64, rtol=0, atol=0)


class TestNaiveTorchCompilability:
    """Test torch.compile compatibility for naive neighbor list functions."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [False, True])
    def test_naive_neighbor_list_compile_no_pbc(self, device, dtype, half_fill):
        """Test that naive_neighbor_list can be compiled with torch.compile."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=50, dtype=dtype, device=device
        )
        cutoff = 3.0
        max_neighbors = 100

        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            50,
            dtype=torch.int32,
            device=device,
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        # Test compiled version
        @torch.compile
        def compiled_naive_neighbor_list(
            positions,
            cutoff,
            neighbor_matrix,
            num_neighbors,
            half_fill,
        ):
            return naive_neighbor_list(
                positions=positions,
                cutoff=cutoff,
                neighbor_matrix=neighbor_matrix,
                num_neighbors=num_neighbors,
                half_fill=half_fill,
            )

        compiled_naive_neighbor_list(
            positions,
            cutoff,
            neighbor_matrix,
            num_neighbors,
            half_fill,
        )

        assert num_neighbors.sum() > 0
        num_rows = positions.shape[0] - int(half_fill)
        for i in range(num_rows):
            assert num_neighbors[i].item() > 0
            neighbor_row = neighbor_matrix[i]
            mask = neighbor_row != 50
            assert neighbor_row[mask].shape == (num_neighbors[i].item(),)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_naive_neighbor_list_compile_pbc(self, device, dtype, half_fill):
        """Test that naive_neighbor_list can be compiled with torch.compile."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=50, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 50
        cell = cell.reshape(1, 3, 3)
        pbc = pbc.reshape(1, 3)
        shift_range_per_dimension, shift_offset, total_shifts = (
            compute_naive_num_shifts(cell, cutoff, pbc)
        )

        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            50,
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

        # Test compiled version
        @torch.compile
        def compiled_naive_neighbor_list(
            positions,
            cutoff,
            cell,
            pbc,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            shift_range_per_dimension,
            shift_offset,
            total_shifts,
            half_fill,
        ):
            return naive_neighbor_list(
                positions=positions,
                cutoff=cutoff,
                cell=cell,
                pbc=pbc,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                num_neighbors=num_neighbors,
                shift_range_per_dimension=shift_range_per_dimension,
                shift_offset=shift_offset,
                total_shifts=total_shifts,
                half_fill=half_fill,
            )

        compiled_naive_neighbor_list(
            positions,
            cutoff,
            cell,
            pbc,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            shift_range_per_dimension,
            shift_offset,
            total_shifts,
            half_fill,
        )

        # Compare results
        assert num_neighbors.sum() > 0
        num_rows = positions.shape[0] - int(half_fill)
        for i in range(num_rows):
            assert num_neighbors[i].item() > 0
            neighbor_row = neighbor_matrix[i]
            mask = neighbor_row != 50
            assert neighbor_row[mask].shape == (num_neighbors[i].item(),)


class TestNaiveMemoryAndPerformance:
    """Test memory usage and performance characteristics of naive implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_memory_scaling(self, device, half_fill):
        """Test that memory usage scales reasonably with system size."""
        import gc

        dtype = torch.float32
        cutoff = 1.1

        # Test different system sizes
        sizes = [10, 20] if device == "cpu" else [50, 100]

        for num_atoms in sizes:
            positions, cell, pbc = create_simple_cubic_system(
                num_atoms=num_atoms, dtype=dtype, device=device
            )
            cell = cell.reshape(1, 3, 3)
            pbc = pbc.reshape(1, 3)

            # Estimate reasonable max_neighbors based on system size and cutoff
            max_neighbors = 100

            # Clear cache before test
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

            # Run batch naive implementation
            neighbor_matrix, num_neighbors, unit_shifts = naive_neighbor_list(
                positions=positions,
                cutoff=cutoff,
                pbc=pbc,
                cell=cell,
                max_neighbors=max_neighbors,
                half_fill=half_fill,
            )

            # Basic checks that output is reasonable
            assert neighbor_matrix.shape == (
                num_atoms,
                max_neighbors,
            )
            assert unit_shifts.shape == (num_atoms, max_neighbors, 3)
            assert num_neighbors.shape == (num_atoms,)
            assert torch.all(num_neighbors >= 0)
            assert torch.all(num_neighbors <= max_neighbors)

            # Clean up
            del neighbor_matrix, unit_shifts, num_neighbors, positions, cell, pbc
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_max_neighbors_overflow_handling(self, device, dtype, half_fill):
        """Test behavior when max_neighbors is exceeded."""

        # Create a dense system with small max_neighbors to force overflow
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cell = cell.reshape(1, 3, 3)
        pbc = pbc.reshape(1, 3)

        cutoff = 2.0  # Large cutoff to find many neighbors
        max_neighbors = 3  # Artificially small to trigger overflow

        # Should not crash, but may not find all neighbors
        neighbor_matrix, num_neighbors, unit_shifts = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        # Should still produce valid output, just potentially incomplete
        assert torch.all(num_neighbors >= 0)
        assert neighbor_matrix.shape == (positions.shape[0], max_neighbors)
        assert unit_shifts.shape == (positions.shape[0], max_neighbors, 3)
        assert num_neighbors.shape == (positions.shape[0],)
        assert neighbor_matrix.device == torch.device(device)
        assert unit_shifts.device == torch.device(device)
        assert num_neighbors.device == torch.device(device)
