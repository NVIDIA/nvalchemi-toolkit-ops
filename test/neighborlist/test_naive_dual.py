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

"""Comprehensive tests for naive dual cutoff neighbor list routines and utilities."""

from importlib import import_module

import pytest
import torch
import warp as wp

from nvalchemiops.neighborlist.naive_dual_cutoff import (
    _fill_naive_neighbor_matrix_dual_cutoff,
    _fill_naive_neighbor_matrix_pbc_dual_cutoff,
    _naive_neighbor_matrix_no_pbc_dual_cutoff,
    _naive_neighbor_matrix_pbc_dual_cutoff,
    naive_neighbor_list_dual_cutoff,
)
from nvalchemiops.neighborlist.neighbor_utils import compute_naive_num_shifts
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

from .test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
    create_random_system,
    create_simple_cubic_system,
)

try:
    _ = import_module("vesin")
    run_vesin_checks = True
except ModuleNotFoundError:
    run_vesin_checks = False


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
                wp_dtype(cutoff1 * cutoff1),  # batch version uses squared cutoffs
                wp_dtype(cutoff2 * cutoff2),  # batch version uses squared cutoffs
                wp_neighbor_matrix1,
                wp_num_neighbors1,
                wp_neighbor_matrix2,
                wp_num_neighbors2,
                half_fill,
            ],
        )

        # Check results
        # Cutoff2 should find at least as many neighbors as cutoff1
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
            # In half_fill mode, only i < j relationships are stored
            assert num_neighbors1[3].item() == 0, (
                f"Atom 3 should have 0 neighbor in cutoff1, got {num_neighbors1[3].item()}"
            )
            assert num_neighbors2[3].item() == 0, (
                f"Atom 3 should have 0 neighbors in cutoff2, got {num_neighbors2[3].item()}"
            )
        else:
            # In full mode, symmetric relationships are stored
            assert num_neighbors1[3].item() == 1, (
                f"Atom 3 should have 1 neighbors in cutoff1, got {num_neighbors1[3].item()}"
            )
            assert num_neighbors2[3].item() == 2, (
                f"Atom 3 should have 2 neighbors in cutoff2, got {num_neighbors2[3].item()}"
            )

        # Check neighbor indices are valid
        for i in range(positions.shape[0]):
            for j in range(num_neighbors1[i].item()):
                neighbor_idx = neighbor_matrix1[i, j].item()
                assert 0 <= neighbor_idx < positions.shape[0], (
                    f"Invalid neighbor index: {neighbor_idx}"
                )
                assert neighbor_idx != i, "Atom should not be its own neighbor"
            for j in range(num_neighbors2[i].item()):
                neighbor_idx = neighbor_matrix2[i, j].item()
                assert 0 <= neighbor_idx < positions.shape[0], (
                    f"Invalid neighbor index: {neighbor_idx}"
                )
                assert neighbor_idx != i, "Atom should not be its own neighbor"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_naive_dual_cutoff_pbc_kernel(self, device, dtype):
        """Test _fill_naive_neighbor_matrix_pbc_dual_cutoff kernel."""
        # Simple system with PBC
        positions = torch.tensor(
            [[0.1, 0.1, 0.1], [1.9, 0.1, 0.1]],  # Should be neighbors via PBC
            dtype=dtype,
            device=device,
        )
        cell = torch.eye(3, dtype=dtype, device=device) * 2.0
        cutoff1 = 0.3  # Smaller cutoff
        cutoff2 = 0.5  # Larger cutoff
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


class TestNaiveDualCutoffUtilityFunctions:
    """Test utility functions used by naive dual cutoff neighbor list."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_naive_neighbor_matrix_no_pbc_no_alloc_dual_cutoff_function(
        self, device, dtype, half_fill
    ):
        """Test _naive_neighbor_matrix_no_pbc_no_alloc_dual_cutoff custom op."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        # Prepare output arrays
        neighbor_matrix1 = torch.full(
            (positions.shape[0], max_neighbors1),
            positions.shape[0],
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix2 = torch.full(
            (positions.shape[0], max_neighbors2),
            positions.shape[0],
            dtype=torch.int32,
            device=device,
        )
        num_neighbors1 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )
        num_neighbors2 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        _naive_neighbor_matrix_no_pbc_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
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
        for i in range(positions.shape[0]):
            for j in range(num_neighbors1[i].item()):
                neighbor_idx = neighbor_matrix1[i, j].item()
                assert 0 <= neighbor_idx < positions.shape[0], (
                    f"Invalid neighbor index: {neighbor_idx}"
                )
                assert neighbor_idx != i, "Atom should not be its own neighbor"
            for j in range(num_neighbors2[i].item()):
                neighbor_idx = neighbor_matrix2[i, j].item()
                assert 0 <= neighbor_idx < positions.shape[0], (
                    f"Invalid neighbor index: {neighbor_idx}"
                )
                assert neighbor_idx != i, "Atom should not be its own neighbor"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_naive_neighbor_matrix_pbc_dual_cutoff_function(
        self, device, dtype, half_fill
    ):
        """Test _naive_neighbor_matrix_pbc_dual_cutoff function."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff1 = 1.0
        cutoff2 = 2.5
        max_neighbors1 = 20
        max_neighbors2 = 35

        shift_range_per_dimension, shift_offset, total_shifts = (
            compute_naive_num_shifts(cell.reshape(1, 3, 3), cutoff2, pbc.reshape(1, 3))
        )

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

        _naive_neighbor_matrix_pbc_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            cell.reshape(1, 3, 3),
            neighbor_matrix1,
            neighbor_matrix2,
            neighbor_matrix_shifts1,
            neighbor_matrix_shifts2,
            num_neighbors1,
            num_neighbors2,
            shift_range_per_dimension,
            shift_offset,
            total_shifts,
            half_fill=half_fill,
        )

        # Check output shapes and types
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
        assert torch.all(num_neighbors2 >= num_neighbors1), (
            "Larger cutoff should find at least as many neighbors"
        )

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


class TestNaiveDualCutoffMainAPI:
    """Test the main naive dual cutoff neighbor list API function."""

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [False, True])
    @pytest.mark.parametrize("pbc_flag", [True, False])
    @pytest.mark.parametrize("preallocate", [False, True])
    @pytest.mark.parametrize("return_neighbor_list", [False, True])
    @pytest.mark.parametrize("fill_value", [-1, 8])
    def test_naive_neighbor_list_dual_cutoff_no_pbc(
        self,
        device,
        dtype,
        half_fill,
        pbc_flag,
        preallocate,
        return_neighbor_list,
        fill_value,
    ):
        """Test naive_neighbor_list_dual_cutoff without periodic boundary conditions."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        if not pbc_flag:
            cell = None
            pbc = None
        if preallocate:
            neighbor_matrix1 = torch.full(
                (positions.shape[0], max_neighbors1),
                fill_value,
                dtype=torch.int32,
                device=device,
            )
            num_neighbors1 = torch.zeros(
                positions.shape[0], dtype=torch.int32, device=device
            )
            neighbor_matrix2 = torch.full(
                (positions.shape[0], max_neighbors2),
                fill_value,
                dtype=torch.int32,
                device=device,
            )
            num_neighbors2 = torch.zeros(
                positions.shape[0], dtype=torch.int32, device=device
            )
            args = (positions, cutoff1, cutoff2)
            kwargs = {
                "fill_value": fill_value,
                "half_fill": half_fill,
                "neighbor_matrix1": neighbor_matrix1,
                "num_neighbors1": num_neighbors1,
                "neighbor_matrix2": neighbor_matrix2,
                "num_neighbors2": num_neighbors2,
                "return_neighbor_list": return_neighbor_list,
            }
            if pbc_flag:
                shift_range_per_dimension, shift_offset, total_shifts = (
                    compute_naive_num_shifts(cell, cutoff2, pbc)
                )
                kwargs["cell"] = cell
                kwargs["pbc"] = pbc
                kwargs["shift_range_per_dimension"] = shift_range_per_dimension
                kwargs["shift_offset"] = shift_offset
                kwargs["total_shifts"] = total_shifts
                neighbor_matrix_shifts1 = torch.zeros(
                    (positions.shape[0], max_neighbors1, 3),
                    dtype=torch.int32,
                    device=device,
                )
                kwargs["neighbor_matrix_shifts1"] = neighbor_matrix_shifts1
                neighbor_matrix_shifts2 = torch.zeros(
                    (positions.shape[0], max_neighbors2, 3),
                    dtype=torch.int32,
                    device=device,
                )
                kwargs["neighbor_matrix_shifts2"] = neighbor_matrix_shifts2
            results = naive_neighbor_list_dual_cutoff(*args, **kwargs)
            if return_neighbor_list:
                if pbc_flag:
                    (
                        neighbor_list1,
                        neighbor_ptr1,
                        neighbor_shifts1,
                        neighbor_list2,
                        neighbor_ptr2,
                        neighbor_shifts2,
                    ) = results
                    num_neighbors1 = neighbor_ptr1[1:] - neighbor_ptr1[:-1]
                    num_neighbors2 = neighbor_ptr2[1:] - neighbor_ptr2[:-1]
                    idx_i1 = neighbor_list1[0]
                    idx_j1 = neighbor_list1[1]
                    idx_i2 = neighbor_list2[0]
                    idx_j2 = neighbor_list2[1]
                    u1 = neighbor_shifts1
                    u2 = neighbor_shifts2
                else:
                    (
                        neighbor_list1,
                        neighbor_ptr1,
                        neighbor_list2,
                        neighbor_ptr2,
                    ) = results
                    num_neighbors1 = neighbor_ptr1[1:] - neighbor_ptr1[:-1]
                    num_neighbors2 = neighbor_ptr2[1:] - neighbor_ptr2[:-1]
                    idx_i1 = neighbor_list1[0]
                    idx_j1 = neighbor_list1[1]
                    idx_i2 = neighbor_list2[0]
                    idx_j2 = neighbor_list2[1]
                    u1 = torch.zeros(
                        (idx_i1.shape[0], 3), dtype=torch.int32, device=device
                    )
                    u2 = torch.zeros(
                        (idx_j2.shape[0], 3), dtype=torch.int32, device=device
                    )
        else:
            args = (positions, cutoff1, cutoff2)
            kwargs = {
                "max_neighbors1": max_neighbors1,
                "max_neighbors2": max_neighbors2,
                "fill_value": fill_value,
                "half_fill": half_fill,
                "return_neighbor_list": return_neighbor_list,
            }
            if pbc_flag:
                kwargs["cell"] = cell
                kwargs["pbc"] = pbc
            results = naive_neighbor_list_dual_cutoff(*args, **kwargs)
            if pbc_flag:
                if return_neighbor_list:
                    (
                        neighbor_list1,
                        neighbor_ptr1,
                        neighbor_shifts1,
                        neighbor_list2,
                        neighbor_ptr2,
                        neighbor_shifts2,
                    ) = results
                    num_neighbors1 = neighbor_ptr1[1:] - neighbor_ptr1[:-1]
                    num_neighbors2 = neighbor_ptr2[1:] - neighbor_ptr2[:-1]
                    idx_i1 = neighbor_list1[0]
                    idx_j1 = neighbor_list1[1]
                    idx_i2 = neighbor_list2[0]
                    idx_j2 = neighbor_list2[1]
                    u1 = neighbor_shifts1
                    u2 = neighbor_shifts2
                else:
                    (
                        neighbor_matrix1,
                        num_neighbors1,
                        neighbor_matrix_shifts1,
                        neighbor_matrix2,
                        num_neighbors2,
                        neighbor_matrix_shifts2,
                    ) = results
            else:
                if return_neighbor_list:
                    (
                        neighbor_list1,
                        neighbor_ptr1,
                        neighbor_list2,
                        neighbor_ptr2,
                    ) = results
                    num_neighbors1 = neighbor_ptr1[1:] - neighbor_ptr1[:-1]
                    num_neighbors2 = neighbor_ptr2[1:] - neighbor_ptr2[:-1]
                    idx_i1 = neighbor_list1[0]
                    idx_j1 = neighbor_list1[1]
                    idx_i2 = neighbor_list2[0]
                    idx_j2 = neighbor_list2[1]
                    u1 = torch.zeros(
                        (idx_i1.shape[0], 3), dtype=torch.int32, device=device
                    )
                    u2 = torch.zeros(
                        (idx_j2.shape[0], 3), dtype=torch.int32, device=device
                    )
                else:
                    (
                        neighbor_matrix1,
                        num_neighbors1,
                        neighbor_matrix2,
                        num_neighbors2,
                    ) = results

        # Check output shapes and types
        assert num_neighbors1.dtype == torch.int32
        assert num_neighbors2.dtype == torch.int32
        assert num_neighbors1.shape == (positions.shape[0],)
        assert num_neighbors2.shape == (positions.shape[0],)
        assert num_neighbors1.device == torch.device(device)
        assert num_neighbors2.device == torch.device(device)
        if return_neighbor_list:
            assert neighbor_list1.dtype == torch.int32
            assert neighbor_list2.dtype == torch.int32
            assert neighbor_list1.shape == (2, num_neighbors1.sum())
            assert neighbor_list2.shape == (2, num_neighbors2.sum())
            assert neighbor_list1.device == torch.device(device)
            assert neighbor_list2.device == torch.device(device)

            if pbc_flag:
                assert u1.dtype == torch.int32
                assert u1.shape == (num_neighbors1.sum(), 3)
                assert u2.dtype == torch.int32
                assert u2.shape == (num_neighbors2.sum(), 3)
                assert u1.device == torch.device(device)
                assert u2.device == torch.device(device)

        else:
            assert neighbor_matrix1.dtype == torch.int32
            assert neighbor_matrix2.dtype == torch.int32
            assert neighbor_matrix1.shape == (
                positions.shape[0],
                max_neighbors1,
            )
            assert neighbor_matrix2.shape == (
                positions.shape[0],
                max_neighbors2,
            )
            assert neighbor_matrix1.device == torch.device(device)
            assert neighbor_matrix2.device == torch.device(device)

            if pbc_flag:
                assert neighbor_matrix_shifts1.dtype == torch.int32
                assert neighbor_matrix_shifts2.dtype == torch.int32
                assert neighbor_matrix_shifts1.shape == (
                    positions.shape[0],
                    max_neighbors1,
                    3,
                )
                assert neighbor_matrix_shifts2.shape == (
                    positions.shape[0],
                    max_neighbors2,
                    3,
                )
                assert neighbor_matrix_shifts1.device == torch.device(device)
                assert neighbor_matrix_shifts2.device == torch.device(device)

        # Get reference result
        i_ref1, j_ref1, u_ref1, _ = brute_force_neighbors(positions, cell, pbc, cutoff1)
        i_ref2, j_ref2, u_ref2, _ = brute_force_neighbors(positions, cell, pbc, cutoff2)

        if return_neighbor_list and not half_fill:
            assert_neighbor_lists_equal(
                (idx_i1, idx_j1, u1),
                (i_ref1, j_ref1, u_ref1),
            )
            assert_neighbor_lists_equal(
                (idx_i2, idx_j2, u2),
                (i_ref2, j_ref2, u_ref2),
            )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_naive_neighbor_list_dual_cutoff_edge_cases(
        self,
        device,
        dtype,
    ):
        """Test edge cases for naive_neighbor_list_dual_cutoff."""
        # Empty system
        positions_empty = torch.empty(0, 3, dtype=dtype, device=device)
        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions=positions_empty,
                cutoff1=1.0,
                cutoff2=1.5,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
            )
        )
        assert neighbor_matrix1.shape == (0, 10)
        assert neighbor_matrix2.shape == (0, 15)
        assert num_neighbors1.shape == (0,)
        assert num_neighbors2.shape == (0,)

        # Single atom
        positions_single = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions=positions_single,
                cutoff1=1.0,
                cutoff2=1.5,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
            )
        )
        assert num_neighbors1[0].item() == 0, "Single atom should have no neighbors"
        assert num_neighbors2[0].item() == 0, "Single atom should have no neighbors"

        # Zero cutoffs
        positions, _, _ = create_simple_cubic_system(
            num_atoms=4, dtype=dtype, device=device
        )
        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions=positions,
                cutoff1=0.0,
                cutoff2=0.0,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
            )
        )
        assert torch.all(num_neighbors1 == 0), "Zero cutoffs should find no neighbors"
        assert torch.all(num_neighbors2 == 0), "Zero cutoffs should find no neighbors"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_naive_neighbor_list_dual_cutoff_error_conditions(
        self, device, dtype, half_fill
    ):
        """Test error conditions for naive_neighbor_list_dual_cutoff."""
        positions, cell, pbc = create_simple_cubic_system(dtype=dtype, device=device)

        # Test mismatched cell and pbc arguments
        with pytest.raises(
            ValueError, match="If cell is provided, pbc must also be provided"
        ):
            naive_neighbor_list_dual_cutoff(
                positions, 1.0, 1.5, pbc=None, cell=cell, max_neighbors1=10
            )

        with pytest.raises(
            ValueError, match="If pbc is provided, cell must also be provided"
        ):
            naive_neighbor_list_dual_cutoff(
                positions, 1.0, 1.5, pbc=pbc, cell=None, max_neighbors1=10
            )

    def test_max_neighbors2_default(self):
        """Test that max_neighbors2 defaults to max_neighbors1 when not provided."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
        )

        # Call without max_neighbors2
        result = naive_neighbor_list_dual_cutoff(
            positions=positions,
            cutoff1=0.5,
            cutoff2=1.5,
            max_neighbors1=10,
            # max_neighbors2 not provided
        )

        # Should not raise an error and should work correctly
        assert (
            len(result) == 4
        )  # neighbor_matrix1, neighbor_matrix2, num_neighbors1, num_neighbors2


class TestNaiveDualCutoffPerformanceAndScaling:
    """Test performance characteristics and scaling of dual cutoff implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_dual_cutoff_scaling_with_system_size(self, device):
        """Test that dual cutoff implementation scales as expected with system size."""
        import time

        dtype = torch.float32
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 50
        max_neighbors2 = 75

        # Test different system sizes
        sizes = [10, 100] if device == "cpu" else [100, 1000]
        times = []

        for num_atoms in sizes:
            positions, cell, pbc = create_simple_cubic_system(
                num_atoms=num_atoms, dtype=dtype, device=device
            )

            # Warm up
            for _ in range(10):
                naive_neighbor_list_dual_cutoff(
                    positions,
                    cutoff1,
                    cutoff2,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc,
                    cell=cell,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            # Time the operation
            start_time = time.time()
            for _ in range(100):
                naive_neighbor_list_dual_cutoff(
                    positions,
                    cutoff1,
                    cutoff2,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc,
                    cell=cell,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            times.append(elapsed)

        # Check that it doesn't grow too fast (should be roughly O(N^2))
        # This is a loose check since we can't expect perfect scaling
        assert times[1] > times[0] * 0.7, "Time should increase with system size"
        if len(times) > 2:
            # Very loose scaling check
            scaling_factor = times[-1] / times[0]
            size_factor = (sizes[-1] / sizes[0]) ** 2
            assert scaling_factor < size_factor * 5, (
                "Scaling should not be much worse than O(N^2)"
            )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_dual_cutoff_cutoff_scaling(self, device):
        """Test scaling with different cutoff values."""
        dtype = torch.float32
        num_atoms = 50
        max_neighbors1 = 100
        max_neighbors2 = 150

        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=num_atoms, dtype=dtype, device=device
        )

        # Test different cutoff pairs
        cutoff_pairs = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5)]
        neighbor_counts1 = []
        neighbor_counts2 = []

        for cutoff1, cutoff2 in cutoff_pairs:
            _, num_neighbors1, _, _, num_neighbors2, _ = (
                naive_neighbor_list_dual_cutoff(
                    positions,
                    cutoff1,
                    cutoff2,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc,
                    cell=cell,
                )
            )
            total_pairs1 = num_neighbors1.sum().item()
            total_pairs2 = num_neighbors2.sum().item()
            neighbor_counts1.append(total_pairs1)
            neighbor_counts2.append(total_pairs2)

        # Check that neighbor count increases with cutoff
        for i in range(1, len(neighbor_counts1)):
            assert neighbor_counts1[i] >= neighbor_counts1[i - 1], (
                f"Neighbor count should increase with cutoff: {neighbor_counts1}"
            )
            assert neighbor_counts2[i] >= neighbor_counts2[i - 1], (
                f"Neighbor count should increase with cutoff: {neighbor_counts2}"
            )

        # Check that cutoff2 always finds at least as many neighbors as cutoff1
        for i in range(len(neighbor_counts1)):
            assert neighbor_counts2[i] >= neighbor_counts1[i], (
                f"Larger cutoff should find at least as many neighbors: {neighbor_counts2[i]} >= {neighbor_counts1[i]}"
            )


class TestNaiveDualCutoffRobustness:
    """Test robustness of dual cutoff implementation to various inputs."""

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
                cutoff1 = 1.0
                cutoff2 = 1.5
                max_neighbors1 = 30
                max_neighbors2 = 50

                # Should not crash
                if pbc_flag:
                    (
                        neighbor_matrix1,
                        num_neighbors1,
                        neighbor_matrix_shifts1,
                        neighbor_matrix2,
                        num_neighbors2,
                        neighbor_matrix_shifts2,
                    ) = naive_neighbor_list_dual_cutoff(
                        positions=positions,
                        cutoff1=cutoff1,
                        cutoff2=cutoff2,
                        max_neighbors1=max_neighbors1,
                        max_neighbors2=max_neighbors2,
                        pbc=pbc,
                        cell=cell,
                        half_fill=half_fill,
                    )
                    assert neighbor_matrix_shifts1.device == torch.device(device)
                    assert neighbor_matrix_shifts2.device == torch.device(device)
                else:
                    (
                        neighbor_matrix1,
                        num_neighbors1,
                        neighbor_matrix2,
                        num_neighbors2,
                    ) = naive_neighbor_list_dual_cutoff(
                        positions=positions,
                        cutoff1=cutoff1,
                        cutoff2=cutoff2,
                        max_neighbors1=max_neighbors1,
                        max_neighbors2=max_neighbors2,
                        pbc=None,
                        cell=None,
                        half_fill=half_fill,
                    )

                # Basic sanity checks
                assert torch.all(num_neighbors1 >= 0)
                assert torch.all(num_neighbors2 >= 0)
                assert torch.all(num_neighbors1 <= max_neighbors1)
                assert torch.all(num_neighbors2 <= max_neighbors2)
                assert torch.all(num_neighbors2 >= num_neighbors1), (
                    "Larger cutoff should find at least as many neighbors"
                )
                assert neighbor_matrix1.device == torch.device(device)
                assert neighbor_matrix2.device == torch.device(device)
                assert num_neighbors1.device == torch.device(device)
                assert num_neighbors2.device == torch.device(device)

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
        cutoff1 = 0.15
        cutoff2 = 0.25
        max_neighbors1 = 15
        max_neighbors2 = 25

        # Should handle extreme aspect ratios
        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = naive_neighbor_list_dual_cutoff(
            positions=positions * torch.tensor([10.0, 0.1, 0.1], device=device),
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc,
            cell=cell,
            half_fill=half_fill,
        )

        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors2 >= num_neighbors1), (
            "Larger cutoff should find at least as many neighbors"
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_large_cutoffs(self, device, dtype, half_fill):
        """Test with very large cutoffs."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )

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
        ) = naive_neighbor_list_dual_cutoff(
            positions=positions,
            cutoff1=large_cutoff1,
            cutoff2=large_cutoff2,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc,
            cell=cell,
        )

        # Should find many neighbors
        assert num_neighbors1.sum() > 0
        assert num_neighbors2.sum() > 0
        assert torch.all(num_neighbors2 >= num_neighbors1), (
            "Larger cutoff should find at least as many neighbors"
        )
        # Each atom should have multiple neighbors (including periodic images)
        assert torch.all(num_neighbors1 > 0)
        assert torch.all(num_neighbors2 > 0)

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

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 30
        max_neighbors2 = 50

        # Get results for both precisions
        _, num_neighbors1_f32, _, _, num_neighbors2_f32, _ = (
            naive_neighbor_list_dual_cutoff(
                positions_f32,
                cutoff1,
                cutoff2,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                pbc=pbc,
                cell=cell_f32,
                half_fill=half_fill,
            )
        )
        _, num_neighbors1_f64, _, _, num_neighbors2_f64, _ = (
            naive_neighbor_list_dual_cutoff(
                positions_f64,
                cutoff1,
                cutoff2,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                pbc=pbc,
                cell=cell_f64,
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


class TestNaiveDualCutoffMemoryAndPerformance:
    """Test memory usage and performance characteristics of dual cutoff implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_memory_scaling(self, device, half_fill):
        """Test that memory usage scales reasonably with system size."""
        import gc

        dtype = torch.float32
        cutoff1 = 1.0
        cutoff2 = 1.5

        # Test different system sizes
        sizes = [10, 100] if device == "cpu" else [100, 200]

        for num_atoms in sizes:
            positions, cell, pbc = create_simple_cubic_system(
                num_atoms=num_atoms, dtype=dtype, device=device
            )
            cell = cell.reshape(1, 3, 3)
            pbc = pbc.reshape(1, 3)

            # Estimate reasonable max_neighbors based on system size and cutoff
            max_neighbors1 = 5 * num_atoms
            max_neighbors2 = 10 * num_atoms

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
            ) = naive_neighbor_list_dual_cutoff(
                positions=positions,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                pbc=pbc,
                cell=cell,
                half_fill=half_fill,
            )

            # Basic checks that output is reasonable
            assert neighbor_matrix1.shape == (num_atoms, max_neighbors1)
            assert neighbor_matrix2.shape == (num_atoms, max_neighbors2)
            assert neighbor_matrix_shifts1.shape == (
                num_atoms,
                max_neighbors1,
                3,
            )
            assert neighbor_matrix_shifts2.shape == (
                num_atoms,
                max_neighbors2,
                3,
            )
            assert num_neighbors1.shape == (num_atoms,)
            assert num_neighbors2.shape == (num_atoms,)
            assert torch.all(num_neighbors1 >= 0)
            assert torch.all(num_neighbors2 >= 0)
            assert torch.all(num_neighbors1 <= max_neighbors1)
            assert torch.all(num_neighbors2 <= max_neighbors2)
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
                positions,
                cell,
                pbc,
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
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cell = cell.reshape(1, 3, 3)
        pbc = pbc.reshape(1, 3)

        cutoff1 = 1.5  # Large cutoffs to find many neighbors
        cutoff2 = 2.0
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
        ) = naive_neighbor_list_dual_cutoff(
            positions=positions,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc,
            cell=cell,
            half_fill=half_fill,
        )

        # Should still produce valid output, just potentially incomplete
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert neighbor_matrix1.shape == (positions.shape[0], max_neighbors1)
        assert neighbor_matrix2.shape == (positions.shape[0], max_neighbors2)
        assert neighbor_matrix_shifts1.shape == (
            positions.shape[0],
            max_neighbors1,
            3,
        )
        assert neighbor_matrix_shifts2.shape == (
            positions.shape[0],
            max_neighbors2,
            3,
        )
        assert num_neighbors1.shape == (positions.shape[0],)
        assert num_neighbors2.shape == (positions.shape[0],)
        assert neighbor_matrix1.device == torch.device(device)
        assert neighbor_matrix2.device == torch.device(device)
        assert neighbor_matrix_shifts1.device == torch.device(device)
        assert neighbor_matrix_shifts2.device == torch.device(device)
        assert num_neighbors1.device == torch.device(device)
        assert num_neighbors2.device == torch.device(device)
