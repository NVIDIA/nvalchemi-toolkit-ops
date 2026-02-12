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

"""Tests for PyTorch bindings of naive dual cutoff neighbor list methods."""

import pytest
import torch

from nvalchemiops.torch.neighbors.naive_dual_cutoff import (
    naive_neighbor_list_dual_cutoff,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
)

from ...test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
    create_simple_cubic_system,
)
from .conftest import requires_vesin


class TestNaiveDualCutoffCorrectness:
    """Test correctness of naive dual cutoff neighbor list against reference."""

    @requires_vesin
    @pytest.mark.parametrize("fill_value", [-1, 8])
    def test_matrix_format_no_pbc(
        self, device, dtype, half_fill, preallocate, fill_value
    ):
        """Test dual cutoff neighbor list in matrix format without PBC."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

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
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1,
                cutoff2,
                fill_value=fill_value,
                half_fill=half_fill,
                neighbor_matrix1=neighbor_matrix1,
                num_neighbors1=num_neighbors1,
                neighbor_matrix2=neighbor_matrix2,
                num_neighbors2=num_neighbors2,
            )
        else:
            neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
                naive_neighbor_list_dual_cutoff(
                    positions,
                    cutoff1,
                    cutoff2,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    fill_value=fill_value,
                    half_fill=half_fill,
                )
            )

        # Verify output shapes and types
        assert neighbor_matrix1.shape == (8, max_neighbors1)
        assert neighbor_matrix2.shape == (8, max_neighbors2)
        assert num_neighbors1.shape == (8,)
        assert num_neighbors2.shape == (8,)
        assert neighbor_matrix1.dtype == torch.int32
        assert neighbor_matrix2.dtype == torch.int32
        assert num_neighbors1.dtype == torch.int32
        assert num_neighbors2.dtype == torch.int32

        # Verify neighbor counts are reasonable
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors1 <= max_neighbors1)
        assert torch.all(num_neighbors2 <= max_neighbors2)
        assert torch.all(num_neighbors2 >= num_neighbors1)

    @requires_vesin
    @pytest.mark.parametrize("fill_value", [-1, 8])
    def test_matrix_format_with_pbc(
        self, device, dtype, half_fill, preallocate, fill_value
    ):
        """Test dual cutoff neighbor list in matrix format with PBC."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        if preallocate:
            shift_range_per_dimension, shift_offset, total_shifts = (
                compute_naive_num_shifts(cell, cutoff2, pbc)
            )
            neighbor_matrix1 = torch.full(
                (positions.shape[0], max_neighbors1),
                fill_value,
                dtype=torch.int32,
                device=device,
            )
            num_neighbors1 = torch.zeros(
                positions.shape[0], dtype=torch.int32, device=device
            )
            neighbor_matrix_shifts1 = torch.zeros(
                (positions.shape[0], max_neighbors1, 3),
                dtype=torch.int32,
                device=device,
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
            neighbor_matrix_shifts2 = torch.zeros(
                (positions.shape[0], max_neighbors2, 3),
                dtype=torch.int32,
                device=device,
            )
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1,
                cutoff2,
                cell=cell,
                pbc=pbc,
                fill_value=fill_value,
                half_fill=half_fill,
                neighbor_matrix1=neighbor_matrix1,
                num_neighbors1=num_neighbors1,
                neighbor_matrix_shifts1=neighbor_matrix_shifts1,
                neighbor_matrix2=neighbor_matrix2,
                num_neighbors2=num_neighbors2,
                neighbor_matrix_shifts2=neighbor_matrix_shifts2,
                shift_range_per_dimension=shift_range_per_dimension,
                shift_offset=shift_offset,
                total_shifts=total_shifts,
            )
        else:
            (
                neighbor_matrix1,
                num_neighbors1,
                neighbor_matrix_shifts1,
                neighbor_matrix2,
                num_neighbors2,
                neighbor_matrix_shifts2,
            ) = naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1,
                cutoff2,
                cell=cell,
                pbc=pbc,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                fill_value=fill_value,
                half_fill=half_fill,
            )

        # Verify output shapes and types
        assert neighbor_matrix1.shape == (8, max_neighbors1)
        assert neighbor_matrix2.shape == (8, max_neighbors2)
        assert neighbor_matrix_shifts1.shape == (8, max_neighbors1, 3)
        assert neighbor_matrix_shifts2.shape == (8, max_neighbors2, 3)
        assert num_neighbors1.shape == (8,)
        assert num_neighbors2.shape == (8,)

        # Verify neighbor counts are reasonable
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors2 >= num_neighbors1)

    @requires_vesin
    def test_list_format_no_pbc_correctness(self, device, dtype, half_fill):
        """Test dual cutoff neighbor list in COO format without PBC against reference."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff1 = 1.0
        cutoff2 = 1.5

        neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1,
                cutoff2,
                max_neighbors1=15,
                max_neighbors2=25,
                half_fill=half_fill,
                return_neighbor_list=True,
            )
        )

        # Verify output format
        assert neighbor_list1.shape[0] == 2
        assert neighbor_list2.shape[0] == 2
        assert neighbor_ptr1.shape == (9,)
        assert neighbor_ptr2.shape == (9,)

        # Compare against reference (only for full fill mode)
        if not half_fill:
            idx_i1, idx_j1 = neighbor_list1[0], neighbor_list1[1]
            idx_i2, idx_j2 = neighbor_list2[0], neighbor_list2[1]
            u1 = torch.zeros((idx_i1.shape[0], 3), dtype=torch.int32, device=device)
            u2 = torch.zeros((idx_i2.shape[0], 3), dtype=torch.int32, device=device)

            i_ref1, j_ref1, u_ref1, _ = brute_force_neighbors(
                positions, None, None, cutoff1
            )
            i_ref2, j_ref2, u_ref2, _ = brute_force_neighbors(
                positions, None, None, cutoff2
            )

            assert_neighbor_lists_equal((idx_i1, idx_j1, u1), (i_ref1, j_ref1, u_ref1))
            assert_neighbor_lists_equal((idx_i2, idx_j2, u2), (i_ref2, j_ref2, u_ref2))

    @requires_vesin
    def test_list_format_with_pbc_correctness(self, device, dtype, half_fill):
        """Test dual cutoff neighbor list in COO format with PBC against reference."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff1 = 1.0
        cutoff2 = 1.5

        (
            neighbor_list1,
            neighbor_ptr1,
            neighbor_shifts1,
            neighbor_list2,
            neighbor_ptr2,
            neighbor_shifts2,
        ) = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            cell=cell,
            pbc=pbc,
            max_neighbors1=15,
            max_neighbors2=25,
            half_fill=half_fill,
            return_neighbor_list=True,
        )

        # Verify output format
        assert neighbor_list1.shape[0] == 2
        assert neighbor_list2.shape[0] == 2
        assert neighbor_ptr1.shape == (9,)
        assert neighbor_ptr2.shape == (9,)
        assert neighbor_shifts1.shape[0] == neighbor_list1.shape[1]
        assert neighbor_shifts2.shape[0] == neighbor_list2.shape[1]

        # Compare against reference (only for full fill mode)
        if not half_fill:
            idx_i1, idx_j1 = neighbor_list1[0], neighbor_list1[1]
            idx_i2, idx_j2 = neighbor_list2[0], neighbor_list2[1]

            i_ref1, j_ref1, u_ref1, _ = brute_force_neighbors(
                positions, cell, pbc, cutoff1
            )
            i_ref2, j_ref2, u_ref2, _ = brute_force_neighbors(
                positions, cell, pbc, cutoff2
            )

            assert_neighbor_lists_equal(
                (idx_i1, idx_j1, neighbor_shifts1),
                (i_ref1, j_ref1, u_ref1),
            )
            assert_neighbor_lists_equal(
                (idx_i2, idx_j2, neighbor_shifts2),
                (i_ref2, j_ref2, u_ref2),
            )


class TestNaiveDualCutoffEdgeCases:
    """Test edge cases for naive dual cutoff neighbor list."""

    def test_empty_system(self, device, dtype):
        """Test dual cutoff neighbor list with empty system."""
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

    def test_single_atom(self, device, dtype):
        """Test dual cutoff neighbor list with single atom."""
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
        assert num_neighbors1[0].item() == 0
        assert num_neighbors2[0].item() == 0

    def test_zero_cutoffs(self, device, dtype):
        """Test dual cutoff neighbor list with zero cutoffs."""
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
        assert torch.all(num_neighbors1 == 0)
        assert torch.all(num_neighbors2 == 0)

    def test_identical_cutoffs(self, device, dtype):
        """Test dual cutoff neighbor list with identical cutoff values."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.5
        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions=positions,
                cutoff1=cutoff,
                cutoff2=cutoff,
                max_neighbors1=15,
                max_neighbors2=15,
                pbc=None,
                cell=None,
            )
        )
        # When cutoffs are identical, neighbor counts should match
        torch.testing.assert_close(num_neighbors1, num_neighbors2, rtol=0, atol=0)


class TestNaiveDualCutoffErrors:
    """Test error conditions for naive dual cutoff neighbor list."""

    def test_cell_without_pbc_error(self, device, dtype):
        """Test that providing cell without pbc raises error."""
        positions, cell, _ = create_simple_cubic_system(dtype=dtype, device=device)

        with pytest.raises(
            ValueError, match="If cell is provided, pbc must also be provided"
        ):
            naive_neighbor_list_dual_cutoff(
                positions, 1.0, 1.5, pbc=None, cell=cell, max_neighbors1=10
            )

    def test_pbc_without_cell_error(self, device, dtype):
        """Test that providing pbc without cell raises error."""
        positions, _, pbc = create_simple_cubic_system(dtype=dtype, device=device)

        with pytest.raises(
            ValueError, match="If pbc is provided, cell must also be provided"
        ):
            naive_neighbor_list_dual_cutoff(
                positions, 1.0, 1.5, pbc=pbc, cell=None, max_neighbors1=10
            )

    def test_negative_cutoff_error(self, device, dtype):
        """Test that negative cutoffs are handled appropriately."""
        positions, _, _ = create_simple_cubic_system(dtype=dtype, device=device)

        # Should either raise error or produce empty neighbor lists
        try:
            neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
                naive_neighbor_list_dual_cutoff(
                    positions,
                    -1.0,
                    -0.5,
                    max_neighbors1=10,
                    max_neighbors2=15,
                )
            )
            # If it doesn't raise, verify no neighbors found
            assert torch.all(num_neighbors1 == 0)
            assert torch.all(num_neighbors2 == 0)
        except (ValueError, RuntimeError):
            # Acceptable to raise error for negative cutoffs
            pass

    def test_cutoff1_greater_than_cutoff2_behavior(self, device, dtype):
        """Test behavior when cutoff1 > cutoff2."""
        positions, _, _ = create_simple_cubic_system(dtype=dtype, device=device)

        # This should work but cutoff2 should find fewer neighbors than cutoff1
        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1=2.0,
                cutoff2=1.0,
                max_neighbors1=25,
                max_neighbors2=15,
            )
        )

        # Verify cutoff2 finds fewer or equal neighbors
        assert torch.all(num_neighbors2 <= num_neighbors1)


class TestNaiveDualCutoffOutputFormats:
    """Test different output formats for naive dual cutoff neighbor list."""

    def test_matrix_format_shapes(self, device, dtype):
        """Test that matrix format returns correct shapes."""
        positions, _, _ = create_simple_cubic_system(dtype=dtype, device=device)
        max_neighbors1 = 12
        max_neighbors2 = 20

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                1.0,
                1.5,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
            )
        )

        assert neighbor_matrix1.shape == (8, max_neighbors1)
        assert neighbor_matrix2.shape == (8, max_neighbors2)
        assert num_neighbors1.shape == (8,)
        assert num_neighbors2.shape == (8,)

    def test_list_format_shapes(self, device, dtype):
        """Test that list format returns correct shapes."""
        positions, _, _ = create_simple_cubic_system(dtype=dtype, device=device)

        neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                1.0,
                1.5,
                max_neighbors1=15,
                max_neighbors2=25,
                return_neighbor_list=True,
            )
        )

        assert neighbor_list1.ndim == 2
        assert neighbor_list2.ndim == 2
        assert neighbor_list1.shape[0] == 2
        assert neighbor_list2.shape[0] == 2
        assert neighbor_ptr1.shape == (9,)
        assert neighbor_ptr2.shape == (9,)

    def test_max_neighbors2_defaults_to_max_neighbors1(self, device, dtype):
        """Test that max_neighbors2 defaults to max_neighbors1 when not provided."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype, device=device
        )

        result = naive_neighbor_list_dual_cutoff(
            positions=positions,
            cutoff1=0.5,
            cutoff2=1.5,
            max_neighbors1=10,
        )

        # Should return 4 tensors and not raise error
        assert len(result) == 4
        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = result
        # Both should have same max_neighbors dimension
        assert neighbor_matrix1.shape[1] == neighbor_matrix2.shape[1] == 10

    def test_larger_cutoff_finds_more_neighbors(self, device, dtype):
        """Test that larger cutoff finds at least as many neighbors."""
        positions, _, _ = create_simple_cubic_system(dtype=dtype, device=device)

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1=1.0,
                cutoff2=1.5,
                max_neighbors1=15,
                max_neighbors2=25,
            )
        )

        # cutoff2 should find at least as many neighbors as cutoff1
        assert torch.all(num_neighbors2 >= num_neighbors1)
