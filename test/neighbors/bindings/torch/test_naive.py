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

"""Tests for PyTorch bindings of naive neighbor list methods."""

from __future__ import annotations

import pytest
import torch

from nvalchemiops.torch.neighbors.naive import naive_neighbor_list
from nvalchemiops.torch.neighbors.neighbor_utils import compute_naive_num_shifts

from ...test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
    create_random_system,
    create_simple_cubic_system,
)
from .conftest import requires_vesin


class TestNaiveCorrectness:
    """Test correctness of naive neighbor list against reference implementation."""

    @requires_vesin
    def test_against_vesin_reference_no_pbc(self, device, dtype):
        """Verify correctness against vesin reference (no PBC)."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1

        # Get our result
        neighbor_list, neighbor_ptr = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=None,
            cell=None,
            max_neighbors=20,
            return_neighbor_list=True,
        )

        idx_i = neighbor_list[0]
        idx_j = neighbor_list[1]
        u = torch.zeros((idx_i.shape[0], 3), dtype=torch.int32, device=device)

        # Get reference result
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(
            positions, cell=None, pbc=None, cutoff=cutoff
        )

        # Compare
        assert_neighbor_lists_equal((idx_i, idx_j, u), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_against_vesin_reference_with_pbc(self, device, dtype):
        """Verify correctness against vesin reference (with PBC)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1

        # Get our result with PBC
        neighbor_list, neighbor_ptr, neighbor_shifts = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=20,
            return_neighbor_list=True,
        )

        idx_i = neighbor_list[0]
        idx_j = neighbor_list[1]
        u = neighbor_shifts

        # Get reference result
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(
            positions, cell=cell, pbc=pbc, cutoff=cutoff
        )

        # Compare
        assert_neighbor_lists_equal((idx_i, idx_j, u), (i_ref, j_ref, u_ref))

    def test_random_systems_basic_correctness(self, device, dtype):
        """Test basic correctness properties on random systems."""
        for pbc_flag in [True, False]:
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

                # Get neighbor matrix format
                if pbc_flag:
                    neighbor_matrix, num_neighbors, unit_shifts = naive_neighbor_list(
                        positions=positions,
                        cutoff=cutoff,
                        pbc=pbc,
                        cell=cell,
                        max_neighbors=max_neighbors,
                    )
                else:
                    neighbor_matrix, num_neighbors = naive_neighbor_list(
                        positions=positions,
                        cutoff=cutoff,
                        pbc=None,
                        cell=None,
                        max_neighbors=max_neighbors,
                    )

                # Verify basic correctness properties
                assert torch.all(num_neighbors >= 0)
                assert torch.all(num_neighbors <= max_neighbors)
                assert neighbor_matrix.device == torch.device(device)
                assert num_neighbors.device == torch.device(device)

    def test_precision_consistency(self, device):
        """Test that float32 and float64 give consistent neighbor counts."""
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
        )
        _, num_neighbors_f64, _ = naive_neighbor_list(
            positions_f64,
            cutoff,
            pbc=pbc,
            cell=cell_f64,
            max_neighbors=max_neighbors,
        )

        # Neighbor counts should be identical
        torch.testing.assert_close(num_neighbors_f32, num_neighbors_f64, rtol=0, atol=0)


class TestNaiveEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_system(self, device, dtype, half_fill):
        """Test behavior with empty position array."""
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

    def test_single_atom(self, device, dtype, half_fill):
        """Test behavior with single atom (should have no neighbors)."""
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

    def test_zero_cutoff(self, device, dtype, half_fill):
        """Test that zero cutoff produces no neighbors."""
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

    def test_large_cutoff_with_pbc(self, device, dtype, half_fill):
        """Test with cutoff larger than cell size."""
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

        # Should find many neighbors (including periodic images)
        assert num_neighbors.sum() > 0
        assert torch.all(num_neighbors > 0)

    def test_extreme_elongated_cell(self, device, dtype, half_fill):
        """Test with extreme cell aspect ratios."""
        positions = torch.rand(10, 3, dtype=dtype, device=device)
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]],
            dtype=dtype,
            device=device,
        ).reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], device=device).reshape(1, 3)
        cutoff = 0.2
        max_neighbors = 20

        # Should handle extreme aspect ratios without crashing
        _, num_neighbors, _ = naive_neighbor_list(
            positions=positions * torch.tensor([10.0, 0.1, 0.1], device=device),
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        assert torch.all(num_neighbors >= 0)

    def test_max_neighbors_overflow(self, device, dtype, half_fill):
        """Test behavior when max_neighbors is too small."""
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

        # Should produce valid output
        assert torch.all(num_neighbors >= 0)
        assert neighbor_matrix.shape == (positions.shape[0], max_neighbors)
        assert unit_shifts.shape == (positions.shape[0], max_neighbors, 3)


class TestNaiveErrors:
    """Test error handling and input validation."""

    def test_mismatched_cell_without_pbc(self, device, dtype):
        """Test that providing cell without pbc raises error."""
        positions, cell, _ = create_simple_cubic_system(dtype=dtype, device=device)

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

    def test_mismatched_pbc_without_cell(self, device, dtype):
        """Test that providing pbc without cell raises error."""
        positions, _, pbc = create_simple_cubic_system(dtype=dtype, device=device)

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


class TestNaiveOutputFormats:
    """Test different output formats (matrix vs list)."""

    def test_matrix_format_output_shapes_no_pbc(self, device, dtype, half_fill):
        """Test neighbor matrix format output shapes (no PBC)."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20

        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=None,
            cell=None,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
            return_neighbor_list=False,
        )

        # Verify shapes and types
        assert neighbor_matrix.dtype == torch.int32
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix.shape == (positions.shape[0], max_neighbors)
        assert num_neighbors.shape == (positions.shape[0],)
        assert neighbor_matrix.device == torch.device(device)
        assert num_neighbors.device == torch.device(device)

    def test_matrix_format_output_shapes_with_pbc(self, device, dtype, half_fill):
        """Test neighbor matrix format output shapes (with PBC)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20

        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
            return_neighbor_list=False,
        )

        # Verify shapes and types
        assert neighbor_matrix.dtype == torch.int32
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix_shifts.dtype == torch.int32
        assert neighbor_matrix.shape == (positions.shape[0], max_neighbors)
        assert num_neighbors.shape == (positions.shape[0],)
        assert neighbor_matrix_shifts.shape == (positions.shape[0], max_neighbors, 3)

    def test_list_format_output_shapes_no_pbc(self, device, dtype):
        """Test neighbor list (COO) format output shapes (no PBC)."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1

        neighbor_list, neighbor_ptr = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=None,
            cell=None,
            max_neighbors=20,
            return_neighbor_list=True,
        )

        # Verify shapes and types
        assert neighbor_list.dtype == torch.int32
        assert neighbor_ptr.dtype == torch.int32
        assert neighbor_list.shape[0] == 2
        assert neighbor_ptr.shape == (positions.shape[0] + 1,)
        assert neighbor_list.device == torch.device(device)
        assert neighbor_ptr.device == torch.device(device)

    def test_list_format_output_shapes_with_pbc(self, device, dtype):
        """Test neighbor list (COO) format output shapes (with PBC)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1

        neighbor_list, neighbor_ptr, neighbor_shifts = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=20,
            return_neighbor_list=True,
        )

        # Verify shapes and types
        assert neighbor_list.dtype == torch.int32
        assert neighbor_ptr.dtype == torch.int32
        assert neighbor_shifts.dtype == torch.int32
        assert neighbor_list.shape[0] == 2
        assert neighbor_ptr.shape == (positions.shape[0] + 1,)
        assert neighbor_shifts.shape[1] == 3

    def test_preallocated_output_no_pbc(self, device, dtype, half_fill):
        """Test with preallocated output tensors (no PBC)."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20
        fill_value = -1

        # Preallocate tensors
        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            fill_value,
            dtype=torch.int32,
            device=device,
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        # Call with preallocated tensors
        _ = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            half_fill=half_fill,
            return_neighbor_list=False,
        )

        # When preallocated, return is None or tuple
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix.dtype == torch.int32
        assert torch.all(num_neighbors >= 0)

    def test_preallocated_output_with_pbc(self, device, dtype, half_fill):
        """Test with preallocated output tensors (with PBC)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20
        fill_value = -1

        shift_range_per_dimension, shift_offset, total_shifts = (
            compute_naive_num_shifts(cell, cutoff, pbc)
        )

        # Preallocate tensors
        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            fill_value,
            dtype=torch.int32,
            device=device,
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (positions.shape[0], max_neighbors, 3),
            dtype=torch.int32,
            device=device,
        )

        # Call with preallocated tensors
        _ = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            shift_range_per_dimension=shift_range_per_dimension,
            shift_offset=shift_offset,
            total_shifts=total_shifts,
            half_fill=half_fill,
            return_neighbor_list=False,
        )

        # Verify output
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix.dtype == torch.int32
        assert neighbor_matrix_shifts.dtype == torch.int32
        assert torch.all(num_neighbors >= 0)

    def test_conversion_between_matrix_and_list_formats(self, device, dtype):
        """Test that matrix and list formats contain same neighbor information."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20

        # Get matrix format
        neighbor_matrix, num_neighbors, _ = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            return_neighbor_list=False,
        )

        # Get list format
        neighbor_list, neighbor_ptr, _ = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            return_neighbor_list=True,
        )

        # Total number of neighbors should match
        matrix_total = num_neighbors.sum().item()
        list_total = neighbor_list.shape[1]
        assert matrix_total == list_total


class TestNaiveCompile:
    """Test torch.compile compatibility."""

    def test_compile_no_pbc(self, device, dtype, half_fill):
        """Test that naive_neighbor_list can be compiled (no PBC)."""
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

        # Verify results
        assert num_neighbors.sum() > 0
        num_rows = positions.shape[0] - int(half_fill)
        for i in range(num_rows):
            assert num_neighbors[i].item() > 0
            neighbor_row = neighbor_matrix[i]
            mask = neighbor_row != 50
            assert neighbor_row[mask].shape == (num_neighbors[i].item(),)

    def test_compile_with_pbc(self, device, dtype, half_fill):
        """Test that naive_neighbor_list can be compiled (with PBC)."""
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

        # Verify results
        assert num_neighbors.sum() > 0
        num_rows = positions.shape[0] - int(half_fill)
        for i in range(num_rows):
            assert num_neighbors[i].item() > 0
            neighbor_row = neighbor_matrix[i]
            mask = neighbor_row != 50
            assert neighbor_row[mask].shape == (num_neighbors[i].item(),)


class TestNaivePerformance:
    """Test performance characteristics and scaling (marked as slow)."""

    @pytest.mark.slow
    def test_scaling_with_system_size(self, device):
        """Test that naive implementation has reasonable scaling with system size."""
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

        # Verify time increases with system size (loose check)
        assert times[1] > times[0] * 0.8, "Time should increase with system size"
        if len(times) > 2:
            # Very loose scaling check - should not be orders of magnitude worse than O(N^2)
            scaling_factor = times[-1] / times[0]
            size_factor = (sizes[-1] / sizes[0]) ** 2
            # Allow 5x deviation from ideal O(N^2) scaling
            assert scaling_factor < size_factor * 5, (
                f"Scaling ({scaling_factor:.2f}) much worse than O(N^2) ({size_factor:.2f})"
            )

    def test_cutoff_scaling(self, device):
        """Test that neighbor count increases with cutoff."""
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

        # Verify neighbor count increases with cutoff
        for i in range(1, len(neighbor_counts)):
            assert neighbor_counts[i] >= neighbor_counts[i - 1], (
                f"Neighbor count should increase with cutoff: {neighbor_counts}"
            )

    @pytest.mark.slow
    def test_memory_scaling(self, device):
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

            max_neighbors = 100

            # Clear cache before test
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

            # Run naive implementation
            neighbor_matrix, num_neighbors, unit_shifts = naive_neighbor_list(
                positions=positions,
                cutoff=cutoff,
                pbc=pbc,
                cell=cell,
                max_neighbors=max_neighbors,
            )

            # Verify output shapes are reasonable
            assert neighbor_matrix.shape == (num_atoms, max_neighbors)
            assert unit_shifts.shape == (num_atoms, max_neighbors, 3)
            assert num_neighbors.shape == (num_atoms,)
            assert torch.all(num_neighbors >= 0)
            assert torch.all(num_neighbors <= max_neighbors)

            # Clean up
            del neighbor_matrix, unit_shifts, num_neighbors, positions, cell, pbc
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()
