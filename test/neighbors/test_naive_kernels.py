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
    get_naive_neighbor_matrix_kernel,
    naive_neighbor_matrix,
    naive_neighbor_matrix_pbc,
)
from nvalchemiops.neighbors.naive.launchers import _scalar_sentinels
from nvalchemiops.neighbors.neighbor_utils import (
    _compute_naive_num_shifts,
    _update_neighbor_matrix,
    compute_inv_cells,
    wrap_positions_single,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

from .test_utils import (
    create_random_system,
    create_simple_cubic_system,
)

_FILL_NAIVE_NO_PBC = {
    half_fill: {
        t: get_naive_neighbor_matrix_kernel(
            t, pbc_mode="none", batched=False, half_fill=half_fill
        )
        for t in (wp.float32, wp.float64, wp.float16)
    }
    for half_fill in (False, True)
}
_FILL_NAIVE_PBC_WRAP = {
    half_fill: {
        t: get_naive_neighbor_matrix_kernel(
            t, pbc_mode="wrap_on_entry", batched=False, half_fill=half_fill
        )
        for t in (wp.float32, wp.float64, wp.float16)
    }
    for half_fill in (False, True)
}

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
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    unit_shift: wp.vec3i,
    max_neighbors: int,
    half_fill: bool,
):
    _update_neighbor_matrix(
        i,
        j,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        unit_shift,
        max_neighbors,
        half_fill,
        False,
    )


@wp.kernel
def _update_neighbor_matrix_with_shifts_kernel(
    i: int,
    j: int,
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    unit_shift: wp.vec3i,
    max_neighbors: int,
    half_fill: bool,
):
    _update_neighbor_matrix(
        i,
        j,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        unit_shift,
        max_neighbors,
        half_fill,
        True,
    )


@pytest.mark.parametrize("half_fill", [True, False])
class TestNaiveKernels:
    """Test individual naive neighbor list kernels."""

    def test_update_neighbor_matrix_kernel(self, half_fill):
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
        wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32)
        wp_unit_shift = wp.vec3i(0, 0, 0)

        ij = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        for i, j in ij:
            wp.launch(
                _update_neighbor_matrix_kernel,
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

    def test_update_neighbor_matrix_with_shifts_kernel(self, half_fill):
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
                _update_neighbor_matrix_with_shifts_kernel,
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
        (
            empty_offsets,
            empty_cell,
            empty_shift_range,
            empty_num_shifts,
            empty_batch_idx,
            empty_batch_ptr,
            empty_target_indices,
            empty_matrix,
            empty_shifts,
            empty_num_neighbors,
            empty_vectors,
            empty_distances,
            empty_pair_params,
            empty_energies,
            empty_forces,
            empty_rebuild_flags,
        ) = _scalar_sentinels(wp_dtype, wp_device)

        # Launch kernel
        wp.launch(
            _FILL_NAIVE_NO_PBC[half_fill][wp_dtype],
            dim=(1, 1, positions.shape[0]),
            device=wp_device,
            inputs=[
                wp_positions,
                empty_offsets,
                wp_dtype(cutoff * cutoff),
                wp_dtype(0.0),
                empty_cell,
                empty_shift_range,
                empty_num_shifts,
                empty_batch_idx,
                empty_batch_ptr,
                empty_target_indices,
                wp_neighbor_matrix,
                empty_shifts,
                wp_num_neighbors,
                empty_matrix,
                empty_shifts,
                empty_num_neighbors,
                empty_vectors,
                empty_distances,
                empty_pair_params,
                empty_energies,
                empty_forces,
                empty_rebuild_flags,
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

        total_atoms = positions.shape[0]
        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell.unsqueeze(0), dtype=wp_mat_dtype)
        wp_shifts = wp.from_torch(shifts, dtype=wp.vec3i)

        # Pre-wrap positions
        wp_inv_cell = wp.empty_like(wp_cell)
        compute_inv_cells(wp_cell, wp_inv_cell, wp_dtype, wp_device)
        wp_positions_wrapped = wp.empty_like(wp_positions)
        wp_per_atom_cell_offsets = wp.empty(
            total_atoms, dtype=wp.vec3i, device=wp_device
        )
        wrap_positions_single(
            wp_positions,
            wp_cell,
            wp_inv_cell,
            wp_positions_wrapped,
            wp_per_atom_cell_offsets,
            wp_dtype,
            wp_device,
        )

        # Output arrays
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
        (
            _empty_offsets,
            _empty_cell,
            _empty_shift_range,
            empty_num_shifts,
            empty_batch_idx,
            empty_batch_ptr,
            empty_target_indices,
            empty_matrix,
            empty_shifts,
            empty_num_neighbors,
            empty_vectors,
            empty_distances,
            empty_pair_params,
            empty_energies,
            empty_forces,
            empty_rebuild_flags,
        ) = _scalar_sentinels(wp_dtype, wp_device)

        # Launch kernel using the typed overload
        wp.launch(
            _FILL_NAIVE_PBC_WRAP[half_fill][wp_dtype],
            dim=(1, len(shifts), total_atoms),
            device=wp_device,
            inputs=[
                wp_positions_wrapped,
                wp_per_atom_cell_offsets,
                wp_dtype(cutoff * cutoff),
                wp_dtype(0.0),
                wp_cell,
                wp_shifts,
                empty_num_shifts,
                empty_batch_idx,
                empty_batch_ptr,
                empty_target_indices,
                wp_neighbor_matrix,
                wp_neighbor_matrix_shifts,
                wp_num_neighbors,
                empty_matrix,
                empty_shifts,
                empty_num_neighbors,
                empty_vectors,
                empty_distances,
                empty_pair_params,
                empty_energies,
                empty_forces,
                empty_rebuild_flags,
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
        shift_range_per_dimension, num_shifts, max_shifts = compute_naive_num_shifts(
            cell, cutoff, pbc
        )

        wp_shift_range = wp.from_torch(shift_range_per_dimension, dtype=wp.vec3i)

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
            wp_shift_range,
            max_shifts,
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

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_naive_neighbor_matrix_pbc_prewrapped(self, device, dtype, half_fill):
        """Test naive_neighbor_matrix_pbc with wrap_positions=False."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 30

        shift_range_per_dimension, num_shifts, max_shifts = compute_naive_num_shifts(
            cell, cutoff, pbc
        )
        wp_shift_range = wp.from_torch(shift_range_per_dimension, dtype=wp.vec3i)

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)

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
        naive_neighbor_matrix_pbc(
            wp_positions,
            cutoff,
            wp_cell,
            wp_shift_range,
            max_shifts,
            wp.from_torch(neighbor_matrix, dtype=wp.int32),
            wp.from_torch(neighbor_matrix_shifts, dtype=wp.vec3i),
            wp.from_torch(num_neighbors, dtype=wp.int32),
            wp_dtype,
            str(device),
            half_fill,
            wrap_positions=False,
        )

        assert torch.all(num_neighbors >= 0), "Neighbor counts should be non-negative"
        assert num_neighbors.sum() > 0, "Should find some neighbors"

        valid_shifts = neighbor_matrix_shifts[neighbor_matrix != -1]
        if len(valid_shifts) > 0:
            assert torch.all(torch.abs(valid_shifts) <= 5), (
                "Unit shifts should be small integers"
            )


class TestNaiveSelectiveRebuildFlags:
    """Test selective rebuild (rebuild_flags) for naive neighbor list warp launchers."""

    def test_no_rebuild_preserves_data(self):
        """All flags False: neighbor data should remain unchanged."""
        device = "cuda:0"
        dtype = torch.float32
        wp_device = device

        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)

        # Initial full build
        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors), -1, dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )
        wp_nm = wp.from_torch(neighbor_matrix, dtype=wp.int32)
        wp_nn = wp.from_torch(num_neighbors, dtype=wp.int32)

        naive_neighbor_matrix(
            wp_positions, cutoff, wp_nm, wp_nn, wp_dtype, wp_device, False
        )

        saved_nm = neighbor_matrix.clone()
        saved_nn = num_neighbors.clone()

        # Rebuild with flag=False: nothing should change
        rebuild_flags = torch.zeros(1, dtype=torch.bool, device=device)
        wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool)

        naive_neighbor_matrix(
            wp_positions,
            cutoff,
            wp_nm,
            wp_nn,
            wp_dtype,
            wp_device,
            False,
            rebuild_flags=wp_rebuild_flags,
        )

        assert torch.equal(num_neighbors, saved_nn), (
            "num_neighbors must be unchanged when rebuild_flags is False"
        )
        for i in range(positions.shape[0]):
            n = num_neighbors[i].item()
            assert torch.equal(neighbor_matrix[i, :n], saved_nm[i, :n]), (
                f"neighbor_matrix row {i} should be unchanged"
            )

    def test_rebuild_updates_data(self):
        """Flag=True: result should match a fresh full rebuild."""
        device = "cuda:0"
        dtype = torch.float32
        wp_device = device

        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)

        # Full build reference
        nm_ref = torch.full(
            (positions.shape[0], max_neighbors), -1, dtype=torch.int32, device=device
        )
        nn_ref = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)
        wp_nm_ref = wp.from_torch(nm_ref, dtype=wp.int32)
        wp_nn_ref = wp.from_torch(nn_ref, dtype=wp.int32)
        naive_neighbor_matrix(
            wp_positions, cutoff, wp_nm_ref, wp_nn_ref, wp_dtype, wp_device, False
        )

        # Build with stale data, then selective rebuild with flag=True
        nm_sel = torch.full(
            (positions.shape[0], max_neighbors), 0, dtype=torch.int32, device=device
        )
        nn_sel = torch.full((positions.shape[0],), 0, dtype=torch.int32, device=device)
        wp_nm_sel = wp.from_torch(nm_sel, dtype=wp.int32)
        wp_nn_sel = wp.from_torch(nn_sel, dtype=wp.int32)

        rebuild_flags = torch.ones(1, dtype=torch.bool, device=device)
        wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool)

        naive_neighbor_matrix(
            wp_positions,
            cutoff,
            wp_nm_sel,
            wp_nn_sel,
            wp_dtype,
            wp_device,
            False,
            rebuild_flags=wp_rebuild_flags,
        )

        assert torch.equal(nn_sel, nn_ref), (
            "num_neighbors should match full rebuild when flag=True"
        )


###########################################################################################
########################### Tiled Naive Kernel Tests #######################################
###########################################################################################

dtypes = [torch.float32, torch.float64]


def _skip_if_cpu(device):
    """Skip tiled kernel tests on CPU because wp.launch_tiled is CUDA-only."""
    if "cpu" in str(device):
        pytest.skip(
            "native_strategy='tile' uses wp.launch_tiled and is CUDA-only; "
            "CPU parameter is not supported"
        )


def _neighbor_shift_sets(
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
) -> list[set[tuple[int, tuple[int, int, int]]]]:
    """Return row-local ``(neighbor, shift)`` sets for order-independent checks."""
    matrix_cpu = neighbor_matrix.detach().cpu()
    shifts_cpu = neighbor_matrix_shifts.detach().cpu()
    counts_cpu = num_neighbors.detach().cpu()
    rows = []
    for row in range(matrix_cpu.shape[0]):
        row_items = set()
        for slot in range(int(counts_cpu[row].item())):
            shift = tuple(int(x) for x in shifts_cpu[row, slot].tolist())
            row_items.add((int(matrix_cpu[row, slot].item()), shift))
        rows.append(row_items)
    return rows


@pytest.mark.parametrize("dtype", dtypes)
class TestNaiveTiledMatchesScalar:
    """Verify that tiled naive kernels produce identical results to scalar kernels."""

    def test_tiled_no_pbc_counts_match(self, device, dtype):
        """Tiled non-PBC neighbor counts should match scalar kernel."""
        _skip_if_cpu(device)
        positions, _, _ = create_simple_cubic_system(
            num_atoms=27, cell_size=3.0, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 30
        num_atoms = positions.shape[0]
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_device = str(device)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)

        # Scalar reference
        nm_ref = torch.full(
            (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        nn_ref = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        naive_neighbor_matrix(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp.from_torch(nm_ref, dtype=wp.int32),
            wp.from_torch(nn_ref, dtype=wp.int32),
            wp_dtype,
            wp_device,
            native_strategy="scalar",
        )

        # Tiled
        nm_tiled = torch.full(
            (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        nn_tiled = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        naive_neighbor_matrix(
            wp_positions,
            cutoff,
            wp.from_torch(nm_tiled, dtype=wp.int32),
            wp.from_torch(nn_tiled, dtype=wp.int32),
            wp_dtype,
            wp_device,
            native_strategy="tile",
        )

        assert torch.equal(nn_tiled, nn_ref), (
            f"Tiled counts differ: {nn_tiled.tolist()} vs {nn_ref.tolist()}"
        )

    def test_tiled_no_pbc_random(self, device, dtype):
        """Tiled should match scalar for random non-PBC system."""
        _skip_if_cpu(device)
        positions, _, _ = create_random_system(
            num_atoms=64, cell_size=5.0, dtype=dtype, device=device, pbc_flag=False
        )
        cutoff = 2.0
        max_neighbors = 50
        num_atoms = positions.shape[0]
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_device = str(device)

        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)

        nm_ref = torch.full(
            (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        nn_ref = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        naive_neighbor_matrix(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp.from_torch(nm_ref, dtype=wp.int32),
            wp.from_torch(nn_ref, dtype=wp.int32),
            wp_dtype,
            wp_device,
            native_strategy="scalar",
        )

        nm_tiled = torch.full(
            (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        nn_tiled = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        naive_neighbor_matrix(
            wp_positions,
            cutoff,
            wp.from_torch(nm_tiled, dtype=wp.int32),
            wp.from_torch(nn_tiled, dtype=wp.int32),
            wp_dtype,
            wp_device,
            native_strategy="tile",
        )

        assert torch.equal(nn_tiled, nn_ref), (
            "Random system: tiled counts differ from scalar"
        )

    def test_tiled_pbc_counts_match(self, device, dtype):
        """Tiled PBC neighbor counts should match scalar kernel."""
        _skip_if_cpu(device)
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=27, cell_size=3.0, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 30
        num_atoms = positions.shape[0]
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)

        shift_range, num_shifts, _ = compute_naive_num_shifts(cell, cutoff, pbc)

        # Note: PBC tiled launcher accesses .ptr for caching, so we must NOT
        # use return_ctype=True for positions/cell passed to the tiled launcher.
        wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)
        wp_cell_obj = wp.from_torch(cell, dtype=wp_mat_dtype)
        wp_shift_range_obj = wp.from_torch(shift_range, dtype=wp.vec3i)

        # Scalar reference
        nm_ref = torch.full(
            (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        ns_ref = torch.zeros(
            (num_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        nn_ref = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        naive_neighbor_matrix_pbc(
            wp_positions,
            cutoff,
            wp_cell_obj,
            wp_shift_range_obj,
            num_shifts,
            wp.from_torch(nm_ref, dtype=wp.int32),
            wp.from_torch(ns_ref, dtype=wp.vec3i),
            wp.from_torch(nn_ref, dtype=wp.int32),
            wp_dtype,
            wp_device,
            native_strategy="scalar",
        )

        # Tiled
        nm_tiled = torch.full(
            (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        ns_tiled = torch.zeros(
            (num_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        nn_tiled = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        naive_neighbor_matrix_pbc(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp_cell_obj,
            wp_shift_range_obj,
            num_shifts,
            wp.from_torch(nm_tiled, dtype=wp.int32),
            wp.from_torch(ns_tiled, dtype=wp.vec3i),
            wp.from_torch(nn_tiled, dtype=wp.int32),
            wp_dtype,
            wp_device,
            native_strategy="tile",
        )

        assert torch.equal(nn_tiled, nn_ref), (
            f"PBC tiled counts differ: {nn_tiled.tolist()} vs {nn_ref.tolist()}"
        )

    @pytest.mark.parametrize("wrap_positions", [True, False])
    def test_tiled_pbc_neighbor_shifts_match_scalar(
        self, device, dtype, wrap_positions
    ):
        """Tiled PBC should match scalar row-local neighbor/shift sets."""
        _skip_if_cpu(device)
        half_fill = False
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)
        box = 2.0
        positions = torch.tensor(
            [
                [0.1, 0.5, 0.5],
                [2.1, 0.5, 0.5],
                [1.9, 0.5, 0.5],
                [0.5, 1.5, 0.5],
            ],
            dtype=dtype,
            device=device,
        )
        if not wrap_positions:
            positions = positions.remainder(box)
        cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * box
        pbc = torch.ones((1, 3), dtype=torch.bool, device=device)
        cutoff = 0.35
        max_neighbors = 12
        num_atoms = positions.shape[0]
        shift_range, num_shifts, _ = compute_naive_num_shifts(cell, cutoff, pbc)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)
        wp_shift_range = wp.from_torch(shift_range, dtype=wp.vec3i)

        nm_scalar = torch.full(
            (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        ns_scalar = torch.zeros(
            (num_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        nn_scalar = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        distances = torch.zeros((num_atoms, max_neighbors), dtype=dtype, device=device)
        naive_neighbor_matrix_pbc(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp_cell,
            wp_shift_range,
            int(num_shifts[0].item()),
            wp.from_torch(nm_scalar, dtype=wp.int32),
            wp.from_torch(ns_scalar, dtype=wp.vec3i),
            wp.from_torch(nn_scalar, dtype=wp.int32),
            wp_dtype,
            wp_device,
            half_fill=half_fill,
            wrap_positions=wrap_positions,
            return_distances=True,
            neighbor_distances=wp.from_torch(distances, dtype=wp_dtype),
            native_strategy="scalar",
        )

        nm_tiled = torch.full(
            (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        ns_tiled = torch.zeros(
            (num_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        nn_tiled = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        naive_neighbor_matrix_pbc(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp_cell,
            wp_shift_range,
            int(num_shifts[0].item()),
            wp.from_torch(nm_tiled, dtype=wp.int32),
            wp.from_torch(ns_tiled, dtype=wp.vec3i),
            wp.from_torch(nn_tiled, dtype=wp.int32),
            wp_dtype,
            wp_device,
            half_fill=half_fill,
            wrap_positions=wrap_positions,
            native_strategy="tile",
        )

        assert _neighbor_shift_sets(nm_tiled, ns_tiled, nn_tiled) == (
            _neighbor_shift_sets(nm_scalar, ns_scalar, nn_scalar)
        )

    def test_tiled_half_fill_no_pbc(self, device, dtype):
        """Tiled half_fill=True should match scalar half_fill counts."""
        _skip_if_cpu(device)
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20
        num_atoms = positions.shape[0]
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_device = str(device)

        # Scalar reference
        nm_ref = torch.full(
            (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        nn_ref = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        naive_neighbor_matrix(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp.from_torch(nm_ref, dtype=wp.int32),
            wp.from_torch(nn_ref, dtype=wp.int32),
            wp_dtype,
            wp_device,
            half_fill=True,
            native_strategy="scalar",
        )

        # Tiled
        nm_tiled = torch.full(
            (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        nn_tiled = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        naive_neighbor_matrix(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp.from_torch(nm_tiled, dtype=wp.int32),
            wp.from_torch(nn_tiled, dtype=wp.int32),
            wp_dtype,
            wp_device,
            half_fill=True,
            native_strategy="tile",
        )

        assert torch.equal(nn_tiled, nn_ref), (
            "Half-fill tiled counts differ from scalar"
        )


@pytest.mark.parametrize("dtype", dtypes)
class TestNaiveTiledSelectiveRebuild:
    """Test selective rebuild for tiled naive kernels."""

    def test_no_rebuild_preserves_data(self, device, dtype):
        """Flag=False should leave existing data untouched."""
        _skip_if_cpu(device)
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20
        num_atoms = positions.shape[0]
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_device = str(device)

        # First build to populate data
        nm = torch.full(
            (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        nn = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        wp_nm = wp.from_torch(nm, dtype=wp.int32)
        wp_nn = wp.from_torch(nn, dtype=wp.int32)
        naive_neighbor_matrix(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp_nm,
            wp_nn,
            wp_dtype,
            wp_device,
            native_strategy="tile",
        )
        saved_nn = nn.clone()

        # Rebuild with flag=False — data should be unchanged
        rebuild_flags = torch.zeros(1, dtype=torch.bool, device=device)
        wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool)
        naive_neighbor_matrix(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp_nm,
            wp_nn,
            wp_dtype,
            wp_device,
            rebuild_flags=wp_rebuild_flags,
            native_strategy="tile",
        )

        assert torch.equal(nn, saved_nn), "num_neighbors unchanged when flag=False"

    def test_rebuild_matches_full(self, device, dtype):
        """Flag=True should produce same results as full build."""
        _skip_if_cpu(device)
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20
        num_atoms = positions.shape[0]
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_device = str(device)

        # Full build reference
        nm_ref = torch.full(
            (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        nn_ref = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        naive_neighbor_matrix(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp.from_torch(nm_ref, dtype=wp.int32),
            wp.from_torch(nn_ref, dtype=wp.int32),
            wp_dtype,
            wp_device,
            native_strategy="tile",
        )

        # Selective build with flag=True
        nm_sel = torch.full(
            (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        nn_sel = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        rebuild_flags = torch.ones(1, dtype=torch.bool, device=device)
        naive_neighbor_matrix(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp.from_torch(nm_sel, dtype=wp.int32),
            wp.from_torch(nn_sel, dtype=wp.int32),
            wp_dtype,
            wp_device,
            rebuild_flags=wp.from_torch(rebuild_flags, dtype=wp.bool),
            native_strategy="tile",
        )

        assert torch.equal(nn_sel, nn_ref), (
            "Selective rebuild should match full build when flag=True"
        )


# ---------------------------------------------------------------------------
# Isolated pair-output kwargs on the naive launchers.  Each test exercises one
# of the partial and pair-output features (target_indices / return_vectors / return_distances)
# in isolation (no pair_fn) against a brute-force reference.
# ---------------------------------------------------------------------------


class TestNaivePairOutputsIsolated:
    """Each pair-output kwarg verified in isolation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_return_distances_only(self, device):
        """``return_distances=True`` writes correct per-pair distances."""
        dtype = torch.float32
        positions, _, _ = create_simple_cubic_system(
            num_atoms=27, dtype=dtype, device=device
        )
        cutoff = 1.5
        max_neighbors = 30
        N = positions.shape[0]

        neighbor_matrix = torch.full(
            (N, max_neighbors), N, dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(N, dtype=torch.int32, device=device)
        distances = torch.zeros((N, max_neighbors), dtype=dtype, device=device)

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        naive_neighbor_matrix(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp.from_torch(neighbor_matrix, dtype=wp.int32),
            wp.from_torch(num_neighbors, dtype=wp.int32),
            wp_dtype,
            str(device),
            return_distances=True,
            neighbor_distances=wp.from_torch(distances, dtype=wp_dtype),
        )

        # Reference: brute-force distance for every emitted neighbor slot.
        for i in range(N):
            n = int(num_neighbors[i].item())
            for slot in range(n):
                j = int(neighbor_matrix[i, slot].item())
                if j >= N:
                    continue
                d_ref = float(torch.linalg.norm(positions[i] - positions[j]))
                d_obs = float(distances[i, slot].item())
                assert abs(d_obs - d_ref) < 1e-4, (
                    f"distance mismatch at i={i}, slot={slot}: obs={d_obs}, ref={d_ref}"
                )
                assert d_obs <= cutoff + 1e-4, "distance must be within cutoff"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_return_vectors_only(self, device):
        """``return_vectors=True`` writes correct per-pair displacements."""
        dtype = torch.float32
        positions, _, _ = create_simple_cubic_system(
            num_atoms=27, dtype=dtype, device=device
        )
        cutoff = 1.5
        max_neighbors = 30
        N = positions.shape[0]

        neighbor_matrix = torch.full(
            (N, max_neighbors), N, dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(N, dtype=torch.int32, device=device)
        vectors = torch.zeros((N, max_neighbors, 3), dtype=dtype, device=device)

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        naive_neighbor_matrix(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp.from_torch(neighbor_matrix, dtype=wp.int32),
            wp.from_torch(num_neighbors, dtype=wp.int32),
            wp_dtype,
            str(device),
            return_vectors=True,
            neighbor_vectors=wp.from_torch(vectors, dtype=wp_vec_dtype),
        )

        # Reference: per-slot r_ij = positions[j] - positions[i].
        for i in range(N):
            n = int(num_neighbors[i].item())
            for slot in range(n):
                j = int(neighbor_matrix[i, slot].item())
                if j >= N:
                    continue
                r_ref = positions[j] - positions[i]
                r_obs = vectors[i, slot]
                assert torch.allclose(r_obs, r_ref, atol=1e-4), (
                    f"vector mismatch at i={i}, slot={slot}: "
                    f"obs={r_obs.tolist()}, ref={r_ref.tolist()}"
                )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_target_indices_only(self, device):
        """``target_indices`` restricts central rows to the index subset."""
        dtype = torch.float32
        positions, _, _ = create_simple_cubic_system(
            num_atoms=27, dtype=dtype, device=device
        )
        cutoff = 1.5
        max_neighbors = 30
        N = positions.shape[0]

        # Pick 3 atoms as targets — rows in the output map to these in order.
        target_idx = torch.tensor([0, 5, 17], dtype=torch.int32, device=device)
        K = int(target_idx.shape[0])

        # Compact output: rows == num_targets.
        partial_matrix = torch.full(
            (K, max_neighbors), N, dtype=torch.int32, device=device
        )
        partial_counts = torch.zeros(K, dtype=torch.int32, device=device)

        # Reference run: full build, then check that partial rows match.
        full_matrix = torch.full(
            (N, max_neighbors), N, dtype=torch.int32, device=device
        )
        full_counts = torch.zeros(N, dtype=torch.int32, device=device)

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        # Full neighbor list.
        naive_neighbor_matrix(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp.from_torch(full_matrix, dtype=wp.int32),
            wp.from_torch(full_counts, dtype=wp.int32),
            wp_dtype,
            str(device),
        )

        # Partial neighbor list at the same atoms.
        naive_neighbor_matrix(
            wp.from_torch(positions, dtype=wp_vec_dtype),
            cutoff,
            wp.from_torch(partial_matrix, dtype=wp.int32),
            wp.from_torch(partial_counts, dtype=wp.int32),
            wp_dtype,
            str(device),
            target_indices=wp.from_torch(target_idx, dtype=wp.int32),
        )

        # Each partial row's count must match the corresponding full row.
        for row, src in enumerate(target_idx.tolist()):
            assert int(partial_counts[row].item()) == int(full_counts[src].item()), (
                f"target row {row} (atom {src}): "
                f"partial count {int(partial_counts[row])} != "
                f"full count {int(full_counts[src])}"
            )
            # Compare neighbor sets (order may differ within row).
            n = int(full_counts[src].item())
            partial_set = set(partial_matrix[row, :n].tolist())
            full_set = set(full_matrix[src, :n].tolist())
            assert partial_set == full_set, (
                f"neighbor sets differ for target row {row} (atom {src}): "
                f"partial={partial_set}, full={full_set}"
            )
