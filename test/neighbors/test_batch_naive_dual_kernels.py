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

from nvalchemiops.neighbors.naive import (
    batch_naive_neighbor_matrix_dual_cutoff,
    batch_naive_neighbor_matrix_pbc_dual_cutoff,
    get_naive_neighbor_matrix_dual_cutoff_kernel,
)
from nvalchemiops.neighbors.naive.launchers import _scalar_sentinels
from nvalchemiops.neighbors.neighbor_utils import (
    compute_inv_cells,
    wrap_positions_batch,
)
from nvalchemiops.torch.neighbors.neighbor_utils import compute_naive_num_shifts
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

from .test_utils import create_batch_systems

_FILL_BATCH_NAIVE_DUAL = {
    half_fill: {
        t: get_naive_neighbor_matrix_dual_cutoff_kernel(
            t, pbc_mode="none", batched=True, half_fill=half_fill
        )
        for t in (wp.float32, wp.float64, wp.float16)
    }
    for half_fill in (False, True)
}
_FILL_BATCH_NAIVE_PBC_DUAL_WRAP = {
    half_fill: {
        t: get_naive_neighbor_matrix_dual_cutoff_kernel(
            t, pbc_mode="wrap_on_entry", batched=True, half_fill=half_fill
        )
        for t in (wp.float32, wp.float64, wp.float16)
    }
    for half_fill in (False, True)
}


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
        (
            empty_offsets,
            empty_cell,
            empty_shift_range,
            empty_num_shifts,
            _empty_batch_idx,
            _empty_batch_ptr,
            empty_target_indices,
            _empty_matrix,
            empty_shifts,
            _empty_num_neighbors,
            empty_vectors,
            empty_distances,
            empty_pair_params,
            empty_energies,
            empty_forces,
            empty_rebuild_flags,
        ) = _scalar_sentinels(wp_dtype, wp_device)

        # Launch kernel
        wp.launch(
            _FILL_BATCH_NAIVE_DUAL[half_fill][wp_dtype],
            dim=(1, 1, positions_batch.shape[0]),
            device=wp_device,
            inputs=[
                wp_positions,
                empty_offsets,
                wp_dtype(cutoff1 * cutoff1),
                wp_dtype(cutoff2 * cutoff2),
                empty_cell,
                empty_shift_range,
                empty_num_shifts,
                wp_batch_idx,
                wp_batch_ptr,
                empty_target_indices,
                wp_neighbor_matrix1,
                empty_shifts,
                wp_num_neighbors1,
                wp_neighbor_matrix2,
                empty_shifts,
                wp_num_neighbors2,
                empty_vectors,
                empty_distances,
                empty_pair_params,
                empty_energies,
                empty_forces,
                empty_rebuild_flags,
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
        num_systems = 2
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=num_systems,
            atoms_per_system=atoms_per_system,
            dtype=dtype,
            device=device,
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 20
        max_atoms_per_system = max(atoms_per_system)

        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cell_batch, cutoff2, pbc_batch
        )

        # Convert to warp types
        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)
        wp_device = str(device)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell_batch, dtype=wp_mat_dtype)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32)
        wp_shift_range = wp.from_torch(shift_range, dtype=wp.vec3i)
        wp_num_shifts = wp.from_torch(num_shifts, dtype=wp.int32)

        # Pre-wrap positions
        inv_cell_arr = torch.zeros_like(cell_batch)
        wp_inv_cell = wp.from_torch(inv_cell_arr, dtype=wp_mat_dtype)
        compute_inv_cells(wp_cell, wp_inv_cell, wp_dtype, wp_device)
        positions_wrapped_arr = torch.zeros_like(positions_batch)
        per_atom_cell_offsets_arr = torch.zeros(
            positions_batch.shape[0], 3, dtype=torch.int32, device=device
        )
        wp_positions_wrapped = wp.from_torch(positions_wrapped_arr, dtype=wp_vec_dtype)
        wp_per_atom_cell_offsets = wp.from_torch(
            per_atom_cell_offsets_arr, dtype=wp.vec3i
        )
        wrap_positions_batch(
            wp_positions,
            wp_cell,
            wp_inv_cell,
            wp_batch_idx,
            wp_positions_wrapped,
            wp_per_atom_cell_offsets,
            wp_dtype,
            wp_device,
        )

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
        (
            _empty_offsets,
            _empty_cell,
            _empty_shift_range,
            _empty_num_shifts,
            _empty_batch_idx,
            _empty_batch_ptr,
            empty_target_indices,
            _empty_matrix,
            _empty_shifts,
            _empty_num_neighbors,
            empty_vectors,
            empty_distances,
            empty_pair_params,
            empty_energies,
            empty_forces,
            empty_rebuild_flags,
        ) = _scalar_sentinels(wp_dtype, wp_device)

        # Launch kernel with 3D dims: (num_systems, max_shifts, max_atoms_per_system)
        wp.launch(
            _FILL_BATCH_NAIVE_PBC_DUAL_WRAP[True][wp_dtype],
            dim=(num_systems, max_shifts, max_atoms_per_system),
            device=wp_device,
            inputs=[
                wp_positions_wrapped,
                wp_per_atom_cell_offsets,
                wp_dtype(cutoff1 * cutoff1),
                wp_dtype(cutoff2 * cutoff2),
                wp_cell,
                wp_shift_range,
                wp_num_shifts,
                wp_batch_idx,
                wp_batch_ptr,
                empty_target_indices,
                wp_neighbor_matrix1,
                wp_neighbor_matrix_shifts1,
                wp_num_neighbors1,
                wp_neighbor_matrix2,
                wp_neighbor_matrix_shifts2,
                wp_num_neighbors2,
                empty_vectors,
                empty_distances,
                empty_pair_params,
                empty_energies,
                empty_forces,
                empty_rebuild_flags,
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
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 30
        max_neighbors2 = 40
        max_atoms_per_system = max(atoms_per_system)

        # Compute shift ranges (on-the-fly API)
        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cell_batch, cutoff2, pbc_batch
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

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell_batch, dtype=get_wp_mat_dtype(dtype))
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32)
        wp_shift_range = wp.from_torch(shift_range, dtype=wp.vec3i)
        wp_num_shifts = wp.from_torch(num_shifts, dtype=wp.int32)
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
            positions=wp_positions,
            cell=wp_cell,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_ptr=wp_batch_ptr,
            batch_idx=wp_batch_idx,
            shift_range=wp_shift_range,
            num_shifts_arr=wp_num_shifts,
            max_shifts_per_system=max_shifts,
            neighbor_matrix1=wp_neighbor_matrix1,
            neighbor_matrix2=wp_neighbor_matrix2,
            neighbor_matrix_shifts1=wp_neighbor_matrix_shifts1,
            neighbor_matrix_shifts2=wp_neighbor_matrix_shifts2,
            num_neighbors1=wp_num_neighbors1,
            num_neighbors2=wp_num_neighbors2,
            wp_dtype=wp_dtype,
            device=str(device),
            max_atoms_per_system=max_atoms_per_system,
            half_fill=half_fill,
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

    def test_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped(
        self, device, dtype, half_fill
    ):
        """Test batch_naive_neighbor_matrix_pbc_dual_cutoff with wrap_positions=False."""
        atoms_per_system = [4, 6]
        num_systems = 2
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=num_systems,
            atoms_per_system=atoms_per_system,
            dtype=dtype,
            device=device,
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 30
        max_neighbors2 = 40
        max_atoms_per_system = max(atoms_per_system)

        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cell_batch, cutoff2, pbc_batch
        )

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell_batch, dtype=get_wp_mat_dtype(dtype))
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32)
        wp_shift_range = wp.from_torch(shift_range, dtype=wp.vec3i)
        wp_num_shifts = wp.from_torch(num_shifts, dtype=wp.int32)

        total_atoms = positions_batch.shape[0]

        neighbor_matrix1 = torch.full(
            (total_atoms, max_neighbors1), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix2 = torch.full(
            (total_atoms, max_neighbors2), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts1 = torch.zeros(
            (total_atoms, max_neighbors1, 3), dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts2 = torch.zeros(
            (total_atoms, max_neighbors2, 3), dtype=torch.int32, device=device
        )
        num_neighbors1 = torch.zeros(total_atoms, dtype=torch.int32, device=device)
        num_neighbors2 = torch.zeros(total_atoms, dtype=torch.int32, device=device)

        batch_naive_neighbor_matrix_pbc_dual_cutoff(
            positions=wp_positions,
            cell=wp_cell,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_ptr=wp_batch_ptr,
            batch_idx=wp_batch_idx,
            shift_range=wp_shift_range,
            num_shifts_arr=wp_num_shifts,
            max_shifts_per_system=max_shifts,
            neighbor_matrix1=wp.from_torch(neighbor_matrix1, dtype=wp.int32),
            neighbor_matrix2=wp.from_torch(neighbor_matrix2, dtype=wp.int32),
            neighbor_matrix_shifts1=wp.from_torch(
                neighbor_matrix_shifts1, dtype=wp.vec3i
            ),
            neighbor_matrix_shifts2=wp.from_torch(
                neighbor_matrix_shifts2, dtype=wp.vec3i
            ),
            num_neighbors1=wp.from_torch(num_neighbors1, dtype=wp.int32),
            num_neighbors2=wp.from_torch(num_neighbors2, dtype=wp.int32),
            wp_dtype=wp_dtype,
            device=str(device),
            max_atoms_per_system=max_atoms_per_system,
            half_fill=half_fill,
            wrap_positions=False,
        )

        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors2 >= num_neighbors1)

        valid_shifts1 = neighbor_matrix_shifts1[neighbor_matrix1 != -1]
        valid_shifts2 = neighbor_matrix_shifts2[neighbor_matrix2 != -1]
        if len(valid_shifts1) > 0:
            assert torch.all(torch.abs(valid_shifts1) <= 5)
        if len(valid_shifts2) > 0:
            assert torch.all(torch.abs(valid_shifts2) <= 5)


class TestBatchNaiveDualCutoffSelectiveRebuildFlags:
    """Test selective rebuild (rebuild_flags) for batch naive dual cutoff warp launchers."""

    def test_no_rebuild_preserves_data(self):
        """All flags False: neighbor data should remain unchanged for all systems."""
        device = "cuda:0"
        dtype = torch.float32

        atoms_per_system = [5, 6]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 20
        max_neighbors2 = 30
        total_atoms = positions_batch.shape[0]

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)

        # Initial full build
        nm1 = torch.full(
            (total_atoms, max_neighbors1), -1, dtype=torch.int32, device=device
        )
        nm2 = torch.full(
            (total_atoms, max_neighbors2), -1, dtype=torch.int32, device=device
        )
        nn1 = torch.zeros(total_atoms, dtype=torch.int32, device=device)
        nn2 = torch.zeros(total_atoms, dtype=torch.int32, device=device)
        wp_nm1 = wp.from_torch(nm1, dtype=wp.int32)
        wp_nm2 = wp.from_torch(nm2, dtype=wp.int32)
        wp_nn1 = wp.from_torch(nn1, dtype=wp.int32)
        wp_nn2 = wp.from_torch(nn2, dtype=wp.int32)

        batch_naive_neighbor_matrix_dual_cutoff(
            wp_positions,
            cutoff1,
            cutoff2,
            wp_batch_idx,
            wp_batch_ptr,
            wp_nm1,
            wp_nn1,
            wp_nm2,
            wp_nn2,
            wp_dtype,
            device,
            False,
        )

        saved_nn1 = nn1.clone()
        saved_nn2 = nn2.clone()

        # Selective rebuild with all flags=False: data should be unchanged
        rebuild_flags = torch.zeros(2, dtype=torch.bool, device=device)
        wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool)

        batch_naive_neighbor_matrix_dual_cutoff(
            wp_positions,
            cutoff1,
            cutoff2,
            wp_batch_idx,
            wp_batch_ptr,
            wp_nm1,
            wp_nn1,
            wp_nm2,
            wp_nn2,
            wp_dtype,
            device,
            False,
            rebuild_flags=wp_rebuild_flags,
        )

        assert torch.equal(nn1, saved_nn1), "nn1 must be unchanged when flags are False"
        assert torch.equal(nn2, saved_nn2), "nn2 must be unchanged when flags are False"

    def test_rebuild_updates_data(self):
        """True flags: rebuilt system data should match a fresh full rebuild."""
        device = "cuda:0"
        dtype = torch.float32

        atoms_per_system = [5, 6]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 20
        max_neighbors2 = 30
        total_atoms = positions_batch.shape[0]

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)

        # Reference: full build
        nm1_ref = torch.full(
            (total_atoms, max_neighbors1), -1, dtype=torch.int32, device=device
        )
        nm2_ref = torch.full(
            (total_atoms, max_neighbors2), -1, dtype=torch.int32, device=device
        )
        nn1_ref = torch.zeros(total_atoms, dtype=torch.int32, device=device)
        nn2_ref = torch.zeros(total_atoms, dtype=torch.int32, device=device)
        wp_nm1_ref = wp.from_torch(nm1_ref, dtype=wp.int32)
        wp_nm2_ref = wp.from_torch(nm2_ref, dtype=wp.int32)
        wp_nn1_ref = wp.from_torch(nn1_ref, dtype=wp.int32)
        wp_nn2_ref = wp.from_torch(nn2_ref, dtype=wp.int32)
        batch_naive_neighbor_matrix_dual_cutoff(
            wp_positions,
            cutoff1,
            cutoff2,
            wp_batch_idx,
            wp_batch_ptr,
            wp_nm1_ref,
            wp_nn1_ref,
            wp_nm2_ref,
            wp_nn2_ref,
            wp_dtype,
            device,
            False,
        )

        # Selective rebuild with all flags=True
        nm1_sel = torch.full(
            (total_atoms, max_neighbors1), 99, dtype=torch.int32, device=device
        )
        nm2_sel = torch.full(
            (total_atoms, max_neighbors2), 99, dtype=torch.int32, device=device
        )
        nn1_sel = torch.full((total_atoms,), 99, dtype=torch.int32, device=device)
        nn2_sel = torch.full((total_atoms,), 99, dtype=torch.int32, device=device)
        wp_nm1_sel = wp.from_torch(nm1_sel, dtype=wp.int32)
        wp_nm2_sel = wp.from_torch(nm2_sel, dtype=wp.int32)
        wp_nn1_sel = wp.from_torch(nn1_sel, dtype=wp.int32)
        wp_nn2_sel = wp.from_torch(nn2_sel, dtype=wp.int32)

        rebuild_flags = torch.ones(2, dtype=torch.bool, device=device)
        wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool)

        batch_naive_neighbor_matrix_dual_cutoff(
            wp_positions,
            cutoff1,
            cutoff2,
            wp_batch_idx,
            wp_batch_ptr,
            wp_nm1_sel,
            wp_nn1_sel,
            wp_nm2_sel,
            wp_nn2_sel,
            wp_dtype,
            device,
            False,
            rebuild_flags=wp_rebuild_flags,
        )

        assert torch.equal(nn1_sel, nn1_ref), (
            "nn1 should match full rebuild when all flags=True"
        )
        assert torch.equal(nn2_sel, nn2_ref), (
            "nn2 should match full rebuild when all flags=True"
        )
