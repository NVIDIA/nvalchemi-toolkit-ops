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
    _fill_batch_naive_neighbor_matrix_pbc_overload,
    batch_naive_neighbor_matrix,
    batch_naive_neighbor_matrix_pbc,
)
from nvalchemiops.neighbors.neighbor_utils import (
    compute_inv_cells,
    wrap_positions_batch,
)
from nvalchemiops.torch.neighbors.neighbor_utils import compute_naive_num_shifts
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

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
        atoms_per_system = [3, 4]
        num_systems = 2
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=num_systems,
            atoms_per_system=atoms_per_system,
            dtype=dtype,
            device=device,
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.0
        max_neighbors = 15
        max_atoms_per_system = max(atoms_per_system)

        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cell_batch, cutoff, pbc_batch
        )

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
        total_atoms = positions_batch.shape[0]
        wp_inv_cell = wp.empty_like(wp_cell)
        compute_inv_cells(wp_cell, wp_inv_cell, wp_dtype, wp_device)
        wp_positions_wrapped = wp.empty_like(wp_positions)
        wp_per_atom_cell_offsets = wp.empty(
            total_atoms, dtype=wp.vec3i, device=wp_device
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

        # Launch kernel using the typed overload (3D: systems x shifts x atoms)
        wp.launch(
            _fill_batch_naive_neighbor_matrix_pbc_overload[wp_dtype],
            dim=(num_systems, max_shifts, max_atoms_per_system),
            device=wp_device,
            inputs=[
                wp_positions_wrapped,
                wp_per_atom_cell_offsets,
                wp_cell,
                wp_dtype(cutoff),
                wp_batch_ptr,
                wp_shift_range,
                wp_num_shifts,
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

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)
        cutoff = 1.2
        max_neighbors = 25
        max_atoms_per_system = max(atoms_per_system)

        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cell_batch, cutoff, pbc_batch
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

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell_batch, dtype=get_wp_mat_dtype(dtype))
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)
        wp_shift_range = wp.from_torch(shift_range, dtype=wp.vec3i)
        wp_num_shifts = wp.from_torch(num_shifts, dtype=wp.int32)
        wp_neighbor_matrix = wp.from_torch(neighbor_matrix, dtype=wp.int32)
        wp_neighbor_matrix_shifts = wp.from_torch(
            neighbor_matrix_shifts, dtype=wp.vec3i
        )
        wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32)

        # Call launcher
        batch_naive_neighbor_matrix_pbc(
            positions=wp_positions,
            cell=wp_cell,
            cutoff=cutoff,
            batch_ptr=wp_batch_ptr,
            batch_idx=wp_batch_idx,
            shift_range=wp_shift_range,
            num_shifts_arr=wp_num_shifts,
            max_shifts_per_system=max_shifts,
            neighbor_matrix=wp_neighbor_matrix,
            neighbor_matrix_shifts=wp_neighbor_matrix_shifts,
            num_neighbors=wp_num_neighbors,
            wp_dtype=wp_dtype,
            device=str(device),
            max_atoms_per_system=max_atoms_per_system,
            half_fill=half_fill,
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

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_matrix_pbc_prewrapped(self, device, dtype, half_fill):
        """Test batch_naive_neighbor_matrix_pbc with wrap_positions=False."""
        atoms_per_system = [4, 5]
        num_systems = 2
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=num_systems,
            atoms_per_system=atoms_per_system,
            dtype=dtype,
            device=device,
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)
        cutoff = 1.2
        max_neighbors = 25
        max_atoms_per_system = max(atoms_per_system)

        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cell_batch, cutoff, pbc_batch
        )

        total_atoms = positions_batch.shape[0]

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell_batch, dtype=get_wp_mat_dtype(dtype))
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)
        wp_shift_range = wp.from_torch(shift_range, dtype=wp.vec3i)
        wp_num_shifts = wp.from_torch(num_shifts, dtype=wp.int32)

        neighbor_matrix = torch.full(
            (total_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (total_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(total_atoms, dtype=torch.int32, device=device)
        batch_naive_neighbor_matrix_pbc(
            positions=wp_positions,
            cell=wp_cell,
            cutoff=cutoff,
            batch_ptr=wp_batch_ptr,
            batch_idx=wp_batch_idx,
            shift_range=wp_shift_range,
            num_shifts_arr=wp_num_shifts,
            max_shifts_per_system=max_shifts,
            neighbor_matrix=wp.from_torch(neighbor_matrix, dtype=wp.int32),
            neighbor_matrix_shifts=wp.from_torch(
                neighbor_matrix_shifts, dtype=wp.vec3i
            ),
            num_neighbors=wp.from_torch(num_neighbors, dtype=wp.int32),
            wp_dtype=wp_dtype,
            device=str(device),
            max_atoms_per_system=max_atoms_per_system,
            half_fill=half_fill,
            wrap_positions=False,
        )

        assert torch.all(num_neighbors >= 0), "Neighbor counts should be non-negative"
        assert num_neighbors.sum().item() > 0, "Should find some neighbors"

        valid_shifts = neighbor_matrix_shifts[neighbor_matrix != -1]
        if len(valid_shifts) > 0:
            assert torch.all(torch.abs(valid_shifts) <= 5), (
                "Unit shifts should be small integers"
            )


class TestBatchNaiveSelectiveRebuildFlags:
    """Test selective rebuild (rebuild_flags) for batch naive neighbor list warp launchers."""

    def test_no_rebuild_preserves_data(self):
        """All flags False: neighbor data should remain unchanged for all systems."""
        device = "cuda:0"
        dtype = torch.float32

        atoms_per_system = [5, 6]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 20
        total_atoms = positions_batch.shape[0]

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)

        # Initial full build
        neighbor_matrix = torch.full(
            (total_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(total_atoms, dtype=torch.int32, device=device)
        wp_nm = wp.from_torch(neighbor_matrix, dtype=wp.int32)
        wp_nn = wp.from_torch(num_neighbors, dtype=wp.int32)

        batch_naive_neighbor_matrix(
            wp_positions,
            cutoff,
            wp_batch_idx,
            wp_batch_ptr,
            wp_nm,
            wp_nn,
            wp_dtype,
            device,
            False,
        )

        saved_nm = neighbor_matrix.clone()
        saved_nn = num_neighbors.clone()

        # Selective rebuild with all flags=False: data should be unchanged
        rebuild_flags = torch.zeros(2, dtype=torch.bool, device=device)
        wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool)

        batch_naive_neighbor_matrix(
            wp_positions,
            cutoff,
            wp_batch_idx,
            wp_batch_ptr,
            wp_nm,
            wp_nn,
            wp_dtype,
            device,
            False,
            rebuild_flags=wp_rebuild_flags,
        )

        assert torch.equal(num_neighbors, saved_nn), (
            "num_neighbors must be unchanged when all rebuild_flags are False"
        )
        for i in range(total_atoms):
            n = num_neighbors[i].item()
            assert torch.equal(neighbor_matrix[i, :n], saved_nm[i, :n]), (
                f"neighbor_matrix row {i} should be unchanged"
            )

    def test_rebuild_updates_data(self):
        """True flags: rebuilt system data should match a fresh full rebuild."""
        device = "cuda:0"
        dtype = torch.float32

        atoms_per_system = [5, 6]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 20
        total_atoms = positions_batch.shape[0]

        wp_dtype = get_wp_dtype(dtype)
        wp_vec_dtype = get_wp_vec_dtype(dtype)

        wp_positions = wp.from_torch(positions_batch, dtype=wp_vec_dtype)
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32)

        # Full build reference
        nm_ref = torch.full(
            (total_atoms, max_neighbors), -1, dtype=torch.int32, device=device
        )
        nn_ref = torch.zeros(total_atoms, dtype=torch.int32, device=device)
        wp_nm_ref = wp.from_torch(nm_ref, dtype=wp.int32)
        wp_nn_ref = wp.from_torch(nn_ref, dtype=wp.int32)
        batch_naive_neighbor_matrix(
            wp_positions,
            cutoff,
            wp_batch_idx,
            wp_batch_ptr,
            wp_nm_ref,
            wp_nn_ref,
            wp_dtype,
            device,
            False,
        )

        # Selective rebuild with all flags=True: result should match reference
        nm_sel = torch.full(
            (total_atoms, max_neighbors), 99, dtype=torch.int32, device=device
        )
        nn_sel = torch.full((total_atoms,), 99, dtype=torch.int32, device=device)
        wp_nm_sel = wp.from_torch(nm_sel, dtype=wp.int32)
        wp_nn_sel = wp.from_torch(nn_sel, dtype=wp.int32)

        rebuild_flags = torch.ones(2, dtype=torch.bool, device=device)
        wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool)

        batch_naive_neighbor_matrix(
            wp_positions,
            cutoff,
            wp_batch_idx,
            wp_batch_ptr,
            wp_nm_sel,
            wp_nn_sel,
            wp_dtype,
            device,
            False,
            rebuild_flags=wp_rebuild_flags,
        )

        assert torch.equal(nn_sel, nn_ref), (
            "num_neighbors should match full rebuild when all flags=True"
        )
