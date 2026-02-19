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

"""Tests for PyTorch bindings of batched cell list neighbor construction methods."""

import pytest
import torch

from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors.batch_cell_list import (
    batch_build_cell_list,
    batch_cell_list,
    batch_query_cell_list,
    estimate_batch_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
)

from ...test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
    create_random_system,
    create_simple_cubic_system,
)
from .conftest import requires_vesin


class TestBatchCellListAPI:
    """Test the main batch cell list API functions."""

    @requires_vesin
    @pytest.mark.parametrize("cutoff", [1.0, 3.0])
    def test_single_system_single_atom(self, device, dtype, cutoff):
        """Test with single system containing single atom (should have no neighbors)."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        cell = (torch.eye(3, dtype=dtype, device=device) * 2.0).reshape(1, 3, 3)
        pbc = torch.tensor([[True, True, True]], device=device)
        batch_idx = torch.tensor([0], dtype=torch.int32, device=device)

        # Test batch_cell_list function
        neighbor_list, _, u = batch_cell_list(
            positions, cutoff, cell, pbc, batch_idx, return_neighbor_list=True
        )

        i, j = neighbor_list

        i_ref, j_ref, u_ref, _ = brute_force_neighbors(
            positions, cell, pbc.squeeze(0), cutoff
        )

        # Results should be identical
        assert_neighbor_lists_equal((i, j, u), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_single_system_two_atoms(self, device, dtype):
        """Test single system with two atoms."""
        # Two atoms within cutoff distance
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype, device=device
        )
        cell = (torch.eye(3, dtype=dtype, device=device) * 2.0).reshape(1, 3, 3)
        pbc = torch.tensor([[True, True, True]], device=device)
        batch_idx = torch.tensor([0, 0], dtype=torch.int32, device=device)
        cutoff = 1.0

        neighbor_list, _, u = batch_cell_list(
            positions, cutoff, cell, pbc, batch_idx, return_neighbor_list=True
        )
        i, j = neighbor_list

        # Should have 2 pairs: (0->1) and (1->0)
        assert len(i) == 2, f"Expected 2 neighbors, got {len(i)}"

        # Compare with brute force reference
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(
            positions, cell, pbc.squeeze(0), cutoff
        )
        assert_neighbor_lists_equal((i, j, u), (i_ref, j_ref, u_ref))

    def test_two_systems_same_structure(self, device, dtype):
        """Test batch with two identical systems."""
        # Create two identical cubic systems
        positions_1, cell_1, pbc_1 = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        positions_2 = positions_1.clone()

        # Concatenate for batch
        positions = torch.cat([positions_1, positions_2], dim=0)
        cell = torch.cat([cell_1, cell_1], dim=0)
        pbc = torch.cat([pbc_1, pbc_1], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(8, dtype=torch.int32, device=device),
                torch.ones(8, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 1.1

        # Test batch_cell_list
        _, neighbor_ptr, _ = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            max_neighbors=10,
            return_neighbor_list=True,
        )
        num_neighbors = neighbor_ptr[1:] - neighbor_ptr[:-1]
        # Each system should have the same number of neighbors
        num_neighbors_sys0 = num_neighbors[:8].sum().item()
        num_neighbors_sys1 = num_neighbors[8:].sum().item()
        assert num_neighbors_sys0 == num_neighbors_sys1, (
            f"Identical systems should have same neighbor counts: "
            f"{num_neighbors_sys0} vs {num_neighbors_sys1}"
        )

    def test_two_systems_different_structures(self, device, dtype):
        """Test batch with two different systems."""
        # System 1: 4 atoms
        positions_1 = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=dtype,
            device=device,
        )
        cell_1 = (torch.eye(3, dtype=dtype, device=device) * 3.0).reshape(1, 3, 3)
        pbc_1 = torch.tensor([[True, True, True]], device=device)

        # System 2: 3 atoms
        positions_2 = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.0, 0.8, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        cell_2 = (torch.eye(3, dtype=dtype, device=device) * 2.5).reshape(1, 3, 3)
        pbc_2 = torch.tensor([[True, True, True]], device=device)

        # Concatenate for batch
        positions = torch.cat([positions_1, positions_2], dim=0)
        cell = torch.cat([cell_1, cell_2], dim=0)
        pbc = torch.cat([pbc_1, pbc_2], dim=0)
        batch_idx = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device
        )
        cutoff = 1.5

        # Test batch_cell_list
        neighbor_list, _, _ = batch_cell_list(
            positions, cutoff, cell, pbc, batch_idx, return_neighbor_list=True
        )
        i, j = neighbor_list

        # Basic checks
        assert i.dtype == torch.int32
        assert j.dtype == torch.int32
        assert i.device.type == device.split(":")[0]

        # Verify neighbors are within their respective systems
        for atom_i, atom_j in zip(i.tolist(), j.tolist()):
            sys_i = batch_idx[atom_i].item()
            sys_j = batch_idx[atom_j].item()
            assert sys_i == sys_j, (
                f"Cross-system neighbors detected: atom {atom_i} (sys {sys_i}) "
                f"-> atom {atom_j} (sys {sys_j})"
            )

    def test_random_batch_systems(self, device, dtype):
        """Test with batch of random systems."""
        atoms_per_system = [10, 15, 12]
        cutoff = 5.0

        positions_list = []
        cells_list = []
        pbcs_list = []
        batch_idx_list = []

        for sys_idx, num_atoms in enumerate(atoms_per_system):
            pos, cell, pbc = create_random_system(
                num_atoms=num_atoms,
                cell_size=3.0,
                dtype=dtype,
                device=device,
                seed=42 + sys_idx,
                pbc_flag=True,
            )
            positions_list.append(pos)
            cells_list.append(cell)
            pbcs_list.append(pbc)
            batch_idx_list.append(
                torch.full((num_atoms,), sys_idx, dtype=torch.int32, device=device)
            )

        positions = torch.cat(positions_list, dim=0)
        cell = torch.cat(cells_list, dim=0)
        pbc = torch.cat(pbcs_list, dim=0)
        batch_idx = torch.cat(batch_idx_list, dim=0)

        max_neighbors = estimate_max_neighbors(
            cutoff, atomic_density=0.35, safety_factor=5.0
        )
        neighbor_list, _, u = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            max_neighbors=max_neighbors,
            return_neighbor_list=True,
        )
        i, j = neighbor_list

        # Basic checks
        assert i.dtype == torch.int32
        assert j.dtype == torch.int32
        assert u.dtype == torch.int32
        assert i.device.type == device.split(":")[0]

        # Check consistency: if (i,j) is a pair, j should be within cutoff of i
        if len(i) > 0:
            for idx in range(min(10, len(i))):
                atom_i, atom_j = i[idx].item(), j[idx].item()
                sys_idx = batch_idx[atom_i].item()
                shift = cell[sys_idx] @ u[idx].to(dtype)
                rij = positions[atom_j] - positions[atom_i] + shift
                dist = torch.norm(rij, dim=0).item()
                assert dist < cutoff + 1e-5, f"Distance {dist} exceeds cutoff {cutoff}"

    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_batch_no_pbc(self, device, dtype, return_neighbor_list):
        """Test batch with no periodic boundary conditions."""
        positions_1, cell_1, _ = create_simple_cubic_system(
            num_atoms=8, cell_size=3.0, dtype=dtype, device=device
        )
        positions_2 = positions_1.clone()

        positions = torch.cat([positions_1, positions_2], dim=0)
        cell = torch.cat([cell_1, cell_1], dim=0)
        pbc = torch.tensor(
            [[False, False, False], [False, False, False]], device=device
        )
        batch_idx = torch.cat(
            [
                torch.zeros(8, dtype=torch.int32, device=device),
                torch.ones(8, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 1.1

        results = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            return_neighbor_list=return_neighbor_list,
        )
        u = results[-1]

        # With no PBC, all shifts should be zero
        if len(u) > 0:
            assert torch.all(u == 0), "All shifts should be zero with no PBC"

    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    @pytest.mark.parametrize("preallocate", [True, False])
    @pytest.mark.parametrize("fill_value", [None, -1])
    def test_batch_mixed_pbc(
        self, device, dtype, return_neighbor_list, preallocate, fill_value
    ):
        """Test batch with mixed periodic boundary conditions."""
        positions_1, cell_1, _ = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        positions_2 = positions_1.clone()

        positions = torch.cat([positions_1, positions_2], dim=0)
        cell = torch.cat([cell_1, cell_1], dim=0)
        pbc = torch.tensor([[True, False, True], [False, True, False]], device=device)
        batch_idx = torch.cat(
            [
                torch.zeros(8, dtype=torch.int32, device=device),
                torch.ones(8, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 3.0

        if preallocate:
            max_neighbors = estimate_max_neighbors(
                cutoff, atomic_density=0.35, safety_factor=5.0
            )
            max_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
                cell, pbc, cutoff
            )
            (
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
            ) = allocate_cell_list(
                positions.shape[0], max_cells, neighbor_search_radius, device
            )
            fill_value = positions.shape[0] if fill_value is None else fill_value
            neighbor_matrix = torch.full(
                (positions.shape[0], max_neighbors),
                fill_value,
                dtype=torch.int32,
                device=device,
            )
            neighbor_matrix_shifts = torch.zeros(
                (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
            )
            num_neighbors = torch.zeros(
                (positions.shape[0],), dtype=torch.int32, device=device
            )

            results = batch_cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                batch_idx,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
                cells_per_dimension=cells_per_dimension,
                neighbor_search_radius=neighbor_search_radius,
                atom_periodic_shifts=atom_periodic_shifts,
                atom_to_cell_mapping=atom_to_cell_mapping,
                atoms_per_cell_count=atoms_per_cell_count,
                cell_atom_start_indices=cell_atom_start_indices,
                cell_atom_list=cell_atom_list,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                num_neighbors=num_neighbors,
            )
        else:
            max_neighbors = estimate_max_neighbors(
                cutoff, atomic_density=0.35, safety_factor=5.0
            )
            results = batch_cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                batch_idx,
                max_neighbors=max_neighbors,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
            )

        if return_neighbor_list:
            neighbor_list, _, u = results
            assert len(neighbor_list) == 2
            assert u[:, 2].sum().item() == 0
            assert (u[:, 0] ** 2).sum().item() > 0
        else:
            _, _, u = results
            assert u[:, :, 2].sum().item() == 0
            assert (u[:, :, 0] ** 2).sum().item() > 0

    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_batch_zero_cutoff(self, device, dtype, return_neighbor_list):
        """Test batch with zero cutoff (should find no neighbors)."""
        positions_1, cell_1, pbc_1 = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        positions = torch.cat([positions_1, positions_1], dim=0)
        cell = torch.cat([cell_1, cell_1], dim=0)
        pbc = torch.cat([pbc_1, pbc_1], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(8, dtype=torch.int32, device=device),
                torch.ones(8, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 0.0

        results = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            return_neighbor_list=return_neighbor_list,
        )
        if return_neighbor_list:
            assert len(results) == 3
            assert results[0].shape == (2, 0)  # neighbor_list
            assert results[1].shape == (17,)  # neighbor_ptr
            assert results[2].shape == (0, 3)  # shifts
        else:
            assert len(results) == 3
            assert results[0].shape[0] == 16
            assert results[-1].sum().item() == 0

    @pytest.mark.parametrize(
        "pbc_flags",
        [
            [[True, True, True], [True, True, True]],
            [[False, False, False], [False, False, False]],
            [[True, False, True], [False, True, False]],
        ],
    )
    @pytest.mark.parametrize("num_atoms", [10, 20])
    @pytest.mark.parametrize("cutoff", [1.0, 3.0])
    def test_batch_scaling_correctness(
        self, pbc_flags, dtype, device, num_atoms, cutoff
    ):
        """Test batch with various sizes and configurations."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions_list = []
        cells_list = []
        pbcs_list = []
        batch_idx_list = []

        for sys_idx, pbc_flag in enumerate(pbc_flags):
            pos, cell, pbc = create_random_system(
                num_atoms=num_atoms,
                cell_size=3.0,
                dtype=dtype,
                device=device,
                seed=42 + sys_idx,
                pbc_flag=pbc_flag,
            )
            positions_list.append(pos)
            cells_list.append(cell)
            pbcs_list.append(pbc)
            batch_idx_list.append(
                torch.full((num_atoms,), sys_idx, dtype=torch.int32, device=device)
            )

        positions = torch.cat(positions_list, dim=0)
        cell = torch.cat(cells_list, dim=0)
        pbc = torch.cat(pbcs_list, dim=0)
        batch_idx = torch.cat(batch_idx_list, dim=0)

        estimated_density = num_atoms / cell[0].det().abs().item()
        max_neighbors = estimate_max_neighbors(
            cutoff, atomic_density=estimated_density, safety_factor=5.0
        )
        neighbor_list, _, u = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            max_neighbors=max_neighbors,
            return_neighbor_list=True,
        )
        i, j = neighbor_list
        S = u.to(dtype)

        # Check consistency: if (i,j) is a pair, j should be within cutoff of i
        if len(i) > 0:
            for idx in range(min(10, len(i))):
                atom_i, atom_j = i[idx].item(), j[idx].item()
                sys_idx = batch_idx[atom_i].item()
                shift = S[idx] @ cell[sys_idx]
                rij = positions[atom_j] - positions[atom_i] + shift
                dist = torch.norm(rij, dim=0).item()
                assert dist < cutoff + 1e-5, f"Distance {dist} exceeds cutoff {cutoff}"


class TestBatchEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_estimate_batch_cell_list_sizes(self, device, dtype):
        """Test that estimate_batch_cell_list_sizes returns the correct values for an empty batch."""
        cell = torch.zeros((0, 3, 3), dtype=dtype, device=device)
        pbc = torch.zeros((0, 3), dtype=torch.bool, device=device)
        cutoff = 1.0
        max_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell, pbc, cutoff
        )
        assert max_cells == 1
        assert neighbor_search_radius.shape == (0, 3)
        assert neighbor_search_radius.dtype == torch.int32
        assert neighbor_search_radius.device == torch.device(device)

        # Now test with negative cutoff
        cell = torch.eye(3, dtype=dtype, device=device).reshape(1, 3, 3)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        cutoff = -1.0
        max_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell, pbc, cutoff
        )
        assert max_cells == 1
        assert neighbor_search_radius.shape == (1, 3)
        assert neighbor_search_radius.dtype == torch.int32
        assert neighbor_search_radius.device == torch.device(device)

    def test_all_negative_volume(self, device, dtype):
        """Check to make sure bindings raises error for all <= 0 volumes"""
        positions = torch.rand((4, 3), device=device, dtype=dtype)
        # cell has zero and negative volumes
        cells = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [
                    [0.2225, 0.6140, 0.7039],
                    [0.4351, 0.3592, 0.8304],
                    [0.1768, 0.0427, 0.3177],
                ],
            ],
            dtype=dtype,
            device=device,
        )
        pbc = torch.ones((4, 3), dtype=bool, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        with pytest.raises(RuntimeError, match="Cells with volume <= 0"):
            _ = batch_cell_list(positions, 3.0, cells, pbc, batch_idx)

    def test_mixed_negative_volume(self, device, dtype):
        """Check that samples with negative volume fail"""
        positions = torch.rand((4, 3), device=device, dtype=dtype)
        # first is positive, second is negative
        cells = torch.tensor(
            [
                [
                    [0.7340, 0.5755, 0.5256],
                    [0.3528, 0.1856, 0.9662],
                    [0.2384, 0.1754, 0.1968],
                ],
                [
                    [0.3681, 0.1729, 0.0691],
                    [0.7392, 0.7962, 0.9036],
                    [0.3154, 0.7710, 0.2854],
                ],
            ],
            dtype=dtype,
            device=device,
        )
        pbc = torch.ones((4, 3), dtype=bool, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        with pytest.raises(RuntimeError, match="Cells with volume <= 0"):
            _ = batch_cell_list(positions, 3.0, cells, pbc, batch_idx)

    def test_empty_batch_build_cell_list(self, device, dtype):
        """Test with empty batch."""
        positions = torch.empty(0, 3, dtype=dtype, device=device)
        cell = torch.eye(3, dtype=dtype, device=device).reshape(1, 3, 3)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        batch_idx = torch.empty(0, dtype=torch.int32, device=device)
        cutoff = 1.0
        cells_per_dimension = torch.tensor([1, 1, 1], dtype=torch.int32, device=device)
        neighbor_search_radius = torch.tensor(
            [1, 1, 1], dtype=torch.int32, device=device
        )
        atom_periodic_shifts = torch.tensor([0, 0, 0], dtype=torch.int32, device=device)
        atom_to_cell_mapping = torch.tensor([0, 0, 0], dtype=torch.int32, device=device)
        atoms_per_cell_count = torch.tensor([0], dtype=torch.int32, device=device)
        cell_atom_start_indices = torch.tensor([0], dtype=torch.int32, device=device)
        cell_atom_list = torch.tensor([], dtype=torch.int32, device=device)
        batch_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )

        assert torch.equal(
            atom_periodic_shifts,
            torch.tensor([0, 0, 0], dtype=torch.int32, device=device),
        )
        assert torch.equal(
            atom_to_cell_mapping,
            torch.tensor([0, 0, 0], dtype=torch.int32, device=device),
        )
        assert torch.equal(
            atoms_per_cell_count, torch.tensor([0], dtype=torch.int32, device=device)
        )
        assert torch.equal(
            cell_atom_start_indices, torch.tensor([0], dtype=torch.int32, device=device)
        )
        assert torch.equal(
            cell_atom_list, torch.tensor([], dtype=torch.int32, device=device)
        )

        # Now test with negative cutoff
        cutoff = -1.0
        batch_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )
        assert torch.equal(
            atom_periodic_shifts,
            torch.tensor([0, 0, 0], dtype=torch.int32, device=device),
        )
        assert torch.equal(
            atom_to_cell_mapping,
            torch.tensor([0, 0, 0], dtype=torch.int32, device=device),
        )
        assert torch.equal(
            atoms_per_cell_count, torch.tensor([0], dtype=torch.int32, device=device)
        )
        assert torch.equal(
            cell_atom_start_indices, torch.tensor([0], dtype=torch.int32, device=device)
        )
        assert torch.equal(
            cell_atom_list, torch.tensor([], dtype=torch.int32, device=device)
        )

    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_empty_batch(self, return_neighbor_list):
        """Test with empty batch."""
        positions = torch.empty(0, 3, dtype=torch.float32)
        cell = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)
        pbc = torch.tensor([[True, True, True]])
        batch_idx = torch.empty(0, dtype=torch.int32)
        cutoff = 1.0

        results = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            return_neighbor_list=return_neighbor_list,
        )
        if return_neighbor_list:
            assert len(results) == 3
            assert results[0].shape == (2, 0)  # neighbor_list
            assert results[1].shape == (1,)  # neighbor_ptr
            assert results[2].shape == (0, 3)  # shifts
        else:
            assert len(results) == 3
            assert results[0].shape[0] == 0  # neighbor_matrix
            assert results[1].shape[0] == 0  # num_neighbors
            assert results[2].shape[0] == 0  # neighbor_matrix_shifts
            assert results[2].shape[2] == 3
            assert results[1].shape == (0,)

    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_batch_dtype_consistency(self, dtype, return_neighbor_list):
        """Test that output dtypes are consistent with inputs."""
        positions = torch.randn(10, 3, dtype=dtype)
        cell = (torch.eye(3, dtype=dtype) * 2.0).reshape(1, 3, 3).repeat(2, 1, 1)
        pbc = torch.tensor([[True, True, True], [True, True, True]], dtype=torch.bool)
        batch_idx = torch.cat(
            [torch.zeros(5, dtype=torch.int32), torch.ones(5, dtype=torch.int32)]
        )
        cutoff = 1.5

        results = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            return_neighbor_list=return_neighbor_list,
        )

        for result in results:
            assert result.dtype == torch.int32

    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_batch_device_consistency(self, device, return_neighbor_list):
        """Test that outputs are on the same device as inputs."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions = torch.randn(10, 3, device=device)
        cell = torch.eye(3, device=device).reshape(1, 3, 3).repeat(2, 1, 1) * 2.0
        pbc = torch.tensor([[True, True, True], [True, True, True]], device=device)
        batch_idx = torch.cat(
            [
                torch.zeros(5, dtype=torch.int32, device=device),
                torch.ones(5, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 1.5

        results = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            return_neighbor_list=return_neighbor_list,
        )
        for result in results:
            assert result.device == torch.device(device)


class TestBatchCellListComponentsAPI:
    """Test the modular batch cell list API functions."""

    def test_batch_build_and_query_cell_list(self, device, dtype):
        """Test building and querying batch cell list separately."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create batch with 2 systems
        positions_1, cell_1, pbc_1 = create_simple_cubic_system(
            dtype=dtype, device=device
        )
        positions_2 = positions_1.clone()

        positions = torch.cat([positions_1, positions_2], dim=0)
        cell = torch.cat([cell_1, cell_1], dim=0)
        pbc = torch.cat([pbc_1, pbc_1], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(8, dtype=torch.int32, device=device),
                torch.ones(8, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 1.1

        # Get size estimates for batch_build_cell_list
        max_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell,
            pbc,
            cutoff,
        )
        max_neighbors = estimate_max_neighbors(cutoff)

        total_atoms = positions.shape[0]

        # Allocate memory for the cell list
        cell_list_cache = allocate_cell_list(
            total_atoms, max_cells, neighbor_search_radius, device
        )

        # Build cell list
        batch_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            *cell_list_cache,
        )

        assert cell_list_cache[0] is not None
        assert cell_list_cache[0].device == torch.device(device)
        assert cell_list_cache[0].dtype == torch.int32
        assert cell_list_cache[0].shape == (2, 3)  # 2 systems, 3 dimensions

        # Query using the cell list
        assert max_neighbors > 0
        neighbor_matrix = torch.full(
            (total_atoms, max_neighbors),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts = torch.zeros(
            (total_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros((total_atoms,), dtype=torch.int32, device=device)
        batch_query_cell_list(
            positions,
            cell,
            pbc,
            cutoff,
            batch_idx,
            *cell_list_cache,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            False,
        )
        assert neighbor_matrix is not None
        assert neighbor_matrix.device == torch.device(device)
        assert neighbor_matrix.dtype == torch.int32
        assert neighbor_matrix.shape == (total_atoms, max_neighbors)
        assert neighbor_matrix_shifts is not None
        assert neighbor_matrix_shifts.device == torch.device(device)
        assert neighbor_matrix_shifts.dtype == torch.int32
        assert neighbor_matrix_shifts.shape == (total_atoms, max_neighbors, 3)
        assert num_neighbors is not None
        assert num_neighbors.device == torch.device(device)
        assert num_neighbors.dtype == torch.int32
        assert num_neighbors.shape == (total_atoms,)

        # Check that we have some neighbors (cubic system should have many)
        valid_neighbors = (neighbor_matrix >= 0).sum()
        assert valid_neighbors > 0

        # Check that the neighbor matrix is correct
        for i in range(total_atoms):
            row_mask = neighbor_matrix[i] >= 0
            assert row_mask.sum() == num_neighbors[i].item()


class TestBatchTorchCompilability:
    """Test torch.compile compatibility for core batch functions."""

    def test_batch_build_cell_list_compile(self, device, dtype):
        """Test that batch_build_cell_list can be compiled with torch.compile."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions_1, cell_1, pbc_1 = create_simple_cubic_system(
            dtype=dtype, device=device
        )
        positions = torch.cat([positions_1, positions_1], dim=0)
        cell = torch.cat([cell_1, cell_1], dim=0)
        pbc = torch.cat([pbc_1, pbc_1], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(8, dtype=torch.int32, device=device),
                torch.ones(8, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 1.1

        # Get size estimates
        max_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell,
            pbc,
            cutoff,
        )

        # Test uncompiled version
        clcu = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius,
            device,
        )
        batch_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            *clcu,
        )

        # Test compiled version
        clcc = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius.clone(),
            device,
        )

        @torch.compile
        def compiled_batch_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ):
            batch_build_cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                batch_idx,
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
            )

        compiled_batch_build_cell_list(positions, cutoff, cell, pbc, batch_idx, *clcc)

        # Compare results (cell_list_cache includes cell_offsets at index [2])
        all_tensors_u = clcu
        all_tensors_c = clcc
        for i, (tensor_uncompiled, tensor_compiled) in enumerate(
            zip(all_tensors_u, all_tensors_c)
        ):
            assert tensor_uncompiled.shape == tensor_compiled.shape, (
                f"Shape mismatch in tensor {i}: {tensor_uncompiled.shape} vs {tensor_compiled.shape}"
            )
            assert tensor_uncompiled.dtype == tensor_compiled.dtype, (
                f"Dtype mismatch in tensor {i}: {tensor_uncompiled.dtype} vs {tensor_compiled.dtype}"
            )
            assert tensor_uncompiled.device == tensor_compiled.device, (
                f"Device mismatch in tensor {i}: {tensor_uncompiled.device} vs {tensor_compiled.device}"
            )
            # For integer tensors, check exact equality
            if tensor_uncompiled.dtype in [torch.int32, torch.int64]:
                assert torch.equal(tensor_uncompiled, tensor_compiled), (
                    f"Value mismatch in tensor {i}"
                )
            else:
                # For float tensors, use tolerance
                assert torch.allclose(
                    tensor_uncompiled,
                    tensor_compiled,
                    rtol=1e-5,
                    atol=1e-6,
                ), f"Value mismatch in tensor {i}"

    @pytest.mark.parametrize("pbc_flag", [False, True])
    def test_batch_query_cell_list_compile(self, device, dtype, pbc_flag):
        """Test that batch_query_cell_list can be compiled with torch.compile."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions_1, cell_1, _ = create_simple_cubic_system(dtype=dtype, device=device)
        positions = torch.cat([positions_1, positions_1], dim=0)
        cell = torch.cat([cell_1, cell_1], dim=0)
        pbc = torch.tensor(
            [[pbc_flag, pbc_flag, pbc_flag], [pbc_flag, pbc_flag, pbc_flag]],
            device=device,
        )
        batch_idx = torch.cat(
            [
                torch.zeros(8, dtype=torch.int32, device=device),
                torch.ones(8, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 3.0

        # Build cell list first
        max_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell,
            pbc,
            cutoff,
        )
        max_neighbors = estimate_max_neighbors(cutoff)

        cell_list_cache_uncompiled = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius,
            device,
        )
        batch_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            *cell_list_cache_uncompiled,
        )

        # Query cell list
        neighbor_matrix_uncompiled = torch.full(
            (positions.shape[0], max_neighbors),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts_uncompiled = torch.zeros(
            (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors_uncompiled = torch.zeros(
            (positions.shape[0],), dtype=torch.int32, device=device
        )

        # Test uncompiled version
        batch_query_cell_list(
            positions,
            cell,
            pbc,
            cutoff,
            batch_idx,
            *cell_list_cache_uncompiled,
            neighbor_matrix_uncompiled,
            neighbor_matrix_shifts_uncompiled,
            num_neighbors_uncompiled,
            False,
        )

        # Test compiled version
        cell_list_cache_compiled = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius.clone(),
            device,
        )
        neighbor_matrix_compiled = torch.full(
            (positions.shape[0], max_neighbors),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts_compiled = torch.zeros(
            (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors_compiled = torch.zeros(
            (positions.shape[0],), dtype=torch.int32, device=device
        )

        @torch.compile
        def compiled_query_cell_list(
            positions,
            cell,
            pbc,
            cutoff,
            batch_idx,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
        ):
            batch_build_cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                batch_idx,
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
            )
            batch_query_cell_list(
                positions,
                cell,
                pbc,
                cutoff,
                batch_idx,
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                False,
            )

        compiled_query_cell_list(
            positions,
            cell,
            pbc,
            cutoff,
            batch_idx,
            *cell_list_cache_compiled,
            neighbor_matrix_compiled,
            neighbor_matrix_shifts_compiled,
            num_neighbors_compiled,
        )

        # Compare results
        for row_idx, (unc_row, cmp_row) in enumerate(
            zip(neighbor_matrix_uncompiled, neighbor_matrix_compiled)
        ):
            unc_row_sorted, indices_uncompiled = torch.sort(unc_row)
            cmp_row_sorted, indices_compiled = torch.sort(cmp_row)
            assert torch.equal(unc_row_sorted, cmp_row_sorted), (
                f"Neighbor matrix mismatch for row {row_idx}"
            )
            assert torch.equal(indices_uncompiled, indices_compiled), (
                f"Indices mismatch for row {row_idx}"
            )

            assert torch.equal(
                neighbor_matrix_shifts_uncompiled[row_idx, indices_uncompiled, 0],
                neighbor_matrix_shifts_compiled[row_idx, indices_compiled, 0],
            ), f"Neighbor matrix shifts mismatch for row {row_idx}"
            assert torch.equal(
                neighbor_matrix_shifts_uncompiled[row_idx, indices_uncompiled, 1],
                neighbor_matrix_shifts_compiled[row_idx, indices_compiled, 1],
            ), f"Neighbor matrix shifts mismatch for row {row_idx}"
            assert torch.equal(
                neighbor_matrix_shifts_uncompiled[row_idx, indices_uncompiled, 2],
                neighbor_matrix_shifts_compiled[row_idx, indices_compiled, 2],
            ), f"Neighbor matrix shifts mismatch for row {row_idx}"
        assert torch.equal(num_neighbors_uncompiled, num_neighbors_compiled), (
            "Number of neighbors mismatch"
        )
