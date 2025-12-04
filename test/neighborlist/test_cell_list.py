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

"""Tests for nvalchemiops neighbor list main API functions."""

from importlib import import_module

import pytest
import torch

from nvalchemiops.neighborlist.cell_list import (
    allocate_cell_list,
    build_cell_list,
    cell_list,
    estimate_cell_list_sizes,
    query_cell_list,
)
from nvalchemiops.neighborlist.neighbor_utils import estimate_max_neighbors

from .test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
    create_nonorthorhombic_system,
    create_random_system,
    create_simple_cubic_system,
)

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda:0")
dtypes = [torch.float32, torch.float64]

try:
    _ = import_module("vesin")
    run_vesin_checks = True
except ModuleNotFoundError:
    run_vesin_checks = False


class TestCellListAPI:
    """Test the main cell list API functions."""

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_single_atom_system(self, device, dtype):
        """Test with single atom (should have no neighbors)."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        cell = (torch.eye(3, dtype=dtype, device=device) * 2.0).reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], device=device)
        cutoff = 3.0

        neighbor_list, _, u = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )
        i, j = neighbor_list
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)

        # Results should be identical
        assert_neighbor_lists_equal((i, j, u), (i_ref, j_ref, u_ref))

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_two_atom_system(self, device, dtype):
        """Test simple two-atom system."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype, device=device
        )
        cell = (torch.eye(3, dtype=dtype, device=device) * 2.0).reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], device=device)
        cutoff = 1.0

        neighbor_list, _, u = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )
        i, j = neighbor_list
        assert len(i) == 2, f"Expected 2 neighbors, got {len(i)}"

        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i, j, u), (i_ref, j_ref, u_ref))

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_cubic_system(self, device, dtype):
        """Test with simple cubic lattice."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        cutoff = 1.1  # Should capture nearest neighbors

        neighbor_list, _, u = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )
        i, j = neighbor_list

        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i, j, u), (i_ref, j_ref, u_ref))

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("pbc_flag", [True, False])
    def test_random_system(self, device, dtype, pbc_flag):
        """Test with random atomic positions."""
        positions, cell, pbc = create_random_system(
            num_atoms=20,
            cell_size=10.0,
            dtype=dtype,
            device=device,
            seed=42,
            pbc_flag=pbc_flag,
        )
        cutoff = 5.0

        neighbor_list, _, u = cell_list(
            positions, cutoff, cell, pbc, max_neighbors=1500, return_neighbor_list=True
        )
        i, j = neighbor_list
        ref_i, ref_j, ref_u, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i, j, u), (ref_i, ref_j, ref_u))

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_no_pbc(self, device, dtype, return_neighbor_list):
        """Test with no periodic boundary conditions."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=3.0, dtype=dtype, device=device
        )
        pbc = torch.tensor([False, False, False], device=device)
        cutoff = 3.0

        results = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            return_neighbor_list=return_neighbor_list,
        )
        u = results[-1]

        # With no PBC, all shifts should be zero
        if len(u) > 0:
            assert torch.all(u == 0), "All shifts should be zero with no PBC"

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    @pytest.mark.parametrize("preallocate", [True, False])
    @pytest.mark.parametrize("fill_value", [None, -1])
    @pytest.mark.parametrize("cell_pbc_shape", [0, 1])
    def test_mixed_pbc(
        self,
        device,
        dtype,
        return_neighbor_list,
        preallocate,
        fill_value,
        cell_pbc_shape,
    ):
        """Test with mixed periodic boundary conditions."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        cutoff = 3.0

        if cell_pbc_shape == 0:
            cell = cell.reshape(3, 3)
            pbc = pbc.reshape(3)
        else:
            cell = cell.reshape(1, 3, 3)
            pbc = pbc.reshape(1, 3)

        if preallocate:
            max_neighbors = estimate_max_neighbors(cutoff)
            max_cells, neighbor_search_radius = estimate_cell_list_sizes(
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

            results = cell_list(
                positions,
                cutoff,
                cell,
                pbc,
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
            results = cell_list(
                positions,
                cutoff,
                cell,
                pbc,
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

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_large_cutoff(self, device, dtype, return_neighbor_list):
        """Test with large cutoff that includes many neighbors."""
        positions, cell, pbc = create_random_system(
            num_atoms=10, cell_size=2.0, dtype=dtype, device=device, seed=123
        )
        cutoff = 5.0  # Large cutoff

        results = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            return_neighbor_list=return_neighbor_list,
        )
        if return_neighbor_list:
            num_pairs = results[0].shape[1]
        else:
            num_pairs = results[1].sum().item()
        assert num_pairs >= 0

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_zero_cutoff(self, device, dtype, return_neighbor_list):
        """Test with zero cutoff (should find no neighbors)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 0.0

        results = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            return_neighbor_list=return_neighbor_list,
        )
        if return_neighbor_list:
            assert len(results) == 3
            assert results[0].shape == (2, 0)  # neighbor_list
            assert results[1].shape == (9,)  # neighbor_ptr
            assert results[2].shape == (0, 3)  # shifts
        else:
            assert len(results) == 3
            assert results[0].shape[0] == 8
            assert results[1].sum().item() == 0

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize(
        "pbc_flag",
        [
            [True, True, True],
            [False, False, False],
            [True, False, True],
            [False, False, True],
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("num_atoms", [10, 20, 50, 100])
    @pytest.mark.parametrize("cutoff", [1.0, 3.0, 5.0])
    @pytest.mark.parametrize("system_type", ["random", "nonorthorhombic"])
    def test_scaling_correctness(
        self, pbc_flag, dtype, device, num_atoms, cutoff, system_type
    ):
        """Test with random atomic positions."""
        if system_type == "random":
            positions, cell, pbc = create_random_system(
                num_atoms=num_atoms,
                cell_size=3.0,
                dtype=dtype,
                device=device,
                seed=42,
                pbc_flag=pbc_flag,
            )

        else:
            positions, cell, pbc = create_nonorthorhombic_system(
                num_atoms=num_atoms,
                a=8.57,
                b=12.9645,
                c=7.2203,
                alpha=90.74,
                beta=115.944,
                gamma=87.663,
                dtype=dtype,
                device=device,
                seed=42,
                pbc_flag=pbc_flag,
            )
            scale_factor = (1.0 / 720.88) ** (1.0 / 3.0)
            cell = cell * scale_factor
            positions = positions * scale_factor

        estimated_density = num_atoms / cell.det().abs().item()
        max_neighbors = estimate_max_neighbors(
            cutoff, atomic_density=estimated_density, safety_factor=5.0
        )
        neighbor_list, _, u = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            max_neighbors=max_neighbors,
            return_neighbor_list=True,
        )
        i, j = neighbor_list
        # Basic checks
        ref_i, ref_j, ref_u, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i, j, u), (ref_i, ref_j, ref_u))

        # Check consistency: if (i,j) is a pair, j should be within cutoff of i
        if len(i) > 0:
            for idx in range(min(10, len(i))):  # Check first 10 pairs
                atom_i, atom_j = i[idx].item(), j[idx].item()
                shift = cell.squeeze(0) @ u[idx].to(dtype)
                rij = positions[atom_j] - positions[atom_i] + shift
                dist = torch.norm(rij, dim=0).item()
                assert dist < cutoff + 1e-5, f"Distance {dist} exceeds cutoff {cutoff}"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_empty_estimate_cell_list_sizes(self, device, dtype):
        """Test that estimate_batch_cell_list_sizes returns the correct values for an empty batch."""
        cell = torch.zeros((0, 3, 3), dtype=dtype, device=device)
        pbc = torch.zeros((0, 3), dtype=torch.bool, device=device)
        cutoff = 1.0
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
        assert max_cells == 1
        assert neighbor_search_radius.shape == (3,)
        assert neighbor_search_radius.dtype == torch.int32
        assert neighbor_search_radius.device == torch.device(device)

        # Now test with negative cutoff
        cell = torch.eye(3, dtype=dtype, device=device).reshape(1, 3, 3)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        cutoff = -1.0
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
        assert max_cells == 1
        assert neighbor_search_radius.shape == (3,)
        assert neighbor_search_radius.dtype == torch.int32
        assert neighbor_search_radius.device == torch.device(device)

    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_empty_system(self, return_neighbor_list):
        """Test with empty coordinate array."""
        positions = torch.empty(0, 3, dtype=torch.float32)
        cell = torch.eye(3, dtype=torch.float32)
        pbc = torch.tensor([True, True, True])
        cutoff = 1.0

        results = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=return_neighbor_list
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

    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_dtype_consistency(self, dtype, return_neighbor_list):
        """Test that output dtypes are consistent with inputs."""
        positions = torch.randn(5, 3, dtype=dtype)
        cell = (torch.eye(3, dtype=dtype) * 2.0).reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], dtype=torch.bool)
        cutoff = 1.5

        results = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=return_neighbor_list
        )

        for result in results:
            assert result.dtype == torch.int32

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_device_consistency(self, device, return_neighbor_list):
        """Test that outputs are on the same device as inputs."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions = torch.randn(5, 3, device=device)
        cell = torch.eye(3, device=device).reshape(1, 3, 3) * 2.0
        pbc = torch.tensor([True, True, True], device=device)
        cutoff = 1.5

        results = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=return_neighbor_list
        )
        for result in results:
            assert result.device == torch.device(device)


class TestCellListComponentsAPI:
    """Test the new modular cell list API functions."""

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_build_and_query_cell_list(self, device, dtype):
        """Test building and querying cell list separately."""
        positions, cell, pbc = create_simple_cubic_system(dtype=dtype, device=device)
        cutoff = 1.1
        pbc = pbc.squeeze(0)
        # Get size estimates for build_cell_list
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(
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
        build_cell_list(positions, cutoff, cell, pbc, *cell_list_cache)

        assert cell_list_cache[0] is not None
        assert cell_list_cache[0].device == torch.device(device)
        assert cell_list_cache[0].dtype == torch.int32
        assert cell_list_cache[0].shape == (3,)

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
        query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            *cell_list_cache,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
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

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_estimate_max_neighbors(self, device, dtype):
        """Test max_neighbors estimation."""
        positions, cell, _ = create_simple_cubic_system(dtype=dtype, device=device)
        cutoff = 1.1
        density = positions.shape[0] / cell.det().abs().item()
        max_neighbors = estimate_max_neighbors(
            cutoff, atomic_density=density, safety_factor=5.0
        )
        assert max_neighbors > 0
        assert isinstance(max_neighbors, int)


class TestTorchCompilability:
    """Test torch.compile compatibility for core functions."""

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_build_cell_list_compile(self, device, dtype):
        """Test that build_cell_list can be compiled with torch.compile."""
        positions, cell, pbc = create_simple_cubic_system(dtype=dtype, device=device)
        cutoff = 1.1
        pbc = pbc.squeeze(0)

        # Get size estimates
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(
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
        build_cell_list(positions, cutoff, cell, pbc, *clcu)

        # Test compiled version
        clcc = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius,
            device,
        )

        @torch.compile
        def compiled_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ):
            build_cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
            )

        compiled_build_cell_list(positions, cutoff, cell, pbc, *clcc)

        # Compare results
        for i, (tensor_uncompiled, tensor_compiled) in enumerate(zip(clcu, clcc)):
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

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("pbc_flag", [False, True])
    def test_query_cell_list_compile(self, device, dtype, pbc_flag):
        """Test that query_cell_list can be compiled with torch.compile."""
        positions, cell, pbc = create_simple_cubic_system(dtype=dtype, device=device)
        cutoff = 3.0
        pbc = torch.tensor([pbc_flag, pbc_flag, pbc_flag], device=device)
        # Build cell list first
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell,
            pbc,
            cutoff,
        )
        max_neighbors = estimate_max_neighbors(
            cutoff,
        )
        cell_list_cache_uncompiled = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius,
            device,
        )
        build_cell_list(positions, cutoff, cell, pbc, *cell_list_cache_uncompiled)

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
        query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            *cell_list_cache_uncompiled,
            neighbor_matrix_uncompiled,
            neighbor_matrix_shifts_uncompiled,
            num_neighbors_uncompiled,
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
            cutoff,
            cell,
            pbc,
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
            build_cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
            )
            query_cell_list(
                positions,
                cutoff,
                cell,
                pbc,
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
            )

        compiled_query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
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
