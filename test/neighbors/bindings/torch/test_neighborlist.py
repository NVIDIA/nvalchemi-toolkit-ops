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

"""API tests for the generic neighbor_list wrapper function."""

import pytest
import torch

from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.batch_cell_list import (
    batch_cell_list,
)
from nvalchemiops.torch.neighbors.batch_naive import (
    batch_naive_neighbor_list,
)
from nvalchemiops.torch.neighbors.batch_naive_dual_cutoff import (
    batch_naive_neighbor_list_dual_cutoff,
)
from nvalchemiops.torch.neighbors.cell_list import (
    cell_list,
)
from nvalchemiops.torch.neighbors.naive import (
    naive_neighbor_list,
)
from nvalchemiops.torch.neighbors.naive_dual_cutoff import (
    naive_neighbor_list_dual_cutoff,
)
from nvalchemiops.torch.neighbors.neighbor_utils import prepare_batch_idx_ptr

from ...test_utils import (
    assert_neighbor_matrix_equal,
    create_random_system,
)


class TestNeighborListAutoSelection:
    """Test automatic method selection based on system size."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_auto_select_naive_small_system(self, dtype, device):
        """Auto-select naive for small systems (< 2000 atoms)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Small system: 100 atoms
        target_density = 0.25
        num_atoms = 100
        volume = num_atoms / target_density
        box_size = volume ** (1 / 3)
        positions = torch.rand(num_atoms, 3, dtype=dtype, device=device) * box_size
        cutoff = 2.0

        # Call wrapper with no method specified
        result = neighbor_list(positions, cutoff, return_neighbor_list=True)

        # Should auto-select "naive" and work correctly
        assert len(result) == 2  # No PBC, so includes neighbor_ptr but no shifts
        neighbor_list_result, neighbor_ptr = result
        assert neighbor_list_result.shape[0] == 2  # COO format
        assert neighbor_ptr.shape[0] == 101
        assert neighbor_ptr[0] == 0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_auto_select_naive_with_pbc(self, dtype, device):
        """Auto-select naive for small systems with PBC."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions, cell, pbc = create_random_system(
            100, 10.0, dtype=dtype, device=device
        )
        cutoff = 2.0

        # Call wrapper with no method specified but with cell and pbc
        result = neighbor_list(
            positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=True
        )

        # Should auto-select "naive" and include shifts
        assert len(result) == 3  # With PBC, includes neighbor_ptr and shifts
        neighbor_list_result, neighbor_ptr, shifts = result
        assert neighbor_list_result.shape[0] == 2
        assert neighbor_ptr.shape[0] == 101
        assert neighbor_ptr[0] == 0
        assert shifts.shape[1] == 3

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_auto_select_cell_list_large_system(self, dtype, device):
        """Auto-select cell_list for large systems (>= 2000 avg atoms per system)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Large system: 2000 atoms
        positions = torch.randn(2000, 3, dtype=dtype, device=device) * 50.0
        cutoff = 2.0

        # Call wrapper with no method specified
        # Should auto-create cell and pbc
        result = neighbor_list(positions, cutoff, return_neighbor_list=True)

        # Should auto-select "cell_list" and work correctly
        assert (
            len(result) == 3
        )  # With PBC (auto-created), so includes neighbor_ptr and shifts
        neighbor_list_result, neighbor_ptr, shifts = result
        assert neighbor_list_result.shape[0] == 2  # COO format
        assert neighbor_ptr.shape[0] == 2001
        assert neighbor_ptr[0] == 0
        assert shifts.shape[1] == 3  # 3D shifts

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_auto_select_naive_dual_cutoff(self, dtype, device):
        """Auto-select naive_dual_cutoff when cutoff2 is provided."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions, cell, pbc = create_random_system(
            100, 10.0, dtype=dtype, device=device
        )
        cutoff1 = 2.5
        cutoff2 = 3.5

        # Call wrapper with cutoff2 but no method specified
        # Need to provide max_neighbors1 for naive_dual_cutoff
        result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            cutoff2=cutoff2,
            max_neighbors1=50,
            max_neighbors2=50,
            return_neighbor_list=True,
        )

        # Should auto-select "naive_dual_cutoff"
        # Returns 8 outputs: nlist1, ptr1, shifts1, nlist2, ptr2, shifts2
        assert len(result) == 6
        nlist1, ptr1, shifts1, nlist2, ptr2, shifts2 = result
        assert nlist1.shape[0] == 2
        assert nlist2.shape[0] == 2
        assert ptr1.shape[0] == 101
        assert ptr2.shape[0] == 101
        assert shifts1.shape[1] == 3
        assert shifts2.shape[1] == 3

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_auto_select_batch_naive(self, dtype, device):
        """Auto-select batch_naive when batch_idx is provided for small system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create batch of small systems
        positions1, cell1, pbc1 = create_random_system(
            50, 10.0, dtype=dtype, device=device
        )
        positions2, cell2, pbc2 = create_random_system(
            30, 10.0, dtype=dtype, device=device
        )
        cutoff = 2.0
        positions = torch.cat([positions1, positions2], dim=0)
        cell = torch.stack([cell1.squeeze(0), cell2.squeeze(0)], dim=0)
        pbc = torch.stack([pbc1.squeeze(0), pbc2.squeeze(0)], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(50, dtype=torch.int32, device=device),
                torch.ones(30, dtype=torch.int32, device=device),
            ]
        )
        batch_ptr = torch.tensor([0, 50, 80], dtype=torch.int32, device=device)

        # Call wrapper with batch_idx but no method specified
        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
        )

        # Should auto-select "batch_naive"
        assert len(result) == 3
        nlist, neighbor_ptr, _ = result
        assert nlist.shape[0] == 2
        assert neighbor_ptr.shape[0] == 81
        assert neighbor_ptr[0] == 0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Requires GPU."
                ),
            ),
        ],
    )
    def test_auto_select_batch_cell_list(self, dtype, device):
        """Auto-select batch_cell_list when batch_idx is provided for large system."""

        # Create batch with avg >= 2000 atoms per system
        positions1 = torch.randn(2500, 3, dtype=dtype, device=device) * 50.0
        positions2 = torch.randn(2500, 3, dtype=dtype, device=device) * 50.0

        positions = torch.cat([positions1, positions2], dim=0).to(device=device)
        cell = (
            torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(2, 1, 1) * 60.0
        )
        pbc = torch.tensor([[True, True, True], [True, True, True]], device=device)
        batch_idx = torch.cat(
            [
                torch.zeros(2500, dtype=torch.int32, device=device),
                torch.ones(2500, dtype=torch.int32, device=device),
            ]
        ).to(device=device)
        batch_ptr = torch.tensor([0, 2500, 5000], dtype=torch.int32, device=device)
        cutoff = 2.0

        # Call wrapper with batch_idx but no method specified
        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
        )

        # Should auto-select "batch_cell_list"
        assert len(result) == 3
        nlist, neighbor_ptr, _ = result
        assert nlist.shape[0] == 2
        assert neighbor_ptr.shape[0] == 5001
        assert neighbor_ptr[0] == 0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_auto_select_batch_naive_dual_cutoff(self, dtype, device):
        """Auto-select batch_naive_dual_cutoff when both cutoff2 and batch_idx are provided."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create batch of small systems
        positions1, cell1, pbc1 = create_random_system(
            50, 10.0, dtype=dtype, device=device
        )
        positions2, cell2, pbc2 = create_random_system(
            30, 10.0, dtype=dtype, device=device
        )

        positions = torch.cat([positions1, positions2], dim=0)
        cell = torch.stack([cell1.squeeze(0), cell2.squeeze(0)], dim=0)
        pbc = torch.stack([pbc1.squeeze(0), pbc2.squeeze(0)], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(50, dtype=torch.int32, device=device),
                torch.ones(30, dtype=torch.int32, device=device),
            ]
        )
        batch_ptr = torch.tensor([0, 50, 80], dtype=torch.int32, device=device)
        cutoff1 = 2.5
        cutoff2 = 3.5

        # Call wrapper with cutoff2 and batch_idx but no method specified
        # Need to provide max_neighbors1 for batch_naive_dual_cutoff
        result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cutoff2=cutoff2,
            max_neighbors1=50,
            max_neighbors2=50,
            return_neighbor_list=True,
        )

        # Should auto-select "batch_naive_dual_cutoff"
        assert len(result) == 6
        nlist1, ptr1, shifts1, nlist2, ptr2, shifts2 = result
        assert nlist1.shape[0] == 2
        assert nlist2.shape[0] == 2
        assert ptr1.shape[0] == 81
        assert ptr2.shape[0] == 81
        assert shifts1.shape[1] == 3
        assert shifts2.shape[1] == 3


class TestNeighborListExplicitMethod:
    """Test explicit method selection."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_explicit_naive(self, dtype, device):
        """Test explicit naive method selection."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions, cell, pbc = create_random_system(
            100, 10.0, dtype=dtype, device=device
        )
        cutoff = 2.0

        # Call wrapper with explicit method
        wrapper_result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            return_neighbor_list=False,
        )

        # Call naive directly
        direct_result = naive_neighbor_list(
            positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=False
        )

        # Results should match
        assert len(wrapper_result) == len(direct_result)
        assert_neighbor_matrix_equal(wrapper_result, direct_result)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_explicit_cell_list(self, dtype, device):
        """Test explicit cell_list method selection."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions, cell, pbc = create_random_system(
            500, 20.0, dtype=dtype, device=device
        )
        cutoff = 2.0

        # Call wrapper with explicit method
        wrapper_result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="cell_list",
            return_neighbor_list=False,
        )

        # Call cell_list directly
        direct_result = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=False
        )

        # Results should match
        assert len(wrapper_result) == len(direct_result)
        assert_neighbor_matrix_equal(wrapper_result, direct_result)


class TestNeighborListBatchProcessing:
    """Test batch processing with batch_idx."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_batch_naive(self, dtype, device):
        """Test batch naive method."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create two small systems
        positions1, cell1, pbc1 = create_random_system(
            50, 10.0, dtype=dtype, device=device
        )
        positions2, cell2, pbc2 = create_random_system(
            30, 10.0, dtype=dtype, device=device
        )
        cutoff = 2.0

        # Combine into batch
        positions = torch.cat([positions1, positions2], dim=0)
        cell = torch.stack([cell1.squeeze(0), cell2.squeeze(0)], dim=0)
        pbc = torch.stack([pbc1.squeeze(0), pbc2.squeeze(0)], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(50, dtype=torch.int32, device=device),
                torch.ones(30, dtype=torch.int32, device=device),
            ]
        )
        batch_ptr = torch.tensor([0, 50, 80], dtype=torch.int32, device=device)

        # Call wrapper with explicit batch_naive method
        wrapper_result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_naive",
            return_neighbor_list=False,
        )

        # Call batch_naive directly
        direct_result = batch_naive_neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            return_neighbor_list=False,
        )

        # Results should match
        assert len(wrapper_result) == len(direct_result)
        assert_neighbor_matrix_equal(wrapper_result, direct_result)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Requires GPU."
                ),
            ),
        ],
    )
    def test_batch_cell_list(self, dtype, device):
        """Test batch cell_list method."""

        # Create two systems
        positions1, cell1, pbc1 = create_random_system(
            200, 15.0, dtype=dtype, device=device
        )
        positions2, cell2, pbc2 = create_random_system(
            150, 15.0, dtype=dtype, device=device
        )
        cutoff = 5.0

        # Combine into batch
        positions = torch.cat([positions1, positions2], dim=0).to(device=device)
        cell = torch.stack([cell1.squeeze(0), cell2.squeeze(0)], dim=0).to(
            device=device
        )
        pbc = torch.stack([pbc1.squeeze(0), pbc2.squeeze(0)], dim=0).to(device=device)
        batch_idx = torch.cat(
            [
                torch.zeros(200, dtype=torch.int32, device=device),
                torch.ones(150, dtype=torch.int32, device=device),
            ]
        )

        # Call wrapper with explicit batch_cell_list method
        wrapper_result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            method="batch_cell_list",
            return_neighbor_list=False,
        )

        # Call batch_cell_list directly
        direct_result = batch_cell_list(
            positions, cutoff, cell, pbc, batch_idx, return_neighbor_list=False
        )

        # Results should match
        assert len(wrapper_result) == len(direct_result)
        assert_neighbor_matrix_equal(wrapper_result, direct_result)


class TestNeighborListDualCutoff:
    """Test dual cutoff functionality."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_naive_dual_cutoff(self, dtype, device):
        """Test naive dual cutoff method."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions, cell, pbc = create_random_system(
            100, 10.0, dtype=dtype, device=device
        )
        cutoff1 = 2.5
        cutoff2 = 3.5

        # Call wrapper with explicit method
        wrapper_result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            cutoff2=cutoff2,
            method="naive_dual_cutoff",
            return_neighbor_list=False,
        )

        # Call naive_dual_cutoff directly
        direct_result = naive_neighbor_list_dual_cutoff(
            positions, cutoff1, cutoff2, cell=cell, pbc=pbc, return_neighbor_list=False
        )

        # Results should match (6 outputs for dual cutoff with PBC)
        wrapper_result1, wrapper_result2 = (
            (wrapper_result[0], wrapper_result[1], wrapper_result[2]),
            (wrapper_result[3], wrapper_result[4], wrapper_result[5]),
        )
        direct_result1, direct_result2 = (
            (direct_result[0], direct_result[1], direct_result[2]),
            (direct_result[3], direct_result[4], direct_result[5]),
        )
        assert_neighbor_matrix_equal(wrapper_result1, direct_result1)
        assert_neighbor_matrix_equal(wrapper_result2, direct_result2)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_batch_naive_dual_cutoff(self, dtype, device):
        """Test batch naive dual cutoff method."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create two small systems
        positions1, cell1, pbc1 = create_random_system(
            50, 10.0, dtype=dtype, device=device
        )
        positions2, cell2, pbc2 = create_random_system(
            30, 10.0, dtype=dtype, device=device
        )

        # Combine into batch
        positions = torch.cat([positions1, positions2], dim=0)
        cell = torch.stack([cell1.squeeze(0), cell2.squeeze(0)], dim=0)
        pbc = torch.stack([pbc1.squeeze(0), pbc2.squeeze(0)], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(50, dtype=torch.int32, device=device),
                torch.ones(30, dtype=torch.int32, device=device),
            ]
        )
        batch_ptr = torch.tensor([0, 50, 80], dtype=torch.int32, device=device)

        cutoff1 = 2.5
        cutoff2 = 3.5

        # Call wrapper
        wrapper_result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cutoff2=cutoff2,
            method="batch_naive_dual_cutoff",
            max_neighbors1=50,
            max_neighbors2=50,
            return_neighbor_list=False,
        )

        # Call batch_naive_dual_cutoff directly
        direct_result = batch_naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=50,
            max_neighbors2=50,
            return_neighbor_list=False,
        )

        # Results should match
        assert len(wrapper_result) == 6
        assert len(direct_result) == 6

        wrapper_result1, wrapper_result2 = (
            (wrapper_result[0], wrapper_result[1], wrapper_result[2]),
            (wrapper_result[3], wrapper_result[4], wrapper_result[5]),
        )
        direct_result1, direct_result2 = (
            (direct_result[0], direct_result[1], direct_result[2]),
            (direct_result[3], direct_result[4], direct_result[5]),
        )
        assert_neighbor_matrix_equal(wrapper_result1, direct_result1)
        assert_neighbor_matrix_equal(wrapper_result2, direct_result2)


class TestNeighborListReturnFormats:
    """Test different return formats (matrix vs list)."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_return_neighbor_matrix(self, dtype, device):
        """Test returning neighbor matrix (default)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions, cell, pbc = create_random_system(
            100, 10.0, dtype=dtype, device=device
        )
        cutoff = 5.0

        # Default: return neighbor matrix
        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            return_neighbor_list=False,
        )

        neighbor_matrix, num_neighbors, shifts = result

        # Check shapes
        assert neighbor_matrix.ndim == 2
        assert neighbor_matrix.shape[0] == 100  # num_atoms
        assert num_neighbors.shape[0] == 100
        assert shifts.ndim == 3
        assert shifts.shape[0] == 100

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_return_neighbor_list_coo(self, dtype, device):
        """Test returning neighbor list in COO format."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions, cell, pbc = create_random_system(
            100, 10.0, dtype=dtype, device=device
        )
        cutoff = 5.0

        # Return neighbor list in COO format
        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            return_neighbor_list=True,
        )

        neighbor_list_coo, neighbor_ptr, shifts = result

        # Check shapes
        assert neighbor_list_coo.shape[0] == 2  # [sources, targets]
        assert neighbor_ptr.shape[0] == 101  # total_atoms + 1
        assert shifts.ndim == 2
        assert shifts.shape[1] == 3  # 3D shifts


class TestNeighborListHalfFill:
    """Test half_fill parameter."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("half_fill", [False, True])
    def test_half_fill_parameter(self, dtype, device, half_fill):
        """Test half_fill parameter is passed through correctly."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions, cell, pbc = create_random_system(
            50, 10.0, dtype=dtype, device=device
        )
        cutoff = 5.0

        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            half_fill=half_fill,
            return_neighbor_list=True,
        )

        neighbor_list_coo, _, _ = result

        # With half_fill=True, should have roughly half the pairs
        # (This is a weak test, but verifies the parameter is used)
        if half_fill:
            # Each pair should appear only once
            # Verify no duplicate pairs (i,j) and (j,i)
            sources = neighbor_list_coo[0].cpu().numpy()
            targets = neighbor_list_coo[1].cpu().numpy()
            pairs = set(zip(sources, targets))
            reverse_pairs = set(zip(targets, sources))

            # With half_fill, should have no overlapping pairs
            overlap = pairs.intersection(reverse_pairs)
            assert len(overlap) == 0, "Half-fill should not have reciprocal pairs"


class TestNeighborListNoPBC:
    """Test neighbor list without periodic boundary conditions."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_no_pbc_naive(self, dtype, device):
        """Test naive without PBC."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create positions without PBC
        positions = torch.randn(100, 3, dtype=dtype, device=device) * 5.0
        cutoff = 3.0

        # Call with no cell/pbc (no PBC)
        result = neighbor_list(
            positions, cutoff, method="naive", return_neighbor_list=True
        )

        # Should return 3 values (includes neighbor_ptr, but no shifts without PBC)
        assert len(result) == 2
        neighbor_list_coo, neighbor_ptr = result
        assert neighbor_list_coo.shape[0] == 2
        assert neighbor_ptr.shape[0] == 101


def _sorted_pairs(neighbor_list_coo):
    """Extract sorted (i, j) pairs from COO neighbor list for order-independent comparison."""
    import numpy as np

    sources = neighbor_list_coo[0].cpu().numpy()
    targets = neighbor_list_coo[1].cpu().numpy()
    idx = np.lexsort([targets, sources])
    return torch.from_numpy(np.stack([sources[idx], targets[idx]], axis=1))


class TestNeighborListBoundingBoxCell:
    """Test bounding-box cell fallback when cell=None for cell_list methods."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Requires GPU."
                ),
            ),
        ],
    )
    def test_cell_list_no_cell_vs_naive(self, dtype, device):
        """Explicit cell_list with cell=None should match naive results."""
        torch.manual_seed(42)
        positions = torch.randn(200, 3, dtype=dtype, device=device) * 10.0
        cutoff = 3.0

        cell_result = neighbor_list(
            positions, cutoff, method="cell_list", return_neighbor_list=True
        )
        naive_result = neighbor_list(
            positions, cutoff, method="naive", return_neighbor_list=True
        )

        # cell_list returns shifts (pbc is auto-created), naive does not
        cell_pairs = _sorted_pairs(cell_result[0])
        naive_pairs = _sorted_pairs(naive_result[0])
        torch.testing.assert_close(cell_pairs, naive_pairs)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Requires GPU."
                ),
            ),
        ],
    )
    def test_auto_dispatch_cell_list_vs_naive(self, dtype, device):
        """Auto-dispatched cell_list (>= 2000 atoms) with cell=None should match naive."""
        torch.manual_seed(42)
        positions = torch.randn(2500, 3, dtype=dtype, device=device) * 50.0
        cutoff = 2.0

        auto_result = neighbor_list(positions, cutoff, return_neighbor_list=True)
        naive_result = neighbor_list(
            positions, cutoff, method="naive", return_neighbor_list=True
        )

        auto_pairs = _sorted_pairs(auto_result[0])
        naive_pairs = _sorted_pairs(naive_result[0])
        torch.testing.assert_close(auto_pairs, naive_pairs)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Requires GPU."
                ),
            ),
        ],
    )
    def test_batch_cell_list_no_cell_vs_batch_naive(self, dtype, device):
        """Batch cell_list with cell=None should match batch naive results."""
        torch.manual_seed(42)
        n1, n2 = 150, 100
        positions1 = torch.randn(n1, 3, dtype=dtype, device=device) * 10.0
        positions2 = torch.randn(n2, 3, dtype=dtype, device=device) * 10.0
        positions = torch.cat([positions1, positions2], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(n1, dtype=torch.int32, device=device),
                torch.ones(n2, dtype=torch.int32, device=device),
            ]
        )
        batch_ptr = torch.tensor([0, n1, n1 + n2], dtype=torch.int32, device=device)
        cutoff = 3.0

        cell_result = neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_cell_list",
            return_neighbor_list=True,
        )
        naive_result = neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_naive",
            return_neighbor_list=True,
        )

        cell_pairs = _sorted_pairs(cell_result[0])
        naive_pairs = _sorted_pairs(naive_result[0])
        torch.testing.assert_close(cell_pairs, naive_pairs)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Requires GPU."
                ),
            ),
        ],
    )
    def test_batch_cell_list_no_cell_batch_ptr_only(self, dtype, device):
        """Batch cell_list with cell=None and only batch_ptr (no batch_idx) should work."""
        torch.manual_seed(42)
        n1, n2 = 150, 100
        positions1 = torch.randn(n1, 3, dtype=dtype, device=device) * 10.0
        positions2 = torch.randn(n2, 3, dtype=dtype, device=device) * 10.0
        positions = torch.cat([positions1, positions2], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(n1, dtype=torch.int32, device=device),
                torch.ones(n2, dtype=torch.int32, device=device),
            ]
        )
        batch_ptr = torch.tensor([0, n1, n1 + n2], dtype=torch.int32, device=device)
        cutoff = 3.0

        # Only batch_ptr supplied (no batch_idx) — previously crashed with
        # AttributeError: 'NoneType' object has no attribute 'unsqueeze'
        cell_result = neighbor_list(
            positions,
            cutoff,
            batch_ptr=batch_ptr,
            method="batch_cell_list",
            return_neighbor_list=True,
        )
        naive_result = neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_naive",
            return_neighbor_list=True,
        )

        cell_pairs = _sorted_pairs(cell_result[0])
        naive_pairs = _sorted_pairs(naive_result[0])
        torch.testing.assert_close(cell_pairs, naive_pairs)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Requires GPU."
                ),
            ),
        ],
    )
    def test_batch_negative_positions(self, dtype, device):
        """Batched bounding-box cell should handle positions with large negative offset."""
        torch.manual_seed(42)
        n1, n2 = 150, 100
        positions1 = torch.randn(n1, 3, dtype=dtype, device=device) * 10.0 - 50.0
        positions2 = torch.randn(n2, 3, dtype=dtype, device=device) * 10.0 - 50.0
        positions = torch.cat([positions1, positions2], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(n1, dtype=torch.int32, device=device),
                torch.ones(n2, dtype=torch.int32, device=device),
            ]
        )
        batch_ptr = torch.tensor([0, n1, n1 + n2], dtype=torch.int32, device=device)
        cutoff = 3.0

        cell_result = neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_cell_list",
            return_neighbor_list=True,
        )
        naive_result = neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_naive",
            return_neighbor_list=True,
        )

        cell_pairs = _sorted_pairs(cell_result[0])
        naive_pairs = _sorted_pairs(naive_result[0])
        torch.testing.assert_close(cell_pairs, naive_pairs)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Requires GPU."
                ),
            ),
        ],
    )
    def test_batch_different_offsets_per_system(self, dtype, device):
        """Batched bounding-box cell with per-system offsets should match batch naive."""
        torch.manual_seed(42)
        n1, n2 = 150, 100
        positions1 = torch.randn(n1, 3, dtype=dtype, device=device) * 10.0
        positions2 = torch.randn(n2, 3, dtype=dtype, device=device) * 10.0 + 100.0
        positions = torch.cat([positions1, positions2], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(n1, dtype=torch.int32, device=device),
                torch.ones(n2, dtype=torch.int32, device=device),
            ]
        )
        batch_ptr = torch.tensor([0, n1, n1 + n2], dtype=torch.int32, device=device)
        cutoff = 3.0

        cell_result = neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_cell_list",
            return_neighbor_list=True,
        )
        naive_result = neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_naive",
            return_neighbor_list=True,
        )

        cell_pairs = _sorted_pairs(cell_result[0])
        naive_pairs = _sorted_pairs(naive_result[0])
        torch.testing.assert_close(cell_pairs, naive_pairs)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Requires GPU."
                ),
            ),
        ],
    )
    def test_batch_cell_list_no_cell_positions_unchanged(self, dtype, device):
        """Calling neighbor_list with batch_cell_list should not mutate input positions."""
        torch.manual_seed(42)
        n1, n2 = 150, 100
        positions1 = torch.randn(n1, 3, dtype=dtype, device=device) * 10.0
        positions2 = torch.randn(n2, 3, dtype=dtype, device=device) * 10.0
        positions = torch.cat([positions1, positions2], dim=0)
        positions_clone = positions.clone()
        batch_idx = torch.cat(
            [
                torch.zeros(n1, dtype=torch.int32, device=device),
                torch.ones(n2, dtype=torch.int32, device=device),
            ]
        )
        batch_ptr = torch.tensor([0, n1, n1 + n2], dtype=torch.int32, device=device)
        cutoff = 3.0

        neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_cell_list",
            return_neighbor_list=True,
        )

        torch.testing.assert_close(positions, positions_clone)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Requires GPU."
                ),
            ),
        ],
    )
    def test_negative_positions(self, dtype, device):
        """Bounding-box cell should handle positions with large negative offset."""
        torch.manual_seed(42)
        positions = torch.randn(200, 3, dtype=dtype, device=device) * 10.0 - 50.0
        cutoff = 3.0

        cell_result = neighbor_list(
            positions, cutoff, method="cell_list", return_neighbor_list=True
        )
        naive_result = neighbor_list(
            positions, cutoff, method="naive", return_neighbor_list=True
        )

        cell_pairs = _sorted_pairs(cell_result[0])
        naive_pairs = _sorted_pairs(naive_result[0])
        torch.testing.assert_close(cell_pairs, naive_pairs)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Requires GPU."
                ),
            ),
        ],
    )
    def test_planar_positions(self, dtype, device):
        """Bounding-box cell should handle planar positions (one dimension constant)."""
        torch.manual_seed(42)
        positions = torch.randn(200, 3, dtype=dtype, device=device) * 10.0
        positions[:, 2] = 0.0  # All atoms on z=0 plane
        cutoff = 3.0

        cell_result = neighbor_list(
            positions, cutoff, method="cell_list", return_neighbor_list=True
        )
        naive_result = neighbor_list(
            positions, cutoff, method="naive", return_neighbor_list=True
        )

        cell_pairs = _sorted_pairs(cell_result[0])
        naive_pairs = _sorted_pairs(naive_result[0])
        torch.testing.assert_close(cell_pairs, naive_pairs)


class TestNeighborListInvalidMethod:
    """Test error handling for invalid method."""

    def test_invalid_method_name(self):
        """Test that invalid method name raises ValueError."""
        positions = torch.randn(10, 3)
        cutoff = 2.0

        with pytest.raises(ValueError, match="Invalid method"):
            neighbor_list(positions, cutoff, method="invalid_method")


class TestNeighborListKwargs:
    """Test kwargs passing to underlying methods."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_kwargs_max_neighbors_naive(self, device):
        """Test passing max_neighbors kwarg to naive method."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions, cell, pbc = create_random_system(
            50, 10.0, dtype=torch.float32, device=device
        )
        cutoff = 5.0

        # Call with explicit max_neighbors
        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            max_neighbors=20,
            return_neighbor_list=False,
        )

        neighbor_matrix, _, _ = result
        # Check that matrix has the specified max_neighbors size
        assert neighbor_matrix.shape[1] == 20

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_kwargs_max_neighbors_cell_list(self, device):
        """Test passing max_neighbors kwarg to cell_list method."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions, cell, pbc = create_random_system(
            100, 15.0, dtype=torch.float32, device=device
        )
        cutoff = 5.0

        # Call with explicit max_neighbors
        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="cell_list",
            max_neighbors=30,
            return_neighbor_list=False,
        )

        neighbor_matrix, _, _ = result
        # Check that matrix has the specified max_neighbors size
        assert neighbor_matrix.shape[1] == 30

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_kwargs_max_neighbors_dual_cutoff(self, device):
        """Test passing max_neighbors1 and max_neighbors2 kwargs to dual cutoff method."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions, cell, pbc = create_random_system(
            50, 10.0, dtype=torch.float32, device=device
        )
        cutoff1 = 2.5
        cutoff2 = 3.5

        # Call with explicit max_neighbors for both cutoffs
        # Note: dual cutoff uses max_neighbors1, not max_neighbors
        result = neighbor_list(
            positions,
            cutoff1,
            cell=cell,
            pbc=pbc,
            cutoff2=cutoff2,
            method="naive_dual_cutoff",
            max_neighbors1=15,
            max_neighbors2=25,
            return_neighbor_list=False,
        )

        # Returns 6 outputs for dual cutoff with matrix format
        nm1, _, _, nm2, _, _ = result
        assert nm1.shape[1] == 15  # First matrix has max_neighbors1
        assert nm2.shape[1] == 25  # Second matrix has max_neighbors2

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_kwargs_preallocated_tensors_naive(self, device):
        """Test passing pre-allocated tensors via kwargs to naive method."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions, cell, pbc = create_random_system(
            50, 10.0, dtype=torch.float32, device=device
        )
        cutoff = 5.0

        # Pre-allocate tensors
        max_neighbors = 20
        neighbor_matrix = torch.full(
            (50, max_neighbors), 50, dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(50, dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (50, max_neighbors, 3), dtype=torch.int32, device=device
        )

        # Call with pre-allocated tensors
        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            return_neighbor_list=False,
        )

        # Should return the same pre-allocated tensors (modified in-place)
        result_nm, result_num, result_shifts = result
        assert result_nm is neighbor_matrix
        assert result_num is num_neighbors
        assert result_shifts is neighbor_matrix_shifts

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_kwargs_invalid_parameter_raises_error(self, device):
        """Test that invalid kwargs raise TypeError."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions = torch.randn(50, 3, dtype=torch.float32, device=device)
        cutoff = 2.0

        # Call with invalid kwarg should raise TypeError
        with pytest.raises(TypeError):
            neighbor_list(
                positions,
                cutoff,
                method="naive",
                invalid_parameter_name=123,  # This doesn't exist
            )

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_kwargs_forwarded_with_auto_selection(self, device):
        """Test that kwargs are forwarded correctly with auto method selection."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions, cell, pbc = create_random_system(
            50, 10.0, dtype=torch.float32, device=device
        )
        cutoff = 5.0

        # Let it auto-select naive, but pass max_neighbors
        result = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=25,
            return_neighbor_list=False,
        )

        neighbor_matrix, _, _ = result
        # Should have respected the max_neighbors kwarg
        assert neighbor_matrix.shape[1] == 25


class TestNeighborListEdgeCases:
    """Test edge cases."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_empty_system(self, device):
        """Test with empty system (0 atoms)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions = torch.empty(0, 3, dtype=torch.float32, device=device)
        cutoff = 2.0

        result = neighbor_list(
            positions, cutoff, method="naive", return_neighbor_list=True
        )

        # Should handle gracefully
        assert len(result) == 2
        neighbor_list_coo, neighbor_ptr = result
        assert neighbor_list_coo.shape[1] == 0  # No pairs
        assert neighbor_ptr.shape[0] == 1
        assert neighbor_ptr[0] == 0

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_single_atom(self, device):
        """Test with single atom system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions = torch.randn(1, 3, dtype=torch.float32, device=device)
        cutoff = 2.0

        result = neighbor_list(
            positions, cutoff, method="naive", return_neighbor_list=True
        )

        # Single atom has no neighbors
        assert len(result) == 2
        neighbor_list_coo, neighbor_ptr = result
        assert neighbor_list_coo.shape[1] == 0  # No pairs
        assert neighbor_ptr.shape[0] == 2
        assert neighbor_ptr[0] == 0


class TestPrepareBatchIdxPtr:
    """Test prepare_batch_idx_ptr function."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        "batch_idx", [None, torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.int32)]
    )
    @pytest.mark.parametrize(
        "batch_ptr", [None, torch.tensor([0, 2, 5, 7], dtype=torch.int32)]
    )
    def testprepare_batch_idx_ptr(self, device, batch_idx, batch_ptr):
        """Test prepare_batch_idx_ptr function."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if batch_idx is not None:
            batch_idx = batch_idx.to(device=device)
        if batch_ptr is not None:
            batch_ptr = batch_ptr.to(device=device)
        num_atoms = 7
        num_systems = 3
        num_atoms_per_system = torch.tensor([2, 3, 2], dtype=torch.int32, device=device)

        if batch_idx is None and batch_ptr is None:
            with pytest.raises(
                ValueError, match="Either batch_idx or batch_ptr must be provided."
            ):
                prepare_batch_idx_ptr(batch_idx, batch_ptr, num_atoms, device)
        else:
            batch_idx, batch_ptr = prepare_batch_idx_ptr(
                batch_idx, batch_ptr, num_atoms, device
            )
            assert batch_idx.shape[0] == num_atoms
            assert batch_ptr.shape[0] == num_systems + 1

            calculated_ptr = torch.zeros(
                num_systems + 1, dtype=torch.int32, device=device
            )
            torch.cumsum(num_atoms_per_system, dim=0, out=calculated_ptr[1:])
            assert torch.all(batch_ptr == calculated_ptr)
