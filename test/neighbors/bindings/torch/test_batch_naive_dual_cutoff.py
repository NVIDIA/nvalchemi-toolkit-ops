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

"""Tests for PyTorch bindings of batched naive dual cutoff neighbor list methods."""

import pytest
import torch

from nvalchemiops.torch.neighbors.batch_naive import (
    batch_naive_neighbor_list,
)
from nvalchemiops.torch.neighbors.batch_naive_dual_cutoff import (
    batch_naive_neighbor_list_dual_cutoff,
)

from ...test_utils import (
    create_batch_systems,
)


def create_batch_idx_and_ptr(
    atoms_per_system: list, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create batch_idx and batch_ptr tensors from atoms_per_system list.

    Parameters
    ----------
    atoms_per_system : list
        Number of atoms in each system
    device : str
        Device to create tensors on

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        batch_idx and batch_ptr tensors
    """
    total_atoms = sum(atoms_per_system)
    batch_idx = torch.zeros(total_atoms, dtype=torch.int32, device=device)
    batch_ptr = torch.zeros(len(atoms_per_system) + 1, dtype=torch.int32, device=device)

    start_idx = 0
    for i, num_atoms in enumerate(atoms_per_system):
        batch_idx[start_idx : start_idx + num_atoms] = i
        batch_ptr[i + 1] = batch_ptr[i] + num_atoms
        start_idx += num_atoms

    return batch_idx, batch_ptr


class TestBatchNaiveDualCutoffCorrectness:
    """Test correctness of batch naive dual cutoff neighbor list."""

    def test_matrix_format_no_pbc(self, device, dtype, half_fill):
        """Test dual cutoff batch neighbor list in matrix format without PBC."""
        atoms_per_system = [6, 8]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 20
        max_neighbors2 = 30

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                pbc=None,
                cell=None,
                half_fill=half_fill,
            )
        )

        # Check output types and shapes
        expected_rows = positions_batch.shape[0]
        assert neighbor_matrix1.dtype == torch.int32
        assert neighbor_matrix2.dtype == torch.int32
        assert num_neighbors1.dtype == torch.int32
        assert num_neighbors2.dtype == torch.int32
        assert neighbor_matrix1.shape == (expected_rows, max_neighbors1)
        assert neighbor_matrix2.shape == (expected_rows, max_neighbors2)
        assert num_neighbors1.shape == (positions_batch.shape[0],)
        assert num_neighbors2.shape == (positions_batch.shape[0],)
        assert neighbor_matrix1.device == torch.device(device)
        assert neighbor_matrix2.device == torch.device(device)

        # Check neighbor counts are reasonable
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors1 <= max_neighbors1)
        assert torch.all(num_neighbors2 <= max_neighbors2)
        assert torch.all(num_neighbors2 >= num_neighbors1)

    def test_matrix_format_with_pbc(self, device, dtype, half_fill):
        """Test dual cutoff batch neighbor list in matrix format with PBC."""
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 30
        max_neighbors2 = 50

        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
        )

        # Check output types and shapes
        expected_rows = positions_batch.shape[0]
        assert neighbor_matrix1.shape == (expected_rows, max_neighbors1)
        assert neighbor_matrix2.shape == (expected_rows, max_neighbors2)
        assert neighbor_matrix_shifts1.shape == (expected_rows, max_neighbors1, 3)
        assert neighbor_matrix_shifts2.shape == (expected_rows, max_neighbors2, 3)
        assert num_neighbors1.shape == (positions_batch.shape[0],)
        assert num_neighbors2.shape == (positions_batch.shape[0],)

        # Check neighbor counts
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors2 >= num_neighbors1)

    def test_consistency_with_single_cutoff(self, device, dtype):
        """Test that dual cutoff results match two separate single cutoff calls."""
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 30
        max_neighbors2 = 50

        # Get dual cutoff result
        (
            neighbor_matrix1_dual,
            num_neighbors1_dual,
            neighbor_matrix_shifts1_dual,
            neighbor_matrix2_dual,
            num_neighbors2_dual,
            neighbor_matrix_shifts2_dual,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=False,
        )

        # Get single cutoff results
        (
            neighbor_matrix1_single,
            num_neighbors1_single,
            neighbor_matrix_shifts1_single,
        ) = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff1,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=False,
        )

        (
            neighbor_matrix2_single,
            num_neighbors2_single,
            neighbor_matrix_shifts2_single,
        ) = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=False,
        )

        # Compare neighbor counts
        torch.testing.assert_close(
            num_neighbors1_dual, num_neighbors1_single, rtol=0, atol=0
        )
        torch.testing.assert_close(
            num_neighbors2_dual, num_neighbors2_single, rtol=0, atol=0
        )

    def test_larger_cutoff_finds_more_neighbors(self, device, dtype):
        """Test that larger cutoff finds at least as many neighbors as smaller cutoff."""
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5

        (
            _,
            num_neighbors1,
            _,
            _,
            num_neighbors2,
            _,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=30,
            max_neighbors2=50,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=False,
        )

        # Verify cutoff2 finds at least as many neighbors
        assert torch.all(num_neighbors2 >= num_neighbors1)


class TestBatchNaiveDualCutoffEdgeCases:
    """Test edge cases for batch naive dual cutoff neighbor list."""

    def test_empty_system(self, device, dtype, half_fill):
        """Test dual cutoff batch neighbor list with empty system."""
        positions_empty = torch.empty(0, 3, dtype=dtype, device=device)
        batch_idx_empty = torch.empty(0, dtype=torch.int32, device=device)
        batch_ptr_empty = torch.tensor([0], dtype=torch.int32, device=device)

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_empty,
                cutoff1=1.0,
                cutoff2=1.5,
                batch_idx=batch_idx_empty,
                batch_ptr=batch_ptr_empty,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
                half_fill=half_fill,
            )
        )
        assert neighbor_matrix1.shape == (0, 10)
        assert neighbor_matrix2.shape == (0, 15)
        assert num_neighbors1.shape == (0,)
        assert num_neighbors2.shape == (0,)

    def test_single_atom_system(self, device, dtype, half_fill):
        """Test dual cutoff batch neighbor list with single atom."""
        positions_single = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        batch_idx_single = torch.tensor([0], dtype=torch.int32, device=device)
        batch_ptr_single = torch.tensor([0, 1], dtype=torch.int32, device=device)

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_single,
                cutoff1=1.0,
                cutoff2=1.5,
                batch_idx=batch_idx_single,
                batch_ptr=batch_ptr_single,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
                half_fill=half_fill,
            )
        )
        assert num_neighbors1[0].item() == 0
        assert num_neighbors2[0].item() == 0

    def test_zero_cutoffs(self, device, dtype, half_fill):
        """Test dual cutoff batch neighbor list with zero cutoffs."""
        atoms_per_system = [4, 4]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=0.0,
                cutoff2=0.0,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
                half_fill=half_fill,
            )
        )
        assert torch.all(num_neighbors1 == 0)
        assert torch.all(num_neighbors2 == 0)

    def test_identical_cutoffs(self, device, dtype, half_fill):
        """Test dual cutoff batch neighbor list with identical cutoff values."""
        atoms_per_system = [6, 8]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.5
        _, num_neighbors1, _, num_neighbors2 = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff,
            cutoff2=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=20,
            max_neighbors2=20,
            pbc=None,
            cell=None,
            half_fill=half_fill,
        )

        # When cutoffs are identical, neighbor counts should match
        torch.testing.assert_close(num_neighbors1, num_neighbors2, rtol=0, atol=0)


class TestBatchNaiveDualCutoffErrors:
    """Test error conditions for batch naive dual cutoff neighbor list."""

    def test_cell_without_pbc_error(self, device, dtype):
        """Test that providing cell without pbc raises error."""
        atoms_per_system = [4, 6]
        positions_batch, cell_batch, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        with pytest.raises(
            ValueError, match="If cell is provided, pbc must also be provided"
        ):
            batch_naive_neighbor_list_dual_cutoff(
                positions_batch,
                1.0,
                1.5,
                batch_idx,
                batch_ptr,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=cell_batch,
            )

    def test_pbc_without_cell_error(self, device, dtype):
        """Test that providing pbc without cell raises error."""
        atoms_per_system = [4, 6]
        positions_batch, _, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        with pytest.raises(
            ValueError, match="If pbc is provided, cell must also be provided"
        ):
            batch_naive_neighbor_list_dual_cutoff(
                positions_batch,
                1.0,
                1.5,
                batch_idx,
                batch_ptr,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=pbc_batch,
                cell=None,
            )

    def test_mismatched_batch_dimensions(self, device, dtype):
        """Test that mismatched batch_idx length raises RuntimeError."""
        atoms_per_system = [4, 6]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Create mismatched batch_idx (wrong total atoms - 5 instead of 10)
        bad_batch_idx = torch.zeros(5, dtype=torch.int32, device=device)

        with pytest.raises(
            RuntimeError, match="batch_idx length.*does not match num_atoms"
        ):
            batch_naive_neighbor_list_dual_cutoff(
                positions_batch,
                1.0,
                1.5,
                bad_batch_idx,
                batch_ptr,
                max_neighbors1=10,
                max_neighbors2=15,
            )


class TestBatchNaiveDualCutoffOutputFormats:
    """Test different output formats for batch naive dual cutoff neighbor list."""

    def test_list_format_no_pbc(self, device, dtype, half_fill):
        """Test dual cutoff batch neighbor list in COO format without PBC."""
        atoms_per_system = [6, 8]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5

        neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=30,
                max_neighbors2=50,
                pbc=None,
                cell=None,
                half_fill=half_fill,
                return_neighbor_list=True,
            )
        )

        # Check that we get neighbor list format (2, N) instead of matrix
        assert neighbor_list1.ndim == 2
        assert neighbor_list2.ndim == 2
        assert neighbor_list1.shape[0] == 2
        assert neighbor_list2.shape[0] == 2
        assert neighbor_list1.dtype == torch.int32
        assert neighbor_list2.dtype == torch.int32

        # Larger cutoff should find at least as many pairs
        assert neighbor_list2.shape[1] >= neighbor_list1.shape[1]

    def test_list_format_with_pbc(self, device, dtype, half_fill):
        """Test dual cutoff batch neighbor list in COO format with PBC."""
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5

        (
            neighbor_list1,
            neighbor_ptr1,
            unit_shifts1,
            neighbor_list2,
            neighbor_ptr2,
            unit_shifts2,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=30,
            max_neighbors2=50,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
            return_neighbor_list=True,
        )

        # Check neighbor list format
        assert neighbor_list1.shape[0] == 2
        assert neighbor_list2.shape[0] == 2
        assert unit_shifts1.shape[0] == neighbor_list1.shape[1]
        assert unit_shifts2.shape[0] == neighbor_list2.shape[1]

        # Larger cutoff should find at least as many pairs
        assert neighbor_list2.shape[1] >= neighbor_list1.shape[1]

    def test_max_neighbors_same_value(self, device, dtype):
        """Test that both matrices have correct shape with same max_neighbors."""
        atoms_per_system = [4, 6]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2,
            atoms_per_system=atoms_per_system,
            dtype=dtype,
            device=device,
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        neighbor_matrix1, _, neighbor_matrix2, _ = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=1.0,
                cutoff2=1.5,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=10,
                max_neighbors2=10,
                pbc=None,
                cell=None,
            )
        )

        assert neighbor_matrix1.shape[1] == neighbor_matrix2.shape[1] == 10


class TestBatchNaiveDualCutoffPerformance:
    """Test performance characteristics of batch naive dual cutoff neighbor list."""

    @pytest.mark.slow
    def test_scaling_with_system_size(self, device):
        """Test that batch dual cutoff scales reasonably with system size."""
        import time

        dtype = torch.float32
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 50
        max_neighbors2 = 80

        # Test different batch sizes (smaller for CPU)
        batch_sizes = (
            [(2, [10, 12], [2.0, 2.0]), (3, [8, 10, 12], [2.0, 2.0, 2.0])]
            if device == "cpu"
            else [
                (3, [20, 25, 30], [2.0, 2.0, 2.0]),
                (4, [15, 20, 25, 30], [2.0, 2.0, 2.0, 2.0]),
            ]
        )
        times = []

        for num_systems, atoms_per_system, cell_sizes in batch_sizes:
            positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                num_systems=num_systems,
                cell_sizes=cell_sizes,
                atoms_per_system=atoms_per_system,
                dtype=dtype,
                device=device,
            )
            batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

            # Warm up
            for _ in range(10):
                batch_naive_neighbor_list_dual_cutoff(
                    positions_batch,
                    cutoff1,
                    cutoff2,
                    batch_idx,
                    batch_ptr,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    half_fill=True,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            # Time the operation
            start_time = time.time()
            for _ in range(100):
                batch_naive_neighbor_list_dual_cutoff(
                    positions_batch,
                    cutoff1,
                    cutoff2,
                    batch_idx,
                    batch_ptr,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    half_fill=True,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            times.append(elapsed)

        # Loose check that time increases with system size
        assert times[1] > times[0] * 0.5

    @pytest.mark.slow
    def test_cutoff_scaling(self, device):
        """Test scaling with different cutoff values."""
        dtype = torch.float32
        atoms_per_system = [15, 20]
        max_neighbors1 = 100
        max_neighbors2 = 150

        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Test different cutoff pairs
        cutoff_pairs = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5)]
        neighbor_counts1 = []
        neighbor_counts2 = []

        for cutoff1, cutoff2 in cutoff_pairs:
            (_, num_neighbors1, _, _, num_neighbors2, _) = (
                batch_naive_neighbor_list_dual_cutoff(
                    positions_batch,
                    cutoff1,
                    cutoff2,
                    batch_idx,
                    batch_ptr,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    half_fill=True,
                )
            )
            total_pairs1 = num_neighbors1.sum().item()
            total_pairs2 = num_neighbors2.sum().item()
            neighbor_counts1.append(total_pairs1)
            neighbor_counts2.append(total_pairs2)

        # Check that neighbor count increases with cutoff
        for i in range(1, len(neighbor_counts1)):
            assert neighbor_counts1[i] >= neighbor_counts1[i - 1]
            assert neighbor_counts2[i] >= neighbor_counts2[i - 1]

        # Check that cutoff2 finds at least as many neighbors as cutoff1
        for count1, count2 in zip(neighbor_counts1, neighbor_counts2):
            assert count2 >= count1

    def test_random_systems_robustness(self, device, dtype, half_fill):
        """Test with random systems of various sizes and configurations."""
        for pbc_flag in [True, False]:
            # Test several random systems
            for seed in [42, 123, 456]:
                atoms_per_system = [15, 20, 18]
                positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                    num_systems=3,
                    atoms_per_system=atoms_per_system,
                    dtype=dtype,
                    device=device,
                    seed=seed,
                    pbc_flag=pbc_flag,
                )
                batch_idx, batch_ptr = create_batch_idx_and_ptr(
                    atoms_per_system, device
                )

                cutoff1 = 1.0
                cutoff2 = 1.5
                max_neighbors1 = 50
                max_neighbors2 = 80

                if not pbc_flag:
                    cell_batch = None
                    pbc_batch = None

                # Should not crash
                result = batch_naive_neighbor_list_dual_cutoff(
                    positions=positions_batch,
                    cutoff1=cutoff1,
                    cutoff2=cutoff2,
                    batch_idx=batch_idx,
                    batch_ptr=batch_ptr,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    half_fill=half_fill,
                )

                if pbc_flag:
                    (
                        _,
                        num_neighbors1,
                        _,
                        _,
                        num_neighbors2,
                        _,
                    ) = result
                else:
                    (
                        _,
                        num_neighbors1,
                        _,
                        num_neighbors2,
                    ) = result

                # Basic sanity checks
                assert torch.all(num_neighbors1 >= 0)
                assert torch.all(num_neighbors2 >= 0)
                assert torch.all(num_neighbors2 >= num_neighbors1)

    def test_extreme_geometries(self, device, dtype, half_fill):
        """Test with extreme cell geometries."""
        atoms_per_system = [8, 10]
        positions_batch = torch.rand(18, 3, dtype=dtype, device=device)
        cell_batch = torch.tensor(
            [
                [[10.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
                [[0.1, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.1]],
            ],
            dtype=dtype,
            device=device,
        )
        pbc_batch = torch.tensor(
            [[True, True, True], [True, True, True]], device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Scale positions to fit in cells
        positions_batch[:8] = positions_batch[:8] * torch.tensor(
            [10.0, 0.1, 0.1], device=device
        )
        positions_batch[8:] = positions_batch[8:] * torch.tensor(
            [0.1, 10.0, 0.1], device=device
        )

        cutoff1 = 0.15
        cutoff2 = 0.25

        (
            _,
            num_neighbors1,
            _,
            _,
            num_neighbors2,
            _,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=20,
            max_neighbors2=30,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
        )

        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors2 >= num_neighbors1)

    def test_large_cutoffs(self, device, dtype, half_fill):
        """Test with very large cutoffs relative to cell size."""
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Cutoffs larger than cell size
        large_cutoff1 = 4.0
        large_cutoff2 = 6.0

        (
            _,
            num_neighbors1,
            _,
            _,
            num_neighbors2,
            _,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=large_cutoff1,
            cutoff2=large_cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=100,
            max_neighbors2=150,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
        )

        # Should find many neighbors
        assert num_neighbors1.sum() > 0
        assert num_neighbors2.sum() > 0
        assert torch.all(num_neighbors2 >= num_neighbors1)

    def test_precision_consistency(self, device, half_fill):
        """Test that float32 and float64 give consistent results."""
        atoms_per_system = [6, 8]
        positions_batch_f32, cell_batch_f32, pbc_batch, _ = create_batch_systems(
            num_systems=2,
            atoms_per_system=atoms_per_system,
            dtype=torch.float32,
            device=device,
        )
        positions_batch_f64 = positions_batch_f32.double()
        cell_batch_f64 = cell_batch_f32.double()
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5

        # Get results for both precisions
        (_, num_neighbors1_f32, _, _, num_neighbors2_f32, _) = (
            batch_naive_neighbor_list_dual_cutoff(
                positions_batch_f32,
                cutoff1,
                cutoff2,
                batch_idx,
                batch_ptr,
                max_neighbors1=50,
                max_neighbors2=80,
                pbc=pbc_batch,
                cell=cell_batch_f32,
                half_fill=half_fill,
            )
        )
        (_, num_neighbors1_f64, _, _, num_neighbors2_f64, _) = (
            batch_naive_neighbor_list_dual_cutoff(
                positions_batch_f64,
                cutoff1,
                cutoff2,
                batch_idx,
                batch_ptr,
                max_neighbors1=50,
                max_neighbors2=80,
                pbc=pbc_batch,
                cell=cell_batch_f64,
                half_fill=half_fill,
            )
        )

        # Neighbor counts should be identical
        torch.testing.assert_close(
            num_neighbors1_f32, num_neighbors1_f64, rtol=0, atol=0
        )
        torch.testing.assert_close(
            num_neighbors2_f32, num_neighbors2_f64, rtol=0, atol=0
        )
