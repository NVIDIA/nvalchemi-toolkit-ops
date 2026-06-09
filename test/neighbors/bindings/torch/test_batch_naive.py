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

"""Tests for PyTorch bindings of batched naive neighbor list methods."""

from unittest import mock

import pytest
import torch

from nvalchemiops.torch.neighbors.batch_naive import (
    batch_naive_neighbor_list,
)
from nvalchemiops.torch.neighbors.neighbor_utils import compute_naive_num_shifts

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
        Device to place tensors on

    Returns
    -------
    batch_idx : torch.Tensor
        System index for each atom (total_atoms,)
    batch_ptr : torch.Tensor
        Start index for each system (num_systems + 1,)
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


def _active_neighbor_shift_rows(
    neighbor_matrix: torch.Tensor,
    shifts: torch.Tensor,
    counts: torch.Tensor,
    atom_index: int,
) -> list[tuple[int, int, int, int]]:
    """Return sorted active ``(neighbor, sx, sy, sz)`` rows for one atom."""
    count = int(counts[atom_index].item())
    rows = torch.cat(
        (
            neighbor_matrix[atom_index, :count].unsqueeze(1),
            shifts[atom_index, :count],
        ),
        dim=1,
    )
    return sorted(tuple(row) for row in rows.detach().cpu().tolist())


class TestBatchNaiveCorrectness:
    """Tests verifying correctness of batch naive neighbor list implementation."""

    def test_basic_without_pbc(self, device, dtype, half_fill):
        """Test basic neighbor list calculation without periodic boundaries."""
        atoms_per_system = [6, 8, 7]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 20

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=None,
            cell=None,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        # Check output types and shapes
        total_atoms = positions_batch.shape[0]
        assert neighbor_matrix.dtype == torch.int32
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix.shape == (total_atoms, max_neighbors)
        assert num_neighbors.shape == (total_atoms,)
        assert neighbor_matrix.device == torch.device(device)
        assert num_neighbors.device == torch.device(device)

        # Check neighbor counts are reasonable
        assert torch.all(num_neighbors >= 0)
        assert torch.all(num_neighbors <= max_neighbors)

    def test_basic_with_pbc(self, device, dtype, half_fill):
        """Test basic neighbor list calculation with periodic boundaries."""
        atoms_per_system = [5, 7, 6]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 30

        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
            batch_naive_neighbor_list(
                positions=positions_batch,
                cutoff=cutoff,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                pbc=pbc_batch,
                cell=cell_batch,
                max_neighbors=max_neighbors,
                half_fill=half_fill,
            )
        )

        # Check output types and shapes
        total_atoms = positions_batch.shape[0]
        assert neighbor_matrix.dtype == torch.int32
        assert neighbor_matrix_shifts.dtype == torch.int32
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix.shape == (total_atoms, max_neighbors)
        assert neighbor_matrix_shifts.shape == (total_atoms, max_neighbors, 3)
        assert num_neighbors.shape == (total_atoms,)
        assert neighbor_matrix.device == torch.device(device)
        assert neighbor_matrix_shifts.device == torch.device(device)
        assert num_neighbors.device == torch.device(device)

        # Check neighbor counts
        assert torch.all(num_neighbors >= 0)

    def test_pbc_uses_forwarded_batch_idx_without_rebuilding(self, device, dtype):
        """Test PBC path does not rebuild batch_idx when both batch tensors are provided."""
        atoms_per_system = [5, 7, 6]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=3,
            atoms_per_system=atoms_per_system,
            dtype=dtype,
            device=device,
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        with mock.patch(
            "nvalchemiops.torch.neighbors.batch_naive.torch.repeat_interleave",
            side_effect=AssertionError("unexpected repeat_interleave call"),
        ):
            neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
                batch_naive_neighbor_list(
                    positions=positions_batch,
                    cutoff=1.2,
                    batch_idx=batch_idx,
                    batch_ptr=batch_ptr,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    max_neighbors=30,
                    half_fill=False,
                )
            )

        total_atoms = positions_batch.shape[0]
        assert neighbor_matrix.shape == (total_atoms, 30)
        assert neighbor_matrix_shifts.shape == (total_atoms, 30, 3)
        assert num_neighbors.shape == (total_atoms,)

    def test_consistency_single_system_no_pbc(self, device, dtype):
        """Test batch gives same results as single system without PBC.

        Compares batch processing of multiple systems with processing
        each system individually to ensure consistency.
        """
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 30

        # Get batch result
        _, num_neighbors_batch, _ = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=pbc_batch,
            cell=cell_batch,
            max_neighbors=max_neighbors,
            half_fill=False,
        )

        # Compare with single system results
        for sys_idx in range(2):
            start_idx = batch_ptr[sys_idx].item()
            end_idx = batch_ptr[sys_idx + 1].item()

            positions_single = positions_batch[start_idx:end_idx]
            pbc_single = pbc_batch[sys_idx : sys_idx + 1]
            cell_single = cell_batch[sys_idx : sys_idx + 1]

            n_atoms_single = positions_single.shape[0]
            batch_idx_single = torch.zeros(
                n_atoms_single, dtype=torch.int32, device=positions_single.device
            )
            batch_ptr_single = torch.tensor(
                [0, n_atoms_single], dtype=torch.int32, device=positions_single.device
            )

            _, num_neighbors_single, _ = batch_naive_neighbor_list(
                positions=positions_single,
                cutoff=cutoff,
                batch_idx=batch_idx_single,
                batch_ptr=batch_ptr_single,
                pbc=pbc_single,
                cell=cell_single,
                max_neighbors=max_neighbors,
                half_fill=False,
            )

            # Neighbor counts should be identical
            torch.testing.assert_close(
                num_neighbors_batch[start_idx:end_idx],
                num_neighbors_single,
                rtol=0,
                atol=0,
            )

    def test_consistency_single_system_with_pbc(self, device, dtype):
        """Test batch gives same results as single system with PBC.

        Verifies that processing systems in a batch gives identical
        results to processing them individually with periodic boundaries.
        """
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 30

        # Get batch result
        _, num_neighbors_batch, _ = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=pbc_batch,
            cell=cell_batch,
            max_neighbors=max_neighbors,
            half_fill=False,
        )

        # Compare with single system results
        for sys_idx in range(2):
            start_idx = batch_ptr[sys_idx].item()
            end_idx = batch_ptr[sys_idx + 1].item()

            positions_single = positions_batch[start_idx:end_idx]
            cell_single = cell_batch[sys_idx : sys_idx + 1]
            pbc_single = pbc_batch[sys_idx : sys_idx + 1]

            n_atoms_single = positions_single.shape[0]
            batch_idx_single = torch.zeros(
                n_atoms_single, dtype=torch.int32, device=positions_single.device
            )
            batch_ptr_single = torch.tensor(
                [0, n_atoms_single], dtype=torch.int32, device=positions_single.device
            )

            _, num_neighbors_single, _ = batch_naive_neighbor_list(
                positions=positions_single,
                cutoff=cutoff,
                batch_idx=batch_idx_single,
                batch_ptr=batch_ptr_single,
                pbc=pbc_single,
                cell=cell_single,
                max_neighbors=max_neighbors,
                half_fill=False,
            )

            # Neighbor counts should be identical
            torch.testing.assert_close(
                num_neighbors_batch[start_idx:end_idx],
                num_neighbors_single,
                rtol=0,
                atol=0,
            )

    def test_precision_consistency(self, device, half_fill):
        """Test float32 and float64 give consistent neighbor counts.

        For the same geometry, both precisions should find the same
        number of neighbors (though distances may differ slightly).
        """
        atoms_per_system = [6, 8, 7]
        positions_batch_f32, cell_batch_f32, pbc_batch, _ = create_batch_systems(
            num_systems=3,
            atoms_per_system=atoms_per_system,
            dtype=torch.float32,
            device=device,
        )
        positions_batch_f64 = positions_batch_f32.double()
        cell_batch_f64 = cell_batch_f32.double()

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 30

        # Get results for both precisions
        _, num_neighbors_f32, _ = batch_naive_neighbor_list(
            positions_batch_f32,
            cutoff,
            pbc=pbc_batch,
            cell=cell_batch_f32,
            max_neighbors=max_neighbors,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            half_fill=half_fill,
        )
        _, num_neighbors_f64, _ = batch_naive_neighbor_list(
            positions_batch_f64,
            cutoff,
            pbc=pbc_batch,
            cell=cell_batch_f64,
            max_neighbors=max_neighbors,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            half_fill=half_fill,
        )

        # Neighbor counts should be identical
        torch.testing.assert_close(num_neighbors_f32, num_neighbors_f64, rtol=0, atol=0)

    def test_random_systems_no_pbc(self, device, dtype, half_fill):
        """Test with random batch systems without PBC."""
        for seed in [42, 123, 456]:
            atoms_per_system = [12, 15, 10, 18]
            positions_batch, _, _, _ = create_batch_systems(
                num_systems=4,
                cell_sizes=[2.0, 2.0, 2.0, 2.0],
                atoms_per_system=atoms_per_system,
                dtype=dtype,
                device=device,
                seed=seed,
                pbc_flag=False,
            )
            batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

            cutoff = 1.3
            max_neighbors = 40

            neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
                positions=positions_batch,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                pbc=None,
                cell=None,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                half_fill=half_fill,
            )

            # Basic sanity checks
            assert torch.all(num_neighbors >= 0)
            assert torch.all(num_neighbors <= max_neighbors)
            assert neighbor_matrix.device == torch.device(device)
            assert num_neighbors.device == torch.device(device)

    def test_random_systems_with_pbc(self, device, dtype, half_fill):
        """Test with random batch systems with PBC."""
        for seed in [42, 123, 456]:
            atoms_per_system = [12, 15, 10, 18]
            positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                num_systems=4,
                cell_sizes=[2.0, 2.0, 2.0, 2.0],
                atoms_per_system=atoms_per_system,
                dtype=dtype,
                device=device,
                seed=seed,
                pbc_flag=True,
            )
            batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

            cutoff = 1.3
            max_neighbors = 40

            neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
                batch_naive_neighbor_list(
                    positions=positions_batch,
                    cutoff=cutoff,
                    max_neighbors=max_neighbors,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    batch_idx=batch_idx,
                    batch_ptr=batch_ptr,
                    half_fill=half_fill,
                )
            )

            # Basic sanity checks
            assert torch.all(num_neighbors >= 0)
            assert torch.all(num_neighbors <= max_neighbors)
            assert neighbor_matrix.device == torch.device(device)
            assert neighbor_matrix_shifts.device == torch.device(device)
            assert num_neighbors.device == torch.device(device)

    def test_mixed_system_sizes(self, device, dtype, half_fill):
        """Test with very different system sizes in same batch.

        Tests batch handling of systems ranging from single atom
        to 30 atoms to verify robustness.
        """
        atoms_per_system = [2, 25, 5, 30, 1]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=5,
            cell_sizes=[2.0, 2.0, 2.0, 2.0, 2.0],
            atoms_per_system=atoms_per_system,
            dtype=dtype,
            device=device,
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.5
        max_neighbors = 50

        _, num_neighbors, _ = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
        )

        # Check that single-atom systems have no neighbors
        for i, num_atoms in enumerate(atoms_per_system):
            if num_atoms == 1:
                start_idx = batch_ptr[i].item()
                assert num_neighbors[start_idx].item() == 0, (
                    f"Single atom at index {start_idx} should have no neighbors"
                )


class TestBatchNaiveEdgeCases:
    """Edge case tests for batch naive neighbor list."""

    def test_empty_batch(self, device, dtype, half_fill):
        """Test with empty batch (no atoms)."""
        positions_empty = torch.empty(0, 3, dtype=dtype, device=device)
        batch_idx_empty = torch.empty(0, dtype=torch.int32, device=device)
        batch_ptr_empty = torch.tensor([0], dtype=torch.int32, device=device)

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions=positions_empty,
            cutoff=1.0,
            batch_idx=batch_idx_empty,
            batch_ptr=batch_ptr_empty,
            max_neighbors=10,
            pbc=None,
            cell=None,
            half_fill=half_fill,
        )

        assert neighbor_matrix.shape == (0, 10)
        assert num_neighbors.shape == (0,)

    def test_single_atom(self, device, dtype, half_fill):
        """Test single system with single atom."""
        positions_single = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        batch_idx_single = torch.tensor([0], dtype=torch.int32, device=device)
        batch_ptr_single = torch.tensor([0, 1], dtype=torch.int32, device=device)

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions=positions_single,
            cutoff=1.0,
            batch_idx=batch_idx_single,
            batch_ptr=batch_ptr_single,
            max_neighbors=10,
            pbc=None,
            cell=None,
            half_fill=half_fill,
        )

        assert num_neighbors[0].item() == 0, "Single atom should have no neighbors"

    def test_zero_cutoff(self, device, dtype, half_fill):
        """Test with zero cutoff should find no neighbors."""
        atoms_per_system = [3, 4]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=0.0,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
            pbc=None,
            cell=None,
            half_fill=half_fill,
        )

        assert torch.all(num_neighbors == 0), "Zero cutoff should find no neighbors"

    def test_single_system_batch(self, device, dtype, half_fill):
        """Test batch with only one system."""
        atoms_per_system = [10]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=1, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 30

        neighbor_matrix, num_neighbors, _ = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=pbc_batch,
            cell=cell_batch,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        # Should work correctly with single system
        assert neighbor_matrix.shape == (10, max_neighbors)
        assert num_neighbors.shape == (10,)
        assert torch.all(num_neighbors >= 0)

    def test_max_neighbors_overflow(self, device, dtype, half_fill):
        """Test behavior when max_neighbors is exceeded.

        Creates dense system with small max_neighbors to ensure
        implementation handles overflow gracefully.
        """
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 2.0  # Large cutoff to find many neighbors
        max_neighbors = 3  # Artificially small to trigger overflow

        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
            batch_naive_neighbor_list(
                positions=positions_batch,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                pbc=pbc_batch,
                cell=cell_batch,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                half_fill=half_fill,
            )
        )

        # Should produce valid output, potentially incomplete
        total_atoms = positions_batch.shape[0]
        assert torch.all(num_neighbors >= 0)
        assert neighbor_matrix.shape == (total_atoms, max_neighbors)
        assert neighbor_matrix_shifts.shape == (total_atoms, max_neighbors, 3)
        assert num_neighbors.shape == (total_atoms,)


class TestBatchNaiveErrors:
    """Input validation and error condition tests."""

    def test_mismatched_cell_pbc_cell_without_pbc(self, device, dtype, half_fill):
        """Test error when cell provided without pbc."""
        atoms_per_system = [4, 5]
        positions_batch, cell_batch, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        with pytest.raises(
            ValueError, match="If cell is provided, pbc must also be provided"
        ):
            batch_naive_neighbor_list(
                positions_batch,
                1.0,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=10,
                pbc=None,
                cell=cell_batch,
                half_fill=half_fill,
            )

    def test_mismatched_cell_pbc_pbc_without_cell(self, device, dtype, half_fill):
        """Test error when pbc provided without cell."""
        atoms_per_system = [4, 5]
        positions_batch, _, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        with pytest.raises(
            ValueError, match="If pbc is provided, cell must also be provided"
        ):
            batch_naive_neighbor_list(
                positions_batch,
                1.0,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=10,
                pbc=pbc_batch,
                cell=None,
                half_fill=half_fill,
            )


class TestBatchNaiveOutputFormats:
    """Tests for different return formats."""

    def test_return_neighbor_list_no_pbc(self, device, dtype, half_fill):
        """Test return_neighbor_list=True without PBC.

        Verifies that COO format (2, N) is returned instead of matrix format.
        """
        atoms_per_system = [4, 6, 5]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 25

        neighbor_list, neighbor_ptr = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors,
            pbc=None,
            cell=None,
            half_fill=half_fill,
            return_neighbor_list=True,
        )

        # Check COO format
        assert neighbor_list.ndim == 2
        assert neighbor_list.shape[0] == 2
        assert neighbor_list.dtype == torch.int32
        assert neighbor_ptr.dtype == torch.int32

    def test_return_neighbor_list_with_pbc(self, device, dtype, half_fill):
        """Test return_neighbor_list=True with PBC.

        Verifies that COO format with shifts is returned.
        """
        atoms_per_system = [4, 6, 5]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 25

        neighbor_list, neighbor_ptr, neighbor_shifts = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
            return_neighbor_list=True,
        )

        # Check COO format with shifts
        assert neighbor_list.ndim == 2
        assert neighbor_list.shape[0] == 2
        assert neighbor_list.dtype == torch.int32
        assert neighbor_ptr.dtype == torch.int32
        assert neighbor_shifts.dtype == torch.int32

    def test_matrix_format_default(self, device, dtype, half_fill):
        """Test default return format is matrix.

        Verifies that without return_neighbor_list=True,
        matrix format is returned.
        """
        atoms_per_system = [5, 7]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 20

        result = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=pbc_batch,
            cell=cell_batch,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        # Should return matrix format
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = result
        total_atoms = sum(atoms_per_system)
        assert neighbor_matrix.shape == (total_atoms, max_neighbors)
        assert num_neighbors.shape == (total_atoms,)
        assert neighbor_matrix_shifts.shape == (total_atoms, max_neighbors, 3)


class TestBatchNaivePerformance:
    """Performance and scaling tests."""

    def test_cutoff_scaling(self, device, dtype, half_fill):
        """Test neighbor count increases with cutoff.

        Verifies that larger cutoff values find more neighbors,
        as expected physically.
        """
        atoms_per_system = [15, 18, 20]
        max_neighbors = 100

        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Test different cutoffs
        cutoffs = [0.8, 1.2, 1.6, 2.0]
        neighbor_counts = []

        for cutoff in cutoffs:
            _, num_neighbors, _ = batch_naive_neighbor_list(
                positions_batch,
                cutoff,
                batch_idx,
                batch_ptr,
                pbc=pbc_batch,
                cell=cell_batch,
                max_neighbors=max_neighbors,
                half_fill=half_fill,
            )
            total_pairs = num_neighbors.sum().item()
            neighbor_counts.append(total_pairs)

        # Neighbor count should increase monotonically with cutoff
        for i in range(1, len(neighbor_counts)):
            assert neighbor_counts[i] >= neighbor_counts[i - 1], (
                f"Neighbor count should increase with cutoff: {neighbor_counts}"
            )

    @pytest.mark.slow
    def test_memory_scaling(self, device, half_fill):
        """Test that memory usage scales reasonably with batch size.

        Verifies that output tensor sizes are correct and
        memory allocation doesn't fail for various batch sizes.
        """
        import gc

        dtype = torch.float32
        cutoff = 1.2

        # Test different batch sizes
        batch_configs = (
            [([8, 10], 2), ([12, 15], 2)]
            if device == "cpu"
            else [([20, 25], 2), ([30, 35], 2)]
        )

        for atoms_per_system, num_systems in batch_configs:
            positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                num_systems=num_systems,
                atoms_per_system=atoms_per_system,
                dtype=dtype,
                device=device,
            )
            batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

            max_neighbors = 40

            # Clear cache before test
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

            # Run batch implementation
            neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
                batch_naive_neighbor_list(
                    positions=positions_batch,
                    cutoff=cutoff,
                    max_neighbors=max_neighbors,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    batch_idx=batch_idx,
                    batch_ptr=batch_ptr,
                    half_fill=half_fill,
                )
            )

            # Check output dimensions are correct
            total_atoms = positions_batch.shape[0]
            assert neighbor_matrix.shape == (total_atoms, max_neighbors)
            assert neighbor_matrix_shifts.shape == (total_atoms, max_neighbors, 3)
            assert num_neighbors.shape == (total_atoms,)
            assert torch.all(num_neighbors >= 0)
            assert torch.all(num_neighbors <= max_neighbors)


class TestBatchNaiveAutograd:
    """Differentiable per-pair distances/vectors with per-system grad_cell."""

    def _make_two_systems(self, device, n_per=4, box=10.0):
        torch.manual_seed(0)
        pos = torch.randn(2 * n_per, 3, dtype=torch.float64, device=device) * 0.15
        batch_idx = torch.tensor(
            [0] * n_per + [1] * n_per,
            dtype=torch.int32,
            device=device,
        )
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .repeat(2, 1, 1)
            * box
        )
        pbc = torch.ones((2, 3), dtype=torch.bool, device=device)
        return pos, cell, pbc, batch_idx, n_per

    def test_forward_returns_differentiable(self, device):
        pos, cell, pbc, batch_idx, n_per = self._make_two_systems(device)
        pos.requires_grad_(True)
        nm, nn, shifts, d, v = batch_naive_neighbor_list(
            pos,
            5.0,
            batch_idx=batch_idx,
            cell=cell,
            pbc=pbc,
            max_neighbors=8,
            max_atoms_per_system=n_per,
            return_distances=True,
            return_vectors=True,
        )
        assert d.requires_grad and v.requires_grad

    @pytest.mark.slow
    def test_gradcheck_distances_wrt_positions(self, device):
        pos, cell, pbc, batch_idx, n_per = self._make_two_systems(device)
        pos.requires_grad_(True)

        def fn(p):
            _, _, _, d, _ = batch_naive_neighbor_list(
                p,
                5.0,
                batch_idx=batch_idx,
                cell=cell,
                pbc=pbc,
                max_neighbors=8,
                max_atoms_per_system=n_per,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        torch.autograd.gradcheck(fn, (pos,), atol=1e-5, eps=1e-6, nondet_tol=1e-7)

    @pytest.mark.slow
    def test_gradcheck_distances_wrt_cell(self, device):
        pos, cell, pbc, batch_idx, n_per = self._make_two_systems(device)
        cell = cell.clone().requires_grad_(True)

        def fn(c):
            _, _, _, d, _ = batch_naive_neighbor_list(
                pos,
                5.0,
                batch_idx=batch_idx,
                cell=c,
                pbc=pbc,
                max_neighbors=8,
                max_atoms_per_system=n_per,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        torch.autograd.gradcheck(fn, (cell,), atol=1e-5, eps=1e-6, nondet_tol=1e-7)

    def test_half_fill_with_pair_outputs(self, device):
        """half_fill=True now combines with per-pair geometry outputs (batched)."""
        pos, cell, pbc, batch_idx, n_per = self._make_two_systems(device)
        nm, _nn, _sh, dist, vec = batch_naive_neighbor_list(
            pos,
            5.0,
            batch_idx=batch_idx,
            cell=cell,
            pbc=pbc,
            max_neighbors=8,
            max_atoms_per_system=n_per,
            return_distances=True,
            return_vectors=True,
            half_fill=True,
        )
        active = nm != pos.shape[0]
        assert int(active.sum()) > 0
        assert torch.all(dist[active] <= 5.0 + 1e-4)
        torch.testing.assert_close(
            dist[active], vec[active].norm(dim=-1), atol=1e-5, rtol=1e-5
        )

    def test_pair_outputs_reject_rebuild_flags(self, device):
        """rebuild_flags stays unsupported with pair outputs (stale cached geometry)."""
        pos, cell, pbc, batch_idx, n_per = self._make_two_systems(device)
        num_systems = int(batch_idx.max().item()) + 1
        with pytest.raises(NotImplementedError, match="rebuild_flags"):
            batch_naive_neighbor_list(
                pos,
                5.0,
                batch_idx=batch_idx,
                cell=cell,
                pbc=pbc,
                max_neighbors=8,
                max_atoms_per_system=n_per,
                return_distances=True,
                rebuild_flags=torch.ones(num_systems, dtype=torch.bool, device=device),
            )

    @pytest.mark.slow
    def test_gradgradcheck_second_order(self, device):
        pos, cell, pbc, batch_idx, n_per = self._make_two_systems(device)
        pos.requires_grad_(True)

        def fn(p):
            *_, d, _ = batch_naive_neighbor_list(
                p,
                5.0,
                batch_idx=batch_idx,
                cell=cell,
                pbc=pbc,
                max_neighbors=8,
                max_atoms_per_system=n_per,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        torch.autograd.gradgradcheck(
            fn,
            (pos,),
            atol=1e-4,
            eps=1e-5,
            nondet_tol=1e-7,
        )

    def test_no_grad_path_unchanged(self, device):
        pos, cell, pbc, batch_idx, n_per = self._make_two_systems(device)
        nm_a, nn_a, sh_a = batch_naive_neighbor_list(
            pos,
            5.0,
            batch_idx=batch_idx,
            cell=cell,
            pbc=pbc,
            max_neighbors=8,
            max_atoms_per_system=n_per,
        )
        nm_b, nn_b, sh_b, d_b, v_b = batch_naive_neighbor_list(
            pos,
            5.0,
            batch_idx=batch_idx,
            cell=cell,
            pbc=pbc,
            max_neighbors=8,
            max_atoms_per_system=n_per,
            return_distances=True,
            return_vectors=True,
        )
        assert not d_b.requires_grad and not v_b.requires_grad
        assert torch.equal(nn_a, nn_b)
        for i in range(nm_a.shape[0]):
            n = nn_a[i].item()
            row_a = sorted(nm_a[i, :n].tolist())
            row_b = sorted(nm_b[i, :n].tolist())
            assert row_a == row_b


class TestBatchNaiveCompile:
    """Torch compile coverage for explicit-buffer batched naive paths."""

    @pytest.mark.slow
    def test_compile_no_pbc_explicit_buffers(self, device, dtype):
        """Compile the batched no-PBC runtime with explicit outputs."""
        atoms_per_system = [5, 6]
        positions, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)
        cutoff = 1.5
        fill_value = positions.shape[0]
        max_neighbors = 30

        def alloc_outputs():
            return (
                torch.full(
                    (fill_value, max_neighbors),
                    fill_value,
                    dtype=torch.int32,
                    device=device,
                ),
                torch.zeros(fill_value, dtype=torch.int32, device=device),
            )

        def run(pos, neighbor_matrix, num_neighbors):
            return batch_naive_neighbor_list(
                pos,
                cutoff,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                fill_value=fill_value,
                neighbor_matrix=neighbor_matrix,
                num_neighbors=num_neighbors,
            )

        eager = run(positions, *alloc_outputs())
        compiled = torch.compile(run)(positions, *alloc_outputs())

        for result, expected in zip(compiled, eager):
            assert torch.equal(result, expected)

    @pytest.mark.slow
    def test_compile_pbc_explicit_buffers(self, device, dtype):
        """Compile the batched PBC runtime with prepared shift metadata."""
        atoms_per_system = [5, 6]
        positions, cell, pbc, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)
        cutoff = 1.5
        fill_value = positions.shape[0]
        max_neighbors = 30
        max_atoms_per_system = max(atoms_per_system)
        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cell, cutoff, pbc
        )

        def alloc_outputs():
            return (
                torch.full(
                    (fill_value, max_neighbors),
                    fill_value,
                    dtype=torch.int32,
                    device=device,
                ),
                torch.zeros(fill_value, dtype=torch.int32, device=device),
                torch.zeros(
                    (fill_value, max_neighbors, 3), dtype=torch.int32, device=device
                ),
                torch.empty_like(positions),
                torch.empty((fill_value, 3), dtype=torch.int32, device=device),
                torch.empty_like(cell),
            )

        def run(
            pos,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            wrapped,
            offsets,
            inv_cell,
        ):
            return batch_naive_neighbor_list(
                pos,
                cutoff,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                pbc=pbc,
                cell=cell,
                fill_value=fill_value,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                num_neighbors=num_neighbors,
                shift_range_per_dimension=shift_range,
                num_shifts_per_system=num_shifts,
                max_shifts_per_system=max_shifts,
                max_atoms_per_system=max_atoms_per_system,
                positions_wrapped_buffer=wrapped,
                per_atom_cell_offsets_buffer=offsets,
                inv_cell_buffer=inv_cell,
            )

        eager = run(positions, *alloc_outputs())
        compiled = torch.compile(run)(positions, *alloc_outputs())

        compiled_matrix, compiled_counts, compiled_shifts = compiled
        eager_matrix, eager_counts, eager_shifts = eager
        assert torch.equal(compiled_counts, eager_counts)
        for atom_index in range(fill_value):
            assert _active_neighbor_shift_rows(
                compiled_matrix,
                compiled_shifts,
                compiled_counts,
                atom_index,
            ) == _active_neighbor_shift_rows(
                eager_matrix,
                eager_shifts,
                eager_counts,
                atom_index,
            )
