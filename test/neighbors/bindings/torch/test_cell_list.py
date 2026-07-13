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

"""Tests for PyTorch bindings of cell list neighbor construction methods."""

import pytest
import torch

from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors.cell_list import (
    build_cell_list,
    cell_list,
    estimate_cell_list_sizes,
    query_cell_list,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
)

from ...test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
    create_nonorthorhombic_system,
    create_random_system,
    create_simple_cubic_system,
)
from .conftest import requires_vesin


def _search_radius_envelope(neighbor_search_radius: torch.Tensor) -> int:
    """Return the number of cell-offset combinations implied by a radius."""
    return int(torch.prod(2 * neighbor_search_radius.to("cpu") + 1).item())


class TestCellListCorrectness:
    """Tests verifying cell list correctness against reference implementations."""

    @requires_vesin
    def test_single_atom_no_neighbors(self, device, dtype):
        """Single atom system should have no neighbors."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        cell = (torch.eye(3, dtype=dtype, device=device) * 2.0).reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], device=device)
        cutoff = 3.0

        neighbor_list, _, u = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )
        i, j = neighbor_list
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)

        assert_neighbor_lists_equal((i, j, u), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_two_atom_system(self, device, dtype):
        """Two atoms within cutoff should be neighbors."""
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

    @requires_vesin
    def test_cubic_system_correctness(self, device, dtype):
        """Simple cubic lattice should match reference implementation."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        cutoff = 1.1  # Captures nearest neighbors

        neighbor_list, _, u = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )
        i, j = neighbor_list

        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i, j, u), (i_ref, j_ref, u_ref))

    @requires_vesin
    @pytest.mark.parametrize("pbc_flag", [True, False])
    def test_random_system_correctness(self, device, dtype, pbc_flag):
        """Random atomic positions should match reference implementation."""
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
            positions,
            cutoff,
            cell,
            pbc,
            max_neighbors=1500,
            return_neighbor_list=True,
        )
        i, j = neighbor_list
        ref_i, ref_j, ref_u, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i, j, u), (ref_i, ref_j, ref_u))

    @requires_vesin
    def test_random_system_distance_validity(self, device, dtype):
        """All neighbor pairs should be within cutoff distance."""
        positions, cell, pbc = create_random_system(
            num_atoms=20,
            cell_size=10.0,
            dtype=dtype,
            device=device,
            seed=42,
            pbc_flag=True,
        )
        cutoff = 5.0

        neighbor_list, _, u = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            max_neighbors=1500,
            return_neighbor_list=True,
        )
        i, j = neighbor_list

        # Verify distances are within cutoff
        if len(i) > 0:
            for idx in range(min(10, len(i))):  # Check first 10 pairs
                atom_i, atom_j = i[idx].item(), j[idx].item()
                shift = cell.squeeze(0) @ u[idx].to(dtype)
                rij = positions[atom_j] - positions[atom_i] + shift
                dist = torch.norm(rij, dim=0).item()
                assert dist < cutoff + 1e-5, f"Distance {dist} exceeds cutoff {cutoff}"

    @requires_vesin
    @pytest.mark.parametrize("pbc_flag", [[True, True, True], [False, False, False]])
    def test_scaling_random_system(self, device, dtype, pbc_flag):
        """Test correctness across different system sizes (random)."""
        for num_atoms in [10, 20]:
            positions, cell, pbc = create_random_system(
                num_atoms=num_atoms,
                cell_size=3.0,
                dtype=dtype,
                device=device,
                seed=42,
                pbc_flag=pbc_flag,
            )
            cutoff = 3.0

            estimated_density = num_atoms / cell.det().abs().item()
            max_neighbors = estimate_max_neighbors(
                cutoff, atomic_density=estimated_density * 5.0
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

            ref_i, ref_j, ref_u, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
            assert_neighbor_lists_equal((i, j, u), (ref_i, ref_j, ref_u))

    @requires_vesin
    @pytest.mark.slow
    @pytest.mark.parametrize("pbc_flag", [[True, True, True], [True, False, True]])
    @pytest.mark.parametrize("num_atoms", [50, 100])
    @pytest.mark.parametrize("cutoff", [1.0, 5.0])
    def test_scaling_nonorthorhombic_system(
        self, device, dtype, pbc_flag, num_atoms, cutoff
    ):
        """Test correctness for non-orthorhombic cells at different scales."""
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
            cutoff, atomic_density=estimated_density * 5.0
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

        ref_i, ref_j, ref_u, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i, j, u), (ref_i, ref_j, ref_u))


class TestCellListEdgeCases:
    """Tests for edge cases: empty systems, single atoms, zero cutoffs."""

    def test_empty_system_neighbor_list_format(self):
        """Empty coordinate array should return empty neighbor list format."""
        positions = torch.empty(0, 3, dtype=torch.float32)
        cell = torch.eye(3, dtype=torch.float32)
        pbc = torch.tensor([True, True, True])
        cutoff = 1.0

        results = cell_list(positions, cutoff, cell, pbc, return_neighbor_list=True)
        assert len(results) == 3
        assert results[0].shape == (2, 0)  # neighbor_list
        assert results[1].shape == (1,)  # neighbor_ptr
        assert results[2].shape == (0, 3)  # shifts

    def test_empty_system_neighbor_matrix_format(self):
        """Empty coordinate array should return empty neighbor matrix format."""
        positions = torch.empty(0, 3, dtype=torch.float32)
        cell = torch.eye(3, dtype=torch.float32)
        pbc = torch.tensor([True, True, True])
        cutoff = 1.0

        results = cell_list(positions, cutoff, cell, pbc, return_neighbor_list=False)
        assert len(results) == 3
        assert results[0].shape[0] == 0  # neighbor_matrix
        assert results[1].shape[0] == 0  # num_neighbors
        assert results[2].shape[0] == 0  # neighbor_matrix_shifts
        assert results[2].shape[2] == 3
        assert results[1].shape == (0,)

    def test_zero_cutoff_neighbor_list_format(self, device, dtype):
        """Zero cutoff should find no neighbors (list format)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 0.0

        results = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            return_neighbor_list=True,
        )
        assert len(results) == 3
        assert results[0].shape == (2, 0)  # neighbor_list
        assert results[1].shape == (9,)  # neighbor_ptr
        assert results[2].shape == (0, 3)  # shifts

    def test_zero_cutoff_neighbor_matrix_format(self, device, dtype):
        """Zero cutoff should find no neighbors (matrix format)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 0.0

        results = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            return_neighbor_list=False,
        )
        assert len(results) == 3
        assert results[0].shape[0] == 8
        assert results[1].sum().item() == 0

    def test_estimate_cell_list_sizes_empty_batch(self, device, dtype):
        """Empty batch should return valid default values."""
        cell = torch.zeros((0, 3, 3), dtype=dtype, device=device)
        pbc = torch.zeros((0, 3), dtype=torch.bool, device=device)
        cutoff = 1.0
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
        assert max_cells == 1
        assert neighbor_search_radius.shape == (3,)
        assert neighbor_search_radius.dtype == torch.int32
        assert neighbor_search_radius.device == torch.device(device)

    def test_estimate_cell_list_sizes_negative_cutoff(self, device, dtype):
        """Negative cutoff should return valid default values."""
        cell = torch.eye(3, dtype=dtype, device=device).reshape(1, 3, 3)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        cutoff = -1.0
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
        assert max_cells == 1
        assert neighbor_search_radius.shape == (3,)
        assert neighbor_search_radius.dtype == torch.int32
        assert neighbor_search_radius.device == torch.device(device)

    def test_estimate_cell_list_sizes_min_cells_one_pbc_shapes(self, device, dtype):
        """Legacy min-cell sizing should treat (3,) and (1, 3) PBC equally."""
        cell = torch.eye(3, dtype=dtype, device=device).reshape(1, 3, 3) * 11.0
        pbc_2d = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        pbc_1d = pbc_2d.reshape(3)
        cutoff = 20.0

        max_cells_2d, neighbor_search_radius_2d = estimate_cell_list_sizes(
            cell,
            pbc_2d,
            cutoff,
            min_cells_per_dimension=1,
        )
        max_cells_1d, neighbor_search_radius_1d = estimate_cell_list_sizes(
            cell,
            pbc_1d,
            cutoff,
            min_cells_per_dimension=1,
        )

        expected_radius = torch.tensor([2, 2, 2], dtype=torch.int32, device=device)
        assert max_cells_2d == 1
        assert max_cells_1d == 1
        assert torch.equal(neighbor_search_radius_2d, expected_radius)
        assert torch.equal(neighbor_search_radius_1d, expected_radius)

    def test_build_cell_list_min_cells_one_uses_legacy_grid(self, device, dtype):
        """build_cell_list should expose the legacy one-cell grid policy."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=dtype,
            device=device,
        )
        cell = torch.eye(3, dtype=dtype, device=device).reshape(1, 3, 3) * 11.0
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
        cutoff = 20.0

        max_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell,
            pbc,
            cutoff,
            min_cells_per_dimension=1,
        )
        cell_list_cache = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius,
            device,
        )

        build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            *cell_list_cache,
            min_cells_per_dimension=1,
        )

        assert torch.equal(
            cell_list_cache[0],
            torch.tensor([1, 1, 1], dtype=torch.int32, device=device),
        )

    def test_atom_centric_cell_list_allocates_legacy_grid(self, device, dtype):
        """Explicit atom-centric cell_list runs on the legacy single-cell grid.

        With ``cutoff`` larger than the box, the grid collapses to one cell; the
        atom-centric path must still enumerate the pair correctly.
        """
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=dtype,
            device=device,
        )
        cell = torch.eye(3, dtype=dtype, device=device).reshape(1, 3, 3) * 11.0
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)

        matrix, num_neighbors, _shifts = cell_list(
            positions,
            20.0,
            cell,
            pbc,
            max_neighbors=1024,
            return_neighbor_list=False,
            strategy="atom_centric",
        )

        assert matrix.shape[0] == 2
        assert num_neighbors.shape == (2,)
        # Both atoms are within cutoff of each other (and periodic images).
        assert int(num_neighbors.min()) >= 1

    def test_large_cutoff(self, device, dtype, return_neighbor_list):
        """Large cutoff that includes many neighbors should work correctly."""
        positions, cell, pbc = create_random_system(
            num_atoms=10, cell_size=2.0, dtype=dtype, device=device, seed=123
        )
        cutoff = 5.0  # Large cutoff

        results = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            max_neighbors=1500,
            return_neighbor_list=return_neighbor_list,
        )
        if return_neighbor_list:
            num_pairs = results[0].shape[1]
        else:
            num_pairs = results[1].sum().item()
        assert num_pairs >= 0


class TestCellListErrors:
    """Tests for input validation and error conditions."""

    def test_zero_volume_cell_raises_error(self, device, dtype):
        """Cell with zero volume (degenerate) should raise RuntimeError."""
        positions = torch.rand((4, 3), device=device, dtype=dtype)
        # Cell has zero volume (linearly dependent rows)
        cells = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            ],
            dtype=dtype,
            device=device,
        )
        pbc = torch.ones((1, 3), dtype=bool, device=device)
        with pytest.raises(RuntimeError, match="Cell with volume == 0"):
            _ = cell_list(
                positions,
                3.0,
                cells,
                pbc,
            )


class TestLeftHandedCells:
    """Tests for left-handed (negative determinant) cell support."""

    def _check_left_handed(
        self,
        positions,
        cell,
        pbc,
        cutoff,
        dtype,
        *,
        max_radius_envelope=None,
        require_periodic_shift=False,
    ):
        """Helper: verify neighbor list and distance equivalence for a left-handed cell."""
        assert cell.det().item() < 0, "Cell should have negative determinant"
        if max_radius_envelope is not None:
            _max_cells, neighbor_search_radius = estimate_cell_list_sizes(
                cell,
                pbc,
                cutoff,
            )
            assert (
                _search_radius_envelope(neighbor_search_radius) <= max_radius_envelope
            )

        estimated_density = positions.shape[0] / cell.det().abs().item()
        max_neighbors = estimate_max_neighbors(
            cutoff, atomic_density=estimated_density * 5.0
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
        ref_i, ref_j, ref_u, _ = brute_force_neighbors(positions, cell, pbc, cutoff)

        # Neighbor list equivalence
        assert_neighbor_lists_equal((i, j, u), (ref_i, ref_j, ref_u))
        if require_periodic_shift:
            assert torch.any(u != 0)

        # Distance equivalence
        if len(i) > 0:
            cell_sq = cell.squeeze(0)
            shifts = u.to(dtype) @ cell_sq
            dists = torch.norm(positions[j] - positions[i] + shifts, dim=1)
            ref_shifts = ref_u.to(dtype) @ cell_sq
            ref_dists = torch.norm(
                positions[ref_j] - positions[ref_i] + ref_shifts, dim=1
            )
            assert torch.allclose(
                dists.sort().values, ref_dists.sort().values, atol=1e-5
            )

    @requires_vesin
    def test_cubic_system(self, device, dtype):
        """Left-handed cubic cell should match brute force."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        cell[..., 0, :] *= -1
        self._check_left_handed(positions, cell, pbc, cutoff=1.1, dtype=dtype)

    @requires_vesin
    @pytest.mark.parametrize("pbc_flag", [True, False])
    def test_random_system(self, device, dtype, pbc_flag):
        """Left-handed random system should match brute force."""
        positions, cell, pbc = create_random_system(
            num_atoms=20,
            cell_size=10.0,
            dtype=dtype,
            device=device,
            seed=42,
            pbc_flag=pbc_flag,
        )
        cell[..., 0, :] *= -1
        self._check_left_handed(positions, cell, pbc, cutoff=5.0, dtype=dtype)

    @requires_vesin
    def test_nonorthorhombic_system(self, device, dtype):
        """Left-handed triclinic cell should match brute force."""
        cell = torch.tensor(
            [
                [
                    [-6.0, 0.2, 0.1],
                    [0.4, 7.0, 0.3],
                    [0.2, 0.5, 8.0],
                ],
            ],
            dtype=dtype,
            device=device,
        )
        frac = torch.tensor(
            [
                [0.05, 0.10, 0.10],
                [0.95, 0.10, 0.10],
                [0.50, 0.05, 0.50],
                [0.50, 0.95, 0.50],
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75],
            ],
            dtype=dtype,
            device=device,
        )
        positions = frac @ cell[0]
        pbc = torch.tensor([[True, True, True]], device=device)
        self._check_left_handed(
            positions,
            cell,
            pbc,
            cutoff=2.0,
            dtype=dtype,
            max_radius_envelope=125,
            require_periodic_shift=True,
        )

    def test_left_handed_estimate_cell_list_sizes(self, device, dtype):
        """estimate_cell_list_sizes should accept left-handed cells."""
        cell = (torch.eye(3, dtype=dtype, device=device) * 5.0).reshape(1, 3, 3)
        cell[..., 0, :] *= -1
        pbc = torch.tensor([True, True, True], device=device)

        max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, 2.0)
        assert max_cells > 0
        assert neighbor_search_radius.shape == (3,)

    def test_estimate_cell_list_sizes_rejects_nonpositive_max_nbins(
        self,
        device,
        dtype,
    ):
        """Invalid cell-bin caps should fail before launching Warp sizing kernels."""
        cell = (torch.eye(3, dtype=dtype, device=device) * 5.0).reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], device=device)

        with pytest.raises(ValueError, match="max_nbins must be positive"):
            estimate_cell_list_sizes(cell, pbc, 2.0, max_nbins=0)


class TestCellListOutputFormats:
    """Tests for different return formats and output configurations."""

    def test_no_pbc_neighbor_list_format(self, device, dtype):
        """No PBC should result in zero shifts (list format)."""
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
            return_neighbor_list=True,
        )
        u = results[-1]

        # With no PBC, all shifts should be zero
        if len(u) > 0:
            assert torch.all(u == 0), "All shifts should be zero with no PBC"

    def test_no_pbc_neighbor_matrix_format(self, device, dtype):
        """No PBC should result in zero shifts at every ACTIVE slot.

        Note: under the always-write design, ``neighbor_matrix_shifts`` is
        allocated via ``torch.empty`` and the kernel writes every active
        slot unconditionally.  Tail slots (column index >= num_neighbors[i])
        are left uninitialized — downstream consumers gate on
        ``neighbor_matrix != fill_value``, so tail values are never read.
        Assertion checks only active slots.
        """
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=3.0, dtype=dtype, device=device
        )
        pbc = torch.tensor([False, False, False], device=device)
        cutoff = 3.0

        nm, nn, nms = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            return_neighbor_list=False,
        )

        # With no PBC, every shift at every active slot must be (0, 0, 0).
        fill_value = positions.shape[0]
        mask = nm != fill_value
        if mask.any():
            assert torch.all(nms[mask] == 0), (
                "All shifts at active slots should be zero with no PBC"
            )

    @pytest.mark.parametrize("cell_pbc_shape", [0, 1])
    @pytest.mark.parametrize("fill_value", [None, -1])
    def test_mixed_pbc_neighbor_list_format(
        self, device, dtype, preallocate, cell_pbc_shape, fill_value
    ):
        """Mixed PBC (e.g., periodic in x,y only) should work correctly (list format)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        # Set PBC only in x and y dimensions
        pbc = torch.tensor([True, True, False], device=device)
        cutoff = 3.0

        if cell_pbc_shape == 0:
            cell = cell.reshape(3, 3)
            pbc = pbc.reshape(3)
        else:
            cell = cell.reshape(1, 3, 3)
            pbc = pbc.reshape(1, 3)

        if preallocate:
            max_neighbors = estimate_max_neighbors(cutoff, atomic_density=0.35 * 5.0)
            max_cells, neighbor_search_radius = estimate_cell_list_sizes(
                cell, pbc, cutoff
            )
            cell_list_cache = allocate_cell_list(
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
                return_neighbor_list=True,
                cells_per_dimension=cell_list_cache[0],
                neighbor_search_radius=cell_list_cache[1],
                atom_periodic_shifts=cell_list_cache[2],
                atom_to_cell_mapping=cell_list_cache[3],
                atoms_per_cell_count=cell_list_cache[4],
                cell_atom_start_indices=cell_list_cache[5],
                cell_atom_list=cell_list_cache[6],
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                num_neighbors=num_neighbors,
            )
        else:
            max_neighbors = estimate_max_neighbors(cutoff, atomic_density=0.35 * 5.0)
            results = cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                max_neighbors=max_neighbors,
                fill_value=fill_value,
                return_neighbor_list=True,
            )

        neighbor_list, _, u = results
        assert len(neighbor_list) == 2
        # z-direction should have no shifts (no PBC)
        assert u[:, 2].sum().item() == 0
        # x-direction should have some shifts (PBC enabled)
        assert (u[:, 0] ** 2).sum().item() > 0

    @pytest.mark.parametrize("fill_value", [None, -1])
    def test_mixed_pbc_neighbor_matrix_format(
        self, device, dtype, preallocate, fill_value
    ):
        """Mixed PBC (e.g., periodic in x,y only) should work correctly (matrix format)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        # Set PBC only in x and y dimensions
        pbc = torch.tensor([True, True, False], device=device)
        cutoff = 3.0

        if preallocate:
            max_neighbors = estimate_max_neighbors(cutoff, atomic_density=0.35 * 5.0)
            max_cells, neighbor_search_radius = estimate_cell_list_sizes(
                cell, pbc, cutoff
            )
            cell_list_cache = allocate_cell_list(
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
                return_neighbor_list=False,
                cells_per_dimension=cell_list_cache[0],
                neighbor_search_radius=cell_list_cache[1],
                atom_periodic_shifts=cell_list_cache[2],
                atom_to_cell_mapping=cell_list_cache[3],
                atoms_per_cell_count=cell_list_cache[4],
                cell_atom_start_indices=cell_list_cache[5],
                cell_atom_list=cell_list_cache[6],
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                num_neighbors=num_neighbors,
            )
        else:
            max_neighbors = estimate_max_neighbors(cutoff, atomic_density=0.35 * 5.0)
            results = cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                max_neighbors=max_neighbors,
                fill_value=fill_value,
                return_neighbor_list=False,
            )

        nm, _, u = results
        # Under the always-write-shifts contract only the active range of
        # ``u`` is written; the tail is uninitialised.  Mask by the
        # fill_value-padded neighbor_matrix (matches the batch sibling).
        fv = positions.shape[0] if fill_value is None else fill_value
        mask = nm != fv
        # z-direction should have no shifts (no PBC)
        assert int(u[..., 2][mask].sum().item()) == 0
        # x-direction should have some shifts (PBC enabled)
        assert int((u[..., 0][mask] ** 2).sum().item()) > 0

    def test_dtype_consistency(self, dtype, return_neighbor_list):
        """Output dtypes should be consistent (int32 for indices)."""
        positions = torch.randn(5, 3, dtype=dtype)
        cell = (torch.eye(3, dtype=dtype) * 2.0).reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], dtype=torch.bool)
        cutoff = 1.5

        results = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=return_neighbor_list
        )

        for result in results:
            assert result.dtype == torch.int32

    def test_device_consistency(self, device, return_neighbor_list):
        """Outputs should be on the same device as inputs."""
        positions = torch.randn(5, 3, device=device)
        cell = torch.eye(3, device=device).reshape(1, 3, 3) * 2.0
        pbc = torch.tensor([True, True, True], device=device)
        cutoff = 1.5

        results = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=return_neighbor_list
        )
        for result in results:
            assert result.device == torch.device(device)


def _sorted_pairs(neighbor_list_coo):
    """Sort COO (source, target) pairs for order-independent comparison."""
    sources = neighbor_list_coo[0]
    targets = neighbor_list_coo[1]
    keys = sources.to(torch.int64) * (
        int(targets.max().item()) + 1 if targets.numel() else 1
    ) + targets.to(torch.int64)
    order = torch.argsort(keys)
    return torch.stack([sources[order], targets[order]], dim=0)


class TestCellListAtomCentricPathEquivalence:
    """``atom_centric_path="sorted"`` matches ``"direct"`` (CUDA-only)."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="cell_list atom-centric direct/sorted path requires CUDA",
    )
    def test_sorted_matches_direct(self):
        """Sorted and direct atom-centric paths produce the same pair set."""
        torch.manual_seed(0)
        device = "cuda"
        positions = torch.rand(300, 3, dtype=torch.float32, device=device) * 20.0
        cell = (torch.eye(3, dtype=torch.float32, device=device) * 20.0).reshape(
            1, 3, 3
        )
        pbc = torch.tensor([[True, True, True]], device=device)
        cutoff = 5.0

        direct = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            strategy="atom_centric",
            atom_centric_path="direct",
            return_neighbor_list=True,
        )
        sorted_ = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            strategy="atom_centric",
            atom_centric_path="sorted",
            return_neighbor_list=True,
        )
        torch.testing.assert_close(
            _sorted_pairs(sorted_[0]).cpu(), _sorted_pairs(direct[0]).cpu()
        )


class TestCellListCompile:
    """Tests for torch.compile compatibility."""

    @pytest.mark.slow
    def test_build_cell_list_compile(self, device, dtype):
        """build_cell_list should be compatible with torch.compile."""
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

    @pytest.mark.parametrize("pbc_flag", [False, True])
    @pytest.mark.slow
    def test_query_cell_list_compile(self, device, dtype, pbc_flag):
        """query_cell_list should be compatible with torch.compile."""
        positions, cell, pbc = create_simple_cubic_system(dtype=dtype, device=device)
        cutoff = 3.0
        pbc = torch.tensor([pbc_flag, pbc_flag, pbc_flag], device=device)
        # Build cell list first
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell,
            pbc,
            cutoff,
        )
        density = positions.shape[0] / cell.det()
        max_neighbors = estimate_max_neighbors(cutoff, atomic_density=density * 2.0)
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

        assert torch.equal(num_neighbors_uncompiled, num_neighbors_compiled), (
            "Number of neighbors mismatch"
        )

    @pytest.mark.slow
    def test_query_cell_list_target_indices_compile(self, device, dtype):
        """target_indices query path should stay behind a custom-op boundary."""
        positions, cell, pbc = create_simple_cubic_system(dtype=dtype, device=device)
        cutoff = 3.0
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell,
            pbc,
            cutoff,
        )
        max_neighbors = estimate_max_neighbors(cutoff)
        target_indices = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)

        def allocate_outputs():
            return (
                torch.full(
                    (target_indices.shape[0], max_neighbors),
                    -1,
                    dtype=torch.int32,
                    device=device,
                ),
                torch.zeros(
                    (target_indices.shape[0], max_neighbors, 3),
                    dtype=torch.int32,
                    device=device,
                ),
                torch.zeros(
                    (target_indices.shape[0],), dtype=torch.int32, device=device
                ),
            )

        cache_eager = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius.clone(),
            device,
        )
        build_cell_list(positions, cutoff, cell, pbc, *cache_eager)
        eager_outputs = allocate_outputs()
        query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            *cache_eager,
            *eager_outputs,
            target_indices=target_indices,
        )

        cache_compiled = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius.clone(),
            device,
        )
        compiled_outputs = allocate_outputs()

        @torch.compile
        def compiled_query(
            positions,
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
            target_indices,
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
                target_indices=target_indices,
            )

        compiled_query(
            positions,
            cell,
            pbc,
            *cache_compiled,
            *compiled_outputs,
            target_indices,
        )

        assert torch.equal(eager_outputs[2], compiled_outputs[2])
        assert torch.equal(
            torch.sort(eager_outputs[0]).values,
            torch.sort(compiled_outputs[0]).values,
        )


class TestCellListComponentsAPI:
    """Tests for the modular cell list API functions."""

    def test_build_and_query_cell_list(self, device, dtype):
        """Building and querying cell list separately should work correctly."""
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

    def test_estimate_max_neighbors(self, device, dtype):
        """Max neighbors estimation should return reasonable values."""
        positions, cell, _ = create_simple_cubic_system(dtype=dtype, device=device)
        cutoff = 1.1
        density = positions.shape[0] / cell.det().abs().item()
        max_neighbors = estimate_max_neighbors(cutoff, atomic_density=density * 5.0)
        assert max_neighbors > 0
        assert isinstance(max_neighbors, int)


class TestCellListSelectiveRebuildFlags:
    """Test selective rebuild (rebuild_flags) for cell list torch bindings."""

    def test_no_rebuild_preserves_data(self, device, dtype):
        """Flag=False: neighbor data should remain unchanged."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        cell = cell.reshape(1, 3, 3)
        pbc_1d = pbc.squeeze(0)
        cutoff = 1.1

        max_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell, pbc_1d, cutoff
        )
        cell_list_cache = allocate_cell_list(
            positions.shape[0], max_cells, neighbor_search_radius, device
        )
        (
            cells_per_dimension,
            neighbor_search_radius_t,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = cell_list_cache

        build_cell_list(
            positions,
            cutoff,
            cell,
            pbc_1d,
            cells_per_dimension,
            neighbor_search_radius_t,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )

        max_neighbors = 10
        nm = torch.full(
            (positions.shape[0], max_neighbors), -1, dtype=torch.int32, device=device
        )
        nm_shifts = torch.zeros(
            (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
        )
        nn = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)

        query_cell_list(
            positions,
            cutoff,
            cell,
            pbc_1d,
            cells_per_dimension,
            neighbor_search_radius_t,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            nm,
            nm_shifts,
            nn,
        )

        saved_nm = nm.clone()
        saved_nn = nn.clone()

        rebuild_flags = torch.zeros(1, dtype=torch.bool, device=device)
        query_cell_list(
            positions,
            cutoff,
            cell,
            pbc_1d,
            cells_per_dimension,
            neighbor_search_radius_t,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            nm,
            nm_shifts,
            nn,
            rebuild_flags=rebuild_flags,
        )

        assert torch.equal(nn, saved_nn), (
            "num_neighbors must be unchanged when rebuild_flags is False"
        )
        for i in range(positions.shape[0]):
            n = nn[i].item()
            assert torch.equal(nm[i, :n], saved_nm[i, :n]), (
                f"neighbor_matrix row {i} should be unchanged"
            )

    def test_rebuild_updates_data(self, device, dtype):
        """Flag=True: result should match a fresh full rebuild."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        cell = cell.reshape(1, 3, 3)
        pbc_1d = pbc.squeeze(0)
        cutoff = 1.1

        max_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell, pbc_1d, cutoff
        )
        cell_list_cache = allocate_cell_list(
            positions.shape[0], max_cells, neighbor_search_radius, device
        )
        (
            cells_per_dimension,
            neighbor_search_radius_t,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = cell_list_cache

        build_cell_list(
            positions,
            cutoff,
            cell,
            pbc_1d,
            cells_per_dimension,
            neighbor_search_radius_t,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )

        max_neighbors = 10

        # Reference: full build
        nm_ref = torch.full(
            (positions.shape[0], max_neighbors), -1, dtype=torch.int32, device=device
        )
        nm_ref_shifts = torch.zeros(
            (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
        )
        nn_ref = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)
        query_cell_list(
            positions,
            cutoff,
            cell,
            pbc_1d,
            cells_per_dimension,
            neighbor_search_radius_t,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            nm_ref,
            nm_ref_shifts,
            nn_ref,
        )

        # Selective rebuild with flag=True
        nm_sel = torch.full(
            (positions.shape[0], max_neighbors), 99, dtype=torch.int32, device=device
        )
        nm_sel_shifts = torch.zeros(
            (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
        )
        nn_sel = torch.full((positions.shape[0],), 99, dtype=torch.int32, device=device)

        rebuild_flags = torch.ones(1, dtype=torch.bool, device=device)
        query_cell_list(
            positions,
            cutoff,
            cell,
            pbc_1d,
            cells_per_dimension,
            neighbor_search_radius_t,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            nm_sel,
            nm_sel_shifts,
            nn_sel,
            rebuild_flags=rebuild_flags,
        )

        assert torch.equal(nn_sel, nn_ref), (
            "num_neighbors should match full rebuild when flag=True"
        )

    @pytest.mark.parametrize("pbc_flag", [[True, True, True], [False, False, False]])
    def test_nonselective_matches_true_rebuild_flag(self, device, dtype, pbc_flag):
        """Non-selective query output should match selective flag=True output."""
        positions, cell, pbc = create_random_system(
            num_atoms=12,
            cell_size=6.0,
            dtype=dtype,
            device=device,
            seed=7,
            pbc_flag=pbc_flag,
        )
        cell = cell.reshape(1, 3, 3)
        pbc = pbc.reshape(3)
        cutoff = 2.0

        max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
        cell_list_cache = allocate_cell_list(
            positions.shape[0], max_cells, neighbor_search_radius, device
        )
        (
            cells_per_dimension,
            neighbor_search_radius_t,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = cell_list_cache

        build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            cells_per_dimension,
            neighbor_search_radius_t,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )

        max_neighbors = 32
        nm_ref = torch.full(
            (positions.shape[0], max_neighbors), -1, dtype=torch.int32, device=device
        )
        shifts_ref = torch.zeros(
            (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
        )
        nn_ref = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)
        query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            cells_per_dimension,
            neighbor_search_radius_t,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            nm_ref,
            shifts_ref,
            nn_ref,
            strategy="atom_centric",
        )

        nm_sel = torch.full_like(nm_ref, -1)
        shifts_sel = torch.zeros_like(shifts_ref)
        nn_sel = torch.full_like(nn_ref, 99)
        rebuild_flags = torch.ones(1, dtype=torch.bool, device=device)
        query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            cells_per_dimension,
            neighbor_search_radius_t,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            nm_sel,
            shifts_sel,
            nn_sel,
            rebuild_flags=rebuild_flags,
            strategy="atom_centric",
        )

        assert torch.equal(nn_sel, nn_ref)
        for row_idx, count in enumerate(nn_ref.detach().cpu().tolist()):
            active_ref = [
                (
                    int(nm_ref[row_idx, col].item()),
                    tuple(int(x) for x in shifts_ref[row_idx, col].cpu().tolist()),
                )
                for col in range(int(count))
            ]
            active_sel = [
                (
                    int(nm_sel[row_idx, col].item()),
                    tuple(int(x) for x in shifts_sel[row_idx, col].cpu().tolist()),
                )
                for col in range(int(count))
            ]
            assert sorted(active_sel) == sorted(active_ref)


class TestCellListAutograd:
    """Autograd path for per-pair distances and vectors.

    When ``return_distances`` / ``return_vectors`` are set, the wrapper
    appends the differentiable per-pair tensors to its return tuple.  The
    backward is computed by ``_NeighborDistanceVectorFn`` via reconstruction
    from positions and cell, which also enables second-order autograd.
    """

    def _make_system(self, device):
        # fp64, small N, atoms well inside cutoff so the neighbor topology is
        # stable under the central-difference perturbations gradcheck applies.
        torch.manual_seed(0)
        N = 6
        positions = torch.randn(N, 3, dtype=torch.float64, device=device) * 0.4
        cell = (torch.eye(3, dtype=torch.float64, device=device) * 4.0).unsqueeze(0)
        pbc = torch.tensor([[True, True, True]], device=device)
        return positions, cell, pbc

    def test_forward_returns_differentiable_distances_and_vectors(self, device):
        positions, cell, pbc = self._make_system(device)
        positions.requires_grad_(True)
        cell.requires_grad_(True)
        nm, nn, shifts, dists, vecs = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            return_distances=True,
            return_vectors=True,
        )
        assert dists.requires_grad and vecs.requires_grad
        assert dists.shape == (positions.shape[0], nm.shape[1])
        assert vecs.shape == (positions.shape[0], nm.shape[1], 3)

    def test_coo_distances_vectors_aligned_and_differentiable(self, device):
        """``return_neighbor_list=True`` repacks per-pair geometry into COO
        order aligned with the neighbor list and keeps the autograd link."""
        positions, cell, pbc = self._make_system(device)
        positions.requires_grad_(True)
        nl, _nptr, nl_shifts, dists, vecs = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            return_neighbor_list=True,
            return_distances=True,
            return_vectors=True,
        )
        num_pairs = nl.shape[1]
        assert dists.shape == (num_pairs,)
        assert vecs.shape == (num_pairs, 3)
        # COO geometry must index-align with the neighbor list.
        i_idx, j_idx = nl[0].long(), nl[1].long()
        rij = (
            positions[j_idx]
            - positions[i_idx]
            + nl_shifts.to(positions.dtype) @ cell[0]
        )
        assert torch.allclose(rij, vecs)
        assert torch.allclose(rij.norm(dim=1), dists)
        # Autograd still flows through the COO outputs.
        dists.pow(2).sum().backward()
        assert positions.grad is not None and torch.isfinite(positions.grad).all()

    def test_return_tuple_shape_extends_with_flags(self, device):
        """Tuple shape changes only when pair-output flags are set; the
        non-flag path keeps the 0.3.1 3-tuple."""
        positions, cell, pbc = self._make_system(device)
        out_default = cell_list(positions, 1.5, cell, pbc)
        assert len(out_default) == 3

        out_d = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            return_distances=True,
        )
        assert len(out_d) == 4

        out_v = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            return_vectors=True,
        )
        assert len(out_v) == 4

        out_dv = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            return_distances=True,
            return_vectors=True,
        )
        assert len(out_dv) == 5

    @pytest.mark.slow
    def test_gradcheck_distances_wrt_positions(self, device):
        positions, cell, pbc = self._make_system(device)
        positions.requires_grad_(True)

        def fn(p):
            _, _, _, d, _ = cell_list(
                p,
                1.5,
                cell,
                pbc,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        # nondet_tol covers atomic_add ordering nondeterminism on CUDA.
        assert torch.autograd.gradcheck(
            fn,
            (positions,),
            atol=1e-5,
            eps=1e-6,
            nondet_tol=1e-7,
        )

    @pytest.mark.slow
    def test_gradcheck_distances_wrt_cell(self, device):
        positions, cell, pbc = self._make_system(device)
        cell = cell.clone().requires_grad_(True)

        def fn(c):
            _, _, _, d, _ = cell_list(
                positions,
                1.5,
                c,
                pbc,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        assert torch.autograd.gradcheck(fn, (cell,), atol=1e-5, eps=1e-6)

    @pytest.mark.slow
    def test_gradcheck_vectors_wrt_positions(self, device):
        positions, cell, pbc = self._make_system(device)
        positions.requires_grad_(True)

        def fn(p):
            _, _, _, _, v = cell_list(
                p,
                1.5,
                cell,
                pbc,
                return_distances=True,
                return_vectors=True,
            )
            # Linear-in-r loss exercises grad_r contributions in backward.
            return v.sum()

        assert torch.autograd.gradcheck(fn, (positions,), atol=1e-5, eps=1e-6)

    @pytest.mark.slow
    def test_gradgradcheck_distances_second_order(self, device):
        """Second-order autograd via reconstruction in backward."""
        positions, cell, pbc = self._make_system(device)
        # Smaller N for faster gradgradcheck.
        positions = positions[:5].clone().requires_grad_(True)

        def fn(p):
            _, _, _, d, _ = cell_list(
                p,
                1.5,
                cell,
                pbc,
                return_distances=True,
                return_vectors=True,
            )
            # Non-linear loss so the Hessian is non-trivial.
            return d.pow(2).sum()

        # nondet_tol covers atomic_add ordering nondeterminism on CUDA.
        assert torch.autograd.gradgradcheck(
            fn,
            (positions,),
            atol=1e-4,
            eps=1e-6,
            nondet_tol=1e-7,
        )

    def test_hessian_vector_product_smoke(self, device):
        """create_graph=True allows constructing an HVP without errors."""
        positions, cell, pbc = self._make_system(device)
        positions = positions[:5].clone().requires_grad_(True)

        _, _, _, d, _ = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            return_distances=True,
            return_vectors=True,
        )
        loss = d.pow(2).sum()
        grad_pos = torch.autograd.grad(loss, positions, create_graph=True)[0]
        # HVP with a random direction vector.
        v = torch.randn_like(positions)
        hvp = torch.autograd.grad((grad_pos * v).sum(), positions, retain_graph=False)[
            0
        ]
        assert torch.isfinite(hvp).all()
        assert hvp.shape == positions.shape

    def test_no_grad_path_unchanged(self, device):
        """When inputs don't require grad, outputs are plain tensors and the
        active portion of the return is numerically equal to the
        non-autograd path.

        Inactive matrix slots (column >= num_neighbors) are uninitialized by
        the kernel and may hold different garbage between independent
        ``torch.empty`` allocations — compare only the active slots.
        """
        positions, cell, pbc = self._make_system(device)

        nm_a, nn_a, sh_a = cell_list(positions, 1.5, cell, pbc)
        nm_b, nn_b, sh_b, d_b, v_b = cell_list(
            positions,
            1.5,
            cell,
            pbc,
            return_distances=True,
            return_vectors=True,
        )
        assert not d_b.requires_grad and not v_b.requires_grad
        assert torch.equal(nn_a, nn_b)
        # Active-slot comparison. The autograd-capable path may emit each row in
        # a different order, so compare the active (neighbor, shift) tuples.
        for row_idx, count in enumerate(nn_a.detach().cpu().tolist()):
            active_a = [
                (
                    int(nm_a[row_idx, col].item()),
                    tuple(int(x) for x in sh_a[row_idx, col].detach().cpu().tolist()),
                )
                for col in range(int(count))
            ]
            active_b = [
                (
                    int(nm_b[row_idx, col].item()),
                    tuple(int(x) for x in sh_b[row_idx, col].detach().cpu().tolist()),
                )
                for col in range(int(count))
            ]
            assert sorted(active_a) == sorted(active_b)


class TestEstimateMaxNeighborsDefaults:
    """Defaults for estimate_max_neighbors (pure Python, framework-agnostic)."""

    def test_nonpositive_cutoff_returns_zero(self):
        assert estimate_max_neighbors(0.0) == 0
        assert estimate_max_neighbors(-1.0) == 0

    @pytest.mark.parametrize("cutoff", [0.5, 1.5, 2.67])
    def test_short_cutoff_respects_default_lower_bound(self, cutoff):
        """Short cutoffs are floored at the default lower bound (16)."""
        assert estimate_max_neighbors(cutoff) >= 16

    @pytest.mark.parametrize("cutoff", [0.5, 1.5, 2.67])
    def test_lower_bound_kwarg_raises_floor(self, cutoff):
        """max_neighbors_lower_bound raises the floor for short cutoffs where the
        density estimate would otherwise fall below it."""
        assert estimate_max_neighbors(cutoff, max_neighbors_lower_bound=64) == 64

    def test_lower_bound_does_not_cap_estimate(self):
        """A lower bound below the density estimate leaves the estimate
        unchanged (it is a floor, not a cap)."""
        cutoff = 8.5
        assert estimate_max_neighbors(
            cutoff, max_neighbors_lower_bound=16
        ) == estimate_max_neighbors(cutoff, max_neighbors_lower_bound=1)

    def test_density_scales_estimate(self):
        """A higher atomic_density must raise the estimate (the sole knob for
        denser / clustered systems)."""
        cutoff = 5.0
        assert estimate_max_neighbors(
            cutoff, atomic_density=0.8
        ) > estimate_max_neighbors(cutoff, atomic_density=0.2)

    @pytest.mark.parametrize("cutoff", [1.0, 3.0, 5.0, 8.5])
    def test_result_is_positive_multiple_of_16(self, cutoff):
        result = estimate_max_neighbors(cutoff)
        assert result > 0
        assert result % 16 == 0
        assert isinstance(result, int)

    def test_safety_factor_deprecated_and_folded(self):
        """safety_factor is deprecated: it must warn and fold into atomic_density
        (identical result to scaling the density directly)."""
        cutoff = 5.0
        with pytest.warns(DeprecationWarning, match="safety_factor"):
            result = estimate_max_neighbors(
                cutoff, atomic_density=0.2, safety_factor=3.0
            )
        assert result == estimate_max_neighbors(cutoff, atomic_density=0.2 * 3.0)

    def test_no_warning_without_safety_factor(self):
        """The default path must not emit a deprecation warning."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            estimate_max_neighbors(5.0, atomic_density=0.3)
