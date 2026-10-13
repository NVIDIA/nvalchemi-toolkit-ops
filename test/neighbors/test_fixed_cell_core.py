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

"""Focused Warp-level checks for prepared fixed-cell geometry reuse."""

import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.cell_list import build_cell_list
from nvalchemiops.neighbors.cell_list.launchers import batch_build_cell_list
from nvalchemiops.neighbors.cluster_tile import (
    build_cluster_tile_list,
    query_cluster_tile,
    query_cluster_tile_coo,
)
from nvalchemiops.neighbors.cluster_tile.launchers import (
    prepare_cluster_tile_geometry,
)
from nvalchemiops.neighbors.naive import naive_neighbor_matrix_pbc

pytestmark = pytest.mark.gpu


@pytest.fixture
def device():
    """Return the CUDA device used by Warp kernel tests."""
    if not torch.cuda.is_available():
        pytest.skip("fixed-cell core tests require CUDA")
    return "cuda:0"


def _wp_view(tensor: torch.Tensor, dtype: type) -> wp.array:
    """Return a contiguous Torch tensor as a Warp array view."""
    return wp.from_torch(tensor.contiguous(), dtype=dtype)


def test_naive_supplied_inverse_buffer_remains_scratch_by_default(device):
    """The public PBC launcher recomputes a supplied inverse buffer by default."""
    cell = torch.eye(3, dtype=torch.float32, device=device).mul_(4.0).unsqueeze(0)
    inverse = torch.full_like(cell, 7.0)
    positions = torch.tensor(
        [[0.1, 0.2, 0.2], [3.7, 0.2, 0.2]],
        dtype=torch.float32,
        device=device,
    )
    wrapped = torch.empty_like(positions)
    offsets = torch.empty((2, 3), dtype=torch.int32, device=device)
    pbc = torch.ones((1, 3), dtype=torch.bool, device=device)
    shift_range = torch.ones((1, 3), dtype=torch.int32, device=device)
    max_shifts = 27
    max_neighbors = max_shifts
    neighbors = torch.full((2, max_neighbors), -1, dtype=torch.int32, device=device)
    neighbor_shifts = torch.zeros(
        (2, max_neighbors, 3), dtype=torch.int32, device=device
    )
    counts = torch.zeros((2,), dtype=torch.int32, device=device)

    naive_neighbor_matrix_pbc(
        _wp_view(positions, wp.vec3f),
        0.5,
        _wp_view(cell, wp.mat33f),
        _wp_view(shift_range, wp.vec3i),
        max_shifts,
        _wp_view(neighbors, wp.int32),
        _wp_view(neighbor_shifts, wp.vec3i),
        _wp_view(counts, wp.int32),
        wp.float32,
        device,
        positions_wrapped_buffer=_wp_view(wrapped, wp.vec3f),
        per_atom_cell_offsets_buffer=_wp_view(offsets, wp.vec3i),
        inv_cell_buffer=_wp_view(inverse, wp.mat33f),
        pbc=_wp_view(pbc, wp.bool),
    )

    torch.testing.assert_close(inverse, torch.linalg.inv(cell))
    torch.testing.assert_close(
        counts, torch.tensor([1, 1], dtype=torch.int32, device=device)
    )
    assert neighbors[0, 0].item() == 1
    assert neighbors[1, 0].item() == 0


def test_cell_list_fixed_geometry_reuses_grid_and_matches_dynamic_build(device):
    """Fixed cell-list construction reads cached grid and inverse arrays."""
    positions = torch.tensor(
        [[0.2, 0.3, 0.4], [1.2, 0.4, 0.5], [2.4, 2.6, 2.5], [3.5, 3.1, 0.7]],
        dtype=torch.float32,
        device=device,
    )
    cell = torch.eye(3, dtype=torch.float32, device=device).mul_(4.0).unsqueeze(0)
    inverse = torch.linalg.inv(cell)
    pbc = torch.ones((3,), dtype=torch.bool, device=device)
    cutoff = 1.5
    num_cells = 64

    def build(fixed: bool, grid: torch.Tensor):
        """Run one cell-list build and return its observable arrays."""
        atom_shifts = torch.zeros((4, 3), dtype=torch.int32, device=device)
        atom_to_cell = torch.zeros_like(atom_shifts)
        occupancy = torch.zeros((num_cells,), dtype=torch.int32, device=device)
        starts = torch.zeros_like(occupancy)
        atom_list = torch.zeros((4,), dtype=torch.int32, device=device)
        build_cell_list(
            _wp_view(positions, wp.vec3f),
            _wp_view(cell, wp.mat33f),
            _wp_view(pbc, wp.bool),
            cutoff,
            _wp_view(grid, wp.int32),
            _wp_view(atom_shifts, wp.vec3i),
            _wp_view(atom_to_cell, wp.vec3i),
            _wp_view(occupancy, wp.int32),
            _wp_view(starts, wp.int32),
            _wp_view(atom_list, wp.int32),
            wp.float32,
            device,
            fixed_cell=fixed,
            inv_cell=_wp_view(inverse, wp.mat33f) if fixed else None,
        )
        return atom_shifts, atom_to_cell, occupancy, starts, atom_list

    dynamic_grid = torch.zeros((3,), dtype=torch.int32, device=device)
    dynamic_outputs = build(False, dynamic_grid)
    cached_grid = dynamic_grid.clone()
    fixed_outputs = build(True, cached_grid)

    for actual, expected in zip(fixed_outputs[:-1], dynamic_outputs[:-1], strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        fixed_outputs[-1].sort().values,
        dynamic_outputs[-1].sort().values,
    )
    torch.testing.assert_close(cached_grid, dynamic_grid)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch_cell_list_fixed_and_dynamic_build_abi(device, dtype):
    """Batched dynamic and fixed launches preserve their grid and inverse inputs."""
    positions = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [7.8, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [4.8, 0.2, 0.3],
            [9.7, 0.2, 0.3],
        ],
        dtype=dtype,
        device=device,
    )
    sides = torch.tensor([8.0, 6.0, 10.0], dtype=dtype, device=device)
    cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * sides[:, None, None]
    inverse = torch.linalg.inv(cell)
    pbc = torch.ones((3, 3), dtype=torch.bool, device=device)
    batch_idx = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int32, device=device)
    cutoff = 1.0
    max_cells_per_system = 512

    def build(
        *,
        fixed: bool,
        grid: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ):
        """Run one real batched Warp build and return its observable arrays."""
        if grid is None:
            grid = torch.zeros((3, 3), dtype=torch.int32, device=device)
        cells_per_system = grid.prod(dim=1).to(torch.int32)
        if offsets is None:
            offsets = torch.cat(
                (
                    torch.zeros((1,), dtype=torch.int32, device=device),
                    cells_per_system.cumsum(0, dtype=torch.int32)[:-1],
                )
            )
        cell_offsets = offsets.clone()
        atom_shifts = torch.zeros((5, 3), dtype=torch.int32, device=device)
        atom_to_cell = torch.zeros_like(atom_shifts)
        occupancy = torch.zeros(
            (3 * max_cells_per_system,), dtype=torch.int32, device=device
        )
        starts = torch.zeros_like(occupancy)
        cell_occupancy = torch.zeros_like(occupancy)
        atom_list = torch.zeros((5,), dtype=torch.int32, device=device)
        batch_build_cell_list(
            _wp_view(positions, wp.vec3f if dtype == torch.float32 else wp.vec3d),
            _wp_view(cell, wp.mat33f if dtype == torch.float32 else wp.mat33d),
            _wp_view(pbc, wp.bool),
            cutoff,
            _wp_view(batch_idx, wp.int32),
            _wp_view(grid, wp.vec3i),
            _wp_view(cell_offsets, wp.int32),
            _wp_view(cells_per_system, wp.int32),
            _wp_view(atom_shifts, wp.vec3i),
            _wp_view(atom_to_cell, wp.vec3i),
            _wp_view(occupancy, wp.int32),
            _wp_view(starts, wp.int32),
            _wp_view(atom_list, wp.int32),
            wp.float32 if dtype == torch.float32 else wp.float64,
            device,
            fixed_cell=fixed,
            inv_cell=_wp_view(
                inverse, wp.mat33f if dtype == torch.float32 else wp.mat33d
            )
            if fixed
            else None,
        )
        cell_occupancy[:-1] = starts[1:] - starts[:-1]
        cell_occupancy[-1] = positions.shape[0] - starts[-1]
        return (
            grid,
            cell_offsets,
            cells_per_system,
            cell_occupancy,
            atom_shifts,
            atom_to_cell,
            starts,
            atom_list,
        )

    dynamic = build(fixed=False)
    fixed_grid = dynamic[0].clone()
    fixed_counts = fixed_grid.prod(dim=1).to(torch.int32)
    fixed_offsets = torch.cat(
        (
            torch.zeros((1,), dtype=torch.int32, device=device),
            fixed_counts.cumsum(0, dtype=torch.int32)[:-1],
        )
    )
    fixed = build(fixed=True, grid=fixed_grid, offsets=fixed_offsets)

    for actual, expected in zip(fixed, dynamic, strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(dynamic[2], dynamic[0].prod(dim=1).to(torch.int32))
    torch.testing.assert_close(dynamic[1], fixed_offsets)
    assert int(dynamic[2][1]) > 0
    torch.testing.assert_close(dynamic[1][2], dynamic[1][1] + dynamic[2][1])
    for system, atom_count in ((0, 2), (1, 0), (2, 3)):
        start = int(dynamic[1][system])
        end = start + int(dynamic[2][system])
        assert int(dynamic[3][start:end].sum()) == atom_count


def test_cluster_tile_prepared_geometry_matches_dynamic_matrix_and_coo(device):
    """Prepared QR and certificates preserve matrix and COO pair sets."""
    natom = 32
    atom_ids = torch.arange(natom, dtype=torch.int32, device=device)
    coords = torch.arange(natom, dtype=torch.float32, device=device)
    positions = torch.stack(
        (
            torch.remainder(coords, 4) * 0.45 + 0.2,
            torch.remainder(torch.floor(coords / 4), 4) * 0.45 + 0.3,
            torch.floor(coords / 16) * 0.7 + 0.4,
        ),
        dim=1,
    )
    cell = torch.tensor(
        [[[5.0, 0.1, 0.0], [0.3, 4.7, 0.1], [0.0, 0.2, 4.5]]],
        dtype=torch.float32,
        device=device,
    )
    inverse = torch.linalg.inv(cell)
    cutoff = 1.0
    qr = torch.empty((1, 15), dtype=torch.float32, device=device)
    axis_aligned = torch.empty((1,), dtype=torch.bool, device=device)
    fractional_certified = torch.empty_like(axis_aligned)
    height_certified = torch.empty_like(axis_aligned)
    bbox_bounds = torch.empty((1, 3), dtype=torch.float32, device=device)
    prepare_cluster_tile_geometry(
        _wp_view(cell, wp.mat33f),
        _wp_view(inverse, wp.mat33f),
        wp.float32(cutoff * cutoff),
        _wp_view(qr, wp.float32),
        _wp_view(axis_aligned, wp.bool),
        _wp_view(fractional_certified, wp.bool),
        _wp_view(height_certified, wp.bool),
        _wp_view(bbox_bounds, wp.float32),
        device,
    )

    def build_tiles(fixed: bool):
        """Build the single self tile for one geometry mode."""
        num_tiles = torch.zeros((1,), dtype=torch.int32, device=device)
        tile_rows = torch.zeros((1,), dtype=torch.int32, device=device)
        tile_cols = torch.zeros((1,), dtype=torch.int32, device=device)
        common = {
            "fixed_cell": fixed,
            "qr": _wp_view(qr, wp.float32) if fixed else None,
            "axis_aligned": _wp_view(axis_aligned, wp.bool) if fixed else None,
            "fractional_rounding_certified": (
                _wp_view(fractional_certified, wp.bool) if fixed else None
            ),
            "qr_height_certified": (
                _wp_view(height_certified, wp.bool) if fixed else None
            ),
            "bbox_cutoff_bounds": (
                _wp_view(bbox_bounds, wp.float32) if fixed else None
            ),
        }
        build_cluster_tile_list(
            _wp_view(positions[:, 0], wp.float32),
            _wp_view(positions[:, 1], wp.float32),
            _wp_view(positions[:, 2], wp.float32),
            _wp_view(cell, wp.mat33f),
            _wp_view(inverse, wp.mat33f),
            cutoff,
            _wp_view(num_tiles, wp.int32),
            _wp_view(tile_rows, wp.int32),
            _wp_view(tile_cols, wp.int32),
            wp.float32,
            device,
            **common,
        )
        return num_tiles, tile_rows, tile_cols, common

    dynamic_tiles = build_tiles(False)
    fixed_tiles = build_tiles(True)
    torch.testing.assert_close(fixed_tiles[0], dynamic_tiles[0])
    torch.testing.assert_close(fixed_tiles[1], dynamic_tiles[1])
    torch.testing.assert_close(fixed_tiles[2], dynamic_tiles[2])

    def query_matrix(fixed: bool, tiles):
        """Run the matrix query and return its topology arrays."""
        counts = torch.zeros((natom,), dtype=torch.int32, device=device)
        neighbors = torch.full((natom, natom), -1, dtype=torch.int32, device=device)
        shifts = torch.zeros((natom, natom, 3), dtype=torch.int32, device=device)
        query_cluster_tile(
            _wp_view(atom_ids, wp.int32),
            _wp_view(positions[:, 0], wp.float32),
            _wp_view(positions[:, 1], wp.float32),
            _wp_view(positions[:, 2], wp.float32),
            _wp_view(tiles[0], wp.int32),
            _wp_view(tiles[1], wp.int32),
            _wp_view(tiles[2], wp.int32),
            _wp_view(cell, wp.mat33f),
            _wp_view(inverse, wp.mat33f),
            cutoff,
            natom,
            _wp_view(neighbors, wp.int32),
            _wp_view(counts, wp.int32),
            _wp_view(shifts, wp.int32),
            wp.float32,
            device,
            n_tiles=1,
            **tiles[3],
        )
        return counts, neighbors, shifts

    dynamic_matrix = query_matrix(False, dynamic_tiles)
    fixed_matrix = query_matrix(True, fixed_tiles)
    for actual, expected in zip(fixed_matrix, dynamic_matrix, strict=True):
        torch.testing.assert_close(actual, expected)

    def query_coo(fixed: bool, tiles):
        """Run generic COO query and return its active pair prefix."""
        pair_counter = torch.zeros((1,), dtype=torch.int32, device=device)
        pairs = torch.full((natom * natom, 2), -1, dtype=torch.int32, device=device)
        pair_shifts = torch.zeros((natom * natom, 3), dtype=torch.int32, device=device)
        query_cluster_tile_coo(
            _wp_view(atom_ids, wp.int32),
            _wp_view(positions[:, 0], wp.float32),
            _wp_view(positions[:, 1], wp.float32),
            _wp_view(positions[:, 2], wp.float32),
            _wp_view(tiles[0], wp.int32),
            _wp_view(tiles[1], wp.int32),
            _wp_view(tiles[2], wp.int32),
            _wp_view(cell, wp.mat33f),
            _wp_view(inverse, wp.mat33f),
            cutoff,
            natom,
            natom * natom,
            _wp_view(pair_counter, wp.int32),
            _wp_view(pairs, wp.int32),
            _wp_view(pair_shifts, wp.int32),
            wp.float32,
            device,
            n_tiles=1,
            **tiles[3],
        )
        count = int(pair_counter.cpu().item())
        return pairs[:count], pair_shifts[:count]

    dynamic_coo = query_coo(False, dynamic_tiles)
    fixed_coo = query_coo(True, fixed_tiles)
    actual_pairs = sorted(
        zip(fixed_coo[0].cpu().tolist(), fixed_coo[1].cpu().tolist(), strict=True)
    )
    expected_pairs = sorted(
        zip(dynamic_coo[0].cpu().tolist(), dynamic_coo[1].cpu().tolist(), strict=True)
    )
    assert actual_pairs == expected_pairs
