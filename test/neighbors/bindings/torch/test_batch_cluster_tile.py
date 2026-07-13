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

"""Tests for the batched cluster-pair tile neighbor list PyTorch bindings."""

import pytest
import torch

import nvalchemiops.torch.neighbors.batch_cluster_tile as batch_cluster_tile_module
from nvalchemiops.neighbors.cluster_tile import (
    estimate_batch_max_tiles_per_group as estimate_core_batch_max_tiles_per_group,
)
from nvalchemiops.neighbors.cluster_tile import (
    estimate_max_tiles_per_group,
)
from nvalchemiops.neighbors.neighbor_utils import NeighborOverflowError
from nvalchemiops.torch.neighbors.batch_cluster_tile import (
    TILE_GROUP_SIZE,
    allocate_batch_cluster_tile_list,
    batch_build_cluster_tile_list,
    batch_cluster_tile_neighbor_list,
    batch_query_cluster_tile,
    estimate_batch_cluster_tile_list_sizes,
    estimate_batch_max_tiles_per_group,
)

from ...test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
)
from .conftest import requires_vesin

# batch_cluster_tile is CUDA + float32 only; override the conftest
# device/dtype fixtures to restrict the parametrize matrix.


@pytest.fixture(params=["cuda:0"], ids=lambda d: d.replace(":", "_"))
def device(request):
    if not torch.cuda.is_available():
        pytest.skip("batch_cluster_tile kernel tests require torch CUDA tensors")
    return request.param


@pytest.fixture(params=[torch.float32], ids=["float32"])
def dtype(request):
    return request.param


def _make_batch(
    sys_sizes: list[int],
    cell_sizes: list[float],
    device: str,
    dtype=torch.float32,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    pos_chunks, cells = [], []
    for sz, L in zip(sys_sizes, cell_sizes):
        pos_chunks.append(torch.rand(sz, 3, dtype=dtype, device=device) * L)
        cells.append(torch.eye(3, dtype=dtype, device=device) * L)
    positions = torch.cat(pos_chunks, dim=0).contiguous()
    cell_batch = torch.stack(cells, dim=0).contiguous()
    bp = [0]
    for sz in sys_sizes:
        bp.append(bp[-1] + sz)
    batch_ptr = torch.tensor(bp, dtype=torch.int32, device=device)
    return positions, cell_batch, batch_ptr


class TestBatchClusterTileValidation:
    """Validate public option combinations rejected before kernel launch."""

    def test_pair_outputs_reject_tile_format(self):
        """Pair-output buffers are not supported with tile-format output."""
        positions, cell_batch, batch_ptr = _make_batch([32], [8.0], device="cpu")

        with pytest.raises(NotImplementedError, match="format='tile'"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                format="tile",
                return_distances=True,
            )

    def test_cutoff2_rejects_non_matrix_and_pair_outputs(self):
        """Dual cutoff is matrix-only and cannot request pair outputs."""
        positions, cell_batch, batch_ptr = _make_batch([32], [8.0], device="cpu")

        with pytest.raises(ValueError, match="format='matrix'"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                format="coo",
                cutoff2=3.0,
            )

        with pytest.raises(ValueError, match="cannot be combined with pair outputs"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                cutoff2=3.0,
                return_vectors=True,
            )

    def test_segmented_offsets_are_all_or_nothing(self):
        """Segmented tile/COO metadata must be passed as paired arrays."""
        positions, cell_batch, batch_ptr = _make_batch([32], [8.0], device="cpu")
        offsets = torch.tensor([0, 64], dtype=torch.int32)
        counts = torch.zeros(1, dtype=torch.int32)

        with pytest.raises(ValueError, match="pair_offsets"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                format="coo",
                pair_offsets=offsets,
            )

        with pytest.raises(ValueError, match="tile_offsets"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                tile_offsets=offsets,
            )

        with pytest.raises(ValueError, match="format='tile'"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                format="tile",
                rebuild_flags=torch.ones(1, dtype=torch.bool),
                tile_offsets=offsets,
                tile_counts=counts,
            )


def test_batch_cluster_tile_max_tiles_per_group_bypasses_sizing(monkeypatch):
    """Explicit max_tiles_per_group skips geometry-aware sizing sync."""

    def fail_sizing(*args, **kwargs):
        del args, kwargs
        raise AssertionError("estimate_batch_max_tiles_per_group should not be called")

    def fake_allocate(batch_ptr, device, *, dtype, max_tiles_per_group):
        del batch_ptr, device, dtype
        assert max_tiles_per_group == 7
        raise RuntimeError("allocation reached")

    monkeypatch.setattr(
        batch_cluster_tile_module, "estimate_batch_max_tiles_per_group", fail_sizing
    )
    monkeypatch.setattr(
        batch_cluster_tile_module, "allocate_batch_cluster_tile_list", fake_allocate
    )

    positions = torch.zeros(64, 3, dtype=torch.float32)
    cell_batch = torch.stack([torch.eye(3), torch.eye(3)]).to(torch.float32) * 10.0
    batch_ptr = torch.tensor([0, 32, 64], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="allocation reached"):
        batch_cluster_tile_module.batch_cluster_tile_neighbor_list(
            positions,
            3.0,
            cell_batch,
            batch_ptr,
            max_tiles_per_group=7,
        )


def test_estimate_batch_max_tiles_per_group_rejects_short_batch_ptr_length():
    """Direct cluster-tile sizing rejects one-entry batch_ptr."""
    batch_ptr = torch.tensor([0], dtype=torch.int32)
    cell_batch = torch.zeros((0, 3, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="batch_ptr.*length at least 2"):
        estimate_batch_max_tiles_per_group(batch_ptr, 3.0, cell_batch)


def test_batch_cluster_tile_max_tiles_per_group_vectorized_helper():
    """Batched max-tile sizing should match the scalar estimator."""
    batch_ptr = torch.tensor([0, 32, 32800], dtype=torch.int32)
    cell_batch = torch.stack(
        [
            torch.eye(3, dtype=torch.float64) * 10.0,
            torch.eye(3, dtype=torch.float64) * 10.0,
        ],
    )
    cutoff = 20.0

    expected = 256
    for start, stop, cell in zip(batch_ptr[:-1], batch_ptr[1:], cell_batch):
        expected = max(
            expected,
            estimate_max_tiles_per_group(
                int((stop - start).item()),
                cutoff,
                float(torch.linalg.det(cell).abs().item()),
            ),
        )

    got = estimate_batch_max_tiles_per_group(
        batch_ptr,
        cutoff,
        cell_batch,
    )

    assert got == expected
    assert got > 256


def test_batch_max_tiles_per_group_uses_float64_volumes_for_float32_cells():
    """Float32 cell inputs should size from float64 determinant volumes."""
    batch_ptr = torch.tensor([0, 100000], dtype=torch.int32)
    cell_batch = torch.tensor(
        [
            [
                [10.0, 0.1, 0.05],
                [0.0, 10.0, 0.08],
                [0.02, 0.03, 10.0],
            ],
        ],
        dtype=torch.float32,
    )
    cutoff = 25.0
    volumes_f64 = torch.linalg.det(cell_batch.to(torch.float64)).abs().view(-1)
    volumes_f32 = torch.linalg.det(cell_batch).abs().view(-1)
    assert volumes_f32.item() != volumes_f64.item()

    expected = estimate_core_batch_max_tiles_per_group(
        batch_ptr,
        cutoff,
        volumes_f64,
    )
    old_estimate = estimate_core_batch_max_tiles_per_group(
        batch_ptr,
        cutoff,
        volumes_f32,
    )
    assert old_estimate != expected
    got = estimate_batch_max_tiles_per_group(
        batch_ptr,
        cutoff,
        cell_batch,
    )

    assert got == expected
    assert got > 256


def test_core_batch_max_tiles_per_group_keeps_batch_floor_for_small_systems():
    """Small batched systems keep the historical compact-buffer floor."""
    assert (
        estimate_core_batch_max_tiles_per_group([0, 1, 2], 2.0, [1000.0, 1000.0]) == 256
    )


def test_core_batch_max_tiles_per_group_validates_cell_volume_count():
    """The core estimator requires one cell volume per batch segment."""
    with pytest.raises(ValueError, match="cell_volumes"):
        estimate_core_batch_max_tiles_per_group([0, 32, 64], 2.0, [1000.0])


def test_core_batch_max_tiles_per_group_validates_monotonic_batch_ptr():
    """The core estimator rejects decreasing batch pointers."""
    with pytest.raises(ValueError, match="non-decreasing"):
        estimate_core_batch_max_tiles_per_group(
            [0, 64, 32],
            2.0,
            [1000.0, 1000.0],
        )


def _canonicalize_matrix_full(
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    shifts: torch.Tensor,
    atom_system: list[int],
    natom: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten per-system matrix output to ALL directed (i, j, shift) triples.

    cluster_tile is full-fill, so each unordered pair appears in both rows;
    this matches vesin's ``full_list=True`` reference directly.
    """
    device = neighbor_matrix.device
    i_list, j_list, u_list = [], [], []
    nm_cpu = neighbor_matrix.cpu()
    nn_cpu = num_neighbors.cpu()
    s_cpu = shifts.cpu()
    for i in range(natom):
        ni = int(nn_cpu[i].item())
        for k in range(ni):
            j = int(nm_cpu[i, k].item())
            if 0 <= j < natom:
                assert atom_system[j] == atom_system[i], (
                    f"cross-system pair i={i} j={j}"
                )
                sh = tuple(int(x) for x in s_cpu[i, k])
                i_list.append(i)
                j_list.append(j)
                u_list.append(sh)
    i_t = torch.tensor(i_list, dtype=torch.int32, device=device)
    j_t = torch.tensor(j_list, dtype=torch.int32, device=device)
    if u_list:
        u_t = torch.tensor(u_list, dtype=torch.int32, device=device).reshape(-1, 3)
    else:
        u_t = torch.zeros((0, 3), dtype=torch.int32, device=device)
    return i_t, j_t, u_t


def _reference_pairs_per_system(
    positions: torch.Tensor,
    cell_batch: torch.Tensor,
    batch_ptr: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Brute-force per-system full-fill reference (global indices)."""
    S = cell_batch.shape[0]
    i_all, j_all, u_all = [], [], []
    for s in range(S):
        a = int(batch_ptr[s].item())
        b = int(batch_ptr[s + 1].item())
        sub = positions[a:b]
        cell_s = cell_batch[s].unsqueeze(0)
        pbc_s = torch.tensor([True, True, True], device=positions.device)
        i_s, j_s, u_s, _ = brute_force_neighbors(sub, cell_s, pbc_s, cutoff)
        i_all.append(i_s.to(torch.int64) + a)
        j_all.append(j_s.to(torch.int64) + a)
        u_all.append(u_s)
    if i_all:
        i_t = torch.cat(i_all).to(torch.int32)
        j_t = torch.cat(j_all).to(torch.int32)
        u_t = torch.cat(u_all)
    else:
        dev = positions.device
        i_t = torch.zeros(0, dtype=torch.int32, device=dev)
        j_t = torch.zeros(0, dtype=torch.int32, device=dev)
        u_t = torch.zeros((0, 3), dtype=torch.int32, device=dev)
    return i_t, j_t, u_t


# =============================================================================
# Correctness
# =============================================================================
class TestBatchTileNeighborListCorrectness:
    @requires_vesin
    def test_single_system_batch(self, device, dtype):
        """Batch of size 1 should match brute-force."""
        positions, cell_batch, batch_ptr = _make_batch(
            [64],
            [10.0],
            device=device,
            dtype=dtype,
            seed=1,
        )
        cutoff = 3.0
        nm, nn, nms = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
        )
        atom_system = [0] * positions.shape[0]
        i_got, j_got, u_got = _canonicalize_matrix_full(
            nm,
            nn,
            nms,
            atom_system,
            positions.shape[0],
        )
        i_ref, j_ref, u_ref = _reference_pairs_per_system(
            positions,
            cell_batch,
            batch_ptr,
            cutoff,
        )
        assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_multi_system_equal_sizes(self, device, dtype):
        """Multiple systems with identical sizes and cells."""
        positions, cell_batch, batch_ptr = _make_batch(
            [64, 64, 64],
            [10.0, 10.0, 10.0],
            device=device,
            dtype=dtype,
            seed=2,
        )
        cutoff = 3.0
        nm, nn, nms = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
        )
        atom_system = sum(([s] * sz for s, sz in enumerate([64, 64, 64])), [])
        i_got, j_got, u_got = _canonicalize_matrix_full(
            nm,
            nn,
            nms,
            atom_system,
            positions.shape[0],
        )
        i_ref, j_ref, u_ref = _reference_pairs_per_system(
            positions,
            cell_batch,
            batch_ptr,
            cutoff,
        )
        assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_partial_sizes(self, device, dtype):
        """Per-system sizes that are NOT multiples of TILE_GROUP_SIZE."""
        sizes = [33, 65, 100]
        positions, cell_batch, batch_ptr = _make_batch(
            sizes,
            [8.0, 10.0, 12.0],
            device=device,
            dtype=dtype,
            seed=3,
        )
        cutoff = 2.5
        nm, nn, nms = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=128,
        )
        atom_system = sum(([s] * sz for s, sz in enumerate(sizes)), [])
        i_got, j_got, u_got = _canonicalize_matrix_full(
            nm,
            nn,
            nms,
            atom_system,
            positions.shape[0],
        )
        i_ref, j_ref, u_ref = _reference_pairs_per_system(
            positions,
            cell_batch,
            batch_ptr,
            cutoff,
        )
        assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_triclinic_cells(self, device, dtype):
        """Moderately skewed triclinic cells."""
        device = device
        torch.manual_seed(4)
        N = 96
        frac = torch.rand(N, 3, dtype=dtype, device=device)
        # Skew the cell off-diagonal.
        cell = torch.eye(3, dtype=dtype, device=device) * 10.0
        cell[0, 1] = 1.0
        cell[1, 2] = 0.5
        positions = (frac @ cell).contiguous()
        cell_batch = cell.unsqueeze(0).contiguous()
        batch_ptr = torch.tensor([0, N], dtype=torch.int32, device=device)
        cutoff = 2.5
        nm, nn, nms = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=128,
        )
        atom_system = [0] * N
        i_got, j_got, u_got = _canonicalize_matrix_full(
            nm,
            nn,
            nms,
            atom_system,
            N,
        )
        i_ref, j_ref, u_ref = _reference_pairs_per_system(
            positions,
            cell_batch,
            batch_ptr,
            cutoff,
        )
        assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    def test_component_API_matches_convenience(self, device, dtype):
        """Explicit allocate + build + to_matrix matches the convenience path."""
        sizes = [64, 96]
        positions, cell_batch, batch_ptr = _make_batch(
            sizes,
            [10.0, 8.0],
            device=device,
            dtype=dtype,
            seed=5,
        )
        cutoff = 2.5
        N = positions.shape[0]

        nm1, nn1, _nms1 = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
        )

        (
            sorted_atom_index,
            sort_inv,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            batch_idx_sorted,
            batch_ptr_padded,
            group_system,
            group_ptr,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
        ) = allocate_batch_cluster_tile_list(
            batch_ptr,
            torch.device(device),
            dtype=dtype,
        )
        batch_build_cluster_tile_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            sorted_atom_index,
            sort_inv,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            batch_idx_sorted,
            batch_ptr_padded,
            group_system,
            group_ptr,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
        )
        nm2 = torch.full((N, 64), N, dtype=torch.int32, device=device)
        nn2 = torch.zeros(N, dtype=torch.int32, device=device)
        nms2 = torch.zeros((N, 64, 3), dtype=torch.int32, device=device)
        batch_query_cluster_tile(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            cell_batch,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            cutoff,
            N,
            nm2,
            nn2,
            nms2,
        )
        torch.testing.assert_close(nn1, nn2)
        # Entries may differ in per-row order; compare sets row-wise.
        for i in range(N):
            n_i = int(nn1[i].item())
            s1 = {int(x.item()) for x in nm1[i, :n_i]}
            s2 = {int(x.item()) for x in nm2[i, :n_i]}
            assert s1 == s2, f"atom {i} neighbor set mismatch"

    def test_tile_buffer_overflow_raises(self, device, dtype):
        """A too-small batch tile buffer must raise, not silently truncate."""
        positions, cell_batch, batch_ptr = _make_batch(
            [256, 256], [8.0, 8.0], device=device, dtype=dtype, seed=9
        )
        cutoff = 4.0
        N = positions.shape[0]
        scratch = allocate_batch_cluster_tile_list(
            batch_ptr, torch.device(device), dtype=dtype, max_tiles_per_group=1
        )
        (
            sorted_atom_index,
            sort_inv,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            batch_idx_sorted,
            batch_ptr_padded,
            group_system,
            group_ptr,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
        ) = scratch
        batch_build_cluster_tile_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            sorted_atom_index,
            sort_inv,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            batch_idx_sorted,
            batch_ptr_padded,
            group_system,
            group_ptr,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
        )
        assert int(num_tiles.item()) > int(tile_row_group.shape[0])
        nm = torch.empty((N, 64), dtype=torch.int32, device=device)
        nn = torch.zeros(N, dtype=torch.int32, device=device)
        nms = torch.empty((N, 64, 3), dtype=torch.int32, device=device)
        with pytest.raises(NeighborOverflowError):
            batch_query_cluster_tile(
                sorted_atom_index,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                cell_batch,
                num_tiles,
                tile_row_group,
                tile_col_group,
                tile_system,
                cutoff,
                N,
                nm,
                nn,
                nms,
            )


# =============================================================================
# Edge cases
# =============================================================================
class TestBatchTileNeighborListEdgeCases:
    def test_empty_cutoff(self, device, dtype):
        positions, cell_batch, batch_ptr = _make_batch(
            [32, 64],
            [5.0, 5.0],
            device=device,
            dtype=dtype,
            seed=6,
        )
        nm, nn, _nms = batch_cluster_tile_neighbor_list(
            positions,
            1e-6,
            cell_batch,
            batch_ptr,
            max_neighbors=16,
        )
        assert int(nn.sum().item()) == 0

    def test_estimate_sizes_consistency(self, device):
        batch_ptr = torch.tensor(
            [0, 33, 98, 198],
            dtype=torch.int32,
            device=device,
        )
        n_padded, ngroup, ngroup_padded, max_tiles, S = (
            estimate_batch_cluster_tile_list_sizes(batch_ptr)
        )
        assert S == 3
        assert n_padded >= 198
        assert n_padded % TILE_GROUP_SIZE == 0
        assert ngroup == n_padded // TILE_GROUP_SIZE
        assert max_tiles >= ngroup
        assert ngroup_padded > ngroup


# =============================================================================
# Output formats: format="tile" and format="coo"
# =============================================================================
class TestBatchTileNeighborListFormats:
    """Cover the ``format="tile"`` and ``format="coo"`` return paths of
    ``batch_cluster_tile_neighbor_list`` (lines 875-940 in batch_cluster_tile.py)."""

    def test_format_tile_returns_eleven_tuple(self, device, dtype):
        sizes = [64, 96]
        positions, cell_batch, batch_ptr = _make_batch(
            sizes,
            [10.0, 10.0],
            device=device,
            dtype=dtype,
            seed=10,
        )
        result = batch_cluster_tile_neighbor_list(
            positions,
            3.0,
            cell_batch,
            batch_ptr,
            format="tile",
        )
        assert isinstance(result, tuple) and len(result) == 11
        (
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            batch_idx_sorted,
            batch_ptr_padded,
            group_ptr,
        ) = result
        N = positions.shape[0]
        n_padded = int(batch_ptr_padded[-1].item())
        assert num_tiles.dtype == torch.int32
        assert num_tiles.numel() == 1
        assert int(num_tiles.item()) <= int(tile_row_group.numel())
        assert sorted_atom_index.shape == (n_padded,)
        assert sorted_pos_x.shape == (n_padded,)
        assert batch_idx_sorted.shape == (n_padded,)
        assert batch_ptr_padded.shape == (len(sizes) + 1,)
        assert group_ptr.shape[0] == len(sizes) + 1
        assert n_padded >= N

    def test_format_coo_returns_three_tuple(self, device, dtype):
        sizes = [64, 64]
        positions, cell_batch, batch_ptr = _make_batch(
            sizes,
            [10.0, 10.0],
            device=device,
            dtype=dtype,
            seed=11,
        )
        cutoff = 3.0
        # matrix-format reference for pair count
        nm, nn, _ = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
        )
        expected_pairs = int(nn.sum().item())

        nl, neighbor_ptr, nls = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_pairs=4096,
            format="coo",
        )
        N = positions.shape[0]
        assert nl.shape[0] == 2
        assert nl.shape[1] == expected_pairs
        assert nls.shape == (expected_pairs, 3)
        assert neighbor_ptr.shape == (N + 1,)
        assert int(neighbor_ptr[0].item()) == 0
        assert int(neighbor_ptr[-1].item()) == expected_pairs
        # Sources match nn (matrix per-atom counts).
        per_atom_from_ptr = (neighbor_ptr[1:] - neighbor_ptr[:-1]).to(torch.int32)
        torch.testing.assert_close(per_atom_from_ptr, nn)

    def test_format_coo_with_preallocated_buffers(self, device, dtype):
        sizes = [48, 48]
        positions, cell_batch, batch_ptr = _make_batch(
            sizes,
            [10.0, 10.0],
            device=device,
            dtype=dtype,
            seed=12,
        )
        cutoff = 3.0
        max_pairs = 2048
        N = positions.shape[0]
        # Caller-provided buffers: neighbor_list is (2, max_pairs).  The
        # implementation transposes it to (max_pairs, 2) internally.
        neighbor_list_buf = torch.empty(
            (max_pairs, 2),
            dtype=torch.int32,
            device=device,
        ).transpose(0, 1)
        neighbor_list_shifts_buf = torch.empty(
            (max_pairs, 3),
            dtype=torch.int32,
            device=device,
        )
        pair_counter_buf = torch.zeros(1, dtype=torch.int32, device=device)
        nl, neighbor_ptr, nls = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_pairs=max_pairs,
            format="coo",
            neighbor_list=neighbor_list_buf,
            neighbor_list_shifts=neighbor_list_shifts_buf,
            pair_counter=pair_counter_buf,
        )
        assert nl.shape[0] == 2
        assert neighbor_ptr.shape == (N + 1,)
        assert int(neighbor_ptr[-1].item()) == nl.shape[1]
        assert nls.shape == (nl.shape[1], 3)

    def test_invalid_format_raises(self, device, dtype):
        positions, cell_batch, batch_ptr = _make_batch(
            [32],
            [10.0],
            device=device,
            dtype=dtype,
            seed=13,
        )
        with pytest.raises(ValueError, match="format"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.5,
                cell_batch,
                batch_ptr,
                format="bogus",
            )


# =============================================================================
# Errors
# =============================================================================
class TestBatchTileNeighborListErrors:
    def test_wrong_dtype(self, device):
        positions, cell_batch, batch_ptr = _make_batch(
            [32],
            [10.0],
            device=device,
            dtype=torch.float32,
            seed=7,
        )
        positions = positions.to(torch.float64)
        with pytest.raises(TypeError):
            batch_cluster_tile_neighbor_list(
                positions,
                2.5,
                cell_batch,
                batch_ptr,
                max_neighbors=32,
            )

    def test_mismatched_batch_ptr(self, device, dtype):
        positions, cell_batch, _ = _make_batch(
            [32],
            [10.0],
            device=device,
            dtype=dtype,
            seed=8,
        )
        # batch_ptr claims a larger total than positions provides.
        bad_bp = torch.tensor([0, 64], dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="batch_ptr"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.5,
                cell_batch,
                bad_bp,
                max_neighbors=32,
            )


# =============================================================================
# torch.compile compatibility
# =============================================================================
class TestBatchClusterTileCompile:
    """Tests for ``torch.compile`` compatibility of the batched cluster-pair
    tile path.  Verifies that the ``@torch.library.custom_op``-decorated
    component shells (``_batch_build_cluster_tile_list``,
    ``_batch_query_cluster_tile``, ``_batch_query_cluster_tile_coo``) survive a
    ``torch.compile`` round-trip.
    """

    @pytest.mark.slow
    def test_batch_cluster_tile_neighbor_list_compile(self, device, dtype):
        """``batch_cluster_tile_neighbor_list`` should be compatible with ``torch.compile``."""
        sizes = [64, 96]
        positions, cell_batch, batch_ptr = _make_batch(
            sizes,
            [10.0, 8.0],
            device=device,
            dtype=dtype,
            seed=11,
        )
        cutoff = 2.5

        nm_uncompiled, nn_uncompiled, _nms_uncompiled = (
            batch_cluster_tile_neighbor_list(
                positions,
                cutoff,
                cell_batch,
                batch_ptr,
                max_neighbors=64,
            )
        )

        @torch.compile
        def compiled_batch_cluster_tile_neighbor_list(
            positions, cutoff, cell_batch, batch_ptr
        ):
            return batch_cluster_tile_neighbor_list(
                positions,
                cutoff,
                cell_batch,
                batch_ptr,
                max_neighbors=64,
            )

        nm_compiled, nn_compiled, _nms_compiled = (
            compiled_batch_cluster_tile_neighbor_list(
                positions, cutoff, cell_batch, batch_ptr
            )
        )

        assert torch.equal(nn_uncompiled, nn_compiled)
        N = positions.shape[0]
        for i in range(N):
            n_i = int(nn_uncompiled[i].item())
            s_uncompiled = {int(x.item()) for x in nm_uncompiled[i, :n_i]}
            s_compiled = {int(x.item()) for x in nm_compiled[i, :n_i]}
            assert s_uncompiled == s_compiled, (
                f"Row {i} neighbor set mismatch under torch.compile"
            )


# =============================================================================
# Components API
# =============================================================================
class TestBatchClusterTileComponentsAPI:
    """Tests for the modular batched tile API functions (allocate + build +
    convert), exercised independently of the convenience wrapper.
    """

    def test_build_and_convert_roundtrip(self, device, dtype):
        """allocate + build + convert + reconvert (with cleared outputs) should
        be deterministic across re-launches against the same state.
        """
        sizes = [48, 80]
        positions, cell_batch, batch_ptr = _make_batch(
            sizes,
            [9.0, 7.0],
            device=device,
            dtype=dtype,
            seed=13,
        )
        cutoff = 2.5
        N = positions.shape[0]

        state = allocate_batch_cluster_tile_list(
            batch_ptr,
            torch.device(device),
            dtype=dtype,
        )
        batch_build_cluster_tile_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            *state,
        )
        nm_a = torch.full((N, 32), N, dtype=torch.int32, device=device)
        nn_a = torch.zeros(N, dtype=torch.int32, device=device)
        nms_a = torch.zeros((N, 32, 3), dtype=torch.int32, device=device)
        batch_query_cluster_tile(
            state[0],  # sorted_atom_index
            state[2],  # sorted_pos_x
            state[3],  # sorted_pos_y
            state[4],  # sorted_pos_z
            cell_batch,
            state[15],  # num_tiles
            state[16],  # tile_row_group
            state[17],  # tile_col_group
            state[18],  # tile_system
            cutoff,
            N,
            nm_a,
            nn_a,
            nms_a,
        )

        # Second conversion into fresh outputs from the same built state.
        nm_b = torch.full((N, 32), N, dtype=torch.int32, device=device)
        nn_b = torch.zeros(N, dtype=torch.int32, device=device)
        nms_b = torch.zeros((N, 32, 3), dtype=torch.int32, device=device)
        batch_query_cluster_tile(
            state[0],
            state[2],
            state[3],
            state[4],
            cell_batch,
            state[15],
            state[16],
            state[17],
            state[18],
            cutoff,
            N,
            nm_b,
            nn_b,
            nms_b,
        )

        assert torch.equal(nn_a, nn_b)
        for i in range(N):
            n_i = int(nn_a[i].item())
            s_a = {int(x.item()) for x in nm_a[i, :n_i]}
            s_b = {int(x.item()) for x in nm_b[i, :n_i]}
            assert s_a == s_b, f"atom {i} re-conversion mismatch"

    def test_allocate_sizes_consistent_with_estimate(self, device, dtype):
        """The shapes returned by ``allocate_batch_cluster_tile_list`` should
        match what ``estimate_batch_cluster_tile_list_sizes`` advertises.
        """
        sizes = [48, 80, 40]
        positions, cell_batch, batch_ptr = _make_batch(
            sizes,
            [9.0, 7.0, 6.0],
            device=device,
            dtype=dtype,
            seed=14,
        )
        del positions, cell_batch  # unused — only batch_ptr shape matters here
        n_padded, ngroup, ngroup_padded, max_tiles, num_systems = (
            estimate_batch_cluster_tile_list_sizes(batch_ptr)
        )
        state = allocate_batch_cluster_tile_list(
            batch_ptr,
            torch.device(device),
            dtype=dtype,
        )
        sorted_atom_index = state[0]
        sorted_pos_x = state[2]
        group_system = state[7]
        group_ctr_x = state[9]
        tile_row_group = state[16]
        # n_padded total: sorted arrays sized by total padded atoms.
        assert sorted_atom_index.shape[0] == n_padded
        assert sorted_pos_x.shape[0] == n_padded
        # group_system has ngroup entries; group_ctr_* have ngroup_padded.
        assert group_system.shape[0] == ngroup
        assert group_ctr_x.shape[0] == ngroup_padded
        # tile row buffers sized by max_tiles.
        assert tile_row_group.shape[0] == max_tiles
        assert num_systems == int(batch_ptr.shape[0]) - 1

    def test_build_wrong_positions_dtype_raises(self, device, dtype):
        sizes = [32]
        positions, cell_batch, batch_ptr = _make_batch(
            sizes,
            [10.0],
            device=device,
            dtype=torch.float32,
            seed=20,
        )
        state = allocate_batch_cluster_tile_list(
            batch_ptr,
            torch.device(device),
            dtype=torch.float32,
        )
        with pytest.raises(TypeError, match="float32"):
            batch_build_cluster_tile_list(
                positions.to(torch.float64),
                2.5,
                cell_batch,
                batch_ptr,
                *state,
            )

    def test_build_wrong_cell_shape_raises(self, device, dtype):
        positions, cell_batch, batch_ptr = _make_batch(
            [32],
            [10.0],
            device=device,
            dtype=dtype,
            seed=21,
        )
        state = allocate_batch_cluster_tile_list(
            batch_ptr,
            torch.device(device),
            dtype=dtype,
        )
        # cell_batch as 2D (3,3) instead of (S, 3, 3) → ValueError
        bad_cell = cell_batch.squeeze(0)
        with pytest.raises(ValueError, match="cell_batch"):
            batch_build_cluster_tile_list(
                positions,
                2.5,
                bad_cell,
                batch_ptr,
                *state,
            )

    def test_build_wrong_batch_ptr_dtype_raises(self, device, dtype):
        positions, cell_batch, batch_ptr = _make_batch(
            [32],
            [10.0],
            device=device,
            dtype=dtype,
            seed=22,
        )
        state = allocate_batch_cluster_tile_list(
            batch_ptr,
            torch.device(device),
            dtype=dtype,
        )
        with pytest.raises(ValueError, match="batch_ptr"):
            batch_build_cluster_tile_list(
                positions,
                2.5,
                cell_batch,
                batch_ptr.to(torch.int64),
                *state,
            )

    def test_build_mismatched_batch_ptr_length_raises(self, device, dtype):
        positions, cell_batch, batch_ptr = _make_batch(
            [32, 32],
            [10.0, 10.0],
            device=device,
            dtype=dtype,
            seed=23,
        )
        state = allocate_batch_cluster_tile_list(
            batch_ptr,
            torch.device(device),
            dtype=dtype,
        )
        # cell_batch has 2 systems but batch_ptr claims 3 → ValueError.
        bad_bp = torch.tensor([0, 16, 32, 64], dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="batch_ptr length"):
            batch_build_cluster_tile_list(
                positions,
                2.5,
                cell_batch,
                bad_bp,
                *state,
            )

    def test_build_with_explicit_inv_cell_batch(self, device, dtype):
        """Passing inv_cell_batch explicitly skips the torch.linalg.inv call."""
        positions, cell_batch, batch_ptr = _make_batch(
            [32, 48],
            [10.0, 10.0],
            device=device,
            dtype=dtype,
            seed=24,
        )
        state = allocate_batch_cluster_tile_list(
            batch_ptr,
            torch.device(device),
            dtype=dtype,
        )
        inv_cell_batch = torch.linalg.inv(cell_batch).contiguous()
        # Should run without error when inv_cell_batch is provided.
        batch_build_cluster_tile_list(
            positions,
            2.5,
            cell_batch,
            batch_ptr,
            *state,
            inv_cell_batch=inv_cell_batch,
        )


class TestBatchClusterTileAutograd:
    """Differentiable per-pair distances/vectors for batch_cluster_tile_neighbor_list."""

    def _make_batch(self, device, n_per=32, box=5.0):
        torch.manual_seed(0)
        pos = torch.randn(2 * n_per, 3, dtype=torch.float32, device=device) * 0.5
        batch_ptr = torch.tensor(
            [0, n_per, 2 * n_per], dtype=torch.int32, device=device
        )
        cell_batch = (
            torch.eye(3, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(2, 1, 1)
            * box
        )
        return pos, cell_batch, batch_ptr

    def test_forward_returns_differentiable(self, device):
        pos, cell_batch, batch_ptr = self._make_batch(device)
        pos.requires_grad_(True)
        nm, nn, shifts, d, v = batch_cluster_tile_neighbor_list(
            pos,
            1.5,
            cell_batch,
            batch_ptr,
            return_distances=True,
            return_vectors=True,
        )
        assert d.requires_grad and v.requires_grad

    def test_grad_positions_finite(self, device):
        pos, cell_batch, batch_ptr = self._make_batch(device)
        pos.requires_grad_(True)
        _, _, _, d, _ = batch_cluster_tile_neighbor_list(
            pos,
            1.5,
            cell_batch,
            batch_ptr,
            return_distances=True,
            return_vectors=True,
        )
        d.sum().backward()
        assert torch.isfinite(pos.grad).all()

    def test_grad_cell_finite(self, device):
        pos, _, batch_ptr = self._make_batch(device)
        cell_batch = (
            torch.eye(3, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(2, 1, 1)
            * 5.0
        )
        cell_batch.requires_grad_(True)
        _, _, _, d, _ = batch_cluster_tile_neighbor_list(
            pos,
            1.5,
            cell_batch,
            batch_ptr,
            return_distances=True,
            return_vectors=True,
        )
        d.sum().backward()
        assert torch.isfinite(cell_batch.grad).all()

    def test_hessian_vector_product_smoke(self, device):
        """fp32 second-order HVP smoke — see TestClusterTileAutograd."""
        pos, cell_batch, batch_ptr = self._make_batch(device)
        pos.requires_grad_(True)

        def loss(p):
            *_, d, _ = batch_cluster_tile_neighbor_list(
                p,
                1.5,
                cell_batch,
                batch_ptr,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        g = torch.autograd.grad(loss(pos), pos, create_graph=True)[0]
        v = torch.randn_like(pos)
        hvp = torch.autograd.grad((g * v).sum(), pos)[0]
        assert torch.isfinite(hvp).all()
        assert hvp.shape == pos.shape

    def test_no_grad_path_unchanged(self, device):
        pos, cell_batch, batch_ptr = self._make_batch(device)
        nm_a, nn_a, sh_a = batch_cluster_tile_neighbor_list(
            pos,
            1.5,
            cell_batch,
            batch_ptr,
        )
        nm_b, nn_b, sh_b, d_b, v_b = batch_cluster_tile_neighbor_list(
            pos,
            1.5,
            cell_batch,
            batch_ptr,
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

    def test_grad_matches_fd_spot_check(self, device):
        """fp32 spot-check on a tight per-system cluster.  See the
        single-system class for the rationale on why we don't use
        ``gradcheck`` here.
        """
        torch.manual_seed(0)
        n_per = 4
        data = torch.randn(2 * n_per, 3, dtype=torch.float32, device=device) * 0.15
        batch_ptr = torch.tensor(
            [0, n_per, 2 * n_per], dtype=torch.int32, device=device
        )
        cell_batch = (
            torch.eye(3, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(2, 1, 1)
            * 20.0
        )
        eps = 1e-3

        def fn(p):
            *_, d, _ = batch_cluster_tile_neighbor_list(
                p,
                5.0,
                cell_batch,
                batch_ptr,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        pos = data.clone().requires_grad_(True)
        ana = torch.autograd.grad(fn(pos), pos)[0]
        fd = torch.zeros_like(ana)
        for i in range(2 * n_per):
            for j in range(3):
                pp = data.clone().requires_grad_(False)
                pp[i, j] += eps
                f_p = fn(pp).item()
                pm = data.clone().requires_grad_(False)
                pm[i, j] -= eps
                f_m = fn(pm).item()
                fd[i, j] = (f_p - f_m) / (2 * eps)
        max_abs_diff = (ana - fd).abs().max().item()
        max_ref = max(ana.abs().max().item(), fd.abs().max().item(), 1.0)
        assert max_abs_diff / max_ref < 5e-2, (
            f"analytical vs FD relative disagreement {max_abs_diff / max_ref:.3e}"
        )
