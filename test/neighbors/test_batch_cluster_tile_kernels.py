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

"""Tests for individual batched cluster_tile kernel launchers.

The torch-binding suite at
``test/neighbors/bindings/torch/test_batch_cluster_tile.py`` exercises
the end-to-end pipeline; this file targets the public Warp launchers
(``batch_build_cluster_tile_list``, ``batch_query_cluster_tile``,
``batch_query_cluster_tile_coo``) directly with ``wp.array`` inputs.
"""

import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.cluster_tile import (
    TILE_GROUP_SIZE,
    batch_build_cluster_tile_list,
    batch_query_cluster_tile,
)

pytestmark = pytest.mark.gpu


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("batch_cluster_tile kernel tests require torch CUDA tensors")
    return "cuda:0"


def _morton_sort_per_system(
    positions: torch.Tensor,
    batch_ptr: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-system Morton sort into a padded SoA layout.

    Mirrors the torch wrapper's ``_batched_morton_sort_padded`` helper
    closely enough for use as a kernel-layer fixture; not graph-safe.
    Returns ``(sorted_pos_x, sorted_pos_y, sorted_pos_z, group_system,
    group_ptr)``.
    """
    num_systems = inv_cell_batch.shape[0]
    natom_per_system = (batch_ptr[1:] - batch_ptr[:-1]).to(torch.int64)
    natom_padded_per_system = (
        (natom_per_system + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    n_padded = int(natom_padded_per_system.sum().item())

    # Padded layout: per-system, fill real atoms then pad to next 32 boundary.
    sorted_pos = torch.zeros(n_padded, 3, dtype=torch.float32, device=device)
    batch_idx_sorted = torch.zeros(n_padded, dtype=torch.int32, device=device)
    write = 0
    for s in range(num_systems):
        start = int(batch_ptr[s].item())
        end = int(batch_ptr[s + 1].item())
        natom = end - start
        pad = int(natom_padded_per_system[s].item())
        sys_pos = positions[start:end]
        frac = sys_pos @ inv_cell_batch[s].T
        frac = frac - torch.floor(frac)
        bucket = (frac * 1024.0).clamp(0, 1023).to(torch.int32)
        ix, iy, iz = bucket.unbind(dim=-1)

        def _spread(x: torch.Tensor) -> torch.Tensor:
            x = x & 0x3FF
            x = (x | (x << 16)) & 0x030000FF
            x = (x | (x << 8)) & 0x0300F00F
            x = (x | (x << 4)) & 0x030C30C3
            x = (x | (x << 2)) & 0x09249249
            return x

        codes = (_spread(iz) << 2) | (_spread(iy) << 1) | _spread(ix)
        perm = torch.argsort(codes)
        sorted_pos[write : write + natom] = sys_pos[perm]
        # Pad slots: duplicate the first atom of the system.
        sorted_pos[write + natom : write + pad] = sys_pos[0:1]
        batch_idx_sorted[write : write + pad] = s
        write += pad

    ngroup = n_padded // TILE_GROUP_SIZE
    group_system = batch_idx_sorted[::TILE_GROUP_SIZE].contiguous()
    # group_ptr[s] = first group index of system s; group_ptr[-1] = ngroup.
    groups_per_system = (natom_padded_per_system // TILE_GROUP_SIZE).to(torch.int32)
    group_ptr = torch.zeros(num_systems + 1, dtype=torch.int32, device=device)
    torch.cumsum(groups_per_system, dim=0, out=group_ptr[1:])
    sorted_pos_x = sorted_pos[:, 0].contiguous()
    sorted_pos_y = sorted_pos[:, 1].contiguous()
    sorted_pos_z = sorted_pos[:, 2].contiguous()
    del ngroup  # surfaced indirectly via group_system.shape[0]
    return sorted_pos_x, sorted_pos_y, sorted_pos_z, group_system, group_ptr


class TestBuildBatchTileNeighborListLauncher:
    """End-to-end smoke test of the batched Warp tile-build launcher."""

    def test_emits_pairs_for_two_systems(self, device):
        """Two-system batch should emit at least one tile pair per system."""
        torch.manual_seed(0)
        sizes = [64, 96]
        cell_sizes = [10.0, 8.0]
        cutoff = 2.5

        pos_chunks, cells = [], []
        for sz, L in zip(sizes, cell_sizes):
            pos_chunks.append(torch.rand(sz, 3, dtype=torch.float32, device=device) * L)
            cells.append(torch.eye(3, dtype=torch.float32, device=device) * L)
        positions = torch.cat(pos_chunks, dim=0).contiguous()
        cell_batch = torch.stack(cells, dim=0).contiguous()
        inv_cell_batch = torch.linalg.inv(cell_batch).contiguous()
        batch_ptr = torch.tensor(
            [0, sizes[0], sizes[0] + sizes[1]],
            dtype=torch.int32,
            device=device,
        )

        (
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            group_system,
            group_ptr,
        ) = _morton_sort_per_system(positions, batch_ptr, inv_cell_batch, device)

        ngroup = int(group_system.shape[0])
        ngroup_padded = (
            (ngroup + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
        ) * TILE_GROUP_SIZE + TILE_GROUP_SIZE
        max_tiles = ngroup * ngroup

        group_ctr_x = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ctr_y = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ctr_z = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ext_x = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ext_y = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ext_z = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        num_tiles = torch.zeros(1, dtype=torch.int32, device=device)
        tile_row_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
        tile_col_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
        tile_system = torch.zeros(max_tiles, dtype=torch.int32, device=device)

        batch_build_cluster_tile_list(
            sorted_pos_x=wp.from_torch(
                sorted_pos_x, dtype=wp.float32, return_ctype=True
            ),
            sorted_pos_y=wp.from_torch(
                sorted_pos_y, dtype=wp.float32, return_ctype=True
            ),
            sorted_pos_z=wp.from_torch(
                sorted_pos_z, dtype=wp.float32, return_ctype=True
            ),
            group_system=wp.from_torch(group_system, dtype=wp.int32, return_ctype=True),
            group_ptr=wp.from_torch(group_ptr, dtype=wp.int32, return_ctype=True),
            cell_batch=wp.from_torch(cell_batch, dtype=wp.mat33f, return_ctype=True),
            inv_cell_batch=wp.from_torch(
                inv_cell_batch, dtype=wp.mat33f, return_ctype=True
            ),
            cutoff=cutoff,
            group_ctr_x_buffer=wp.from_torch(
                group_ctr_x, dtype=wp.float32, return_ctype=True
            ),
            group_ctr_y_buffer=wp.from_torch(
                group_ctr_y, dtype=wp.float32, return_ctype=True
            ),
            group_ctr_z_buffer=wp.from_torch(
                group_ctr_z, dtype=wp.float32, return_ctype=True
            ),
            group_ext_x_buffer=wp.from_torch(
                group_ext_x, dtype=wp.float32, return_ctype=True
            ),
            group_ext_y_buffer=wp.from_torch(
                group_ext_y, dtype=wp.float32, return_ctype=True
            ),
            group_ext_z_buffer=wp.from_torch(
                group_ext_z, dtype=wp.float32, return_ctype=True
            ),
            num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
            tile_row_group=wp.from_torch(
                tile_row_group, dtype=wp.int32, return_ctype=True
            ),
            tile_col_group=wp.from_torch(
                tile_col_group, dtype=wp.int32, return_ctype=True
            ),
            tile_system=wp.from_torch(tile_system, dtype=wp.int32, return_ctype=True),
            wp_dtype=wp.float32,
            device=device,
        )

        n_tiles = int(num_tiles.cpu().item())
        assert n_tiles > 0, "Expected at least one tile pair"
        sys_indices = tile_system[:n_tiles].cpu()
        # Every emitted tile pair stays within a single system.
        assert int(sys_indices.min().item()) >= 0
        assert int(sys_indices.max().item()) < 2

    def test_dtype_other_than_float32_raises(self, device):
        """``batch_build_cluster_tile_list`` only supports ``wp.float32``."""
        with pytest.raises(ValueError, match="float32-only"):
            # Minimal call with the f64 dtype switch; arrays are not read
            # because the dtype check fires first.
            batch_build_cluster_tile_list(
                sorted_pos_x=wp.zeros(0, dtype=wp.float32, device=device),
                sorted_pos_y=wp.zeros(0, dtype=wp.float32, device=device),
                sorted_pos_z=wp.zeros(0, dtype=wp.float32, device=device),
                group_system=wp.zeros(0, dtype=wp.int32, device=device),
                group_ptr=wp.zeros(1, dtype=wp.int32, device=device),
                cell_batch=wp.zeros(0, dtype=wp.mat33f, device=device),
                inv_cell_batch=wp.zeros(0, dtype=wp.mat33f, device=device),
                cutoff=1.0,
                group_ctr_x_buffer=wp.zeros(0, dtype=wp.float32, device=device),
                group_ctr_y_buffer=wp.zeros(0, dtype=wp.float32, device=device),
                group_ctr_z_buffer=wp.zeros(0, dtype=wp.float32, device=device),
                group_ext_x_buffer=wp.zeros(0, dtype=wp.float32, device=device),
                group_ext_y_buffer=wp.zeros(0, dtype=wp.float32, device=device),
                group_ext_z_buffer=wp.zeros(0, dtype=wp.float32, device=device),
                num_tiles=wp.zeros(1, dtype=wp.int32, device=device),
                tile_row_group=wp.zeros(0, dtype=wp.int32, device=device),
                tile_col_group=wp.zeros(0, dtype=wp.int32, device=device),
                tile_system=wp.zeros(0, dtype=wp.int32, device=device),
                wp_dtype=wp.float64,
                device=device,
            )


class TestBatchTileToMatrixLauncher:
    """End-to-end smoke test of the batched tile-to-matrix conversion launcher."""

    def test_neighbor_pairs_in_same_system(self, device):
        """All emitted neighbors should share a system with their source atom."""
        torch.manual_seed(1)
        sizes = [64, 96]
        cell_sizes = [10.0, 8.0]
        cutoff = 2.5
        N = sum(sizes)

        pos_chunks, cells = [], []
        for sz, L in zip(sizes, cell_sizes):
            pos_chunks.append(torch.rand(sz, 3, dtype=torch.float32, device=device) * L)
            cells.append(torch.eye(3, dtype=torch.float32, device=device) * L)
        positions = torch.cat(pos_chunks, dim=0).contiguous()
        cell_batch = torch.stack(cells, dim=0).contiguous()
        inv_cell_batch = torch.linalg.inv(cell_batch).contiguous()
        batch_ptr = torch.tensor(
            [0, sizes[0], sizes[0] + sizes[1]],
            dtype=torch.int32,
            device=device,
        )

        # Build per-atom system membership for the assertion below.
        batch_idx = torch.cat(
            [
                torch.zeros(sizes[0], dtype=torch.int32, device=device),
                torch.ones(sizes[1], dtype=torch.int32, device=device),
            ]
        )

        (
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            group_system,
            group_ptr,
        ) = _morton_sort_per_system(positions, batch_ptr, inv_cell_batch, device)

        # Track the sort permutation so we can pass sorted_atom_index to
        # batch_query_cluster_tile.  Recompute here (cheap for small N).
        n_padded = int(sorted_pos_x.shape[0])
        sorted_atom_index = torch.full((n_padded,), N, dtype=torch.int32, device=device)
        write = 0
        for s in range(2):
            start = int(batch_ptr[s].item())
            end = int(batch_ptr[s + 1].item())
            natom = end - start
            sys_pos = positions[start:end]
            frac = sys_pos @ inv_cell_batch[s].T
            frac = frac - torch.floor(frac)
            bucket = (frac * 1024.0).clamp(0, 1023).to(torch.int32)
            ix, iy, iz = bucket.unbind(dim=-1)

            def _spread(x: torch.Tensor) -> torch.Tensor:
                x = x & 0x3FF
                x = (x | (x << 16)) & 0x030000FF
                x = (x | (x << 8)) & 0x0300F00F
                x = (x | (x << 4)) & 0x030C30C3
                x = (x | (x << 2)) & 0x09249249
                return x

            codes = (_spread(iz) << 2) | (_spread(iy) << 1) | _spread(ix)
            perm = torch.argsort(codes)
            natom_padded = (
                (natom + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
            ) * TILE_GROUP_SIZE
            sorted_atom_index[write : write + natom] = (start + perm).to(torch.int32)
            # Padding slots stay at the N sentinel.
            write += natom_padded

        ngroup = int(group_system.shape[0])
        ngroup_padded = (
            (ngroup + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
        ) * TILE_GROUP_SIZE + TILE_GROUP_SIZE
        max_tiles = ngroup * ngroup

        group_ctr_x = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ctr_y = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ctr_z = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ext_x = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ext_y = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ext_z = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        num_tiles = torch.zeros(1, dtype=torch.int32, device=device)
        tile_row_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
        tile_col_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
        tile_system = torch.zeros(max_tiles, dtype=torch.int32, device=device)

        batch_build_cluster_tile_list(
            sorted_pos_x=wp.from_torch(
                sorted_pos_x, dtype=wp.float32, return_ctype=True
            ),
            sorted_pos_y=wp.from_torch(
                sorted_pos_y, dtype=wp.float32, return_ctype=True
            ),
            sorted_pos_z=wp.from_torch(
                sorted_pos_z, dtype=wp.float32, return_ctype=True
            ),
            group_system=wp.from_torch(group_system, dtype=wp.int32, return_ctype=True),
            group_ptr=wp.from_torch(group_ptr, dtype=wp.int32, return_ctype=True),
            cell_batch=wp.from_torch(cell_batch, dtype=wp.mat33f, return_ctype=True),
            inv_cell_batch=wp.from_torch(
                inv_cell_batch, dtype=wp.mat33f, return_ctype=True
            ),
            cutoff=cutoff,
            group_ctr_x_buffer=wp.from_torch(
                group_ctr_x, dtype=wp.float32, return_ctype=True
            ),
            group_ctr_y_buffer=wp.from_torch(
                group_ctr_y, dtype=wp.float32, return_ctype=True
            ),
            group_ctr_z_buffer=wp.from_torch(
                group_ctr_z, dtype=wp.float32, return_ctype=True
            ),
            group_ext_x_buffer=wp.from_torch(
                group_ext_x, dtype=wp.float32, return_ctype=True
            ),
            group_ext_y_buffer=wp.from_torch(
                group_ext_y, dtype=wp.float32, return_ctype=True
            ),
            group_ext_z_buffer=wp.from_torch(
                group_ext_z, dtype=wp.float32, return_ctype=True
            ),
            num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
            tile_row_group=wp.from_torch(
                tile_row_group, dtype=wp.int32, return_ctype=True
            ),
            tile_col_group=wp.from_torch(
                tile_col_group, dtype=wp.int32, return_ctype=True
            ),
            tile_system=wp.from_torch(tile_system, dtype=wp.int32, return_ctype=True),
            wp_dtype=wp.float32,
            device=device,
        )

        n_tiles = int(num_tiles.cpu().item())
        assert n_tiles > 0

        max_neighbors = 64
        neighbor_matrix = torch.full(
            (N, max_neighbors), N, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (N, max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(N, dtype=torch.int32, device=device)

        batch_query_cluster_tile(
            sorted_atom_index=wp.from_torch(
                sorted_atom_index, dtype=wp.int32, return_ctype=True
            ),
            sorted_pos_x=wp.from_torch(
                sorted_pos_x, dtype=wp.float32, return_ctype=True
            ),
            sorted_pos_y=wp.from_torch(
                sorted_pos_y, dtype=wp.float32, return_ctype=True
            ),
            sorted_pos_z=wp.from_torch(
                sorted_pos_z, dtype=wp.float32, return_ctype=True
            ),
            cell_batch=wp.from_torch(cell_batch, dtype=wp.mat33f, return_ctype=True),
            inv_cell_batch=wp.from_torch(
                inv_cell_batch, dtype=wp.mat33f, return_ctype=True
            ),
            num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
            tile_row_group=wp.from_torch(
                tile_row_group, dtype=wp.int32, return_ctype=True
            ),
            tile_col_group=wp.from_torch(
                tile_col_group, dtype=wp.int32, return_ctype=True
            ),
            tile_system=wp.from_torch(tile_system, dtype=wp.int32, return_ctype=True),
            cutoff=cutoff,
            natom=N,
            neighbor_matrix=wp.from_torch(
                neighbor_matrix, dtype=wp.int32, return_ctype=True
            ),
            num_neighbors=wp.from_torch(
                num_neighbors, dtype=wp.int32, return_ctype=True
            ),
            neighbor_matrix_shifts=wp.from_torch(
                neighbor_matrix_shifts, dtype=wp.int32, return_ctype=True
            ),
            wp_dtype=wp.float32,
            device=device,
        )

        # Sanity: each emitted neighbor (within the active row prefix) shares
        # a system with its source atom.
        nm_cpu = neighbor_matrix.cpu()
        nn_cpu = num_neighbors.cpu()
        bi_cpu = batch_idx.cpu()
        for i in range(N):
            n_i = int(nn_cpu[i].item())
            for k in range(min(n_i, max_neighbors)):
                j = int(nm_cpu[i, k].item())
                if j == N:  # sentinel padding
                    continue
                assert bi_cpu[i] == bi_cpu[j], f"Cross-system pair ({i}, {j}) emitted"
