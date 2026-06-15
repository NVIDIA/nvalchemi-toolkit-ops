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

"""Tests for individual single-system cluster_tile kernel launchers.

The torch-binding suite at
``test/neighbors/bindings/torch/test_cluster_tile.py`` exercises the
end-to-end pipeline; this file targets the Warp support helpers and public launchers
(``_compute_morton``, ``_permute_gather_soa``, ``_tile_sort_pairs``,
``build_cluster_tile_list``, ``query_cluster_tile``, ``query_cluster_tile_coo``)
directly with ``wp.array`` inputs.
"""

import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.cluster_tile import (
    TILE_GROUP_SIZE,
    build_cluster_tile_list,
    query_cluster_tile,
)
from nvalchemiops.neighbors.cluster_tile.kernels import _bbox_distance_sq
from nvalchemiops.neighbors.cluster_tile.launchers import (
    _compute_morton,
    _permute_gather_soa,
    _tile_sort_pairs,
)

pytestmark = pytest.mark.gpu


@wp.kernel(enable_backward=False)
def _bbox_distance_sq_test_kernel(
    d: wp.array(dtype=wp.vec3f),
    rg_ext: wp.array(dtype=wp.vec3f),
    cg_ext: wp.array(dtype=wp.vec3f),
    out: wp.array(dtype=wp.float32),
) -> None:
    """Evaluate ``_bbox_distance_sq`` for test vectors."""
    i = wp.tid()
    out[i] = _bbox_distance_sq(d[i], rg_ext[i], cg_ext[i])


def _mat33f_from_torch(mat: torch.Tensor):
    """Zero-copy view a ``(1, 3, 3)`` or ``(3, 3)`` torch tensor as a
    ``wp.array(dtype=wp.mat33f, shape=(1,))``.  Cluster-tile kernels read
    ``cell`` / ``inv_cell`` as length-1 arrays and dereference ``[0]``
    inside the kernel body.
    """
    if mat.ndim == 2:
        mat = mat.unsqueeze(0)
    return wp.from_torch(
        mat.detach().contiguous().to(torch.float32),
        dtype=wp.mat33f,
        return_ctype=True,
    )


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("cluster_tile kernel tests require torch CUDA tensors")
    return "cuda:0"


class TestBBoxDistanceSqHelper:
    """Tests for the cluster bounding-box distance helper."""

    def test_overlap_and_gap_cases(self, device):
        """Helper returns zero for overlap and squared gap otherwise."""
        d = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 5.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        rg_ext = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cg_ext = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.5, 2.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        out = torch.empty(3, dtype=torch.float32, device=device)
        wp.launch(
            kernel=_bbox_distance_sq_test_kernel,
            dim=3,
            inputs=[
                wp.from_torch(d, dtype=wp.vec3f, return_ctype=True),
                wp.from_torch(rg_ext, dtype=wp.vec3f, return_ctype=True),
                wp.from_torch(cg_ext, dtype=wp.vec3f, return_ctype=True),
                wp.from_torch(out, dtype=wp.float32, return_ctype=True),
            ],
            device=device,
        )
        expected = torch.tensor([0.0, 1.0, 6.25], dtype=torch.float32, device=device)
        torch.testing.assert_close(out, expected)


class TestComputeMortonLauncher:
    """Smoke + invariant tests for the fused Morton + identity-init kernel."""

    def test_codes_are_in_30bit_range(self, device):
        """Real atoms get codes in ``[0, 2**30)``; padding slots use the
        sentinel ``0x40000000`` so they sort after any real entry.
        """
        torch.manual_seed(0)
        N = 32
        n_padded = N  # multiple of TILE_GROUP_SIZE
        positions = torch.rand(n_padded, 3, dtype=torch.float32, device=device) * 4.0
        cell = torch.eye(3, dtype=torch.float32, device=device) * 4.0
        inv_cell_torch = torch.linalg.inv(cell).contiguous()

        morton_codes = torch.zeros(n_padded, dtype=torch.int32, device=device)
        sorted_atom_index = torch.zeros(n_padded, dtype=torch.int32, device=device)
        num_neighbors = torch.zeros(N, dtype=torch.int32, device=device)
        num_tiles = torch.zeros(1, dtype=torch.int32, device=device)

        wp_positions = wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True)
        wp_codes = wp.from_torch(morton_codes, dtype=wp.int32, return_ctype=True)
        wp_sorted_idx = wp.from_torch(
            sorted_atom_index, dtype=wp.int32, return_ctype=True
        )
        wp_nn = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)
        wp_num_tiles = wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True)
        wp_inv_cell = _mat33f_from_torch(inv_cell_torch)

        _compute_morton(
            wp_positions,
            wp_inv_cell,
            N,
            wp_codes,
            wp_sorted_idx,
            wp_nn,
            wp_num_tiles,
            device,
        )

        codes_cpu = morton_codes.cpu()
        # All real atoms have a 30-bit Morton code (< 0x40000000).
        assert (codes_cpu[:N] < 0x40000000).all(), "Real Morton code exceeds 30 bits"
        # The identity-init outputs are zeroed/initialised by the same kernel.
        assert torch.equal(
            sorted_atom_index.cpu(),
            torch.arange(n_padded, dtype=torch.int32),
        )
        assert int(num_tiles.cpu().item()) == 0
        assert int(num_neighbors.cpu().sum().item()) == 0

    def test_padding_slots_use_sentinel(self, device):
        """Padding slots beyond ``natom`` get the 0x40000000 sentinel."""
        torch.manual_seed(1)
        N = 33
        n_padded = TILE_GROUP_SIZE * 2  # 64
        # Pad ``positions`` to ``n_padded`` (any in-cell value is fine).
        positions = torch.rand(n_padded, 3, dtype=torch.float32, device=device) * 4.0
        cell = torch.eye(3, dtype=torch.float32, device=device) * 4.0
        inv_cell_torch = torch.linalg.inv(cell).contiguous()
        morton_codes = torch.zeros(n_padded, dtype=torch.int32, device=device)
        sorted_atom_index = torch.zeros(n_padded, dtype=torch.int32, device=device)
        num_neighbors = torch.zeros(N, dtype=torch.int32, device=device)
        num_tiles = torch.zeros(1, dtype=torch.int32, device=device)

        _compute_morton(
            wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True),
            _mat33f_from_torch(inv_cell_torch),
            N,
            wp.from_torch(morton_codes, dtype=wp.int32, return_ctype=True),
            wp.from_torch(sorted_atom_index, dtype=wp.int32, return_ctype=True),
            wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True),
            wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
            device,
        )

        codes_cpu = morton_codes.cpu()
        # Padding slots ``[N, n_padded)`` carry the sentinel.
        assert (codes_cpu[N:n_padded] == 0x40000000).all(), (
            "Padding Morton code is not 0x40000000"
        )


class TestPermuteGatherSoaLauncher:
    """Tests for the AoS-to-SoA permute-and-gather launcher."""

    def test_identity_permutation(self, device):
        """Identity permutation yields ``sorted_pos_*[i] = positions[i]``."""
        torch.manual_seed(2)
        N = 32
        positions = torch.rand(N, 3, dtype=torch.float32, device=device)
        sorted_atom_index = torch.arange(N, dtype=torch.int32, device=device)
        sorted_pos_x = torch.zeros(N, dtype=torch.float32, device=device)
        sorted_pos_y = torch.zeros(N, dtype=torch.float32, device=device)
        sorted_pos_z = torch.zeros(N, dtype=torch.float32, device=device)

        _permute_gather_soa(
            wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True),
            wp.from_torch(sorted_atom_index, dtype=wp.int32, return_ctype=True),
            N,
            wp.from_torch(sorted_pos_x, dtype=wp.float32, return_ctype=True),
            wp.from_torch(sorted_pos_y, dtype=wp.float32, return_ctype=True),
            wp.from_torch(sorted_pos_z, dtype=wp.float32, return_ctype=True),
            device,
        )

        torch.testing.assert_close(sorted_pos_x, positions[:, 0])
        torch.testing.assert_close(sorted_pos_y, positions[:, 1])
        torch.testing.assert_close(sorted_pos_z, positions[:, 2])

    def test_reverse_permutation(self, device):
        """Reverse permutation yields the AoS rows in reversed order."""
        N = 8
        positions = torch.arange(N * 3, dtype=torch.float32, device=device).reshape(
            N, 3
        )
        sorted_atom_index = torch.arange(
            N - 1, -1, -1, dtype=torch.int32, device=device
        )
        sorted_pos_x = torch.zeros(N, dtype=torch.float32, device=device)
        sorted_pos_y = torch.zeros(N, dtype=torch.float32, device=device)
        sorted_pos_z = torch.zeros(N, dtype=torch.float32, device=device)

        _permute_gather_soa(
            wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True),
            wp.from_torch(sorted_atom_index, dtype=wp.int32, return_ctype=True),
            N,
            wp.from_torch(sorted_pos_x, dtype=wp.float32, return_ctype=True),
            wp.from_torch(sorted_pos_y, dtype=wp.float32, return_ctype=True),
            wp.from_torch(sorted_pos_z, dtype=wp.float32, return_ctype=True),
            device,
        )

        torch.testing.assert_close(sorted_pos_x, positions.flip(0)[:, 0])


class TestTileSortPairsLauncher:
    """Tests for the bitonic-sort key-value launcher."""

    @pytest.mark.parametrize("N", [1024, 2048])
    def test_supported_sizes_launch(self, device, N):
        """The N=1024 and N=2048 specializations should launch and sort."""
        torch.manual_seed(N)
        keys = torch.randint(0, 1 << 30, (N,), dtype=torch.int32, device=device)
        values = torch.arange(N, dtype=torch.int32, device=device)
        keys_sorted_ref, perm = torch.sort(keys)

        ok = _tile_sort_pairs(
            wp.from_torch(keys, dtype=wp.int32, return_ctype=True),
            wp.from_torch(values, dtype=wp.int32, return_ctype=True),
            N,
            device,
        )
        assert ok, f"_tile_sort_pairs should have a specialization at N={N}"
        torch.testing.assert_close(keys, keys_sorted_ref)
        torch.testing.assert_close(values, perm.to(dtype=torch.int32))

    def test_unsupported_size_returns_false(self, device):
        """Unsupported ``natom`` should leave inputs untouched and return False."""
        N = 777  # not a specialization
        keys = torch.randint(0, 100, (N,), dtype=torch.int32, device=device)
        values = torch.arange(N, dtype=torch.int32, device=device)
        keys_before = keys.clone()
        values_before = values.clone()

        ok = _tile_sort_pairs(
            wp.from_torch(keys, dtype=wp.int32, return_ctype=True),
            wp.from_torch(values, dtype=wp.int32, return_ctype=True),
            N,
            device,
        )
        assert not ok, "Unsupported size should report no work done"
        assert torch.equal(keys, keys_before)
        assert torch.equal(values, values_before)


class TestBuildTileNeighborListLauncher:
    """End-to-end smoke test of the public Warp tile-build launcher."""

    def test_emits_pairs(self, device):
        """A 64-atom cubic system should emit at least one tile pair."""
        torch.manual_seed(3)
        n_padded = 64
        positions = torch.rand(n_padded, 3, dtype=torch.float32, device=device) * 4.0
        cell_torch = torch.eye(3, dtype=torch.float32, device=device) * 4.0
        inv_cell_torch = torch.linalg.inv(cell_torch).contiguous()
        cutoff = 2.5
        natom = n_padded
        ngroup = n_padded // TILE_GROUP_SIZE
        ngroup_padded = TILE_GROUP_SIZE * 2
        max_tiles = ngroup * ngroup

        # Morton + sort + gather upstream (use the launchers under test).
        morton_codes = torch.zeros(n_padded, dtype=torch.int32, device=device)
        sorted_atom_index = torch.zeros(n_padded, dtype=torch.int32, device=device)
        sorted_pos_x = torch.zeros(n_padded, dtype=torch.float32, device=device)
        sorted_pos_y = torch.zeros(n_padded, dtype=torch.float32, device=device)
        sorted_pos_z = torch.zeros(n_padded, dtype=torch.float32, device=device)
        num_neighbors = torch.zeros(natom, dtype=torch.int32, device=device)
        num_tiles = torch.zeros(1, dtype=torch.int32, device=device)

        _compute_morton(
            wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True),
            _mat33f_from_torch(inv_cell_torch),
            natom,
            wp.from_torch(morton_codes, dtype=wp.int32, return_ctype=True),
            wp.from_torch(sorted_atom_index, dtype=wp.int32, return_ctype=True),
            wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True),
            wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
            device,
        )
        # Single-block bitonic sort works at n_padded=1024/2048; for N=64
        # fall back to a torch-side argsort, which is enough for this test.
        perm = torch.argsort(morton_codes)
        sorted_atom_index.copy_(perm.to(torch.int32))
        _permute_gather_soa(
            wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True),
            wp.from_torch(sorted_atom_index, dtype=wp.int32, return_ctype=True),
            natom,
            wp.from_torch(sorted_pos_x, dtype=wp.float32, return_ctype=True),
            wp.from_torch(sorted_pos_y, dtype=wp.float32, return_ctype=True),
            wp.from_torch(sorted_pos_z, dtype=wp.float32, return_ctype=True),
            device,
        )

        # Allocate the tile-build state arrays (centers, extents, tile lists).
        group_ctr_x = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ctr_y = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ctr_z = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ext_x = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ext_y = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        group_ext_z = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
        tile_row_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
        tile_col_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)

        build_cluster_tile_list(
            sorted_pos_x=wp.from_torch(
                sorted_pos_x, dtype=wp.float32, return_ctype=True
            ),
            sorted_pos_y=wp.from_torch(
                sorted_pos_y, dtype=wp.float32, return_ctype=True
            ),
            sorted_pos_z=wp.from_torch(
                sorted_pos_z, dtype=wp.float32, return_ctype=True
            ),
            cell=_mat33f_from_torch(cell_torch),
            inv_cell=_mat33f_from_torch(inv_cell_torch),
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
            wp_dtype=wp.float32,
            device=device,
        )

        n_tiles = int(num_tiles.cpu().item())
        assert n_tiles > 0, "Expected at least one tile pair"
        # Tile indices fall within [0, ngroup).
        used = tile_row_group[:n_tiles].cpu()
        used_col = tile_col_group[:n_tiles].cpu()
        assert int(used.max().item()) < ngroup
        assert int(used_col.max().item()) < ngroup

        # query_cluster_tile smoke: converted neighbor matrix shape matches.
        max_neighbors = 32
        nm = torch.full((natom, max_neighbors), natom, dtype=torch.int32, device=device)
        nn = torch.zeros(natom, dtype=torch.int32, device=device)
        nms = torch.zeros((natom, max_neighbors, 3), dtype=torch.int32, device=device)
        query_cluster_tile(
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
            num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
            tile_row_group=wp.from_torch(
                tile_row_group, dtype=wp.int32, return_ctype=True
            ),
            tile_col_group=wp.from_torch(
                tile_col_group, dtype=wp.int32, return_ctype=True
            ),
            cell=_mat33f_from_torch(cell_torch),
            inv_cell=_mat33f_from_torch(inv_cell_torch),
            cutoff=cutoff,
            natom=natom,
            neighbor_matrix=wp.from_torch(nm, dtype=wp.int32, return_ctype=True),
            neighbor_matrix_shifts=wp.from_torch(
                nms, dtype=wp.int32, return_ctype=True
            ),
            num_neighbors=wp.from_torch(nn, dtype=wp.int32, return_ctype=True),
            wp_dtype=wp.float32,
            device=device,
        )
        assert int(nn.sum().item()) > 0, "Expected at least one neighbor pair emitted"
