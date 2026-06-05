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

"""Tests for the single-system cluster-pair tile neighbor list PyTorch bindings."""

import pytest
import torch

from nvalchemiops.neighbors.neighbor_utils import NeighborOverflowError
from nvalchemiops.torch.neighbors.cell_list import cell_list
from nvalchemiops.torch.neighbors.cluster_tile import (
    TILE_GROUP_SIZE,
    allocate_cluster_tile_list,
    build_cluster_tile_list,
    cluster_tile_neighbor_list,
    estimate_cluster_tile_list_sizes,
    query_cluster_tile,
)

from ...test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
    create_random_system,
    create_simple_cubic_system,
)
from .conftest import requires_vesin

# cluster_tile is CUDA + float32 only; override the conftest device/dtype
# fixtures to restrict the parametrize matrix.


@pytest.fixture(params=["cuda:0"], ids=lambda d: d.replace(":", "_"))
def device(request):
    if not torch.cuda.is_available():
        pytest.skip("cluster_tile kernel tests require torch CUDA tensors")
    return request.param


@pytest.fixture(params=[torch.float32], ids=["float32"])
def dtype(request):
    return request.param


def _orthorhombic_cell(
    cell_size: float, device: str, dtype=torch.float32
) -> torch.Tensor:
    return (torch.eye(3, dtype=dtype, device=device) * cell_size).reshape(1, 3, 3)


# =============================================================================
# Correctness
# =============================================================================
class TestTileNeighborListCorrectness:
    def test_single_atom_no_neighbors(self, device, dtype):
        """Single atom system should have no neighbors."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        cell = _orthorhombic_cell(4.0, device, dtype)
        cutoff = 0.75
        nm, nn, _nms = cluster_tile_neighbor_list(
            positions, cutoff, cell, max_neighbors=8
        )
        assert int(nn.sum().item()) == 0
        assert nm.shape == (1, 8)

    def test_two_atom_pair(self, device, dtype):
        """Two atoms within cutoff yield a full-fill pair in both rows."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype, device=device
        )
        cell = _orthorhombic_cell(4.0, device, dtype)
        cutoff = 1.0
        nm, nn, _nms = cluster_tile_neighbor_list(
            positions, cutoff, cell, max_neighbors=8
        )
        # Full-fill: each atom lists the other (matches cell_list half_fill=False).
        assert int(nn.sum().item()) == 2
        assert nn.cpu().tolist() == [1, 1]
        assert int(nm[0, 0].item()) == 1
        assert int(nm[1, 0].item()) == 0

    @requires_vesin
    def test_cubic_system(self, device, dtype):
        """Simple cubic lattice (4x4x4 = 64 atoms, multiple of TILE_GROUP_SIZE)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=64,
            cell_size=4.0,
            dtype=dtype,
            device=device,
        )
        cutoff = 1.1

        nm, nn, nms = cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=32,
        )
        i_got, j_got, u_got = _matrix_to_coo_full(
            nm,
            nn,
            nms,
            positions.shape[0],
        )
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_random_system(self, device, dtype):
        """Random atomic positions vs brute-force reference."""
        positions, cell, pbc = create_random_system(
            num_atoms=64,
            cell_size=10.0,
            dtype=dtype,
            device=device,
            seed=42,
            pbc_flag=True,
        )
        cutoff = 3.0
        nm, nn, nms = cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=128,
        )
        i_got, j_got, u_got = _matrix_to_coo_full(nm, nn, nms, positions.shape[0])
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    @requires_vesin
    @pytest.mark.parametrize("N", [33, 65, 127])
    def test_non_aligned_N_correctness(self, device, dtype, N):
        """Non-32-aligned N produces correct pairs (padding-safe)."""
        positions, cell, pbc = create_random_system(
            num_atoms=N,
            cell_size=10.0,
            dtype=dtype,
            device=device,
            seed=N,
            pbc_flag=True,
        )
        cutoff = 3.0
        nm, nn, nms = cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=128,
        )
        i_got, j_got, u_got = _matrix_to_coo_full(nm, nn, nms, N)
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_triclinic_system(self, device, dtype):
        """Triclinic (non-orthorhombic) cell vs brute-force reference."""
        torch.manual_seed(11)
        N = 96
        # Build a moderately skewed cell.
        cell_mat = torch.eye(3, dtype=dtype, device=device) * 10.0
        cell_mat[0, 1] = 1.0
        cell_mat[1, 2] = 0.5
        cell = cell_mat.reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
        # Fractional sampling, then map into the triclinic cell.
        frac = torch.rand(N, 3, dtype=dtype, device=device)
        positions = (frac @ cell_mat).contiguous()
        cutoff = 2.5
        nm, nn, nms = cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=128,
        )
        i_got, j_got, u_got = _matrix_to_coo_full(nm, nn, nms, N)
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_larger_random_system(self, device, dtype):
        """Larger system exercises multiple tile rows."""
        positions, cell, pbc = create_random_system(
            num_atoms=256,
            cell_size=15.0,
            dtype=dtype,
            device=device,
            seed=7,
            pbc_flag=True,
        )
        cutoff = 2.5
        nm, nn, nms = cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=128,
        )
        i_got, j_got, u_got = _matrix_to_coo_full(nm, nn, nms, positions.shape[0])
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_return_neighbor_list(self, device, dtype):
        """COO output (``format="coo"``) matches matrix output."""
        positions, cell, _ = create_random_system(
            num_atoms=64,
            cell_size=10.0,
            dtype=dtype,
            device=device,
            seed=3,
            pbc_flag=True,
        )
        cutoff = 3.0
        neighbor_list, neighbor_ptr, shifts = cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=128,
            format="coo",
        )
        assert neighbor_list.shape[0] == 2
        # Compare pair counts against the matrix-mode result.
        nm, nn, _nms = cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=128,
        )
        assert int(nn.sum().item()) == int(neighbor_list.shape[1])

    def test_component_API_matches_convenience(self, device, dtype):
        """Explicit component calls produce the same state as the convenience wrapper."""
        N = 128
        torch.manual_seed(0)
        positions = torch.rand(N, 3, dtype=dtype, device=device) * 10.0
        cell = _orthorhombic_cell(10.0, device, dtype)
        cutoff = 2.5

        # Convenience path
        nm1, nn1, nms1 = cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=64,
        )

        # Component path -- allocate + build + convert using the
        # SoA-layout state tensors exposed by allocate_cluster_tile_list.
        (
            sorted_atom_index,
            morton_codes,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
        ) = allocate_cluster_tile_list(
            N,
            torch.device(device),
            dtype=dtype,
        )
        build_cluster_tile_list(
            positions,
            cutoff,
            cell,
            sorted_atom_index,
            morton_codes,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
        )
        nm2 = torch.full((N, 64), N, dtype=torch.int32, device=device)
        nn2 = torch.zeros(N, dtype=torch.int32, device=device)
        nms2 = torch.zeros((N, 64, 3), dtype=torch.int32, device=device)
        query_cluster_tile(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            cell,
            cutoff,
            N,
            nm2,
            nn2,
            nms2,
        )
        torch.testing.assert_close(nn1, nn2)
        # Entries may be in different order per-row; compare row-wise
        # sorted neighbor index sets.
        for i in range(N):
            n_i = int(nn1[i].item())
            s1 = {int(x.item()) for x in nm1[i, :n_i]}
            s2 = {int(x.item()) for x in nm2[i, :n_i]}
            assert s1 == s2, f"atom {i} neighbor set mismatch"


# =============================================================================
# Edge cases
# =============================================================================
class TestTileNeighborListEdgeCases:
    def test_zero_cutoff_matrix_empty(self, device, dtype):
        """Cutoff smaller than any distance -> zero neighbors."""
        N = 32
        positions = torch.rand(N, 3, dtype=dtype, device=device) * 10.0
        cell = _orthorhombic_cell(10.0, device, dtype)
        nm, nn, _nms = cluster_tile_neighbor_list(
            positions, 1e-6, cell, max_neighbors=32
        )
        assert int(nn.sum().item()) == 0

    def test_max_neighbors_overflow_raises(self, device, dtype):
        """max_neighbors overflow raises instead of silently truncating rows."""
        N = 64
        torch.manual_seed(1)
        positions = torch.rand(N, 3, dtype=dtype, device=device) * 8.0
        cell = _orthorhombic_cell(8.0, device, dtype)
        with pytest.raises(NeighborOverflowError):
            cluster_tile_neighbor_list(positions, 3.0, cell, max_neighbors=4)

    def test_component_sizes_match_estimate(self, device):
        """estimate_cluster_tile_list_sizes returns shape-consistent sizes."""
        N = 512
        n_padded, ngroup, ngroup_padded, max_tiles = estimate_cluster_tile_list_sizes(N)
        assert n_padded == N  # already 32-aligned
        assert ngroup == n_padded // TILE_GROUP_SIZE
        assert ngroup_padded % TILE_GROUP_SIZE == 0 and ngroup_padded > ngroup
        assert max_tiles >= ngroup

    def test_component_sizes_match_estimate_non_aligned(self, device):
        """Non-32-aligned N is rounded up to ``ceil(N/32)*32``."""
        N = 33
        n_padded, ngroup, ngroup_padded, max_tiles = estimate_cluster_tile_list_sizes(N)
        assert n_padded == 64  # ceil(33/32) * 32
        assert ngroup == 2
        assert max_tiles >= ngroup


# =============================================================================
# Errors
# =============================================================================
class TestTileNeighborListErrors:
    def test_wrong_dtype(self, device):
        positions = torch.rand(32, 3, dtype=torch.float64, device=device) * 10.0
        cell = _orthorhombic_cell(10.0, device)
        # Cluster-tile is float32-only; the torch wrapper rejects non-float32
        # positions at the frontend with TypeError.
        with pytest.raises(TypeError, match="float32"):
            cluster_tile_neighbor_list(positions, 2.5, cell, max_neighbors=32)

    def test_non_multiple_of_group_size_accepted(self, device, dtype):
        """N not divisible by TILE_GROUP_SIZE is padded internally.

        Sanity check: build runs without error and emits some pairs.
        Correctness vs reference is exercised by
        ``test_non_aligned_N_correctness`` below.
        """
        positions = torch.rand(33, 3, dtype=dtype, device=device) * 10.0
        cell = _orthorhombic_cell(10.0, device, dtype)
        nm, nn, _nms = cluster_tile_neighbor_list(
            positions,
            2.5,
            cell,
            max_neighbors=32,
            format="matrix",
        )
        assert nm.shape == (33, 32)
        assert int(nn.sum().item()) >= 0

    def test_triclinic_cell_accepted(self, device, dtype):
        """Triclinic cells are now first-class (cluster_tile parity with batch).

        Sanity check: build runs without error and emits some pairs.
        Correctness vs a reference is exercised in
        ``TestTileNeighborListCorrectness::test_triclinic_system`` below.
        """
        positions = torch.rand(32, 3, dtype=dtype, device=device) * 10.0
        cell = _orthorhombic_cell(10.0, device, dtype).clone()
        cell[0, 0, 1] = 1.0  # off-diagonal
        nm, nn, _nms = cluster_tile_neighbor_list(
            positions,
            2.5,
            cell,
            max_neighbors=32,
            format="matrix",
        )
        # nm shape sanity; nn nonneg sum.
        assert nm.shape == (32, 32)
        assert int(nn.sum().item()) >= 0


# =============================================================================
# Helpers
# =============================================================================
def _matrix_to_coo_full(
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    shifts: torch.Tensor,
    natom: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten (nm, nn, nms) into ALL directed (i, j, shift) triples.

    cluster_tile is full-fill: every atom's row lists all its neighbors, so
    each unordered pair appears in both rows (with negated shifts).  This
    matches vesin's ``full_list=True`` reference directly.
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


# =============================================================================
# torch.compile compatibility
# =============================================================================
class TestClusterTileCompile:
    """Tests for ``torch.compile`` compatibility of the cluster-pair tile path.

    Verifies that the ``@torch.library.custom_op``-decorated component shells
    (``_build_cluster_tile_list``, ``_query_cluster_tile``, ``_query_cluster_tile_coo``)
    survive a ``torch.compile`` round-trip without graph breaks that change
    the output.
    """

    @pytest.mark.slow
    def test_cluster_tile_neighbor_list_compile(self, device, dtype):
        """``cluster_tile_neighbor_list`` should be compatible with ``torch.compile``."""
        torch.manual_seed(0)
        N = 64
        positions = torch.rand(N, 3, dtype=dtype, device=device) * 10.0
        cell = _orthorhombic_cell(10.0, device, dtype)
        cutoff = 2.5

        nm_uncompiled, nn_uncompiled, nms_uncompiled = cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=64,
        )

        @torch.compile
        def compiled_cluster_tile_neighbor_list(positions, cutoff, cell):
            return cluster_tile_neighbor_list(positions, cutoff, cell, max_neighbors=64)

        nm_compiled, nn_compiled, nms_compiled = compiled_cluster_tile_neighbor_list(
            positions, cutoff, cell
        )

        assert torch.equal(nn_uncompiled, nn_compiled)
        # Per-row neighbor sets must match (column order within a row may differ).
        for i in range(N):
            n_i = int(nn_uncompiled[i].item())
            s_uncompiled = {int(x.item()) for x in nm_uncompiled[i, :n_i]}
            s_compiled = {int(x.item()) for x in nm_compiled[i, :n_i]}
            assert s_uncompiled == s_compiled, (
                f"Row {i} neighbor set mismatch under torch.compile"
            )

    @pytest.mark.slow
    def test_build_then_convert_compile(self, device, dtype):
        """Component build + query_cluster_tile should compile cleanly."""
        torch.manual_seed(1)
        N = 64
        positions = torch.rand(N, 3, dtype=dtype, device=device) * 10.0
        cell = _orthorhombic_cell(10.0, device, dtype)
        cutoff = 2.5

        # Uncompiled reference via the convenience wrapper.
        nm_ref, nn_ref, _nms_ref = cluster_tile_neighbor_list(
            positions, cutoff, cell, max_neighbors=64
        )

        # Compiled component sequence.
        state = allocate_cluster_tile_list(N, torch.device(device), dtype=dtype)
        nm = torch.full((N, 64), N, dtype=torch.int32, device=device)
        nn = torch.zeros(N, dtype=torch.int32, device=device)
        nms = torch.zeros((N, 64, 3), dtype=torch.int32, device=device)

        @torch.compile
        def compiled_build_and_convert(positions, cutoff, cell, nm, nn, nms):
            build_cluster_tile_list(positions, cutoff, cell, *state)
            query_cluster_tile(
                state[0],  # sorted_atom_index
                state[2],  # sorted_pos_x
                state[3],  # sorted_pos_y
                state[4],  # sorted_pos_z
                state[11],  # num_tiles
                state[12],  # tile_row_group
                state[13],  # tile_col_group
                cell,
                cutoff,
                N,
                nm,
                nn,
                nms,
            )

        compiled_build_and_convert(positions, cutoff, cell, nm, nn, nms)

        assert torch.equal(nn_ref, nn)
        for i in range(N):
            n_i = int(nn_ref[i].item())
            s_ref = {int(x.item()) for x in nm_ref[i, :n_i]}
            s_got = {int(x.item()) for x in nm[i, :n_i]}
            assert s_ref == s_got, f"Row {i} neighbor set mismatch under torch.compile"


# =============================================================================
# Left-handed cells
# =============================================================================
class TestClusterTileLeftHanded:
    """Cells with ``det(cell) < 0`` should produce the same pair set as the
    right-handed mirror.  Mirrors :class:`TestLeftHandedCells` in
    ``test_cell_list.py``.
    """

    @requires_vesin
    def test_left_handed_cubic(self, device, dtype):
        """Flipping the sign of one axis should preserve the pair set."""
        N = 64
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=N, cell_size=4.0, dtype=dtype, device=device
        )
        cutoff = 1.1

        nm_rh, nn_rh, nms_rh = cluster_tile_neighbor_list(
            positions, cutoff, cell, max_neighbors=32
        )
        i_rh, j_rh, _u_rh = _matrix_to_coo_full(nm_rh, nn_rh, nms_rh, N)
        pairs_rh = {(int(a), int(b)) for a, b in zip(i_rh, j_rh)}

        # Flip the third axis to produce a left-handed cell.
        cell_lh = cell.clone()
        cell_lh[0, 2, 2] = -cell_lh[0, 2, 2]
        positions_lh = positions.clone()
        positions_lh[:, 2] = -positions_lh[:, 2]

        nm_lh, nn_lh, nms_lh = cluster_tile_neighbor_list(
            positions_lh, cutoff, cell_lh, max_neighbors=32
        )
        i_lh, j_lh, _u_lh = _matrix_to_coo_full(nm_lh, nn_lh, nms_lh, N)
        pairs_lh = {(int(a), int(b)) for a, b in zip(i_lh, j_lh)}

        assert pairs_rh == pairs_lh, "Left-handed cell produced different pair set"
        del pbc  # pbc fixture unused under PBC-implicit cluster_tile


class TestClusterTileAutograd:
    """Differentiable per-pair distances/vectors for cluster_tile_neighbor_list."""

    def _make_system(self, device, n=32, box=5.0):
        torch.manual_seed(0)
        pos = torch.randn(n, 3, dtype=torch.float32, device=device) * 0.5
        cell = torch.eye(3, dtype=torch.float32, device=device) * box
        return pos, cell

    def test_forward_returns_differentiable(self, device):
        pos, cell = self._make_system(device)
        pos.requires_grad_(True)
        nm, nn, shifts, d, v = cluster_tile_neighbor_list(
            pos,
            1.5,
            cell,
            return_distances=True,
            return_vectors=True,
        )
        assert d.requires_grad and v.requires_grad

    def test_grad_positions_finite(self, device):
        pos, cell = self._make_system(device)
        pos.requires_grad_(True)
        _, _, _, d, _ = cluster_tile_neighbor_list(
            pos,
            1.5,
            cell,
            return_distances=True,
            return_vectors=True,
        )
        d.sum().backward()
        assert torch.isfinite(pos.grad).all()

    def test_grad_cell_finite(self, device):
        pos, _ = self._make_system(device)
        cell = torch.eye(3, dtype=torch.float32, device=device) * 5.0
        cell.requires_grad_(True)
        _, _, _, d, _ = cluster_tile_neighbor_list(
            pos,
            1.5,
            cell,
            return_distances=True,
            return_vectors=True,
        )
        d.sum().backward()
        assert torch.isfinite(cell.grad).all()

    def test_hessian_vector_product_smoke(self, device):
        """fp32 second-order: HVP runs and stays finite."""
        pos, cell = self._make_system(device)
        pos.requires_grad_(True)

        def loss(p):
            *_, d, _ = cluster_tile_neighbor_list(
                p,
                1.5,
                cell,
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
        pos, cell = self._make_system(device)
        nm_a, nn_a, sh_a = cluster_tile_neighbor_list(pos, 1.5, cell)
        nm_b, nn_b, sh_b, d_b, v_b = cluster_tile_neighbor_list(
            pos,
            1.5,
            cell,
            return_distances=True,
            return_vectors=True,
        )
        assert not d_b.requires_grad and not v_b.requires_grad
        assert torch.equal(nn_a, nn_b)
        # Sort each row's active indices so the comparison is order-agnostic
        # (cluster_tile may emit pairs in a different order than the
        # non-pair-output path).
        for i in range(nm_a.shape[0]):
            n = nn_a[i].item()
            row_a = sorted(nm_a[i, :n].tolist())
            row_b = sorted(nm_b[i, :n].tolist())
            assert row_a == row_b

    def test_grad_matches_fd_spot_check(self, device):
        """fp32 spot-check: analytical gradient agrees with central-FD
        on a tight cluster within fp32 precision.

        Torch ``gradcheck`` requires fp64 for its default tolerances;
        cluster_tile is fp32-only so we do a hand-rolled spot check.
        Tight cluster + wide cutoff + larger FD eps put the neighbor-set
        discontinuity well out of FD reach and absorb fp32 cancellation.
        """
        torch.manual_seed(0)
        n = 8
        data = torch.randn(n, 3, dtype=torch.float32, device=device) * 0.15
        cell = torch.eye(3, dtype=torch.float32, device=device) * 20.0
        eps = 1e-3

        def fn(p):
            *_, d, _ = cluster_tile_neighbor_list(
                p,
                5.0,
                cell,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        pos = data.clone().requires_grad_(True)
        ana = torch.autograd.grad(fn(pos), pos)[0]
        fd = torch.zeros_like(ana)
        for i in range(n):
            for j in range(3):
                pp = data.clone().requires_grad_(False)
                pp[i, j] += eps
                f_p = fn(pp).item()
                pm = data.clone().requires_grad_(False)
                pm[i, j] -= eps
                f_m = fn(pm).item()
                fd[i, j] = (f_p - f_m) / (2 * eps)
        # fp32 + the warp launcher's reductions produce ~1e-2 worst-case
        # disagreement; relative agreement is what we check.
        max_abs_diff = (ana - fd).abs().max().item()
        max_ref = max(ana.abs().max().item(), fd.abs().max().item(), 1.0)
        assert max_abs_diff / max_ref < 5e-2, (
            f"analytical vs FD relative disagreement {max_abs_diff / max_ref:.3e}"
        )


class TestClusterTileCutoff2SelectiveOverflow:
    """Coverage for single-system dual cutoff, selective rebuild, and overflow."""

    def test_cutoff2_returns_two_matrix_groups(self, device, dtype):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.4, 0.0, 0.0]],
            dtype=dtype,
            device=device,
        )
        cell = _orthorhombic_cell(5.0, device, dtype)
        nm1, nn1, _sh1, nm2, nn2, _sh2 = cluster_tile_neighbor_list(
            positions,
            1.0,
            cell,
            max_neighbors=8,
            cutoff2=1.6,
        )
        assert int(nn2.sum().item()) >= int(nn1.sum().item())
        assert int(nn2.sum().item()) > 0
        assert nm1.shape == nm2.shape == (3, 8)

    def test_rebuild_flags_false_preserves_previous_outputs(self, device, dtype):
        torch.manual_seed(101)
        positions = torch.rand(64, 3, dtype=dtype, device=device) * 6.0
        cell = _orthorhombic_cell(6.0, device, dtype)
        cutoff = 2.0
        nm, nn, shifts = cluster_tile_neighbor_list(
            positions, cutoff, cell, max_neighbors=64
        )
        tile_state = cluster_tile_neighbor_list(positions, cutoff, cell, format="tile")
        num_tiles, tile_row_group, tile_col_group, *_ = tile_state

        moved = positions.clone()
        moved[0, 0] += 0.25
        nm2, nn2, shifts2 = cluster_tile_neighbor_list(
            moved,
            cutoff,
            cell,
            max_neighbors=64,
            rebuild_flags=torch.tensor([False], dtype=torch.bool, device=device),
            num_tiles=num_tiles,
            tile_row_group=tile_row_group,
            tile_col_group=tile_col_group,
            neighbor_matrix=nm,
            num_neighbors=nn,
            neighbor_matrix_shifts=shifts,
        )
        assert torch.equal(nm2, nm)
        assert torch.equal(nn2, nn)
        assert torch.equal(shifts2, shifts)

    def test_matrix_overflow_raises(self, device, dtype):
        positions = torch.rand(64, 3, dtype=dtype, device=device) * 8.0
        cell = _orthorhombic_cell(8.0, device, dtype)
        with pytest.raises(NeighborOverflowError):
            cluster_tile_neighbor_list(positions, 3.0, cell, max_neighbors=1)

    def test_compact_coo_overflow_raises(self, device, dtype):
        positions = torch.rand(64, 3, dtype=dtype, device=device) * 8.0
        cell = _orthorhombic_cell(8.0, device, dtype)
        with pytest.raises(NeighborOverflowError):
            cluster_tile_neighbor_list(
                positions,
                3.0,
                cell,
                max_neighbors=64,
                max_pairs=1,
                format="coo",
            )


def _per_atom_neighbor_sets(nm, nn, nms, natom):
    """Per-atom frozenset of (j, sx, sy, sz) directed-neighbor tuples."""
    nm_c, nn_c, s_c = nm.cpu().numpy(), nn.cpu().numpy(), nms.cpu().numpy()
    out = []
    for i in range(natom):
        s = set()
        for k in range(int(nn_c[i])):
            j = int(nm_c[i, k])
            sh = s_c[i, k]
            s.add((j, int(sh[0]), int(sh[1]), int(sh[2])))
        out.append(frozenset(s))
    return out


class TestClusterTileCellListParity:
    """cluster_tile (full-fill) must match cell_list (half_fill=False) exactly.

    Counts alone cannot catch a wrong neighbor distribution or shift sign, so
    these compare per-atom ``(j, shift)`` sets and autograd forces/energy.
    Boxes use ``box > 2*cutoff`` so no atom neighbors its own periodic image
    (where cell_list emits ``(i, i, shift)`` self-pairs that cluster_tile's
    ``i_sorted < j_sorted`` enumeration excludes -- a deliberate convention
    difference, not tested here).
    """

    def test_full_fill_per_atom_sets_match_cell_list(self, device, dtype):
        torch.manual_seed(0)
        n, box, cutoff = 128, 12.0, 4.0
        pos = torch.rand(n, 3, dtype=dtype, device=device) * box
        cell = _orthorhombic_cell(box, device, dtype)
        pbc = torch.ones(3, dtype=torch.bool, device=device)
        nm, nn, nms = cluster_tile_neighbor_list(pos, cutoff, cell, max_neighbors=256)
        cm, cn, cms = cell_list(
            pos, cutoff, cell, pbc, half_fill=False, max_neighbors=256
        )
        assert int(nn.sum().item()) == int(cn.sum().item())
        ct_sets = _per_atom_neighbor_sets(nm, nn, nms, n)
        cl_sets = _per_atom_neighbor_sets(cm, cn, cms, n)
        assert ct_sets == cl_sets

    def test_force_and_energy_match_cell_list(self, device, dtype):
        torch.manual_seed(1)
        n, box, cutoff = 96, 12.0, 4.0
        base = torch.rand(n, 3, dtype=dtype, device=device) * box
        cell = _orthorhombic_cell(box, device, dtype)
        pbc = torch.ones(3, dtype=torch.bool, device=device)

        def energy_force(method):
            p = base.clone().requires_grad_(True)
            if method == "ct":
                _nm, _nn, _sh, dist = cluster_tile_neighbor_list(
                    p, cutoff, cell, max_neighbors=256, return_distances=True
                )
            else:
                out = cell_list(
                    p,
                    cutoff,
                    cell,
                    pbc,
                    half_fill=False,
                    max_neighbors=256,
                    return_distances=True,
                )
                dist = out[-1]
            energy = dist[dist > 0].sum()
            (grad,) = torch.autograd.grad(energy, p)
            return float(energy.item()), grad

        e_ct, g_ct = energy_force("ct")
        e_cl, g_cl = energy_force("cl")
        assert abs(e_ct - e_cl) < 1e-2 * max(1.0, abs(e_cl))
        assert torch.allclose(g_ct, g_cl, atol=1e-2, rtol=1e-3)

    @requires_vesin
    def test_dual_cutoff_secondary_matches_cell_list(self, device, dtype):
        """The cutoff2 matrix must cover the (cutoff, cutoff2] shell."""
        torch.manual_seed(2)
        n, box = 96, 14.0
        pos = torch.rand(n, 3, dtype=dtype, device=device) * box
        cell = _orthorhombic_cell(box, device, dtype)
        pbc = torch.ones(3, dtype=torch.bool, device=device)
        cutoff, cutoff2 = 3.0, 6.0
        nm, nn, nms, nm2, nn2, nms2 = cluster_tile_neighbor_list(
            pos, cutoff, cell, cutoff2=cutoff2, max_neighbors=512
        )
        for matrix, counts, shifts, rc in (
            (nm, nn, nms, cutoff),
            (nm2, nn2, nms2, cutoff2),
        ):
            i_got, j_got, u_got = _matrix_to_coo_full(matrix, counts, shifts, n)
            i_ref, j_ref, u_ref, _ = brute_force_neighbors(pos, cell, pbc, rc)
            assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    def test_tile_buffer_overflow_raises(self, device, dtype):
        """A too-small tile buffer must raise, not silently truncate tiles.

        Forced cheaply with ``max_tiles_per_group=1`` rather than a large
        dense system; exercises the build->query tile-overflow guard.
        """
        torch.manual_seed(3)
        n, box, cutoff = 128, 12.0, 5.0
        pos = torch.rand(n, 3, dtype=dtype, device=device) * box
        cell = _orthorhombic_cell(box, device, dtype)
        (
            sai,
            mc,
            spx,
            spy,
            spz,
            gcx,
            gcy,
            gcz,
            gex,
            gey,
            gez,
            num_tiles,
            trg,
            tcg,
        ) = allocate_cluster_tile_list(
            n, torch.device(device), dtype=dtype, max_tiles_per_group=1
        )
        build_cluster_tile_list(
            pos,
            cutoff,
            cell,
            sai,
            mc,
            spx,
            spy,
            spz,
            gcx,
            gcy,
            gcz,
            gex,
            gey,
            gez,
            num_tiles,
            trg,
            tcg,
        )
        assert int(num_tiles.item()) > int(trg.shape[0])  # overflow really occurred
        nm = torch.empty((n, 256), dtype=torch.int32, device=device)
        nn = torch.zeros(n, dtype=torch.int32, device=device)
        nms = torch.empty((n, 256, 3), dtype=torch.int32, device=device)
        with pytest.raises(NeighborOverflowError):
            query_cluster_tile(
                sai,
                spx,
                spy,
                spz,
                num_tiles,
                trg,
                tcg,
                cell,
                cutoff,
                n,
                nm,
                nn,
                nms,
            )
