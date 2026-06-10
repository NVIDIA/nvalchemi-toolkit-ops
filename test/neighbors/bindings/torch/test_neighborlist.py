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

import nvalchemiops.torch.neighbors as neighbor_module
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
    """Test automatic method selection based on estimated neighbor density."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_auto_select_cell_list_sparse_no_cell(self, dtype, device):
        """Cell-less auto dispatch is correct at method-dependent COO arity."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is required for this test parameter")

        # Small system: 100 atoms
        target_density = 0.25
        num_atoms = 100
        volume = num_atoms / target_density
        box_size = volume ** (1 / 3)
        positions = torch.rand(num_atoms, 3, dtype=dtype, device=device) * box_size
        cutoff = 2.0

        # Call wrapper with no method specified
        result = neighbor_list(
            positions, cutoff, max_neighbors=64, return_neighbor_list=True
        )

        # Cell-less COO arity is method-dependent (naive -> 2-tuple, cell_list ->
        # 3-tuple with zeroed shifts). The COO list/ptr always live at [0]/[1].
        assert len(result) in (2, 3)
        neighbor_list_result, neighbor_ptr = result[0], result[1]
        assert neighbor_list_result.shape[0] == 2  # COO format
        assert neighbor_ptr.shape[0] == 101
        assert neighbor_ptr[0] == 0
        if len(result) == 3:
            assert result[2].shape[1] == 3  # shifts present only for cell_list

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_auto_select_cell_list_sparse_with_pbc(self, dtype, device):
        """Auto-select cell_list for sparse systems with PBC."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is required for this test parameter")

        positions, cell, pbc = create_random_system(
            100, 10.0, dtype=dtype, device=device
        )
        cutoff = 2.0

        # Call wrapper with no method specified but with cell and pbc
        result = neighbor_list(
            positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=True
        )

        # Should include shifts because a periodic cell was provided.
        assert len(result) == 3  # With PBC, includes neighbor_ptr and shifts
        neighbor_list_result, neighbor_ptr, shifts = result
        assert neighbor_list_result.shape[0] == 2
        assert neighbor_ptr.shape[0] == 101
        assert neighbor_ptr[0] == 0
        assert shifts.shape[1] == 3

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_auto_select_cell_list_large_sparse_system(self, dtype, device):
        """Auto-select cell_list for large sparse systems."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")

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
    def test_auto_select_batch_cell_list_sparse(self, dtype, device):
        """Auto-select batch_cell_list for sparse batched systems."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is required for this test parameter")

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

        # Should auto-select a batched method and include shifts from the input PBC.
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
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
                ),
            ),
        ],
    )
    def test_auto_select_batch_cell_list_large_sparse(self, dtype, device):
        """Auto-select batch_cell_list for large sparse batched systems."""

        # Create sparse batch
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

    def test_auto_dispatch_dense_geometry_uses_naive(self, monkeypatch):
        """Dense geometry selects the naive implementation."""
        seen = {}

        def fake_naive(positions, cutoff, **kwargs):
            del positions, cutoff, kwargs
            seen["method"] = "naive"
            return "naive"

        def fail_cell_list(*args, **kwargs):
            del args, kwargs
            raise AssertionError("cell_list should not be selected")

        monkeypatch.setattr(neighbor_module, "naive_neighbor_list", fake_naive)
        monkeypatch.setattr(neighbor_module, "cell_list", fail_cell_list)

        positions = torch.zeros(1000, 3, dtype=torch.float32)
        cell = torch.eye(3, dtype=torch.float32) * 10.0
        pbc = torch.zeros(3, dtype=torch.bool)

        assert (
            neighbor_module.neighbor_list(positions, 5.0, cell=cell, pbc=pbc) == "naive"
        )
        assert seen["method"] == "naive"

    def test_auto_dispatch_sparse_geometry_uses_cell_list(self, monkeypatch):
        """Sparse geometry selects the cell-list implementation."""
        seen = {}

        def fail_naive(*args, **kwargs):
            del args, kwargs
            raise AssertionError("naive should not be selected")

        def fake_cell_list(positions, cutoff, cell, pbc, **kwargs):
            del positions, cutoff, cell, pbc, kwargs
            seen["method"] = "cell_list"
            return "cell_list"

        monkeypatch.setattr(neighbor_module, "naive_neighbor_list", fail_naive)
        monkeypatch.setattr(neighbor_module, "cell_list", fake_cell_list)

        positions = torch.zeros(1000, 3, dtype=torch.float32)
        cell = torch.eye(3, dtype=torch.float32) * 100.0
        pbc = torch.zeros(3, dtype=torch.bool)

        assert (
            neighbor_module.neighbor_list(positions, 2.0, cell=cell, pbc=pbc)
            == "cell_list"
        )
        assert seen["method"] == "cell_list"

    def test_auto_dispatch_batched_uses_max_expected_neighbors(self, monkeypatch):
        """Batched geometry uses the densest system for method selection."""
        seen = {}

        def fake_batch_naive(positions, cutoff, **kwargs):
            del positions, cutoff, kwargs
            seen["method"] = "batch_naive"
            return "batch_naive"

        def fail_batch_cell_list(*args, **kwargs):
            del args, kwargs
            raise AssertionError("batch_cell_list should not be selected")

        monkeypatch.setattr(
            neighbor_module, "batch_naive_neighbor_list", fake_batch_naive
        )
        monkeypatch.setattr(neighbor_module, "batch_cell_list", fail_batch_cell_list)

        positions = torch.zeros(1100, 3, dtype=torch.float32)
        cell = torch.stack(
            [
                torch.eye(3, dtype=torch.float32) * 10.0,
                torch.eye(3, dtype=torch.float32) * 100.0,
            ]
        )
        pbc = torch.zeros(2, 3, dtype=torch.bool)
        batch_idx = torch.cat(
            [
                torch.zeros(1000, dtype=torch.int32),
                torch.ones(100, dtype=torch.int32),
            ]
        )
        batch_ptr = torch.tensor([0, 1000, 1100], dtype=torch.int32)

        assert (
            neighbor_module.neighbor_list(
                positions,
                5.0,
                cell=cell,
                pbc=pbc,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
            )
            == "batch_naive"
        )
        assert seen["method"] == "batch_naive"

    def test_auto_dispatch_single_system_batch_unbatches_naive(self, monkeypatch):
        """Single-system batched inputs dispatch to unbatched naive."""

        def fake_naive(positions, cutoff, **kwargs):
            del positions, cutoff
            assert kwargs["cell"].shape == (3, 3)
            assert kwargs["pbc"].shape == (3,)
            return "naive"

        def fail_batch_naive(*args, **kwargs):
            del args, kwargs
            raise AssertionError("batch_naive should not be selected")

        monkeypatch.setattr(neighbor_module, "naive_neighbor_list", fake_naive)
        monkeypatch.setattr(
            neighbor_module, "batch_naive_neighbor_list", fail_batch_naive
        )

        positions = torch.zeros(1000, 3, dtype=torch.float32)
        cell = (torch.eye(3, dtype=torch.float32) * 10.0).reshape(1, 3, 3)
        pbc = torch.zeros(1, 3, dtype=torch.bool)
        batch_idx = torch.zeros(1000, dtype=torch.int32)
        batch_ptr = torch.tensor([0, 1000], dtype=torch.int32)

        assert (
            neighbor_module.neighbor_list(
                positions,
                5.0,
                cell=cell,
                pbc=pbc,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
            )
            == "naive"
        )

    def test_auto_dispatch_single_system_batch_unbatches_cell_list(self, monkeypatch):
        """Single-system batched inputs dispatch to unbatched cell_list."""

        def fail_batch_cell_list(*args, **kwargs):
            del args, kwargs
            raise AssertionError("batch_cell_list should not be selected")

        def fake_cell_list(positions, cutoff, cell, pbc, **kwargs):
            del positions, cutoff, kwargs
            assert cell.shape == (3, 3)
            assert pbc.shape == (3,)
            return "cell_list"

        monkeypatch.setattr(neighbor_module, "batch_cell_list", fail_batch_cell_list)
        monkeypatch.setattr(neighbor_module, "cell_list", fake_cell_list)

        positions = torch.zeros(1000, 3, dtype=torch.float32)
        cell = (torch.eye(3, dtype=torch.float32) * 100.0).reshape(1, 3, 3)
        pbc = torch.zeros(1, 3, dtype=torch.bool)
        batch_idx = torch.zeros(1000, dtype=torch.int32)
        batch_ptr = torch.tensor([0, 1000], dtype=torch.int32)

        assert (
            neighbor_module.neighbor_list(
                positions,
                2.0,
                cell=cell,
                pbc=pbc,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
            )
            == "cell_list"
        )

    def test_auto_dispatch_sparse_periodic_float32_uses_cell_list(self, monkeypatch):
        """Sparse periodic inputs below the cluster-tile gate use cell_list."""

        def fail_cluster_tile(*args, **kwargs):
            del args, kwargs
            raise AssertionError("cluster_tile should not be selected")

        def fail_naive(*args, **kwargs):
            del args, kwargs
            raise AssertionError("naive should not be selected")

        def fake_cell_list(positions, cutoff, cell, pbc, **kwargs):
            del positions, cutoff, cell, pbc, kwargs
            return "cell_list"

        monkeypatch.setattr(
            neighbor_module, "cluster_tile_neighbor_list", fail_cluster_tile
        )
        monkeypatch.setattr(
            neighbor_module, "batch_cluster_tile_neighbor_list", fail_cluster_tile
        )
        monkeypatch.setattr(neighbor_module, "naive_neighbor_list", fail_naive)
        monkeypatch.setattr(neighbor_module, "cell_list", fake_cell_list)

        positions = torch.zeros(2048, 3, dtype=torch.float32)
        cell = torch.eye(3, dtype=torch.float32) * 30.0
        pbc = torch.ones(3, dtype=torch.bool)

        assert (
            neighbor_module.neighbor_list(positions, 3.0, cell=cell, pbc=pbc)
            == "cell_list"
        )

    def test_auto_dispatch_routes_cluster_tile_decision(self, monkeypatch):
        """Auto-dispatch can route a feasible selector decision to cluster_tile."""

        def fake_auto_method(*args, **kwargs):
            del args, kwargs
            return "cluster_tile"

        def fail_naive(*args, **kwargs):
            del args, kwargs
            raise AssertionError("naive should not be selected")

        def fail_cell_list(*args, **kwargs):
            del args, kwargs
            raise AssertionError("cell_list should not be selected")

        def fake_cluster_tile(positions, cutoff, cell, **kwargs):
            del positions, cutoff, cell, kwargs
            return "cluster_tile"

        monkeypatch.setattr(
            neighbor_module, "_auto_method_from_geometry", fake_auto_method
        )
        monkeypatch.setattr(neighbor_module, "naive_neighbor_list", fail_naive)
        monkeypatch.setattr(neighbor_module, "cell_list", fail_cell_list)
        monkeypatch.setattr(
            neighbor_module, "cluster_tile_neighbor_list", fake_cluster_tile
        )

        positions = torch.zeros(128, 3, dtype=torch.float32)
        cell = torch.eye(3, dtype=torch.float32) * 30.0
        pbc = torch.ones(3, dtype=torch.bool)

        assert (
            neighbor_module.neighbor_list(positions, 3.0, cell=cell, pbc=pbc)
            == "cluster_tile"
        )

    def test_auto_dispatch_routes_batch_cluster_tile_decision(self, monkeypatch):
        """Auto-dispatch prefixes a feasible batched cluster-tile decision."""

        def fake_auto_method(*args, **kwargs):
            del args, kwargs
            return "cluster_tile"

        def fake_batch_cluster_tile(positions, cutoff, cell, batch_ptr, **kwargs):
            del positions, cutoff, cell, batch_ptr, kwargs
            return "batch_cluster_tile"

        monkeypatch.setattr(
            neighbor_module, "_auto_method_from_geometry", fake_auto_method
        )
        monkeypatch.setattr(
            neighbor_module, "batch_cluster_tile_neighbor_list", fake_batch_cluster_tile
        )

        positions = torch.zeros(8, 3, dtype=torch.float32)
        cell = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)
        cell = cell.expand(2, -1, -1).contiguous() * 30.0
        pbc = torch.ones((2, 3), dtype=torch.bool)
        batch_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
        batch_ptr = torch.tensor([0, 4, 8], dtype=torch.int32)

        assert (
            neighbor_module.neighbor_list(
                positions,
                3.0,
                cell=cell,
                pbc=pbc,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
            )
            == "batch_cluster_tile"
        )

    def test_explicit_batch_cluster_tile_broadcasts_shared_cell(self, monkeypatch):
        """Explicit batch_cluster_tile accepts a shared (3, 3) cell."""
        captured = {}

        def fake_batch_cluster_tile(positions, cutoff, cell, batch_ptr, **kwargs):
            del positions, cutoff, kwargs
            captured["cell"] = cell
            captured["batch_ptr"] = batch_ptr
            return "batch_cluster_tile"

        monkeypatch.setattr(
            neighbor_module, "batch_cluster_tile_neighbor_list", fake_batch_cluster_tile
        )

        positions = torch.zeros(64, 3, dtype=torch.float32)
        batch_ptr = torch.tensor([0, 32, 64], dtype=torch.int32)
        cell = torch.eye(3, dtype=torch.float32) * 10.0
        pbc = torch.ones(2, 3, dtype=torch.bool)

        assert (
            neighbor_module.neighbor_list(
                positions,
                3.0,
                cell=cell,
                pbc=pbc,
                batch_ptr=batch_ptr,
                method="batch_cluster_tile",
            )
            == "batch_cluster_tile"
        )
        assert captured["cell"].shape == (2, 3, 3)
        assert captured["cell"].is_contiguous()
        torch.testing.assert_close(captured["cell"][0], cell)
        torch.testing.assert_close(captured["cell"][1], cell)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_auto_select_batch_naive_dual_cutoff(self, dtype, device):
        """Auto-select batch_naive_dual_cutoff when both cutoff2 and batch_idx are provided."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is required for this test parameter")

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

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_auto_select_batch_ptr_only_no_cell(self, dtype, device):
        """method=None + batch_ptr-only (no batch_idx, no cell) hits the
        ``elif batch_ptr is not None`` branch in __init__.py dispatch."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is required for this test parameter")

        positions = torch.randn(80, 3, dtype=dtype, device=device) * 5.0
        batch_ptr = torch.tensor([0, 50, 80], dtype=torch.int32, device=device)
        result = neighbor_list(
            positions,
            cutoff=2.0,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
        )
        # Cell-less batch COO arity is method-dependent (batch_naive -> 2-tuple,
        # batch_cell_list -> 3-tuple). The COO list/ptr always live at [0]/[1].
        assert len(result) in (2, 3)
        nlist, neighbor_ptr = result[0], result[1]
        assert nlist.shape[0] == 2
        assert neighbor_ptr.shape[0] == 81

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_auto_select_batch_idx_only_no_cell(self, dtype, device):
        """method=None + batch_idx-only (no batch_ptr, no cell) hits the
        ``elif batch_idx is not None`` branch in __init__.py dispatch."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is required for this test parameter")

        positions = torch.randn(80, 3, dtype=dtype, device=device) * 5.0
        batch_idx = torch.cat(
            [
                torch.zeros(50, dtype=torch.int32, device=device),
                torch.ones(30, dtype=torch.int32, device=device),
            ]
        )
        result = neighbor_list(
            positions,
            cutoff=2.0,
            batch_idx=batch_idx,
            return_neighbor_list=True,
        )
        # Cell-less batch COO arity is method-dependent (batch_naive -> 2-tuple,
        # batch_cell_list -> 3-tuple). The COO list/ptr always live at [0]/[1].
        assert len(result) in (2, 3)
        nlist, neighbor_ptr = result[0], result[1]
        assert nlist.shape[0] == 2
        assert neighbor_ptr.shape[0] == 81

    def test_cell_without_pbc_raises(self):
        """`cell` provided without `pbc` must raise (line 271-276)."""
        positions = torch.randn(10, 3, dtype=torch.float32)
        cell = torch.eye(3, dtype=torch.float32) * 10.0
        with pytest.raises(ValueError, match="`pbc` is required"):
            neighbor_list(positions, cutoff=2.0, cell=cell)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_explicit_cluster_tile_no_cell_raises(self, dtype):
        """method='cluster_tile' without a cell must raise, not synthesize a box.

        cluster_tile is PBC-implicit; synthesizing a bounding-box cell and forcing
        PBC would emit spurious wrap-around pairs.  No cell/pbc -> pbc=None ->
        ``_reject_unsupported_cluster_tile_combo`` raises before any cell handling.
        """
        positions = torch.randn(256, 3, dtype=dtype) * 5.0
        with pytest.raises(NotImplementedError):
            neighbor_list(
                positions,
                cutoff=2.0,
                method="cluster_tile",
                return_neighbor_list=True,
            )

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_explicit_batch_cluster_tile_no_cell_raises(self, dtype):
        """method='batch_cluster_tile' without a cell must raise (see single-system)."""
        positions = torch.randn(256, 3, dtype=dtype) * 5.0
        batch_ptr = torch.tensor([0, 128, 256], dtype=torch.int32)
        with pytest.raises(NotImplementedError):
            neighbor_list(
                positions,
                cutoff=2.0,
                method="batch_cluster_tile",
                batch_ptr=batch_ptr,
                return_neighbor_list=True,
            )

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_explicit_cluster_tile_accepts_cutoff2(self, dtype):
        """method='cluster_tile' accepts cutoff2 on matrix output."""
        if not torch.cuda.is_available():
            pytest.skip("cluster_tile requires CUDA")
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.4, 0.0, 0.0]],
            dtype=dtype,
            device="cuda",
        )
        cell = torch.eye(3, dtype=dtype, device="cuda") * 5.0
        pbc = torch.ones(3, dtype=torch.bool, device="cuda")
        result = neighbor_list(
            positions,
            cutoff=1.0,
            cell=cell,
            pbc=pbc,
            method="cluster_tile",
            cutoff2=1.6,
            max_neighbors=8,
            return_neighbor_list=False,
        )
        assert len(result) == 6

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_explicit_batch_cluster_tile_accepts_cutoff2(self, dtype):
        """method='batch_cluster_tile' accepts cutoff2 on matrix output."""
        if not torch.cuda.is_available():
            pytest.skip("batch_cluster_tile requires CUDA")
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [1.4, 0.0, 0.0],
            ],
            dtype=dtype,
            device="cuda",
        )
        cell = torch.eye(3, dtype=dtype, device="cuda").repeat(2, 1, 1) * 5.0
        pbc = torch.ones((2, 3), dtype=torch.bool, device="cuda")
        batch_ptr = torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda")
        result = neighbor_list(
            positions,
            cutoff=1.0,
            cell=cell,
            pbc=pbc,
            method="batch_cluster_tile",
            batch_ptr=batch_ptr,
            cutoff2=1.6,
            max_neighbors=8,
            return_neighbor_list=False,
        )
        assert len(result) == 6

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_explicit_cluster_tile_rebuild_flags_raise_clear_state_error(self, dtype):
        """method='cluster_tile' exposes rebuild_flags without TypeError."""
        if not torch.cuda.is_available():
            pytest.skip("cluster_tile requires CUDA")
        positions = torch.randn(64, 3, dtype=dtype, device="cuda") * 5.0
        cell = torch.eye(3, dtype=dtype, device="cuda") * 10.0
        pbc = torch.ones(3, dtype=torch.bool, device="cuda")
        with pytest.raises(ValueError, match="previous cluster_tile state"):
            neighbor_list(
                positions,
                cutoff=2.0,
                cell=cell,
                pbc=pbc,
                method="cluster_tile",
                max_neighbors=32,
                rebuild_flags=torch.tensor([True], dtype=torch.bool, device="cuda"),
                return_neighbor_list=False,
            )


class TestNeighborListExplicitMethod:
    """Test explicit method selection."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_explicit_naive(self, dtype, device):
        """Test explicit naive method selection."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")

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

    @pytest.mark.parametrize(
        ("method", "expected_route", "expected_options"),
        [
            ("naive", "batch_naive", {"native_strategy": "auto"}),
            ("cell_list", "batch_cell_list", {"strategy": "auto"}),
            ("cluster_tile", "batch_cluster_tile", {}),
            ("naive_dual_cutoff", "batch_naive_dual_cutoff", {}),
            ("naive_scalar", "batch_naive", {"native_strategy": "scalar"}),
            ("naive_tile", "batch_naive", {"native_strategy": "tile"}),
            (
                "cell_list_atom_centric",
                "batch_cell_list",
                {"strategy": "atom_centric", "atom_centric_path": "direct"},
            ),
            (
                "cell_list_pair_centric",
                "batch_cell_list",
                {"strategy": "pair_centric", "atom_centric_path": "sorted"},
            ),
        ],
    )
    @pytest.mark.parametrize("batch_arg", ["batch_idx", "batch_ptr", "both"])
    def test_explicit_unbatched_method_promotes_with_batch_metadata(
        self, method, expected_route, expected_options, batch_arg, device, monkeypatch
    ):
        """Explicit single-system methods route to their batch equivalents."""
        positions = torch.zeros(6, 3, dtype=torch.float32, device=device)
        cell = torch.eye(3, dtype=torch.float32, device=device).repeat(2, 1, 1) * 8.0
        pbc = torch.ones(2, 3, dtype=torch.bool, device=device)
        kwargs = {
            "batch_idx": torch.tensor(
                [0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device
            ),
            "batch_ptr": torch.tensor([0, 3, 6], dtype=torch.int32, device=device),
        }
        if batch_arg == "batch_idx":
            kwargs.pop("batch_ptr")
        elif batch_arg == "batch_ptr":
            kwargs.pop("batch_idx")

        seen = {}

        def fake_batch_naive(*args, **call_kwargs):
            seen["route"] = "batch_naive"
            seen["kwargs"] = call_kwargs
            return "batch_naive"

        def fake_batch_cell_list(*args, **call_kwargs):
            seen["route"] = "batch_cell_list"
            seen["args"] = args
            seen["kwargs"] = call_kwargs
            return "batch_cell_list"

        def fake_batch_cluster_tile(*args, **call_kwargs):
            seen["route"] = "batch_cluster_tile"
            seen["args"] = args
            seen["kwargs"] = call_kwargs
            return "batch_cluster_tile"

        def fake_batch_naive_dual_cutoff(*args, **call_kwargs):
            seen["route"] = "batch_naive_dual_cutoff"
            seen["kwargs"] = call_kwargs
            return "batch_naive_dual_cutoff"

        monkeypatch.setattr(
            neighbor_module, "batch_naive_neighbor_list", fake_batch_naive
        )
        monkeypatch.setattr(neighbor_module, "batch_cell_list", fake_batch_cell_list)
        monkeypatch.setattr(
            neighbor_module,
            "batch_cluster_tile_neighbor_list",
            fake_batch_cluster_tile,
        )
        monkeypatch.setattr(
            neighbor_module,
            "batch_naive_neighbor_list_dual_cutoff",
            fake_batch_naive_dual_cutoff,
        )

        result = neighbor_list(
            positions,
            2.0,
            cutoff2=3.0 if method == "naive_dual_cutoff" else None,
            cell=cell,
            pbc=pbc,
            method=method,
            **kwargs,
        )

        assert result == expected_route
        assert seen["route"] == expected_route
        for key, expected in expected_options.items():
            assert seen["kwargs"][key] == expected

    @pytest.mark.parametrize("method", ["naive", "cell_list"])
    def test_explicit_unbatched_method_without_batch_metadata_stays_unbatched(
        self, method, device, monkeypatch
    ):
        """A 3D cell alone is not batch metadata for explicit methods."""
        positions = torch.zeros(6, 3, dtype=torch.float32, device=device)
        cell = torch.eye(3, dtype=torch.float32, device=device).repeat(2, 1, 1) * 8.0
        pbc = torch.zeros(2, 3, dtype=torch.bool, device=device)

        def fake_unbatched(*args, **kwargs):
            return "unbatched"

        def fail_batched(*args, **kwargs):
            raise AssertionError("batch method should not be selected")

        monkeypatch.setattr(neighbor_module, "batch_naive_neighbor_list", fail_batched)
        monkeypatch.setattr(neighbor_module, "batch_cell_list", fail_batched)
        monkeypatch.setattr(neighbor_module, "naive_neighbor_list", fake_unbatched)
        monkeypatch.setattr(neighbor_module, "cell_list", fake_unbatched)

        assert (
            neighbor_list(positions, 2.0, cell=cell, pbc=pbc, method=method)
            == "unbatched"
        )

    @pytest.mark.parametrize(
        ("method", "expected_route"),
        [("naive", "batch_naive"), ("cell_list", "batch_cell_list")],
    )
    @pytest.mark.parametrize("batch_arg", ["batch_idx", "batch_ptr", "both"])
    def test_explicit_unbatched_method_promotes_single_system_batch_metadata(
        self, method, expected_route, batch_arg, device, monkeypatch
    ):
        """Even one-system batch metadata is still explicit batch metadata."""
        positions = torch.zeros(6, 3, dtype=torch.float32, device=device)
        cell = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3) * 8.0
        pbc = torch.zeros(1, 3, dtype=torch.bool, device=device)
        kwargs = {
            "batch_idx": torch.zeros(6, dtype=torch.int32, device=device),
            "batch_ptr": torch.tensor([0, 6], dtype=torch.int32, device=device),
        }
        if batch_arg == "batch_idx":
            kwargs.pop("batch_ptr")
        elif batch_arg == "batch_ptr":
            kwargs.pop("batch_idx")

        def fake_batch_naive(*args, **call_kwargs):
            return "batch_naive"

        def fake_batch_cell_list(*args, **call_kwargs):
            return "batch_cell_list"

        monkeypatch.setattr(
            neighbor_module, "batch_naive_neighbor_list", fake_batch_naive
        )
        monkeypatch.setattr(neighbor_module, "batch_cell_list", fake_batch_cell_list)

        assert (
            neighbor_list(positions, 2.0, cell=cell, pbc=pbc, method=method, **kwargs)
            == expected_route
        )

    def test_invalid_method_with_batch_metadata_stays_invalid(self, device):
        """Unknown method names are not promoted into generated batch names."""
        positions = torch.zeros(6, 3, dtype=torch.float32, device=device)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device)

        with pytest.raises(ValueError, match="Invalid method"):
            neighbor_list(positions, 2.0, method="not_a_method", batch_idx=batch_idx)

    def test_promoted_naive_does_not_cross_batch_boundaries(self, device):
        """Promoted naive honors batch boundaries for overlapping coordinates."""
        molecule = torch.tensor(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        positions = torch.cat([molecule, molecule], dim=0)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device)
        batch_ptr = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)

        pairs, _ptr = neighbor_list(
            positions,
            1.1,
            method="naive",
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
            max_neighbors=4,
        )

        assert pairs.shape[1] == 8
        assert torch.all(batch_idx[pairs[0].long()] == batch_idx[pairs[1].long()])

    def test_method_none_and_batch_method_accept_batch_metadata(self, device):
        """Auto and explicit batch methods accept batch metadata."""
        positions = torch.zeros(6, 3, dtype=torch.float32, device=device)
        cell = torch.eye(3, dtype=torch.float32, device=device).repeat(2, 1, 1)
        pbc = torch.zeros(2, 3, dtype=torch.bool, device=device)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device)
        batch_ptr = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)

        auto_result = neighbor_list(
            positions,
            2.0,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=8,
        )
        batch_result = neighbor_list(
            positions,
            2.0,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_naive",
            max_neighbors=8,
        )

        assert auto_result[0].shape == (6, 8)
        assert batch_result[0].shape == (6, 8)


class TestNeighborListBatchProcessing:
    """Test batch processing with batch_idx."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_batch_naive(self, dtype, device):
        """Test batch naive method."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is required for this test parameter")

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
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
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
        positions = torch.cat([positions1, positions2], dim=0)
        cell = torch.stack([cell1.squeeze(0), cell2.squeeze(0)], dim=0)
        pbc = torch.stack([pbc1.squeeze(0), pbc2.squeeze(0)], dim=0)
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
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")

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
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
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
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
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
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
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
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
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
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
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
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
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
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
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
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
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
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
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


class TestNeighborListFineGrainedMethodEquivalence:
    """Fine-grained ``method=`` names match their base method's pair set.

    The fine-grained strategy names (e.g. ``naive_tile``,
    ``cell_list_pair_centric``) route to the same base kernel as
    ``naive`` / ``cell_list`` with sub-options pinned, so they must produce
    an identical neighbor set on the same geometry.
    """

    def _periodic_float32_system(self, device):
        torch.manual_seed(42)
        positions = torch.rand(256, 3, dtype=torch.float32, device=device) * 20.0
        cell = torch.eye(3, dtype=torch.float32, device=device) * 20.0
        pbc = torch.ones(3, dtype=torch.bool, device=device)
        return positions, cell, pbc

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("method", ["naive_scalar", "naive_tile"])
    def test_naive_suboptions_match_naive(self, device, method):
        """``naive_scalar`` / ``naive_tile`` match the base ``naive`` pair set."""
        positions, cell, pbc = self._periodic_float32_system(device)
        cutoff = 5.0

        base = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="naive",
            return_neighbor_list=True,
        )
        fine = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method=method,
            return_neighbor_list=True,
        )
        torch.testing.assert_close(_sorted_pairs(fine[0]), _sorted_pairs(base[0]))

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
                ),
            ),
        ],
    )
    def test_cell_list_atom_centric_matches_cell_list(self, device):
        """``cell_list_atom_centric`` matches the base ``cell_list`` pair set."""
        positions, cell, pbc = self._periodic_float32_system(device)
        cutoff = 5.0

        base = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="cell_list",
            return_neighbor_list=True,
        )
        fine = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="cell_list_atom_centric",
            return_neighbor_list=True,
        )
        torch.testing.assert_close(_sorted_pairs(fine[0]), _sorted_pairs(base[0]))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="cell_list pair_centric requires CUDA"
    )
    def test_cell_list_pair_centric_matches_cell_list(self):
        """``cell_list_pair_centric`` (CUDA-only) matches the base ``cell_list``."""
        positions, cell, pbc = self._periodic_float32_system("cuda")
        cutoff = 5.0

        base = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="cell_list",
            return_neighbor_list=True,
        )
        fine = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="cell_list_pair_centric",
            return_neighbor_list=True,
        )
        torch.testing.assert_close(_sorted_pairs(fine[0]), _sorted_pairs(base[0]))

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
                ),
            ),
        ],
    )
    def test_batch_naive_tile_matches_batch_naive(self, device):
        """Batched ``batch_naive_tile`` matches the base ``batch_naive`` pair set."""
        torch.manual_seed(42)
        n1, n2 = 128, 96
        positions = torch.rand(n1 + n2, 3, dtype=torch.float32, device=device) * 20.0
        cell = (
            torch.eye(3, dtype=torch.float32, device=device)
            .reshape(1, 3, 3)
            .expand(2, -1, -1)
            .contiguous()
            * 20.0
        )
        pbc = torch.ones((2, 3), dtype=torch.bool, device=device)
        batch_idx = torch.cat(
            [
                torch.zeros(n1, dtype=torch.int32, device=device),
                torch.ones(n2, dtype=torch.int32, device=device),
            ]
        )
        batch_ptr = torch.tensor([0, n1, n1 + n2], dtype=torch.int32, device=device)
        cutoff = 5.0

        base = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_naive",
            return_neighbor_list=True,
        )
        fine = neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_naive_tile",
            return_neighbor_list=True,
        )
        torch.testing.assert_close(_sorted_pairs(fine[0]), _sorted_pairs(base[0]))


class TestNeighborListEmptyNoCell:
    """B2: empty positions with ``cell=None`` returns empty outputs, not a raise."""

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is required for this test parameter",
                ),
            ),
        ],
    )
    def test_empty_positions_no_cell_auto_dispatch(self, device):
        """``method=None`` + (0, 3) positions + ``cell=None`` must not raise."""
        positions = torch.empty(0, 3, dtype=torch.float32, device=device)
        cutoff = 2.0

        result = neighbor_list(positions, cutoff, return_neighbor_list=True)

        # COO list/ptr always at [0]/[1]; empty system -> no pairs, ptr=[0].
        neighbor_list_coo, neighbor_ptr = result[0], result[1]
        assert neighbor_list_coo.shape[1] == 0
        assert neighbor_ptr.shape[0] == 1
        assert int(neighbor_ptr[0]) == 0


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
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")

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
            pytest.skip("CUDA is required for this test parameter")
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


def test_suggest_then_run_under_torch_compile():
    """``suggest`` (host-only) then an explicit-method run survives torch.compile.

    The estimation call is made outside the compiled region; feeding its
    returned strategy name straight back as ``method=`` must produce the same
    matrix-format result compiled as eager.
    """
    from nvalchemiops.torch.neighbors import suggest_neighbor_list_method
    from nvalchemiops.torch.neighbors.neighbor_utils import compute_naive_num_shifts

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    n = 256
    positions = torch.rand(n, 3, dtype=torch.float32, device=device) * 20.0
    cell = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3) * 20.0
    pbc = torch.ones(3, dtype=torch.bool, device=device)
    batch_ptr = torch.tensor([0, n], dtype=torch.int32, device=device)

    method = suggest_neighbor_list_method(batch_ptr, cell, pbc, 5.0)
    shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
        cell, 5.0, pbc.reshape(1, 3)
    )

    def alloc_outputs():
        return (
            torch.full((n, 128), n, dtype=torch.int32, device=device),
            torch.zeros(n, dtype=torch.int32, device=device),
            torch.zeros((n, 128, 3), dtype=torch.int32, device=device),
        )

    def run(pos, neighbor_matrix, num_neighbors, neighbor_matrix_shifts):
        return neighbor_list(
            pos,
            5.0,
            cell=cell,
            pbc=pbc,
            method=method,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            shift_range_per_dimension=shift_range,
            num_shifts_per_system=num_shifts,
            max_shifts_per_system=max_shifts,
        )

    eager = run(positions, *alloc_outputs())
    compiled = torch.compile(run)(positions, *alloc_outputs())

    assert torch.equal(eager[1], compiled[1])  # num_neighbors agree


def test_host_only_suggest_rejects_inside_torch_compile():
    """Strategy estimation must happen before Dynamo traces the runtime call."""
    from nvalchemiops.torch.neighbors import suggest_neighbor_list_method

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_ptr = torch.tensor([0, 4], dtype=torch.int32, device=device)
    cell = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3)
    pbc = torch.ones(3, dtype=torch.bool, device=device)

    def run(ptr, cell_arg, pbc_arg):
        return suggest_neighbor_list_method(ptr, cell_arg, pbc_arg, 1.0)

    with pytest.raises(RuntimeError, match="host-only neighbor-list helper"):
        torch.compile(run)(batch_ptr, cell, pbc)


def test_host_only_auto_method_rejects_inside_torch_compile():
    """Auto dispatch is host-only; compiled callers must pass method= explicitly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    positions = torch.rand(4, 3, dtype=torch.float32, device=device)
    cell = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3)
    pbc = torch.ones(3, dtype=torch.bool, device=device)

    def run(pos):
        return neighbor_list(pos, 1.0, cell=cell, pbc=pbc)

    with pytest.raises(RuntimeError, match="neighbor_list\\(method=None\\)"):
        torch.compile(run)(positions)


def test_host_only_naive_shift_metadata_rejects_inside_torch_compile():
    """Naive PBC shift metadata is prepared once and reused inside compile."""
    from nvalchemiops.torch.neighbors.neighbor_utils import compute_naive_num_shifts

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cell = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3)
    pbc = torch.ones((1, 3), dtype=torch.bool, device=device)

    def run(cell_arg, pbc_arg):
        shift_range, num_shifts, _max_shifts = compute_naive_num_shifts(
            cell_arg, 1.0, pbc_arg
        )
        return shift_range, num_shifts

    with pytest.raises(RuntimeError, match="compute_naive_num_shifts"):
        torch.compile(run)(cell, pbc)
