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

"""Tests for PyTorch bindings of naive neighbor list methods."""

from __future__ import annotations

import pytest
import torch
import warp as wp

from nvalchemiops.torch.neighbors.naive import (
    _naive_neighbor_matrix_no_pbc,
    _naive_neighbor_matrix_pbc,
    naive_neighbor_list,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    NeighborOverflowError,
    compute_naive_num_shifts,
)

from ...test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
    create_random_system,
    create_simple_cubic_system,
)
from .conftest import requires_vesin


def _sorted_row_multisets(
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor | None = None,
) -> list[list[tuple[int, ...]]]:
    """Return active neighbor rows as sorted multisets for parity checks."""
    rows = []
    for row, count_tensor in enumerate(num_neighbors):
        count = int(count_tensor)
        values = []
        for col in range(count):
            item = (int(neighbor_matrix[row, col]),)
            if neighbor_matrix_shifts is not None:
                item += tuple(int(value) for value in neighbor_matrix_shifts[row, col])
            values.append(item)
        rows.append(sorted(values))
    return rows


class TestNaiveCorrectness:
    """Test correctness of naive neighbor list against reference implementation."""

    @requires_vesin
    def test_against_vesin_reference_no_pbc(self, device, dtype):
        """Verify correctness against vesin reference (no PBC)."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1

        # Get our result
        neighbor_list, neighbor_ptr = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=None,
            cell=None,
            max_neighbors=20,
            return_neighbor_list=True,
        )

        idx_i = neighbor_list[0]
        idx_j = neighbor_list[1]
        u = torch.zeros((idx_i.shape[0], 3), dtype=torch.int32, device=device)

        # Get reference result
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(
            positions, cell=None, pbc=None, cutoff=cutoff
        )

        # Compare
        assert_neighbor_lists_equal((idx_i, idx_j, u), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_against_vesin_reference_with_pbc(self, device, dtype):
        """Verify correctness against vesin reference (with PBC)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1

        # Get our result with PBC
        neighbor_list, neighbor_ptr, neighbor_shifts = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=20,
            return_neighbor_list=True,
        )

        idx_i = neighbor_list[0]
        idx_j = neighbor_list[1]
        u = neighbor_shifts

        # Get reference result
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(
            positions, cell=cell, pbc=pbc, cutoff=cutoff
        )

        # Compare
        assert_neighbor_lists_equal((idx_i, idx_j, u), (i_ref, j_ref, u_ref))

    def test_random_systems_basic_correctness(self, device, dtype):
        """Test basic correctness properties on random systems."""
        for pbc_flag in [True, False]:
            for seed in [42, 123, 456]:
                positions, cell, pbc = create_random_system(
                    num_atoms=20,
                    cell_size=3.0,
                    dtype=dtype,
                    device=device,
                    seed=seed,
                    pbc_flag=pbc_flag,
                )
                cutoff = 1.2
                max_neighbors = 50

                # Get neighbor matrix format
                if pbc_flag:
                    neighbor_matrix, num_neighbors, unit_shifts = naive_neighbor_list(
                        positions=positions,
                        cutoff=cutoff,
                        pbc=pbc,
                        cell=cell,
                        max_neighbors=max_neighbors,
                    )
                else:
                    neighbor_matrix, num_neighbors = naive_neighbor_list(
                        positions=positions,
                        cutoff=cutoff,
                        pbc=None,
                        cell=None,
                        max_neighbors=max_neighbors,
                    )

                # Verify basic correctness properties
                assert torch.all(num_neighbors >= 0)
                assert torch.all(num_neighbors <= max_neighbors)
                assert neighbor_matrix.device == torch.device(device)
                assert num_neighbors.device == torch.device(device)

    def test_precision_consistency(self, device):
        """Test that float32 and float64 give consistent neighbor counts."""
        positions_f32, cell_f32, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=torch.float32, device=device
        )
        positions_f64 = positions_f32.double()
        cell_f64 = cell_f32.double()

        cutoff = 1.1
        max_neighbors = 50

        # Get results for both precisions
        _, num_neighbors_f32, _ = naive_neighbor_list(
            positions_f32,
            cutoff,
            pbc=pbc,
            cell=cell_f32,
            max_neighbors=max_neighbors,
        )
        _, num_neighbors_f64, _ = naive_neighbor_list(
            positions_f64,
            cutoff,
            pbc=pbc,
            cell=cell_f64,
            max_neighbors=max_neighbors,
        )

        # Neighbor counts should be identical
        torch.testing.assert_close(num_neighbors_f32, num_neighbors_f64, rtol=0, atol=0)

    def test_target_indices_matrix_compact_rows(self, device):
        """target_indices returns compact rows matching selected full rows."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        target_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)

        full_nm, full_nn = naive_neighbor_list(positions, 0.75, max_neighbors=4)
        partial_nm, partial_nn = naive_neighbor_list(
            positions,
            0.75,
            max_neighbors=4,
            target_indices=target_indices,
            neighbor_matrix_shifts=torch.zeros(
                (target_indices.shape[0], 4, 3),
                dtype=torch.int32,
                device=device,
            ),
        )

        assert partial_nm.shape == (2, 4)
        torch.testing.assert_close(partial_nn, full_nn[target_indices.long()])
        for row, atom in enumerate(target_indices.cpu().tolist()):
            count = int(partial_nn[row])
            torch.testing.assert_close(
                torch.sort(partial_nm[row, :count].cpu()).values,
                torch.sort(full_nm[atom, : int(full_nn[atom])].cpu()).values,
            )

    def test_target_indices_coo_uses_compact_source_rows(self, device):
        """COO source rows are compact target rows, not original atom ids."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        target_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)

        neighbor_list, neighbor_ptr = naive_neighbor_list(
            positions,
            0.75,
            max_neighbors=4,
            target_indices=target_indices,
            return_neighbor_list=True,
        )

        assert neighbor_ptr.shape == (3,)
        assert set(neighbor_list[0].cpu().tolist()) == {0, 1}
        assert set(map(tuple, neighbor_list.T.cpu().tolist())) == {(0, 3), (1, 1)}

    @pytest.mark.gpu
    def test_target_indices_compile_fullgraph_with_compact_buffers(self, device):
        """target_indices without pair_fn stays behind a fullgraph custom op."""
        if not str(device).startswith("cuda"):
            pytest.skip("CUDA is required for torch.compile fullgraph Warp check")
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        target_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
        neighbor_matrix = torch.full((2, 4), 4, dtype=torch.int32, device=device)
        num_neighbors = torch.zeros((2,), dtype=torch.int32, device=device)

        @torch.compile(fullgraph=True)
        def _run(pos, nm, nn):
            return naive_neighbor_list(
                pos,
                0.75,
                neighbor_matrix=nm,
                num_neighbors=nn,
                target_indices=target_indices,
            )

        partial_nm, partial_nn = _run(
            positions, neighbor_matrix.clone(), num_neighbors.clone()
        )
        full_nm, full_nn = naive_neighbor_list(positions, 0.75, max_neighbors=4)
        assert partial_nm.shape == (2, 4)
        torch.testing.assert_close(partial_nn, full_nn[target_indices.long()])
        for row, atom in enumerate(target_indices.cpu().tolist()):
            count = int(partial_nn[row])
            torch.testing.assert_close(
                torch.sort(partial_nm[row, :count].cpu()).values,
                torch.sort(full_nm[atom, : int(full_nn[atom])].cpu()).values,
            )

    @pytest.mark.gpu
    def test_target_indices_compile_fullgraph_pbc_pair_geometry(self, device):
        """PBC target_indices fullgraph path supports geometry buffers."""
        if not str(device).startswith("cuda"):
            pytest.skip("CUDA is required for torch.compile fullgraph Warp check")
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [9.5, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        cell = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0) * 10.0
        pbc = torch.tensor([[True, True, True]], device=device)
        target_indices = torch.tensor([0], dtype=torch.int32, device=device)
        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(cell, 1.0, pbc)
        neighbor_matrix = torch.full((1, 8), 3, dtype=torch.int32, device=device)
        num_neighbors = torch.zeros((1,), dtype=torch.int32, device=device)
        shifts = torch.zeros((1, 8, 3), dtype=torch.int32, device=device)
        distances = torch.zeros((1, 8), dtype=torch.float32, device=device)
        vectors = torch.zeros((1, 8, 3), dtype=torch.float32, device=device)

        @torch.compile(fullgraph=True)
        def _run(pos, nm, nn, nms, dist, vec):
            return naive_neighbor_list(
                pos,
                1.0,
                cell=cell,
                pbc=pbc,
                neighbor_matrix=nm,
                num_neighbors=nn,
                neighbor_matrix_shifts=nms,
                shift_range_per_dimension=shift_range,
                num_shifts_per_system=num_shifts,
                max_shifts_per_system=max_shifts,
                target_indices=target_indices,
                return_distances=True,
                return_vectors=True,
                neighbor_distances=dist,
                neighbor_vectors=vec,
            )

        partial_nm, partial_nn, partial_shifts, partial_dist, partial_vec = _run(
            positions,
            neighbor_matrix.clone(),
            num_neighbors.clone(),
            shifts.clone(),
            distances.clone(),
            vectors.clone(),
        )
        assert partial_nm.shape == (1, 8)
        assert partial_shifts.shape == (1, 8, 3)
        assert partial_dist.shape == (1, 8)
        assert partial_vec.shape == (1, 8, 3)
        assert int(partial_nn[0]) >= 1

    @pytest.mark.gpu
    def test_target_indices_tile_compile_fullgraph_runtime_targets(self, device):
        """Tiled topology custom op accepts runtime compact targets."""
        if not str(device).startswith("cuda"):
            pytest.skip("CUDA is required for tiled fullgraph coverage.")
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        @torch.compile(fullgraph=True)
        def run(pos, nm, nn, targets):
            return naive_neighbor_list(
                pos,
                0.75,
                neighbor_matrix=nm,
                num_neighbors=nn,
                target_indices=targets,
                strategy="tile",
            )

        for targets in (
            torch.tensor([2, 0], dtype=torch.int32, device=device),
            torch.tensor([0, 2], dtype=torch.int32, device=device),
        ):
            tiled_nm, tiled_nn = run(
                positions,
                torch.full((2, 4), 4, dtype=torch.int32, device=device),
                torch.zeros((2,), dtype=torch.int32, device=device),
                targets,
            )
            scalar_nm, scalar_nn = naive_neighbor_list(
                positions,
                0.75,
                max_neighbors=4,
                target_indices=targets,
                strategy="scalar",
            )
            torch.testing.assert_close(tiled_nn, scalar_nn, rtol=0, atol=0)
            assert _sorted_row_multisets(tiled_nm, tiled_nn) == _sorted_row_multisets(
                scalar_nm,
                scalar_nn,
            )

    @pytest.mark.gpu
    def test_target_indices_tile_compile_fullgraph_pbc_runtime_targets(self, device):
        """Tiled PBC topology custom op accepts runtime compact targets."""
        if not str(device).startswith("cuda"):
            pytest.skip("CUDA is required for tiled fullgraph coverage.")
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        cell = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0) * 4.0
        pbc = torch.ones((1, 3), dtype=torch.bool, device=device)
        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(cell, 1.1, pbc)

        @torch.compile(fullgraph=True)
        def run(pos, nm, nms, nn, targets):
            return naive_neighbor_list(
                pos,
                1.1,
                cell=cell,
                pbc=pbc,
                neighbor_matrix=nm,
                neighbor_matrix_shifts=nms,
                num_neighbors=nn,
                shift_range_per_dimension=shift_range,
                num_shifts_per_system=num_shifts,
                max_shifts_per_system=max_shifts,
                target_indices=targets,
                strategy="tile",
            )

        for targets in (
            torch.tensor([2, 0], dtype=torch.int32, device=device),
            torch.tensor([0, 2], dtype=torch.int32, device=device),
        ):
            tiled_nm, tiled_nn, tiled_shifts = run(
                positions,
                torch.full((2, 32), 4, dtype=torch.int32, device=device),
                torch.zeros((2, 32, 3), dtype=torch.int32, device=device),
                torch.zeros((2,), dtype=torch.int32, device=device),
                targets,
            )
            scalar_nm, scalar_nn, scalar_shifts = naive_neighbor_list(
                positions,
                1.1,
                cell=cell,
                pbc=pbc,
                max_neighbors=32,
                neighbor_matrix_shifts=torch.zeros(
                    (2, 32, 3),
                    dtype=torch.int32,
                    device=device,
                ),
                target_indices=targets,
                strategy="scalar",
            )
            torch.testing.assert_close(tiled_nn, scalar_nn, rtol=0, atol=0)
            assert _sorted_row_multisets(
                tiled_nm,
                tiled_nn,
                tiled_shifts,
            ) == _sorted_row_multisets(scalar_nm, scalar_nn, scalar_shifts)

    def test_target_indices_rejects_full_size_user_buffers(self, device):
        """Partial lists require compact user buffers."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        with pytest.raises(ValueError, match="neighbor_matrix"):
            naive_neighbor_list(
                positions,
                0.75,
                neighbor_matrix=torch.full((4, 4), 4, dtype=torch.int32, device=device),
                num_neighbors=torch.zeros((2,), dtype=torch.int32, device=device),
                target_indices=torch.tensor([2, 0], dtype=torch.int32, device=device),
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [False, True])
    @pytest.mark.parametrize("pbc_mode", ["none", "wrapped", "prewrapped"])
    def test_target_indices_tile_matches_scalar(
        self,
        device,
        dtype,
        half_fill,
        pbc_mode,
    ):
        """Explicit tile matches scalar compact topology for all PBC modes."""
        if not str(device).startswith("cuda"):
            pytest.skip("Tiled partial parity is CUDA-only.")
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        target_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
        kwargs = {}
        if pbc_mode != "none":
            cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * 4.0
            pbc = torch.ones((1, 3), dtype=torch.bool, device=device)
            shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
                cell,
                1.1,
                pbc,
            )
            kwargs = {
                "cell": cell,
                "pbc": pbc,
                "wrap_positions": pbc_mode == "wrapped",
                "shift_range_per_dimension": shift_range,
                "num_shifts_per_system": num_shifts,
                "max_shifts_per_system": max_shifts,
            }
        scalar = naive_neighbor_list(
            positions,
            1.1,
            max_neighbors=32,
            half_fill=half_fill,
            target_indices=target_indices,
            strategy="scalar",
            **kwargs,
        )
        tiled = naive_neighbor_list(
            positions,
            1.1,
            max_neighbors=32,
            half_fill=half_fill,
            target_indices=target_indices,
            strategy="tile",
            **kwargs,
        )
        torch.testing.assert_close(scalar[1], tiled[1], rtol=0, atol=0)
        scalar_shifts = scalar[2] if pbc_mode != "none" else None
        tiled_shifts = tiled[2] if pbc_mode != "none" else None
        assert _sorted_row_multisets(scalar[0], scalar[1], scalar_shifts) == (
            _sorted_row_multisets(tiled[0], tiled[1], tiled_shifts)
        )

    def test_target_indices_tile_rejects_cpu(self, device):
        """Explicit partial tile rejects CPU before native dispatch."""
        if str(device).startswith("cuda"):
            pytest.skip("CPU-only rejection case.")
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        with pytest.raises(ValueError, match="requires CUDA"):
            naive_neighbor_list(
                positions,
                1.0,
                max_neighbors=4,
                target_indices=torch.tensor([0], dtype=torch.int32, device=device),
                strategy="tile",
            )

    @pytest.mark.gpu
    def test_target_indices_rejects_cross_device(self, device):
        """Partial targets must share the positions device."""
        if not str(device).startswith("cuda"):
            pytest.skip("CUDA is required for cross-device coverage.")
        positions = torch.zeros((3, 3), dtype=torch.float32, device=device)
        targets = torch.tensor([0], dtype=torch.int32, device="cpu")
        with pytest.raises(ValueError, match="same device"):
            naive_neighbor_list(
                positions,
                1.0,
                max_neighbors=4,
                target_indices=targets,
                strategy="scalar",
            )

    @pytest.mark.gpu
    def test_target_indices_rejects_cross_device_output_buffers(self, device):
        """Compact output buffers must share the positions device."""
        if not str(device).startswith("cuda"):
            pytest.skip("CUDA is required for cross-device coverage.")
        positions = torch.zeros((3, 3), dtype=torch.float32, device=device)
        targets = torch.tensor([0], dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="same device"):
            naive_neighbor_list(
                positions,
                1.0,
                max_neighbors=4,
                target_indices=targets,
                neighbor_matrix=torch.empty((1, 4), dtype=torch.int32),
                num_neighbors=torch.empty((1,), dtype=torch.int32),
                strategy="scalar",
            )

    @pytest.mark.gpu
    def test_target_indices_geometry_rejects_cross_device_output_buffer(self, device):
        """Partial geometry output buffers must share the positions device."""
        if not str(device).startswith("cuda"):
            pytest.skip("CUDA is required for cross-device coverage.")
        positions = torch.zeros((3, 3), dtype=torch.float32, device=device)
        targets = torch.tensor([0], dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="same device"):
            naive_neighbor_list(
                positions,
                1.0,
                max_neighbors=4,
                target_indices=targets,
                neighbor_matrix=torch.empty((1, 4), dtype=torch.int32, device=device),
                num_neighbors=torch.empty((1,), dtype=torch.int32, device=device),
                neighbor_distances=torch.empty((1, 4), dtype=torch.float32),
                return_distances=True,
                strategy="scalar",
            )

    def test_target_indices_auto_forwards_native_dispatch(self, device, monkeypatch):
        """Topology-only auto resolves to scalar and preserves compact targets."""
        seen = {}

        def fake_launcher(**kwargs):
            seen.update(kwargs)

        monkeypatch.setattr(
            "nvalchemiops.torch.neighbors.naive._naive_neighbor_matrix_no_pbc",
            fake_launcher,
        )
        positions = torch.zeros((3, 3), dtype=torch.float32, device=device)
        targets = torch.tensor([2, 0], dtype=torch.int32, device=device)
        naive_neighbor_list(
            positions,
            1.0,
            max_neighbors=4,
            target_indices=targets,
            strategy="auto",
        )
        assert seen["strategy"] == "scalar"
        assert seen["target_indices"] is targets

    @pytest.mark.gpu
    def test_target_indices_tile_overflow_contract(self, device):
        """Partial tile preserves uncapped counts and compact overflow behavior."""
        if not str(device).startswith("cuda"):
            pytest.skip("CUDA is required for tiled overflow coverage.")
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        targets = torch.tensor([0, 3], dtype=torch.int32, device=device)
        matrix, counts = naive_neighbor_list(
            positions,
            1.0,
            max_neighbors=1,
            fill_value=-7,
            target_indices=targets,
            strategy="tile",
        )
        assert counts.tolist() == [2, 0]
        assert int(counts[1]) == 0
        assert int(matrix[1, 0]) == -7
        with pytest.raises(NeighborOverflowError):
            naive_neighbor_list(
                positions,
                1.0,
                max_neighbors=1,
                fill_value=-7,
                target_indices=targets,
                strategy="tile",
                return_neighbor_list=True,
            )


class TestNaiveEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_system(self, device, dtype, half_fill):
        """Test behavior with empty position array."""
        positions_empty = torch.empty(0, 3, dtype=dtype, device=device)
        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions=positions_empty,
            cutoff=1.0,
            pbc=None,
            cell=None,
            max_neighbors=10,
            half_fill=half_fill,
        )
        assert neighbor_matrix.shape == (0, 10)
        assert num_neighbors.shape == (0,)

    def test_single_atom(self, device, dtype, half_fill):
        """Test behavior with single atom (should have no neighbors)."""
        positions_single = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions=positions_single,
            cutoff=1.0,
            pbc=None,
            cell=None,
            max_neighbors=10,
            half_fill=half_fill,
        )
        assert num_neighbors[0].item() == 0, "Single atom should have no neighbors"

    def test_zero_cutoff(self, device, dtype, half_fill):
        """Test that zero cutoff produces no neighbors."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=4, dtype=dtype, device=device
        )
        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions=positions,
            cutoff=0.0,
            pbc=None,
            cell=None,
            max_neighbors=10,
            half_fill=half_fill,
        )
        assert torch.all(num_neighbors == 0), "Zero cutoff should find no neighbors"

    def test_large_cutoff_with_pbc(self, device, dtype, half_fill):
        """Test with cutoff larger than cell size."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )

        # Cutoff larger than cell size
        large_cutoff = 5.0
        max_neighbors = 200

        _, num_neighbors, _ = naive_neighbor_list(
            positions=positions,
            cutoff=large_cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        # Should find many neighbors (including periodic images)
        assert num_neighbors.sum() > 0
        assert torch.all(num_neighbors > 0)

    def test_extreme_elongated_cell(self, device, dtype, half_fill):
        """Test with extreme cell aspect ratios."""
        positions = torch.rand(10, 3, dtype=dtype, device=device)
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]],
            dtype=dtype,
            device=device,
        ).reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], device=device).reshape(1, 3)
        cutoff = 0.2
        max_neighbors = 20

        # Should handle extreme aspect ratios without crashing
        _, num_neighbors, _ = naive_neighbor_list(
            positions=positions * torch.tensor([10.0, 0.1, 0.1], device=device),
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        assert torch.all(num_neighbors >= 0)

    def test_max_neighbors_overflow(self, device, dtype, half_fill):
        """Test behavior when max_neighbors is too small."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cell = cell.reshape(1, 3, 3)
        pbc = pbc.reshape(1, 3)

        cutoff = 2.0  # Large cutoff to find many neighbors
        max_neighbors = 3  # Artificially small to trigger overflow

        # Should not crash, but may not find all neighbors
        neighbor_matrix, num_neighbors, unit_shifts = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        # Should produce valid output
        assert torch.all(num_neighbors >= 0)
        assert neighbor_matrix.shape == (positions.shape[0], max_neighbors)
        assert unit_shifts.shape == (positions.shape[0], max_neighbors, 3)


class TestNaiveErrors:
    """Test error handling and input validation."""

    def test_mismatched_cell_without_pbc(self, device, dtype):
        """Test that providing cell without pbc raises error."""
        positions, cell, _ = create_simple_cubic_system(dtype=dtype, device=device)

        with pytest.raises(
            ValueError, match="If cell is provided, pbc must also be provided"
        ):
            naive_neighbor_list(
                positions,
                1.0,
                pbc=None,
                cell=cell,
                max_neighbors=10,
            )

    def test_mismatched_pbc_without_cell(self, device, dtype):
        """Test that providing pbc without cell raises error."""
        positions, _, pbc = create_simple_cubic_system(dtype=dtype, device=device)

        with pytest.raises(
            ValueError, match="If pbc is provided, cell must also be provided"
        ):
            naive_neighbor_list(
                positions,
                1.0,
                pbc=pbc,
                cell=None,
                max_neighbors=10,
            )


class TestNaiveOutputFormats:
    """Test different output formats (matrix vs list)."""

    def test_matrix_format_output_shapes_no_pbc(self, device, dtype, half_fill):
        """Test neighbor matrix format output shapes (no PBC)."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20

        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=None,
            cell=None,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
            return_neighbor_list=False,
        )

        # Verify shapes and types
        assert neighbor_matrix.dtype == torch.int32
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix.shape == (positions.shape[0], max_neighbors)
        assert num_neighbors.shape == (positions.shape[0],)
        assert neighbor_matrix.device == torch.device(device)
        assert num_neighbors.device == torch.device(device)

    def test_matrix_format_output_shapes_with_pbc(self, device, dtype, half_fill):
        """Test neighbor matrix format output shapes (with PBC)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20

        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
            return_neighbor_list=False,
        )

        # Verify shapes and types
        assert neighbor_matrix.dtype == torch.int32
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix_shifts.dtype == torch.int32
        assert neighbor_matrix.shape == (positions.shape[0], max_neighbors)
        assert num_neighbors.shape == (positions.shape[0],)
        assert neighbor_matrix_shifts.shape == (positions.shape[0], max_neighbors, 3)

    def test_list_format_output_shapes_no_pbc(self, device, dtype):
        """Test neighbor list (COO) format output shapes (no PBC)."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1

        neighbor_list, neighbor_ptr = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=None,
            cell=None,
            max_neighbors=20,
            return_neighbor_list=True,
        )

        # Verify shapes and types
        assert neighbor_list.dtype == torch.int32
        assert neighbor_ptr.dtype == torch.int32
        assert neighbor_list.shape[0] == 2
        assert neighbor_ptr.shape == (positions.shape[0] + 1,)
        assert neighbor_list.device == torch.device(device)
        assert neighbor_ptr.device == torch.device(device)

    def test_list_format_output_shapes_with_pbc(self, device, dtype):
        """Test neighbor list (COO) format output shapes (with PBC)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1

        neighbor_list, neighbor_ptr, neighbor_shifts = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=20,
            return_neighbor_list=True,
        )

        # Verify shapes and types
        assert neighbor_list.dtype == torch.int32
        assert neighbor_ptr.dtype == torch.int32
        assert neighbor_shifts.dtype == torch.int32
        assert neighbor_list.shape[0] == 2
        assert neighbor_ptr.shape == (positions.shape[0] + 1,)
        assert neighbor_shifts.shape[1] == 3

    def test_preallocated_output_no_pbc(self, device, dtype, half_fill):
        """Test with preallocated output tensors (no PBC)."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20
        fill_value = -1

        # Preallocate tensors
        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            fill_value,
            dtype=torch.int32,
            device=device,
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        # Call with preallocated tensors
        _ = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            half_fill=half_fill,
            return_neighbor_list=False,
        )

        # When preallocated, return is None or tuple
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix.dtype == torch.int32
        assert torch.all(num_neighbors >= 0)

    def test_preallocated_output_with_pbc(self, device, dtype, half_fill):
        """Test with preallocated output tensors (with PBC)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20
        fill_value = -1

        shift_range_per_dimension, num_shifts, max_shifts = compute_naive_num_shifts(
            cell, cutoff, pbc
        )

        # Preallocate tensors
        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            fill_value,
            dtype=torch.int32,
            device=device,
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (positions.shape[0], max_neighbors, 3),
            dtype=torch.int32,
            device=device,
        )

        # Call with preallocated tensors
        _ = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            shift_range_per_dimension=shift_range_per_dimension,
            num_shifts_per_system=num_shifts,
            max_shifts_per_system=max_shifts,
            half_fill=half_fill,
            return_neighbor_list=False,
        )

        # Verify output
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix.dtype == torch.int32
        assert neighbor_matrix_shifts.dtype == torch.int32
        assert torch.all(num_neighbors >= 0)

    def test_conversion_between_matrix_and_list_formats(self, device, dtype):
        """Test that matrix and list formats contain same neighbor information."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20

        # Get matrix format
        neighbor_matrix, num_neighbors, _ = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            return_neighbor_list=False,
        )

        # Get list format
        neighbor_list, neighbor_ptr, _ = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            return_neighbor_list=True,
        )

        # Total number of neighbors should match
        matrix_total = num_neighbors.sum().item()
        list_total = neighbor_list.shape[1]
        assert matrix_total == list_total


class TestNaiveCompile:
    """Test torch.compile compatibility."""

    @pytest.mark.slow
    def test_compile_no_pbc(self, device, dtype, half_fill):
        """Test that naive_neighbor_list can be compiled (no PBC)."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=50, dtype=dtype, device=device
        )
        cutoff = 3.0
        max_neighbors = 100

        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            50,
            dtype=torch.int32,
            device=device,
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        # Test compiled version
        @torch.compile
        def compiled_naive_neighbor_list(
            positions,
            cutoff,
            neighbor_matrix,
            num_neighbors,
            half_fill,
        ):
            return naive_neighbor_list(
                positions=positions,
                cutoff=cutoff,
                neighbor_matrix=neighbor_matrix,
                num_neighbors=num_neighbors,
                half_fill=half_fill,
            )

        compiled_naive_neighbor_list(
            positions,
            cutoff,
            neighbor_matrix,
            num_neighbors,
            half_fill,
        )

        # Verify results
        assert num_neighbors.sum() > 0
        num_rows = positions.shape[0] - int(half_fill)
        for i in range(num_rows):
            assert num_neighbors[i].item() > 0
            neighbor_row = neighbor_matrix[i]
            mask = neighbor_row != 50
            assert neighbor_row[mask].shape == (num_neighbors[i].item(),)


class TestNaiveCudaGraph:
    """CUDA graph capture coverage for explicit-buffer naive neighbor paths."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for CUDA graph capture"
    )
    def test_pbc_explicit_buffers_cuda_graph_capture(self):
        """Capture the PBC runtime path with prepared metadata and scratch."""
        device = torch.device("cuda")
        dtype = torch.float64
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 50
        fill_value = positions.shape[0]
        cell = cell.reshape(1, 3, 3)
        pbc = pbc.reshape(1, 3)
        shift_range_per_dimension, num_shifts, max_shifts = compute_naive_num_shifts(
            cell, cutoff, pbc
        )

        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            fill_value,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts = torch.zeros(
            (positions.shape[0], max_neighbors, 3),
            dtype=torch.int32,
            device=device,
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )
        wrapped_positions = torch.empty_like(positions)
        per_atom_cell_offsets = torch.empty(
            (positions.shape[0], 3), dtype=torch.int32, device=device
        )
        inv_cell = torch.empty_like(cell)

        def run() -> None:
            neighbor_matrix.fill_(fill_value)
            neighbor_matrix_shifts.zero_()
            num_neighbors.zero_()
            naive_neighbor_list(
                positions=positions,
                cutoff=cutoff,
                cell=cell,
                pbc=pbc,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                num_neighbors=num_neighbors,
                shift_range_per_dimension=shift_range_per_dimension,
                num_shifts_per_system=num_shifts,
                max_shifts_per_system=max_shifts,
                positions_wrapped_buffer=wrapped_positions,
                per_atom_cell_offsets_buffer=per_atom_cell_offsets,
                inv_cell_buffer=inv_cell,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)

        wp_device = wp.get_device(str(device))
        wp_stream = wp.stream_from_torch(stream)
        with torch.cuda.stream(stream), wp.ScopedStream(wp_stream):
            wp.capture_begin(wp_device, wp_stream)
            run()
            graph = wp.capture_end(wp_device, wp_stream)
        wp.capture_launch(graph)
        torch.cuda.synchronize()

        assert num_neighbors.sum() > 0
        assert torch.all(num_neighbors > 0)

    @pytest.mark.slow
    def test_compile_with_pbc(self, device, dtype, half_fill):
        """Test that naive_neighbor_list can be compiled (with PBC)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=50, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 50
        cell = cell.reshape(1, 3, 3)
        pbc = pbc.reshape(1, 3)
        shift_range_per_dimension, num_shifts, max_shifts = compute_naive_num_shifts(
            cell, cutoff, pbc
        )

        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            50,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts = torch.zeros(
            (positions.shape[0], max_neighbors, 3),
            dtype=torch.int32,
            device=device,
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        # Test compiled version
        @torch.compile
        def compiled_naive_neighbor_list(
            positions,
            cutoff,
            cell,
            pbc,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            shift_range_per_dimension,
            num_shifts,
            max_shifts,
            half_fill,
        ):
            return naive_neighbor_list(
                positions=positions,
                cutoff=cutoff,
                cell=cell,
                pbc=pbc,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                num_neighbors=num_neighbors,
                shift_range_per_dimension=shift_range_per_dimension,
                num_shifts_per_system=num_shifts,
                max_shifts_per_system=max_shifts,
                half_fill=half_fill,
            )

        compiled_naive_neighbor_list(
            positions,
            cutoff,
            cell,
            pbc,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            shift_range_per_dimension,
            num_shifts,
            max_shifts,
            half_fill,
        )

        # Verify results
        assert num_neighbors.sum() > 0
        num_rows = positions.shape[0] - int(half_fill)
        for i in range(num_rows):
            assert num_neighbors[i].item() > 0
            neighbor_row = neighbor_matrix[i]
            mask = neighbor_row != 50
            assert neighbor_row[mask].shape == (num_neighbors[i].item(),)


class TestNaivePerformance:
    """Test performance characteristics and scaling (marked as slow)."""

    def test_cutoff_scaling(self, device):
        """Test that neighbor count increases with cutoff."""
        dtype = torch.float32
        num_atoms = 50
        max_neighbors = 200

        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=num_atoms, dtype=dtype, device=device
        )

        # Test different cutoffs
        cutoffs = [0.5, 1.0, 1.5, 2.0]
        neighbor_counts = []

        for cutoff in cutoffs:
            _, num_neighbors, _ = naive_neighbor_list(
                positions,
                cutoff,
                pbc=pbc,
                cell=cell,
                max_neighbors=max_neighbors,
            )
            total_pairs = num_neighbors.sum().item()
            neighbor_counts.append(total_pairs)

        # Verify neighbor count increases with cutoff
        for i in range(1, len(neighbor_counts)):
            assert neighbor_counts[i] >= neighbor_counts[i - 1], (
                f"Neighbor count should increase with cutoff: {neighbor_counts}"
            )

    @pytest.mark.slow
    def test_memory_scaling(self, device):
        """Test that memory usage scales reasonably with system size."""
        import gc

        dtype = torch.float32
        cutoff = 1.1

        # Test different system sizes
        sizes = [10, 20] if device == "cpu" else [50, 100]

        for num_atoms in sizes:
            positions, cell, pbc = create_simple_cubic_system(
                num_atoms=num_atoms, dtype=dtype, device=device
            )
            cell = cell.reshape(1, 3, 3)
            pbc = pbc.reshape(1, 3)

            max_neighbors = 100

            # Clear cache before test
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

            # Run naive implementation
            neighbor_matrix, num_neighbors, unit_shifts = naive_neighbor_list(
                positions=positions,
                cutoff=cutoff,
                pbc=pbc,
                cell=cell,
                max_neighbors=max_neighbors,
            )

            # Verify output shapes are reasonable
            assert neighbor_matrix.shape == (num_atoms, max_neighbors)
            assert unit_shifts.shape == (num_atoms, max_neighbors, 3)
            assert num_neighbors.shape == (num_atoms,)
            assert torch.all(num_neighbors >= 0)
            assert torch.all(num_neighbors <= max_neighbors)

            # Clean up
            del neighbor_matrix, unit_shifts, num_neighbors, positions, cell, pbc
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()


class TestNaiveSelectiveRebuildFlags:
    """Test selective rebuild (rebuild_flags) for naive_neighbor_list torch binding."""

    def test_partial_rebuild_flags_are_rejected(self, device):
        """Compact rows cannot be combined with selective rebuild flags."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        with pytest.raises(
            NotImplementedError,
            match=r"^Partial neighbor lists do not support rebuild_flags$",
        ):
            naive_neighbor_list(
                positions,
                1.0,
                max_neighbors=4,
                target_indices=torch.tensor([0], dtype=torch.int32, device=device),
                rebuild_flags=torch.ones(1, dtype=torch.bool, device=device),
            )

    def test_no_pbc_wrapper_rejects_partial_rebuild_before_mutation(self, device):
        """The no-PBC custom op preserves counts on an unsupported request."""
        positions = torch.zeros((2, 3), dtype=torch.float32, device=device)
        neighbor_matrix = torch.full((1, 4), 2, dtype=torch.int32, device=device)
        num_neighbors = torch.full((1,), 37, dtype=torch.int32, device=device)

        with pytest.raises(
            NotImplementedError,
            match=r"^Partial neighbor lists do not support rebuild_flags$",
        ):
            _naive_neighbor_matrix_no_pbc(
                positions=positions,
                cutoff=1.0,
                neighbor_matrix=neighbor_matrix,
                num_neighbors=num_neighbors,
                rebuild_flags=torch.ones(1, dtype=torch.bool, device=device),
                target_indices=torch.tensor([0], dtype=torch.int32, device=device),
            )

        assert num_neighbors.tolist() == [37]

    def test_pbc_wrapper_rejects_partial_rebuild_before_mutation(self, device):
        """The PBC custom op preserves counts on an unsupported request."""
        positions = torch.zeros((2, 3), dtype=torch.float32, device=device)
        cell = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
        pbc = torch.ones((1, 3), dtype=torch.bool, device=device)
        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(cell, 1.0, pbc)
        neighbor_matrix = torch.full((1, 4), 2, dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (1, 4, 3),
            dtype=torch.int32,
            device=device,
        )
        num_neighbors = torch.full((1,), 37, dtype=torch.int32, device=device)

        with pytest.raises(
            NotImplementedError,
            match=r"^Partial neighbor lists do not support rebuild_flags$",
        ):
            _naive_neighbor_matrix_pbc(
                positions=positions,
                cutoff=1.0,
                cell=cell,
                pbc=pbc,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                num_neighbors=num_neighbors,
                shift_range_per_dimension=shift_range,
                num_shifts_per_system=num_shifts,
                max_shifts_per_system=max_shifts,
                rebuild_flags=torch.ones(1, dtype=torch.bool, device=device),
                target_indices=torch.tensor([0], dtype=torch.int32, device=device),
            )

        assert num_neighbors.tolist() == [37]

    def test_no_rebuild_preserves_data(self, device, dtype):
        """Flag=False: neighbor data should remain unchanged."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20

        # Initial full build (pre-allocated output)
        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors), -1, dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
        )

        saved_nm = neighbor_matrix.clone()
        saved_nn = num_neighbors.clone()

        # Selective rebuild with flag=False: data should be unchanged
        rebuild_flags = torch.zeros(1, dtype=torch.bool, device=device)
        naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            rebuild_flags=rebuild_flags,
        )

        assert torch.equal(num_neighbors, saved_nn), (
            "num_neighbors must be unchanged when rebuild_flags is False"
        )
        for i in range(positions.shape[0]):
            n = num_neighbors[i].item()
            assert torch.equal(neighbor_matrix[i, :n], saved_nm[i, :n]), (
                f"neighbor_matrix row {i} should be unchanged"
            )

    def test_rebuild_updates_data(self, device, dtype):
        """Flag=True: result should match a fresh full rebuild."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20

        # Reference: full build
        nm_ref = torch.full(
            (positions.shape[0], max_neighbors), -1, dtype=torch.int32, device=device
        )
        nn_ref = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)
        naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm_ref,
            num_neighbors=nn_ref,
        )

        # Selective rebuild with flag=True: should match reference
        nm_sel = torch.full(
            (positions.shape[0], max_neighbors), 99, dtype=torch.int32, device=device
        )
        nn_sel = torch.full((positions.shape[0],), 99, dtype=torch.int32, device=device)

        rebuild_flags = torch.ones(1, dtype=torch.bool, device=device)
        naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm_sel,
            num_neighbors=nn_sel,
            rebuild_flags=rebuild_flags,
        )

        assert torch.equal(nn_sel, nn_ref), (
            "num_neighbors should match full rebuild when flag=True"
        )


class TestNaiveAutograd:
    """Differentiable per-pair distances / vectors via the autograd primitive."""

    def _make_system(self, device, n=6, box=4.0):
        torch.manual_seed(0)
        pos = torch.randn(n, 3, dtype=torch.float64, device=device) * 0.3
        cell = torch.eye(3, dtype=torch.float64, device=device) * box
        pbc = torch.tensor([True, True, True], device=device)
        return pos, cell, pbc

    def test_forward_returns_differentiable_no_pbc(self, device):
        pos, _, _ = self._make_system(device)
        pos.requires_grad_(True)
        nm, nn, d, v = naive_neighbor_list(
            pos,
            1.5,
            max_neighbors=8,
            return_distances=True,
            return_vectors=True,
        )
        assert d.requires_grad and v.requires_grad

    def test_forward_returns_differentiable_pbc(self, device):
        pos, cell, pbc = self._make_system(device)
        pos.requires_grad_(True)
        nm, nn, shifts, d, v = naive_neighbor_list(
            pos,
            1.5,
            cell=cell,
            pbc=pbc,
            max_neighbors=8,
            return_distances=True,
            return_vectors=True,
        )
        assert d.requires_grad and v.requires_grad

    @pytest.mark.slow
    def test_partial_no_pbc_distance_gradcheck_matches_selected_full_rows(self, device):
        """Compact no-PBC distance gradients use each target's source atom row."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.7, 0.2, 0.0],
                [2.4, 0.0, 0.0],
                [3.2, -0.3, 0.0],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        target_indices = torch.tensor([3, 0], dtype=torch.int32, device=device)

        def partial_loss(pos):
            _, _, distances = naive_neighbor_list(
                pos,
                1.1,
                max_neighbors=4,
                target_indices=target_indices,
                return_distances=True,
            )
            return distances.sum()

        assert torch.autograd.gradcheck(
            partial_loss,
            (positions,),
            atol=1e-5,
            eps=1e-6,
            nondet_tol=1e-7,
        )
        partial_grad = torch.autograd.grad(partial_loss(positions), positions)[0]
        _, _, full_distances = naive_neighbor_list(
            positions,
            1.1,
            max_neighbors=4,
            return_distances=True,
        )
        selected_full_loss = full_distances[target_indices.long()].sum()
        selected_full_grad = torch.autograd.grad(selected_full_loss, positions)[0]

        torch.testing.assert_close(partial_loss(positions), selected_full_loss)
        torch.testing.assert_close(partial_grad, selected_full_grad)

    @pytest.mark.slow
    def test_gradcheck_distances_wrt_positions(self, device):
        pos, cell, pbc = self._make_system(device)
        pos.requires_grad_(True)

        def fn(p):
            _, _, _, d, _ = naive_neighbor_list(
                p,
                1.5,
                cell=cell,
                pbc=pbc,
                max_neighbors=8,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        torch.autograd.gradcheck(fn, (pos,), atol=1e-5, eps=1e-6, nondet_tol=1e-7)

    @pytest.mark.slow
    def test_gradcheck_distances_wrt_cell(self, device):
        pos, cell, pbc = self._make_system(device)
        cell = cell.clone().requires_grad_(True)

        def fn(c):
            _, _, _, d, _ = naive_neighbor_list(
                pos,
                1.5,
                cell=c,
                pbc=pbc,
                max_neighbors=8,
                return_distances=True,
                return_vectors=True,
            )
            return d.sum()

        torch.autograd.gradcheck(fn, (cell,), atol=1e-5, eps=1e-6, nondet_tol=1e-7)

    def test_half_fill_with_pair_outputs(self, device):
        """half_fill=True now combines with per-pair geometry outputs; each emitted
        pair carries a correct distance/vector (self-consistent: ``|vec| == dist``)."""
        pos, cell, pbc = self._make_system(device)
        nm, _nn, _sh, dist, vec = naive_neighbor_list(
            pos,
            1.5,
            cell=cell,
            pbc=pbc,
            max_neighbors=8,
            return_distances=True,
            return_vectors=True,
            half_fill=True,
        )
        active = nm != pos.shape[0]
        assert int(active.sum()) > 0
        assert torch.all(dist[active] <= 1.5 + 1e-4)
        torch.testing.assert_close(
            dist[active], vec[active].norm(dim=-1), atol=1e-5, rtol=1e-5
        )

    def test_pair_outputs_reject_rebuild_flags(self, device):
        """rebuild_flags stays unsupported with pair outputs (stale cached geometry)."""
        pos, cell, pbc = self._make_system(device)
        with pytest.raises(NotImplementedError, match="rebuild_flags"):
            naive_neighbor_list(
                pos,
                1.5,
                cell=cell,
                pbc=pbc,
                max_neighbors=8,
                return_distances=True,
                rebuild_flags=torch.ones(1, dtype=torch.bool, device=device),
            )

    @pytest.mark.slow
    def test_gradgradcheck_second_order(self, device):
        """Second-order autograd: gradient-of-gradient is also correct."""
        pos, cell, pbc = self._make_system(device)
        pos.requires_grad_(True)

        def fn(p):
            *_, d, _ = naive_neighbor_list(
                p,
                1.5,
                cell=cell,
                pbc=pbc,
                max_neighbors=8,
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
        """Non-grad inputs through the autograd path: outputs are plain
        tensors and the active slots match the non-autograd path.

        The two kernel specializations may emit neighbors in different
        orders, so compare as sets per row.
        """
        pos, cell, pbc = self._make_system(device)
        nm_a, nn_a, sh_a = naive_neighbor_list(
            pos,
            1.5,
            cell=cell,
            pbc=pbc,
            max_neighbors=8,
        )
        nm_b, nn_b, sh_b, d_b, v_b = naive_neighbor_list(
            pos,
            1.5,
            cell=cell,
            pbc=pbc,
            max_neighbors=8,
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
