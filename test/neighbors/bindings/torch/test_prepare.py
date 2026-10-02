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

"""Tests for :func:`prepare_neighbor_list_method`."""

from __future__ import annotations

import pytest
import torch

from nvalchemiops.torch.neighbors import (
    NeighborOverflowError,
    neighbor_list,
    prepare_neighbor_list_method,
)

from ...test_utils import (
    count_neighbors_reference,
    create_batch_systems,
    create_random_system,
)

# Every naive and cell-list strategy, coarse and fine-grained.
SINGLE_METHODS = (
    "naive",
    "naive_scalar",
    "naive_tile",
    "cell_list",
    "cell_list_atom_centric",
    "cell_list_pair_centric",
)
BATCH_METHODS = tuple(f"batch_{name}" for name in SINGLE_METHODS)
# cluster_tile is guarded: CUDA, float32 and fully periodic only. It is listed
# separately so those guards do not constrain the other strategies.
CLUSTER_TILE_METHODS = ("cluster_tile",)
BATCH_CLUSTER_TILE_METHODS = ("batch_cluster_tile",)
DUAL_METHODS = ("naive_dual_cutoff",)
BATCH_DUAL_METHODS = ("batch_naive_dual_cutoff",)


def periodic_system(device, num_atoms=60, cell_size=6.0):
    """A periodic box with enough neighbors to exercise capacity handling."""
    return create_random_system(
        num_atoms=num_atoms,
        cell_size=cell_size,
        dtype=torch.float64,
        device=device,
    )


_FREE_PBC = torch.tensor([False, False, False])


def _free_cell(device, size=60.0):
    """A box large enough that no periodic image is within any test cutoff."""
    return torch.eye(3, dtype=torch.float64, device=device) * size


def _skip_unsupported(method, device):
    """Skip strategies a device genuinely cannot run."""
    if "pair_centric" in method and device == "cpu":
        pytest.skip("pair_centric kernels use CUDA block scheduling")
    if "cluster_tile" in method and device == "cpu":
        pytest.skip("cluster_tile requires CUDA")


def reference_pairs(positions, cell, pbc, cutoff):
    """Total pair count from the shared brute-force reference."""
    return int(count_neighbors_reference(positions, cell, pbc, cutoff).sum())


def reference_pairs_batched(positions, cell, pbc, ptr, cutoff):
    """Sum the per-system reference pair counts."""
    return sum(
        reference_pairs(positions[a:b], cell[i], pbc[i], cutoff)
        for i, (a, b) in enumerate(zip(ptr[:-1].tolist(), ptr[1:].tolist()))
    )


class TestPrepareNeighborListMethod:
    """Behaviour of the preallocation helper."""

    @pytest.mark.parametrize("half_fill", [False, True])
    def test_matches_reference_periodic(self, device, half_fill):
        """Prepared buffers reproduce a brute-force periodic pair count."""
        positions, cell, pbc = periodic_system(device)
        cutoff = 2.5
        kwargs = prepare_neighbor_list_method(
            positions, cutoff, cell=cell, pbc=pbc, half_fill=half_fill
        )
        _matrix, counts, _shifts = neighbor_list(positions, cutoff, **kwargs)
        expected = reference_pairs(positions, cell, pbc, cutoff)
        if half_fill:
            expected //= 2
        assert int(counts.sum()) == expected

    @pytest.mark.parametrize("method", ["naive", "cell_list"])
    def test_explicit_method_is_honored(self, device, method):
        """An explicit method is forwarded and still produces correct counts."""
        positions, cell, pbc = periodic_system(device)
        cutoff = 2.5
        kwargs = prepare_neighbor_list_method(
            positions, cutoff, cell=cell, pbc=pbc, method=method
        )
        assert kwargs["method"] == method
        _matrix, counts, _shifts = neighbor_list(positions, cutoff, **kwargs)
        assert int(counts.sum()) == reference_pairs(positions, cell, pbc, cutoff)

    def test_free_boundary_omits_cell_and_pbc(self, device):
        """Without a cell the result carries neither cell nor pbc."""
        positions, _cell, _pbc = create_random_system(
            num_atoms=60, dtype=torch.float64, device=device
        )
        cutoff = 2.5
        kwargs = prepare_neighbor_list_method(positions, cutoff)
        assert "cell" not in kwargs
        assert "pbc" not in kwargs
        result = neighbor_list(positions, cutoff, **kwargs)
        # Counts are always the second element; the arity itself is not
        # asserted here because it varies by strategy (see
        # test_free_boundary_arity_varies_by_strategy).
        assert int(result[1].sum()) == reference_pairs(
            positions, _free_cell(device), _FREE_PBC.to(device), cutoff
        )

    def test_all_false_pbc_is_dropped(self, device):
        """An all-False pbc selects the cheaper aperiodic path.

        The periodic kernels are chosen by ``pbc is not None`` rather than by
        value, so forwarding an all-False pbc would search periodic images
        that cannot exist.
        """
        positions, _cell, _pbc = create_random_system(
            num_atoms=60, dtype=torch.float64, device=device
        )
        cell = (torch.eye(3, dtype=torch.float64, device=device) * 40.0).unsqueeze(0)
        pbc = torch.tensor([False, False, False], device=device)

        kwargs = prepare_neighbor_list_method(positions, 2.5, cell=cell, pbc=pbc)
        assert "cell" not in kwargs
        assert "pbc" not in kwargs
        assert "neighbor_matrix_shifts" not in kwargs
        result = neighbor_list(positions, 2.5, **kwargs)
        assert int(result[1].sum()) == reference_pairs(
            positions, _free_cell(device), _FREE_PBC.to(device), 2.5
        )

    def test_free_boundary_arity_varies_by_strategy(self, device):
        """Pin the free-boundary return arity, which is strategy dependent.

        Without a cell, ``naive`` returns ``(matrix, counts)`` while
        ``cell_list`` synthesizes a bounding box and returns
        ``(matrix, counts, shifts)``. Because the auto-selector chooses
        differently per device and system size, callers that do not fix
        ``method`` must index the result rather than unpack it.
        """
        positions, _cell, _pbc = create_random_system(
            num_atoms=60, dtype=torch.float64, device=device
        )
        expected = reference_pairs(
            positions, _free_cell(device), _FREE_PBC.to(device), 2.5
        )

        naive = neighbor_list(
            positions,
            2.5,
            **prepare_neighbor_list_method(positions, 2.5, method="naive"),
        )
        cells = neighbor_list(
            positions,
            2.5,
            **prepare_neighbor_list_method(positions, 2.5, method="cell_list"),
        )
        assert len(naive) == 2
        assert len(cells) == 3
        assert int(naive[1].sum()) == expected
        assert int(cells[1].sum()) == expected

    def test_explicit_max_neighbors_is_honored(self, device):
        """An explicit capacity is used verbatim, without second-guessing."""
        positions, cell, pbc = periodic_system(device)
        kwargs = prepare_neighbor_list_method(
            positions, 2.5, cell=cell, pbc=pbc, max_neighbors=4
        )
        assert kwargs["max_neighbors"] == 4
        assert kwargs["neighbor_matrix"].shape[1] == 4

    def test_num_neighbors_reports_capacity_shortfall(self, device):
        """Counts expose an under-sized capacity, which is the caller's check.

        The matrix path does not synchronize to check capacity itself, so
        ``num_neighbors`` reports what was *found* rather than what was
        stored. Comparing it against ``max_neighbors`` is how a caller
        detects that the buffers need re-preparing.
        """
        positions, cell, pbc = periodic_system(device)
        cutoff = 2.5
        kwargs = prepare_neighbor_list_method(
            positions, cutoff, cell=cell, pbc=pbc, max_neighbors=8
        )
        _matrix, counts, _shifts = neighbor_list(positions, cutoff, **kwargs)
        assert int(counts.max()) > kwargs["max_neighbors"]

    def test_estimated_capacity_holds_for_this_geometry(self, device):
        """The default estimate accommodates this lattice and cutoff."""
        positions, cell, pbc = periodic_system(device)
        cutoff = 2.5
        kwargs = prepare_neighbor_list_method(positions, cutoff, cell=cell, pbc=pbc)
        matrix, counts, _shifts = neighbor_list(positions, cutoff, **kwargs)

        assert int(counts.max()) <= kwargs["max_neighbors"]
        fill_value = positions.shape[0]
        assert int((matrix != fill_value).sum()) == int(counts.sum())

    def test_mismatched_cell_and_pbc_raises(self, device):
        """cell and pbc must be supplied together."""
        positions, cell, _pbc = periodic_system(device)
        with pytest.raises(ValueError, match="together"):
            prepare_neighbor_list_method(positions, 4.0, cell=cell)

    def test_cell_list_state_is_allocated(self, device):
        """The cell-list strategy receives its grid and sort scratch."""
        positions, cell, pbc = periodic_system(device)
        kwargs = prepare_neighbor_list_method(
            positions, 2.5, cell=cell, pbc=pbc, method="cell_list"
        )
        for key in (
            "cells_per_dimension",
            "neighbor_search_radius",
            "atom_periodic_shifts",
            "atom_to_cell_mapping",
            "atoms_per_cell_count",
            "cell_atom_start_indices",
            "cell_atom_list",
            "sorted_positions",
            "sorted_shifts",
        ):
            assert key in kwargs, f"missing cell-list buffer {key!r}"

    def test_naive_periodic_state_is_allocated(self, device):
        """The naive periodic strategy receives shift metadata and wrap scratch."""
        positions, cell, pbc = periodic_system(device)
        kwargs = prepare_neighbor_list_method(
            positions, 2.5, cell=cell, pbc=pbc, method="naive"
        )
        for key in (
            "shift_range_per_dimension",
            "num_shifts_per_system",
            "max_shifts_per_system",
            "positions_wrapped_buffer",
            "per_atom_cell_offsets_buffer",
            "inv_cell_buffer",
        ):
            assert key in kwargs, f"missing naive buffer {key!r}"

    def test_return_neighbor_list_round_trips(self, device):
        """COO output is prepared for and agrees with the matrix pair count."""
        positions, cell, pbc = periodic_system(device)
        cutoff = 2.5
        kwargs = prepare_neighbor_list_method(
            positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=True
        )
        assert kwargs["return_neighbor_list"] is True

        pairs, ptr, shifts = neighbor_list(positions, cutoff, **kwargs)
        expected = reference_pairs(positions, cell, pbc, cutoff)
        assert pairs.shape == (2, expected)
        assert shifts.shape == (expected, 3)
        assert int(ptr[-1]) == expected

    def test_return_neighbor_list_raises_on_capacity_shortfall(self, device):
        """COO output reports a shortfall rather than dropping pairs.

        Unlike the matrix path, the COO build cannot proceed with a truncated
        matrix -- the pair list would simply be missing entries -- so it
        raises. The synchronization that detects this is already paid for by
        the compaction.
        """
        positions, cell, pbc = periodic_system(device)
        kwargs = prepare_neighbor_list_method(
            positions,
            4.0,
            cell=cell,
            pbc=pbc,
            max_neighbors=8,
            return_neighbor_list=True,
        )
        with pytest.raises(NeighborOverflowError):
            neighbor_list(positions, 2.5, **kwargs)

    def test_return_neighbor_list_with_sufficient_capacity(self, device):
        """An adequate capacity yields the full COO pair list."""
        positions, cell, pbc = periodic_system(device)
        cutoff = 2.5
        expected = reference_pairs(positions, cell, pbc, cutoff)
        kwargs = prepare_neighbor_list_method(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=256,
            return_neighbor_list=True,
        )
        pairs, _ptr, _shifts = neighbor_list(positions, cutoff, **kwargs)
        assert pairs.shape[1] == expected

    def test_return_neighbor_list_absent_when_not_requested(self, device):
        """The key is omitted entirely for the default matrix output."""
        positions, cell, pbc = periodic_system(device)
        kwargs = prepare_neighbor_list_method(positions, 2.5, cell=cell, pbc=pbc)
        assert "return_neighbor_list" not in kwargs

    @pytest.mark.parametrize("method", SINGLE_METHODS)
    def test_every_single_system_method(self, device, method):
        """Each naive and cell-list strategy is prepared for and runs."""
        _skip_unsupported(method, device)
        positions, cell, pbc = periodic_system(device)
        cutoff = 2.5
        kwargs = prepare_neighbor_list_method(
            positions, cutoff, cell=cell, pbc=pbc, method=method
        )
        _matrix, counts, _shifts = neighbor_list(positions, cutoff, **kwargs)
        assert int(counts.sum()) == reference_pairs(positions, cell, pbc, cutoff)

    @pytest.mark.parametrize("method", BATCH_METHODS)
    def test_every_batched_method(self, device, method):
        """Each batched naive and cell-list strategy is prepared for and runs."""
        _skip_unsupported(method, device)
        positions, cell, pbc, batch_ptr = create_batch_systems(
            num_systems=4,
            atoms_per_system=[40, 55, 48, 60],
            cell_sizes=[6.0, 6.5, 6.0, 7.0],
            dtype=torch.float64,
            device=device,
        )
        batch_ptr = batch_ptr.to(torch.int32)
        cutoff = 2.5
        kwargs = prepare_neighbor_list_method(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            method=method,
            batch_ptr=batch_ptr,
        )
        _matrix, counts, _shifts = neighbor_list(positions, cutoff, **kwargs)
        expected = reference_pairs_batched(positions, cell, pbc, batch_ptr, cutoff)
        assert int(counts.sum()) == expected

    @pytest.mark.parametrize("method", DUAL_METHODS)
    def test_every_dual_cutoff_method(self, device, method):
        """Dual-cutoff strategies get both numbered output sets."""
        positions, cell, pbc = periodic_system(device)
        # cutoff is the inner radius and cutoff2 the outer; inverting them
        # silently returns the smaller radius for both output sets.
        cutoff, cutoff2 = 1.8, 2.5
        kwargs = prepare_neighbor_list_method(
            positions, cutoff, cutoff2=cutoff2, cell=cell, pbc=pbc, method=method
        )
        for key in ("neighbor_matrix1", "neighbor_matrix2", "num_neighbors1"):
            assert key in kwargs, f"missing dual-cutoff buffer {key!r}"
        assert "neighbor_matrix" not in kwargs

        _m1, counts1, _s1, _m2, counts2, _s2 = neighbor_list(
            positions, cutoff, **kwargs
        )
        assert int(counts1.sum()) == reference_pairs(positions, cell, pbc, cutoff)
        assert int(counts2.sum()) == reference_pairs(positions, cell, pbc, cutoff2)

    @pytest.mark.parametrize("method", BATCH_DUAL_METHODS)
    def test_every_batched_dual_cutoff_method(self, device, method):
        """Batched dual-cutoff strategies combine both extensions."""
        positions, cell, pbc, batch_ptr = create_batch_systems(
            num_systems=3,
            atoms_per_system=[40, 55, 48],
            cell_sizes=[6.0, 6.5, 6.0],
            dtype=torch.float64,
            device=device,
        )
        batch_ptr = batch_ptr.to(torch.int32)
        cutoff, cutoff2 = 1.8, 2.5
        kwargs = prepare_neighbor_list_method(
            positions,
            cutoff,
            cutoff2=cutoff2,
            cell=cell,
            pbc=pbc,
            method=method,
            batch_ptr=batch_ptr,
        )
        _m1, counts1, _s1, _m2, counts2, _s2 = neighbor_list(
            positions, cutoff, **kwargs
        )
        assert int(counts1.sum()) == reference_pairs_batched(
            positions, cell, pbc, batch_ptr, cutoff
        )
        assert int(counts2.sum()) == reference_pairs_batched(
            positions, cell, pbc, batch_ptr, cutoff2
        )

    def test_batched_cell_list_state_is_batch_shaped(self, device):
        """Batched grids are sized per system, not with the single-system path.

        ``estimate_cell_list_sizes`` rejects a ``(num_systems, 3)`` pbc, so the
        batched estimator has to be used instead.
        """
        positions, cell, pbc, batch_ptr = create_batch_systems(
            num_systems=4,
            atoms_per_system=[40, 55, 48, 60],
            cell_sizes=[6.0, 6.5, 6.0, 7.0],
            dtype=torch.float64,
            device=device,
        )
        batch_ptr = batch_ptr.to(torch.int32)
        kwargs = prepare_neighbor_list_method(
            positions,
            2.5,
            cell=cell,
            pbc=pbc,
            method="batch_cell_list",
            batch_ptr=batch_ptr,
        )
        assert kwargs["neighbor_search_radius"].shape == (4, 3)
        assert "cell_offsets" in kwargs

    @pytest.mark.parametrize("method", CLUSTER_TILE_METHODS)
    def test_cluster_tile_single_system(self, device, method):
        """cluster_tile is prepared for under its CUDA/float32/periodic guards."""
        _skip_unsupported(method, device)
        positions, cell, pbc = create_random_system(
            num_atoms=60, cell_size=6.0, dtype=torch.float32, device=device
        )
        cutoff = 2.5
        kwargs = prepare_neighbor_list_method(
            positions, cutoff, cell=cell, pbc=pbc, method=method
        )
        _matrix, counts, _shifts = neighbor_list(positions, cutoff, **kwargs)
        assert int(counts.sum()) == reference_pairs(positions, cell, pbc, cutoff)

    @pytest.mark.parametrize("method", BATCH_CLUSTER_TILE_METHODS)
    def test_cluster_tile_batched(self, device, method):
        """Batched cluster_tile needs a contiguous batch and float32 positions."""
        _skip_unsupported(method, device)
        positions, cell, pbc, batch_ptr = create_batch_systems(
            num_systems=3,
            atoms_per_system=[40, 55, 48],
            cell_sizes=[6.0, 6.5, 6.0],
            dtype=torch.float32,
            device=device,
        )
        batch_ptr = batch_ptr.to(torch.int32)
        cutoff = 2.5
        kwargs = prepare_neighbor_list_method(
            positions, cutoff, cell=cell, pbc=pbc, method=method, batch_ptr=batch_ptr
        )
        _matrix, counts, _shifts = neighbor_list(positions, cutoff, **kwargs)
        expected = reference_pairs_batched(positions, cell, pbc, batch_ptr, cutoff)
        assert int(counts.sum()) == expected

    def test_buffers_are_reused_across_calls(self, device):
        """Repeated calls write into the same tensors rather than new ones."""
        positions, cell, pbc = periodic_system(device)
        kwargs = prepare_neighbor_list_method(positions, 2.5, cell=cell, pbc=pbc)
        first = neighbor_list(positions, 2.5, **kwargs)
        second = neighbor_list(positions, 2.5, **kwargs)
        assert first[0].data_ptr() == kwargs["neighbor_matrix"].data_ptr()
        assert second[0].data_ptr() == kwargs["neighbor_matrix"].data_ptr()
        assert int(first[1].sum()) == int(second[1].sum())
