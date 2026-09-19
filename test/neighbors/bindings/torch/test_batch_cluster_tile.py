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

import os
import subprocess
import sys
import tempfile
import textwrap

import pytest
import torch

import nvalchemiops.torch.neighbors.batch_cluster_tile as batch_cluster_tile_module
from nvalchemiops.neighbors.cluster_tile import (
    estimate_batch_max_tiles_per_group as estimate_core_batch_max_tiles_per_group,
)
from nvalchemiops.neighbors.cluster_tile import (
    estimate_max_tiles_per_group,
)
from nvalchemiops.neighbors.neighbor_utils import (
    NeighborOverflowError,
    TileBufferOverflow,
)
from nvalchemiops.torch.neighbors.batch_cluster_tile import (
    TILE_GROUP_SIZE,
    allocate_batch_cluster_tile_list,
    batch_build_cluster_tile_list,
    batch_cluster_tile_neighbor_list,
    batch_query_cluster_tile,
    estimate_batch_cluster_tile_list_sizes,
    estimate_batch_cluster_tile_segments,
    estimate_batch_max_tiles_per_group,
)
from nvalchemiops.torch.neighbors.neighbor_utils import _validate_segmented_coo_state

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


def _scratch_kwargs(
    scratch: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    """Map allocator outputs to combined build/query keyword arguments."""
    names = (
        "sorted_atom_index",
        "sort_inv",
        "sorted_pos_x",
        "sorted_pos_y",
        "sorted_pos_z",
        "batch_idx_sorted",
        "batch_ptr_padded",
        "group_system",
        "group_ptr",
        "group_ctr_x",
        "group_ctr_y",
        "group_ctr_z",
        "group_ext_x",
        "group_ext_y",
        "group_ext_z",
        "num_tiles",
        "tile_row_group",
        "tile_col_group",
        "tile_system",
    )
    return dict(zip(names, scratch))


def _matrix_pair_sets(
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_shifts: torch.Tensor,
) -> list[frozenset[tuple[int, int, int, int]]]:
    """Return order-independent ``(target, shift)`` rows for matrix output."""
    matrix = neighbor_matrix.cpu()
    counts = num_neighbors.cpu()
    shifts = neighbor_shifts.cpu()
    return [
        frozenset(
            (
                int(matrix[source, slot]),
                int(shifts[source, slot, 0]),
                int(shifts[source, slot, 1]),
                int(shifts[source, slot, 2]),
            )
            for slot in range(int(counts[source]))
        )
        for source in range(matrix.shape[0])
    ]


def _compact_coo_pair_sets(
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    shifts: torch.Tensor,
) -> list[frozenset[tuple[int, int, int, int]]]:
    """Validate compact CSR ownership and return order-independent rows."""
    pointer = neighbor_ptr.cpu().tolist()
    pairs = neighbor_list.cpu()
    shifts_cpu = shifts.cpu()
    assert pointer[0] == 0
    assert pointer[-1] == neighbor_list.shape[1]
    assert all(start <= stop for start, stop in zip(pointer[:-1], pointer[1:]))
    rows = []
    for source, (start, stop) in enumerate(zip(pointer[:-1], pointer[1:])):
        assert torch.all(pairs[0, start:stop] == source)
        rows.append(
            frozenset(
                (
                    int(pairs[1, slot]),
                    int(shifts_cpu[slot, 0]),
                    int(shifts_cpu[slot, 1]),
                    int(shifts_cpu[slot, 2]),
                )
                for slot in range(start, stop)
            )
        )
    return rows


class TestBatchClusterTileValidation:
    """Validate public option combinations rejected before kernel launch."""

    @staticmethod
    def _empty_query_args() -> tuple[tuple[torch.Tensor | object, ...], dict]:
        """Return CPU component inputs suitable for pre-launch validation."""
        natom, max_neighbors = 2, 4
        args = (
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
            torch.eye(3, dtype=torch.float32)[None],
            torch.zeros(1, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            2.0,
            natom,
            torch.empty((natom, max_neighbors), dtype=torch.int32),
            torch.zeros(natom, dtype=torch.int32),
            torch.empty((natom, max_neighbors, 3), dtype=torch.int32),
        )
        return args, {"natom": natom, "max_neighbors": max_neighbors}

    def test_query_requires_complete_secondary_outputs(self):
        """A batched dual-cutoff query requires its complete output triple."""
        args, sizes = self._empty_query_args()
        secondary = torch.empty(
            (sizes["natom"], sizes["max_neighbors"]), dtype=torch.int32
        )

        with pytest.raises(ValueError, match="cutoff2 requires"):
            batch_query_cluster_tile(*args, cutoff2=3.0)
        with pytest.raises(ValueError, match="must be supplied together"):
            batch_query_cluster_tile(
                *args,
                cutoff2=3.0,
                neighbor_matrix2=secondary,
            )
        with pytest.raises(ValueError, match="neighbor_matrix_shifts2"):
            batch_query_cluster_tile(
                *args,
                cutoff2=3.0,
                neighbor_matrix2=secondary,
                num_neighbors2=torch.zeros(sizes["natom"], dtype=torch.int32),
                neighbor_matrix_shifts2=torch.empty(
                    (sizes["natom"], sizes["max_neighbors"], 2),
                    dtype=torch.int32,
                ),
            )

    @pytest.mark.parametrize(
        ("flag", "buffer_name", "bad_shape"),
        [
            ("return_vectors", "neighbor_vectors", (2, 4, 2)),
            ("return_distances", "neighbor_distances", (2, 3)),
        ],
    )
    def test_query_requires_shaped_geometry_outputs(self, flag, buffer_name, bad_shape):
        """Enabled batched geometry requires a correctly shaped buffer."""
        args, _ = self._empty_query_args()

        with pytest.raises(ValueError, match=f"{buffer_name} is required"):
            batch_query_cluster_tile(*args, **{flag: True})
        with pytest.raises(ValueError, match=buffer_name):
            batch_query_cluster_tile(
                *args,
                **{
                    flag: True,
                    buffer_name: torch.empty(bad_shape, dtype=torch.float32),
                },
            )

    def test_wrapper_rejects_partial_secondary_outputs_before_build(self):
        """The batched wrapper rejects partially caller-owned cutoff2 state."""
        positions = torch.zeros((2, 3), dtype=torch.float32)
        cell_batch = torch.eye(3, dtype=torch.float32)[None]
        batch_ptr = torch.tensor([0, 2], dtype=torch.int32)

        with pytest.raises(ValueError, match="must be supplied together"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                max_neighbors=4,
                cutoff2=3.0,
                neighbor_matrix2=torch.empty((2, 4), dtype=torch.int32),
            )

    def test_selective_coo_bootstrap_rejects_partial_state(self):
        """All-true COO bootstrap does not mix caller and allocated state."""
        positions, cell_batch, batch_ptr = _make_batch([32], [8.0], device="cpu")

        with pytest.raises(ValueError, match="complete caller-owned state"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                format="coo",
                rebuild_flags=torch.ones(1, dtype=torch.bool),
                return_state=True,
                neighbor_list=torch.empty((2, 64), dtype=torch.int32),
            )

    def test_segmented_coo_validation_allows_nonselective_state(self):
        """Segmented COO metadata does not require selective rebuild flags."""
        device = torch.device("cpu")
        capacity = 8

        result = _validate_segmented_coo_state(
            device=device,
            num_systems=1,
            neighbor_list=torch.empty((2, capacity), dtype=torch.int32),
            neighbor_list_shifts=torch.empty((capacity, 3), dtype=torch.int32),
            pair_offsets=torch.tensor([0, capacity], dtype=torch.int32),
            pair_counts=torch.zeros(1, dtype=torch.int32),
            rebuild_flags=None,
        )

        assert result == capacity

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

    def test_return_state_requires_rebuild_flags(self):
        """State return is rejected without selective-rebuild inputs."""
        positions, cell_batch, batch_ptr = _make_batch([32], [8.0], device="cpu")

        with pytest.raises(ValueError, match="requires rebuild_flags"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                return_state=True,
            )

    def test_segmented_coo_rejects_undersized_topology_before_build(self, monkeypatch):
        """Segmented COO capacity mismatches fail before a Warp build launch."""
        positions, cell_batch, batch_ptr = _make_batch([32], [8.0], device="cpu")
        pair_offsets = torch.tensor([0, 64], dtype=torch.int32)
        tile_offsets = torch.tensor([0, 4], dtype=torch.int32)

        def fail_build(*args, **kwargs):
            del args, kwargs
            raise AssertionError("segmented COO validation must run before build")

        monkeypatch.setattr(
            batch_cluster_tile_module,
            "batch_build_cluster_tile_list",
            fail_build,
        )

        with pytest.raises(ValueError, match="neighbor_list.*capacity"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                format="coo",
                rebuild_flags=torch.ones(1, dtype=torch.bool),
                neighbor_list=torch.empty((2, 1), dtype=torch.int32),
                neighbor_list_shifts=torch.empty((64, 3), dtype=torch.int32),
                pair_offsets=pair_offsets,
                pair_counts=torch.zeros(1, dtype=torch.int32),
                tile_offsets=tile_offsets,
                tile_counts=torch.zeros(1, dtype=torch.int32),
            )

    def test_segmented_coo_false_flag_requires_prior_state(self, monkeypatch):
        """A skipped system cannot preserve implicitly allocated COO state."""
        positions, cell_batch, batch_ptr = _make_batch([32], [8.0], device="cpu")

        def fail_build(*args, **kwargs):
            del args, kwargs
            raise AssertionError("selective state validation must run before build")

        monkeypatch.setattr(
            batch_cluster_tile_module,
            "batch_build_cluster_tile_list",
            fail_build,
        )

        with pytest.raises(ValueError, match="caller-owned"):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                format="coo",
                rebuild_flags=torch.zeros(1, dtype=torch.bool),
                pair_offsets=torch.tensor([0, 64], dtype=torch.int32),
                pair_counts=torch.zeros(1, dtype=torch.int32),
                tile_offsets=torch.tensor([0, 4], dtype=torch.int32),
                tile_counts=torch.zeros(1, dtype=torch.int32),
            )

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (
                lambda kwargs: kwargs.update(
                    pair_offsets=torch.tensor([1, 64], dtype=torch.int32),
                ),
                "pair_offsets must start",
            ),
            (
                lambda kwargs: kwargs.update(
                    pair_counts=torch.tensor([65], dtype=torch.int32),
                ),
                "pair_counts must lie",
            ),
            (
                lambda kwargs: kwargs.update(
                    neighbor_list_shifts=torch.empty((63, 3), dtype=torch.int32),
                ),
                "neighbor_list_shifts",
            ),
            (
                lambda kwargs: kwargs.update(
                    rebuild_flags=torch.ones(1, dtype=torch.int32),
                ),
                "rebuild_flags",
            ),
        ],
    )
    def test_segmented_coo_rejects_malformed_state_before_build(
        self,
        monkeypatch,
        mutate,
        message,
    ):
        """Malformed segmented metadata is rejected before any Warp launch."""
        positions, cell_batch, batch_ptr = _make_batch([32], [8.0], device="cpu")
        kwargs = {
            "format": "coo",
            "rebuild_flags": torch.ones(1, dtype=torch.bool),
            "neighbor_list": torch.empty((2, 64), dtype=torch.int32),
            "neighbor_list_shifts": torch.empty((64, 3), dtype=torch.int32),
            "pair_offsets": torch.tensor([0, 64], dtype=torch.int32),
            "pair_counts": torch.zeros(1, dtype=torch.int32),
            "tile_offsets": torch.tensor([0, 4], dtype=torch.int32),
            "tile_counts": torch.zeros(1, dtype=torch.int32),
        }
        mutate(kwargs)

        def fail_build(*args, **kwargs):
            del args, kwargs
            raise AssertionError("segmented COO validation must run before build")

        monkeypatch.setattr(
            batch_cluster_tile_module,
            "batch_build_cluster_tile_list",
            fail_build,
        )
        with pytest.raises(ValueError, match=message):
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                **kwargs,
            )


def test_batch_cluster_tile_max_tiles_per_group_bypasses_sizing(monkeypatch):
    """An explicit allocation factor avoids synchronizing geometry to the host."""

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


def _batch_ptr_from_sizes(sizes: list[int]) -> torch.Tensor:
    """Create a CPU batch pointer from per-system atom counts."""
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    return torch.tensor(offsets, dtype=torch.int32)


@pytest.mark.parametrize(
    ("sizes", "max_tiles_per_group", "expected_capacity"),
    [
        ([0, 0, 0], 2, 0),
        ([0, 1, 31, 32, 33, 63, 64, 65], 2, 21),
        ([192, 32, 32], 3, 20),
        ([192, 32, 32], 8, 38),
    ],
)
def test_compact_tile_capacity_sums_per_system(
    sizes, max_tiles_per_group, expected_capacity
):
    """Compact capacity sums bounded contributions from each system."""
    batch_ptr = _batch_ptr_from_sizes(sizes)
    *_, max_tiles, _ = estimate_batch_cluster_tile_list_sizes(
        batch_ptr,
        max_tiles_per_group=max_tiles_per_group,
    )
    assert max_tiles == expected_capacity


def test_compact_tile_capacity_matches_single_system_formula():
    """One-system batching agrees with the single-system capacity formula."""
    batch_ptr = _batch_ptr_from_sizes([191])
    *_, max_tiles, _ = estimate_batch_cluster_tile_list_sizes(
        batch_ptr,
        max_tiles_per_group=3,
    )
    assert max_tiles == 6 * min(6, 3)


def test_compact_tile_capacity_many_small_systems_stays_linear():
    """One thousand one-group systems allocate one thousand tile entries."""
    batch_ptr = _batch_ptr_from_sizes([1] * 1000)
    *_, max_tiles, _ = estimate_batch_cluster_tile_list_sizes(batch_ptr)
    assert max_tiles == 1000


def test_compact_tile_capacity_matches_summed_segmented_capacities():
    """Compact and segmented sizing sum the same per-system contributions."""
    batch_ptr = _batch_ptr_from_sizes([0, 1, 33, 96])
    *_, compact_capacity, _ = estimate_batch_cluster_tile_list_sizes(
        batch_ptr,
        max_tiles_per_group=2,
    )
    segmented_capacities, *_ = estimate_batch_cluster_tile_segments(
        batch_ptr,
        max_neighbors=8,
        max_tiles_per_group=2,
    )
    assert compact_capacity == int(segmented_capacities.sum().item())


def test_compact_allocator_uses_exact_capacity_for_all_tile_arrays():
    """All three pooled tile arrays use the summed compact capacity."""
    batch_ptr = _batch_ptr_from_sizes([192, 32, 32])
    state = allocate_batch_cluster_tile_list(
        batch_ptr,
        torch.device("cpu"),
        max_tiles_per_group=3,
    )
    assert state[16].shape == (20,)
    assert state[17].shape == (20,)
    assert state[18].shape == (20,)


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
    def test_current_stream_consumes_event_gated_geometry_input(
        self, device, dtype, torch_stream_runner
    ):
        """Batched geometry writes stay on the caller's current stream."""
        source = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        positions = torch.empty_like(source)
        cell_batch = torch.eye(3, dtype=dtype, device=device).repeat(2, 1, 1) * 8.0
        batch_ptr = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
        _, snapshot, expected = torch_stream_runner(
            source,
            positions,
            lambda value: batch_cluster_tile_neighbor_list(
                value,
                1.0,
                cell_batch,
                batch_ptr,
                max_neighbors=8,
                return_vectors=True,
                return_distances=True,
            ),
        )
        matrix, counts, shifts, distances, vectors = snapshot
        (
            expected_matrix,
            expected_counts,
            expected_shifts,
            expected_distances,
            expected_vectors,
        ) = expected
        torch.testing.assert_close(counts, expected_counts)
        assert torch.any(counts > 0)
        active = torch.arange(matrix.shape[1], device=device)[None, :] < counts[:, None]
        torch.testing.assert_close(matrix[active], expected_matrix[active])
        torch.testing.assert_close(shifts[active], expected_shifts[active])
        torch.testing.assert_close(distances, expected_distances)
        torch.testing.assert_close(vectors, expected_vectors)

    def test_compact_coo_current_stream_geometry(
        self, device, dtype, torch_stream_runner
    ):
        """Batched exact COO fill and prefix packing use the caller's stream."""
        source = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        positions = torch.empty_like(source)
        cell_batch = torch.eye(3, dtype=dtype, device=device).repeat(2, 1, 1) * 8.0
        batch_ptr = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
        _, snapshot, expected = torch_stream_runner(
            source,
            positions,
            lambda value: batch_cluster_tile_neighbor_list(
                value,
                1.0,
                cell_batch,
                batch_ptr,
                format="coo",
                max_pairs=16,
                max_tiles_per_group=1,
                return_vectors=True,
                return_distances=True,
            ),
        )
        for result, reference in zip(snapshot, expected, strict=True):
            torch.testing.assert_close(result, reference)

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

    @pytest.mark.parametrize("cutoff2", [0.91, 4.0])
    @pytest.mark.parametrize(
        "compiled",
        [False, pytest.param(True, marks=pytest.mark.slow)],
    )
    @requires_vesin
    def test_default_capacity_uses_larger_dual_cutoff(
        self, device, dtype, cutoff2, compiled
    ):
        """Reversed and equal dual cutoffs retain independently referenced rows."""
        positions = torch.arange(40, dtype=dtype, device=device).reshape(-1, 1)
        positions = torch.cat(
            (positions * 0.05, torch.zeros((40, 2), dtype=dtype, device=device)),
            dim=1,
        )
        cell_batch = torch.eye(3, dtype=dtype, device=device)[None] * 10.0
        batch_ptr = torch.tensor([0, 40], dtype=torch.int32, device=device)
        if compiled:
            scratch = _scratch_kwargs(
                allocate_batch_cluster_tile_list(
                    batch_ptr,
                    torch.device(device),
                    dtype=dtype,
                    max_tiles_per_group=2,
                )
            )

            @torch.compile(fullgraph=True)
            def run(runtime_positions):
                return batch_cluster_tile_neighbor_list(
                    runtime_positions,
                    4.0,
                    cell_batch,
                    batch_ptr,
                    cutoff2=cutoff2,
                    max_tiles_per_group=2,
                    **scratch,
                )

            out = run(positions)
        else:
            out = batch_cluster_tile_neighbor_list(
                positions, 4.0, cell_batch, batch_ptr, cutoff2=cutoff2
            )
        atom_system = [0] * positions.shape[0]
        for offset, reference_cutoff in ((0, 4.0), (3, cutoff2)):
            got = _canonicalize_matrix_full(
                *out[offset : offset + 3], atom_system, positions.shape[0]
            )
            reference = _reference_pairs_per_system(
                positions, cell_batch, batch_ptr, reference_cutoff
            )
            assert_neighbor_lists_equal(got, reference)

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
        """All batched output formats report one compact tile requirement."""
        positions = torch.zeros((96, 3), dtype=dtype, device=device)
        cell_batch = torch.eye(3, dtype=dtype, device=device).repeat(2, 1, 1) * 12.0
        batch_ptr = torch.tensor([0, 32, 96], dtype=torch.int32, device=device)
        cutoff = 5.0
        required_counts = {}
        for format in ("matrix", "coo", "tile"):
            with pytest.raises(TileBufferOverflow) as caught:
                batch_cluster_tile_neighbor_list(
                    positions,
                    cutoff,
                    cell_batch,
                    batch_ptr,
                    max_neighbors=256,
                    max_tiles_per_group=1,
                    format=format,
                )
            required_counts[format] = caught.value.num_tiles
            assert caught.value.num_tiles > caught.value.max_tiles
            assert caught.value.system_index is None
        assert len(set(required_counts.values())) == 1

        segmented_positions = torch.zeros((160, 3), dtype=dtype, device=device)
        segmented_cells = torch.eye(3, dtype=dtype, device=device).repeat(2, 1, 1)
        segmented_cells *= 12.0
        segmented_ptr = torch.tensor([0, 32, 160], dtype=torch.int32, device=device)
        scratch = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                segmented_ptr,
                torch.device(device),
                dtype=dtype,
                max_tiles_per_group=1,
            )
        )
        with pytest.raises(TileBufferOverflow) as segmented:
            batch_cluster_tile_neighbor_list(
                segmented_positions,
                cutoff,
                segmented_cells,
                segmented_ptr,
                max_neighbors=256,
                rebuild_flags=torch.ones(2, dtype=torch.bool, device=device),
                neighbor_matrix=torch.empty(
                    (160, 256), dtype=torch.int32, device=device
                ),
                num_neighbors=torch.zeros(160, dtype=torch.int32, device=device),
                neighbor_matrix_shifts=torch.empty(
                    (160, 256, 3), dtype=torch.int32, device=device
                ),
                tile_offsets=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
                tile_counts=torch.zeros(2, dtype=torch.int32, device=device),
                max_tiles_per_group=1,
                **scratch,
            )
        assert segmented.value.system_index == 1
        assert segmented.value.max_tiles == 1
        assert segmented.value.num_tiles > segmented.value.max_tiles
        result = batch_cluster_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            format="tile",
            max_tiles_per_group=2,
        )
        assert int(result[0].item()) == next(iter(required_counts.values()))
        assert int(result[0].item()) <= result[1].shape[0]

    def test_compact_mixed_system_capacity_reports_actual_overflow(self, device, dtype):
        """Tighter compact sizing reports the pooled count without truncation."""
        positions = torch.zeros((256, 3), dtype=dtype, device=device)
        cell_batch = torch.eye(3, dtype=dtype, device=device).repeat(3, 1, 1) * 100.0
        batch_ptr = torch.tensor([0, 192, 224, 256], dtype=torch.int32, device=device)

        with pytest.raises(TileBufferOverflow) as caught:
            batch_cluster_tile_neighbor_list(
                positions,
                1.0,
                cell_batch,
                batch_ptr,
                format="tile",
                max_tiles_per_group=3,
            )
        assert caught.value.max_tiles == 20
        assert caught.value.num_tiles == 23
        assert caught.value.system_index is None

        result = batch_cluster_tile_neighbor_list(
            positions,
            1.0,
            cell_batch,
            batch_ptr,
            format="tile",
            max_tiles_per_group=4,
        )
        assert int(result[0].item()) == 23
        assert result[1].shape == (26,)
        assert result[2].shape == (26,)
        assert result[3].shape == (26,)

    def test_compact_capacity_is_pooled_across_systems(self, device, dtype):
        """One system may exceed its sizing term when total capacity is sufficient."""
        positions = torch.zeros((384, 3), dtype=dtype, device=device)
        for group in range(6):
            positions[192 + 32 * group : 192 + 32 * (group + 1), 0] = 10.0 * (group + 1)
        cell_batch = torch.eye(3, dtype=dtype, device=device).repeat(2, 1, 1) * 1000.0
        batch_ptr = torch.tensor([0, 192, 384], dtype=torch.int32, device=device)

        result = batch_cluster_tile_neighbor_list(
            positions,
            1.0,
            cell_batch,
            batch_ptr,
            format="tile",
            max_tiles_per_group=3,
        )

        assert result[1].shape == (36,)
        assert int(result[0].item()) == 27
        per_system = torch.bincount(result[3][:27].to(torch.long), minlength=2)
        assert per_system.tolist() == [21, 6]

    def test_oversized_caller_scratch_remains_the_launch_capacity(self, device, dtype):
        """An explicit factor does not shrink complete caller-owned scratch."""
        positions = torch.zeros((384, 3), dtype=dtype, device=device)
        cell_batch = torch.eye(3, dtype=dtype, device=device).repeat(2, 1, 1) * 100.0
        batch_ptr = torch.tensor([0, 192, 384], dtype=torch.int32, device=device)
        scratch = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr,
                torch.device(device),
                dtype=dtype,
                max_tiles_per_group=6,
            )
        )

        result = batch_cluster_tile_neighbor_list(
            positions,
            1.0,
            cell_batch,
            batch_ptr,
            format="tile",
            max_tiles_per_group=1,
            **scratch,
        )

        assert result[1] is scratch["tile_row_group"]
        assert result[1].shape == (72,)
        assert int(result[0].item()) == 42

    def test_matrix_overflow_reports_eager_fields(self, device, dtype):
        """Batched matrix overflow identifies the compact capacity and count."""
        positions = torch.zeros((128, 3), dtype=dtype, device=device)
        cell_batch = torch.eye(3, dtype=dtype, device=device).repeat(2, 1, 1) * 8.0
        batch_ptr = torch.tensor([0, 64, 128], dtype=torch.int32, device=device)

        with pytest.raises(NeighborOverflowError) as caught:
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                max_neighbors=1,
            )
        assert caught.value.max_neighbors == 1
        assert caught.value.num_neighbors > 1
        assert caught.value.system_index is None

    def test_compact_coo_overflow_reports_eager_fields(self, device, dtype):
        """Batched compact COO overflow reports its fixed pair capacity."""
        positions = torch.zeros((128, 3), dtype=dtype, device=device)
        cell_batch = torch.eye(3, dtype=dtype, device=device).repeat(2, 1, 1) * 8.0
        batch_ptr = torch.tensor([0, 64, 128], dtype=torch.int32, device=device)

        with pytest.raises(NeighborOverflowError) as caught:
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                max_neighbors=64,
                max_pairs=1,
                format="coo",
            )
        assert caught.value.max_neighbors == 1
        assert caught.value.num_neighbors > 1
        assert caught.value.system_index is None

    def test_segmented_coo_overflow_reports_first_system(self, device, dtype):
        """Batched segmented COO overflow reports the first overflowing system."""
        positions = torch.zeros((128, 3), dtype=dtype, device=device)
        cell_batch = torch.eye(3, dtype=dtype, device=device).repeat(2, 1, 1) * 8.0
        batch_ptr = torch.tensor([0, 64, 128], dtype=torch.int32, device=device)

        with pytest.raises(NeighborOverflowError) as caught:
            batch_cluster_tile_neighbor_list(
                positions,
                2.0,
                cell_batch,
                batch_ptr,
                max_neighbors=64,
                format="coo",
                rebuild_flags=torch.ones(2, dtype=torch.bool, device=device),
                neighbor_list=torch.empty((2, 2), dtype=torch.int32, device=device),
                neighbor_list_shifts=torch.empty(
                    (2, 3), dtype=torch.int32, device=device
                ),
                pair_offsets=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
                pair_counts=torch.zeros(2, dtype=torch.int32, device=device),
            )
        assert caught.value.max_neighbors == 1
        assert caught.value.num_neighbors > 1
        assert caught.value.system_index == 0


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

    def test_selective_matrix_return_state_appends_batched_state(self, device, dtype):
        """Selective matrix calls can return reusable batched tile state."""
        positions, cell_batch, batch_ptr = _make_batch(
            [64], [8.0], device=device, dtype=dtype, seed=43
        )

        out = batch_cluster_tile_neighbor_list(
            positions,
            2.0,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            rebuild_flags=torch.ones(1, dtype=torch.bool, device=device),
            return_state=True,
        )

        assert len(out) == 9
        neighbor_matrix, num_neighbors, shifts, *state = out
        assert neighbor_matrix.shape == (64, 64)
        assert num_neighbors.shape == (64,)
        assert shifts.shape == (64, 64, 3)
        tile_offsets, tile_counts, num_tiles, row, col, system = state
        assert tile_offsets.shape == (2,)
        assert tile_counts.shape == (1,)
        assert num_tiles.shape == (1,)
        assert row.ndim == col.ndim == system.ndim == 1

    def test_selective_matrix_default_arity_is_unchanged(self, device, dtype):
        """Selective matrix calls keep the three-array default return."""
        positions, cell_batch, batch_ptr = _make_batch(
            [64], [8.0], device=device, dtype=dtype, seed=44
        )

        out = batch_cluster_tile_neighbor_list(
            positions,
            2.0,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            rebuild_flags=torch.ones(1, dtype=torch.bool, device=device),
        )

        assert len(out) == 3

    @pytest.mark.parametrize("rebuild_flag", [False, True])
    def test_returned_matrix_state_can_be_passed_to_next_call(
        self, device, dtype, rebuild_flag
    ):
        """The public state suffix is sufficient for the next selective call."""
        positions, cell_batch, batch_ptr = _make_batch(
            [64], [8.0], device=device, dtype=dtype, seed=48
        )
        first = batch_cluster_tile_neighbor_list(
            positions,
            2.0,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            rebuild_flags=torch.ones(1, dtype=torch.bool, device=device),
            return_state=True,
        )
        matrix, counts, shifts, *state = first
        tile_offsets, tile_counts, num_tiles, row, col, system = state

        second = batch_cluster_tile_neighbor_list(
            positions,
            2.0,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            rebuild_flags=torch.tensor([rebuild_flag], dtype=torch.bool, device=device),
            neighbor_matrix=matrix,
            num_neighbors=counts,
            neighbor_matrix_shifts=shifts,
            tile_offsets=tile_offsets,
            tile_counts=tile_counts,
            num_tiles=num_tiles,
            tile_row_group=row,
            tile_col_group=col,
            tile_system=system,
            return_state=True,
        )

        for returned, supplied in zip(second[-6:], state):
            assert returned is supplied

    def test_return_state_mixed_flags_preserve_unflagged_system(self, device, dtype):
        """Mixed flags keep unflagged rows while returning reusable state."""
        positions, cell_batch, batch_ptr = _make_batch(
            [64, 64], [8.0, 8.0], device=device, dtype=dtype, seed=49
        )
        first = batch_cluster_tile_neighbor_list(
            positions,
            2.0,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            rebuild_flags=torch.ones(2, dtype=torch.bool, device=device),
            return_state=True,
        )
        matrix, counts, shifts, *state = first
        system0_matrix = matrix[:64].clone()
        system0_counts = counts[:64].clone()
        system0_shifts = shifts[:64].clone()
        system0_tile_count = state[1][0].clone()
        moved = positions.clone()
        moved[:64] = torch.roll(moved[:64], shifts=1, dims=0)
        moved[64:, 0] += 0.2

        second = batch_cluster_tile_neighbor_list(
            moved,
            2.0,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            rebuild_flags=torch.tensor([False, True], dtype=torch.bool, device=device),
            neighbor_matrix=matrix,
            num_neighbors=counts,
            neighbor_matrix_shifts=shifts,
            tile_offsets=state[0],
            tile_counts=state[1],
            num_tiles=state[2],
            tile_row_group=state[3],
            tile_col_group=state[4],
            tile_system=state[5],
            return_state=True,
        )

        assert len(second) == 9
        torch.testing.assert_close(second[0][:64], system0_matrix)
        torch.testing.assert_close(second[1][:64], system0_counts)
        torch.testing.assert_close(second[2][:64], system0_shifts)
        torch.testing.assert_close(second[4][0], system0_tile_count)
        for returned, supplied in zip(second[-6:], state):
            assert returned is supplied

    @pytest.mark.parametrize("rebuild_flag", [False, True])
    def test_selective_matrix_state_aliases_caller_buffers(
        self, device, dtype, rebuild_flag
    ):
        """Returned matrix state aliases explicitly supplied scratch buffers."""
        positions, cell_batch, batch_ptr = _make_batch(
            [64], [8.0], device=device, dtype=dtype, seed=45
        )
        scratch_kwargs = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr, torch.device(device), dtype=dtype
            )
        )
        _tile_caps, tile_offsets, _pair_caps, _pair_offsets = (
            estimate_batch_cluster_tile_segments(batch_ptr, max_neighbors=64)
        )
        tile_counts = torch.zeros(1, dtype=torch.int32, device=device)
        matrix = torch.full((64, 64), 64, dtype=torch.int32, device=device)
        counts = torch.zeros(64, dtype=torch.int32, device=device)
        shifts = torch.zeros((64, 64, 3), dtype=torch.int32, device=device)

        out = batch_cluster_tile_neighbor_list(
            positions,
            2.0,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            rebuild_flags=torch.tensor([rebuild_flag], dtype=torch.bool, device=device),
            neighbor_matrix=matrix,
            num_neighbors=counts,
            neighbor_matrix_shifts=shifts,
            tile_offsets=tile_offsets,
            tile_counts=tile_counts,
            return_state=True,
            **scratch_kwargs,
        )

        assert len(out) == 9
        for returned, supplied in zip(
            out[-6:],
            (
                tile_offsets,
                tile_counts,
                scratch_kwargs["num_tiles"],
                scratch_kwargs["tile_row_group"],
                scratch_kwargs["tile_col_group"],
                scratch_kwargs["tile_system"],
            ),
        ):
            assert returned is supplied

    @pytest.mark.parametrize("rebuild_flag", [False, True])
    def test_selective_segmented_coo_return_state(self, device, dtype, rebuild_flag):
        """Segmented COO appends six caller-owned batched state buffers."""
        positions, cell_batch, batch_ptr = _make_batch(
            [64], [8.0], device=device, dtype=dtype, seed=46
        )
        scratch_kwargs = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr, torch.device(device), dtype=dtype
            )
        )
        _tile_caps, tile_offsets, _pair_caps, pair_offsets = (
            estimate_batch_cluster_tile_segments(batch_ptr, max_neighbors=64)
        )
        max_pairs = int(pair_offsets[-1].item())
        tile_counts = torch.zeros(1, dtype=torch.int32, device=device)
        pair_counts = torch.zeros(1, dtype=torch.int32, device=device)

        out = batch_cluster_tile_neighbor_list(
            positions,
            2.0,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            format="coo",
            rebuild_flags=torch.tensor([rebuild_flag], dtype=torch.bool, device=device),
            neighbor_list=torch.empty((2, max_pairs), dtype=torch.int32, device=device),
            neighbor_list_shifts=torch.empty(
                (max_pairs, 3), dtype=torch.int32, device=device
            ),
            pair_counter=torch.zeros(1, dtype=torch.int32, device=device),
            tile_offsets=tile_offsets,
            tile_counts=tile_counts,
            pair_offsets=pair_offsets,
            pair_counts=pair_counts,
            return_state=True,
            **scratch_kwargs,
        )

        assert len(out) == 10
        assert out[1] is pair_offsets
        assert out[2] is pair_counts
        for returned, supplied in zip(
            out[-6:],
            (
                tile_offsets,
                tile_counts,
                scratch_kwargs["num_tiles"],
                scratch_kwargs["tile_row_group"],
                scratch_kwargs["tile_col_group"],
                scratch_kwargs["tile_system"],
            ),
        ):
            assert returned is supplied

    def test_selective_segmented_coo_all_true_bootstrap_allocates_state(
        self, device, dtype
    ):
        """An all-true batched COO call allocates persistent state."""
        positions, cell_batch, batch_ptr = _make_batch(
            [64], [8.0], device=device, dtype=dtype, seed=46
        )

        result = batch_cluster_tile_neighbor_list(
            positions,
            2.0,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            format="coo",
            rebuild_flags=torch.ones(1, dtype=torch.bool, device=device),
            return_state=True,
        )

        assert len(result) == 10
        neighbor_list, pair_offsets, pair_counts, shifts, *state = result
        assert neighbor_list.shape[0] == 2
        assert shifts.shape == (neighbor_list.shape[1], 3)
        assert pair_offsets.shape == (2,)
        assert pair_counts.shape == (1,)

        reused = batch_cluster_tile_neighbor_list(
            positions,
            2.0,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            format="coo",
            rebuild_flags=torch.zeros(1, dtype=torch.bool, device=device),
            return_state=True,
            neighbor_list=neighbor_list,
            neighbor_list_shifts=shifts,
            pair_offsets=pair_offsets,
            pair_counts=pair_counts,
            tile_offsets=state[0],
            tile_counts=state[1],
            num_tiles=state[2],
            tile_row_group=state[3],
            tile_col_group=state[4],
            tile_system=state[5],
        )

        assert all(reused[index] is result[index] for index in range(10))

    def test_selective_dual_cutoff_return_state(self, device, dtype):
        """Dual-cutoff matrix output appends state after both matrix triples."""
        positions, cell_batch, batch_ptr = _make_batch(
            [64], [8.0], device=device, dtype=dtype, seed=47
        )
        scratch_kwargs = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr, torch.device(device), dtype=dtype
            )
        )
        _tile_caps, tile_offsets, _pair_caps, _pair_offsets = (
            estimate_batch_cluster_tile_segments(batch_ptr, max_neighbors=64)
        )
        tile_counts = torch.zeros(1, dtype=torch.int32, device=device)
        matrix1 = torch.full((64, 64), 64, dtype=torch.int32, device=device)
        counts1 = torch.zeros(64, dtype=torch.int32, device=device)
        shifts1 = torch.zeros((64, 64, 3), dtype=torch.int32, device=device)
        matrix2 = matrix1.clone()
        counts2 = counts1.clone()
        shifts2 = shifts1.clone()

        out = batch_cluster_tile_neighbor_list(
            positions,
            1.5,
            cell_batch,
            batch_ptr,
            cutoff2=2.0,
            max_neighbors=64,
            rebuild_flags=torch.ones(1, dtype=torch.bool, device=device),
            neighbor_matrix=matrix1,
            num_neighbors=counts1,
            neighbor_matrix_shifts=shifts1,
            neighbor_matrix2=matrix2,
            num_neighbors2=counts2,
            neighbor_matrix_shifts2=shifts2,
            tile_offsets=tile_offsets,
            tile_counts=tile_counts,
            return_state=True,
            **scratch_kwargs,
        )

        assert len(out) == 12
        assert out[6] is tile_offsets
        assert out[7] is tile_counts
        assert out[8] is scratch_kwargs["num_tiles"]

    def test_format_coo_returns_three_tuple(self, device, dtype):
        sizes = [33, 67, 19]
        positions, cell_batch, batch_ptr = _make_batch(
            sizes,
            [10.0, 8.0, 12.0],
            device=device,
            dtype=dtype,
            seed=11,
        )
        cutoff = 3.0
        # matrix-format reference for pair count
        nm, nn, matrix_shifts = batch_cluster_tile_neighbor_list(
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
        assert _compact_coo_pair_sets(nl, neighbor_ptr, nls) == _matrix_pair_sets(
            nm, nn, matrix_shifts
        )
        atom_system = torch.bucketize(
            torch.arange(N, device=device, dtype=torch.int32),
            batch_ptr[1:],
            right=True,
        )
        assert torch.equal(atom_system[nl[0].long()], atom_system[nl[1].long()])

    def test_compact_coo_mixed_dense_batch_rows(self, device, dtype):
        """Empty, singleton, and unequal dense systems keep exact CSR rows."""
        sizes = [0, 1, 33, 65]
        batch_ptr = torch.tensor([0, 0, 1, 34, 99], dtype=torch.int32, device=device)
        positions = torch.zeros((batch_ptr[-1].item(), 3), dtype=dtype, device=device)
        cell_batch = torch.eye(3, dtype=dtype, device=device).repeat(4, 1, 1) * 8.0
        expected_counts = torch.cat(
            [
                torch.full((size,), size - 1, dtype=torch.int32, device=device)
                for size in sizes
                if size > 0
            ]
        )
        expected_pairs = int(expected_counts.sum().item())

        pairs, pointer, shifts = batch_cluster_tile_neighbor_list(
            positions,
            1.0,
            cell_batch,
            batch_ptr,
            format="coo",
            max_pairs=expected_pairs,
            max_tiles_per_group=3,
        )

        assert pairs.shape == (2, expected_pairs)
        torch.testing.assert_close(pointer[1:] - pointer[:-1], expected_counts)
        rows = _compact_coo_pair_sets(pairs, pointer, shifts)
        assert [len(row) for row in rows] == expected_counts.cpu().tolist()
        atom_system = torch.bucketize(
            torch.arange(positions.shape[0], dtype=torch.int32, device=device),
            batch_ptr[1:],
            right=True,
        )
        assert torch.equal(atom_system[pairs[0].long()], atom_system[pairs[1].long()])

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
    def test_compact_per_system_capacity_matches_eager_and_fullgraph(
        self, device, dtype
    ):
        """The exact pooled allocation supports equal eager and compiled tiles."""
        positions = torch.zeros((256, 3), dtype=dtype, device=device)
        cell_batch = torch.eye(3, dtype=dtype, device=device).repeat(3, 1, 1) * 100.0
        batch_ptr = torch.tensor([0, 192, 224, 256], dtype=torch.int32, device=device)
        compiled_scratch = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr,
                torch.device(device),
                dtype=dtype,
                max_tiles_per_group=4,
            )
        )

        @torch.compile(fullgraph=True)
        def run(runtime_positions):
            return batch_cluster_tile_neighbor_list(
                runtime_positions,
                1.0,
                cell_batch,
                batch_ptr,
                format="tile",
                max_tiles_per_group=4,
                **compiled_scratch,
            )

        eager = batch_cluster_tile_neighbor_list(
            positions,
            1.0,
            cell_batch,
            batch_ptr,
            format="tile",
            max_tiles_per_group=4,
        )
        compiled = run(positions)
        expected_count = int(eager[0].item())
        actual_count = int(compiled[0].item())
        assert expected_count == actual_count == 23
        assert eager[1].shape == compiled[1].shape == (26,)
        expected_tiles = set(
            zip(
                eager[1][:expected_count].tolist(),
                eager[2][:expected_count].tolist(),
                eager[3][:expected_count].tolist(),
                strict=True,
            )
        )
        actual_tiles = set(
            zip(
                compiled[1][:actual_count].tolist(),
                compiled[2][:actual_count].tolist(),
                compiled[3][:actual_count].tolist(),
                strict=True,
            )
        )
        assert actual_tiles == expected_tiles

    @pytest.mark.slow
    def test_batch_cluster_tile_neighbor_list_compile(self, device, dtype):
        """Prepared batch matrix topology supports fullgraph compilation."""
        sizes = [64, 96]
        positions, cell_batch, batch_ptr = _make_batch(
            sizes,
            [10.0, 8.0],
            device=device,
            dtype=dtype,
            seed=11,
        )
        cutoff = 2.5

        eager_scratch = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr, torch.device(device), dtype=dtype, max_tiles_per_group=4
            )
        )
        compiled_scratch = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr, torch.device(device), dtype=dtype, max_tiles_per_group=4
            )
        )
        nm_uncompiled, nn_uncompiled, _nms_uncompiled = (
            batch_cluster_tile_neighbor_list(
                positions,
                cutoff,
                cell_batch,
                batch_ptr,
                max_neighbors=64,
                **eager_scratch,
            )
        )

        @torch.compile(fullgraph=True)
        def compiled_batch_cluster_tile_neighbor_list(
            positions, cutoff, cell_batch, batch_ptr
        ):
            return batch_cluster_tile_neighbor_list(
                positions,
                cutoff,
                cell_batch,
                batch_ptr,
                max_neighbors=64,
                **compiled_scratch,
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

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", ["tile", "dual"])
    def test_batch_cluster_tile_other_formats_fullgraph(self, device, dtype, mode):
        """Prepared tile and dual-matrix batch formats support fullgraph."""
        positions, cell_batch, batch_ptr = _make_batch(
            [33, 67, 19], [10.0, 8.0, 12.0], device=device, dtype=dtype, seed=11
        )
        eager_scratch = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr, torch.device(device), dtype=dtype, max_tiles_per_group=4
            )
        )
        compiled_scratch = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr, torch.device(device), dtype=dtype, max_tiles_per_group=4
            )
        )
        kwargs = (
            {"format": "tile"}
            if mode == "tile"
            else {"max_neighbors": 64, "cutoff2": 2.5}
        )
        eager = batch_cluster_tile_neighbor_list(
            positions, 2.0, cell_batch, batch_ptr, **kwargs, **eager_scratch
        )

        @torch.compile(fullgraph=True)
        def run(runtime_positions):
            return batch_cluster_tile_neighbor_list(
                runtime_positions,
                2.0,
                cell_batch,
                batch_ptr,
                **kwargs,
                **compiled_scratch,
            )

        compiled = run(positions)
        if mode == "tile":
            assert torch.equal(eager[0], compiled[0])
            active = int(eager[0].item())
            assert sorted(
                zip(*[x[:active].cpu().tolist() for x in eager[1:4]])
            ) == sorted(zip(*[x[:active].cpu().tolist() for x in compiled[1:4]]))
            for expected, actual in zip(eager[4:], compiled[4:]):
                assert torch.equal(expected, actual)
        else:
            for start in (0, 3):
                assert torch.equal(eager[start + 1], compiled[start + 1])
                assert _matrix_pair_sets(
                    *eager[start : start + 3]
                ) == _matrix_pair_sets(*compiled[start : start + 3])

    @pytest.mark.slow
    def test_compact_coo_fullgraph_dynamic_unequal_batch(self, device, dtype):
        """One fullgraph call handles changing exact batch lengths."""
        batch_ptr = torch.tensor([0, 3, 7, 9], dtype=torch.int32, device=device)
        cell_batch = torch.eye(3, dtype=dtype, device=device).repeat(3, 1, 1) * 20.0
        close = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        empty = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        scratch = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr, torch.device(device), dtype=dtype, max_tiles_per_group=1
            )
        )

        @torch.compile(fullgraph=True)
        def run(runtime_positions):
            return batch_cluster_tile_neighbor_list(
                runtime_positions,
                1.0,
                cell_batch,
                batch_ptr,
                format="coo",
                max_neighbors=8,
                max_pairs=72,
                **scratch,
            )

        atom_system = torch.bucketize(
            torch.arange(close.shape[0], dtype=torch.int32, device=device),
            batch_ptr[1:],
            right=True,
        )
        counts = []
        for positions in (close, empty):
            eager = batch_cluster_tile_neighbor_list(
                positions,
                1.0,
                cell_batch,
                batch_ptr,
                format="coo",
                max_neighbors=8,
                max_pairs=72,
                max_tiles_per_group=1,
            )
            compiled = run(positions)
            assert _compact_coo_pair_sets(*compiled) == _compact_coo_pair_sets(*eager)
            assert torch.equal(
                atom_system[compiled[0][0].long()],
                atom_system[compiled[0][1].long()],
            )
            counts.append(compiled[0].shape[1])
        assert counts[0] > 0
        assert counts[1] == 0

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("return_distances", "return_vectors"),
        [(True, False), (False, True), (True, True)],
    )
    def test_compact_coo_geometry_fullgraph_return_contract(
        self, device, dtype, return_distances, return_vectors
    ):
        """Batched exact COO returns exact geometry without output buffers."""
        batch_ptr = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        empty = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        cell_batch = torch.stack(
            (
                torch.eye(3, dtype=dtype, device=device) * 8.0,
                torch.diag(torch.tensor([9.0, 8.0, 7.0], device=device)),
            )
        )
        scratch = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr,
                torch.device(device),
                dtype=dtype,
                max_tiles_per_group=1,
            )
        )

        def call(runtime_positions):
            return batch_cluster_tile_neighbor_list(
                runtime_positions,
                1.0,
                cell_batch,
                batch_ptr,
                format="coo",
                max_pairs=16,
                return_distances=return_distances,
                return_vectors=return_vectors,
                **scratch,
            )

        compiled_call = torch.compile(call, fullgraph=True)
        atom_system = torch.bucketize(
            torch.arange(positions.shape[0], dtype=torch.int32, device=device),
            batch_ptr[1:],
            right=True,
        )
        for runtime_positions in (positions, empty):
            for result in (
                call(runtime_positions),
                compiled_call(runtime_positions),
            ):
                pairs, pointer, shifts, *geometry = result
                _compact_coo_pair_sets(pairs, pointer, shifts)
                pair_system = atom_system[pairs[0].long()]
                expected_vectors = (
                    runtime_positions[pairs[1].long()]
                    - runtime_positions[pairs[0].long()]
                )
                expected_vectors = expected_vectors + torch.einsum(
                    "pa,pab->pb",
                    shifts.to(dtype),
                    cell_batch[pair_system.long()],
                )
                expected_length = 3 + int(return_distances) + int(return_vectors)
                assert len(result) == expected_length
                geometry_index = 0
                if return_distances:
                    distances = geometry[geometry_index]
                    geometry_index += 1
                    assert distances.shape == (pairs.shape[1],)
                    torch.testing.assert_close(distances, expected_vectors.norm(dim=-1))
                if return_vectors:
                    vectors = geometry[geometry_index]
                    assert vectors.shape == (pairs.shape[1], 3)
                    torch.testing.assert_close(vectors, expected_vectors)

    @pytest.mark.slow
    def test_compact_coo_fullgraph_empty_geometry(self, device, dtype):
        """Batched fullgraph COO returns exact empty geometry."""
        positions = torch.zeros((1, 3), dtype=dtype, device=device)
        cell_batch = torch.eye(3, dtype=dtype, device=device).repeat(2, 1, 1) * 8.0
        batch_ptr = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
        scratch = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr,
                torch.device(device),
                dtype=dtype,
                max_tiles_per_group=1,
            )
        )

        @torch.compile(fullgraph=True)
        def run(runtime_positions):
            return batch_cluster_tile_neighbor_list(
                runtime_positions,
                1.0,
                cell_batch,
                batch_ptr,
                format="coo",
                max_pairs=8,
                return_distances=True,
                return_vectors=True,
                **scratch,
            )

        pairs, pointer, shifts, distances, vectors = run(positions)
        assert pairs.shape == (2, 0)
        assert pointer.tolist() == [0, 0]
        assert shifts.shape == (0, 3)
        assert distances.shape == (0,)
        assert vectors.shape == (0, 3)

    @pytest.mark.slow
    @pytest.mark.parametrize("partial", [False, True])
    def test_batch_cluster_tile_fullgraph_requires_complete_scratch(
        self, device, dtype, partial
    ):
        """Compiled convenience calls require the complete allocator state."""
        positions, cell_batch, batch_ptr = _make_batch(
            [2, 3], [8.0, 8.0], device=device, dtype=dtype, seed=52
        )
        kwargs = {}
        if partial:
            kwargs["sorted_atom_index"] = allocate_batch_cluster_tile_list(
                batch_ptr, torch.device(device), dtype=dtype
            )[0]

        @torch.compile(fullgraph=True)
        def run(runtime_positions):
            return batch_cluster_tile_neighbor_list(
                runtime_positions, 1.0, cell_batch, batch_ptr, max_neighbors=8, **kwargs
            )

        with pytest.raises(
            RuntimeError,
            match=(
                "compiled batch_cluster_tile_neighbor_list requires complete scratch "
                "from allocate_batch_cluster_tile_list"
            ),
        ):
            run(positions)

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    def test_batch_cluster_tile_fullgraph_bad_batch_total_is_isolated(self):
        """Compiled batch-total assertions fail in an isolated CUDA process."""
        script = textwrap.dedent(
            """
            import torch
            from nvalchemiops.torch.neighbors.batch_cluster_tile import (
                allocate_batch_cluster_tile_list,
                batch_cluster_tile_neighbor_list,
            )

            device = torch.device("cuda")
            positions = torch.zeros((9, 3), dtype=torch.float32, device=device)
            cell_batch = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3) * 20.0
            allocation_ptr = torch.tensor([0, 9], dtype=torch.int32, device=device)
            bad_batch_ptr = torch.tensor([0, 8], dtype=torch.int32, device=device)
            names = (
                "sorted_atom_index", "sort_inv", "sorted_pos_x", "sorted_pos_y",
                "sorted_pos_z", "batch_idx_sorted", "batch_ptr_padded", "group_system",
                "group_ptr", "group_ctr_x", "group_ctr_y", "group_ctr_z", "group_ext_x",
                "group_ext_y", "group_ext_z", "num_tiles", "tile_row_group",
                "tile_col_group", "tile_system",
            )
            scratch = dict(zip(names, allocate_batch_cluster_tile_list(
                allocation_ptr, device, dtype=torch.float32, max_tiles_per_group=1
            )))

            @torch.compile(fullgraph=True)
            def run(values):
                return batch_cluster_tile_neighbor_list(
                    values, 1.0, cell_batch, bad_batch_ptr, max_neighbors=8, **scratch
                )

            run(positions)
            torch.cuda.synchronize()
            """
        )
        with tempfile.TemporaryDirectory() as cache_dir:
            env = os.environ.copy()
            env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_dir, "inductor")
            env["WARP_CACHE_PATH"] = os.path.join(cache_dir, "warp")
            result = subprocess.run(  # noqa: S603 - test isolates CUDA assertions
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        assert result.returncode != 0
        assert "batch_ptr[-1] must equal positions.shape[0]" in (
            result.stdout + result.stderr
        )

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    def test_batch_cluster_tile_fullgraph_bad_padded_layout_is_isolated(self):
        """Compiled padded-layout assertions fail in an isolated CUDA process."""
        script = textwrap.dedent(
            """
            import torch
            from nvalchemiops.torch.neighbors.batch_cluster_tile import (
                allocate_batch_cluster_tile_list,
                batch_cluster_tile_neighbor_list,
            )

            device = torch.device("cuda")
            positions = torch.zeros((64, 3), dtype=torch.float32, device=device)
            cell_batch = torch.eye(3, dtype=torch.float32, device=device).repeat(2, 1, 1) * 20.0
            allocation_ptr = torch.tensor([0, 33, 64], dtype=torch.int32, device=device)
            batch_ptr = torch.tensor([0, 32, 64], dtype=torch.int32, device=device)
            names = (
                "sorted_atom_index", "sort_inv", "sorted_pos_x", "sorted_pos_y",
                "sorted_pos_z", "batch_idx_sorted", "batch_ptr_padded", "group_system",
                "group_ptr", "group_ctr_x", "group_ctr_y", "group_ctr_z", "group_ext_x",
                "group_ext_y", "group_ext_z", "num_tiles", "tile_row_group",
                "tile_col_group", "tile_system",
            )
            scratch = dict(zip(names, allocate_batch_cluster_tile_list(
                allocation_ptr, device, dtype=torch.float32, max_tiles_per_group=1
            )))

            @torch.compile(fullgraph=True)
            def run(values):
                return batch_cluster_tile_neighbor_list(
                    values, 1.0, cell_batch, batch_ptr, max_neighbors=8, **scratch
                )

            run(positions)
            torch.cuda.synchronize()
            """
        )
        with tempfile.TemporaryDirectory() as cache_dir:
            env = os.environ.copy()
            env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_dir, "inductor")
            env["WARP_CACHE_PATH"] = os.path.join(cache_dir, "warp")
            result = subprocess.run(  # noqa: S603 - test isolates CUDA assertions
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        assert result.returncode != 0
        assert "scratch padded atom length must match the current batch_ptr" in (
            result.stdout + result.stderr
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("dual", [False, True])
    def test_batch_cluster_tile_selective_matrix_fullgraph_preserves_false_rows(
        self, device, dtype, dual
    ):
        """Compiled selective batch matrix updates only true systems."""
        batch_ptr = torch.tensor([0, 2, 6, 9], dtype=torch.int32, device=device)
        cell_batch = torch.eye(3, dtype=dtype, device=device).repeat(3, 1, 1) * 20.0
        initial = torch.zeros((9, 3), dtype=dtype, device=device)
        moved = initial.clone()
        moved[2:6, 0] = torch.arange(4, dtype=dtype, device=device) * 3.0
        scratch = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr, torch.device(device), dtype=dtype, max_tiles_per_group=1
            )
        )
        _tile_caps, tile_offsets, _pair_caps, _pair_offsets = (
            estimate_batch_cluster_tile_segments(
                batch_ptr, max_neighbors=8, max_tiles_per_group=1
            )
        )
        tile_counts = torch.zeros(3, dtype=torch.int32, device=device)
        primary = (
            torch.full((9, 8), 9, dtype=torch.int32, device=device),
            torch.zeros(9, dtype=torch.int32, device=device),
            torch.zeros((9, 8, 3), dtype=torch.int32, device=device),
        )
        vectors = torch.full((9, 8, 3), -7.0, dtype=dtype, device=device)
        distances = torch.full((9, 8), -7.0, dtype=dtype, device=device)
        secondary = tuple(tensor.clone() for tensor in primary)
        kwargs = {
            "max_neighbors": 8,
            "neighbor_matrix": primary[0],
            "num_neighbors": primary[1],
            "neighbor_matrix_shifts": primary[2],
            "tile_offsets": tile_offsets,
            "tile_counts": tile_counts,
            "return_state": True,
            **scratch,
        }
        if not dual:
            kwargs.update(
                return_vectors=True,
                return_distances=True,
                neighbor_vectors=vectors,
                neighbor_distances=distances,
            )
        if dual:
            kwargs.update(
                cutoff2=2.0,
                neighbor_matrix2=secondary[0],
                num_neighbors2=secondary[1],
                neighbor_matrix_shifts2=secondary[2],
            )
        batch_cluster_tile_neighbor_list(
            initial,
            1.0,
            cell_batch,
            batch_ptr,
            rebuild_flags=torch.ones(3, dtype=torch.bool, device=device),
            **kwargs,
        )
        false_rows = torch.tensor([0, 1, 6, 7, 8], device=device)
        snapshots = [tuple(tensor[false_rows].clone() for tensor in primary)]
        geometry_snapshot = (vectors[false_rows].clone(), distances[false_rows].clone())
        if dual:
            snapshots.append(tuple(tensor[false_rows].clone() for tensor in secondary))

        @torch.compile(fullgraph=True)
        def run(runtime_positions, runtime_flags):
            return batch_cluster_tile_neighbor_list(
                runtime_positions,
                1.0,
                cell_batch,
                batch_ptr,
                rebuild_flags=runtime_flags,
                **kwargs,
            )

        result = run(moved, torch.tensor([False, True, False], device=device))
        output_groups = (primary, secondary) if dual else (primary,)
        reference = batch_cluster_tile_neighbor_list(
            moved,
            1.0,
            cell_batch,
            batch_ptr,
            max_neighbors=8,
            cutoff2=2.0 if dual else None,
            max_tiles_per_group=1,
        )
        for group, snapshot in zip(output_groups, snapshots):
            for tensor, expected in zip(group, snapshot):
                assert torch.equal(tensor[false_rows], expected)
        if not dual:
            assert torch.equal(vectors[false_rows], geometry_snapshot[0])
            assert torch.equal(distances[false_rows], geometry_snapshot[1])
            assert all(result[index] is tensor for index, tensor in enumerate(primary))
        for group_index, group in enumerate(output_groups):
            expected = reference[group_index * 3 : group_index * 3 + 3]
            assert torch.equal(group[1][2:6], expected[1][2:6])
            assert _matrix_pair_sets(*group)[2:6] == _matrix_pair_sets(*expected)[2:6]
        if not dual:
            active = torch.arange(8, device=device)[None, :] < primary[1][:, None]
            safe_neighbors = torch.where(active, primary[0], 0).to(torch.long)
            batch_idx = torch.repeat_interleave(
                torch.arange(3, device=device), batch_ptr[1:] - batch_ptr[:-1]
            )
            expected_vectors = moved[safe_neighbors] - moved[:, None]
            expected_vectors = expected_vectors + torch.einsum(
                "nma,nab->nmb", primary[2].to(dtype), cell_batch[batch_idx]
            )
            expected_vectors = torch.where(
                active[..., None], expected_vectors, torch.zeros_like(expected_vectors)
            )
            torch.testing.assert_close(vectors[2:6], expected_vectors[2:6])
            torch.testing.assert_close(
                distances[2:6], expected_vectors.norm(dim=-1)[2:6]
            )
        state_start = 6 if dual else 3
        state = (
            tile_offsets,
            tile_counts,
            scratch["num_tiles"],
            scratch["tile_row_group"],
            scratch["tile_col_group"],
            scratch["tile_system"],
        )
        assert all(result[state_start + i] is value for i, value in enumerate(state))


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

    @pytest.mark.slow
    def test_direct_matrix_geometry_query_fullgraph(self, device, dtype):
        """Direct batched query declares Warp geometry-buffer mutation."""
        positions, cell_batch, batch_ptr = _make_batch(
            [32, 32], [10.0, 10.0], device=device, dtype=dtype, seed=2
        )
        N, cutoff = positions.shape[0], 2.5
        state = allocate_batch_cluster_tile_list(
            batch_ptr, torch.device(device), dtype=dtype
        )
        matrix = torch.full((N, 64), N, dtype=torch.int32, device=device)
        counts = torch.zeros(N, dtype=torch.int32, device=device)
        shifts = torch.zeros((N, 64, 3), dtype=torch.int32, device=device)
        vectors = torch.full((N, 64, 3), -7.0, dtype=dtype, device=device)
        distances = torch.full((N, 64), -7.0, dtype=dtype, device=device)

        @torch.compile(fullgraph=True)
        def run(runtime_positions):
            batch_build_cluster_tile_list(
                runtime_positions, cutoff, cell_batch, batch_ptr, *state
            )
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
                matrix,
                counts,
                shifts,
                return_vectors=True,
                return_distances=True,
                neighbor_vectors=vectors,
                neighbor_distances=distances,
            )

        run(positions)
        active = torch.arange(64, device=device)[None, :] < counts[:, None]
        assert torch.equal(distances[~active], torch.zeros_like(distances[~active]))
        assert torch.equal(vectors[~active], torch.zeros_like(vectors[~active]))

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

    @pytest.mark.parametrize("kind", ["dtype", "device", "rank", "length"])
    def test_build_rejects_malformed_scratch_before_mutation(self, device, dtype, kind):
        """Malformed allocator scratch is rejected without changing any buffer."""
        positions, cell_batch, batch_ptr = _make_batch(
            [32], [10.0], device=device, dtype=dtype, seed=25
        )
        state = list(
            allocate_batch_cluster_tile_list(
                batch_ptr, torch.device(device), dtype=dtype
            )
        )
        for tensor in state:
            tensor.fill_(7)
        if kind == "dtype":
            state[0] = state[0].to(torch.float32)
        elif kind == "device":
            state[0] = state[0].cpu()
        elif kind == "rank":
            state[2] = state[2].reshape(-1, 1)
        else:
            state[1] = state[1][:-1]
        snapshots = [tensor.clone() for tensor in state]
        with pytest.raises(ValueError, match="must"):
            batch_build_cluster_tile_list(positions, 2.5, cell_batch, batch_ptr, *state)
        assert all(
            torch.equal(snapshot, tensor)
            for snapshot, tensor in zip(
                snapshots,
                state,
            )
        )

    def test_build_rejects_current_partition_with_incompatible_padding(
        self, device, dtype
    ):
        """Scratch capacity follows the current padded total, not atom total alone."""
        positions, cell_batch, batch_ptr = _make_batch(
            [32, 32], [10.0, 10.0], device=device, dtype=dtype, seed=26
        )
        allocation_ptr = torch.tensor([0, 33, 64], dtype=torch.int32, device=device)
        state = allocate_batch_cluster_tile_list(
            allocation_ptr, torch.device(device), dtype=dtype
        )
        with pytest.raises(ValueError, match="padded atom length"):
            batch_build_cluster_tile_list(positions, 2.5, cell_batch, batch_ptr, *state)

    def test_build_accepts_different_partition_with_same_padding(self, device, dtype):
        """Complete scratch is reusable when the current padded layout fits."""
        allocation_ptr = torch.tensor([0, 33, 64], dtype=torch.int32, device=device)
        positions, cell_batch, batch_ptr = _make_batch(
            [34, 30], [10.0, 10.0], device=device, dtype=dtype, seed=27
        )
        state = allocate_batch_cluster_tile_list(
            allocation_ptr, torch.device(device), dtype=dtype
        )
        batch_build_cluster_tile_list(positions, 2.5, cell_batch, batch_ptr, *state)
        assert int(state[15].item()) >= 0

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

    def test_no_grad_uses_nondifferentiable_geometry(self, device):
        """Disabled grad mode returns batched geometry without an autograd graph."""
        pos, cell_batch, batch_ptr = self._make_batch(device)
        pos.requires_grad_(True)
        cell_batch.requires_grad_(True)

        with torch.no_grad():
            matrix, counts, shifts, distances, vectors = (
                batch_cluster_tile_neighbor_list(
                    pos,
                    1.5,
                    cell_batch,
                    batch_ptr,
                    max_neighbors=64,
                    return_distances=True,
                    return_vectors=True,
                )
            )

        active = torch.arange(matrix.shape[1], device=device)[None, :] < counts[:, None]
        batch_idx = torch.repeat_interleave(
            torch.arange(cell_batch.shape[0], device=device),
            (batch_ptr[1:] - batch_ptr[:-1]).to(torch.long),
        )
        assert not distances.requires_grad
        assert not vectors.requires_grad
        assert torch.equal(distances[~active], torch.zeros_like(distances[~active]))
        assert torch.equal(vectors[~active], torch.zeros_like(vectors[~active]))
        safe_neighbors = torch.where(active, matrix, 0).to(torch.long)
        expected = pos[safe_neighbors] - pos[:, None]
        expected = expected + torch.einsum(
            "nma,nab->nmb", shifts.to(pos.dtype), cell_batch[batch_idx]
        )
        torch.testing.assert_close(vectors[active], expected[active])
        torch.testing.assert_close(distances[active], expected.norm(dim=-1)[active])

    def test_matrix_geometry_buffers_are_detached_snapshots(self, device):
        """Differentiable batched returns leave reusable buffers graph-free."""
        pos, cell_batch, batch_ptr = self._make_batch(device)
        pos.requires_grad_(True)
        cell_batch.requires_grad_(True)
        vectors = torch.full((pos.shape[0], 64, 3), -3.0, device=device)
        distances = torch.full((pos.shape[0], 64), -3.0, device=device)

        output = batch_cluster_tile_neighbor_list(
            pos,
            1.5,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            return_distances=True,
            return_vectors=True,
            neighbor_vectors=vectors,
            neighbor_distances=distances,
        )
        returned_distances, returned_vectors = output[3:]
        assert returned_distances is not distances
        assert returned_vectors is not vectors
        assert returned_distances.data_ptr() != distances.data_ptr()
        assert returned_vectors.data_ptr() != vectors.data_ptr()
        torch.testing.assert_close(distances, returned_distances.detach())
        torch.testing.assert_close(vectors, returned_vectors.detach())
        assert not distances.requires_grad and distances.grad_fn is None
        assert not vectors.requires_grad and vectors.grad_fn is None
        gradients = torch.autograd.grad(
            returned_distances.sum() + returned_vectors.square().sum(),
            (pos, cell_batch),
        )
        assert all(torch.isfinite(value).all() for value in gradients)

        with torch.no_grad():
            no_grad_output = batch_cluster_tile_neighbor_list(
                pos,
                1.5,
                cell_batch,
                batch_ptr,
                max_neighbors=64,
                return_distances=True,
                return_vectors=True,
                neighbor_vectors=vectors,
                neighbor_distances=distances,
            )
        assert no_grad_output[3] is distances
        assert no_grad_output[4] is vectors
        assert not distances.requires_grad and distances.grad_fn is None
        assert not vectors.requires_grad and vectors.grad_fn is None

    @pytest.mark.parametrize(
        ("buffer_name", "shape"),
        [
            ("neighbor_distances", (64, 64)),
            ("neighbor_vectors", (64, 64, 3)),
        ],
    )
    def test_matrix_geometry_rejects_grad_tracked_buffers(
        self, device, buffer_name, shape
    ):
        """Batched geometry buffers reject autograd metadata before mutation."""
        pos, cell_batch, batch_ptr = self._make_batch(device)
        buffer = torch.full(shape, -7.0, device=device, requires_grad=True)
        before = buffer.detach().clone()
        kwargs = {
            "return_distances": buffer_name == "neighbor_distances",
            "return_vectors": buffer_name == "neighbor_vectors",
            buffer_name: buffer,
        }

        with pytest.raises(
            ValueError,
            match=rf"{buffer_name} must not require gradients",
        ):
            batch_cluster_tile_neighbor_list(
                pos,
                1.5,
                cell_batch,
                batch_ptr,
                max_neighbors=64,
                **kwargs,
            )
        torch.testing.assert_close(buffer.detach(), before)

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

    @pytest.mark.parametrize(
        ("return_distances", "return_vectors"),
        [(False, True), (True, False), (True, True)],
    )
    @pytest.mark.slow
    def test_matrix_geometry_fullgraph_matches_eager(
        self, device, return_distances, return_vectors
    ):
        """Compiled batched matrix geometry uses the source-system cell."""
        pos, cell_batch, batch_ptr = self._make_batch(device)
        scratch = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr, torch.device(device), dtype=pos.dtype, max_tiles_per_group=4
            )
        )

        @torch.compile(fullgraph=True)
        def run(runtime_positions, runtime_cell):
            return batch_cluster_tile_neighbor_list(
                runtime_positions,
                1.5,
                runtime_cell,
                batch_ptr,
                max_neighbors=64,
                return_distances=return_distances,
                return_vectors=return_vectors,
                **scratch,
            )

        eager = batch_cluster_tile_neighbor_list(
            pos,
            1.5,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
            max_tiles_per_group=4,
            return_distances=return_distances,
            return_vectors=return_vectors,
        )
        compiled = run(pos, cell_batch)
        matrix, counts, shifts = compiled[:3]
        assert all(not value.requires_grad for value in compiled[:3])
        assert torch.equal(eager[1], counts)
        active = torch.arange(matrix.shape[1], device=device)[None, :] < counts[:, None]
        neighbors = torch.where(active, matrix, 0).to(torch.long)
        batch_idx = torch.repeat_interleave(
            torch.arange(cell_batch.shape[0], device=device),
            (batch_ptr[1:] - batch_ptr[:-1]).to(torch.long),
            output_size=pos.shape[0],
        )
        expected_vectors = pos[neighbors] - pos[:, None]
        expected_vectors = expected_vectors + torch.einsum(
            "nma,nab->nmb", shifts.to(pos.dtype), cell_batch[batch_idx]
        )
        expected_vectors = torch.where(
            active[..., None], expected_vectors, torch.zeros_like(expected_vectors)
        )
        offset = 3
        if return_distances:
            torch.testing.assert_close(compiled[offset], expected_vectors.norm(dim=-1))
            offset += 1
        if return_vectors:
            torch.testing.assert_close(compiled[offset], expected_vectors)

    @pytest.mark.slow
    def test_matrix_geometry_compiled_position_and_cell_gradients(self, device):
        """Compiled batched matrix geometry preserves position and cell gradients."""
        pos, cell_batch, batch_ptr = self._make_batch(device)
        distance_buffer = torch.empty((pos.shape[0], 64), device=device)
        scratch = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr, torch.device(device), dtype=pos.dtype, max_tiles_per_group=4
            )
        )

        def eager_loss(runtime_positions, runtime_cell):
            return batch_cluster_tile_neighbor_list(
                runtime_positions,
                1.5,
                runtime_cell,
                batch_ptr,
                max_neighbors=64,
                max_tiles_per_group=4,
                return_distances=True,
            )[3].sum()

        @torch.compile(fullgraph=True)
        def compiled_geometry(runtime_positions, runtime_cell):
            return batch_cluster_tile_neighbor_list(
                runtime_positions,
                1.5,
                runtime_cell,
                batch_ptr,
                max_neighbors=64,
                return_distances=True,
                neighbor_distances=distance_buffer,
                **scratch,
            )[3]

        eager_pos = pos.clone().requires_grad_(True)
        eager_cell = cell_batch.clone().requires_grad_(True)
        expected = torch.autograd.grad(
            eager_loss(eager_pos, eager_cell), (eager_pos, eager_cell)
        )
        compiled_pos = pos.clone().requires_grad_(True)
        compiled_cell = cell_batch.clone().requires_grad_(True)
        actual_distances = compiled_geometry(compiled_pos, compiled_cell)
        actual = torch.autograd.grad(
            actual_distances.sum(), (compiled_pos, compiled_cell)
        )
        torch.testing.assert_close(actual, expected)
        assert actual_distances.data_ptr() != distance_buffer.data_ptr()
        torch.testing.assert_close(distance_buffer, actual_distances.detach())
        assert not distance_buffer.requires_grad and distance_buffer.grad_fn is None

    @pytest.mark.slow
    @pytest.mark.parametrize("compiled", [False, True])
    @pytest.mark.parametrize(
        ("return_distances", "return_vectors"),
        [(True, False), (False, True), (True, True)],
        ids=["distances", "vectors", "both"],
    )
    def test_compact_coo_geometry_buffer_lifecycle(
        self, device, compiled, return_distances, return_vectors
    ):
        """Batched exact COO buffers stay detached across grad-mode changes."""
        batch_ptr = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
        cell_batch = torch.stack(
            (
                torch.eye(3, dtype=torch.float32, device=device) * 20.0,
                torch.diag(torch.tensor([22.0, 20.0, 18.0], device=device)),
            )
        )
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        scratch = _scratch_kwargs(
            allocate_batch_cluster_tile_list(
                batch_ptr,
                torch.device(device),
                dtype=positions.dtype,
                max_tiles_per_group=1,
            )
        )

        def call(runtime_positions, runtime_cell, vectors, distances):
            return batch_cluster_tile_neighbor_list(
                runtime_positions,
                1.0,
                runtime_cell,
                batch_ptr,
                format="coo",
                max_neighbors=8,
                max_pairs=32,
                return_vectors=return_vectors,
                return_distances=return_distances,
                neighbor_vectors=vectors,
                neighbor_distances=distances,
                **scratch,
            )

        run = torch.compile(call, fullgraph=True) if compiled else call
        vectors = torch.full((32, 3), float("nan"), device=device)
        distances = torch.full((32,), float("nan"), device=device)
        grad_positions = positions.clone().requires_grad_(True)
        grad_cell = cell_batch.clone().requires_grad_(True)
        output = run(
            grad_positions,
            grad_cell,
            vectors,
            distances,
        )
        pairs, pointer, shifts = output[:3]
        _compact_coo_pair_sets(pairs, pointer, shifts)
        count = pairs.shape[1]
        atom_system = torch.bucketize(
            torch.arange(positions.shape[0], dtype=torch.int32, device=device),
            batch_ptr[1:],
            right=True,
        )
        pair_system = atom_system[pairs[0].long()]
        expected_vectors = (
            grad_positions[pairs[1].long()] - grad_positions[pairs[0].long()]
        )
        expected_vectors = expected_vectors + torch.einsum(
            "pa,pab->pb", shifts.to(positions.dtype), grad_cell[pair_system.long()]
        )
        output_index = 3
        if return_distances:
            exact_distances = output[output_index]
            output_index += 1
            torch.testing.assert_close(distances[:count], expected_vectors.norm(dim=-1))
            torch.testing.assert_close(exact_distances, expected_vectors.norm(dim=-1))
            assert exact_distances.data_ptr() != distances.data_ptr()
        if return_vectors:
            exact_vectors = output[output_index]
            torch.testing.assert_close(vectors[:count], expected_vectors)
            torch.testing.assert_close(exact_vectors, expected_vectors)
            assert exact_vectors.data_ptr() != vectors.data_ptr()
        gradients = torch.autograd.grad(
            sum(tensor.square().sum() for tensor in output[3:]),
            (grad_positions, grad_cell),
        )
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert not vectors.requires_grad and vectors.grad_fn is None
        assert not distances.requires_grad and distances.grad_fn is None

        with torch.no_grad():
            no_grad_output = run(positions + 0.05, cell_batch, vectors, distances)
        assert all(not tensor.requires_grad for tensor in no_grad_output[3:])
        assert not vectors.requires_grad and vectors.grad_fn is None
        assert not distances.requires_grad and distances.grad_fn is None

        final_positions = (positions + 0.1).requires_grad_(True)
        final_cell = cell_batch.clone().requires_grad_(True)
        final_output = run(final_positions, final_cell, vectors, distances)
        final_gradients = torch.autograd.grad(
            sum(tensor.square().sum() for tensor in final_output[3:]),
            (final_positions, final_cell),
        )
        assert all(torch.isfinite(gradient).all() for gradient in final_gradients)
        assert not vectors.requires_grad and vectors.grad_fn is None
        assert not distances.requires_grad and distances.grad_fn is None

    @pytest.mark.parametrize(
        ("buffer_name", "shape"),
        [
            ("neighbor_distances", (32,)),
            ("neighbor_vectors", (32, 3)),
        ],
    )
    def test_compact_coo_rejects_grad_tracked_geometry_buffers(
        self, device, buffer_name, shape
    ):
        """Batched exact COO rejects grad-tracked capacity buffers."""
        batch_ptr = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cell_batch = torch.eye(3, device=device).repeat(2, 1, 1) * 20.0
        buffer = torch.full(shape, -7.0, device=device, requires_grad=True)
        before = buffer.detach().clone()
        kwargs = {
            "return_distances": buffer_name == "neighbor_distances",
            "return_vectors": buffer_name == "neighbor_vectors",
            buffer_name: buffer,
        }

        with pytest.raises(
            ValueError,
            match=rf"{buffer_name} must not require gradients",
        ):
            batch_cluster_tile_neighbor_list(
                positions,
                1.0,
                cell_batch,
                batch_ptr,
                format="coo",
                max_neighbors=8,
                max_pairs=32,
                max_tiles_per_group=1,
                **kwargs,
            )
        torch.testing.assert_close(buffer.detach(), before)
