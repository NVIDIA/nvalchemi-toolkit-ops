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

"""Public-contract tests for prepared Torch cluster-tile execution."""

import inspect
from dataclasses import FrozenInstanceError, is_dataclass

import pytest
import torch

import nvalchemiops.torch.neighbors.prepared_cluster_tile as prepared_module
from nvalchemiops.torch.neighbors import (
    ClusterTileState,
    batch_cluster_tile_neighbor_list,
    cluster_tile_neighbor_list,
    cluster_tile_neighbor_list_prepared,
    prepare_cluster_tile,
)


def _inputs(
    batched: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return deterministic single or batched CUDA inputs."""
    torch.manual_seed(11)
    if batched:
        positions = torch.rand((48, 3), dtype=torch.float32, device="cuda") * 8.0
        cell = torch.eye(3, dtype=torch.float32, device="cuda").repeat(2, 1, 1)
        cell *= 8.0
        batch_ptr = torch.tensor([0, 17, 48], dtype=torch.int32, device="cuda")
        return positions, cell, batch_ptr
    positions = torch.rand((32, 3), dtype=torch.float32, device="cuda") * 8.0
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 8.0
    return positions, cell, None


def _direct(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_ptr: torch.Tensor | None,
    *,
    format: str,
    cutoff2: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run the matching direct route with the test capacities."""
    kwargs = {
        "format": format,
        "max_neighbors": 32,
        "max_pairs": 2048,
        "max_tiles_per_group": 4,
        "cutoff2": cutoff2,
    }
    if batch_ptr is None:
        return cluster_tile_neighbor_list(positions, 1.2, cell, **kwargs)
    return batch_cluster_tile_neighbor_list(
        positions,
        1.2,
        cell,
        batch_ptr,
        **kwargs,
    )


def _matrix_records(
    output: tuple[torch.Tensor, ...],
    start: int = 0,
) -> list[list[tuple[int, int, int, int]]]:
    """Return order-independent active matrix records by row."""
    matrix, counts, shifts = output[start : start + 3]
    records = []
    for source, count in enumerate(counts.cpu().tolist()):
        targets = matrix[source, :count].cpu().tolist()
        row_shifts = shifts[source, :count].cpu().tolist()
        records.append(
            sorted((target, *shift) for target, shift in zip(targets, row_shifts))
        )
    return records


def _reference_matrix_records(
    positions: torch.Tensor,
    cutoff: float,
    batch_ptr: torch.Tensor | None = None,
) -> list[list[tuple[int, int, int, int]]]:
    """Return matrix records from an independent non-periodic reference."""
    values = positions.cpu().tolist()
    cutoff_squared = cutoff * cutoff
    segments = (
        zip(batch_ptr.cpu().tolist(), batch_ptr.cpu().tolist()[1:])
        if batch_ptr is not None
        else [(0, len(values))]
    )
    records = [[] for _ in values]
    for start, stop in segments:
        for source in range(start, stop):
            records[source] = sorted(
                (target, 0, 0, 0)
                for target in range(start, stop)
                if target != source
                and sum(
                    (coordinate - source_coordinate) ** 2
                    for coordinate, source_coordinate in zip(
                        values[target], values[source]
                    )
                )
                <= cutoff_squared
            )
    return records


def _coo_records(output: tuple[torch.Tensor, ...]) -> list[tuple[int, ...]]:
    """Validate CSR ownership and return order-independent COO records."""
    pairs, pointer, shifts = output
    assert pointer.tolist()[0] == 0
    assert int(pointer[-1].item()) == pairs.shape[1]
    assert torch.all(pointer[1:] >= pointer[:-1])
    for source, (start, stop) in enumerate(
        zip(pointer[:-1].cpu().tolist(), pointer[1:].cpu().tolist())
    ):
        assert torch.all(pairs[0, start:stop] == source)
    return sorted(
        (source, target, *shift)
        for (source, target), shift in zip(
            pairs.transpose(0, 1).cpu().tolist(), shifts.cpu().tolist()
        )
    )


def _tile_records(
    output: tuple[torch.Tensor, ...],
    *,
    batched: bool,
) -> tuple[list[tuple[int, ...]], tuple[torch.Tensor, ...]]:
    """Return active tile records and deterministic sorting outputs."""
    count = int(output[0].item())
    tile_fields = 3 if batched else 2
    records = sorted(
        zip(*(value[:count].cpu().tolist() for value in output[1 : 1 + tile_fields]))
    )
    return records, output[1 + tile_fields :]


def _assert_same(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    *,
    format: str,
    batched: bool,
    dual: bool = False,
) -> None:
    """Compare the defined contents of a cluster-tile result."""
    assert len(actual) == len(expected)
    if format == "matrix":
        starts = (0, 3) if dual else (0,)
        for start in starts:
            assert torch.equal(actual[start + 1], expected[start + 1])
            assert _matrix_records(actual, start) == _matrix_records(expected, start)
    elif format == "coo":
        assert torch.equal(actual[1], expected[1])
        assert _coo_records(actual) == _coo_records(expected)
    else:
        actual_tiles, actual_sort = _tile_records(actual, batched=batched)
        expected_tiles, expected_sort = _tile_records(expected, batched=batched)
        assert actual_tiles == expected_tiles
        assert all(
            torch.equal(left, right) for left, right in zip(actual_sort, expected_sort)
        )


@pytest.mark.gpu
@pytest.mark.parametrize("format", ["tile", "matrix", "coo"])
@pytest.mark.parametrize("batched", [False, True])
def test_prepared_eager_matches_direct(format: str, batched: bool) -> None:
    """Prepared eager execution preserves every direct return tuple."""
    positions, cell, batch_ptr = _inputs(batched)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format=format,
        batch_ptr=batch_ptr,
        max_neighbors=32,
        max_pairs=2048,
        max_tiles_per_group=4,
    )
    actual = cluster_tile_neighbor_list_prepared(positions, cell, state)
    expected = _direct(positions, cell, batch_ptr, format=format)
    _assert_same(actual, expected, format=format, batched=batched)


@pytest.mark.gpu
@pytest.mark.parametrize("batched", [False, True])
def test_prepared_dual_matrix_matches_direct(batched: bool) -> None:
    """Prepared dual-cutoff matrices retain both direct output triples."""
    positions, cell, batch_ptr = _inputs(batched)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        batch_ptr=batch_ptr,
        max_neighbors=32,
        cutoff2=1.6,
        max_tiles_per_group=4,
    )
    actual = cluster_tile_neighbor_list_prepared(positions, cell, state)
    expected = _direct(
        positions,
        cell,
        batch_ptr,
        format="matrix",
        cutoff2=1.6,
    )
    _assert_same(actual, expected, format="matrix", batched=batched, dual=True)


@pytest.mark.gpu
@pytest.mark.parametrize(("cutoff", "cutoff2"), [(4.0, 0.91), (4.0, 4.0)])
@pytest.mark.parametrize("batched", [False, True])
def test_prepared_dual_matrix_default_capacity_matches_reference(
    cutoff: float,
    cutoff2: float,
    batched: bool,
) -> None:
    """Default dual-cutoff capacity covers both independently referenced cutoffs."""
    indices = torch.arange(80 if batched else 40, dtype=torch.float32, device="cuda")
    local = indices.remainder(40)
    positions = torch.stack(
        (
            local * 0.05,
            torch.zeros_like(local),
            torch.zeros_like(local),
        ),
        dim=1,
    )
    if batched:
        cell = torch.eye(3, dtype=torch.float32, device="cuda").repeat(2, 1, 1) * 20.0
        batch_ptr = torch.tensor([0, 40, 80], dtype=torch.int32, device="cuda")
    else:
        cell = torch.eye(3, dtype=torch.float32, device="cuda") * 20.0
        batch_ptr = None
    state = prepare_cluster_tile(
        positions,
        cutoff,
        cell,
        format="matrix",
        batch_ptr=batch_ptr,
        cutoff2=cutoff2,
        max_tiles_per_group=4,
    )
    actual = cluster_tile_neighbor_list_prepared(positions, cell, state)
    assert state.max_neighbors == 64
    assert _matrix_records(actual, 0) == _reference_matrix_records(
        positions, cutoff, batch_ptr
    )
    assert _matrix_records(actual, 3) == _reference_matrix_records(
        positions, cutoff2, batch_ptr
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("batched", [False, True])
def test_prepared_dual_matrix_fullgraph(batched: bool) -> None:
    """Closure-captured dual-cutoff state preserves both matrix triples."""
    positions, cell, batch_ptr = _inputs(batched)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        batch_ptr=batch_ptr,
        max_neighbors=32,
        cutoff2=1.6,
        max_tiles_per_group=4,
    )

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return cluster_tile_neighbor_list_prepared(values, box, state)

    actual = run(positions, cell)
    expected = _direct(
        positions,
        cell,
        batch_ptr,
        format="matrix",
        cutoff2=1.6,
    )
    _assert_same(actual, expected, format="matrix", batched=batched, dual=True)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("format", ["tile", "matrix", "coo"])
@pytest.mark.parametrize("batched", [False, True])
def test_prepared_fullgraph_closure_matches_direct(
    format: str,
    batched: bool,
) -> None:
    """A closure-captured state supports each nonselective format."""
    positions, cell, batch_ptr = _inputs(batched)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format=format,
        batch_ptr=batch_ptr,
        max_neighbors=32,
        max_pairs=2048,
        max_tiles_per_group=4,
    )
    torch.compiler.reset()

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return cluster_tile_neighbor_list_prepared(values, box, state)

    actual = run(positions, cell)
    expected = _direct(positions, cell, batch_ptr, format=format)
    _assert_same(actual, expected, format=format, batched=batched)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("batched", [False, True])
def test_prepared_fullgraph_exact_coo_changes_size(batched: bool) -> None:
    """One compiled prepared call returns different exact COO sizes."""
    close = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    far = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    if batched:
        cell = torch.eye(3, dtype=torch.float32, device="cuda").repeat(2, 1, 1)
        cell *= 20.0
        batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    else:
        cell = torch.eye(3, dtype=torch.float32, device="cuda") * 20.0
        batch_ptr = None
    state = prepare_cluster_tile(
        close,
        1.0,
        cell,
        format="coo",
        batch_ptr=batch_ptr,
        max_neighbors=8,
        max_pairs=16,
        max_tiles_per_group=1,
    )

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return cluster_tile_neighbor_list_prepared(values, cell, state)

    close_output = run(close)
    far_output = run(far)
    assert close_output[0].shape[1] == (4 if batched else 12)
    assert far_output[0].shape[1] == 0
    _coo_records(close_output)
    _coo_records(far_output)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("batched", [False, True])
def test_prepared_geometry_and_gradients(batched: bool) -> None:
    """Prepared matrix geometry keeps position and cell gradients."""
    positions, cell, batch_ptr = _inputs(batched)
    state = prepare_cluster_tile(
        positions,
        1.5,
        cell,
        format="matrix",
        batch_ptr=batch_ptr,
        max_neighbors=32,
        return_vectors=True,
        return_distances=True,
        max_tiles_per_group=4,
    )

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return cluster_tile_neighbor_list_prepared(values, box, state)

    grad_positions = positions.clone().requires_grad_(True)
    grad_cell = cell.clone().requires_grad_(True)
    output = run(grad_positions, grad_cell)
    assert output[3] is state.neighbor_distances
    assert output[4] is state.neighbor_vectors
    gradients = torch.autograd.grad(output[3].sum(), (grad_positions, grad_cell))
    assert all(torch.isfinite(value).all() for value in gradients)


@pytest.mark.gpu
@pytest.mark.slow
def test_prepared_exact_coo_geometry_is_aligned() -> None:
    """Prepared COO returns exact geometry and updates reusable buffers."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 20.0
    state = prepare_cluster_tile(
        positions,
        1.0,
        cell,
        format="coo",
        max_neighbors=8,
        max_pairs=16,
        return_vectors=True,
        return_distances=True,
        max_tiles_per_group=1,
    )

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return cluster_tile_neighbor_list_prepared(values, box, state)

    grad_positions = positions.clone().requires_grad_(True)
    grad_cell = cell.clone().requires_grad_(True)
    pairs, _, shifts, distances, vectors = run(grad_positions, grad_cell)
    count = pairs.shape[1]
    expected = (
        grad_positions[pairs[1].long()]
        - grad_positions[pairs[0].long()]
        + shifts.to(grad_positions.dtype) @ grad_cell
    )
    torch.testing.assert_close(vectors[:count], expected)
    torch.testing.assert_close(distances[:count], expected.norm(dim=-1))
    torch.testing.assert_close(state.neighbor_vectors[:count], expected)
    torch.testing.assert_close(state.neighbor_distances[:count], expected.norm(dim=-1))
    assert vectors.data_ptr() != state.neighbor_vectors.data_ptr()
    assert distances.data_ptr() != state.neighbor_distances.data_ptr()
    gradients = torch.autograd.grad(
        distances.sum(),
        (grad_positions, grad_cell),
    )
    assert all(torch.isfinite(value).all() for value in gradients)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("return_vectors", "return_distances"),
    [(True, False), (False, True), (True, True)],
)
def test_prepared_rejects_dual_cutoff_geometry_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
    return_vectors: bool,
    return_distances: bool,
) -> None:
    """Dual-cutoff geometry fails before prepared scratch is allocated."""
    positions, cell, _ = _inputs(False)

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("dual-cutoff geometry reached scratch allocation")

    monkeypatch.setattr(
        prepared_module,
        "allocate_cluster_tile_list",
        unexpected_allocation,
    )
    with pytest.raises(
        ValueError,
        match="cutoff2 cannot be combined with return_vectors or return_distances",
    ):
        prepare_cluster_tile(
            positions,
            1.2,
            cell,
            format="matrix",
            cutoff2=1.6,
            return_vectors=return_vectors,
            return_distances=return_distances,
        )


@pytest.mark.gpu
def test_prepared_batch_partition_metadata_is_cached() -> None:
    """Prepared batches reuse only partition-derived metadata tensors."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda").repeat(3, 1, 1) * 8.0
    batch_ptr = torch.tensor([0, 0, 2, 5], dtype=torch.int32, device="cuda")
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        batch_ptr=batch_ptr,
        max_neighbors=8,
        max_tiles_per_group=2,
    )
    metadata = state._partition_metadata
    assert metadata is not None
    assert metadata._fields == (
        "atom_system",
        "batch_ptr_padded",
        "padded_slot_system",
        "real_sorted_rank_to_padded_slot",
        "group_ptr",
        "group_system",
    )
    assert metadata.atom_system.tolist() == [1, 1, 2, 2, 2]
    assert metadata.batch_ptr_padded.tolist() == [0, 0, 32, 64]
    assert metadata.padded_slot_system.tolist() == [1] * 32 + [2] * 32
    assert metadata.real_sorted_rank_to_padded_slot.tolist() == [0, 1, 32, 33, 34]
    assert metadata.group_ptr.tolist() == [0, 0, 1, 2]
    assert metadata.group_system.tolist() == [1, 2]
    assert metadata.padded_slot_system is state._scratch[5]
    assert metadata.batch_ptr_padded is state._scratch[6]
    assert metadata.group_system is state._scratch[7]
    assert metadata.group_ptr is state._scratch[8]

    metadata_tensors = tuple(metadata)
    metadata_values = tuple(value.clone() for value in metadata_tensors)
    metadata_pointers = tuple(value.data_ptr() for value in metadata_tensors)
    cluster_tile_neighbor_list_prepared(positions, cell, state)
    sorted_positions_before = state._scratch[2].clone()

    changed_positions = positions.clone()
    changed_positions[1, 0] = 0.9
    changed_positions[3, 1] = 0.7
    changed_cell = cell.clone()
    changed_cell[2, 0, 0] = 9.0
    actual = cluster_tile_neighbor_list_prepared(
        changed_positions,
        changed_cell,
        state,
    )
    expected = batch_cluster_tile_neighbor_list(
        changed_positions,
        1.2,
        changed_cell,
        batch_ptr,
        format="matrix",
        max_neighbors=8,
        max_tiles_per_group=2,
    )
    _assert_same(actual, expected, format="matrix", batched=True)
    assert not torch.equal(sorted_positions_before, state._scratch[2])
    assert tuple(value.data_ptr() for value in metadata_tensors) == metadata_pointers
    for value, expected_value in zip(metadata_tensors, metadata_values):
        assert torch.equal(value, expected_value)


@pytest.mark.gpu
def test_prepared_storage_is_owned_and_reused() -> None:
    """States own distinct buffers and reuse each borrowed matrix result."""
    positions, cell, _ = _inputs(False)
    first = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    second = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    first_output = cluster_tile_neighbor_list_prepared(positions, cell, first)
    second_output = cluster_tile_neighbor_list_prepared(positions, cell, second)
    reused_output = cluster_tile_neighbor_list_prepared(positions * 0.9, cell, first)
    assert first_output[0] is reused_output[0]
    assert first_output[1] is reused_output[1]
    assert first_output[0].data_ptr() != second_output[0].data_ptr()

    coo_state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="coo",
        max_neighbors=32,
        max_pairs=1024,
        max_tiles_per_group=4,
    )
    first_coo = cluster_tile_neighbor_list_prepared(positions, cell, coo_state)
    second_coo = cluster_tile_neighbor_list_prepared(positions, cell, coo_state)
    assert first_coo[0].data_ptr() != second_coo[0].data_ptr()


@pytest.mark.gpu
def test_preparation_owns_default_capacities() -> None:
    """Preparation estimates omitted capacities and owns the resulting buffers."""
    positions, cell, _ = _inputs(False)
    state = prepare_cluster_tile(positions, 1.2, cell, format="coo")
    output = cluster_tile_neighbor_list_prepared(positions, cell, state)
    assert state.max_neighbors >= 32
    assert state.max_pairs == positions.shape[0] * state.max_neighbors
    assert state.max_tiles_per_group > 0
    _coo_records(output)


@pytest.mark.gpu
def test_prepared_configuration_is_frozen() -> None:
    """The public state is one frozen slotted dataclass."""
    positions, cell, _ = _inputs(False)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        max_neighbors=32,
        return_distances=True,
        max_tiles_per_group=4,
    )
    assert is_dataclass(state)
    assert hasattr(ClusterTileState, "__slots__")
    with pytest.raises(FrozenInstanceError):
        state.cutoff = 2.0
    with pytest.raises((AttributeError, TypeError)):
        state.neighbor_distances = torch.empty_like(state.neighbor_distances)


@pytest.mark.gpu
def test_prepared_rejects_static_input_mismatches() -> None:
    """Preparation and execution reject mismatches before launch."""
    positions, cell, _ = _inputs(False)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    with pytest.raises(ValueError, match="positions shape"):
        cluster_tile_neighbor_list_prepared(positions[:-1], cell, state)
    with pytest.raises(TypeError, match="positions dtype"):
        cluster_tile_neighbor_list_prepared(positions.double(), cell, state)
    with pytest.raises(ValueError, match="positions device"):
        cluster_tile_neighbor_list_prepared(positions.cpu(), cell, state)
    with pytest.raises(ValueError, match="cell shape"):
        cluster_tile_neighbor_list_prepared(positions, cell.unsqueeze(0), state)
    with pytest.raises(TypeError, match="cell dtype"):
        cluster_tile_neighbor_list_prepared(positions, cell.double(), state)
    with pytest.raises(ValueError, match="cell device"):
        cluster_tile_neighbor_list_prepared(positions, cell.cpu(), state)
    bad_ptr = torch.tensor([0, 16, 31], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="end at N"):
        prepare_cluster_tile(
            positions,
            1.2,
            cell.repeat(2, 1, 1),
            format="matrix",
            batch_ptr=bad_ptr,
        )


def test_prepared_api_has_no_caller_owned_storage_or_pair_callback() -> None:
    """Preparation exposes only the approved minimal public arguments."""
    parameters = set(inspect.signature(prepare_cluster_tile).parameters)
    assert parameters == {
        "positions",
        "cutoff",
        "cell",
        "format",
        "batch_ptr",
        "max_neighbors",
        "fill_value",
        "max_pairs",
        "cutoff2",
        "return_vectors",
        "return_distances",
        "max_tiles_per_group",
    }
    assert set(inspect.signature(cluster_tile_neighbor_list_prepared).parameters) == {
        "positions",
        "cell",
        "state",
    }


@pytest.mark.gpu
@pytest.mark.parametrize("format", ["tile", "matrix", "coo"])
def test_prepared_empty_single_system(format: str) -> None:
    """Prepared execution preserves empty-system behavior."""
    positions = torch.empty((0, 3), dtype=torch.float32, device="cuda")
    cell = torch.eye(3, dtype=torch.float32, device="cuda")
    state = prepare_cluster_tile(
        positions,
        1.0,
        cell,
        format=format,
        max_neighbors=8,
        max_tiles_per_group=1,
    )
    output = cluster_tile_neighbor_list_prepared(positions, cell, state)
    if format == "tile":
        assert int(output[0].item()) == 0
    elif format == "matrix":
        assert output[0].shape == (0, 8)
        assert output[1].numel() == 0
    else:
        assert output[0].shape == (2, 0)
        assert output[1].tolist() == [0]
