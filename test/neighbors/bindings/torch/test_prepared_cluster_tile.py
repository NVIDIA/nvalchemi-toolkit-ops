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
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError, is_dataclass

import pytest
import torch

import nvalchemiops.torch.neighbors.prepared_cluster_tile as prepared_module
from nvalchemiops.torch.neighbors import (
    ClusterTileState,
    NeighborOverflowError,
    batch_cluster_tile_neighbor_list,
    cluster_tile_neighbor_list,
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


def _prepared_neighbor_list(
    positions: torch.Tensor,
    cell: torch.Tensor,
    state: ClusterTileState,
    *,
    rebuild_flags: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Route prepared storage through its matching public API."""
    if state.is_batched:
        return batch_cluster_tile_neighbor_list(
            positions,
            None,
            cell,
            None,
            rebuild_flags=rebuild_flags,
            state=state,
        )
    return cluster_tile_neighbor_list(
        positions,
        None,
        cell,
        rebuild_flags=rebuild_flags,
        state=state,
    )


def _run_uninitialized_selective_fullgraph() -> subprocess.CompletedProcess[str]:
    """Run one asynchronous initialization failure in a fresh process."""
    script = textwrap.dedent(
        """
        import torch
        from nvalchemiops.torch.neighbors import (
            cluster_tile_neighbor_list,
            prepare_cluster_tile,
        )

        positions = torch.rand((32, 3), dtype=torch.float32, device="cuda")
        cell = torch.eye(3, dtype=torch.float32, device="cuda") * 8.0
        state = prepare_cluster_tile(
            positions,
            1.0,
            cell,
            format="matrix",
            selective=True,
            max_neighbors=32,
            max_tiles_per_group=2,
        )

        @torch.compile(fullgraph=True)
        def run(values, flags):
            return cluster_tile_neighbor_list(
                values, None, cell, rebuild_flags=flags, state=state
            )

        flags = torch.zeros(1, dtype=torch.bool, device="cuda")
        print("SELECTIVE_CALL_STARTED", flush=True)
        run(positions, flags)
        torch.cuda.synchronize()
        print("SELECTIVE_CALL_RETURNED", flush=True)
        """
    )
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _run_direct_eager_capture() -> subprocess.CompletedProcess[str]:
    """Attempt unsupported direct eager prepared execution during capture."""
    script = textwrap.dedent(
        """
        import torch
        from nvalchemiops.torch.neighbors import (
            cluster_tile_neighbor_list,
            prepare_cluster_tile,
        )

        positions = torch.rand((32, 3), dtype=torch.float32, device="cuda")
        cell = torch.eye(3, dtype=torch.float32, device="cuda") * 8.0
        state = prepare_cluster_tile(
            positions,
            1.0,
            cell,
            format="matrix",
            max_neighbors=32,
            max_tiles_per_group=2,
        )
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            cluster_tile_neighbor_list(positions, None, cell, state=state)
        """
    )
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _run_singular_prepared_fullgraph(
    batched: bool,
) -> subprocess.CompletedProcess[str]:
    """Run one compiled prepared call with a singular current cell."""
    route = (
        "batch_cluster_tile_neighbor_list" if batched else "cluster_tile_neighbor_list"
    )
    batch_setup = (
        "batch_ptr = torch.tensor([0, 16, 32], dtype=torch.int32, device='cuda')\n"
        "cell = torch.eye(3, dtype=torch.float32, device='cuda').repeat(2, 1, 1)"
        if batched
        else "batch_ptr = None\ncell = torch.eye(3, dtype=torch.float32, device='cuda')"
    ).replace("\n", "\n        ")
    singular_update = "cell[1].zero_()" if batched else "cell.zero_()"
    call = (
        "batch_cluster_tile_neighbor_list(values, None, box, None, state=state)"
        if batched
        else "cluster_tile_neighbor_list(values, None, box, state=state)"
    )
    script = textwrap.dedent(
        f"""
        import torch
        from nvalchemiops.torch.neighbors import (
            {route},
            prepare_cluster_tile,
        )

        positions = torch.rand((32, 3), dtype=torch.float32, device="cuda")
        {batch_setup}
        state = prepare_cluster_tile(
            positions,
            1.0,
            cell,
            format="matrix",
            batch_ptr=batch_ptr,
            max_neighbors=32,
            max_tiles_per_group=2,
        )

        @torch.compile(fullgraph=True)
        def run(values, box):
            return {call}

        run(positions, cell)
        torch.cuda.synchronize()
        {singular_update}
        print("SINGULAR_CALL_STARTED", flush=True)
        run(positions, cell)
        torch.cuda.synchronize()
        print("SINGULAR_CALL_RETURNED", flush=True)
        """
    )
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


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


def test_public_routes_require_legacy_inputs_without_state() -> None:
    """Both public routes reject omitted legacy inputs before execution."""
    positions = torch.empty((0, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="cutoff, cell"):
        cluster_tile_neighbor_list(positions)
    with pytest.raises(ValueError, match="cutoff, cell_batch, batch_ptr"):
        batch_cluster_tile_neighbor_list(positions)


def test_public_routes_reject_non_state_objects_before_execution() -> None:
    """The state keyword has one explicit public runtime type."""
    positions = torch.empty((0, 3), dtype=torch.float32)
    cell = torch.empty((3, 3), dtype=torch.float32)
    with pytest.raises(TypeError, match="ClusterTileState"):
        cluster_tile_neighbor_list(positions, None, cell, state=object())
    with pytest.raises(TypeError, match="ClusterTileState"):
        batch_cluster_tile_neighbor_list(
            positions, None, cell.unsqueeze(0), None, state=object()
        )


@pytest.mark.gpu
def test_public_routes_reject_wrong_state_partition_and_return_state() -> None:
    """State route selection and result ownership are explicit."""
    positions, cell, _ = _inputs(False)
    single_state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    batch_positions, batch_cell, batch_ptr = _inputs(True)
    batch_state = prepare_cluster_tile(
        batch_positions,
        1.2,
        batch_cell,
        format="matrix",
        batch_ptr=batch_ptr,
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    with pytest.raises(ValueError, match="unbatched"):
        cluster_tile_neighbor_list(positions, None, cell, state=batch_state)
    with pytest.raises(ValueError, match="batched"):
        batch_cluster_tile_neighbor_list(
            batch_positions, None, batch_cell, None, state=single_state
        )
    with pytest.raises(ValueError, match="return_state"):
        cluster_tile_neighbor_list(
            positions, None, cell, return_state=True, state=single_state
        )


@pytest.mark.gpu
def test_state_overrides_static_configuration_and_batch_ptr() -> None:
    """Prepared configuration overrides redundant static caller options."""
    positions, cell, _ = _inputs(False)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    output = cluster_tile_neighbor_list(
        positions,
        -1.0,
        cell,
        format="tile",
        max_neighbors=1,
        max_pairs=1,
        fill_value=-1,
        cutoff2=0.1,
        return_vectors=True,
        return_distances=True,
        pair_fn=object(),
        max_tiles_per_group=1,
        state=state,
    )
    assert len(output) == 3

    batch_positions, batch_cell, batch_ptr = _inputs(True)
    batch_state = prepare_cluster_tile(
        batch_positions,
        1.2,
        batch_cell,
        format="matrix",
        batch_ptr=batch_ptr,
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    batch_output = batch_cluster_tile_neighbor_list(
        batch_positions,
        -1.0,
        batch_cell,
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        format="tile",
        max_neighbors=1,
        max_pairs=1,
        fill_value=-1,
        cutoff2=0.1,
        return_vectors=True,
        return_distances=True,
        pair_fn=object(),
        max_tiles_per_group=1,
        state=batch_state,
    )
    assert len(batch_output) == 3


@pytest.mark.gpu
def test_state_aggregates_return_state_and_storage_conflicts() -> None:
    """Each public route reports every state-owned conflict in one error."""
    positions, cell, _ = _inputs(False)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    with pytest.raises(ValueError) as single_error:
        cluster_tile_neighbor_list(
            positions,
            None,
            cell,
            return_state=True,
            neighbor_matrix=torch.empty(0, device="cuda"),
            pair_forces=torch.empty(0, device="cuda"),
            state=state,
        )
    for name in ("return_state", "neighbor_matrix", "pair_forces"):
        assert name in str(single_error.value)

    batch_positions, batch_cell, batch_ptr = _inputs(True)
    batch_state = prepare_cluster_tile(
        batch_positions,
        1.2,
        batch_cell,
        format="matrix",
        batch_ptr=batch_ptr,
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    with pytest.raises(ValueError) as batch_error:
        batch_cluster_tile_neighbor_list(
            batch_positions,
            None,
            batch_cell,
            None,
            return_state=True,
            inv_cell_batch=torch.empty(0, device="cuda"),
            tile_counts=torch.empty(0, device="cuda"),
            state=batch_state,
        )
    for name in ("return_state", "inv_cell_batch", "tile_counts"):
        assert name in str(batch_error.value)


@pytest.mark.gpu
def test_state_rejects_pair_params_without_prepared_callback() -> None:
    """Prepared routes reject dynamic pair parameters without callback storage."""
    positions, cell, _ = _inputs(False)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    with pytest.raises(ValueError, match="pair_params.*prepared pair callbacks"):
        cluster_tile_neighbor_list(
            positions,
            None,
            cell,
            pair_params=torch.empty(0, device="cuda"),
            state=state,
        )

    batch_positions, batch_cell, batch_ptr = _inputs(True)
    batch_state = prepare_cluster_tile(
        batch_positions,
        1.2,
        batch_cell,
        format="matrix",
        batch_ptr=batch_ptr,
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    with pytest.raises(ValueError, match="pair_params.*prepared pair callbacks"):
        batch_cluster_tile_neighbor_list(
            batch_positions,
            None,
            batch_cell,
            None,
            pair_params=torch.empty(0, device="cuda"),
            state=batch_state,
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "argument_name",
    [
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
        "neighbor_matrix2",
        "neighbor_matrix_shifts2",
        "num_neighbors2",
        "neighbor_list",
        "neighbor_list_shifts",
        "pair_offsets",
        "pair_counts",
        "pair_counter",
        "sorted_atom_index",
        "morton_codes",
        "sorted_pos_x",
        "sorted_pos_y",
        "sorted_pos_z",
        "group_ctr_x",
        "group_ctr_y",
        "group_ctr_z",
        "group_ext_x",
        "group_ext_y",
        "group_ext_z",
        "num_tiles",
        "tile_row_group",
        "tile_col_group",
        "neighbor_vectors",
        "neighbor_distances",
        "pair_energies",
        "pair_forces",
    ],
)
def test_single_state_rejects_all_caller_owned_storage(argument_name: str) -> None:
    """Every single-route caller-owned buffer conflicts with state storage."""
    positions, cell, _ = _inputs(False)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    with pytest.raises(ValueError, match=argument_name):
        cluster_tile_neighbor_list(
            positions,
            None,
            cell,
            state=state,
            **{argument_name: torch.empty(0, device="cuda")},
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "argument_name",
    [
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
        "neighbor_matrix2",
        "neighbor_matrix_shifts2",
        "num_neighbors2",
        "neighbor_list",
        "neighbor_list_shifts",
        "pair_counter",
        "pair_offsets",
        "pair_counts",
        "inv_cell_batch",
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
        "tile_offsets",
        "tile_counts",
        "neighbor_vectors",
        "neighbor_distances",
        "pair_energies",
        "pair_forces",
    ],
)
def test_batch_state_rejects_all_caller_owned_storage(argument_name: str) -> None:
    """Every batch-route caller-owned buffer conflicts with state storage."""
    positions, cell, batch_ptr = _inputs(True)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        batch_ptr=batch_ptr,
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    with pytest.raises(ValueError, match=argument_name):
        batch_cluster_tile_neighbor_list(
            positions,
            None,
            cell,
            None,
            state=state,
            **{argument_name: torch.empty(0, device="cuda")},
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
    actual = _prepared_neighbor_list(positions, cell, state)
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
    actual = _prepared_neighbor_list(positions, cell, state)
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
    actual = _prepared_neighbor_list(positions, cell, state)
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
        return _prepared_neighbor_list(values, box, state)

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
        return _prepared_neighbor_list(values, box, state)

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
        return _prepared_neighbor_list(values, cell, state)

    close_output = run(close)
    far_output = run(far)
    assert close_output[0].shape[1] == (4 if batched else 12)
    assert far_output[0].shape[1] == 0
    _coo_records(close_output)
    _coo_records(far_output)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
@pytest.mark.parametrize(
    ("return_distances", "return_vectors"),
    [(True, False), (False, True), (True, True)],
    ids=["distances", "vectors", "both"],
)
def test_prepared_matrix_geometry_buffer_lifecycle(
    batched: bool,
    compiled: bool,
    return_distances: bool,
    return_vectors: bool,
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Prepared matrix buffers stay detached across grad-mode changes."""
    positions, cell, batch_ptr = _inputs(batched)
    state = prepare_cluster_tile(
        positions,
        1.5,
        cell,
        format="matrix",
        batch_ptr=batch_ptr,
        max_neighbors=32,
        return_vectors=return_vectors,
        return_distances=return_distances,
        max_tiles_per_group=4,
    )

    def call(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _prepared_neighbor_list(values, box, state)

    if compiled:
        torch.compiler.reset()
        run = torch.compile(call, fullgraph=True)
    else:
        run = call

    def direct(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        kwargs = {
            "format": "matrix",
            "max_neighbors": 32,
            "max_tiles_per_group": 4,
            "return_vectors": return_vectors,
            "return_distances": return_distances,
        }
        if batch_ptr is None:
            return cluster_tile_neighbor_list(values, 1.5, box, **kwargs)
        return batch_cluster_tile_neighbor_list(
            values,
            1.5,
            box,
            batch_ptr,
            **kwargs,
        )

    def geometry_loss(output: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Return an order-independent loss over requested geometry."""
        output_index = 3
        terms = []
        if return_distances:
            terms.append(output[output_index].sum())
            output_index += 1
        if return_vectors:
            terms.append(output[output_index].square().sum())
        return sum(terms)

    def assert_geometry_values(
        output: tuple[torch.Tensor, ...],
        values: torch.Tensor,
        box: torch.Tensor,
    ) -> None:
        """Check active matrix geometry against current positions and cells."""
        matrix, counts, shifts = output[:3]
        output_index = 3
        distances = output[output_index] if return_distances else None
        output_index += int(return_distances)
        vectors = output[output_index] if return_vectors else None
        for source, count in enumerate(counts.cpu().tolist()):
            if count == 0:
                continue
            targets = matrix[source, :count].long()
            source_cell = (
                box
                if batch_ptr is None
                else box[
                    torch.bucketize(
                        torch.tensor(source, dtype=torch.int32, device=values.device),
                        batch_ptr[1:],
                        right=True,
                    ).long()
                ]
            )
            expected_vectors = values[targets] - values[source]
            expected_vectors = expected_vectors + (
                shifts[source, :count].to(values.dtype) @ source_cell
            )
            if vectors is not None:
                torch.testing.assert_close(vectors[source, :count], expected_vectors)
            if distances is not None:
                torch.testing.assert_close(
                    distances[source, :count], expected_vectors.norm(dim=-1)
                )

    def assert_snapshot_contract(
        output: tuple[torch.Tensor, ...], *, aliases: bool
    ) -> None:
        """Check returned geometry against the state-owned snapshots."""
        output_index = 3
        if return_distances:
            returned_distances = output[output_index]
            output_index += 1
            assert state.neighbor_distances is not None
            torch.testing.assert_close(
                state.neighbor_distances, returned_distances.detach()
            )
            if aliases:
                assert returned_distances is state.neighbor_distances
            else:
                assert returned_distances is not state.neighbor_distances
                assert (
                    returned_distances.data_ptr() != state.neighbor_distances.data_ptr()
                )
            assert not state.neighbor_distances.requires_grad
            assert state.neighbor_distances.grad_fn is None
        if return_vectors:
            returned_vectors = output[output_index]
            assert state.neighbor_vectors is not None
            torch.testing.assert_close(
                state.neighbor_vectors, returned_vectors.detach()
            )
            if aliases:
                assert returned_vectors is state.neighbor_vectors
            else:
                assert returned_vectors is not state.neighbor_vectors
                assert returned_vectors.data_ptr() != state.neighbor_vectors.data_ptr()
            assert not state.neighbor_vectors.requires_grad
            assert state.neighbor_vectors.grad_fn is None

    def gradients_for(
        runner,
        values: torch.Tensor,
        box: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Run one differentiable call and consume its graph."""
        output = runner(values, box)
        gradients = torch.autograd.grad(geometry_loss(output), (values, box))
        return output, gradients

    grad_positions = positions.clone().requires_grad_(True)
    grad_cell = cell.clone().requires_grad_(True)
    output, gradients = gradients_for(run, grad_positions, grad_cell)
    assert_geometry_values(output, grad_positions, grad_cell)
    assert_snapshot_contract(output, aliases=False)

    reference_positions = positions.clone().requires_grad_(True)
    reference_cell = cell.clone().requires_grad_(True)
    reference_output, reference_gradients = gradients_for(
        direct, reference_positions, reference_cell
    )
    assert _matrix_records(output) == _matrix_records(reference_output)
    torch.testing.assert_close(gradients, reference_gradients)

    warning_count = len(recwarn)
    no_grad_positions = positions + 0.03
    no_grad_cell = cell * 1.01
    with torch.no_grad():
        no_grad_output = run(no_grad_positions, no_grad_cell)
    assert_geometry_values(no_grad_output, no_grad_positions, no_grad_cell)
    assert_snapshot_contract(no_grad_output, aliases=True)
    assert all(not tensor.requires_grad for tensor in no_grad_output[3:])
    assert all(
        "not a leaf Tensor" not in str(warning.message)
        for warning in recwarn.list[warning_count:]
    )

    final_positions = (positions + 0.06).requires_grad_(True)
    final_cell = (cell * 0.99).requires_grad_(True)
    final_output, final_gradients = gradients_for(run, final_positions, final_cell)
    assert_geometry_values(final_output, final_positions, final_cell)
    assert_snapshot_contract(final_output, aliases=False)

    final_reference_positions = (positions + 0.06).requires_grad_(True)
    final_reference_cell = (cell * 0.99).requires_grad_(True)
    final_reference_output, final_reference_gradients = gradients_for(
        direct, final_reference_positions, final_reference_cell
    )
    assert _matrix_records(final_output) == _matrix_records(final_reference_output)
    torch.testing.assert_close(final_gradients, final_reference_gradients)


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
        return _prepared_neighbor_list(values, box, state)

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
    assert not state.neighbor_vectors.requires_grad
    assert state.neighbor_vectors.grad_fn is None
    assert not state.neighbor_distances.requires_grad
    assert state.neighbor_distances.grad_fn is None
    gradients = torch.autograd.grad(
        distances.sum(),
        (grad_positions, grad_cell),
    )
    assert all(torch.isfinite(value).all() for value in gradients)


@pytest.mark.gpu
@pytest.mark.slow
def test_prepared_batch_exact_coo_geometry_uses_cached_partition() -> None:
    """Prepared batch COO geometry stays current with cached atom ownership."""
    positions, cell, batch_ptr = _inputs(True)
    assert batch_ptr is not None
    state = prepare_cluster_tile(
        positions,
        1.5,
        cell,
        format="coo",
        batch_ptr=batch_ptr,
        max_neighbors=32,
        max_pairs=2048,
        return_vectors=True,
        return_distances=True,
        max_tiles_per_group=4,
    )
    metadata = state._partition_metadata
    assert metadata is not None
    metadata_tensors = (
        metadata.atom_system,
        metadata.batch_ptr_padded,
        metadata.padded_slot_system,
        metadata.real_sorted_rank_to_padded_slot,
        metadata.group_ptr,
        metadata.group_system,
    )
    metadata_values = tuple(value.clone() for value in metadata_tensors)
    metadata_pointers = tuple(value.data_ptr() for value in metadata_tensors)

    def call(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _prepared_neighbor_list(values, box, state)

    torch.compiler.reset()
    run = torch.compile(call, fullgraph=True)

    def assert_current_geometry(
        output: tuple[torch.Tensor, ...],
        values: torch.Tensor,
        box: torch.Tensor,
    ) -> None:
        """Check exact geometry against current batched inputs."""
        pairs, pointer, shifts, distances, vectors = output
        _coo_records((pairs, pointer, shifts))
        pair_system = metadata.atom_system[pairs[0].long()].long()
        expected_vectors = values[pairs[1].long()] - values[pairs[0].long()]
        expected_vectors = expected_vectors + torch.einsum(
            "pa,pab->pb", shifts.to(values.dtype), box[pair_system]
        )
        torch.testing.assert_close(vectors, expected_vectors)
        torch.testing.assert_close(distances, expected_vectors.norm(dim=-1))
        count = pairs.shape[1]
        assert state.neighbor_vectors is not None
        assert state.neighbor_distances is not None
        torch.testing.assert_close(state.neighbor_vectors[:count], vectors.detach())
        torch.testing.assert_close(state.neighbor_distances[:count], distances.detach())
        assert vectors.data_ptr() != state.neighbor_vectors.data_ptr()
        assert distances.data_ptr() != state.neighbor_distances.data_ptr()

    grad_positions = positions.clone().requires_grad_(True)
    grad_cell = cell.clone().requires_grad_(True)
    output = run(grad_positions, grad_cell)
    assert_current_geometry(output, grad_positions, grad_cell)
    gradients = torch.autograd.grad(
        output[3].sum() + output[4].square().sum(),
        (grad_positions, grad_cell),
    )

    reference_positions = positions.clone().requires_grad_(True)
    reference_cell = cell.clone().requires_grad_(True)
    reference = batch_cluster_tile_neighbor_list(
        reference_positions,
        1.5,
        reference_cell,
        batch_ptr,
        format="coo",
        max_neighbors=32,
        max_pairs=2048,
        max_tiles_per_group=4,
        return_vectors=True,
        return_distances=True,
    )
    reference_gradients = torch.autograd.grad(
        reference[3].sum() + reference[4].square().sum(),
        (reference_positions, reference_cell),
    )
    assert _coo_records(output[:3]) == _coo_records(reference[:3])
    torch.testing.assert_close(gradients, reference_gradients)

    changed_positions = (positions * 0.97).requires_grad_(True)
    changed_cell = (cell * 1.03).requires_grad_(True)
    changed_output = run(changed_positions, changed_cell)
    assert_current_geometry(changed_output, changed_positions, changed_cell)
    changed_gradients = torch.autograd.grad(
        changed_output[3].sum() + changed_output[4].square().sum(),
        (changed_positions, changed_cell),
    )
    assert all(torch.isfinite(value).all() for value in changed_gradients)
    assert state._partition_metadata is metadata
    assert tuple(value.data_ptr() for value in metadata_tensors) == metadata_pointers
    for value, expected_value in zip(metadata_tensors, metadata_values):
        assert torch.equal(value, expected_value)


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
    prepared_batch_ptr = batch_ptr.clone()
    batch_ptr.fill_(5)
    metadata = state._partition_metadata
    assert metadata is not None
    metadata_tensors = (
        metadata.atom_system,
        metadata.batch_ptr_padded,
        metadata.padded_slot_system,
        metadata.real_sorted_rank_to_padded_slot,
        metadata.group_ptr,
        metadata.group_system,
    )
    metadata_values = tuple(value.clone() for value in metadata_tensors)
    metadata_pointers = tuple(value.data_ptr() for value in metadata_tensors)
    _prepared_neighbor_list(positions, cell, state)

    changed_positions = positions.clone()
    changed_positions[1, 0] = 0.9
    changed_positions[3, 1] = 0.7
    changed_cell = cell.clone()
    changed_cell[2, 0, 0] = 9.0
    actual = _prepared_neighbor_list(
        changed_positions,
        changed_cell,
        state,
    )
    expected = batch_cluster_tile_neighbor_list(
        changed_positions,
        1.2,
        changed_cell,
        prepared_batch_ptr,
        format="matrix",
        max_neighbors=8,
        max_tiles_per_group=2,
    )
    _assert_same(actual, expected, format="matrix", batched=True)
    assert state._partition_metadata is metadata
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
    first_output = _prepared_neighbor_list(positions, cell, first)
    second_output = _prepared_neighbor_list(positions, cell, second)
    reused_output = _prepared_neighbor_list(positions * 0.9, cell, first)
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
    first_coo = _prepared_neighbor_list(positions, cell, coo_state)
    second_coo = _prepared_neighbor_list(positions, cell, coo_state)
    assert first_coo[0].data_ptr() != second_coo[0].data_ptr()


@pytest.mark.gpu
def test_preparation_owns_default_capacities() -> None:
    """Preparation estimates omitted capacities and owns the resulting buffers."""
    positions, cell, _ = _inputs(False)
    state = prepare_cluster_tile(positions, 1.2, cell, format="coo")
    output = _prepared_neighbor_list(positions, cell, state)
    assert state.max_neighbors >= 32
    assert state.max_pairs == positions.shape[0] * state.max_neighbors
    assert state.max_tiles_per_group > 0
    _coo_records(output)


@pytest.mark.gpu
def test_selective_single_matrix_initializes_and_preserves() -> None:
    """A single matrix must rebuild once before false can preserve it."""
    positions, cell, _ = _inputs(False)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        selective=True,
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    false = torch.zeros(1, dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError, match="cannot preserve uninitialized"):
        _prepared_neighbor_list(
            positions,
            cell,
            state,
            rebuild_flags=false,
        )

    true = torch.ones(1, dtype=torch.bool, device="cuda")
    initial = _prepared_neighbor_list(
        positions,
        cell,
        state,
        rebuild_flags=true,
    )
    snapshot = tuple(value.clone() for value in initial)
    preserved = _prepared_neighbor_list(
        positions * 0.5,
        cell,
        state,
        rebuild_flags=false,
    )
    assert all(torch.equal(value, saved) for value, saved in zip(preserved, snapshot))


@pytest.mark.gpu
def test_selective_partial_batch_rebuild_preserves_false_rows() -> None:
    """Eager selective batching changes only the flagged system rows."""
    positions, cell, batch_ptr = _inputs(True)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        batch_ptr=batch_ptr,
        selective=True,
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    all_true = torch.ones(2, dtype=torch.bool, device="cuda")
    initial = _prepared_neighbor_list(
        positions,
        cell,
        state,
        rebuild_flags=all_true,
    )
    snapshot = tuple(value.clone() for value in initial)
    changed = positions.clone()
    changed[17:] *= 0.5
    mixed = torch.tensor([False, True], dtype=torch.bool, device="cuda")
    output = _prepared_neighbor_list(
        changed,
        cell,
        state,
        rebuild_flags=mixed,
    )
    for value, saved in zip(output, snapshot):
        assert torch.equal(value[:17], saved[:17])
    expected = _direct(changed, cell, batch_ptr, format="matrix")
    assert _matrix_records(output)[17:] == _matrix_records(expected)[17:]


@pytest.mark.gpu
@pytest.mark.slow
def test_selective_dual_matrix_fullgraph_preserves_false_rows() -> None:
    """Compiled selective dual cutoff preserves both unflagged triples."""
    positions, cell, batch_ptr = _inputs(True)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        batch_ptr=batch_ptr,
        selective=True,
        cutoff2=1.6,
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    all_true = torch.ones(2, dtype=torch.bool, device="cuda")
    initial = _prepared_neighbor_list(
        positions,
        cell,
        state,
        rebuild_flags=all_true,
    )
    snapshot = tuple(value.clone() for value in initial)
    changed = positions.clone()
    changed[17:] *= 0.5
    mixed = torch.tensor([False, True], dtype=torch.bool, device="cuda")

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor, flags: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _prepared_neighbor_list(
            values,
            cell,
            state,
            rebuild_flags=flags,
        )

    output = run(changed, mixed)
    torch.cuda.synchronize()
    for value, saved in zip(output, snapshot):
        assert torch.equal(value[:17], saved[:17])
    expected = _direct(
        changed,
        cell,
        batch_ptr,
        format="matrix",
        cutoff2=1.6,
    )
    for start in (0, 3):
        assert (
            _matrix_records(output, start)[17:] == _matrix_records(expected, start)[17:]
        )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("batched", [False, True])
def test_selective_compiled_all_false_preserves_all_state(batched: bool) -> None:
    """Ordinary compiled all-false execution leaves prepared storage unchanged."""
    positions, cell, batch_ptr = _inputs(batched)
    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        batch_ptr=batch_ptr,
        selective=True,
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    flags = torch.ones(state.num_systems, dtype=torch.bool, device="cuda")
    _prepared_neighbor_list(positions, cell, state, rebuild_flags=flags)

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor, rebuild: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _prepared_neighbor_list(
            values,
            cell,
            state,
            rebuild_flags=rebuild,
        )

    run(positions, flags)
    torch.cuda.synchronize()
    storage = (*state._scratch, *state._topology)
    pointers = tuple(value.data_ptr() for value in storage)
    snapshot = tuple(value.clone() for value in storage)
    flags.zero_()
    run(positions * 0.25, flags)
    torch.cuda.synchronize()
    assert tuple(value.data_ptr() for value in storage) == pointers
    assert all(torch.equal(value, saved) for value, saved in zip(storage, snapshot))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_prepared_matrix_cuda_graph_replay(
    batched: bool,
    dual: bool,
    selective: bool,
) -> None:
    """A warmed compiled matrix call observes new values on graph replay."""
    positions, cell, batch_ptr = _inputs(batched)
    live_positions = positions.clone()
    live_cell = cell.clone()
    state = prepare_cluster_tile(
        live_positions,
        1.2,
        live_cell,
        format="matrix",
        batch_ptr=batch_ptr,
        selective=selective,
        cutoff2=1.6 if dual else None,
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    flags = torch.ones(state.num_systems, dtype=torch.bool, device="cuda")
    if selective:
        _prepared_neighbor_list(
            live_positions,
            live_cell,
            state,
            rebuild_flags=flags,
        )

        @torch.compile(fullgraph=True)
        def run(
            values: torch.Tensor,
            box: torch.Tensor,
            rebuild: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            return _prepared_neighbor_list(
                values,
                box,
                state,
                rebuild_flags=rebuild,
            )

        def execute() -> tuple[torch.Tensor, ...]:
            return run(live_positions, live_cell, flags)

    else:

        @torch.compile(fullgraph=True)
        def run(
            values: torch.Tensor,
            box: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            return _prepared_neighbor_list(values, box, state)

        def execute() -> tuple[torch.Tensor, ...]:
            return run(live_positions, live_cell)

    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        execute()
    warm_stream.synchronize()

    if selective:
        flags.zero_()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_output = execute()

    topology_pointers = tuple(value.data_ptr() for value in state._topology)
    assert tuple(value.data_ptr() for value in graph_output) == topology_pointers

    changed = positions * 0.5
    live_positions.copy_(changed)
    if selective:
        flags.fill_(True)
    graph.replay()
    torch.cuda.synchronize()
    expected = _direct(
        changed,
        live_cell,
        batch_ptr,
        format="matrix",
        cutoff2=1.6 if dual else None,
    )
    _assert_same(
        graph_output,
        expected,
        format="matrix",
        batched=batched,
        dual=dual,
    )
    assert tuple(value.data_ptr() for value in graph_output) == topology_pointers

    if selective:
        snapshot = tuple(value.clone() for value in graph_output)
        live_positions.copy_(positions * 0.2)
        flags.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert all(
            torch.equal(value, saved) for value, saved in zip(graph_output, snapshot)
        )

    cell_probe_positions = positions.clone()
    cell_probe_positions[0] = torch.tensor([0.1, 0.0, 0.0], device="cuda")
    cell_probe_positions[1] = torch.tensor([7.9, 0.0, 0.0], device="cuda")
    if batched:
        cell_probe_positions[17] = torch.tensor([0.1, 0.0, 0.0], device="cuda")
        cell_probe_positions[18] = torch.tensor([7.9, 0.0, 0.0], device="cuda")
    live_positions.copy_(cell_probe_positions)
    live_cell.copy_(cell)
    if selective:
        flags.fill_(True)
    graph.replay()
    torch.cuda.synchronize()
    original_expected = _direct(
        live_positions,
        live_cell,
        batch_ptr,
        format="matrix",
        cutoff2=1.6 if dual else None,
    )
    _assert_same(
        graph_output,
        original_expected,
        format="matrix",
        batched=batched,
        dual=dual,
    )
    original_cell_records = _matrix_records(graph_output)

    changed_cell = cell * 1.25
    live_cell.copy_(changed_cell)
    graph.replay()
    torch.cuda.synchronize()
    expected = _direct(
        live_positions,
        changed_cell,
        batch_ptr,
        format="matrix",
        cutoff2=1.6 if dual else None,
    )
    _assert_same(
        graph_output,
        expected,
        format="matrix",
        batched=batched,
        dual=dual,
    )
    assert _matrix_records(graph_output) != original_cell_records

    if selective and batched:
        snapshot = tuple(value.clone() for value in graph_output)
        live_positions[:17].copy_(positions[:17] * 0.15)
        flags.copy_(torch.tensor([True, False], device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        expected = _direct(
            live_positions,
            live_cell,
            batch_ptr,
            format="matrix",
            cutoff2=1.6 if dual else None,
        )
        starts = (0, 3) if dual else (0,)
        for start in starts:
            assert (
                _matrix_records(graph_output, start)[:17]
                == _matrix_records(expected, start)[:17]
            )
            assert all(
                torch.equal(value[17:], saved[17:])
                for value, saved in zip(
                    graph_output[start : start + 3],
                    snapshot[start : start + 3],
                )
            )

        snapshot = tuple(value.clone() for value in graph_output)
        live_positions[17:].copy_(positions[17:] * 0.1)
        flags.copy_(torch.tensor([False, True], device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        expected = _direct(
            live_positions,
            live_cell,
            batch_ptr,
            format="matrix",
            cutoff2=1.6 if dual else None,
        )
        for start in starts:
            assert all(
                torch.equal(value[:17], saved[:17])
                for value, saved in zip(
                    graph_output[start : start + 3],
                    snapshot[start : start + 3],
                )
            )
            assert (
                _matrix_records(graph_output, start)[17:]
                == _matrix_records(expected, start)[17:]
            )

    assert tuple(value.data_ptr() for value in graph_output) == topology_pointers


@pytest.mark.gpu
@pytest.mark.slow
def test_direct_eager_prepared_capture_has_actionable_error() -> None:
    """Direct eager prepared capture directs callers to the compiled route."""
    result = _run_direct_eager_capture()
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "direct eager prepared execution cannot be captured" in output
    assert "torch.compile(fullgraph=True)" in output
    assert "matrix-topology callable" in output


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("batched", [False, True])
def test_compiled_prepared_singular_cell_asserts_on_device(batched: bool) -> None:
    """Compiled inverse validation rejects singular current cells asynchronously."""
    result = _run_singular_prepared_fullgraph(batched)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "SINGULAR_CALL_STARTED" in output
    assert "SINGULAR_CALL_RETURNED" not in output
    assert "non-singular" in output or "device-side assert" in output.lower()


@pytest.mark.gpu
def test_failed_selective_rebuild_invalidates_preservation() -> None:
    """A failed eager rebuild cannot expose partially overwritten topology."""
    positions = torch.arange(32, dtype=torch.float32, device="cuda")[
        :, None
    ] * torch.tensor([3.0, 0.0, 0.0], device="cuda")
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 100.0
    state = prepare_cluster_tile(
        positions,
        1.0,
        cell,
        format="matrix",
        selective=True,
        max_neighbors=1,
        max_tiles_per_group=1,
    )
    true = torch.ones(1, dtype=torch.bool, device="cuda")
    false = torch.zeros(1, dtype=torch.bool, device="cuda")
    _prepared_neighbor_list(
        positions,
        cell,
        state,
        rebuild_flags=true,
    )
    dense = torch.zeros_like(positions)
    with pytest.raises(NeighborOverflowError):
        _prepared_neighbor_list(
            dense,
            cell,
            state,
            rebuild_flags=true,
        )
    with pytest.raises(ValueError, match="cannot preserve uninitialized"):
        _prepared_neighbor_list(
            dense,
            cell,
            state,
            rebuild_flags=false,
        )


@pytest.mark.gpu
def test_selective_configuration_and_flags_are_restricted() -> None:
    """Selective preparation accepts only matrix topology and valid flags."""
    positions, cell, _ = _inputs(False)
    with pytest.raises(ValueError, match="matrix output only"):
        prepare_cluster_tile(
            positions,
            1.2,
            cell,
            format="coo",
            selective=True,
        )
    with pytest.raises(ValueError, match="matrix output only"):
        prepare_cluster_tile(
            positions,
            1.2,
            cell,
            format="tile",
            selective=True,
        )
    with pytest.raises(ValueError, match="does not support geometry"):
        prepare_cluster_tile(
            positions,
            1.2,
            cell,
            format="matrix",
            selective=True,
            return_vectors=True,
        )

    state = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        selective=True,
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    with pytest.raises(ValueError, match="requires rebuild_flags"):
        _prepared_neighbor_list(positions, cell, state)
    with pytest.raises(ValueError, match=r"shape \(num_systems,\)"):
        _prepared_neighbor_list(
            positions,
            cell,
            state,
            rebuild_flags=torch.ones(2, dtype=torch.bool, device="cuda"),
        )
    with pytest.raises(ValueError, match="bool tensor"):
        _prepared_neighbor_list(
            positions,
            cell,
            state,
            rebuild_flags=torch.ones(1, dtype=torch.int32, device="cuda"),
        )
    with pytest.raises(ValueError, match="prepared device"):
        _prepared_neighbor_list(
            positions,
            cell,
            state,
            rebuild_flags=torch.ones(1, dtype=torch.bool),
        )

    plain = prepare_cluster_tile(
        positions,
        1.2,
        cell,
        format="matrix",
        max_neighbors=32,
        max_tiles_per_group=4,
    )
    with pytest.raises(ValueError, match="requires a selective"):
        _prepared_neighbor_list(
            positions,
            cell,
            plain,
            rebuild_flags=torch.ones(1, dtype=torch.bool, device="cuda"),
        )


@pytest.mark.gpu
@pytest.mark.slow
def test_selective_fullgraph_false_before_init_is_isolated() -> None:
    """Compiled preservation before initialization fails without returning."""
    result = _run_uninitialized_selective_fullgraph()
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "SELECTIVE_CALL_STARTED" in output
    assert "SELECTIVE_CALL_RETURNED" not in output
    assert (
        "cannot preserve uninitialized" in output
        or "device-side assert" in output.lower()
    )


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
        _prepared_neighbor_list(positions[:-1], cell, state)
    with pytest.raises(TypeError, match="positions dtype"):
        _prepared_neighbor_list(positions.double(), cell, state)
    with pytest.raises(ValueError, match="positions device"):
        _prepared_neighbor_list(positions.cpu(), cell, state)
    with pytest.raises(ValueError, match="cell shape"):
        _prepared_neighbor_list(positions, cell.unsqueeze(0), state)
    with pytest.raises(TypeError, match="cell dtype"):
        _prepared_neighbor_list(positions, cell.double(), state)
    with pytest.raises(ValueError, match="cell device"):
        _prepared_neighbor_list(positions, cell.cpu(), state)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "values",
    ([1, 16, 32], [0, 16, 31], [0, 24, 16, 32], [0, -1, 32]),
)
def test_prepared_rejects_invalid_batch_partitions(values: list[int]) -> None:
    """Preparation rejects invalid cumulative batch partitions."""
    positions, cell, _ = _inputs(False)
    batch_ptr = torch.tensor(values, dtype=torch.int32, device="cuda")
    cell_batch = cell.repeat(len(values) - 1, 1, 1)
    with pytest.raises(
        ValueError,
        match=(
            "batch_ptr must start at 0, end at positions.shape\\[0\\], "
            "and be non-decreasing"
        ),
    ):
        prepare_cluster_tile(
            positions,
            1.2,
            cell_batch,
            format="matrix",
            batch_ptr=batch_ptr,
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "kind",
    ("dtype", "device", "rank", "length"),
)
def test_prepared_rejects_invalid_batch_partition_structure(
    kind: str,
) -> None:
    """Preparation enforces the documented batch pointer representation."""
    positions, cell, _ = _inputs(False)
    if kind == "dtype":
        batch_ptr = torch.tensor([0, 16, 32], dtype=torch.int64, device="cuda")
    elif kind == "device":
        batch_ptr = torch.tensor([0, 16, 32], dtype=torch.int32)
    elif kind == "rank":
        batch_ptr = torch.tensor([[0, 16, 32]], dtype=torch.int32, device="cuda")
    else:
        batch_ptr = torch.tensor([0], dtype=torch.int32, device="cuda")
    with pytest.raises(
        ValueError,
        match="batch_ptr must be a CUDA int32 tensor with shape",
    ):
        prepare_cluster_tile(
            positions,
            1.2,
            cell.repeat(2, 1, 1),
            format="matrix",
            batch_ptr=batch_ptr,
        )


def test_prepared_api_has_no_caller_owned_storage_or_pair_callback() -> None:
    """Prepared routing preserves the public positional and state signatures."""
    parameters = set(inspect.signature(prepare_cluster_tile).parameters)
    assert parameters == {
        "positions",
        "cutoff",
        "cell",
        "format",
        "batch_ptr",
        "selective",
        "max_neighbors",
        "fill_value",
        "max_pairs",
        "cutoff2",
        "return_vectors",
        "return_distances",
        "max_tiles_per_group",
    }
    single = inspect.signature(cluster_tile_neighbor_list).parameters
    batch = inspect.signature(batch_cluster_tile_neighbor_list).parameters
    assert single["cutoff"].default is None
    assert single["cell"].default is None
    assert single["state"].kind is inspect.Parameter.KEYWORD_ONLY
    assert batch["cutoff"].default is None
    assert batch["cell_batch"].default is None
    assert batch["batch_ptr"].default is None
    assert batch["max_tiles_per_group"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert batch["state"].kind is inspect.Parameter.KEYWORD_ONLY


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
    output = _prepared_neighbor_list(positions, cell, state)
    if format == "tile":
        assert int(output[0].item()) == 0
    elif format == "matrix":
        assert output[0].shape == (0, 8)
        assert output[1].numel() == 0
    else:
        assert output[0].shape == (2, 0)
        assert output[1].tolist() == [0]
