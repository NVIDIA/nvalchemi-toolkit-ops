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

"""Compiler-boundary tests for coherent Torch prepared neighbor-list routes."""

import subprocess
import sys

import pytest
import torch

from nvalchemiops.torch.neighbors import (
    batch_cluster_tile_neighbor_list,
    cluster_tile_neighbor_list,
    neighbor_list,
    prepare_neighbor_list,
)
from test.neighbors.test_utils import assert_neighbor_matrix_equal

from .prepared_test_helpers import inputs as _inputs
from .prepared_test_helpers import prepare as _prepare


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batched", [False, True])
def test_prepared_cluster_selective_coo_support_matrix(batched: bool) -> None:
    """Selective COO is compiled for one system and eager-only for batches."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda").repeat(2, 1, 1) * 20.0
    pbc = torch.ones((2, 3), dtype=torch.bool, device="cuda")
    batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    if not batched:
        positions = positions[:2]
        cell = cell[:1]
        pbc = pbc[:1]
        batch_ptr = None
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_cluster_tile" if batched else "cluster_tile",
        format="coo",
        return_neighbor_list=True,
        selective=True,
        max_neighbors=8,
        max_tiles_per_group=1,
    )
    assert state.supports_compilation is not batched
    result = neighbor_list(
        positions,
        cell=cell,
        state=state,
        rebuild_flags=torch.ones(2 if batched else 1, dtype=torch.bool, device="cuda"),
    )
    assert len(result) == 4
    assert state.pair_offsets is result[1]
    assert state.pair_counts is result[2]
    preserved = neighbor_list(
        positions,
        cell=cell,
        state=state,
        rebuild_flags=torch.zeros(2 if batched else 1, dtype=torch.bool, device="cuda"),
    )
    assert preserved[0] is result[0]
    assert preserved[1] is state.pair_offsets
    assert preserved[2] is state.pair_counts
    assert preserved[3] is result[3]
    if not batched:

        @torch.compile(fullgraph=True)
        def run(
            values: torch.Tensor, box: torch.Tensor, flags: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            return neighbor_list(values, cell=box, state=state, rebuild_flags=flags)

        compiled = run(
            positions,
            cell,
            torch.ones(1, dtype=torch.bool, device="cuda"),
        )
        assert len(compiled) == 4


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_fullgraph_mirrors_6_to_2_to_0_and_preserves_old_results() -> None:
    """One compiled callable handles changing COO sizes and state references."""
    close, middle, far, cell = _inputs()
    close.requires_grad_(True)
    cell.requires_grad_(True)
    state = _prepare(close, cell)

    @torch.compile(fullgraph=True)
    def legacy(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return cluster_tile_neighbor_list(
            values,
            1.0,
            box,
            max_neighbors=8,
            max_pairs=32,
            format="coo",
            max_tiles_per_group=1,
            return_vectors=True,
            return_distances=True,
        )

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return neighbor_list(values, cell=box, state=state)

    legacy(close, cell)
    legacy(middle, cell)
    legacy(far, cell)
    first = run(close, cell)
    geometry_loss = first[-2].sum() + first[-1].square().sum()
    geometry_loss.backward()
    assert close.grad is not None and torch.isfinite(close.grad).all()
    assert cell.grad is not None and torch.isfinite(cell.grad).all()
    first_values = tuple(value.detach().clone() for value in first)
    second = run(middle, cell)
    assert second[0].shape == (2, 2)
    assert first[0] is not second[0]
    assert state.neighbor_list is second[0]
    assert state.neighbor_ptr is second[1]
    third = run(far, cell)
    assert third[0].shape == (2, 0)
    assert state.neighbor_list is third[0]
    assert state.neighbor_ptr is third[1]
    for actual, expected in zip(first, first_values):
        torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_naive_dual_fullgraph_reuses_larger_cutoff_shift_storage() -> None:
    """Dual naive fullgraph execution uses coverage sized for the larger cutoff."""
    positions = torch.tensor(
        [[0.1, 0.1, 0.1], [3.9, 0.1, 0.1]], dtype=torch.float32, device="cuda"
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 4.0
    state = prepare_neighbor_list(
        positions,
        0.5,
        cutoff2=0.9,
        cell=cell,
        pbc=torch.ones(3, dtype=torch.bool, device="cuda"),
        method="naive",
        max_neighbors=32,
        max_neighbors2=32,
    )

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return neighbor_list(values, cell=box, state=state)

    first = run(positions, cell)
    updated_cell = torch.eye(3, dtype=torch.float32, device="cuda") * 4.5
    second = run(positions, updated_cell)
    expected = neighbor_list(
        positions,
        0.5,
        cutoff2=0.9,
        cell=updated_cell,
        pbc=torch.ones(3, dtype=torch.bool, device="cuda"),
        method="naive_dual_cutoff",
        max_neighbors1=32,
        max_neighbors2=32,
    )
    for actual, reference in zip(second, expected, strict=True):
        torch.testing.assert_close(actual, reference)
    assert first[0] is state.neighbor_matrix1
    assert second[3] is state.neighbor_matrix2


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_naive_compiled_coverage_failure_isolated_process() -> None:
    """Compiled insufficient periodic-image coverage fails in a fresh process."""
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            """
import torch
from nvalchemiops.torch.neighbors import neighbor_list, prepare_neighbor_list

positions = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], device='cuda')
prepared_cell = torch.eye(3, device='cuda') * 4.0
runtime_cell = torch.eye(3, device='cuda') * 0.5
state = prepare_neighbor_list(
    positions, 0.7, cell=prepared_cell,
    pbc=torch.ones(3, dtype=torch.bool, device='cuda'),
    method='naive', max_neighbors=64,
)

@torch.compile(fullgraph=True)
def run(values, box):
    return neighbor_list(values, cell=box, state=state)

run(positions, runtime_cell)
torch.cuda.synchronize()
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "periodic-image coverage" in result.stderr


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_naive_guard_preserves_geometry_backward() -> None:
    """The periodic coverage guard preserves eager coordinate gradients."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=torch.float32, device="cuda"
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
    prepared_positions = positions.clone().requires_grad_()
    state = prepare_neighbor_list(
        prepared_positions,
        1.0,
        cell=cell,
        pbc=torch.ones(3, dtype=torch.bool, device="cuda"),
        method="naive",
        max_neighbors=8,
        return_vectors=True,
        return_distances=True,
    )

    prepared = neighbor_list(prepared_positions, cell=cell, state=state)
    (prepared[-2].square().sum() + prepared[-1].sum()).backward()
    prepared_grad = prepared_positions.grad.detach().clone()
    direct_positions = positions.clone().requires_grad_()
    direct = neighbor_list(
        direct_positions,
        1.0,
        cell=cell,
        pbc=torch.ones(3, dtype=torch.bool, device="cuda"),
        method="naive",
        max_neighbors=8,
        return_vectors=True,
        return_distances=True,
    )
    (direct[-2].square().sum() + direct[-1].sum()).backward()
    torch.testing.assert_close(prepared_grad, direct_positions.grad)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_single_cluster_coo_fullgraph_accepts_batched_cell_shape() -> None:
    """Single prepared COO compilation accepts the public ``(1, 3, 3)`` cell form."""
    close, _, _, cell = _inputs()
    cell = cell.unsqueeze(0)
    state = _prepare(close, cell)

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return neighbor_list(values, cell=box, state=state)

    result = run(close, cell)
    assert result[0].shape == (2, 6)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_synthesized_batch_cell_list_fullgraph_and_validation_is_recoverable() -> (
    None
):
    """Synthesized batched cells compile and preflight errors preserve results."""
    positions = torch.rand((8, 3), dtype=torch.float32, device="cuda")
    state = prepare_neighbor_list(
        positions,
        1.0,
        method="batch_cell_list",
        batch_ptr=torch.tensor([0, 3, 8], dtype=torch.int32, device="cuda"),
        max_neighbors=8,
    )

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return neighbor_list(values, state=state)

    first = run(positions)
    with pytest.raises(ValueError, match="positions shape"):
        neighbor_list(positions[:-1], state=state)
    assert state.initialized.all()
    assert state.neighbor_matrix is first[0]
    assert state.num_neighbors is first[1]
    assert state.neighbor_matrix_shifts is first[2]
    assert run(positions)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_batched_cluster_tile_fullgraph_smoke() -> None:
    """The batched cluster-tile state supports fullgraph execution."""
    positions = torch.rand((32, 3), dtype=torch.float32, device="cuda") * 4.0
    cell = torch.eye(3, dtype=torch.float32, device="cuda").repeat(2, 1, 1) * 5.0
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=torch.ones((2, 3), dtype=torch.bool, device="cuda"),
        batch_ptr=torch.tensor([0, 16, 32], dtype=torch.int32, device="cuda"),
        method="batch_cluster_tile",
        format="matrix",
        max_neighbors=8,
        max_tiles_per_group=1,
    )
    assert state.supports_compilation and state.compilation_blocker is None

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return neighbor_list(values, cell=box, state=state)

    result = run(positions, cell)
    assert result[0].shape == (32, 8)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_batched_cluster_coo_fullgraph_publishes_tile_segments() -> None:
    """Batched compact COO keeps its tile segments inside the compiled boundary."""
    positions = torch.rand((32, 3), dtype=torch.float32, device="cuda") * 4.0
    cell = torch.eye(3, dtype=torch.float32, device="cuda").repeat(2, 1, 1) * 5.0
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=torch.ones((2, 3), dtype=torch.bool, device="cuda"),
        batch_ptr=torch.tensor([0, 16, 32], dtype=torch.int32, device="cuda"),
        method="batch_cluster_tile",
        format="coo",
        return_neighbor_list=True,
        max_pairs=256,
        max_tiles_per_group=1,
    )
    assert state.supports_compilation and state.compilation_blocker is None

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return neighbor_list(values, cell=box, state=state)

    result = run(positions, cell)
    assert result[0].shape[0] == 2
    assert state.tile_offsets is not None
    assert state.tile_counts is not None
    assert state.tile_offsets.shape == (3,)
    assert state.tile_counts.shape == (2,)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_batched_cluster_coo_geometry_fullgraph() -> None:
    """Batched compact COO geometry outputs stay within the compiled boundary."""
    positions = torch.rand((32, 3), dtype=torch.float32, device="cuda") * 4.0
    cell = torch.eye(3, dtype=torch.float32, device="cuda").repeat(2, 1, 1) * 5.0
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=torch.ones((2, 3), dtype=torch.bool, device="cuda"),
        batch_ptr=torch.tensor([0, 16, 32], dtype=torch.int32, device="cuda"),
        method="batch_cluster_tile",
        format="coo",
        return_neighbor_list=True,
        return_vectors=True,
        max_pairs=256,
        max_tiles_per_group=1,
    )
    assert state.supports_compilation and state.compilation_blocker is None
    assert state.neighbor_vectors is not None
    vectors = state.neighbor_vectors
    vectors.fill_(torch.nan)

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return neighbor_list(values, cell=box, state=state)

    result = run(positions, cell)
    assert len(result) == 4
    assert torch.isfinite(state.neighbor_vectors).any()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_cluster_matrix_cuda_graph_replay() -> None:
    """A warmed general prepared matrix route supports CUDA Graph replay."""
    positions = torch.rand((32, 3), dtype=torch.float32, device="cuda") * 4.0
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=torch.ones(3, dtype=torch.bool, device="cuda"),
        method="cluster_tile",
        format="matrix",
        max_neighbors=8,
        max_tiles_per_group=1,
    )

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return neighbor_list(values, cell=box, state=state)

    run(positions, cell)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(positions, cell)
    graph.replay()
    torch.cuda.synchronize()
    assert captured[0] is state.neighbor_matrix
    assert captured[1] is state.num_neighbors
    assert int(captured[1].sum()) >= 0


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("batched", "dual_cutoff", "selective"),
    [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
)
def test_prepared_cluster_matrix_fullgraph_cuda_graph_contract(
    batched: bool, dual_cutoff: bool, selective: bool
) -> None:
    """Compiled prepared matrix variants replay through a CUDA Graph."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0], [1.2, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    prepare_kwargs: dict[str, object] = {}
    if batched:
        positions = torch.cat((positions, positions + 2.0))
        cell = cell.repeat(2, 1, 1)
        pbc = torch.ones((2, 3), dtype=torch.bool, device="cuda")
        prepare_kwargs["batch_ptr"] = torch.tensor(
            [0, 4, 8], dtype=torch.int32, device="cuda"
        )
    if dual_cutoff:
        prepare_kwargs["cutoff2"] = 0.7
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        method="batch_cluster_tile" if batched else "cluster_tile",
        format="matrix",
        selective=selective,
        max_neighbors=8,
        max_tiles_per_group=1,
        **prepare_kwargs,
    )
    flags = torch.ones(2 if batched else 1, dtype=torch.bool, device="cuda")

    triples = ((0, 1, 2), (3, 4, 5)) if dual_cutoff else ((0, 1, 2),)

    def assert_rows(
        actual: tuple[torch.Tensor, ...],
        expected: tuple[torch.Tensor, ...],
        rows: slice,
    ) -> None:
        for matrix, counts, shifts in triples:
            assert_neighbor_matrix_equal(
                (actual[matrix][rows], actual[counts][rows], actual[shifts][rows]),
                (
                    expected[matrix][rows],
                    expected[counts][rows],
                    expected[shifts][rows],
                ),
            )

    def direct() -> tuple[torch.Tensor, ...]:
        kwargs = dict(
            format="matrix",
            max_neighbors=8,
            max_tiles_per_group=1,
            cutoff2=0.7 if dual_cutoff else None,
        )
        if batched:
            return batch_cluster_tile_neighbor_list(
                positions, 1.0, cell, prepare_kwargs["batch_ptr"], **kwargs
            )
        return cluster_tile_neighbor_list(positions, 1.0, cell, **kwargs)

    if selective:

        @torch.compile(fullgraph=True)
        def run(
            values: torch.Tensor, box: torch.Tensor, current_flags: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            return neighbor_list(
                values, cell=box, state=state, rebuild_flags=current_flags
            )

        run(positions, cell, flags)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(positions, cell, flags)
        baseline = tuple(value.clone() for value in captured)
        flags.zero_()
        positions[0, 0].add_(2.0)
        cell[..., 0, 0].mul_(1.1)
        graph.replay()
        torch.cuda.synchronize()
        assert_rows(captured, baseline, slice(None))
        flags[0] = True
        positions[0, 0].add_(0.2)
        cell[..., 0, 0].mul_(1.1)
        graph.replay()
        torch.cuda.synchronize()
        selected = slice(0, 4) if batched else slice(None)
        assert_rows(captured, direct(), selected)
        if batched:
            assert_rows(captured, baseline, slice(4, None))
    else:

        @torch.compile(fullgraph=True)
        def run(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return neighbor_list(values, cell=box, state=state)

        run(positions, cell)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(positions, cell)
        positions[0, 0].add_(0.2)
        cell[..., 0, 0].mul_(1.1)
        graph.replay()
        torch.cuda.synchronize()
        assert_rows(captured, direct(), slice(None))

    if dual_cutoff:
        assert captured[0] is state.neighbor_matrix1
        assert captured[1] is state.num_neighbors1
        assert captured[3] is state.neighbor_matrix2
        assert captured[4] is state.num_neighbors2
    else:
        assert captured[0] is state.neighbor_matrix
        assert captured[1] is state.num_neighbors
    assert int(captured[1].sum()) >= 0


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_cluster_cuda_graph_replay_reads_updated_boundary_cell() -> None:
    """CUDA Graph replay uses the current cell for minimum-image topology."""
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [4.9, 0.0, 0.0]], dtype=torch.float32, device="cuda"
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=torch.ones(3, dtype=torch.bool, device="cuda"),
        method="cluster_tile",
        format="matrix",
        max_neighbors=8,
        max_tiles_per_group=1,
    )

    @torch.compile(fullgraph=True)
    def run(values: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return neighbor_list(values, cell=box, state=state)

    run(positions, cell)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(positions, cell)
    baseline = tuple(value.clone() for value in captured)
    cell.copy_(torch.eye(3, dtype=torch.float32, device="cuda") * 10.0)
    graph.replay()
    torch.cuda.synchronize()
    direct = cluster_tile_neighbor_list(
        positions,
        1.0,
        cell,
        format="matrix",
        max_neighbors=8,
        max_tiles_per_group=1,
    )
    assert_neighbor_matrix_equal(captured, direct)
    assert not torch.equal(captured[1], baseline[1])


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("batched", "partial", "capture", "single_batched_cell"),
    [
        (False, False, False, True),
        (True, False, False, False),
        (True, True, False, False),
        (False, False, True, False),
    ],
)
def test_prepared_cluster_compiled_preservation_validates_current_cell(
    batched: bool, partial: bool, capture: bool, single_batched_cell: bool
) -> None:
    """Compiled preservation rejects singular current cells in an isolated CUDA process."""
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            f"""
import torch

from nvalchemiops.torch.neighbors import neighbor_list, prepare_neighbor_list

batched = {batched!r}
partial = {partial!r}
capture = {capture!r}
single_batched_cell = {single_batched_cell!r}
positions = torch.tensor(
    [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=torch.float32, device="cuda"
)
cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
pbc = torch.ones(3, dtype=torch.bool, device="cuda")
kwargs = {{}}
if batched:
    positions = torch.cat((positions, positions + 2.0))
    cell = cell.repeat(2, 1, 1)
    pbc = torch.ones((2, 3), dtype=torch.bool, device="cuda")
    kwargs["batch_ptr"] = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
elif single_batched_cell:
    cell = cell.unsqueeze(0)
state = prepare_neighbor_list(
    positions, 1.0, cell=cell, pbc=pbc,
    method="batch_cluster_tile" if batched else "cluster_tile",
    format="matrix", selective=True, max_neighbors=8, max_tiles_per_group=1,
    **kwargs,
)
flags = torch.ones(2 if batched else 1, dtype=torch.bool, device="cuda")

@torch.compile(fullgraph=True)
def run(values, box, current_flags):
    return neighbor_list(values, cell=box, state=state, rebuild_flags=current_flags)

run(positions, cell, flags)[1].cpu()
flags.zero_()
if partial:
    flags[0] = True
if capture:
    flags.fill_(True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(positions, cell, flags)
    captured[1].cpu()
    flags.zero_()
    cell.zero_()
    graph.replay()
    captured[1].cpu()
else:
    if batched:
        cell[-1].zero_()
    else:
        cell.zero_()
    run(positions, cell, flags)[1].cpu()
torch.cuda.synchronize()
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "cell matr" in result.stderr
    assert "must be non-singular" in result.stderr
