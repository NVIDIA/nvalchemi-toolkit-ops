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

"""Public-contract tests for coherent Torch prepared neighbor-list routes."""

import inspect
from itertools import product

import pytest
import torch

from nvalchemiops.torch.neighbors import (
    NeighborListState,
    NeighborOverflowError,
    neighbor_list,
    prepare_neighbor_list,
)
from nvalchemiops.torch.neighbors.batch_cluster_tile import (
    batch_cluster_tile_neighbor_list,
)
from nvalchemiops.torch.neighbors.cluster_tile import cluster_tile_neighbor_list

from ...test_utils import assert_neighbor_matrix_equal
from .prepared_test_helpers import inputs as _inputs
from .prepared_test_helpers import prepare as _prepare
from .prepared_test_helpers import prepared_sum_pair_fn as _prepared_sum_pair_fn


def _assert_cluster_route_equal(
    prepared: tuple[torch.Tensor, ...], direct: tuple[torch.Tensor, ...], format: str
) -> None:
    """Compare public cluster output without reading unspecified matrix tails."""
    assert len(prepared) == len(direct)
    if format == "matrix":
        for start in range(0, len(prepared), 3):
            assert_neighbor_matrix_equal(
                prepared[start : start + 3], direct[start : start + 3]
            )
        return
    if format == "tile":
        torch.testing.assert_close(prepared[0], direct[0])
        for got, expected in zip(
            prepared[3 if len(prepared) == 7 else 4 :],
            direct[3 if len(direct) == 7 else 4 :],
            strict=True,
        ):
            torch.testing.assert_close(got, expected)
        return
    for got, expected in zip(prepared, direct, strict=True):
        torch.testing.assert_close(got, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("format", ["matrix", "tile", "coo"])
@pytest.mark.parametrize("cutoff2", [None, 1.25])
@pytest.mark.parametrize("fixed_cell", [False, True])
def test_prepared_single_cluster_route_matches_direct(
    format: str, cutoff2: float | None, fixed_cell: bool
) -> None:
    """Supported single prepared cluster outputs match the direct route."""
    if cutoff2 is not None and format != "matrix":
        pytest.skip("dual cutoff is matrix-only")
    positions, _, _, cell = _inputs()
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    kwargs = dict(
        method="cluster_tile",
        format=format,
        return_neighbor_list=format == "coo",
        max_neighbors=8,
        max_pairs=32,
        max_tiles_per_group=1,
        cutoff2=cutoff2,
    )
    state = prepare_neighbor_list(
        positions, 1.0, cell=cell, pbc=pbc, fixed_cell=fixed_cell, **kwargs
    )
    prepared = neighbor_list(positions, cell=cell, state=state)
    direct = cluster_tile_neighbor_list(
        positions,
        1.0,
        cell,
        **{
            key: value
            for key, value in kwargs.items()
            if key not in {"method", "return_neighbor_list"}
        },
    )
    _assert_cluster_route_equal(prepared, direct, format)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("format", ["matrix", "tile", "coo"])
@pytest.mark.parametrize("cutoff2", [None, 1.25])
@pytest.mark.parametrize("fixed_cell", [False, True])
def test_prepared_batch_cluster_route_matches_direct(
    format: str, cutoff2: float | None, fixed_cell: bool
) -> None:
    """Supported batched prepared cluster outputs match the direct route."""
    if cutoff2 is not None and format != "matrix":
        pytest.skip("dual cutoff is matrix-only")
    positions, _, _, single_cell = _inputs()
    positions = torch.cat((positions[:2], positions[:2] + 4.0))
    batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    cell = single_cell.repeat(2, 1, 1)
    pbc = torch.ones((2, 3), dtype=torch.bool, device="cuda")
    kwargs = dict(
        method="batch_cluster_tile",
        format=format,
        return_neighbor_list=format == "coo",
        max_neighbors=8,
        max_pairs=32,
        max_tiles_per_group=1,
        cutoff2=cutoff2,
    )
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        fixed_cell=fixed_cell,
        **kwargs,
    )
    prepared = neighbor_list(positions, cell=cell, state=state)
    direct_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"method", "return_neighbor_list"}
    }
    if format == "coo":
        assert state.pair_offsets is not None
        direct_kwargs.update(
            neighbor_list=torch.empty_like(prepared[0]),
            neighbor_list_shifts=torch.empty_like(prepared[3]),
            pair_offsets=state.pair_offsets,
            pair_counts=torch.zeros_like(state.pair_counts),
        )
    direct = batch_cluster_tile_neighbor_list(
        positions,
        1.0,
        cell,
        batch_ptr,
        **direct_kwargs,
    )
    if format == "coo":
        torch.testing.assert_close(prepared[1], direct[1])
        torch.testing.assert_close(prepared[2], direct[2])
        for start, count in zip(prepared[1][:-1], prepared[2], strict=True):
            begin = int(start)
            end = begin + int(count)
            torch.testing.assert_close(
                prepared[0][:, begin:end], direct[0][:, begin:end]
            )
            torch.testing.assert_close(prepared[3][begin:end], direct[3][begin:end])
        return
    _assert_cluster_route_equal(prepared, direct, format)


def test_generic_public_workflow_exports() -> None:
    """The generic factory and state expose the prepared public workflow."""
    import nvalchemiops.torch.neighbors as neighbors

    assert hasattr(neighbors, "NeighborListState")
    assert hasattr(neighbors, "prepare_neighbor_list")
    assert "state" in inspect.signature(neighbor_list).parameters
    assert (
        inspect.signature(neighbor_list).parameters["state"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    with pytest.raises(TypeError, match="unexpected keyword argument 'coo_capacity'"):
        prepare_neighbor_list(
            torch.zeros((2, 3), dtype=torch.float32),
            1.0,
            method="naive",
            coo_capacity=4,
        )
    for name in (
        "method",
        "strategy",
        "format",
        "coo_layout",
        "is_batched",
        "num_atoms",
        "num_systems",
        "cutoff",
        "cutoff2",
        "half_fill",
        "fill_value",
        "wrap_positions",
        "selective",
        "return_vectors",
        "return_distances",
        "span_margin",
        "supports_compilation",
        "compilation_blocker",
        "initialized",
    ):
        assert hasattr(NeighborListState, name)

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for prepared state allocation")
    close, _, _, cell = _inputs()
    state = _prepare(close, cell)
    assert state.neighbor_list is None
    assert state.neighbor_ptr is None
    assert state.neighbor_list_shifts is None
    assert state.neighbor_vectors is None
    assert state.neighbor_distances is None
    assert state.neighbor_matrix is None
    assert state.neighbor_matrix1 is None
    assert state.neighbor_matrix2 is None
    assert state.pair_offsets is None
    assert state.pair_counts is None
    assert state.num_tiles is None
    assert state.tile_row_group is None
    assert state.tile_col_group is None
    assert state.sorted_atom_index is None
    assert not bool(state.initialized.any())

    neighbor_list(close, cell=cell, state=state)
    assert state.num_tiles is not None
    assert state.tile_row_group is not None
    assert state.tile_col_group is not None
    assert state.sorted_atom_index is not None


def test_public_contract_rejects_non_state_and_caller_storage() -> None:
    """Prepared routing validates state identity and caller-owned buffers."""
    positions = torch.empty((0, 3), dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float32)
    with pytest.raises(
        TypeError,
        match="state must be a NeighborListState returned by prepare_neighbor_list",
    ):
        neighbor_list(positions, cell=cell, state=object())

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for prepared state allocation")
    close, _, _, cell = _inputs()
    state = _prepare(close, cell, return_vectors=False, return_distances=False)
    with pytest.raises(
        ValueError,
        match="caller-owned output or scratch buffers cannot be used with state",
    ):
        neighbor_list(
            close,
            cell=cell,
            state=state,
            neighbor_list=torch.empty((2, 32), dtype=torch.int32, device="cuda"),
        )
    with pytest.raises(
        ValueError,
        match="caller-owned output or scratch buffers cannot be used with state",
    ):
        neighbor_list(
            close,
            cell=cell,
            state=state,
            neighbor_matrix1=torch.empty(
                (close.shape[0], 8), dtype=torch.int32, device="cuda"
            ),
        )
    assert neighbor_list(close, state=state)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        neighbor_list(close, cell=cell, state=state, unknown_state_option=True)
    assert neighbor_list(close, cell=cell, state=state)
    with pytest.raises(ValueError, match="positions shape does not match"):
        neighbor_list(close[:-1], cell=cell, state=state)
    assert neighbor_list(close, cell=cell, state=state)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_eager_mirrors_exact_coo_results() -> None:
    """Eager execution publishes every exact topology and geometry result."""
    close, middle, far, cell = _inputs()
    state = _prepare(close, cell)
    for positions, pair_count in ((close, 6), (middle, 2), (far, 0)):
        result = neighbor_list(positions, cell=cell, state=state)
        assert result[0].shape == (2, pair_count)
        assert result[1].shape == (4,)
        assert result[2].shape == (pair_count, 3)
        assert state.neighbor_list is result[0]
        assert state.neighbor_ptr is result[1]
        assert state.neighbor_list_shifts is result[2]
        assert state.neighbor_vectors is not None
        assert state.neighbor_distances is not None
        assert state.neighbor_vectors.shape == (pair_count, 3)
        assert state.neighbor_distances.shape == (pair_count,)
        assert state.initialized


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_matrix_pair_outputs_publish_state_buffers() -> None:
    """Matrix pair geometry and callback outputs remain visible on the state."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 20.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    pair_params = torch.arange(1, 4, dtype=torch.float32, device="cuda").reshape(-1, 1)
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        method="cluster_tile",
        max_neighbors=8,
        max_tiles_per_group=1,
        return_vectors=True,
        return_distances=True,
        pair_fn=_prepared_sum_pair_fn,
    )

    result = neighbor_list(
        positions,
        cell=cell,
        state=state,
        pair_params=pair_params,
    )

    assert len(result) == 3
    assert state.neighbor_matrix is result[0]
    assert state.num_neighbors is result[1]
    assert state.neighbor_matrix_shifts is result[2]
    assert state.neighbor_vectors is not None
    assert state.neighbor_distances is not None
    assert state.pair_energies is not None
    assert state.pair_forces is not None
    for source, count in enumerate(state.num_neighbors.tolist()):
        for slot in range(count):
            target = int(state.neighbor_matrix[source, slot])
            vector = positions[target] - positions[source]
            distance = vector.norm()
            torch.testing.assert_close(state.neighbor_vectors[source, slot], vector)
            torch.testing.assert_close(state.neighbor_distances[source, slot], distance)
            expected_energy = pair_params[source, 0] + pair_params[target, 0] + distance
            torch.testing.assert_close(
                state.pair_energies[source, slot], expected_energy
            )
            torch.testing.assert_close(state.pair_forces[source, slot], -vector)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("format", ["matrix", "coo"])
@pytest.mark.parametrize("fixed_cell", [False, True])
def test_prepared_cluster_callback_matches_direct_buffers(
    batched: bool, format: str, fixed_cell: bool
) -> None:
    """Prepared eager cluster callbacks match direct state-owned pair outputs."""
    positions, _, _, single_cell = _inputs()
    positions = positions[:2]
    cell = single_cell
    batch_ptr = None
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    if batched:
        positions = torch.cat((positions, positions + 4.0))
        cell = single_cell.repeat(2, 1, 1)
        batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
        pbc = torch.ones((2, 3), dtype=torch.bool, device="cuda")
    pair_params = torch.arange(
        positions.shape[0], dtype=torch.float32, device="cuda"
    ).reshape(-1, 1)
    common = dict(
        format=format,
        max_neighbors=8,
        max_pairs=32,
        max_tiles_per_group=1,
        pair_fn=_prepared_sum_pair_fn,
        pair_params=pair_params,
    )
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        fixed_cell=fixed_cell,
        method="batch_cluster_tile" if batched else "cluster_tile",
        return_neighbor_list=format == "coo",
        **{key: value for key, value in common.items() if key != "pair_params"},
    )
    prepared = neighbor_list(positions, cell=cell, state=state, pair_params=pair_params)
    pair_shape = (32,) if format == "coo" else (positions.shape[0], 8)
    energies = torch.empty(pair_shape, dtype=torch.float32, device="cuda")
    forces = torch.empty((*pair_shape, 3), dtype=torch.float32, device="cuda")
    direct_fn = (
        batch_cluster_tile_neighbor_list if batched else cluster_tile_neighbor_list
    )
    direct_kwargs: dict[str, object] = dict(
        pair_energies=energies,
        pair_forces=forces,
        **common,
    )
    if batched and format == "coo":
        assert state.pair_offsets is not None and state.pair_counts is not None
        direct_kwargs.update(
            neighbor_list=torch.empty_like(prepared[0]),
            neighbor_list_shifts=torch.empty_like(prepared[3]),
            pair_offsets=state.pair_offsets,
            pair_counts=torch.zeros_like(state.pair_counts),
        )
    direct = direct_fn(
        positions,
        1.0,
        cell,
        *(() if not batched else (batch_ptr,)),
        **direct_kwargs,
    )
    assert state.pair_energies is not None and state.pair_forces is not None
    if batched and format == "coo":
        torch.testing.assert_close(prepared[1], direct[1])
        torch.testing.assert_close(prepared[2], direct[2])
        for start, count in zip(prepared[1][:-1], prepared[2], strict=True):
            begin = int(start)
            end = begin + int(count)
            torch.testing.assert_close(
                prepared[0][:, begin:end], direct[0][:, begin:end]
            )
            torch.testing.assert_close(prepared[3][begin:end], direct[3][begin:end])
            torch.testing.assert_close(
                state.pair_energies[begin:end], energies[begin:end]
            )
            torch.testing.assert_close(state.pair_forces[begin:end], forces[begin:end])
    else:
        _assert_cluster_route_equal(prepared, direct, format)
    if format == "coo" and not batched:
        pair_count = prepared[0].shape[-1]
        torch.testing.assert_close(state.pair_energies, energies[:pair_count])
        torch.testing.assert_close(state.pair_forces, forces[:pair_count])
    elif format == "matrix":
        for source, count in enumerate(prepared[1].tolist()):
            torch.testing.assert_close(
                state.pair_energies[source, :count], energies[source, :count]
            )
            torch.testing.assert_close(
                state.pair_forces[source, :count], forces[source, :count]
            )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_cluster_tile_uses_current_torch_stream(torch_stream_runner) -> None:
    """Prepared single-system custom ops consume inputs from the caller stream."""
    source_positions, _, _, cell = _inputs()
    state = _prepare(source_positions, cell)
    target_positions = torch.empty_like(source_positions)

    def run(values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return neighbor_list(values, cell=cell, state=state)

    _actual, snapshot, expected = torch_stream_runner(
        source_positions,
        target_positions,
        run,
    )
    for result, reference in zip(snapshot, expected, strict=True):
        torch.testing.assert_close(result, reference)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_configuration_is_frozen_and_runtime_geometry_validated() -> None:
    """Configuration cannot be rebound while runtime geometry remains dynamic."""
    close, _, _, cell = _inputs()
    state = _prepare(close, cell, return_vectors=False, return_distances=False)
    with pytest.raises(AttributeError):
        state.cutoff = 2.0
    with pytest.raises(AttributeError):
        del state.cutoff
    assert state.cutoff == 1.0
    with pytest.raises(ValueError, match="positions shape"):
        neighbor_list(close[:-1], cell=cell, state=state)
    state = _prepare(close, cell, return_vectors=False, return_distances=False)
    with pytest.raises(ValueError, match="cell shape"):
        neighbor_list(close, cell=cell.unsqueeze(0), state=state)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("method", "extra", "expected_outputs"),
    [
        ("naive", {}, 3),
        ("naive", {"cutoff2": 1.5, "max_neighbors2": 8}, 6),
        ("cell_list", {}, 3),
        ("batch_naive", {}, 3),
        ("batch_cell_list", {}, 3),
    ],
)
def test_prepared_existing_non_tile_routes(
    method: str,
    extra: dict[str, object],
    expected_outputs: int,
) -> None:
    """Prepared routing preserves the existing non-cluster Torch route tuples."""
    positions = torch.rand((8, 3), dtype=torch.float32, device="cuda") * 4.0
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    kwargs: dict[str, object] = {"method": method, "max_neighbors": 8}
    kwargs.update(extra)
    if method.startswith("batch_"):
        kwargs.update(
            batch_ptr=torch.tensor([0, 3, 8], dtype=torch.int32, device="cuda"),
            cell=cell.repeat(2, 1, 1),
            pbc=pbc.repeat(2, 1),
        )
    else:
        kwargs.update(cell=cell, pbc=pbc)
    state = prepare_neighbor_list(positions, 1.0, **kwargs)
    result = neighbor_list(positions, cell=kwargs["cell"], state=state)
    assert len(result) == expected_outputs
    assert state.initialized.all()
    if "cutoff2" in extra:
        assert state.neighbor_matrix is None
        assert state.neighbor_matrix1 is result[0]
        assert state.num_neighbors1 is result[1]
        assert state.neighbor_matrix_shifts1 is result[2]
        assert state.neighbor_matrix2 is result[3]
        assert state.num_neighbors2 is result[4]
        assert state.neighbor_matrix_shifts2 is result[5]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batched", [False, True])
def test_prepared_cell_list_reuses_topology_storage(batched: bool) -> None:
    """Prepared cell-list calls reuse their state-owned matrix buffers."""
    positions = torch.tensor(
        [[0.1, 0.1, 0.1], [0.4, 0.1, 0.1], [2.1, 0.1, 0.1], [2.4, 0.1, 0.1]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
    kwargs: dict[str, object] = {
        "cell": cell,
        "pbc": torch.ones(3, dtype=torch.bool, device="cuda"),
        "method": "cell_list",
        "max_neighbors": 8,
    }
    if batched:
        kwargs.update(
            cell=cell.repeat(2, 1, 1),
            pbc=torch.ones((2, 3), dtype=torch.bool, device="cuda"),
            batch_ptr=torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda"),
            method="batch_cell_list",
        )
    state = prepare_neighbor_list(positions, 0.8, **kwargs)
    first = neighbor_list(positions, cell=kwargs["cell"], state=state)
    first_ids = tuple(id(value) for value in first)
    second = neighbor_list(positions + 0.02, cell=kwargs["cell"], state=state)
    assert tuple(id(value) for value in second) == first_ids
    assert state.neighbor_matrix is second[0]
    assert state.num_neighbors is second[1]
    assert state.neighbor_matrix_shifts is second[2]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("method", ["naive", "cell_list"])
def test_prepared_geometry_reuse_keeps_each_backward_connected(method: str) -> None:
    """Ordinary backward-before-reuse calls match independent route gradients."""
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 4.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    state = prepare_neighbor_list(
        torch.tensor(
            [[0.1, 1.0, 1.0], [0.5, 1.0, 1.0], [3.9, 1.0, 1.0]],
            dtype=torch.float32,
            device="cuda",
        ),
        0.75,
        cell=cell,
        pbc=pbc,
        method=method,
        max_neighbors=8,
        return_vectors=True,
        return_distances=True,
    )
    frames = (
        [[0.1, 1.0, 1.0], [0.5, 1.0, 1.0], [3.9, 1.0, 1.0]],
        [[0.2, 1.0, 1.0], [0.65, 1.0, 1.0], [3.8, 1.0, 1.0]],
    )
    for frame in frames:
        positions = torch.tensor(
            frame, dtype=torch.float32, device="cuda", requires_grad=True
        )
        runtime_cell = cell.detach().clone().requires_grad_()
        result = neighbor_list(positions, cell=runtime_cell, state=state)
        assert state.neighbor_distances is result[-2]
        assert state.neighbor_vectors is result[-1]
        active = result[0] != state.fill_value
        loss = result[-2][active].sum() + result[-1][active].square().sum()
        loss.backward()
        actual_position_grad = positions.grad.detach().clone()
        actual_cell_grad = runtime_cell.grad.detach().clone()

        reference_positions = positions.detach().clone().requires_grad_()
        reference_cell = runtime_cell.detach().clone().requires_grad_()
        reference = neighbor_list(
            reference_positions,
            0.75,
            cell=reference_cell,
            pbc=pbc,
            method=method,
            max_neighbors=8,
            return_vectors=True,
            return_distances=True,
        )
        reference_active = reference[0] != state.fill_value
        reference_loss = (
            reference[-2][reference_active].sum()
            + reference[-1][reference_active].square().sum()
        )
        reference_loss.backward()
        torch.testing.assert_close(actual_position_grad, reference_positions.grad)
        torch.testing.assert_close(actual_cell_grad, reference_cell.grad)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("method", ["naive", "cell_list"])
def test_prepared_geometry_clone_survives_changed_pairs_before_backward(
    method: str,
) -> None:
    """Cloned earlier geometry keeps its gradients after pair and shift changes."""
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 4.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    initial = torch.tensor(
        [[0.1, 1.0, 1.0], [3.9, 1.0, 1.0], [2.0, 1.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
    )
    state = prepare_neighbor_list(
        initial,
        0.55,
        cell=cell,
        pbc=pbc,
        method=method,
        max_neighbors=8,
        return_vectors=True,
        return_distances=True,
    )
    frame0 = initial.clone().requires_grad_()
    frame0_cell = cell.clone().requires_grad_()
    first = neighbor_list(frame0, cell=frame0_cell, state=state)
    first_active = (first[0] != state.fill_value).clone()
    assert torch.count_nonzero(first[2]) > 0
    saved_distances = first[-2].clone()
    saved_vectors = first[-1].clone()
    saved_loss = (
        saved_distances[first_active].sum() + saved_vectors[first_active].square().sum()
    )

    frame1 = torch.tensor(
        [[0.1, 1.0, 1.0], [0.5, 1.0, 1.0], [0.9, 1.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    second = neighbor_list(frame1, cell=frame0_cell, state=state)
    second_active = second[0] != state.fill_value
    assert int(second_active.sum()) != int(first_active.sum())
    assert torch.count_nonzero(second[2]) == 0
    assert state.neighbor_distances is second[-2]
    assert state.neighbor_vectors is second[-1]
    saved_loss.backward()

    reference_positions = initial.clone().requires_grad_()
    reference_cell = cell.clone().requires_grad_()
    reference = neighbor_list(
        reference_positions,
        0.55,
        cell=reference_cell,
        pbc=pbc,
        method=method,
        max_neighbors=8,
        return_vectors=True,
        return_distances=True,
    )
    reference_active = reference[0] != state.fill_value
    reference_loss = (
        reference[-2][reference_active].sum()
        + reference[-1][reference_active].square().sum()
    )
    reference_loss.backward()
    torch.testing.assert_close(frame0.grad, reference_positions.grad)
    torch.testing.assert_close(frame0_cell.grad, reference_cell.grad)
    assert frame1.grad is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("method", "batched"),
    [
        ("naive", False),
        ("batch_naive", True),
        ("cell_list", False),
        ("batch_cell_list", True),
    ],
)
@pytest.mark.parametrize("fixed_cell", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_prepared_periodic_geometry_gradients_across_routes(
    method: str, batched: bool, fixed_cell: bool, dtype: torch.dtype
) -> None:
    """Single and batch naive/cell-list geometry differentiates for both cell modes."""
    positions = torch.tensor(
        [[0.1, 1.0, 1.0], [7.9, 1.0, 1.0], [4.1, 1.0, 1.0], [3.9, 1.0, 1.0]],
        dtype=dtype,
        device="cuda",
    )
    base_cell = torch.eye(3, dtype=dtype, device="cuda") * 8.0
    if batched:
        cell = base_cell.repeat(2, 1, 1)
        pbc = torch.ones((2, 3), dtype=torch.bool, device="cuda")
        batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    else:
        positions = positions[:2]
        cell = base_cell
        pbc = torch.ones(3, dtype=torch.bool, device="cuda")
        batch_ptr = None
    state = prepare_neighbor_list(
        positions,
        0.5,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method=method,
        fixed_cell=fixed_cell,
        max_neighbors=8,
        return_vectors=True,
        return_distances=True,
    )
    runtime_positions = positions.detach().clone().requires_grad_()
    runtime_cell = cell.detach().clone().requires_grad_()
    result = neighbor_list(runtime_positions, cell=runtime_cell, state=state)
    active = result[0] != state.fill_value
    assert torch.count_nonzero(result[2][active]) > 0
    assert state.neighbor_distances is result[-2]
    assert state.neighbor_vectors is result[-1]
    loss = result[-2][active].sum() + result[-1][active].square().sum()
    loss.backward()
    assert runtime_positions.grad is not None
    assert runtime_positions.grad.abs().sum() > 0
    assert runtime_cell.grad is not None
    assert runtime_cell.grad.abs().sum() > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_selective_geometry_preserves_stationary_unselected_system() -> None:
    """A selective rebuild preserves borrowed geometry for an unchanged system."""
    positions = torch.tensor(
        [[0.1, 1.0, 1.0], [7.9, 1.0, 1.0], [4.1, 1.0, 1.0], [3.9, 1.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda").repeat(2, 1, 1) * 8.0
    pbc = torch.ones((2, 3), dtype=torch.bool, device="cuda")
    batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    state = prepare_neighbor_list(
        positions,
        0.5,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_cell_list",
        max_neighbors=8,
        return_vectors=True,
        return_distances=True,
        selective=True,
    )
    all_systems = torch.ones(2, dtype=torch.bool, device="cuda")
    selected_system_only = torch.tensor([True, False], device="cuda")
    first_positions = positions.clone().requires_grad_()
    runtime_cell = cell.clone().requires_grad_()
    first = neighbor_list(
        first_positions,
        cell=runtime_cell,
        state=state,
        rebuild_flags=all_systems,
    )
    old_topology = tuple(value[2:].clone() for value in first[:3])
    old_distances = first[-2][2:].clone()
    old_vectors = first[-1][2:].clone()

    moved_positions = positions.clone()
    moved_positions[:2] = torch.tensor(
        [[2.1, 1.0, 1.0], [2.3, 1.0, 1.0]], dtype=torch.float32, device="cuda"
    )
    runtime_positions = moved_positions.requires_grad_()
    second = neighbor_list(
        runtime_positions,
        cell=runtime_cell,
        state=state,
        rebuild_flags=selected_system_only,
    )
    for actual, expected in zip(
        (value[2:] for value in second[:3]), old_topology, strict=True
    ):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(second[-2][2:], old_distances)
    torch.testing.assert_close(second[-1][2:], old_vectors)
    assert state.neighbor_distances is second[-2]
    assert state.neighbor_vectors is second[-1]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("strategy", ["atom_centric", "pair_centric"])
def test_prepared_cell_list_rejects_insufficient_runtime_radius(
    batched: bool, strategy: str
) -> None:
    """Runtime grid radii cannot exceed the prepared search coverage."""
    positions = torch.tensor(
        [[0.05, 0.1, 0.1], [0.4, 0.1, 0.1], [0.05, 0.1, 0.1], [0.4, 0.1, 0.1]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 8.0
    runtime_cell = cell.clone()
    runtime_cell[0, 0] = 0.8
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    kwargs: dict[str, object] = {
        "cell": cell,
        "pbc": pbc,
        "method": "cell_list",
        "strategy": strategy,
        "max_neighbors": 32,
    }
    runtime_kwargs: dict[str, object] = {
        "cell": runtime_cell,
        "pbc": pbc,
        "method": "naive",
        "max_neighbors": 32,
    }
    if batched:
        batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
        kwargs.update(
            cell=cell.repeat(2, 1, 1),
            pbc=pbc.repeat(2, 1),
            batch_ptr=batch_ptr,
            method="batch_cell_list",
        )
        runtime_kwargs.update(
            cell=runtime_cell.repeat(2, 1, 1),
            pbc=pbc.repeat(2, 1),
            batch_ptr=batch_ptr,
            method="batch_naive",
        )

    state = prepare_neighbor_list(positions, 1.0, **kwargs)

    # A moderate cell shrink is still covered by the prepared radius and has
    # the same public neighbor topology as a fresh naive build.
    safe_cell = kwargs["cell"].clone()
    safe_cell.mul_(0.5)
    safe = neighbor_list(positions, cell=safe_cell, state=state)
    safe_reference = neighbor_list(
        positions,
        1.0,
        **{
            **runtime_kwargs,
            "cell": safe_cell,
            "method": "batch_naive" if batched else "naive",
        },
    )
    assert_neighbor_matrix_equal(safe[:3], safe_reference[:3])

    # The state-free route finds cross-boundary images in the smaller runtime
    # box; the prepared route must reject before it can return a partial matrix.
    runtime = kwargs["cell"].clone()
    runtime[..., 0, 0] = 0.8
    reference = neighbor_list(
        positions,
        1.0,
        **{**runtime_kwargs, "cell": runtime},
    )
    assert any(
        bool(torch.any(reference[2][row, : int(reference[1][row]), 0] != 0).item())
        for row in range(reference[1].shape[0])
    )
    with pytest.raises(ValueError, match="cell-list search coverage.*re-prepare"):
        neighbor_list(positions, cell=runtime, state=state)
    assert not state.initialized.any()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_cell_list_rejects_cached_grid_policy_mismatch() -> None:
    """Runtime grid minima cannot silently omit neighbors after a cell change."""
    positions = torch.tensor(
        [[0.1, 0.1, 0.1], [1.05, 0.1, 0.1]], dtype=torch.float32, device="cuda"
    )
    prepared_cell = torch.eye(3, dtype=torch.float32, device="cuda") * 8.0
    runtime_cell = torch.eye(3, dtype=torch.float32, device="cuda") * 2.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    reference = neighbor_list(
        positions,
        1.0,
        cell=runtime_cell,
        pbc=pbc,
        method="naive",
        max_neighbors=32,
    )
    torch.testing.assert_close(
        reference[1], torch.ones(2, dtype=torch.int32, device="cuda")
    )

    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=prepared_cell,
        pbc=pbc,
        method="cell_list",
        strategy="atom_centric",
        max_neighbors=32,
    )
    with pytest.raises(ValueError, match="cell-list search coverage.*re-prepare"):
        neighbor_list(positions, cell=runtime_cell, state=state)
    assert not state.initialized.any()

    refreshed = prepare_neighbor_list(
        positions,
        1.0,
        cell=runtime_cell,
        pbc=pbc,
        method="cell_list",
        strategy="atom_centric",
        max_neighbors=32,
    )
    actual = neighbor_list(positions, cell=runtime_cell, state=refreshed)
    assert_neighbor_matrix_equal(actual[:3], reference[:3])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_cell_list_radius_guard_uses_reciprocal_face_heights() -> None:
    """Skew-cell reciprocal heights contribute to required grid radii."""
    positions = torch.tensor(
        [[0.1, 0.1, 0.1], [0.4, 0.1, 0.1]], dtype=torch.float32, device="cuda"
    )
    prepared_cell = torch.eye(3, dtype=torch.float32, device="cuda") * 10.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=prepared_cell,
        pbc=pbc,
        method="cell_list",
        strategy="atom_centric",
        max_neighbors=32,
    )
    skew_cell = torch.tensor(
        [[10.0, 0.0, 0.0], [9.8, 0.4, 0.0], [0.0, 0.0, 10.0]],
        dtype=torch.float32,
        device="cuda",
    )
    with pytest.raises(ValueError, match="cell-list search coverage"):
        neighbor_list(positions, cell=skew_cell, state=state)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_cell_list_radius_guard_supports_float64() -> None:
    """The prepared coverage guard supports float64 cells and positions."""
    positions = torch.tensor(
        [[0.1, 0.1, 0.1], [0.4, 0.1, 0.1]], dtype=torch.float64, device="cuda"
    )
    cell = torch.eye(3, dtype=torch.float64, device="cuda") * 8.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        method="cell_list",
        strategy="atom_centric",
        max_neighbors=8,
    )
    safe_cell = cell * 0.5
    prepared = neighbor_list(positions, cell=safe_cell, state=state)
    reference = neighbor_list(
        positions,
        1.0,
        cell=safe_cell,
        pbc=pbc,
        method="naive",
        max_neighbors=8,
    )
    assert_neighbor_matrix_equal(prepared[:3], reference[:3])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("strategy", ["atom_centric", "pair_centric"])
def test_prepared_batch_cell_list_radius_guard_checks_selected_systems_only(
    strategy: str,
) -> None:
    """An unselected system may use a wider runtime radius without rebuilding."""
    positions = torch.tensor(
        [[0.05, 0.1, 0.1], [0.4, 0.1, 0.1], [0.05, 0.1, 0.1], [0.4, 0.1, 0.1]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda").repeat(2, 1, 1) * 8.0
    pbc = torch.ones((2, 3), dtype=torch.bool, device="cuda")
    batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_cell_list",
        strategy=strategy,
        max_neighbors=32,
        selective=True,
    )
    first = neighbor_list(
        positions,
        cell=cell,
        state=state,
        rebuild_flags=torch.ones(2, dtype=torch.bool, device="cuda"),
    )
    preserved_rows = tuple(value[2:].clone() for value in first[:3])
    runtime_cell = cell.clone()
    runtime_cell[1, 0, 0] = 0.8
    selected_first_only = neighbor_list(
        positions,
        cell=runtime_cell,
        state=state,
        rebuild_flags=torch.tensor([True, False], dtype=torch.bool, device="cuda"),
    )
    for actual, expected in zip(
        (value[2:] for value in selected_first_only[:3]), preserved_rows, strict=True
    ):
        torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="cell-list search coverage.*system 1"):
        neighbor_list(
            positions,
            cell=runtime_cell,
            state=state,
            rebuild_flags=torch.tensor([False, True], dtype=torch.bool, device="cuda"),
        )
    assert not state.initialized.any()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("format", ["matrix", "coo"])
@pytest.mark.parametrize("fixed_cell", [False, True])
def test_prepared_cell_list_partial_rows_match_direct(
    batched: bool, format: str, fixed_cell: bool
) -> None:
    """Prepared cell lists keep the direct compact target-row contract."""
    positions = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.4, 0.1, 0.1],
            [1.1, 0.1, 0.1],
            [1.4, 0.1, 0.1],
            [2.1, 0.1, 0.1],
            [2.4, 0.1, 0.1],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    target_indices = torch.tensor([1, 4], dtype=torch.int32, device="cuda")
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    kwargs: dict[str, object] = {
        "cell": cell,
        "pbc": pbc,
        "method": "cell_list",
        "target_indices": target_indices,
        "max_neighbors": 8,
        "format": format,
        "return_neighbor_list": format == "coo",
        "return_vectors": True,
        "return_distances": True,
    }
    direct_kwargs = dict(kwargs)
    direct_kwargs.pop("format")
    if batched:
        batch_ptr = torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda")
        kwargs.update(
            cell=cell.repeat(2, 1, 1),
            pbc=pbc.repeat(2, 1),
            batch_ptr=batch_ptr,
            method="batch_cell_list",
        )
        direct_kwargs.update(
            cell=kwargs["cell"],
            pbc=kwargs["pbc"],
            batch_ptr=batch_ptr,
            method="batch_cell_list",
        )
    state = prepare_neighbor_list(positions, 0.8, fixed_cell=fixed_cell, **kwargs)
    prepared = neighbor_list(positions, cell=kwargs["cell"], state=state)
    direct = neighbor_list(positions, 0.8, **direct_kwargs)
    assert prepared[1].shape == (
        target_indices.numel() + 1 if format == "coo" else target_indices.numel(),
    )
    if format == "matrix":
        assert_neighbor_matrix_equal(prepared[:3], direct[:3])
        for row, count in enumerate(prepared[1].tolist()):
            for actual, expected in zip(prepared[3:], direct[3:], strict=True):
                torch.testing.assert_close(actual[row, :count], expected[row, :count])
    else:
        for actual, expected in zip(prepared, direct, strict=True):
            torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(("cutoff", "cutoff2"), [(0.75, 1.25), (1.25, 0.75)])
@pytest.mark.parametrize("fixed_cell", [False, True])
def test_prepared_cluster_dual_omitted_capacity_uses_larger_cutoff(
    batched: bool, cutoff: float, cutoff2: float, fixed_cell: bool
) -> None:
    """Automatic cluster matrix capacity covers either cutoff ordering."""
    positions, _, _, single_cell = _inputs()
    cell = single_cell
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    kwargs: dict[str, object] = {
        "cell": cell,
        "pbc": pbc,
        "method": "cluster_tile",
        "cutoff2": cutoff2,
        "format": "matrix",
        "max_tiles_per_group": 1,
    }
    direct = cluster_tile_neighbor_list
    direct_args: tuple[torch.Tensor, ...] = (positions, cutoff, cell)
    if batched:
        positions = torch.cat((positions[:2], positions[:2] + 4.0))
        batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
        cell = single_cell.repeat(2, 1, 1)
        kwargs.update(
            cell=cell,
            pbc=pbc.repeat(2, 1),
            batch_ptr=batch_ptr,
            method="batch_cluster_tile",
        )
        direct = batch_cluster_tile_neighbor_list
        direct_args = (positions, cutoff, cell, batch_ptr)
    state = prepare_neighbor_list(positions, cutoff, fixed_cell=fixed_cell, **kwargs)
    prepared = neighbor_list(positions, cell=cell, state=state)
    expected = direct(
        *direct_args,
        cutoff2=cutoff2,
        max_neighbors=state.neighbor_matrix1.shape[1],
        max_tiles_per_group=1,
        format="matrix",
    )
    _assert_cluster_route_equal(prepared, expected, "matrix")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dual", [False, True])
def test_prepared_nonperiodic_naive_omits_shift_storage(
    batched: bool, dual: bool
) -> None:
    """Nonperiodic prepared naive routes retain their legacy shift-free tuple."""
    positions = torch.rand((8, 3), dtype=torch.float32, device="cuda")
    kwargs: dict[str, object] = {
        "method": "batch_naive" if batched else "naive",
        "max_neighbors": 8,
    }
    if batched:
        kwargs["batch_ptr"] = torch.tensor([0, 3, 8], dtype=torch.int32, device="cuda")
    if dual:
        kwargs.update(cutoff2=1.5, max_neighbors2=8)
    state = prepare_neighbor_list(positions, 1.0, **kwargs)
    result = neighbor_list(positions, state=state)
    assert len(result) == (4 if dual else 2)
    assert state.neighbor_matrix_shifts is None
    if dual:
        assert state.neighbor_matrix_shifts1 is None
        assert state.neighbor_matrix_shifts2 is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_dual_naive_coo_result_properties() -> None:
    """Dual COO preparation publishes both suffix groups without changing its tuple."""
    positions = torch.rand((8, 3), dtype=torch.float32, device="cuda") * 4.0
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=torch.ones(3, dtype=torch.bool, device="cuda"),
        cutoff2=1.5,
        method="naive",
        max_neighbors=8,
        max_neighbors2=8,
        return_neighbor_list=True,
    )
    assert (
        state.neighbor_list is state.neighbor_ptr is state.neighbor_list_shifts is None
    )
    result = neighbor_list(positions, cell=cell, state=state)
    assert len(result) == 6
    assert state.neighbor_list1 is result[0]
    assert state.neighbor_ptr1 is result[1]
    assert state.neighbor_list_shifts1 is result[2]
    assert state.neighbor_list2 is result[3]
    assert state.neighbor_ptr2 is result[4]
    assert state.neighbor_list_shifts2 is result[5]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("cell", "pbc"),
    [
        (torch.diag(torch.tensor([4.0, 5.0, 6.0])), torch.tensor([True, True, True])),
        (
            torch.tensor([[4.0, 0.5, 0.0], [0.0, 5.0, 0.25], [0.0, 0.0, 6.0]]),
            torch.tensor([True, True, True]),
        ),
        (torch.diag(torch.tensor([4.0, 5.0, 6.0])), torch.tensor([True, False, True])),
    ],
)
def test_prepared_naive_periodic_geometry_matches_direct(
    cell: torch.Tensor, pbc: torch.Tensor
) -> None:
    """Prepared naive matrix output preserves direct periodic topology and shifts."""
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [3.9, 0.2, 0.3], [1.2, 2.1, 0.4]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = cell.to(device="cuda", dtype=torch.float32)
    pbc = pbc.to(device="cuda")
    state = prepare_neighbor_list(
        positions,
        0.7,
        cell=cell,
        pbc=pbc,
        method="naive",
        max_neighbors=64,
    )
    actual = neighbor_list(positions, cell=cell, state=state)
    expected = neighbor_list(
        positions,
        0.7,
        cell=cell,
        pbc=pbc,
        method="naive",
        max_neighbors=64,
    )
    assert_neighbor_matrix_equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_batch_naive_periodic_guard_handles_heterogeneous_cells() -> None:
    """Batched prepared naive uses each system's current cell for coverage."""
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [3.9, 0.0, 0.0], [0.2, 0.0, 0.0], [1.8, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.diag_embed(
        torch.tensor([[4.0, 4.0, 4.0], [2.0, 3.0, 5.0]], device="cuda")
    )
    pbc = torch.tensor([[True, True, True], [True, False, True]], device="cuda")
    batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    state = prepare_neighbor_list(
        positions,
        0.7,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_naive",
        max_neighbors=32,
    )
    actual = neighbor_list(positions, cell=cell, state=state)
    expected = neighbor_list(
        positions,
        0.7,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_naive",
        max_neighbors=32,
    )
    assert_neighbor_matrix_equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_naive_guard_rejects_insufficient_periodic_coverage() -> None:
    """Runtime cells requiring more images fail with a re-preparation request."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float32, device="cuda"
    )
    prepared_cell = torch.eye(3, device="cuda") * 4.0
    runtime_cell = torch.eye(3, device="cuda") * 0.5
    state = prepare_neighbor_list(
        positions,
        0.7,
        cell=prepared_cell,
        pbc=torch.ones(3, dtype=torch.bool, device="cuda"),
        method="naive",
        max_neighbors=64,
    )
    with pytest.raises(ValueError, match="periodic-image coverage.*re-prepare"):
        neighbor_list(positions, cell=runtime_cell, state=state)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_naive_selective_guard_checks_selected_systems_only() -> None:
    """Selective prepared naive validates only systems requested for rebuild."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, device="cuda").repeat(2, 1, 1) * 4.0
    pbc = torch.ones((2, 3), dtype=torch.bool, device="cuda")
    batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    state = prepare_neighbor_list(
        positions,
        0.7,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_naive",
        max_neighbors=32,
        selective=True,
    )
    neighbor_list(
        positions,
        cell=cell,
        state=state,
        rebuild_flags=torch.ones(2, dtype=torch.bool, device="cuda"),
    )
    bad_cell = cell.clone()
    bad_cell[1].mul_(0.125)
    preserved = neighbor_list(
        positions,
        cell=bad_cell,
        state=state,
        rebuild_flags=torch.tensor([True, False], dtype=torch.bool, device="cuda"),
    )
    assert preserved[0] is state.neighbor_matrix
    failing_state = prepare_neighbor_list(
        positions,
        0.7,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_naive",
        max_neighbors=32,
        selective=True,
    )
    neighbor_list(
        positions,
        cell=cell,
        state=failing_state,
        rebuild_flags=torch.ones(2, dtype=torch.bool, device="cuda"),
    )
    with pytest.raises(ValueError, match="periodic-image coverage"):
        neighbor_list(
            positions,
            cell=bad_cell,
            state=failing_state,
            rebuild_flags=torch.tensor([False, True], dtype=torch.bool, device="cuda"),
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_preparation_rejects_unsupported_cluster_and_reroutes_batch_cell_dual() -> None:
    """Preparation matches direct cluster limits and legacy batch dual dispatch."""
    positions = torch.rand((8, 3), dtype=torch.float32, device="cuda")
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    common = dict(cell=cell, pbc=pbc, method="cluster_tile", max_tiles_per_group=1)
    with pytest.raises(NotImplementedError, match="half_fill=True is not supported"):
        prepare_neighbor_list(positions, 1.0, half_fill=True, **common)
    with pytest.raises(ValueError, match="target_indices is not supported"):
        prepare_neighbor_list(
            positions,
            1.0,
            target_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
            **common,
        )
    with pytest.raises(ValueError, match="cutoff2 is supported only"):
        prepare_neighbor_list(
            positions,
            1.0,
            cutoff2=1.5,
            format="coo",
            return_neighbor_list=True,
            **common,
        )
    with pytest.raises(ValueError, match="max_neighbors2 is not supported"):
        prepare_neighbor_list(positions, 1.0, max_neighbors2=8, **common)

    batch_cell = cell.repeat(2, 1, 1)
    state = prepare_neighbor_list(
        positions,
        1.0,
        cutoff2=1.5,
        cell=batch_cell,
        pbc=pbc.repeat(2, 1),
        batch_ptr=torch.tensor([0, 3, 8], dtype=torch.int32, device="cuda"),
        method="batch_cell_list",
        max_neighbors=8,
        max_neighbors2=8,
    )
    result = neighbor_list(positions, cell=batch_cell, state=state)
    assert state.method == "batch_naive_dual_cutoff" and len(result) == 6
    assert state.neighbor_matrix1 is result[0] and state.neighbor_matrix2 is result[3]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    ("requested_strategy", "resolved_strategy"),
    [
        ("atom_centric", "cell_list_atom_centric"),
        ("pair_centric", "cell_list_pair_centric"),
    ],
)
def test_prepared_generic_cell_strategy_matches_fine_grained_route(
    batched: bool,
    requested_strategy: str,
    resolved_strategy: str,
) -> None:
    """Generic cell-list strategies resolve once and preserve topology."""
    close, _, _, cell = _inputs()
    if batched:
        positions = torch.cat((close, close + 4.0), dim=0)
        batch_ptr = torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda")
        cell = cell.repeat(2, 1, 1)
        pbc = torch.ones((2, 3), dtype=torch.bool, device="cuda")
    else:
        positions = close
        batch_ptr = None
        pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    method = "batch_cell_list" if batched else "cell_list"
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method=method,
        strategy=requested_strategy,
        max_neighbors=8,
    )
    expected_method = f"batch_{resolved_strategy}" if batched else resolved_strategy
    assert state.strategy == expected_method
    with pytest.raises(AttributeError):
        state.strategy = "cell_list_atom_centric"

    prepared = neighbor_list(positions, cell=cell, state=state)
    expected = neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method=expected_method,
        max_neighbors=8,
    )
    torch.testing.assert_close(prepared[1], expected[1])
    for row, count in enumerate(prepared[1].tolist()):
        actual_row = torch.sort(prepared[0][row, :count])[0]
        expected_row = torch.sort(expected[0][row, :count])[0]
        torch.testing.assert_close(actual_row, expected_row)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batched", [False, True])
def test_prepared_generic_naive_strategy_matches_fine_grained_route(
    batched: bool,
) -> None:
    """Generic naive strategies are forwarded to their selected kernels."""
    positions, _, _, _ = _inputs()
    batch_ptr = None
    if batched:
        positions = torch.cat((positions, positions + 4.0), dim=0)
        batch_ptr = torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda")
    method = "batch_naive" if batched else "naive"
    for requested_strategy, resolved_strategy in (
        ("scalar", "naive_scalar"),
        ("tile", "naive_tile"),
    ):
        state = prepare_neighbor_list(
            positions,
            1.0,
            method=method,
            batch_ptr=batch_ptr,
            strategy=requested_strategy,
            max_neighbors=8,
        )
        expected_method = f"batch_{resolved_strategy}" if batched else resolved_strategy
        assert state.strategy == expected_method
        prepared = neighbor_list(positions, state=state)
        expected = neighbor_list(
            positions,
            1.0,
            batch_ptr=batch_ptr,
            method=expected_method,
            max_neighbors=8,
        )
        torch.testing.assert_close(prepared[1], expected[1])
        for row, count in enumerate(prepared[1].tolist()):
            actual_row = torch.sort(prepared[0][row, :count])[0]
            expected_row = torch.sort(expected[0][row, :count])[0]
            torch.testing.assert_close(actual_row, expected_row)


def test_prepared_generic_naive_tile_strategy_reaches_cpu_launcher() -> None:
    """The prepared topology path forwards its resolved kernel strategy."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)
    state = prepare_neighbor_list(
        positions,
        1.0,
        method="naive",
        strategy="tile",
        max_neighbors=4,
    )

    with pytest.raises(ValueError, match="strategy='tile' requires CUDA"):
        neighbor_list(positions, state=state)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_nonperiodic_span_margin_is_translation_invariant() -> None:
    """A synthesized cell-list state bounds span, rather than absolute position."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device="cuda"
    )
    state = prepare_neighbor_list(
        positions,
        1.0,
        method="cell_list",
        max_neighbors=8,
        span_margin=0.5,
    )
    exact_bound = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 1.0, 1.0]], dtype=torch.float32, device="cuda"
    )
    neighbor_list(exact_bound, state=state)
    translated = positions + 5.0
    prepared = neighbor_list(translated, state=state)
    eager = neighbor_list(translated, 1.0, method="cell_list", max_neighbors=8)
    torch.testing.assert_close(prepared[1], eager[1])
    for row, count in enumerate(prepared[1].tolist()):
        torch.testing.assert_close(prepared[0][row, :count], eager[0][row, :count])
        torch.testing.assert_close(prepared[2][row, :count], eager[2][row, :count])
    with pytest.raises(
        ValueError,
        match="prepared nonperiodic span exceeded for system 0 on axis 0",
    ):
        neighbor_list(
            torch.tensor(
                [[0.0, 0.0, 0.0], [1.6, 1.0, 1.0]],
                dtype=torch.float32,
                device="cuda",
            ),
            state=state,
        )
    assert not state.initialized.any()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_span_margin_empty_zero_span_and_geometry_gradient_parity() -> None:
    """Prepared synthesized cells handle degenerate exemplars and preserve geometry gradients."""
    empty = torch.empty((0, 3), dtype=torch.float32, device="cuda")
    empty_state = prepare_neighbor_list(
        empty, 1.0, method="cell_list", max_neighbors=8, span_margin=0.25
    )
    assert empty_state.span_margin == 0.25
    neighbor_list(empty, state=empty_state)

    zero_span = torch.zeros((2, 3), dtype=torch.float32, device="cuda")
    zero_state = prepare_neighbor_list(
        zero_span, 1.0, method="cell_list", max_neighbors=8, span_margin=0.0
    )
    neighbor_list(zero_span, state=zero_state)

    values = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    state = prepare_neighbor_list(
        values.detach(),
        1.0,
        method="cell_list",
        max_neighbors=8,
        return_vectors=True,
        return_distances=True,
        span_margin=0.1,
    )
    prepared = neighbor_list(values, state=state)
    prepared_loss = (
        state.neighbor_distances.sum() + state.neighbor_vectors.square().sum()
    )
    prepared_grad = torch.autograd.grad(prepared_loss, values)[0]
    reference = values.detach().clone().requires_grad_(True)
    eager = neighbor_list(
        reference,
        1.0,
        method="cell_list",
        max_neighbors=8,
        return_vectors=True,
        return_distances=True,
    )
    eager_loss = eager[-2].sum() + eager[-1].square().sum()
    eager_grad = torch.autograd.grad(eager_loss, reference)[0]
    torch.testing.assert_close(prepared[0], eager[0])
    torch.testing.assert_close(prepared_grad, eager_grad)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_batched_span_margin_checks_selected_systems_only() -> None:
    """A selective batched cell list preserves an unselected out-of-bound system."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    state = prepare_neighbor_list(
        positions,
        1.0,
        method="batch_cell_list",
        batch_ptr=torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda"),
        max_neighbors=8,
        span_margin=0.1,
        selective=True,
    )
    all_selected = torch.ones(2, dtype=torch.bool, device="cuda")
    neighbor_list(positions, state=state, rebuild_flags=all_selected)
    moved = positions.clone()
    moved[3, 0] = 1.2
    keep_second = torch.tensor([True, False], dtype=torch.bool, device="cuda")
    neighbor_list(moved, state=state, rebuild_flags=keep_second)
    with pytest.raises(ValueError, match="system 1 on axis 0"):
        neighbor_list(moved, state=state, rebuild_flags=all_selected)
    assert not state.initialized.any()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("format", ["matrix"])
@pytest.mark.parametrize("fixed_cell", [False, True])
def test_prepared_selective_storage_and_failure_invalidation(
    batched: bool, format: str, fixed_cell: bool
) -> None:
    """Selective routes retain untouched systems and clear state after overflow."""
    positions = torch.rand((32, 3), dtype=torch.float32, device="cuda") * 4.0
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    kwargs: dict[str, object] = {
        "method": "batch_cluster_tile" if batched else "cluster_tile",
        "format": format,
        "return_neighbor_list": format == "coo",
        "max_neighbors": 8,
        "max_pairs": 256,
        "max_tiles_per_group": 1,
        "selective": True,
    }
    if batched:
        kwargs.update(
            cell=cell.repeat(2, 1, 1),
            pbc=pbc.repeat(2, 1),
            batch_ptr=torch.tensor([0, 16, 32], dtype=torch.int32, device="cuda"),
        )
        flags = torch.tensor([True, True], dtype=torch.bool, device="cuda")
    else:
        kwargs.update(cell=cell, pbc=pbc)
        flags = torch.ones(1, dtype=torch.bool, device="cuda")
    state = prepare_neighbor_list(positions, 1.0, fixed_cell=fixed_cell, **kwargs)
    if format == "coo":
        assert state.neighbor_list is not None and state.neighbor_ptr is None
        assert state.pair_offsets is not None and state.pair_counts is not None
    else:
        assert state.neighbor_matrix is not None
    with pytest.raises(ValueError, match="cannot preserve uninitialized system"):
        neighbor_list(positions, cell=kwargs["cell"], state=state, rebuild_flags=~flags)
    with pytest.raises(
        RuntimeError,
        match="prepared Torch neighbor-list state is invalid; call prepare_neighbor_list again",
    ):
        neighbor_list(positions, cell=kwargs["cell"], state=state, rebuild_flags=flags)
    state = prepare_neighbor_list(positions, 1.0, fixed_cell=fixed_cell, **kwargs)
    first = neighbor_list(
        positions, cell=kwargs["cell"], state=state, rebuild_flags=flags
    )
    before = first[0].clone()
    preserve = torch.zeros_like(flags)
    second = neighbor_list(
        positions + 0.1, cell=kwargs["cell"], state=state, rebuild_flags=preserve
    )
    torch.testing.assert_close(second[0], before)
    overflowing = torch.zeros_like(positions)
    with pytest.raises(NeighborOverflowError, match="larger than the maximum allowed"):
        neighbor_list(
            overflowing,
            cell=kwargs["cell"],
            state=state,
            rebuild_flags=flags,
        )
    assert not state.initialized.any()
    assert all(
        getattr(state, name) is None
        for name in ("neighbor_matrix", "neighbor_list", "pair_offsets", "num_tiles")
    )
    with pytest.raises(
        RuntimeError,
        match="prepared Torch neighbor-list state is invalid; call prepare_neighbor_list again",
    ):
        neighbor_list(positions, cell=kwargs["cell"], state=state, rebuild_flags=flags)
    replacement = prepare_neighbor_list(positions, 1.0, fixed_cell=fixed_cell, **kwargs)
    assert neighbor_list(
        positions, cell=kwargs["cell"], state=replacement, rebuild_flags=flags
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_batched_cluster_tile_shared_cell_matches_state_free() -> None:
    """Prepared batched cluster tiles retain the public shared-cell contract."""
    positions = torch.rand((32, 3), dtype=torch.float32, device="cuda") * 4.0
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    batch_ptr = torch.tensor([0, 16, 32], dtype=torch.int32, device="cuda")
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_cluster_tile",
        format="matrix",
        max_neighbors=8,
        max_tiles_per_group=1,
    )
    prepared = neighbor_list(positions, cell=cell, state=state)
    eager = neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_cluster_tile",
        max_neighbors=8,
        max_tiles_per_group=1,
    )
    torch.testing.assert_close(prepared[1], eager[1])
    for row, count in enumerate(prepared[1].tolist()):
        torch.testing.assert_close(prepared[0][row, :count], eager[0][row, :count])
        torch.testing.assert_close(prepared[2][row, :count], eager[2][row, :count])
    assert state.neighbor_matrix is prepared[0]
    assert state.num_neighbors is prepared[1]
    assert state.neighbor_matrix_shifts is prepared[2]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_batch_partition_is_independent_of_caller_tensor() -> None:
    """Prepared batch execution retains the partition supplied at preparation."""
    positions, _, _, single_cell = _inputs()
    positions = torch.cat((positions[:2], positions[:2] + 4.0))
    batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    cell = single_cell.repeat(2, 1, 1)
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=torch.ones((2, 3), dtype=torch.bool, device="cuda"),
        batch_ptr=batch_ptr,
        method="batch_cluster_tile",
        max_neighbors=8,
        max_tiles_per_group=1,
    )
    batch_ptr[1] = 1
    prepared = neighbor_list(positions, cell=cell, state=state)
    direct = batch_cluster_tile_neighbor_list(
        positions,
        1.0,
        cell,
        torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda"),
        max_neighbors=8,
        max_tiles_per_group=1,
    )
    _assert_cluster_route_equal(prepared, direct, "matrix")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_batched_cluster_geometry_is_detached_snapshot_across_reuse() -> None:
    """Batched prepared geometry remains differentiable while snapshots are replaced."""
    positions, _, _, single_cell = _inputs()
    positions = torch.cat((positions[:2], positions[:2] + 4.0)).requires_grad_()
    cell = single_cell.repeat(2, 1, 1).requires_grad_()
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=torch.ones((2, 3), dtype=torch.bool, device="cuda"),
        batch_ptr=torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda"),
        method="batch_cluster_tile",
        max_neighbors=8,
        max_tiles_per_group=1,
        return_vectors=True,
        return_distances=True,
    )
    first = neighbor_list(positions, cell=cell, state=state)
    first_values = tuple(value.detach().clone() for value in first)
    (first[-2].sum() + first[-1].square().sum()).backward()
    assert positions.grad is not None and cell.grad is not None
    assert (
        state.neighbor_vectors is not None and not state.neighbor_vectors.requires_grad
    )
    second = neighbor_list(positions.detach() + 0.1, cell=cell.detach(), state=state)
    assert state.neighbor_vectors is second[-1]
    for actual, expected in zip(first, first_values, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("format", ["tile", "matrix", "coo"])
def test_prepared_empty_cluster_executes(format: str) -> None:
    """Empty prepared cluster routes preserve their ordinary output contracts."""
    positions = torch.empty((0, 3), dtype=torch.float32, device="cuda")
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=torch.ones(3, dtype=torch.bool, device="cuda"),
        method="cluster_tile",
        format=format,
        return_neighbor_list=format == "coo",
        max_neighbors=8,
        **({} if format == "coo" else {"max_pairs": 8}),
        max_tiles_per_group=1,
    )
    result = neighbor_list(positions, cell=cell, state=state)
    if format == "matrix":
        assert result[0].shape == (0, 8)
        assert result[1].shape == (0,)
    elif format == "coo":
        assert result[0].shape == (2, 0)
        assert result[1].shape == (1,)
    else:
        assert int(result[0]) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_tile_schema_and_state_precedence() -> None:
    """Tile metadata is published and redundant execution configuration is inert."""
    positions = torch.rand((32, 3), dtype=torch.float32, device="cuda") * 4.0
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        method="cluster_tile",
        format="matrix",
        max_neighbors=8,
        max_tiles_per_group=1,
    )
    result = neighbor_list(
        positions,
        99.0,
        cell=cell,
        state=state,
        method="naive",
        format="coo",
        max_neighbors=1,
        return_vectors=True,
        pair_fn=object(),
    )
    assert len(result) == 3
    contradictory_snapshot = tuple(value.clone() for value in result)
    state_only = neighbor_list(positions, cell=cell, state=state)
    assert len(state_only) == len(contradictory_snapshot)
    for observed, expected in zip(contradictory_snapshot, state_only, strict=True):
        torch.testing.assert_close(observed, expected)
    for name in (
        "num_tiles",
        "tile_row_group",
        "tile_col_group",
        "sorted_atom_index",
        "sorted_pos_x",
        "sorted_pos_y",
        "sorted_pos_z",
    ):
        assert getattr(state, name) is not None
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        neighbor_list(positions, cell=cell, state=state, unknown_state_option=True)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_tile_output_and_format_contradictions() -> None:
    """Prepared tile output owns its auxiliary buffers and rejects contradictions."""
    positions = torch.rand((32, 3), dtype=torch.float32, device="cuda") * 4.0
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 5.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        method="cluster_tile",
        format="tile",
        max_neighbors=8,
        max_tiles_per_group=1,
    )
    result = neighbor_list(positions, cell=cell, state=state)
    assert len(result) == 7
    for name in (
        "num_tiles",
        "tile_row_group",
        "tile_col_group",
        "sorted_atom_index",
        "sorted_pos_x",
        "sorted_pos_y",
        "sorted_pos_z",
    ):
        assert getattr(state, name) is not None

    with pytest.raises(ValueError, match="format contradicts return_neighbor_list"):
        prepare_neighbor_list(
            positions,
            1.0,
            cell=cell,
            pbc=pbc,
            method="cluster_tile",
            format="matrix",
            return_neighbor_list=True,
            max_neighbors=8,
            max_tiles_per_group=1,
        )
    with pytest.raises(ValueError, match="format contradicts return_neighbor_list"):
        prepare_neighbor_list(
            positions,
            1.0,
            cell=cell,
            pbc=pbc,
            method="cluster_tile",
            format="coo",
            return_neighbor_list=False,
            max_neighbors=8,
            max_tiles_per_group=1,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("method", ["naive", "cell_list"])
def test_fixed_cell_noncluster_routes_match_dynamic(method: str) -> None:
    """Fixed-cell naive and cell-list routes match their dynamic behavior."""
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [0.6, 0.2, 0.3], [2.0, 2.0, 2.0], [2.4, 2.0, 2.0]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 8.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    kwargs = dict(method=method, max_neighbors=8)
    fixed_state = prepare_neighbor_list(
        positions, 1.0, cell=cell, pbc=pbc, fixed_cell=True, **kwargs
    )
    dynamic_state = prepare_neighbor_list(positions, 1.0, cell=cell, pbc=pbc, **kwargs)

    assert fixed_state.fixed_cell is True
    with pytest.raises(AttributeError):
        fixed_state.fixed_cell = False
    fixed = neighbor_list(positions, cell=cell.clone(), state=fixed_state)
    dynamic = neighbor_list(positions, cell=cell.clone(), state=dynamic_state)
    assert_neighbor_matrix_equal(fixed[:3], dynamic[:3])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_cell_public_validation_and_execution_keyword() -> None:
    """The fixed-cell option is prepare-only and validates explicit cells."""
    positions = torch.zeros((2, 3), dtype=torch.float32, device="cuda")
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 6.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    parameter = inspect.signature(prepare_neighbor_list).parameters["fixed_cell"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(ValueError, match="Boolean"):
        prepare_neighbor_list(positions, 1.0, cell=cell, fixed_cell=1)
    with pytest.raises(ValueError, match="explicit cell"):
        prepare_neighbor_list(positions, 1.0, method="naive", fixed_cell=True)
    with pytest.raises(ValueError, match="explicit cell"):
        prepare_neighbor_list(positions, 1.0, method="cell_list", fixed_cell=True)
    state = prepare_neighbor_list(
        positions, 1.0, cell=cell, pbc=pbc, method="naive", fixed_cell=True
    )
    with pytest.raises(TypeError):
        neighbor_list(positions, cell=cell, state=state, fixed_cell=True)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_cell_cluster_matrix_reuses_topology_and_keeps_cell_gradient() -> None:
    """Fixed cluster matrix geometry uses cached topology and live cell values."""
    positions = torch.tensor(
        [[0.2, 1.0, 1.0], [9.8, 1.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    cell = (torch.eye(3, dtype=torch.float32, device="cuda") * 10.0).requires_grad_()
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        method="cluster_tile",
        fixed_cell=True,
        max_neighbors=4,
        max_tiles_per_group=1,
        return_vectors=True,
        return_distances=True,
    )
    runtime_cell = cell.detach().clone().requires_grad_()
    result = neighbor_list(positions, cell=runtime_cell, state=state)
    assert result[1].tolist() == [1, 1]
    assert torch.count_nonzero(result[2]) > 0
    torch.testing.assert_close(result[3][:, 0], torch.full((2,), 0.4, device="cuda"))
    for source in range(2):
        target = int(result[0][source, 0])
        reconstructed = positions.detach()[target] - positions.detach()[source]
        reconstructed = (
            reconstructed
            + result[2][source, 0].to(torch.float32) @ runtime_cell.detach()
        )
        torch.testing.assert_close(result[4][source, 0], reconstructed)
    result[3].sum().backward()
    assert runtime_cell.grad is not None
    assert runtime_cell.grad.abs().sum() > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_cell_batch_cluster_shared_cell_keeps_live_gradient() -> None:
    """Shared batched cluster geometry uses detached topology and live cell input."""
    positions = torch.tensor(
        [[0.2, 1.0, 1.0], [9.8, 1.0, 1.0], [0.2, 1.0, 1.0], [9.8, 1.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    cell = (torch.eye(3, dtype=torch.float32, device="cuda") * 10.0).requires_grad_()
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    state = prepare_neighbor_list(
        positions,
        1.0,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_cluster_tile",
        fixed_cell=True,
        max_neighbors=4,
        max_tiles_per_group=1,
        return_vectors=True,
        return_distances=True,
    )
    runtime_cell = cell.detach().clone().requires_grad_()
    result = neighbor_list(positions, cell=runtime_cell, state=state)
    assert result[1].tolist() == [1, 1, 1, 1]
    torch.testing.assert_close(result[3][:, 0], torch.full((4,), 0.4, device="cuda"))
    assert torch.count_nonzero(result[2]) > 0
    result[3].sum().backward()
    assert runtime_cell.grad is not None
    assert runtime_cell.grad.abs().sum() > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_cell_cluster_coo_direct_csr_matches_dynamic() -> None:
    """Prepared cluster COO uses cached geometry in both CSR passes."""
    positions, _, _, cell = _inputs()
    positions = positions[:2]
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    kwargs = dict(
        method="cluster_tile",
        format="coo",
        return_neighbor_list=True,
        max_neighbors=4,
        max_pairs=8,
        max_tiles_per_group=1,
    )
    fixed = prepare_neighbor_list(
        positions, 1.0, cell=cell, pbc=pbc, fixed_cell=True, **kwargs
    )
    dynamic = prepare_neighbor_list(positions, 1.0, cell=cell, pbc=pbc, **kwargs)
    got = neighbor_list(positions, cell=cell.clone(), state=fixed)
    expected = neighbor_list(positions, cell=cell.clone(), state=dynamic)
    torch.testing.assert_close(got[1], expected[1])
    active = int(got[1][-1])
    torch.testing.assert_close(got[0][:, :active], expected[0][:, :active])
    torch.testing.assert_close(got[2][:active], expected[2][:active])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("strategy", "wrap_positions", "partial"),
    [
        ("scalar", True, False),
        ("tile", True, False),
        ("scalar", False, False),
        ("scalar", True, True),
    ],
)
def test_fixed_cell_naive_scalar_tile_partial_and_wrap(
    strategy: str, wrap_positions: bool, partial: bool
) -> None:
    """Fixed naive scalar, tile, partial, and wrapping paths retain parity."""
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [7.9, 0.2, 0.3], [3.0, 3.0, 3.0]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 8.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    target_indices = (
        torch.tensor([1, 2], dtype=torch.int32, device="cuda") if partial else None
    )
    kwargs = dict(
        cell=cell,
        pbc=pbc,
        method="naive",
        strategy=strategy,
        target_indices=target_indices,
        wrap_positions=wrap_positions,
        max_neighbors=8,
    )
    fixed = prepare_neighbor_list(positions, 0.5, fixed_cell=True, **kwargs)
    got = neighbor_list(positions, cell=cell.clone(), state=fixed)
    direct_kwargs = {key: value for key, value in kwargs.items() if key != "strategy"}
    direct_kwargs["method"] = f"naive_{strategy}"
    expected = neighbor_list(positions, 0.5, **direct_kwargs)
    assert_neighbor_matrix_equal(got, expected)
    if not partial:
        assert torch.count_nonzero(got[2]) > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dual_cutoff", [False, True])
def test_fixed_cell_naive_batch_and_dual_cutoff(
    batched: bool, dual_cutoff: bool
) -> None:
    """Fixed single and batch naive routes cache periodic inverse geometry."""
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [7.9, 0.2, 0.3]], dtype=torch.float32, device="cuda"
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 8.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    batch_ptr = None
    if batched:
        positions = torch.cat((positions, positions + 2.0))
        batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
        cell = cell.repeat(2, 1, 1)
        pbc = pbc.repeat(2, 1)
    kwargs = dict(
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_naive" if batched else "naive",
        max_neighbors=8,
        cutoff2=0.9 if dual_cutoff else None,
        max_neighbors2=8 if dual_cutoff else None,
    )
    fixed = prepare_neighbor_list(positions, 0.5, fixed_cell=True, **kwargs)
    got = neighbor_list(positions, cell=cell.clone(), state=fixed)
    direct_kwargs = dict(kwargs)
    if dual_cutoff:
        direct_kwargs["method"] = (
            "batch_naive_dual_cutoff" if batched else "naive_dual_cutoff"
        )
        direct_kwargs["max_neighbors1"] = direct_kwargs.pop("max_neighbors")
    else:
        direct_kwargs.pop("max_neighbors2")
    expected = neighbor_list(positions, 0.5, **direct_kwargs)
    for start in range(0, len(got), 3):
        assert_neighbor_matrix_equal(
            got[start : start + 3], expected[start : start + 3]
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("strategy", ["scalar", "tile"])
def test_fixed_cell_batch_naive_explicit_strategies(strategy: str) -> None:
    """Fixed batched naive forwards both supported execution strategies."""
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [7.9, 0.2, 0.3], [1.1, 0.2, 0.3], [6.9, 0.2, 0.3]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda").repeat(2, 1, 1) * 8.0
    pbc = torch.ones((2, 3), dtype=torch.bool, device="cuda")
    batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    kwargs = dict(
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_naive",
        strategy=strategy,
        max_neighbors=8,
    )
    state = prepare_neighbor_list(positions, 0.5, fixed_cell=True, **kwargs)
    actual = neighbor_list(positions, cell=cell.clone(), state=state)
    direct_kwargs = {key: value for key, value in kwargs.items() if key != "strategy"}
    direct_kwargs["method"] = f"batch_naive_{strategy}"
    expected = neighbor_list(positions, 0.5, **direct_kwargs)
    assert_neighbor_matrix_equal(actual, expected)
    assert torch.count_nonzero(actual[2]) > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("strategy", ["atom_centric", "pair_centric"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_cell_cell_list_grid_reuse_rebuilds_occupancy(
    batched: bool, strategy: str, dtype: torch.dtype
) -> None:
    """Cached cell-list grids remain fixed while atom occupancy is rebuilt."""
    positions = torch.tensor(
        [[0.1, 0.1, 0.1], [0.5, 0.1, 0.1], [2.1, 0.1, 0.1], [2.5, 0.1, 0.1]],
        dtype=dtype,
        device="cuda",
    )
    cell = torch.eye(3, dtype=dtype, device="cuda") * 8.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    batch_ptr = None
    if batched:
        positions = torch.cat((positions, positions + 4.0))
        cell = cell.repeat(2, 1, 1)
        pbc = pbc.repeat(2, 1)
        batch_ptr = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
    kwargs = dict(
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method="batch_cell_list" if batched else "cell_list",
        strategy=strategy,
        max_neighbors=8,
    )
    fixed = prepare_neighbor_list(positions, 0.8, fixed_cell=True, **kwargs)
    direct_kwargs = {key: value for key, value in kwargs.items() if key != "strategy"}
    route_prefix = "batch_cell_list" if batched else "cell_list"
    direct_kwargs["method"] = f"{route_prefix}_{strategy}"
    moved = positions.clone()
    moved[1::4, 0] += 1.5
    moved[3::4, 0] += 1.5
    first = neighbor_list(positions, cell=cell.clone(), state=fixed)
    first_counts = first[1].clone()
    rebuilt = neighbor_list(moved, cell=cell.clone(), state=fixed)
    expected = neighbor_list(moved, 0.8, **direct_kwargs)
    assert_neighbor_matrix_equal(rebuilt[:3], expected[:3])
    assert not torch.equal(first_counts, rebuilt[1])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("method", "batched", "dual_cutoff"),
    [
        ("naive", False, False),
        ("batch_naive", True, False),
        ("naive", False, True),
        ("batch_naive", True, True),
    ],
)
def test_fixed_cell_nonperiodic_naive_accepts_explicit_cell_without_pbc(
    method: str, batched: bool, dual_cutoff: bool
) -> None:
    """Explicit unused cells pass runtime validation across nonperiodic naive routes."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 6.0
    batch_ptr = None
    if batched:
        positions = torch.cat((positions, positions + 3.0))
        cell = cell.repeat(2, 1, 1)
        batch_ptr = torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda")
    options = dict(
        cell=cell,
        batch_ptr=batch_ptr,
        method=method,
        max_neighbors=4,
        cutoff2=1.25 if dual_cutoff else None,
        max_neighbors2=4 if dual_cutoff else None,
    )
    with pytest.raises(ValueError, match="pbc is required when cell is provided"):
        prepare_neighbor_list(positions, 0.75, **options)
    fixed = prepare_neighbor_list(positions, 0.75, fixed_cell=True, **options)
    expected_options = dict(options)
    expected_options.pop("cell")
    if dual_cutoff:
        expected_options["method"] = (
            "batch_naive_dual_cutoff" if batched else "naive_dual_cutoff"
        )
        expected_options["max_neighbors1"] = expected_options.pop("max_neighbors")
    else:
        expected_options.pop("max_neighbors2")
    expected = neighbor_list(positions, 0.75, **expected_options)
    actual = neighbor_list(positions, cell=cell.clone(), state=fixed)
    if dual_cutoff:
        for start in (0, 2):
            torch.testing.assert_close(actual[start], expected[start])
            torch.testing.assert_close(actual[start + 1], expected[start + 1])
    else:
        assert_neighbor_matrix_equal(actual[:3], expected[:3])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("method", ["naive", "cell_list"])
def test_fixed_cell_periodic_geometry_keeps_position_and_cell_gradients(
    method: str,
) -> None:
    """Cached inverse geometry does not detach differentiable live pair geometry."""
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [7.8, 0.2, 0.3]],
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    cell = (torch.eye(3, dtype=torch.float32, device="cuda") * 8.0).requires_grad_()
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    state = prepare_neighbor_list(
        positions.detach(),
        0.5,
        cell=cell.detach(),
        pbc=pbc,
        method=method,
        fixed_cell=True,
        max_neighbors=8,
        return_vectors=True,
        return_distances=True,
    )
    runtime_cell = cell.detach().clone().requires_grad_()
    result = neighbor_list(positions, cell=runtime_cell, state=state)
    assert torch.count_nonzero(result[2]) > 0
    result[-2][0, 0].backward()
    assert positions.grad is not None and positions.grad.abs().sum() > 0
    assert runtime_cell.grad is not None and runtime_cell.grad.abs().sum() > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("method", "batched", "strategy"),
    [
        ("naive", False, "scalar"),
        ("batch_naive", True, "scalar"),
        ("cell_list", False, "atom_centric"),
        ("batch_cell_list", True, "atom_centric"),
        ("cell_list", False, "pair_centric"),
    ],
)
def test_fixed_cell_pair_callbacks_match_periodic_geometry(
    method: str, batched: bool, strategy: str
) -> None:
    """Fixed naive and cell-list callbacks preserve periodic pair geometry."""
    system_positions = torch.tensor(
        [[0.1, 0.2, 0.3], [7.8, 0.2, 0.3]], dtype=torch.float32, device="cuda"
    )
    positions = (
        torch.cat((system_positions, system_positions + 2.0))
        if batched
        else system_positions
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 8.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    batch_ptr = None
    if batched:
        cell = cell.repeat(2, 1, 1)
        pbc = pbc.repeat(2, 1)
        batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    cell_batch = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pair_params = torch.arange(
        positions.shape[0], dtype=torch.float32, device="cuda"
    ).reshape(-1, 1)
    state = prepare_neighbor_list(
        positions,
        0.5,
        cell=cell,
        pbc=pbc,
        batch_ptr=batch_ptr,
        method=method,
        strategy=strategy,
        fixed_cell=True,
        max_neighbors=4,
        pair_fn=_prepared_sum_pair_fn,
    )
    result = neighbor_list(
        positions,
        cell=cell.clone(),
        state=state,
        pair_params=pair_params,
    )
    assert result[1].tolist() == [1] * positions.shape[0]
    assert state.pair_energies is not None and state.pair_forces is not None
    assert state.neighbor_matrix_shifts is not None
    for source in range(positions.shape[0]):
        target = int(result[0][source, 0].item())
        shift = state.neighbor_matrix_shifts[source, 0].to(positions.dtype)
        system = source // 2 if batched else 0
        vector = positions[target] - positions[source] + shift @ cell_batch[system]
        distance = vector.norm()
        expected_energy = pair_params[source, 0] + pair_params[target, 0] + distance
        torch.testing.assert_close(state.pair_energies[source, 0], expected_energy)
        torch.testing.assert_close(state.pair_forces[source, 0], -vector)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rotated", [False, True])
def test_fixed_cell_cluster_skew_geometry_matches_lattice_oracle(rotated: bool) -> None:
    """Skew triclinic shifts and vectors match a small independent lattice oracle."""
    base_cell = torch.tensor(
        [[5.0, 0.0, 0.0], [4.7, 1.0, 0.0], [0.0, 0.0, 5.0]],
        dtype=torch.float32,
        device="cuda",
    )
    positions = torch.tensor(
        [[0.3, 0.1, 1.0], [0.5, 1.0, 1.0]], dtype=torch.float32, device="cuda"
    )
    if rotated:
        angle = torch.tensor(0.63, device="cuda")
        rotation = torch.stack(
            (
                torch.stack((angle.cos(), -angle.sin(), angle.new_zeros(()))),
                torch.stack((angle.sin(), angle.cos(), angle.new_zeros(()))),
                torch.tensor([0.0, 0.0, 1.0], device="cuda"),
            )
        )
        base_cell = base_cell @ rotation
        positions = positions @ rotation
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    state = prepare_neighbor_list(
        positions,
        0.6,
        cell=base_cell,
        pbc=pbc,
        method="cluster_tile",
        fixed_cell=True,
        max_neighbors=4,
        max_tiles_per_group=1,
        return_vectors=True,
        return_distances=True,
    )
    result = neighbor_list(positions, cell=base_cell.clone(), state=state)
    shifts = result[2]
    distances = result[3]
    vectors = result[4]
    lattice_shifts = torch.tensor(
        list(product(range(-1, 2), repeat=3)), dtype=base_cell.dtype, device="cuda"
    )
    oracle_vectors = positions[1] - positions[0] + lattice_shifts @ base_cell
    oracle_mask = oracle_vectors.norm(dim=1) < 0.6
    assert int(oracle_mask.sum()) == 1
    oracle_shift = lattice_shifts[oracle_mask][0].to(torch.int32)
    oracle_vector = oracle_vectors[oracle_mask][0]
    assert result[1].tolist() == [1, 1]
    torch.testing.assert_close(shifts[:, 0], torch.stack((oracle_shift, -oracle_shift)))
    torch.testing.assert_close(vectors[0, 0], oracle_vector)
    torch.testing.assert_close(vectors[1, 0], -oracle_vector)
    torch.testing.assert_close(
        distances[:, 0], oracle_vector.norm().expand(2), atol=1e-5, rtol=1e-5
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("method", ["naive", "cell_list", "cluster_tile"])
def test_fixed_cell_fullgraph_accepts_fixed_and_dynamic_states(method: str) -> None:
    """One fullgraph wrapper accepts fixed and dynamic states for every route."""
    positions, cell, fixed_state, dynamic_state = _fixed_cell_compile_inputs(method)

    @torch.compile(fullgraph=True)
    def run(
        values: torch.Tensor, box: torch.Tensor, prepared_state: NeighborListState
    ) -> tuple[torch.Tensor, ...]:
        return neighbor_list(values, cell=box, state=prepared_state)

    fixed = run(positions, cell, fixed_state)
    dynamic = run(positions, cell, dynamic_state)
    assert_neighbor_matrix_equal(fixed, dynamic)
    assert fixed[1].tolist() == [1, 1]
    positions[1, 0] = 4.0
    rebuilt_fixed = run(positions, cell, fixed_state)
    rebuilt_dynamic = run(positions, cell, dynamic_state)
    assert_neighbor_matrix_equal(rebuilt_fixed, rebuilt_dynamic)
    assert rebuilt_fixed[1].tolist() == [0, 0]


def _fixed_cell_compile_inputs(
    method: str,
) -> tuple[torch.Tensor, torch.Tensor, NeighborListState, NeighborListState]:
    """Build shared inputs for prepared fixed/dynamic compiler tests."""
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [7.9, 0.2, 0.3]], dtype=torch.float32, device="cuda"
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 8.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    route_options = {"max_tiles_per_group": 1} if method == "cluster_tile" else {}
    fixed_state = prepare_neighbor_list(
        positions,
        0.5,
        cell=cell,
        pbc=pbc,
        method=method,
        fixed_cell=True,
        max_neighbors=8,
        **route_options,
    )
    dynamic_state = prepare_neighbor_list(
        positions,
        0.5,
        cell=cell,
        pbc=pbc,
        method=method,
        max_neighbors=8,
        **route_options,
    )
    return positions, cell, fixed_state, dynamic_state


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("method", ["naive", "cluster_tile"])
def test_fixed_cell_cuda_graph_replay_rebuilds_positions(method: str) -> None:
    """Supported fixed routes replay with changed positions and cached topology."""
    positions, cell, fixed_state, _ = _fixed_cell_compile_inputs(method)

    @torch.compile(fullgraph=True)
    def run(
        values: torch.Tensor, box: torch.Tensor, prepared_state: NeighborListState
    ) -> tuple[torch.Tensor, ...]:
        return neighbor_list(values, cell=box, state=prepared_state)

    first = run(positions, cell, fixed_state)
    assert first[1].tolist() == [1, 1]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(positions, cell, fixed_state)
    positions[1, 0] = 4.0
    graph.replay()
    torch.cuda.synchronize()
    assert captured[1].tolist() == [0, 0]
