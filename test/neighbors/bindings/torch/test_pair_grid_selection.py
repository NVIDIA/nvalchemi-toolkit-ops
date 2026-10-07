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


"""Validate grid selection through GPU sizing and public NL calls."""

import pytest
import torch

from nvalchemiops.neighbors.base_dispatch import DEFAULT_BATCH_MAX_NBINS
from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors._cell_grid import _select_pair_grid
from nvalchemiops.torch.neighbors.batch_cell_list import (
    batch_cell_list,
    estimate_batch_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.batch_naive import batch_naive_neighbor_list
from nvalchemiops.torch.neighbors.cell_list import cell_list, estimate_cell_list_sizes
from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _workspace_kwargs(atoms, cells, radius):
    """Allocate the public cell workspace and name its reusable arrays."""
    names = (
        "cells_per_dimension",
        "neighbor_search_radius",
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "atoms_per_cell_count",
        "cell_atom_start_indices",
        "cell_atom_list",
    )
    return dict(
        zip(names, allocate_cell_list(atoms, cells, radius, radius.device), strict=True)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch_grid_counts_partially_occupied_warps(dtype):
    """Nineteen source lanes use one warp; sixty-four use two neighbor loops."""
    cell = torch.eye(3, device="cuda", dtype=dtype)[None] * 21.978
    pbc = torch.ones((1, 3), device="cuda", dtype=torch.bool)
    ptr = torch.tensor([0, 4096], device="cuda", dtype=torch.int32)
    _, _, grid = _select_pair_grid(cell, pbc, 6.0, None, ptr, 216, 4)
    assert grid.tolist() == [[6, 6, 6]]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "atoms,side,cutoff", [(4394, 53.547, 25.0), (8192, 43.956, 25.0)]
)
def test_batch_grid_preserves_source_passes(dtype, atoms, side, cutoff):
    """Coarsening preserves the configured source-pass count in dense cells."""
    cell = torch.eye(3, device="cuda", dtype=dtype)[None] * side
    pbc = torch.ones((1, 3), device="cuda", dtype=torch.bool)
    ptr = torch.tensor([0, atoms], device="cuda", dtype=torch.int32)
    _, _, grid = _select_pair_grid(cell, pbc, cutoff, None, ptr, 64, 4)
    assert grid.tolist() == [[4, 4, 4]]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "atoms,side,cutoff,expected", [(16384, 55.382, 15.0, 7), (65536, 87.912, 25.0, 6)]
)
def test_dense_batch_accounts_for_neighbor_loop(dtype, atoms, side, cutoff, expected):
    """Dense cell loops retain the finer grid when coarse cells increase work."""
    cell = torch.eye(3, device="cuda", dtype=dtype)[None] * side
    pbc = torch.ones((1, 3), device="cuda", dtype=torch.bool)
    ptr = torch.tensor([0, atoms], device="cuda", dtype=torch.int32)
    _, _, grid = _select_pair_grid(cell, pbc, cutoff, None, ptr, 8192, 4)
    assert grid.tolist() == [[expected] * 3]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_grid_rejects_more_neighbor_work(dtype):
    """A smaller stencil must also account for its longer neighbor-atom loop."""
    cell = torch.eye(3, device="cuda", dtype=dtype)[None] * 12
    pbc = torch.ones((1, 3), device="cuda", dtype=torch.bool)
    ptr = torch.tensor([0, 512], device="cuda", dtype=torch.int32)
    _, _, grid = _select_pair_grid(cell, pbc, 6.0, None, ptr, 64, 4, single_system=True)
    # Two cells per axis reduce offsets 125 -> 27 but increase pair checks
    # 512,000 -> 884,736 and the per-source neighbor loop 8 -> 64.
    assert grid.tolist() == [[4, 4, 4]]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("spare_atoms", [0, 3])
def test_single_workspace_overwrites_previous_contents(dtype, spare_atoms):
    """Poisoned buffers rebuild physical pairs and preserve zeroed spare tails."""
    atoms = 32
    generator = torch.Generator(device="cuda").manual_seed(53)
    fractions = torch.rand((atoms, 3), generator=generator, device="cuda", dtype=dtype)
    cell = torch.eye(3, device="cuda", dtype=dtype) * 12
    pbc = torch.ones(3, device="cuda", dtype=torch.bool)
    capacity, radius = estimate_cell_list_sizes(cell, pbc, 6.0)
    workspace = _workspace_kwargs(atoms + spare_atoms, capacity, radius)
    pointers = {name: value.data_ptr() for name, value in workspace.items()}
    for side in (12, 18, 12):
        cell.copy_(torch.eye(3, device="cuda", dtype=dtype) * side)
        common = dict(
            positions=fractions * side,
            cell=cell,
            pbc=pbc,
            cutoff=6.0,
            max_neighbors=128,
            return_neighbor_list=True,
        )
        expected = cell_list(grid_policy="adaptive", **common, strategy="atom_centric")
        for value in workspace.values():
            value.fill_(7)
        actual = cell_list(
            grid_policy="adaptive", **common, strategy="pair_centric", **workspace
        )
        assert signature(actual) == signature(expected)
        assert int(workspace["atoms_per_cell_count"].sum()) == atoms
        for name in ("atom_periodic_shifts", "atom_to_cell_mapping", "cell_atom_list"):
            assert torch.all(workspace[name][atoms:] == 0)
        assert {name: value.data_ptr() for name, value in workspace.items()} == pointers


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("reuse_workspace", [False, True])
def test_adaptive_default_strategy_calls_select_grid(batched, public, reuse_workspace):
    """Default cell calls refresh caller grids and preserve physical pairs."""
    atoms = 4096 if batched else 512
    systems = 2 if batched else 1
    side = 21.978 if batched else 17.445
    expected = 6 if batched else 4
    generator = torch.Generator(device="cuda").manual_seed(37)
    positions = torch.rand((atoms * systems, 3), generator=generator, device="cuda")
    positions *= side
    cell = torch.eye(3, device="cuda") * side
    pbc = torch.ones(3, device="cuda", dtype=torch.bool)
    common = dict(
        positions=positions,
        cutoff=6.0,
        cell=cell,
        pbc=pbc,
        max_neighbors=1024,
        return_neighbor_list=True,
    )
    if batched:
        common.update(
            cell=cell.repeat(systems, 1, 1),
            pbc=pbc.repeat(systems, 1),
            batch_idx=torch.arange(
                systems, device="cuda", dtype=torch.int32
            ).repeat_interleave(atoms),
            batch_ptr=torch.arange(systems + 1, device="cuda", dtype=torch.int32)
            * atoms,
        )
    estimate = estimate_batch_cell_list_sizes if batched else estimate_cell_list_sizes
    capacity, radius = estimate(common["cell"], common["pbc"], 6.0)
    workspace = _workspace_kwargs(atoms * systems, capacity, radius)
    call = batch_cell_list if batched else cell_list
    reference = call(grid_policy="adaptive", **common, strategy="atom_centric")
    if public:
        common["method"] = "batch_cell_list" if batched else "cell_list"
        call = neighbor_list
    actual = call(
        grid_policy="adaptive", **common, **(workspace if reuse_workspace else {})
    )
    if reuse_workspace:
        assert (
            workspace["cells_per_dimension"].reshape(systems, 3).tolist()
            == [[expected] * 3] * systems
        )
    assert signature(actual) == signature(reference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("strategy", ["auto", "pair_centric"])
def test_compiled_single_grid_tracks_box(dtype, strategy):
    """Single-system fullgraph calls update grids and radii in fixed storage."""
    atoms = 512
    generator = torch.Generator(device="cuda").manual_seed(43)
    fractions = torch.rand((atoms, 3), generator=generator, device="cuda", dtype=dtype)
    cell = torch.eye(3, device="cuda", dtype=dtype) * 17.445
    pbc = torch.ones(3, device="cuda", dtype=torch.bool)
    positions = fractions * 17.445
    capacity, radius = estimate_cell_list_sizes(cell, pbc, 6.0)
    workspace = _workspace_kwargs(atoms, capacity, radius)
    outputs = dict(
        neighbor_matrix=torch.empty((atoms, atoms), device="cuda", dtype=torch.int32),
        neighbor_matrix_shifts=torch.empty(
            (atoms, atoms, 3), device="cuda", dtype=torch.int32
        ),
        num_neighbors=torch.zeros(atoms, device="cuda", dtype=torch.int32),
    )
    pointers = {name: value.data_ptr() for name, value in (workspace | outputs).items()}
    query = torch.compile(cell_list, fullgraph=True)
    for scale in (1.0, 2.0, 0.9):
        cell.copy_(torch.eye(3, device="cuda", dtype=dtype) * (17.445 * scale))
        positions.copy_(fractions * (17.445 * scale))
        kwargs = dict(
            positions=positions, cell=cell, pbc=pbc, cutoff=6.0, max_neighbors=atoms
        )
        expected = cell_list(grid_policy="adaptive", **kwargs, strategy="atom_centric")
        actual = query(
            grid_policy="adaptive", **kwargs, strategy=strategy, **workspace, **outputs
        )
        assert workspace["cells_per_dimension"].tolist() == (
            [2] * 3 if scale == 2.0 else [4] * 3
        )
        assert workspace["neighbor_search_radius"].tolist() == (
            [1] * 3 if scale == 2.0 else [2] * 3
        )
        torch.testing.assert_close(actual[1], expected[1])
        for row in range(atoms):
            count = int(actual[1][row])
            actual_pairs = torch.cat(
                (actual[0][row, :count, None], actual[2][row, :count]), dim=1
            )
            expected_pairs = torch.cat(
                (expected[0][row, :count, None], expected[2][row, :count]), dim=1
            )
            assert sorted(actual_pairs.tolist()) == sorted(expected_pairs.tolist())
        assert {
            name: value.data_ptr() for name, value in (workspace | outputs).items()
        } == pointers


@pytest.mark.parametrize(
    "atoms,side,expected", [(512, 17.445, 2), (1024, 21.978, 3), (2048, 27.691, 4)]
)
def test_public_reused_workspace_selects_same_grid(atoms, side, expected):
    """Reused public buffers receive the selected grid, radii, and same pairs."""
    generator = torch.Generator(device="cuda").manual_seed(19)
    positions = torch.rand((atoms * 2, 3), generator=generator, device="cuda") * side
    cell = torch.eye(3, device="cuda").repeat(2, 1, 1) * side
    pbc = torch.ones((2, 3), device="cuda", dtype=torch.bool)
    idx = torch.arange(2, device="cuda", dtype=torch.int32).repeat_interleave(atoms)
    ptr = torch.arange(3, device="cuda", dtype=torch.int32) * atoms
    capacity, radius = estimate_batch_cell_list_sizes(cell, pbc, 6.0)
    workspace = _workspace_kwargs(atoms * 2, capacity, radius)
    pointers = {name: value.data_ptr() for name, value in workspace.items()}
    common = dict(
        positions=positions,
        cutoff=6.0,
        cell=cell,
        pbc=pbc,
        batch_idx=idx,
        batch_ptr=ptr,
        max_neighbors=256,
        method="batch_cell_list_pair_centric",
        return_neighbor_list=True,
    )
    reference = neighbor_list(grid_policy="adaptive", **common)
    actual = neighbor_list(grid_policy="adaptive", **common, **workspace)
    assert workspace["cells_per_dimension"].tolist() == [[expected] * 3] * 2
    assert signature(actual) == signature(reference)
    assert {name: value.data_ptr() for name, value in workspace.items()} == pointers


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nondefault_stream", [False, True])
@pytest.mark.parametrize("direct_without_ptr", [False, True])
def test_compiled_reused_workspace_selects_current_grid(
    dtype, nondefault_stream, direct_without_ptr
):
    """Fullgraph calls refresh selected grids for changed boxes in fixed buffers."""
    stream = torch.cuda.Stream() if nondefault_stream else torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        atoms = 1024
        generator = torch.Generator(device="cuda").manual_seed(29)
        fraction = torch.rand(
            (atoms * 2, 3), generator=generator, device="cuda", dtype=dtype
        )
        positions = torch.empty_like(fraction)
        cell = torch.eye(3, device="cuda", dtype=dtype).repeat(2, 1, 1) * 21.978
        base_cell = cell.clone()
        pbc = torch.ones((2, 3), device="cuda", dtype=torch.bool)
        idx = torch.arange(2, device="cuda", dtype=torch.int32).repeat_interleave(atoms)
        ptr = torch.arange(3, device="cuda", dtype=torch.int32) * atoms
        capacity, radius = estimate_batch_cell_list_sizes(cell, pbc, 6.0)
        workspace = _workspace_kwargs(atoms * 2, capacity, radius)
        outputs = dict(
            neighbor_matrix=torch.empty(
                (atoms * 2, atoms), device="cuda", dtype=torch.int32
            ),
            neighbor_matrix_shifts=torch.empty(
                (atoms * 2, atoms, 3), device="cuda", dtype=torch.int32
            ),
            num_neighbors=torch.zeros(atoms * 2, device="cuda", dtype=torch.int32),
        )
        common = dict(
            positions=positions,
            cutoff=6.0,
            cell=cell,
            pbc=pbc,
            batch_idx=idx,
            batch_ptr=ptr,
            max_neighbors=atoms,
            method="batch_cell_list_pair_centric",
        )
        pointers = {
            name: value.data_ptr() for name, value in (workspace | outputs).items()
        }
        compiled_kwargs = common.copy()
        if direct_without_ptr:
            compiled_kwargs.pop("batch_ptr")
            compiled_kwargs.pop("method")
            compiled_kwargs["strategy"] = "pair_centric"
        compiled_query = torch.compile(
            batch_cell_list if direct_without_ptr else neighbor_list, fullgraph=True
        )
        for scale in (1.0, 1.5, 0.9, 1.0):
            cell.copy_(base_cell * scale)
            positions.copy_(fraction * (21.978 * scale))
            expected = neighbor_list(grid_policy="adaptive", **common)
            _, expected_radius, expected_grid = _select_pair_grid(
                cell, pbc, 6.0, idx, ptr, capacity // 2, 4
            )
            actual = compiled_query(
                grid_policy="adaptive", **compiled_kwargs, **workspace, **outputs
            )
            torch.testing.assert_close(workspace["cells_per_dimension"], expected_grid)
            torch.testing.assert_close(
                workspace["neighbor_search_radius"], expected_radius
            )
            torch.testing.assert_close(actual[1], expected[1])
            actual_host = tuple(value.cpu() for value in actual)
            expected_host = tuple(value.cpu() for value in expected)
            for row in range(atoms * 2):
                count = int(actual_host[1][row])
                actual_pairs = torch.cat(
                    (actual_host[0][row, :count, None], actual_host[2][row, :count]),
                    dim=1,
                )
                expected_pairs = torch.cat(
                    (
                        expected_host[0][row, :count, None],
                        expected_host[2][row, :count],
                    ),
                    dim=1,
                )
                assert sorted(actual_pairs.tolist()) == sorted(expected_pairs.tolist())
            assert {
                name: value.data_ptr() for name, value in (workspace | outputs).items()
            } == pointers


@pytest.mark.parametrize("capacity,expected", [(2, 1), (3, 1), (16, 2)])
def test_reused_workspace_capacity_and_current_radii(capacity, expected):
    """Select within supplied capacity and refresh radii before building."""
    gen = torch.Generator(device="cuda").manual_seed(23)
    pos = torch.rand((32, 3), generator=gen, device="cuda") * 12
    cell = torch.eye(3, device="cuda").repeat(2, 1, 1) * 12
    pbc = torch.ones((2, 3), device="cuda", dtype=torch.bool)
    idx = torch.arange(2, device="cuda", dtype=torch.int32).repeat_interleave(16)
    ptr = torch.tensor([0, 16, 32], device="cuda", dtype=torch.int32)
    radius = torch.zeros((2, 3), device="cuda", dtype=torch.int32)
    common = dict(
        positions=pos,
        cutoff=6.0,
        cell=cell,
        pbc=pbc,
        batch_idx=idx,
        batch_ptr=ptr,
        method="batch_cell_list_pair_centric",
        return_neighbor_list=True,
        max_neighbors=128,
    )
    reference = neighbor_list(grid_policy="adaptive", **common)
    workspace = _workspace_kwargs(32, capacity, radius)
    actual = neighbor_list(grid_policy="adaptive", **common, **workspace)
    assert workspace["cells_per_dimension"].tolist() == [[expected] * 3] * 2
    assert workspace["neighbor_search_radius"].tolist() == [[1, 1, 1]] * 2
    assert signature(actual) == signature(reference)
    too_small = _workspace_kwargs(32, 1, radius)
    with pytest.raises(ValueError, match="cell workspace.*capacity"):
        neighbor_list(grid_policy="adaptive", **common, **too_small)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("single_system", [False, True])
@pytest.mark.parametrize(
    "atoms,side,cutoff,expected",
    [
        (512, 17.445, 6, 2),
        (1024, 21.978, 6, 3),
        (2048, 27.691, 6, 4),
        (512, 17.445, 15, 2),
        (2048, 27.691, 15, 4),
        (2048, 27.691, 25, 4),
        (686, 28.833, 25, 3),
        (4394, 53.547, 25, 4),
    ],
)
def test_original_choices(dtype, single_system, atoms, side, cutoff, expected):
    """Balance stencil visits and source passes in the measured transition cases."""
    cell = torch.eye(3, device="cuda", dtype=dtype).unsqueeze(0) * side
    pbc = torch.ones((1, 3), device="cuda", dtype=torch.bool)
    ptr = torch.tensor([0, atoms], device="cuda", dtype=torch.int32)
    total, radius, grid = _select_pair_grid(
        cell,
        pbc,
        cutoff,
        None,
        ptr,
        DEFAULT_BATCH_MAX_NBINS,
        4,
        single_system=single_system,
    )
    if single_system:
        expected = 6 if (atoms, cutoff) == (1024, 6) else 4
    assert grid.tolist() == [[expected] * 3]
    assert total == expected**3


@pytest.mark.parametrize("atoms", [0, 1, 17, 512, 1024])
def test_single_scalar_population_matches_boundaries(atoms):
    """Passing the known population avoids scratch without changing selection."""
    cell = torch.eye(3, device="cuda")[None] * 12
    pbc = torch.ones((1, 3), device="cuda", dtype=torch.bool)
    ptr = torch.tensor([0, atoms], device="cuda", dtype=torch.int32)
    expected = _select_pair_grid(cell, pbc, 6.0, None, ptr, 64, 4, single_system=True)
    actual = _select_pair_grid(
        cell, pbc, 6.0, None, None, 64, 4, single_system=True, num_atoms=atoms
    )
    assert actual[0] == expected[0]
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2])


@pytest.mark.parametrize("cap", [1, 3, 27, 64, 8192])
def test_grid_capacity_and_changing_population(cap):
    """Respect allocation limits and read updated populations every call."""
    cell = torch.eye(3, device="cuda").unsqueeze(0) * 17.445
    pbc = torch.ones((1, 3), device="cuda", dtype=torch.bool)
    ptr = torch.tensor([0, 512], device="cuda", dtype=torch.int32)
    for atoms, expected in ((512, 2), (2048, 4), (512, 2)):
        ptr[1] = atoms
        total, radii, grid = _select_pair_grid(cell, pbc, 15.0, None, ptr, cap, 4)
        assert 1 <= total <= cap
        assert total == int(grid.prod())
        assert torch.all(radii >= torch.ceil(15.0 * grid / 17.445))
        if cap >= 64:
            assert grid.tolist() == [[expected] * 3]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("invalid_system", [0, 1])
def test_selector_rejects_one_degenerate_cell(dtype, invalid_system):
    """Fused sizing rejects a zero-volume cell within a valid batch."""
    cell = torch.eye(3, device="cuda", dtype=dtype).repeat(2, 1, 1) * 17.445
    cell[invalid_system, 0].zero_()
    pbc = torch.ones((2, 3), device="cuda", dtype=torch.bool)
    ptr = torch.tensor([0, 512, 1024], device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="volume == 0.0"):
        _select_pair_grid(cell, pbc, 6.0, None, ptr, DEFAULT_BATCH_MAX_NBINS, 4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_allocation_total_exceeds_int32(dtype):
    """Accumulate valid per-system cell counts without int32 overflow."""
    systems = 3000
    cell = torch.eye(3, device="cuda", dtype=dtype).repeat(systems, 1, 1) * 100
    pbc = torch.ones((systems, 3), device="cuda", dtype=torch.bool)
    ptr = torch.arange(systems + 1, device="cuda", dtype=torch.int32)
    total, _, grid = _select_pair_grid(cell, pbc, 1.0, None, ptr, 1_000_000, 4)
    assert total > torch.iinfo(torch.int32).max
    assert total == int(grid.to(torch.int64).prod(dim=1).sum())


@pytest.mark.parametrize(
    "systems,invalid",
    [(0, False)]
    + [
        (systems, invalid)
        for systems in (1, 7, 63, 64, 65, 129)
        for invalid in (False, True)
    ],
)
def test_allocation_reads_one_validated_scalar(systems, invalid):
    """Automatic sizing reports valid sizes and errors through one scalar read."""
    cell = torch.eye(3, device="cuda").repeat(systems, 1, 1) * 12
    pbc = torch.ones((systems, 3), device="cuda", dtype=torch.bool)
    ptr = torch.arange(systems + 1, device="cuda", dtype=torch.int32) * 32
    _select_pair_grid(cell, pbc, 6.0, None, ptr, 64, 4)
    if invalid:
        cell[-1, 0].zero_()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        if invalid:
            with pytest.raises(RuntimeError, match="volume == 0.0"):
                _select_pair_grid(cell, pbc, 6.0, None, ptr, 64, 4)
        else:
            total, _, grid = _select_pair_grid(cell, pbc, 6.0, None, ptr, 64, 4)
    reads = sum(
        event.count
        for event in profile.key_averages()
        if event.key == "aten::_local_scalar_dense"
    )
    assert reads == 1
    if not invalid:
        assert total == int(grid.to(torch.int64).prod(dim=1).sum())


@pytest.mark.parametrize("supplied_ptr", [False, True])
@pytest.mark.parametrize("reuse_workspace", [False, True])
def test_direct_binding_pair_geometry(supplied_ptr, reuse_workspace):
    """Direct calls with either batch representation preserve differentiable geometry."""
    generator = torch.Generator(device="cuda").manual_seed(17)
    positions = (
        torch.rand((64, 3), device="cuda", generator=generator) * 12
    ).requires_grad_()
    cell = (torch.eye(3, device="cuda").repeat(2, 1, 1) * 12).requires_grad_()
    pbc = torch.ones((2, 3), device="cuda", dtype=torch.bool)
    idx = torch.arange(2, device="cuda", dtype=torch.int32).repeat_interleave(32)
    ptr = (
        torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
        if supplied_ptr
        else None
    )
    common = dict(
        positions=positions,
        cutoff=6.0,
        cell=cell,
        pbc=pbc,
        batch_idx=idx,
        max_neighbors=256,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
    )
    workspace = {}
    if reuse_workspace:
        capacity, radius = estimate_batch_cell_list_sizes(cell, pbc, 6.0)
        workspace = _workspace_kwargs(64, capacity, radius)
    actual = batch_cell_list(
        grid_policy="adaptive",
        **common,
        **workspace,
        batch_ptr=ptr,
        strategy="pair_centric",
    )
    reference = batch_naive_neighbor_list(
        **common, max_atoms_per_system=32, strategy="scalar"
    )
    assert signature(actual) == signature(reference)

    def loss(output):
        """Combine distance and vector outputs for a position/cell gradient check."""
        return output[3].square().sum() + output[4].square().sum()

    actual_grad = torch.autograd.grad(
        loss(actual), (positions, cell), retain_graph=True
    )
    expected_grad = torch.autograd.grad(loss(reference), (positions, cell))
    for value, expected in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(value, expected, rtol=1e-4, atol=1e-4)


def signature(result):
    """Return sorted physical pairs and periodic images."""
    indices, _, shifts = result[:3]
    return sorted(torch.cat((indices.T, shifts), dim=1).cpu().tolist())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nondefault_stream", [False, True])
@pytest.mark.parametrize("reuse_workspace", [False, True])
def test_uneven_current_boxes(dtype, nondefault_stream, reuse_workspace):
    """Match exhaustive pairs as uneven boxes and poisoned workspaces change."""
    stream = torch.cuda.Stream() if nondefault_stream else torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        counts = [17, 0, 129]
        ptr = torch.tensor([0, 17, 17, 146], device="cuda", dtype=torch.int32)
        batch_idx = torch.repeat_interleave(
            torch.arange(3, device="cuda", dtype=torch.int32),
            torch.tensor(counts, device="cuda"),
        )
        gen = torch.Generator(device="cuda").manual_seed(31)
        fraction = torch.rand((146, 3), device="cuda", dtype=dtype, generator=gen)
        pbc = torch.tensor(
            [[True, True, True], [False, False, False], [True, False, True]],
            device="cuda",
        )
        base = torch.tensor(
            [
                [[16, 0, 0], [0, 7, 0], [0, 0, 9]],
                [[8, 0, 0], [0, 8, 0], [0, 0, 8]],
                [[-17, 2, 1], [0, 12, 2], [0, 1, 10]],
            ],
            device="cuda",
            dtype=dtype,
        )
        cell = base.clone()
        pos = torch.empty_like(fraction)
        workspace = (
            _workspace_kwargs(
                146,
                len(counts) * DEFAULT_BATCH_MAX_NBINS,
                torch.zeros((len(counts), 3), device="cuda", dtype=torch.int32),
            )
            if reuse_workspace
            else {}
        )
        for scale in (0.95, 1.0, 1.05, 1.5, 0.95):
            cell.copy_(base * scale)
            pos.copy_(
                torch.bmm(fraction.unsqueeze(1), cell[batch_idx.long()]).squeeze(1)
            )
            common = dict(
                positions=pos,
                cutoff=6.0,
                cell=cell,
                pbc=pbc,
                batch_idx=batch_idx,
                max_neighbors=1024,
                return_neighbor_list=True,
            )
            for value in workspace.values():
                value.fill_(-17)
            actual = neighbor_list(
                grid_policy="adaptive",
                **common,
                batch_ptr=ptr,
                method="batch_cell_list_pair_centric",
                **workspace,
            )
            expected = batch_naive_neighbor_list(
                **common, max_atoms_per_system=max(counts), strategy="scalar"
            )
            assert signature(actual) == signature(expected)
        total, radius, grid = _select_pair_grid(
            cell, pbc, 6.0, batch_idx, ptr, DEFAULT_BATCH_MAX_NBINS, 4
        )
        assert total == int(grid.prod(dim=1).sum())


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("reuse_workspace", [False, True])
def test_grid_policy_default_and_opt_in(batched, public, reuse_workspace):
    """Default grids retain configured sizing while adaptive preserves pairs."""
    systems = 3 if batched else 1
    populations = [17, 0, 32] if batched else [32]
    atoms = sum(populations)
    generator = torch.Generator(device="cuda").manual_seed(71)
    positions = torch.rand((atoms, 3), generator=generator, device="cuda") * 12
    cell = torch.eye(3, device="cuda").repeat(systems, 1, 1) * 12
    pbc = torch.ones((systems, 3), device="cuda", dtype=torch.bool)
    common = dict(
        positions=positions,
        cutoff=6.0,
        cell=cell,
        pbc=pbc,
        max_neighbors=64,
        return_neighbor_list=True,
    )
    if batched:
        common.update(
            batch_idx=torch.repeat_interleave(
                torch.arange(systems, device="cuda", dtype=torch.int32),
                torch.tensor(populations, device="cuda"),
            ),
            batch_ptr=torch.tensor([0, 17, 17, 49], device="cuda", dtype=torch.int32),
        )
    call = neighbor_list if public else (batch_cell_list if batched else cell_list)
    if public:
        common["method"] = (
            "batch_cell_list_pair_centric" if batched else "cell_list_pair_centric"
        )
    else:
        common["strategy"] = "pair_centric"
    results = []
    grids = []
    for policy in (None, "configured", "adaptive"):
        radius = torch.full(
            (systems, 3) if batched else (3,), 2, device="cuda", dtype=torch.int32
        )
        workspace = _workspace_kwargs(atoms, 64 * systems, radius)
        policy_kwargs = {} if policy is None else {"grid_policy": policy}
        result = call(
            **common, **(workspace if reuse_workspace else {}), **policy_kwargs
        )
        results.append(signature(result))
        if reuse_workspace:
            grids.append(workspace["cells_per_dimension"].clone())
    assert results[0] == results[1] == results[2]
    if reuse_workspace:
        torch.testing.assert_close(grids[0], torch.full_like(grids[0], 4))
        torch.testing.assert_close(grids[0], grids[1])
        if batched:
            assert not torch.equal(grids[1], grids[2])
        else:
            # This geometry increases estimated pair work on coarser grids.
            torch.testing.assert_close(grids[1], grids[2])
    with pytest.raises(ValueError, match="grid_policy"):
        call(**common, grid_policy="unknown")
