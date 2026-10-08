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

"""Target cell-list queries refresh search radii for the current grid."""

import pytest
import torch

from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.batch_cell_list import (
    batch_cell_list,
    estimate_batch_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.batch_naive import batch_naive_neighbor_list
from nvalchemiops.torch.neighbors.cell_list import (
    cell_list,
    estimate_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.naive import naive_neighbor_list
from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_WORKSPACE_NAMES = (
    "cells_per_dimension",
    "neighbor_search_radius",
    "atom_periodic_shifts",
    "atom_to_cell_mapping",
    "atoms_per_cell_count",
    "cell_atom_start_indices",
    "cell_atom_list",
)


def _workspace_kwargs(total_atoms, cell, pbc, cutoff):
    """Allocate reusable batch cell-list storage for the supplied geometry."""
    capacity, radius = estimate_batch_cell_list_sizes(cell, pbc, cutoff)
    return dict(
        zip(
            _WORKSPACE_NAMES,
            allocate_cell_list(total_atoms, capacity, radius, cell.device),
            strict=True,
        )
    )


def _call(public, positions, cell, pbc, batch_idx, batch_ptr, cutoff, **kwargs):
    """Run either public dispatch or the direct batched cell-list binding."""
    common = dict(
        positions=positions,
        cutoff=cutoff,
        cell=cell,
        pbc=pbc,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        max_neighbors=kwargs.pop("max_neighbors", 256),
    )
    if public:
        method = kwargs.pop("method", "batch_cell_list_pair_centric")
        return neighbor_list(method=method, **common, **kwargs)
    strategy = kwargs.pop("strategy", "pair_centric")
    return batch_cell_list(strategy=strategy, **common, **kwargs)


def _assert_matches_naive(actual, expected):
    """Compare counts and sorted (neighbor, periodic image) rows."""
    actual_matrix, actual_counts, actual_images = actual[:3]
    expected_matrix, expected_counts, expected_images = expected[:3]
    torch.testing.assert_close(actual_counts, expected_counts)
    actual_counts = actual_counts.cpu().tolist()
    expected_counts = expected_counts.cpu().tolist()
    actual_matrix = actual_matrix.cpu()
    actual_images = actual_images.cpu()
    expected_matrix = expected_matrix.cpu()
    expected_images = expected_images.cpu()
    for row, (actual_count, expected_count) in enumerate(
        zip(actual_counts, expected_counts, strict=True)
    ):
        assert actual_count <= actual_matrix.shape[1]
        assert expected_count <= expected_matrix.shape[1]
        actual_pairs = sorted(
            (
                int(actual_matrix[row, col]),
                *map(int, actual_images[row, col].tolist()),
            )
            for col in range(actual_count)
        )
        expected_pairs = sorted(
            (
                int(expected_matrix[row, col]),
                *map(int, expected_images[row, col].tolist()),
            )
            for col in range(expected_count)
        )
        assert actual_pairs == expected_pairs


def _naive(positions, cell, pbc, batch_idx, batch_ptr, cutoff, target_indices, width):
    """Compute an independent scalar reference for selected batch rows."""
    return batch_naive_neighbor_list(
        positions=positions,
        cutoff=cutoff,
        cell=cell,
        pbc=pbc,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        target_indices=target_indices,
        max_neighbors=width,
        max_atoms_per_system=int((batch_ptr[1:] - batch_ptr[:-1]).max()),
        strategy="scalar",
    )


def _small_batch(side=12.0, skew=False, dtype=torch.float32):
    """Make two deterministic systems and compact targets in reverse order."""
    generator = torch.Generator(device="cuda").manual_seed(91)
    fractions = torch.rand((64, 3), generator=generator, device="cuda", dtype=dtype)
    if skew:
        base = torch.tensor(
            [[side, 0.0, 0.0], [0.22 * side, 0.91 * side, 0.0], [0.0, 0.0, side]],
            device="cuda",
            dtype=dtype,
        )
        cell = base.repeat(2, 1, 1)
        pbc = torch.tensor([[True, False, True], [True, False, True]], device="cuda")
    else:
        cell = torch.eye(3, device="cuda", dtype=dtype).repeat(2, 1, 1) * side
        pbc = torch.ones((2, 3), device="cuda", dtype=torch.bool)
    positions = torch.cat((fractions[:32] @ cell[0], fractions[32:] @ cell[1]))
    batch_idx = torch.arange(2, device="cuda", dtype=torch.int32).repeat_interleave(32)
    batch_ptr = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
    target_indices = torch.tensor([49, 5, 33, 0], device="cuda", dtype=torch.int32)
    return positions, cell, pbc, batch_idx, batch_ptr, target_indices


@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize(
    "transition,initial_side,initial_cutoff,target_side,target_cutoff,target_policy,dtype",
    [
        ("unchanged", 12.0, 6.0, 12.0, 6.0, None, torch.float32),
        ("unchanged", 12.0, 6.0, 12.0, 6.0, "configured", torch.float32),
        ("unchanged", 12.0, 6.0, 12.0, 6.0, "adaptive", torch.float32),
        ("cutoff_growth", 12.0, 3.0, 12.0, 6.0, "adaptive", torch.float32),
        ("cell_shrink", 18.0, 6.0, 12.0, 6.0, "adaptive", torch.float32),
        ("unchanged", 12.0, 6.0, 12.0, 6.0, "adaptive", torch.float64),
    ],
)
def test_full_adaptive_workspace_reuse_for_target_queries(
    public,
    transition,
    initial_side,
    initial_cutoff,
    target_side,
    target_cutoff,
    target_policy,
    dtype,
):
    """Reused adaptive workspaces cover target rows after geometry transitions."""
    positions, cell, pbc, batch_idx, batch_ptr, targets = _small_batch(
        initial_side, dtype=dtype
    )
    workspace = _workspace_kwargs(64, cell, pbc, max(initial_cutoff, target_cutoff))
    seed = _call(
        public,
        positions,
        cell,
        pbc,
        batch_idx,
        batch_ptr,
        initial_cutoff,
        grid_policy="adaptive",
        **workspace,
    )
    seed_reference = _naive(
        positions, cell, pbc, batch_idx, batch_ptr, initial_cutoff, None, 256
    )
    _assert_matches_naive(seed, seed_reference)

    if transition == "cell_shrink":
        cell.copy_(cell * (target_side / initial_side))
        positions.mul_(target_side / initial_side)
    policy_kwargs = {} if target_policy is None else {"grid_policy": target_policy}
    actual = _call(
        public,
        positions,
        cell,
        pbc,
        batch_idx,
        batch_ptr,
        target_cutoff,
        target_indices=targets,
        **policy_kwargs,
        **workspace,
    )
    expected = _naive(
        positions,
        cell,
        pbc,
        batch_idx,
        batch_ptr,
        target_cutoff,
        targets,
        256,
    )
    _assert_matches_naive(actual, expected)

    fresh = _call(
        public,
        positions,
        cell,
        pbc,
        batch_idx,
        batch_ptr,
        target_cutoff,
        target_indices=targets,
        **policy_kwargs,
    )
    _assert_matches_naive(
        fresh,
        _naive(
            positions,
            cell,
            pbc,
            batch_idx,
            batch_ptr,
            target_cutoff,
            targets,
            fresh[0].shape[1],
        ),
    )


@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_configured_target_and_skew_mixed_pbc(public, dtype):
    """Configured target sizing after an adaptive full build handles skewed mixed PBC."""
    positions, cell, pbc, batch_idx, batch_ptr, targets = _small_batch(
        12.0, skew=True, dtype=dtype
    )
    workspace = _workspace_kwargs(64, cell, pbc, 6.0)
    full = _call(
        public,
        positions,
        cell,
        pbc,
        batch_idx,
        batch_ptr,
        6.0,
        grid_policy="adaptive",
        **workspace,
    )
    _assert_matches_naive(
        full, _naive(positions, cell, pbc, batch_idx, batch_ptr, 6.0, None, 256)
    )
    targeted = _call(
        public,
        positions,
        cell,
        pbc,
        batch_idx,
        batch_ptr,
        6.0,
        target_indices=targets,
        grid_policy="configured",
        **workspace,
    )
    _assert_matches_naive(
        targeted,
        _naive(positions, cell, pbc, batch_idx, batch_ptr, 6.0, targets, 256),
    )


@pytest.mark.parametrize(
    "public,compiled", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("target_policy", [None, "configured", "adaptive"])
def test_single_full_adaptive_workspace_reuse_for_targets(
    public, compiled, target_policy
):
    """Single-system target queries refresh the reused full adaptive grid radius."""
    atoms = 513
    generator = torch.Generator(device="cuda").manual_seed(151)
    sides = torch.tensor([9.0, 9.0, 384.0], device="cuda")
    positions = torch.rand((atoms, 3), generator=generator, device="cuda") * sides
    # This direct-image pair spans three configured cells but two adaptive cells.
    positions[:2] = torch.tensor([[1.98, 1.0, 1.0], [7.02, 1.0, 1.0]], device="cuda")
    cell = torch.diag(sides)
    pbc = torch.ones(3, device="cuda", dtype=torch.bool)
    targets = torch.tensor([0, 1, 17, 512], device="cuda", dtype=torch.int32)
    capacity, radius = estimate_cell_list_sizes(cell, pbc, 6.0)
    workspace = dict(
        zip(
            _WORKSPACE_NAMES,
            allocate_cell_list(atoms, capacity, radius, cell.device),
            strict=True,
        )
    )

    target_call = torch.compile(cell_list, fullgraph=True) if compiled else cell_list

    def call(target_indices=None, **kwargs):
        common = dict(
            positions=positions,
            cutoff=6.0,
            cell=cell,
            pbc=pbc,
            max_neighbors=256,
            target_indices=target_indices,
            **kwargs,
        )
        if public:
            return neighbor_list(method="cell_list_pair_centric", **common)
        fn = target_call if target_indices is not None else cell_list
        return fn(strategy="pair_centric", **common)

    seed = call(grid_policy="adaptive", **workspace)
    seed_reference = naive_neighbor_list(
        positions=positions,
        cutoff=6.0,
        cell=cell,
        pbc=pbc,
        max_neighbors=256,
        strategy="scalar",
    )
    _assert_matches_naive(seed, seed_reference)

    target_kwargs = {} if target_policy is None else {"grid_policy": target_policy}
    actual = call(target_indices=targets, **target_kwargs, **workspace)
    expected = naive_neighbor_list(
        positions=positions,
        cutoff=6.0,
        cell=cell,
        pbc=pbc,
        max_neighbors=256,
        target_indices=targets,
        strategy="scalar",
    )
    _assert_matches_naive(actual, expected)


def test_auto_target_query_uneven_system_populations():
    """Auto dispatch handles compact targets in uneven pair-centric systems."""
    counts = (512, 8192)
    sides = (21.978, 40.0)
    generator = torch.Generator(device="cuda").manual_seed(117)
    fractions = torch.rand((sum(counts), 3), generator=generator, device="cuda")
    cells = torch.stack([torch.eye(3, device="cuda") * side for side in sides], dim=0)
    positions = torch.cat(
        (
            fractions[: counts[0]] @ cells[0],
            fractions[counts[0] :] @ cells[1],
        )
    )
    pbc = torch.ones((2, 3), device="cuda", dtype=torch.bool)
    batch_idx = torch.cat(
        (
            torch.zeros(counts[0], device="cuda", dtype=torch.int32),
            torch.ones(counts[1], device="cuda", dtype=torch.int32),
        )
    )
    batch_ptr = torch.tensor(
        [0, counts[0], sum(counts)], device="cuda", dtype=torch.int32
    )
    targets = torch.tensor([8191, 0, 8703, 512], device="cuda", dtype=torch.int32)
    width = 512
    workspace = _workspace_kwargs(sum(counts), cells, pbc, 6.0)
    common = dict(
        positions=positions,
        cutoff=6.0,
        cell=cells,
        pbc=pbc,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        max_neighbors=width,
        grid_policy="adaptive",
    )
    full = neighbor_list(method="batch_cell_list", **common, **workspace)
    full_reference = _naive(
        positions, cells, pbc, batch_idx, batch_ptr, 6.0, None, width
    )
    _assert_matches_naive(full, full_reference)
    actual = neighbor_list(
        method="batch_cell_list",
        **common,
        target_indices=targets,
        **workspace,
    )
    expected = _naive(positions, cells, pbc, batch_idx, batch_ptr, 6.0, targets, width)
    _assert_matches_naive(actual, expected)


def test_compiled_target_reuses_adaptive_workspace():
    """A warmed fullgraph target route refreshes and reuses supplied storage."""
    positions, cell, pbc, batch_idx, batch_ptr, targets = _small_batch(12.0)
    workspace = _workspace_kwargs(64, cell, pbc, 6.0)
    _call(
        False,
        positions,
        cell,
        pbc,
        batch_idx,
        batch_ptr,
        6.0,
        grid_policy="adaptive",
        **workspace,
    )
    compiled = torch.compile(batch_cell_list, fullgraph=True)
    kwargs = dict(
        positions=positions,
        cutoff=6.0,
        cell=cell,
        pbc=pbc,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        max_neighbors=256,
        target_indices=targets,
        strategy="pair_centric",
        grid_policy="adaptive",
        **workspace,
    )
    compiled(**kwargs)
    actual = compiled(**kwargs)
    expected = _naive(positions, cell, pbc, batch_idx, batch_ptr, 6.0, targets, 256)
    _assert_matches_naive(actual, expected)


@pytest.mark.parametrize("empty_targets,half_fill", [(False, True), (True, False)])
def test_target_radius_nonperiodic_single_cell_axis(empty_targets, half_fill):
    """Target reuse covers half lists and empty rows with a nonperiodic thin axis."""
    positions, cell, pbc, batch_idx, batch_ptr, targets = _small_batch()
    positions[:, 1] *= 1.0 / 12.0
    cell[:, 1, 1] = 1.0
    pbc[:, 1] = False
    workspace = _workspace_kwargs(64, cell, pbc, 6.0)
    _call(
        False,
        positions,
        cell,
        pbc,
        batch_idx,
        batch_ptr,
        6.0,
        grid_policy="adaptive",
        **workspace,
    )
    targets = torch.arange(63, -1, -1, device="cuda", dtype=torch.int32)
    if empty_targets:
        targets = targets[:0]
    actual = _call(
        False,
        positions,
        cell,
        pbc,
        batch_idx,
        batch_ptr,
        6.0,
        target_indices=targets,
        half_fill=half_fill,
        **workspace,
    )
    expected = batch_naive_neighbor_list(
        positions=positions,
        cutoff=6.0,
        cell=cell,
        pbc=pbc,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        target_indices=targets,
        max_neighbors=256,
        max_atoms_per_system=32,
        half_fill=half_fill,
        strategy="scalar",
    )
    if half_fill:
        # Cell and naive half lists orient pairs differently. Compare the
        # physical undirected pairs across all requested rows.
        def canonical_pairs(result):
            matrix, counts, images = (v.cpu() for v in result[:3])
            sources = targets.cpu().tolist()
            pairs = []
            for row, count in enumerate(counts.tolist()):
                for col in range(count):
                    i, j = sources[row], int(matrix[row, col])
                    image = tuple(images[row, col].tolist())
                    pairs.append(min((i, j, *image), (j, i, *(-v for v in image))))
            return sorted(pairs)

        assert canonical_pairs(actual) == canonical_pairs(expected)
    else:
        _assert_matches_naive(actual, expected)
    assert torch.equal(
        workspace["cells_per_dimension"][:, 1],
        torch.ones(2, device="cuda", dtype=torch.int32),
    )
    assert torch.equal(
        workspace["neighbor_search_radius"][:, 1],
        torch.zeros(2, device="cuda", dtype=torch.int32),
    )
