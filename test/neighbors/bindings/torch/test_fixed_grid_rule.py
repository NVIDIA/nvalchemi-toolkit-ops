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


"""Changing-box correctness and host-planning regression checks."""

import pytest
import torch

from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.batch_naive import batch_naive_neighbor_list


def _pair_signature(output, half_fill):
    """Compare physical pairs with a consistent half-list orientation."""
    edges, _, shifts = output[:3]
    rows = torch.cat((edges.T, shifts), dim=1).cpu().tolist()
    if half_fill:
        rows = [
            row if row[0] <= row[1] else [row[1], row[0], *(-s for s in row[2:])]
            for row in rows
        ]
    return sorted(rows)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("strategy", ["atom_centric", "pair_centric"])
@pytest.mark.parametrize("half_fill", [False, True])
@pytest.mark.parametrize("grid_policy", ["configured", "adaptive"])
def test_fixed_grid_uses_current_box(dtype, strategy, half_fill, grid_policy):
    """Update mixed-handed boxes across bin boundaries and match exhaustive pairs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for pair-centric cell-list kernels")
    generator = torch.Generator(device="cuda").manual_seed(9)
    fractional = torch.rand((2, 16, 3), generator=generator, device="cuda", dtype=dtype)
    cells = torch.empty((2, 3, 3), dtype=dtype, device="cuda")
    positions = torch.empty((32, 3), dtype=dtype, device="cuda")
    batch_idx = torch.arange(2, device="cuda", dtype=torch.int32).repeat_interleave(16)
    batch_ptr = torch.tensor([0, 16, 32], device="cuda", dtype=torch.int32)
    pbc = torch.tensor([[True, True, True], [True, False, True]], device="cuda")
    for side in (11.9, 12.1, 17.9, 18.1, 23.9, 24.1, 11.9):
        current = torch.eye(3, device="cuda", dtype=dtype).repeat(2, 1, 1) * side
        current[1, 0, 1] = side / 8
        current[1, 0].neg_()
        cells.copy_(current)
        positions.copy_((fractional @ cells).reshape(-1, 3))
        common = dict(
            positions=positions,
            cutoff=6.0,
            cell=cells,
            pbc=pbc,
            batch_idx=batch_idx,
            max_neighbors=256,
            half_fill=half_fill,
            return_neighbor_list=True,
        )
        reference = batch_naive_neighbor_list(
            **common, max_atoms_per_system=16, strategy="scalar"
        )
        actual = neighbor_list(
            **common,
            batch_ptr=batch_ptr,
            method="batch_cell_list_" + strategy,
            grid_policy=grid_policy,
        )
        assert _pair_signature(actual, half_fill) == _pair_signature(
            reference, half_fill
        )
    common["return_neighbor_list"] = False
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        neighbor_list(
            **common,
            batch_ptr=batch_ptr,
            method="batch_cell_list_" + strategy,
            grid_policy=grid_policy,
        )
    operations = {event.key for event in profile.key_averages()}
    assert "aten::bincount" not in operations
    assert sum(
        event.count
        for event in profile.key_averages()
        if event.key == "aten::linalg_cross"
    ) == (
        0
        if grid_policy == "adaptive" and strategy == "pair_centric" and not half_fill
        else 1
    )
