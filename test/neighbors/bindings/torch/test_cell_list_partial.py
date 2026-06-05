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

"""Regression tests for the cell_list ``target_indices`` (partial) path.

The compact partial output has ``num_targets`` rows (not ``total_atoms``).
These tests check the partial path returns the same neighbors as the full
matrix for the target rows, and that an undersized output buffer raises a
clean error (the pre-launch shape guard) instead of an out-of-bounds write.
"""

import pytest
import torch

from nvalchemiops.torch.neighbors.cell_list import cell_list

from ...test_utils import brute_force_neighbors, create_simple_cubic_system


def _partial_pair_sets(nm, nn, targets):
    """Set of (target_atom, neighbor) pairs from a compact partial result."""
    nm = nm.cpu()
    nn = nn.cpu()
    targets = targets.cpu()
    width = nm.shape[1]
    pairs = set()
    for r in range(nm.shape[0]):
        ti = int(targets[r])
        for k in range(min(int(nn[r]), width)):
            pairs.add((ti, int(nm[r, k])))
    return pairs


def test_cell_list_target_indices_matches_brute_force(device, dtype):
    """Compact partial rows must match an independent brute-force reference."""
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=dtype, device=device
    )
    pbc = pbc.reshape(3)
    cutoff = 1.1
    n = positions.shape[0]
    max_neighbors = 24  # well above the ~6 nearest neighbors of this lattice

    targets = torch.arange(0, n, 2, dtype=torch.int32, device=device)
    nt = int(targets.shape[0])
    nm = torch.full((nt, max_neighbors), n, dtype=torch.int32, device=device)
    nms = torch.zeros((nt, max_neighbors, 3), dtype=torch.int32, device=device)
    nn = torch.zeros((nt,), dtype=torch.int32, device=device)
    nm_p, nn_p, _ = cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        max_neighbors=max_neighbors,
        fill_value=n,
        neighbor_matrix=nm,
        neighbor_matrix_shifts=nms,
        num_neighbors=nn,
        target_indices=targets,
    )
    if "cuda" in str(device):
        torch.cuda.synchronize()

    # Independent O(N^2) reference, restricted to the target rows.
    i_ref, j_ref, _u, _s = brute_force_neighbors(positions, cell, pbc, cutoff)
    target_set = {int(t) for t in targets.cpu()}
    ref_pairs = {
        (int(i), int(j))
        for i, j in zip(i_ref.cpu(), j_ref.cpu())
        if int(i) in target_set
    }

    assert _partial_pair_sets(nm_p, nn_p, targets) == ref_pairs


def test_cell_list_target_indices_undersized_buffer_raises(device, dtype):
    """An output matrix smaller than ``num_targets`` rows must fail cleanly."""
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=dtype, device=device
    )
    pbc = pbc.reshape(3)
    n = positions.shape[0]
    w = 8
    targets = torch.arange(0, n, 2, dtype=torch.int32, device=device)
    nt = int(targets.shape[0])
    nm = torch.full((nt - 1, w), n, dtype=torch.int32, device=device)  # too small
    nms = torch.zeros((nt - 1, w, 3), dtype=torch.int32, device=device)
    nn = torch.zeros((nt - 1,), dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="rows"):
        cell_list(
            positions,
            cutoff=1.1,
            cell=cell,
            pbc=pbc,
            max_neighbors=w,
            fill_value=n,
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            num_neighbors=nn,
            target_indices=targets,
        )
