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

"""Shared fixtures for Torch prepared-state public-contract tests."""

import torch
import warp as wp

from nvalchemiops.torch.neighbors import NeighborListState, prepare_neighbor_list


@wp.func
def prepared_sum_pair_fn(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    """Return an easily checked energy and force for prepared matrix output."""
    return pair_params[i, 0] + pair_params[j, 0] + distance, -r_ij


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return three geometries with 6, 2, and 0 directed pairs."""
    close = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    middle = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    far = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    cell = torch.eye(3, dtype=torch.float32, device="cuda") * 20.0
    return close, middle, far, cell


def prepare(
    positions: torch.Tensor,
    cell: torch.Tensor,
    *,
    return_vectors: bool = True,
    return_distances: bool = True,
) -> NeighborListState:
    """Create the generic prepared cluster-tile state used by both test files."""
    return prepare_neighbor_list(
        positions,
        1.0,
        cell,
        torch.ones(3, dtype=torch.bool, device=positions.device),
        method="cluster_tile",
        return_neighbor_list=True,
        max_neighbors=8,
        max_pairs=32,
        max_tiles_per_group=1,
        return_vectors=return_vectors,
        return_distances=return_distances,
    )
