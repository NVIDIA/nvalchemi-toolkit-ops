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


"""Torch-owned storage and stream handling for pair-centric grid sizing."""

import torch
import warp as wp

from nvalchemiops.neighbors.cell_list._grid_selection import _get_pair_grid_kernel
from nvalchemiops.neighbors.cell_list.launchers import _PAIR_CENTRIC_BLOCK_DIM
from nvalchemiops.neighbors.neighbor_utils import empty_sentinel
from nvalchemiops.torch._warp_op_helpers import scoped_torch_warp_stream
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype

__all__ = []


def _pair_grid_boundaries(
    cell: torch.Tensor,
    batch_idx: torch.Tensor | None,
    batch_ptr: torch.Tensor | None,
) -> torch.Tensor:
    """Return contiguous int32 system boundaries for the current population."""
    systems = cell.shape[0]
    if batch_ptr is None:
        if torch.compiler.is_compiling():
            counts = torch.zeros(systems, device=cell.device, dtype=torch.int32)
            counts.scatter_add_(
                0,
                batch_idx.to(torch.int64),
                torch.ones_like(batch_idx, dtype=torch.int32),
            )
        else:
            counts = torch.bincount(batch_idx.to(torch.int64), minlength=systems)
        batch_ptr = torch.zeros(systems + 1, device=cell.device, dtype=torch.int32)
        torch.cumsum(counts, dim=0, out=batch_ptr[1:])
    elif batch_ptr.ndim != 1 or batch_ptr.shape[0] != systems + 1:
        raise ValueError("batch_ptr must have shape (num_systems + 1,)")
    return batch_ptr.to(device=cell.device, dtype=torch.int32).contiguous()


@scoped_torch_warp_stream
def _select_pair_grid(
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch_idx: torch.Tensor,
    batch_ptr: torch.Tensor | None,
    max_nbins: int,
    minimum: int,
    single_system: bool = False,
    num_atoms: int | None = None,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Size current batched cells using geometry and per-system populations.

    Supplied boundaries provide populations directly. Direct binding callers
    with only batch indices use their histogram, including empty systems.
    Returns total allocation size, search radii and the selected grid tensor.
    Torch owns every buffer and the current stream through the sizing launch.
    Single-system callers can pass the known ``num_atoms`` directly, avoiding
    allocation and transfer of a two-element boundary tensor.
    """
    if max_nbins <= 0:
        raise ValueError("max_nbins must be positive")
    systems = cell.shape[0]
    scalar_population = single_system and num_atoms is not None
    if not scalar_population:
        batch_ptr = _pair_grid_boundaries(cell, batch_idx, batch_ptr)
    dtype = get_wp_dtype(cell.dtype)
    grids = torch.empty((systems, 3), device=cell.device, dtype=torch.int32)
    radii = torch.empty_like(grids)
    counts = torch.empty((systems,), device=cell.device, dtype=torch.int32)
    wp.launch(
        _get_pair_grid_kernel(
            dtype, _PAIR_CENTRIC_BLOCK_DIM, single_system, scalar_population
        ),
        dim=systems,
        device=str(cell.device),
        inputs=[
            wp.from_torch(cell, dtype=get_wp_mat_dtype(cell.dtype)),
            wp.from_torch(pbc, dtype=wp.bool),
            empty_sentinel(1, wp.int32, str(cell.device))
            if scalar_population
            else wp.from_torch(batch_ptr, dtype=wp.int32),
            num_atoms if scalar_population else 0,
            dtype(cutoff),
            max_nbins,
            minimum,
            wp.from_torch(grids, dtype=wp.vec3i),
            wp.from_torch(radii, dtype=wp.vec3i),
            wp.from_torch(counts, dtype=wp.int32),
        ],
    )
    cell_counts = counts.cpu().tolist()
    if any(count < 1 for count in cell_counts):
        raise RuntimeError(
            "Cells with volume == 0.0 detected and are not supported."
            " Please pass unit cells with `det(cell) != 0.0`."
        )
    total = sum(cell_counts)
    return total, radii, grids
