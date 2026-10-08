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

from nvalchemiops.neighbors.cell_list._grid_selection import (
    _get_pair_grid_kernel,
    _validated_cell_total,
)
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
    """Return cumulative atom counts for each system.

    Parameters
    ----------
    cell : torch.Tensor
        Batch cell matrices. The leading dimension gives the number of systems.
    batch_idx : torch.Tensor or None
        System index for each atom. Required when ``batch_ptr`` is omitted.
    batch_ptr : torch.Tensor or None
        Optional cumulative system boundaries. A supplied tensor provides the
        populations directly.

    Returns
    -------
    torch.Tensor
        Contiguous int32 boundaries with shape ``(num_systems + 1,)``.
    """
    systems = cell.shape[0]
    if batch_ptr is None:
        if batch_idx is None:
            raise ValueError("batch_idx is required when batch_ptr is omitted")
        counts = torch.zeros(systems, device=cell.device, dtype=torch.int32)
        counts.scatter_add_(
            0,
            batch_idx.to(torch.int64),
            torch.ones_like(batch_idx, dtype=torch.int32),
        )
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
    batch_idx: torch.Tensor | None,
    batch_ptr: torch.Tensor | None,
    max_nbins: int,
    minimum: int,
    single_system: bool = False,
    num_atoms: int | None = None,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Select cell grids from cell geometry and per-system populations.

    Parameters
    ----------
    cell : torch.Tensor
        Cell matrices with shape ``(num_systems, 3, 3)``.
    pbc : torch.Tensor
        Periodic boundary flags with shape ``(num_systems, 3)``.
    cutoff : float
        Neighbor search cutoff distance.
    batch_idx : torch.Tensor or None
        System index for each atom. Required when ``batch_ptr`` is omitted,
        except when ``single_system`` and ``num_atoms`` provide the population.
    batch_ptr : torch.Tensor or None
        Optional cumulative system boundaries for atom populations.
    max_nbins : int
        Maximum number of cells allowed per system.
    minimum : int
        Configured per-axis minimum used for the compatibility candidate grid.
    single_system : bool, default=False
        Select the single-system kernel specialization.
    num_atoms : int or None, default=None
        Known single-system population, avoiding construction of boundaries.

    Returns
    -------
    tuple[int, torch.Tensor, torch.Tensor]
        Total cell allocation size, search radii, and selected grid dimensions.

    Notes
    -----
    Counts are validated and summed on the device. Automatic allocation reads
    one int64 size; zero reports an invalid cell in a nonempty batch. A single
    system can use ``num_atoms`` directly, avoiding a two-element boundary
    tensor.
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
    allocation_size = torch.empty(1, device=cell.device, dtype=torch.int64)
    wp.launch_tiled(
        _validated_cell_total,
        dim=1,
        block_dim=_PAIR_CENTRIC_BLOCK_DIM,
        device=str(cell.device),
        inputs=[wp.from_torch(counts), wp.from_torch(allocation_size)],
    )
    total = int(allocation_size.item())
    if systems > 0 and total == 0:
        raise RuntimeError(
            "Cells with volume == 0.0 detected and are not supported."
            " Please pass unit cells with `det(cell) != 0.0`."
        )
    return total, radii, grids
