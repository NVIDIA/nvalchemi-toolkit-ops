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


"""Private PyTorch neighbor-list dispatch helpers."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import warp as wp

from nvalchemiops.neighbors.base_dispatch import (
    FEATURE_BATCHED,
    FEATURE_CUDA,
    FEATURE_POSITIONS_FLOAT32,
    neighbor_list_strategy_run_args,
    optional_outputs_mask,
)
from nvalchemiops.neighbors.base_dispatch import (
    estimate_neighbor_list_costs as _estimate_neighbor_list_costs_wp,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    _raise_if_compiling_host_only,
    synthesize_cell_for_batch,
    synthesize_cell_for_ss,
)
from nvalchemiops.torch.types import get_wp_mat_dtype

__all__ = [
    "_auto_base_method_from_geometry",
    "_auto_method_from_geometry",
    "_reject_unsupported_cluster_tile_combo",
    "_squeeze_single_system_cell_pbc",
    "broadcast_shared_cell_for_batch",
    "estimate_neighbor_list_costs",
    "suggest_neighbor_list_method",
]


def _as_batched_cell(cell: torch.Tensor) -> torch.Tensor:
    """Return ``cell`` with an explicit leading system dimension."""
    return cell if cell.ndim == 3 else cell.unsqueeze(0)


def _squeeze_single_system_cell_pbc(
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Convert single-system batched ``cell`` / ``pbc`` tensors to unbatched shape."""
    if cell is not None and cell.ndim == 3 and cell.shape[0] == 1:
        cell = cell[0]
    if pbc is not None and pbc.ndim == 2 and pbc.shape[0] == 1:
        pbc = pbc[0]
    return cell, pbc


def broadcast_shared_cell_for_batch(
    cell: torch.Tensor,
    num_systems: int,
) -> torch.Tensor:
    """Return one cell per system for batched cluster-tile calls."""
    if cell.ndim == 3:
        return cell
    if cell.ndim != 2:
        raise ValueError(
            "cell for method='batch_cluster_tile' must have shape (3, 3) "
            "or (num_systems, 3, 3)"
        )
    return cell.unsqueeze(0).expand(int(num_systems), -1, -1).contiguous()


def _selector_batch_ptr_from_geometry(
    positions: torch.Tensor,
    batch_idx: torch.Tensor | None,
    batch_ptr: torch.Tensor | None,
    num_systems: int,
) -> torch.Tensor:
    """Return an int32 batch pointer for selector metadata."""
    if batch_ptr is not None:
        return batch_ptr.to(dtype=torch.int32).detach().contiguous()
    if batch_idx is not None:
        counts = torch.bincount(batch_idx, minlength=int(num_systems)).to(torch.int32)
        return torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=positions.device),
                counts.cumsum(dim=0),
            ]
        )

    base_count = positions.shape[0] // max(int(num_systems), 1)
    counts = torch.full(
        (int(num_systems),),
        base_count,
        dtype=torch.int32,
        device=positions.device,
    )
    if num_systems > 0:
        counts[0] += positions.shape[0] - base_count * int(num_systems)
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=positions.device),
            counts.cumsum(dim=0),
        ]
    )


def _normalize_selector_cell_pbc(
    cell: torch.Tensor,
    pbc: torch.Tensor,
    num_systems: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous selector metadata with one cell per system."""
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
    if cell.ndim != 3 or cell.shape[-2:] != (3, 3):
        raise ValueError("cell must have shape (3, 3) or (num_systems, 3, 3)")
    if cell.shape[0] == 1 and num_systems > 1:
        cell = cell.expand(num_systems, -1, -1)
    if cell.shape[0] != num_systems:
        raise ValueError("cell must contain one matrix per system")

    if pbc.ndim not in (1, 2):
        raise ValueError("pbc must have shape (3,) or (num_systems, 3)")
    if pbc.ndim == 2 and pbc.shape[0] == 1 and num_systems > 1:
        pbc = pbc.expand(num_systems, -1)
    return cell.detach().contiguous(), pbc.detach().to(dtype=torch.bool).contiguous()


def estimate_neighbor_list_costs(
    batch_ptr: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    *,
    batch_idx: torch.Tensor | None = None,
    max_nbins: int | None = None,
    optional_outputs: Iterable[str] | None = None,
    cutoff2: float | None = None,
    half_fill: bool = False,
    return_neighbor_list: bool = False,
    target_indices: torch.Tensor | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    use_pair_fn: bool = False,
    rebuild_flags: torch.Tensor | None = None,
    wrap_positions: bool = True,
    positions_dtype: torch.dtype | None = None,
) -> list[tuple[str, float]]:
    """Report feasible Torch neighbor-list strategies and their estimated cost.

    Parameters
    ----------
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=torch.int32
        Cumulative atom counts. ``batch_ptr[-1]`` is the total atom count.
    cell : torch.Tensor, shape (3, 3) or (num_systems, 3, 3)
        Per-system cells, or one shared cell to broadcast.
    pbc : torch.Tensor, shape (3,) or (num_systems, 3), dtype=bool
        Shared or per-system PBC flags.
    cutoff : float
        Neighbor cutoff.
    max_nbins : int, optional
        Per-system cell-list cell cap. Defaults to the same cap used by the
        active single-system or batched frontend.
    optional_outputs : iterable of str, optional
        Public neighbor-list option names to include in feasibility checks.
    target_indices : torch.Tensor, optional
        Public partial-row source indices.  Its length is used to score
        targeted naive/cell-list work.
    positions_dtype : torch.dtype, optional
        Position dtype used for feature feasibility.  Standalone calls default
        to ``cell.dtype``.

    Returns
    -------
    list of (str, float)
        Feasible strategies and their relative estimated cost (lower is
        faster), sorted cheapest-first.  Host-only: call outside
        ``torch.compile`` and pass the chosen name as an explicit ``method=``.
    """
    _raise_if_compiling_host_only(
        "estimate_neighbor_list_costs",
        "Call it before compiling, then pass the selected strategy name as "
        "an explicit method= argument to the compiled neighbor-list call.",
    )
    if batch_ptr.ndim != 1:
        raise ValueError("batch_ptr must be a 1-D tensor")
    num_systems = max(int(batch_ptr.shape[0]) - 1, 0)
    cell, pbc = _normalize_selector_cell_pbc(cell, pbc, num_systems)
    batch_ptr = batch_ptr.detach().to(dtype=torch.int32).contiguous()
    if batch_idx is not None:
        batch_idx = batch_idx.detach().to(dtype=torch.int32).contiguous()

    wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32, requires_grad=False)
    wp_batch_idx = (
        wp.from_torch(batch_idx, dtype=wp.int32, requires_grad=False)
        if batch_idx is not None
        else None
    )
    wp_cell = wp.from_torch(
        cell, dtype=get_wp_mat_dtype(cell.dtype), requires_grad=False
    )
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, requires_grad=False)
    feature_mask = 0
    if cell.device.type == "cuda":
        feature_mask |= FEATURE_CUDA
    position_dtype = cell.dtype if positions_dtype is None else positions_dtype
    if position_dtype == torch.float32:
        feature_mask |= FEATURE_POSITIONS_FLOAT32
    if num_systems > 1:
        feature_mask |= FEATURE_BATCHED
    options = optional_outputs_mask(
        optional_outputs,
        cutoff2=cutoff2,
        half_fill=half_fill,
        return_neighbor_list=return_neighbor_list,
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        use_pair_fn=use_pair_fn,
        rebuild_flags=rebuild_flags,
        wrap_positions=wrap_positions,
    )
    return _estimate_neighbor_list_costs_wp(
        wp_batch_ptr,
        wp_cell,
        wp_pbc,
        cutoff,
        batch_idx=wp_batch_idx,
        max_nbins=max_nbins,
        option_mask=options,
        feature_mask=feature_mask,
        target_count=(
            int(target_indices.shape[0]) if target_indices is not None else None
        ),
    )


def suggest_neighbor_list_method(*args, **kwargs) -> str:
    """Return the cheapest feasible Torch neighbor-list strategy name.

    Thin wrapper over :func:`estimate_neighbor_list_costs` returning only the
    top-ranked strategy.  Same arguments and same host-only sync caveat.
    """
    return estimate_neighbor_list_costs(*args, **kwargs)[0][0]


def _auto_method_from_geometry(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
    batch_idx: torch.Tensor | None,
    batch_ptr: torch.Tensor | None,
    num_systems: int,
    *,
    optional_outputs: Iterable[str] | None = None,
    cutoff2: float | None = None,
    half_fill: bool = False,
    return_neighbor_list: bool = False,
    target_indices: torch.Tensor | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    use_pair_fn: bool = False,
    rebuild_flags: torch.Tensor | None = None,
    wrap_positions: bool = True,
) -> str:
    """Return the cheapest base strategy name for the geometry (no ``batch_``).

    The ``batch_`` prefix is added by the caller from ``num_systems``; this
    returns the unbatched fine-grained name, e.g. ``"naive_tile"``.
    """
    _raise_if_compiling_host_only(
        "neighbor_list(method=None)",
        "Choose the method before compiling with suggest_neighbor_list_method "
        "or pass method='naive', method='cell_list', or another explicit "
        "strategy name.",
    )
    if positions.shape[0] == 0:
        return "cell_list_atom_centric"

    selector_batch_ptr = _selector_batch_ptr_from_geometry(
        positions, batch_idx, batch_ptr, num_systems
    )
    if cell is None:
        if batch_idx is not None and num_systems > 1:
            _, selector_cell, selector_pbc = synthesize_cell_for_batch(
                positions, batch_idx, selector_batch_ptr, cutoff
            )
        else:
            _, selector_cell, selector_pbc = synthesize_cell_for_ss(positions, cutoff)
            if num_systems > 1:
                selector_cell = selector_cell.expand(num_systems, -1, -1)
                selector_pbc = selector_pbc.expand(num_systems, -1)
    else:
        if pbc is None:
            raise ValueError("pbc is required when cell is provided")
        selector_cell = cell
        selector_pbc = pbc

    name = suggest_neighbor_list_method(
        selector_batch_ptr,
        selector_cell,
        selector_pbc,
        cutoff,
        batch_idx=batch_idx,
        optional_outputs=optional_outputs,
        cutoff2=cutoff2,
        half_fill=half_fill,
        return_neighbor_list=return_neighbor_list,
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        use_pair_fn=use_pair_fn,
        rebuild_flags=rebuild_flags,
        wrap_positions=wrap_positions,
        positions_dtype=positions.dtype,
    )
    return name[len("batch_") :] if name.startswith("batch_") else name


def _auto_base_method_from_geometry(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
    batch_idx: torch.Tensor | None,
    batch_ptr: torch.Tensor | None,
    num_systems: int,
) -> str:
    """Pick the base method (``naive`` / ``cell_list`` / ``cluster_tile``)."""
    name = _auto_method_from_geometry(
        positions, cutoff, cell, pbc, batch_idx, batch_ptr, num_systems
    )
    return neighbor_list_strategy_run_args(name)[0]


def _reject_unsupported_cluster_tile_combo(
    pbc: torch.Tensor | None,
    half_fill: bool,
) -> None:
    """Raise ``NotImplementedError`` for unsupported explicit cluster-tile combos."""
    if pbc is None:
        raise NotImplementedError(
            "method='cluster_tile' / 'batch_cluster_tile' is "
            "fundamentally PBC-implicit; non-periodic systems (pbc=None) "
            "are not supported.  Use method='naive' or 'cell_list', "
            "or pass a cell with fully periodic pbc."
        )
    try:
        all_periodic = bool(pbc.all().item())
    except RuntimeError:
        all_periodic = True
    if not all_periodic:
        raise NotImplementedError(
            "method='cluster_tile' / 'batch_cluster_tile' is "
            "fundamentally PBC-implicit; pbc with any False entry "
            "is not supported.  Use method='naive' or 'cell_list'."
        )
    if half_fill:
        raise NotImplementedError(
            "method='cluster_tile' / 'batch_cluster_tile' uses "
            "tile-level upper-triangular iteration; atom-level "
            "half_fill=True is not supported.  Use method='naive' or "
            "'cell_list' for half-fill output."
        )
