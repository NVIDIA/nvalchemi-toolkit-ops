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


"""Buffer preparation for a chosen PyTorch neighbor-list strategy."""

from __future__ import annotations

from typing import Any

import torch

from nvalchemiops.neighbors.prepare import (
    _CELL_LIST_STATE_KEYS,
    _MIN_EXTENT,
    _density_from_volume,
    _neighbor_capacity,
    _output_buffer_specs,
    _split_batched_method,
    _strategy_scratch_kind,
)
from nvalchemiops.torch.neighbors._dispatch import (
    suggest_neighbor_list_method,
)
from nvalchemiops.torch.neighbors.batch_cell_list import (
    estimate_batch_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.cell_list import (
    allocate_query_sort_scratch,
    estimate_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.cluster_tile import _cell_volume
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
    compute_naive_num_shifts,
    synthesize_cell_for_ss,
)

__all__ = ["prepare_neighbor_list_method"]


def _atomic_density(positions: torch.Tensor, cell: torch.Tensor | None) -> float:
    """Mean number density, from the cell when there is one, else the extent."""
    total_atoms = int(positions.shape[0])
    volume = _cell_volume(cell) if cell is not None else 0.0
    if total_atoms and volume <= 0.0:
        extent = positions.max(dim=0).values - positions.min(dim=0).values
        volume = float(extent.clamp(min=_MIN_EXTENT).prod())
    return _density_from_volume(total_atoms, volume)


def _allocate_outputs(
    kwargs: dict[str, Any],
    num_rows: int,
    max_neighbors: int,
    device: torch.device,
    periodic: bool,
    fill_value: int,
    suffix: str = "",
) -> None:
    """Fill ``kwargs`` with freshly sized matrix outputs.

    Shapes and names come from
    :func:`nvalchemiops.neighbors.prepare._output_buffer_specs`; this adds only
    the Torch allocation.
    """
    kwargs[f"max_neighbors{suffix}"] = max_neighbors
    for spec in _output_buffer_specs(num_rows, max_neighbors, periodic, suffix):
        fill = fill_value if spec.fill is not None else 0
        kwargs[spec.key] = torch.full(
            spec.shape, fill, dtype=torch.int32, device=device
        )
    if not periodic:
        kwargs.pop(f"neighbor_matrix_shifts{suffix}", None)


def prepare_neighbor_list_method(
    positions: torch.Tensor,
    cutoff: float,
    *,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    method: str | None = None,
    batch_idx: torch.Tensor | None = None,
    batch_ptr: torch.Tensor | None = None,
    cutoff2: float | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    max_neighbors: int | None = None,
    max_neighbors2: int | None = None,
    target_indices: torch.Tensor | None = None,
    return_neighbor_list: bool = False,
) -> dict[str, Any]:
    """Allocate every buffer a neighbor-list strategy needs, ready to splat.

    Companion to :func:`suggest_neighbor_list_method`: that picks *which*
    strategy to run, this allocates *what* the strategy needs so the hot loop
    neither allocates nor synchronizes.

    >>> kwargs = prepare_neighbor_list_method(positions, cutoff, cell=cell, pbc=pbc)
    >>> for _ in range(n_steps):
    ...     matrix, counts, shifts = neighbor_list(positions, cutoff, **kwargs)

    The returned mapping carries ``method``, ``half_fill``, the matrix outputs,
    and the strategy-specific scratch, so the call site needs only ``positions``
    and ``cutoff``. It also carries ``cell`` / ``pbc`` when they apply, which is
    why they are not passed again at the call site.

    This function reads tensor values and therefore synchronizes. Call it once
    during setup, outside any ``torch.compile`` region, and reuse the result.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates. Used for shape, device, dtype, and for the density
        the default ``max_neighbors`` is derived from. Not read for values
        beyond that, so the buffers do not depend on this particular frame.
    cutoff : float
        Cutoff distance in Cartesian units.
    cell : torch.Tensor, shape (3, 3) or (num_systems, 3, 3), optional
        Cell matrix. Must be given together with ``pbc``.
    pbc : torch.Tensor, shape (3,) or (num_systems, 3), dtype=torch.bool, optional
        Periodicity flags. Must be given together with ``cell``.
    method : str, optional
        Strategy name. When omitted, :func:`suggest_neighbor_list_method`
        chooses one. Fine-grained names such as ``"cell_list_atom_centric"``
        are accepted and forwarded unchanged.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=torch.int32, optional
        Per-atom system index for batched inputs.
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=torch.int32, optional
        Segment boundaries for batched inputs.
    cutoff2 : float, optional
        Second cutoff. Selects the dual-cutoff methods, whose buffers are
        named ``neighbor_matrix1`` / ``neighbor_matrix2`` and so on rather
        than the plain single-cutoff names.
    half_fill : bool, default=False
        Store each pair once.
    fill_value : int, optional
        Padding value for unused neighbor slots. Defaults to ``total_atoms``.
    max_neighbors : int, optional
        Override the estimated per-row capacity for ``cutoff``.
    max_neighbors2 : int, optional
        Override the estimated per-row capacity for ``cutoff2``.
    target_indices : torch.Tensor, optional
        Restrict source rows to this subset; output rows become
        ``len(target_indices)``.
    return_neighbor_list : bool, default=False
        Prepare for COO ``(idx_i, idx_j)`` output rather than matrix output.
        Affects which strategies are feasible, so it is forwarded to the
        selector as well as to :func:`neighbor_list`. The neighbor matrix is
        still allocated, because the COO pairs are compacted from it.

    Returns
    -------
    dict
        Keyword arguments for :func:`neighbor_list`, splattable with ``**``.

    Raises
    ------
    ValueError
        If exactly one of ``cell`` and ``pbc`` is provided.

    Notes
    -----
    Capacity is an estimate, not a guarantee. ``max_neighbors`` is the mean
    neighbour count for the system's density plus a Poisson allowance for the
    busiest row, and the matrix path does not synchronize to check it, so the
    caller owns the check: compare ``num_neighbors`` against ``max_neighbors``
    and re-prepare when it grows.
    This matters across a trajectory, where a capacity that fits the first
    frame need not fit later ones.

    When ``pbc`` is present but every entry is False, ``cell`` and ``pbc`` are
    omitted from the result. The periodic kernels are selected by ``pbc is not
    None`` rather than by value, so passing an all-False ``pbc`` would search
    for periodic images that cannot exist. Deciding it here costs one
    synchronization at setup instead of one per call.

    See Also
    --------
    suggest_neighbor_list_method : Choose the strategy this function prepares for.
    neighbor_list : Consumer of the returned mapping.
    """
    if (cell is None) != (pbc is None):
        raise ValueError("cell and pbc must be provided together, or neither")

    device = positions.device
    total_atoms = int(positions.shape[0])
    num_rows = total_atoms if target_indices is None else int(target_indices.shape[0])

    # Value check: an all-False pbc must not select the periodic kernels.
    periodic = pbc is not None and bool(pbc.any())

    if method is None:
        selector_batch_ptr = batch_ptr
        if selector_batch_ptr is None:
            selector_batch_ptr = torch.tensor(
                [0, total_atoms], dtype=torch.int32, device=device
            )
        # Omit cell/pbc for a free boundary rather than synthesizing a box,
        # so the cost model prices the aperiodic path.
        method = suggest_neighbor_list_method(
            selector_batch_ptr,
            cell if periodic else None,
            pbc if periodic else None,
            cutoff,
            positions=positions,
            batch_idx=batch_idx,
            half_fill=half_fill,
            target_indices=target_indices,
            return_neighbor_list=return_neighbor_list,
            positions_dtype=positions.dtype,
        )

    kwargs: dict[str, Any] = {"method": method, "half_fill": half_fill}
    if return_neighbor_list:
        kwargs["return_neighbor_list"] = True
    if cutoff2 is not None:
        kwargs["cutoff2"] = cutoff2
    if periodic:
        kwargs["cell"] = cell
        kwargs["pbc"] = pbc
    if batch_idx is not None:
        kwargs["batch_idx"] = batch_idx
    if batch_ptr is not None:
        kwargs["batch_ptr"] = batch_ptr
    if target_indices is not None:
        kwargs["target_indices"] = target_indices
    if fill_value is not None:
        kwargs["fill_value"] = fill_value

    base, prefixed = _split_batched_method(method)
    batched = prefixed or batch_idx is not None or batch_ptr is not None
    scratch = _strategy_scratch_kind(method, periodic)

    if scratch == "cell_list":
        # Size the grid against the same bounding box the dispatcher
        # synthesizes. It derives from ``positions``, so a free-boundary system
        # whose extent grows substantially needs a fresh call.
        grid_cell, grid_pbc = cell, pbc
        if grid_cell is None:
            _, grid_cell, grid_pbc = synthesize_cell_for_ss(positions, cutoff)
        if batched:
            # The batched search radius is (num_systems, 3), which the
            # single-system estimator rejects.
            grid_cell = grid_cell.reshape(-1, 3, 3)
            grid_pbc = grid_pbc.reshape(-1, 3)
            if grid_cell.shape[0] == 1 and batch_ptr is not None:
                num_systems = int(batch_ptr.shape[0]) - 1
                grid_cell = grid_cell.expand(num_systems, -1, -1)
                grid_pbc = grid_pbc.expand(num_systems, -1)
            max_total_cells, search_radius = estimate_batch_cell_list_sizes(
                grid_cell, grid_pbc, cutoff
            )
            kwargs["cell_offsets"] = torch.zeros(
                grid_cell.shape[0], dtype=torch.int32, device=device
            )
        else:
            max_total_cells, search_radius = estimate_cell_list_sizes(
                grid_cell, grid_pbc, cutoff
            )
        kwargs.update(
            zip(
                _CELL_LIST_STATE_KEYS,
                allocate_cell_list(total_atoms, max_total_cells, search_radius, device),
            )
        )
        if not batched:
            # Query-sort scratch is single-system only.
            sorted_positions, sorted_shifts = allocate_query_sort_scratch(
                total_atoms, dtype=positions.dtype, device=device
            )
            kwargs["sorted_positions"] = sorted_positions
            kwargs["sorted_shifts"] = sorted_shifts
    elif scratch == "naive_periodic":
        # Deriving the shift range internally costs one sync per call.
        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cell.reshape(-1, 3, 3), cutoff, pbc.reshape(-1, 3)
        )
        kwargs["shift_range_per_dimension"] = shift_range
        kwargs["num_shifts_per_system"] = num_shifts
        kwargs["max_shifts_per_system"] = max_shifts
        # Without these the launcher allocates per call, which also prevents
        # CUDA graph capture.
        kwargs["positions_wrapped_buffer"] = torch.empty_like(positions)
        kwargs["per_atom_cell_offsets_buffer"] = torch.empty(
            (total_atoms, 3), dtype=torch.int32, device=device
        )
        kwargs["inv_cell_buffer"] = torch.empty_like(cell.reshape(-1, 3, 3))

    # No half_fill adjustment: a half list's rows are asymmetric, so the
    # full-list estimate remains the safe upper bound.
    density = _atomic_density(positions, cell)
    resolved_fill = total_atoms if fill_value is None else fill_value

    def _capacity(requested: int | None, radius: float) -> int:
        return _neighbor_capacity(radius, density, requested)

    if cutoff2 is None:
        _allocate_outputs(
            kwargs,
            num_rows,
            _capacity(max_neighbors, cutoff),
            device,
            periodic,
            resolved_fill,
        )
    else:
        # Dual-cutoff methods take two independently sized output sets.
        for suffix, radius, requested in (
            ("1", cutoff, max_neighbors),
            ("2", cutoff2, max_neighbors2),
        ):
            _allocate_outputs(
                kwargs,
                num_rows,
                _capacity(requested, radius),
                device,
                periodic,
                resolved_fill,
                suffix=suffix,
            )

    return kwargs
