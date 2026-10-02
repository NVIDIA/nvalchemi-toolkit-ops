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


"""Buffer preparation for a chosen JAX neighbor-list strategy."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from nvalchemiops.jax.neighbors._dispatch import (
    _synthesize_cell_for_geometry,
    suggest_neighbor_list_method,
)
from nvalchemiops.jax.neighbors.batch_cell_list import (
    estimate_batch_cell_list_sizes,
)
from nvalchemiops.jax.neighbors.cell_list import estimate_cell_list_sizes
from nvalchemiops.jax.neighbors.neighbor_utils import (
    allocate_cell_list,
    compute_naive_num_shifts,
)
from nvalchemiops.neighbors.prepare import (
    _CELL_LIST_STATE_KEYS,
    _MIN_EXTENT,
    _density_from_volume,
    _neighbor_capacity,
    _output_buffer_specs,
    _split_batched_method,
    _strategy_scratch_kind,
)

__all__ = ["prepare_neighbor_list_method"]


def _atomic_density(positions: jax.Array, cell: jax.Array | None) -> float:
    """Mean number density, from the cell when there is one, else the extent."""
    total_atoms = int(positions.shape[0])
    volume = 0.0
    if cell is not None:
        volume = float(jnp.abs(jnp.linalg.det(cell.reshape(-1, 3, 3)[0])))
    if total_atoms and volume <= 0.0:
        extent = jnp.max(positions, axis=0) - jnp.min(positions, axis=0)
        volume = float(jnp.prod(jnp.maximum(extent, _MIN_EXTENT)))
    return _density_from_volume(total_atoms, volume)


def _allocate_outputs(
    kwargs: dict[str, Any],
    num_rows: int,
    max_neighbors: int,
    periodic: bool,
    fill_value: int,
    suffix: str = "",
) -> None:
    """Fill ``kwargs`` with freshly sized matrix outputs.

    Shapes and names come from
    :func:`nvalchemiops.neighbors.prepare._output_buffer_specs`; this adds only
    the JAX allocation.
    """
    kwargs[f"max_neighbors{suffix}"] = max_neighbors
    for spec in _output_buffer_specs(num_rows, max_neighbors, periodic, suffix):
        fill = fill_value if spec.fill is not None else 0
        kwargs[spec.key] = jnp.full(spec.shape, fill, dtype=jnp.int32)
    if not periodic:
        kwargs.pop(f"neighbor_matrix_shifts{suffix}", None)


def prepare_neighbor_list_method(
    positions: jax.Array,
    cutoff: float,
    *,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    method: str | None = None,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    cutoff2: float | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    max_neighbors: int | None = None,
    max_neighbors2: int | None = None,
    target_indices: jax.Array | None = None,
    return_neighbor_list: bool = False,
) -> dict[str, Any]:
    """Allocate every buffer a neighbor-list strategy needs, ready to splat.

    Companion to :func:`suggest_neighbor_list_method`: that picks *which*
    strategy to run, this allocates *what* the strategy needs so the hot loop
    does the selection, capacity estimation and grid sizing once rather than
    per call.

    >>> kwargs = prepare_neighbor_list_method(positions, cutoff, cell=cell, pbc=pbc)
    >>> for _ in range(n_steps):
    ...     matrix, counts, shifts = neighbor_list(positions, cutoff, **kwargs)

    This is the setup half of the workflow :func:`neighbor_list` describes: it
    runs eagerly and inspects host values, then hands back fixed capacities and
    buffers for a compiled, method-specific call.

    Buffers behave differently here than in the Torch binding. JAX is
    functional: the output arrays are donated so XLA can alias their storage,
    but they are **not** written in place, so results must be read from the
    return value rather than from the mapping. Re-splatting the same mapping is
    therefore correct and cheap, but it does not accumulate state.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic positions. Used for sizing and, when ``cell`` is omitted, for
        the synthesized bounding box.
    cutoff : float
        Neighbour cutoff.
    cell : jax.Array, optional
        Simulation cell; omit together with ``pbc`` for a free boundary.
    pbc : jax.Array, optional
        Periodic flags; omit together with ``cell``.
    method : str, optional
        Strategy to prepare for. Chosen by
        :func:`suggest_neighbor_list_method` when omitted.
    batch_idx : jax.Array, optional
        System index per atom.
    batch_ptr : jax.Array, optional
        Cumulative atom counts per system.
    cutoff2 : float, optional
        Second cutoff, for the dual-cutoff strategies.
    half_fill : bool, default=False
        Store each pair once.
    fill_value : int, optional
        Padding value for unused matrix entries. Defaults to the atom count.
    max_neighbors : int, optional
        Override the estimated per-row capacity for ``cutoff``.
    max_neighbors2 : int, optional
        Override the estimated per-row capacity for ``cutoff2``.
    target_indices : jax.Array, optional
        Restrict source rows to this subset; output rows become
        ``len(target_indices)``.
    return_neighbor_list : bool, default=False
        Prepare for COO ``(idx_i, idx_j)`` output rather than matrix output.
        Affects which strategies are feasible, so it is forwarded to the
        selector as well as to :func:`neighbor_list`.

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
    busiest row, so the caller owns the check: compare ``num_neighbors``
    against ``max_neighbors`` and re-prepare when it grows. This matters across
    a trajectory, where a capacity that fits the first frame need not fit later
    ones.

    When ``pbc`` is present but every entry is False, ``cell`` and ``pbc`` are
    omitted from the result, because the periodic kernels are selected by
    ``pbc is not None`` rather than by value.

    See Also
    --------
    suggest_neighbor_list_method : Choose the strategy this function prepares for.
    neighbor_list : Consumer of the returned mapping.
    nvalchemiops.torch.neighbors.prepare_neighbor_list_method : Torch equivalent.
    """
    if (cell is None) != (pbc is None):
        raise ValueError("cell and pbc must be provided together, or neither")

    total_atoms = int(positions.shape[0])
    num_rows = total_atoms if target_indices is None else int(target_indices.shape[0])

    # Value check: an all-False pbc must not select the periodic kernels.
    periodic = pbc is not None and bool(jnp.any(pbc))

    if method is None:
        selector_batch_ptr = batch_ptr
        if selector_batch_ptr is None:
            selector_batch_ptr = jnp.array([0, total_atoms], dtype=jnp.int32)
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

    # ``batch_cell_list`` builds its own grid, so the cell-list scratch does
    # not apply to it.
    cluster_tile = base.startswith("cluster_tile")
    takes_grid_state = scratch == "cell_list" and not batched

    if scratch == "cell_list":
        # Size the grid against the same bounding box the dispatcher
        # synthesizes. It derives from ``positions``, so a free-boundary system
        # whose extent grows substantially needs a fresh call.
        grid_cell, grid_pbc = cell, pbc
        if grid_cell is None:
            ptr = batch_ptr
            if ptr is None:
                ptr = jnp.array([0, total_atoms], dtype=jnp.int32)
            grid_cell, grid_pbc = _synthesize_cell_for_geometry(
                positions, batch_idx, ptr.astype(jnp.int32), cutoff
            )
        if takes_grid_state:
            # The JAX estimator also returns the realized cells per dimension.
            max_total_cells, _, search_radius = estimate_cell_list_sizes(
                positions, grid_cell.reshape(1, 3, 3), cutoff, grid_pbc.reshape(1, 3)
            )
            kwargs.update(
                zip(
                    _CELL_LIST_STATE_KEYS,
                    allocate_cell_list(total_atoms, max_total_cells, search_radius),
                )
            )
            # JAX names this ``sorted_atom_periodic_shifts``, Torch
            # ``sorted_shifts``.
            kwargs["sorted_positions"] = jnp.empty(
                (total_atoms, 3), dtype=positions.dtype
            )
            kwargs["sorted_atom_periodic_shifts"] = jnp.empty(
                (total_atoms, 3), dtype=jnp.int32
            )
        else:
            # Pin the grid capacity to this geometry rather than letting it
            # be re-derived per call.
            max_total_cells, _, _ = estimate_batch_cell_list_sizes(
                positions,
                batch_ptr=batch_ptr,
                batch_idx=batch_idx,
                cell=grid_cell.reshape(-1, 3, 3),
                cutoff=cutoff,
                pbc=grid_pbc.reshape(-1, 3),
            )
            kwargs["max_total_cells"] = int(max_total_cells)
    elif scratch == "naive_periodic":
        # Deriving the shift range internally costs one sync per call.
        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cell.reshape(-1, 3, 3), cutoff, pbc.reshape(-1, 3)
        )
        kwargs["shift_range_per_dimension"] = shift_range
        kwargs["num_shifts_per_system"] = num_shifts
        kwargs["max_shifts_per_system"] = max_shifts
        # Without these the launcher allocates per call.
        kwargs["positions_wrapped_buffer"] = jnp.empty_like(positions)
        kwargs["per_atom_cell_offsets_buffer"] = jnp.empty(
            (total_atoms, 3), dtype=jnp.int32
        )
        kwargs["inv_cell_buffer"] = jnp.empty_like(cell.reshape(-1, 3, 3))

    # No half_fill adjustment: a half list's rows are asymmetric, so the
    # full-list estimate remains the safe upper bound.
    density = _atomic_density(positions, cell)
    resolved_fill = total_atoms if fill_value is None else fill_value

    if cluster_tile:
        # cluster_tile numbers its second matrix ``...2`` with no ``...1``.
        capacity = _neighbor_capacity(cutoff, density, max_neighbors)
        _allocate_outputs(kwargs, num_rows, capacity, periodic, resolved_fill)
        if cutoff2 is not None:
            capacity2 = _neighbor_capacity(cutoff2, density, max_neighbors2)
            kwargs["max_neighbors2"] = capacity2
            _allocate_outputs(
                kwargs, num_rows, capacity2, periodic, resolved_fill, suffix="2"
            )
        return kwargs

    if cutoff2 is None:
        _allocate_outputs(
            kwargs,
            num_rows,
            _neighbor_capacity(cutoff, density, max_neighbors),
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
                _neighbor_capacity(radius, density, requested),
                periodic,
                resolved_fill,
                suffix=suffix,
            )

    return kwargs
