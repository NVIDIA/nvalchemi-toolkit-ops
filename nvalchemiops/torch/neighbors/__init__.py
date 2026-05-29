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

"""PyTorch neighbor list API.

This module provides the main entry point for PyTorch users of the neighbor list API.
"""

from __future__ import annotations

import torch

# Batch cell list functions
from nvalchemiops.torch.neighbors.batch_cell_list import (
    batch_cell_list,
    estimate_batch_cell_list_sizes,
)

# Batched cluster-pair tile functions
from nvalchemiops.torch.neighbors.batch_cluster_tile import (
    batch_cluster_tile_neighbor_list,
)

# Batch naive functions
from nvalchemiops.torch.neighbors.batch_naive import (
    batch_naive_neighbor_list,
)

# Batch naive dual cutoff functions
from nvalchemiops.torch.neighbors.batch_naive_dual_cutoff import (
    batch_naive_neighbor_list_dual_cutoff,
)

# Unbatched cell list functions
from nvalchemiops.torch.neighbors.cell_list import (
    cell_list,
    estimate_cell_list_sizes,
)

# Unbatched cluster-pair tile functions
from nvalchemiops.torch.neighbors.cluster_tile import (
    cluster_tile_neighbor_list,
)

# Unbatched naive functions
from nvalchemiops.torch.neighbors.naive import (
    naive_neighbor_list,
)

# Unbatched naive dual cutoff functions
from nvalchemiops.torch.neighbors.naive_dual_cutoff import (
    naive_neighbor_list_dual_cutoff,
)

# Utility functions
from nvalchemiops.torch.neighbors.neighbor_utils import (
    prepare_batch_idx_ptr,
    synthesize_cell_for_batch,
    synthesize_cell_for_ss,
)

_CLUSTER_TILE_AUTO_BLOCKED_KWARGS = {
    "cutoff2",
    "format",
    "max_pairs",
    "neighbor_distances",
    "neighbor_list",
    "neighbor_list_shifts",
    "neighbor_vectors",
    "pair_counter",
    "pair_energies",
    "pair_fn",
    "pair_forces",
    "pair_params",
    "return_distances",
    "return_vectors",
    "target_indices",
}


def _has_active_cluster_tile_blocker(kwargs: dict) -> bool:
    """Return True when kwargs request behavior outside safe auto-dispatch."""
    for name in _CLUSTER_TILE_AUTO_BLOCKED_KWARGS:
        value = kwargs.get(name)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                return True
            continue
        return True
    return False


def _reject_unsupported_cluster_tile_combo(
    pbc: torch.Tensor | None,
    half_fill: bool,
) -> None:
    """Raise NotImplementedError for combos cluster_tile cannot honor.

    Cluster-tile is fundamentally PBC-implicit (it enumerates tile pairs
    inside a periodic shift range) and tile-level upper-triangular
    (atom-level full-fill within tiles, but no per-pair half-fill).
    Honoring non-periodic boundaries or ``half_fill=True`` would require
    post-processing passes that change the algorithm's identity, so the
    dispatcher rejects them explicitly instead of silently dropping.
    """
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
        all_periodic = True  # let downstream handle malformed pbc
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


def _cluster_tile_auto_eligible(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
    cutoff2: float | None,
    half_fill: bool,
    kwargs: dict,
) -> bool:
    """Return True when ``method=None`` may safely select cluster-pair tile.

    Cluster-tile is PBC-implicit and tile-level upper-triangular, so
    auto-dispatch never picks it when ``pbc`` has any False entry or
    ``half_fill=True``.  Explicit ``method='cluster_tile'`` callers get
    a clear ``NotImplementedError`` in those cases (see the dispatcher
    match branch below).
    """
    if cutoff2 is not None:
        return False
    if half_fill:
        return False
    if positions.dtype != torch.float32 or positions.device.type != "cuda":
        return False
    if cell is None or pbc is None or _has_active_cluster_tile_blocker(kwargs):
        return False
    try:
        return bool(pbc.to(device=positions.device).all().item())
    except RuntimeError:
        return False


def neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    batch_ptr: torch.Tensor | None = None,
    cutoff2: float | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    method: str | None = None,
    wrap_positions: bool = True,
    **kwargs: dict,
):
    """Compute neighbor list using the appropriate method based on the provided parameters.

    This is the main entry point for PyTorch users of the neighbor list API. It automatically
    selects the most appropriate algorithm (naive O(N²) or cell list O(N)) based on system
    size and parameters.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Concatenated atomic coordinates for all systems in Cartesian space.
        Each row represents one atom's (x, y, z) position.
        Unwrapped (box-crossing) coordinates are supported when PBC is used;
        the kernel wraps positions internally.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    cell : torch.Tensor, shape (3, 3) or (num_systems, 3, 3), optional
        Cell matrix defining the simulation box.
    pbc : torch.Tensor, shape (3,) or (num_systems, 3), dtype=torch.bool, optional
        Periodic boundary condition flags for each dimension.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=torch.int32, optional
        System index for each atom.  Must be **sorted by system** (i.e.,
        atoms in system 0 first, then system 1, and so on).  Interleaved
        layouts are not supported by ``cluster_tile`` / ``batch_cluster_tile``
        and will silently emit cross-system pairs.  For ``cell_list`` /
        ``naive`` methods, interleaved layouts work but ``batch_ptr``
        will still be derived assuming a contiguous layout.
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=torch.int32, optional
        Cumulative atom counts defining system boundaries.
    cutoff2 : float, optional
        Second cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    half_fill : bool, optional
        If True, only store half of the neighbor relationships to avoid double counting.
        Another half could be reconstructed by swapping source and target indices and inverting unit shifts.
    fill_value : int | None, optional
        Value to fill the neighbor matrix with. Default is total_atoms.
    return_neighbor_list : bool, optional - default = False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format by
        creating a mask over the fill_value, which can incur a performance penalty.
        We recommend using the neighbor matrix format,
        and only convert to a neighbor list format if absolutely necessary.
    method : str | None, optional
        Method to use for neighbor list computation.
        Choices: "naive", "cell_list", "cluster_tile", "batch_naive",
        "batch_cell_list", "batch_cluster_tile", "naive_dual_cutoff",
        "batch_naive_dual_cutoff".  For large fully periodic CUDA float32
        single-cutoff systems, ``method=None`` may select ``cluster_tile`` /
        ``batch_cluster_tile`` when no partial or pair-output kwargs are active.
        If None, a default method will be chosen based on average atoms per
        system (cluster_tile/cell_list when >= 2000, naive otherwise). When only
        ``batch_idx`` is provided (no ``batch_ptr`` or 3-D ``cell``),
        auto-selection reads ``batch_idx[-1]`` which triggers a
        device-to-host synchronization. To avoid this, pass ``batch_ptr``,
        a 3-D ``cell`` array, or specify ``method`` explicitly.
    wrap_positions : bool, default=True
        If True, wrap input positions into the primary cell before
        neighbor search. Set to False when positions are already
        wrapped (e.g. by a preceding integration step) to save two
        GPU kernel launches per call. Only applies to naive methods; cell list
        methods handle wrapping internally.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the method.

        max_neighbors : int, optional
            Maximum number of neighbors per atom.
            Can be provided to aid in allocation for both naive and cell list methods.
        max_neighbors2 : int, optional
            Maximum number of neighbors per atom within cutoff2.
            Can be provided to aid in allocation for naive dual cutoff method.
        neighbor_matrix : torch.Tensor, optional
            Pre-allocated tensor of shape (total_atoms, max_neighbors) for neighbor indices.
            Can be provided to avoid reallocation for both naive and cell list methods.
        neighbor_matrix_shifts : torch.Tensor, optional
            Pre-allocated tensor of shape (total_atoms, max_neighbors, 3) for shift vectors.
            Can be provided to avoid reallocation for both naive and cell list methods.
        num_neighbors : torch.Tensor, optional
            Pre-allocated tensor of shape (total_atoms,) for neighbor counts.
            Can be provided to avoid reallocation for both naive and cell list methods.
        shift_range_per_dimension : torch.Tensor, optional
            Pre-allocated tensor of shape (1, 3) for shift range in each dimension.
            Can be provided to avoid reallocation for naive methods.
        num_shifts_per_system : torch.Tensor, optional
            Pre-computed tensor of shape (num_systems,) for the number of periodic
            shifts per system. Can be provided to avoid recomputation for naive methods.
        max_shifts_per_system : int, optional
            Maximum per-system shift count.
            Can be provided to avoid recomputation for naive methods.
        cells_per_dimension : torch.Tensor, optional
            Pre-allocated tensor of shape (3,) for number of cells in x, y, z directions.
            Can be provided to avoid reallocation for cell list construction.
        neighbor_search_radius : torch.Tensor, optional
            Pre-allocated tensor of shape (3,) for radius of neighboring cells to search
            in each dimension. Can be provided to avoid reallocation for cell list construction.
        atom_periodic_shifts : torch.Tensor, optional
            Pre-allocated tensor of shape (total_atoms, 3) for periodic boundary crossings
            for each atom. Can be provided to avoid reallocation for cell list construction.
        atom_to_cell_mapping : torch.Tensor, optional
            Pre-allocated tensor of shape (total_atoms, 3) for cell coordinates for each atom.
            Can be provided to avoid reallocation for cell list construction.
        atoms_per_cell_count : torch.Tensor, optional
            Pre-allocated tensor of shape (max_total_cells,) for number of atoms in each cell.
            Can be provided to avoid reallocation for cell list construction.
        cell_atom_start_indices : torch.Tensor, optional
            Pre-allocated tensor of shape (max_total_cells,) for starting index in
            cell_atom_list for each cell. Can be provided to avoid reallocation for
            cell list construction.
        cell_atom_list : torch.Tensor, optional
            Pre-allocated tensor of shape (total_atoms,) for flattened list of atom
            indices organized by cell. Can be provided to avoid reallocation for
            cell list construction.
        max_atoms_per_system : int, optional
            Maximum number of atoms per system. Used in batch naive implementation
            with PBC. If not provided, it will be computed automatically.
            Can be provided to avoid CUDA synchronization.

    Returns
    -------
    results : tuple of torch.Tensor
        Variable-length tuple depending on input parameters. The return pattern follows:

        **Single cutoff:**
          - No PBC, matrix format: ``(neighbor_matrix, num_neighbors)``
          - No PBC, list format: ``(neighbor_list, neighbor_ptr)``
          - With PBC, matrix format: ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``
          - With PBC, list format: ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``

        **Dual cutoff:**
          - No PBC, matrix format: ``(neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2)``
          - No PBC, list format: ``(neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2)``
          - With PBC, matrix format: ``(neighbor_matrix1, num_neighbors1, neighbor_matrix_shifts1, neighbor_matrix2, num_neighbors2, neighbor_matrix_shifts2)``
          - With PBC, list format: ``(neighbor_list1, neighbor_ptr1, neighbor_list_shifts1, neighbor_list2, neighbor_ptr2, neighbor_list_shifts2)``

        **Components returned:**

        - **neighbor_data** (tensor): Neighbor indices, format depends on ``return_neighbor_list``:

            - If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix``
              with shape (total_atoms, max_neighbors), dtype int32. Each row i contains
              indices of atom i's neighbors.
            - If ``return_neighbor_list=True``: Returns ``neighbor_list`` with shape
              (2, num_pairs), dtype int32, in COO format [source_atoms, target_atoms].

        - **num_neighbor_data** (tensor): Information about the number of neighbors for each atom,
          format depends on ``return_neighbor_list``:

            - If ``return_neighbor_list=False`` (default): Returns ``num_neighbors`` with shape (total_atoms,), dtype int32.
              Count of neighbors found for each atom.
            - If ``return_neighbor_list=True``: Returns ``neighbor_ptr`` with shape (total_atoms + 1,), dtype int32.
              CSR-style pointer arrays where ``neighbor_ptr_data[i]`` to ``neighbor_ptr_data[i+1]`` gives the range of
              neighbors for atom i in the flattened neighbor list.

        - **neighbor_shift_data** (tensor, optional): Periodic shift vectors, only when ``pbc`` is provided:
          format depends on ``return_neighbor_list``:

            - If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix_shifts`` with
              shape (total_atoms, max_neighbors, 3), dtype int32.
            - If ``return_neighbor_list=True``: Returns ``unit_shifts`` with shape
              (num_pairs, 3), dtype int32.

        When ``cutoff2`` is provided, the pattern repeats for the second cutoff with interleaved
        components (neighbor_data2, num_neighbor_data2, neighbor_shift_data2) appended to the tuple.

    Examples
    --------
    Single cutoff, matrix format, with PBC::

        >>> nm, num, shifts = neighbor_list(pos, 5.0, cell=cell, pbc=pbc)

    Single cutoff, list format, no PBC::

        >>> nlist, ptr = neighbor_list(pos, 5.0, return_neighbor_list=True)

    Dual cutoff, matrix format, with PBC::

        >>> nm1, num1, sh1, nm2, num2, sh2 = neighbor_list(
        ...     pos, 2.5, cutoff2=5.0, cell=cell, pbc=pbc
        ... )

    See Also
    --------
    naive_neighbor_list : Direct access to naive O(N²) algorithm
    cell_list : Direct access to cell list O(N) algorithm
    batch_naive_neighbor_list : Batched naive algorithm
    batch_cell_list : Batched cell list algorithm
    """
    if cell is not None and pbc is None:
        raise ValueError(
            "`pbc` is required when `cell` is provided. "
            "Pass a boolean tensor of shape (3,) or (num_systems, 3), "
            "e.g. pbc=torch.tensor([True, True, True])."
        )

    if method is None:
        total_atoms = positions.shape[0]

        # Compute average atoms per system for method selection.
        num_systems = 1
        if cell is not None and cell.ndim == 3:
            # cell shape is (num_systems, 3, 3)
            num_systems = cell.shape[0]
        elif batch_ptr is not None:
            # batch_ptr shape is (num_systems + 1,)
            num_systems = batch_ptr.shape[0] - 1
        elif batch_idx is not None:
            # NOTE: reading batch_idx[-1] triggers a GPU-to-CPU sync
            # assume sorted batch_idx
            num_systems = max(1, batch_idx[-1].item() + 1)
        avg_atoms = total_atoms // num_systems

        if cutoff2 is not None:
            method = "naive_dual_cutoff"

        elif avg_atoms >= 2000 and _cluster_tile_auto_eligible(
            positions,
            cell,
            pbc,
            cutoff2,
            half_fill,
            kwargs,
        ):
            method = "cluster_tile"
        elif avg_atoms >= 2000:
            method = "cell_list"
        else:
            method = "naive"

        if batch_idx is not None or batch_ptr is not None:
            method = "batch_" + method
            batch_idx, batch_ptr = prepare_batch_idx_ptr(
                batch_idx, batch_ptr, total_atoms, positions.device
            )
    match method:
        case "naive":
            return naive_neighbor_list(
                positions,
                cutoff,
                pbc=pbc,
                cell=cell,
                half_fill=half_fill,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
                wrap_positions=wrap_positions,
                **kwargs,
            )
        case "cell_list":
            if cell is None:
                positions, cell, pbc = synthesize_cell_for_ss(positions, cutoff)
            return cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                half_fill=half_fill,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
                **kwargs,
            )
        case "batch_naive":
            return batch_naive_neighbor_list(
                positions,
                cutoff,
                pbc=pbc,
                cell=cell,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                half_fill=half_fill,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
                wrap_positions=wrap_positions,
                **kwargs,
            )
        case "batch_cell_list":
            if batch_idx is None or batch_ptr is None:
                batch_idx, batch_ptr = prepare_batch_idx_ptr(
                    batch_idx, batch_ptr, positions.shape[0], positions.device
                )
            if cell is None:
                positions, cell, pbc = synthesize_cell_for_batch(
                    positions, batch_idx, batch_ptr, cutoff
                )
            return batch_cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                batch_idx,
                half_fill=half_fill,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
                **kwargs,
            )
        case "cluster_tile":
            # format="tile" is reachable only via cluster_tile_neighbor_list directly.
            # Reject before any cell handling (mirrors the JAX dispatch): cluster_tile is
            # PBC-implicit, so a missing cell / non-periodic input must error rather than
            # synthesize a tiny box and force PBC (which would emit spurious wrap-around pairs).
            _reject_unsupported_cluster_tile_combo(pbc, half_fill)
            if cell is None:
                raise ValueError("cell is required for method='cluster_tile'")
            return cluster_tile_neighbor_list(
                positions,
                cutoff,
                cell,
                fill_value=fill_value,
                format="coo" if return_neighbor_list else "matrix",
                cutoff2=cutoff2,
                **kwargs,
            )
        case "batch_cluster_tile":
            # Reject before any cell handling (mirrors the JAX dispatch); see the
            # single-system case above for why cluster_tile must not synthesize a cell.
            _reject_unsupported_cluster_tile_combo(pbc, half_fill)
            if batch_idx is None or batch_ptr is None:
                batch_idx, batch_ptr = prepare_batch_idx_ptr(
                    batch_idx, batch_ptr, positions.shape[0], positions.device
                )
            if cell is None:
                raise ValueError("cell is required for method='batch_cluster_tile'")
            return batch_cluster_tile_neighbor_list(
                positions,
                cutoff,
                cell,
                batch_ptr,
                fill_value=fill_value,
                format="coo" if return_neighbor_list else "matrix",
                cutoff2=cutoff2,
                **kwargs,
            )
        case "naive_dual_cutoff":
            return naive_neighbor_list_dual_cutoff(
                positions,
                cutoff,
                cutoff2,
                pbc=pbc,
                cell=cell,
                half_fill=half_fill,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
                wrap_positions=wrap_positions,
                **kwargs,
            )
        case "batch_naive_dual_cutoff":
            return batch_naive_neighbor_list_dual_cutoff(
                positions,
                cutoff,
                cutoff2,
                pbc=pbc,
                cell=cell,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                half_fill=half_fill,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
                wrap_positions=wrap_positions,
                **kwargs,
            )
        case _:
            raise ValueError(f"Invalid method: {method}")


__all__ = [
    # High-level API
    "neighbor_list",
    # Unbatched algorithms
    "cell_list",
    "naive_neighbor_list",
    "naive_neighbor_list_dual_cutoff",
    "cluster_tile_neighbor_list",
    "estimate_cell_list_sizes",
    # Batched algorithms
    "batch_cell_list",
    "batch_naive_neighbor_list",
    "batch_naive_neighbor_list_dual_cutoff",
    "batch_cluster_tile_neighbor_list",
    "estimate_batch_cell_list_sizes",
]
