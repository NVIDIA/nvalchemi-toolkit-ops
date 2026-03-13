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

"""JAX neighbor list API.

This module provides JAX bindings for neighbor list computation and related utilities
for both single and batched systems.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Batch cell list functions
from nvalchemiops.jax.neighbors.batch_cell_list import (
    batch_build_cell_list,
    batch_cell_list,
    batch_query_cell_list,
    estimate_batch_cell_list_sizes,
)

# Batch naive functions
from nvalchemiops.jax.neighbors.batch_naive import (
    batch_naive_neighbor_list,
)

# Batch naive dual cutoff functions
from nvalchemiops.jax.neighbors.batch_naive_dual_cutoff import (
    batch_naive_neighbor_list_dual_cutoff,
)

# Unbatched cell list functions
from nvalchemiops.jax.neighbors.cell_list import (
    build_cell_list,
    cell_list,
    estimate_cell_list_sizes,
    query_cell_list,
)

# Unbatched naive functions
from nvalchemiops.jax.neighbors.naive import (
    naive_neighbor_list,
)

# Unbatched naive dual cutoff functions
from nvalchemiops.jax.neighbors.naive_dual_cutoff import (
    naive_neighbor_list_dual_cutoff,
)

# Utility functions
from nvalchemiops.jax.neighbors.neighbor_utils import (
    NeighborOverflowError,
    allocate_cell_list,
    compute_naive_num_shifts,
    estimate_max_neighbors,
    get_neighbor_list_from_neighbor_matrix,
    prepare_batch_idx_ptr,
)

# Rebuild detection
from nvalchemiops.jax.neighbors.rebuild_detection import (
    batch_cell_list_needs_rebuild,
    batch_neighbor_list_needs_rebuild,
    cell_list_needs_rebuild,
    check_batch_cell_list_rebuild_needed,
    check_batch_neighbor_list_rebuild_needed,
    check_cell_list_rebuild_needed,
    check_neighbor_list_rebuild_needed,
    neighbor_list_needs_rebuild,
)


def neighbor_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    cutoff2: float | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    method: str | None = None,
    wrap_positions: bool = True,
    **kwargs: dict,
):
    """Compute neighbor list using the appropriate method based on the provided parameters.

    This is the main entry point for JAX users of the neighbor list API. It automatically
    selects the most appropriate algorithm (naive O(N²) or cell list O(N)) based on system
    size and parameters.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3)
        Concatenated atomic coordinates for all systems in Cartesian space.
        Each row represents one atom's (x, y, z) position.
        Unwrapped (box-crossing) coordinates are supported when PBC is used;
        the kernel wraps positions internally.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    cell : jax.Array, shape (3, 3) or (num_systems, 3, 3), optional
        Cell matrix defining the simulation box.
    pbc : jax.Array, shape (3,) or (num_systems, 3), dtype=bool, optional
        Periodic boundary condition flags for each dimension.
    batch_idx : jax.Array, shape (total_atoms,), dtype=jnp.int32, optional
        System index for each atom.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=jnp.int32, optional
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
        Choices: "naive", "cell_list", "batch_naive", "batch_cell_list", "naive_dual_cutoff", "batch_naive_dual_cutoff".
        If None, a default method will be chosen based on average atoms per
        system (cell_list when >= 2000, naive otherwise). When only
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
        neighbor_matrix : jax.Array, optional
            Pre-shaped array of shape (total_atoms, max_neighbors) for neighbor indices.
            Can be provided to hint buffer reuse to XLA for both naive and cell list methods.
        neighbor_matrix_shifts : jax.Array, optional
            Pre-shaped array of shape (total_atoms, max_neighbors, 3) for shift vectors.
            Can be provided to hint buffer reuse to XLA for both naive and cell list methods.
        num_neighbors : jax.Array, optional
            Pre-shaped array of shape (total_atoms,) for neighbor counts.
            Can be provided to hint buffer reuse to XLA for both naive and cell list methods.
        shift_range_per_dimension : jax.Array, optional
            Pre-computed array of shape (1, 3) for shift range in each dimension.
            Can be provided to avoid recomputation for naive methods.
        num_shifts_per_system : jax.Array, optional
            Pre-computed array of shape (num_systems,) for the number of periodic
            shifts per system. Can be provided to avoid recomputation for naive methods.
        max_shifts_per_system : int, optional
            Maximum per-system shift count.
            Can be provided to avoid recomputation for naive methods.
        cells_per_dimension : jax.Array, optional
            Pre-computed array of shape (3,) for number of cells in x, y, z directions.
            Can be provided to hint buffer reuse to XLA for cell list construction.
        neighbor_search_radius : jax.Array, optional
            Pre-computed array of shape (3,) for radius of neighboring cells to search
            in each dimension. Can be provided to hint buffer reuse to XLA for cell list construction.
        atom_periodic_shifts : jax.Array, optional
            Pre-shaped array of shape (total_atoms, 3) for periodic boundary crossings
            for each atom. Can be provided to hint buffer reuse to XLA for cell list construction.
        atom_to_cell_mapping : jax.Array, optional
            Pre-shaped array of shape (total_atoms, 3) for cell coordinates for each atom.
            Can be provided to hint buffer reuse to XLA for cell list construction.
        atoms_per_cell_count : jax.Array, optional
            Pre-shaped array of shape (max_total_cells,) for number of atoms in each cell.
            Can be provided to hint buffer reuse to XLA for cell list construction.
        cell_atom_start_indices : jax.Array, optional
            Pre-shaped array of shape (max_total_cells,) for starting index in
            cell_atom_list for each cell. Can be provided to hint buffer reuse to XLA for
            cell list construction.
        cell_atom_list : jax.Array, optional
            Pre-shaped array of shape (total_atoms,) for flattened list of atom
            indices organized by cell. Can be provided to hint buffer reuse to XLA for
            cell list construction.
        max_atoms_per_system : int, optional
            Maximum number of atoms per system. Used in batch naive implementation
            with PBC. If not provided, it will be computed automatically.
            Can be provided to avoid CUDA synchronization.

    Returns
    -------
    results : tuple of jax.Array
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

        - **neighbor_data** (array): Neighbor indices, format depends on ``return_neighbor_list``:

            - If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix``
              with shape (total_atoms, max_neighbors), dtype int32. Each row i contains
              indices of atom i's neighbors.
            - If ``return_neighbor_list=True``: Returns ``neighbor_list`` with shape
              (2, num_pairs), dtype int32, in COO format [source_atoms, target_atoms].

        - **num_neighbor_data** (array): Information about the number of neighbors for each atom,
          format depends on ``return_neighbor_list``:

            - If ``return_neighbor_list=False`` (default): Returns ``num_neighbors`` with shape (total_atoms,), dtype int32.
              Count of neighbors found for each atom.
            - If ``return_neighbor_list=True``: Returns ``neighbor_ptr`` with shape (total_atoms + 1,), dtype int32.
              CSR-style pointer arrays where ``neighbor_ptr_data[i]`` to ``neighbor_ptr_data[i+1]`` gives the range of
              neighbors for atom i in the flattened neighbor list.

        - **neighbor_shift_data** (array, optional): Periodic shift vectors, only when ``pbc`` is provided:
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
    if method is None:
        total_atoms = positions.shape[0]

        num_systems = 1
        if cell is not None and cell.ndim == 3:
            num_systems = cell.shape[0]
        elif batch_ptr is not None:
            num_systems = batch_ptr.shape[0] - 1
        elif batch_idx is not None:
            num_systems = max(1, int(batch_idx[-1]) + 1)
        avg_atoms = total_atoms // num_systems

        if cutoff2 is not None:
            method = "naive_dual_cutoff"

        elif avg_atoms >= 2000:
            method = "cell_list"
            if cell is None or pbc is None:
                cell = jnp.eye(3, dtype=positions.dtype).reshape(1, 3, 3)
                pbc = jnp.array([[False, False, False]])
        else:
            method = "naive"

        if batch_idx is not None or batch_ptr is not None:
            method = "batch_" + method
            batch_idx, batch_ptr = prepare_batch_idx_ptr(
                batch_idx, batch_ptr, total_atoms
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
            # NOTE: JAX cell_list does not yet support half_fill/fill_value
            # (unlike Torch). These parameters are silently ignored here.
            # See JAX_FINAL.md for tracking.
            return cell_list(
                positions,
                cutoff,
                cell,
                pbc,
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
            # NOTE: JAX batch_cell_list does not yet support half_fill/fill_value
            # (unlike Torch). These parameters are silently ignored here.
            return batch_cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                batch_idx,
                batch_ptr=batch_ptr,
                return_neighbor_list=return_neighbor_list,
                **kwargs,
            )
        case "naive_dual_cutoff":
            if cutoff2 is None:
                raise ValueError(
                    "cutoff2 must be provided for naive_dual_cutoff method"
                )
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
            if cutoff2 is None:
                raise ValueError(
                    "cutoff2 must be provided for batch_naive_dual_cutoff method"
                )
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
    # Unbatched neighbor list
    "naive_neighbor_list",
    "naive_neighbor_list_dual_cutoff",
    "estimate_cell_list_sizes",
    "build_cell_list",
    "query_cell_list",
    "cell_list",
    # Batched neighbor list
    "batch_naive_neighbor_list",
    "batch_naive_neighbor_list_dual_cutoff",
    "estimate_batch_cell_list_sizes",
    "batch_build_cell_list",
    "batch_query_cell_list",
    "batch_cell_list",
    # Rebuild detection
    "cell_list_needs_rebuild",
    "neighbor_list_needs_rebuild",
    "check_cell_list_rebuild_needed",
    "check_neighbor_list_rebuild_needed",
    # Utilities
    "compute_naive_num_shifts",
    "get_neighbor_list_from_neighbor_matrix",
    "prepare_batch_idx_ptr",
    "allocate_cell_list",
    "estimate_max_neighbors",
    "NeighborOverflowError",
]
