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

"""JAX bindings for batched cell list O(N) neighbor list construction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel

from nvalchemiops.jax.neighbors.neighbor_utils import (
    allocate_cell_list,
    get_neighbor_list_from_neighbor_matrix,
    prepare_batch_idx_ptr,
)
from nvalchemiops.neighbors.batch_cell_list import (
    _batch_cell_list_bin_atoms_overload,
    _batch_cell_list_build_neighbor_matrix_overload,
    _batch_cell_list_construct_bin_size_overload,
    _batch_cell_list_count_atoms_per_bin_overload,
    _compute_cells_per_system,
)
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# Build step 1: Construct bin sizes (per system)
_jax_batch_construct_bin_size_f32 = jax_kernel(
    _batch_cell_list_construct_bin_size_overload[wp.float32],
    num_outputs=1,
    in_out_argnames=["cells_per_dimension"],
    enable_backward=False,
)
_jax_batch_construct_bin_size_f64 = jax_kernel(
    _batch_cell_list_construct_bin_size_overload[wp.float64],
    num_outputs=1,
    in_out_argnames=["cells_per_dimension"],
    enable_backward=False,
)

# Helper: Compute cells per system
_jax_compute_cells_per_system = jax_kernel(
    _compute_cells_per_system,
    num_outputs=1,
    in_out_argnames=["cells_per_system"],
    enable_backward=False,
)

# Build step 2: Count atoms per bin
_jax_batch_count_atoms_per_bin_f32 = jax_kernel(
    _batch_cell_list_count_atoms_per_bin_overload[wp.float32],
    num_outputs=2,
    in_out_argnames=["atoms_per_cell_count", "atom_periodic_shifts"],
    enable_backward=False,
)
_jax_batch_count_atoms_per_bin_f64 = jax_kernel(
    _batch_cell_list_count_atoms_per_bin_overload[wp.float64],
    num_outputs=2,
    in_out_argnames=["atoms_per_cell_count", "atom_periodic_shifts"],
    enable_backward=False,
)

# Build step 3: Bin atoms into cells
_jax_batch_bin_atoms_f32 = jax_kernel(
    _batch_cell_list_bin_atoms_overload[wp.float32],
    num_outputs=3,
    in_out_argnames=["atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_list"],
    enable_backward=False,
)
_jax_batch_bin_atoms_f64 = jax_kernel(
    _batch_cell_list_bin_atoms_overload[wp.float64],
    num_outputs=3,
    in_out_argnames=["atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_list"],
    enable_backward=False,
)

# Query: Build neighbor matrix from batch cell list
_jax_batch_build_neighbor_matrix_f32 = jax_kernel(
    _batch_cell_list_build_neighbor_matrix_overload[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)
_jax_batch_build_neighbor_matrix_f64 = jax_kernel(
    _batch_cell_list_build_neighbor_matrix_overload[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)

__all__ = [
    "batch_cell_list",
    "batch_build_cell_list",
    "batch_query_cell_list",
    "estimate_batch_cell_list_sizes",
]


def estimate_batch_cell_list_sizes(
    positions: jax.Array,
    batch_ptr: jax.Array | None = None,
    batch_idx: jax.Array | None = None,
    cell: jax.Array | None = None,
    cutoff: float = 5.0,
    pbc: jax.Array | None = None,
    buffer_factor: float = 1.5,
) -> tuple[int, jax.Array, jax.Array]:
    """Estimate required batch cell list sizes.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        Batch indices for each atom.
    cell : jax.Array, shape (num_systems, 3, 3), dtype=float32 or float64, optional
        Cell matrices for each system.
    cutoff : float, optional
        Cutoff distance. Default is 5.0.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool, optional
        PBC flags.
    buffer_factor : float, optional
        Buffer multiplier. Default is 1.5.

    Returns
    -------
    max_total_cells : int
        Maximum total cells to allocate.
    cells_per_dimension : jax.Array, shape (num_systems, 3)
        Cells per dimension for each system.
    neighbor_search_radius : jax.Array, shape (num_systems, 3)
        Search radius for each system.

    .. warning::

        This function is **not compatible with** ``jax.jit``. The returned
        ``max_total_cells`` is used to determine array allocation sizes, which
        must be concrete (statically known) at JAX trace time. When using
        ``batch_cell_list`` or ``batch_build_cell_list`` inside ``jax.jit``,
        provide ``max_total_cells`` explicitly to bypass this function.
    """

    # Prepare batch info
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0]
    )
    num_systems = batch_ptr.shape[0] - 1

    # Simple estimation per system
    max_total_cells = 0
    cells_per_dim_list = []
    search_radius_list = []

    for sys_idx in range(num_systems):
        start_idx = batch_ptr[sys_idx]
        end_idx = batch_ptr[sys_idx + 1]
        num_atoms_in_sys = end_idx - start_idx

        if num_atoms_in_sys == 0:
            cells_per_dim_list.append(jnp.ones(3, dtype=jnp.int32))
            search_radius_list.append(jnp.ones(3, dtype=jnp.int32))
            continue

        # Volume estimation
        if cell is not None:
            det = jnp.linalg.det(cell[sys_idx])
            volume = jnp.abs(det)
        else:
            volume = 1000.0  # Default assumption

        cell_volume = cutoff**3
        # TODO: This estimation derives array sizes from traced input data (cell
        # geometry), which is fundamentally incompatible with jax.jit compilation.
        # The JAX bindings need a refactored usage pattern where sizing is always
        # performed outside the JIT boundary, or a fixed upper-bound allocation
        # strategy is adopted.
        num_cells_est = max(int(volume / cell_volume * buffer_factor), 8)
        max_total_cells += num_cells_est

        cells_per_dim = jnp.ceil(num_cells_est ** (1 / 3)).astype(jnp.int32)
        cells_per_dim_list.append(cells_per_dim * jnp.ones(3, dtype=jnp.int32))
        search_radius_list.append(jnp.ones(3, dtype=jnp.int32))

    cells_per_dimension = jnp.stack(cells_per_dim_list, axis=0)
    neighbor_search_radius = jnp.stack(search_radius_list, axis=0)

    return max_total_cells, cells_per_dimension, neighbor_search_radius


def batch_build_cell_list(
    positions: jax.Array,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    cutoff: float = 5.0,
    max_total_cells: int | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Build spatial cell lists for batch of systems.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        Batch indices.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts.
    cell : jax.Array, shape (num_systems, 3, 3), dtype=float32 or float64, optional
        Cell matrices.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool, optional
        PBC flags.
    cutoff : float, optional
        Cutoff distance. Default is 5.0.
    max_total_cells : int, optional
        Maximum cells. If None, will be estimated.

    Returns
    -------
    cells_per_dimension : jax.Array, shape (num_systems, 3), dtype=int32
        Number of cells in x, y, z directions for each system.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom.
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom.
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell.
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32
        Starting index in ``cell_atom_list`` for each cell.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell.
    neighbor_search_radius : jax.Array, shape (num_systems, 3), dtype=int32
        Search radius in neighboring cells for each system.
    cell_origin : jax.Array, shape (3,), dtype same as positions
        Cell origin point (currently zeros).

    Notes
    -----
    When calling inside ``jax.jit``, ``max_total_cells`` **must** be provided
    to avoid calling ``estimate_batch_cell_list_sizes``, which is not JIT-compatible.
    """

    # Prepare batch info
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0]
    )
    num_systems = batch_ptr.shape[0] - 1

    if max_total_cells is None:
        max_total_cells, cells_per_dim_est, neighbor_search_radius = (
            estimate_batch_cell_list_sizes(
                positions, batch_ptr, batch_idx, cell, cutoff, pbc
            )
        )
        # Ensure neighbor_search_radius is on the correct device
        neighbor_search_radius = neighbor_search_radius
    else:
        neighbor_search_radius = jnp.ones((num_systems, 3), dtype=jnp.int32)

    # Allocate cell list tensors
    (
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    ) = allocate_cell_list(
        positions.shape[0],
        max_total_cells,
        neighbor_search_radius,
    )

    # Select kernels based on dtype
    if positions.dtype == jnp.float64:
        _construct = _jax_batch_construct_bin_size_f64
        _count = _jax_batch_count_atoms_per_bin_f64
        _bin = _jax_batch_bin_atoms_f64
    else:
        _construct = _jax_batch_construct_bin_size_f32
        _count = _jax_batch_count_atoms_per_bin_f32
        _bin = _jax_batch_bin_atoms_f32
        positions = positions.astype(jnp.float32)

    # Ensure cell dtype matches positions
    if cell is not None and cell.dtype != positions.dtype:
        cell = cell.astype(positions.dtype)

    # Ensure pbc is bool with shape (num_systems, 3)
    if pbc is not None:
        pbc_bool = pbc.astype(jnp.bool_)
    else:
        pbc_bool = jnp.ones((num_systems, 3), dtype=jnp.bool_)

    total_atoms = positions.shape[0]

    # Step 1: Construct bin sizes (one thread per system)
    (cells_per_dimension,) = _construct(
        cell,
        pbc_bool,
        cells_per_dimension,
        float(cutoff),
        int(max_total_cells),
        launch_dims=(num_systems,),
    )

    # Step 2: Compute cells_per_system and cell_offsets
    cells_per_system = jnp.zeros(num_systems, dtype=jnp.int32)
    (cells_per_system,) = _jax_compute_cells_per_system(
        cells_per_dimension,
        cells_per_system,
        launch_dims=(num_systems,),
    )
    cell_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(cells_per_system[:-1]),
        ]
    )

    # Step 3: Count atoms per bin
    atoms_per_cell_count, atom_periodic_shifts = _count(
        positions,
        cell,
        pbc_bool,
        batch_idx,
        cells_per_dimension,
        cell_offsets,
        atoms_per_cell_count,
        atom_periodic_shifts,
        launch_dims=(total_atoms,),
    )

    # Step 4: Compute exclusive prefix sum (replaces wp.utils.array_scan)
    cell_atom_start_indices = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(atoms_per_cell_count[:-1]),
        ]
    )

    # Step 5: Zero counts before second pass
    atoms_per_cell_count = jnp.zeros_like(atoms_per_cell_count)

    # Step 6: Bin atoms
    atom_to_cell_mapping, atoms_per_cell_count, cell_atom_list = _bin(
        positions,
        cell,
        pbc_bool,
        batch_idx,
        cells_per_dimension,
        cell_offsets,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        launch_dims=(total_atoms,),
    )

    cell_origin = jnp.zeros(3, dtype=positions.dtype)

    return (
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_search_radius,
        cell_origin,
    )


def batch_query_cell_list(
    positions: jax.Array,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    cutoff: float = 5.0,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    cells_per_dimension: jax.Array | None = None,
    atom_periodic_shifts: jax.Array | None = None,
    atom_to_cell_mapping: jax.Array | None = None,
    cell_atom_start_indices: jax.Array | None = None,
    cell_atom_list: jax.Array | None = None,
    atoms_per_cell_count: jax.Array | None = None,
    neighbor_search_radius: jax.Array | None = None,
    max_neighbors: int | None = None,
    neighbor_matrix: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Query batch cell lists to find neighbors.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        Batch indices.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts.
    cutoff : float, optional
        Cutoff distance.
    cell : jax.Array, shape (num_systems, 3, 3), dtype=float32 or float64, optional
        Cell matrices.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool, optional
        PBC flags.
    cells_per_dimension : jax.Array, shape (num_systems, 3), dtype=int32, optional
        Cells per dimension.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32, optional
        Periodic shifts for each atom (output from ``batch_build_cell_list``).
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32, optional
        Cell mappings.
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32, optional
        Start indices.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32, optional
        Cell atom list.
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32, optional
        Number of atoms assigned to each cell. Output from ``batch_build_cell_list``.
    neighbor_search_radius : jax.Array, shape (num_systems, 3), dtype=int32, optional
        Search radius.
    max_neighbors : int, optional
        Maximum neighbors per atom.
    neighbor_matrix : jax.Array, shape (total_atoms, max_neighbors), dtype=int32, optional
        Pre-allocated neighbor matrix.
    num_neighbors : jax.Array, shape (total_atoms,), dtype=int32, optional
        Pre-allocated neighbors count array.
    neighbor_matrix_shifts : jax.Array, shape (total_atoms, max_neighbors, 3), dtype=int32, optional
        Pre-allocated shift vectors array. Pass in a pre-shaped array to hint buffer
        reuse to XLA; note that JAX returns a new array rather than mutating the input.

    Returns
    -------
    neighbor_matrix : jax.Array, shape (total_atoms, max_neighbors), dtype=int32
        Neighbor matrix.
    num_neighbors : jax.Array, shape (total_atoms,), dtype=int32
        Neighbors count.
    neighbor_matrix_shifts : jax.Array, shape (total_atoms, max_neighbors, 3), dtype=int32
        Periodic shifts for each neighbor relationship.
    """
    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(cutoff)

    # Prepare batch info
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0]
    )
    num_systems = batch_ptr.shape[0] - 1

    if neighbor_matrix is None:
        neighbor_matrix = jnp.full(
            (positions.shape[0], max_neighbors),
            positions.shape[0],
            dtype=jnp.int32,
        )
    else:
        neighbor_matrix = neighbor_matrix.at[:].set(positions.shape[0])

    if num_neighbors is None:
        num_neighbors = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    else:
        num_neighbors = num_neighbors.at[:].set(0)

    # Select kernel based on dtype
    if positions.dtype == jnp.float64:
        _query_kernel = _jax_batch_build_neighbor_matrix_f64
    else:
        _query_kernel = _jax_batch_build_neighbor_matrix_f32
        positions = positions.astype(jnp.float32)

    # Ensure cell dtype matches positions
    if cell is not None and cell.dtype != positions.dtype:
        cell = cell.astype(positions.dtype)

    # Ensure pbc is bool with shape (num_systems, 3)
    if pbc is not None:
        pbc_bool = pbc.astype(jnp.bool_)
    else:
        pbc_bool = jnp.ones((num_systems, 3), dtype=jnp.bool_)

    total_atoms = positions.shape[0]

    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = jnp.zeros(
            (total_atoms, max_neighbors, 3),
            dtype=jnp.int32,
        )
    else:
        neighbor_matrix_shifts = neighbor_matrix_shifts.at[:].set(0)

    if atoms_per_cell_count is None:
        max_total_cells = cell_atom_start_indices.shape[0]
        atoms_per_cell_count = jnp.zeros(max_total_cells, dtype=jnp.int32)

    # Compute cell_offsets from cells_per_dimension
    cells_per_system = jnp.prod(cells_per_dimension, axis=1)
    cell_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(cells_per_system[:-1], dtype=jnp.int32),
        ]
    )

    neighbor_matrix, neighbor_matrix_shifts, num_neighbors = _query_kernel(
        positions,
        cell,
        pbc_bool,
        batch_idx,
        float(cutoff),
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        cell_offsets,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        False,  # half_fill
        launch_dims=(total_atoms,),
    )

    return neighbor_matrix, num_neighbors, neighbor_matrix_shifts


def batch_cell_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    max_neighbors: int | None = None,
    max_total_cells: int | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    return_neighbor_list: bool = False,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, tuple]:
    """Build and query spatial cell lists for batch of systems.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates.
    cutoff : float
        Cutoff distance for neighbor detection.
    cell : jax.Array, shape (num_systems, 3, 3), dtype=float32 or float64, optional
        Cell matrices defining lattice vectors. Default is identity matrix.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool, optional
        Periodic boundary condition flags. Default is all True.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        Batch indices for each atom.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts defining system boundaries.
    max_neighbors : int, optional
        Maximum number of neighbors per atom. If None, will be estimated.
    max_total_cells : int, optional
        Maximum number of cells to allocate. If None, will be estimated.
    neighbor_matrix_shifts : jax.Array, shape (total_atoms, max_neighbors, 3), dtype=int32, optional
        Pre-allocated shift vectors array. If None, will be allocated internally.
        Pass in a pre-shaped array to hint buffer reuse to XLA; note that JAX returns
        a new array rather than mutating the input.
    return_neighbor_list : bool, optional
        If True, convert result to COO neighbor list format. Default is False.

    Returns
    -------
    neighbor_data : jax.Array
        If ``return_neighbor_list=False`` (default): ``neighbor_matrix`` with shape
        (total_atoms, max_neighbors), dtype int32.
        If ``return_neighbor_list=True``: ``neighbor_list`` with shape
        (2, num_pairs), dtype int32, in COO format.
    neighbor_count : jax.Array
        If ``return_neighbor_list=False``: ``num_neighbors`` with shape
        (total_atoms,), dtype int32.
        If ``return_neighbor_list=True``: ``neighbor_ptr`` with shape
        (total_atoms + 1,), dtype int32.
    shift_data : jax.Array
        If ``return_neighbor_list=False`` (default): ``neighbor_matrix_shifts`` with shape
        (total_atoms, max_neighbors, 3), dtype int32.
        If ``return_neighbor_list=True``: ``neighbor_list_shifts`` with shape
        (num_pairs, 3), dtype int32.
        Periodic shift vectors for each neighbor relationship.

    See Also
    --------
    batch_build_cell_list : Build cell list separately
    batch_query_cell_list : Query cell list separately
    batch_naive_neighbor_list : Naive O(N^2) method
    """

    # Prepare batch info
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0]
    )

    # Build cell list
    (
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_search_radius,
        cell_origin,
    ) = batch_build_cell_list(
        positions,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        cell=cell,
        pbc=pbc,
        cutoff=cutoff,
        max_total_cells=max_total_cells,
    )

    # Query cell list
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = batch_query_cell_list(
        positions=positions,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        cutoff=cutoff,
        cell=cell,
        pbc=pbc,
        cells_per_dimension=cells_per_dimension,
        atom_periodic_shifts=atom_periodic_shifts,
        atom_to_cell_mapping=atom_to_cell_mapping,
        atoms_per_cell_count=atoms_per_cell_count,
        cell_atom_start_indices=cell_atom_start_indices,
        cell_atom_list=cell_atom_list,
        neighbor_search_radius=neighbor_search_radius,
        max_neighbors=max_neighbors,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
    )

    if return_neighbor_list:
        neighbor_list, neighbor_ptr, neighbor_list_shifts = (
            get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                neighbor_shift_matrix=neighbor_matrix_shifts,
                fill_value=positions.shape[0],
            )
        )
        return neighbor_list, neighbor_ptr, neighbor_list_shifts
    else:
        return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
