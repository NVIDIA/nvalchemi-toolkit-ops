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

from nvalchemiops.jax.neighbors._autograd import (
    _build_index_residuals,
    _NeighborForwardOutput,
    _route_pair_outputs,
)
from nvalchemiops.jax.neighbors.neighbor_utils import (
    allocate_cell_list,
    get_neighbor_list_from_neighbor_matrix,
    prepare_batch_idx_ptr,
)
from nvalchemiops.neighbors.cell_list import (
    get_build_cell_list_kernel,
    get_cell_list_cells_per_system_kernel,
    get_cell_list_gather_kernel,
    get_query_cell_list_kernel,
)
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.neighbors.output_args import (
    _has_partial_or_pair_outputs,
)

# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# Build step 1: Construct bin sizes (per system)
_jax_batch_construct_bin_size_f32 = jax_kernel(
    get_build_cell_list_kernel("construct_bin_size", wp.float32, batched=True),
    num_outputs=1,
    in_out_argnames=["cells_per_dimension_batch"],
    enable_backward=False,
)
_jax_batch_construct_bin_size_f64 = jax_kernel(
    get_build_cell_list_kernel("construct_bin_size", wp.float64, batched=True),
    num_outputs=1,
    in_out_argnames=["cells_per_dimension_batch"],
    enable_backward=False,
)

# Helper: Compute cells per system
_jax_compute_cells_per_system = jax_kernel(
    get_cell_list_cells_per_system_kernel(),
    num_outputs=1,
    in_out_argnames=["cells_per_system"],
    enable_backward=False,
)

# Build step 2: Count atoms per bin
_jax_batch_count_atoms_per_bin_f32 = jax_kernel(
    get_build_cell_list_kernel("count_atoms", wp.float32, batched=True),
    num_outputs=2,
    in_out_argnames=["atoms_per_cell_count", "atom_periodic_shifts"],
    enable_backward=False,
)
_jax_batch_count_atoms_per_bin_f64 = jax_kernel(
    get_build_cell_list_kernel("count_atoms", wp.float64, batched=True),
    num_outputs=2,
    in_out_argnames=["atoms_per_cell_count", "atom_periodic_shifts"],
    enable_backward=False,
)

# Build step 3: Bin atoms into cells
_jax_batch_bin_atoms_f32 = jax_kernel(
    get_build_cell_list_kernel("bin_atoms", wp.float32, batched=True),
    num_outputs=3,
    in_out_argnames=["atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_list"],
    enable_backward=False,
)
_jax_batch_bin_atoms_f64 = jax_kernel(
    get_build_cell_list_kernel("bin_atoms", wp.float64, batched=True),
    num_outputs=3,
    in_out_argnames=["atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_list"],
    enable_backward=False,
)

# Gather: pack positions + atom_periodic_shifts into per-cell-contiguous layout
# (cell_atom_list permutation) for coalesced reads by the sorted-build kernel.
_jax_batch_gather_positions_by_cell_f32 = jax_kernel(
    get_cell_list_gather_kernel(wp.float32),
    num_outputs=2,
    in_out_argnames=["sorted_positions", "sorted_shifts"],
    enable_backward=False,
)
_jax_batch_gather_positions_by_cell_f64 = jax_kernel(
    get_cell_list_gather_kernel(wp.float64),
    num_outputs=2,
    in_out_argnames=["sorted_positions", "sorted_shifts"],
    enable_backward=False,
)

# Query: sorted-reads atom-centric batch neighbor matrix kernel.  The same
# kernel handles selective and non-selective callers via the ``rebuild_flags``
# array (always-True for non-selective; per-system bool array otherwise).
_jax_batch_build_neighbor_matrix_local_count_sorted_f32 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float32,
        strategy="atom_centric",
        batched=True,
        selective=True,
        partial=False,
        return_vectors=False,
        return_distances=False,
        pair_fn=None,
    ),
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)
_jax_batch_build_neighbor_matrix_local_count_sorted_f64 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float64,
        strategy="atom_centric",
        batched=True,
        selective=True,
        partial=False,
        return_vectors=False,
        return_distances=False,
        pair_fn=None,
    ),
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)

# Pair-output variants — consumed by the autograd path when
# ``return_distances`` / ``return_vectors`` is set.  The bytes written into
# ``neighbor_vectors`` / ``neighbor_distances`` are differentiable via the
# JAX autograd primitive in :mod:`nvalchemiops.jax.neighbors._autograd`.
_jax_batch_build_neighbor_matrix_local_count_sorted_pair_f32 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float32,
        strategy="atom_centric",
        batched=True,
        selective=True,
        partial=False,
        return_vectors=True,
        return_distances=True,
        pair_fn=None,
    ),
    num_outputs=5,
    in_out_argnames=[
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)
_jax_batch_build_neighbor_matrix_local_count_sorted_pair_f64 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float64,
        strategy="atom_centric",
        batched=True,
        selective=True,
        partial=False,
        return_vectors=True,
        return_distances=True,
        pair_fn=None,
    ),
    num_outputs=5,
    in_out_argnames=[
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
        "neighbor_vectors",
        "neighbor_distances",
    ],
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
    target_indices: jax.Array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: jax.Array | None = None,
    neighbor_vectors: jax.Array | None = None,
    neighbor_distances: jax.Array | None = None,
    pair_energies: jax.Array | None = None,
    pair_forces: jax.Array | None = None,
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
    if _has_partial_or_pair_outputs(
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    ):
        raise NotImplementedError(
            "batch_build_cell_list does not accept return_distances / "
            "return_vectors / target_indices / pair_fn-related kwargs.  "
            "Use the top-level batch_cell_list() wrapper, which routes "
            "pair outputs through the JAX autograd path, or call the "
            "warp factory directly for low-level access.",
        )

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
    empty_bool1d = jnp.zeros((0,), dtype=jnp.bool_)
    empty_i32 = jnp.zeros((0,), dtype=jnp.int32)

    total_atoms = positions.shape[0]

    # Step 1: Construct bin sizes (one thread per system)
    (cells_per_dimension,) = _construct(
        cell,
        empty_bool1d,
        pbc_bool,
        empty_i32,
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
            jnp.cumsum(cells_per_system[:-1], dtype=jnp.int32),
        ]
    )

    # Step 3: Count atoms per bin
    atoms_per_cell_count, atom_periodic_shifts = _count(
        positions,
        cell,
        empty_bool1d,
        pbc_bool,
        batch_idx,
        empty_i32,
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
            jnp.cumsum(atoms_per_cell_count[:-1], dtype=jnp.int32),
        ]
    )

    # Step 5: Zero counts before second pass
    atoms_per_cell_count = jnp.zeros_like(atoms_per_cell_count)

    # Step 6: Bin atoms
    atom_to_cell_mapping, atoms_per_cell_count, cell_atom_list = _bin(
        positions,
        cell,
        empty_bool1d,
        pbc_bool,
        batch_idx,
        empty_i32,
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
    rebuild_flags: jax.Array | None = None,
    target_indices: jax.Array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: jax.Array | None = None,
    neighbor_vectors: jax.Array | None = None,
    neighbor_distances: jax.Array | None = None,
    pair_energies: jax.Array | None = None,
    pair_forces: jax.Array | None = None,
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

    if _has_partial_or_pair_outputs(
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    ):
        raise NotImplementedError(
            "batch_query_cell_list does not accept return_distances / "
            "return_vectors / target_indices / pair_fn-related kwargs.  "
            "Use the top-level batch_cell_list() wrapper, which routes "
            "pair outputs through the JAX autograd path, or call the "
            "warp factory directly for low-level access.",
        )

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
    elif rebuild_flags is None:
        neighbor_matrix = neighbor_matrix.at[:].set(jnp.int32(positions.shape[0]))

    if num_neighbors is None:
        num_neighbors = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    elif rebuild_flags is None:
        num_neighbors = num_neighbors.at[:].set(jnp.int32(0))

    # Select kernels based on dtype; same sorted-reads kernel for selective
    # and non-selective (controlled by ``rebuild_flags``).
    if positions.dtype == jnp.float64:
        _gather_kernel = _jax_batch_gather_positions_by_cell_f64
        _sorted_build_kernel = _jax_batch_build_neighbor_matrix_local_count_sorted_f64
    else:
        _gather_kernel = _jax_batch_gather_positions_by_cell_f32
        _sorted_build_kernel = _jax_batch_build_neighbor_matrix_local_count_sorted_f32
        positions = positions.astype(jnp.float32)

    # Ensure cell dtype matches positions
    if cell is not None and cell.dtype != positions.dtype:
        cell = cell.astype(positions.dtype)

    # Ensure pbc is bool with shape (num_systems, 3)
    if pbc is not None:
        pbc_bool = pbc.astype(jnp.bool_)
    else:
        pbc_bool = jnp.ones((num_systems, 3), dtype=jnp.bool_)
    empty_bool1d = jnp.zeros((0,), dtype=jnp.bool_)
    empty_i32 = jnp.zeros((0,), dtype=jnp.int32)
    empty_scalar2d = jnp.zeros((0, 0), dtype=positions.dtype)
    empty_vec_matrix = jnp.zeros((0, 0, 3), dtype=positions.dtype)

    total_atoms = positions.shape[0]

    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = jnp.zeros(
            (total_atoms, max_neighbors, 3),
            dtype=jnp.int32,
        )
    elif rebuild_flags is None:
        neighbor_matrix_shifts = neighbor_matrix_shifts.at[:].set(jnp.int32(0))

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

    batch_idx_i32 = batch_idx.astype(jnp.int32)

    if rebuild_flags is not None:
        rf = rebuild_flags.astype(jnp.bool_)
        atom_rebuild = rf[batch_idx_i32]
        num_neighbors = jnp.where(
            atom_rebuild, jnp.zeros_like(num_neighbors), num_neighbors
        )
    else:
        rf = jnp.ones((num_systems,), dtype=jnp.bool_)

    sorted_positions = jnp.zeros((total_atoms, 3), dtype=positions.dtype)
    sorted_atom_periodic_shifts = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
    sorted_positions, sorted_atom_periodic_shifts = _gather_kernel(
        positions,
        atom_periodic_shifts,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        launch_dims=(total_atoms,),
    )

    neighbor_matrix, neighbor_matrix_shifts, num_neighbors = _sorted_build_kernel(
        positions,
        atom_periodic_shifts,
        sorted_positions,
        sorted_atom_periodic_shifts,
        cell,
        empty_bool1d,
        pbc_bool,
        batch_idx_i32,
        float(cutoff),
        empty_i32,
        cells_per_dimension,
        empty_i32,
        neighbor_search_radius,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        cell_offsets,
        empty_i32,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        empty_vec_matrix,
        empty_scalar2d,
        empty_scalar2d,
        empty_scalar2d,
        empty_vec_matrix,
        False,  # half_fill
        rf,
        launch_dims=(total_atoms,),
    )

    return neighbor_matrix, num_neighbors, neighbor_matrix_shifts


def _batch_cell_list_pair_outputs_forward(
    positions: jax.Array,
    cell: jax.Array,
    *,
    pbc_bool: jax.Array,
    batch_idx_i32: jax.Array,
    cells_per_dimension: jax.Array,
    atom_periodic_shifts: jax.Array,
    atom_to_cell_mapping: jax.Array,
    atoms_per_cell_count: jax.Array,
    cell_atom_start_indices: jax.Array,
    cell_atom_list: jax.Array,
    cell_offsets: jax.Array,
    neighbor_search_radius: jax.Array,
    neighbor_matrix: jax.Array,
    neighbor_matrix_shifts: jax.Array,
    num_neighbors: jax.Array,
    neighbor_vectors: jax.Array,
    neighbor_distances: jax.Array,
    cutoff: float,
) -> _NeighborForwardOutput:
    """Forward closure consumed by ``_route_pair_outputs``.

    Runs the gather + batched pair-output kernel.  The Warp launches do not
    propagate gradients across the JAX boundary, so positions/cell are
    detached here; the autograd primitive's reconstruction backward receives
    the live tensors separately.
    """
    positions = jax.lax.stop_gradient(positions)
    cell = jax.lax.stop_gradient(cell)

    if positions.dtype == jnp.float64:
        gather_kernel = _jax_batch_gather_positions_by_cell_f64
        pair_kernel = _jax_batch_build_neighbor_matrix_local_count_sorted_pair_f64
    else:
        gather_kernel = _jax_batch_gather_positions_by_cell_f32
        pair_kernel = _jax_batch_build_neighbor_matrix_local_count_sorted_pair_f32

    total_atoms = positions.shape[0]
    num_systems = pbc_bool.shape[0]
    sorted_positions = jnp.zeros((total_atoms, 3), dtype=positions.dtype)
    sorted_atom_periodic_shifts = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
    sorted_positions, sorted_atom_periodic_shifts = gather_kernel(
        positions,
        atom_periodic_shifts,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        launch_dims=(total_atoms,),
    )

    empty_bool1d = jnp.zeros((0,), dtype=jnp.bool_)
    empty_i32 = jnp.zeros((0,), dtype=jnp.int32)
    empty_scalar2d = jnp.zeros((0, 0), dtype=positions.dtype)
    empty_vec_matrix = jnp.zeros((0, 0, 3), dtype=positions.dtype)
    rf = jnp.ones((num_systems,), dtype=jnp.bool_)

    nm_out, nms_out, nn_out, nv_out, nd_out = pair_kernel(
        positions,
        atom_periodic_shifts,
        sorted_positions,
        sorted_atom_periodic_shifts,
        cell,
        empty_bool1d,
        pbc_bool,
        batch_idx_i32,
        float(cutoff),
        empty_i32,
        cells_per_dimension,
        empty_i32,
        neighbor_search_radius,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        cell_offsets,
        empty_i32,  # target_indices
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        neighbor_vectors,
        neighbor_distances,
        empty_scalar2d,  # pair_params
        empty_scalar2d,  # pair_energies
        empty_vec_matrix,  # pair_forces
        False,  # half_fill
        rf,
        launch_dims=(total_atoms,),
    )

    i_idx, j_idx, shifts_ret, _, mask_ = _build_index_residuals(
        nm_out,
        nn_out,
        nms_out,
    )
    K, M = nm_out.shape
    return _NeighborForwardOutput(
        distances=nd_out,
        vectors=nv_out,
        extra_outputs=(nm_out, nn_out, nms_out),
        i_idx=i_idx,
        j_idx=j_idx,
        shifts=shifts_ret,
        batch_idx=batch_idx_i32,
        active_mask=mask_,
        matrix_shape=(K, M),
    )


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
    target_indices: jax.Array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: jax.Array | None = None,
    neighbor_vectors: jax.Array | None = None,
    neighbor_distances: jax.Array | None = None,
    pair_energies: jax.Array | None = None,
    pair_forces: jax.Array | None = None,
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

    has_pair_outputs = _has_partial_or_pair_outputs(
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )
    if has_pair_outputs and (
        pair_fn is not None
        or pair_params is not None
        or pair_energies is not None
        or pair_forces is not None
        or target_indices is not None
    ):
        raise NotImplementedError(
            "pair_fn / pair_params / pair_energies / pair_forces / "
            "target_indices are not yet wired through the JAX "
            "batch_cell_list binding.  Only return_distances and "
            "return_vectors are supported in this pass.",
        )

    # Preserve LIVE positions/cell for the autograd primitive; the warp
    # kernels are non-differentiable across the JAX boundary so we detach
    # them for the cell-list build.
    positions_for_grad = positions
    cell_for_grad = cell
    if has_pair_outputs:
        positions = jax.lax.stop_gradient(positions)
        cell = jax.lax.stop_gradient(cell)

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

    if has_pair_outputs:
        num_systems = batch_ptr.shape[0] - 1
        if pbc is not None:
            pbc_bool = pbc.astype(jnp.bool_)
        else:
            pbc_bool = jnp.ones((num_systems, 3), dtype=jnp.bool_)
        if max_neighbors is None:
            max_neighbors = estimate_max_neighbors(cutoff)
        total_atoms = positions.shape[0]
        if neighbor_matrix_shifts is None:
            neighbor_matrix_shifts = jnp.zeros(
                (total_atoms, max_neighbors, 3), dtype=jnp.int32
            )
        nm = jnp.full((total_atoms, max_neighbors), total_atoms, dtype=jnp.int32)
        nn = jnp.zeros(total_atoms, dtype=jnp.int32)
        if return_distances and neighbor_distances is None:
            neighbor_distances = jnp.zeros(
                (total_atoms, max_neighbors), dtype=positions.dtype
            )
        if return_vectors and neighbor_vectors is None:
            neighbor_vectors = jnp.zeros(
                (total_atoms, max_neighbors, 3), dtype=positions.dtype
            )
        if neighbor_distances is None:
            neighbor_distances = jnp.zeros(
                (total_atoms, max_neighbors), dtype=positions.dtype
            )
        if neighbor_vectors is None:
            neighbor_vectors = jnp.zeros(
                (total_atoms, max_neighbors, 3), dtype=positions.dtype
            )
        cells_per_system = jnp.prod(cells_per_dimension, axis=1)
        cell_offsets = jnp.concatenate(
            [
                jnp.array([0], dtype=jnp.int32),
                jnp.cumsum(cells_per_system[:-1], dtype=jnp.int32),
            ]
        )
        batch_idx_i32 = batch_idx.astype(jnp.int32)
        forward_kwargs = {
            "pbc_bool": pbc_bool,
            "batch_idx_i32": batch_idx_i32,
            "cells_per_dimension": cells_per_dimension,
            "atom_periodic_shifts": atom_periodic_shifts,
            "atom_to_cell_mapping": atom_to_cell_mapping,
            "atoms_per_cell_count": atoms_per_cell_count,
            "cell_atom_start_indices": cell_atom_start_indices,
            "cell_atom_list": cell_atom_list,
            "cell_offsets": cell_offsets,
            "neighbor_search_radius": neighbor_search_radius,
            "neighbor_matrix": nm,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
            "num_neighbors": nn,
            "neighbor_vectors": neighbor_vectors,
            "neighbor_distances": neighbor_distances,
            "cutoff": cutoff,
        }
        distances_out, vectors_out, nm_out, nn_out, shifts_out = _route_pair_outputs(
            positions_for_grad,
            cell_for_grad,
            _batch_cell_list_pair_outputs_forward,
            forward_kwargs,
        )
        if return_neighbor_list:
            nl, nptr, nl_shifts = get_neighbor_list_from_neighbor_matrix(
                nm_out,
                num_neighbors=nn_out,
                neighbor_shift_matrix=shifts_out,
                fill_value=total_atoms,
            )
            base = (nl, nptr, nl_shifts)
        else:
            base = (nm_out, nn_out, shifts_out)
        if return_distances and return_vectors:
            return (*base, distances_out, vectors_out)
        if return_distances:
            return (*base, distances_out)
        return (*base, vectors_out)

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
