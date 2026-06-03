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


"""Warp-facing launchers for cell-list neighbor lists."""

from __future__ import annotations

import warp as wp

from nvalchemiops.neighbors.cell_list.dispatch import (
    PAIR_CENTRIC_MAX_LINEAR_LAUNCH,
    compute_batch_pair_centric_n_outer,
    is_pair_centric_launch_safe,
    is_pair_centric_parallelism_sufficient,
    pair_centric_launch_size,
    select_batch_cell_list_strategy,
    select_cell_list_strategy,
)
from nvalchemiops.neighbors.cell_list.kernels import (
    _build_cell_to_system_map,
    _compute_cells_per_system,
    _fill_target_row_lookup,
    _reset_target_row_lookup,
    get_build_cell_list_kernel,
    get_cell_list_cells_per_system_kernel,
    get_cell_list_gather_kernel,
    get_query_cell_list_kernel,
)
from nvalchemiops.neighbors.neighbor_utils import (
    dtype_info,
    get_gather_positions_and_shifts_kernel,
    selective_zero_num_neighbors,
)
from nvalchemiops.neighbors.neighbor_utils import empty_sentinel as _empty_sentinel
from nvalchemiops.neighbors.output_args import (
    _prepare_pair_output_args,
)

__all__ = [
    "PAIR_CENTRIC_MAX_LINEAR_LAUNCH",
    "batch_build_cell_list",
    "batch_query_cell_list",
    "batch_query_cell_list_pair_centric_sorted",
    "build_cell_list",
    "compute_batch_pair_centric_n_outer",
    "is_pair_centric_launch_safe",
    "is_pair_centric_parallelism_sufficient",
    "pair_centric_launch_size",
    "get_cell_list_cells_per_system_kernel",
    "get_build_cell_list_kernel",
    "get_cell_list_gather_kernel",
    "get_query_cell_list_kernel",
    "query_cell_list",
    "query_cell_list_atom_centric_sorted",
    "query_cell_list_pair_centric_sorted",
    "select_batch_cell_list_strategy",
    "select_cell_list_strategy",
]

# Inert ``max_radius`` placeholder for the single-system pair-centric launch
# (only the batched kernel reads ``max_radius``).  Hoisted to module scope so
# the launch reuses one host-side value instead of rebuilding it per call.
_ZERO_RADIUS = wp.vec3i(0, 0, 0)


def _pair_centric_unsafe_message(
    total_cells: int,
    n_outer: int,
    block_dim: int,
) -> str:
    """Return a user-facing message for an unsafe pair-centric launch."""
    launch_size = pair_centric_launch_size(total_cells, n_outer, block_dim)
    return (
        "strategy='pair_centric' would require "
        f"{launch_size} logical threads "
        f"({int(total_cells)} cells * {int(n_outer) + 1} offsets * "
        f"{int(block_dim)} threads), exceeding the safe linear launch limit "
        f"of {PAIR_CENTRIC_MAX_LINEAR_LAUNCH}."
    )


def _raise_unsafe_pair_centric_launch(
    total_cells: int,
    n_outer: int,
    block_dim: int,
) -> None:
    """Raise for raw pair-centric launchers that cannot fall back."""
    raise ValueError(_pair_centric_unsafe_message(total_cells, n_outer, block_dim))


def _prepare_target_row_lookup(
    target_indices: wp.array | None,
    target_row_lookup: wp.array | None,
    total_atoms: int,
    device: str,
) -> wp.array:
    """Return atom-id to compact target-row lookup for pair-centric kernels.

    When ``target_indices`` is provided and ``target_row_lookup`` is omitted,
    this helper allocates a transient ``(total_atoms,)`` int32 scratch array.
    CUDA graph/capture callers should pass caller-owned scratch explicitly.
    """
    if target_indices is None:
        return _empty_sentinel(1, wp.int32, device)
    lookup = target_row_lookup
    if lookup is None:
        lookup = wp.empty((int(total_atoms),), dtype=wp.int32, device=device)
    wp.launch(
        _reset_target_row_lookup,
        dim=int(total_atoms),
        inputs=[lookup],
        device=device,
    )
    wp.launch(
        _fill_target_row_lookup,
        dim=int(target_indices.shape[0]),
        inputs=[target_indices, lookup],
        device=device,
    )
    return lookup


def build_cell_list(
    positions: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    cells_per_dimension: wp.array,
    atom_periodic_shifts: wp.array,
    atom_to_cell_mapping: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    wp_dtype: type,
    device: str,
    min_cells_per_dimension: int = 4,
) -> None:
    """Core warp launcher for building spatial cell list.

    Constructs a spatial decomposition data structure for efficient neighbor searching
    using pure warp operations. This function launches warp kernels to organize atoms
    into spatial cells.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix defining the simulation box.
    pbc : wp.array, shape (3,), dtype=wp.bool
        Periodic boundary condition flags for x, y, z directions.
    cutoff : float
        Maximum distance for neighbor search.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        OUTPUT: Number of cells created in x, y, z directions.
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: Periodic boundary crossings for each atom.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: 3D cell coordinates assigned to each atom.
    atoms_per_cell_count : wp.array, shape (max_total_cells,), dtype=wp.int32
        OUTPUT: Number of atoms in each cell. Must be zeroed by caller before first use.
    cell_atom_start_indices : wp.array, shape (max_total_cells,), dtype=wp.int32
        OUTPUT: Array for cell start offsets. Caller provides pre-allocated
        and zeroed array.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Flattened list of atom indices organized by cell.
    wp_dtype : type
        Warp dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    min_cells_per_dimension : int, default 4
        Lower bound for the per-axis cell count. Pass 1 for the legacy grid rule.

    Notes
    -----
    - This is a low-level Warp interface. The caller must ensure
      ``atoms_per_cell_count`` is zeroed before calling.
    - This function handles the cumsum internally using ``wp.utils.array_scan``.
    - For framework bindings, use the torch/jax wrappers instead.

    See Also
    --------
    query_cell_list : Query cell list to build neighbor matrix (call after this)
    """
    total_atoms = positions.shape[0]
    max_total_cells = atoms_per_cell_count.shape[0]
    wp_cutoff = wp_dtype(cutoff)

    wp.launch(
        get_build_cell_list_kernel(
            "construct_bin_size",
            wp_dtype,
            min_cells_per_dimension=int(min_cells_per_dimension),
        ),
        dim=1,
        device=device,
        inputs=(
            cell,
            pbc,
            _empty_sentinel(2, wp.bool, device),
            cells_per_dimension,
            _empty_sentinel(1, wp.vec3i, device),
            wp_cutoff,
            max_total_cells,
        ),
    )

    wp.launch(
        get_build_cell_list_kernel("count_atoms", wp_dtype),
        dim=total_atoms,
        inputs=[
            positions,
            cell,
            pbc,
            _empty_sentinel(2, wp.bool, device),
            _empty_sentinel(1, wp.int32, device),
            cells_per_dimension,
            _empty_sentinel(1, wp.vec3i, device),
            _empty_sentinel(1, wp.int32, device),
            atoms_per_cell_count,
            atom_periodic_shifts,
        ],
        device=device,
    )

    wp.utils.array_scan(atoms_per_cell_count, cell_atom_start_indices, inclusive=False)

    atoms_per_cell_count.zero_()

    wp.launch(
        get_build_cell_list_kernel("bin_atoms", wp_dtype),
        dim=total_atoms,
        inputs=[
            positions,
            cell,
            pbc,
            _empty_sentinel(2, wp.bool, device),
            _empty_sentinel(1, wp.int32, device),
            cells_per_dimension,
            _empty_sentinel(1, wp.vec3i, device),
            _empty_sentinel(1, wp.int32, device),
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ],
        device=device,
    )


def query_cell_list_atom_centric_sorted(
    sorted_positions: wp.array,
    sorted_atom_periodic_shifts: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    cells_per_dimension: wp.array,
    neighbor_search_radius: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    atom_to_cell_mapping: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    rebuild_flags: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = True,
    *,
    positions: wp.array | None = None,
    atom_periodic_shifts: wp.array | None = None,
    target_indices: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
    atom_centric_path: str = "sorted",
    selective: bool = True,
) -> None:
    """Atom-centric query with sorted-position reads.

    The caller pre-gathers ``sorted_positions`` and
    ``sorted_atom_periodic_shifts`` via ``gather_fused``.  Output writes
    target the public ``neighbor_matrix[atom_idx, n]`` indexed by the
    original atom index, so downstream code is unaware of the reordering.

    ``half_fill=True`` iterates the half-shell + ``j > i`` self filter;
    ``half_fill=False`` iterates the full shell + ``j != i`` self filter.
    The local-counter (per-thread register) is valid in both modes
    because each thread is the unique writer to its own row.

    ``rebuild_flags`` is a caller-allocated 1-element ``wp.bool`` array for
    selective rebuilds.  Non-selective callers pass a sentinel array and set
    ``selective=False`` so the kernel specialization does not read it.

    Parameters
    ----------
    sorted_positions : wp.array, shape (num_atoms,), dtype=wp.vec3*
        Cell-contiguous reordered positions.
    sorted_atom_periodic_shifts : wp.array, shape (num_atoms,), dtype=wp.vec3i
        Cell-contiguous reordered shifts.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Simulation cell matrix.
    pbc : wp.array, shape (3,), dtype=wp.bool
        Per-axis periodic-boundary flags.
    cutoff : float
        Neighbor search cutoff.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        Cell grid shape.
    neighbor_search_radius : wp.array, shape (3,), dtype=wp.int32
        Per-axis search radius.
    atoms_per_cell_count : wp.array, shape (num_cells,), dtype=wp.int32
        Atoms per cell.
    cell_atom_start_indices : wp.array, shape (num_cells,), dtype=wp.int32
        CSR cell-start indices.
    cell_atom_list : wp.array, shape (num_atoms,), dtype=wp.int32
        CSR atom indices.
    atom_to_cell_mapping : wp.array, shape (num_atoms,), dtype=wp.vec3i
        Atom -> cell coordinate map.
    neighbor_matrix : wp.array, shape (num_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: per-atom neighbor indices.
    neighbor_matrix_shifts : wp.array, shape (num_atoms, max_neighbors), dtype=wp.vec3i
        OUTPUT: per-pair lattice shift.
    num_neighbors : wp.array, shape (num_atoms,), dtype=wp.int32
        OUTPUT: per-atom neighbor count.
    rebuild_flags : wp.array, shape (1,), dtype=wp.bool
        Selective-rebuild flag.
    wp_dtype : type
        Warp scalar dtype.
    device : str
        Warp device string.
    half_fill : bool, default ``True``
        Half-shell + ``j > i`` filter when True.
    positions, atom_periodic_shifts : wp.array, optional
        Unsorted positions / shifts.  Required when ``target_indices`` is
        provided so the central-atom read does not need to invert the
        sorted layout.
    target_indices : wp.array, shape (num_targets,), dtype=wp.int32, optional
        Restrict source atoms to a subset.  Switches the kernel to a
        per-row iteration over ``target_indices`` and consults ``positions``
        / ``atom_periodic_shifts`` for central-atom reads.
    return_vectors, return_distances : bool, default ``False``
        Write per-pair displacement vectors / distances to
        ``neighbor_vectors`` / ``neighbor_distances``.
    pair_fn : callable, optional
        Module-scope ``@wp.func`` of signature
        ``(r_ij, distance, pair_params, i, j) -> (energy, force)``.  When
        provided, the launcher requires ``pair_params``,
        ``pair_energies`` and ``pair_forces``.
    pair_params : wp.array, shape (num_atoms, num_parameters), optional
        Per-atom pair-function parameters.  Dtype must match positions.
    neighbor_vectors : wp.array, shape (num_atoms, max_neighbors), dtype=wp.vec3*, optional
        OUTPUT buffer for per-pair displacements (only with ``return_vectors``).
    neighbor_distances : wp.array, shape (num_atoms, max_neighbors), dtype=wp_dtype, optional
        OUTPUT buffer for per-pair distances (only with ``return_distances``).
    pair_energies : wp.array, shape (num_atoms, max_neighbors), dtype=wp_dtype, optional
        OUTPUT buffer for per-pair energies (only with ``pair_fn``).
    pair_forces : wp.array, shape (num_atoms, max_neighbors), dtype=wp.vec3*, optional
        OUTPUT buffer for per-pair forces (only with ``pair_fn``).
    selective : bool, default ``True``
        Whether ``rebuild_flags`` controls kernel execution.

    Notes
    -----
    The same atom-centric kernel factory handles the default path, compact
    ``target_indices`` rows, optional vector/distance buffers, and ``pair_fn``
    slot outputs.
    """

    partial = target_indices is not None
    if atom_centric_path not in {"direct", "sorted"}:
        raise ValueError(
            f"atom_centric_path must be 'direct' | 'sorted', got {atom_centric_path!r}",
        )
    if (partial or atom_centric_path == "direct") and (
        positions is None or atom_periodic_shifts is None
    ):
        raise ValueError(
            "positions and atom_periodic_shifts are required for the "
            f"atom-centric {atom_centric_path!r} path",
        )
    positions_arg = positions if positions is not None else sorted_positions
    atom_periodic_shifts_arg = (
        atom_periodic_shifts
        if atom_periodic_shifts is not None
        else sorted_atom_periodic_shifts
    )
    target_indices_arg = (
        target_indices
        if target_indices is not None
        else _empty_sentinel(1, wp.int32, device)
    )
    (
        neighbor_vectors_arg,
        neighbor_distances_arg,
        pair_params_arg,
        pair_energies_arg,
        pair_forces_arg,
    ) = _prepare_pair_output_args(
        wp_dtype,
        device,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )
    kernel = get_query_cell_list_kernel(
        wp_dtype,
        strategy="atom_centric",
        batched=False,
        selective=bool(selective),
        partial=partial,
        half_fill=bool(half_fill),
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        atom_centric_path=atom_centric_path,
    )
    if partial:
        dim = int(target_indices.shape[0])
    elif atom_centric_path == "direct":
        dim = int(positions_arg.shape[0])
    else:
        dim = int(sorted_positions.shape[0])
    wp.launch(
        kernel,
        dim=dim,
        inputs=[
            positions_arg,
            atom_periodic_shifts_arg,
            sorted_positions,
            sorted_atom_periodic_shifts,
            cell,
            pbc,
            _empty_sentinel(2, wp.bool, device),
            _empty_sentinel(1, wp.int32, device),
            wp_dtype(cutoff),
            cells_per_dimension,
            _empty_sentinel(1, wp.vec3i, device),
            neighbor_search_radius,
            _empty_sentinel(1, wp.vec3i, device),
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            _empty_sentinel(1, wp.int32, device),
            target_indices_arg,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            neighbor_vectors_arg,
            neighbor_distances_arg,
            pair_params_arg,
            pair_energies_arg,
            pair_forces_arg,
            rebuild_flags,
        ],
        device=device,
    )


def query_cell_list_pair_centric_sorted(
    sorted_positions: wp.array,
    sorted_atom_periodic_shifts: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    cells_per_dimension: wp.array,
    neighbor_search_radius: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    rebuild_flags: wp.array,
    wp_dtype: type,
    device: str,
    n_outer: int,
    block_dim: int = 64,
    half_fill: bool = True,
    *,
    target_indices: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
    target_row_lookup: wp.array | None = None,
    selective: bool = True,
) -> None:
    """Pair-centric query: one block per (source cell, offset).

    One factory-generated kernel handles both same-cell pairs and outer
    cell-cell pairs.  ``offset_idx == 0`` is the same-cell case; nonzero
    offsets are decoded on-the-fly from ``neighbor_search_radius`` and
    ``half_fill`` via :func:`_decode_shift_index` /
    :func:`_decode_full_shift_index`.

    The kernel uses per-emit ``wp.atomic_add(num_neighbors, atom_i, 1)``
    because multiple blocks (different outer offsets) may target the
    same source atom.

    Pair-set output is identical to the atom-centric kernel; per-row
    ordering inside ``neighbor_matrix`` is non-deterministic across
    builds.

    Parameters
    ----------
    sorted_positions, sorted_atom_periodic_shifts : wp.array
        Pre-gathered sorted-by-cell views (output of ``gather_fused``).
    neighbor_search_radius : wp.array(dtype=wp.int32), shape (3,)
        Per-axis radius ``(Rx, Ry, Rz)``.  The kernel decodes each block's
        outer-offset index into a ``(dx, dy, dz)`` vector on-the-fly via
        the shared shift-index decoders.
    n_outer : int
        Number of non-self outer cell offsets - must match
        ``compute_batch_pair_centric_n_outer(R, half_fill)``.  Caller computes
        this once per geometry change (the only host-side dependency on
        the per-axis radius).
    half_fill : bool, default True
        Half-shell + ``j > i`` self filter when True; full shell + ``j != i``
        self filter when False.  Same semantics as the atom-centric kernel.
    block_dim : int, default 64
        Threads per CUDA block.  Should bracket typical
        atoms-per-cell counts; cells with more atoms see lanes
        stride-loop.
    rebuild_flags : wp.array(dtype=wp.bool), shape (1,)
        Caller-allocated GPU-resident flag.  ``False`` makes every block
        return immediately and outputs are preserved.  ``True`` runs the
        full build - caller must zero ``num_neighbors`` first so the
        per-emit atomic_adds start from 0.  Non-selective callers pass a
        sentinel array and set ``selective=False``.

    Notes
    -----
    The pair-centric kernel factory supports compact ``target_indices`` rows,
    optional vector/distance buffers, and ``pair_fn`` slot outputs directly.
    It remains CUDA-only because it maps one logical block to each
    ``(source_cell, offset)`` pair.
    """

    if "cpu" in str(device).lower():
        raise ValueError(
            "strategy='pair_centric' is not supported on CPU "
            "(kernels use CUDA block scheduling).  Pass 'atom_centric' instead.",
        )
    total_atoms = int(sorted_positions.shape[0])
    total_cells = int(atoms_per_cell_count.shape[0])
    block_dim_int = int(block_dim)
    n_offsets_int = int(n_outer) + 1
    if not is_pair_centric_launch_safe(total_cells, int(n_outer), block_dim_int):
        _raise_unsafe_pair_centric_launch(total_cells, int(n_outer), block_dim_int)
    partial = target_indices is not None
    target_row_lookup_arg = _prepare_target_row_lookup(
        target_indices,
        target_row_lookup,
        total_atoms,
        device,
    )
    (
        neighbor_vectors_arg,
        neighbor_distances_arg,
        pair_params_arg,
        pair_energies_arg,
        pair_forces_arg,
    ) = _prepare_pair_output_args(
        wp_dtype,
        device,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )
    kernel = get_query_cell_list_kernel(
        wp_dtype,
        strategy="pair_centric",
        batched=False,
        selective=bool(selective),
        partial=partial,
        half_fill=bool(half_fill),
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
    )
    wp.launch(
        kernel,
        dim=total_cells * n_offsets_int * block_dim_int,
        block_dim=block_dim_int,
        inputs=[
            sorted_positions,
            sorted_atom_periodic_shifts,
            cell,
            pbc,
            _empty_sentinel(2, wp.bool, device),
            wp_dtype(cutoff),
            cells_per_dimension,
            _empty_sentinel(1, wp.vec3i, device),
            neighbor_search_radius,
            _empty_sentinel(1, wp.vec3i, device),
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            _empty_sentinel(1, wp.int32, device),
            _empty_sentinel(1, wp.int32, device),
            target_row_lookup_arg,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            neighbor_vectors_arg,
            neighbor_distances_arg,
            pair_params_arg,
            pair_energies_arg,
            pair_forces_arg,
            block_dim_int,
            total_cells,
            n_offsets_int,
            _ZERO_RADIUS,
            rebuild_flags,
        ],
        device=device,
    )


def query_cell_list(
    positions: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    cells_per_dimension: wp.array,
    neighbor_search_radius: wp.array,
    atom_periodic_shifts: wp.array,
    atom_to_cell_mapping: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    *,
    sorted_positions: wp.array | None = None,
    sorted_atom_periodic_shifts: wp.array | None = None,
    strategy: str = "atom_centric",
    atom_centric_path: str = "auto",
    n_outer: int | None = None,
    target_indices: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
    target_row_lookup: wp.array | None = None,
) -> None:
    """Core warp launcher for querying spatial cell list to build neighbor matrix.

    Uses pre-built cell list data structures to efficiently find all atom pairs
    within the specified cutoff distance using pure warp operations.  Output
    arrays (``neighbor_matrix``, ``neighbor_matrix_shifts``, ``num_neighbors``)
    are caller-allocated.  The per-cell-contiguous gather scratch
    (``sorted_positions``, ``sorted_atom_periodic_shifts``) and the
    1-element ``rebuild_flags`` are optional: when omitted, this launcher
    uses the non-selective kernel specialization.  Graph/capture callers should
    pass caller-owned scratch explicitly to keep allocations out of the
    captured region.  If ``target_indices`` is used with pair-centric mode
    and ``target_row_lookup`` is omitted, this launcher allocates a transient
    lookup scratch array.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix for periodic boundary coordinate shifts.
    pbc : wp.array, shape (3,), dtype=wp.bool
        Periodic boundary condition flags.
    cutoff : float
        Maximum distance for considering atoms as neighbors.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        Number of cells in x, y, z directions from build_cell_list.
    neighbor_search_radius : wp.array, shape (3,), dtype=wp.int32
        Radius of neighboring cells to search in each dimension.
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        Periodic boundary crossings for each atom from build_cell_list.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        3D cell coordinates for each atom from build_cell_list.
    atoms_per_cell_count : wp.array, shape (max_total_cells,), dtype=wp.int32
        Number of atoms in each cell from build_cell_list.
    cell_atom_start_indices : wp.array, shape (max_total_cells,), dtype=wp.int32
        Starting index in cell_atom_list for each cell from build_cell_list.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        Flattened list of atom indices organized by cell from build_cell_list.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors, 3), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT (atomic): per-atom neighbor counts.  Accumulated via
        ``wp.atomic_add``, so non-selective callers must zero it before the
        call (selective callers zero it via ``rebuild_flags``).
    wp_dtype : type
        Warp dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    sorted_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*, optional
        Per-cell-contiguous gather scratch.  ``gather_fused`` writes into
        it each call.  Allocated transiently when omitted; graph/capture
        callers should pass caller-owned scratch.
    sorted_atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i, optional
        Per-cell-contiguous gather scratch.  ``gather_fused`` writes into
        it each call.  Allocated transiently when omitted; graph/capture
        callers should pass caller-owned scratch.
    rebuild_flags : wp.array, shape (1,), dtype=wp.bool, optional
        1-element flag.  ``False`` makes the kernel return immediately.
        When omitted, this launcher uses the non-selective kernel
        specialization and does not read a rebuild flag.
    half_fill : bool, default=False
        If True, only store half of the neighbor relationships (i < j).
    strategy : {"atom_centric", "pair_centric"}, default "atom_centric"
        Selects which of the two sorted fast-path kernels to launch.
        Both produce identical pair sets for either ``half_fill`` value;
        per-row ordering inside ``neighbor_matrix`` differs.

        ``"pair_centric"`` requires ``n_outer`` (the host-side count of
        non-self outer cell offsets at the per-axis radius - caller
        precomputes via :func:`compute_batch_pair_centric_n_outer`).  Auto-
        strategy selection (sync-free) lives at the torch-wrapper layer where the
        sync is already paid; direct-warp callers pick explicitly.
        Pair-centric is CUDA-only - CPU callers must use atom-centric.
    n_outer : int, optional
        Required when ``strategy="pair_centric"``.  Number of non-self
        outer cell offsets at the per-axis search radius - see
        :func:`compute_batch_pair_centric_n_outer` for the closed form.
    target_indices : wp.array, shape (num_targets,), dtype=wp.int32, optional
        Restrict central rows to a subset of atom indices.  Output rows are
        compact and follow ``target_indices`` order for both strategies.
    return_vectors, return_distances : bool, default ``False``
        Write per-pair displacement vectors / distances into
        ``neighbor_vectors`` / ``neighbor_distances``.
    pair_fn : callable, optional
        Module-scope ``@wp.func`` of signature
        ``(r_ij, distance, pair_params, i, j) -> (energy, force)``.
    pair_params : wp.array, shape (num_atoms, num_parameters), optional
        Per-atom pair-function parameters; required with ``pair_fn``.
    neighbor_vectors, neighbor_distances : wp.array, optional
        OUTPUT buffers for per-pair displacements / distances.
    pair_energies, pair_forces : wp.array, optional
        OUTPUT buffers for per-pair energies / forces; required with
        ``pair_fn``.

    Notes
    -----
    - Output and scratch arrays are caller-owned except for the optional
      transient ``target_row_lookup`` allocation described above.
      ``num_neighbors`` must be zeroed before each pair-centric call
      (atomic_add semantics).  Shifts output uses the always-write
      contract (no prefill required).
    - Both strategies support compact ``target_indices`` rows, optional
      vector/distance buffers, and ``pair_fn`` slot outputs.

    See Also
    --------
    build_cell_list                     : Build cell list (call before this)
    query_cell_list_atom_centric_sorted  : Atom-centric kernel (both half_fill modes)
    query_cell_list_pair_centric_sorted : Pair-centric alternative
    select_cell_list_strategy          : Sync-free (N, cutoff) auto-selection rule
    compute_batch_pair_centric_n_outer : Closed-form for ``n_outer``
    """

    total_atoms = positions.shape[0]

    selective = rebuild_flags is not None
    rebuild_flags_arg = (
        rebuild_flags
        if rebuild_flags is not None
        else _empty_sentinel(1, wp.bool, device)
    )

    cpu_only = "cpu" in str(device).lower()
    if strategy == "atom_centric":
        chosen = "atom_centric"
    elif strategy == "pair_centric":
        if cpu_only:
            raise ValueError(
                "strategy='pair_centric' is not supported on CPU "
                "(kernels use CUDA block scheduling).  Pass "
                "'atom_centric' instead.",
            )
        if n_outer is None:
            raise ValueError(
                "strategy='pair_centric' requires n_outer.  Compute via "
                "compute_batch_pair_centric_n_outer((Rx, Ry, Rz), half_fill).",
            )
        block_dim = 64
        total_cells = int(atoms_per_cell_count.shape[0])
        if not is_pair_centric_launch_safe(total_cells, int(n_outer), block_dim):
            _raise_unsafe_pair_centric_launch(total_cells, int(n_outer), block_dim)
        chosen = "pair_centric"
    else:
        raise ValueError(
            f"strategy must be 'atom_centric' | 'pair_centric', got {strategy!r}",
        )

    if atom_centric_path == "auto":
        atom_centric_path = "direct"
    elif atom_centric_path not in {"direct", "sorted"}:
        raise ValueError(
            "atom_centric_path must be 'auto' | 'direct' | 'sorted', "
            f"got {atom_centric_path!r}",
        )
    needs_sorted = chosen == "pair_centric" or atom_centric_path == "sorted"
    _vec_dtype, _ = dtype_info(wp_dtype)
    if needs_sorted:
        if (sorted_positions is None) != (sorted_atom_periodic_shifts is None):
            raise ValueError(
                "Pass both sorted_positions and sorted_atom_periodic_shifts, "
                "or neither - got a mixed state.",
            )
        if sorted_positions is None:
            sorted_positions = wp.empty(
                int(total_atoms), dtype=_vec_dtype, device=device
            )
        if sorted_atom_periodic_shifts is None:
            sorted_atom_periodic_shifts = wp.empty(
                int(total_atoms), dtype=wp.vec3i, device=device
            )
    else:
        sorted_positions = _empty_sentinel(1, _vec_dtype, device)
        sorted_atom_periodic_shifts = _empty_sentinel(1, wp.vec3i, device)

    if needs_sorted:
        wp.launch(
            get_gather_positions_and_shifts_kernel(wp_dtype),
            dim=total_atoms,
            inputs=[
                positions,
                atom_periodic_shifts,
                cell_atom_list,
                sorted_positions,
                sorted_atom_periodic_shifts,
            ],
            device=device,
        )
    if chosen == "pair_centric":
        query_cell_list_pair_centric_sorted(
            sorted_positions=sorted_positions,
            sorted_atom_periodic_shifts=sorted_atom_periodic_shifts,
            cell=cell,
            pbc=pbc,
            cutoff=float(cutoff),
            cells_per_dimension=cells_per_dimension,
            neighbor_search_radius=neighbor_search_radius,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            rebuild_flags=rebuild_flags_arg,
            wp_dtype=wp_dtype,
            device=device,
            n_outer=int(n_outer),
            block_dim=64,
            half_fill=bool(half_fill),
            target_indices=target_indices,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
            pair_params=pair_params,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
            target_row_lookup=target_row_lookup,
            selective=selective,
        )
    else:
        query_cell_list_atom_centric_sorted(
            sorted_positions=sorted_positions,
            sorted_atom_periodic_shifts=sorted_atom_periodic_shifts,
            cell=cell,
            pbc=pbc,
            cutoff=float(cutoff),
            cells_per_dimension=cells_per_dimension,
            neighbor_search_radius=neighbor_search_radius,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            atom_to_cell_mapping=atom_to_cell_mapping,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            rebuild_flags=rebuild_flags_arg,
            wp_dtype=wp_dtype,
            device=device,
            half_fill=bool(half_fill),
            positions=positions,
            atom_periodic_shifts=atom_periodic_shifts,
            target_indices=target_indices,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
            pair_params=pair_params,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
            atom_centric_path=atom_centric_path,
            selective=selective,
        )


# ----------------------------------------------------------------------------
# Public launchers - batched path
# ----------------------------------------------------------------------------


def batch_build_cell_list(
    positions: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    batch_idx: wp.array,
    cells_per_dimension: wp.array,
    cell_offsets: wp.array,
    cells_per_system: wp.array,
    atom_periodic_shifts: wp.array,
    atom_to_cell_mapping: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    wp_dtype: type,
    device: str,
    min_cells_per_dimension: int = 4,
) -> None:
    """Core warp launcher for building batch spatial cell lists.

    Constructs spatial decomposition data structures for multiple systems using
    pure warp operations. This function launches warp kernels to organize atoms
    into spatial cells across all systems in the batch.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in the batch.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrices for each system in the batch.
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool
        Periodic boundary condition flags for each system and dimension.
    cutoff : float
        Neighbor search cutoff distance.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        OUTPUT: Number of cells in x, y, z directions for each system.
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        OUTPUT: Starting index in global cell arrays for each system.
        Computed internally via exclusive scan of cells_per_dimension products.
    cells_per_system : wp.array, shape (num_systems,), dtype=wp.int32
        SCRATCH: Temporary buffer for total cells per system.
        Used as input to exclusive scan for computing cell_offsets.
        Must be pre-allocated by caller.
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: Periodic boundary crossings for each atom.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: 3D cell coordinates assigned to each atom.
    atoms_per_cell_count : wp.array, shape (max_total_cells,), dtype=wp.int32
        OUTPUT: Number of atoms in each cell. Must be zeroed by caller before first use.
    cell_atom_start_indices : wp.array, shape (max_total_cells,), dtype=wp.int32
        OUTPUT: Starting index in cell_atom_list for each cell's atoms.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Flattened list of atom indices organized by cell.
    wp_dtype : type
        Warp dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    min_cells_per_dimension : int, default 4
        Lower bound for the per-axis cell count. Pass 1 for the legacy grid rule.

    Notes
    -----
    - This is a low-level warp interface. Caller must ensure atoms_per_cell_count is zeroed.
    - cell_offsets is computed internally after cells_per_dimension is determined.
    - This function handles the internal cumsum for cell_atom_start_indices using wp.utils.array_scan.
    - For framework bindings, use the torch/jax wrappers instead.

    See Also
    --------
    batch_query_cell_list : Query cell list to build neighbor matrix (call after this)
    """
    total_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    max_total_cells = atoms_per_cell_count.shape[0]
    wp_cutoff = wp_dtype(cutoff)

    wp.launch(
        get_build_cell_list_kernel(
            "construct_bin_size",
            wp_dtype,
            batched=True,
            min_cells_per_dimension=int(min_cells_per_dimension),
        ),
        dim=num_systems,
        device=device,
        inputs=(
            cell,
            _empty_sentinel(1, wp.bool, device),
            pbc,
            _empty_sentinel(1, wp.int32, device),
            cells_per_dimension,
            wp_cutoff,
            max_total_cells,
        ),
    )

    wp.launch(
        _compute_cells_per_system,
        dim=num_systems,
        device=device,
        inputs=(cells_per_dimension, cells_per_system),
    )
    wp.utils.array_scan(cells_per_system, cell_offsets, inclusive=False)

    wp.launch(
        get_build_cell_list_kernel("count_atoms", wp_dtype, batched=True),
        dim=total_atoms,
        inputs=(
            positions,
            cell,
            _empty_sentinel(1, wp.bool, device),
            pbc,
            batch_idx,
            _empty_sentinel(1, wp.int32, device),
            cells_per_dimension,
            cell_offsets,
            atoms_per_cell_count,
            atom_periodic_shifts,
        ),
        device=device,
    )

    wp.utils.array_scan(atoms_per_cell_count, cell_atom_start_indices, inclusive=False)

    atoms_per_cell_count.zero_()

    wp.launch(
        get_build_cell_list_kernel("bin_atoms", wp_dtype, batched=True),
        dim=total_atoms,
        inputs=(
            positions,
            cell,
            _empty_sentinel(1, wp.bool, device),
            pbc,
            batch_idx,
            _empty_sentinel(1, wp.int32, device),
            cells_per_dimension,
            cell_offsets,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ),
        device=device,
    )


def batch_query_cell_list_pair_centric_sorted(
    positions: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    cells_per_dimension: wp.array,
    neighbor_search_radius: wp.array,
    cell_offsets: wp.array,
    cells_per_system: wp.array,
    atom_periodic_shifts: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    sorted_positions: wp.array,
    sorted_atom_periodic_shifts: wp.array,
    cell_to_system: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    total_cells: int,
    n_outer: int,
    R_max: tuple[int, int, int],
    half_fill: bool = True,
    block_dim: int = 64,
    *,
    target_indices: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
    rebuild_flags: wp.array | None = None,
    target_row_lookup: wp.array | None = None,
) -> None:
    """Core warp launcher for the pair-centric batched cell-list query.

    Launches ``_batch_pair_centric_outer`` over ``total_cells x
    (n_outer + 1)`` blocks of ``block_dim`` threads.  ``offset_idx == 0``
    is the self-cell entry (within-cell pairs, with the appropriate filter
    for ``half_fill``); ``offset_idx > 0`` are the outer-shell offsets
    decoded on-the-fly from ``R_max``.  Per-system out-of-range offsets
    early-return.

    Per-emit ``wp.atomic_add(num_neighbors, atom_i, 1)`` because multiple
    blocks may write to the same atom's row; caller must pre-zero
    ``num_neighbors``.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrices for each system.
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool
        Periodic boundary condition flags.
    cutoff : float
        Neighbor search cutoff distance.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Cells per axis for each system.  From :func:`batch_build_cell_list`.
    neighbor_search_radius : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Per-system search radius.  From :func:`batch_build_cell_list`.
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        Starting global cell index per system.  From :func:`batch_build_cell_list`.
    cells_per_system : wp.array, shape (num_systems,), dtype=wp.int32
        Number of cells per system.  From :func:`batch_build_cell_list`.
    atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Per-atom PBC shift vectors.  From :func:`batch_build_cell_list`.
    atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
        Atoms per cell.  From :func:`batch_build_cell_list`.
    cell_atom_start_indices : wp.array, shape (total_cells,), dtype=wp.int32
        Cell start indices into ``cell_atom_list``.  From :func:`batch_build_cell_list`.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        Cell-organized atom indices.  From :func:`batch_build_cell_list`.
    sorted_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT (scratch): cell-contiguous reordered positions.  Caller
        allocates; ``_gather_positions_by_cell`` overwrites each call.
    sorted_atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
        OUTPUT (scratch): cell-contiguous reordered shifts.
    cell_to_system : wp.array, shape (total_cells,), dtype=wp.int32
        OUTPUT (scratch): global-cell -> system map.  Caller allocates;
        ``_build_cell_to_system_map`` overwrites each call.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: neighbor indices.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors, 3), dtype=wp.vec3i
        OUTPUT: per-pair PBC shift vectors.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT (atomic): per-atom neighbor counts.  Caller must zero
        before this call.
    wp_dtype : type
        Warp scalar dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string (e.g. ``"cuda:0"``).
    total_cells : int
        Sum of ``cells_per_system``.  Caller computes once per geometry change.
    n_outer : int
        Non-self outer cell offsets at ``R_max``.  From
        :func:`compute_batch_pair_centric_n_outer`.
    R_max : tuple[int, int, int]
        Cross-system maximum per-axis search radius.
    half_fill : bool, default True
        If True, half-shell + ``j > i`` self filter; if False, full shell
        + ``j != i`` self filter.
    block_dim : int, default 64
        Threads per CUDA block.

    Notes
    -----
    - Pair-centric kernels are CUDA-only.
    - The same pair-centric kernel factory supports compact ``target_indices``
      rows, optional vector/distance buffers, ``pair_fn`` slot outputs, and
      selective rebuild flags.

    See Also
    --------
    batch_query_cell_list : Atom-centric alternative
    compute_batch_pair_centric_n_outer : Computes ``n_outer``
    """

    if "cpu" in str(device).lower():
        raise ValueError(
            "strategy='pair_centric' is not supported on CPU "
            "(kernels use CUDA block scheduling).  Pass 'atom_centric' instead.",
        )
    natom = int(positions.shape[0])
    num_systems = int(cell.shape[0])
    block_dim_int = int(block_dim)
    n_offsets_int = int(n_outer) + 1
    if not is_pair_centric_launch_safe(int(total_cells), int(n_outer), block_dim_int):
        _raise_unsafe_pair_centric_launch(int(total_cells), int(n_outer), block_dim_int)
    R_max_vec = wp.vec3i(int(R_max[0]), int(R_max[1]), int(R_max[2]))
    partial = target_indices is not None
    rebuild_flags_arg = (
        rebuild_flags
        if rebuild_flags is not None
        else _empty_sentinel(1, wp.bool, device)
    )
    target_row_lookup_arg = _prepare_target_row_lookup(
        target_indices,
        target_row_lookup,
        natom,
        device,
    )
    (
        neighbor_vectors_arg,
        neighbor_distances_arg,
        pair_params_arg,
        pair_energies_arg,
        pair_forces_arg,
    ) = _prepare_pair_output_args(
        wp_dtype,
        device,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )

    wp.launch(
        _build_cell_to_system_map,
        dim=num_systems,
        inputs=[cell_offsets, cells_per_system, cell_to_system],
        device=device,
    )

    wp.launch(
        get_cell_list_gather_kernel(wp_dtype),
        dim=natom,
        inputs=[
            positions,
            atom_periodic_shifts,
            cell_atom_list,
            sorted_positions,
            sorted_atom_periodic_shifts,
        ],
        device=device,
    )

    kernel = get_query_cell_list_kernel(
        wp_dtype,
        strategy="pair_centric",
        batched=True,
        selective=rebuild_flags is not None,
        partial=partial,
        half_fill=bool(half_fill),
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
    )
    wp.launch(
        kernel,
        dim=total_cells * n_offsets_int * block_dim_int,
        block_dim=block_dim_int,
        inputs=[
            sorted_positions,
            sorted_atom_periodic_shifts,
            cell,
            _empty_sentinel(1, wp.bool, device),
            pbc,
            wp_dtype(cutoff),
            _empty_sentinel(1, wp.int32, device),
            cells_per_dimension,
            _empty_sentinel(1, wp.int32, device),
            neighbor_search_radius,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            cell_offsets,
            cell_to_system,
            target_row_lookup_arg,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            neighbor_vectors_arg,
            neighbor_distances_arg,
            pair_params_arg,
            pair_energies_arg,
            pair_forces_arg,
            block_dim_int,
            total_cells,
            n_offsets_int,
            R_max_vec,
            rebuild_flags_arg,
        ],
        device=device,
    )


def batch_query_cell_list(
    positions: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    batch_idx: wp.array,
    cells_per_dimension: wp.array,
    neighbor_search_radius: wp.array,
    cell_offsets: wp.array,
    atom_periodic_shifts: wp.array,
    atom_to_cell_mapping: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    *,
    sorted_positions: wp.array | None = None,
    sorted_atom_periodic_shifts: wp.array | None = None,
    strategy: str = "atom_centric",
    atom_centric_path: str = "auto",
    cells_per_system: wp.array | None = None,
    cell_to_system: wp.array | None = None,
    n_outer: int | None = None,
    R_max: tuple[int, int, int] | None = None,
    total_cells: int | None = None,
    target_indices: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
    target_row_lookup: wp.array | None = None,
) -> None:
    """Core warp launcher for querying batch spatial cell lists to build neighbor matrices.

    Uses pre-built cell list data structures to efficiently find all atom pairs
    within the specified cutoff distance for multiple systems using pure warp
    operations.  Mirrors the single-system :func:`query_cell_list` signature:
    ``strategy`` selects which of the two batch query kernels to launch;
    pair-centric requires additional caller-allocated scratch + metadata
    that the atom-centric path doesn't need.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in the batch.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrices for each system in the batch.
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool
        Periodic boundary condition flags for each system and dimension.
    cutoff : float
        Neighbor search cutoff distance.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Number of cells in x, y, z directions for each system.
    neighbor_search_radius : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Radius of neighboring cells to search for each system.
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        Starting index in global cell arrays for each system.
        Output from batch_build_cell_list.
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        Periodic boundary crossings for each atom. Output from batch_build_cell_list.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        3D cell coordinates for each atom. Output from batch_build_cell_list.
    atoms_per_cell_count : wp.array, shape (max_total_cells,), dtype=wp.int32
        Number of atoms in each cell. Output from batch_build_cell_list.
    cell_atom_start_indices : wp.array, shape (max_total_cells,), dtype=wp.int32
        Starting index in cell_atom_list for each cell. Output from batch_build_cell_list.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        Flattened list of atom indices organized by cell. Output from batch_build_cell_list.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors, 3), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT (atomic): per-atom neighbor counts.  Accumulated via
        ``wp.atomic_add``, so non-selective callers must zero it before the
        call (selective callers zero it via ``rebuild_flags``).
    wp_dtype : type
        Warp dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store half of the neighbor relationships (i < j).
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool, optional
        Per-system rebuild flags. If provided, only systems where
        ``rebuild_flags[i]`` is True are processed; others are skipped on
        the GPU without CPU sync.  When omitted, the non-selective kernel
        specialization is launched and the caller is responsible for
        pre-zeroing ``num_neighbors``.
    sorted_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*, optional
        Per-cell-contiguous gather scratch.  Allocated transiently when
        omitted; graph/capture callers should pass caller-owned scratch.
    sorted_atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i, optional
        Per-cell-contiguous gather scratch.  Allocated transiently when
        omitted; graph/capture callers should pass caller-owned scratch.
    target_indices : wp.array, shape (num_targets,), dtype=wp.int32, optional
        Restrict central rows to a subset of atom indices.  Output rows are
        compact and follow ``target_indices`` order for both strategies.
    return_vectors, return_distances : bool, default ``False``
        Write per-pair displacement vectors / distances into the
        ``neighbor_vectors`` / ``neighbor_distances`` kwargs.
    pair_fn : callable, optional
        Module-scope ``@wp.func`` of signature
        ``(r_ij, distance, pair_params, i, j) -> (energy, force)``.
    pair_params : wp.array, shape (num_atoms, num_parameters), optional
        Per-atom pair-function parameters; required with ``pair_fn``.
    neighbor_vectors, neighbor_distances : wp.array, optional
        OUTPUT buffers for per-pair displacements / distances.
    pair_energies, pair_forces : wp.array, optional
        OUTPUT buffers for per-pair energies / forces; required with
        ``pair_fn``.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - Both atom-centric and pair-centric paths consume the per-cell-contiguous
      ``sorted_positions`` / ``sorted_atom_periodic_shifts`` scratch.  The
      selected path fills that scratch before launching its neighbor kernel.
    - Both strategies support compact ``target_indices`` rows, optional
      vector/distance buffers, ``pair_fn`` slot outputs, and selective rebuild
      flags.

    See Also
    --------
    batch_build_cell_list : Build cell list data structures (call before this)
    batch_query_cell_list_pair_centric_sorted : Pair-centric alternative (CUDA only).
    """

    total_atoms = positions.shape[0]

    selective = rebuild_flags is not None
    rebuild_flags_arg = (
        rebuild_flags
        if rebuild_flags is not None
        else _empty_sentinel(1, wp.bool, device)
    )

    if strategy == "pair_centric":
        missing = [
            name
            for name, val in (
                ("cells_per_system", cells_per_system),
                ("cell_to_system", cell_to_system),
                ("n_outer", n_outer),
                ("R_max", R_max),
                ("total_cells", total_cells),
            )
            if val is None
        ]
        if missing:
            raise ValueError(
                f"strategy='pair_centric' requires the following kwargs to "
                f"be provided (caller-allocated scratch + sync-computed "
                f"metadata): {missing}.  See compute_batch_pair_centric_n_outer "
                f"for n_outer.",
            )
        block_dim = 64
        if not is_pair_centric_launch_safe(int(total_cells), int(n_outer), block_dim):
            _raise_unsafe_pair_centric_launch(int(total_cells), int(n_outer), block_dim)
    elif strategy != "atom_centric":
        raise ValueError(
            f"strategy must be 'atom_centric' | 'pair_centric', got {strategy!r}",
        )

    if atom_centric_path == "auto":
        atom_centric_path = "direct"
    elif atom_centric_path not in {"direct", "sorted"}:
        raise ValueError(
            "atom_centric_path must be 'auto' | 'direct' | 'sorted', "
            f"got {atom_centric_path!r}",
        )
    needs_sorted = strategy == "pair_centric" or atom_centric_path == "sorted"
    _vec_dtype, _ = dtype_info(wp_dtype)
    if needs_sorted:
        if (sorted_positions is None) != (sorted_atom_periodic_shifts is None):
            raise ValueError(
                "Pass both sorted_positions and sorted_atom_periodic_shifts, "
                "or neither - got a mixed state.",
            )
        if sorted_positions is None:
            sorted_positions = wp.empty(
                int(total_atoms), dtype=_vec_dtype, device=device
            )
        if sorted_atom_periodic_shifts is None:
            sorted_atom_periodic_shifts = wp.empty(
                int(total_atoms), dtype=wp.vec3i, device=device
            )
    else:
        sorted_positions = _empty_sentinel(1, _vec_dtype, device)
        sorted_atom_periodic_shifts = _empty_sentinel(1, wp.vec3i, device)

    if strategy == "pair_centric":
        batch_query_cell_list_pair_centric_sorted(
            positions=positions,
            cell=cell,
            pbc=pbc,
            cutoff=cutoff,
            cells_per_dimension=cells_per_dimension,
            neighbor_search_radius=neighbor_search_radius,
            cell_offsets=cell_offsets,
            cells_per_system=cells_per_system,
            atom_periodic_shifts=atom_periodic_shifts,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            sorted_positions=sorted_positions,
            sorted_atom_periodic_shifts=sorted_atom_periodic_shifts,
            cell_to_system=cell_to_system,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            wp_dtype=wp_dtype,
            device=device,
            total_cells=int(total_cells),
            n_outer=int(n_outer),
            R_max=R_max,
            half_fill=bool(half_fill),
            target_indices=target_indices,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
            pair_params=pair_params,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
            rebuild_flags=rebuild_flags,
            target_row_lookup=target_row_lookup,
        )
        return

    if needs_sorted:
        wp.launch(
            get_cell_list_gather_kernel(wp_dtype),
            dim=total_atoms,
            inputs=[
                positions,
                atom_periodic_shifts,
                cell_atom_list,
                sorted_positions,
                sorted_atom_periodic_shifts,
            ],
            device=device,
        )

    partial = target_indices is not None
    if selective and not partial:
        selective_zero_num_neighbors(num_neighbors, batch_idx, rebuild_flags, device)
    target_indices_arg = (
        target_indices
        if target_indices is not None
        else _empty_sentinel(1, wp.int32, device)
    )
    (
        neighbor_vectors_arg,
        neighbor_distances_arg,
        pair_params_arg,
        pair_energies_arg,
        pair_forces_arg,
    ) = _prepare_pair_output_args(
        wp_dtype,
        device,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )
    kernel = get_query_cell_list_kernel(
        wp_dtype,
        strategy="atom_centric",
        batched=True,
        selective=selective,
        partial=partial,
        half_fill=bool(half_fill),
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        atom_centric_path=atom_centric_path,
    )
    dim = int(target_indices.shape[0]) if partial else total_atoms
    wp.launch(
        kernel,
        dim=dim,
        inputs=[
            positions,
            atom_periodic_shifts,
            sorted_positions,
            sorted_atom_periodic_shifts,
            cell,
            _empty_sentinel(1, wp.bool, device),
            pbc,
            batch_idx,
            wp_dtype(cutoff),
            _empty_sentinel(1, wp.int32, device),
            cells_per_dimension,
            _empty_sentinel(1, wp.int32, device),
            neighbor_search_radius,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            cell_offsets,
            target_indices_arg,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            neighbor_vectors_arg,
            neighbor_distances_arg,
            pair_params_arg,
            pair_energies_arg,
            pair_forces_arg,
            rebuild_flags_arg,
        ],
        device=device,
    )
