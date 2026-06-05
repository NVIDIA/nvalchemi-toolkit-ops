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

"""PyTorch bindings for batched cell list neighbor construction.

The torch wrapper auto-selects between two batch query kernels:

* **atom-centric** (:mod:`nvalchemiops.neighbors.batch_cell_list`) -
  baseline 1 thread/atom; thread-local-counter optimisation.  Best at
  large total atoms with small per-system cutoff (cutoff=6 MLIP regime
  with many systems).
* **pair-centric** (:func:`nvalchemiops.neighbors.batch_cell_list.batch_query_cell_list_pair_centric_sorted`) -
  one block per ``(source_cell, outer_offset)``; per-emit
  ``atomic_add(num_neighbors, atom_i, 1)`` trades thread-local-counter
  for ``ncell x n_outer`` parallelism.  Best at moderate-to-large
  cutoff and / or few-large-systems batches.

Auto-select uses sync-free quantities (``total_atoms``, ``num_systems``,
``cutoff``); the ``total_cells`` Python int is already paid by
:func:`estimate_batch_cell_list_sizes` at allocation time.  Defaults
are calibrated empirically; overrides are exposed via environment
variables - see :func:`select_batch_cell_list_strategy`.
"""

from __future__ import annotations

import warnings

import torch
import warp as wp

from nvalchemiops.neighbors.cell_list import (
    PAIR_CENTRIC_MAX_LINEAR_LAUNCH,
    compute_batch_pair_centric_n_outer,
    get_build_cell_list_kernel,
    is_pair_centric_launch_safe,
    is_pair_centric_parallelism_sufficient,
    pair_centric_launch_size,
    select_batch_cell_list_strategy,
)
from nvalchemiops.neighbors.cell_list import (
    batch_build_cell_list as wp_batch_build_cell_list,
)
from nvalchemiops.neighbors.cell_list import (
    batch_query_cell_list as wp_batch_query_cell_list,
)
from nvalchemiops.neighbors.neighbor_utils import empty_sentinel, estimate_max_neighbors
from nvalchemiops.neighbors.neighbor_utils import (
    fill_neighbor_matrix_tail as wp_fill_neighbor_matrix_tail,
)
from nvalchemiops.neighbors.output_args import (
    _has_partial_or_pair_outputs,
)
from nvalchemiops.torch.neighbors._autograd import (
    _flatten_active_pairs,
    _NeighborForwardOutput,
    _route_pair_outputs,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
    coo_pack_pair_geometry,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = [
    "estimate_batch_cell_list_sizes",
    "batch_build_cell_list",
    "batch_query_cell_list",
    "batch_cell_list",
]


def _pair_centric_unsafe_message(
    total_cells: int,
    n_outer: int,
    block_dim: int = 64,
) -> str:
    """Return the unsafe pair-centric launch message."""
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
    block_dim: int = 64,
) -> None:
    """Raise when an explicit pair-centric request is unsafe."""
    raise ValueError(_pair_centric_unsafe_message(total_cells, n_outer, block_dim))


def _resolve_atom_centric_path(atom_centric_path: str) -> str:
    """Resolve an atom-centric path argument; ``"auto"`` defaults to ``"direct"``."""
    if atom_centric_path == "auto":
        return "direct"
    if atom_centric_path in {"direct", "sorted"}:
        return atom_centric_path
    raise ValueError(
        "atom_centric_path must be 'auto' | 'direct' | 'sorted', "
        f"got {atom_centric_path!r}",
    )


def estimate_batch_cell_list_sizes(
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    max_nbins: int = 8192,
    min_cells_per_dimension: int = 4,
) -> tuple[int, torch.Tensor]:
    """Estimate memory allocation sizes for batch cell list construction.

    Analyzes a batch of systems to determine conservative memory
    allocation requirements for torch.compile-friendly batch cell list building.
    Uses system sizes, cutoff distance, and safety factors to prevent overflow.

    Parameters
    ----------
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Unit cell matrices for each system in the batch.
    pbc : torch.Tensor, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    cutoff : float
        Neighbor search cutoff distance.
    max_nbins : int, default=8192
        Maximum number of cells to allocate per system.
    min_cells_per_dimension : int, default=4
        Minimum adaptive cell count per periodic dimension.

    Returns
    -------
    max_total_cells_across_batch : int
        Estimated maximum total cells needed across all systems combined.
    neighbor_search_radius : torch.Tensor, shape (num_systems, 3), dtype=int32
        Radius of neighboring cells to search for each system.

    Notes
    -----
    - Currently, only unit cells with a positive determinant (i.e. with
      positive volume) are supported. For non-periodic systems, pass an identity
      cell.
    - Estimates assume roughly uniform atomic distribution within each system
    - Cell sizes are determined by the smallest cutoff to ensure neighbor completeness
    - For degenerate cells or empty systems, returns conservative fallback values

    See Also
    --------
    nvalchemiops.neighbors.batch_cell_list.batch_build_cell_list : Core warp launcher
    allocate_cell_list : Allocates tensors based on these estimates
    batch_build_cell_list : High-level wrapper that uses these estimates
    """
    if max_nbins <= 0:
        raise ValueError("max_nbins must be positive")
    if cell.numel() > 0 and torch.any(cell.det().abs() == 0.0):
        raise RuntimeError(
            "Cells with volume == 0.0 detected and are not supported."
            " Please pass unit cells with `det(cell) != 0.0`."
        )
    num_systems = cell.shape[0]

    if num_systems == 0 or cutoff <= 0:
        return 1, torch.zeros((num_systems, 3), device=cell.device, dtype=torch.int32)

    dtype = cell.dtype
    device = cell.device
    wp_device = str(device)
    wp_dtype = get_wp_dtype(dtype)
    wp_mat_dtype = get_wp_mat_dtype(dtype)

    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)

    max_total_cells = torch.zeros(num_systems, device=device, dtype=torch.int32)
    wp_max_total_cells = wp.from_torch(
        max_total_cells, dtype=wp.int32, return_ctype=True
    )
    neighbor_search_radius = torch.zeros(
        (num_systems, 3), dtype=torch.int32, device=device
    )
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.vec3i, return_ctype=True
    )

    wp.launch(
        get_build_cell_list_kernel(
            "estimate_sizes",
            wp_dtype,
            batched=True,
            min_cells_per_dimension=int(min_cells_per_dimension),
        ),
        dim=num_systems,
        inputs=[
            wp_cell,
            empty_sentinel(1, wp.bool, wp_device),
            wp_pbc,
            wp_dtype(cutoff),
            max_nbins,
            wp_max_total_cells,
            empty_sentinel(1, wp.int32, wp_device),
            wp_neighbor_search_radius,
        ],
        device=wp_device,
    )

    return (
        max_total_cells.sum().item(),
        neighbor_search_radius,
    )


@torch.library.custom_op(
    "nvalchemiops::batch_build_cell_list",
    mutates_args=(
        "cells_per_dimension",
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "atoms_per_cell_count",
        "cell_atom_start_indices",
        "cell_atom_list",
    ),
)
def _batch_build_cell_list_op(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    min_cells_per_dimension: int = 4,
) -> None:
    """Internal custom op for building batch spatial cell lists.

    This function is torch compilable.

    See Also
    --------
    nvalchemiops.neighbors.batch_cell_list.batch_build_cell_list : Core warp launcher
    batch_build_cell_list : High-level wrapper function
    """
    device = positions.device
    num_systems = cell.shape[0]

    # Handle empty case
    if positions.shape[0] == 0 or cutoff <= 0:
        return

    # Get warp dtype of input tensors
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_device = str(device)

    # Convert to warp arrays
    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
    wp_batch_idx = wp.from_torch(
        batch_idx.to(dtype=torch.int32), dtype=wp.int32, return_ctype=True
    )

    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.vec3i, return_ctype=True
    )

    # Allocate cell_offsets internally (shape num_systems, not num_systems+1)
    cell_offsets = torch.zeros(num_systems, dtype=torch.int32, device=device)
    wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32)

    # Allocate cells_per_system scratch buffer
    cells_per_system = torch.zeros(num_systems, dtype=torch.int32, device=device)
    wp_cells_per_system = wp.from_torch(cells_per_system, dtype=wp.int32)

    wp_atom_periodic_shifts = wp.from_torch(
        atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
    )
    # underlying warp launcher relies on Python API for array_scan
    # so `return_ctype` is omitted
    wp_atoms_per_cell_count = wp.from_torch(atoms_per_cell_count, dtype=wp.int32)
    wp_cell_atom_start_indices = wp.from_torch(cell_atom_start_indices, dtype=wp.int32)
    wp_cell_atom_list = wp.from_torch(cell_atom_list, dtype=wp.int32, return_ctype=True)

    # Zero atoms_per_cell_count before building
    atoms_per_cell_count.zero_()

    # Call core warp launcher
    wp_batch_build_cell_list(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=cutoff,
        batch_idx=wp_batch_idx,
        cells_per_dimension=wp_cells_per_dimension,
        cell_offsets=wp_cell_offsets,
        cells_per_system=wp_cells_per_system,
        atom_periodic_shifts=wp_atom_periodic_shifts,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        atoms_per_cell_count=wp_atoms_per_cell_count,
        cell_atom_start_indices=wp_cell_atom_start_indices,
        cell_atom_list=wp_cell_atom_list,
        wp_dtype=wp_dtype,
        device=wp_device,
        min_cells_per_dimension=int(min_cells_per_dimension),
    )


def batch_build_cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    min_cells_per_dimension: int = 4,
) -> None:
    """Build batch spatial cell lists with fixed allocation sizes for torch.compile compatibility.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Concatenated atomic coordinates for all systems in the batch.
    cutoff : float
        Neighbor search cutoff distance.
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Unit cell matrices for each system in the batch.
    pbc : torch.Tensor, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=int32
        System index for each atom.
    cells_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=int32
        OUTPUT: Number of cells in x, y, z directions for each system.
    neighbor_search_radius : torch.Tensor, shape (num_systems, 3), dtype=int32
        Radius of neighboring cells to search in each dimension. Passed through
        from allocate_cell_list for API continuity but not used in this function.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        OUTPUT: Periodic boundary crossings for each atom across all systems.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        OUTPUT: 3D cell coordinates assigned to each atom across all systems.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        OUTPUT: Number of atoms in each cell across all systems.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        OUTPUT: Starting index in global cell arrays for each system (CSR format).
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32
        OUTPUT: Flattened list of atom indices organized by cell across all systems.
    min_cells_per_dimension : int, default=4
        Minimum adaptive cell count per periodic dimension.

    See Also
    --------
    nvalchemiops.neighbors.batch_cell_list.batch_build_cell_list : Core warp launcher
    estimate_batch_cell_list_sizes : Estimate memory requirements
    batch_query_cell_list : Query the built cell list for neighbors
    batch_cell_list : High-level function that builds and queries in one call
    """
    return _batch_build_cell_list_op(
        positions,
        cutoff,
        cell,
        pbc,
        batch_idx,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        min_cells_per_dimension,
    )


@torch.library.custom_op(
    "nvalchemiops::batch_query_cell_list",
    mutates_args=("neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"),
)
def _batch_query_cell_list_op(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    half_fill: bool = False,
    fill_value: int | None = None,
    algorithm: str = "auto",
    atom_centric_path: str = "auto",
) -> None:
    """Internal custom op for querying batch spatial cell lists to build neighbor matrices.

    This function is torch compilable.

    When ``fill_value`` is provided, the op writes ``fill_value`` into
    ``neighbor_matrix[i, num_neighbors[i]..max_neighbors-1]`` after the
    query kernel (CUDA only), letting callers skip the upstream
    ``neighbor_matrix.fill_(fill_value) + neighbor_matrix_shifts.zero_()``
    prefills.  Mirrors the single-system skip-prefill design.

    ``strategy`` mirrors the single-system :func:`cell_list` knob:

    - ``"auto"`` (default) - apply :func:`select_batch_cell_list_strategy`.
    - ``"atom_centric"`` - force atom-centric.
    - ``"pair_centric"`` - force pair-centric (CUDA only; CPU raises).

    See Also
    --------
    nvalchemiops.neighbors.batch_cell_list.batch_query_cell_list : Core warp launcher
    batch_query_cell_list : High-level wrapper function
    """
    device = positions.device
    strategy = algorithm
    num_systems = cell.shape[0]

    # Handle empty case
    if positions.shape[0] == 0 or cutoff <= 0:
        return

    # Get warp dtypes and arrays
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_device = str(device)

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
    wp_batch_idx = wp.from_torch(
        batch_idx.to(dtype=torch.int32), dtype=wp.int32, return_ctype=True
    )

    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.vec3i, return_ctype=True
    )
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.vec3i, return_ctype=True
    )

    #  cell_offsets[i] = sum of cells for systems 0..i-1
    cells_per_system = cells_per_dimension.prod(dim=1)
    cell_offsets = torch.zeros(num_systems, dtype=torch.int32, device=device)
    if num_systems > 1:
        torch.cumsum(cells_per_system[:-1], dim=0, out=cell_offsets[1:])
    # cell_offsets[0] is already 0 from zeros initialization
    wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32, return_ctype=True)

    wp_atom_periodic_shifts = wp.from_torch(
        atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
    )
    wp_atoms_per_cell_count = wp.from_torch(
        atoms_per_cell_count, dtype=wp.int32, return_ctype=True
    )
    wp_cell_atom_start_indices = wp.from_torch(
        cell_atom_start_indices, dtype=wp.int32, return_ctype=True
    )
    wp_cell_atom_list = wp.from_torch(cell_atom_list, dtype=wp.int32, return_ctype=True)

    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix_shifts = wp.from_torch(
        neighbor_matrix_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)

    # Atom-centric vs pair-centric (pair-centric is CUDA-only).
    total_atoms = positions.shape[0]
    atom_centric_path = _resolve_atom_centric_path(atom_centric_path)

    cpu_only = device.type != "cuda"
    if strategy == "auto":
        use_pair_centric = (not cpu_only) and (
            select_batch_cell_list_strategy(
                total_atoms=int(total_atoms),
                num_systems=int(num_systems),
                cutoff=float(cutoff),
            )
            == "pair_centric"
        )
    elif strategy == "atom_centric":
        use_pair_centric = False
    elif strategy == "pair_centric":
        if cpu_only:
            raise ValueError(
                "strategy='pair_centric' is not supported on CPU "
                "(kernels use CUDA block scheduling).  Pass 'auto' or "
                "'atom_centric' instead.",
            )
        use_pair_centric = True
    else:
        raise ValueError(
            f"strategy must be 'auto' | 'atom_centric' | 'pair_centric', "
            f"got {strategy!r}",
        )

    wp_sorted_pos = None
    wp_sorted_shifts = None

    wp_cells_per_system = None
    wp_cell_to_system = None
    total_cells = None
    n_outer = None
    R_max = None
    if use_pair_centric:
        total_cells = int(cells_per_system.sum().item())
        R_max_t = neighbor_search_radius.max(dim=0).values.tolist()
        R_max = (int(R_max_t[0]), int(R_max_t[1]), int(R_max_t[2]))
        n_outer = compute_batch_pair_centric_n_outer(R_max, bool(half_fill))
        if not is_pair_centric_launch_safe(total_cells, n_outer):
            if strategy == "pair_centric":
                _raise_unsafe_pair_centric_launch(total_cells, n_outer)
            use_pair_centric = False
            total_cells = None
            n_outer = None
            R_max = None
        elif strategy == "auto" and not is_pair_centric_parallelism_sufficient(
            int(total_atoms), total_cells, n_outer
        ):
            use_pair_centric = False
            total_cells = None
            n_outer = None
            R_max = None
        else:
            wp_cells_per_system = wp.from_torch(
                cells_per_system.to(dtype=torch.int32),
                dtype=wp.int32,
                return_ctype=True,
            )
            cell_to_system_t = torch.zeros(
                max(total_cells, 1), dtype=torch.int32, device=device
            )
            wp_cell_to_system = wp.from_torch(
                cell_to_system_t, dtype=wp.int32, return_ctype=True
            )

    if use_pair_centric or atom_centric_path == "sorted":
        sorted_positions_t = torch.empty(
            (int(total_atoms), 3), dtype=positions.dtype, device=device
        )
        sorted_shifts_t = torch.empty(
            (int(total_atoms), 3), dtype=torch.int32, device=device
        )
        wp_sorted_pos = wp.from_torch(
            sorted_positions_t, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_sorted_shifts = wp.from_torch(
            sorted_shifts_t, dtype=wp.vec3i, return_ctype=True
        )

    wp_batch_query_cell_list(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=cutoff,
        batch_idx=wp_batch_idx,
        cells_per_dimension=wp_cells_per_dimension,
        neighbor_search_radius=wp_neighbor_search_radius,
        cell_offsets=wp_cell_offsets,
        atom_periodic_shifts=wp_atom_periodic_shifts,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        atoms_per_cell_count=wp_atoms_per_cell_count,
        cell_atom_start_indices=wp_cell_atom_start_indices,
        cell_atom_list=wp_cell_atom_list,
        sorted_positions=wp_sorted_pos,
        sorted_atom_periodic_shifts=wp_sorted_shifts,
        neighbor_matrix=wp_neighbor_matrix,
        neighbor_matrix_shifts=wp_neighbor_matrix_shifts,
        num_neighbors=wp_num_neighbors,
        rebuild_flags=None,
        wp_dtype=wp_dtype,
        device=wp_device,
        half_fill=half_fill,
        strategy="pair_centric" if use_pair_centric else "atom_centric",
        atom_centric_path=atom_centric_path,
        cells_per_system=wp_cells_per_system,
        cell_to_system=wp_cell_to_system,
        total_cells=total_cells,
        n_outer=n_outer,
        R_max=R_max,
    )

    # Coalesced tail fill (CUDA only - the kernel uses wp.launch_tiled
    # which silently mis-runs on CPU; CPU callers prefill in
    # ``batch_cell_list`` above).  Mirrors the single-system pattern
    # in ``_query_cell_list_op``.
    if fill_value is not None and wp_device != "cpu":
        max_neighbors = int(neighbor_matrix.shape[1])
        if max_neighbors > 0:
            wp_fill_neighbor_matrix_tail(
                wp_num_neighbors,
                # Row count must be the OUTPUT matrix's row count, not
                # ``total_atoms``: the ``target_indices`` (partial) path writes
                # compact ``num_targets`` rows, so ``total_atoms`` would launch
                # the tail-fill out of bounds over rows [num_targets, N).
                int(neighbor_matrix.shape[0]),
                max_neighbors,
                int(fill_value),
                wp_neighbor_matrix,
                wp_device,
            )


@torch.library.custom_op(
    "nvalchemiops::batch_query_cell_list_selective",
    mutates_args=("neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"),
)
def _batch_query_cell_list_selective_op(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    rebuild_flags: torch.Tensor,
    half_fill: bool = False,
    atom_centric_path: str = "auto",
) -> None:
    """Internal custom op for querying batch cell lists with per-system selective skip.

    Only systems with rebuild_flags[i] == True are recomputed on the GPU.
    Existing neighbor data for non-rebuilt systems is preserved without CPU-GPU sync.

    This function is torch compilable.

    See Also
    --------
    nvalchemiops.neighbors.batch_cell_list.batch_query_cell_list : Core warp launcher
    batch_query_cell_list : High-level wrapper function
    """
    device = positions.device
    num_systems = cell.shape[0]

    if positions.shape[0] == 0 or cutoff <= 0:
        return

    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_device = str(device)

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
    wp_batch_idx = wp.from_torch(
        batch_idx.to(dtype=torch.int32), dtype=wp.int32, return_ctype=True
    )
    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.vec3i, return_ctype=True
    )
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.vec3i, return_ctype=True
    )

    cells_per_system = cells_per_dimension.prod(dim=1)
    cell_offsets = torch.zeros(num_systems, dtype=torch.int32, device=device)
    if num_systems > 1:
        torch.cumsum(cells_per_system[:-1], dim=0, out=cell_offsets[1:])
    wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32, return_ctype=True)

    wp_atom_periodic_shifts = wp.from_torch(
        atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
    )
    wp_atoms_per_cell_count = wp.from_torch(
        atoms_per_cell_count, dtype=wp.int32, return_ctype=True
    )
    wp_cell_atom_start_indices = wp.from_torch(
        cell_atom_start_indices, dtype=wp.int32, return_ctype=True
    )
    wp_cell_atom_list = wp.from_torch(cell_atom_list, dtype=wp.int32, return_ctype=True)
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix_shifts = wp.from_torch(
        neighbor_matrix_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)
    wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True)

    wp_sorted_pos = None
    wp_sorted_shifts = None
    atom_centric_path = _resolve_atom_centric_path(atom_centric_path)

    wp_batch_query_cell_list(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=cutoff,
        batch_idx=wp_batch_idx,
        cells_per_dimension=wp_cells_per_dimension,
        neighbor_search_radius=wp_neighbor_search_radius,
        cell_offsets=wp_cell_offsets,
        atom_periodic_shifts=wp_atom_periodic_shifts,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        atoms_per_cell_count=wp_atoms_per_cell_count,
        cell_atom_start_indices=wp_cell_atom_start_indices,
        cell_atom_list=wp_cell_atom_list,
        sorted_positions=wp_sorted_pos,
        sorted_atom_periodic_shifts=wp_sorted_shifts,
        neighbor_matrix=wp_neighbor_matrix,
        neighbor_matrix_shifts=wp_neighbor_matrix_shifts,
        num_neighbors=wp_num_neighbors,
        wp_dtype=wp_dtype,
        device=wp_device,
        half_fill=half_fill,
        rebuild_flags=wp_rebuild_flags,
        atom_centric_path=atom_centric_path,
    )


def batch_query_cell_list(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    half_fill: bool = False,
    rebuild_flags: torch.Tensor | None = None,
    fill_value: int | None = None,
    strategy: str = "auto",
    atom_centric_path: str = "auto",
    target_indices: torch.Tensor | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: torch.Tensor | None = None,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
) -> None:
    """Query batch spatial cell lists to build neighbor matrices for multiple systems.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Concatenated Cartesian coordinates for all systems in the batch.
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Unit cell matrices for each system in the batch.
    pbc : torch.Tensor, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags.
    cutoff : float
        Neighbor search cutoff distance.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=int32
        System index for each atom.
    cells_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=int32
        Number of cells in x, y, z directions for each system.
    neighbor_search_radius : torch.Tensor, shape (num_systems, 3), dtype=int32
        Radius of neighboring cells to search.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings per atom from batch_build_cell_list.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates per atom from batch_build_cell_list.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        Number of atoms per cell from batch_build_cell_list.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        Starting index per cell from batch_build_cell_list.
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32
        Atom list organized by cell from batch_build_cell_list.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=int32
        OUTPUT: Neighbor matrix to be filled.
    neighbor_matrix_shifts : torch.Tensor, shape (total_atoms, max_neighbors, 3), dtype=int32
        OUTPUT: Shift vectors for each neighbor relationship.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=int32
        OUTPUT: Number of neighbors per atom.
    half_fill : bool, default=False
        If True, only store half of the neighbor relationships.
    rebuild_flags : torch.Tensor, shape (num_systems,), dtype=torch.bool, optional
        Per-system rebuild flags. If provided, only systems with True are processed
        on the GPU; existing neighbor data for other systems is preserved.
    fill_value : int, optional
        If provided AND ``rebuild_flags`` is None, the operation writes
        ``fill_value`` into the unused-column tail of ``neighbor_matrix``
        after the kernel runs (CUDA only), letting callers skip the
        ``neighbor_matrix.fill_(fill_value) + neighbor_matrix_shifts.zero_()``
        prefills.  Mirrors the single-system skip-prefill design.
    strategy : {"auto", "atom_centric", "pair_centric"}, default "auto"
        Forces one of the two warp-level batch cell-list kernels.
        ``"auto"`` applies the sync-free dispatch rule
        (:func:`select_batch_cell_list_strategy`).  Pair-centric requires CUDA.
    atom_centric_path : {"auto", "direct", "sorted"}, default "auto"
        Selects the atom-centric implementation path. ``"auto"`` resolves to
        ``"direct"``.

    See Also
    --------
    nvalchemiops.neighbors.batch_cell_list.batch_query_cell_list : Core warp launcher
    batch_build_cell_list : Builds the cell list data structures
    batch_cell_list : High-level function that builds and queries in one call
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
        # Optional per-neighbor outputs bypass the torch custom op (which
        # cannot carry a callable ``pair_fn``) and call the warp factory
        # directly while preserving the requested strategy.
        _batch_query_cell_list_optional(
            positions,
            cell,
            pbc,
            cutoff,
            batch_idx,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            half_fill=half_fill,
            rebuild_flags=rebuild_flags,
            fill_value=fill_value,
            strategy=strategy,
            atom_centric_path=atom_centric_path,
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
        return None
    if rebuild_flags is None:
        return _batch_query_cell_list_op(
            positions,
            cell,
            pbc,
            cutoff,
            batch_idx,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            half_fill,
            fill_value,
            strategy,
            atom_centric_path,
        )
    return _batch_query_cell_list_selective_op(
        positions,
        cell,
        pbc,
        cutoff,
        batch_idx,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        rebuild_flags,
        half_fill,
        atom_centric_path,
    )


def _batch_query_cell_list_optional(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    *,
    half_fill: bool,
    rebuild_flags: torch.Tensor | None,
    fill_value: int | None,
    strategy: str,
    atom_centric_path: str,
    target_indices: torch.Tensor | None,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
    pair_params: torch.Tensor | None,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
    pair_energies: torch.Tensor | None,
    pair_forces: torch.Tensor | None,
) -> None:
    """Route to the warp factory when optional per-neighbor outputs are used.

    Bypasses the torch ``@torch.library.custom_op`` boundary (which
    cannot carry a callable ``pair_fn``) and calls
    :func:`wp_batch_query_cell_list` directly.  Caller-supplied buffers
    are converted via :func:`wp.from_torch`; omitted scratch is allocated
    fresh as torch tensors for this call.
    """
    device = positions.device
    num_systems = cell.shape[0]
    total_atoms = positions.shape[0]
    if total_atoms == 0 or cutoff <= 0:
        return

    # The query writes one output row per source atom: ``num_targets`` compact
    # rows when ``target_indices`` is given, else ``total_atoms``.  Validate the
    # caller-owned output buffers cover that many rows *before* launching, so an
    # undersized (e.g. compact ``target_indices``) buffer raises a clean error
    # instead of an out-of-bounds device write that corrupts the CUDA context.
    n_out_rows = (
        int(target_indices.shape[0]) if target_indices is not None else total_atoms
    )
    if int(neighbor_matrix.shape[0]) < n_out_rows:
        raise ValueError(
            f"neighbor_matrix has {int(neighbor_matrix.shape[0])} rows but the "
            f"{'partial target_indices' if target_indices is not None else 'full'}"
            f" query writes {n_out_rows} rows; allocate at least that many."
        )
    if int(num_neighbors.shape[0]) < int(neighbor_matrix.shape[0]):
        raise ValueError(
            "num_neighbors must have at least as many rows as neighbor_matrix "
            f"(got {int(num_neighbors.shape[0])} vs {int(neighbor_matrix.shape[0])})."
        )

    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_device = str(device)

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
    wp_batch_idx = wp.from_torch(
        batch_idx.to(dtype=torch.int32), dtype=wp.int32, return_ctype=True
    )
    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.vec3i, return_ctype=True
    )
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.vec3i, return_ctype=True
    )

    cells_per_system = cells_per_dimension.prod(dim=1)
    cell_offsets = torch.zeros(num_systems, dtype=torch.int32, device=device)
    if num_systems > 1:
        torch.cumsum(cells_per_system[:-1], dim=0, out=cell_offsets[1:])
    wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32, return_ctype=True)

    wp_atom_periodic_shifts = wp.from_torch(
        atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
    )
    wp_atoms_per_cell_count = wp.from_torch(
        atoms_per_cell_count, dtype=wp.int32, return_ctype=True
    )
    wp_cell_atom_start_indices = wp.from_torch(
        cell_atom_start_indices, dtype=wp.int32, return_ctype=True
    )
    wp_cell_atom_list = wp.from_torch(cell_atom_list, dtype=wp.int32, return_ctype=True)
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix_shifts = wp.from_torch(
        neighbor_matrix_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)

    wp_sorted_pos = None
    wp_sorted_shifts = None

    if rebuild_flags is not None:
        wp_rebuild_flags = wp.from_torch(
            rebuild_flags, dtype=wp.bool, return_ctype=True
        )
    else:
        wp_rebuild_flags = None

    wp_target_indices = (
        wp.from_torch(target_indices, dtype=wp.int32, return_ctype=True)
        if target_indices is not None
        else None
    )
    # Pair-output buffers are validated by ``_prepare_pair_output_args`` in the
    # launcher (which dereferences ``pair_params.dtype``), so they must be real
    # Warp arrays, not ``return_ctype`` launch structs.  ``wp.from_torch`` without
    # ``return_ctype`` still aliases the torch tensor zero-copy, so kernel writes
    # land in the output buffers.
    wp_pair_params = (
        wp.from_torch(pair_params, dtype=wp_dtype) if pair_params is not None else None
    )
    wp_neighbor_vectors = (
        wp.from_torch(neighbor_vectors, dtype=wp_vec_dtype)
        if neighbor_vectors is not None
        else None
    )
    wp_neighbor_distances = (
        wp.from_torch(neighbor_distances, dtype=wp_dtype)
        if neighbor_distances is not None
        else None
    )
    wp_pair_energies = (
        wp.from_torch(pair_energies, dtype=wp_dtype)
        if pair_energies is not None
        else None
    )
    wp_pair_forces = (
        wp.from_torch(pair_forces, dtype=wp_vec_dtype)
        if pair_forces is not None
        else None
    )

    atom_centric_path = _resolve_atom_centric_path(atom_centric_path)

    cpu_only = device.type != "cuda"
    if strategy == "auto":
        use_pair_centric = (not cpu_only) and (
            select_batch_cell_list_strategy(
                total_atoms=int(total_atoms),
                num_systems=int(num_systems),
                cutoff=float(cutoff),
            )
            == "pair_centric"
        )
    elif strategy == "atom_centric":
        use_pair_centric = False
    elif strategy == "pair_centric":
        if cpu_only:
            raise ValueError(
                "strategy='pair_centric' is not supported on CPU "
                "(kernels use CUDA block scheduling).  Pass 'auto' or "
                "'atom_centric' instead.",
            )
        use_pair_centric = True
    else:
        raise ValueError(
            f"strategy must be 'auto' | 'atom_centric' | 'pair_centric', "
            f"got {strategy!r}",
        )

    wp_cells_per_system = None
    wp_cell_to_system = None
    total_cells = None
    n_outer = None
    R_max = None
    if use_pair_centric:
        total_cells = int(cells_per_system.sum().item())
        R_max_t = neighbor_search_radius.max(dim=0).values.tolist()
        R_max = (int(R_max_t[0]), int(R_max_t[1]), int(R_max_t[2]))
        n_outer = compute_batch_pair_centric_n_outer(R_max, bool(half_fill))
        if not is_pair_centric_launch_safe(total_cells, n_outer):
            if strategy == "pair_centric":
                _raise_unsafe_pair_centric_launch(total_cells, n_outer)
            use_pair_centric = False
            total_cells = None
            n_outer = None
            R_max = None
        elif strategy == "auto" and not is_pair_centric_parallelism_sufficient(
            int(total_atoms), total_cells, n_outer
        ):
            use_pair_centric = False
            total_cells = None
            n_outer = None
            R_max = None
        else:
            wp_cells_per_system = wp.from_torch(
                cells_per_system.to(dtype=torch.int32),
                dtype=wp.int32,
                return_ctype=True,
            )
            cell_to_system_t = torch.zeros(
                max(total_cells, 1), dtype=torch.int32, device=device
            )
            wp_cell_to_system = wp.from_torch(
                cell_to_system_t, dtype=wp.int32, return_ctype=True
            )

    if use_pair_centric or atom_centric_path == "sorted":
        sorted_positions_t = torch.empty(
            (int(total_atoms), 3), dtype=positions.dtype, device=device
        )
        sorted_shifts_t = torch.empty(
            (int(total_atoms), 3), dtype=torch.int32, device=device
        )
        wp_sorted_pos = wp.from_torch(
            sorted_positions_t, dtype=wp_vec_dtype, return_ctype=True
        )
        wp_sorted_shifts = wp.from_torch(
            sorted_shifts_t, dtype=wp.vec3i, return_ctype=True
        )

    wp_batch_query_cell_list(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=float(cutoff),
        batch_idx=wp_batch_idx,
        cells_per_dimension=wp_cells_per_dimension,
        neighbor_search_radius=wp_neighbor_search_radius,
        cell_offsets=wp_cell_offsets,
        atom_periodic_shifts=wp_atom_periodic_shifts,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        atoms_per_cell_count=wp_atoms_per_cell_count,
        cell_atom_start_indices=wp_cell_atom_start_indices,
        cell_atom_list=wp_cell_atom_list,
        sorted_positions=wp_sorted_pos,
        sorted_atom_periodic_shifts=wp_sorted_shifts,
        neighbor_matrix=wp_neighbor_matrix,
        neighbor_matrix_shifts=wp_neighbor_matrix_shifts,
        num_neighbors=wp_num_neighbors,
        rebuild_flags=wp_rebuild_flags,
        wp_dtype=wp_dtype,
        device=wp_device,
        half_fill=bool(half_fill),
        strategy="pair_centric" if use_pair_centric else "atom_centric",
        atom_centric_path=atom_centric_path,
        cells_per_system=wp_cells_per_system,
        cell_to_system=wp_cell_to_system,
        total_cells=total_cells,
        n_outer=n_outer,
        R_max=R_max,
        target_indices=wp_target_indices,
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        pair_fn=pair_fn,
        pair_params=wp_pair_params,
        neighbor_vectors=wp_neighbor_vectors,
        neighbor_distances=wp_neighbor_distances,
        pair_energies=wp_pair_energies,
        pair_forces=wp_pair_forces,
    )

    if fill_value is not None and rebuild_flags is None and wp_device != "cpu":
        max_neighbors = int(neighbor_matrix.shape[1])
        if max_neighbors > 0:
            wp_fill_neighbor_matrix_tail(
                wp_num_neighbors,
                # Row count must be the OUTPUT matrix's row count, not
                # ``total_atoms``: the ``target_indices`` (partial) path writes
                # compact ``num_targets`` rows, so ``total_atoms`` would launch
                # the tail-fill out of bounds over rows [num_targets, N).
                int(neighbor_matrix.shape[0]),
                max_neighbors,
                int(fill_value),
                wp_neighbor_matrix,
                wp_device,
            )


def batch_cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    num_neighbors: torch.Tensor | None = None,
    cells_per_dimension: torch.Tensor | None = None,
    neighbor_search_radius: torch.Tensor | None = None,
    cell_offsets: torch.Tensor | None = None,
    atom_periodic_shifts: torch.Tensor | None = None,
    atom_to_cell_mapping: torch.Tensor | None = None,
    atoms_per_cell_count: torch.Tensor | None = None,
    cell_atom_start_indices: torch.Tensor | None = None,
    cell_atom_list: torch.Tensor | None = None,
    rebuild_flags: torch.Tensor | None = None,
    strategy: str = "auto",
    atom_centric_path: str = "auto",
    target_indices: torch.Tensor | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: torch.Tensor | None = None,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor]
):
    """Build complete batch neighbor matrices using spatial cell list acceleration.

    High-level convenience function that processes multiple systems
    simultaneously. Automatically estimates memory requirements, builds batch
    spatial cell list data structures, and queries them to produce complete
    neighbor matrices for all systems.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Concatenated atomic coordinates for all systems in the batch.
    cutoff : float
        Neighbor search cutoff distance.
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Unit cell matrices for each system in the batch.
    pbc : torch.Tensor, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=int32
        System index for each atom.
    max_neighbors : int or None, optional
        Maximum number of neighbors per atom. If None, automatically estimated.
    half_fill : bool, default=False
        If True, only fill half of the neighbor matrix.
    fill_value : int | None, optional
        Value to use for padding empty neighbor slots in the matrix. Default is total_atoms.
    return_neighbor_list : bool, optional - default=False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format.
    cells_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=int32, optional
        Pre-allocated tensor for cell dimensions.
    neighbor_search_radius : torch.Tensor, shape (num_systems, 3), dtype=int32, optional
        Pre-allocated tensor for search radius.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32, optional
        Pre-allocated tensor for periodic shifts.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32, optional
        Pre-allocated tensor for cell mapping.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32, optional
        Pre-allocated tensor for atom counts.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32, optional
        Pre-allocated tensor for start indices.
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32, optional
        Pre-allocated tensor for atom list.
    rebuild_flags : torch.Tensor, shape (num_systems,), dtype=torch.bool, optional
        Per-system rebuild flags produced by ``batch_cell_list_needs_rebuild``.
        If provided, only systems where rebuild_flags[i] is True are recomputed;
        existing data in ``neighbor_matrix`` and ``num_neighbors`` is preserved for
        non-rebuilt systems entirely on the GPU (no CPU-GPU sync). When this is used,
        pre-allocated ``neighbor_matrix`` and ``num_neighbors`` tensors must be provided
        and will not be globally zeroed - only rebuilt-system entries are reset.

    Returns
    -------
    results : tuple of torch.Tensor
        Variable-length tuple with neighbor data in matrix or list format.

    See Also
    --------
    nvalchemiops.neighbors.batch_cell_list.batch_build_cell_list : Core warp launcher for building
    nvalchemiops.neighbors.batch_cell_list.batch_query_cell_list : Core warp launcher for querying
    batch_naive_neighbor_list : O(N^2) method for small systems
    """

    total_atoms = positions.shape[0]
    device = positions.device
    if device == "cpu":
        warnings.warn(
            "The CPU version of `batch_cell_list` is known to experience"
            " issues with memory allocation and under investigation. Please"
            " ensure tensor provided as `positions` is on GPU."
        )

    # Handle empty case
    if total_atoms <= 0 or cutoff <= 0:
        if return_neighbor_list:
            return (
                torch.zeros((2, 0), dtype=torch.int32, device=device),
                torch.zeros((total_atoms + 1,), dtype=torch.int32, device=device),
                torch.zeros((0, 3), dtype=torch.int32, device=device),
            )
        else:
            return (
                torch.full((total_atoms, 0), -1, dtype=torch.int32, device=device),
                torch.zeros((total_atoms,), dtype=torch.int32, device=device),
                torch.zeros((total_atoms, 0, 3), dtype=torch.int32, device=device),
            )

    if max_neighbors is None and neighbor_matrix is None:
        max_neighbors = estimate_max_neighbors(cutoff)

    if fill_value is None:
        fill_value = total_atoms

    # CPU prefills; CUDA tail-fills (``wp.launch_tiled`` mis-runs on CPU).
    is_cpu = device.type == "cpu"
    if neighbor_matrix is None:
        if is_cpu:
            neighbor_matrix = torch.full(
                (total_atoms, max_neighbors),
                fill_value,
                dtype=torch.int32,
                device=device,
            )
        else:
            neighbor_matrix = torch.empty(
                (total_atoms, max_neighbors), dtype=torch.int32, device=device
            )
    elif is_cpu and rebuild_flags is None:
        neighbor_matrix.fill_(fill_value)
    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = torch.empty(
            (total_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
    if num_neighbors is None:
        num_neighbors = torch.zeros((total_atoms,), dtype=torch.int32, device=device)
    elif rebuild_flags is None:
        num_neighbors.zero_()

    # Allocate cell list if needed.  Explicit atom-centric queries use the
    # legacy 1-cell minimum; auto/pair-centric keep the current 4-cell policy.
    allocated_cell_list = (
        cells_per_dimension is None
        or neighbor_search_radius is None
        or atom_periodic_shifts is None
        or atom_to_cell_mapping is None
        or atoms_per_cell_count is None
        or cell_atom_start_indices is None
        or cell_atom_list is None
    )
    cell_list_min_cells = 1 if strategy == "atom_centric" else 4
    if allocated_cell_list:
        max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell,
            pbc,
            cutoff,
            min_cells_per_dimension=cell_list_min_cells,
        )
        (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = allocate_cell_list(
            total_atoms,
            max_total_cells,
            neighbor_search_radius,
            device,
        )
        cell_list_cache = (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )
    else:
        # Caller-provided caches are assumed to have been sized with the
        # default public estimate policy.
        cell_list_min_cells = 4
        # atoms_per_cell_count is atomic_add'd; the rest are fully overwritten.
        atoms_per_cell_count.zero_()
        cell_list_cache = (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )

    # Build batch cell list with fixed allocations
    batch_build_cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        batch_idx,
        *cell_list_cache,
        min_cells_per_dimension=cell_list_min_cells,
    )

    if return_vectors or return_distances:
        if return_distances and neighbor_distances is None:
            neighbor_distances = torch.zeros(
                (total_atoms, max_neighbors), dtype=positions.dtype, device=device
            )
        if return_vectors and neighbor_vectors is None:
            neighbor_vectors = torch.zeros(
                (total_atoms, max_neighbors, 3),
                dtype=positions.dtype,
                device=device,
            )
        # ``pair_fn`` energy/force buffers are optional: allocate them like the
        # neighbor matrix when not supplied, so they can be returned.
        if pair_fn is not None and pair_energies is None:
            pair_energies = torch.zeros(
                (total_atoms, max_neighbors), dtype=positions.dtype, device=device
            )
        if pair_fn is not None and pair_forces is None:
            pair_forces = torch.zeros(
                (total_atoms, max_neighbors, 3), dtype=positions.dtype, device=device
            )

        forward_kwargs = {
            "cutoff": cutoff,
            "pbc": pbc,
            "batch_idx": batch_idx,
            "cell_list_cache": cell_list_cache,
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
            "num_neighbors": num_neighbors,
            "half_fill": half_fill,
            "rebuild_flags": rebuild_flags,
            "fill_value": fill_value,
            "strategy": strategy,
            "atom_centric_path": atom_centric_path,
            "target_indices": target_indices,
            "return_vectors": return_vectors,
            "return_distances": return_distances,
            "pair_fn": pair_fn,
            "pair_params": pair_params,
            "neighbor_vectors": neighbor_vectors,
            "neighbor_distances": neighbor_distances,
            "pair_energies": pair_energies,
            "pair_forces": pair_forces,
        }
        distances_out, vectors_out, nm_out, nn_out, shifts_out = _route_pair_outputs(
            positions,
            cell,
            _batch_cell_list_query_forward,
            forward_kwargs,
        )

        if return_neighbor_list:
            nl, nptr, nl_shifts = get_neighbor_list_from_neighbor_matrix(
                nm_out,
                num_neighbors=nn_out,
                neighbor_shift_matrix=shifts_out,
                fill_value=fill_value,
            )
            base = (nl, nptr, nl_shifts)
            # Repack per-pair outputs into the same COO order as ``nl`` so they
            # index-align with it; ``index_select`` keeps the autograd link.
            # ``pair_fn`` also fills the caller's matrix buffers in place.
            active = nm_out != fill_value
            distances_out, vectors_out = coo_pack_pair_geometry(
                active, distances_out, vectors_out
            )
            pe_out, pf_out = coo_pack_pair_geometry(active, pair_energies, pair_forces)
        else:
            base = (nm_out, nn_out, shifts_out)
            pe_out, pf_out = pair_energies, pair_forces

        tail: list[torch.Tensor] = []
        if return_distances:
            tail.append(distances_out)
        if return_vectors:
            tail.append(vectors_out)
        if pair_fn is not None:
            tail.extend((pe_out, pf_out))
        return (*base, *tail)

    # Query neighbor lists
    batch_query_cell_list(
        positions,
        cell,
        pbc,
        cutoff,
        batch_idx,
        *cell_list_cache,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
        rebuild_flags,
        fill_value,
        strategy,
        atom_centric_path,
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

    if return_neighbor_list:
        neighbor_list, neighbor_ptr, neighbor_list_shifts = (
            get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                neighbor_shift_matrix=neighbor_matrix_shifts,
                fill_value=fill_value,
            )
        )
        return neighbor_list, neighbor_ptr, neighbor_list_shifts
    else:
        return neighbor_matrix, num_neighbors, neighbor_matrix_shifts


def _batch_cell_list_query_forward(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    *,
    cutoff: float,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_list_cache: tuple,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    half_fill: bool,
    rebuild_flags: torch.Tensor | None,
    fill_value: int,
    strategy: str,
    atom_centric_path: str,
    target_indices: torch.Tensor | None,
    return_vectors: bool,
    return_distances: bool,
    pair_fn,
    pair_params: torch.Tensor | None,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
    pair_energies: torch.Tensor | None,
    pair_forces: torch.Tensor | None,
) -> _NeighborForwardOutput:
    """Forward closure consumed by ``_NeighborDistanceVectorFn`` (batched)."""
    batch_query_cell_list(
        positions,
        cell,
        pbc,
        cutoff,
        batch_idx,
        *cell_list_cache,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
        rebuild_flags,
        fill_value,
        strategy,
        atom_centric_path,
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
    i_idx, j_idx, shifts_flat, batch_idx_flat, mask = _flatten_active_pairs(
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
        target_indices=target_indices,
        batch_idx=batch_idx,
    )
    K, M = neighbor_matrix.shape
    return _NeighborForwardOutput(
        distances=neighbor_distances,
        vectors=neighbor_vectors,
        extra_outputs=(neighbor_matrix, num_neighbors, neighbor_matrix_shifts),
        i_idx_flat=i_idx,
        j_idx_flat=j_idx,
        shifts_flat=shifts_flat,
        batch_idx_flat=batch_idx_flat,
        active_mask=mask,
        matrix_shape=(K, M),
    )
