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

"""PyTorch bindings for the CSR cell-list neighbor builder.

The CSR path stores per-cell atom lists in three tensors
(``cell_atom_start_indices``, ``cell_atom_list``, ``atoms_per_cell_count``)
with variable per-cell occupancy.  It handles arbitrary cell geometry
(orthorhombic + triclinic), arbitrary periodic-boundary settings, both
half-fill modes, and selective rebuild.

Two query kernels share this build:

* **atom-centric** - one thread per atom.  The default direct
  (full-fill) path accumulates counts via ``wp.atomic_add`` on
  ``num_neighbors``, so the order of neighbors within a given row is
  unspecified.  Best at large N.
* **pair-centric** - one CUDA block per ``(source_cell, offset)``;
  offset zero handles same-cell pairs.  Per-emit ``atomic_add`` on
  ``num_neighbors``.  Best at small/medium N or large cutoff (more
  cell-level parallelism than atom-level).

Auto-select uses sync-free quantities (``natom``, ``cutoff``).  See
:func:`select_cell_list_strategy` for the 3-clause rule.  Pin a strategy
per-call via ``cell_list(..., strategy="pair_centric")``.
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.neighbors.cell_list import (
    PAIR_CENTRIC_MAX_LINEAR_LAUNCH,
    compute_batch_pair_centric_n_outer,
    get_build_cell_list_kernel,
    is_pair_centric_launch_safe,
    is_pair_centric_parallelism_sufficient,
    pair_centric_launch_size,
    select_cell_list_strategy,
)
from nvalchemiops.neighbors.cell_list import (
    build_cell_list as wp_build_cell_list,
)
from nvalchemiops.neighbors.cell_list import (
    query_cell_list as wp_query_cell_list,
)
from nvalchemiops.neighbors.neighbor_utils import (
    empty_sentinel,
    estimate_max_neighbors,
    selective_zero_num_neighbors_single,
)
from nvalchemiops.neighbors.neighbor_utils import (
    fill_neighbor_matrix_tail as wp_fill_neighbor_matrix_tail,
)
from nvalchemiops.neighbors.output_args import (
    _has_partial_or_pair_outputs,
)
from nvalchemiops.torch._warp_op_helpers import register_noop_fake
from nvalchemiops.torch.neighbors._autograd import (
    _flatten_active_pairs,
    _NeighborForwardOutput,
    _route_pair_outputs,
)
from nvalchemiops.torch.neighbors._compiled_pair_fn import (
    CompiledPairFn,
    is_compiled_pair_fn,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    _validate_pair_params_present,
    allocate_cell_list,
    coo_pack_pair_geometry,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = [
    "allocate_query_sort_scratch",
    "build_cell_list",
    "cell_list",
    "estimate_cell_list_sizes",
    "query_cell_list",
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


def allocate_query_sort_scratch(
    total_atoms: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the sort-side scratch tensors consumed by
    ``query_cell_list``.

    Required by both atom-centric and pair-centric query paths when the
    call is wrapped in a captured CUDA graph.  Allocate once during
    setup and pass the returned tensors to ``query_cell_list(...,
    sorted_positions=..., sorted_shifts=...)`` so the captured region
    does no allocation of its own.

    Returns
    -------
    sorted_positions : torch.Tensor, shape ``(total_atoms, 3)``, dtype=``dtype``
        Per-cell-contiguous gathered positions.  Written by
        ``gather_fused`` each call.
    sorted_shifts : torch.Tensor, shape ``(total_atoms, 3)``, dtype=int32
        Per-cell-contiguous gathered periodic shifts.  Written by
        ``gather_fused`` each call.

    Parameters
    ----------
    total_atoms : int
    dtype : torch.dtype
        Must match the positions dtype passed to ``query_cell_list``.
    device : torch.device | str
    """
    sorted_positions = torch.empty(
        (int(total_atoms), 3),
        dtype=dtype,
        device=device,
    )
    sorted_shifts = torch.empty(
        (int(total_atoms), 3),
        dtype=torch.int32,
        device=device,
    )
    return sorted_positions, sorted_shifts


def estimate_cell_list_sizes(
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    max_nbins: int = 524288,
    min_cells_per_dimension: int = 4,
) -> tuple[int, torch.Tensor]:
    """Estimate allocation sizes for torch.compile-friendly cell list construction.

    Provides conservative estimates for maximum memory allocations needed when
    building cell lists with fixed-size tensors to avoid dynamic allocation
    and graph breaks in torch.compile.

    This function is not torch.compile compatible because it returns an integer
    received from using torch.Tensor.item()

    Parameters
    ----------
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix defining the simulation box.
    pbc : torch.Tensor, shape (3,) or (1, 3), dtype=bool
        Flags indicating periodic boundary conditions in x, y, z directions.
    cutoff : float
        Maximum distance for neighbor search, determines minimum cell size.
    max_nbins : int, default=524288
        Cap on total cells.  When the natural cell-grid (box / cutoff)^3
        exceeds this cap, the kernel halves cells/dim iteratively until it
        fits - which inflates the *atoms-per-cell* count and quadratically
        increases inner-loop work.  Cells/dim arrays cost ~4 MB at this
        cap (2 x max_nbins x 4 bytes).
    min_cells_per_dimension : int, default=4
        Lower bound for the per-axis cell count. Pass 1 for the legacy grid
        rule used by explicit atom-centric benchmarks.

    Returns
    -------
    max_total_cells : int
        Estimated maximum number of cells needed for spatial decomposition.
        For degenerate cells, returns the total number of atoms.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        Radius of neighboring cells to search in each dimension.

    Notes
    -----
    - Cell size is determined by the cutoff distance to ensure neighboring
      cells contain all potential neighbors. The estimation assumes roughly
      cubic cells and uniform atomic distribution.
    - Currently, only unit cells with a positive determinant (i.e. with
      positive volume) are supported. For non-periodic systems, pass an identity
      cell.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Core warp launcher
    allocate_cell_list : Allocates tensors based on these estimates
    build_cell_list : High-level wrapper that uses these estimates
    """
    if max_nbins <= 0:
        raise ValueError("max_nbins must be positive")
    if cell.numel() > 0 and cell.det().abs() == 0.0:
        raise RuntimeError(
            "Cell with volume == 0.0 detected and is not supported."
            " Please pass unit cells with `det(cell) != 0.0`."
        )
    dtype = cell.dtype
    device = cell.device

    if (cell.ndim == 3 and cell.shape[0] == 0) or cutoff <= 0:
        return 1, torch.zeros((3,), dtype=torch.int32, device=device)

    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
    pbc = pbc.reshape(3)

    wp_device = str(device)
    wp_dtype = get_wp_dtype(dtype)
    wp_mat_dtype = get_wp_mat_dtype(dtype)
    wp_cell = wp.from_torch(
        cell, dtype=wp_mat_dtype, requires_grad=False, return_ctype=True
    )
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, requires_grad=False, return_ctype=True)

    max_total_cells = torch.zeros(1, device=device, dtype=torch.int32)
    wp_max_total_cells = wp.from_torch(
        max_total_cells, dtype=wp.int32, requires_grad=False, return_ctype=True
    )

    neighbor_search_radius = torch.zeros((3,), dtype=torch.int32, device=device)
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.int32, requires_grad=False, return_ctype=True
    )

    wp.launch(
        get_build_cell_list_kernel(
            "estimate_sizes",
            wp_dtype,
            min_cells_per_dimension=int(min_cells_per_dimension),
        ),
        dim=1,
        inputs=[
            wp_cell,
            wp_pbc,
            empty_sentinel(2, wp.bool, wp_device),
            wp_dtype(cutoff),
            max_nbins,
            wp_max_total_cells,
            wp_neighbor_search_radius,
            empty_sentinel(1, wp.vec3i, wp_device),
        ],
        device=wp_device,
    )

    return (
        max_total_cells.item(),
        neighbor_search_radius,
    )


@torch.library.custom_op(
    "nvalchemiops::build_cell_list",
    mutates_args=(
        "cells_per_dimension",
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "atoms_per_cell_count",
        "cell_atom_start_indices",
        "cell_atom_list",
    ),
)
def _build_cell_list_op(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    min_cells_per_dimension: int = 4,
) -> None:
    """Internal custom op for building spatial cell list.

    This function is torch compilable.

    Notes
    -----
    The neighbor_search_radius is not an input parameter because it's computed
    internally by the warp launcher and doesn't need to be passed in.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Core warp launcher
    build_cell_list : High-level wrapper function
    """
    total_atoms = positions.shape[0]
    device = positions.device

    # Handle empty case
    if total_atoms == 0:
        return

    cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pbc = pbc.reshape(3)

    # Get warp dtypes and arrays
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_device = str(device)

    wp_positions = wp.from_torch(
        positions, dtype=wp_vec_dtype, requires_grad=False, return_ctype=True
    )
    wp_cell = wp.from_torch(
        cell, dtype=wp_mat_dtype, requires_grad=False, return_ctype=True
    )
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, requires_grad=False, return_ctype=True)

    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_atom_periodic_shifts = wp.from_torch(
        atom_periodic_shifts, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    # underlying warp launcher relies on Python API for array_scan
    # so `return_ctype` is omitted
    wp_atoms_per_cell_count = wp.from_torch(
        atoms_per_cell_count, dtype=wp.int32, requires_grad=False
    )
    wp_cell_atom_start_indices = wp.from_torch(
        cell_atom_start_indices, dtype=wp.int32, requires_grad=False
    )
    wp_cell_atom_list = wp.from_torch(
        cell_atom_list, dtype=wp.int32, requires_grad=False, return_ctype=True
    )

    atoms_per_cell_count.zero_()
    wp_build_cell_list(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=cutoff,
        cells_per_dimension=wp_cells_per_dimension,
        atom_periodic_shifts=wp_atom_periodic_shifts,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        atoms_per_cell_count=wp_atoms_per_cell_count,
        cell_atom_start_indices=wp_cell_atom_start_indices,
        cell_atom_list=wp_cell_atom_list,
        wp_dtype=wp_dtype,
        device=wp_device,
        min_cells_per_dimension=int(min_cells_per_dimension),
    )


@_build_cell_list_op.register_fake
def _(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    min_cells_per_dimension: int = 4,
) -> None:
    return None


def build_cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    min_cells_per_dimension: int = 4,
) -> None:
    """Build spatial cell list with fixed allocation sizes for torch.compile compatibility.

    Constructs a spatial decomposition data structure for efficient neighbor searching.
    Uses fixed-size memory allocations to prevent dynamic tensor creation that would
    cause graph breaks in torch.compile.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates in Cartesian space where total_atoms is the number of atoms.
        Must be float32, float64, or float16 dtype.
    cutoff : float
        Maximum distance for neighbor search. Determines minimum cell size.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix defining the simulation box. Each row represents a
        lattice vector in Cartesian coordinates. Must match positions dtype.
    pbc : torch.Tensor, shape (3,) or (1, 3), dtype=bool
        Flags indicating periodic boundary conditions in x, y, z directions.
        True enables PBC, False disables it for that dimension.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        OUTPUT: Number of cells created in x, y, z directions.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        Radius of neighboring cells to search in each dimension. Passed through
        from allocate_cell_list for API continuity but not used in this function.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        OUTPUT: Periodic boundary crossings for each atom.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        OUTPUT: 3D cell coordinates assigned to each atom.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        OUTPUT: Number of atoms in each cell. Only first 'total_cells' entries are valid.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        OUTPUT: Starting index in cell_atom_list for each cell's atoms.
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32
        OUTPUT: Flattened list of atom indices organized by cell. Use with start_indices
        to extract atoms for each cell.
    min_cells_per_dimension : int, default=4
        Lower bound for the per-axis cell count. Pass 1 for the legacy grid
        rule used by explicit atom-centric benchmarks.

    Notes
    -----
    - This function is torch.compile compatible and uses only static tensor shapes
    - Memory usage is determined by max_total_cells
    - For optimal performance, use estimates from estimate_cell_list_sizes()
    - Cell list must be rebuilt when atoms move between cells or PBC/cell changes

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Core warp launcher
    estimate_cell_list_sizes : Estimate memory requirements
    query_cell_list : Query the built cell list for neighbors
    cell_list : High-level function that builds and queries in one call
    """
    return _build_cell_list_op(
        positions,
        cutoff,
        cell,
        pbc,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        min_cells_per_dimension,
    )


@torch.library.custom_op(
    "nvalchemiops::query_cell_list",
    mutates_args=("neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"),
)
def _query_cell_list_op(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
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
    algorithm: str = "auto",
    atom_centric_path: str = "auto",
    sorted_positions: torch.Tensor | None = None,
    sorted_shifts: torch.Tensor | None = None,
) -> None:
    """Internal custom op for querying spatial cell list to build neighbor matrix.

    This function is torch compilable.

    When ``fill_value`` is provided and ``rebuild_flags`` is None, the
    operation also writes ``fill_value`` into ``neighbor_matrix[i,
    num_neighbors[i]..max_neighbors-1]`` after the query kernel, letting
    callers skip ``neighbor_matrix.fill_(fill_value) +
    neighbor_matrix_shifts.zero_()`` (~60% of the per-step CUDA time at
    large N + cutoff).  ``neighbor_matrix_shifts`` is intentionally NOT
    tail-filled - downstream consumers gate on
    ``neighbor_matrix != fill_value`` and never read tail entries.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.query_cell_list : Core warp launcher
    query_cell_list : High-level wrapper function
    """
    total_atoms = positions.shape[0]
    device = positions.device
    strategy = algorithm

    # Handle empty case
    if total_atoms == 0:
        return

    cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pbc = pbc.reshape(3)

    # Get warp dtypes and arrays
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_device = str(device)

    wp_positions = wp.from_torch(
        positions, dtype=wp_vec_dtype, requires_grad=False, return_ctype=True
    )
    wp_cell = wp.from_torch(
        cell, dtype=wp_mat_dtype, requires_grad=False, return_ctype=True
    )
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, requires_grad=False, return_ctype=True)

    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_atom_periodic_shifts = wp.from_torch(
        atom_periodic_shifts, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_atoms_per_cell_count = wp.from_torch(
        atoms_per_cell_count, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_cell_atom_start_indices = wp.from_torch(
        cell_atom_start_indices, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_cell_atom_list = wp.from_torch(
        cell_atom_list, dtype=wp.int32, requires_grad=False, return_ctype=True
    )

    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix_shifts = wp.from_torch(
        neighbor_matrix_shifts, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(
        num_neighbors, dtype=wp.int32, requires_grad=False, return_ctype=True
    )

    if rebuild_flags is not None:
        wp_rebuild_flags = wp.from_torch(
            rebuild_flags, dtype=wp.bool, requires_grad=False, return_ctype=True
        )
        selective_zero_num_neighbors_single(
            wp_num_neighbors, wp_rebuild_flags, wp_device
        )
    else:
        wp_rebuild_flags = None

    # Pair-centric kernels are CUDA-only; see :func:`select_cell_list_strategy`.
    cpu_only = device.type == "cpu"
    if strategy == "auto":
        chosen = (
            "atom_centric"
            if cpu_only
            else select_cell_list_strategy(int(total_atoms), float(cutoff))
        )
    elif strategy == "atom_centric":
        chosen = "atom_centric"
    elif strategy == "pair_centric":
        if cpu_only:
            raise ValueError(
                "strategy='pair_centric' is not supported on CPU "
                "(kernels use CUDA block scheduling).  Pass 'auto' or "
                "'atom_centric' instead.",
            )
        chosen = "pair_centric"
    else:
        raise ValueError(
            f"strategy must be 'auto' | 'atom_centric' | 'pair_centric', "
            f"got {strategy!r}",
        )
    use_pair = chosen == "pair_centric"
    atom_centric_path = _resolve_atom_centric_path(atom_centric_path)

    # Caller-allocated sort scratch - both or neither.  Mixed state raises so
    # a half-graph capture can't silently fall back to an internal allocation.
    _sort_set = {sorted_positions is not None, sorted_shifts is not None}
    if len(_sort_set) != 1:
        raise ValueError(
            "Pass both sorted_positions and sorted_shifts, or neither - "
            "got a mixed state.",
        )
    sort_scratch_provided = sorted_positions is not None
    wp_sorted_positions = None
    wp_sorted_shifts = None

    n_outer = None
    if use_pair:
        # n_outer is the only host-side dependency on the per-axis radius;
        # the kernel decodes (dx, dy, dz) on-the-fly via the shared shift-
        # index decoders.  One ``.item()`` sync per call - same cost as the
        # old offset-table path, with no allocation.
        Rx = int(neighbor_search_radius[0].item())
        Ry = int(neighbor_search_radius[1].item())
        Rz = int(neighbor_search_radius[2].item())
        n_outer = compute_batch_pair_centric_n_outer((Rx, Ry, Rz), bool(half_fill))
        total_cells = int(atoms_per_cell_count.shape[0])
        if not is_pair_centric_launch_safe(total_cells, n_outer):
            if strategy == "pair_centric":
                _raise_unsafe_pair_centric_launch(total_cells, n_outer)
            chosen = "atom_centric"
            use_pair = False
            n_outer = None
        elif strategy == "auto" and not is_pair_centric_parallelism_sufficient(
            int(total_atoms), total_cells, n_outer
        ):
            chosen = "atom_centric"
            use_pair = False
            n_outer = None

    needs_sorted = use_pair or atom_centric_path == "sorted"
    if needs_sorted:
        if sort_scratch_provided:
            sorted_positions_t = sorted_positions
            sorted_shifts_t = sorted_shifts
        else:
            sorted_positions_t = torch.empty(
                (int(total_atoms), 3), dtype=positions.dtype, device=device
            )
            sorted_shifts_t = torch.empty(
                (int(total_atoms), 3), dtype=torch.int32, device=device
            )
        wp_sorted_positions = wp.from_torch(
            sorted_positions_t,
            dtype=wp_vec_dtype,
            requires_grad=False,
            return_ctype=True,
        )
        wp_sorted_shifts = wp.from_torch(
            sorted_shifts_t, dtype=wp.vec3i, requires_grad=False, return_ctype=True
        )

    wp_query_cell_list(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=cutoff,
        cells_per_dimension=wp_cells_per_dimension,
        neighbor_search_radius=wp_neighbor_search_radius,
        atom_periodic_shifts=wp_atom_periodic_shifts,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        atoms_per_cell_count=wp_atoms_per_cell_count,
        cell_atom_start_indices=wp_cell_atom_start_indices,
        cell_atom_list=wp_cell_atom_list,
        sorted_positions=wp_sorted_positions,
        sorted_atom_periodic_shifts=wp_sorted_shifts,
        neighbor_matrix=wp_neighbor_matrix,
        neighbor_matrix_shifts=wp_neighbor_matrix_shifts,
        num_neighbors=wp_num_neighbors,
        rebuild_flags=wp_rebuild_flags,
        wp_dtype=wp_dtype,
        device=wp_device,
        half_fill=bool(half_fill),
        strategy=chosen,
        n_outer=n_outer,
        atom_centric_path=atom_centric_path,
    )

    # Coalesced tail fill (CUDA only - the kernel uses wp.launch_tiled which
    # silently mis-runs on CPU; CPU callers prefill in ``cell_list`` above).
    # Skipped when ``rebuild_flags`` is provided - those callers own
    # buffer prefill explicitly.
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


def query_cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
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
    sorted_positions: torch.Tensor | None = None,
    sorted_shifts: torch.Tensor | None = None,
    target_indices: torch.Tensor | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | CompiledPairFn | None = None,
    pair_params: torch.Tensor | None = None,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
) -> None:
    """Query spatial cell list to build neighbor matrix with distance constraints.

    Uses pre-built cell list data structures to efficiently find all atom pairs
    within the specified cutoff distance. Handles periodic boundary conditions
    and returns neighbor matrix format.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates in Cartesian space.
    cutoff : float
        Maximum distance for considering atoms as neighbors.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix for periodic boundary coordinate shifts.
    pbc : torch.Tensor, shape (3,) or (1, 3), dtype=bool
        Periodic boundary condition flags.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        Number of cells in x, y, z directions from build_cell_list.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        Shifts to search from build_cell_list.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom from build_cell_list.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom from build_cell_list.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell from build_cell_list.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        Starting index in cell_atom_list for each cell from build_cell_list.
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell from build_cell_list.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Must be pre-allocated.
    neighbor_matrix_shifts : torch.Tensor, shape (total_atoms, max_neighbors, 3), dtype=int32
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
        Must be pre-allocated.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=int32
        OUTPUT: Number of neighbors found for each atom.
        Must be pre-allocated.
    half_fill : bool, default=False
        If True, only store half of the neighbor relationships.
    rebuild_flags : torch.Tensor, shape () or (1,), dtype=torch.bool, optional
        If provided, controls whether the neighbor list is recomputed.
        When the flag is False the kernel is skipped and the pre-allocated output
        tensors are returned unchanged.  When the flag is True (or when this
        argument is None) the query proceeds as normal.
        Note: providing this argument disables torch.compile compatibility.
    fill_value : int, optional
        If provided AND ``rebuild_flags`` is None, the operation writes
        ``fill_value`` into the unused-column tail of ``neighbor_matrix``
        after the kernel runs, letting callers skip the
        ``neighbor_matrix.fill_(fill_value) + neighbor_matrix_shifts.zero_()``
        prefills.  Drops ~60 % of the per-step CUDA cost at large N/cutoff.
    strategy : {"auto", "atom_centric", "pair_centric"}, default "auto"
        Selects which of the two cell-list query kernels to launch.  See
        :func:`select_cell_list_strategy` for the "auto" rule.  Both strategies
        return identical pair sets for either ``half_fill`` value;
        per-row ordering inside ``neighbor_matrix`` differs.
    atom_centric_path : {"auto", "direct", "sorted"}, default "auto"
        Selects the atom-centric implementation path when
        ``strategy="atom_centric"``.  ``"auto"`` resolves to ``"direct"``.
    sorted_positions, sorted_shifts : torch.Tensor, optional
        Pre-allocated scratch (shape ``(total_atoms, 3)``) used by both
        atom-centric and pair-centric paths.  Allocate via
        :func:`allocate_query_sort_scratch`.  Both or neither.

        When NOT provided, the function allocates a fresh torch tensor
        per call.  Pass the allocated tensors for graphed workflows so
        the captured region does no allocation of its own.

        Graph capture: use ``wp.capture_begin/end`` with stream
        alignment (``wp.ScopedStream(wp.stream_from_torch(side_stream))``).
        ``torch.cuda.graph`` will NOT work because ``build_cell_list``
        invokes ``wp.utils.array_scan`` (CUB) which allocates its
        workspace via ``cudaMallocAsync``; that allocator is not
        permitted by ``torch.cuda.graph`` capture but is fine under
        Warp's stream-capture flavor.
    target_indices : torch.Tensor, shape (num_targets,), dtype=int32, optional
        Restrict central rows to a subset of atom indices.  Output rows are
        compact and follow ``target_indices`` order.
    return_vectors, return_distances : bool, default ``False``
        Write per-pair displacement vectors / distances into
        ``neighbor_vectors`` / ``neighbor_distances``.
    pair_fn : callable, optional
        Module-scope ``@wp.func`` of signature
        ``(r_ij, distance, pair_params, i, j) -> (energy, force)``.
    pair_params : torch.Tensor, shape (num_atoms, num_parameters), optional
        Per-atom pair-function parameters; required with ``pair_fn``.
    neighbor_vectors, neighbor_distances : torch.Tensor, optional
        OUTPUT buffers for per-pair displacements / distances.
    pair_energies, pair_forces : torch.Tensor, optional
        OUTPUT buffers for per-pair energies / forces; required with ``pair_fn``.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.query_cell_list : Core warp launcher
    build_cell_list : Builds the cell list data structures
    cell_list : High-level function that builds and queries in one call
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
        _validate_pair_params_present(pair_fn, pair_params)
        if (
            pair_fn is None
            and pair_params is None
            and pair_energies is None
            and pair_forces is None
        ):
            return _query_cell_list_optional_no_pair_fn_op(
                positions,
                cutoff,
                cell,
                pbc,
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
                sorted_positions,
                sorted_shifts,
                target_indices,
                neighbor_vectors,
                neighbor_distances,
                half_fill,
                fill_value,
                strategy,
                atom_centric_path,
                return_vectors,
                return_distances,
            )
        if is_compiled_pair_fn(pair_fn):
            op = pair_fn.get_or_register(
                "query_cell_list_optional_pair",
                _register_compiled_query_cell_list_optional_pair_op,
            )
            return op(
                positions,
                cutoff,
                cell,
                pbc,
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
                sorted_positions,
                sorted_shifts,
                target_indices,
                neighbor_vectors,
                neighbor_distances,
                pair_params,
                pair_energies,
                pair_forces,
                half_fill,
                fill_value,
                strategy,
                atom_centric_path,
                return_vectors,
                return_distances,
            )
        if torch.compiler.is_compiling():
            raise NotImplementedError(
                "cell_list pair_fn outputs are eager-only because callable Warp "
                "functions cannot cross a torch.library.custom_op schema boundary.",
            )
        # Optional per-neighbor outputs bypass the torch custom op (which
        # cannot carry a callable ``pair_fn``) and call the warp factory
        # directly while preserving the requested strategy.
        _query_cell_list_optional(
            positions,
            cutoff,
            cell,
            pbc,
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
            sorted_positions=sorted_positions,
            sorted_shifts=sorted_shifts,
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
    if (
        not torch.compiler.is_compiling()
        and strategy == "atom_centric"
        and sorted_positions is None
        and sorted_shifts is None
        and _resolve_atom_centric_path(atom_centric_path) == "direct"
    ):
        _query_cell_list_direct_eager(
            positions,
            cutoff,
            cell,
            pbc,
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
            atom_centric_path="direct",
        )
        return None
    return _query_cell_list_op(
        positions,
        cutoff,
        cell,
        pbc,
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
        rebuild_flags,
        fill_value,
        strategy,
        atom_centric_path,
        sorted_positions,
        sorted_shifts,
    )


@_query_cell_list_op.register_fake
def _(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
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
    algorithm: str = "auto",
    atom_centric_path: str = "auto",
    sorted_positions: torch.Tensor | None = None,
    sorted_shifts: torch.Tensor | None = None,
) -> None:
    return None


def _query_cell_list_direct_eager(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
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
    atom_centric_path: str,
) -> None:
    """Eager fast path for explicit atom-centric direct queries.

    This keeps the common benchmark/runtime path off the generic custom-op
    boundary while preserving that boundary for ``torch.compile``.
    """
    total_atoms = positions.shape[0]
    device = positions.device
    if total_atoms == 0:
        return

    cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pbc = pbc.reshape(3)

    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_device = str(device)

    wp_positions = wp.from_torch(
        positions, dtype=wp_vec_dtype, requires_grad=False, return_ctype=True
    )
    wp_cell = wp.from_torch(
        cell, dtype=wp_mat_dtype, requires_grad=False, return_ctype=True
    )
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, requires_grad=False, return_ctype=True)
    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_atom_periodic_shifts = wp.from_torch(
        atom_periodic_shifts, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_atoms_per_cell_count = wp.from_torch(
        atoms_per_cell_count, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_cell_atom_start_indices = wp.from_torch(
        cell_atom_start_indices, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_cell_atom_list = wp.from_torch(
        cell_atom_list, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix_shifts = wp.from_torch(
        neighbor_matrix_shifts, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(
        num_neighbors, dtype=wp.int32, requires_grad=False, return_ctype=True
    )

    if rebuild_flags is not None:
        wp_rebuild_flags = wp.from_torch(
            rebuild_flags, dtype=wp.bool, requires_grad=False, return_ctype=True
        )
        selective_zero_num_neighbors_single(
            wp_num_neighbors, wp_rebuild_flags, wp_device
        )
    else:
        wp_rebuild_flags = None

    wp_query_cell_list(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=float(cutoff),
        cells_per_dimension=wp_cells_per_dimension,
        neighbor_search_radius=wp_neighbor_search_radius,
        atom_periodic_shifts=wp_atom_periodic_shifts,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        atoms_per_cell_count=wp_atoms_per_cell_count,
        cell_atom_start_indices=wp_cell_atom_start_indices,
        cell_atom_list=wp_cell_atom_list,
        neighbor_matrix=wp_neighbor_matrix,
        neighbor_matrix_shifts=wp_neighbor_matrix_shifts,
        num_neighbors=wp_num_neighbors,
        wp_dtype=wp_dtype,
        device=wp_device,
        half_fill=bool(half_fill),
        rebuild_flags=wp_rebuild_flags,
        strategy="atom_centric",
        atom_centric_path=atom_centric_path,
    )

    if fill_value is not None and rebuild_flags is None and wp_device != "cpu":
        max_neighbors = int(neighbor_matrix.shape[1])
        if max_neighbors > 0:
            wp_fill_neighbor_matrix_tail(
                wp_num_neighbors,
                int(neighbor_matrix.shape[0]),
                max_neighbors,
                int(fill_value),
                wp_neighbor_matrix,
                wp_device,
            )


@torch.library.custom_op(
    "nvalchemiops::query_cell_list_optional_no_pair_fn",
    mutates_args=(
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
        "neighbor_vectors",
        "neighbor_distances",
    ),
)
def _query_cell_list_optional_no_pair_fn_op(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
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
    rebuild_flags: torch.Tensor | None,
    sorted_positions: torch.Tensor | None,
    sorted_shifts: torch.Tensor | None,
    target_indices: torch.Tensor | None,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
    half_fill: bool,
    fill_value: int | None,
    strategy: str,
    atom_centric_path: str,
    return_vectors: bool,
    return_distances: bool,
) -> None:
    _query_cell_list_optional(
        positions,
        cutoff,
        cell,
        pbc,
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
        sorted_positions=sorted_positions,
        sorted_shifts=sorted_shifts,
        strategy=strategy,
        atom_centric_path=atom_centric_path,
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=None,
        pair_params=None,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=None,
        pair_forces=None,
    )


@_query_cell_list_optional_no_pair_fn_op.register_fake
def _(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
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
    rebuild_flags: torch.Tensor | None,
    sorted_positions: torch.Tensor | None,
    sorted_shifts: torch.Tensor | None,
    target_indices: torch.Tensor | None,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
    half_fill: bool,
    fill_value: int | None,
    strategy: str,
    atom_centric_path: str,
    return_vectors: bool,
    return_distances: bool,
) -> None:
    return None


def _register_compiled_query_cell_list_optional_pair_op(compiled: CompiledPairFn):
    """Register a pair_fn-specialized cell-list query custom op."""

    @torch.library.custom_op(
        f"nvalchemiops::{compiled.op_name('query_cell_list_optional_pair')}",
        mutates_args=(
            "neighbor_matrix",
            "neighbor_matrix_shifts",
            "num_neighbors",
            "neighbor_vectors",
            "neighbor_distances",
            "pair_energies",
            "pair_forces",
        ),
    )
    def _compiled_query_cell_list_optional_pair(
        positions: torch.Tensor,
        cutoff: float,
        cell: torch.Tensor,
        pbc: torch.Tensor,
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
        rebuild_flags: torch.Tensor | None,
        sorted_positions: torch.Tensor | None,
        sorted_shifts: torch.Tensor | None,
        target_indices: torch.Tensor | None,
        neighbor_vectors: torch.Tensor,
        neighbor_distances: torch.Tensor,
        pair_params: torch.Tensor,
        pair_energies: torch.Tensor,
        pair_forces: torch.Tensor,
        half_fill: bool,
        fill_value: int | None,
        strategy: str,
        atom_centric_path: str,
        return_vectors: bool,
        return_distances: bool,
    ) -> None:
        _query_cell_list_optional(
            positions,
            cutoff,
            cell,
            pbc,
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
            sorted_positions=sorted_positions,
            sorted_shifts=sorted_shifts,
            strategy=strategy,
            atom_centric_path=atom_centric_path,
            target_indices=target_indices,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=compiled.pair_fn,
            pair_params=pair_params,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
        )

    register_noop_fake(_compiled_query_cell_list_optional_pair)
    return _compiled_query_cell_list_optional_pair


def _query_cell_list_optional(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
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
    sorted_positions: torch.Tensor | None,
    sorted_shifts: torch.Tensor | None,
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

    The torch ``@torch.library.custom_op`` boundary cannot carry a
    callable ``pair_fn``; this helper bypasses it and calls
    :func:`wp_query_cell_list` directly.  Caller-supplied scratch + output
    buffers are converted via :func:`wp.from_torch`; omitted scratch is
    allocated fresh as a torch tensor for this call.
    """
    total_atoms = positions.shape[0]
    device = positions.device
    if total_atoms == 0:
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

    cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pbc = pbc.reshape(3)

    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_device = str(device)

    wp_positions = wp.from_torch(
        positions, dtype=wp_vec_dtype, requires_grad=False, return_ctype=True
    )
    wp_cell = wp.from_torch(
        cell, dtype=wp_mat_dtype, requires_grad=False, return_ctype=True
    )
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, requires_grad=False, return_ctype=True)
    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_atom_periodic_shifts = wp.from_torch(
        atom_periodic_shifts, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_atoms_per_cell_count = wp.from_torch(
        atoms_per_cell_count, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_cell_atom_start_indices = wp.from_torch(
        cell_atom_start_indices, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_cell_atom_list = wp.from_torch(
        cell_atom_list, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix_shifts = wp.from_torch(
        neighbor_matrix_shifts, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(
        num_neighbors, dtype=wp.int32, requires_grad=False, return_ctype=True
    )

    if rebuild_flags is not None:
        wp_rebuild_flags = wp.from_torch(
            rebuild_flags, dtype=wp.bool, requires_grad=False, return_ctype=True
        )
        selective_zero_num_neighbors_single(
            wp_num_neighbors, wp_rebuild_flags, wp_device
        )
    else:
        wp_rebuild_flags = None

    wp_sorted_positions = None
    wp_sorted_shifts = None

    # Optional torch buffers -> warp arrays (only when supplied).
    wp_target_indices = (
        wp.from_torch(
            target_indices, dtype=wp.int32, requires_grad=False, return_ctype=True
        )
        if target_indices is not None
        else None
    )
    # Pair-output buffers are validated by ``_prepare_pair_output_args`` in the
    # launcher (which dereferences ``pair_params.dtype``), so they must be real
    # Warp arrays, not ``return_ctype`` launch structs.  ``wp.from_torch`` without
    # ``return_ctype`` still aliases the torch tensor zero-copy, so kernel writes
    # land in the output buffers.
    wp_pair_params = (
        wp.from_torch(pair_params, dtype=wp_dtype, requires_grad=False)
        if pair_params is not None
        else None
    )
    wp_neighbor_vectors = (
        wp.from_torch(neighbor_vectors, dtype=wp_vec_dtype, requires_grad=False)
        if neighbor_vectors is not None
        else None
    )
    wp_neighbor_distances = (
        wp.from_torch(neighbor_distances, dtype=wp_dtype, requires_grad=False)
        if neighbor_distances is not None
        else None
    )
    wp_pair_energies = (
        wp.from_torch(pair_energies, dtype=wp_dtype, requires_grad=False)
        if pair_energies is not None
        else None
    )
    wp_pair_forces = (
        wp.from_torch(pair_forces, dtype=wp_vec_dtype, requires_grad=False)
        if pair_forces is not None
        else None
    )

    atom_centric_path = _resolve_atom_centric_path(atom_centric_path)

    cpu_only = device.type == "cpu"
    if strategy == "auto":
        chosen = (
            "atom_centric"
            if cpu_only
            else select_cell_list_strategy(int(total_atoms), float(cutoff))
        )
    elif strategy == "atom_centric":
        chosen = "atom_centric"
    elif strategy == "pair_centric":
        if cpu_only:
            raise ValueError(
                "strategy='pair_centric' is not supported on CPU "
                "(kernels use CUDA block scheduling).  Pass 'auto' or "
                "'atom_centric' instead.",
            )
        chosen = "pair_centric"
    else:
        raise ValueError(
            f"strategy must be 'auto' | 'atom_centric' | 'pair_centric', "
            f"got {strategy!r}",
        )

    n_outer = None
    if chosen == "pair_centric":
        Rx = int(neighbor_search_radius[0].item())
        Ry = int(neighbor_search_radius[1].item())
        Rz = int(neighbor_search_radius[2].item())
        n_outer = compute_batch_pair_centric_n_outer((Rx, Ry, Rz), bool(half_fill))
        total_cells = int(atoms_per_cell_count.shape[0])
        if not is_pair_centric_launch_safe(total_cells, n_outer):
            if strategy == "pair_centric":
                _raise_unsafe_pair_centric_launch(total_cells, n_outer)
            chosen = "atom_centric"
            n_outer = None
        elif strategy == "auto" and not is_pair_centric_parallelism_sufficient(
            int(total_atoms), total_cells, n_outer
        ):
            chosen = "atom_centric"
            n_outer = None

    if chosen == "pair_centric" or atom_centric_path == "sorted":
        if sorted_positions is None:
            sorted_positions = torch.empty(
                (int(total_atoms), 3), dtype=positions.dtype, device=device
            )
        if sorted_shifts is None:
            sorted_shifts = torch.empty(
                (int(total_atoms), 3), dtype=torch.int32, device=device
            )
        wp_sorted_positions = wp.from_torch(
            sorted_positions,
            dtype=wp_vec_dtype,
            requires_grad=False,
            return_ctype=True,
        )
        wp_sorted_shifts = wp.from_torch(
            sorted_shifts, dtype=wp.vec3i, requires_grad=False, return_ctype=True
        )

    wp_query_cell_list(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=float(cutoff),
        cells_per_dimension=wp_cells_per_dimension,
        neighbor_search_radius=wp_neighbor_search_radius,
        atom_periodic_shifts=wp_atom_periodic_shifts,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        atoms_per_cell_count=wp_atoms_per_cell_count,
        cell_atom_start_indices=wp_cell_atom_start_indices,
        cell_atom_list=wp_cell_atom_list,
        sorted_positions=wp_sorted_positions,
        sorted_atom_periodic_shifts=wp_sorted_shifts,
        neighbor_matrix=wp_neighbor_matrix,
        neighbor_matrix_shifts=wp_neighbor_matrix_shifts,
        num_neighbors=wp_num_neighbors,
        rebuild_flags=wp_rebuild_flags,
        wp_dtype=wp_dtype,
        device=wp_device,
        half_fill=bool(half_fill),
        strategy=chosen,
        n_outer=n_outer,
        atom_centric_path=atom_centric_path,
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


def cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    num_neighbors: torch.Tensor | None = None,
    cells_per_dimension: torch.Tensor | None = None,
    neighbor_search_radius: torch.Tensor | None = None,
    atom_periodic_shifts: torch.Tensor | None = None,
    atom_to_cell_mapping: torch.Tensor | None = None,
    atoms_per_cell_count: torch.Tensor | None = None,
    cell_atom_start_indices: torch.Tensor | None = None,
    cell_atom_list: torch.Tensor | None = None,
    rebuild_flags: torch.Tensor | None = None,
    strategy: str = "auto",
    atom_centric_path: str = "auto",
    sorted_positions: torch.Tensor | None = None,
    sorted_shifts: torch.Tensor | None = None,
    target_indices: torch.Tensor | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | CompiledPairFn | None = None,
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
    """Build complete neighbor matrix using spatial cell list acceleration.

    High-level convenience function that automatically estimates memory requirements,
    builds spatial cell list data structures, and queries them to produce a complete
    neighbor matrix. Combines build_cell_list and query_cell_list operations.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates in Cartesian space where total_atoms is the number of atoms.
    cutoff : float
        Maximum distance for neighbor search.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix defining the simulation box. Each row represents a
        lattice vector in Cartesian coordinates.
    pbc : torch.Tensor, shape (3,) or (1, 3), dtype=bool
        Flags indicating periodic boundary conditions in x, y, z directions.
    max_neighbors : int, optional
        Maximum number of neighbors per atom. If not provided, will be estimated automatically.
    half_fill : bool, optional
        If True, only fill half of the neighbor matrix. Default is False.
    fill_value : int | None, optional
        Value to fill the neighbor matrix with. Default is total_atoms.
    return_neighbor_list : bool, optional - default = False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format by
        creating a mask over the fill_value, which can incur a performance penalty.
        We recommend using the neighbor matrix format,
        and only convert to a neighbor list format if absolutely necessary.
    neighbor_matrix : torch.Tensor, optional
        Pre-allocated tensor of shape (total_atoms, max_neighbors) for neighbor indices.
        If None, allocated internally.
    neighbor_matrix_shifts : torch.Tensor, optional
        Pre-allocated tensor of shape (total_atoms, max_neighbors, 3) for shift vectors.
        If None, allocated internally.
    num_neighbors : torch.Tensor, optional
        Pre-allocated tensor of shape (total_atoms,) for neighbor counts.
        If None, allocated internally.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32, optional
        Number of cells in x, y, z directions.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32, optional
        Radius of neighboring cells to search in each dimension.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32, optional
        Periodic boundary crossings for each atom.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32, optional
        Cell coordinates for each atom.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32, optional
        Number of atoms in each cell.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32, optional
        Starting index in cell_atom_list for each cell.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32, optional
        Flattened list of atom indices organized by cell.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    rebuild_flags : torch.Tensor, shape () or (1,), dtype=torch.bool, optional
        If provided, controls whether the neighbor list is recomputed.
        When the flag is False the existing ``neighbor_matrix``, ``num_neighbors``,
        and ``neighbor_matrix_shifts`` tensors are returned unchanged and all
        kernel launches are skipped.  When the flag is True (or when this argument
        is None) the neighbor list is recomputed as normal.
    strategy : {"auto", "atom_centric", "pair_centric"}, default "auto"
        Cell-list query kernel selection.  Both strategies return identical
        pair sets; per-row ordering inside ``neighbor_matrix`` differs.
        See :func:`nvalchemiops.neighbors.cell_list.select_cell_list_strategy`
        for the ``"auto"`` rule.  Pair-centric is CUDA-only.
    atom_centric_path : {"auto", "direct", "sorted"}, default "auto"
        Atom-centric implementation path.  ``"auto"`` resolves to ``"direct"``.
    sorted_positions, sorted_shifts : torch.Tensor, optional
        Pre-allocated sort-side scratch (shape ``(total_atoms, 3)``).
        Pass both to make the call graph-capture safe.  Allocate via
        :func:`allocate_query_sort_scratch`.  Both or neither.

    Returns
    -------
    results : tuple of torch.Tensor
        Variable-length tuple depending on input parameters. The return pattern follows:

        - Matrix format (default): ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``
        - List format (return_neighbor_list=True): ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``

    Notes
    -----
    - This is the main user-facing API for cell list neighbor construction
    - Uses automatic memory allocation estimation for torch.compile compatibility
    - For advanced users who want to cache cell lists, use build_cell_list and query_cell_list separately
    - Returns appropriate empty tensors for systems with <= 1 atom or cutoff <= 0

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Core warp launcher for building
    nvalchemiops.neighbors.cell_list.query_cell_list : Core warp launcher for querying
    naive_neighbor_list : O(N^2) method for small systems
    """

    total_atoms = positions.shape[0]
    device = positions.device
    if pbc is None:
        raise ValueError(
            "cell_list requires `pbc` to be specified. "
            "Pass a boolean tensor of shape (3,) or (1, 3), "
            "e.g. pbc=torch.tensor([True, True, True])."
        )
    cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pbc = pbc.reshape(3)

    if is_compiled_pair_fn(pair_fn) and torch.compiler.is_compiling():
        if return_neighbor_list:
            raise NotImplementedError(
                "CompiledPairFn supports torch.compile(fullgraph=True) for "
                "matrix neighbor-list output only; use return_neighbor_list=False.",
            )
        missing = [
            name
            for name, value in (
                ("neighbor_matrix", neighbor_matrix),
                ("neighbor_matrix_shifts", neighbor_matrix_shifts),
                ("num_neighbors", num_neighbors),
                ("cells_per_dimension", cells_per_dimension),
                ("neighbor_search_radius", neighbor_search_radius),
                ("atom_periodic_shifts", atom_periodic_shifts),
                ("atom_to_cell_mapping", atom_to_cell_mapping),
                ("atoms_per_cell_count", atoms_per_cell_count),
                ("cell_atom_start_indices", cell_atom_start_indices),
                ("cell_atom_list", cell_atom_list),
                ("sorted_positions", sorted_positions),
                ("sorted_shifts", sorted_shifts),
                ("neighbor_vectors", neighbor_vectors),
                ("neighbor_distances", neighbor_distances),
                ("pair_params", pair_params),
                ("pair_energies", pair_energies),
                ("pair_forces", pair_forces),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "CompiledPairFn under torch.compile(fullgraph=True) requires "
                "fixed-shape caller-provided buffers/metadata; missing "
                f"{', '.join(missing)}.",
            )
    _validate_pair_params_present(pair_fn, pair_params)

    if fill_value is None:
        fill_value = total_atoms
    num_rows = (
        int(target_indices.shape[0]) if target_indices is not None else total_atoms
    )

    # Handle empty case
    if total_atoms <= 0 or cutoff <= 0:
        if return_neighbor_list:
            return (
                torch.zeros((2, 0), dtype=torch.int32, device=device),
                torch.zeros((num_rows + 1,), dtype=torch.int32, device=device),
                torch.zeros((0, 3), dtype=torch.int32, device=device),
            )
        else:
            return (
                torch.full((num_rows, 0), fill_value, dtype=torch.int32, device=device),
                torch.zeros((num_rows,), dtype=torch.int32, device=device),
                torch.zeros((num_rows, 0, 3), dtype=torch.int32, device=device),
            )

    if max_neighbors is None and (
        neighbor_matrix is None
        or neighbor_matrix_shifts is None
        or num_neighbors is None
    ):
        max_neighbors = estimate_max_neighbors(cutoff)

    # CPU prefills; CUDA tail-fills (``wp.launch_tiled`` mis-runs on CPU).
    is_cpu = str(device) == "cpu"
    if neighbor_matrix is None:
        if is_cpu:
            neighbor_matrix = torch.full(
                (num_rows, max_neighbors),
                fill_value,
                dtype=torch.int32,
                device=device,
            )
        else:
            neighbor_matrix = torch.empty(
                (num_rows, max_neighbors), dtype=torch.int32, device=device
            )
    elif is_cpu and rebuild_flags is None:
        neighbor_matrix.fill_(fill_value)
    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = torch.empty(
            (num_rows, max_neighbors, 3), dtype=torch.int32, device=device
        )
    if num_neighbors is None:
        num_neighbors = torch.zeros((num_rows,), dtype=torch.int32, device=device)
    elif rebuild_flags is None:
        num_neighbors.zero_()

    # Allocate cell list if needed. Explicit atom-centric queries use the
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
        max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell,
            pbc,
            cutoff,
            min_cells_per_dimension=cell_list_min_cells,
        )
        cell_list_cache = allocate_cell_list(
            total_atoms,
            max_total_cells,
            neighbor_search_radius,
            device,
        )
    else:
        # Caller-provided caches are assumed to have been sized with the
        # default public estimate policy.
        cell_list_min_cells = 4
        cells_per_dimension.zero_()
        atom_periodic_shifts.zero_()
        atom_to_cell_mapping.zero_()
        atoms_per_cell_count.zero_()
        cell_atom_start_indices.zero_()
        cell_atom_list.zero_()
        cell_list_cache = (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )

    build_cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        *cell_list_cache,
        min_cells_per_dimension=cell_list_min_cells,
    )

    if return_vectors or return_distances or pair_fn is not None:
        # Pair_fn receives distance/vector values as local kernel variables;
        # these matrix buffers are only public geometry outputs.
        if return_distances and neighbor_distances is None:
            neighbor_distances = torch.zeros(
                (num_rows, max_neighbors), dtype=positions.dtype, device=device
            )
        if return_vectors and neighbor_vectors is None:
            neighbor_vectors = torch.zeros(
                (num_rows, max_neighbors, 3),
                dtype=positions.dtype,
                device=device,
            )
        # ``pair_fn`` energy/force buffers are optional: allocate them like the
        # neighbor matrix when the caller did not supply them, so they can be
        # returned.
        if pair_fn is not None and pair_energies is None:
            pair_energies = torch.zeros(
                (num_rows, max_neighbors), dtype=positions.dtype, device=device
            )
        if pair_fn is not None and pair_forces is None:
            pair_forces = torch.zeros(
                (num_rows, max_neighbors, 3), dtype=positions.dtype, device=device
            )
        forward_kwargs = {
            "cutoff": cutoff,
            "pbc": pbc,
            "cell_list_cache": cell_list_cache,
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
            "num_neighbors": num_neighbors,
            "half_fill": half_fill,
            "rebuild_flags": rebuild_flags,
            "fill_value": fill_value,
            "strategy": strategy,
            "atom_centric_path": atom_centric_path,
            "sorted_positions": sorted_positions,
            "sorted_shifts": sorted_shifts,
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
            _cell_list_query_forward,
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
            # Repack the per-pair outputs into the same COO order as the
            # neighbor list so they index-align with ``nl``; ``index_select``
            # keeps the autograd link.  ``pair_fn`` also fills the caller's
            # matrix buffers in place (those stay matrix layout).
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

    query_cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        *cell_list_cache,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
        rebuild_flags,
        fill_value,
        strategy,
        atom_centric_path,
        sorted_positions,
        sorted_shifts,
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


def _cell_list_query_forward(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    *,
    cutoff: float,
    pbc: torch.Tensor,
    cell_list_cache: tuple,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    half_fill: bool,
    rebuild_flags: torch.Tensor | None,
    fill_value: int,
    strategy: str,
    atom_centric_path: str,
    sorted_positions: torch.Tensor | None,
    sorted_shifts: torch.Tensor | None,
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
    """Forward closure consumed by ``_NeighborDistanceVectorFn``.

    Runs the existing ``query_cell_list`` warp launcher (which writes into the
    pre-allocated output buffers) and then flattens the active matrix slots
    into the per-pair index arrays the backward needs.  The warp kernel does
    not participate in torch autograd; differentiability is added by the
    Function's reconstruction-based backward.
    """
    query_cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        *cell_list_cache,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
        rebuild_flags,
        fill_value,
        strategy,
        atom_centric_path,
        sorted_positions,
        sorted_shifts,
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
