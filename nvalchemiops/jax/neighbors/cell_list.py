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

"""JAX bindings for unbatched cell list O(N) neighbor list construction."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import GraphMode, jax_callable, jax_kernel

from nvalchemiops.jax.neighbors.neighbor_utils import (
    _validate_graph_mode,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.neighbors.cell_list import (
    _cell_list_bin_atoms_overload,
    _cell_list_build_neighbor_matrix_overload,
    _cell_list_build_neighbor_matrix_selective_overload,
    _cell_list_construct_bin_size_overload,
    _cell_list_count_atoms_per_bin_overload,
)
from nvalchemiops.neighbors.cell_list import (
    build_cell_list as _warp_build_cell_list,
)
from nvalchemiops.neighbors.cell_list import (
    query_cell_list as _warp_query_cell_list,
)
from nvalchemiops.neighbors.neighbor_utils import (
    _selective_zero_num_neighbors_single,
    estimate_max_neighbors,
)

# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# Build step 1: Construct bin sizes
_jax_construct_bin_size_f32 = jax_kernel(
    _cell_list_construct_bin_size_overload[wp.float32],
    num_outputs=1,
    in_out_argnames=["cells_per_dimension"],
    enable_backward=False,
)
_jax_construct_bin_size_f64 = jax_kernel(
    _cell_list_construct_bin_size_overload[wp.float64],
    num_outputs=1,
    in_out_argnames=["cells_per_dimension"],
    enable_backward=False,
)

# Build step 2: Count atoms per bin
_jax_count_atoms_per_bin_f32 = jax_kernel(
    _cell_list_count_atoms_per_bin_overload[wp.float32],
    num_outputs=2,
    in_out_argnames=["atoms_per_cell_count", "atom_periodic_shifts"],
    enable_backward=False,
)
_jax_count_atoms_per_bin_f64 = jax_kernel(
    _cell_list_count_atoms_per_bin_overload[wp.float64],
    num_outputs=2,
    in_out_argnames=["atoms_per_cell_count", "atom_periodic_shifts"],
    enable_backward=False,
)

# Build step 3: Bin atoms into cells
_jax_bin_atoms_f32 = jax_kernel(
    _cell_list_bin_atoms_overload[wp.float32],
    num_outputs=3,
    in_out_argnames=["atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_list"],
    enable_backward=False,
)
_jax_bin_atoms_f64 = jax_kernel(
    _cell_list_bin_atoms_overload[wp.float64],
    num_outputs=3,
    in_out_argnames=["atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_list"],
    enable_backward=False,
)

# Query: Build neighbor matrix from cell list
_jax_build_neighbor_matrix_f32 = jax_kernel(
    _cell_list_build_neighbor_matrix_overload[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)
_jax_build_neighbor_matrix_f64 = jax_kernel(
    _cell_list_build_neighbor_matrix_overload[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)

# Selective query: Build neighbor matrix from cell list (skips non-rebuilt systems)
_jax_build_neighbor_matrix_selective_f32 = jax_kernel(
    _cell_list_build_neighbor_matrix_selective_overload[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)
_jax_build_neighbor_matrix_selective_f64 = jax_kernel(
    _cell_list_build_neighbor_matrix_selective_overload[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)

__all__ = [
    "cell_list",
    "build_cell_list",
    "query_cell_list",
    "estimate_cell_list_sizes",
]


def _reset_query_outputs(
    neighbor_matrix,
    neighbor_matrix_shifts,
    num_neighbors,
    fill_value,
) -> None:
    """Reset query outputs inside the Warp callback."""
    neighbor_matrix.fill_(fill_value)
    num_neighbors.zero_()
    neighbor_matrix_shifts.zero_()


def _run_graph_build_cell_list(
    positions,
    cell,
    pbc,
    cells_per_dimension,
    atom_periodic_shifts,
    atom_to_cell_mapping,
    atoms_per_cell_count,
    cell_atom_start_indices,
    cell_atom_list,
    cutoff,
    wp_dtype,
) -> None:
    """Execute the fused cell-list build callback."""
    atoms_per_cell_count.zero_()
    # Graph-capture contract: ``_warp_build_cell_list`` is the Python-level
    # Warp launcher and takes a device string. Inside this jax_callable body
    # the device is a Python constant captured once at CUDA-graph capture
    # time, so subsequent replays reuse the same device. If the launcher
    # signature ever grows a stream/context parameter, that argument must
    # also be hoisted out of the per-replay path here.
    _warp_build_cell_list(
        positions,
        cell,
        pbc,
        cutoff,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        wp_dtype,
        str(positions.device),
    )


def _run_graph_query_cell_list(
    positions,
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
    cutoff,
    fill_value,
    wp_dtype,
    rebuild_flags=None,
) -> None:
    """Execute the fused cell-list query callback."""
    if rebuild_flags is None:
        _reset_query_outputs(
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            fill_value,
        )
    else:
        wp.launch(
            kernel=_selective_zero_num_neighbors_single,
            dim=num_neighbors.shape[0],
            inputs=[num_neighbors, rebuild_flags],
        )

    # Graph-capture contract: ``_warp_query_cell_list`` is the Python-level
    # Warp launcher and takes a device string. Inside this jax_callable body
    # the device is a Python constant captured once at CUDA-graph capture
    # time, so subsequent replays reuse the same device. If the launcher
    # signature ever grows a stream/context parameter, that argument must
    # also be hoisted out of the per-replay path here.
    _warp_query_cell_list(
        positions,
        cell,
        pbc,
        cutoff,
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
        wp_dtype,
        str(positions.device),
        half_fill=False,
        rebuild_flags=rebuild_flags,
    )


def _run_graph_cell_list(
    positions,
    cell,
    pbc,
    neighbor_search_radius,
    cells_per_dimension,
    atom_periodic_shifts,
    atom_to_cell_mapping,
    atoms_per_cell_count,
    cell_atom_start_indices,
    cell_atom_list,
    neighbor_matrix,
    neighbor_matrix_shifts,
    num_neighbors,
    cutoff,
    fill_value,
    wp_dtype,
) -> None:
    """Execute the fused cell-list build+query callback."""
    _run_graph_build_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        cutoff,
        wp_dtype,
    )
    _run_graph_query_cell_list(
        positions,
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
        cutoff,
        fill_value,
        wp_dtype,
    )


def _graph_build_cell_list_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
) -> None:
    _run_graph_build_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        cutoff,
        wp.float32,
    )


def _graph_build_cell_list_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
) -> None:
    _run_graph_build_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        cutoff,
        wp.float64,
    )


def _graph_query_cell_list_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    fill_value: wp.int32,
) -> None:
    _run_graph_query_cell_list(
        positions,
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
        cutoff,
        fill_value,
        wp.float32,
    )


def _graph_query_cell_list_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    fill_value: wp.int32,
) -> None:
    _run_graph_query_cell_list(
        positions,
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
        cutoff,
        fill_value,
        wp.float64,
    )


def _graph_query_cell_list_selective_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    fill_value: wp.int32,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    _run_graph_query_cell_list(
        positions,
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
        cutoff,
        fill_value,
        wp.float32,
        rebuild_flags=rebuild_flags,
    )


def _graph_query_cell_list_selective_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    fill_value: wp.int32,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    _run_graph_query_cell_list(
        positions,
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
        cutoff,
        fill_value,
        wp.float64,
        rebuild_flags=rebuild_flags,
    )


def _graph_cell_list_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    cells_per_dimension: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    fill_value: wp.int32,
) -> None:
    _run_graph_cell_list(
        positions,
        cell,
        pbc,
        neighbor_search_radius,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp.float32,
    )


def _graph_cell_list_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    cells_per_dimension: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    fill_value: wp.int32,
) -> None:
    _run_graph_cell_list(
        positions,
        cell,
        pbc,
        neighbor_search_radius,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp.float64,
    )


_jax_graph_build_cell_list_f32 = jax_callable(
    _graph_build_cell_list_f32,
    num_outputs=6,
    in_out_argnames=[
        "cells_per_dimension",
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "atoms_per_cell_count",
        "cell_atom_start_indices",
        "cell_atom_list",
    ],
    graph_mode=GraphMode.WARP,
)
_jax_graph_build_cell_list_f64 = jax_callable(
    _graph_build_cell_list_f64,
    num_outputs=6,
    in_out_argnames=[
        "cells_per_dimension",
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "atoms_per_cell_count",
        "cell_atom_start_indices",
        "cell_atom_list",
    ],
    graph_mode=GraphMode.WARP,
)
_jax_graph_query_cell_list_f32 = jax_callable(
    _graph_query_cell_list_f32,
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    graph_mode=GraphMode.WARP,
)
_jax_graph_query_cell_list_f64 = jax_callable(
    _graph_query_cell_list_f64,
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    graph_mode=GraphMode.WARP,
)
_jax_graph_query_cell_list_selective_f32 = jax_callable(
    _graph_query_cell_list_selective_f32,
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    graph_mode=GraphMode.WARP,
)
_jax_graph_query_cell_list_selective_f64 = jax_callable(
    _graph_query_cell_list_selective_f64,
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    graph_mode=GraphMode.WARP,
)
_jax_graph_cell_list_f32 = jax_callable(
    _graph_cell_list_f32,
    num_outputs=9,
    in_out_argnames=[
        "cells_per_dimension",
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "atoms_per_cell_count",
        "cell_atom_start_indices",
        "cell_atom_list",
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
    ],
    graph_mode=GraphMode.WARP,
)
_jax_graph_cell_list_f64 = jax_callable(
    _graph_cell_list_f64,
    num_outputs=9,
    in_out_argnames=[
        "cells_per_dimension",
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "atoms_per_cell_count",
        "cell_atom_start_indices",
        "cell_atom_list",
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
    ],
    graph_mode=GraphMode.WARP,
)


def estimate_cell_list_sizes(
    positions: jax.Array,
    cell: jax.Array,
    cutoff: float,
    pbc: jax.Array | None = None,
    buffer_factor: float = 1.5,
) -> tuple[int, jax.Array, jax.Array]:
    """Estimate required cell list sizes based on atomic density.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space.
    cell : jax.Array, shape (1, 3, 3), dtype=float32 or float64
        Cell matrix defining lattice vectors.
    cutoff : float
        Cutoff distance for neighbor searching.
    pbc : jax.Array, shape (3,) or (1, 3), dtype=bool, optional
        Periodic boundary condition flags. Default is all True.
    buffer_factor : float, optional
        Buffer multiplier for cell count estimation. Default is 1.5.

    Returns
    -------
    max_total_cells : int
        Maximum total number of cells to allocate.
    cells_per_dimension : jax.Array, shape (3,) or (1, 3), dtype=int32
        Estimated number of cells in each dimension.
    neighbor_search_radius : jax.Array, shape (3,), dtype=int32
        Estimated search radius in neighboring cells.

    Notes
    -----
    This function estimates cell list parameters based on atomic positions and
    density. The actual number of cells used will be determined during cell
    list construction.

    .. warning::

        This function is **not compatible with** ``jax.jit``. The returned
        ``max_total_cells`` is used to determine array allocation sizes, which
        must be concrete (statically known) at JAX trace time. When using
        ``cell_list`` or ``build_cell_list`` inside ``jax.jit``, provide
        ``max_total_cells`` explicitly to bypass this function.
    """
    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    if pbc is None:
        pbc = jnp.ones((1, 3), dtype=jnp.bool_)
    if pbc.ndim == 1:
        pbc = pbc[jnp.newaxis, :]

    # Simple estimation: compute total volume and estimate cell volume
    # Cell volume = det(cell_matrix)
    det = jnp.linalg.det(cell[0])
    volume = jnp.abs(det)
    cell_volume = cutoff**3
    # TODO: This estimation derives array sizes from traced input data (cell
    # geometry), which is fundamentally incompatible with jax.jit compilation.
    # The JAX bindings need a refactored usage pattern where sizing is always
    # performed outside the JIT boundary, or a fixed upper-bound allocation
    # strategy is adopted.
    num_cells_est = jnp.int32(volume / cell_volume * buffer_factor)
    max_total_cells = jnp.max(jnp.array([num_cells_est, 8]))  # Minimum 8 cells

    # Compute cells_per_dimension and neighbor_search_radius from cell geometry,
    # mirroring the Warp _estimate_cell_list_sizes kernel used by the Torch path.
    inverse_cell_transpose = jnp.linalg.inv(cell[0]).T
    face_distances = 1.0 / jnp.linalg.norm(inverse_cell_transpose, axis=1)
    cells_per_dimension = jnp.maximum(jnp.int32(face_distances / cutoff), 1)

    pbc_squeezed = pbc.squeeze()[:3] if pbc.ndim > 1 else pbc[:3]
    neighbor_search_radius = jnp.where(
        (cells_per_dimension == 1) & ~pbc_squeezed,
        jnp.zeros(3, dtype=jnp.int32),
        jnp.int32(jnp.ceil(cutoff * cells_per_dimension / face_distances)),
    )

    return max_total_cells, cells_per_dimension, neighbor_search_radius


def build_cell_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array,
    pbc: jax.Array,
    cells_per_dimension: jax.Array | None = None,
    neighbor_search_radius: jax.Array | None = None,
    atom_periodic_shifts: jax.Array | None = None,
    atom_to_cell_mapping: jax.Array | None = None,
    atoms_per_cell_count: jax.Array | None = None,
    cell_atom_start_indices: jax.Array | None = None,
    cell_atom_list: jax.Array | None = None,
    max_total_cells: int | None = None,
    graph_mode: Literal["none", "warp"] = "none",
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Build spatial cell list for efficient neighbor searching.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor searching. Must be positive.
    cell : jax.Array, shape (1, 3, 3), dtype=float32 or float64
        Cell matrix defining lattice vectors.
    pbc : jax.Array, shape (3,) or (1, 3), dtype=bool
        Periodic boundary condition flags.
    cells_per_dimension : jax.Array, shape (3,), dtype=int32, optional
        OUTPUT: Number of cells in x, y, z directions. If None, allocated.
    neighbor_search_radius : jax.Array, shape (3,), dtype=int32, optional
        Search radius in neighboring cells. If None, allocated.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32, optional
        OUTPUT: Periodic boundary crossings for each atom. If None, allocated.
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32, optional
        OUTPUT: 3D cell coordinates for each atom. If None, allocated.
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32, optional
        OUTPUT: Number of atoms in each cell. If None, allocated.
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32, optional
        OUTPUT: Starting index in cell_atom_list for each cell. If None, allocated.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32, optional
        OUTPUT: Flattened list of atom indices organized by cell. If None, allocated.
    max_total_cells : int, optional
        Maximum number of cells to allocate. If None, will be estimated.

    Returns
    -------
    cells_per_dimension : jax.Array, shape (3,), dtype=int32
        Number of cells in x, y, z directions.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom.
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom.
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell.
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32
        Starting index in cell_atom_list for each cell.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell.
    neighbor_search_radius : jax.Array, shape (3,), dtype=int32
        Search radius in neighboring cells.

    Notes
    -----
    When calling inside ``jax.jit``, ``max_total_cells`` **must** be provided
    to avoid calling ``estimate_cell_list_sizes``, which is not JIT-compatible.

    ``graph_mode="warp"`` uses a fused ``jax_callable`` that captures the full
    Warp-side build sequence. For replay-friendly usage inside ``jax.jit``,
    donate and reuse the optional cell-list buffers.

    See Also
    --------
    query_cell_list : Query the built cell list for neighbors
    """
    graph_mode = _validate_graph_mode(graph_mode)

    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    if pbc.ndim == 1:
        pbc = pbc[jnp.newaxis, :]

    if max_total_cells is None:
        max_total_cells, _, neighbor_search_radius_est = estimate_cell_list_sizes(
            positions, cell, cutoff, pbc
        )
        if neighbor_search_radius is None:
            neighbor_search_radius = neighbor_search_radius_est
    else:
        if neighbor_search_radius is None:
            neighbor_search_radius = jnp.ones(3, dtype=jnp.int32)

    # Allocate cell list tensors if not provided
    if cells_per_dimension is None:
        cells_per_dimension = jnp.ones(3, dtype=jnp.int32)
    if atom_periodic_shifts is None:
        atom_periodic_shifts = jnp.zeros((positions.shape[0], 3), dtype=jnp.int32)
    if atom_to_cell_mapping is None:
        atom_to_cell_mapping = jnp.zeros((positions.shape[0], 3), dtype=jnp.int32)
    if atoms_per_cell_count is None:
        atoms_per_cell_count = jnp.zeros(max_total_cells, dtype=jnp.int32)
    elif graph_mode == "none":
        atoms_per_cell_count = atoms_per_cell_count.at[:].set(jnp.int32(0))
    if cell_atom_start_indices is None:
        cell_atom_start_indices = jnp.zeros(max_total_cells, dtype=jnp.int32)
    if cell_atom_list is None:
        cell_atom_list = jnp.zeros(positions.shape[0], dtype=jnp.int32)

    # Select kernels based on dtype
    if positions.dtype == jnp.float64:
        _construct_bin_size = _jax_construct_bin_size_f64
        _count_atoms = _jax_count_atoms_per_bin_f64
        _bin_atoms = _jax_bin_atoms_f64
    else:
        _construct_bin_size = _jax_construct_bin_size_f32
        _count_atoms = _jax_count_atoms_per_bin_f32
        _bin_atoms = _jax_bin_atoms_f32
        positions = positions.astype(jnp.float32)

    # Ensure cell dtype matches positions dtype so warp overload dispatch is consistent
    if cell.dtype != positions.dtype:
        cell = cell.astype(positions.dtype)

    total_atoms = positions.shape[0]

    # Squeeze pbc to 1D for kernel (kernels expect shape (3,))
    pbc_1d = pbc.squeeze() if pbc.ndim == 2 else pbc
    pbc_bool = pbc_1d.astype(jnp.bool_)

    if graph_mode == "warp":
        graph_build = (
            _jax_graph_build_cell_list_f64
            if positions.dtype == jnp.float64
            else _jax_graph_build_cell_list_f32
        )
        (
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = graph_build(
            positions,
            cell,
            pbc_bool,
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            float(cutoff),
        )
    else:
        # Step 1: Construct bin sizes
        (cells_per_dimension,) = _construct_bin_size(
            cell,
            pbc_bool,
            cells_per_dimension,
            float(cutoff),
            int(max_total_cells),
            launch_dims=(1,),
        )

        # Step 2: Count atoms per bin
        atoms_per_cell_count, atom_periodic_shifts = _count_atoms(
            positions,
            cell,
            pbc_bool,
            cells_per_dimension,
            atoms_per_cell_count,
            atom_periodic_shifts,
            launch_dims=(total_atoms,),
        )

        # Step 3: Compute exclusive prefix sum (replaces wp.utils.array_scan)
        cell_atom_start_indices = jnp.concatenate(
            [
                jnp.array([0], dtype=jnp.int32),
                jnp.cumsum(atoms_per_cell_count[:-1], dtype=jnp.int32),
            ]
        )

        # Step 4: Zero counts before second pass
        atoms_per_cell_count = jnp.zeros_like(atoms_per_cell_count)

        # Step 5: Bin atoms
        atom_to_cell_mapping, atoms_per_cell_count, cell_atom_list = _bin_atoms(
            positions,
            cell,
            pbc_bool,
            cells_per_dimension,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            launch_dims=(total_atoms,),
        )

    return (
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_search_radius,
    )


def query_cell_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array,
    pbc: jax.Array,
    cells_per_dimension: jax.Array,
    atom_periodic_shifts: jax.Array,
    atom_to_cell_mapping: jax.Array,
    atoms_per_cell_count: jax.Array,
    cell_atom_start_indices: jax.Array,
    cell_atom_list: jax.Array,
    neighbor_search_radius: jax.Array,
    max_neighbors: int | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
    rebuild_flags: jax.Array | None = None,
    graph_mode: Literal["none", "warp"] = "none",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Query cell list to find neighbors within cutoff.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection.
    cell : jax.Array, shape (1, 3, 3), dtype=float32 or float64
        Cell matrix defining lattice vectors.
    pbc : jax.Array, shape (3,) or (1, 3), dtype=bool
        Periodic boundary condition flags.
    cells_per_dimension : jax.Array, shape (3,), dtype=int32
        Number of cells in each dimension.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom (output from ``build_cell_list``).
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom.
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell (output from ``build_cell_list``).
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32
        Starting index in cell_atom_list for each cell.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell.
    neighbor_search_radius : jax.Array, shape (3,), dtype=int32
        Search radius in neighboring cells.
    max_neighbors : int, optional
        Maximum number of neighbors per atom.
    neighbor_matrix : jax.Array, optional
        Pre-allocated neighbor matrix.
    num_neighbors : jax.Array, optional
        Pre-allocated neighbors count array.

    Returns
    -------
    neighbor_matrix : jax.Array, shape (total_atoms, max_neighbors), dtype=int32
        Neighbor matrix with neighbor atom indices.
    num_neighbors : jax.Array, shape (total_atoms,), dtype=int32
        Number of neighbors found for each atom.
    neighbor_matrix_shifts : jax.Array, shape (total_atoms, max_neighbors, 3), dtype=int32
        Periodic shift vectors for each neighbor relationship.

    See Also
    --------
    build_cell_list : Build cell list before querying
    cell_list : Combined build and query operation
    """
    graph_mode = _validate_graph_mode(graph_mode)

    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(cutoff)

    if neighbor_matrix is None:
        neighbor_matrix = jnp.full(
            (positions.shape[0], max_neighbors),
            positions.shape[0],
            dtype=jnp.int32,
        )
    elif rebuild_flags is None and graph_mode == "none":
        neighbor_matrix = neighbor_matrix.at[:].set(jnp.int32(positions.shape[0]))

    if num_neighbors is None:
        num_neighbors = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    elif rebuild_flags is None and graph_mode == "none":
        num_neighbors = num_neighbors.at[:].set(jnp.int32(0))

    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = jnp.zeros(
            (positions.shape[0], max_neighbors, 3),
            dtype=jnp.int32,
        )
    elif rebuild_flags is None and graph_mode == "none":
        neighbor_matrix_shifts = neighbor_matrix_shifts.at[:].set(jnp.int32(0))

    # Select kernel based on dtype
    if positions.dtype == jnp.float64:
        _query_kernel = _jax_build_neighbor_matrix_f64
        _query_kernel_selective = _jax_build_neighbor_matrix_selective_f64
    else:
        _query_kernel = _jax_build_neighbor_matrix_f32
        _query_kernel_selective = _jax_build_neighbor_matrix_selective_f32
        positions = positions.astype(jnp.float32)

    # Ensure cell dtype matches positions dtype so warp overload dispatch is consistent
    if cell.dtype != positions.dtype:
        cell = cell.astype(positions.dtype)

    total_atoms = positions.shape[0]

    # Squeeze pbc to 1D for kernel (kernels expect shape (3,))
    pbc_1d = pbc.squeeze() if pbc.ndim == 2 else pbc
    pbc_bool = pbc_1d.astype(jnp.bool_)

    if graph_mode == "warp":
        fill_value = int(positions.shape[0])
        if rebuild_flags is not None:
            rf = rebuild_flags.flatten()[:1].astype(jnp.bool_)
            graph_query = (
                _jax_graph_query_cell_list_selective_f64
                if positions.dtype == jnp.float64
                else _jax_graph_query_cell_list_selective_f32
            )
            neighbor_matrix, neighbor_matrix_shifts, num_neighbors = graph_query(
                positions,
                cell,
                pbc_bool,
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
                float(cutoff),
                fill_value,
                rf,
            )
        else:
            graph_query = (
                _jax_graph_query_cell_list_f64
                if positions.dtype == jnp.float64
                else _jax_graph_query_cell_list_f32
            )
            neighbor_matrix, neighbor_matrix_shifts, num_neighbors = graph_query(
                positions,
                cell,
                pbc_bool,
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
                float(cutoff),
                fill_value,
            )
    elif rebuild_flags is not None:
        rf = rebuild_flags.flatten()[:1].astype(jnp.bool_)
        num_neighbors = jnp.where(rf[0], jnp.zeros_like(num_neighbors), num_neighbors)
        neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
            _query_kernel_selective(
                positions,
                cell,
                pbc_bool,
                float(cutoff),
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
                False,  # half_fill
                rf,
                launch_dims=(total_atoms,),
            )
        )
    else:
        neighbor_matrix, neighbor_matrix_shifts, num_neighbors = _query_kernel(
            positions,
            cell,
            pbc_bool,
            float(cutoff),
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
            False,  # half_fill
            launch_dims=(total_atoms,),
        )

    return neighbor_matrix, num_neighbors, neighbor_matrix_shifts


def cell_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    max_neighbors: int | None = None,
    max_total_cells: int | None = None,
    return_neighbor_list: bool = False,
    cells_per_dimension: jax.Array | None = None,
    neighbor_search_radius: jax.Array | None = None,
    atom_periodic_shifts: jax.Array | None = None,
    atom_to_cell_mapping: jax.Array | None = None,
    atoms_per_cell_count: jax.Array | None = None,
    cell_atom_start_indices: jax.Array | None = None,
    cell_atom_list: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
    graph_mode: Literal["none", "warp"] = "none",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build and query spatial cell list for efficient neighbor finding.

    This is a convenience function that combines build_cell_list and query_cell_list
    in a single call.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection.
    cell : jax.Array, shape (1, 3, 3), dtype=float32 or float64, optional
        Cell matrix defining lattice vectors. Default is identity matrix.
    pbc : jax.Array, shape (3,) or (1, 3), dtype=bool, optional
        Periodic boundary condition flags. Default is all True.
    max_neighbors : int, optional
        Maximum number of neighbors per atom. If None, will be estimated.
    max_total_cells : int, optional
        Maximum number of cells to allocate. If None, will be estimated.
    return_neighbor_list : bool, optional
        If True, convert result to COO neighbor list format. Default is False.
    graph_mode : {"none", "warp"}, default="none"
        Execution mode for the underlying Warp launches. ``"warp"`` is
        intended for jitted call sites that donate and reuse the optional
        cell-list caches and output buffers.

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
        If ``return_neighbor_list=False``: ``neighbor_matrix_shifts`` with shape
        (total_atoms, max_neighbors, 3), dtype int32.
        If ``return_neighbor_list=True``: ``neighbor_list_shifts`` with shape
        (num_pairs, 3), dtype int32.

    See Also
    --------
    build_cell_list : Build cell list separately
    query_cell_list : Query cell list separately
    naive_neighbor_list : Naive O(N^2) method
    """
    graph_mode = _validate_graph_mode(graph_mode)

    if cell is None:
        cell = jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, :, :]
    if pbc is None:
        pbc = jnp.ones((1, 3), dtype=jnp.bool_)

    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    if pbc.ndim == 1:
        pbc = pbc[jnp.newaxis, :]

    if max_neighbors is None and (
        neighbor_matrix is None
        or neighbor_matrix_shifts is None
        or num_neighbors is None
    ):
        max_neighbors = estimate_max_neighbors(cutoff)

    if max_total_cells is None:
        max_total_cells, _, neighbor_search_radius_est = estimate_cell_list_sizes(
            positions, cell, cutoff, pbc
        )
        if neighbor_search_radius is None:
            neighbor_search_radius = neighbor_search_radius_est
    elif neighbor_search_radius is None:
        neighbor_search_radius = jnp.ones(3, dtype=jnp.int32)

    if cells_per_dimension is None:
        cells_per_dimension = jnp.ones(3, dtype=jnp.int32)
    if atom_periodic_shifts is None:
        atom_periodic_shifts = jnp.zeros((positions.shape[0], 3), dtype=jnp.int32)
    if atom_to_cell_mapping is None:
        atom_to_cell_mapping = jnp.zeros((positions.shape[0], 3), dtype=jnp.int32)
    if atoms_per_cell_count is None:
        atoms_per_cell_count = jnp.zeros(max_total_cells, dtype=jnp.int32)
    elif graph_mode == "none":
        atoms_per_cell_count = atoms_per_cell_count.at[:].set(jnp.int32(0))
    if cell_atom_start_indices is None:
        cell_atom_start_indices = jnp.zeros(max_total_cells, dtype=jnp.int32)
    if cell_atom_list is None:
        cell_atom_list = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    if neighbor_matrix is None:
        neighbor_matrix = jnp.full(
            (positions.shape[0], max_neighbors),
            positions.shape[0],
            dtype=jnp.int32,
        )
    elif graph_mode == "none":
        neighbor_matrix = neighbor_matrix.at[:].set(jnp.int32(positions.shape[0]))
    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = jnp.zeros(
            (positions.shape[0], max_neighbors, 3),
            dtype=jnp.int32,
        )
    elif graph_mode == "none":
        neighbor_matrix_shifts = neighbor_matrix_shifts.at[:].set(jnp.int32(0))
    if num_neighbors is None:
        num_neighbors = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    elif graph_mode == "none":
        num_neighbors = num_neighbors.at[:].set(jnp.int32(0))

    if positions.dtype != jnp.float64:
        positions = positions.astype(jnp.float32)
    if cell.dtype != positions.dtype:
        cell = cell.astype(positions.dtype)
    pbc_1d = pbc.squeeze() if pbc.ndim == 2 else pbc
    pbc_bool = pbc_1d.astype(jnp.bool_)

    if graph_mode == "warp":
        graph_cell_list = (
            _jax_graph_cell_list_f64
            if positions.dtype == jnp.float64
            else _jax_graph_cell_list_f32
        )
        (
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
        ) = graph_cell_list(
            positions,
            cell,
            pbc_bool,
            neighbor_search_radius,
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            float(cutoff),
            int(positions.shape[0]),
        )
    else:
        (
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_search_radius,
        ) = build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            cells_per_dimension=cells_per_dimension,
            neighbor_search_radius=neighbor_search_radius,
            atom_periodic_shifts=atom_periodic_shifts,
            atom_to_cell_mapping=atom_to_cell_mapping,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            max_total_cells=max_total_cells,
            graph_mode="none",
        )

        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = query_cell_list(
            positions=positions,
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
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            graph_mode="none",
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
        return (
            neighbor_list,
            neighbor_ptr,
            neighbor_list_shifts,
        )
    else:
        return (
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        )
