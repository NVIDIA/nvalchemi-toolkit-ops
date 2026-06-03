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


"""Generated Warp kernels and factories for cell-list neighbor lists."""

from functools import lru_cache
from typing import Any, Literal

import warp as wp

from nvalchemiops.math import wpdivmod
from nvalchemiops.neighbors.neighbor_utils import (
    DTYPE_INFO_ALL,
    _append_specialization_doc,
    _decode_full_shift_index,
    _decode_shift_index,
    kernel_specialization_name,
    require_supported_dtype,
    set_fn_doc,
    set_fn_name,
)

_SUPPORTED_DTYPES = (wp.float32, wp.float64)
_DTYPE_INFO: dict[type, tuple[type, type]] = {
    dtype: DTYPE_INFO_ALL[dtype] for dtype in _SUPPORTED_DTYPES
}


def _require_supported_dtype(wp_dtype: type) -> None:
    """Validate that the cell-list factories support ``wp_dtype``."""
    require_supported_dtype(wp_dtype, _SUPPORTED_DTYPES)


def _pair_output_features(
    *,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> tuple[str, ...]:
    """Return specialization-name tokens for pair-output options."""
    return tuple(
        feature
        for feature in (
            "vectors" if return_vectors else "",
            "distances" if return_distances else "",
            "pair_fn" if pair_fn is not None else "",
        )
        if feature
    )


def _cell_list_build_base_name(stage: str, *, batched: bool) -> str:
    """Return the 0.3.1-style base name for a cell-list build stage."""
    prefix = "_batch_cell_list" if batched else "_cell_list"
    match stage:
        case "estimate_sizes":
            return (
                "_batch_estimate_cell_list_sizes"
                if batched
                else "_estimate_cell_list_sizes"
            )
        case "construct_bin_size":
            return f"{prefix}_construct_bin_size"
        case "count_atoms":
            return f"{prefix}_count_atoms_per_bin"
        case "bin_atoms":
            return f"{prefix}_bin_atoms"
        case _:
            raise ValueError(f"Unknown cell-list build stage {stage!r}")


def _cell_list_neighbor_base_name(*, batched: bool, selective: bool) -> str:
    """Return the 0.3.1-style base name for cell-list neighbor rows."""
    base = (
        "_batch_cell_list_build_neighbor_matrix"
        if batched
        else "_cell_list_build_neighbor_matrix"
    )
    if selective:
        base = f"{base}_selective"
    return base


__all__ = [
    "_build_cell_to_system_map",
    "_fill_target_row_lookup",
    "_reset_target_row_lookup",
    "get_cell_list_cells_per_system_kernel",
    "get_build_cell_list_kernel",
    "get_cell_list_gather_kernel",
    "get_query_cell_list_kernel",
]


@lru_cache(maxsize=None)
def _make_estimate_cell_list_sizes_kernel(
    wp_dtype: type,
    *,
    batched: bool,
    min_cells_per_dimension: int,
):
    """Build the ``estimate_cell_list_sizes`` kernel for the dtype/mode.

    The single-system and batched specializations share one kernel body.
    ``BATCHED`` is a compile-time constant, so Warp eliminates the unused
    branch for each cached specialization. Unused single/batch arrays are
    passed as zero-size sentinels by launchers and bindings.
    """
    _require_supported_dtype(wp_dtype)
    _, mat_dtype = _DTYPE_INFO[wp_dtype]
    BATCHED = wp.constant(bool(batched))
    MIN_CELLS_PER_DIMENSION = wp.constant(int(min_cells_per_dimension))

    @wp.kernel(enable_backward=False)
    def _kernel(
        cell: wp.array(dtype=mat_dtype),
        pbc_single: wp.array(dtype=wp.bool),
        pbc_batch: wp.array2d(dtype=wp.bool),
        cell_size: wp_dtype,
        max_nbins: wp.int32,
        number_of_cells: wp.array(dtype=wp.int32),
        neighbor_search_radius_single: wp.array(dtype=wp.int32),
        neighbor_search_radius_batch: wp.array(dtype=wp.vec3i),
    ) -> None:
        """Estimate cell-list allocation sizes

        Parameters
        ----------
        cell : wp.array, shape (num_systems,), dtype=wp.mat33*
            Cell matrices defining simulation boxes.
        pbc_single : wp.array, shape (3,), dtype=wp.bool
            Single-system PBC flags. Zero-size sentinel in batched mode.
        pbc_batch : wp.array, shape (num_systems, 3), dtype=wp.bool
            Batched PBC flags. Zero-size sentinel in single-system mode.
        cell_size : float
            Target cell size used for decomposition.
        max_nbins : wp.int32
            Maximum cell count allowed per system.
        number_of_cells : wp.array, shape (num_systems,), dtype=wp.int32
            OUTPUT: Estimated cell count per system.
        neighbor_search_radius_single : wp.array, shape (3,), dtype=wp.int32
            OUTPUT: Single-system neighbor search radius. Sentinel in batched mode.
        neighbor_search_radius_batch : wp.array, shape (num_systems,), dtype=wp.vec3i
            OUTPUT: Batched neighbor search radii. Sentinel in single-system mode.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: One thread per system in batched mode; one sentinel thread in single-system mode.
        - Modifies: ``number_of_cells`` and the active neighbor-search-radius output.
        ``BATCHED`` is a static specialization. Inactive single/batch arrays are
        zero-size sentinels and are not read.

        See Also
        --------
        get_build_cell_list_kernel : Return the specialized cell-list build kernel for this stage.
        """
        system_idx = wp.int32(0)
        if BATCHED:
            system_idx = wp.tid()

        inverse_cell_transpose = wp.transpose(wp.inverse(cell[system_idx]))
        cells_per_dimension = wp.vec3i(0, 0, 0)
        pbc_x = wp.bool(False)
        pbc_y = wp.bool(False)
        pbc_z = wp.bool(False)
        if BATCHED:
            pbc_x = pbc_batch[system_idx, 0]
            pbc_y = pbc_batch[system_idx, 1]
            pbc_z = pbc_batch[system_idx, 2]
        else:
            pbc_x = pbc_single[0]
            pbc_y = pbc_single[1]
            pbc_z = pbc_single[2]

        for dim in range(3):
            face_distance = type(cell_size)(1.0) / wp.length(
                inverse_cell_transpose[dim]
            )
            cells_per_dimension[dim] = max(wp.int32(face_distance / cell_size), 1)

        for dim in range(3):
            pbc_dim = pbc_z
            if dim == 0:
                pbc_dim = pbc_x
            elif dim == 1:
                pbc_dim = pbc_y
            if MIN_CELLS_PER_DIMENSION > 1 and (
                pbc_dim or cells_per_dimension[dim] > 1
            ):
                while cells_per_dimension[dim] < MIN_CELLS_PER_DIMENSION:
                    cells_per_dimension[dim] = cells_per_dimension[dim] * 2

        total_cells = int(
            cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
        )
        while total_cells > max_nbins:
            for dim in range(3):
                cells_per_dimension[dim] = max(cells_per_dimension[dim] // 2, 1)
            total_cells = int(
                cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
            )

        search_radius = wp.vec3i(0, 0, 0)
        for dim in range(3):
            face_distance = type(cell_size)(1.0) / wp.length(
                inverse_cell_transpose[dim]
            )
            pbc_dim = pbc_z
            if dim == 0:
                pbc_dim = pbc_x
            elif dim == 1:
                pbc_dim = pbc_y
            if cells_per_dimension[dim] == 1 and not pbc_dim:
                search_radius[dim] = 0
            else:
                search_radius[dim] = wp.int32(
                    wp.ceil(
                        cell_size
                        * type(cell_size)(cells_per_dimension[dim])
                        / face_distance
                    )
                )

        number_of_cells[system_idx] = total_cells
        if BATCHED:
            neighbor_search_radius_batch[system_idx] = search_radius
        else:
            neighbor_search_radius_single[0] = search_radius[0]
            neighbor_search_radius_single[1] = search_radius[1]
            neighbor_search_radius_single[2] = search_radius[2]

    name = kernel_specialization_name(
        _cell_list_build_base_name("estimate_sizes", batched=batched),
        wp_dtype=wp_dtype,
        features=(f"mincells{int(min_cells_per_dimension)}",),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp_dtype,
            entries=(
                ("stage", "estimate_sizes"),
                ("batched", bool(batched)),
                ("min_cells_per_dimension", int(min_cells_per_dimension)),
            ),
        ),
    )


@lru_cache(maxsize=None)
def _make_construct_bin_size_kernel(
    wp_dtype: type,
    *,
    batched: bool,
    min_cells_per_dimension: int,
):
    """Build the ``construct_bin_size`` kernel for the dtype/mode."""
    _require_supported_dtype(wp_dtype)
    _, mat_dtype = _DTYPE_INFO[wp_dtype]
    BATCHED = wp.constant(bool(batched))
    MIN_CELLS_PER_DIMENSION = wp.constant(int(min_cells_per_dimension))

    @wp.kernel(enable_backward=False)
    def _kernel(
        cell: wp.array(dtype=mat_dtype),
        pbc_single: wp.array(dtype=wp.bool),
        pbc_batch: wp.array2d(dtype=wp.bool),
        cells_per_dimension_single: wp.array(dtype=wp.int32),
        cells_per_dimension_batch: wp.array(dtype=wp.vec3i),
        target_cell_size: wp_dtype,
        max_total_cells: wp.int32,
    ) -> None:
        """Determine optimal spatial decomposition parameters

        Parameters
        ----------
        cell : wp.array, shape (num_systems,), dtype=wp.mat33*
            Cell matrices defining simulation boxes.
        pbc_single : wp.array, shape (3,), dtype=wp.bool
            Single-system PBC flags. Zero-size sentinel in batched mode.
        pbc_batch : wp.array, shape (num_systems, 3), dtype=wp.bool
            Batched PBC flags. Zero-size sentinel in single-system mode.
        cells_per_dimension_single : wp.array, shape (3,), dtype=wp.int32
            OUTPUT: Single-system cell counts per dimension.
        cells_per_dimension_batch : wp.array, shape (num_systems,), dtype=wp.vec3i
            OUTPUT: Batched cell counts per dimension.
        target_cell_size : float
            Desired cell size for the spatial grid.
        max_total_cells : wp.int32
            Maximum total cells allowed across active systems.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: One thread per system in batched mode; one sentinel thread in single-system mode.
        - Modifies: The active ``cells_per_dimension_single`` or ``cells_per_dimension_batch`` output.
        ``BATCHED`` is a static specialization. Inactive single/batch arrays are
        zero-size sentinels and are not read.

        See Also
        --------
        get_build_cell_list_kernel : Return the specialized cell-list build kernel for this stage.
        """
        system_idx = wp.int32(0)
        num_systems = wp.int32(1)
        if BATCHED:
            system_idx = wp.tid()
            num_systems = cell.shape[0]

        inverse_cell_transpose = wp.transpose(wp.inverse(cell[system_idx]))
        cells_per_dimension = wp.vec3i(0, 0, 0)
        pbc_x = wp.bool(False)
        pbc_y = wp.bool(False)
        pbc_z = wp.bool(False)
        if BATCHED:
            pbc_x = pbc_batch[system_idx, 0]
            pbc_y = pbc_batch[system_idx, 1]
            pbc_z = pbc_batch[system_idx, 2]
        else:
            pbc_x = pbc_single[0]
            pbc_y = pbc_single[1]
            pbc_z = pbc_single[2]

        for dim in range(3):
            face_distance = type(target_cell_size)(1.0) / wp.length(
                inverse_cell_transpose[dim]
            )
            cells_per_dimension[dim] = max(
                wp.int32(face_distance / target_cell_size), 1
            )

        for dim in range(3):
            pbc_dim = pbc_z
            if dim == 0:
                pbc_dim = pbc_x
            elif dim == 1:
                pbc_dim = pbc_y
            if MIN_CELLS_PER_DIMENSION > 1 and (
                pbc_dim or cells_per_dimension[dim] > 1
            ):
                while cells_per_dimension[dim] < MIN_CELLS_PER_DIMENSION:
                    cells_per_dimension[dim] = cells_per_dimension[dim] * 2

        total_cells = int(
            cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
        )
        while total_cells * num_systems > max_total_cells:
            for dim in range(3):
                cells_per_dimension[dim] = max(cells_per_dimension[dim] // 2, 1)
            total_cells = int(
                cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
            )

        if BATCHED:
            cells_per_dimension_batch[system_idx] = cells_per_dimension
        else:
            cells_per_dimension_single[0] = cells_per_dimension[0]
            cells_per_dimension_single[1] = cells_per_dimension[1]
            cells_per_dimension_single[2] = cells_per_dimension[2]

    name = kernel_specialization_name(
        _cell_list_build_base_name("construct_bin_size", batched=batched),
        wp_dtype=wp_dtype,
        features=(f"mincells{int(min_cells_per_dimension)}",),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp_dtype,
            entries=(
                ("stage", "construct_bin_size"),
                ("batched", bool(batched)),
                ("min_cells_per_dimension", int(min_cells_per_dimension)),
            ),
        ),
    )


@lru_cache(maxsize=None)
def _make_count_atoms_per_bin_kernel(wp_dtype: type, *, batched: bool):
    """Build the ``count_atoms_per_bin`` kernel for the dtype/mode."""
    _require_supported_dtype(wp_dtype)
    vec_dtype, mat_dtype = _DTYPE_INFO[wp_dtype]
    BATCHED = wp.constant(bool(batched))

    @wp.kernel(enable_backward=False)
    def _kernel(
        positions: wp.array(dtype=vec_dtype),
        cell: wp.array(dtype=mat_dtype),
        pbc_single: wp.array(dtype=wp.bool),
        pbc_batch: wp.array2d(dtype=wp.bool),
        batch_idx: wp.array(dtype=wp.int32),
        cells_per_dimension_single: wp.array(dtype=wp.int32),
        cells_per_dimension_batch: wp.array(dtype=wp.vec3i),
        cell_offsets: wp.array(dtype=wp.int32),
        atoms_per_cell_count: wp.array(dtype=wp.int32),
        atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    ) -> None:
        """Count atoms in each spatial cell and compute periodic shifts

        Parameters
        ----------
        positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
            Atomic coordinates in Cartesian space.
        cell : wp.array, shape (num_systems,), dtype=wp.mat33*
            Cell matrices for coordinate transforms.
        pbc_single : wp.array, shape (3,), dtype=wp.bool
            Single-system PBC flags. Zero-size sentinel in batched mode.
        pbc_batch : wp.array, shape (num_systems, 3), dtype=wp.bool
            Batched PBC flags. Zero-size sentinel in single-system mode.
        batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
            System index for each atom. Sentinel in single-system mode.
        cells_per_dimension_single : wp.array, shape (3,), dtype=wp.int32
            Single-system cell counts per dimension.
        cells_per_dimension_batch : wp.array, shape (num_systems,), dtype=wp.vec3i
            Batched cell counts per dimension.
        cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
            Global cell offset per system. Sentinel in single-system mode.
        atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
            OUTPUT: Atom counts per global cell.
        atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
            OUTPUT: Periodic cell shifts per atom.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: One thread per atom.
        - Modifies: ``atoms_per_cell_count`` atomically and ``atom_periodic_shifts``.
        ``BATCHED`` is a static specialization. Inactive single/batch arrays are
        zero-size sentinels and are not read.

        See Also
        --------
        get_build_cell_list_kernel : Return the specialized cell-list build kernel for this stage.
        """
        atom_idx = wp.tid()
        system_idx = wp.int32(0)
        s_cell_offset = wp.int32(0)
        s_cells_per_dimension = wp.vec3i(0, 0, 0)
        pbc_x = wp.bool(False)
        pbc_y = wp.bool(False)
        pbc_z = wp.bool(False)
        fractional_position = positions[atom_idx]
        if BATCHED:
            system_idx = batch_idx[atom_idx]
            s_cell_offset = cell_offsets[system_idx]
            s_cells_per_dimension = cells_per_dimension_batch[system_idx]
            pbc_x = pbc_batch[system_idx, 0]
            pbc_y = pbc_batch[system_idx, 1]
            pbc_z = pbc_batch[system_idx, 2]
            fractional_position = positions[atom_idx] * wp.inverse(cell[system_idx])
        else:
            s_cells_per_dimension = wp.vec3i(
                cells_per_dimension_single[0],
                cells_per_dimension_single[1],
                cells_per_dimension_single[2],
            )
            pbc_x = pbc_single[0]
            pbc_y = pbc_single[1]
            pbc_z = pbc_single[2]
            fractional_position = (
                wp.transpose(wp.inverse(cell[0])) * positions[atom_idx]
            )

        cell_coords = wp.vec3i(0, 0, 0)
        for dim in range(3):
            cell_coords[dim] = wp.int32(
                wp.floor(
                    fractional_position[dim]
                    * type(fractional_position[dim])(s_cells_per_dimension[dim])
                )
            )
            pbc_dim = pbc_z
            if dim == 0:
                pbc_dim = pbc_x
            elif dim == 1:
                pbc_dim = pbc_y
            if pbc_dim:
                quotient, remainder = wpdivmod(
                    cell_coords[dim], s_cells_per_dimension[dim]
                )
                atom_periodic_shifts[atom_idx][dim] = quotient
                cell_coords[dim] = remainder
            else:
                atom_periodic_shifts[atom_idx][dim] = 0
                cell_coords[dim] = wp.clamp(
                    cell_coords[dim], 0, s_cells_per_dimension[dim] - 1
                )

        linear_cell_index = (
            s_cell_offset
            + cell_coords[0]
            + s_cells_per_dimension[0]
            * (cell_coords[1] + s_cells_per_dimension[1] * cell_coords[2])
        )
        wp.atomic_add(atoms_per_cell_count, linear_cell_index, 1)

    name = kernel_specialization_name(
        _cell_list_build_base_name("count_atoms", batched=batched),
        wp_dtype=wp_dtype,
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp_dtype,
            entries=(
                ("stage", "count_atoms"),
                ("batched", bool(batched)),
            ),
        ),
    )


@lru_cache(maxsize=None)
def _make_bin_atoms_kernel(wp_dtype: type, *, batched: bool):
    """Build the ``bin_atoms`` kernel for the dtype/mode."""
    _require_supported_dtype(wp_dtype)
    vec_dtype, mat_dtype = _DTYPE_INFO[wp_dtype]
    BATCHED = wp.constant(bool(batched))

    @wp.kernel(enable_backward=False)
    def _kernel(
        positions: wp.array(dtype=vec_dtype),
        cell: wp.array(dtype=mat_dtype),
        pbc_single: wp.array(dtype=wp.bool),
        pbc_batch: wp.array2d(dtype=wp.bool),
        batch_idx: wp.array(dtype=wp.int32),
        cells_per_dimension_single: wp.array(dtype=wp.int32),
        cells_per_dimension_batch: wp.array(dtype=wp.vec3i),
        cell_offsets: wp.array(dtype=wp.int32),
        atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
        atoms_per_cell_count: wp.array(dtype=wp.int32),
        cell_atom_start_indices: wp.array(dtype=wp.int32),
        cell_atom_list: wp.array(dtype=wp.int32),
    ) -> None:
        """Assign atoms to spatial cells and build cell-contiguous storage

        Parameters
        ----------
        positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
            Atomic coordinates in Cartesian space.
        cell : wp.array, shape (num_systems,), dtype=wp.mat33*
            Cell matrices for coordinate transforms.
        pbc_single : wp.array, shape (3,), dtype=wp.bool
            Single-system PBC flags. Zero-size sentinel in batched mode.
        pbc_batch : wp.array, shape (num_systems, 3), dtype=wp.bool
            Batched PBC flags. Zero-size sentinel in single-system mode.
        batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
            System index for each atom. Sentinel in single-system mode.
        cells_per_dimension_single : wp.array, shape (3,), dtype=wp.int32
            Single-system cell counts per dimension.
        cells_per_dimension_batch : wp.array, shape (num_systems,), dtype=wp.vec3i
            Batched cell counts per dimension.
        cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
            Global cell offset per system. Sentinel in single-system mode.
        atom_to_cell_mapping : wp.array, shape (total_atoms,), dtype=wp.vec3i
            OUTPUT: Cell coordinates assigned to each atom.
        atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
            MODIFIED: Per-cell insertion counters.
        cell_atom_start_indices : wp.array, shape (total_cells,), dtype=wp.int32
            Starting offsets for each cell in ``cell_atom_list``.
        cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
            OUTPUT: Atom indices in cell-contiguous order.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: One thread per atom.
        - Modifies: ``atom_to_cell_mapping``, ``atoms_per_cell_count`` atomically, and ``cell_atom_list``.
        ``BATCHED`` is a static specialization. Inactive single/batch arrays are
        zero-size sentinels and are not read.

        See Also
        --------
        get_build_cell_list_kernel : Return the specialized cell-list build kernel for this stage.
        """
        atom_idx = wp.tid()
        if atom_idx >= positions.shape[0]:
            return

        system_idx = wp.int32(0)
        s_cell_offset = wp.int32(0)
        s_cells_per_dimension = wp.vec3i(0, 0, 0)
        pbc_x = wp.bool(False)
        pbc_y = wp.bool(False)
        pbc_z = wp.bool(False)
        fractional_position = positions[atom_idx]
        if BATCHED:
            system_idx = batch_idx[atom_idx]
            s_cell_offset = cell_offsets[system_idx]
            s_cells_per_dimension = cells_per_dimension_batch[system_idx]
            pbc_x = pbc_batch[system_idx, 0]
            pbc_y = pbc_batch[system_idx, 1]
            pbc_z = pbc_batch[system_idx, 2]
            fractional_position = positions[atom_idx] * wp.inverse(cell[system_idx])
        else:
            s_cells_per_dimension = wp.vec3i(
                cells_per_dimension_single[0],
                cells_per_dimension_single[1],
                cells_per_dimension_single[2],
            )
            pbc_x = pbc_single[0]
            pbc_y = pbc_single[1]
            pbc_z = pbc_single[2]
            fractional_position = (
                wp.transpose(wp.inverse(cell[0])) * positions[atom_idx]
            )

        cell_coords = wp.vec3i(0, 0, 0)
        for dim in range(3):
            cell_coords[dim] = wp.int32(
                wp.floor(
                    fractional_position[dim]
                    * type(fractional_position[dim])(s_cells_per_dimension[dim])
                )
            )
            pbc_dim = pbc_z
            if dim == 0:
                pbc_dim = pbc_x
            elif dim == 1:
                pbc_dim = pbc_y
            if pbc_dim:
                _, remainder = wpdivmod(cell_coords[dim], s_cells_per_dimension[dim])
                cell_coords[dim] = remainder
            else:
                cell_coords[dim] = wp.clamp(
                    cell_coords[dim], 0, s_cells_per_dimension[dim] - 1
                )

        atom_to_cell_mapping[atom_idx] = cell_coords
        linear_cell_index = (
            s_cell_offset
            + cell_coords[0]
            + s_cells_per_dimension[0]
            * (cell_coords[1] + s_cells_per_dimension[1] * cell_coords[2])
        )
        position_in_cell = wp.atomic_add(atoms_per_cell_count, linear_cell_index, 1)
        final_list_index = cell_atom_start_indices[linear_cell_index] + position_in_cell
        cell_atom_list[final_list_index] = atom_idx

    name = kernel_specialization_name(
        _cell_list_build_base_name("bin_atoms", batched=batched),
        wp_dtype=wp_dtype,
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp_dtype,
            entries=(
                ("stage", "bin_atoms"),
                ("batched", bool(batched)),
            ),
        ),
    )


# ----------------------------------------------------------------------------
# Neighbor-matrix kernel factories
# ----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _reset_target_row_lookup(target_row_lookup: wp.array(dtype=wp.int32)) -> None:
    """Reset atom-id to compact target-row lookup entries

    Parameters
    ----------
    target_row_lookup : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Lookup entries reset to ``-1``.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: one thread per atom-id entry.
    - Modifies: target_row_lookup.

    See Also
    --------
    _prepare_target_row_lookup : Prepare the partial-row lookup used by cell-list neighbor search.
    """
    atom_idx = wp.tid()
    target_row_lookup[atom_idx] = -1


@wp.kernel(enable_backward=False)
def _fill_target_row_lookup(
    target_indices: wp.array(dtype=wp.int32),
    target_row_lookup: wp.array(dtype=wp.int32),
) -> None:
    """Fill atom-id to compact target-row lookup from target indices

    Parameters
    ----------
    target_indices : wp.array, shape (n_targets,), dtype=wp.int32
        Target atom indices in compact row order.
    target_row_lookup : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Atom-id to compact row lookup.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: one thread per target row.
    - Modifies: target_row_lookup.

    See Also
    --------
    _prepare_target_row_lookup : Prepare the partial-row lookup used by cell-list neighbor search.
    """
    row = wp.tid()
    target_row_lookup[target_indices[row]] = row


@lru_cache(maxsize=None)
def _make_store_neighbor_fn(
    wp_dtype: type,
    *,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
):
    """Build a per-slot store helper for one dtype/output combination."""
    _require_supported_dtype(wp_dtype)
    vec_dtype, _ = _DTYPE_INFO[wp_dtype]
    RETURN_VECTORS = wp.constant(bool(return_vectors))
    RETURN_DISTANCES = wp.constant(bool(return_distances))
    HAS_PAIR_FN = wp.constant(pair_fn is not None)

    @wp.func
    def _store_neighbor(
        output_row: wp.int32,
        output_slot: wp.int32,
        atom_idx: wp.int32,
        neighbor_atom_idx: wp.int32,
        shift: wp.vec3i,
        dr: vec_dtype,
        distance_sq: wp_dtype,
        neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
        neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
        neighbor_vectors: wp.array(dtype=vec_dtype, ndim=2),
        neighbor_distances: wp.array(dtype=wp_dtype, ndim=2),
        pair_params: wp.array(dtype=wp_dtype, ndim=2),
        pair_energies: wp.array(dtype=wp_dtype, ndim=2),
        pair_forces: wp.array(dtype=vec_dtype, ndim=2),
    ) -> None:
        """Store a cell-list neighbor pair

        Parameters
        ----------
        output_row : wp.int32
            Row in the output neighbor buffers.
        output_slot : wp.int32
            Column or slot in the output neighbor buffers.
        atom_idx : wp.int32
            Source atom index passed to an optional pair function.
        neighbor_atom_idx : wp.int32
            Neighbor atom index written to ``neighbor_matrix``.
        shift : wp.vec3i
            Periodic shift vector stored with the pair.
        dr : wp.vec3*
            Cartesian displacement from source atom to neighbor atom.
        distance_sq : float
            Squared pair distance.
        neighbor_matrix : wp.array, shape (rows, max_neighbors), dtype=wp.int32
            OUTPUT: Neighbor atom indices.
        neighbor_matrix_shifts : wp.array, shape (rows, max_neighbors), dtype=wp.vec3i
            OUTPUT: Periodic shift vectors.
        neighbor_vectors : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*
            OUTPUT: Optional displacement vectors. Sentinel when disabled.
        neighbor_distances : wp.array, shape (rows, max_neighbors), dtype=wp.float*
            OUTPUT: Optional pair distances. Sentinel when disabled.
        pair_params : wp.array, shape (total_atoms, K), dtype=wp.float*
            Pair-function parameters. Sentinel when no pair function is active.
        pair_energies : wp.array, shape (rows, max_neighbors), dtype=wp.float*
            OUTPUT: Optional pair-function energies. Sentinel when disabled.
        pair_forces : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*
            OUTPUT: Optional pair-function forces. Sentinel when disabled.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        ``RETURN_VECTORS``, ``RETURN_DISTANCES``, and ``HAS_PAIR_FN`` are static
        specializations. Inactive output buffers are zero-size sentinels and are not read.
        """
        neighbor_matrix[output_row, output_slot] = neighbor_atom_idx
        neighbor_matrix_shifts[output_row, output_slot] = shift
        if RETURN_VECTORS:
            neighbor_vectors[output_row, output_slot] = dr
        if RETURN_DISTANCES or HAS_PAIR_FN:
            distance = wp.sqrt(distance_sq)
            if RETURN_DISTANCES:
                neighbor_distances[output_row, output_slot] = distance
            if HAS_PAIR_FN:
                pair_energy, pair_force = pair_fn(
                    dr,
                    distance,
                    pair_params,
                    atom_idx,
                    neighbor_atom_idx,
                )
                pair_energies[output_row, output_slot] = pair_energy
                pair_forces[output_row, output_slot] = pair_force

    name = kernel_specialization_name(
        "_cell_list_store_neighbor",
        wp_dtype=wp_dtype,
        features=_pair_output_features(
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
        ),
    )
    return set_fn_doc(
        set_fn_name(_store_neighbor, name),
        _append_specialization_doc(
            _store_neighbor.__doc__,
            dtype=wp_dtype,
            entries=(
                ("return_vectors", bool(return_vectors)),
                ("return_distances", bool(return_distances)),
                ("pair_fn", pair_fn is not None),
            ),
        ),
    )


@lru_cache(maxsize=None)
def _make_atom_centric_kernel(
    wp_dtype: type,
    *,
    batched: bool,
    selective: bool,
    partial: bool,
    half_fill: bool,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
    atom_centric_path: str,
) -> wp.Kernel:
    """Build the atom-centric neighbor-matrix kernel."""
    _require_supported_dtype(wp_dtype)
    vec_dtype, mat_dtype = _DTYPE_INFO[wp_dtype]
    BATCHED = wp.constant(bool(batched))
    SELECTIVE = wp.constant(bool(selective))
    PARTIAL = wp.constant(bool(partial))
    HALF_FILL = wp.constant(bool(half_fill))
    ATOM_CENTRIC_DIRECT = wp.constant(atom_centric_path == "direct")
    SYMMETRIC_FULL_FILL = wp.constant(
        atom_centric_path == "direct"
        and not bool(half_fill)
        and not bool(partial)
        and not bool(return_vectors)
        and not bool(return_distances)
        and pair_fn is None
    )
    store_neighbor = _make_store_neighbor_fn(
        wp_dtype,
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        pair_fn=pair_fn,
    )

    @wp.kernel(enable_backward=False, module="unique")
    def _kernel(
        positions: wp.array(dtype=vec_dtype),
        atom_periodic_shifts: wp.array(dtype=wp.vec3i),
        sorted_positions: wp.array(dtype=vec_dtype),
        sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
        cell: wp.array(dtype=mat_dtype),
        pbc_single: wp.array(dtype=wp.bool),
        pbc_batch: wp.array2d(dtype=wp.bool),
        batch_idx: wp.array(dtype=wp.int32),
        cutoff: wp_dtype,
        cells_per_dimension_single: wp.array(dtype=wp.int32),
        cells_per_dimension_batch: wp.array(dtype=wp.vec3i),
        neighbor_search_radius_single: wp.array(dtype=wp.int32),
        neighbor_search_radius_batch: wp.array(dtype=wp.vec3i),
        atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
        atoms_per_cell_count: wp.array(dtype=wp.int32),
        cell_atom_start_indices: wp.array(dtype=wp.int32),
        cell_atom_list: wp.array(dtype=wp.int32),
        cell_offsets: wp.array(dtype=wp.int32),
        target_indices: wp.array(dtype=wp.int32),
        neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
        neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
        num_neighbors: wp.array(dtype=wp.int32),
        neighbor_vectors: wp.array(dtype=vec_dtype, ndim=2),
        neighbor_distances: wp.array(dtype=wp_dtype, ndim=2),
        pair_params: wp.array(dtype=wp_dtype, ndim=2),
        pair_energies: wp.array(dtype=wp_dtype, ndim=2),
        pair_forces: wp.array(dtype=vec_dtype, ndim=2),
        rebuild_flags: wp.array(dtype=wp.bool),
    ) -> None:
        """Build cell-list neighbor-matrix rows with one source atom per thread

        Parameters
        ----------
        positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
            Atomic coordinates in original ordering.
        atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
            Periodic shifts in original ordering.
        sorted_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
            Cell-contiguous positions.
        sorted_atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
            Cell-contiguous periodic shifts.
        cell : wp.array, shape (num_systems,), dtype=wp.mat33*
            Cell matrices defining lattice vectors.
        pbc_single : wp.array, shape (3,), dtype=wp.bool
            Single-system PBC flags. Sentinel in batched mode.
        pbc_batch : wp.array, shape (num_systems, 3), dtype=wp.bool
            Batched PBC flags. Sentinel in single-system mode.
        batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
            System index per atom. Sentinel in single-system mode.
        cutoff : float
            Neighbor cutoff distance.
        cells_per_dimension_single : wp.array, shape (3,), dtype=wp.int32
            Single-system cell counts per dimension.
        cells_per_dimension_batch : wp.array, shape (num_systems,), dtype=wp.vec3i
            Batched cell counts per dimension.
        neighbor_search_radius_single : wp.array, shape (3,), dtype=wp.int32
            Single-system neighbor-search radius.
        neighbor_search_radius_batch : wp.array, shape (num_systems,), dtype=wp.vec3i
            Batched neighbor-search radius.
        atom_to_cell_mapping : wp.array, shape (total_atoms,), dtype=wp.vec3i
            Cell coordinates assigned to each atom.
        atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
            Atom counts per cell.
        cell_atom_start_indices : wp.array, shape (total_cells,), dtype=wp.int32
            Cell start offsets into ``cell_atom_list``.
        cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
            Atom indices in cell-contiguous order.
        cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
            Global cell offset per system. Sentinel in single-system mode.
        target_indices : wp.array, shape (n_targets,), dtype=wp.int32
            Compact target rows for partial neighbor lists. Sentinel for full mode.
        neighbor_matrix : wp.array, shape (rows, max_neighbors), dtype=wp.int32
            OUTPUT: Neighbor atom indices.
        neighbor_matrix_shifts : wp.array, shape (rows, max_neighbors), dtype=wp.vec3i
            OUTPUT: Periodic shift vectors.
        num_neighbors : wp.array, shape (rows,), dtype=wp.int32
            OUTPUT: Neighbor counts per row.
        neighbor_vectors : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*
            OUTPUT: Optional displacement vectors. Sentinel when disabled.
        neighbor_distances : wp.array, shape (rows, max_neighbors), dtype=wp.float*
            OUTPUT: Optional pair distances. Sentinel when disabled.
        pair_params : wp.array, shape (total_atoms, K), dtype=wp.float*
            Pair-function parameters. Sentinel when no pair function is active.
        pair_energies : wp.array, shape (rows, max_neighbors), dtype=wp.float*
            OUTPUT: Optional pair-function energies. Sentinel when disabled.
        pair_forces : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*
            OUTPUT: Optional pair-function forces. Sentinel when disabled.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            Selective rebuild flags. Sentinel for non-selective specializations.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: One thread per cell-list slot or compact target row.
        - Modifies: ``neighbor_matrix``, ``neighbor_matrix_shifts``, ``num_neighbors``, and enabled pair-output buffers.
        Batching, selective rebuild, partial rows, and pair outputs are static
        specializations. Inactive arguments are zero-size sentinels and are not read.

        See Also
        --------
        get_query_cell_list_kernel : Return the specialized atom-centric neighbor-search kernel.
        """
        row = wp.tid()
        atom_idx = wp.int32(0)
        output_row = wp.int32(0)
        if PARTIAL:
            if row >= target_indices.shape[0]:
                return
            atom_idx = target_indices[row]
            output_row = row
        elif ATOM_CENTRIC_DIRECT:
            if row >= positions.shape[0]:
                return
            atom_idx = row
            output_row = atom_idx
        else:
            if row >= sorted_positions.shape[0]:
                return
            atom_idx = cell_atom_list[row]
            if atom_idx >= sorted_positions.shape[0]:
                return
            output_row = atom_idx

        system_idx = wp.int32(0)
        if BATCHED:
            system_idx = batch_idx[atom_idx]
            if SELECTIVE and not rebuild_flags[system_idx]:
                return
        else:
            if SELECTIVE and not rebuild_flags[0]:
                return

        central_atom_position = positions[atom_idx]
        central_atom_shift = atom_periodic_shifts[atom_idx]
        if not ATOM_CENTRIC_DIRECT and not PARTIAL:
            central_atom_position = sorted_positions[row]
            central_atom_shift = sorted_atom_periodic_shifts[row]
        central_atom_cell_coords = atom_to_cell_mapping[atom_idx]

        cutoff_distance_sq = cutoff * cutoff
        max_neighbors = neighbor_matrix.shape[1]

        cell_mat = cell[system_idx]
        cell_transpose = wp.transpose(cell_mat)
        s_cell_offset = wp.int32(0)
        s_cells_per_dimension = wp.vec3i(0, 0, 0)
        s_neighbor_search_radius = wp.vec3i(0, 0, 0)
        pbc_x = wp.bool(False)
        pbc_y = wp.bool(False)
        pbc_z = wp.bool(False)
        if BATCHED:
            s_cell_offset = cell_offsets[system_idx]
            s_cells_per_dimension = cells_per_dimension_batch[system_idx]
            s_neighbor_search_radius = neighbor_search_radius_batch[system_idx]
            pbc_x = pbc_batch[system_idx, 0]
            pbc_y = pbc_batch[system_idx, 1]
            pbc_z = pbc_batch[system_idx, 2]
        else:
            s_cells_per_dimension = wp.vec3i(
                cells_per_dimension_single[0],
                cells_per_dimension_single[1],
                cells_per_dimension_single[2],
            )
            s_neighbor_search_radius = wp.vec3i(
                neighbor_search_radius_single[0],
                neighbor_search_radius_single[1],
                neighbor_search_radius_single[2],
            )
            pbc_x = pbc_single[0]
            pbc_y = pbc_single[1]
            pbc_z = pbc_single[2]

        cpd_x = s_cells_per_dimension[0]
        cpd_y = s_cells_per_dimension[1]
        cpd_z = s_cells_per_dimension[2]

        dx_lo = wp.int32(0)
        if not HALF_FILL and not SYMMETRIC_FULL_FILL:
            dx_lo = -s_neighbor_search_radius[0]

        n = wp.int32(0)
        for dx in range(dx_lo, s_neighbor_search_radius[0] + 1):
            for dy in range(
                -s_neighbor_search_radius[1], s_neighbor_search_radius[1] + 1
            ):
                for dz in range(
                    -s_neighbor_search_radius[2], s_neighbor_search_radius[2] + 1
                ):
                    if HALF_FILL or SYMMETRIC_FULL_FILL:
                        if not (
                            dx > 0
                            or (dx == 0 and dy > 0)
                            or (dx == 0 and dy == 0 and dz >= 0)
                        ):
                            continue
                    target_x = central_atom_cell_coords[0] + dx
                    target_y = central_atom_cell_coords[1] + dy
                    target_z = central_atom_cell_coords[2] + dz

                    if not pbc_x and (target_x < 0 or target_x >= cpd_x):
                        continue
                    if not pbc_y and (target_y < 0 or target_y >= cpd_y):
                        continue
                    if not pbc_z and (target_z < 0 or target_z >= cpd_z):
                        continue

                    cs_x, wc_x = wpdivmod(target_x, cpd_x)
                    cs_y, wc_y = wpdivmod(target_y, cpd_y)
                    cs_z, wc_z = wpdivmod(target_z, cpd_z)
                    linear_cell_index = (
                        s_cell_offset + wc_x + cpd_x * (wc_y + cpd_y * wc_z)
                    )
                    cell_start_index = cell_atom_start_indices[linear_cell_index]
                    num_atoms_in_cell = atoms_per_cell_count[linear_cell_index]

                    for cell_atom_idx in range(num_atoms_in_cell):
                        j_slot = cell_start_index + cell_atom_idx
                        neighbor_atom_idx = cell_atom_list[j_slot]
                        neighbor_atom_shift = atom_periodic_shifts[neighbor_atom_idx]
                        neighbor_pos = positions[neighbor_atom_idx]
                        if not ATOM_CENTRIC_DIRECT:
                            neighbor_atom_shift = sorted_atom_periodic_shifts[j_slot]
                            neighbor_pos = sorted_positions[j_slot]

                        shift_x = cs_x
                        shift_y = cs_y
                        shift_z = cs_z
                        if pbc_x:
                            shift_x += central_atom_shift[0] - neighbor_atom_shift[0]
                        else:
                            shift_x = 0
                        if pbc_y:
                            shift_y += central_atom_shift[1] - neighbor_atom_shift[1]
                        else:
                            shift_y = 0
                        if pbc_z:
                            shift_z += central_atom_shift[2] - neighbor_atom_shift[2]
                        else:
                            shift_z = 0

                        if dx == 0 and dy == 0 and dz == 0:
                            if HALF_FILL or SYMMETRIC_FULL_FILL:
                                if neighbor_atom_idx <= atom_idx:
                                    continue
                            else:
                                if neighbor_atom_idx == atom_idx:
                                    continue

                        if shift_x == 0 and shift_y == 0 and shift_z == 0:
                            dr = neighbor_pos - central_atom_position
                        else:
                            fractional_shift = type(central_atom_position)(
                                type(cutoff)(shift_x),
                                type(cutoff)(shift_y),
                                type(cutoff)(shift_z),
                            )
                            cartesian_shift = cell_transpose * fractional_shift
                            dr = neighbor_pos - central_atom_position + cartesian_shift
                        distance_sq = wp.dot(dr, dr)

                        if distance_sq < cutoff_distance_sq:
                            if SYMMETRIC_FULL_FILL:
                                pos_i = wp.atomic_add(num_neighbors, atom_idx, 1)
                                if pos_i < max_neighbors:
                                    store_neighbor(
                                        atom_idx,
                                        pos_i,
                                        atom_idx,
                                        neighbor_atom_idx,
                                        wp.vec3i(shift_x, shift_y, shift_z),
                                        dr,
                                        distance_sq,
                                        neighbor_matrix,
                                        neighbor_matrix_shifts,
                                        neighbor_vectors,
                                        neighbor_distances,
                                        pair_params,
                                        pair_energies,
                                        pair_forces,
                                    )
                                pos_j = wp.atomic_add(
                                    num_neighbors, neighbor_atom_idx, 1
                                )
                                if pos_j < max_neighbors:
                                    store_neighbor(
                                        neighbor_atom_idx,
                                        pos_j,
                                        neighbor_atom_idx,
                                        atom_idx,
                                        wp.vec3i(-shift_x, -shift_y, -shift_z),
                                        -dr,
                                        distance_sq,
                                        neighbor_matrix,
                                        neighbor_matrix_shifts,
                                        neighbor_vectors,
                                        neighbor_distances,
                                        pair_params,
                                        pair_energies,
                                        pair_forces,
                                    )
                            elif n < max_neighbors:
                                store_neighbor(
                                    output_row,
                                    n,
                                    atom_idx,
                                    neighbor_atom_idx,
                                    wp.vec3i(shift_x, shift_y, shift_z),
                                    dr,
                                    distance_sq,
                                    neighbor_matrix,
                                    neighbor_matrix_shifts,
                                    neighbor_vectors,
                                    neighbor_distances,
                                    pair_params,
                                    pair_energies,
                                    pair_forces,
                                )
                            if not SYMMETRIC_FULL_FILL:
                                n += 1

        if not SYMMETRIC_FULL_FILL:
            num_neighbors[output_row] = n

    name = kernel_specialization_name(
        _cell_list_neighbor_base_name(
            batched=bool(batched),
            selective=bool(selective),
        ),
        wp_dtype=wp_dtype,
        features=(
            "atom_centric",
            # The default ("sorted") path carries the canonical 0.3.1-style name
            # (no path token); only the codegen-distinct "direct" path adds a
            # token so the two kernels keep separate Warp cache keys.
            "direct" if atom_centric_path == "direct" else "",
            "symmetric_full"
            if (
                atom_centric_path == "direct"
                and not bool(half_fill)
                and not bool(partial)
                and not bool(return_vectors)
                and not bool(return_distances)
                and pair_fn is None
            )
            else "",
            "half" if bool(half_fill) else "",
            "partial" if partial else "",
            *_pair_output_features(
                return_vectors=return_vectors,
                return_distances=return_distances,
                pair_fn=pair_fn,
            ),
        ),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp_dtype,
            entries=(
                ("strategy", "atom_centric"),
                ("atom_centric_path", atom_centric_path),
                ("batched", bool(batched)),
                ("selective", bool(selective)),
                ("partial", bool(partial)),
                ("half_fill", bool(half_fill)),
                ("return_vectors", bool(return_vectors)),
                ("return_distances", bool(return_distances)),
                ("pair_fn", pair_fn is not None),
            ),
        ),
    )


@lru_cache(maxsize=None)
def _make_pair_centric_kernel(
    wp_dtype: type,
    *,
    batched: bool,
    selective: bool,
    partial: bool,
    half_fill: bool,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
) -> wp.Kernel:
    """Build the pair-centric neighbor-matrix kernel."""
    _require_supported_dtype(wp_dtype)
    vec_dtype, mat_dtype = _DTYPE_INFO[wp_dtype]
    BATCHED = wp.constant(bool(batched))
    SELECTIVE = wp.constant(bool(selective))
    PARTIAL = wp.constant(bool(partial))
    HALF_FILL = wp.constant(bool(half_fill))
    store_neighbor = _make_store_neighbor_fn(
        wp_dtype,
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        pair_fn=pair_fn,
    )

    @wp.kernel(enable_backward=False, module="unique")
    def _kernel(
        sorted_positions: wp.array(dtype=vec_dtype),
        sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
        cell: wp.array(dtype=mat_dtype),
        pbc_single: wp.array(dtype=wp.bool),
        pbc_batch: wp.array2d(dtype=wp.bool),
        cutoff: wp_dtype,
        cells_per_dimension_single: wp.array(dtype=wp.int32),
        cells_per_dimension_batch: wp.array(dtype=wp.vec3i),
        neighbor_search_radius_single: wp.array(dtype=wp.int32),
        neighbor_search_radius_batch: wp.array(dtype=wp.vec3i),
        atoms_per_cell_count: wp.array(dtype=wp.int32),
        cell_atom_start_indices: wp.array(dtype=wp.int32),
        cell_atom_list: wp.array(dtype=wp.int32),
        cell_offsets: wp.array(dtype=wp.int32),
        cell_to_system: wp.array(dtype=wp.int32),
        target_row_lookup: wp.array(dtype=wp.int32),
        neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
        neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
        num_neighbors: wp.array(dtype=wp.int32),
        neighbor_vectors: wp.array(dtype=vec_dtype, ndim=2),
        neighbor_distances: wp.array(dtype=wp_dtype, ndim=2),
        pair_params: wp.array(dtype=wp_dtype, ndim=2),
        pair_energies: wp.array(dtype=wp_dtype, ndim=2),
        pair_forces: wp.array(dtype=vec_dtype, ndim=2),
        block_dim_const: wp.int32,
        total_cells: wp.int32,
        n_offsets: wp.int32,
        max_radius: wp.vec3i,
        rebuild_flags: wp.array(dtype=wp.bool),
    ) -> None:
        """Build cell-list neighbor-matrix rows with one block per cell/offset pair

        Parameters
        ----------
        sorted_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
            Cell-contiguous positions.
        sorted_atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
            Cell-contiguous periodic shifts.
        cell : wp.array, shape (num_systems,), dtype=wp.mat33*
            Cell matrices defining lattice vectors.
        pbc_single : wp.array, shape (3,), dtype=wp.bool
            Single-system PBC flags. Sentinel in batched mode.
        pbc_batch : wp.array, shape (num_systems, 3), dtype=wp.bool
            Batched PBC flags. Sentinel in single-system mode.
        cutoff : float
            Neighbor cutoff distance.
        cells_per_dimension_single : wp.array, shape (3,), dtype=wp.int32
            Single-system cell counts per dimension.
        cells_per_dimension_batch : wp.array, shape (num_systems,), dtype=wp.vec3i
            Batched cell counts per dimension.
        neighbor_search_radius_single : wp.array, shape (3,), dtype=wp.int32
            Single-system neighbor-search radius.
        neighbor_search_radius_batch : wp.array, shape (num_systems,), dtype=wp.vec3i
            Batched neighbor-search radius.
        atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
            Atom counts per cell.
        cell_atom_start_indices : wp.array, shape (total_cells,), dtype=wp.int32
            Cell start offsets into ``cell_atom_list``.
        cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
            Atom indices in cell-contiguous order.
        cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
            Global cell offset per system. Sentinel in single-system mode.
        cell_to_system : wp.array, shape (total_cells,), dtype=wp.int32
            System index for each global cell. Sentinel in single-system mode.
        target_row_lookup : wp.array, shape (total_atoms,), dtype=wp.int32
            Atom-id to compact target-row lookup for partial mode.
        neighbor_matrix : wp.array, shape (rows, max_neighbors), dtype=wp.int32
            OUTPUT: Neighbor atom indices.
        neighbor_matrix_shifts : wp.array, shape (rows, max_neighbors), dtype=wp.vec3i
            OUTPUT: Periodic shift vectors.
        num_neighbors : wp.array, shape (rows,), dtype=wp.int32
            OUTPUT: Neighbor counts per row, updated atomically.
        neighbor_vectors : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*
            OUTPUT: Optional displacement vectors. Sentinel when disabled.
        neighbor_distances : wp.array, shape (rows, max_neighbors), dtype=wp.float*
            OUTPUT: Optional pair distances. Sentinel when disabled.
        pair_params : wp.array, shape (total_atoms, K), dtype=wp.float*
            Pair-function parameters. Sentinel when no pair function is active.
        pair_energies : wp.array, shape (rows, max_neighbors), dtype=wp.float*
            OUTPUT: Optional pair-function energies. Sentinel when disabled.
        pair_forces : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*
            OUTPUT: Optional pair-function forces. Sentinel when disabled.
        block_dim_const : wp.int32
            Runtime copy of the CUDA block dimension.
        total_cells : wp.int32
            Number of global cells to traverse.
        n_offsets : wp.int32
            Number of neighbor-cell offsets encoded in the launch grid.
        max_radius : wp.vec3i
            Maximum search radius used to decode batched offsets.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            Selective rebuild flags. Sentinel for non-selective specializations.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: One CUDA block per source-cell/offset pair, with lanes covering source-cell atoms.
        - Modifies: ``neighbor_matrix``, ``neighbor_matrix_shifts``, ``num_neighbors`` atomically, and enabled pair-output buffers.
        Batching, selective rebuild, partial rows, and pair outputs are static
        specializations. Pair-centric execution is CUDA-only at the launcher boundary.

        See Also
        --------
        get_query_cell_list_kernel : Return the specialized pair-centric neighbor-search kernel.
        """
        tid = wp.tid()
        bid = tid / block_dim_const
        lane = tid - bid * block_dim_const
        source_cell = bid / n_offsets
        offset_idx = bid % n_offsets
        if source_cell >= total_cells:
            return

        system_idx = wp.int32(0)
        s_cell_offset = wp.int32(0)
        s_cpd = wp.vec3i(0, 0, 0)
        s_nsr = wp.vec3i(0, 0, 0)
        pbc_x = wp.bool(False)
        pbc_y = wp.bool(False)
        pbc_z = wp.bool(False)
        if BATCHED:
            system_idx = cell_to_system[source_cell]
            if SELECTIVE and not rebuild_flags[system_idx]:
                return
            s_cell_offset = cell_offsets[system_idx]
            s_cpd = cells_per_dimension_batch[system_idx]
            s_nsr = neighbor_search_radius_batch[system_idx]
            pbc_x = pbc_batch[system_idx, 0]
            pbc_y = pbc_batch[system_idx, 1]
            pbc_z = pbc_batch[system_idx, 2]
        else:
            if SELECTIVE and not rebuild_flags[0]:
                return
            s_cpd = wp.vec3i(
                cells_per_dimension_single[0],
                cells_per_dimension_single[1],
                cells_per_dimension_single[2],
            )
            s_nsr = wp.vec3i(
                neighbor_search_radius_single[0],
                neighbor_search_radius_single[1],
                neighbor_search_radius_single[2],
            )
            pbc_x = pbc_single[0]
            pbc_y = pbc_single[1]
            pbc_z = pbc_single[2]

        src_count = atoms_per_cell_count[source_cell]
        if src_count == 0:
            return
        src_start = cell_atom_start_indices[source_cell]

        cpd_x = s_cpd[0]
        cpd_y = s_cpd[1]
        cpd_z = s_cpd[2]
        cpd_xy = cpd_x * cpd_y

        local_cell = source_cell - s_cell_offset
        cax = local_cell % cpd_x
        cay = (local_cell / cpd_x) % cpd_y
        caz = local_cell / cpd_xy

        decode_radius = max_radius
        if not BATCHED:
            decode_radius = s_nsr

        offset_vec = wp.vec3i(0, 0, 0)
        if HALF_FILL:
            offset_vec = _decode_shift_index(offset_idx, decode_radius)
        else:
            if offset_idx == 0:
                offset_vec = wp.vec3i(0, 0, 0)
            else:
                offset_vec = _decode_full_shift_index(offset_idx - 1, decode_radius)
        dx_v = offset_vec[0]
        dy_v = offset_vec[1]
        dz_v = offset_vec[2]
        is_self = dx_v == 0 and dy_v == 0 and dz_v == 0

        if dx_v > s_nsr[0] or dx_v < -s_nsr[0]:
            return
        if dy_v > s_nsr[1] or dy_v < -s_nsr[1]:
            return
        if dz_v > s_nsr[2] or dz_v < -s_nsr[2]:
            return

        target_x = cax + dx_v
        target_y = cay + dy_v
        target_z = caz + dz_v

        if not pbc_x and (target_x < 0 or target_x >= cpd_x):
            return
        if not pbc_y and (target_y < 0 or target_y >= cpd_y):
            return
        if not pbc_z and (target_z < 0 or target_z >= cpd_z):
            return

        cs_x_base, wc_x = wpdivmod(target_x, cpd_x)
        cs_y_base, wc_y = wpdivmod(target_y, cpd_y)
        cs_z_base, wc_z = wpdivmod(target_z, cpd_z)

        nbr_cell = s_cell_offset + wc_x + cpd_x * (wc_y + cpd_y * wc_z)
        nbr_count = atoms_per_cell_count[nbr_cell]
        if nbr_count == 0:
            return
        nbr_start = cell_atom_start_indices[nbr_cell]

        cutoff_distance_sq = cutoff * cutoff
        max_neighbors = neighbor_matrix.shape[1]
        cell_transpose = wp.transpose(cell[system_idx])

        slot = lane
        while slot < src_count:
            s = src_start + slot
            atom_idx = cell_atom_list[s]
            output_row = atom_idx
            if PARTIAL:
                output_row = target_row_lookup[atom_idx]
                if output_row < 0:
                    slot += block_dim_const
                    continue
            central_atom_position = sorted_positions[s]
            central_atom_shift = sorted_atom_periodic_shifts[s]

            for j_local in range(nbr_count):
                if is_self and j_local == slot:
                    continue
                j_slot = nbr_start + j_local
                neighbor_atom_idx = cell_atom_list[j_slot]
                if is_self and HALF_FILL and neighbor_atom_idx <= atom_idx:
                    continue
                neighbor_atom_shift = sorted_atom_periodic_shifts[j_slot]
                neighbor_pos = sorted_positions[j_slot]

                shift_x = cs_x_base
                shift_y = cs_y_base
                shift_z = cs_z_base

                if pbc_x:
                    shift_x += central_atom_shift[0] - neighbor_atom_shift[0]
                else:
                    shift_x = 0

                if pbc_y:
                    shift_y += central_atom_shift[1] - neighbor_atom_shift[1]
                else:
                    shift_y = 0

                if pbc_z:
                    shift_z += central_atom_shift[2] - neighbor_atom_shift[2]
                else:
                    shift_z = 0

                if shift_x == 0 and shift_y == 0 and shift_z == 0:
                    dr = neighbor_pos - central_atom_position
                else:
                    fractional_shift = type(central_atom_position)(
                        type(central_atom_position[0])(shift_x),
                        type(central_atom_position[0])(shift_y),
                        type(central_atom_position[0])(shift_z),
                    )
                    cartesian_shift = cell_transpose * fractional_shift
                    dr = neighbor_pos - central_atom_position + cartesian_shift
                distance_sq = wp.dot(dr, dr)

                if distance_sq < cutoff_distance_sq:
                    output_slot = wp.atomic_add(num_neighbors, output_row, 1)
                    if output_slot < max_neighbors:
                        store_neighbor(
                            output_row,
                            output_slot,
                            atom_idx,
                            neighbor_atom_idx,
                            wp.vec3i(shift_x, shift_y, shift_z),
                            dr,
                            distance_sq,
                            neighbor_matrix,
                            neighbor_matrix_shifts,
                            neighbor_vectors,
                            neighbor_distances,
                            pair_params,
                            pair_energies,
                            pair_forces,
                        )
            slot += block_dim_const

    name = kernel_specialization_name(
        _cell_list_neighbor_base_name(
            batched=bool(batched),
            selective=bool(selective),
        ),
        wp_dtype=wp_dtype,
        features=(
            "pair_centric",
            "half" if bool(half_fill) else "",
            "partial" if partial else "",
            *_pair_output_features(
                return_vectors=return_vectors,
                return_distances=return_distances,
                pair_fn=pair_fn,
            ),
        ),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp_dtype,
            entries=(
                ("strategy", "pair_centric"),
                ("batched", bool(batched)),
                ("selective", bool(selective)),
                ("partial", bool(partial)),
                ("half_fill", bool(half_fill)),
                ("return_vectors", bool(return_vectors)),
                ("return_distances", bool(return_distances)),
                ("pair_fn", pair_fn is not None),
            ),
        ),
    )


def get_query_cell_list_kernel(
    wp_dtype: type,
    *,
    strategy: str = "atom_centric",
    batched: bool = False,
    selective: bool = False,
    partial: bool = False,
    half_fill: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    atom_centric_path: str = "sorted",
) -> wp.Kernel:
    """Return a cached cell-list neighbor-matrix kernel."""
    _require_supported_dtype(wp_dtype)
    if strategy == "atom_centric":
        if atom_centric_path not in {"direct", "sorted"}:
            raise ValueError(
                "atom_centric_path must be 'direct' | 'sorted', "
                f"got {atom_centric_path!r}",
            )
        return _make_atom_centric_kernel(
            wp_dtype,
            batched=bool(batched),
            selective=bool(selective),
            partial=bool(partial),
            half_fill=bool(half_fill),
            return_vectors=bool(return_vectors),
            return_distances=bool(return_distances),
            pair_fn=pair_fn,
            atom_centric_path=atom_centric_path,
        )
    if strategy == "pair_centric":
        return _make_pair_centric_kernel(
            wp_dtype,
            batched=bool(batched),
            selective=bool(selective),
            partial=bool(partial),
            half_fill=bool(half_fill),
            return_vectors=bool(return_vectors),
            return_distances=bool(return_distances),
            pair_fn=pair_fn,
        )
    raise ValueError(
        f"strategy must be 'atom_centric' | 'pair_centric', got {strategy!r}"
    )


def get_build_cell_list_kernel(
    stage: Literal["estimate_sizes", "construct_bin_size", "count_atoms", "bin_atoms"],
    wp_dtype: type,
    *,
    batched: bool = False,
    min_cells_per_dimension: int = 4,
) -> wp.Kernel:
    """Return a cached cell-list CSR build kernel.

    Parameters
    ----------
    stage : {"estimate_sizes", "construct_bin_size", "count_atoms", "bin_atoms"}
        Cell-list build stage to select.
    wp_dtype : type
        Warp scalar dtype (``wp.float32`` or ``wp.float64``).
    batched : bool, default False
        Select the batched static specialization.
    min_cells_per_dimension : int, default 4
        Lower bound for the per-axis cell count in sizing stages.  Pass 1 for
        the legacy cell-grid rule.

    Returns
    -------
    wp.Kernel
        Cached Warp kernel for the requested CSR build stage.
    """
    _require_supported_dtype(wp_dtype)
    stage_name = str(stage)
    batched_mode = bool(batched)

    match stage_name:
        case "estimate_sizes":
            return _make_estimate_cell_list_sizes_kernel(
                wp_dtype,
                batched=batched_mode,
                min_cells_per_dimension=int(min_cells_per_dimension),
            )
        case "construct_bin_size":
            return _make_construct_bin_size_kernel(
                wp_dtype,
                batched=batched_mode,
                min_cells_per_dimension=int(min_cells_per_dimension),
            )
        case "count_atoms":
            return _make_count_atoms_per_bin_kernel(wp_dtype, batched=batched_mode)
        case "bin_atoms":
            return _make_bin_atoms_kernel(wp_dtype, batched=batched_mode)

    raise ValueError(
        "stage must be 'estimate_sizes', 'construct_bin_size', "
        f"'count_atoms', or 'bin_atoms'; got {stage!r}"
    )


@wp.kernel(enable_backward=False)
def _gather_positions_by_cell(
    positions: wp.array(dtype=Any),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=Any),
    sorted_shifts: wp.array(dtype=wp.vec3i),
) -> None:
    """Reorder per-atom positions and shifts into cell-contiguous layout

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Per-atom positions in original ordering.
    atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Per-atom integer PBC shift vectors in original ordering.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        Cell-contiguous atom indices output by ``batch_build_cell_list``.
    sorted_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT: positions gathered in cell-contiguous order.
    sorted_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
        OUTPUT: shifts gathered in cell-contiguous order.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: One thread per cell-list slot (dim=total_atoms)
    - Modifies: sorted_positions, sorted_shifts

    See Also
    --------
    get_cell_list_gather_kernel : Return the specialized sorted-position gather kernel.
    """
    idx = wp.tid()
    atom_idx = cell_atom_list[idx]
    sorted_positions[idx] = positions[atom_idx]
    sorted_shifts[idx] = atom_periodic_shifts[atom_idx]


@wp.kernel(enable_backward=False)
def _compute_cells_per_system(
    cells_per_dimension: wp.array(dtype=wp.vec3i),
    cells_per_system: wp.array(dtype=wp.int32),
) -> None:
    """Compute total cells per system from cell dimension vectors

    Parameters
    ----------
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Number of cells in x, y, z directions for each system.
    cells_per_system : wp.array, shape (num_systems,), dtype=wp.int32
        OUTPUT: Total number of cells for each system.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: One thread per system (dim=num_systems)
    - Modifies: cells_per_system

    See Also
    --------
    get_cell_list_cells_per_system_kernel : Return the specialized cells-per-system kernel.
    """
    system_idx = wp.tid()
    dims = cells_per_dimension[system_idx]
    cells_per_system[system_idx] = dims[0] * dims[1] * dims[2]


def get_cell_list_cells_per_system_kernel() -> wp.Kernel:
    """Return the cell-list helper kernel computing cells per system."""
    return _compute_cells_per_system


@wp.kernel(enable_backward=False)
def _build_cell_to_system_map(
    cell_offsets: wp.array(dtype=wp.int32),
    cells_per_system: wp.array(dtype=wp.int32),
    cell_to_system: wp.array(dtype=wp.int32),
) -> None:
    """Build a global-cell-index to system-index lookup table

    Parameters
    ----------
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        Starting global cell index of each system.
    cells_per_system : wp.array, shape (num_systems,), dtype=wp.int32
        Number of cells in each system.
    cell_to_system : wp.array, shape (total_cells,), dtype=wp.int32
        OUTPUT: system index for each global cell.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: One thread per system (dim=num_systems)
    - Modifies: cell_to_system

    See Also
    --------
    batch_query_cell_list_pair_centric_sorted : Launch batched pair-centric search that uses the cell-to-system map.
    """
    system_idx = wp.tid()
    offset = cell_offsets[system_idx]
    count = cells_per_system[system_idx]
    for cell_idx in range(count):
        cell_to_system[offset + cell_idx] = system_idx


@lru_cache(maxsize=None)
def get_cell_list_gather_kernel(wp_dtype: type) -> wp.Kernel:
    """Return a cached gather kernel for cell-contiguous position reads."""
    _require_supported_dtype(wp_dtype)
    vec_dtype = _DTYPE_INFO[wp_dtype][0]
    kernel = wp.overload(
        _gather_positions_by_cell,
        [
            wp.array(dtype=vec_dtype),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=vec_dtype),
            wp.array(dtype=wp.vec3i),
        ],
    )
    name = kernel_specialization_name(
        "_cell_list_gather_positions_by_cell",
        wp_dtype=wp_dtype,
    )
    return set_fn_doc(
        set_fn_name(kernel, name),
        _append_specialization_doc(
            kernel.__doc__,
            dtype=wp_dtype,
            entries=(("operation", "cell_list_gather_positions_by_cell"),),
        ),
    )
