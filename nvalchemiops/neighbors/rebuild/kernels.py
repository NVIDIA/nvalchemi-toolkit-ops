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

"""Warp kernel factories for rebuild detection."""

from functools import lru_cache
from typing import Any

import warp as wp

from nvalchemiops.math import wpdivmod
from nvalchemiops.neighbors.neighbor_utils import (
    _append_specialization_doc,
    dtype_info,
    kernel_specialization_name,
    require_supported_dtype,
    set_fn_doc,
    set_fn_name,
)

__all__ = [
    "get_cell_list_rebuild_kernel",
    "get_neighbor_list_rebuild_kernel",
]


@wp.func
def _cell_coords_from_fractional(
    fractional_position: Any,
    cells_per_dimension: wp.vec3i,
    pbc_x: bool,
    pbc_y: bool,
    pbc_z: bool,
) -> wp.vec3i:
    """Map fractional coordinates to cell-list coordinates

    Parameters
    ----------
    fractional_position : wp.vec3*
        Position in fractional cell coordinates.
    cells_per_dimension : wp.vec3i
        Number of cells in each dimension.
    pbc_x : bool
        Whether the x direction is periodic.
    pbc_y : bool
        Whether the y direction is periodic.
    pbc_z : bool
        Whether the z direction is periodic.

    Returns
    -------
    wp.vec3i
        Wrapped or clamped cell coordinates.
    """
    coords = wp.vec3i(0, 0, 0)
    for dim in range(3):
        coords[dim] = wp.int32(
            wp.floor(
                fractional_position[dim]
                * type(fractional_position[dim])(cells_per_dimension[dim])
            )
        )
        pbc_dim = pbc_z
        if dim == 0:
            pbc_dim = pbc_x
        elif dim == 1:
            pbc_dim = pbc_y
        if pbc_dim:
            _quotient, remainder = wpdivmod(coords[dim], cells_per_dimension[dim])
            coords[dim] = remainder
        else:
            coords[dim] = wp.clamp(coords[dim], 0, cells_per_dimension[dim] - 1)
    return coords


@wp.func
def _cell_coords_changed(
    current_cell_coords: wp.vec3i, reference_cell_coords: wp.vec3i
) -> bool:
    """Return whether cell-list coordinates changed

    Parameters
    ----------
    current_cell_coords : wp.vec3i
        Cell coordinates computed from the current position.
    reference_cell_coords : wp.vec3i
        Cell coordinates stored from the previous rebuild.

    Returns
    -------
    bool
        True when any coordinate component changed.
    """
    return (
        current_cell_coords[0] != reference_cell_coords[0]
        or current_cell_coords[1] != reference_cell_coords[1]
        or current_cell_coords[2] != reference_cell_coords[2]
    )


@wp.func
def _minimum_image_displacement(
    delta: Any,
    cell: Any,
    cell_inv: Any,
    pbc_x: bool,
    pbc_y: bool,
    pbc_z: bool,
):
    """Apply the minimum-image convention to a displacement

    Parameters
    ----------
    delta : wp.vec3*
        Cartesian displacement before minimum-image wrapping.
    cell : wp.mat33*
        Cell matrix.
    cell_inv : wp.mat33*
        Inverse cell matrix.
    pbc_x : bool
        Whether the x direction is periodic.
    pbc_y : bool
        Whether the y direction is periodic.
    pbc_z : bool
        Whether the z direction is periodic.

    Returns
    -------
    wp.vec3*
        Minimum-image Cartesian displacement.
    """
    delta_frac = delta * cell_inv
    if pbc_x:
        delta_frac[0] -= wp.floor(delta_frac[0] + type(delta_frac[0])(0.5))
    if pbc_y:
        delta_frac[1] -= wp.floor(delta_frac[1] + type(delta_frac[1])(0.5))
    if pbc_z:
        delta_frac[2] -= wp.floor(delta_frac[2] + type(delta_frac[2])(0.5))
    return delta_frac * cell


@wp.func
def _moved_beyond_skin(displacement: Any, skin_distance_threshold: Any) -> bool:
    """Return whether a displacement exceeds the rebuild threshold

    Parameters
    ----------
    displacement : wp.vec3*
        Cartesian displacement from reference position.
    skin_distance_threshold : float
        Movement threshold that triggers rebuild.

    Returns
    -------
    bool
        True when ``displacement`` length is larger than ``skin_distance_threshold``.
    """
    return wp.length(displacement) > skin_distance_threshold


def _make_neighbor_rebuild_kernel(
    wp_dtype: type,
    *,
    batched: bool,
    pbc: bool,
) -> wp.Kernel:
    """Build a neighbor-list rebuild detection kernel.

    The returned kernel has one superset signature for both batched and
    single-system launchers. ``batched`` and ``pbc`` are captured as static
    ``wp.constant`` axes, so Warp can eliminate unused branches while public
    launchers pass zero-size sentinel arrays for inactive inputs.

    Parameters
    ----------
    wp_dtype : type
        Warp scalar dtype (``wp.float16``, ``wp.float32``, or ``wp.float64``).
    batched : bool
        Select a per-system ``rebuild_flags[batch_idx[i]]`` contract instead
        of the single-system ``rebuild_flags[0]`` contract.
    pbc : bool
        Select minimum-image displacement using ``cell``, ``cell_inv`` and PBC
        flags instead of raw Euclidean displacement.

    Returns
    -------
    wp.Kernel
        Specialized rebuild-detection kernel.
    """
    require_supported_dtype(wp_dtype)
    vec_dtype, mat_dtype = dtype_info(wp_dtype)
    BATCHED = wp.constant(bool(batched))
    PBC = wp.constant(bool(pbc))

    @wp.kernel(enable_backward=False)
    def _kernel(
        reference_positions: wp.array(dtype=vec_dtype),
        current_positions: wp.array(dtype=vec_dtype),
        batch_idx: wp.array(dtype=wp.int32),
        cell: wp.array(dtype=mat_dtype),
        cell_inv: wp.array(dtype=mat_dtype),
        pbc_flags: wp.array(dtype=wp.bool),
        pbc_flags_batch: wp.array2d(dtype=wp.bool),
        skin_distance_threshold: wp_dtype,
        rebuild_flags: wp.array(dtype=wp.bool),
    ) -> None:
        """Detect atoms moved beyond skin distance

        Parameters
        ----------
        reference_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
            Reference positions from the previous neighbor-list rebuild.
        current_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
            Current positions to compare against ``reference_positions``.
        batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
            System index per atom. Zero-size sentinel in single-system mode.
        cell : wp.array, shape (num_systems,), dtype=wp.mat33*
            Cell matrices for PBC minimum-image displacement.
        cell_inv : wp.array, shape (num_systems,), dtype=wp.mat33*
            Inverse cell matrices for PBC minimum-image displacement.
        pbc_flags : wp.array, shape (3,), dtype=wp.bool
            Single-system PBC flags. Sentinel in batched or no-PBC modes.
        pbc_flags_batch : wp.array, shape (num_systems, 3), dtype=wp.bool
            Batched PBC flags. Sentinel in single-system or no-PBC modes.
        skin_distance_threshold : float
            Movement threshold that triggers a rebuild.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            OUTPUT: Rebuild flag per system.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: One thread per atom.
        - Modifies: ``rebuild_flags``; either entry ``0`` for single-system launches or ``batch_idx[atom]`` for batched launches.
        ``BATCHED`` and ``PBC`` are static specializations. Unused superset
        inputs are zero-size sentinels and are not read.

        See Also
        --------
        get_neighbor_list_rebuild_kernel : Return the specialized neighbor-list rebuild detection kernel.
        """
        atom_idx = wp.tid()
        if atom_idx >= reference_positions.shape[0]:
            return

        isys = wp.int32(0)
        if BATCHED:
            isys = batch_idx[atom_idx]

        if rebuild_flags[isys]:
            return

        displacement = current_positions[atom_idx] - reference_positions[atom_idx]
        if PBC:
            if BATCHED:
                pbc_x = pbc_flags_batch[isys, 0]
                pbc_y = pbc_flags_batch[isys, 1]
                pbc_z = pbc_flags_batch[isys, 2]
            else:
                pbc_x = pbc_flags[0]
                pbc_y = pbc_flags[1]
                pbc_z = pbc_flags[2]
            displacement = _minimum_image_displacement(
                displacement,
                cell[isys],
                cell_inv[isys],
                pbc_x,
                pbc_y,
                pbc_z,
            )

        if _moved_beyond_skin(displacement, skin_distance_threshold):
            rebuild_flags[isys] = True

    base = (
        "_check_batch_atoms_moved_beyond_skin"
        if batched
        else "_check_atoms_moved_beyond_skin"
    )
    if pbc:
        base = f"{base}_pbc"
    name = kernel_specialization_name(base, wp_dtype=wp_dtype)
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp_dtype,
            entries=(
                ("operation", "neighbor_list_rebuild"),
                ("batched", bool(batched)),
                ("pbc", bool(pbc)),
            ),
        ),
    )


def _make_cell_rebuild_kernel(
    wp_dtype: type,
    *,
    batched: bool,
) -> wp.Kernel:
    """Build a cell-list rebuild detection kernel.

    The returned kernel has one superset signature for both batched and
    single-system launchers. ``batched`` is captured as a static
    ``wp.constant`` axis, so Warp can eliminate the inactive shape path while
    public launchers pass zero-size sentinel arrays for unused inputs.

    Parameters
    ----------
    wp_dtype : type
        Warp scalar dtype (``wp.float16``, ``wp.float32``, or ``wp.float64``).
    batched : bool
        Select per-system ``cells_per_dimension`` / PBC arrays and
        ``rebuild_flags[batch_idx[i]]`` instead of the single-system arrays.

    Returns
    -------
    wp.Kernel
        Specialized cell-list rebuild-detection kernel.
    """
    require_supported_dtype(wp_dtype)
    vec_dtype, mat_dtype = dtype_info(wp_dtype)
    BATCHED = wp.constant(bool(batched))

    @wp.kernel(enable_backward=False)
    def _kernel(
        current_positions: wp.array(dtype=vec_dtype),
        cell: wp.array(dtype=mat_dtype),
        atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
        batch_idx: wp.array(dtype=wp.int32),
        cells_per_dimension: wp.array(dtype=wp.int32),
        cells_per_dimension_batch: wp.array(dtype=wp.vec3i),
        pbc_flags: wp.array(dtype=wp.bool),
        pbc_flags_batch: wp.array2d(dtype=wp.bool),
        rebuild_flags: wp.array(dtype=wp.bool),
    ) -> None:
        """Detect atoms moved between cell-list cells

        Parameters
        ----------
        current_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
            Current atomic positions.
        cell : wp.array, shape (num_systems,), dtype=wp.mat33*
            Cell matrices for fractional-coordinate transforms.
        atom_to_cell_mapping : wp.array, shape (total_atoms,), dtype=wp.vec3i
            Previously stored cell coordinates for each atom.
        batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
            System index per atom. Zero-size sentinel in single-system mode.
        cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
            Single-system cell counts per dimension.
        cells_per_dimension_batch : wp.array, shape (num_systems,), dtype=wp.vec3i
            Batched cell counts per dimension.
        pbc_flags : wp.array, shape (3,), dtype=wp.bool
            Single-system PBC flags. Sentinel in batched mode.
        pbc_flags_batch : wp.array, shape (num_systems, 3), dtype=wp.bool
            Batched PBC flags. Sentinel in single-system mode.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            OUTPUT: Rebuild flag per system.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: One thread per atom.
        - Modifies: ``rebuild_flags``; either entry ``0`` for single-system launches or ``batch_idx[atom]`` for batched launches.
        ``BATCHED`` is a static specialization. Unused superset inputs are
        zero-size sentinels and are not read.

        See Also
        --------
        get_cell_list_rebuild_kernel : Return the specialized cell-list rebuild detection kernel.
        """
        atom_idx = wp.tid()
        if atom_idx >= current_positions.shape[0]:
            return

        isys = wp.int32(0)
        if BATCHED:
            isys = batch_idx[atom_idx]

        if rebuild_flags[isys]:
            return

        if BATCHED:
            inv_cell = wp.inverse(cell[isys])
            fractional_position = current_positions[atom_idx] * inv_cell
            current_cell_coords = _cell_coords_from_fractional(
                fractional_position,
                cells_per_dimension_batch[isys],
                pbc_flags_batch[isys, 0],
                pbc_flags_batch[isys, 1],
                pbc_flags_batch[isys, 2],
            )
        else:
            inverse_cell_transpose = wp.transpose(wp.inverse(cell[0]))
            fractional_position = inverse_cell_transpose * current_positions[atom_idx]
            current_cell_coords = _cell_coords_from_fractional(
                fractional_position,
                wp.vec3i(
                    cells_per_dimension[0],
                    cells_per_dimension[1],
                    cells_per_dimension[2],
                ),
                pbc_flags[0],
                pbc_flags[1],
                pbc_flags[2],
            )

        if _cell_coords_changed(current_cell_coords, atom_to_cell_mapping[atom_idx]):
            rebuild_flags[isys] = True

    base = (
        "_check_batch_atoms_changed_cells" if batched else "_check_atoms_changed_cells"
    )
    name = kernel_specialization_name(base, wp_dtype=wp_dtype)
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp_dtype,
            entries=(
                ("operation", "cell_list_rebuild"),
                ("batched", bool(batched)),
            ),
        ),
    )


@lru_cache(maxsize=None)
def get_neighbor_list_rebuild_kernel(
    wp_dtype: type,
    *,
    batched: bool = False,
    pbc: bool = False,
) -> wp.Kernel:
    """Return a cached neighbor-list rebuild detection kernel.

    Parameters
    ----------
    wp_dtype : type
        Warp scalar dtype (``wp.float16``, ``wp.float32``, or ``wp.float64``).
    batched : bool, optional
        Select the batched per-system kernel variant.
    pbc : bool, optional
        Select the minimum-image neighbor-list kernel.

    Returns
    -------
    wp.Kernel
        The compiled and cached neighbor-list rebuild detection kernel.
    """
    require_supported_dtype(wp_dtype)
    return _make_neighbor_rebuild_kernel(wp_dtype, batched=bool(batched), pbc=bool(pbc))


@lru_cache(maxsize=None)
def get_cell_list_rebuild_kernel(
    wp_dtype: type,
    *,
    batched: bool = False,
) -> wp.Kernel:
    """Return a cached cell-list rebuild detection kernel.

    Parameters
    ----------
    wp_dtype : type
        Warp scalar dtype (``wp.float16``, ``wp.float32``, or ``wp.float64``).
    batched : bool, optional
        Select the batched per-system kernel variant.

    Returns
    -------
    wp.Kernel
        The compiled and cached cell-list rebuild detection kernel.
    """
    require_supported_dtype(wp_dtype)
    return _make_cell_rebuild_kernel(wp_dtype, batched=bool(batched))
