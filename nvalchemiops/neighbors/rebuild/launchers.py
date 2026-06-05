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

"""Core Warp launchers for rebuild detection.

This module preserves the public rebuild-detection entry points while kernel
factories live in :mod:`nvalchemiops.neighbors.rebuild.kernels`.
"""

import warp as wp

from nvalchemiops.neighbors.neighbor_utils import (
    dtype_info,
    update_ref_positions,
    update_ref_positions_batch,
)
from nvalchemiops.neighbors.neighbor_utils import (
    empty_sentinel as _empty_sentinel,
)
from nvalchemiops.neighbors.rebuild.kernels import (
    get_cell_list_rebuild_kernel,
    get_neighbor_list_rebuild_kernel,
)

__all__ = [
    "check_cell_list_rebuild",
    "check_neighbor_list_rebuild",
    "check_batch_neighbor_list_rebuild",
    "check_batch_cell_list_rebuild",
    "get_cell_list_rebuild_kernel",
    "get_neighbor_list_rebuild_kernel",
]


def _validate_pbc_params(
    cell: wp.array | None,
    cell_inv: wp.array | None,
    pbc: wp.array | None,
) -> bool:
    """Validate the optional MIC argument group and return whether PBC is active."""
    pbc_params = (cell, cell_inv, pbc)
    if any(p is not None for p in pbc_params) and not all(
        p is not None for p in pbc_params
    ):
        raise ValueError(
            "cell, cell_inv, and pbc must all be provided together to enable MIC "
            "displacement checking. Received a partial set."
        )
    return cell is not None


def _matrix_sentinel(wp_dtype: type, device: str) -> wp.array:
    """Return a zero-size matrix sentinel for ``wp_dtype``/``device``."""
    _vec_dtype, mat_dtype = dtype_info(wp_dtype)
    return _empty_sentinel(1, mat_dtype, device)


def _launch_neighbor_list_rebuild(
    reference_positions: wp.array,
    current_positions: wp.array,
    batch_idx: wp.array | None,
    skin_distance_threshold: float,
    rebuild_flags: wp.array,
    wp_dtype: type,
    device: str,
    *,
    batched: bool,
    cell: wp.array | None,
    cell_inv: wp.array | None,
    pbc: wp.array | None,
) -> None:
    """Launch the shape-uniform neighbor-list rebuild kernel."""
    use_pbc = _validate_pbc_params(cell, cell_inv, pbc)
    if batched and batch_idx is None:
        raise ValueError("batch_idx is required for batched rebuild detection")

    if use_pbc:
        cell_input = cell
        cell_inv_input = cell_inv
        pbc_flags = _empty_sentinel(1, wp.bool, device) if batched else pbc
        pbc_flags_batch = pbc if batched else _empty_sentinel(2, wp.bool, device)
    else:
        cell_input = _matrix_sentinel(wp_dtype, device)
        cell_inv_input = _matrix_sentinel(wp_dtype, device)
        pbc_flags = _empty_sentinel(1, wp.bool, device)
        pbc_flags_batch = _empty_sentinel(2, wp.bool, device)

    wp.launch(
        kernel=get_neighbor_list_rebuild_kernel(wp_dtype, batched=batched, pbc=use_pbc),
        dim=reference_positions.shape[0],
        inputs=[
            reference_positions,
            current_positions,
            batch_idx if batched else _empty_sentinel(1, wp.int32, device),
            cell_input,
            cell_inv_input,
            pbc_flags,
            pbc_flags_batch,
            wp_dtype(skin_distance_threshold),
            rebuild_flags,
        ],
        device=device,
    )


def _launch_cell_list_rebuild(
    current_positions: wp.array,
    atom_to_cell_mapping: wp.array,
    batch_idx: wp.array | None,
    cells_per_dimension: wp.array,
    cell: wp.array,
    pbc: wp.array,
    rebuild_flags: wp.array,
    wp_dtype: type,
    device: str,
    *,
    batched: bool,
) -> None:
    """Launch the shape-uniform cell-list rebuild kernel."""
    if batched and batch_idx is None:
        raise ValueError(
            "batch_idx is required for batched cell-list rebuild detection"
        )

    wp.launch(
        kernel=get_cell_list_rebuild_kernel(wp_dtype, batched=batched),
        dim=current_positions.shape[0],
        inputs=[
            current_positions,
            cell,
            atom_to_cell_mapping,
            batch_idx if batched else _empty_sentinel(1, wp.int32, device),
            _empty_sentinel(1, wp.int32, device) if batched else cells_per_dimension,
            cells_per_dimension if batched else _empty_sentinel(1, wp.vec3i, device),
            _empty_sentinel(1, wp.bool, device) if batched else pbc,
            pbc if batched else _empty_sentinel(2, wp.bool, device),
            rebuild_flags,
        ],
        device=device,
    )


def check_cell_list_rebuild(
    current_positions: wp.array,
    atom_to_cell_mapping: wp.array,
    cells_per_dimension: wp.array,
    cell: wp.array,
    pbc: wp.array,
    rebuild_flag: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for detecting if cell list needs rebuilding.

    Checks if any atoms have moved between spatial cells since the cell list was built.

    Parameters
    ----------
    current_positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Current atomic coordinates in Cartesian space.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        Previously computed cell coordinates for each atom.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        Number of cells in x, y, z directions.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix for coordinate transformations.
    pbc : wp.array, shape (3,), dtype=wp.bool
        Periodic boundary condition flags.
    rebuild_flag : wp.array, shape (1,), dtype=wp.bool
        OUTPUT: Flag set to True if rebuild is needed.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - rebuild_flag must be pre-allocated and initialized to False by caller.

    See Also
    --------
    get_cell_list_rebuild_kernel : Factory-selected cell-list rebuild detection kernel.
    """
    _launch_cell_list_rebuild(
        current_positions,
        atom_to_cell_mapping,
        None,
        cells_per_dimension,
        cell,
        pbc,
        rebuild_flag,
        wp_dtype,
        device,
        batched=False,
    )


def check_neighbor_list_rebuild(
    reference_positions: wp.array,
    current_positions: wp.array,
    skin_distance_threshold: float,
    rebuild_flag: wp.array,
    wp_dtype: type,
    device: str,
    update_reference_positions: bool = False,
    cell: wp.array | None = None,
    cell_inv: wp.array | None = None,
    pbc: wp.array | None = None,
) -> None:
    """Core warp launcher for detecting if neighbor list needs rebuilding.

    Checks if any atoms have moved beyond the skin distance since the neighbor
    list was built.  When ``cell``, ``cell_inv`` and ``pbc`` are all provided
    the check uses minimum-image convention (MIC) so that atoms crossing
    periodic boundaries are not spuriously flagged.

    Parameters
    ----------
    reference_positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic positions when the neighbor list was last built.
    current_positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Current atomic positions to compare against reference.
    skin_distance_threshold : float
        Maximum allowed displacement before neighbor list becomes invalid.
    rebuild_flag : wp.array, shape (1,), dtype=wp.bool
        OUTPUT: Flag set to True if rebuild is needed.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    update_reference_positions : bool, optional
        If True, overwrite ``reference_positions`` with ``current_positions``
        for all atoms when a rebuild is detected. The update runs in a second
        kernel launch after the detection kernel, so every atom is guaranteed
        to be updated with no race conditions. Default False.
    cell : wp.array or None, optional
        Unit cell matrix, shape (1,), dtype=wp.mat33*.  Required together with
        ``cell_inv`` and ``pbc`` to enable MIC displacement.
    cell_inv : wp.array or None, optional
        Precomputed inverse of the cell matrix, same shape/dtype as ``cell``.
    pbc : wp.array or None, optional
        Periodic boundary condition flags, shape (3,), dtype=wp.bool.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - rebuild_flag must be pre-allocated and initialized to False by caller.

    Raises
    ------
    ValueError
        If only a subset of ``cell``, ``cell_inv``, and ``pbc`` are provided.
        All three must be supplied together to enable MIC displacement.

    See Also
    --------
    get_neighbor_list_rebuild_kernel : Factory-selected neighbor-list rebuild detection kernel.
    update_ref_positions : Standalone reference-position update launcher.
    """
    _launch_neighbor_list_rebuild(
        reference_positions,
        current_positions,
        None,
        skin_distance_threshold,
        rebuild_flag,
        wp_dtype,
        device,
        batched=False,
        cell=cell,
        cell_inv=cell_inv,
        pbc=pbc,
    )
    if update_reference_positions:
        update_ref_positions(
            current_positions, rebuild_flag, reference_positions, wp_dtype, device
        )


def check_batch_neighbor_list_rebuild(
    reference_positions: wp.array,
    current_positions: wp.array,
    batch_idx: wp.array,
    skin_distance_threshold: float,
    rebuild_flags: wp.array,
    wp_dtype: type,
    device: str,
    update_reference_positions: bool = False,
    cell: wp.array | None = None,
    cell_inv: wp.array | None = None,
    pbc: wp.array | None = None,
) -> None:
    """Core warp launcher for detecting per-system neighbor list rebuild needs.

    Checks if any atoms in each system have moved beyond the skin distance since
    the neighbor list was built. Sets per-system rebuild flags on GPU without
    requiring CPU synchronization.

    When ``cell``, ``cell_inv`` and ``pbc`` are all provided the check uses
    minimum-image convention (MIC) so that atoms crossing periodic boundaries
    are not spuriously flagged.

    Parameters
    ----------
    reference_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Atomic positions when each system's neighbor list was last built.
    current_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Current atomic positions to compare against reference.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    skin_distance_threshold : float
        Maximum allowed displacement before neighbor list becomes invalid.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        OUTPUT: Per-system flags set to True if rebuild is needed.
        Must be pre-allocated and initialized to False by caller.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    update_reference_positions : bool, optional
        If True, overwrite ``reference_positions`` with ``current_positions``
        for all atoms in rebuilt systems when a rebuild is detected. The update
        runs in a second kernel launch after the detection kernel, so every atom
        in each rebuilt system is guaranteed to be updated with no race
        conditions. Default False.
    cell : wp.array or None, optional
        Per-system cell matrices, shape (num_systems,), dtype=wp.mat33*.
        Required together with ``cell_inv`` and ``pbc`` to enable MIC.
    cell_inv : wp.array or None, optional
        Precomputed per-system inverse cell matrices, same shape/dtype as
        ``cell``.
    pbc : wp.array or None, optional
        Per-system PBC flags, shape (num_systems, 3), dtype=wp.bool (2D).

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - rebuild_flags must be pre-allocated and initialized to False by caller.
    - No CPU-GPU synchronization required; flags are written entirely on GPU.

    Raises
    ------
    ValueError
        If only a subset of ``cell``, ``cell_inv``, and ``pbc`` are provided.
        All three must be supplied together to enable MIC displacement.

    See Also
    --------
    get_neighbor_list_rebuild_kernel : Factory-selected batched neighbor-list rebuild detection kernel.
    update_ref_positions_batch : Standalone reference-position update launcher.
    """
    _launch_neighbor_list_rebuild(
        reference_positions,
        current_positions,
        batch_idx,
        skin_distance_threshold,
        rebuild_flags,
        wp_dtype,
        device,
        batched=True,
        cell=cell,
        cell_inv=cell_inv,
        pbc=pbc,
    )
    if update_reference_positions:
        update_ref_positions_batch(
            current_positions,
            rebuild_flags,
            batch_idx,
            reference_positions,
            wp_dtype,
            device,
        )


def check_batch_cell_list_rebuild(
    current_positions: wp.array,
    atom_to_cell_mapping: wp.array,
    batch_idx: wp.array,
    cells_per_dimension: wp.array,
    cell: wp.array,
    pbc: wp.array,
    rebuild_flags: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for detecting per-system cell list rebuild needs.

    Checks if any atoms in each system have moved between spatial cells since
    the cell list was built. Sets per-system rebuild flags on GPU without
    requiring CPU synchronization.

    Parameters
    ----------
    current_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Current Cartesian coordinates.
    atom_to_cell_mapping : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Previously computed cell coordinates for each atom.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    cells_per_dimension : wp.array, shape (num_systems,), dtype=wp.vec3i
        Number of cells in x, y, z directions for each system.
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Per-system unit cell matrices for coordinate transformations.
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool
        Per-system periodic boundary condition flags (2D array).
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        OUTPUT: Per-system flags set to True if rebuild is needed.
        Must be pre-allocated and initialized to False by caller.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - rebuild_flags must be pre-allocated and initialized to False by caller.
    - No CPU-GPU synchronization required; flags are written entirely on GPU.

    See Also
    --------
    get_cell_list_rebuild_kernel : Factory-selected batched cell-list rebuild detection kernel.
    """
    _launch_cell_list_rebuild(
        current_positions,
        atom_to_cell_mapping,
        batch_idx,
        cells_per_dimension,
        cell,
        pbc,
        rebuild_flags,
        wp_dtype,
        device,
        batched=True,
    )
