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

"""
Cell Utilities for NPT/NPH Simulations.

This module provides utilities for manipulating simulation cells (periodic boxes)
in molecular dynamics simulations with variable cell volume/shape (NPT, NPH ensembles).

The cell is represented as a (B, 3, 3) array of matrices where each matrix contains
lattice vectors as columns:
    cell[b] = [a, b, c]  (column vectors for system b)

Even single-system simulations use shape (1, 3, 3).

Fractional coordinates s relate to Cartesian coordinates r by:
    r = cell @ s
    s = cell_inv @ r

Key concepts:
- Cell volume: V = det(cell)
- Cell inverse: For coordinate transformations
- Strain tensor: Deformation from reference cell
- Position remapping: Maintain fractional coordinates when cell changes

All kernels are dtype-agnostic and support both float32 and float64 cell matrices.
Functions that require cell_inv accept it as a required parameter; callers
must pre-compute via ``compute_cell_inverse`` to avoid redundant inverse
computations in MD loops.
"""

from __future__ import annotations

from typing import Any

import warp as wp

__all__ = [
    # Cell properties
    "compute_cell_volume",
    "compute_cell_inverse",
    # Strain operations
    "compute_strain_tensor",
    "apply_strain_to_cell",
    # Position operations
    "scale_positions_with_cell",
    "wrap_positions_to_cell",
    "cartesian_to_fractional",
    "fractional_to_cartesian",
    # Non-mutating variants
    "scale_positions_with_cell_out",
    "wrap_positions_to_cell_out",
]


# ==============================================================================
# Cell Property Kernels
# ==============================================================================


@wp.kernel
def _compute_cell_volume_kernel(
    cells: wp.array(dtype=Any),
    volumes: wp.array(dtype=Any),
):
    """Compute cell volume V = det(cell) = a · (b × c).

    Launch Grid
    -----------
    dim = [num_systems]

    Parameters
    ----------
    cells : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell matrices. Shape (B,).
    volumes : wp.array(dtype=wp.float32 or wp.float64)
        Output volumes. Shape (B,).
    """
    sys_id = wp.tid()
    cell = cells[sys_id]

    # Cell columns are lattice vectors a, b, c
    a0 = cell[0, 0]
    a1 = cell[1, 0]
    a2 = cell[2, 0]
    b0 = cell[0, 1]
    b1 = cell[1, 1]
    b2 = cell[2, 1]
    c0 = cell[0, 2]
    c1 = cell[1, 2]
    c2 = cell[2, 2]

    # det = a · (b × c)
    det = a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0)

    volumes[sys_id] = wp.abs(det)


@wp.kernel
def _compute_cell_inverse_kernel(
    cells: wp.array(dtype=Any),
    cells_inv: wp.array(dtype=Any),
):
    """Compute cell inverse for coordinate transformations.

    Launch Grid
    -----------
    dim = [num_systems]

    Parameters
    ----------
    cells : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell matrices. Shape (B,).
    cells_inv : wp.array(dtype=wp.mat33f or wp.mat33d)
        Output cell inverses. Shape (B,).
    """
    sys_id = wp.tid()
    cell = cells[sys_id]

    # Cell elements
    a00 = cell[0, 0]
    a01 = cell[0, 1]
    a02 = cell[0, 2]
    a10 = cell[1, 0]
    a11 = cell[1, 1]
    a12 = cell[1, 2]
    a20 = cell[2, 0]
    a21 = cell[2, 1]
    a22 = cell[2, 2]

    # Determinant
    det = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )

    inv_det = type(a00)(1.0) / wp.max(det, type(a00)(1e-10))

    # Adjugate matrix / det
    inv00 = (a11 * a22 - a12 * a21) * inv_det
    inv01 = (a02 * a21 - a01 * a22) * inv_det
    inv02 = (a01 * a12 - a02 * a11) * inv_det
    inv10 = (a12 * a20 - a10 * a22) * inv_det
    inv11 = (a00 * a22 - a02 * a20) * inv_det
    inv12 = (a02 * a10 - a00 * a12) * inv_det
    inv20 = (a10 * a21 - a11 * a20) * inv_det
    inv21 = (a01 * a20 - a00 * a21) * inv_det
    inv22 = (a00 * a11 - a01 * a10) * inv_det

    cells_inv[sys_id] = type(cell)(
        inv00, inv01, inv02, inv10, inv11, inv12, inv20, inv21, inv22
    )


@wp.kernel
def _compute_strain_tensor_kernel(
    cells: wp.array(dtype=Any),
    cells_ref_inv: wp.array(dtype=Any),
    strains: wp.array(dtype=Any),
):
    """Compute strain tensor: ε = cell @ cell_ref_inv - I.

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    cell = cells[sys_id]
    cell_ref_inv = cells_ref_inv[sys_id]

    # Compute cell @ cell_ref_inv
    m = wp.mul(cell, cell_ref_inv)
    m -= wp.identity(3, dtype=cell.dtype)
    strains[sys_id] = m


@wp.kernel
def _apply_strain_to_cell_kernel(
    cells: wp.array(dtype=Any),
    strains: wp.array(dtype=Any),
    cells_out: wp.array(dtype=Any),
):
    """Apply strain: cell_new = (I + strain) @ cell.

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    cell = cells[sys_id]
    strain = strains[sys_id]

    cells_out[sys_id] = wp.mul(wp.identity(3, dtype=cell.dtype) + strain, cell)


# ==============================================================================
# Position Transformation Kernels
# ==============================================================================


@wp.kernel
def _scale_positions_single_kernel(
    positions: wp.array(dtype=Any),
    cell_old_inv: wp.array(dtype=Any),
    cell_new: wp.array(dtype=Any),
):
    """Scale positions for single system (no batch_idx).

    r_new = cell_new @ cell_old_inv @ r_old

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    r = positions[atom_idx]
    # Single system: always index 0
    coi = cell_old_inv[0]
    cn = cell_new[0]

    positions[atom_idx] = wp.mul(wp.mul(cn, coi), r)


@wp.kernel
def _scale_positions_kernel(
    positions: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cells_old_inv: wp.array(dtype=Any),
    cells_new: wp.array(dtype=Any),
):
    """Scale positions from old cell to new cell maintaining fractional coords.

    r_new = cell_new @ cell_old_inv @ r_old

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    r = positions[atom_idx]
    cell_old_inv = cells_old_inv[sys_id]
    cell_new = cells_new[sys_id]

    positions[atom_idx] = wp.mul(wp.mul(cell_new, cell_old_inv), r)


@wp.kernel
def _scale_positions_out_single_kernel(
    positions: wp.array(dtype=Any),
    cell_old_inv: wp.array(dtype=Any),
    cell_new: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
):
    """Scale positions to output array for single system.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    r = positions[atom_idx]
    coi = cell_old_inv[0]
    cn = cell_new[0]

    positions_out[atom_idx] = wp.mul(wp.mul(cn, coi), r)


@wp.kernel
def _scale_positions_out_kernel(
    positions: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cells_old_inv: wp.array(dtype=Any),
    cells_new: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
):
    """Scale positions to output array.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    r = positions[atom_idx]
    cell_old_inv = cells_old_inv[sys_id]
    cell_new = cells_new[sys_id]

    positions_out[atom_idx] = wp.mul(wp.mul(cell_new, cell_old_inv), r)


# ==============================================================================
# Wrapping Kernels
# ==============================================================================


@wp.kernel
def _wrap_positions_single_kernel(
    positions: wp.array(dtype=Any),
    cell_inv: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
):
    """Wrap positions for single system (no batch_idx).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    r = positions[atom_idx]
    ci = cell_inv[0]
    c = cell[0]

    # Convert to fractional: s = cell_inv @ r
    s = wp.mul(ci, r)

    # Wrap to [0, 1) using floor
    s_wrapped = type(s)(
        s[0] - wp.floor(s[0]), s[1] - wp.floor(s[1]), s[2] - wp.floor(s[2])
    )

    # Convert back to Cartesian: r_new = cell @ s_wrapped
    positions[atom_idx] = wp.mul(c, s_wrapped)


@wp.kernel
def _wrap_positions_kernel(
    positions: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cells_inv: wp.array(dtype=Any),
    cells: wp.array(dtype=Any),
):
    """Wrap positions into the primary cell [0, 1) in fractional coordinates.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    r = positions[atom_idx]
    cell_inv = cells_inv[sys_id]
    cell = cells[sys_id]

    # Convert to fractional: s = cell_inv @ r
    s = wp.mul(cell_inv, r)

    # Wrap to [0, 1) using floor
    s_wrapped = type(s)(
        s[0] - wp.floor(s[0]), s[1] - wp.floor(s[1]), s[2] - wp.floor(s[2])
    )

    # Convert back to Cartesian: r_new = cell @ s_wrapped
    positions[atom_idx] = wp.mul(cell, s_wrapped)


@wp.kernel
def _wrap_positions_out_single_kernel(
    positions: wp.array(dtype=Any),
    cell_inv: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
):
    """Wrap positions to output array for single system.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    r = positions[atom_idx]
    ci = cell_inv[0]
    c = cell[0]

    s = wp.mul(ci, r)
    s_wrapped = type(s)(
        s[0] - wp.floor(s[0]), s[1] - wp.floor(s[1]), s[2] - wp.floor(s[2])
    )
    positions_out[atom_idx] = wp.mul(c, s_wrapped)


@wp.kernel
def _wrap_positions_out_kernel(
    positions: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cells_inv: wp.array(dtype=Any),
    cells: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
):
    """Wrap positions to output array.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    r = positions[atom_idx]
    cell_inv = cells_inv[sys_id]
    cell = cells[sys_id]

    # Convert to fractional: s = cell_inv @ r
    s = wp.mul(cell_inv, r)

    # Wrap to [0, 1) using floor
    s_wrapped = type(s)(
        s[0] - wp.floor(s[0]), s[1] - wp.floor(s[1]), s[2] - wp.floor(s[2])
    )

    # Convert back to Cartesian: r_new = cell @ s_wrapped
    positions_out[atom_idx] = wp.mul(cell, s_wrapped)


# ==============================================================================
# Coordinate Transformation Kernels
# ==============================================================================


@wp.kernel
def _cartesian_to_fractional_single_kernel(
    positions: wp.array(dtype=Any),
    cell_inv: wp.array(dtype=Any),
    fractional: wp.array(dtype=Any),
):
    """Convert Cartesian to fractional for single system.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    r = positions[atom_idx]
    ci = cell_inv[0]

    fractional[atom_idx] = wp.mul(ci, r)


@wp.kernel
def _cartesian_to_fractional_kernel(
    positions: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cells_inv: wp.array(dtype=Any),
    fractional: wp.array(dtype=Any),
):
    """Convert Cartesian coordinates to fractional coordinates.

    s = cell_inv @ r

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    r = positions[atom_idx]
    cell_inv = cells_inv[sys_id]

    fractional[atom_idx] = wp.mul(cell_inv, r)


@wp.kernel
def _fractional_to_cartesian_single_kernel(
    fractional: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
):
    """Convert fractional to Cartesian for single system.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    s = fractional[atom_idx]
    c = cell[0]

    positions[atom_idx] = wp.mul(c, s)


@wp.kernel
def _fractional_to_cartesian_kernel(
    fractional: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cells: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
):
    """Convert fractional coordinates to Cartesian coordinates.

    r = cell @ s

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    s = fractional[atom_idx]
    cell = cells[sys_id]

    positions[atom_idx] = wp.mul(cell, s)


# ==============================================================================
# Kernel Overloads for Explicit Typing
# ==============================================================================

_T = [wp.float32, wp.float64]  # Scalar types
_V = [wp.vec3f, wp.vec3d]  # Vector types
_M = [wp.mat33f, wp.mat33d]  # Matrix types

# Cell property kernel overloads
_compute_cell_volume_kernel_overload = {}
_compute_cell_inverse_kernel_overload = {}
_compute_strain_tensor_kernel_overload = {}
_apply_strain_to_cell_kernel_overload = {}

# Position scaling kernel overloads
_scale_positions_single_kernel_overload = {}
_scale_positions_kernel_overload = {}
_scale_positions_out_single_kernel_overload = {}
_scale_positions_out_kernel_overload = {}

# Wrapping kernel overloads
_wrap_positions_single_kernel_overload = {}
_wrap_positions_kernel_overload = {}
_wrap_positions_out_single_kernel_overload = {}
_wrap_positions_out_kernel_overload = {}

# Coordinate conversion kernel overloads
_cartesian_to_fractional_single_kernel_overload = {}
_cartesian_to_fractional_kernel_overload = {}
_fractional_to_cartesian_single_kernel_overload = {}
_fractional_to_cartesian_kernel_overload = {}

for t, v, m in zip(_T, _V, _M):
    # Cell property kernels
    _compute_cell_volume_kernel_overload[m] = wp.overload(
        _compute_cell_volume_kernel,
        [wp.array(dtype=m), wp.array(dtype=t)],
    )
    _compute_cell_inverse_kernel_overload[m] = wp.overload(
        _compute_cell_inverse_kernel,
        [wp.array(dtype=m), wp.array(dtype=m)],
    )
    _compute_strain_tensor_kernel_overload[m] = wp.overload(
        _compute_strain_tensor_kernel,
        [wp.array(dtype=m), wp.array(dtype=m), wp.array(dtype=m)],
    )
    _apply_strain_to_cell_kernel_overload[m] = wp.overload(
        _apply_strain_to_cell_kernel,
        [wp.array(dtype=m), wp.array(dtype=m), wp.array(dtype=m)],
    )

    # Position scaling kernels
    _scale_positions_single_kernel_overload[v] = wp.overload(
        _scale_positions_single_kernel,
        [wp.array(dtype=v), wp.array(dtype=m), wp.array(dtype=m)],
    )
    _scale_positions_kernel_overload[v] = wp.overload(
        _scale_positions_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=m),
            wp.array(dtype=m),
        ],
    )
    _scale_positions_out_single_kernel_overload[v] = wp.overload(
        _scale_positions_out_single_kernel,
        [wp.array(dtype=v), wp.array(dtype=m), wp.array(dtype=m), wp.array(dtype=v)],
    )
    _scale_positions_out_kernel_overload[v] = wp.overload(
        _scale_positions_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=m),
            wp.array(dtype=m),
            wp.array(dtype=v),
        ],
    )

    # Wrapping kernels
    _wrap_positions_single_kernel_overload[v] = wp.overload(
        _wrap_positions_single_kernel,
        [wp.array(dtype=v), wp.array(dtype=m), wp.array(dtype=m)],
    )
    _wrap_positions_kernel_overload[v] = wp.overload(
        _wrap_positions_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=m),
            wp.array(dtype=m),
        ],
    )
    _wrap_positions_out_single_kernel_overload[v] = wp.overload(
        _wrap_positions_out_single_kernel,
        [wp.array(dtype=v), wp.array(dtype=m), wp.array(dtype=m), wp.array(dtype=v)],
    )
    _wrap_positions_out_kernel_overload[v] = wp.overload(
        _wrap_positions_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=m),
            wp.array(dtype=m),
            wp.array(dtype=v),
        ],
    )

    # Coordinate conversion kernels
    _cartesian_to_fractional_single_kernel_overload[v] = wp.overload(
        _cartesian_to_fractional_single_kernel,
        [wp.array(dtype=v), wp.array(dtype=m), wp.array(dtype=v)],
    )
    _cartesian_to_fractional_kernel_overload[v] = wp.overload(
        _cartesian_to_fractional_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=m),
            wp.array(dtype=v),
        ],
    )
    _fractional_to_cartesian_single_kernel_overload[v] = wp.overload(
        _fractional_to_cartesian_single_kernel,
        [wp.array(dtype=v), wp.array(dtype=m), wp.array(dtype=v)],
    )
    _fractional_to_cartesian_kernel_overload[v] = wp.overload(
        _fractional_to_cartesian_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=m),
            wp.array(dtype=v),
        ],
    )


# ==============================================================================
# Functional Interfaces
# ==============================================================================


def compute_cell_volume(
    cells: wp.array,
    volumes: wp.array,
    device: str = None,
) -> wp.array:
    r"""
    Compute cell volume :math:`V = |\det(cell)|`.

    Parameters
    ----------
    cells : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell matrices. Shape (B,) where B is number of systems.
        Even single systems use shape (1,).
    volumes : wp.array
        Output array for volumes. Shape (B,). Caller must pre-allocate.
    device : str, optional
        Warp device. If None, inferred from cells.

    Returns
    -------
    wp.array
        Cell volumes. Shape (B,).
    """
    if device is None:
        device = cells.device

    num_systems = cells.shape[0]

    mat_dtype = cells.dtype
    wp.launch(
        _compute_cell_volume_kernel_overload[mat_dtype],
        dim=num_systems,
        inputs=[cells, volumes],
        device=device,
    )

    return volumes


def compute_cell_inverse(
    cells: wp.array,
    cells_inv: wp.array,
    device: str = None,
) -> wp.array:
    """
    Compute cell inverse matrices for coordinate transformations.

    Parameters
    ----------
    cells : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell matrices. Shape (B,).
    cells_inv : wp.array
        Output array for inverses. Shape (B,). Caller must pre-allocate.
    device : str, optional
        Warp device. If None, inferred from cells.

    Returns
    -------
    wp.array
        Cell inverse matrices. Shape (B,).
    """
    if device is None:
        device = cells.device

    num_systems = cells.shape[0]

    mat_dtype = cells.dtype
    wp.launch(
        _compute_cell_inverse_kernel_overload[mat_dtype],
        dim=num_systems,
        inputs=[cells, cells_inv],
        device=device,
    )

    return cells_inv


def compute_strain_tensor(
    cells: wp.array,
    cells_ref_inv: wp.array,
    strains: wp.array,
    device: str = None,
) -> wp.array:
    """
    Compute strain tensor from current and reference cells.

    The strain tensor ε is defined by: cell = (I + ε) @ cell_ref
    So: ε = cell @ cell_ref_inv - I

    Parameters
    ----------
    cells : wp.array(dtype=wp.mat33f or wp.mat33d)
        Current cell matrices. Shape (B,).
    cells_ref_inv : wp.array
        Pre-computed inverse of reference cells. Shape (B,).
        Caller must pre-compute via ``compute_cell_inverse``.
    strains : wp.array
        Output strain tensors. Shape (B,). Caller must pre-allocate.
    device : str, optional
        Warp device. If None, inferred from cells.

    Returns
    -------
    wp.array
        Strain tensors. Shape (B,).
    """
    if device is None:
        device = cells.device

    num_systems = cells.shape[0]

    mat_dtype = cells.dtype
    wp.launch(
        _compute_strain_tensor_kernel_overload[mat_dtype],
        dim=num_systems,
        inputs=[cells, cells_ref_inv, strains],
        device=device,
    )

    return strains


def apply_strain_to_cell(
    cells: wp.array,
    strains: wp.array,
    cells_out: wp.array,
    device: str = None,
) -> wp.array:
    """
    Apply strain tensor to cell: cell_new = (I + strain) @ cell.

    Parameters
    ----------
    cells : wp.array(dtype=wp.mat33f or wp.mat33d)
        Current cell matrices. Shape (B,).
    strains : wp.array
        Strain tensors to apply. Shape (B,).
    cells_out : wp.array
        Output cell matrices. Shape (B,). Caller must pre-allocate.
    device : str, optional
        Warp device. If None, inferred from cells.

    Returns
    -------
    wp.array
        Updated cell matrices. Shape (B,).
    """
    if device is None:
        device = cells.device

    num_systems = cells.shape[0]

    mat_dtype = cells.dtype
    wp.launch(
        _apply_strain_to_cell_kernel_overload[mat_dtype],
        dim=num_systems,
        inputs=[cells, strains, cells_out],
        device=device,
    )

    return cells_out


def scale_positions_with_cell(
    positions: wp.array,
    cells_new: wp.array,
    cells_old_inv: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> None:
    """
    Scale positions when cell changes, maintaining fractional coordinates (in-place).

    r_new = cell_new @ cell_old_inv @ r_old

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,). MODIFIED in-place.
    cells_new : wp.array
        New cell matrices. Shape (B,).
    cells_old_inv : wp.array
        Pre-computed inverse of old cell matrices. Shape (B,).
        Caller must pre-compute via ``compute_cell_inverse``.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Shape (N,). If None, assumes single system.
    device : str, optional
        Warp device. If None, inferred from positions.
    """
    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]

    vec_dtype = positions.dtype
    if batch_idx is None:
        # Single-system kernel
        wp.launch(
            _scale_positions_single_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, cells_old_inv, cells_new],
            device=device,
        )
    else:
        # Batched kernel
        wp.launch(
            _scale_positions_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, batch_idx, cells_old_inv, cells_new],
            device=device,
        )


def scale_positions_with_cell_out(
    positions: wp.array,
    cells_new: wp.array,
    cells_old_inv: wp.array,
    positions_out: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Scale positions when cell changes (non-mutating).

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,).
    cells_new : wp.array
        New cell matrices. Shape (B,).
    cells_old_inv : wp.array
        Pre-computed inverse of old cell matrices. Shape (B,).
        Caller must pre-compute via ``compute_cell_inverse``.
    positions_out : wp.array
        Output positions. Shape (N,). Caller must pre-allocate.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Shape (N,). If None, assumes single system.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Scaled positions.
    """
    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]

    vec_dtype = positions.dtype
    if batch_idx is None:
        wp.launch(
            _scale_positions_out_single_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, cells_old_inv, cells_new, positions_out],
            device=device,
        )
    else:
        wp.launch(
            _scale_positions_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, batch_idx, cells_old_inv, cells_new, positions_out],
            device=device,
        )

    return positions_out


def wrap_positions_to_cell(
    positions: wp.array,
    cells: wp.array,
    cells_inv: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> None:
    """
    Wrap positions into primary cell [0, 1) in fractional coordinates (in-place).

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,). MODIFIED in-place.
    cells : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell matrices. Shape (B,).
    cells_inv : wp.array
        Pre-computed inverse of cells. Shape (B,).
        Caller must pre-compute via ``compute_cell_inverse``.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Shape (N,). If None, assumes single system.
    device : str, optional
        Warp device.
    """
    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]

    vec_dtype = positions.dtype
    if batch_idx is None:
        wp.launch(
            _wrap_positions_single_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, cells_inv, cells],
            device=device,
        )
    else:
        wp.launch(
            _wrap_positions_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, batch_idx, cells_inv, cells],
            device=device,
        )


def wrap_positions_to_cell_out(
    positions: wp.array,
    cells: wp.array,
    cells_inv: wp.array,
    positions_out: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Wrap positions into primary cell (non-mutating).

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,).
    cells : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell matrices. Shape (B,).
    cells_inv : wp.array
        Pre-computed inverse of cells. Shape (B,).
        Caller must pre-compute via ``compute_cell_inverse``.
    positions_out : wp.array
        Output positions. Shape (N,). Caller must pre-allocate.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Shape (N,). If None, assumes single system.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Wrapped positions.
    """
    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]

    vec_dtype = positions.dtype
    if batch_idx is None:
        wp.launch(
            _wrap_positions_out_single_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, cells_inv, cells, positions_out],
            device=device,
        )
    else:
        wp.launch(
            _wrap_positions_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, batch_idx, cells_inv, cells, positions_out],
            device=device,
        )

    return positions_out


def cartesian_to_fractional(
    positions: wp.array,
    cells_inv: wp.array,
    fractional: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Convert Cartesian coordinates to fractional coordinates.

    s = cell_inv @ r

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Cartesian positions. Shape (N,).
    cells_inv : wp.array
        Pre-computed inverse of cells. Shape (B,).
        Caller must pre-compute via ``compute_cell_inverse``.
    fractional : wp.array
        Output fractional coordinates. Shape (N,). Caller must pre-allocate.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Shape (N,). If None, assumes single system.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Fractional coordinates.
    """
    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]

    vec_dtype = positions.dtype
    if batch_idx is None:
        wp.launch(
            _cartesian_to_fractional_single_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, cells_inv, fractional],
            device=device,
        )
    else:
        wp.launch(
            _cartesian_to_fractional_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, batch_idx, cells_inv, fractional],
            device=device,
        )

    return fractional


def fractional_to_cartesian(
    fractional: wp.array,
    cells: wp.array,
    positions: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Convert fractional coordinates to Cartesian coordinates.

    r = cell @ s

    Parameters
    ----------
    fractional : wp.array(dtype=wp.vec3f or wp.vec3d)
        Fractional coordinates. Shape (N,).
    cells : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell matrices. Shape (B,).
    positions : wp.array
        Output Cartesian positions. Shape (N,). Caller must pre-allocate.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Shape (N,). If None, assumes single system.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Cartesian positions.
    """
    if device is None:
        device = fractional.device

    num_atoms = fractional.shape[0]

    vec_dtype = fractional.dtype
    if batch_idx is None:
        wp.launch(
            _fractional_to_cartesian_single_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[fractional, cells, positions],
            device=device,
        )
    else:
        wp.launch(
            _fractional_to_cartesian_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[fractional, batch_idx, cells, positions],
            device=device,
        )

    return positions
