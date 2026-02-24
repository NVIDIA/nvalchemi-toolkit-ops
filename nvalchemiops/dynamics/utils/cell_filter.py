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
Cell Filter Utilities for Variable-Cell Optimization.

This module provides utilities for combining atomic and cell degrees of freedom
into extended arrays, enabling standard optimizers (FIRE, BFGS, etc.) to perform
variable-cell optimization without modification.

The approach follows the "filter" pattern:
- Atomic positions (3N DOFs) + cell parameters (6 DOFs) → extended positions (3N + 6)
- Atomic forces + stress tensor → extended forces (3N + 6)
- Standard optimizer operates on extended arrays
- Results are unpacked back to atomic positions and cell

Key features:
- Cell alignment to upper-triangular form for stability
- 6-DOF cell representation (upper-triangular: a, b*cos(γ), b*sin(γ), c1, c2, c3)
- Stress-to-cell-force conversion with proper volume scaling
- batch_idx/atom_ptr extension for batched systems

Usage workflow:
1. align_cell() - One-time preprocessing to put cell in standard form
2. extend_batch_idx() or extend_atom_ptr() - Update batching arrays for extended DOFs
3. pack_*() - Combine atomic + cell DOFs into extended arrays
4. Run optimizer step on extended arrays
5. unpack_*() - Extract atomic positions and cell from extended arrays
6. Compute forces/stress with your calculator
7. pack_forces_with_cell() - Combine forces and stress for next step
"""

from __future__ import annotations

from typing import Any

import warp as wp

from nvalchemiops.batch_utils import atom_ptr_to_batch_idx

__all__ = [
    # Cell alignment
    "align_cell",
    # Batch index extension
    "extend_batch_idx",
    "extend_atom_ptr",
    # Pack utilities
    "pack_positions_with_cell",
    "pack_velocities_with_cell",
    "pack_forces_with_cell",
    "pack_masses_with_cell",
    # Unpack utilities
    "unpack_positions_with_cell",
    "unpack_velocities_with_cell",
    # Stress conversion
    "stress_to_cell_force",
]


# ==============================================================================
# Cell Alignment Kernel
# ==============================================================================


@wp.kernel
def _align_cell_kernel(
    cell: wp.array(dtype=Any),
    transform: wp.array(dtype=Any),
):
    r"""Align cell to upper-triangular (right-handed) form.

    Transforms the cell matrix to the standard upper-triangular form:

    .. math::

        \mathbf{H} = \begin{pmatrix}
            a & 0 & 0 \\
            b\cos\gamma & b\sin\gamma & 0 \\
            c_1 & c_2 & c_3
        \end{pmatrix}

    where a, b, c are lattice vector lengths and γ is the angle between a and b.

    This representation:
    - Reduces rotational ambiguity (improves optimization stability)
    - Has 6 independent parameters instead of 9
    - Is the standard form expected by many MD codes

    The transformation matrix is computed such that:
        new_positions = old_positions @ transform

    Parameters
    ----------
    cell : wp.array, shape (B,), dtype=wp.mat33*
        Cell matrices (in-place, will be overwritten with aligned cells).
    transform : wp.array, shape (B,), dtype=wp.mat33*
        Output transformation matrices for position update.
        Should be initialized to identity matrices.

    Launch Grid
    -----------
    dim = [num_systems]

    Notes
    -----
    - Adapted from alchemistudio2 implementation.
    - Handles negative volume cells by flipping sign.
    - After this kernel, positions should be updated: pos = pos @ transform
    """
    tid = wp.tid()

    if tid >= cell.shape[0]:
        return

    _cell = cell[tid]
    vol = wp.determinant(_cell)

    # Handle zero volume (degenerate cell)
    if vol == type(_cell[0, 0])(0.0):
        return

    # Ensure right-handed cell
    if vol < type(_cell[0, 0])(0.0):
        _cell = type(_cell[0, 0])(-1.0) * _cell

    _one = type(_cell[0, 0])(1.0)
    _zero = type(_cell[0, 0])(0.0)

    # Compute lattice parameters
    a = wp.length(_cell[0])
    b = wp.length(_cell[1])
    c = wp.length(_cell[2])

    # Compute angles (cosines)
    cos_alpha = wp.dot(_cell[1], _cell[2]) / (b * c)  # angle between b and c
    cos_beta = wp.dot(_cell[0], _cell[2]) / (a * c)  # angle between a and c
    cos_gamma = wp.dot(_cell[0], _cell[1]) / (a * b)  # angle between a and b

    sin_gamma = wp.sqrt(wp.max(_zero, _one - cos_gamma * cos_gamma))

    # Compute c vector components in aligned frame
    c1 = c * cos_beta
    c2 = (c * (cos_alpha - cos_beta * cos_gamma)) / sin_gamma
    c3 = wp.sqrt(wp.max(_zero, c * c - c1 * c1 - c2 * c2))

    # Construct aligned cell (upper triangular)
    cell_r = type(_cell)(
        a,
        _zero,
        _zero,
        b * cos_gamma,
        b * sin_gamma,
        _zero,
        c1,
        c2,
        c3,
    )

    # Compute transformation matrix: r = cell_r_inv @ original_cell
    cell_r_inv = wp.inverse(cell_r)
    r = cell_r_inv * _cell

    # Store results
    cell[tid] = cell_r
    transform[tid] = r


@wp.kernel
def _apply_transform_single_kernel(
    positions: wp.array(dtype=Any),
    transform: wp.array(dtype=Any),
):
    """Apply transformation matrix to positions for single-system cell alignment.

    Computes: positions[i] = transform @ positions[i]

    This is called after _align_cell_kernel to rotate atomic positions so they
    maintain their fractional coordinates in the new aligned cell frame.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=vec3f or vec3d
        Atomic positions. Modified in-place.
    transform : wp.array, shape (1,), dtype=mat33f or mat33d
        Transformation matrix from _align_cell_kernel.

    Launch Grid
    -----------
    dim = num_atoms
    """
    idx = wp.tid()
    r = positions[idx]
    T = transform[0]
    positions[idx] = wp.mul(T, r)


@wp.kernel
def _apply_transform_kernel(
    positions: wp.array(dtype=Any),
    transform: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
):
    """Apply transformation matrices to positions for batched cell alignment.

    Computes: positions[i] = transform[sys] @ positions[i]
    where sys = batch_idx[i].

    This is called after _align_cell_kernel to rotate atomic positions so they
    maintain their fractional coordinates in their respective aligned cell frames.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=vec3f or vec3d
        Concatenated atomic positions. Modified in-place.
    transform : wp.array, shape (num_systems,), dtype=mat33f or mat33d
        Per-system transformation matrices from _align_cell_kernel.
    batch_idx : wp.array, shape (total_atoms,), dtype=int32
        System index for each atom.

    Launch Grid
    -----------
    dim = total_atoms
    """
    idx = wp.tid()
    sys = batch_idx[idx]
    r = positions[idx]
    T = transform[sys]
    positions[idx] = wp.mul(T, r)


# ==============================================================================
# Pack/Unpack Kernels for Extended Arrays
# ==============================================================================


@wp.kernel
def _pack_positions_kernel(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    extended: wp.array(dtype=Any),
    num_atoms: wp.int32,
):
    """Pack atomic positions and cell into extended position array (single system).

    Combines N atomic positions with 6 cell parameters (stored as 2 vec3s) into
    a single extended array of shape (N + 2,). The cell is assumed to be in
    upper-triangular form from align_cell().

    Cell packing format:
        extended[N]   = [H[0,0], H[1,0], H[2,0]] = [a, b*cos(γ), c1]
        extended[N+1] = [H[1,1], H[2,1], H[2,2]] = [b*sin(γ), c2, c3]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=vec3f or vec3d
        Atomic positions.
    cell : wp.array, shape (1,), dtype=mat33f or mat33d
        Cell matrix (should be upper-triangular from align_cell).
    extended : wp.array, shape (N + 2,), dtype=vec3f or vec3d
        OUTPUT: Extended position array. Modified in-place.
    num_atoms : wp.int32
        Number of atoms (N).

    Launch Grid
    -----------
    dim = num_atoms + 2
    """
    idx = wp.tid()

    if idx < num_atoms:
        # Copy atomic positions
        extended[idx] = positions[idx]
    elif idx == num_atoms:
        # First cell vec3: [a, b*cos(γ), c1] = [H[0,0], H[1,0], H[2,0]]
        H = cell[0]
        extended[idx] = type(positions[0])(H[0, 0], H[1, 0], H[2, 0])
    elif idx == num_atoms + 1:
        # Second cell vec3: [b*sin(γ), c2, c3] = [H[1,1], H[2,1], H[2,2]]
        H = cell[0]
        extended[idx] = type(positions[0])(H[1, 1], H[2, 1], H[2, 2])
    else:
        return


@wp.kernel
def _unpack_positions_kernel(
    extended: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    num_atoms: wp.int32,
):
    """Unpack extended position array to atomic positions and cell (single system).

    Extracts N atomic positions and reconstructs the upper-triangular cell matrix
    from the extended array. This is the inverse of _pack_positions_kernel.

    Parameters
    ----------
    extended : wp.array, shape (N + 2,), dtype=vec3f or vec3d
        Extended position array.
    positions : wp.array, shape (N,), dtype=vec3f or vec3d
        OUTPUT: Atomic positions. Modified in-place.
    cell : wp.array, shape (1,), dtype=mat33f or mat33d
        OUTPUT: Reconstructed upper-triangular cell matrix. Modified in-place.
    num_atoms : wp.int32
        Number of atoms (N).

    Launch Grid
    -----------
    dim = num_atoms + 2
    """
    idx = wp.tid()

    if idx < num_atoms:
        # Copy atomic positions
        positions[idx] = extended[idx]
    elif idx == num_atoms:
        # Reconstruct cell from packed format
        # Need both vec3s, so thread num_atoms does the full reconstruction
        v1 = extended[num_atoms]  # [a, b*cos(γ), c1]
        v2 = extended[num_atoms + 1]  # [b*sin(γ), c2, c3]

        # Upper triangular cell:
        # [a,       0,    0   ]
        # [b*cos(γ), b*sin(γ), 0   ]
        # [c1,      c2,   c3  ]
        _zero = type(v1[0])(0.0)
        cell[0] = type(cell[0])(
            v1[0],
            _zero,
            _zero,  # Row 0
            v1[1],
            v2[0],
            _zero,  # Row 1
            v1[2],
            v2[1],
            v2[2],  # Row 2
        )
    else:
        return


@wp.kernel
def _pack_forces_kernel(
    forces: wp.array(dtype=Any),
    cell_force: wp.array(dtype=Any),
    extended: wp.array(dtype=Any),
    num_atoms: wp.int32,
):
    """Pack atomic forces and cell force into extended force array (single system).

    Combines N atomic forces with 6 cell force components (stored as 2 vec3s)
    into a single extended array. Cell force is typically computed from
    stress_to_cell_force().

    Cell force packing format (same as positions):
        extended[N]   = [Fc[0,0], Fc[1,0], Fc[2,0]]
        extended[N+1] = [Fc[1,1], Fc[2,1], Fc[2,2]]

    Parameters
    ----------
    forces : wp.array, shape (N,), dtype=vec3f or vec3d
        Atomic forces.
    cell_force : wp.array, shape (1,), dtype=mat33f or mat33d
        Cell force matrix (from stress_to_cell_force).
    extended : wp.array, shape (N + 2,), dtype=vec3f or vec3d
        OUTPUT: Extended force array. Modified in-place.
    num_atoms : wp.int32
        Number of atoms (N).

    Launch Grid
    -----------
    dim = num_atoms + 2
    """
    idx = wp.tid()

    if idx < num_atoms:
        extended[idx] = forces[idx]
    elif idx == num_atoms:
        # First cell force vec3
        Fc = cell_force[0]
        extended[idx] = type(forces[0])(Fc[0, 0], Fc[1, 0], Fc[2, 0])
    elif idx == num_atoms + 1:
        # Second cell force vec3
        Fc = cell_force[0]
        extended[idx] = type(forces[0])(Fc[1, 1], Fc[2, 1], Fc[2, 2])
    else:
        return


@wp.kernel
def _pack_masses_kernel(
    masses: wp.array(dtype=Any),
    cell_mass: wp.array(dtype=Any),
    extended: wp.array(dtype=Any),
    num_atoms: wp.int32,
):
    """Pack atomic masses and cell mass into extended mass array (single system).

    Combines N atomic masses with 2 cell mass entries (for the 6 cell DOFs
    represented as 2 vec3s). The cell mass controls the relative response
    speed of cell parameters during optimization.

    Parameters
    ----------
    masses : wp.array, shape (N,), dtype=float32 or float64
        Atomic masses.
    cell_mass : wp.array, shape (1,), dtype=float32 or float64
        Mass for cell DOFs (scalar, same value used for both cell vec3 entries).
    extended : wp.array, shape (N + 2,), dtype=float32 or float64
        OUTPUT: Extended mass array. Modified in-place.
    num_atoms : wp.int32
        Number of atoms (N).

    Launch Grid
    -----------
    dim = num_atoms + 2
    """
    idx = wp.tid()

    if idx < num_atoms:
        extended[idx] = masses[idx]
    else:
        # Cell DOFs get the cell mass
        extended[idx] = cell_mass[0]


# ==============================================================================
# Batched Pack/Unpack Kernels (for use with atom_ptr + batch_idx)
# ==============================================================================
# Two-kernel pattern per pack/unpack operation:
#   - Atom kernel (dim=N): each thread copies one atom position/force/velocity
#   - Cell kernel (dim=M): each thread writes/reads 2 cell DOFs per system


@wp.kernel(enable_backward=False)
def _pack_atoms_batched_kernel(
    src: wp.array(dtype=Any),
    extended: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    atom_ptr: wp.array(dtype=wp.int32),
    ext_atom_ptr: wp.array(dtype=wp.int32),
):
    """Copy one atom from src to its interleaved position in extended array.

    Launch Grid: dim = N (total atoms).
    """
    i = wp.tid()
    s = batch_idx[i]
    local_idx = i - atom_ptr[s]
    extended[ext_atom_ptr[s] + local_idx] = src[i]


@wp.kernel(enable_backward=False)
def _pack_cell_dofs_kernel(
    cells: wp.array(dtype=Any),
    extended: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    ext_atom_ptr: wp.array(dtype=wp.int32),
):
    """Write 2 cell DOF vec3s for one system into extended array.

    Launch Grid: dim = M (num_systems).
    """
    sys = wp.tid()
    n_atoms_sys = atom_ptr[sys + 1] - atom_ptr[sys]
    ext_start = ext_atom_ptr[sys]
    H = cells[sys]
    extended[ext_start + n_atoms_sys] = type(extended[0])(H[0, 0], H[1, 0], H[2, 0])
    extended[ext_start + n_atoms_sys + 1] = type(extended[0])(H[1, 1], H[2, 1], H[2, 2])


@wp.kernel(enable_backward=False)
def _pack_cell_force_dofs_kernel(
    cell_forces: wp.array(dtype=Any),
    extended: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    ext_atom_ptr: wp.array(dtype=wp.int32),
):
    """Write 2 cell force DOF vec3s for one system into extended array.

    Launch Grid: dim = M (num_systems).
    """
    sys = wp.tid()
    n_atoms_sys = atom_ptr[sys + 1] - atom_ptr[sys]
    ext_start = ext_atom_ptr[sys]
    Fc = cell_forces[sys]
    extended[ext_start + n_atoms_sys] = type(extended[0])(Fc[0, 0], Fc[1, 0], Fc[2, 0])
    extended[ext_start + n_atoms_sys + 1] = type(extended[0])(
        Fc[1, 1], Fc[2, 1], Fc[2, 2]
    )


@wp.kernel(enable_backward=False)
def _unpack_atoms_batched_kernel(
    extended: wp.array(dtype=Any),
    dst: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    atom_ptr: wp.array(dtype=wp.int32),
    ext_atom_ptr: wp.array(dtype=wp.int32),
):
    """Copy one atom from its interleaved position in extended array to dst.

    Launch Grid: dim = N (total atoms).
    """
    i = wp.tid()
    s = batch_idx[i]
    local_idx = i - atom_ptr[s]
    dst[i] = extended[ext_atom_ptr[s] + local_idx]


@wp.kernel(enable_backward=False)
def _unpack_cell_dofs_kernel(
    extended: wp.array(dtype=Any),
    cells: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    ext_atom_ptr: wp.array(dtype=wp.int32),
):
    """Read 2 cell DOF vec3s for one system from extended array and reconstruct cell.

    Launch Grid: dim = M (num_systems).
    """
    sys = wp.tid()
    n_atoms_sys = atom_ptr[sys + 1] - atom_ptr[sys]
    ext_start = ext_atom_ptr[sys]
    v1 = extended[ext_start + n_atoms_sys]
    v2 = extended[ext_start + n_atoms_sys + 1]

    _zero = type(v1[0])(0.0)
    cells[sys] = type(cells[0])(
        v1[0],
        _zero,
        _zero,
        v1[1],
        v2[0],
        _zero,
        v1[2],
        v2[1],
        v2[2],
    )


@wp.kernel
def _pack_masses_batched_kernel(
    masses: wp.array(dtype=Any),
    cell_masses: wp.array(dtype=Any),
    extended: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    ext_atom_ptr: wp.array(dtype=wp.int32),
):
    """Pack atomic masses and cell masses into extended array for batched systems.

    Each thread handles one complete system, copying its atomic masses and
    appending the cell mass entries. The cell mass controls the relative
    response speed of cell parameters during optimization.

    Parameters
    ----------
    masses : wp.array, shape (total_atoms,), dtype=float32 or float64
        Concatenated atomic masses for all systems.
    cell_masses : wp.array, shape (num_systems,), dtype=float32 or float64
        Cell mass for each system.
    extended : wp.array, shape (total_atoms + 2*num_systems,), dtype=float32 or float64
        OUTPUT: Extended mass array. Modified in-place.
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        CSR-style pointer for original masses.
    ext_atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        CSR-style pointer for extended array (from extend_atom_ptr).

    Launch Grid
    -----------
    dim = num_systems
    """
    sys = wp.tid()

    orig_start = atom_ptr[sys]
    n_atoms_sys = atom_ptr[sys + 1] - orig_start
    ext_start = ext_atom_ptr[sys]

    # Copy atomic masses (serial within thread)
    for i in range(n_atoms_sys):
        extended[ext_start + i] = masses[orig_start + i]

    # Cell DOFs get the cell mass for this system
    extended[ext_start + n_atoms_sys] = cell_masses[sys]
    extended[ext_start + n_atoms_sys + 1] = cell_masses[sys]


# ==============================================================================
# Batch Index Extension Kernels
# ==============================================================================


@wp.kernel
def _extend_batch_idx_kernel(
    batch_idx: wp.array(dtype=wp.int32),
    extended_batch_idx: wp.array(dtype=wp.int32),
    num_atoms: wp.int32,
    num_systems: wp.int32,
):
    """Extend batch_idx to include cell DOFs for variable-cell optimization.

    Atomic indices keep their original system assignment. Cell DOFs are appended
    after all atoms, with 2 DOFs per system assigned to their respective systems.

    Extended layout:
        [atom_0_sys, atom_1_sys, ..., atom_N-1_sys,   <- original atoms
         sys_0, sys_0,                                 <- system 0 cell DOFs
         sys_1, sys_1,                                 <- system 1 cell DOFs
         ...]

    Parameters
    ----------
    batch_idx : wp.array, shape (num_atoms,), dtype=int32
        Original system index for each atom.
    extended_batch_idx : wp.array, shape (num_atoms + 2*num_systems,), dtype=int32
        OUTPUT: Extended batch index including cell DOFs. Modified in-place.
    num_atoms : wp.int32
        Total number of atoms across all systems (N).
    num_systems : wp.int32
        Number of systems (B).

    Launch Grid
    -----------
    dim = num_atoms + 2 * num_systems
    """
    idx = wp.tid()

    if idx < num_atoms:
        # Atomic positions keep their original batch_idx
        extended_batch_idx[idx] = batch_idx[idx]
    else:
        # Cell DOFs: idx = num_atoms + 2*sys + offset (offset = 0 or 1)
        cell_idx = idx - num_atoms
        sys = cell_idx / 2
        extended_batch_idx[idx] = sys


@wp.kernel
def _extend_atom_ptr_kernel(
    atom_ptr: wp.array(dtype=wp.int32),
    extended_atom_ptr: wp.array(dtype=wp.int32),
):
    """Extend atom_ptr to include cell DOFs for variable-cell optimization.

    Each system's range is extended by 2 entries (for the 6 cell DOFs stored
    as 2 vec3s). The offset increases by 2 for each system.

    Transformation:
        extended_atom_ptr[sys] = atom_ptr[sys] + 2 * sys

    Example:
        atom_ptr     = [0, 50, 100]    # 2 systems with 50 atoms each
        ext_atom_ptr = [0, 52, 104]    # 50+2=52, 100+4=104

    Parameters
    ----------
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        Original CSR-style atom pointers.
    extended_atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        OUTPUT: Extended CSR-style pointers. Modified in-place.

    Launch Grid
    -----------
    dim = num_systems + 1
    """
    sys = wp.tid()
    extended_atom_ptr[sys] = atom_ptr[sys] + 2 * sys


# ==============================================================================
# Stress to Cell Force Conversion Kernel
# ==============================================================================


@wp.kernel
def _stress_to_cell_force_kernel(
    stress: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    cell_force: wp.array(dtype=Any),
    keep_aligned: wp.bool,
):
    r"""Convert stress tensor to cell force for optimization.

    The cell "force" is computed as:

    .. math::

        \mathbf{F}_{\text{cell}} = -V \cdot \boldsymbol{\sigma} \cdot (\mathbf{H}^{-1})^T

    where V is cell volume, σ is the stress tensor, and H is the cell matrix.

    For upper-triangular cells, this simplifies since H^{-1} is also upper-triangular.

    Parameters
    ----------
    stress : wp.array, shape (B,), dtype=wp.mat33*
        Stress tensor in tension-positive (negative for compression)
        convention, in energy/volume units.  For zero-pressure relaxation
        this is typically ``virial / V`` where virial = −Σ r⊗F from the
        LJ kernel.  For finite external pressure use
        ``P_ext − P_internal`` (see ``virial_to_stress``).
    cell : wp.array, shape (B,), dtype=wp.mat33*
        Cell matrices (should be upper-triangular from align_cell).
    volume : wp.array, shape (B,), dtype=wp.float*
        Cell volumes.
    cell_force : wp.array, shape (B,), dtype=wp.mat33*
        Output cell force matrices.
    keep_aligned : wp.bool
        If True, zero out upper-triangular off-diagonal elements [0,1], [0,2], [1,2]
        to prevent the cell from rotating away from upper-triangular form.

    Launch Grid
    -----------
    dim = [num_systems]

    Notes
    -----
    - The stress follows a tension-positive sign convention: negative values
      indicate compression, positive values indicate tension / expansion.
    - The negative prefactor in the formula ensures correct equilibration:
      negative stress (compression) produces a positive cell force that
      expands the cell, while positive stress (tension) produces a negative
      cell force that contracts the cell.
    - When keep_aligned=True, the upper off-diagonal elements are zeroed to
      maintain the upper-triangular cell representation from align_cell().
      This is essential for stable variable-cell optimization.
    """
    sys = wp.tid()

    V = volume[sys]
    S = stress[sys]
    H = cell[sys]

    # Compute H^{-1}
    H_inv = wp.inverse(H)

    # F_cell = -V * S @ H_inv^T
    # Note: in warp, H_inv * x is matrix-vector, we need transpose
    H_inv_T = wp.transpose(H_inv)
    Fc = type(S[0, 0])(-1.0) * V * wp.mul(S, H_inv_T)

    # Zero upper off-diagonal to keep cell aligned (upper-triangular)
    if keep_aligned:
        _zero = type(S[0, 0])(0.0)
        Fc_aligned = type(Fc)(
            Fc[0, 0],
            _zero,
            _zero,  # Row 0: keep [0,0], zero [0,1] and [0,2]
            Fc[1, 0],
            Fc[1, 1],
            _zero,  # Row 1: keep [1,0] and [1,1], zero [1,2]
            Fc[2, 0],
            Fc[2, 1],
            Fc[2, 2],  # Row 2: keep all
        )
        cell_force[sys] = Fc_aligned
    else:
        cell_force[sys] = Fc


# ==============================================================================
# Kernel Overloads for Explicit Typing
# ==============================================================================

_T = [wp.float32, wp.float64]  # Scalar types
_V = [wp.vec3f, wp.vec3d]  # Vector types
_M = [wp.mat33f, wp.mat33d]  # Matrix types

# Cell alignment kernel overloads
_align_cell_kernel_overload = {}
_apply_transform_single_kernel_overload = {}
_apply_transform_kernel_overload = {}

# Pack/unpack kernel overloads (single system)
_pack_positions_kernel_overload = {}
_unpack_positions_kernel_overload = {}
_pack_forces_kernel_overload = {}
_pack_masses_kernel_overload = {}

# Pack/unpack kernel overloads (batched with atom_ptr + batch_idx)
_pack_atoms_batched_kernel_overload = {}
_pack_cell_dofs_kernel_overload = {}
_pack_cell_force_dofs_kernel_overload = {}
_unpack_atoms_batched_kernel_overload = {}
_unpack_cell_dofs_kernel_overload = {}
_pack_masses_batched_kernel_overload = {}

# Stress to cell force kernel overloads
_stress_to_cell_force_kernel_overload = {}

for t, v, m in zip(_T, _V, _M):
    # Cell alignment kernels
    _align_cell_kernel_overload[m] = wp.overload(
        _align_cell_kernel,
        [wp.array(dtype=m), wp.array(dtype=m)],
    )
    _apply_transform_single_kernel_overload[v] = wp.overload(
        _apply_transform_single_kernel,
        [wp.array(dtype=v), wp.array(dtype=m)],
    )
    _apply_transform_kernel_overload[v] = wp.overload(
        _apply_transform_kernel,
        [wp.array(dtype=v), wp.array(dtype=m), wp.array(dtype=wp.int32)],
    )

    # Pack/unpack kernels
    _pack_positions_kernel_overload[v] = wp.overload(
        _pack_positions_kernel,
        [wp.array(dtype=v), wp.array(dtype=m), wp.array(dtype=v), wp.int32],
    )
    _unpack_positions_kernel_overload[v] = wp.overload(
        _unpack_positions_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=m), wp.int32],
    )
    _pack_forces_kernel_overload[v] = wp.overload(
        _pack_forces_kernel,
        [wp.array(dtype=v), wp.array(dtype=m), wp.array(dtype=v), wp.int32],
    )
    _pack_masses_kernel_overload[t] = wp.overload(
        _pack_masses_kernel,
        [wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=t), wp.int32],
    )

    # Batched pack/unpack kernels (atom_ptr + batch_idx)
    _i32 = wp.array(dtype=wp.int32)
    _pack_atoms_batched_kernel_overload[v] = wp.overload(
        _pack_atoms_batched_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), _i32, _i32, _i32],
    )
    _pack_cell_dofs_kernel_overload[v] = wp.overload(
        _pack_cell_dofs_kernel,
        [wp.array(dtype=m), wp.array(dtype=v), _i32, _i32],
    )
    _pack_cell_force_dofs_kernel_overload[v] = wp.overload(
        _pack_cell_force_dofs_kernel,
        [wp.array(dtype=m), wp.array(dtype=v), _i32, _i32],
    )
    _unpack_atoms_batched_kernel_overload[v] = wp.overload(
        _unpack_atoms_batched_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), _i32, _i32, _i32],
    )
    _unpack_cell_dofs_kernel_overload[v] = wp.overload(
        _unpack_cell_dofs_kernel,
        [wp.array(dtype=v), wp.array(dtype=m), _i32, _i32],
    )
    _pack_masses_batched_kernel_overload[t] = wp.overload(
        _pack_masses_batched_kernel,
        [
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
        ],
    )

    # Stress to cell force kernel
    _stress_to_cell_force_kernel_overload[m] = wp.overload(
        _stress_to_cell_force_kernel,
        [
            wp.array(dtype=m),
            wp.array(dtype=m),
            wp.array(dtype=t),
            wp.array(dtype=m),
            wp.bool,
        ],
    )


# ==============================================================================
# Functional Interfaces
# ==============================================================================


def align_cell(
    positions: wp.array,
    cell: wp.array,
    transform: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> tuple[wp.array, wp.array]:
    """
    Align cell to upper-triangular form and transform positions accordingly.

    This is a one-time preprocessing step before variable-cell optimization.
    The cell is transformed to the standard upper-triangular form, and
    positions are rotated to maintain their fractional coordinates.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,). Modified in-place.
    cell : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell matrices. Shape (B,). Modified in-place.
    transform : wp.array(dtype=wp.mat33f or wp.mat33d)
        Scratch array for rotation transform. Shape (B,).
        Caller must pre-allocate.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Shape (N,). If None, assumes single system.
    device : str, optional
        Warp device. If None, inferred from positions.

    Returns
    -------
    tuple[wp.array, wp.array]
        (positions, cell) - same arrays, modified in-place for convenience.

    Example
    -------
    >>> # Before optimization loop
    >>> transform = wp.zeros(1, dtype=wp.mat33d, device=device)
    >>> positions, cell = align_cell(positions, cell, transform)
    """
    if device is None:
        device = positions.device

    num_systems = cell.shape[0]
    num_atoms = positions.shape[0]

    mat_dtype = cell.dtype
    vec_dtype = positions.dtype

    # Align cell and compute transform
    wp.launch(
        _align_cell_kernel_overload[mat_dtype],
        dim=num_systems,
        inputs=[cell, transform],
        device=device,
    )

    # Apply transform to positions: pos_new = pos @ transform
    if batch_idx is None:
        # Single system
        wp.launch(
            _apply_transform_single_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, transform],
            device=device,
        )
    else:
        wp.launch(
            _apply_transform_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, transform, batch_idx],
            device=device,
        )

    return positions, cell


def extend_batch_idx(
    batch_idx: wp.array,
    num_atoms: int,
    num_systems: int,
    extended_batch_idx: wp.array,
    device: str = None,
) -> wp.array:
    """
    Extend batch_idx to include cell DOFs for variable-cell optimization.

    For each system, 2 additional "atoms" (representing the 6 cell DOFs as 2 vec3s)
    are appended. The extended batch_idx assigns these cell DOFs to their
    respective systems.

    Parameters
    ----------
    batch_idx : wp.array(dtype=wp.int32)
        Original batch index for atoms. Shape (N,).
    num_atoms : int
        Number of atoms (N).
    num_systems : int
        Number of systems (B).
    extended_batch_idx : wp.array
        Output extended batch index. Shape (N + 2*B,).
        Caller must pre-allocate.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Extended batch index. Shape (N + 2*B,).

    Example
    -------
    >>> # Original: 100 atoms across 2 systems
    >>> # Extended: 100 + 4 = 104 "atoms" (2 cell DOFs per system)
    >>> ext_batch_idx = wp.zeros(104, dtype=wp.int32, device=device)
    >>> extend_batch_idx(batch_idx, num_atoms=100, num_systems=2, extended_batch_idx=ext_batch_idx)
    """
    if device is None:
        device = batch_idx.device

    extended_size = num_atoms + 2 * num_systems

    wp.launch(
        _extend_batch_idx_kernel,
        dim=extended_size,
        inputs=[batch_idx, extended_batch_idx, num_atoms, num_systems],
        device=device,
    )

    return extended_batch_idx


def extend_atom_ptr(
    atom_ptr: wp.array,
    extended_atom_ptr: wp.array,
    device: str = None,
) -> wp.array:
    """
    Extend atom_ptr to include cell DOFs for variable-cell optimization.

    Each system gets 2 additional DOFs (representing 6 cell parameters as 2 vec3s),
    so the CSR pointers are adjusted: extended_atom_ptr[sys] = atom_ptr[sys] + 2*sys.

    Parameters
    ----------
    atom_ptr : wp.array(dtype=wp.int32)
        Original CSR pointers. Shape (B+1,).
    extended_atom_ptr : wp.array
        Output extended CSR pointers. Shape (B+1,).
        Caller must pre-allocate.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Extended CSR pointers. Shape (B+1,).

    Example
    -------
    >>> # Original: atom_ptr = [0, 50, 100] (2 systems, 50 atoms each)
    >>> # Extended: [0, 52, 104] (50+2=52, 100+4=104)
    >>> ext_atom_ptr = wp.zeros(3, dtype=wp.int32, device=device)
    >>> extend_atom_ptr(atom_ptr, ext_atom_ptr)
    """
    if device is None:
        device = atom_ptr.device

    num_systems_plus_one = atom_ptr.shape[0]

    wp.launch(
        _extend_atom_ptr_kernel,
        dim=num_systems_plus_one,
        inputs=[atom_ptr, extended_atom_ptr],
        device=device,
    )

    return extended_atom_ptr


def pack_positions_with_cell(
    positions: wp.array,
    cell: wp.array,
    extended: wp.array,
    atom_ptr: wp.array = None,
    ext_atom_ptr: wp.array = None,
    device: str = None,
    batch_idx: wp.array = None,
) -> wp.array:
    """
    Pack atomic positions and cell into extended position array.

    Single-system mode (atom_ptr=None):
        The extended array has shape (N + 2,) with dtype vec3*, where:
        - First N entries: atomic positions
        - Entry N: [a, b*cos(γ), c1] (first 3 cell parameters)
        - Entry N+1: [b*sin(γ), c2, c3] (remaining 3 cell parameters)

    Batched mode (atom_ptr provided):
        Positions are concatenated across systems, cells have shape (B,).
        The extended array interleaves each system's positions with its cell DOFs.
        Both atom_ptr and ext_atom_ptr must be provided.  Optionally pass
        batch_idx to avoid recomputing it internally.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,) for single system or (total_atoms,) for batched.
    cell : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell matrix (should be upper-triangular from align_cell).
        Shape (1,) for single system or (B,) for batched.
    extended : wp.array
        Output extended array. Caller must pre-allocate.
        Shape (N+2,) for single, (N+2*B,) for batched.
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style atom pointers. Shape (B+1,). If provided, enables batched mode.
    ext_atom_ptr : wp.array(dtype=wp.int32), optional
        Extended atom pointers from extend_atom_ptr(). Shape (B+1,).
        Required if atom_ptr is provided.
    device : str, optional
        Warp device.
    batch_idx : wp.array(dtype=wp.int32), optional
        Sorted system index per atom. Shape (N,). Computed from atom_ptr
        if not provided.

    Returns
    -------
    wp.array
        Extended position array.
    """
    if device is None:
        device = positions.device

    vec_dtype = positions.dtype

    if atom_ptr is None:
        # Single system mode
        num_atoms = positions.shape[0]
        wp.launch(
            _pack_positions_kernel_overload[vec_dtype],
            dim=num_atoms + 2,
            inputs=[positions, cell, extended, num_atoms],
            device=device,
        )
    else:
        # Batched mode
        N = positions.shape[0]
        M = atom_ptr.shape[0] - 1
        if batch_idx is None:
            batch_idx = wp.empty(N, dtype=wp.int32, device=device)
            atom_ptr_to_batch_idx(atom_ptr, batch_idx)
        wp.launch(
            _pack_atoms_batched_kernel_overload[vec_dtype],
            dim=N,
            inputs=[positions, extended, batch_idx, atom_ptr, ext_atom_ptr],
            device=device,
        )
        wp.launch(
            _pack_cell_dofs_kernel_overload[vec_dtype],
            dim=M,
            inputs=[cell, extended, atom_ptr, ext_atom_ptr],
            device=device,
        )

    return extended


def unpack_positions_with_cell(
    extended: wp.array,
    positions: wp.array,
    cell: wp.array,
    num_atoms: int = None,
    atom_ptr: wp.array = None,
    ext_atom_ptr: wp.array = None,
    device: str = None,
    batch_idx: wp.array = None,
) -> tuple[wp.array, wp.array]:
    """
    Unpack extended position array to atomic positions and cell.

    Single-system mode (atom_ptr=None):
        Unpacks extended array of shape (N + 2,) to positions (N,) and cell (1,).
        Requires num_atoms to be specified.

    Batched mode (atom_ptr provided):
        Unpacks extended array to concatenated positions (total_atoms,) and
        cells (B,). Both atom_ptr and ext_atom_ptr must be provided.
        Optionally pass batch_idx to avoid recomputing it internally.

    Parameters
    ----------
    extended : wp.array(dtype=wp.vec3f or wp.vec3d)
        Extended position array.
    positions : wp.array
        Output atomic positions. Caller must pre-allocate.
    cell : wp.array
        Output cell matrix. Caller must pre-allocate.
    num_atoms : int, optional
        Number of atoms (N). Required for single-system mode.
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style atom pointers. Shape (B+1,). If provided, enables batched mode.
    ext_atom_ptr : wp.array(dtype=wp.int32), optional
        Extended atom pointers from extend_atom_ptr(). Shape (B+1,).
        Required if atom_ptr is provided.
    device : str, optional
        Warp device.
    batch_idx : wp.array(dtype=wp.int32), optional
        Sorted system index per atom. Shape (N,). Computed from atom_ptr
        if not provided.

    Returns
    -------
    tuple[wp.array, wp.array]
        (positions, cell)
    """
    if device is None:
        device = extended.device

    vec_dtype = extended.dtype

    if atom_ptr is None:
        # Single system mode
        if num_atoms is None:
            raise ValueError("num_atoms is required for single-system mode")
        wp.launch(
            _unpack_positions_kernel_overload[vec_dtype],
            dim=num_atoms + 2,
            inputs=[extended, positions, cell, num_atoms],
            device=device,
        )
    else:
        # Batched mode
        N = positions.shape[0]
        M = atom_ptr.shape[0] - 1
        if batch_idx is None:
            batch_idx = wp.empty(N, dtype=wp.int32, device=device)
            atom_ptr_to_batch_idx(atom_ptr, batch_idx)
        wp.launch(
            _unpack_atoms_batched_kernel_overload[vec_dtype],
            dim=N,
            inputs=[extended, positions, batch_idx, atom_ptr, ext_atom_ptr],
            device=device,
        )
        wp.launch(
            _unpack_cell_dofs_kernel_overload[vec_dtype],
            dim=M,
            inputs=[extended, cell, atom_ptr, ext_atom_ptr],
            device=device,
        )

    return positions, cell


def pack_velocities_with_cell(
    velocities: wp.array,
    cell_velocity: wp.array,
    extended: wp.array,
    atom_ptr: wp.array = None,
    ext_atom_ptr: wp.array = None,
    device: str = None,
    batch_idx: wp.array = None,
) -> wp.array:
    """
    Pack atomic velocities and cell velocity into extended velocity array.

    Single-system mode (atom_ptr=None):
        Extended array has shape (N + 2,).

    Batched mode (atom_ptr provided):
        Velocities are concatenated across systems, cell velocities have shape (B,).
        Both atom_ptr and ext_atom_ptr must be provided.  Optionally pass
        batch_idx to avoid recomputing it internally.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,) for single system or (total_atoms,) for batched.
    cell_velocity : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell velocity matrix. Shape (1,) for single system or (B,) for batched.
    extended : wp.array
        Output extended array. Caller must pre-allocate.
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style atom pointers. Shape (B+1,). If provided, enables batched mode.
    ext_atom_ptr : wp.array(dtype=wp.int32), optional
        Extended atom pointers from extend_atom_ptr(). Shape (B+1,).
        Required if atom_ptr is provided.
    device : str, optional
        Warp device.
    batch_idx : wp.array(dtype=wp.int32), optional
        Sorted system index per atom. Computed from atom_ptr if not provided.

    Returns
    -------
    wp.array
        Extended velocity array.
    """
    # Reuse pack_positions_with_cell - same packing format
    return pack_positions_with_cell(
        velocities, cell_velocity, extended, atom_ptr, ext_atom_ptr, device, batch_idx
    )


def unpack_velocities_with_cell(
    extended: wp.array,
    velocities: wp.array,
    cell_velocity: wp.array,
    num_atoms: int = None,
    atom_ptr: wp.array = None,
    ext_atom_ptr: wp.array = None,
    device: str = None,
    batch_idx: wp.array = None,
) -> tuple[wp.array, wp.array]:
    """
    Unpack extended velocity array to atomic velocities and cell velocity.

    Single-system mode (atom_ptr=None):
        Unpacks extended array of shape (N + 2,). Requires num_atoms.

    Batched mode (atom_ptr provided):
        Unpacks to concatenated velocities (total_atoms,) and cell velocities (B,).
        Both atom_ptr and ext_atom_ptr must be provided.  Optionally pass
        batch_idx to avoid recomputing it internally.

    Parameters
    ----------
    extended : wp.array(dtype=wp.vec3f or wp.vec3d)
        Extended velocity array.
    velocities : wp.array
        Output atomic velocities. Caller must pre-allocate.
    cell_velocity : wp.array
        Output cell velocity matrix. Caller must pre-allocate.
    num_atoms : int, optional
        Number of atoms (N). Required for single-system mode.
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style atom pointers. Shape (B+1,). If provided, enables batched mode.
    ext_atom_ptr : wp.array(dtype=wp.int32), optional
        Extended atom pointers from extend_atom_ptr(). Shape (B+1,).
        Required if atom_ptr is provided.
    device : str, optional
        Warp device.
    batch_idx : wp.array(dtype=wp.int32), optional
        Sorted system index per atom. Computed from atom_ptr if not provided.

    Returns
    -------
    tuple[wp.array, wp.array]
        (velocities, cell_velocity)
    """
    return unpack_positions_with_cell(
        extended,
        velocities,
        cell_velocity,
        num_atoms,
        atom_ptr,
        ext_atom_ptr,
        device,
        batch_idx,
    )


def pack_forces_with_cell(
    forces: wp.array,
    cell_force: wp.array,
    extended: wp.array,
    atom_ptr: wp.array = None,
    ext_atom_ptr: wp.array = None,
    device: str = None,
    batch_idx: wp.array = None,
) -> wp.array:
    """
    Pack atomic forces and cell force into extended force array.

    Single-system mode (atom_ptr=None):
        Extended array has shape (N + 2,).

    Batched mode (atom_ptr provided):
        Forces are concatenated across systems, cell forces have shape (B,).
        Both atom_ptr and ext_atom_ptr must be provided.  Optionally pass
        batch_idx to avoid recomputing it internally.

    Parameters
    ----------
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic forces. Shape (N,) for single system or (total_atoms,) for batched.
    cell_force : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell force matrix (from stress_to_cell_force).
        Shape (1,) for single system or (B,) for batched.
    extended : wp.array
        Output extended array. Caller must pre-allocate.
        Shape (N+2,) for single, (N+2*B,) for batched.
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style atom pointers. Shape (B+1,). If provided, enables batched mode.
    ext_atom_ptr : wp.array(dtype=wp.int32), optional
        Extended atom pointers from extend_atom_ptr(). Shape (B+1,).
        Required if atom_ptr is provided.
    device : str, optional
        Warp device.
    batch_idx : wp.array(dtype=wp.int32), optional
        Sorted system index per atom. Computed from atom_ptr if not provided.

    Returns
    -------
    wp.array
        Extended force array.
    """
    if device is None:
        device = forces.device

    vec_dtype = forces.dtype

    if atom_ptr is None:
        # Single system mode
        num_atoms = forces.shape[0]
        wp.launch(
            _pack_forces_kernel_overload[vec_dtype],
            dim=num_atoms + 2,
            inputs=[forces, cell_force, extended, num_atoms],
            device=device,
        )
    else:
        # Batched mode
        N = forces.shape[0]
        M = atom_ptr.shape[0] - 1
        if batch_idx is None:
            batch_idx = wp.empty(N, dtype=wp.int32, device=device)
            atom_ptr_to_batch_idx(atom_ptr, batch_idx)
        wp.launch(
            _pack_atoms_batched_kernel_overload[vec_dtype],
            dim=N,
            inputs=[forces, extended, batch_idx, atom_ptr, ext_atom_ptr],
            device=device,
        )
        wp.launch(
            _pack_cell_force_dofs_kernel_overload[vec_dtype],
            dim=M,
            inputs=[cell_force, extended, atom_ptr, ext_atom_ptr],
            device=device,
        )

    return extended


def pack_masses_with_cell(
    masses: wp.array,
    cell_mass_arr: wp.array,
    extended: wp.array,
    atom_ptr: wp.array = None,
    ext_atom_ptr: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Pack atomic masses and cell mass into extended mass array.

    Single-system mode (atom_ptr=None):
        Extended array has shape (N + 2,).

    Batched mode (atom_ptr provided):
        Masses are concatenated across systems. Cell mass is applied to all systems.
        Both atom_ptr and ext_atom_ptr must be provided.

    Parameters
    ----------
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,) for single system or (total_atoms,) for batched.
    cell_mass_arr : wp.array
        Cell mass as a warp array. Shape (1,) for single system or (B,) for batched.
        Caller must pre-allocate.
    extended : wp.array
        Output extended array. Caller must pre-allocate.
        Shape (N+2,) for single, (N+2*B,) for batched.
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style atom pointers. Shape (B+1,). If provided, enables batched mode.
    ext_atom_ptr : wp.array(dtype=wp.int32), optional
        Extended atom pointers from extend_atom_ptr(). Shape (B+1,).
        Required if atom_ptr is provided.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Extended mass array.
    """
    if device is None:
        device = masses.device

    scalar_dtype = masses.dtype

    if atom_ptr is None:
        # Single system mode
        num_atoms = masses.shape[0]

        wp.launch(
            _pack_masses_kernel_overload[scalar_dtype],
            dim=num_atoms + 2,
            inputs=[masses, cell_mass_arr, extended, num_atoms],
            device=device,
        )
    else:
        # Batched mode with atom_ptr
        num_systems = atom_ptr.shape[0] - 1

        # Launch with num_systems threads (each handles one system)
        wp.launch(
            _pack_masses_batched_kernel_overload[scalar_dtype],
            dim=num_systems,
            inputs=[masses, cell_mass_arr, extended, atom_ptr, ext_atom_ptr],
            device=device,
        )

    return extended


def stress_to_cell_force(
    stress: wp.array,
    cell: wp.array,
    volume: wp.array,
    cell_force: wp.array,
    keep_aligned: bool = True,
    device: str = None,
) -> wp.array:
    r"""
    Convert stress tensor to cell force for optimization.

    Computes: F_cell = -V * σ * (H^{-1})^T

    This is the "force" on the cell that, when minimized, leads to
    zero stress (pressure equilibration).

    Parameters
    ----------
    stress : wp.array(dtype=wp.mat33f or wp.mat33d)
        Stress tensor. Shape (B,).
        Convention: positive values indicate compression.
    cell : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell matrices. Shape (B,).
    volume : wp.array
        Cell volumes. Shape (B,). Caller must pre-compute via
        ``compute_cell_volume``.
    cell_force : wp.array
        Output cell force matrices. Shape (B,). Caller must pre-allocate.
    keep_aligned : bool, default=True
        If True, zero out upper-triangular off-diagonal elements [0,1], [0,2], [1,2]
        of the cell force. This is **essential** to prevent the cell from rotating
        away from the upper-triangular form established by `align_cell()`.
        Only set to False if you know what you're doing.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Cell force matrices. Shape (B,).

    Notes
    -----
    The `keep_aligned=True` behavior zeros out forces on the upper off-diagonal
    elements of the cell matrix:

    .. code-block:: text

        Cell force structure (keep_aligned=True):
        [F00,  0,   0 ]
        [F10, F11,  0 ]
        [F20, F21, F22]

    This prevents the optimizer from introducing rotations that would break
    the upper-triangular cell representation from `align_cell()`.
    """
    if device is None:
        device = stress.device

    num_systems = stress.shape[0]
    mat_dtype = stress.dtype

    wp.launch(
        _stress_to_cell_force_kernel_overload[mat_dtype],
        dim=num_systems,
        inputs=[stress, cell, volume, cell_force, keep_aligned],
        device=device,
    )

    return cell_force
