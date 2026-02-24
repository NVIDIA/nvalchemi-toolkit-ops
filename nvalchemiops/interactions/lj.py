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
Lennard-Jones Potential
=======================

This module implements GPU-accelerated Lennard-Jones (LJ) energy and force
calculations using Warp kernels with neighbor lists for O(N) scaling.

Mathematical Formulation
------------------------

The Lennard-Jones potential describes the interaction between a pair of
neutral atoms or molecules:

.. math::

    V_{LJ}(r) = 4\\epsilon \\left[ \\left(\\frac{\\sigma}{r}\\right)^{12}
                - \\left(\\frac{\\sigma}{r}\\right)^{6} \\right]

where:
- :math:`\\epsilon` is the depth of the potential well (energy)
- :math:`\\sigma` is the distance at which the potential is zero
- :math:`r` is the interparticle distance

The force is the negative gradient:

.. math::

    F_{LJ}(r) = -\\frac{dV}{dr} = \\frac{24\\epsilon}{r}
                \\left[ 2\\left(\\frac{\\sigma}{r}\\right)^{12}
                - \\left(\\frac{\\sigma}{r}\\right)^{6} \\right]

For the virial tensor (needed for pressure calculations in NPT/NPH):

.. math::

    W_{\\alpha\\beta} = \\sum_{i<j} r_{ij,\\alpha} \\cdot F_{ij,\\beta}

.. note::
    This implementation assumes a **half neighbor list** where each pair (i, j)
    appears only once (i.e., only for i < j or only for i > j). Newton's third
    law is applied to accumulate forces on both atoms.

Neighbor Formats
----------------

This module supports two neighbor formats:

1. **Neighbor List (CSR format)**: `idx_j` contains target atom indices,
   `neighbor_ptr` contains CSR row pointers where neighbor_ptr[i] to
   neighbor_ptr[i+1] gives the range of neighbors for atom i.

2. **Neighbor Matrix**: `neighbor_matrix` is shape (N, max_neighbors) where
   each row contains neighbor indices for that atom.

API Structure
-------------

Public Wrappers:
    - `lj_energy()`: Compute energies only
    - `lj_forces()`: Compute forces only (convenience)
    - `lj_energy_forces()`: Compute both energies and forces
    - `lj_energy_forces_virial()`: Compute energies, forces, and virial tensor

References
----------
- Lennard-Jones, J. E. (1924). Proc. R. Soc. A, 106, 463-477
- Allen & Tildesley, "Computer Simulation of Liquids" (1987)

Examples
--------
>>> import warp as wp
>>> import numpy as np
>>> from nvalchemiops.interactions.lj import lj_energy_forces
>>>
>>> # Create simple FCC argon system
>>> positions = wp.array(pos_np, dtype=wp.vec3d, device="cuda:0")
>>> cell = wp.array(cell_np, dtype=wp.mat33d, device="cuda:0")
>>>
>>> # Compute LJ energy and forces
>>> energies, forces = lj_energy_forces(
...     positions=positions,
...     cell=cell,
...     neighbor_matrix=neighbor_matrix,
...     neighbor_matrix_shifts=neighbor_shifts,
...     num_neighbors=num_neighbors,
...     epsilon=0.0104,  # eV (argon)
...     sigma=3.40,      # Angstrom
...     cutoff=8.5,      # Angstrom (2.5*sigma)
...     fill_value=num_atoms,
... )
>>> print(f"Total energy: {energies.numpy().sum():.4f} eV")
"""

from __future__ import annotations

from typing import Any

import warp as wp

from nvalchemiops.interactions.switching import switch_c2

__all__ = [
    "lj_energy",
    "lj_forces",
    "lj_energy_forces",
    "lj_energy_forces_virial",
]


# ==============================================================================
# Helper Functions (dtype-flexible)
# ==============================================================================


@wp.func
def _lj_energy_pair(
    sigma_over_r: wp.float64,
    epsilon: wp.float64,
) -> wp.float64:
    """Compute LJ pair energy.

    Formula: V = 4 * epsilon * (s^12 - s^6) where s = sigma/r
    """
    s2 = sigma_over_r * sigma_over_r
    s6 = s2 * s2 * s2
    s12 = s6 * s6
    return wp.float64(4.0) * epsilon * (s12 - s6)


@wp.func
def _lj_force_over_r(
    sigma_over_r: wp.float64,
    epsilon: wp.float64,
    r_sq: wp.float64,
) -> wp.float64:
    """Compute LJ force magnitude divided by r.

    Formula: F/r = 24 * epsilon / r^2 * (2*s^12 - s^6) where s = sigma/r
    """
    s2 = sigma_over_r * sigma_over_r
    s6 = s2 * s2 * s2
    s12 = s6 * s6
    return wp.float64(24.0) * epsilon * (wp.float64(2.0) * s12 - s6) / r_sq


@wp.func
def _switch_params(
    cutoff: wp.float64,
    switch_width: wp.float64,
) -> tuple[wp.float64, wp.float64]:
    """Return (r_on, r_cut). If switch_width <= 0, returns (cutoff, cutoff)."""
    if switch_width <= wp.float64(0.0):
        return cutoff, cutoff
    r_on = cutoff - switch_width
    if r_on < wp.float64(0.0):
        r_on = wp.float64(0.0)
    return r_on, cutoff


# ==============================================================================
# Warp Kernels - Neighbor Matrix Format
# ==============================================================================


@wp.kernel
def _lj_energy_matrix_kernel(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    epsilon: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    switch_width: wp.array(dtype=Any),
    half_neighbor_list: wp.bool,
    fill_value: wp.int32,
    atomic_energies: wp.array(dtype=Any),
):
    r"""Compute Lennard-Jones energies using neighbor matrix format.

    This kernel supports both **half** and **full** neighbor matrices via the
    `half_neighbor_list` flag:

    - **Half neighbor list**: each pair (i, j) appears once. The per-pair energy
      \(V_{ij}\) is split evenly across i and j (adds 0.5 * V to each).
    - **Full neighbor list**: each pair appears twice (once in i's row and once
      in j's row). The kernel adds 0.5 * V to the current atom only, so the
      total pair energy is still counted exactly once overall.

    Switching: if `switch_width > 0`, energy is multiplied by a C2 switching
    function between r_on = cutoff - switch_width and r_cut = cutoff.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atom positions (Cartesian, in the same length units as sigma/cutoff).
    cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix (used to convert integer shift vectors to Cartesian).
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices for each atom; padded entries are expected to be
        `fill_value` (or >= N).
    neighbor_matrix_shifts : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Integer lattice shifts corresponding to each neighbor entry.
    num_neighbors : wp.array, shape (N,), dtype=wp.int32
        Valid neighbor count per atom (for iterating only filled slots).
    epsilon, sigma, cutoff, switch_width : wp.array, shape (1,), dtype=float32/float64
        Scalar LJ parameters packed as 1-element device arrays.
    half_neighbor_list : wp.bool
        True if the neighbor matrix contains each pair once; False if pairs are duplicated.
    fill_value : wp.int32
        Sentinel value used to pad `neighbor_matrix` rows.
    atomic_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: per-atom energies accumulated atomically (matches input precision).

    Notes
    -----
    - Internal math is performed in float64 for stability; output is cast to match input dtype.
    - Pairs closer than ~1e-5 (r^2 < 1e-10) are skipped for safety.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    cell_t = wp.transpose(cell[0])
    eps = wp.float64(epsilon[0])
    sig = wp.float64(sigma[0])
    cut = wp.float64(cutoff[0])
    sw = wp.float64(switch_width[0])
    cutoff_sq = cut * cut
    r_on, r_cut = _switch_params(cut, sw)

    n_neighbors = num_neighbors[atom_i]

    for neighbor_slot in range(n_neighbors):
        if neighbor_slot >= max_neighbors:
            break

        j = neighbor_matrix[atom_i, neighbor_slot]
        if j >= fill_value or j >= num_atoms:
            continue

        rj = positions[j]

        # Compute shift vector for periodic boundaries
        shift = neighbor_matrix_shifts[atom_i, neighbor_slot]
        shift_vec = cell_t * type(ri)(
            type(ri[0])(shift[0]),
            type(ri[0])(shift[1]),
            type(ri[0])(shift[2]),
        )

        # r_ij = r_i - r_j - shift (vector from j to i)
        r_ij = ri - rj - shift_vec
        r_sq = wp.float64(wp.dot(r_ij, r_ij))

        if r_sq >= cutoff_sq or r_sq < wp.float64(1e-10):
            continue

        r = wp.sqrt(r_sq)
        sigma_over_r = sig / r

        # Energy: V = 4*eps*(s12 - s6), each pair counted once
        pair_energy_raw = _lj_energy_pair(sigma_over_r, eps)
        if sw > wp.float64(0.0) and r > r_on:
            s, ds_dr = switch_c2(r, r_on, r_cut)
            pair_energy = s * pair_energy_raw
        else:
            pair_energy = pair_energy_raw
        # Energy accounting:
        # - If half neighbor list: each pair appears once; split 1/2 to i and 1/2 to j.
        # - If full neighbor list: each pair appears twice; add 1/2 to i only.
        # Cast from float64 accumulator back to output dtype
        half_energy = wp.float64(0.5) * pair_energy
        wp.atomic_add(atomic_energies, atom_i, type(atomic_energies[0])(half_energy))
        if half_neighbor_list:
            wp.atomic_add(atomic_energies, j, type(atomic_energies[0])(half_energy))

    # (energies accumulated per-pair)


@wp.kernel
def _lj_energy_forces_matrix_kernel(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    epsilon: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    switch_width: wp.array(dtype=Any),
    half_neighbor_list: wp.bool,
    fill_value: wp.int32,
    atomic_energies: wp.array(dtype=Any),
    atomic_forces: wp.array(dtype=Any),
):
    """Compute Lennard-Jones energies and forces using neighbor matrix format.

    Energy accounting matches `_lj_energy_matrix_kernel` (see its docstring).

    Forces are accumulated as:
    - **Half neighbor list**: applies Newton's 3rd law (updates both i and j).
    - **Full neighbor list**: updates only i (since j will process its own row).

    Switching: if `switch_width > 0`, both energy and force are smoothly switched
    to zero at cutoff using a C2 continuous switching function.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atom positions in Cartesian space.
    cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix (for periodic shift conversion).
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices for each atom.
    neighbor_matrix_shifts : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Integer lattice shifts for each neighbor entry.
    num_neighbors : wp.array, shape (N,), dtype=wp.int32
        Valid neighbor count per atom.
    epsilon, sigma, cutoff, switch_width : wp.array, shape (1,), dtype=float32/float64
        Scalar LJ parameters packed as 1-element arrays.
    half_neighbor_list : wp.bool
        True if each pair appears once; False if pairs appear twice.
    fill_value : wp.int32
        Sentinel value used to pad `neighbor_matrix`.
    atomic_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: per-atom energies (matches input precision).
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: per-atom forces accumulated atomically (matches positions dtype).

    Notes
    -----
    - Internal math is performed in float64 for stability; output is cast to match input dtype.
    - The kernel accumulates force on i in registers and performs one atomic add at the end.
    - Pairs with r^2 >= cutoff^2 (or extremely small separation) are skipped.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    cell_t = wp.transpose(cell[0])
    eps = wp.float64(epsilon[0])
    sig = wp.float64(sigma[0])
    cut = wp.float64(cutoff[0])
    sw = wp.float64(switch_width[0])
    cutoff_sq = cut * cut
    r_on, r_cut = _switch_params(cut, sw)

    force_acc = type(ri)(
        type(ri[0])(0.0),
        type(ri[0])(0.0),
        type(ri[0])(0.0),
    )

    n_neighbors = num_neighbors[atom_i]

    for neighbor_slot in range(n_neighbors):
        if neighbor_slot >= max_neighbors:
            break

        j = neighbor_matrix[atom_i, neighbor_slot]
        if j >= fill_value or j >= num_atoms:
            continue

        rj = positions[j]

        shift = neighbor_matrix_shifts[atom_i, neighbor_slot]
        shift_vec = cell_t * type(ri)(
            type(ri[0])(shift[0]),
            type(ri[0])(shift[1]),
            type(ri[0])(shift[2]),
        )

        r_ij = ri - rj - shift_vec
        r_sq = wp.float64(wp.dot(r_ij, r_ij))

        if r_sq >= cutoff_sq or r_sq < wp.float64(1e-10):
            continue

        r = wp.sqrt(r_sq)
        sigma_over_r = sig / r

        # Raw energy/force
        pair_energy_raw = _lj_energy_pair(sigma_over_r, eps)
        force_mag_over_r_raw = _lj_force_over_r(sigma_over_r, eps, r_sq)

        # Optional C2 switching
        if sw > wp.float64(0.0) and r > r_on:
            s, ds_dr = switch_c2(r, r_on, r_cut)
            pair_energy = s * pair_energy_raw
            force_mag_over_r = s * force_mag_over_r_raw + (-pair_energy_raw * ds_dr) / r
        else:
            pair_energy = pair_energy_raw
            force_mag_over_r = force_mag_over_r_raw

        # Energies: see note in _lj_energy_matrix_kernel
        # Cast from float64 accumulator back to output dtype
        half_energy = wp.float64(0.5) * pair_energy
        wp.atomic_add(atomic_energies, atom_i, type(atomic_energies[0])(half_energy))
        if half_neighbor_list:
            wp.atomic_add(atomic_energies, j, type(atomic_energies[0])(half_energy))

        force_ij = type(ri)(
            type(ri[0])(force_mag_over_r) * r_ij[0],
            type(ri[0])(force_mag_over_r) * r_ij[1],
            type(ri[0])(force_mag_over_r) * r_ij[2],
        )

        force_acc += force_ij
        # Forces:
        # - half neighbor list: apply Newton's 3rd law (update j here)
        # - full neighbor list: j will be handled by its own row, so don't update j
        if half_neighbor_list:
            wp.atomic_sub(atomic_forces, j, force_ij)
    wp.atomic_add(atomic_forces, atom_i, force_acc)


@wp.kernel
def _lj_energy_forces_virial_matrix_kernel(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    epsilon: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    switch_width: wp.array(dtype=Any),
    half_neighbor_list: wp.bool,
    fill_value: wp.int32,
    atomic_energies: wp.array(dtype=Any),
    atomic_forces: wp.array(dtype=Any),
    virial: wp.array(dtype=Any),
):
    r"""Compute Lennard-Jones energies, forces, and virial (neighbor matrix).

    Energy/force handling matches `_lj_energy_forces_matrix_kernel`.

    Virial tensor is accumulated as:

    \[
      W_{\\alpha\\beta} = -\\sum_{i<j} r_{ij,\\alpha} F_{ij,\\beta}
    \]

    The output uses a flattened 9-element layout:
    `[xx, xy, xz, yx, yy, yz, zx, zy, zz]`.

    For `half_neighbor_list=False` (full neighbor matrix), each pair appears twice,
    so the virial contribution is scaled by 0.5 per edge to avoid double-counting.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atom positions in Cartesian space.
    cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices.
    neighbor_matrix_shifts : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Periodic image shifts (integer lattice vectors).
    num_neighbors : wp.array, shape (N,), dtype=wp.int32
        Valid neighbor count per atom.
    epsilon, sigma, cutoff, switch_width : wp.array, shape (1,), dtype=float32/float64
        Scalar LJ parameters packed as 1-element arrays.
    half_neighbor_list : wp.bool
        True for half neighbor list; False for full neighbor list.
    fill_value : wp.int32
        Sentinel value used to pad `neighbor_matrix`.
    atomic_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: per-atom energies (matches input precision).
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: per-atom forces.
    virial : wp.array, shape (9,), dtype=wp.float32 or wp.float64
        OUTPUT: global virial tensor (flattened 3x3, matches input precision).

    Notes
    -----
    - Internal accumulation uses float64 registers; output is cast to match input dtype.
    - Uses the same C2 switching as the energy/forces kernels when `switch_width > 0`.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    cell_t = wp.transpose(cell[0])
    eps = wp.float64(epsilon[0])
    sig = wp.float64(sigma[0])
    cut = wp.float64(cutoff[0])
    sw = wp.float64(switch_width[0])
    cutoff_sq = cut * cut
    r_on, r_cut = _switch_params(cut, sw)

    force_acc = type(ri)(
        type(ri[0])(0.0),
        type(ri[0])(0.0),
        type(ri[0])(0.0),
    )

    # Local virial accumulator (9 components)
    vir_xx = wp.float64(0.0)
    vir_xy = wp.float64(0.0)
    vir_xz = wp.float64(0.0)
    vir_yx = wp.float64(0.0)
    vir_yy = wp.float64(0.0)
    vir_yz = wp.float64(0.0)
    vir_zx = wp.float64(0.0)
    vir_zy = wp.float64(0.0)
    vir_zz = wp.float64(0.0)

    n_neighbors = num_neighbors[atom_i]

    for neighbor_slot in range(n_neighbors):
        if neighbor_slot >= max_neighbors:
            break

        j = neighbor_matrix[atom_i, neighbor_slot]
        if j >= fill_value or j >= num_atoms:
            continue

        rj = positions[j]

        shift = neighbor_matrix_shifts[atom_i, neighbor_slot]
        shift_vec = cell_t * type(ri)(
            type(ri[0])(shift[0]),
            type(ri[0])(shift[1]),
            type(ri[0])(shift[2]),
        )

        r_ij = ri - rj - shift_vec
        r_sq = wp.float64(wp.dot(r_ij, r_ij))

        if r_sq >= cutoff_sq or r_sq < wp.float64(1e-10):
            continue

        r = wp.sqrt(r_sq)
        sigma_over_r = sig / r

        pair_energy_raw = _lj_energy_pair(sigma_over_r, eps)
        force_mag_over_r_raw = _lj_force_over_r(sigma_over_r, eps, r_sq)

        if sw > wp.float64(0.0) and r > r_on:
            s, ds_dr = switch_c2(r, r_on, r_cut)
            pair_energy = s * pair_energy_raw
            force_mag_over_r = s * force_mag_over_r_raw + (-pair_energy_raw * ds_dr) / r
        else:
            pair_energy = pair_energy_raw
            force_mag_over_r = force_mag_over_r_raw

        # Energies: see note in _lj_energy_matrix_kernel
        # Cast from float64 accumulator back to output dtype
        half_energy = wp.float64(0.5) * pair_energy
        wp.atomic_add(atomic_energies, atom_i, type(atomic_energies[0])(half_energy))
        if half_neighbor_list:
            wp.atomic_add(atomic_energies, j, type(atomic_energies[0])(half_energy))

        force_ij = type(ri)(
            type(ri[0])(force_mag_over_r) * r_ij[0],
            type(ri[0])(force_mag_over_r) * r_ij[1],
            type(ri[0])(force_mag_over_r) * r_ij[2],
        )

        force_acc += force_ij
        if half_neighbor_list:
            wp.atomic_sub(atomic_forces, j, force_ij)

        # Virial: W_αβ = r_ij,α * F_ij,β (float64 for accuracy)
        r_ij_0 = wp.float64(r_ij[0])
        r_ij_1 = wp.float64(r_ij[1])
        r_ij_2 = wp.float64(r_ij[2])
        f_ij_0 = wp.float64(force_ij[0])
        f_ij_1 = wp.float64(force_ij[1])
        f_ij_2 = wp.float64(force_ij[2])

        # Virial scaling:
        # - half neighbor list: each pair once, keep full contribution
        # - full neighbor list: each pair twice, so take 1/2 per edge
        vir_scale = wp.float64(1.0) if half_neighbor_list else wp.float64(0.5)
        vir_xx += vir_scale * (r_ij_0 * f_ij_0)
        vir_xy += vir_scale * (r_ij_0 * f_ij_1)
        vir_xz += vir_scale * (r_ij_0 * f_ij_2)
        vir_yx += vir_scale * (r_ij_1 * f_ij_0)
        vir_yy += vir_scale * (r_ij_1 * f_ij_1)
        vir_yz += vir_scale * (r_ij_1 * f_ij_2)
        vir_zx += vir_scale * (r_ij_2 * f_ij_0)
        vir_zy += vir_scale * (r_ij_2 * f_ij_1)
        vir_zz += vir_scale * (r_ij_2 * f_ij_2)
    wp.atomic_add(atomic_forces, atom_i, force_acc)

    # Accumulate virial (negative sign convention: W = -Σ r ⊗ F)
    # Cast from float64 accumulator back to output dtype

    wp.atomic_sub(virial, 0, type(virial[0])(vir_xx))
    wp.atomic_sub(virial, 1, type(virial[0])(vir_xy))
    wp.atomic_sub(virial, 2, type(virial[0])(vir_xz))
    wp.atomic_sub(virial, 3, type(virial[0])(vir_yx))
    wp.atomic_sub(virial, 4, type(virial[0])(vir_yy))
    wp.atomic_sub(virial, 5, type(virial[0])(vir_yz))
    wp.atomic_sub(virial, 6, type(virial[0])(vir_zx))
    wp.atomic_sub(virial, 7, type(virial[0])(vir_zy))
    wp.atomic_sub(virial, 8, type(virial[0])(vir_zz))


# ==============================================================================
# Warp Kernels - Neighbor List (CSR) Format
# ==============================================================================


@wp.kernel
def _lj_energy_list_kernel(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    epsilon: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    switch_width: wp.array(dtype=Any),
    half_neighbor_list: wp.bool,
    atomic_energies: wp.array(dtype=Any),
):
    """Compute Lennard-Jones energies using neighbor list (CSR) format.

    Neighbor list is provided in CSR form via (`neighbor_ptr`, `idx_j`, `unit_shifts`).
    Energy accounting follows the same convention as the neighbor matrix kernels:

    - **Half neighbor list**: adds 0.5 * V_ij to both i and j.
    - **Full neighbor list**: adds 0.5 * V_ij to i only (since the reverse edge exists).

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atom positions in Cartesian space.
    cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Flattened neighbor indices (CSR adjacency list).
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers. Neighbors for atom i are `idx_j[ptr[i]:ptr[i+1]]`.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Integer lattice shifts for each edge in CSR order.
    epsilon, sigma, cutoff, switch_width : wp.array, shape (1,), dtype=float32/float64
        Scalar LJ parameters packed as 1-element arrays.
    half_neighbor_list : wp.bool
        True if each pair appears once; False if edges are duplicated.
    atomic_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: per-atom energies in float64.

    Notes
    -----
    - Uses float64 internally for energy; output is float64.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]

    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    cell_t = wp.transpose(cell[0])
    eps = wp.float64(epsilon[0])
    sig = wp.float64(sigma[0])
    cut = wp.float64(cutoff[0])
    sw = wp.float64(switch_width[0])
    cutoff_sq = cut * cut
    r_on, r_cut = _switch_params(cut, sw)

    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]
        rj = positions[j]

        shift = unit_shifts[edge_idx]
        shift_vec = cell_t * type(ri)(
            type(ri[0])(shift[0]),
            type(ri[0])(shift[1]),
            type(ri[0])(shift[2]),
        )

        r_ij = ri - rj - shift_vec
        r_sq = wp.float64(wp.dot(r_ij, r_ij))

        if r_sq >= cutoff_sq or r_sq < wp.float64(1e-10):
            continue

        r = wp.sqrt(r_sq)
        sigma_over_r = sig / r

        pair_energy_raw = _lj_energy_pair(sigma_over_r, eps)
        if sw > wp.float64(0.0) and r > r_on:
            s, ds_dr = switch_c2(r, r_on, r_cut)
            pair_energy = s * pair_energy_raw
        else:
            pair_energy = pair_energy_raw

        # Energies: same convention as matrix kernels
        # Cast from float64 accumulator back to output dtype
        half_energy = wp.float64(0.5) * pair_energy
        wp.atomic_add(atomic_energies, atom_i, type(atomic_energies[0])(half_energy))
        if half_neighbor_list:
            wp.atomic_add(atomic_energies, j, type(atomic_energies[0])(half_energy))


@wp.kernel
def _lj_energy_forces_list_kernel(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    epsilon: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    switch_width: wp.array(dtype=Any),
    half_neighbor_list: wp.bool,
    atomic_energies: wp.array(dtype=Any),
    atomic_forces: wp.array(dtype=Any),
):
    """Compute Lennard-Jones energies and forces using neighbor list (CSR) format.

    Energy accounting matches `_lj_energy_list_kernel`.

    Forces are accumulated as:
    - **Half neighbor list**: applies Newton's 3rd law (updates i and j).
    - **Full neighbor list**: updates only i (j will process its own row).

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atom positions.
    cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Flattened neighbor indices.
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Integer lattice shifts for each edge.
    epsilon, sigma, cutoff, switch_width : wp.array, shape (1,), dtype=float32/float64
        Scalar LJ parameters packed as 1-element arrays.
    half_neighbor_list : wp.bool
        True if each pair appears once; False if edges are duplicated.
    atomic_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: per-atom energies.
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: per-atom forces.

    Notes
    -----
    - Switching is applied identically to the neighbor matrix forces kernel.
    - Force on i is accumulated locally and written once to reduce atomic contention.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]

    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    cell_t = wp.transpose(cell[0])
    eps = wp.float64(epsilon[0])
    sig = wp.float64(sigma[0])
    cut = wp.float64(cutoff[0])
    sw = wp.float64(switch_width[0])
    cutoff_sq = cut * cut
    r_on, r_cut = _switch_params(cut, sw)

    force_acc = type(ri)(
        type(ri[0])(0.0),
        type(ri[0])(0.0),
        type(ri[0])(0.0),
    )

    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]
        rj = positions[j]

        shift = unit_shifts[edge_idx]
        shift_vec = cell_t * type(ri)(
            type(ri[0])(shift[0]),
            type(ri[0])(shift[1]),
            type(ri[0])(shift[2]),
        )

        r_ij = ri - rj - shift_vec
        r_sq = wp.float64(wp.dot(r_ij, r_ij))

        if r_sq >= cutoff_sq or r_sq < wp.float64(1e-10):
            continue

        r = wp.sqrt(r_sq)
        sigma_over_r = sig / r

        pair_energy_raw = _lj_energy_pair(sigma_over_r, eps)
        force_mag_over_r_raw = _lj_force_over_r(sigma_over_r, eps, r_sq)

        if sw > wp.float64(0.0) and r > r_on:
            s, ds_dr = switch_c2(r, r_on, r_cut)
            pair_energy = s * pair_energy_raw
            force_mag_over_r = s * force_mag_over_r_raw + (-pair_energy_raw * ds_dr) / r
        else:
            pair_energy = pair_energy_raw
            force_mag_over_r = force_mag_over_r_raw

        # Cast from float64 accumulator back to output dtype
        half_energy = wp.float64(0.5) * pair_energy
        wp.atomic_add(atomic_energies, atom_i, type(atomic_energies[0])(half_energy))
        if half_neighbor_list:
            wp.atomic_add(atomic_energies, j, type(atomic_energies[0])(half_energy))
        force_ij = type(ri)(
            type(ri[0])(force_mag_over_r) * r_ij[0],
            type(ri[0])(force_mag_over_r) * r_ij[1],
            type(ri[0])(force_mag_over_r) * r_ij[2],
        )

        force_acc += force_ij
        if half_neighbor_list:
            wp.atomic_sub(atomic_forces, j, force_ij)
    wp.atomic_add(atomic_forces, atom_i, force_acc)


@wp.kernel
def _lj_energy_forces_virial_list_kernel(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    epsilon: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    switch_width: wp.array(dtype=Any),
    half_neighbor_list: wp.bool,
    atomic_energies: wp.array(dtype=Any),
    atomic_forces: wp.array(dtype=Any),
    virial: wp.array(dtype=Any),
):
    r"""Compute Lennard-Jones energies, forces, and virial (CSR neighbor list).

    Energy/force handling matches `_lj_energy_forces_list_kernel`.

    Virial is accumulated into a global 9-vector (flattened 3x3) using the
    negative sign convention:

    \[
      W = -\\sum r \\otimes F
    \]

    For full neighbor lists (`half_neighbor_list=False`), virial contributions are
    scaled by 0.5 per edge to avoid double-counting.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atom positions.
    cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Flattened neighbor indices.
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Integer lattice shifts.
    epsilon, sigma, cutoff, switch_width : wp.array, shape (1,), dtype=float32/float64
        Scalar LJ parameters packed as 1-element arrays.
    half_neighbor_list : wp.bool
        True for half neighbor list; False for full neighbor list.
    atomic_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: per-atom energies.
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: per-atom forces.
    virial : wp.array, shape (9,), dtype=wp.float64
        OUTPUT: global virial tensor, flattened as `[xx, xy, xz, yx, yy, yz, zx, zy, zz]`.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]

    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    cell_t = wp.transpose(cell[0])
    eps = wp.float64(epsilon[0])
    sig = wp.float64(sigma[0])
    cut = wp.float64(cutoff[0])
    sw = wp.float64(switch_width[0])
    cutoff_sq = cut * cut
    r_on, r_cut = _switch_params(cut, sw)

    force_acc = type(ri)(
        type(ri[0])(0.0),
        type(ri[0])(0.0),
        type(ri[0])(0.0),
    )

    vir_xx = wp.float64(0.0)
    vir_xy = wp.float64(0.0)
    vir_xz = wp.float64(0.0)
    vir_yx = wp.float64(0.0)
    vir_yy = wp.float64(0.0)
    vir_yz = wp.float64(0.0)
    vir_zx = wp.float64(0.0)
    vir_zy = wp.float64(0.0)
    vir_zz = wp.float64(0.0)

    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]
        rj = positions[j]

        shift = unit_shifts[edge_idx]
        shift_vec = cell_t * type(ri)(
            type(ri[0])(shift[0]),
            type(ri[0])(shift[1]),
            type(ri[0])(shift[2]),
        )

        r_ij = ri - rj - shift_vec
        r_sq = wp.float64(wp.dot(r_ij, r_ij))

        if r_sq >= cutoff_sq or r_sq < wp.float64(1e-10):
            continue

        r = wp.sqrt(r_sq)
        sigma_over_r = sig / r

        pair_energy_raw = _lj_energy_pair(sigma_over_r, eps)
        force_mag_over_r_raw = _lj_force_over_r(sigma_over_r, eps, r_sq)

        if sw > wp.float64(0.0) and r > r_on:
            s, ds_dr = switch_c2(r, r_on, r_cut)
            pair_energy = s * pair_energy_raw
            force_mag_over_r = s * force_mag_over_r_raw + (-pair_energy_raw * ds_dr) / r
        else:
            pair_energy = pair_energy_raw
            force_mag_over_r = force_mag_over_r_raw

        # Cast from float64 accumulator back to output dtype
        half_energy = wp.float64(0.5) * pair_energy
        wp.atomic_add(atomic_energies, atom_i, type(atomic_energies[0])(half_energy))
        if half_neighbor_list:
            wp.atomic_add(atomic_energies, j, type(atomic_energies[0])(half_energy))
        force_ij = type(ri)(
            type(ri[0])(force_mag_over_r) * r_ij[0],
            type(ri[0])(force_mag_over_r) * r_ij[1],
            type(ri[0])(force_mag_over_r) * r_ij[2],
        )

        force_acc += force_ij
        if half_neighbor_list:
            wp.atomic_sub(atomic_forces, j, force_ij)

        # Virial
        r_ij_0 = wp.float64(r_ij[0])
        r_ij_1 = wp.float64(r_ij[1])
        r_ij_2 = wp.float64(r_ij[2])
        f_ij_0 = wp.float64(force_ij[0])
        f_ij_1 = wp.float64(force_ij[1])
        f_ij_2 = wp.float64(force_ij[2])

        vir_scale = wp.float64(1.0) if half_neighbor_list else wp.float64(0.5)
        vir_xx += vir_scale * (r_ij_0 * f_ij_0)
        vir_xy += vir_scale * (r_ij_0 * f_ij_1)
        vir_xz += vir_scale * (r_ij_0 * f_ij_2)
        vir_yx += vir_scale * (r_ij_1 * f_ij_0)
        vir_yy += vir_scale * (r_ij_1 * f_ij_1)
        vir_yz += vir_scale * (r_ij_1 * f_ij_2)
        vir_zx += vir_scale * (r_ij_2 * f_ij_0)
        vir_zy += vir_scale * (r_ij_2 * f_ij_1)
        vir_zz += vir_scale * (r_ij_2 * f_ij_2)
    wp.atomic_add(atomic_forces, atom_i, force_acc)

    # Cast from float64 accumulator back to output dtype
    wp.atomic_sub(virial, 0, type(virial[0])(vir_xx))
    wp.atomic_sub(virial, 1, type(virial[0])(vir_xy))
    wp.atomic_sub(virial, 2, type(virial[0])(vir_xz))
    wp.atomic_sub(virial, 3, type(virial[0])(vir_yx))
    wp.atomic_sub(virial, 4, type(virial[0])(vir_yy))
    wp.atomic_sub(virial, 5, type(virial[0])(vir_yz))
    wp.atomic_sub(virial, 6, type(virial[0])(vir_zx))
    wp.atomic_sub(virial, 7, type(virial[0])(vir_zy))
    wp.atomic_sub(virial, 8, type(virial[0])(vir_zz))


# ==============================================================================
# Warp Kernels - Batched (Neighbor Matrix)
# ==============================================================================


@wp.kernel
def _batch_lj_energy_forces_matrix_kernel(
    positions: wp.array(dtype=Any),
    cells: wp.array(dtype=Any),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    batch_idx: wp.array(dtype=wp.int32),
    epsilon: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    switch_width: wp.array(dtype=Any),
    half_neighbor_list: wp.bool,
    fill_value: wp.int32,
    atomic_energies: wp.array(dtype=Any),
    atomic_forces: wp.array(dtype=Any),
):
    """Compute Lennard-Jones energies and forces for batched systems (neighbor matrix).

    This is the batched analogue of `_lj_energy_forces_matrix_kernel`, where each
    atom belongs to a system `system_id = batch_idx[atom_i]` and periodic shifts are
    converted using `cells[system_id]`.

    Launch Grid
    -----------
    dim = [num_atoms_total]

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Concatenated atom positions across all systems.
    cells : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Unit cell per system.
    neighbor_matrix : wp.array2d, shape (N_total, max_neighbors), dtype=wp.int32
        Neighbor indices for each atom.
    neighbor_matrix_shifts : wp.array2d, shape (N_total, max_neighbors), dtype=wp.vec3i
        Integer lattice shifts for each neighbor entry.
    num_neighbors : wp.array, shape (N_total,), dtype=wp.int32
        Neighbor count per atom.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System id for each atom (0..B-1).
    epsilon, sigma, cutoff, switch_width : wp.array, shape (1,), dtype=float32/float64
        Scalar LJ parameters packed as 1-element arrays (shared across systems).
    half_neighbor_list : wp.bool
        True for half neighbor list; False for full neighbor list.
    fill_value : wp.int32
        Padding sentinel for `neighbor_matrix`.
    atomic_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: per-atom energies.
    atomic_forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: per-atom forces.

    Notes
    -----
    - Energies are float64; forces match the positions dtype.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_i >= num_atoms:
        return

    system_id = batch_idx[atom_i]
    ri = positions[atom_i]
    cell_t = wp.transpose(cells[system_id])
    eps = wp.float64(epsilon[0])
    sig = wp.float64(sigma[0])
    cut = wp.float64(cutoff[0])
    sw = wp.float64(switch_width[0])
    cutoff_sq = cut * cut
    r_on, r_cut = _switch_params(cut, sw)

    force_acc = type(ri)(
        type(ri[0])(0.0),
        type(ri[0])(0.0),
        type(ri[0])(0.0),
    )

    n_neighbors = num_neighbors[atom_i]

    for neighbor_slot in range(n_neighbors):
        if neighbor_slot >= max_neighbors:
            break

        j = neighbor_matrix[atom_i, neighbor_slot]
        if j >= fill_value or j >= num_atoms:
            continue

        rj = positions[j]

        shift = neighbor_matrix_shifts[atom_i, neighbor_slot]
        shift_vec = cell_t * type(ri)(
            type(ri[0])(shift[0]),
            type(ri[0])(shift[1]),
            type(ri[0])(shift[2]),
        )

        r_ij = ri - rj - shift_vec
        r_sq = wp.float64(wp.dot(r_ij, r_ij))

        if r_sq >= cutoff_sq or r_sq < wp.float64(1e-10):
            continue

        r = wp.sqrt(r_sq)
        sigma_over_r = sig / r

        pair_energy_raw = _lj_energy_pair(sigma_over_r, eps)
        force_mag_over_r_raw = _lj_force_over_r(sigma_over_r, eps, r_sq)

        if sw > wp.float64(0.0) and r > r_on:
            s, ds_dr = switch_c2(r, r_on, r_cut)
            pair_energy = s * pair_energy_raw
            force_mag_over_r = s * force_mag_over_r_raw + (-pair_energy_raw * ds_dr) / r
        else:
            pair_energy = pair_energy_raw
            force_mag_over_r = force_mag_over_r_raw

        # Cast from float64 accumulator back to output dtype
        half_energy = wp.float64(0.5) * pair_energy
        wp.atomic_add(atomic_energies, atom_i, type(atomic_energies[0])(half_energy))
        if half_neighbor_list:
            wp.atomic_add(atomic_energies, j, type(atomic_energies[0])(half_energy))
        force_ij = type(ri)(
            type(ri[0])(force_mag_over_r) * r_ij[0],
            type(ri[0])(force_mag_over_r) * r_ij[1],
            type(ri[0])(force_mag_over_r) * r_ij[2],
        )

        force_acc += force_ij
        if half_neighbor_list:
            wp.atomic_sub(atomic_forces, j, force_ij)
    wp.atomic_add(atomic_forces, atom_i, force_acc)


@wp.kernel
def _batch_lj_energy_forces_virial_matrix_kernel(
    positions: wp.array(dtype=Any),
    cells: wp.array(dtype=Any),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    batch_idx: wp.array(dtype=wp.int32),
    epsilon: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    switch_width: wp.array(dtype=Any),
    half_neighbor_list: wp.bool,
    fill_value: wp.int32,
    atomic_energies: wp.array(dtype=Any),
    atomic_forces: wp.array(dtype=Any),
    virial: wp.array2d(dtype=Any),
):
    """Compute Lennard-Jones energies, forces, and virial for batched systems.

    Batched analogue of `_lj_energy_forces_virial_matrix_kernel`.

    Virial is accumulated per-system into `virial[system_id, :]` with shape (B, 9)
    using the flattened layout `[xx, xy, xz, yx, yy, yz, zx, zy, zz]`.

    Launch Grid
    -----------
    dim = [num_atoms_total]

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Concatenated atom positions.
    cells : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Unit cell per system.
    neighbor_matrix, neighbor_matrix_shifts, num_neighbors : arrays
        Neighbor matrix data for all atoms (see `_batch_lj_energy_forces_matrix_kernel`).
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System id per atom.
    epsilon, sigma, cutoff, switch_width : wp.array, shape (1,), dtype=float32/float64
        Scalar LJ parameters packed as 1-element arrays.
    half_neighbor_list : wp.bool
        True for half neighbor list; False for full neighbor list.
    fill_value : wp.int32
        Padding sentinel.
    atomic_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: per-atom energies.
    atomic_forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: per-atom forces.
    virial : wp.array2d, shape (B, 9), dtype=wp.float64
        OUTPUT: per-system virial tensor.

    Notes
    -----
    - For full neighbor matrices, virial contributions are scaled by 0.5 per edge.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_i >= num_atoms:
        return

    system_id = batch_idx[atom_i]
    ri = positions[atom_i]
    cell_t = wp.transpose(cells[system_id])
    eps = wp.float64(epsilon[0])
    sig = wp.float64(sigma[0])
    cut = wp.float64(cutoff[0])
    sw = wp.float64(switch_width[0])
    cutoff_sq = cut * cut
    r_on, r_cut = _switch_params(cut, sw)

    force_acc = type(ri)(
        type(ri[0])(0.0),
        type(ri[0])(0.0),
        type(ri[0])(0.0),
    )

    vir_xx = wp.float64(0.0)
    vir_xy = wp.float64(0.0)
    vir_xz = wp.float64(0.0)
    vir_yx = wp.float64(0.0)
    vir_yy = wp.float64(0.0)
    vir_yz = wp.float64(0.0)
    vir_zx = wp.float64(0.0)
    vir_zy = wp.float64(0.0)
    vir_zz = wp.float64(0.0)

    n_neighbors = num_neighbors[atom_i]

    for neighbor_slot in range(n_neighbors):
        if neighbor_slot >= max_neighbors:
            break

        j = neighbor_matrix[atom_i, neighbor_slot]
        if j >= fill_value or j >= num_atoms:
            continue

        rj = positions[j]

        shift = neighbor_matrix_shifts[atom_i, neighbor_slot]
        shift_vec = cell_t * type(ri)(
            type(ri[0])(shift[0]),
            type(ri[0])(shift[1]),
            type(ri[0])(shift[2]),
        )

        r_ij = ri - rj - shift_vec
        r_sq = wp.float64(wp.dot(r_ij, r_ij))

        if r_sq >= cutoff_sq or r_sq < wp.float64(1e-10):
            continue

        r = wp.sqrt(r_sq)
        sigma_over_r = sig / r

        pair_energy_raw = _lj_energy_pair(sigma_over_r, eps)
        force_mag_over_r_raw = _lj_force_over_r(sigma_over_r, eps, r_sq)

        if sw > wp.float64(0.0) and r > r_on:
            s, ds_dr = switch_c2(r, r_on, r_cut)
            pair_energy = s * pair_energy_raw
            force_mag_over_r = s * force_mag_over_r_raw + (-pair_energy_raw * ds_dr) / r
        else:
            pair_energy = pair_energy_raw
            force_mag_over_r = force_mag_over_r_raw

        # Cast from float64 accumulator back to output dtype
        half_energy = wp.float64(0.5) * pair_energy
        wp.atomic_add(atomic_energies, atom_i, type(atomic_energies[0])(half_energy))
        if half_neighbor_list:
            wp.atomic_add(atomic_energies, j, type(atomic_energies[0])(half_energy))
        force_ij = type(ri)(
            type(ri[0])(force_mag_over_r) * r_ij[0],
            type(ri[0])(force_mag_over_r) * r_ij[1],
            type(ri[0])(force_mag_over_r) * r_ij[2],
        )

        force_acc += force_ij
        if half_neighbor_list:
            wp.atomic_sub(atomic_forces, j, force_ij)

        r_ij_0 = wp.float64(r_ij[0])
        r_ij_1 = wp.float64(r_ij[1])
        r_ij_2 = wp.float64(r_ij[2])
        f_ij_0 = wp.float64(force_ij[0])
        f_ij_1 = wp.float64(force_ij[1])
        f_ij_2 = wp.float64(force_ij[2])

        vir_scale = wp.float64(1.0) if half_neighbor_list else wp.float64(0.5)
        vir_xx += vir_scale * (r_ij_0 * f_ij_0)
        vir_xy += vir_scale * (r_ij_0 * f_ij_1)
        vir_xz += vir_scale * (r_ij_0 * f_ij_2)
        vir_yx += vir_scale * (r_ij_1 * f_ij_0)
        vir_yy += vir_scale * (r_ij_1 * f_ij_1)
        vir_yz += vir_scale * (r_ij_1 * f_ij_2)
        vir_zx += vir_scale * (r_ij_2 * f_ij_0)
        vir_zy += vir_scale * (r_ij_2 * f_ij_1)
        vir_zz += vir_scale * (r_ij_2 * f_ij_2)
    wp.atomic_add(atomic_forces, atom_i, force_acc)

    # Cast from float64 accumulator back to output dtype
    wp.atomic_sub(virial, system_id, 0, type(virial[0][0])(vir_xx))
    wp.atomic_sub(virial, system_id, 1, type(virial[0][0])(vir_xy))
    wp.atomic_sub(virial, system_id, 2, type(virial[0][0])(vir_xz))
    wp.atomic_sub(virial, system_id, 3, type(virial[0][0])(vir_yx))
    wp.atomic_sub(virial, system_id, 4, type(virial[0][0])(vir_yy))
    wp.atomic_sub(virial, system_id, 5, type(virial[0][0])(vir_yz))
    wp.atomic_sub(virial, system_id, 6, type(virial[0][0])(vir_zx))
    wp.atomic_sub(virial, system_id, 7, type(virial[0][0])(vir_zy))
    wp.atomic_sub(virial, system_id, 8, type(virial[0][0])(vir_zz))


# ==============================================================================
# Kernel Overloads (float32/float64)
# ==============================================================================

_T = [wp.float32, wp.float64]
_V = [wp.vec3f, wp.vec3d]
_M = [wp.mat33f, wp.mat33d]

# Overload dictionaries
_lj_energy_matrix_kernel_overload = {}
_lj_energy_forces_matrix_kernel_overload = {}
_lj_energy_forces_virial_matrix_kernel_overload = {}
_lj_energy_list_kernel_overload = {}
_lj_energy_forces_list_kernel_overload = {}
_lj_energy_forces_virial_list_kernel_overload = {}
_batch_lj_energy_forces_matrix_kernel_overload = {}
_batch_lj_energy_forces_virial_matrix_kernel_overload = {}

for t, v, m in zip(_T, _V, _M):
    # Neighbor matrix kernels
    _lj_energy_matrix_kernel_overload[t] = wp.overload(
        _lj_energy_matrix_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=m),  # cell
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # neighbor_matrix_shifts
            wp.array(dtype=wp.int32),  # num_neighbors
            wp.array(dtype=t),  # epsilon
            wp.array(dtype=t),  # sigma
            wp.array(dtype=t),  # cutoff
            wp.array(dtype=t),  # switch_width
            wp.bool,  # half_neighbor_list
            wp.int32,  # fill_value
            wp.array(dtype=t),  # atomic_energies (matches input dtype)
        ],
    )

    _lj_energy_forces_matrix_kernel_overload[t] = wp.overload(
        _lj_energy_forces_matrix_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=m),  # cell
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # neighbor_matrix_shifts
            wp.array(dtype=wp.int32),  # num_neighbors
            wp.array(dtype=t),  # epsilon
            wp.array(dtype=t),  # sigma
            wp.array(dtype=t),  # cutoff
            wp.array(dtype=t),  # switch_width
            wp.bool,  # half_neighbor_list
            wp.int32,  # fill_value
            wp.array(dtype=t),  # atomic_energies (matches input dtype)
            wp.array(dtype=v),  # atomic_forces
        ],
    )

    _lj_energy_forces_virial_matrix_kernel_overload[t] = wp.overload(
        _lj_energy_forces_virial_matrix_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=m),  # cell
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # neighbor_matrix_shifts
            wp.array(dtype=wp.int32),  # num_neighbors
            wp.array(dtype=t),  # epsilon
            wp.array(dtype=t),  # sigma
            wp.array(dtype=t),  # cutoff
            wp.array(dtype=t),  # switch_width
            wp.bool,  # half_neighbor_list
            wp.int32,  # fill_value
            wp.array(dtype=t),  # atomic_energies (matches input dtype)
            wp.array(dtype=v),  # atomic_forces
            wp.array(dtype=t),  # virial (matches input dtype)
        ],
    )

    # Neighbor list (CSR) kernels
    _lj_energy_list_kernel_overload[t] = wp.overload(
        _lj_energy_list_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.int32),  # idx_j
            wp.array(dtype=wp.int32),  # neighbor_ptr
            wp.array(dtype=wp.vec3i),  # unit_shifts
            wp.array(dtype=t),  # epsilon
            wp.array(dtype=t),  # sigma
            wp.array(dtype=t),  # cutoff
            wp.array(dtype=t),  # switch_width
            wp.bool,  # half_neighbor_list
            wp.array(dtype=t),  # atomic_energies (matches input dtype)
        ],
    )

    _lj_energy_forces_list_kernel_overload[t] = wp.overload(
        _lj_energy_forces_list_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.int32),  # idx_j
            wp.array(dtype=wp.int32),  # neighbor_ptr
            wp.array(dtype=wp.vec3i),  # unit_shifts
            wp.array(dtype=t),  # epsilon
            wp.array(dtype=t),  # sigma
            wp.array(dtype=t),  # cutoff
            wp.array(dtype=t),  # switch_width
            wp.bool,  # half_neighbor_list
            wp.array(dtype=t),  # atomic_energies (matches input dtype)
            wp.array(dtype=v),  # atomic_forces
        ],
    )

    _lj_energy_forces_virial_list_kernel_overload[t] = wp.overload(
        _lj_energy_forces_virial_list_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.int32),  # idx_j
            wp.array(dtype=wp.int32),  # neighbor_ptr
            wp.array(dtype=wp.vec3i),  # unit_shifts
            wp.array(dtype=t),  # epsilon
            wp.array(dtype=t),  # sigma
            wp.array(dtype=t),  # cutoff
            wp.array(dtype=t),  # switch_width
            wp.bool,  # half_neighbor_list
            wp.array(dtype=t),  # atomic_energies (matches input dtype)
            wp.array(dtype=v),  # atomic_forces
            wp.array(dtype=t),  # virial (matches input dtype)
        ],
    )

    # Batched kernels
    _batch_lj_energy_forces_matrix_kernel_overload[t] = wp.overload(
        _batch_lj_energy_forces_matrix_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=m),  # cells
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # neighbor_matrix_shifts
            wp.array(dtype=wp.int32),  # num_neighbors
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # epsilon
            wp.array(dtype=t),  # sigma
            wp.array(dtype=t),  # cutoff
            wp.array(dtype=t),  # switch_width
            wp.bool,  # half_neighbor_list
            wp.int32,  # fill_value
            wp.array(dtype=t),  # atomic_energies (matches input dtype)
            wp.array(dtype=v),  # atomic_forces
        ],
    )

    _batch_lj_energy_forces_virial_matrix_kernel_overload[t] = wp.overload(
        _batch_lj_energy_forces_virial_matrix_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=m),  # cells
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # neighbor_matrix_shifts
            wp.array(dtype=wp.int32),  # num_neighbors
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # epsilon
            wp.array(dtype=t),  # sigma
            wp.array(dtype=t),  # cutoff
            wp.array(dtype=t),  # switch_width
            wp.bool,  # half_neighbor_list
            wp.int32,  # fill_value
            wp.array(dtype=t),  # atomic_energies (matches input dtype)
            wp.array(dtype=v),  # atomic_forces
            wp.array2d(dtype=t),  # virial (matches input dtype)
        ],
    )


# ==============================================================================
# Public API
# ==============================================================================


def lj_energy(
    positions: wp.array,
    cell: wp.array,
    epsilon: float,
    sigma: float,
    cutoff: float,
    neighbor_matrix: wp.array | None = None,
    neighbor_matrix_shifts: wp.array | None = None,
    num_neighbors: wp.array | None = None,
    fill_value: int | None = None,
    neighbor_list: wp.array | None = None,
    neighbor_ptr: wp.array | None = None,
    neighbor_shifts: wp.array | None = None,
    switch_width: float = 0.0,
    half_neighbor_list: bool = True,
    device: str | None = None,
) -> wp.array:
    """Compute Lennard-Jones energies.

    Parameters
    ----------
    positions : wp.array, dtype=wp.vec3f or wp.vec3d
        Atomic coordinates. Shape (N,).
    cell : wp.array, dtype=wp.mat33f or wp.mat33d
        Unit cell matrix. Shape (1,).
    epsilon : float
        LJ energy parameter (well depth).
    sigma : float
        LJ length parameter (zero-crossing distance).
    cutoff : float
        Cutoff distance for interactions.
    neighbor_matrix : wp.array | None, shape (N, max_neighbors)
        Neighbor indices in matrix format.
    neighbor_matrix_shifts : wp.array | None, shape (N, max_neighbors)
        Periodic shifts for neighbor matrix format.
    num_neighbors : wp.array | None, shape (N,)
        Number of neighbors per atom (for matrix format).
    fill_value : int | None
        Fill value for neighbor matrix padding.
    neighbor_list : wp.array | None, shape (2, num_pairs)
        Neighbor pairs in COO format (alternative to matrix).
    neighbor_ptr : wp.array | None, shape (N+1,)
        CSR row pointers for neighbor list format.
    neighbor_shifts : wp.array | None, shape (num_pairs,)
        Periodic shifts for neighbor list format.
    device : str | None
        Warp device. If None, inferred from positions.

    Returns
    -------
    energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Per-atom energies (matches input positions dtype). Sum to get total energy.
    """
    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]
    is_f32 = positions.dtype == wp.vec3f
    scalar_dtype = wp.float32 if is_f32 else wp.float64

    # Check which format is provided
    use_matrix = neighbor_matrix is not None
    use_list = neighbor_list is not None

    if not use_matrix and not use_list:
        raise ValueError("Must provide either neighbor_matrix or neighbor_list")

    # Allocate output (matches input dtype for flexibility)
    energies = wp.zeros(num_atoms, dtype=scalar_dtype, device=device)

    # Wrap scalar parameters as arrays
    wp_epsilon = wp.array([epsilon], dtype=scalar_dtype, device=device)
    wp_sigma = wp.array([sigma], dtype=scalar_dtype, device=device)
    wp_cutoff = wp.array([cutoff], dtype=scalar_dtype, device=device)
    wp_switch_width = wp.array([switch_width], dtype=scalar_dtype, device=device)
    wp_half = wp.bool(half_neighbor_list)

    if use_matrix:
        if fill_value is None:
            fill_value = num_atoms

        wp.launch(
            _lj_energy_matrix_kernel_overload[scalar_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                cell,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                wp_epsilon,
                wp_sigma,
                wp_cutoff,
                wp_switch_width,
                wp_half,
                wp.int32(fill_value),
                energies,
            ],
            device=device,
        )
    else:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr required for neighbor_list format")
        idx_j = (
            neighbor_list[1].contiguous() if neighbor_list.ndim == 2 else neighbor_list
        )

        wp.launch(
            _lj_energy_list_kernel_overload[scalar_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                cell,
                idx_j,
                neighbor_ptr,
                neighbor_shifts,
                wp_epsilon,
                wp_sigma,
                wp_cutoff,
                wp_switch_width,
                wp_half,
                energies,
            ],
            device=device,
        )

    return energies


def lj_forces(
    positions: wp.array,
    cell: wp.array,
    epsilon: float,
    sigma: float,
    cutoff: float,
    neighbor_matrix: wp.array | None = None,
    neighbor_matrix_shifts: wp.array | None = None,
    num_neighbors: wp.array | None = None,
    fill_value: int | None = None,
    neighbor_list: wp.array | None = None,
    neighbor_ptr: wp.array | None = None,
    neighbor_shifts: wp.array | None = None,
    switch_width: float = 0.0,
    half_neighbor_list: bool = True,
    device: str | None = None,
) -> wp.array:
    """Compute Lennard-Jones forces.

    Convenience wrapper that returns only forces (no energies).
    See lj_energy_forces for parameter descriptions.

    Returns
    -------
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Forces on each atom (matches positions dtype).
    """
    _, forces = lj_energy_forces(
        positions=positions,
        cell=cell,
        epsilon=epsilon,
        sigma=sigma,
        cutoff=cutoff,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        num_neighbors=num_neighbors,
        fill_value=fill_value,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        switch_width=switch_width,
        half_neighbor_list=half_neighbor_list,
        device=device,
    )
    return forces


def lj_energy_forces(
    positions: wp.array,
    cell: wp.array,
    epsilon: float,
    sigma: float,
    cutoff: float,
    neighbor_matrix: wp.array | None = None,
    neighbor_matrix_shifts: wp.array | None = None,
    num_neighbors: wp.array | None = None,
    fill_value: int | None = None,
    neighbor_list: wp.array | None = None,
    neighbor_ptr: wp.array | None = None,
    neighbor_shifts: wp.array | None = None,
    batch_idx: wp.array | None = None,
    switch_width: float = 0.0,
    half_neighbor_list: bool = True,
    device: str | None = None,
) -> tuple[wp.array, wp.array]:
    """Compute Lennard-Jones energies and forces.

    Parameters
    ----------
    positions : wp.array, dtype=wp.vec3f or wp.vec3d
        Atomic coordinates. Shape (N,).
    cell : wp.array, dtype=wp.mat33f or wp.mat33d
        Unit cell matrix. Shape (1,) or (B,) for batched.
    epsilon : float
        LJ energy parameter (well depth).
    sigma : float
        LJ length parameter (zero-crossing distance).
    cutoff : float
        Cutoff distance for interactions.
    neighbor_matrix : wp.array | None, shape (N, max_neighbors)
        Neighbor indices in matrix format.
    neighbor_matrix_shifts : wp.array | None, shape (N, max_neighbors)
        Periodic shifts for neighbor matrix format.
    num_neighbors : wp.array | None, shape (N,)
        Number of neighbors per atom (for matrix format).
    fill_value : int | None
        Fill value for neighbor matrix padding.
    neighbor_list : wp.array | None, shape (2, num_pairs)
        Neighbor pairs in COO format (alternative to matrix).
    neighbor_ptr : wp.array | None, shape (N+1,)
        CSR row pointers for neighbor list format.
    neighbor_shifts : wp.array | None, shape (num_pairs,)
        Periodic shifts for neighbor list format.
    batch_idx : wp.array | None, shape (N,)
        Batch indices for each atom. Required for batched mode.
    device : str | None
        Warp device. If None, inferred from positions.

    Returns
    -------
    energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Per-atom energies (matches input positions dtype).
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Forces on each atom.
    """
    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]
    is_f32 = positions.dtype == wp.vec3f
    scalar_dtype = wp.float32 if is_f32 else wp.float64
    vec_dtype = wp.vec3f if is_f32 else wp.vec3d

    use_matrix = neighbor_matrix is not None
    use_list = neighbor_list is not None
    is_batched = batch_idx is not None

    if not use_matrix and not use_list:
        raise ValueError("Must provide either neighbor_matrix or neighbor_list")

    # Allocate outputs (matches input dtype for flexibility)
    energies = wp.zeros(num_atoms, dtype=scalar_dtype, device=device)
    forces = wp.zeros(num_atoms, dtype=vec_dtype, device=device)

    # Wrap scalar parameters
    wp_epsilon = wp.array([epsilon], dtype=scalar_dtype, device=device)
    wp_sigma = wp.array([sigma], dtype=scalar_dtype, device=device)
    wp_cutoff = wp.array([cutoff], dtype=scalar_dtype, device=device)
    wp_switch_width = wp.array([switch_width], dtype=scalar_dtype, device=device)
    wp_half = wp.bool(half_neighbor_list)

    if use_matrix:
        if fill_value is None:
            fill_value = num_atoms

        if is_batched:
            wp.launch(
                _batch_lj_energy_forces_matrix_kernel_overload[scalar_dtype],
                dim=num_atoms,
                inputs=[
                    positions,
                    cell,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    batch_idx,
                    wp_epsilon,
                    wp_sigma,
                    wp_cutoff,
                    wp_switch_width,
                    wp_half,
                    wp.int32(fill_value),
                    energies,
                    forces,
                ],
                device=device,
            )
        else:
            wp.launch(
                _lj_energy_forces_matrix_kernel_overload[scalar_dtype],
                dim=num_atoms,
                inputs=[
                    positions,
                    cell,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    wp_epsilon,
                    wp_sigma,
                    wp_cutoff,
                    wp_switch_width,
                    wp_half,
                    wp.int32(fill_value),
                    energies,
                    forces,
                ],
                device=device,
            )
    else:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr required for neighbor_list format")
        idx_j = (
            neighbor_list[1].contiguous() if neighbor_list.ndim == 2 else neighbor_list
        )

        wp.launch(
            _lj_energy_forces_list_kernel_overload[scalar_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                cell,
                idx_j,
                neighbor_ptr,
                neighbor_shifts,
                wp_epsilon,
                wp_sigma,
                wp_cutoff,
                wp_switch_width,
                wp_half,
                energies,
                forces,
            ],
            device=device,
        )

    return energies, forces


def lj_energy_forces_virial(
    positions: wp.array,
    cell: wp.array,
    epsilon: float,
    sigma: float,
    cutoff: float,
    neighbor_matrix: wp.array | None = None,
    neighbor_matrix_shifts: wp.array | None = None,
    num_neighbors: wp.array | None = None,
    fill_value: int | None = None,
    neighbor_list: wp.array | None = None,
    neighbor_ptr: wp.array | None = None,
    neighbor_shifts: wp.array | None = None,
    batch_idx: wp.array | None = None,
    switch_width: float = 0.0,
    half_neighbor_list: bool = True,
    device: str | None = None,
) -> tuple[wp.array, wp.array, wp.array]:
    """Compute Lennard-Jones energies, forces, and virial tensor.

    The virial tensor is needed for pressure/stress calculations in NPT/NPH.

    Parameters
    ----------
    positions : wp.array, dtype=wp.vec3f or wp.vec3d
        Atomic coordinates. Shape (N,).
    cell : wp.array, dtype=wp.mat33f or wp.mat33d
        Unit cell matrix. Shape (1,) or (B,) for batched.
    epsilon : float
        LJ energy parameter (well depth).
    sigma : float
        LJ length parameter (zero-crossing distance).
    cutoff : float
        Cutoff distance for interactions.
    neighbor_matrix : wp.array | None, shape (N, max_neighbors)
        Neighbor indices in matrix format.
    neighbor_matrix_shifts : wp.array | None, shape (N, max_neighbors)
        Periodic shifts for neighbor matrix format.
    num_neighbors : wp.array | None, shape (N,)
        Number of neighbors per atom (for matrix format).
    fill_value : int | None
        Fill value for neighbor matrix padding.
    neighbor_list : wp.array | None, shape (2, num_pairs)
        Neighbor pairs in COO format (alternative to matrix).
    neighbor_ptr : wp.array | None, shape (N+1,)
        CSR row pointers for neighbor list format.
    neighbor_shifts : wp.array | None, shape (num_pairs,)
        Periodic shifts for neighbor list format.
    batch_idx : wp.array | None, shape (N,)
        Batch indices for each atom. Required for batched mode.
    num_systems : int
        Number of systems (for batched virial allocation).
    device : str | None
        Warp device. If None, inferred from positions.

    Returns
    -------
    energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Per-atom energies (matches input positions dtype).
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Forces on each atom.
    virial : wp.array, shape (9,) or (B, 9), dtype=wp.float32 or wp.float64
        Virial tensor [xx, xy, xz, yx, yy, yz, zx, zy, zz] (matches input dtype).
        For batched, shape is (num_systems, 9).
    """
    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]
    is_f32 = positions.dtype == wp.vec3f
    scalar_dtype = wp.float32 if is_f32 else wp.float64
    vec_dtype = wp.vec3f if is_f32 else wp.vec3d

    use_matrix = neighbor_matrix is not None
    use_list = neighbor_list is not None
    is_batched = batch_idx is not None

    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
    num_systems = cell.shape[0]

    if not use_matrix and not use_list:
        raise ValueError("Must provide either neighbor_matrix or neighbor_list")

    # Allocate outputs (matches input dtype for flexibility)
    energies = wp.zeros(num_atoms, dtype=scalar_dtype, device=device)
    forces = wp.zeros(num_atoms, dtype=vec_dtype, device=device)

    if is_batched:
        virial = wp.zeros((num_systems, 9), dtype=scalar_dtype, device=device)
    else:
        virial = wp.zeros(9, dtype=scalar_dtype, device=device)

    # Wrap scalar parameters
    wp_epsilon = wp.array([epsilon], dtype=scalar_dtype, device=device)
    wp_sigma = wp.array([sigma], dtype=scalar_dtype, device=device)
    wp_cutoff = wp.array([cutoff], dtype=scalar_dtype, device=device)
    wp_switch_width = wp.array([switch_width], dtype=scalar_dtype, device=device)
    wp_half = wp.bool(half_neighbor_list)

    if use_matrix:
        if fill_value is None:
            fill_value = num_atoms

        if is_batched:
            wp.launch(
                _batch_lj_energy_forces_virial_matrix_kernel_overload[scalar_dtype],
                dim=num_atoms,
                inputs=[
                    positions,
                    cell,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    batch_idx,
                    wp_epsilon,
                    wp_sigma,
                    wp_cutoff,
                    wp_switch_width,
                    wp_half,
                    wp.int32(fill_value),
                    energies,
                    forces,
                    virial,
                ],
                device=device,
            )
        else:
            wp.launch(
                _lj_energy_forces_virial_matrix_kernel_overload[scalar_dtype],
                dim=num_atoms,
                inputs=[
                    positions,
                    cell,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    wp_epsilon,
                    wp_sigma,
                    wp_cutoff,
                    wp_switch_width,
                    wp_half,
                    wp.int32(fill_value),
                    energies,
                    forces,
                    virial,
                ],
                device=device,
            )
    else:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr required for neighbor_list format")
        idx_j = (
            neighbor_list[1].contiguous() if neighbor_list.ndim == 2 else neighbor_list
        )

        wp.launch(
            _lj_energy_forces_virial_list_kernel_overload[scalar_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                cell,
                idx_j,
                neighbor_ptr,
                neighbor_shifts,
                wp_epsilon,
                wp_sigma,
                wp_cutoff,
                wp_switch_width,
                wp_half,
                energies,
                forces,
                virial,
            ],
            device=device,
        )

    return energies, forces, virial
