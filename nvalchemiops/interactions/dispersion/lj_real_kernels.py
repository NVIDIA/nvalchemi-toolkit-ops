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

r"""
Pairwise Dispersion (r^-6) Interactions - Warp Kernel Implementation
====================================================================

This module implements direct pairwise dispersion (van der Waals attractive
:math:`r^{-6}`) energy and force calculations using Warp GPU/CPU kernels. It
includes both the plain :math:`-C_6/r^6` interaction and the
:math:`\beta`-damped real-space term used by dispersion PME (LJ-PME).

It is the dispersion analog of
``nvalchemiops.interactions.electrostatics.coulomb``: the neighbor-list / matrix
machinery, batching, accumulation conventions, and launcher signatures are
identical; only the pair kernel math differs (Coulomb :math:`1/r` becomes
dispersion :math:`-C_6/r^6`).

Mathematical Formulation
------------------------

Dispersion uses the **geometric** combination rule so that the pair coefficient
is separable into a per-atom "dispersion charge"

.. math::

    b_i = \sqrt{C_{6,i}}, \qquad C_{6,ij} = \sqrt{C_{6,i} C_{6,j}} = b_i b_j .

The kernels therefore take ``b`` (the per-atom :math:`\sqrt{C_6}`) in exactly
the place Coulomb takes ``charges``. The conversion from per-atom
:math:`\sigma/\epsilon` to :math:`b_i = 2\sqrt{\epsilon_i}\,\sigma_i^3` is done
in the framework binding layer.

1. Plain dispersion (``alpha = 0``):

   .. math::

       E_{ij} = -\frac{C_{6,ij}}{r^6},
       \qquad
       \mathbf{F}_{i} = -\frac{6 C_{6,ij}}{r^8}\,\mathbf{r}_{ij}.

2. :math:`\beta`-damped real-space term (``alpha = beta > 0``), the short-range
   half of the :math:`r^{-6}` Ewald split (in 't Veld / Essmann LJ-PME):

   .. math::

       S(r) = \left(1 + \beta^2 r^2 + \tfrac12 \beta^4 r^4\right) e^{-\beta^2 r^2},
       \qquad
       E_{ij} = -\frac{C_{6,ij}\, S(r)}{r^6},

   .. math::

       \mathbf{F}_{i}
       = -C_{6,ij}\left[\frac{\beta^6 e^{-\beta^2 r^2}}{r^2}
         + \frac{6 S(r)}{r^8}\right]\mathbf{r}_{ij}.

   As :math:`\beta \to 0`, :math:`S \to 1` and both reduce to the plain forms.

.. note::
   As in ``coulomb.py``, a pair factor of ``0.5`` is applied per processed edge
   and per-atom energy is accumulated on ``atom_i`` while forces are split
   between ``i`` and ``j`` via Newton's third law. With a symmetric neighbor
   list (both ``(i, j)`` and ``(j, i)`` present) this yields the correct total
   energy and forces; a half list yields half the total energy. This matches
   the Coulomb convention exactly.

References
----------
- in 't Veld, Ismail & Grest, J. Chem. Phys. 127, 144711 (2007).
- Essmann et al., J. Chem. Phys. 103, 8577 (1995) - smooth PME.
- https://manual.gromacs.org/current/reference-manual/functions/long-range-vdw.html
"""

from __future__ import annotations

import warp as wp

__all__ = [
    # Warp launchers (framework-agnostic public API)
    "lj_dispersion_energy",
    "lj_dispersion_energy_forces",
    "lj_dispersion_energy_matrix",
    "lj_dispersion_energy_forces_matrix",
    "batch_lj_dispersion_energy",
    "batch_lj_dispersion_energy_forces",
    "batch_lj_dispersion_energy_matrix",
    "batch_lj_dispersion_energy_forces_matrix",
]


# ==============================================================================
# Shared pair functions
# ==============================================================================


@wp.func
def _disp_pair_energy(
    prefactor: wp.float64, r2: wp.float64, alpha: wp.float64
) -> wp.float64:
    """Per-edge dispersion energy contribution.

    Returns ``-prefactor * S(r) / r^6`` where ``prefactor = 0.5 * b_i * b_j`` and
    ``S(r) = (1 + x^2 + 0.5 x^4) exp(-x^2)`` with ``x = alpha * r`` (``S = 1`` for
    ``alpha = 0``).
    """
    inv_r2 = wp.float64(1.0) / r2
    inv_r6 = inv_r2 * inv_r2 * inv_r2

    if alpha > wp.float64(0.0):
        x2 = alpha * alpha * r2
        screen = (wp.float64(1.0) + x2 + wp.float64(0.5) * x2 * x2) * wp.exp(-x2)
        return -prefactor * screen * inv_r6

    return -prefactor * inv_r6


@wp.func
def _disp_pair_force_over_r(
    prefactor: wp.float64, r2: wp.float64, alpha: wp.float64
) -> wp.float64:
    """Scalar ``force_mag_over_r`` such that ``force_ij = force_mag_over_r * r_ij``.

    Encodes the gradient of :func:`_disp_pair_energy`. For ``alpha = 0`` this is
    ``-6 * prefactor / r^8``; for ``alpha > 0`` it adds the screened terms.
    """
    inv_r2 = wp.float64(1.0) / r2
    inv_r6 = inv_r2 * inv_r2 * inv_r2

    if alpha > wp.float64(0.0):
        b2 = alpha * alpha
        x2 = b2 * r2
        exp_term = wp.exp(-x2)
        screen = (wp.float64(1.0) + x2 + wp.float64(0.5) * x2 * x2) * exp_term
        beta6 = b2 * b2 * b2
        return (
            -prefactor * (beta6 * exp_term + wp.float64(6.0) * screen * inv_r6) * inv_r2
        )

    return -wp.float64(6.0) * prefactor * inv_r6 * inv_r2


# ==============================================================================
# Warp Kernels - Energy Only (Neighbor List Format)
# ==============================================================================


@wp.kernel
def _lj_dispersion_energy_kernel(
    positions: wp.array(dtype=wp.vec3d),
    b: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    cutoff: wp.float64,
    alpha: wp.float64,
    energies: wp.array(dtype=wp.float64),
):
    """Compute dispersion energies (plain or beta-damped) using CSR neighbor list."""
    atom_i = wp.tid()
    num_atoms = positions.shape[0]

    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    bi = b[atom_i]
    cell_t = wp.transpose(cell[0])

    energy_acc = wp.float64(0.0)

    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]

        rj = positions[j]
        bj = b[j]

        shift_vec = cell_t * type(ri)(unit_shifts[edge_idx])
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        prefactor = wp.float64(0.5) * bi * bj
        energy_acc += _disp_pair_energy(prefactor, r * r, alpha)

    wp.atomic_add(energies, atom_i, energy_acc)


@wp.kernel
def _lj_dispersion_energy_forces_kernel(
    positions: wp.array(dtype=wp.vec3d),
    b: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    cutoff: wp.float64,
    alpha: wp.float64,
    energies: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=wp.vec3d),
):
    """Compute dispersion energies and forces using CSR neighbor list."""
    atom_i = wp.tid()
    num_atoms = positions.shape[0]

    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    bi = b[atom_i]
    cell_t = wp.transpose(cell[0])

    energy_acc = wp.float64(0.0)
    force_acc = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]

        rj = positions[j]
        bj = b[j]

        shift_vec = cell_t * type(ri)(unit_shifts[edge_idx])
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        r2 = r * r
        prefactor = wp.float64(0.5) * bi * bj
        energy_acc += _disp_pair_energy(prefactor, r2, alpha)
        force_ij = _disp_pair_force_over_r(prefactor, r2, alpha) * r_ij

        force_acc += force_ij
        wp.atomic_add(forces, j, -force_ij)

    wp.atomic_add(energies, atom_i, energy_acc)
    wp.atomic_add(forces, atom_i, force_acc)


# ==============================================================================
# Warp Kernels - Neighbor Matrix Format
# ==============================================================================


@wp.kernel
def _lj_dispersion_energy_matrix_kernel(
    positions: wp.array(dtype=wp.vec3d),
    b: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    cutoff: wp.float64,
    alpha: wp.float64,
    fill_value: wp.int32,
    atomic_energies: wp.array(dtype=wp.float64),
):
    """Compute dispersion energies using neighbor matrix format."""
    atom_idx = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_idx >= num_atoms:
        return

    ri = positions[atom_idx]
    bi = b[atom_idx]
    cell_t = wp.transpose(cell[0])

    energy_acc = wp.float64(0.0)

    for neighbor_slot in range(max_neighbors):
        j = neighbor_matrix[atom_idx, neighbor_slot]
        if j >= fill_value or j >= num_atoms:
            continue

        rj = positions[j]
        bj = b[j]

        shift = neighbor_matrix_shifts[atom_idx, neighbor_slot]
        shift_vec = cell_t * type(ri)(shift)
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        prefactor = wp.float64(0.5) * bi * bj
        energy_acc += _disp_pair_energy(prefactor, r * r, alpha)

    wp.atomic_add(atomic_energies, atom_idx, energy_acc)


@wp.kernel
def _lj_dispersion_energy_forces_matrix_kernel(
    positions: wp.array(dtype=wp.vec3d),
    b: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    cutoff: wp.float64,
    alpha: wp.float64,
    fill_value: wp.int32,
    atomic_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=wp.vec3d),
):
    """Compute dispersion energies and forces using neighbor matrix format."""
    atom_idx = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_idx >= num_atoms:
        return

    ri = positions[atom_idx]
    bi = b[atom_idx]
    cell_t = wp.transpose(cell[0])

    energy_acc = wp.float64(0.0)
    force_acc = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

    for neighbor_slot in range(max_neighbors):
        j = neighbor_matrix[atom_idx, neighbor_slot]
        if j >= fill_value or j >= num_atoms:
            continue

        rj = positions[j]
        bj = b[j]

        shift = neighbor_matrix_shifts[atom_idx, neighbor_slot]
        shift_vec = cell_t * type(ri)(shift)
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        r2 = r * r
        prefactor = wp.float64(0.5) * bi * bj
        energy_acc += _disp_pair_energy(prefactor, r2, alpha)
        force_ij = _disp_pair_force_over_r(prefactor, r2, alpha) * r_ij

        force_acc += force_ij
        wp.atomic_add(atomic_forces, j, -force_ij)

    wp.atomic_add(atomic_energies, atom_idx, energy_acc)
    wp.atomic_add(atomic_forces, atom_idx, force_acc)


# ==============================================================================
# Warp Kernels - Batch Versions (Neighbor List Format)
# ==============================================================================


@wp.kernel
def _batch_lj_dispersion_energy_kernel(
    positions: wp.array(dtype=wp.vec3d),
    b: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    batch_idx: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    alpha: wp.float64,
    energies: wp.array(dtype=wp.float64),
):
    """Compute dispersion energies for batched systems using CSR neighbor list."""
    atom_i = wp.tid()
    num_atoms = positions.shape[0]

    if atom_i >= num_atoms:
        return

    system_id = batch_idx[atom_i]
    ri = positions[atom_i]
    bi = b[atom_i]
    cell_t = wp.transpose(cell[system_id])

    energy_acc = wp.float64(0.0)

    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]

        rj = positions[j]
        bj = b[j]

        shift_vec = cell_t * type(ri)(unit_shifts[edge_idx])
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        prefactor = wp.float64(0.5) * bi * bj
        energy_acc += _disp_pair_energy(prefactor, r * r, alpha)

    wp.atomic_add(energies, atom_i, energy_acc)


@wp.kernel
def _batch_lj_dispersion_energy_forces_kernel(
    positions: wp.array(dtype=wp.vec3d),
    b: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    batch_idx: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    alpha: wp.float64,
    energies: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=wp.vec3d),
):
    """Compute dispersion energies and forces for batched systems using CSR list."""
    atom_i = wp.tid()
    num_atoms = positions.shape[0]

    if atom_i >= num_atoms:
        return

    system_id = batch_idx[atom_i]
    ri = positions[atom_i]
    bi = b[atom_i]
    cell_t = wp.transpose(cell[system_id])

    energy_acc = wp.float64(0.0)
    force_acc = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]

        rj = positions[j]
        bj = b[j]

        shift_vec = cell_t * type(ri)(unit_shifts[edge_idx])
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        r2 = r * r
        prefactor = wp.float64(0.5) * bi * bj
        energy_acc += _disp_pair_energy(prefactor, r2, alpha)
        force_ij = _disp_pair_force_over_r(prefactor, r2, alpha) * r_ij

        force_acc += force_ij
        wp.atomic_add(forces, j, -force_ij)

    wp.atomic_add(energies, atom_i, energy_acc)
    wp.atomic_add(forces, atom_i, force_acc)


# ==============================================================================
# Warp Kernels - Batch Versions (Neighbor Matrix Format)
# ==============================================================================


@wp.kernel
def _batch_lj_dispersion_energy_matrix_kernel(
    positions: wp.array(dtype=wp.vec3d),
    b: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    batch_idx: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    alpha: wp.float64,
    fill_value: wp.int32,
    atomic_energies: wp.array(dtype=wp.float64),
):
    """Compute dispersion energies for batched systems using neighbor matrix."""
    atom_idx = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_idx >= num_atoms:
        return

    system_id = batch_idx[atom_idx]
    ri = positions[atom_idx]
    bi = b[atom_idx]
    cell_t = wp.transpose(cell[system_id])

    energy_acc = wp.float64(0.0)

    for neighbor_slot in range(max_neighbors):
        j = neighbor_matrix[atom_idx, neighbor_slot]
        if j >= fill_value or j >= num_atoms:
            continue

        rj = positions[j]
        bj = b[j]

        shift = neighbor_matrix_shifts[atom_idx, neighbor_slot]
        shift_vec = cell_t * type(ri)(shift)
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        prefactor = wp.float64(0.5) * bi * bj
        energy_acc += _disp_pair_energy(prefactor, r * r, alpha)

    wp.atomic_add(atomic_energies, atom_idx, energy_acc)


@wp.kernel
def _batch_lj_dispersion_energy_forces_matrix_kernel(
    positions: wp.array(dtype=wp.vec3d),
    b: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    batch_idx: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    alpha: wp.float64,
    fill_value: wp.int32,
    atomic_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=wp.vec3d),
):
    """Compute dispersion energies and forces for batched systems using matrix."""
    atom_idx = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_idx >= num_atoms:
        return

    system_id = batch_idx[atom_idx]
    ri = positions[atom_idx]
    bi = b[atom_idx]
    cell_t = wp.transpose(cell[system_id])

    energy_acc = wp.float64(0.0)
    force_acc = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

    for neighbor_slot in range(max_neighbors):
        j = neighbor_matrix[atom_idx, neighbor_slot]
        if j >= fill_value or j >= num_atoms:
            continue

        rj = positions[j]
        bj = b[j]

        shift = neighbor_matrix_shifts[atom_idx, neighbor_slot]
        shift_vec = cell_t * type(ri)(shift)
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        r2 = r * r
        prefactor = wp.float64(0.5) * bi * bj
        energy_acc += _disp_pair_energy(prefactor, r2, alpha)
        force_ij = _disp_pair_force_over_r(prefactor, r2, alpha) * r_ij

        force_acc += force_ij
        wp.atomic_add(atomic_forces, j, -force_ij)

    wp.atomic_add(atomic_energies, atom_idx, energy_acc)
    wp.atomic_add(atomic_forces, atom_idx, force_acc)


# ==============================================================================
# Warp Launchers (Framework-Agnostic)
# ==============================================================================


def lj_dispersion_energy(
    positions: wp.array,
    b: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    cutoff: float,
    alpha: float,
    energies: wp.array,
    device: str | None = None,
) -> None:
    """Launch dispersion energy kernel using CSR neighbor list format.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3d
        Atomic positions.
    b : wp.array, shape (N,), dtype=wp.float64
        Per-atom dispersion charge ``b_i = sqrt(C6_i)``.
    cell : wp.array, shape (1,), dtype=wp.mat33d
        Unit cell matrix.
    idx_j, neighbor_ptr, unit_shifts : wp.array
        CSR neighbor list (destination indices, row pointers, integer shifts).
    cutoff : float
        Real-space cutoff distance.
    alpha : float
        Dispersion Ewald splitting parameter ``beta`` (0.0 for plain ``-C6/r^6``).
    energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: per-atom energies (pre-allocated, zeroed).
    device : str, optional
        Warp device. If None, inferred from positions.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _lj_dispersion_energy_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            b,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            wp.float64(cutoff),
            wp.float64(alpha),
            energies,
        ],
        device=device,
    )


def lj_dispersion_energy_forces(
    positions: wp.array,
    b: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    cutoff: float,
    alpha: float,
    energies: wp.array,
    forces: wp.array,
    device: str | None = None,
) -> None:
    """Launch dispersion energy + forces kernel using CSR neighbor list format."""
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _lj_dispersion_energy_forces_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            b,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            wp.float64(cutoff),
            wp.float64(alpha),
            energies,
            forces,
        ],
        device=device,
    )


def lj_dispersion_energy_matrix(
    positions: wp.array,
    b: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    cutoff: float,
    alpha: float,
    fill_value: int,
    energies: wp.array,
    device: str | None = None,
) -> None:
    """Launch dispersion energy kernel using neighbor matrix format."""
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _lj_dispersion_energy_matrix_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            b,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            wp.float64(cutoff),
            wp.float64(alpha),
            wp.int32(fill_value),
            energies,
        ],
        device=device,
    )


def lj_dispersion_energy_forces_matrix(
    positions: wp.array,
    b: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    cutoff: float,
    alpha: float,
    fill_value: int,
    energies: wp.array,
    forces: wp.array,
    device: str | None = None,
) -> None:
    """Launch dispersion energy + forces kernel using neighbor matrix format."""
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _lj_dispersion_energy_forces_matrix_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            b,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            wp.float64(cutoff),
            wp.float64(alpha),
            wp.int32(fill_value),
            energies,
            forces,
        ],
        device=device,
    )


def batch_lj_dispersion_energy(
    positions: wp.array,
    b: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    batch_idx: wp.array,
    cutoff: float,
    alpha: float,
    energies: wp.array,
    device: str | None = None,
) -> None:
    """Launch batched dispersion energy kernel using CSR neighbor list format."""
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _batch_lj_dispersion_energy_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            b,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
            wp.float64(cutoff),
            wp.float64(alpha),
            energies,
        ],
        device=device,
    )


def batch_lj_dispersion_energy_forces(
    positions: wp.array,
    b: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    batch_idx: wp.array,
    cutoff: float,
    alpha: float,
    energies: wp.array,
    forces: wp.array,
    device: str | None = None,
) -> None:
    """Launch batched dispersion energy + forces kernel using CSR list format."""
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _batch_lj_dispersion_energy_forces_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            b,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
            wp.float64(cutoff),
            wp.float64(alpha),
            energies,
            forces,
        ],
        device=device,
    )


def batch_lj_dispersion_energy_matrix(
    positions: wp.array,
    b: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    batch_idx: wp.array,
    cutoff: float,
    alpha: float,
    fill_value: int,
    energies: wp.array,
    device: str | None = None,
) -> None:
    """Launch batched dispersion energy kernel using neighbor matrix format."""
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _batch_lj_dispersion_energy_matrix_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            b,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            batch_idx,
            wp.float64(cutoff),
            wp.float64(alpha),
            wp.int32(fill_value),
            energies,
        ],
        device=device,
    )


def batch_lj_dispersion_energy_forces_matrix(
    positions: wp.array,
    b: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    batch_idx: wp.array,
    cutoff: float,
    alpha: float,
    fill_value: int,
    energies: wp.array,
    forces: wp.array,
    device: str | None = None,
) -> None:
    """Launch batched dispersion energy + forces kernel using neighbor matrix."""
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _batch_lj_dispersion_energy_forces_matrix_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            b,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            batch_idx,
            wp.float64(cutoff),
            wp.float64(alpha),
            wp.int32(fill_value),
            energies,
            forces,
        ],
        device=device,
    )
