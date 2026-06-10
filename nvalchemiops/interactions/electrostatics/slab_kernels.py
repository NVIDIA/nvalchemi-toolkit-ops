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
Yeh-Berkowitz Slab Correction Kernels
=====================================

This module provides Warp kernels for the Yeh-Berkowitz slab
correction, enabling accurate electrostatics for 2D periodic (slab) systems.

The slab correction removes spurious interactions between periodic images
along the non-periodic direction when using 3D Ewald methods for systems that
are only periodic in two dimensions. For triclinic cells, positions are
projected onto the normal of the two periodic cell vectors.

MATHEMATICAL FORMULATION
========================

Let :math:`\mathbf{n}` be the unit normal to the periodic plane,
:math:`z_i = \mathbf{r}_i \cdot \mathbf{n}`, and
:math:`L = |\mathbf{h}_k \cdot \mathbf{n}|`, where :math:`\mathbf{h}_k`
is the non-periodic cell vector selected by pbc. Per-atom energy:

.. math::

    E_{\\text{slab},i} = \\frac{2\\pi}{V} q_i
    \\left[ z_i M - \\frac{1}{2}(M_2 + Q z_i^2) - \\frac{Q}{12} L^2 \\right]

Per-atom force:

.. math::

    \\mathbf{F}_{\\text{slab},i} = -\\frac{4\\pi}{V} q_i (M - Q z_i) \\mathbf{n}

Per-atom charge gradient:

.. math::

    \\frac{\\partial E_{\\text{slab}}}{\\partial q_i} = \\frac{4\\pi}{V}
    \\left[ z_i M - \\frac{1}{2}(M_2 + Q z_i^2) - \\frac{Q}{12} L^2 \\right]

Per-atom virial contribution under the normal-following affine strain
convention :math:`\mathbf{r}' = \mathbf{F}\mathbf{r}`,
:math:`\mathbf{h}' = \mathbf{F}\mathbf{h}`:

.. math::

    \\mathbf{W}_{\\text{slab},i} =
    E_{\\text{slab},i}(\\mathbf{I} - 2\\mathbf{n}\\mathbf{n}^{T})

where :math:`M = \\sum_j q_j z_j`, :math:`M_2 = \\sum_j q_j z_j^2`,
:math:`Q = \\sum_j q_j`, :math:`V = |\\det(\\mathbf{h})|`,
:math:`L = |\\mathbf{h}_k \\cdot \mathbf{n}|`.

CELL GEOMETRY
=============

Orthorhombic and triclinic cells are supported. The pbc tensor selects the
non-periodic cell vector. The slab normal is recomputed from the two periodic
cell vectors for each system, so tilted periodic planes use the correct
normal-following geometry.

NON-NEUTRAL SYSTEMS
===================

For systems with net charge :math:`Q \\ne 0`, the slab correction follows
the Ballenegger et al. (2009) Eq. 29 convention: a uniform-volume
neutralizing background charge density :math:`\\rho_b = -Q/V` (the same
convention used by standard 3D Ewald). Other conventions (uniform plane,
pinned dipole) yield different additive constants.

PER-SYSTEM PBC
==============

Each batch system carries its own pbc tensor of shape (3,) with True for
periodic directions and False for the non-periodic direction. The kernels
inspect pbc[system_id] to determine the non-periodic axis without any
host/device synchronization. Systems with pbc patterns other than exactly
one False entry (e.g., fully 3D periodic [T, T, T] or 1D periodic) yield
zero contribution.

KERNEL ORGANIZATION
===================

Moment Reduction:
    _slab_reduce_moments_kernel: Accumulate projected M, M2, Q_total per system

Per-Atom Correction:
    Ewald-style split kernels cover energy, energy+forces, and
    energy+forces+charge gradients. Separate virial-capable launch paths avoid
    passing unused output buffers for force-only or charge-gradient-only calls.

Both kernels handle single-system and batched calculations via batch_idx.
For single systems, pass batch_idx = zeros(N, dtype=int32).

REFERENCES
==========

- Yeh, I.-C. & Berkowitz, M. L. (1999). J. Chem. Phys. 111, 3155-3162.
  (Original slab correction for neutral systems)
- Ballenegger, V., Arnold, A. & Cerdà, J. J. (2009). J. Chem. Phys. 131, 094107.
  (Extension to non-neutral systems via background charge correction, Eq. 29)
"""

import math
from typing import Any

import warp as wp

# Mathematical constants
PI = wp.constant(wp.float64(math.pi))
TWOPI = wp.constant(wp.float64(2.0 * math.pi))
FOURPI = wp.constant(wp.float64(4.0 * math.pi))


###########################################################################################
########################### Moment Reduction Kernel #######################################
###########################################################################################


@wp.kernel
def _slab_reduce_moments_kernel(
    positions: wp.array(dtype=Any),  # (N,) vec3
    charges: wp.array(dtype=Any),  # (N,)
    batch_idx: wp.array(dtype=wp.int32),  # (N,)
    pbc: wp.array2d(dtype=wp.bool),  # (B, 3) per-system pbc
    cell: wp.array(dtype=Any),  # (B,) mat33
    mz: wp.array2d(dtype=wp.float64),  # (B, 3) OUTPUT -- projected M in slab-axis slot
    mz2: wp.array2d(
        dtype=wp.float64
    ),  # (B, 3) OUTPUT -- projected M2 in slab-axis slot
    qtotal: wp.array(dtype=wp.float64),  # (B,) OUTPUT -- total charge per system
):
    """Accumulate charge moments along each system's non-periodic axis.

    Each thread processes one atom and accumulates its contributions to its
    system's moments using atomic additions. The non-periodic axis is
    determined per-system from pbc[system_id] entirely on-device.

    Launch Grid
    -----------
    dim = [N_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom (0 to B-1). For single systems, all zeros.
    pbc : wp.array2d, shape (B, 3), dtype=wp.bool
        Per-system periodic boundary conditions. True for periodic directions,
        False for the non-periodic (slab) direction. Systems with patterns
        other than exactly one False entry contribute zero.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system cell matrices. The slab normal is computed from the two
        periodic cell vectors.
    mz : wp.array2d, shape (B, 3), dtype=wp.float64
        OUTPUT: Per-system projected dipole M = sum_i q_i (r_i dot n),
        stored in the non-periodic axis slot.
        Must be zero-initialized before launch.
    mz2 : wp.array2d, shape (B, 3), dtype=wp.float64
        OUTPUT: Per-system projected moment M2 = sum_i q_i (r_i dot n)^2,
        stored in the non-periodic axis slot.
        Must be zero-initialized before launch.
    qtotal : wp.array, shape (B,), dtype=wp.float64
        OUTPUT: Per-system total charge.
        Must be zero-initialized before launch.

    Notes
    -----
    - All accumulations use float64 for numerical stability.
    - Output arrays must be zero-initialized before kernel launch.
    - Atoms in non-slab systems contribute nothing; the kernel determines
      slab geometry per-system from pbc without any host sync.
    """
    atom_idx = wp.tid()

    system_id = batch_idx[atom_idx]
    p0 = pbc[system_id, 0]
    p1 = pbc[system_id, 1]
    p2 = pbc[system_id, 2]

    q = charges[atom_idx]
    pos = positions[atom_idx]

    # Determine the non-periodic axis (the index where pbc is False).
    # Slab geometry has exactly one False entry. Other patterns
    # (fully 3D periodic, 1D periodic, no periodicity) contribute zero.
    axis_idx = wp.int32(2)
    is_slab = False

    if (not p0) and p1 and p2:
        axis_idx = wp.int32(0)
        is_slab = True
    elif p0 and (not p1) and p2:
        axis_idx = wp.int32(1)
        is_slab = True
    elif p0 and p1 and (not p2):
        axis_idx = wp.int32(2)
        is_slab = True

    if is_slab:
        cell_b = cell[system_id]

        # Pick periodic cell vectors by the cyclic convention:
        # axis 0 -> cross(h1, h2), axis 1 -> cross(h2, h0),
        # axis 2 -> cross(h0, h1). This reduces to +x/+y/+z for
        # right-handed axis-aligned cells.
        periodic_a = cell_b[0]
        periodic_b = cell_b[1]
        if axis_idx == wp.int32(0):
            periodic_a = cell_b[1]
            periodic_b = cell_b[2]
        elif axis_idx == wp.int32(1):
            periodic_a = cell_b[2]
            periodic_b = cell_b[0]

        normal_raw = wp.cross(periodic_a, periodic_b)
        normal = normal_raw / wp.length(normal_raw)
        z = wp.dot(pos, normal)

        q_f64 = wp.float64(q)
        z_f64 = wp.float64(z)
        m_contrib = q_f64 * z_f64
        m2_contrib = m_contrib * z_f64

        if axis_idx == wp.int32(0):
            wp.atomic_add(mz, system_id, 0, m_contrib)
            wp.atomic_add(mz2, system_id, 0, m2_contrib)
        elif axis_idx == wp.int32(1):
            wp.atomic_add(mz, system_id, 1, m_contrib)
            wp.atomic_add(mz2, system_id, 1, m2_contrib)
        else:
            wp.atomic_add(mz, system_id, 2, m_contrib)
            wp.atomic_add(mz2, system_id, 2, m2_contrib)

        wp.atomic_add(qtotal, system_id, wp.float64(q))


@wp.kernel
def _slab_precompute_geometry_kernel(
    pbc: wp.array2d(dtype=wp.bool),
    cell: wp.array(dtype=Any),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    slab_volume: wp.array(dtype=wp.float64),
    slab_height_sq: wp.array(dtype=wp.float64),
):
    """Precompute per-system slab geometry consumed by atom-major kernels."""
    system_id = wp.tid()
    p0 = pbc[system_id, 0]
    p1 = pbc[system_id, 1]
    p2 = pbc[system_id, 2]

    axis_idx = wp.int32(-1)
    if (not p0) and p1 and p2:
        axis_idx = wp.int32(0)
    elif p0 and (not p1) and p2:
        axis_idx = wp.int32(1)
    elif p0 and p1 and (not p2):
        axis_idx = wp.int32(2)

    slab_axis[system_id] = axis_idx
    if axis_idx < wp.int32(0):
        slab_normal[system_id] = wp.vec3d(0.0, 0.0, 0.0)
        slab_volume[system_id] = wp.float64(0.0)
        slab_height_sq[system_id] = wp.float64(0.0)
        return

    cell_b = cell[system_id]
    periodic_a = cell_b[0]
    periodic_b = cell_b[1]
    nonperiodic_c = cell_b[2]
    if axis_idx == wp.int32(0):
        periodic_a = cell_b[1]
        periodic_b = cell_b[2]
        nonperiodic_c = cell_b[0]
    elif axis_idx == wp.int32(1):
        periodic_a = cell_b[2]
        periodic_b = cell_b[0]
        nonperiodic_c = cell_b[1]

    normal_raw = wp.cross(periodic_a, periodic_b)
    normal = normal_raw / wp.length(normal_raw)
    c_dot_n = wp.dot(nonperiodic_c, normal)

    slab_normal[system_id] = wp.vec3d(
        wp.float64(normal[0]),
        wp.float64(normal[1]),
        wp.float64(normal[2]),
    )
    slab_volume[system_id] = wp.abs(wp.float64(wp.determinant(cell_b)))
    slab_height_sq[system_id] = wp.float64(c_dot_n) * wp.float64(c_dot_n)


###########################################################################################
########################### Per-Atom Slab Correction Kernel ###############################
###########################################################################################


@wp.func
def _slab_correction_terms_precomputed(
    atom_idx: wp.int32,
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    slab_volume: wp.array(dtype=wp.float64),
    slab_height_sq: wp.array(dtype=wp.float64),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
) -> tuple[
    bool,
    wp.int32,
    wp.int32,
    Any,
    wp.float64,
    wp.float64,
    wp.float64,
    wp.float64,
    wp.float64,
    wp.float64,
]:
    """Compute common slab terms from per-system precomputed geometry."""
    system_id = batch_idx[atom_idx]
    axis_idx = slab_axis[system_id]
    pos = positions[atom_idx]
    q = charges[atom_idx]

    zero = wp.float64(0.0)
    normal_d = slab_normal[system_id]
    normal = type(pos)(
        type(pos[0])(normal_d[0]),
        type(pos[0])(normal_d[1]),
        type(pos[0])(normal_d[2]),
    )
    if axis_idx < wp.int32(0):
        return False, system_id, axis_idx, normal, zero, zero, zero, zero, zero, zero

    vol = slab_volume[system_id]
    height_sq = slab_height_sq[system_id]
    z = wp.dot(pos, normal)

    mz_val = mz[system_id, 2]
    mz2_val = mz2[system_id, 2]
    if axis_idx == wp.int32(0):
        mz_val = mz[system_id, 0]
        mz2_val = mz2[system_id, 0]
    elif axis_idx == wp.int32(1):
        mz_val = mz[system_id, 1]
        mz2_val = mz2[system_id, 1]
    qtot = qtotal[system_id]

    z_f64 = wp.float64(z)
    q_f64 = wp.float64(q)
    bracket = (
        z_f64 * mz_val
        - wp.float64(0.5) * (mz2_val + qtot * z_f64 * z_f64)
        - qtot / wp.float64(12.0) * height_sq
    )
    e_slab = (TWOPI / vol) * q_f64 * bracket

    return (
        True,
        system_id,
        axis_idx,
        normal,
        vol,
        e_slab,
        bracket,
        z_f64,
        q_f64,
        mz_val,
    )


@wp.func
def _slab_add_force(
    atom_idx: wp.int32,
    normal: Any,
    vol: wp.float64,
    z_f64: wp.float64,
    q_f64: wp.float64,
    mz_val: wp.float64,
    qtot: wp.float64,
    forces: wp.array(dtype=Any),
):
    """Accumulate one atom's slab force."""
    f_slab_mag = -(FOURPI / vol) * q_f64 * (mz_val - qtot * z_f64)
    f_slab = type(normal)(
        type(normal[0])(f_slab_mag * wp.float64(normal[0])),
        type(normal[0])(f_slab_mag * wp.float64(normal[1])),
        type(normal[0])(f_slab_mag * wp.float64(normal[2])),
    )
    wp.atomic_add(forces, atom_idx, f_slab)


@wp.func
def _slab_add_charge_grad(
    atom_idx: wp.int32,
    vol: wp.float64,
    bracket: wp.float64,
    charge_grads: wp.array(dtype=wp.float64),
):
    """Accumulate one atom's slab charge gradient."""
    wp.atomic_add(charge_grads, atom_idx, (FOURPI / vol) * bracket)


@wp.func
def _slab_add_virial(
    system_id: wp.int32,
    normal: Any,
    e_slab: wp.float64,
    virial: wp.array(dtype=Any),
):
    """Accumulate one atom's normal-following slab virial."""
    n0 = wp.float64(normal[0])
    n1 = wp.float64(normal[1])
    n2 = wp.float64(normal[2])
    two = wp.float64(2.0)
    one = wp.float64(1.0)

    virial_mat = wp.mat33d(
        e_slab * (one - two * n0 * n0),
        e_slab * (-two * n0 * n1),
        e_slab * (-two * n0 * n2),
        e_slab * (-two * n1 * n0),
        e_slab * (one - two * n1 * n1),
        e_slab * (-two * n1 * n2),
        e_slab * (-two * n2 * n0),
        e_slab * (-two * n2 * n1),
        e_slab * (one - two * n2 * n2),
    )
    wp.atomic_add(virial, system_id, type(virial[0])(virial_mat))


@wp.kernel
def _slab_correction_energy_kernel(
    positions: wp.array(dtype=Any),  # (N,) vec3
    charges: wp.array(dtype=Any),  # (N,)
    batch_idx: wp.array(dtype=wp.int32),  # (N,)
    slab_axis: wp.array(dtype=wp.int32),  # (B,) precomputed nonperiodic axis
    slab_normal: wp.array(dtype=wp.vec3d),  # (B,) precomputed slab normal
    slab_volume: wp.array(dtype=wp.float64),  # (B,) precomputed volume
    slab_height_sq: wp.array(dtype=wp.float64),  # (B,) precomputed height^2
    mz: wp.array2d(dtype=wp.float64),  # (B, 3) projected M in slab-axis slot
    mz2: wp.array2d(dtype=wp.float64),  # (B, 3) projected M2 in slab-axis slot
    qtotal: wp.array(dtype=wp.float64),  # (B,) precomputed total charge
    energy_in: wp.array(dtype=wp.float64),  # (N,) input energies
    energy_out: wp.array(dtype=wp.float64),  # (N,) OUTPUT: energy_in + slab correction
):
    """Apply the slab energy correction."""
    atom_idx = wp.tid()
    (
        is_slab,
        _energy_system_id,
        _energy_axis,
        _energy_normal,
        _energy_vol,
        e_slab,
        _energy_bracket,
        _energy_z,
        _energy_q,
        _energy_mz,
    ) = _slab_correction_terms_precomputed(
        atom_idx,
        positions,
        charges,
        batch_idx,
        slab_axis,
        slab_normal,
        slab_volume,
        slab_height_sq,
        mz,
        mz2,
        qtotal,
    )
    if not is_slab:
        energy_out[atom_idx] = energy_in[atom_idx]
        return
    energy_out[atom_idx] = energy_in[atom_idx] + e_slab


@wp.kernel
def _slab_correction_energy_forces_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    slab_volume: wp.array(dtype=wp.float64),
    slab_height_sq: wp.array(dtype=wp.float64),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    energy_in: wp.array(dtype=wp.float64),
    energy_out: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=Any),
):
    """Apply slab energy and force corrections."""
    atom_idx = wp.tid()
    (
        is_slab,
        system_id,
        _axis_idx,
        normal,
        vol,
        e_slab,
        _force_bracket,
        z_f64,
        q_f64,
        mz_val,
    ) = _slab_correction_terms_precomputed(
        atom_idx,
        positions,
        charges,
        batch_idx,
        slab_axis,
        slab_normal,
        slab_volume,
        slab_height_sq,
        mz,
        mz2,
        qtotal,
    )
    if not is_slab:
        energy_out[atom_idx] = energy_in[atom_idx]
        return
    energy_out[atom_idx] = energy_in[atom_idx] + e_slab
    _slab_add_force(
        atom_idx, normal, vol, z_f64, q_f64, mz_val, qtotal[system_id], forces
    )


@wp.kernel
def _slab_correction_energy_forces_virial_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    slab_volume: wp.array(dtype=wp.float64),
    slab_height_sq: wp.array(dtype=wp.float64),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    energy_in: wp.array(dtype=wp.float64),
    energy_out: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=Any),
    virial: wp.array(dtype=Any),
):
    """Apply slab energy, force, and virial corrections."""
    atom_idx = wp.tid()
    (
        is_slab,
        system_id,
        _axis_idx,
        normal,
        vol,
        e_slab,
        _force_bracket,
        z_f64,
        q_f64,
        mz_val,
    ) = _slab_correction_terms_precomputed(
        atom_idx,
        positions,
        charges,
        batch_idx,
        slab_axis,
        slab_normal,
        slab_volume,
        slab_height_sq,
        mz,
        mz2,
        qtotal,
    )
    if not is_slab:
        energy_out[atom_idx] = energy_in[atom_idx]
        return
    energy_out[atom_idx] = energy_in[atom_idx] + e_slab
    _slab_add_force(
        atom_idx, normal, vol, z_f64, q_f64, mz_val, qtotal[system_id], forces
    )
    _slab_add_virial(system_id, normal, e_slab, virial)


@wp.kernel
def _slab_correction_energy_virial_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    slab_volume: wp.array(dtype=wp.float64),
    slab_height_sq: wp.array(dtype=wp.float64),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    energy_in: wp.array(dtype=wp.float64),
    energy_out: wp.array(dtype=wp.float64),
    virial: wp.array(dtype=Any),
):
    """Apply slab energy and virial corrections without touching force output."""
    atom_idx = wp.tid()
    (
        is_slab,
        system_id,
        _axis_idx,
        normal,
        _vol,
        e_slab,
        _bracket,
        _z_f64,
        _q_f64,
        _mz_val,
    ) = _slab_correction_terms_precomputed(
        atom_idx,
        positions,
        charges,
        batch_idx,
        slab_axis,
        slab_normal,
        slab_volume,
        slab_height_sq,
        mz,
        mz2,
        qtotal,
    )
    if not is_slab:
        energy_out[atom_idx] = energy_in[atom_idx]
        return
    energy_out[atom_idx] = energy_in[atom_idx] + e_slab
    _slab_add_virial(system_id, normal, e_slab, virial)


@wp.kernel
def _slab_correction_energy_forces_charge_grad_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    slab_volume: wp.array(dtype=wp.float64),
    slab_height_sq: wp.array(dtype=wp.float64),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    energy_in: wp.array(dtype=wp.float64),
    energy_out: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=Any),
    charge_grads: wp.array(dtype=wp.float64),
):
    """Apply slab energy, force, and charge-gradient corrections."""
    atom_idx = wp.tid()
    (
        is_slab,
        system_id,
        _axis_idx,
        normal,
        vol,
        e_slab,
        bracket,
        z_f64,
        q_f64,
        mz_val,
    ) = _slab_correction_terms_precomputed(
        atom_idx,
        positions,
        charges,
        batch_idx,
        slab_axis,
        slab_normal,
        slab_volume,
        slab_height_sq,
        mz,
        mz2,
        qtotal,
    )
    if not is_slab:
        energy_out[atom_idx] = energy_in[atom_idx]
        return
    energy_out[atom_idx] = energy_in[atom_idx] + e_slab
    _slab_add_force(
        atom_idx, normal, vol, z_f64, q_f64, mz_val, qtotal[system_id], forces
    )
    _slab_add_charge_grad(atom_idx, vol, bracket, charge_grads)


@wp.kernel
def _slab_correction_energy_forces_charge_grad_virial_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    slab_volume: wp.array(dtype=wp.float64),
    slab_height_sq: wp.array(dtype=wp.float64),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    energy_in: wp.array(dtype=wp.float64),
    energy_out: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=Any),
    charge_grads: wp.array(dtype=wp.float64),
    virial: wp.array(dtype=Any),
):
    """Apply slab energy, force, charge-gradient, and virial corrections."""
    atom_idx = wp.tid()
    (
        is_slab,
        system_id,
        _axis_idx,
        normal,
        vol,
        e_slab,
        bracket,
        z_f64,
        q_f64,
        mz_val,
    ) = _slab_correction_terms_precomputed(
        atom_idx,
        positions,
        charges,
        batch_idx,
        slab_axis,
        slab_normal,
        slab_volume,
        slab_height_sq,
        mz,
        mz2,
        qtotal,
    )
    if not is_slab:
        energy_out[atom_idx] = energy_in[atom_idx]
        return
    energy_out[atom_idx] = energy_in[atom_idx] + e_slab
    _slab_add_force(
        atom_idx, normal, vol, z_f64, q_f64, mz_val, qtotal[system_id], forces
    )
    _slab_add_charge_grad(atom_idx, vol, bracket, charge_grads)
    _slab_add_virial(system_id, normal, e_slab, virial)


@wp.kernel
def _slab_correction_energy_charge_grad_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    slab_volume: wp.array(dtype=wp.float64),
    slab_height_sq: wp.array(dtype=wp.float64),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    energy_in: wp.array(dtype=wp.float64),
    energy_out: wp.array(dtype=wp.float64),
    charge_grads: wp.array(dtype=wp.float64),
):
    """Apply slab energy and charge-gradient corrections without force output."""
    atom_idx = wp.tid()
    (
        is_slab,
        system_id,
        _axis_idx,
        normal,
        vol,
        e_slab,
        bracket,
        _z_f64,
        _q_f64,
        _mz_val,
    ) = _slab_correction_terms_precomputed(
        atom_idx,
        positions,
        charges,
        batch_idx,
        slab_axis,
        slab_normal,
        slab_volume,
        slab_height_sq,
        mz,
        mz2,
        qtotal,
    )
    if not is_slab:
        energy_out[atom_idx] = energy_in[atom_idx]
        return
    energy_out[atom_idx] = energy_in[atom_idx] + e_slab
    _slab_add_charge_grad(atom_idx, vol, bracket, charge_grads)


@wp.kernel
def _slab_correction_energy_charge_grad_virial_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    slab_volume: wp.array(dtype=wp.float64),
    slab_height_sq: wp.array(dtype=wp.float64),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    energy_in: wp.array(dtype=wp.float64),
    energy_out: wp.array(dtype=wp.float64),
    charge_grads: wp.array(dtype=wp.float64),
    virial: wp.array(dtype=Any),
):
    """Apply slab energy, charge-gradient, and virial corrections."""
    atom_idx = wp.tid()
    (
        is_slab,
        system_id,
        _axis_idx,
        normal,
        vol,
        e_slab,
        bracket,
        _z_f64,
        _q_f64,
        _mz_val,
    ) = _slab_correction_terms_precomputed(
        atom_idx,
        positions,
        charges,
        batch_idx,
        slab_axis,
        slab_normal,
        slab_volume,
        slab_height_sq,
        mz,
        mz2,
        qtotal,
    )
    if not is_slab:
        energy_out[atom_idx] = energy_in[atom_idx]
        return
    energy_out[atom_idx] = energy_in[atom_idx] + e_slab
    _slab_add_charge_grad(atom_idx, vol, bracket, charge_grads)
    _slab_add_virial(system_id, normal, e_slab, virial)


@wp.kernel
def _slab_correction_backward_atoms_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    slab_volume: wp.array(dtype=wp.float64),
    slab_height_sq: wp.array(dtype=wp.float64),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    grad_system: wp.array(dtype=wp.float64),
    grad_positions: wp.array(dtype=Any),
    grad_charges: wp.array(dtype=wp.float64),
    grad_normal: wp.array(dtype=wp.vec3d),
):
    """Accumulate first-order slab energy gradients for atoms.

    Thread launch
    -------------
    One thread per atom. The thread writes the atom's position and charge
    gradient and atomically accumulates the slab-normal cotangent needed by the
    cell-gradient kernel.

    Modifies
    --------
    ``grad_positions``, ``grad_charges``, and ``grad_normal``.
    """
    atom_idx = wp.tid()
    is_slab, system_id, axis_idx, normal, vol, e_slab, bracket, z_f64, q_f64, mz_val = (
        _slab_correction_terms_precomputed(
            atom_idx,
            positions,
            charges,
            batch_idx,
            slab_axis,
            slab_normal,
            slab_volume,
            slab_height_sq,
            mz,
            mz2,
            qtotal,
        )
    )
    if not is_slab:
        grad_positions[atom_idx] = type(grad_positions[0])(
            type(grad_positions[0][0])(0.0),
            type(grad_positions[0][0])(0.0),
            type(grad_positions[0][0])(0.0),
        )
        grad_charges[atom_idx] = wp.float64(0.0)
        return

    g = grad_system[system_id]
    qtot = qtotal[system_id]
    d_e_dz = g * (FOURPI / vol) * q_f64 * (mz_val - qtot * z_f64)
    grad_pos_f64 = wp.vec3d(
        d_e_dz * wp.float64(normal[0]),
        d_e_dz * wp.float64(normal[1]),
        d_e_dz * wp.float64(normal[2]),
    )
    grad_positions[atom_idx] = type(grad_positions[0])(grad_pos_f64)
    grad_charges[atom_idx] = g * (FOURPI / vol) * bracket

    pos = positions[atom_idx]
    grad_n = wp.vec3d(
        d_e_dz * wp.float64(pos[0]),
        d_e_dz * wp.float64(pos[1]),
        d_e_dz * wp.float64(pos[2]),
    )
    wp.atomic_add(grad_normal, system_id, grad_n)


@wp.kernel
def _slab_correction_backward_cell_kernel(
    pbc: wp.array2d(dtype=wp.bool),
    cell: wp.array(dtype=Any),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    grad_system: wp.array(dtype=wp.float64),
    grad_normal: wp.array(dtype=wp.vec3d),
    grad_cell: wp.array(dtype=Any),
):
    """Convert slab normal/volume cotangents into literal cell gradients.

    Thread launch
    -------------
    One thread per system.

    Modifies
    --------
    ``grad_cell``.
    """
    system_id = wp.tid()
    p0 = pbc[system_id, 0]
    p1 = pbc[system_id, 1]
    p2 = pbc[system_id, 2]

    axis_idx = wp.int32(2)
    is_slab = False
    if (not p0) and p1 and p2:
        axis_idx = wp.int32(0)
        is_slab = True
    elif p0 and (not p1) and p2:
        axis_idx = wp.int32(1)
        is_slab = True
    elif p0 and p1 and (not p2):
        axis_idx = wp.int32(2)
        is_slab = True

    zero_mat = wp.mat33d(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    if not is_slab:
        grad_cell[system_id] = type(grad_cell[0])(zero_mat)
        return

    cell_b = cell[system_id]
    h0 = cell_b[0]
    h1 = cell_b[1]
    h2 = cell_b[2]
    periodic_a = h0
    periodic_b = h1
    nonperiodic_c = h2
    if axis_idx == wp.int32(0):
        periodic_a = h1
        periodic_b = h2
        nonperiodic_c = h0
    elif axis_idx == wp.int32(1):
        periodic_a = h2
        periodic_b = h0
        nonperiodic_c = h1

    normal_raw = wp.cross(periodic_a, periodic_b)
    area = wp.length(normal_raw)
    normal = normal_raw / area
    det_h = wp.determinant(cell_b)
    det_sign = wp.float64(1.0)
    if wp.float64(det_h) < wp.float64(0.0):
        det_sign = wp.float64(-1.0)
    vol = wp.abs(wp.float64(det_h))
    c_dot_n = wp.dot(nonperiodic_c, normal)
    l_f64 = wp.float64(c_dot_n)
    height_sq = l_f64 * l_f64

    mz_val = mz[system_id, 2]
    mz2_val = mz2[system_id, 2]
    if axis_idx == wp.int32(0):
        mz_val = mz[system_id, 0]
        mz2_val = mz2[system_id, 0]
    elif axis_idx == wp.int32(1):
        mz_val = mz[system_id, 1]
        mz2_val = mz2[system_id, 1]
    qtot = qtotal[system_id]
    g = grad_system[system_id]

    energy_inner = mz_val * mz_val - qtot * mz2_val
    energy_inner = energy_inner - qtot * qtot * height_sq / wp.float64(12.0)
    d_e_dv = -g * TWOPI * energy_inner / (vol * vol)
    d_e_dh = -g * (TWOPI / vol) * qtot * qtot / wp.float64(12.0)

    grad_n = grad_normal[system_id] + wp.vec3d(
        d_e_dh * wp.float64(2.0) * l_f64 * wp.float64(nonperiodic_c[0]),
        d_e_dh * wp.float64(2.0) * l_f64 * wp.float64(nonperiodic_c[1]),
        d_e_dh * wp.float64(2.0) * l_f64 * wp.float64(nonperiodic_c[2]),
    )
    grad_c_extra = wp.vec3d(
        d_e_dh * wp.float64(2.0) * l_f64 * wp.float64(normal[0]),
        d_e_dh * wp.float64(2.0) * l_f64 * wp.float64(normal[1]),
        d_e_dh * wp.float64(2.0) * l_f64 * wp.float64(normal[2]),
    )

    n64 = wp.vec3d(
        wp.float64(normal[0]),
        wp.float64(normal[1]),
        wp.float64(normal[2]),
    )
    grad_u = (grad_n - n64 * wp.dot(n64, grad_n)) / wp.float64(area)
    grad_a = wp.cross(
        wp.vec3d(
            wp.float64(periodic_b[0]),
            wp.float64(periodic_b[1]),
            wp.float64(periodic_b[2]),
        ),
        grad_u,
    )
    grad_b = wp.cross(
        grad_u,
        wp.vec3d(
            wp.float64(periodic_a[0]),
            wp.float64(periodic_a[1]),
            wp.float64(periodic_a[2]),
        ),
    )

    grad_det = d_e_dv * det_sign
    vol_g0 = grad_det * wp.cross(
        wp.vec3d(wp.float64(h1[0]), wp.float64(h1[1]), wp.float64(h1[2])),
        wp.vec3d(wp.float64(h2[0]), wp.float64(h2[1]), wp.float64(h2[2])),
    )
    vol_g1 = grad_det * wp.cross(
        wp.vec3d(wp.float64(h2[0]), wp.float64(h2[1]), wp.float64(h2[2])),
        wp.vec3d(wp.float64(h0[0]), wp.float64(h0[1]), wp.float64(h0[2])),
    )
    vol_g2 = grad_det * wp.cross(
        wp.vec3d(wp.float64(h0[0]), wp.float64(h0[1]), wp.float64(h0[2])),
        wp.vec3d(wp.float64(h1[0]), wp.float64(h1[1]), wp.float64(h1[2])),
    )

    row0 = vol_g0
    row1 = vol_g1
    row2 = vol_g2
    if axis_idx == wp.int32(0):
        row0 = row0 + grad_c_extra
        row1 = row1 + grad_a
        row2 = row2 + grad_b
    elif axis_idx == wp.int32(1):
        row0 = row0 + grad_b
        row1 = row1 + grad_c_extra
        row2 = row2 + grad_a
    else:
        row0 = row0 + grad_a
        row1 = row1 + grad_b
        row2 = row2 + grad_c_extra

    grad = wp.mat33d(
        row0[0],
        row0[1],
        row0[2],
        row1[0],
        row1[1],
        row1[2],
        row2[0],
        row2[1],
        row2[2],
    )
    grad_cell[system_id] = type(grad_cell[0])(grad)


@wp.kernel
def _slab_directional_geometry_kernel(
    pbc: wp.array2d(dtype=wp.bool),
    cell: wp.array(dtype=Any),
    h_cell: wp.array(dtype=Any),
    dnormal: wp.array(dtype=wp.vec3d),
    dvolume: wp.array(dtype=wp.float64),
    dheight_sq: wp.array(dtype=wp.float64),
):
    """Compute per-system slab geometry directional derivatives.

    Thread launch
    -------------
    One thread per system.

    Modifies
    --------
    ``dnormal``, ``dvolume``, and ``dheight_sq``.
    """
    system_id = wp.tid()
    p0 = pbc[system_id, 0]
    p1 = pbc[system_id, 1]
    p2 = pbc[system_id, 2]

    axis_idx = wp.int32(2)
    is_slab = False
    if (not p0) and p1 and p2:
        axis_idx = wp.int32(0)
        is_slab = True
    elif p0 and (not p1) and p2:
        axis_idx = wp.int32(1)
        is_slab = True
    elif p0 and p1 and (not p2):
        axis_idx = wp.int32(2)
        is_slab = True

    zero_vec = wp.vec3d(0.0, 0.0, 0.0)
    if not is_slab:
        dnormal[system_id] = zero_vec
        dvolume[system_id] = wp.float64(0.0)
        dheight_sq[system_id] = wp.float64(0.0)
        return

    cell_b = cell[system_id]
    h_cell_b = h_cell[system_id]
    h0 = cell_b[0]
    h1 = cell_b[1]
    h2 = cell_b[2]
    hh0 = h_cell_b[0]
    hh1 = h_cell_b[1]
    hh2 = h_cell_b[2]

    h0d = wp.vec3d(wp.float64(h0[0]), wp.float64(h0[1]), wp.float64(h0[2]))
    h1d = wp.vec3d(wp.float64(h1[0]), wp.float64(h1[1]), wp.float64(h1[2]))
    h2d = wp.vec3d(wp.float64(h2[0]), wp.float64(h2[1]), wp.float64(h2[2]))
    hh0d = wp.vec3d(wp.float64(hh0[0]), wp.float64(hh0[1]), wp.float64(hh0[2]))
    hh1d = wp.vec3d(wp.float64(hh1[0]), wp.float64(hh1[1]), wp.float64(hh1[2]))
    hh2d = wp.vec3d(wp.float64(hh2[0]), wp.float64(hh2[1]), wp.float64(hh2[2]))

    periodic_a = h0d
    periodic_b = h1d
    nonperiodic_c = h2d
    h_periodic_a = hh0d
    h_periodic_b = hh1d
    h_nonperiodic_c = hh2d
    if axis_idx == wp.int32(0):
        periodic_a = h1d
        periodic_b = h2d
        nonperiodic_c = h0d
        h_periodic_a = hh1d
        h_periodic_b = hh2d
        h_nonperiodic_c = hh0d
    elif axis_idx == wp.int32(1):
        periodic_a = h2d
        periodic_b = h0d
        nonperiodic_c = h1d
        h_periodic_a = hh2d
        h_periodic_b = hh0d
        h_nonperiodic_c = hh1d

    periodic_a64 = wp.vec3d(
        wp.float64(periodic_a[0]),
        wp.float64(periodic_a[1]),
        wp.float64(periodic_a[2]),
    )
    periodic_b64 = wp.vec3d(
        wp.float64(periodic_b[0]),
        wp.float64(periodic_b[1]),
        wp.float64(periodic_b[2]),
    )
    normal_raw = wp.cross(periodic_a64, periodic_b64)
    area = wp.length(normal_raw)
    n64 = normal_raw / area
    normal = n64
    d_normal_raw = wp.cross(h_periodic_a, periodic_b) + wp.cross(
        periodic_a, h_periodic_b
    )
    d_area = wp.dot(normal, d_normal_raw)
    d_normal = (d_normal_raw - normal * d_area) / area

    det_h = wp.determinant(cell_b)
    det_sign = wp.float64(1.0)
    if wp.float64(det_h) < wp.float64(0.0):
        det_sign = wp.float64(-1.0)
    d_det = wp.dot(hh0d, wp.cross(h1d, h2d))
    d_det = d_det + wp.dot(h0d, wp.cross(hh1d, h2d))
    d_det = d_det + wp.dot(h0d, wp.cross(h1d, hh2d))
    d_vol = det_sign * d_det

    l_f64 = wp.dot(nonperiodic_c, normal)
    d_l = wp.dot(h_nonperiodic_c, normal) + wp.dot(nonperiodic_c, d_normal)

    dnormal[system_id] = d_normal
    dvolume[system_id] = d_vol
    dheight_sq[system_id] = wp.float64(2.0) * l_f64 * d_l


@wp.kernel
def _slab_directional_moments_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    h_positions: wp.array(dtype=Any),
    h_charges: wp.array(dtype=wp.float64),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    dnormal: wp.array(dtype=wp.vec3d),
    dmz: wp.array2d(dtype=wp.float64),
    dmz2: wp.array2d(dtype=wp.float64),
    dqtotal: wp.array(dtype=wp.float64),
):
    """Accumulate directional derivatives of slab moments.

    Thread launch
    -------------
    One thread per atom.

    Modifies
    --------
    ``dmz``, ``dmz2``, and ``dqtotal``.
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    axis_idx = slab_axis[system_id]
    if axis_idx < wp.int32(0):
        return

    normal = slab_normal[system_id]
    n64 = wp.vec3d(
        wp.float64(normal[0]),
        wp.float64(normal[1]),
        wp.float64(normal[2]),
    )
    dn = dnormal[system_id]
    pos = positions[atom_idx]
    h_pos = h_positions[atom_idx]
    pos64 = wp.vec3d(
        wp.float64(pos[0]),
        wp.float64(pos[1]),
        wp.float64(pos[2]),
    )
    h_pos64 = wp.vec3d(
        wp.float64(h_pos[0]),
        wp.float64(h_pos[1]),
        wp.float64(h_pos[2]),
    )
    z = wp.dot(pos64, n64)
    dz = wp.dot(h_pos64, n64) + wp.dot(pos64, dn)
    q = wp.float64(charges[atom_idx])
    hq = h_charges[atom_idx]
    dm = hq * z + q * dz
    dm2 = hq * z * z + wp.float64(2.0) * q * z * dz

    if axis_idx == wp.int32(0):
        wp.atomic_add(dmz, system_id, 0, dm)
        wp.atomic_add(dmz2, system_id, 0, dm2)
    elif axis_idx == wp.int32(1):
        wp.atomic_add(dmz, system_id, 1, dm)
        wp.atomic_add(dmz2, system_id, 1, dm2)
    else:
        wp.atomic_add(dmz, system_id, 2, dm)
        wp.atomic_add(dmz2, system_id, 2, dm2)
    wp.atomic_add(dqtotal, system_id, hq)


@wp.kernel
def _slab_weighted_moments_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    weights: wp.array(dtype=wp.float64),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    weighted_qtotal: wp.array(dtype=wp.float64),
    weighted_mz: wp.array2d(dtype=wp.float64),
    weighted_mz2: wp.array2d(dtype=wp.float64),
):
    """Accumulate weighted slab moments for non-uniform cotangents.

    Thread launch
    -------------
    One thread per atom.

    Modifies
    --------
    ``weighted_qtotal``, ``weighted_mz``, and ``weighted_mz2``.
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    axis_idx = slab_axis[system_id]
    if axis_idx < wp.int32(0):
        return

    normal = slab_normal[system_id]
    pos = positions[atom_idx]
    pos64 = wp.vec3d(
        wp.float64(pos[0]),
        wp.float64(pos[1]),
        wp.float64(pos[2]),
    )
    n64 = wp.vec3d(
        wp.float64(normal[0]),
        wp.float64(normal[1]),
        wp.float64(normal[2]),
    )
    z = wp.dot(pos64, n64)
    q = wp.float64(charges[atom_idx])
    wq = weights[atom_idx] * q
    wqz = wq * z
    wqz2 = wqz * z

    wp.atomic_add(weighted_qtotal, system_id, wq)
    if axis_idx == wp.int32(0):
        wp.atomic_add(weighted_mz, system_id, 0, wqz)
        wp.atomic_add(weighted_mz2, system_id, 0, wqz2)
    elif axis_idx == wp.int32(1):
        wp.atomic_add(weighted_mz, system_id, 1, wqz)
        wp.atomic_add(weighted_mz2, system_id, 1, wqz2)
    else:
        wp.atomic_add(weighted_mz, system_id, 2, wqz)
        wp.atomic_add(weighted_mz2, system_id, 2, wqz2)


@wp.kernel
def _slab_directional_weighted_moments_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    weights: wp.array(dtype=wp.float64),
    h_positions: wp.array(dtype=Any),
    h_charges: wp.array(dtype=wp.float64),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    dnormal: wp.array(dtype=wp.vec3d),
    d_weighted_qtotal: wp.array(dtype=wp.float64),
    d_weighted_mz: wp.array2d(dtype=wp.float64),
    d_weighted_mz2: wp.array2d(dtype=wp.float64),
):
    """Accumulate weighted moment directional derivatives.

    Thread launch
    -------------
    One thread per atom.

    Modifies
    --------
    ``d_weighted_qtotal``, ``d_weighted_mz``, and ``d_weighted_mz2``.
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    axis_idx = slab_axis[system_id]
    if axis_idx < wp.int32(0):
        return

    normal = slab_normal[system_id]
    n64 = wp.vec3d(
        wp.float64(normal[0]),
        wp.float64(normal[1]),
        wp.float64(normal[2]),
    )
    dn = dnormal[system_id]
    pos = positions[atom_idx]
    h_pos = h_positions[atom_idx]
    pos64 = wp.vec3d(
        wp.float64(pos[0]),
        wp.float64(pos[1]),
        wp.float64(pos[2]),
    )
    h_pos64 = wp.vec3d(
        wp.float64(h_pos[0]),
        wp.float64(h_pos[1]),
        wp.float64(h_pos[2]),
    )
    z = wp.dot(pos64, n64)
    dz = wp.dot(h_pos64, n64) + wp.dot(pos64, dn)
    weight = weights[atom_idx]
    q = wp.float64(charges[atom_idx])
    hq = h_charges[atom_idx]
    d_wq = weight * hq
    d_wqz = weight * (hq * z + q * dz)
    d_wqz2 = weight * (hq * z * z + wp.float64(2.0) * q * z * dz)

    wp.atomic_add(d_weighted_qtotal, system_id, d_wq)
    if axis_idx == wp.int32(0):
        wp.atomic_add(d_weighted_mz, system_id, 0, d_wqz)
        wp.atomic_add(d_weighted_mz2, system_id, 0, d_wqz2)
    elif axis_idx == wp.int32(1):
        wp.atomic_add(d_weighted_mz, system_id, 1, d_wqz)
        wp.atomic_add(d_weighted_mz2, system_id, 1, d_wqz2)
    else:
        wp.atomic_add(d_weighted_mz, system_id, 2, d_wqz)
        wp.atomic_add(d_weighted_mz2, system_id, 2, d_wqz2)


@wp.kernel
def _slab_correction_weighted_backward_atoms_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    weights: wp.array(dtype=wp.float64),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    slab_volume: wp.array(dtype=wp.float64),
    slab_height_sq: wp.array(dtype=wp.float64),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    weighted_mz: wp.array2d(dtype=wp.float64),
    weighted_mz2: wp.array2d(dtype=wp.float64),
    weighted_qtotal: wp.array(dtype=wp.float64),
    grad_positions: wp.array(dtype=Any),
    grad_charges: wp.array(dtype=wp.float64),
    grad_normal: wp.array(dtype=wp.vec3d),
):
    """Compute atom VJP terms for non-uniform slab cotangents.

    Thread launch
    -------------
    One thread per atom.

    Modifies
    --------
    ``grad_positions``, ``grad_charges``, and ``grad_normal``.
    """
    atom_idx = wp.tid()
    is_slab, system_id, axis_idx, normal, vol, e_slab, bracket, z_f64, q_f64, mz_val = (
        _slab_correction_terms_precomputed(
            atom_idx,
            positions,
            charges,
            batch_idx,
            slab_axis,
            slab_normal,
            slab_volume,
            slab_height_sq,
            mz,
            mz2,
            qtotal,
        )
    )
    if not is_slab:
        grad_positions[atom_idx] = type(grad_positions[0])(
            type(grad_positions[0][0])(0.0),
            type(grad_positions[0][0])(0.0),
            type(grad_positions[0][0])(0.0),
        )
        grad_charges[atom_idx] = wp.float64(0.0)
        return

    n64 = wp.vec3d(
        wp.float64(normal[0]),
        wp.float64(normal[1]),
        wp.float64(normal[2]),
    )

    qtot = qtotal[system_id]
    mz2_val = mz2[system_id, 2]
    wmz_val = weighted_mz[system_id, 2]
    wmz2_val = weighted_mz2[system_id, 2]
    if axis_idx == wp.int32(0):
        mz2_val = mz2[system_id, 0]
        wmz_val = weighted_mz[system_id, 0]
        wmz2_val = weighted_mz2[system_id, 0]
    elif axis_idx == wp.int32(1):
        mz2_val = mz2[system_id, 1]
        wmz_val = weighted_mz[system_id, 1]
        wmz2_val = weighted_mz2[system_id, 1]
    wqtot = weighted_qtotal[system_id]
    weight = weights[atom_idx]
    height_sq = slab_height_sq[system_id]

    s_val = wmz_val + weight * mz_val - z_f64 * (wqtot + weight * qtot)
    d_e_dz = (TWOPI / vol) * q_f64 * s_val
    grad_positions[atom_idx] = type(grad_positions[0])(
        wp.vec3d(d_e_dz * n64[0], d_e_dz * n64[1], d_e_dz * n64[2])
    )

    charge_bracket = z_f64 * (wmz_val + weight * mz_val)
    charge_bracket = charge_bracket - wp.float64(0.5) * (
        z_f64 * z_f64 * wqtot
        + weight * mz2_val
        + wmz2_val
        + qtot * weight * z_f64 * z_f64
    )
    charge_bracket = charge_bracket - (
        height_sq * (wqtot + weight * qtot) / wp.float64(12.0)
    )
    grad_charges[atom_idx] = (TWOPI / vol) * charge_bracket

    pos = positions[atom_idx]
    pos64 = wp.vec3d(
        wp.float64(pos[0]),
        wp.float64(pos[1]),
        wp.float64(pos[2]),
    )
    wp.atomic_add(grad_normal, system_id, d_e_dz * pos64)


@wp.kernel
def _slab_correction_weighted_backward_cell_kernel(
    pbc: wp.array2d(dtype=wp.bool),
    cell: wp.array(dtype=Any),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    weighted_mz: wp.array2d(dtype=wp.float64),
    weighted_mz2: wp.array2d(dtype=wp.float64),
    weighted_qtotal: wp.array(dtype=wp.float64),
    grad_normal: wp.array(dtype=wp.vec3d),
    grad_cell: wp.array(dtype=Any),
):
    """Convert weighted slab VJP normal/volume terms into cell gradients.

    Thread launch
    -------------
    One thread per system.

    Modifies
    --------
    ``grad_cell``.
    """
    system_id = wp.tid()
    p0 = pbc[system_id, 0]
    p1 = pbc[system_id, 1]
    p2 = pbc[system_id, 2]

    axis_idx = wp.int32(2)
    is_slab = False
    if (not p0) and p1 and p2:
        axis_idx = wp.int32(0)
        is_slab = True
    elif p0 and (not p1) and p2:
        axis_idx = wp.int32(1)
        is_slab = True
    elif p0 and p1 and (not p2):
        axis_idx = wp.int32(2)
        is_slab = True

    zero_mat = wp.mat33d(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    if not is_slab:
        grad_cell[system_id] = type(grad_cell[0])(zero_mat)
        return

    cell_b = cell[system_id]
    h0 = cell_b[0]
    h1 = cell_b[1]
    h2 = cell_b[2]
    periodic_a = h0
    periodic_b = h1
    nonperiodic_c = h2
    if axis_idx == wp.int32(0):
        periodic_a = h1
        periodic_b = h2
        nonperiodic_c = h0
    elif axis_idx == wp.int32(1):
        periodic_a = h2
        periodic_b = h0
        nonperiodic_c = h1

    periodic_a64 = wp.vec3d(
        wp.float64(periodic_a[0]),
        wp.float64(periodic_a[1]),
        wp.float64(periodic_a[2]),
    )
    periodic_b64 = wp.vec3d(
        wp.float64(periodic_b[0]),
        wp.float64(periodic_b[1]),
        wp.float64(periodic_b[2]),
    )
    normal_raw = wp.cross(periodic_a64, periodic_b64)
    area = wp.length(normal_raw)
    n64 = normal_raw / area
    det_h = wp.determinant(cell_b)
    det_sign = wp.float64(1.0)
    if wp.float64(det_h) < wp.float64(0.0):
        det_sign = wp.float64(-1.0)
    vol = wp.abs(wp.float64(det_h))

    c64 = wp.vec3d(
        wp.float64(nonperiodic_c[0]),
        wp.float64(nonperiodic_c[1]),
        wp.float64(nonperiodic_c[2]),
    )
    l_f64 = wp.dot(c64, n64)
    height_sq = l_f64 * l_f64

    mz_val = mz[system_id, 2]
    mz2_val = mz2[system_id, 2]
    wmz_val = weighted_mz[system_id, 2]
    wmz2_val = weighted_mz2[system_id, 2]
    if axis_idx == wp.int32(0):
        mz_val = mz[system_id, 0]
        mz2_val = mz2[system_id, 0]
        wmz_val = weighted_mz[system_id, 0]
        wmz2_val = weighted_mz2[system_id, 0]
    elif axis_idx == wp.int32(1):
        mz_val = mz[system_id, 1]
        mz2_val = mz2[system_id, 1]
        wmz_val = weighted_mz[system_id, 1]
        wmz2_val = weighted_mz2[system_id, 1]
    qtot = qtotal[system_id]
    wqtot = weighted_qtotal[system_id]

    energy_inner = mz_val * wmz_val
    energy_inner = energy_inner - wp.float64(0.5) * mz2_val * wqtot
    energy_inner = energy_inner - wp.float64(0.5) * qtot * wmz2_val
    energy_inner = energy_inner - qtot * wqtot * height_sq / wp.float64(12.0)
    d_e_dv = -(TWOPI * energy_inner) / (vol * vol)
    d_e_dh = -((TWOPI / vol) * qtot * wqtot) / wp.float64(12.0)

    grad_n = grad_normal[system_id] + wp.float64(2.0) * d_e_dh * l_f64 * c64
    grad_c_extra = wp.float64(2.0) * d_e_dh * l_f64 * n64

    dot_ng = wp.dot(n64, grad_n)
    proj = grad_n - n64 * dot_ng
    grad_u = proj / wp.float64(area)
    grad_a = wp.cross(periodic_b64, grad_u)
    grad_b = wp.cross(grad_u, periodic_a64)

    grad_det = d_e_dv * det_sign
    h0d = wp.vec3d(wp.float64(h0[0]), wp.float64(h0[1]), wp.float64(h0[2]))
    h1d = wp.vec3d(wp.float64(h1[0]), wp.float64(h1[1]), wp.float64(h1[2]))
    h2d = wp.vec3d(wp.float64(h2[0]), wp.float64(h2[1]), wp.float64(h2[2]))
    vol_g0 = grad_det * wp.cross(h1d, h2d)
    vol_g1 = grad_det * wp.cross(h2d, h0d)
    vol_g2 = grad_det * wp.cross(h0d, h1d)

    row0 = vol_g0
    row1 = vol_g1
    row2 = vol_g2
    if axis_idx == wp.int32(0):
        row0 = row0 + grad_c_extra
        row1 = row1 + grad_a
        row2 = row2 + grad_b
    elif axis_idx == wp.int32(1):
        row0 = row0 + grad_b
        row1 = row1 + grad_c_extra
        row2 = row2 + grad_a
    else:
        row0 = row0 + grad_a
        row1 = row1 + grad_b
        row2 = row2 + grad_c_extra

    grad = wp.mat33d(
        row0[0],
        row0[1],
        row0[2],
        row1[0],
        row1[1],
        row1[2],
        row2[0],
        row2[1],
        row2[2],
    )
    grad_cell[system_id] = type(grad_cell[0])(grad)


@wp.kernel
def _slab_correction_double_backward_atoms_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    h_positions: wp.array(dtype=Any),
    h_charges: wp.array(dtype=wp.float64),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    slab_volume: wp.array(dtype=wp.float64),
    slab_height_sq: wp.array(dtype=wp.float64),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    dmz: wp.array2d(dtype=wp.float64),
    dmz2: wp.array2d(dtype=wp.float64),
    dqtotal: wp.array(dtype=wp.float64),
    dnormal: wp.array(dtype=wp.vec3d),
    dvolume: wp.array(dtype=wp.float64),
    dheight_sq: wp.array(dtype=wp.float64),
    grad_system: wp.array(dtype=wp.float64),
    grad_positions: wp.array(dtype=Any),
    grad_charges: wp.array(dtype=wp.float64),
    grad_normal: wp.array(dtype=wp.vec3d),
    h_grad_normal: wp.array(dtype=wp.vec3d),
):
    """Compute atom HVP terms and normal cotangents for slab double backward.

    Thread launch
    -------------
    One thread per atom.

    Modifies
    --------
    ``grad_positions``, ``grad_charges``, ``grad_normal``, and
    ``h_grad_normal``.
    """
    atom_idx = wp.tid()
    is_slab, system_id, axis_idx, normal, vol, e_slab, bracket, z_f64, q_f64, mz_val = (
        _slab_correction_terms_precomputed(
            atom_idx,
            positions,
            charges,
            batch_idx,
            slab_axis,
            slab_normal,
            slab_volume,
            slab_height_sq,
            mz,
            mz2,
            qtotal,
        )
    )
    if not is_slab:
        grad_positions[atom_idx] = type(grad_positions[0])(
            type(grad_positions[0][0])(0.0),
            type(grad_positions[0][0])(0.0),
            type(grad_positions[0][0])(0.0),
        )
        grad_charges[atom_idx] = wp.float64(0.0)
        return

    n64 = wp.vec3d(
        wp.float64(normal[0]),
        wp.float64(normal[1]),
        wp.float64(normal[2]),
    )
    pos = positions[atom_idx]
    h_pos = h_positions[atom_idx]
    pos64 = wp.vec3d(
        wp.float64(pos[0]),
        wp.float64(pos[1]),
        wp.float64(pos[2]),
    )
    h_pos64 = wp.vec3d(
        wp.float64(h_pos[0]),
        wp.float64(h_pos[1]),
        wp.float64(h_pos[2]),
    )

    qtot = qtotal[system_id]
    dmz_val = dmz[system_id, 2]
    dmz2_val = dmz2[system_id, 2]
    if axis_idx == wp.int32(0):
        dmz_val = dmz[system_id, 0]
        dmz2_val = dmz2[system_id, 0]
    elif axis_idx == wp.int32(1):
        dmz_val = dmz[system_id, 1]
        dmz2_val = dmz2[system_id, 1]
    dqtot = dqtotal[system_id]
    dn = dnormal[system_id]
    dvol = dvolume[system_id]
    dz = wp.dot(h_pos64, n64) + wp.dot(pos64, dn)
    height_sq = slab_height_sq[system_id]

    g = grad_system[system_id]
    hq = h_charges[atom_idx]
    base = mz_val - qtot * z_f64
    common = g * (FOURPI / vol)
    d_e_dz = common * q_f64 * base
    d_d_e_dz = common * (
        hq * base
        + q_f64 * (dmz_val - dqtot * z_f64 - qtot * dz)
        - q_f64 * base * dvol / vol
    )

    grad_pos_f64 = d_d_e_dz * n64 + d_e_dz * dn
    grad_positions[atom_idx] = type(grad_positions[0])(grad_pos_f64)

    dbracket = dz * mz_val + z_f64 * dmz_val
    dbracket = dbracket - wp.float64(0.5) * (
        dmz2_val + dqtot * z_f64 * z_f64 + wp.float64(2.0) * qtot * z_f64 * dz
    )
    dbracket = dbracket - (
        dqtot * height_sq + qtot * dheight_sq[system_id]
    ) / wp.float64(12.0)
    grad_charges[atom_idx] = common * (dbracket - bracket * dvol / vol)

    wp.atomic_add(grad_normal, system_id, d_e_dz * pos64)
    wp.atomic_add(h_grad_normal, system_id, d_d_e_dz * pos64 + d_e_dz * h_pos64)


@wp.kernel
def _slab_correction_double_backward_cell_kernel(
    pbc: wp.array2d(dtype=wp.bool),
    cell: wp.array(dtype=Any),
    h_cell: wp.array(dtype=Any),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    dmz: wp.array2d(dtype=wp.float64),
    dmz2: wp.array2d(dtype=wp.float64),
    dqtotal: wp.array(dtype=wp.float64),
    grad_system: wp.array(dtype=wp.float64),
    grad_normal: wp.array(dtype=wp.vec3d),
    h_grad_normal: wp.array(dtype=wp.vec3d),
    grad_cell: wp.array(dtype=Any),
):
    """Compute literal-cell slab HVP terms.

    Thread launch
    -------------
    One thread per system.

    Modifies
    --------
    ``grad_cell``.
    """
    system_id = wp.tid()
    p0 = pbc[system_id, 0]
    p1 = pbc[system_id, 1]
    p2 = pbc[system_id, 2]

    axis_idx = wp.int32(2)
    is_slab = False
    if (not p0) and p1 and p2:
        axis_idx = wp.int32(0)
        is_slab = True
    elif p0 and (not p1) and p2:
        axis_idx = wp.int32(1)
        is_slab = True
    elif p0 and p1 and (not p2):
        axis_idx = wp.int32(2)
        is_slab = True

    zero_mat = wp.mat33d(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    if not is_slab:
        grad_cell[system_id] = type(grad_cell[0])(zero_mat)
        return

    cell_b = cell[system_id]
    h_cell_b = h_cell[system_id]
    h0 = cell_b[0]
    h1 = cell_b[1]
    h2 = cell_b[2]
    hh0 = h_cell_b[0]
    hh1 = h_cell_b[1]
    hh2 = h_cell_b[2]

    h0d = wp.vec3d(wp.float64(h0[0]), wp.float64(h0[1]), wp.float64(h0[2]))
    h1d = wp.vec3d(wp.float64(h1[0]), wp.float64(h1[1]), wp.float64(h1[2]))
    h2d = wp.vec3d(wp.float64(h2[0]), wp.float64(h2[1]), wp.float64(h2[2]))
    hh0d = wp.vec3d(wp.float64(hh0[0]), wp.float64(hh0[1]), wp.float64(hh0[2]))
    hh1d = wp.vec3d(wp.float64(hh1[0]), wp.float64(hh1[1]), wp.float64(hh1[2]))
    hh2d = wp.vec3d(wp.float64(hh2[0]), wp.float64(hh2[1]), wp.float64(hh2[2]))

    periodic_a = h0d
    periodic_b = h1d
    nonperiodic_c = h2d
    h_periodic_a = hh0d
    h_periodic_b = hh1d
    h_nonperiodic_c = hh2d
    if axis_idx == wp.int32(0):
        periodic_a = h1d
        periodic_b = h2d
        nonperiodic_c = h0d
        h_periodic_a = hh1d
        h_periodic_b = hh2d
        h_nonperiodic_c = hh0d
    elif axis_idx == wp.int32(1):
        periodic_a = h2d
        periodic_b = h0d
        nonperiodic_c = h1d
        h_periodic_a = hh2d
        h_periodic_b = hh0d
        h_nonperiodic_c = hh1d

    normal_raw = wp.cross(periodic_a, periodic_b)
    area = wp.length(normal_raw)
    normal = normal_raw / area
    d_normal_raw = wp.cross(h_periodic_a, periodic_b) + wp.cross(
        periodic_a, h_periodic_b
    )
    d_area = wp.dot(normal, d_normal_raw)
    d_normal = (d_normal_raw - normal * d_area) / area

    det_h = wp.determinant(cell_b)
    det_sign = wp.float64(1.0)
    if wp.float64(det_h) < wp.float64(0.0):
        det_sign = wp.float64(-1.0)
    vol = wp.abs(wp.float64(det_h))
    d_det = wp.dot(hh0d, wp.cross(h1d, h2d))
    d_det = d_det + wp.dot(h0d, wp.cross(hh1d, h2d))
    d_det = d_det + wp.dot(h0d, wp.cross(h1d, hh2d))
    d_vol = det_sign * d_det
    l_f64 = wp.dot(nonperiodic_c, normal)
    d_l = wp.dot(h_nonperiodic_c, normal) + wp.dot(nonperiodic_c, d_normal)
    height_sq = l_f64 * l_f64

    mz_val = mz[system_id, 2]
    mz2_val = mz2[system_id, 2]
    dmz_val = dmz[system_id, 2]
    dmz2_val = dmz2[system_id, 2]
    if axis_idx == wp.int32(0):
        mz_val = mz[system_id, 0]
        mz2_val = mz2[system_id, 0]
        dmz_val = dmz[system_id, 0]
        dmz2_val = dmz2[system_id, 0]
    elif axis_idx == wp.int32(1):
        mz_val = mz[system_id, 1]
        mz2_val = mz2[system_id, 1]
        dmz_val = dmz[system_id, 1]
        dmz2_val = dmz2[system_id, 1]
    qtot = qtotal[system_id]
    dqtot = dqtotal[system_id]
    g = grad_system[system_id]

    energy_inner = mz_val * mz_val - qtot * mz2_val
    energy_inner = energy_inner - qtot * qtot * height_sq / wp.float64(12.0)
    d_energy_inner = wp.float64(2.0) * mz_val * dmz_val - dqtot * mz2_val
    d_energy_inner = d_energy_inner - qtot * dmz2_val
    d_energy_inner = d_energy_inner - (
        wp.float64(2.0) * qtot * dqtot * height_sq
        + qtot * qtot * wp.float64(2.0) * l_f64 * d_l
    ) / wp.float64(12.0)

    d_e_dv = -(g * TWOPI * energy_inner) / (vol * vol)
    h_d_e_dv = -(
        g
        * TWOPI
        * (
            d_energy_inner / (vol * vol)
            - wp.float64(2.0) * energy_inner * d_vol / (vol * vol * vol)
        )
    )
    d_e_dh = -(g * (TWOPI / vol) * qtot * qtot) / wp.float64(12.0)
    h_d_e_dh = -(
        g
        * (TWOPI / wp.float64(12.0))
        * (wp.float64(2.0) * qtot * dqtot / vol - qtot * qtot * d_vol / (vol * vol))
    )

    grad_n = grad_normal[system_id] + wp.float64(2.0) * d_e_dh * l_f64 * nonperiodic_c
    h_grad_n = h_grad_normal[system_id] + wp.float64(2.0) * (
        h_d_e_dh * l_f64 * nonperiodic_c
        + d_e_dh * d_l * nonperiodic_c
        + d_e_dh * l_f64 * h_nonperiodic_c
    )
    h_grad_c_extra = wp.float64(2.0) * (
        h_d_e_dh * l_f64 * normal + d_e_dh * d_l * normal + d_e_dh * l_f64 * d_normal
    )

    dot_ng = wp.dot(normal, grad_n)
    h_dot_ng = wp.dot(d_normal, grad_n) + wp.dot(normal, h_grad_n)
    proj = grad_n - normal * dot_ng
    h_proj = h_grad_n - d_normal * dot_ng - normal * h_dot_ng
    grad_u = proj / area
    h_grad_u = h_proj / area - proj * d_area / (area * area)
    h_grad_a = wp.cross(h_periodic_b, grad_u) + wp.cross(periodic_b, h_grad_u)
    h_grad_b = wp.cross(h_grad_u, periodic_a) + wp.cross(grad_u, h_periodic_a)

    grad_det = d_e_dv * det_sign
    h_grad_det = h_d_e_dv * det_sign
    vol_cross0 = wp.cross(h1d, h2d)
    vol_cross1 = wp.cross(h2d, h0d)
    vol_cross2 = wp.cross(h0d, h1d)
    h_vol_g0 = h_grad_det * vol_cross0 + grad_det * (
        wp.cross(hh1d, h2d) + wp.cross(h1d, hh2d)
    )
    h_vol_g1 = h_grad_det * vol_cross1 + grad_det * (
        wp.cross(hh2d, h0d) + wp.cross(h2d, hh0d)
    )
    h_vol_g2 = h_grad_det * vol_cross2 + grad_det * (
        wp.cross(hh0d, h1d) + wp.cross(h0d, hh1d)
    )

    row0 = h_vol_g0
    row1 = h_vol_g1
    row2 = h_vol_g2
    if axis_idx == wp.int32(0):
        row0 = row0 + h_grad_c_extra
        row1 = row1 + h_grad_a
        row2 = row2 + h_grad_b
    elif axis_idx == wp.int32(1):
        row0 = row0 + h_grad_b
        row1 = row1 + h_grad_c_extra
        row2 = row2 + h_grad_a
    else:
        row0 = row0 + h_grad_a
        row1 = row1 + h_grad_b
        row2 = row2 + h_grad_c_extra

    grad = wp.mat33d(
        row0[0],
        row0[1],
        row0[2],
        row1[0],
        row1[1],
        row1[2],
        row2[0],
        row2[1],
        row2[2],
    )
    grad_cell[system_id] = type(grad_cell[0])(grad)


@wp.kernel
def _slab_correction_weighted_double_backward_atoms_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    weights: wp.array(dtype=wp.float64),
    h_positions: wp.array(dtype=Any),
    h_charges: wp.array(dtype=wp.float64),
    batch_idx: wp.array(dtype=wp.int32),
    slab_axis: wp.array(dtype=wp.int32),
    slab_normal: wp.array(dtype=wp.vec3d),
    slab_volume: wp.array(dtype=wp.float64),
    slab_height_sq: wp.array(dtype=wp.float64),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    weighted_mz: wp.array2d(dtype=wp.float64),
    weighted_mz2: wp.array2d(dtype=wp.float64),
    weighted_qtotal: wp.array(dtype=wp.float64),
    dmz: wp.array2d(dtype=wp.float64),
    dmz2: wp.array2d(dtype=wp.float64),
    dqtotal: wp.array(dtype=wp.float64),
    d_weighted_mz: wp.array2d(dtype=wp.float64),
    d_weighted_mz2: wp.array2d(dtype=wp.float64),
    d_weighted_qtotal: wp.array(dtype=wp.float64),
    dnormal: wp.array(dtype=wp.vec3d),
    dvolume: wp.array(dtype=wp.float64),
    dheight_sq: wp.array(dtype=wp.float64),
    grad_grad_energy: wp.array(dtype=wp.float64),
    grad_positions: wp.array(dtype=Any),
    grad_charges: wp.array(dtype=wp.float64),
    grad_normal: wp.array(dtype=wp.vec3d),
    h_grad_normal: wp.array(dtype=wp.vec3d),
):
    """Compute atom HVP terms for non-uniform slab cotangents.

    Thread launch
    -------------
    One thread per atom.

    Modifies
    --------
    ``grad_grad_energy``, ``grad_positions``, ``grad_charges``,
    ``grad_normal``, and ``h_grad_normal``.
    """
    atom_idx = wp.tid()
    is_slab, system_id, axis_idx, normal, vol, e_slab, bracket, z_f64, q_f64, mz_val = (
        _slab_correction_terms_precomputed(
            atom_idx,
            positions,
            charges,
            batch_idx,
            slab_axis,
            slab_normal,
            slab_volume,
            slab_height_sq,
            mz,
            mz2,
            qtotal,
        )
    )
    if not is_slab:
        grad_grad_energy[atom_idx] = wp.float64(0.0)
        grad_positions[atom_idx] = type(grad_positions[0])(
            type(grad_positions[0][0])(0.0),
            type(grad_positions[0][0])(0.0),
            type(grad_positions[0][0])(0.0),
        )
        grad_charges[atom_idx] = wp.float64(0.0)
        return

    n64 = wp.vec3d(
        wp.float64(normal[0]),
        wp.float64(normal[1]),
        wp.float64(normal[2]),
    )
    pos = positions[atom_idx]
    h_pos = h_positions[atom_idx]
    pos64 = wp.vec3d(
        wp.float64(pos[0]),
        wp.float64(pos[1]),
        wp.float64(pos[2]),
    )
    h_pos64 = wp.vec3d(
        wp.float64(h_pos[0]),
        wp.float64(h_pos[1]),
        wp.float64(h_pos[2]),
    )

    qtot = qtotal[system_id]
    mz2_val = mz2[system_id, 2]
    dmz_val = dmz[system_id, 2]
    dmz2_val = dmz2[system_id, 2]
    wmz_val = weighted_mz[system_id, 2]
    wmz2_val = weighted_mz2[system_id, 2]
    dwmz_val = d_weighted_mz[system_id, 2]
    dwmz2_val = d_weighted_mz2[system_id, 2]
    if axis_idx == wp.int32(0):
        mz2_val = mz2[system_id, 0]
        dmz_val = dmz[system_id, 0]
        dmz2_val = dmz2[system_id, 0]
        wmz_val = weighted_mz[system_id, 0]
        wmz2_val = weighted_mz2[system_id, 0]
        dwmz_val = d_weighted_mz[system_id, 0]
        dwmz2_val = d_weighted_mz2[system_id, 0]
    elif axis_idx == wp.int32(1):
        mz2_val = mz2[system_id, 1]
        dmz_val = dmz[system_id, 1]
        dmz2_val = dmz2[system_id, 1]
        wmz_val = weighted_mz[system_id, 1]
        wmz2_val = weighted_mz2[system_id, 1]
        dwmz_val = d_weighted_mz[system_id, 1]
        dwmz2_val = d_weighted_mz2[system_id, 1]

    dqtot = dqtotal[system_id]
    wqtot = weighted_qtotal[system_id]
    dwqtot = d_weighted_qtotal[system_id]
    dn = dnormal[system_id]
    dvol = dvolume[system_id]
    dz = wp.dot(h_pos64, n64) + wp.dot(pos64, dn)
    height_sq = slab_height_sq[system_id]
    dheight = dheight_sq[system_id]
    weight = weights[atom_idx]
    hq = h_charges[atom_idx]

    base_s = wmz_val + weight * mz_val - z_f64 * (wqtot + weight * qtot)
    d_s = dwmz_val + weight * dmz_val
    d_s = d_s - dz * (wqtot + weight * qtot)
    d_s = d_s - z_f64 * (dwqtot + weight * dqtot)
    common = TWOPI / vol
    d_e_dz = common * q_f64 * base_s
    d_d_e_dz = common * (hq * base_s + q_f64 * d_s - q_f64 * base_s * dvol / vol)

    grad_positions[atom_idx] = type(grad_positions[0])(
        wp.vec3d(
            d_d_e_dz * n64[0] + d_e_dz * dn[0],
            d_d_e_dz * n64[1] + d_e_dz * dn[1],
            d_d_e_dz * n64[2] + d_e_dz * dn[2],
        )
    )

    charge_bracket = z_f64 * (wmz_val + weight * mz_val)
    charge_bracket = charge_bracket - wp.float64(0.5) * (
        z_f64 * z_f64 * wqtot
        + weight * mz2_val
        + wmz2_val
        + qtot * weight * z_f64 * z_f64
    )
    charge_bracket = charge_bracket - (
        height_sq * (wqtot + weight * qtot) / wp.float64(12.0)
    )
    d_charge_bracket = dz * (wmz_val + weight * mz_val)
    d_charge_bracket = d_charge_bracket + z_f64 * (dwmz_val + weight * dmz_val)
    d_charge_bracket = d_charge_bracket - wp.float64(0.5) * (
        wp.float64(2.0) * z_f64 * dz * wqtot
        + z_f64 * z_f64 * dwqtot
        + weight * dmz2_val
        + dwmz2_val
        + dqtot * weight * z_f64 * z_f64
        + qtot * weight * wp.float64(2.0) * z_f64 * dz
    )
    d_charge_bracket = d_charge_bracket - (
        dheight * (wqtot + weight * qtot) + height_sq * (dwqtot + weight * dqtot)
    ) / wp.float64(12.0)
    grad_charges[atom_idx] = common * (d_charge_bracket - charge_bracket * dvol / vol)

    dbracket = dz * mz_val + z_f64 * dmz_val
    dbracket = dbracket - wp.float64(0.5) * (
        dmz2_val + dqtot * z_f64 * z_f64 + wp.float64(2.0) * qtot * z_f64 * dz
    )
    dbracket = dbracket - (dqtot * height_sq + qtot * dheight) / wp.float64(12.0)
    grad_grad_energy[atom_idx] = common * (
        hq * bracket + q_f64 * dbracket - q_f64 * bracket * dvol / vol
    )

    wp.atomic_add(grad_normal, system_id, d_e_dz * pos64)
    wp.atomic_add(h_grad_normal, system_id, d_d_e_dz * pos64 + d_e_dz * h_pos64)


@wp.kernel
def _slab_correction_weighted_double_backward_cell_kernel(
    pbc: wp.array2d(dtype=wp.bool),
    cell: wp.array(dtype=Any),
    h_cell: wp.array(dtype=Any),
    mz: wp.array2d(dtype=wp.float64),
    mz2: wp.array2d(dtype=wp.float64),
    qtotal: wp.array(dtype=wp.float64),
    weighted_mz: wp.array2d(dtype=wp.float64),
    weighted_mz2: wp.array2d(dtype=wp.float64),
    weighted_qtotal: wp.array(dtype=wp.float64),
    dmz: wp.array2d(dtype=wp.float64),
    dmz2: wp.array2d(dtype=wp.float64),
    dqtotal: wp.array(dtype=wp.float64),
    d_weighted_mz: wp.array2d(dtype=wp.float64),
    d_weighted_mz2: wp.array2d(dtype=wp.float64),
    d_weighted_qtotal: wp.array(dtype=wp.float64),
    grad_normal: wp.array(dtype=wp.vec3d),
    h_grad_normal: wp.array(dtype=wp.vec3d),
    grad_cell: wp.array(dtype=Any),
):
    """Compute weighted literal-cell slab HVP terms.

    Thread launch
    -------------
    One thread per system.

    Modifies
    --------
    ``grad_cell``.
    """
    system_id = wp.tid()
    p0 = pbc[system_id, 0]
    p1 = pbc[system_id, 1]
    p2 = pbc[system_id, 2]

    axis_idx = wp.int32(2)
    is_slab = False
    if (not p0) and p1 and p2:
        axis_idx = wp.int32(0)
        is_slab = True
    elif p0 and (not p1) and p2:
        axis_idx = wp.int32(1)
        is_slab = True
    elif p0 and p1 and (not p2):
        axis_idx = wp.int32(2)
        is_slab = True

    zero_mat = wp.mat33d(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    if not is_slab:
        grad_cell[system_id] = type(grad_cell[0])(zero_mat)
        return

    cell_b = cell[system_id]
    h_cell_b = h_cell[system_id]
    h0 = cell_b[0]
    h1 = cell_b[1]
    h2 = cell_b[2]
    hh0 = h_cell_b[0]
    hh1 = h_cell_b[1]
    hh2 = h_cell_b[2]

    h0d = wp.vec3d(wp.float64(h0[0]), wp.float64(h0[1]), wp.float64(h0[2]))
    h1d = wp.vec3d(wp.float64(h1[0]), wp.float64(h1[1]), wp.float64(h1[2]))
    h2d = wp.vec3d(wp.float64(h2[0]), wp.float64(h2[1]), wp.float64(h2[2]))
    hh0d = wp.vec3d(wp.float64(hh0[0]), wp.float64(hh0[1]), wp.float64(hh0[2]))
    hh1d = wp.vec3d(wp.float64(hh1[0]), wp.float64(hh1[1]), wp.float64(hh1[2]))
    hh2d = wp.vec3d(wp.float64(hh2[0]), wp.float64(hh2[1]), wp.float64(hh2[2]))

    periodic_a = h0d
    periodic_b = h1d
    nonperiodic_c = h2d
    h_periodic_a = hh0d
    h_periodic_b = hh1d
    h_nonperiodic_c = hh2d
    if axis_idx == wp.int32(0):
        periodic_a = h1d
        periodic_b = h2d
        nonperiodic_c = h0d
        h_periodic_a = hh1d
        h_periodic_b = hh2d
        h_nonperiodic_c = hh0d
    elif axis_idx == wp.int32(1):
        periodic_a = h2d
        periodic_b = h0d
        nonperiodic_c = h1d
        h_periodic_a = hh2d
        h_periodic_b = hh0d
        h_nonperiodic_c = hh1d

    normal_raw = wp.cross(periodic_a, periodic_b)
    area = wp.length(normal_raw)
    normal = normal_raw / area
    d_normal_raw = wp.cross(h_periodic_a, periodic_b) + wp.cross(
        periodic_a, h_periodic_b
    )
    d_area = wp.dot(normal, d_normal_raw)
    d_normal = (d_normal_raw - normal * d_area) / area

    det_h = wp.determinant(cell_b)
    det_sign = wp.float64(1.0)
    if wp.float64(det_h) < wp.float64(0.0):
        det_sign = wp.float64(-1.0)
    vol = wp.abs(wp.float64(det_h))
    d_det = wp.dot(hh0d, wp.cross(h1d, h2d))
    d_det = d_det + wp.dot(h0d, wp.cross(hh1d, h2d))
    d_det = d_det + wp.dot(h0d, wp.cross(h1d, hh2d))
    d_vol = det_sign * d_det
    l_f64 = wp.dot(nonperiodic_c, normal)
    d_l = wp.dot(h_nonperiodic_c, normal) + wp.dot(nonperiodic_c, d_normal)
    height_sq = l_f64 * l_f64

    mz_val = mz[system_id, 2]
    mz2_val = mz2[system_id, 2]
    dmz_val = dmz[system_id, 2]
    dmz2_val = dmz2[system_id, 2]
    wmz_val = weighted_mz[system_id, 2]
    wmz2_val = weighted_mz2[system_id, 2]
    dwmz_val = d_weighted_mz[system_id, 2]
    dwmz2_val = d_weighted_mz2[system_id, 2]
    if axis_idx == wp.int32(0):
        mz_val = mz[system_id, 0]
        mz2_val = mz2[system_id, 0]
        dmz_val = dmz[system_id, 0]
        dmz2_val = dmz2[system_id, 0]
        wmz_val = weighted_mz[system_id, 0]
        wmz2_val = weighted_mz2[system_id, 0]
        dwmz_val = d_weighted_mz[system_id, 0]
        dwmz2_val = d_weighted_mz2[system_id, 0]
    elif axis_idx == wp.int32(1):
        mz_val = mz[system_id, 1]
        mz2_val = mz2[system_id, 1]
        dmz_val = dmz[system_id, 1]
        dmz2_val = dmz2[system_id, 1]
        wmz_val = weighted_mz[system_id, 1]
        wmz2_val = weighted_mz2[system_id, 1]
        dwmz_val = d_weighted_mz[system_id, 1]
        dwmz2_val = d_weighted_mz2[system_id, 1]
    qtot = qtotal[system_id]
    dqtot = dqtotal[system_id]
    wqtot = weighted_qtotal[system_id]
    dwqtot = d_weighted_qtotal[system_id]

    energy_inner = mz_val * wmz_val
    energy_inner = energy_inner - wp.float64(0.5) * mz2_val * wqtot
    energy_inner = energy_inner - wp.float64(0.5) * qtot * wmz2_val
    energy_inner = energy_inner - qtot * wqtot * height_sq / wp.float64(12.0)
    d_energy_inner = dmz_val * wmz_val + mz_val * dwmz_val
    d_energy_inner = d_energy_inner - wp.float64(0.5) * (
        dmz2_val * wqtot + mz2_val * dwqtot
    )
    d_energy_inner = d_energy_inner - wp.float64(0.5) * (
        dqtot * wmz2_val + qtot * dwmz2_val
    )
    d_energy_inner = d_energy_inner - (
        (dqtot * wqtot + qtot * dwqtot) * height_sq
        + qtot * wqtot * wp.float64(2.0) * l_f64 * d_l
    ) / wp.float64(12.0)

    d_e_dv = -(TWOPI * energy_inner) / (vol * vol)
    h_d_e_dv = -(
        TWOPI
        * (
            d_energy_inner / (vol * vol)
            - wp.float64(2.0) * energy_inner * d_vol / (vol * vol * vol)
        )
    )
    d_e_dh = -((TWOPI / vol) * qtot * wqtot) / wp.float64(12.0)
    h_d_e_dh = -(
        (TWOPI / wp.float64(12.0))
        * ((dqtot * wqtot + qtot * dwqtot) / vol - qtot * wqtot * d_vol / (vol * vol))
    )

    grad_n = grad_normal[system_id] + wp.float64(2.0) * d_e_dh * l_f64 * nonperiodic_c
    h_grad_n = h_grad_normal[system_id] + wp.float64(2.0) * (
        h_d_e_dh * l_f64 * nonperiodic_c
        + d_e_dh * d_l * nonperiodic_c
        + d_e_dh * l_f64 * h_nonperiodic_c
    )
    h_grad_c_extra = wp.float64(2.0) * (
        h_d_e_dh * l_f64 * normal + d_e_dh * d_l * normal + d_e_dh * l_f64 * d_normal
    )

    dot_ng = wp.dot(normal, grad_n)
    h_dot_ng = wp.dot(d_normal, grad_n) + wp.dot(normal, h_grad_n)
    proj = grad_n - normal * dot_ng
    h_proj = h_grad_n - d_normal * dot_ng - normal * h_dot_ng
    grad_u = proj / area
    h_grad_u = h_proj / area - proj * d_area / (area * area)
    h_grad_a = wp.cross(h_periodic_b, grad_u) + wp.cross(periodic_b, h_grad_u)
    h_grad_b = wp.cross(h_grad_u, periodic_a) + wp.cross(grad_u, h_periodic_a)

    grad_det = d_e_dv * det_sign
    h_grad_det = h_d_e_dv * det_sign
    vol_cross0 = wp.cross(h1d, h2d)
    vol_cross1 = wp.cross(h2d, h0d)
    vol_cross2 = wp.cross(h0d, h1d)
    h_vol_g0 = h_grad_det * vol_cross0 + grad_det * (
        wp.cross(hh1d, h2d) + wp.cross(h1d, hh2d)
    )
    h_vol_g1 = h_grad_det * vol_cross1 + grad_det * (
        wp.cross(hh2d, h0d) + wp.cross(h2d, hh0d)
    )
    h_vol_g2 = h_grad_det * vol_cross2 + grad_det * (
        wp.cross(hh0d, h1d) + wp.cross(h0d, hh1d)
    )

    row0 = h_vol_g0
    row1 = h_vol_g1
    row2 = h_vol_g2
    if axis_idx == wp.int32(0):
        row0 = row0 + h_grad_c_extra
        row1 = row1 + h_grad_a
        row2 = row2 + h_grad_b
    elif axis_idx == wp.int32(1):
        row0 = row0 + h_grad_b
        row1 = row1 + h_grad_c_extra
        row2 = row2 + h_grad_a
    else:
        row0 = row0 + h_grad_a
        row1 = row1 + h_grad_b
        row2 = row2 + h_grad_c_extra

    grad = wp.mat33d(
        row0[0],
        row0[1],
        row0[2],
        row1[0],
        row1[1],
        row1[2],
        row2[0],
        row2[1],
        row2[2],
    )
    grad_cell[system_id] = type(grad_cell[0])(grad)


###########################################################################################
########################### Overload Registration #########################################
###########################################################################################

# Type aliases (matching ewald_kernels.py convention)
_T = [wp.float32, wp.float64]
_V = [wp.vec3f, wp.vec3d]
_M = [wp.mat33f, wp.mat33d]

# Overload dictionaries
_slab_reduce_moments_kernel_overload = {}
_slab_precompute_geometry_kernel_overload = {}
_slab_correction_energy_kernel_overload = {}
_slab_correction_energy_forces_kernel_overload = {}
_slab_correction_energy_forces_virial_kernel_overload = {}
_slab_correction_energy_virial_kernel_overload = {}
_slab_correction_energy_forces_charge_grad_kernel_overload = {}
_slab_correction_energy_forces_charge_grad_virial_kernel_overload = {}
_slab_correction_energy_charge_grad_kernel_overload = {}
_slab_correction_energy_charge_grad_virial_kernel_overload = {}
_slab_correction_backward_atoms_kernel_overload = {}
_slab_correction_backward_cell_kernel_overload = {}
_slab_directional_geometry_kernel_overload = {}
_slab_directional_moments_kernel_overload = {}
_slab_weighted_moments_kernel_overload = {}
_slab_directional_weighted_moments_kernel_overload = {}
_slab_correction_weighted_backward_atoms_kernel_overload = {}
_slab_correction_weighted_backward_cell_kernel_overload = {}
_slab_correction_double_backward_atoms_kernel_overload = {}
_slab_correction_double_backward_cell_kernel_overload = {}
_slab_correction_weighted_double_backward_atoms_kernel_overload = {}
_slab_correction_weighted_double_backward_cell_kernel_overload = {}

for t, v, m in zip(_T, _V, _M):
    _slab_reduce_moments_kernel_overload[t] = wp.overload(
        _slab_reduce_moments_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array2d(dtype=wp.bool),  # pbc
            wp.array(dtype=m),  # cell (mat33)
            wp.array2d(dtype=wp.float64),  # mz (B, 3)
            wp.array2d(dtype=wp.float64),  # mz2 (B, 3)
            wp.array(dtype=wp.float64),  # qtotal
        ],
    )

    _slab_precompute_geometry_kernel_overload[t] = wp.overload(
        _slab_precompute_geometry_kernel,
        [
            wp.array2d(dtype=wp.bool),  # pbc
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.float64),  # slab_volume
            wp.array(dtype=wp.float64),  # slab_height_sq
        ],
    )

    _slab_correction_energy_kernel_overload[t] = wp.overload(
        _slab_correction_energy_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.float64),  # slab_volume
            wp.array(dtype=wp.float64),  # slab_height_sq
            wp.array2d(dtype=wp.float64),  # mz (B, 3)
            wp.array2d(dtype=wp.float64),  # mz2 (B, 3)
            wp.array(dtype=wp.float64),  # qtotal
            wp.array(dtype=wp.float64),  # energy_in
            wp.array(dtype=wp.float64),  # energy_out
        ],
    )

    _slab_correction_energy_forces_kernel_overload[t] = wp.overload(
        _slab_correction_energy_forces_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.float64),  # slab_volume
            wp.array(dtype=wp.float64),  # slab_height_sq
            wp.array2d(dtype=wp.float64),  # mz (B, 3)
            wp.array2d(dtype=wp.float64),  # mz2 (B, 3)
            wp.array(dtype=wp.float64),  # qtotal
            wp.array(dtype=wp.float64),  # energy_in
            wp.array(dtype=wp.float64),  # energy_out
            wp.array(dtype=v),  # forces
        ],
    )

    _slab_correction_energy_forces_virial_kernel_overload[t] = wp.overload(
        _slab_correction_energy_forces_virial_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.float64),  # slab_volume
            wp.array(dtype=wp.float64),  # slab_height_sq
            wp.array2d(dtype=wp.float64),  # mz (B, 3)
            wp.array2d(dtype=wp.float64),  # mz2 (B, 3)
            wp.array(dtype=wp.float64),  # qtotal
            wp.array(dtype=wp.float64),  # energy_in
            wp.array(dtype=wp.float64),  # energy_out
            wp.array(dtype=v),  # forces
            wp.array(dtype=m),  # virial
        ],
    )

    _slab_correction_energy_virial_kernel_overload[t] = wp.overload(
        _slab_correction_energy_virial_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.float64),  # slab_volume
            wp.array(dtype=wp.float64),  # slab_height_sq
            wp.array2d(dtype=wp.float64),  # mz (B, 3)
            wp.array2d(dtype=wp.float64),  # mz2 (B, 3)
            wp.array(dtype=wp.float64),  # qtotal
            wp.array(dtype=wp.float64),  # energy_in
            wp.array(dtype=wp.float64),  # energy_out
            wp.array(dtype=m),  # virial
        ],
    )

    _slab_correction_energy_forces_charge_grad_kernel_overload[t] = wp.overload(
        _slab_correction_energy_forces_charge_grad_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.float64),  # slab_volume
            wp.array(dtype=wp.float64),  # slab_height_sq
            wp.array2d(dtype=wp.float64),  # mz (B, 3)
            wp.array2d(dtype=wp.float64),  # mz2 (B, 3)
            wp.array(dtype=wp.float64),  # qtotal
            wp.array(dtype=wp.float64),  # energy_in
            wp.array(dtype=wp.float64),  # energy_out
            wp.array(dtype=v),  # forces
            wp.array(dtype=wp.float64),  # charge_grads
        ],
    )

    _slab_correction_energy_forces_charge_grad_virial_kernel_overload[t] = wp.overload(
        _slab_correction_energy_forces_charge_grad_virial_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.float64),  # slab_volume
            wp.array(dtype=wp.float64),  # slab_height_sq
            wp.array2d(dtype=wp.float64),  # mz (B, 3)
            wp.array2d(dtype=wp.float64),  # mz2 (B, 3)
            wp.array(dtype=wp.float64),  # qtotal
            wp.array(dtype=wp.float64),  # energy_in
            wp.array(dtype=wp.float64),  # energy_out
            wp.array(dtype=v),  # forces
            wp.array(dtype=wp.float64),  # charge_grads
            wp.array(dtype=m),  # virial
        ],
    )

    _slab_correction_energy_charge_grad_kernel_overload[t] = wp.overload(
        _slab_correction_energy_charge_grad_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.float64),  # slab_volume
            wp.array(dtype=wp.float64),  # slab_height_sq
            wp.array2d(dtype=wp.float64),  # mz (B, 3)
            wp.array2d(dtype=wp.float64),  # mz2 (B, 3)
            wp.array(dtype=wp.float64),  # qtotal
            wp.array(dtype=wp.float64),  # energy_in
            wp.array(dtype=wp.float64),  # energy_out
            wp.array(dtype=wp.float64),  # charge_grads
        ],
    )

    _slab_correction_energy_charge_grad_virial_kernel_overload[t] = wp.overload(
        _slab_correction_energy_charge_grad_virial_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.float64),  # slab_volume
            wp.array(dtype=wp.float64),  # slab_height_sq
            wp.array2d(dtype=wp.float64),  # mz (B, 3)
            wp.array2d(dtype=wp.float64),  # mz2 (B, 3)
            wp.array(dtype=wp.float64),  # qtotal
            wp.array(dtype=wp.float64),  # energy_in
            wp.array(dtype=wp.float64),  # energy_out
            wp.array(dtype=wp.float64),  # charge_grads
            wp.array(dtype=m),  # virial
        ],
    )

    _slab_correction_backward_atoms_kernel_overload[t] = wp.overload(
        _slab_correction_backward_atoms_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.float64),  # slab_volume
            wp.array(dtype=wp.float64),  # slab_height_sq
            wp.array2d(dtype=wp.float64),  # mz
            wp.array2d(dtype=wp.float64),  # mz2
            wp.array(dtype=wp.float64),  # qtotal
            wp.array(dtype=wp.float64),  # grad_system
            wp.array(dtype=v),  # grad_positions
            wp.array(dtype=wp.float64),  # grad_charges
            wp.array(dtype=wp.vec3d),  # grad_normal
        ],
    )

    _slab_correction_backward_cell_kernel_overload[t] = wp.overload(
        _slab_correction_backward_cell_kernel,
        [
            wp.array2d(dtype=wp.bool),  # pbc
            wp.array(dtype=m),  # cell
            wp.array2d(dtype=wp.float64),  # mz
            wp.array2d(dtype=wp.float64),  # mz2
            wp.array(dtype=wp.float64),  # qtotal
            wp.array(dtype=wp.float64),  # grad_system
            wp.array(dtype=wp.vec3d),  # grad_normal
            wp.array(dtype=m),  # grad_cell
        ],
    )

    _slab_directional_geometry_kernel_overload[t] = wp.overload(
        _slab_directional_geometry_kernel,
        [
            wp.array2d(dtype=wp.bool),  # pbc
            wp.array(dtype=m),  # cell
            wp.array(dtype=m),  # h_cell
            wp.array(dtype=wp.vec3d),  # dnormal
            wp.array(dtype=wp.float64),  # dvolume
            wp.array(dtype=wp.float64),  # dheight_sq
        ],
    )

    _slab_directional_moments_kernel_overload[t] = wp.overload(
        _slab_directional_moments_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=v),  # h_positions
            wp.array(dtype=wp.float64),  # h_charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.vec3d),  # dnormal
            wp.array2d(dtype=wp.float64),  # dmz
            wp.array2d(dtype=wp.float64),  # dmz2
            wp.array(dtype=wp.float64),  # dqtotal
        ],
    )

    _slab_weighted_moments_kernel_overload[t] = wp.overload(
        _slab_weighted_moments_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.float64),  # weights
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.float64),  # weighted_qtotal
            wp.array2d(dtype=wp.float64),  # weighted_mz
            wp.array2d(dtype=wp.float64),  # weighted_mz2
        ],
    )

    _slab_directional_weighted_moments_kernel_overload[t] = wp.overload(
        _slab_directional_weighted_moments_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.float64),  # weights
            wp.array(dtype=v),  # h_positions
            wp.array(dtype=wp.float64),  # h_charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.vec3d),  # dnormal
            wp.array(dtype=wp.float64),  # d_weighted_qtotal
            wp.array2d(dtype=wp.float64),  # d_weighted_mz
            wp.array2d(dtype=wp.float64),  # d_weighted_mz2
        ],
    )

    _slab_correction_weighted_backward_atoms_kernel_overload[t] = wp.overload(
        _slab_correction_weighted_backward_atoms_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.float64),  # weights
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.float64),  # slab_volume
            wp.array(dtype=wp.float64),  # slab_height_sq
            wp.array2d(dtype=wp.float64),  # mz
            wp.array2d(dtype=wp.float64),  # mz2
            wp.array(dtype=wp.float64),  # qtotal
            wp.array2d(dtype=wp.float64),  # weighted_mz
            wp.array2d(dtype=wp.float64),  # weighted_mz2
            wp.array(dtype=wp.float64),  # weighted_qtotal
            wp.array(dtype=v),  # grad_positions
            wp.array(dtype=wp.float64),  # grad_charges
            wp.array(dtype=wp.vec3d),  # grad_normal
        ],
    )

    _slab_correction_weighted_backward_cell_kernel_overload[t] = wp.overload(
        _slab_correction_weighted_backward_cell_kernel,
        [
            wp.array2d(dtype=wp.bool),  # pbc
            wp.array(dtype=m),  # cell
            wp.array2d(dtype=wp.float64),  # mz
            wp.array2d(dtype=wp.float64),  # mz2
            wp.array(dtype=wp.float64),  # qtotal
            wp.array2d(dtype=wp.float64),  # weighted_mz
            wp.array2d(dtype=wp.float64),  # weighted_mz2
            wp.array(dtype=wp.float64),  # weighted_qtotal
            wp.array(dtype=wp.vec3d),  # grad_normal
            wp.array(dtype=m),  # grad_cell
        ],
    )

    _slab_correction_double_backward_atoms_kernel_overload[t] = wp.overload(
        _slab_correction_double_backward_atoms_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=v),  # h_positions
            wp.array(dtype=wp.float64),  # h_charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.float64),  # slab_volume
            wp.array(dtype=wp.float64),  # slab_height_sq
            wp.array2d(dtype=wp.float64),  # mz
            wp.array2d(dtype=wp.float64),  # mz2
            wp.array(dtype=wp.float64),  # qtotal
            wp.array2d(dtype=wp.float64),  # dmz
            wp.array2d(dtype=wp.float64),  # dmz2
            wp.array(dtype=wp.float64),  # dqtotal
            wp.array(dtype=wp.vec3d),  # dnormal
            wp.array(dtype=wp.float64),  # dvolume
            wp.array(dtype=wp.float64),  # dheight_sq
            wp.array(dtype=wp.float64),  # grad_system
            wp.array(dtype=v),  # grad_positions
            wp.array(dtype=wp.float64),  # grad_charges
            wp.array(dtype=wp.vec3d),  # grad_normal
            wp.array(dtype=wp.vec3d),  # h_grad_normal
        ],
    )

    _slab_correction_weighted_double_backward_atoms_kernel_overload[t] = wp.overload(
        _slab_correction_weighted_double_backward_atoms_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.float64),  # weights
            wp.array(dtype=v),  # h_positions
            wp.array(dtype=wp.float64),  # h_charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # slab_axis
            wp.array(dtype=wp.vec3d),  # slab_normal
            wp.array(dtype=wp.float64),  # slab_volume
            wp.array(dtype=wp.float64),  # slab_height_sq
            wp.array2d(dtype=wp.float64),  # mz
            wp.array2d(dtype=wp.float64),  # mz2
            wp.array(dtype=wp.float64),  # qtotal
            wp.array2d(dtype=wp.float64),  # weighted_mz
            wp.array2d(dtype=wp.float64),  # weighted_mz2
            wp.array(dtype=wp.float64),  # weighted_qtotal
            wp.array2d(dtype=wp.float64),  # dmz
            wp.array2d(dtype=wp.float64),  # dmz2
            wp.array(dtype=wp.float64),  # dqtotal
            wp.array2d(dtype=wp.float64),  # d_weighted_mz
            wp.array2d(dtype=wp.float64),  # d_weighted_mz2
            wp.array(dtype=wp.float64),  # d_weighted_qtotal
            wp.array(dtype=wp.vec3d),  # dnormal
            wp.array(dtype=wp.float64),  # dvolume
            wp.array(dtype=wp.float64),  # dheight_sq
            wp.array(dtype=wp.float64),  # grad_grad_energy
            wp.array(dtype=v),  # grad_positions
            wp.array(dtype=wp.float64),  # grad_charges
            wp.array(dtype=wp.vec3d),  # grad_normal
            wp.array(dtype=wp.vec3d),  # h_grad_normal
        ],
    )

    _slab_correction_weighted_double_backward_cell_kernel_overload[t] = wp.overload(
        _slab_correction_weighted_double_backward_cell_kernel,
        [
            wp.array2d(dtype=wp.bool),  # pbc
            wp.array(dtype=m),  # cell
            wp.array(dtype=m),  # h_cell
            wp.array2d(dtype=wp.float64),  # mz
            wp.array2d(dtype=wp.float64),  # mz2
            wp.array(dtype=wp.float64),  # qtotal
            wp.array2d(dtype=wp.float64),  # weighted_mz
            wp.array2d(dtype=wp.float64),  # weighted_mz2
            wp.array(dtype=wp.float64),  # weighted_qtotal
            wp.array2d(dtype=wp.float64),  # dmz
            wp.array2d(dtype=wp.float64),  # dmz2
            wp.array(dtype=wp.float64),  # dqtotal
            wp.array2d(dtype=wp.float64),  # d_weighted_mz
            wp.array2d(dtype=wp.float64),  # d_weighted_mz2
            wp.array(dtype=wp.float64),  # d_weighted_qtotal
            wp.array(dtype=wp.vec3d),  # grad_normal
            wp.array(dtype=wp.vec3d),  # h_grad_normal
            wp.array(dtype=m),  # grad_cell
        ],
    )

    _slab_correction_double_backward_cell_kernel_overload[t] = wp.overload(
        _slab_correction_double_backward_cell_kernel,
        [
            wp.array2d(dtype=wp.bool),  # pbc
            wp.array(dtype=m),  # cell
            wp.array(dtype=m),  # h_cell
            wp.array2d(dtype=wp.float64),  # mz
            wp.array2d(dtype=wp.float64),  # mz2
            wp.array(dtype=wp.float64),  # qtotal
            wp.array2d(dtype=wp.float64),  # dmz
            wp.array2d(dtype=wp.float64),  # dmz2
            wp.array(dtype=wp.float64),  # dqtotal
            wp.array(dtype=wp.float64),  # grad_system
            wp.array(dtype=wp.vec3d),  # grad_normal
            wp.array(dtype=wp.vec3d),  # h_grad_normal
            wp.array(dtype=m),  # grad_cell
        ],
    )


###########################################################################################
########################### Launcher Functions ############################################
###########################################################################################


def slab_reduce_moments(
    positions: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    pbc: wp.array,
    cell: wp.array,
    mz: wp.array,
    mz2: wp.array,
    qtotal: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch kernel to accumulate slab correction moments.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom.
    pbc : wp.array, shape (B, 3), dtype=wp.bool
        Per-system periodic boundary conditions.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system cell matrices.
    mz : wp.array, shape (B, 3), dtype=wp.float64
        OUTPUT: Projected dipole moment in the non-periodic axis slot.
        Must be zero-initialized.
    mz2 : wp.array, shape (B, 3), dtype=wp.float64
        OUTPUT: Projected second moment in the non-periodic axis slot.
        Must be zero-initialized.
    qtotal : wp.array, shape (B,), dtype=wp.float64
        OUTPUT: Total charge. Must be zero-initialized.
    wp_dtype : type
        Warp scalar type (wp.float32 or wp.float64).
    device : str, optional
        Warp device.
    """
    num_atoms = charges.shape[0]
    if device is None:
        device = str(charges.device)

    wp.launch(
        _slab_reduce_moments_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[positions, charges, batch_idx, pbc, cell, mz, mz2, qtotal],
        device=device,
    )


def slab_precompute_geometry(
    pbc: wp.array,
    cell: wp.array,
    slab_axis: wp.array,
    slab_normal: wp.array,
    slab_volume: wp.array,
    slab_height_sq: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Fill caller-owned per-system slab geometry for atom-major kernels."""
    num_systems = cell.shape[0]
    if device is None:
        device = str(cell.device)
    if num_systems > 0:
        wp.launch(
            _slab_precompute_geometry_kernel_overload[wp_dtype],
            dim=num_systems,
            inputs=[
                pbc,
                cell,
                slab_axis,
                slab_normal,
                slab_volume,
                slab_height_sq,
            ],
            device=device,
        )


def _launch_slab_correction(
    positions: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    pbc: wp.array,
    cell: wp.array,
    mz: wp.array,
    mz2: wp.array,
    qtotal: wp.array,
    slab_axis: wp.array,
    slab_normal: wp.array,
    slab_volume: wp.array,
    slab_height_sq: wp.array,
    energy_in: wp.array,
    energy_out: wp.array,
    wp_dtype: type,
    *,
    forces: wp.array | None = None,
    charge_grads: wp.array | None = None,
    virial: wp.array | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    device: str | None = None,
) -> None:
    """Launch an Ewald-style slab correction kernel for the requested outputs."""
    num_atoms = charges.shape[0]
    if device is None:
        device = str(charges.device)
    kernel_forces = forces if compute_forces else None
    kernel_virial = virial if compute_virial else None
    if compute_forces and kernel_forces is None:
        raise ValueError("forces output is required when compute_forces=True")
    if compute_charge_gradients and charge_grads is None:
        raise ValueError(
            "charge_grads output is required when compute_charge_gradients=True"
        )
    if compute_virial and kernel_virial is None:
        raise ValueError("virial output is required when compute_virial=True")

    common_inputs = [
        positions,
        charges,
        batch_idx,
        slab_axis,
        slab_normal,
        slab_volume,
        slab_height_sq,
        mz,
        mz2,
        qtotal,
        energy_in,
        energy_out,
    ]

    if compute_forces and compute_charge_gradients:
        if compute_virial:
            kernel = _slab_correction_energy_forces_charge_grad_virial_kernel_overload[
                wp_dtype
            ]
            inputs = [*common_inputs, kernel_forces, charge_grads, kernel_virial]
        else:
            kernel = _slab_correction_energy_forces_charge_grad_kernel_overload[
                wp_dtype
            ]
            inputs = [*common_inputs, kernel_forces, charge_grads]
    elif compute_forces:
        if compute_virial:
            kernel = _slab_correction_energy_forces_virial_kernel_overload[wp_dtype]
            inputs = [*common_inputs, kernel_forces, kernel_virial]
        else:
            kernel = _slab_correction_energy_forces_kernel_overload[wp_dtype]
            inputs = [*common_inputs, kernel_forces]
    elif compute_charge_gradients:
        if compute_virial:
            kernel = _slab_correction_energy_charge_grad_virial_kernel_overload[
                wp_dtype
            ]
            inputs = [*common_inputs, charge_grads, kernel_virial]
        else:
            kernel = _slab_correction_energy_charge_grad_kernel_overload[wp_dtype]
            inputs = [*common_inputs, charge_grads]
    elif compute_virial:
        kernel = _slab_correction_energy_virial_kernel_overload[wp_dtype]
        inputs = [*common_inputs, kernel_virial]
    else:
        kernel = _slab_correction_energy_kernel_overload[wp_dtype]
        inputs = common_inputs

    wp.launch(kernel, dim=num_atoms, inputs=inputs, device=device)


def slab_correction(
    positions: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    pbc: wp.array,
    cell: wp.array,
    mz: wp.array,
    mz2: wp.array,
    qtotal: wp.array,
    slab_axis: wp.array,
    slab_normal: wp.array,
    slab_volume: wp.array,
    slab_height_sq: wp.array,
    energy_in: wp.array,
    energy_out: wp.array,
    forces: wp.array,
    charge_grads: wp.array,
    virial: wp.array,
    wp_dtype: type,
    compute_forces: bool = True,
    compute_charge_gradients: bool = True,
    compute_virial: bool = True,
    device: str | None = None,
) -> None:
    """Launch split slab correction kernels for requested outputs.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom.
    pbc : wp.array, shape (B, 3), dtype=wp.bool
        Per-system periodic boundary conditions.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system cell matrices used for cell-gradient and virial geometry.
    mz : wp.array, shape (B, 3), dtype=wp.float64
        Per-system projected dipole moment (from slab_reduce_moments).
    mz2 : wp.array, shape (B, 3), dtype=wp.float64
        Per-system projected second moment (from slab_reduce_moments).
    qtotal : wp.array, shape (B,), dtype=wp.float64
        Per-system total charge (from slab_reduce_moments).
    slab_axis, slab_normal, slab_volume, slab_height_sq
        Caller-owned slab geometry buffers filled by
        :func:`slab_precompute_geometry`.
    energy_in : wp.array, shape (N,), dtype=wp.float64
        Input per-atom energies.
    energy_out : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Corrected per-atom energies.
    compute_forces : bool, default=True
        If True, compute and accumulate slab forces.
    compute_charge_gradients : bool, default=True
        If True, compute and accumulate slab charge gradients.
    compute_virial : bool, default=True
        If True, compute and accumulate slab virial.
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Forces (slab contribution accumulated).
    charge_grads : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Charge gradients (slab contribution accumulated).
    virial : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Virial tensor (slab contribution accumulated).
    wp_dtype : type
        Warp scalar type (wp.float32 or wp.float64).
    device : str, optional
        Warp device.
    """
    _launch_slab_correction(
        positions=positions,
        charges=charges,
        batch_idx=batch_idx,
        pbc=pbc,
        cell=cell,
        mz=mz,
        mz2=mz2,
        qtotal=qtotal,
        slab_axis=slab_axis,
        slab_normal=slab_normal,
        slab_volume=slab_volume,
        slab_height_sq=slab_height_sq,
        energy_in=energy_in,
        energy_out=energy_out,
        wp_dtype=wp_dtype,
        forces=forces,
        charge_grads=charge_grads,
        virial=virial,
        compute_forces=compute_forces,
        compute_charge_gradients=compute_charge_gradients,
        compute_virial=compute_virial,
        device=device,
    )


def slab_correction_backward(
    positions: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    pbc: wp.array,
    cell: wp.array,
    mz: wp.array,
    mz2: wp.array,
    qtotal: wp.array,
    slab_axis: wp.array,
    slab_normal: wp.array,
    slab_volume: wp.array,
    slab_height_sq: wp.array,
    grad_system: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_normal: wp.array,
    grad_cell: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch slab energy backward kernels.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom.
    pbc : wp.array, shape (B, 3), dtype=wp.bool
        Per-system periodic boundary conditions.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system cell matrices.
    mz, mz2, qtotal
        Per-system slab moments from :func:`slab_reduce_moments`.
    slab_axis, slab_normal, slab_volume, slab_height_sq
        Caller-owned slab geometry buffers from :func:`slab_precompute_geometry`.
    grad_system : wp.array, shape (B,), dtype=wp.float64
        Per-system cotangent for the total slab energy.
    grad_positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Literal ``dE/dR``.
    grad_charges : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Literal ``dE/dq``.
    grad_normal : wp.array, shape (B,), dtype=wp.vec3d
        Scratch storage, zero-initialized by the caller.
    grad_cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Literal ``dE/dcell``.
    wp_dtype : type
        Warp scalar type (wp.float32 or wp.float64).
    device : str, optional
        Warp device.
    """
    num_atoms = charges.shape[0]
    num_systems = cell.shape[0]
    if device is None:
        device = str(charges.device)

    if num_atoms > 0:
        wp.launch(
            _slab_correction_backward_atoms_kernel_overload[wp_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                charges,
                batch_idx,
                slab_axis,
                slab_normal,
                slab_volume,
                slab_height_sq,
                mz,
                mz2,
                qtotal,
                grad_system,
                grad_positions,
                grad_charges,
                grad_normal,
            ],
            device=device,
        )

    wp.launch(
        _slab_correction_backward_cell_kernel_overload[wp_dtype],
        dim=num_systems,
        inputs=[
            pbc,
            cell,
            mz,
            mz2,
            qtotal,
            grad_system,
            grad_normal,
            grad_cell,
        ],
        device=device,
    )


def _slab_correction_weighted_backward(
    positions: wp.array,
    charges: wp.array,
    weights: wp.array,
    batch_idx: wp.array,
    pbc: wp.array,
    cell: wp.array,
    mz: wp.array,
    mz2: wp.array,
    qtotal: wp.array,
    slab_axis: wp.array,
    slab_normal: wp.array,
    slab_volume: wp.array,
    slab_height_sq: wp.array,
    weighted_mz: wp.array,
    weighted_mz2: wp.array,
    weighted_qtotal: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_normal: wp.array,
    grad_cell: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch weighted slab VJP kernels for non-uniform cotangents."""
    num_atoms = charges.shape[0]
    num_systems = cell.shape[0]
    if device is None:
        device = str(charges.device)

    if num_atoms > 0:
        wp.launch(
            _slab_weighted_moments_kernel_overload[wp_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                charges,
                weights,
                batch_idx,
                slab_axis,
                slab_normal,
                weighted_qtotal,
                weighted_mz,
                weighted_mz2,
            ],
            device=device,
        )
        wp.launch(
            _slab_correction_weighted_backward_atoms_kernel_overload[wp_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                charges,
                weights,
                batch_idx,
                slab_axis,
                slab_normal,
                slab_volume,
                slab_height_sq,
                mz,
                mz2,
                qtotal,
                weighted_mz,
                weighted_mz2,
                weighted_qtotal,
                grad_positions,
                grad_charges,
                grad_normal,
            ],
            device=device,
        )

    if num_systems > 0:
        wp.launch(
            _slab_correction_weighted_backward_cell_kernel_overload[wp_dtype],
            dim=num_systems,
            inputs=[
                pbc,
                cell,
                mz,
                mz2,
                qtotal,
                weighted_mz,
                weighted_mz2,
                weighted_qtotal,
                grad_normal,
                grad_cell,
            ],
            device=device,
        )


def slab_correction_double_backward(
    positions: wp.array,
    charges: wp.array,
    h_positions: wp.array,
    h_charges: wp.array,
    h_cell: wp.array,
    batch_idx: wp.array,
    pbc: wp.array,
    cell: wp.array,
    mz: wp.array,
    mz2: wp.array,
    qtotal: wp.array,
    slab_axis: wp.array,
    slab_normal: wp.array,
    slab_volume: wp.array,
    slab_height_sq: wp.array,
    grad_system: wp.array,
    dmz: wp.array,
    dmz2: wp.array,
    dqtotal: wp.array,
    dnormal: wp.array,
    dvolume: wp.array,
    dheight_sq: wp.array,
    grad_normal: wp.array,
    h_grad_normal: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_cell: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch analytic slab energy double-backward kernels.

    Parameters
    ----------
    positions, charges, cell, batch_idx, pbc
        Primal slab inputs.
    h_positions, h_charges, h_cell
        Direction/cotangent arrays for the first-order gradient outputs.
    mz, mz2, qtotal
        Base slab moments from :func:`slab_reduce_moments`.
    slab_axis, slab_normal, slab_volume, slab_height_sq
        Caller-owned slab geometry buffers from :func:`slab_precompute_geometry`.
    grad_system
        Per-system cotangent for total slab energy.
    dmz, dmz2, dqtotal, dnormal, dvolume, dheight_sq
        Scratch arrays, zero-initialized by the caller.
    grad_normal, h_grad_normal
        Scratch arrays, zero-initialized by the caller.
    grad_positions, grad_charges, grad_cell
        OUTPUT: Literal HVP values for positions, charges, and cell.
    wp_dtype
        Warp scalar type (wp.float32 or wp.float64).
    device : str, optional
        Warp device.
    """
    num_atoms = charges.shape[0]
    num_systems = cell.shape[0]
    if device is None:
        device = str(charges.device)

    if num_systems > 0:
        wp.launch(
            _slab_directional_geometry_kernel_overload[wp_dtype],
            dim=num_systems,
            inputs=[pbc, cell, h_cell, dnormal, dvolume, dheight_sq],
            device=device,
        )

    if num_atoms > 0:
        wp.launch(
            _slab_directional_moments_kernel_overload[wp_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                charges,
                h_positions,
                h_charges,
                batch_idx,
                slab_axis,
                slab_normal,
                dnormal,
                dmz,
                dmz2,
                dqtotal,
            ],
            device=device,
        )
        wp.launch(
            _slab_correction_double_backward_atoms_kernel_overload[wp_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                charges,
                h_positions,
                h_charges,
                batch_idx,
                slab_axis,
                slab_normal,
                slab_volume,
                slab_height_sq,
                mz,
                mz2,
                qtotal,
                dmz,
                dmz2,
                dqtotal,
                dnormal,
                dvolume,
                dheight_sq,
                grad_system,
                grad_positions,
                grad_charges,
                grad_normal,
                h_grad_normal,
            ],
            device=device,
        )

    if num_systems > 0:
        wp.launch(
            _slab_correction_double_backward_cell_kernel_overload[wp_dtype],
            dim=num_systems,
            inputs=[
                pbc,
                cell,
                h_cell,
                mz,
                mz2,
                qtotal,
                dmz,
                dmz2,
                dqtotal,
                grad_system,
                grad_normal,
                h_grad_normal,
                grad_cell,
            ],
            device=device,
        )


def _slab_correction_weighted_double_backward(
    positions: wp.array,
    charges: wp.array,
    weights: wp.array,
    h_positions: wp.array,
    h_charges: wp.array,
    h_cell: wp.array,
    batch_idx: wp.array,
    pbc: wp.array,
    cell: wp.array,
    mz: wp.array,
    mz2: wp.array,
    qtotal: wp.array,
    slab_axis: wp.array,
    slab_normal: wp.array,
    slab_volume: wp.array,
    slab_height_sq: wp.array,
    weighted_mz: wp.array,
    weighted_mz2: wp.array,
    weighted_qtotal: wp.array,
    dmz: wp.array,
    dmz2: wp.array,
    dqtotal: wp.array,
    d_weighted_mz: wp.array,
    d_weighted_mz2: wp.array,
    d_weighted_qtotal: wp.array,
    dnormal: wp.array,
    dvolume: wp.array,
    dheight_sq: wp.array,
    grad_normal: wp.array,
    h_grad_normal: wp.array,
    grad_grad_energy: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_cell: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch weighted slab HVP kernels for non-uniform cotangents."""
    num_atoms = charges.shape[0]
    num_systems = cell.shape[0]
    if device is None:
        device = str(charges.device)

    if num_atoms > 0:
        wp.launch(
            _slab_weighted_moments_kernel_overload[wp_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                charges,
                weights,
                batch_idx,
                slab_axis,
                slab_normal,
                weighted_qtotal,
                weighted_mz,
                weighted_mz2,
            ],
            device=device,
        )

    if num_systems > 0:
        wp.launch(
            _slab_directional_geometry_kernel_overload[wp_dtype],
            dim=num_systems,
            inputs=[pbc, cell, h_cell, dnormal, dvolume, dheight_sq],
            device=device,
        )

    if num_atoms > 0:
        wp.launch(
            _slab_directional_moments_kernel_overload[wp_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                charges,
                h_positions,
                h_charges,
                batch_idx,
                slab_axis,
                slab_normal,
                dnormal,
                dmz,
                dmz2,
                dqtotal,
            ],
            device=device,
        )

        wp.launch(
            _slab_directional_weighted_moments_kernel_overload[wp_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                charges,
                weights,
                h_positions,
                h_charges,
                batch_idx,
                slab_axis,
                slab_normal,
                dnormal,
                d_weighted_qtotal,
                d_weighted_mz,
                d_weighted_mz2,
            ],
            device=device,
        )
        wp.launch(
            _slab_correction_weighted_double_backward_atoms_kernel_overload[wp_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                charges,
                weights,
                h_positions,
                h_charges,
                batch_idx,
                slab_axis,
                slab_normal,
                slab_volume,
                slab_height_sq,
                mz,
                mz2,
                qtotal,
                weighted_mz,
                weighted_mz2,
                weighted_qtotal,
                dmz,
                dmz2,
                dqtotal,
                d_weighted_mz,
                d_weighted_mz2,
                d_weighted_qtotal,
                dnormal,
                dvolume,
                dheight_sq,
                grad_grad_energy,
                grad_positions,
                grad_charges,
                grad_normal,
                h_grad_normal,
            ],
            device=device,
        )

    if num_systems > 0:
        wp.launch(
            _slab_correction_weighted_double_backward_cell_kernel_overload[wp_dtype],
            dim=num_systems,
            inputs=[
                pbc,
                cell,
                h_cell,
                mz,
                mz2,
                qtotal,
                weighted_mz,
                weighted_mz2,
                weighted_qtotal,
                dmz,
                dmz2,
                dqtotal,
                d_weighted_mz,
                d_weighted_mz2,
                d_weighted_qtotal,
                grad_normal,
                h_grad_normal,
                grad_cell,
            ],
            device=device,
        )


__all__ = [
    "slab_reduce_moments",
    "slab_precompute_geometry",
    "slab_correction",
    "slab_correction_backward",
    "slab_correction_double_backward",
]
