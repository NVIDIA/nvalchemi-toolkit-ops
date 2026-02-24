# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Damped Shifted Force (DSF) Electrostatics - Warp Kernel Implementation
======================================================================

This module implements the Damped Shifted Force (DSF) method for pairwise
:math:`\mathcal{O}(N)` electrostatic summation using Warp GPU/CPU kernels.
The DSF method ensures both potential energy and forces smoothly vanish at a
defined cutoff radius :math:`R_c`.

Mathematical Formulation
------------------------

1. DSF Pair Potential:
   The potential energy for a pair of charges at distance :math:`r \le R_c`:

   .. math::

       V(r) = q_i q_j \left[ \frac{\text{erfc}(\alpha r)}{r}
       - \frac{\text{erfc}(\alpha R_c)}{R_c}
       + \left( \frac{\text{erfc}(\alpha R_c)}{R_c^2}
       + \frac{2\alpha}{\sqrt{\pi}} \frac{e^{-\alpha^2 R_c^2}}{R_c}
       \right)(r - R_c) \right]

2. DSF Force:
   The force between charges at distance :math:`r \le R_c`:

   .. math::

       \mathbf{F}(r) = q_i q_j \left[ \left(
       \frac{\text{erfc}(\alpha r)}{r^2}
       + \frac{2\alpha}{\sqrt{\pi}} \frac{e^{-\alpha^2 r^2}}{r}
       \right) - \left(
       \frac{\text{erfc}(\alpha R_c)}{R_c^2}
       + \frac{2\alpha}{\sqrt{\pi}} \frac{e^{-\alpha^2 R_c^2}}{R_c}
       \right) \right] \hat{r}_{ij}

3. Self-Energy Correction:

   .. math::

       U_i^{\text{self}} = -\left(
       \frac{\text{erfc}(\alpha R_c)}{2 R_c}
       + \frac{\alpha}{\sqrt{\pi}} \right) q_i^2

Architecture
------------
This module provides two layers:

1. **Warp Kernels** (pure Warp, framework-agnostic):
   - ``_dsf_csr_single_kernel``, ``_dsf_csr_batch_kernel`` (CSR neighbor list)
   - ``_dsf_matrix_single_kernel``, ``_dsf_matrix_batch_kernel`` (neighbor matrix)
   Each kernel accepts a ``use_pbc`` flag for periodic boundary conditions.

2. **Warp Launchers** (framework-agnostic API):
   - ``dsf_csr()`` (CSR format, optional PBC via ``cell``/``unit_shifts``)
   - ``dsf_matrix()`` (neighbor matrix format, optional PBC)
   Launchers use the dynamics launch system (``dispatch_family``) for
   automatic dtype and execution-mode dispatch.

For PyTorch integration, see ``nvalchemiops.torch.interactions.electrostatics.dsf``.

.. note::
   This implementation assumes a **full neighbor list** where each pair (i, j)
   appears in both directions (i->j and j->i). The 0.5 factor for pair energy
   and the -0.5 factor for virial account for this double counting.

.. note::
   When using batched mode, ``batch_idx`` must be sorted in non-decreasing
   order. Callers are responsible for providing sorted indices.

Precision
---------
All kernels support float32 and float64 via Warp overloads. Positions, charges,
cutoff, alpha, forces, virial, and charge gradients use the input precision.
Energy output arrays are always float64.
Internal accumulators are always float64 for numerical stability.

Neighbor Formats
----------------

1. **Neighbor List (CSR format)**: ``idx_j`` is shape (num_pairs,) containing
   destination indices, ``neighbor_ptr`` is shape (N+1,) with CSR row pointers.

2. **Neighbor Matrix**: ``neighbor_matrix`` is shape (N, max_neighbors) where
   each row contains neighbor indices for that atom.

References
----------
- Fennell & Gezelter, J. Chem. Phys. 124, 234104 (2006)
- Wolf et al., J. Chem. Phys. 110, 8254 (1999)
"""

from __future__ import annotations

import math

import warp as wp
from warp.types import Any

from nvalchemiops.dynamics.utils.launch_helpers import (
    KernelFamily,
    launch_family,
    resolve_execution_mode,
)
from nvalchemiops.math import wp_erfc
from nvalchemiops.warp_dispatch import register_overloads

__all__ = [
    "dsf_csr",
    "dsf_matrix",
]

PI = math.pi
SQRT_PI = math.sqrt(PI)
TWO_OVER_SQRT_PI = 2.0 / SQRT_PI
ONE_OVER_SQRT_PI = 1.0 / SQRT_PI

_VEC_TO_SCALAR = {wp.vec3f: wp.float32, wp.vec3d: wp.float64}
_VEC_TO_MAT = {wp.vec3f: wp.mat33f, wp.vec3d: wp.mat33d}


# ==============================================================================
# Warp Kernels - CSR Neighbor List Format
# ==============================================================================


@wp.kernel(enable_backward=False)
def _dsf_csr_single_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    cutoff: Any,
    alpha: Any,
    use_pbc: bool,
    compute_forces: bool,
    compute_virial: bool,
    compute_charge_grad: bool,
    energy: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=Any),
    virial: wp.array(dtype=Any),
    charge_grad: wp.array(dtype=Any),
):
    """DSF electrostatics, CSR format, single system.

    Launch dim = [num_atoms]. System index is always 0.
    For non-PBC calls, ``cell`` and ``unit_shifts`` are zero-sized arrays
    and ``use_pbc`` is False.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]

    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    qi = charges[atom_i]
    two_over_sqrt_pi = type(qi)(TWO_OVER_SQRT_PI)
    one_over_sqrt_pi = type(qi)(ONE_OVER_SQRT_PI)
    zero = type(qi)(0.0)
    one = type(qi)(1.0)
    half = type(qi)(0.5)
    two = type(qi)(2.0)
    eps = type(qi)(1e-10)

    alpha_rc = alpha * cutoff
    if alpha > zero:
        erfc_rc = wp_erfc(alpha_rc)
        exp_rc = wp.exp(-alpha_rc * alpha_rc)
    else:
        erfc_rc = one
        exp_rc = one

    v_shift = erfc_rc / cutoff
    force_shift = (
        erfc_rc / (cutoff * cutoff) + two_over_sqrt_pi * alpha * exp_rc / cutoff
    )
    self_coeff = -(v_shift / two + alpha * one_over_sqrt_pi)

    energy_pair_acc = wp.float64(0.0)
    cg_acc = zero
    force_acc = type(ri)(zero, zero, zero)

    if compute_virial:
        virial_acc = wp.mat33d()

    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]
        rj = positions[j]
        qj = charges[j]

        r_ij = ri - rj
        if use_pbc:
            shift = unit_shifts[edge_idx]
            shift_vec = (
                type(ri)(
                    type(qi)(shift[0]),
                    type(qi)(shift[1]),
                    type(qi)(shift[2]),
                )
                * cell[0]
            )
            r_ij = r_ij - shift_vec

        r = wp.length(r_ij)

        if r >= cutoff or r < eps:
            continue

        alpha_r = alpha * r
        if alpha > zero:
            erfc_r = wp_erfc(alpha_r)
            exp_r = wp.exp(-alpha_r * alpha_r)
        else:
            erfc_r = one
            exp_r = one

        v_pair = erfc_r / r - v_shift + force_shift * (r - cutoff)

        energy_pair_acc += wp.float64(qi * qj * v_pair)
        if compute_charge_grad:
            cg_acc += qj * v_pair

        if compute_forces:
            force_factor = (
                erfc_r / (r * r) + two_over_sqrt_pi * alpha * exp_r / r - force_shift
            )
            f_ij = qi * qj * force_factor / r * r_ij
            force_acc += f_ij

            if compute_virial:
                virial_acc += wp.outer(
                    wp.vec3d(
                        wp.float64(f_ij[0]), wp.float64(f_ij[1]), wp.float64(f_ij[2])
                    ),
                    wp.vec3d(
                        wp.float64(r_ij[0]), wp.float64(r_ij[1]), wp.float64(r_ij[2])
                    ),
                )

    self_energy = self_coeff * qi * qi

    wp.atomic_add(
        energy,
        0,
        wp.float64(0.5) * energy_pair_acc + wp.float64(self_energy),
    )

    if compute_charge_grad:
        charge_grad[atom_i] = cg_acc + two * self_coeff * qi

    if compute_forces:
        forces[atom_i] = force_acc

    if compute_virial:
        virial_out = type(virial[0])(
            type(qi)(virial_acc[0, 0]),
            type(qi)(virial_acc[0, 1]),
            type(qi)(virial_acc[0, 2]),
            type(qi)(virial_acc[1, 0]),
            type(qi)(virial_acc[1, 1]),
            type(qi)(virial_acc[1, 2]),
            type(qi)(virial_acc[2, 0]),
            type(qi)(virial_acc[2, 1]),
            type(qi)(virial_acc[2, 2]),
        )
        wp.atomic_add(virial, 0, -half * virial_out)


@wp.kernel(enable_backward=False)
def _dsf_csr_batch_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    batch_idx: wp.array(dtype=wp.int32),
    cutoff: Any,
    alpha: Any,
    use_pbc: bool,
    compute_forces: bool,
    compute_virial: bool,
    compute_charge_grad: bool,
    energy: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=Any),
    virial: wp.array(dtype=Any),
    charge_grad: wp.array(dtype=Any),
):
    """DSF electrostatics, CSR format, batched.

    Launch dim = [num_atoms]. System index from ``batch_idx[atom_i]``.
    ``batch_idx`` must be sorted in non-decreasing order.
    For non-PBC calls, ``cell`` and ``unit_shifts`` are zero-sized arrays
    and ``use_pbc`` is False.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]

    if atom_i >= num_atoms:
        return

    system_id = batch_idx[atom_i]
    ri = positions[atom_i]
    qi = charges[atom_i]
    two_over_sqrt_pi = type(qi)(TWO_OVER_SQRT_PI)
    one_over_sqrt_pi = type(qi)(ONE_OVER_SQRT_PI)
    zero = type(qi)(0.0)
    one = type(qi)(1.0)
    half = type(qi)(0.5)
    two = type(qi)(2.0)
    eps = type(qi)(1e-10)

    alpha_rc = alpha * cutoff
    if alpha > zero:
        erfc_rc = wp_erfc(alpha_rc)
        exp_rc = wp.exp(-alpha_rc * alpha_rc)
    else:
        erfc_rc = one
        exp_rc = one

    v_shift = erfc_rc / cutoff
    force_shift = (
        erfc_rc / (cutoff * cutoff) + two_over_sqrt_pi * alpha * exp_rc / cutoff
    )
    self_coeff = -(v_shift / two + alpha * one_over_sqrt_pi)

    energy_pair_acc = wp.float64(0.0)
    cg_acc = zero
    force_acc = type(ri)(zero, zero, zero)

    if compute_virial:
        virial_acc = wp.mat33d()

    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]
        rj = positions[j]
        qj = charges[j]

        r_ij = ri - rj
        if use_pbc:
            shift = unit_shifts[edge_idx]
            shift_vec = (
                type(ri)(
                    type(qi)(shift[0]),
                    type(qi)(shift[1]),
                    type(qi)(shift[2]),
                )
                * cell[system_id]
            )
            r_ij = r_ij - shift_vec

        r = wp.length(r_ij)

        if r >= cutoff or r < eps:
            continue

        alpha_r = alpha * r
        if alpha > zero:
            erfc_r = wp_erfc(alpha_r)
            exp_r = wp.exp(-alpha_r * alpha_r)
        else:
            erfc_r = one
            exp_r = one

        v_pair = erfc_r / r - v_shift + force_shift * (r - cutoff)

        energy_pair_acc += wp.float64(qi * qj * v_pair)
        if compute_charge_grad:
            cg_acc += qj * v_pair

        if compute_forces:
            force_factor = (
                erfc_r / (r * r) + two_over_sqrt_pi * alpha * exp_r / r - force_shift
            )
            f_ij = qi * qj * force_factor / r * r_ij
            force_acc += f_ij

            if compute_virial:
                virial_acc += wp.outer(
                    wp.vec3d(
                        wp.float64(f_ij[0]), wp.float64(f_ij[1]), wp.float64(f_ij[2])
                    ),
                    wp.vec3d(
                        wp.float64(r_ij[0]), wp.float64(r_ij[1]), wp.float64(r_ij[2])
                    ),
                )

    self_energy = self_coeff * qi * qi

    wp.atomic_add(
        energy,
        system_id,
        wp.float64(0.5) * energy_pair_acc + wp.float64(self_energy),
    )

    if compute_charge_grad:
        charge_grad[atom_i] = cg_acc + two * self_coeff * qi

    if compute_forces:
        forces[atom_i] = force_acc

    if compute_virial:
        virial_out = type(virial[0])(
            type(qi)(virial_acc[0, 0]),
            type(qi)(virial_acc[0, 1]),
            type(qi)(virial_acc[0, 2]),
            type(qi)(virial_acc[1, 0]),
            type(qi)(virial_acc[1, 1]),
            type(qi)(virial_acc[1, 2]),
            type(qi)(virial_acc[2, 0]),
            type(qi)(virial_acc[2, 1]),
            type(qi)(virial_acc[2, 2]),
        )
        wp.atomic_add(virial, system_id, -half * virial_out)


# ==============================================================================
# Warp Kernels - Neighbor Matrix Format
# ==============================================================================


@wp.kernel(enable_backward=False)
def _dsf_matrix_single_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    fill_value: wp.int32,
    cutoff: Any,
    alpha: Any,
    use_pbc: bool,
    compute_forces: bool,
    compute_virial: bool,
    compute_charge_grad: bool,
    energy: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=Any),
    virial: wp.array(dtype=Any),
    charge_grad: wp.array(dtype=Any),
):
    """DSF electrostatics, neighbor matrix format, single system.

    Launch dim = [num_atoms]. System index is always 0.
    For non-PBC calls, ``cell`` and ``neighbor_matrix_shifts`` are zero-sized
    arrays and ``use_pbc`` is False.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    qi = charges[atom_i]
    two_over_sqrt_pi = type(qi)(TWO_OVER_SQRT_PI)
    one_over_sqrt_pi = type(qi)(ONE_OVER_SQRT_PI)
    zero = type(qi)(0.0)
    one = type(qi)(1.0)
    half = type(qi)(0.5)
    two = type(qi)(2.0)
    eps = type(qi)(1e-10)

    alpha_rc = alpha * cutoff
    if alpha > zero:
        erfc_rc = wp_erfc(alpha_rc)
        exp_rc = wp.exp(-alpha_rc * alpha_rc)
    else:
        erfc_rc = one
        exp_rc = one

    v_shift = erfc_rc / cutoff
    force_shift = (
        erfc_rc / (cutoff * cutoff) + two_over_sqrt_pi * alpha * exp_rc / cutoff
    )
    self_coeff = -(v_shift / two + alpha * one_over_sqrt_pi)

    energy_pair_acc = wp.float64(0.0)
    cg_acc = zero
    force_acc = type(ri)(zero, zero, zero)

    if compute_virial:
        virial_acc = wp.mat33d()

    for neighbor_slot in range(max_neighbors):
        j = neighbor_matrix[atom_i, neighbor_slot]
        if j == fill_value or j >= num_atoms:
            continue

        rj = positions[j]
        qj = charges[j]

        r_ij = ri - rj
        if use_pbc:
            shift = neighbor_matrix_shifts[atom_i, neighbor_slot]
            shift_vec = (
                type(ri)(
                    type(qi)(shift[0]),
                    type(qi)(shift[1]),
                    type(qi)(shift[2]),
                )
                * cell[0]
            )
            r_ij = r_ij - shift_vec

        r = wp.length(r_ij)

        if r >= cutoff or r < eps:
            continue

        alpha_r = alpha * r
        if alpha > zero:
            erfc_r = wp_erfc(alpha_r)
            exp_r = wp.exp(-alpha_r * alpha_r)
        else:
            erfc_r = one
            exp_r = one

        v_pair = erfc_r / r - v_shift + force_shift * (r - cutoff)

        energy_pair_acc += wp.float64(qi * qj * v_pair)
        if compute_charge_grad:
            cg_acc += qj * v_pair

        if compute_forces:
            force_factor = (
                erfc_r / (r * r) + two_over_sqrt_pi * alpha * exp_r / r - force_shift
            )
            f_ij = qi * qj * force_factor / r * r_ij
            force_acc += f_ij

            if compute_virial:
                virial_acc += wp.outer(
                    wp.vec3d(
                        wp.float64(f_ij[0]), wp.float64(f_ij[1]), wp.float64(f_ij[2])
                    ),
                    wp.vec3d(
                        wp.float64(r_ij[0]), wp.float64(r_ij[1]), wp.float64(r_ij[2])
                    ),
                )

    self_energy = self_coeff * qi * qi

    wp.atomic_add(
        energy,
        0,
        wp.float64(0.5) * energy_pair_acc + wp.float64(self_energy),
    )

    if compute_charge_grad:
        charge_grad[atom_i] = cg_acc + two * self_coeff * qi

    if compute_forces:
        forces[atom_i] = force_acc

    if compute_virial:
        virial_out = type(virial[0])(
            type(qi)(virial_acc[0, 0]),
            type(qi)(virial_acc[0, 1]),
            type(qi)(virial_acc[0, 2]),
            type(qi)(virial_acc[1, 0]),
            type(qi)(virial_acc[1, 1]),
            type(qi)(virial_acc[1, 2]),
            type(qi)(virial_acc[2, 0]),
            type(qi)(virial_acc[2, 1]),
            type(qi)(virial_acc[2, 2]),
        )
        wp.atomic_add(virial, 0, -half * virial_out)


@wp.kernel(enable_backward=False)
def _dsf_matrix_batch_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    batch_idx: wp.array(dtype=wp.int32),
    fill_value: wp.int32,
    cutoff: Any,
    alpha: Any,
    use_pbc: bool,
    compute_forces: bool,
    compute_virial: bool,
    compute_charge_grad: bool,
    energy: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=Any),
    virial: wp.array(dtype=Any),
    charge_grad: wp.array(dtype=Any),
):
    """DSF electrostatics, neighbor matrix format, batched.

    Launch dim = [num_atoms]. System index from ``batch_idx[atom_i]``.
    ``batch_idx`` must be sorted in non-decreasing order.
    For non-PBC calls, ``cell`` and ``neighbor_matrix_shifts`` are zero-sized
    arrays and ``use_pbc`` is False.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_i >= num_atoms:
        return

    system_id = batch_idx[atom_i]
    ri = positions[atom_i]
    qi = charges[atom_i]
    two_over_sqrt_pi = type(qi)(TWO_OVER_SQRT_PI)
    one_over_sqrt_pi = type(qi)(ONE_OVER_SQRT_PI)
    zero = type(qi)(0.0)
    one = type(qi)(1.0)
    half = type(qi)(0.5)
    two = type(qi)(2.0)
    eps = type(qi)(1e-10)

    alpha_rc = alpha * cutoff
    if alpha > zero:
        erfc_rc = wp_erfc(alpha_rc)
        exp_rc = wp.exp(-alpha_rc * alpha_rc)
    else:
        erfc_rc = one
        exp_rc = one

    v_shift = erfc_rc / cutoff
    force_shift = (
        erfc_rc / (cutoff * cutoff) + two_over_sqrt_pi * alpha * exp_rc / cutoff
    )
    self_coeff = -(v_shift / two + alpha * one_over_sqrt_pi)

    energy_pair_acc = wp.float64(0.0)
    cg_acc = zero
    force_acc = type(ri)(zero, zero, zero)

    if compute_virial:
        virial_acc = wp.mat33d()

    for neighbor_slot in range(max_neighbors):
        j = neighbor_matrix[atom_i, neighbor_slot]
        if j == fill_value or j >= num_atoms:
            continue

        rj = positions[j]
        qj = charges[j]

        r_ij = ri - rj
        if use_pbc:
            shift = neighbor_matrix_shifts[atom_i, neighbor_slot]
            shift_vec = (
                type(ri)(
                    type(qi)(shift[0]),
                    type(qi)(shift[1]),
                    type(qi)(shift[2]),
                )
                * cell[system_id]
            )
            r_ij = r_ij - shift_vec

        r = wp.length(r_ij)

        if r >= cutoff or r < eps:
            continue

        alpha_r = alpha * r
        if alpha > zero:
            erfc_r = wp_erfc(alpha_r)
            exp_r = wp.exp(-alpha_r * alpha_r)
        else:
            erfc_r = one
            exp_r = one

        v_pair = erfc_r / r - v_shift + force_shift * (r - cutoff)

        energy_pair_acc += wp.float64(qi * qj * v_pair)
        if compute_charge_grad:
            cg_acc += qj * v_pair

        if compute_forces:
            force_factor = (
                erfc_r / (r * r) + two_over_sqrt_pi * alpha * exp_r / r - force_shift
            )
            f_ij = qi * qj * force_factor / r * r_ij
            force_acc += f_ij

            if compute_virial:
                virial_acc += wp.outer(
                    wp.vec3d(
                        wp.float64(f_ij[0]), wp.float64(f_ij[1]), wp.float64(f_ij[2])
                    ),
                    wp.vec3d(
                        wp.float64(r_ij[0]), wp.float64(r_ij[1]), wp.float64(r_ij[2])
                    ),
                )

    self_energy = self_coeff * qi * qi

    wp.atomic_add(
        energy,
        system_id,
        wp.float64(0.5) * energy_pair_acc + wp.float64(self_energy),
    )

    if compute_charge_grad:
        charge_grad[atom_i] = cg_acc + two * self_coeff * qi

    if compute_forces:
        forces[atom_i] = force_acc

    if compute_virial:
        virial_out = type(virial[0])(
            type(qi)(virial_acc[0, 0]),
            type(qi)(virial_acc[0, 1]),
            type(qi)(virial_acc[0, 2]),
            type(qi)(virial_acc[1, 0]),
            type(qi)(virial_acc[1, 1]),
            type(qi)(virial_acc[1, 2]),
            type(qi)(virial_acc[2, 0]),
            type(qi)(virial_acc[2, 1]),
            type(qi)(virial_acc[2, 2]),
        )
        wp.atomic_add(virial, system_id, -half * virial_out)


# ==============================================================================
# Overload Registration & KernelFamily Construction
# ==============================================================================


def _csr_single_sig(v, t):
    m = _VEC_TO_MAT[v]
    return [
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=m),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.vec3i),
        t,
        t,
        bool,
        bool,
        bool,
        bool,
        wp.array(dtype=wp.float64),
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=t),
    ]


def _csr_batch_sig(v, t):
    m = _VEC_TO_MAT[v]
    return [
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=m),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.vec3i),
        wp.array(dtype=wp.int32),
        t,
        t,
        bool,
        bool,
        bool,
        bool,
        wp.array(dtype=wp.float64),
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=t),
    ]


def _matrix_single_sig(v, t):
    m = _VEC_TO_MAT[v]
    return [
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=m),
        wp.array2d(dtype=wp.int32),
        wp.array2d(dtype=wp.vec3i),
        wp.int32,
        t,
        t,
        bool,
        bool,
        bool,
        bool,
        wp.array(dtype=wp.float64),
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=t),
    ]


def _matrix_batch_sig(v, t):
    m = _VEC_TO_MAT[v]
    return [
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=m),
        wp.array2d(dtype=wp.int32),
        wp.array2d(dtype=wp.vec3i),
        wp.array(dtype=wp.int32),
        wp.int32,
        t,
        t,
        bool,
        bool,
        bool,
        bool,
        wp.array(dtype=wp.float64),
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=t),
    ]


_csr_single_overloads = register_overloads(_dsf_csr_single_kernel, _csr_single_sig)
_csr_batch_overloads = register_overloads(_dsf_csr_batch_kernel, _csr_batch_sig)
_matrix_single_overloads = register_overloads(
    _dsf_matrix_single_kernel, _matrix_single_sig
)
_matrix_batch_overloads = register_overloads(
    _dsf_matrix_batch_kernel, _matrix_batch_sig
)

_dsf_csr_families = {
    v: KernelFamily(
        single=_csr_single_overloads[v],
        batch_idx=_csr_batch_overloads[v],
        atom_ptr=None,
    )
    for v in [wp.vec3f, wp.vec3d]
}

_dsf_matrix_families = {
    v: KernelFamily(
        single=_matrix_single_overloads[v],
        batch_idx=_matrix_batch_overloads[v],
        atom_ptr=None,
    )
    for v in [wp.vec3f, wp.vec3d]
}


# ==============================================================================
# Warp Launchers
# ==============================================================================


def dsf_csr(
    positions: wp.array,
    charges: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    cutoff: float,
    alpha: float,
    energy: wp.array,
    forces: wp.array,
    virial: wp.array,
    charge_grad: wp.array,
    cell: wp.array | None = None,
    unit_shifts: wp.array | None = None,
    device: str | None = None,
    batch_idx: wp.array | None = None,
    compute_forces: bool = True,
    compute_virial: bool = False,
    compute_charge_grad: bool = False,
    wp_scalar_type: type | None = None,
) -> None:
    """Launch DSF calculation using CSR neighbor list format.

    Handles both periodic and non-periodic systems via optional ``cell``
    and ``unit_shifts`` parameters. Dispatches to single-system or batched
    kernel based on ``batch_idx``.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Destination atom indices in CSR format.
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers.
    cutoff : float
        Cutoff radius.
    alpha : float
        Damping parameter (0.0 for shifted-force bare Coulomb).
    energy : wp.array, shape (num_systems,), dtype=wp.float64
        OUTPUT: Per-system energies. Must be pre-allocated.
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces. Must be pre-allocated.
    virial : wp.array, shape (num_systems,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Per-system virial tensor. Must be pre-allocated.
    charge_grad : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-atom charge gradients dE/dq_i.
    cell : wp.array or None
        Unit cell matrices for PBC. If None, non-periodic.
    unit_shifts : wp.array or None
        Integer unit cell shifts for PBC. Required when ``cell`` is not None.
    device : str, optional
        Warp device. If None, inferred from positions.
    batch_idx : wp.array or None
        System index for each atom. Must be sorted. If None, single system.
    compute_forces : bool, default True
        Whether to compute forces.
    compute_virial : bool, default False
        Whether to compute virial tensor.
    compute_charge_grad : bool, default False
        Whether to compute charge gradients dE/dq_i.
    wp_scalar_type : type, optional
        Warp scalar type. If None, inferred from positions.dtype.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    if wp_scalar_type is None:
        dtype = getattr(positions, "dtype", None)
        if dtype not in _VEC_TO_SCALAR:
            raise ValueError(
                f"Unrecognized positions dtype {dtype}. "
                f"Expected one of {list(_VEC_TO_SCALAR.keys())}. "
                "For ctype arrays, pass wp_scalar_type explicitly."
            )
        wp_scalar_type = _VEC_TO_SCALAR[dtype]

    _SCALAR_TO_VEC = {wp.float32: wp.vec3f, wp.float64: wp.vec3d}
    vec_dtype = _SCALAR_TO_VEC[wp_scalar_type]
    mat_type = _VEC_TO_MAT[vec_dtype]

    use_pbc = cell is not None
    if use_pbc and unit_shifts is None:
        raise ValueError(
            "unit_shifts is required when cell is provided "
            "(periodic boundary conditions)"
        )
    if not use_pbc:
        cell = wp.empty(shape=(0,), dtype=mat_type, device=device)
        unit_shifts = wp.empty(shape=(0,), dtype=wp.vec3i, device=device)

    typed_cutoff = wp_scalar_type(cutoff)
    typed_alpha = wp_scalar_type(alpha)

    mode = resolve_execution_mode(batch_idx, None)

    common_inputs = [
        positions,
        charges,
        cell,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    ]
    flags = [
        typed_cutoff,
        typed_alpha,
        use_pbc,
        compute_forces,
        compute_virial,
        compute_charge_grad,
    ]
    outputs = [energy, forces, virial, charge_grad]

    launch_family(
        _dsf_csr_families[vec_dtype],
        mode=mode,
        dim=num_atoms,
        inputs_single=common_inputs + flags + outputs,
        inputs_batch=common_inputs + [batch_idx] + flags + outputs,
        device=device,
    )


def dsf_matrix(
    positions: wp.array,
    charges: wp.array,
    neighbor_matrix: wp.array,
    cutoff: float,
    alpha: float,
    fill_value: int,
    energy: wp.array,
    forces: wp.array,
    virial: wp.array,
    charge_grad: wp.array,
    cell: wp.array | None = None,
    neighbor_matrix_shifts: wp.array | None = None,
    device: str | None = None,
    batch_idx: wp.array | None = None,
    compute_forces: bool = True,
    compute_virial: bool = False,
    compute_charge_grad: bool = False,
    wp_scalar_type: type | None = None,
) -> None:
    """Launch DSF calculation using neighbor matrix format.

    Handles both periodic and non-periodic systems via optional ``cell``
    and ``neighbor_matrix_shifts`` parameters. Dispatches to single-system
    or batched kernel based on ``batch_idx``.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices. Padding entries have values >= fill_value.
    cutoff : float
        Cutoff radius.
    alpha : float
        Damping parameter.
    fill_value : int
        Value indicating padding in neighbor_matrix.
    energy : wp.array, shape (num_systems,), dtype=wp.float64
        OUTPUT: Per-system energies. Must be pre-allocated.
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces. Must be pre-allocated.
    virial : wp.array, shape (num_systems,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Per-system virial tensor. Must be pre-allocated.
    charge_grad : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-atom charge gradients dE/dq_i.
    cell : wp.array or None
        Unit cell matrices for PBC. If None, non-periodic.
    neighbor_matrix_shifts : wp.array2d or None
        Integer unit cell shifts for PBC. Required when ``cell`` is not None.
    device : str, optional
        Warp device. If None, inferred from positions.
    batch_idx : wp.array or None
        System index for each atom. Must be sorted. If None, single system.
    compute_forces : bool, default True
        Whether to compute forces.
    compute_virial : bool, default False
        Whether to compute virial tensor.
    compute_charge_grad : bool, default False
        Whether to compute charge gradients dE/dq_i.
    wp_scalar_type : type, optional
        Warp scalar type. If None, inferred from positions.dtype.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    if wp_scalar_type is None:
        dtype = getattr(positions, "dtype", None)
        if dtype not in _VEC_TO_SCALAR:
            raise ValueError(
                f"Unrecognized positions dtype {dtype}. "
                f"Expected one of {list(_VEC_TO_SCALAR.keys())}. "
                "For ctype arrays, pass wp_scalar_type explicitly."
            )
        wp_scalar_type = _VEC_TO_SCALAR[dtype]

    _SCALAR_TO_VEC = {wp.float32: wp.vec3f, wp.float64: wp.vec3d}
    vec_dtype = _SCALAR_TO_VEC[wp_scalar_type]
    mat_type = _VEC_TO_MAT[vec_dtype]

    use_pbc = cell is not None
    if use_pbc and neighbor_matrix_shifts is None:
        raise ValueError(
            "neighbor_matrix_shifts is required when cell is provided "
            "(periodic boundary conditions)"
        )
    if not use_pbc:
        cell = wp.empty(shape=(0,), dtype=mat_type, device=device)
        neighbor_matrix_shifts = wp.empty(shape=(0, 0), dtype=wp.vec3i, device=device)

    typed_cutoff = wp_scalar_type(cutoff)
    typed_alpha = wp_scalar_type(alpha)

    mode = resolve_execution_mode(batch_idx, None)

    common_inputs = [
        positions,
        charges,
        cell,
        neighbor_matrix,
        neighbor_matrix_shifts,
    ]
    flags = [
        wp.int32(fill_value),
        typed_cutoff,
        typed_alpha,
        use_pbc,
        compute_forces,
        compute_virial,
        compute_charge_grad,
    ]
    outputs = [energy, forces, virial, charge_grad]

    launch_family(
        _dsf_matrix_families[vec_dtype],
        mode=mode,
        dim=num_atoms,
        inputs_single=common_inputs + flags + outputs,
        inputs_batch=common_inputs + [batch_idx] + flags + outputs,
        device=device,
    )
