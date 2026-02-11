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
   - ``_dsf_kernel``, ``_dsf_pbc_kernel`` (CSR neighbor list format)
   - ``_dsf_matrix_kernel``, ``_dsf_matrix_pbc_kernel`` (neighbor matrix format)

2. **Warp Launchers** (framework-agnostic API):
   - ``dsf()``, ``dsf_pbc()`` (CSR format)
   - ``dsf_matrix()``, ``dsf_matrix_pbc()`` (neighbor matrix format)

For PyTorch integration, see ``nvalchemiops.torch.interactions.electrostatics.dsf``.

.. note::
   This implementation assumes a **full neighbor list** where each pair (i, j)
   appears in both directions (i->j and j->i). The 0.5 factor for pair energy
   and the -0.5 factor for virial account for this double counting.

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

from nvalchemiops.math import wp_erfc

__all__ = [
    # Warp launchers (framework-agnostic public API)
    "dsf",
    "dsf_pbc",
    "dsf_matrix",
    "dsf_matrix_pbc",
]

# Mathematical constants
PI = math.pi
SQRT_PI = math.sqrt(PI)
TWO_OVER_SQRT_PI = 2.0 / SQRT_PI
ONE_OVER_SQRT_PI = 1.0 / SQRT_PI

# Warp dtype mappings
_VEC_TO_SCALAR = {wp.vec3f: wp.float32, wp.vec3d: wp.float64}


# ==============================================================================
# Warp Kernels - CSR Neighbor List Format
# ==============================================================================


@wp.kernel(enable_backward=False)
def _dsf_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    batch_idx: wp.array(dtype=wp.int32),
    cutoff: Any,
    alpha: Any,
    compute_forces: bool,
    compute_virial: bool,
    compute_charge_grad: bool,
    energy: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=Any),
    virial: wp.array(dtype=Any),
    charge_grad: wp.array(dtype=Any),
):
    """Compute DSF electrostatic energy, forces, virial, and charge gradients.

    Non-periodic variant using CSR neighbor list format. Each thread processes
    one atom and loops over its neighbors. Energy is accumulated per-system.
    Forces are written per-atom. Charge gradients (dE/dq) are per-atom.

    Arithmetic runs in input precision; energy accumulator is always float64.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Target atom indices for each pair (flattened CSR data).
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom.
    cutoff : wp.float32 or wp.float64
        Cutoff radius Rc.
    alpha : wp.float32 or wp.float64
        Damping parameter (0.0 for shifted-force bare Coulomb).
    compute_forces : bool
        Whether to compute forces.
    compute_virial : bool
        Whether to compute virial (always False for non-PBC).
    compute_charge_grad : bool
        Whether to compute charge gradients dE/dq_i.
    energy : wp.array, shape (num_systems,), dtype=wp.float64
        OUTPUT: Per-system total energy (always float64).
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces (input precision).
    virial : wp.array, shape (num_systems,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Per-system virial tensor (input precision).
    charge_grad : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-atom charge gradients dE/dq_i (input precision).
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

    # Precompute cutoff constants (input precision)
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

    # Accumulators: energy in float64, others in input precision
    energy_pair_acc = wp.float64(0.0)
    cg_acc = zero
    force_acc = type(ri)(zero, zero, zero)

    if compute_virial:
        virial_acc = wp.mat33d()

    # Iterate over neighbors using CSR pointers
    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]
        rj = positions[j]
        qj = charges[j]

        r_ij = ri - rj
        r = wp.length(r_ij)

        if r >= cutoff or r < eps:
            continue

        # Compute erfc and exp for this distance (input precision)
        alpha_r = alpha * r
        if alpha > zero:
            erfc_r = wp_erfc(alpha_r)
            exp_r = wp.exp(-alpha_r * alpha_r)
        else:
            erfc_r = one
            exp_r = one

        # DSF pair potential (excluding qi*qj)
        v_pair = erfc_r / r - v_shift + force_shift * (r - cutoff)

        # Accumulate pair energy (float64) and charge gradient (input precision)
        energy_pair_acc += wp.float64(qi * qj * v_pair)
        if compute_charge_grad:
            cg_acc += qj * v_pair

        if compute_forces:
            # DSF force factor: erfc(ar)/r^2 + 2a/sqrt(pi)*exp(-a^2r^2)/r - force_shift
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

    # Self-energy: C * qi^2
    self_energy = self_coeff * qi * qi

    # Total energy for this atom: 0.5 * pair_sum (double counting) + self
    wp.atomic_add(
        energy,
        batch_idx[atom_i],
        wp.float64(0.5) * energy_pair_acc + wp.float64(self_energy),
    )

    # Charge gradient: dE/dq_i = sum_j qj*V(rij) + 2*C*qi
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
        wp.atomic_add(virial, batch_idx[atom_i], -half * virial_out)


@wp.kernel(enable_backward=False)
def _dsf_pbc_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    batch_idx: wp.array(dtype=wp.int32),
    cutoff: Any,
    alpha: Any,
    compute_forces: bool,
    compute_virial: bool,
    compute_charge_grad: bool,
    energy: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=Any),
    virial: wp.array(dtype=Any),
    charge_grad: wp.array(dtype=Any),
):
    """Compute DSF electrostatic energy, forces, virial, and charge gradients.

    Periodic boundary conditions variant using CSR neighbor list format. Each
    thread processes one atom and loops over its neighbors. Pair displacements
    are corrected by integer cell shifts for minimum-image convention.

    Arithmetic runs in input precision; energy accumulator is always float64.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (num_systems,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices for periodic boundary conditions.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Target atom indices for each pair (flattened CSR data).
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Integer unit cell shift vectors for each pair.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom.
    cutoff : wp.float32 or wp.float64
        Cutoff radius Rc.
    alpha : wp.float32 or wp.float64
        Damping parameter (0.0 for shifted-force bare Coulomb).
    compute_forces : bool
        Whether to compute forces.
    compute_virial : bool
        Whether to compute virial.
    compute_charge_grad : bool
        Whether to compute charge gradients dE/dq_i.
    energy : wp.array, shape (num_systems,), dtype=wp.float64
        OUTPUT: Per-system total energy (always float64).
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces (input precision).
    virial : wp.array, shape (num_systems,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Per-system virial tensor (input precision).
    charge_grad : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-atom charge gradients dE/dq_i (input precision).
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

    # Precompute cutoff constants (input precision)
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

    # Accumulators
    energy_pair_acc = wp.float64(0.0)
    cg_acc = zero
    force_acc = type(ri)(zero, zero, zero)

    if compute_virial:
        virial_acc = wp.mat33d()

    # Iterate over neighbors using CSR pointers
    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]
        rj = positions[j]
        qj = charges[j]

        # Apply periodic shift
        shift = unit_shifts[edge_idx]
        shift_vec = (
            type(ri)(
                type(qi)(shift[0]),
                type(qi)(shift[1]),
                type(qi)(shift[2]),
            )
            * cell[system_id]
        )
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < eps:
            continue

        # Compute erfc and exp for this distance (input precision)
        alpha_r = alpha * r
        if alpha > zero:
            erfc_r = wp_erfc(alpha_r)
            exp_r = wp.exp(-alpha_r * alpha_r)
        else:
            erfc_r = one
            exp_r = one

        # DSF pair potential (excluding qi*qj)
        v_pair = erfc_r / r - v_shift + force_shift * (r - cutoff)

        # Accumulate pair energy (float64) and charge gradient (input precision)
        energy_pair_acc += wp.float64(qi * qj * v_pair)
        if compute_charge_grad:
            cg_acc += qj * v_pair

        if compute_forces:
            # DSF force factor
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

    # Self-energy
    self_energy = self_coeff * qi * qi

    # Write outputs
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
def _dsf_matrix_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    batch_idx: wp.array(dtype=wp.int32),
    fill_value: wp.int32,
    cutoff: Any,
    alpha: Any,
    compute_forces: bool,
    compute_virial: bool,
    compute_charge_grad: bool,
    energy: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=Any),
    virial: wp.array(dtype=Any),
    charge_grad: wp.array(dtype=Any),
):
    """Compute DSF electrostatic energy, forces, virial, and charge gradients.

    Non-periodic variant using neighbor matrix format. Each thread processes
    one atom and loops over its neighbor columns. Padding entries (fill_value)
    are skipped.

    Arithmetic runs in input precision; energy accumulator is always float64.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Dense neighbor matrix with padding.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom.
    fill_value : wp.int32
        Padding value in neighbor_matrix (typically num_atoms).
    cutoff : wp.float32 or wp.float64
        Cutoff radius Rc.
    alpha : wp.float32 or wp.float64
        Damping parameter (0.0 for shifted-force bare Coulomb).
    compute_forces : bool
        Whether to compute forces.
    compute_virial : bool
        Whether to compute virial (always False for non-PBC).
    compute_charge_grad : bool
        Whether to compute charge gradients dE/dq_i.
    energy : wp.array, shape (num_systems,), dtype=wp.float64
        OUTPUT: Per-system total energy (always float64).
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces (input precision).
    virial : wp.array, shape (num_systems,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Per-system virial tensor (input precision).
    charge_grad : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-atom charge gradients dE/dq_i (input precision).
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

    # Precompute cutoff constants (input precision)
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

    # Accumulators
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

    # Self-energy
    self_energy = self_coeff * qi * qi

    wp.atomic_add(
        energy,
        batch_idx[atom_i],
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
        wp.atomic_add(virial, batch_idx[atom_i], -half * virial_out)


@wp.kernel(enable_backward=False)
def _dsf_matrix_pbc_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    batch_idx: wp.array(dtype=wp.int32),
    fill_value: wp.int32,
    cutoff: Any,
    alpha: Any,
    compute_forces: bool,
    compute_virial: bool,
    compute_charge_grad: bool,
    energy: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=Any),
    virial: wp.array(dtype=Any),
    charge_grad: wp.array(dtype=Any),
):
    """Compute DSF electrostatic energy, forces, virial, and charge gradients.

    Periodic boundary conditions variant using neighbor matrix format. Each
    thread processes one atom and loops over its neighbor columns. Pair
    displacements are corrected by integer cell shifts for minimum-image
    convention. Padding entries (fill_value) are skipped.

    Arithmetic runs in input precision; energy accumulator is always float64.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (num_systems,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices for periodic boundary conditions.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Dense neighbor matrix with padding.
    neighbor_matrix_shifts : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Integer unit cell shift vectors for each neighbor entry.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom.
    fill_value : wp.int32
        Padding value in neighbor_matrix (typically num_atoms).
    cutoff : wp.float32 or wp.float64
        Cutoff radius Rc.
    alpha : wp.float32 or wp.float64
        Damping parameter (0.0 for shifted-force bare Coulomb).
    compute_forces : bool
        Whether to compute forces.
    compute_virial : bool
        Whether to compute virial.
    compute_charge_grad : bool
        Whether to compute charge gradients dE/dq_i.
    energy : wp.array, shape (num_systems,), dtype=wp.float64
        OUTPUT: Per-system total energy (always float64).
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces (input precision).
    virial : wp.array, shape (num_systems,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Per-system virial tensor (input precision).
    charge_grad : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-atom charge gradients dE/dq_i (input precision).
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

    # Precompute cutoff constants (input precision)
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

    # Accumulators
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

        # Apply periodic shift
        shift = neighbor_matrix_shifts[atom_i, neighbor_slot]
        shift_vec = (
            type(ri)(
                type(qi)(shift[0]),
                type(qi)(shift[1]),
                type(qi)(shift[2]),
            )
            * cell[system_id]
        )
        r_ij = ri - rj - shift_vec
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

    # Self-energy
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
# Kernel Overloads (float32 + float64)
# ==============================================================================

_T = [wp.float32, wp.float64]
_V = [wp.vec3f, wp.vec3d]
_M = [wp.mat33f, wp.mat33d]

_dsf_kernel_overload = {}
_dsf_pbc_kernel_overload = {}
_dsf_matrix_kernel_overload = {}
_dsf_matrix_pbc_kernel_overload = {}

for t, v, m in zip(_T, _V, _M):
    _dsf_kernel_overload[t] = wp.overload(
        _dsf_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # idx_j
            wp.array(dtype=wp.int32),  # neighbor_ptr
            wp.array(dtype=wp.int32),  # batch_idx
            t,  # cutoff (scalar)
            t,  # alpha (scalar)
            bool,  # compute_forces
            bool,  # compute_virial
            bool,  # compute_charge_grad
            wp.array(dtype=wp.float64),  # energy (always float64)
            wp.array(dtype=v),  # forces
            wp.array(dtype=m),  # virial
            wp.array(dtype=t),  # charge_grad
        ],
    )

    _dsf_pbc_kernel_overload[t] = wp.overload(
        _dsf_pbc_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.int32),  # idx_j
            wp.array(dtype=wp.int32),  # neighbor_ptr
            wp.array(dtype=wp.vec3i),  # unit_shifts
            wp.array(dtype=wp.int32),  # batch_idx
            t,  # cutoff (scalar)
            t,  # alpha (scalar)
            bool,  # compute_forces
            bool,  # compute_virial
            bool,  # compute_charge_grad
            wp.array(dtype=wp.float64),  # energy (always float64)
            wp.array(dtype=v),  # forces
            wp.array(dtype=m),  # virial
            wp.array(dtype=t),  # charge_grad
        ],
    )

    _dsf_matrix_kernel_overload[t] = wp.overload(
        _dsf_matrix_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array(dtype=wp.int32),  # batch_idx
            wp.int32,  # fill_value
            t,  # cutoff (scalar)
            t,  # alpha (scalar)
            bool,  # compute_forces
            bool,  # compute_virial
            bool,  # compute_charge_grad
            wp.array(dtype=wp.float64),  # energy (always float64)
            wp.array(dtype=v),  # forces
            wp.array(dtype=m),  # virial
            wp.array(dtype=t),  # charge_grad
        ],
    )

    _dsf_matrix_pbc_kernel_overload[t] = wp.overload(
        _dsf_matrix_pbc_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # neighbor_matrix_shifts
            wp.array(dtype=wp.int32),  # batch_idx
            wp.int32,  # fill_value
            t,  # cutoff (scalar)
            t,  # alpha (scalar)
            bool,  # compute_forces
            bool,  # compute_virial
            bool,  # compute_charge_grad
            wp.array(dtype=wp.float64),  # energy (always float64)
            wp.array(dtype=v),  # forces
            wp.array(dtype=m),  # virial
            wp.array(dtype=t),  # charge_grad
        ],
    )


# ==============================================================================
# Warp Launchers (Framework-Agnostic)
# ==============================================================================


def _get_scalar_type(positions: wp.array) -> type:
    """Infer scalar Warp type from positions array dtype.

    Works with regular wp.array objects. For ctype arrays (from
    wp.from_torch with return_ctype=True), pass wp_scalar_type
    explicitly to launcher functions.

    Raises
    ------
    ValueError
        If positions dtype is not recognized (wp.vec3f or wp.vec3d).
    """
    dtype = getattr(positions, "dtype", None)
    if dtype not in _VEC_TO_SCALAR:
        raise ValueError(
            f"Unrecognized positions dtype {dtype}. "
            f"Expected one of {list(_VEC_TO_SCALAR.keys())}. "
            "For ctype arrays, pass wp_scalar_type explicitly."
        )
    return _VEC_TO_SCALAR[dtype]


def dsf(
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
    device: str | None = None,
    batch_idx: wp.array | None = None,
    compute_forces: bool = True,
    compute_charge_grad: bool = False,
    wp_scalar_type: type | None = None,
) -> None:
    """Launch DSF calculation using CSR neighbor list format (non-periodic).

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
        OUTPUT: Per-system energies. Must be pre-allocated and zeroed.
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces. Must be pre-allocated and zeroed.
    virial : wp.array, shape (num_systems,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Per-system virial. Must be pre-allocated.
    charge_grad : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-atom charge gradients dE/dq_i.
    device : str, optional
        Warp device. If None, inferred from positions.
    batch_idx : wp.array, shape (N,), dtype=wp.int32, optional
        System index for each atom. If None, single system assumed.
    compute_forces : bool, default True
        Whether to compute forces.
    compute_charge_grad : bool, default False
        Whether to compute charge gradients dE/dq_i.
    wp_scalar_type : type, optional
        Warp scalar type (wp.float32 or wp.float64). If None, inferred
        from positions.dtype.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    if batch_idx is None:
        batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=device)

    if wp_scalar_type is None:
        wp_scalar_type = _get_scalar_type(positions)

    wp.launch(
        _dsf_kernel_overload[wp_scalar_type],
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            idx_j,
            neighbor_ptr,
            batch_idx,
            wp_scalar_type(cutoff),
            wp_scalar_type(alpha),
            compute_forces,
            False,  # compute_virial always False for non-PBC
            compute_charge_grad,
        ],
        outputs=[energy, forces, virial, charge_grad],
        device=device,
    )


def dsf_pbc(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    cutoff: float,
    alpha: float,
    energy: wp.array,
    forces: wp.array,
    virial: wp.array,
    charge_grad: wp.array,
    device: str | None = None,
    batch_idx: wp.array | None = None,
    compute_forces: bool = True,
    compute_virial: bool = False,
    compute_charge_grad: bool = False,
    wp_scalar_type: type | None = None,
) -> None:
    """Launch DSF calculation using CSR neighbor list format with PBC.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices for each system.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Destination atom indices in CSR format.
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Integer unit cell shifts for PBC.
    cutoff : float
        Cutoff radius.
    alpha : float
        Damping parameter.
    energy : wp.array, shape (num_systems,), dtype=wp.float64
        OUTPUT: Per-system energies. Must be pre-allocated and zeroed.
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces. Must be pre-allocated and zeroed.
    virial : wp.array, shape (num_systems,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Per-system virial tensor. Must be pre-allocated and zeroed.
    charge_grad : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-atom charge gradients dE/dq_i.
    device : str, optional
        Warp device. If None, inferred from positions.
    batch_idx : wp.array, shape (N,), dtype=wp.int32, optional
        System index for each atom. If None, single system assumed.
    compute_forces : bool, default True
        Whether to compute forces.
    compute_virial : bool, default False
        Whether to compute virial tensor.
    compute_charge_grad : bool, default False
        Whether to compute charge gradients dE/dq_i.
    wp_scalar_type : type, optional
        Warp scalar type (wp.float32 or wp.float64). If None, inferred
        from positions.dtype.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    if batch_idx is None:
        batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=device)

    if wp_scalar_type is None:
        wp_scalar_type = _get_scalar_type(positions)

    wp.launch(
        _dsf_pbc_kernel_overload[wp_scalar_type],
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
            wp_scalar_type(cutoff),
            wp_scalar_type(alpha),
            compute_forces,
            compute_virial,
            compute_charge_grad,
        ],
        outputs=[energy, forces, virial, charge_grad],
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
    device: str | None = None,
    batch_idx: wp.array | None = None,
    compute_forces: bool = True,
    compute_charge_grad: bool = False,
    wp_scalar_type: type | None = None,
) -> None:
    """Launch DSF calculation using neighbor matrix format (non-periodic).

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
        OUTPUT: Per-system energies. Must be pre-allocated and zeroed.
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces. Must be pre-allocated and zeroed.
    virial : wp.array, shape (num_systems,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Per-system virial.
    charge_grad : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-atom charge gradients dE/dq_i.
    device : str, optional
        Warp device. If None, inferred from positions.
    batch_idx : wp.array, shape (N,), dtype=wp.int32, optional
        System index for each atom. If None, single system assumed.
    compute_forces : bool, default True
        Whether to compute forces.
    compute_charge_grad : bool, default False
        Whether to compute charge gradients dE/dq_i.
    wp_scalar_type : type, optional
        Warp scalar type (wp.float32 or wp.float64). If None, inferred
        from positions.dtype.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    if batch_idx is None:
        batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=device)

    if wp_scalar_type is None:
        wp_scalar_type = _get_scalar_type(positions)

    wp.launch(
        _dsf_matrix_kernel_overload[wp_scalar_type],
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            neighbor_matrix,
            batch_idx,
            wp.int32(fill_value),
            wp_scalar_type(cutoff),
            wp_scalar_type(alpha),
            compute_forces,
            False,  # compute_virial always False for non-PBC
            compute_charge_grad,
        ],
        outputs=[energy, forces, virial, charge_grad],
        device=device,
    )


def dsf_matrix_pbc(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    cutoff: float,
    alpha: float,
    fill_value: int,
    energy: wp.array,
    forces: wp.array,
    virial: wp.array,
    charge_grad: wp.array,
    device: str | None = None,
    batch_idx: wp.array | None = None,
    compute_forces: bool = True,
    compute_virial: bool = False,
    compute_charge_grad: bool = False,
    wp_scalar_type: type | None = None,
) -> None:
    """Launch DSF calculation using neighbor matrix format with PBC.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices for each system.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices.
    neighbor_matrix_shifts : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Integer unit cell shifts.
    cutoff : float
        Cutoff radius.
    alpha : float
        Damping parameter.
    fill_value : int
        Value indicating padding in neighbor_matrix.
    energy : wp.array, shape (num_systems,), dtype=wp.float64
        OUTPUT: Per-system energies. Must be pre-allocated and zeroed.
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces. Must be pre-allocated and zeroed.
    virial : wp.array, shape (num_systems,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Per-system virial tensor. Must be pre-allocated and zeroed.
    charge_grad : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-atom charge gradients dE/dq_i.
    device : str, optional
        Warp device. If None, inferred from positions.
    batch_idx : wp.array, shape (N,), dtype=wp.int32, optional
        System index for each atom. If None, single system assumed.
    compute_forces : bool, default True
        Whether to compute forces.
    compute_virial : bool, default False
        Whether to compute virial tensor.
    compute_charge_grad : bool, default False
        Whether to compute charge gradients dE/dq_i.
    wp_scalar_type : type, optional
        Warp scalar type (wp.float32 or wp.float64). If None, inferred
        from positions.dtype.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    if batch_idx is None:
        batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=device)

    if wp_scalar_type is None:
        wp_scalar_type = _get_scalar_type(positions)

    wp.launch(
        _dsf_matrix_pbc_kernel_overload[wp_scalar_type],
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            batch_idx,
            wp.int32(fill_value),
            wp_scalar_type(cutoff),
            wp_scalar_type(alpha),
            compute_forces,
            compute_virial,
            compute_charge_grad,
        ],
        outputs=[energy, forces, virial, charge_grad],
        device=device,
    )
