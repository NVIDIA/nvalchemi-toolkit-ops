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
SHAKE and RATTLE Constraint Algorithms
======================================

GPU-accelerated Warp kernels for holonomic bond constraints in molecular dynamics.

This module provides both mutating (in-place) and non-mutating versions
of each kernel for gradient tracking compatibility.

MATHEMATICAL FORMULATION
========================

SHAKE Algorithm (Position Constraints)
--------------------------------------

For each constrained bond (i, j) with target distance d_ij:

1. Compute constraint violation: σ = |r_ij|² - d_ij²
2. Compute Lagrange multiplier: λ = σ / (2 * (1/m_i + 1/m_j) * r_ij · r_ij_old)
3. Update positions:
   - r_i += λ * r_ij_old / m_i
   - r_j -= λ * r_ij_old / m_j

Iterate until all constraints are satisfied within tolerance.

RATTLE Algorithm (Velocity Constraints)
---------------------------------------

After SHAKE, velocities must also satisfy constraints:

For each constrained bond (i, j):

1. Compute velocity constraint violation: κ = v_ij · r_ij
2. Compute Lagrange multiplier: μ = κ / ((1/m_i + 1/m_j) * |r_ij|²)
3. Update velocities:
   - v_i -= μ * r_ij / m_i
   - v_j += μ * r_ij / m_j

USAGE
=====

SHAKE is applied after position updates but before force calculation:

    # Position update (unconstrained)
    velocity_verlet_position_update(positions, velocities, forces, masses, dt)

    # Apply SHAKE to fix bond lengths
    shake_converged = shake_constraints(
        positions, positions_old, masses,
        bond_pairs, bond_lengths, tolerance=1e-6, max_iter=100
    )

    # Compute forces at constrained positions
    forces = compute_forces(positions)

    # Velocity update
    velocity_verlet_velocity_finalize(velocities, forces, masses, dt)

    # Apply RATTLE to fix velocity constraints
    rattle_constraints(positions, velocities, masses, bond_pairs, bond_lengths)

REFERENCES
==========

- Ryckaert et al. (1977). J. Comput. Phys. 23, 327 (SHAKE)
- Andersen (1983). J. Comput. Phys. 52, 24 (RATTLE)
- Allen & Tildesley (1987). Computer Simulation of Liquids
"""

from __future__ import annotations

import os
from typing import Any

import warp as wp

__all__ = [
    # SHAKE - Mutating
    "shake_constraints",
    "shake_iteration",
    # SHAKE - Non-mutating
    "shake_constraints_out",
    "shake_iteration_out",
    # RATTLE - Mutating
    "rattle_constraints",
    "rattle_iteration",
    # RATTLE - Non-mutating
    "rattle_constraints_out",
    "rattle_iteration_out",
]


# ==============================================================================
# SHAKE Kernels
# ==============================================================================

# Tile block size for cooperative reductions
TILE_DIM = int(os.getenv("NVALCHEMIOPS_DYNAMICS_TILE_DIM", 256))


@wp.kernel
def _shake_iteration_kernel(
    positions: wp.array(dtype=Any),
    positions_old: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    bond_atom_i: wp.array(dtype=wp.int32),
    bond_atom_j: wp.array(dtype=wp.int32),
    bond_lengths_sq: wp.array(dtype=Any),
    max_error: wp.array(dtype=wp.float64),
):
    """Single SHAKE iteration for one bond (in-place).

    Updates positions to satisfy bond length constraint.

    Launch Grid
    -----------
    dim = [num_bonds]

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Current positions. MODIFIED in-place.
    positions_old : wp.array(dtype=wp.vec3f or wp.vec3d)
        Positions before unconstrained update.
    masses : wp.array
        Atomic masses.
    bond_atom_i : wp.array(dtype=wp.int32)
        First atom index for each bond.
    bond_atom_j : wp.array(dtype=wp.int32)
        Second atom index for each bond.
    bond_lengths_sq : wp.array
        Target bond length squared for each bond.
    max_error : wp.array(dtype=wp.float64)
        Maximum constraint error (atomic max).

    Notes
    -----
    This kernel uses atomic operations on positions, which may cause
    race conditions when atoms participate in multiple bonds.
    """
    bond_idx = wp.tid()

    i = bond_atom_i[bond_idx]
    j = bond_atom_j[bond_idx]

    r_i = positions[i]
    r_j = positions[j]
    r_i_old = positions_old[i]
    r_j_old = positions_old[j]
    m_i = masses[i]
    m_j = masses[j]
    d_sq = bond_lengths_sq[bond_idx]

    # Current bond vector
    r_ij = r_i - r_j

    # Old bond vector
    r_ij_old = r_i_old - r_j_old

    # Current distance squared
    r_sq = wp.dot(r_ij, r_ij)

    # Constraint violation
    sigma = r_sq - d_sq

    # Track maximum error
    wp.atomic_max(max_error, 0, wp.abs(wp.float64(sigma)))

    # Dot product r_ij · r_ij_old
    dot = wp.dot(r_ij, r_ij_old)

    # Inverse masses
    inv_m_i = type(m_i)(1.0) / m_i
    inv_m_j = type(m_j)(1.0) / m_j

    # Lagrange multiplier
    denom = type(dot)(2.0) * (inv_m_i + inv_m_j) * dot
    if wp.abs(denom) > type(denom)(1e-30):
        lam = sigma / denom

        # Position corrections
        corr_i = lam * r_ij_old * inv_m_i
        corr_j = lam * r_ij_old * inv_m_j

        # Apply corrections atomically
        wp.atomic_sub(positions, i, corr_i)
        wp.atomic_add(positions, j, corr_j)


@wp.kernel
def _shake_iteration_tiled_kernel(
    positions: wp.array(dtype=Any),
    positions_old: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    bond_atom_i: wp.array(dtype=wp.int32),
    bond_atom_j: wp.array(dtype=wp.int32),
    bond_lengths_sq: wp.array(dtype=Any),
    max_error: wp.array(dtype=wp.float64),
):
    """Single SHAKE iteration with tile reductions for error tracking.

    Updates positions to satisfy bond length constraint using atomic operations
    for position corrections. Uses tile reductions for max error computation.

    Launch Grid: dim = [num_bonds], block_dim = TILE_DIM

    Notes
    -----
    - Atomic operations on positions still required due to data dependencies
    - Tile reductions reduce atomic contention for max_error tracking
    - Data dependencies (atoms in multiple bonds) limit parallelization
    """
    bond_idx = wp.tid()

    i = bond_atom_i[bond_idx]
    j = bond_atom_j[bond_idx]

    r_i = positions[i]
    r_j = positions[j]
    r_i_old = positions_old[i]
    r_j_old = positions_old[j]
    m_i = masses[i]
    m_j = masses[j]
    d_sq = bond_lengths_sq[bond_idx]

    # Current bond vector
    r_ij = r_i - r_j

    # Old bond vector
    r_ij_old = r_i_old - r_j_old

    # Current distance squared
    r_sq = wp.dot(r_ij, r_ij)

    # Constraint violation
    sigma = r_sq - d_sq

    # Compute local error
    local_error = wp.abs(wp.float64(sigma))

    # Tile reduction for max error
    t_error = wp.tile(local_error)
    max_tile_error = wp.tile_reduce(wp.max, t_error)
    block_max_error = max_tile_error[0]

    # Only first thread in block updates max error
    if bond_idx % TILE_DIM == 0:
        wp.atomic_max(max_error, 0, block_max_error)

    # Dot product r_ij · r_ij_old
    dot = wp.dot(r_ij, r_ij_old)

    # Inverse masses
    inv_m_i = type(m_i)(1.0) / m_i
    inv_m_j = type(m_j)(1.0) / m_j

    # Lagrange multiplier
    denom = type(dot)(2.0) * (inv_m_i + inv_m_j) * dot
    if wp.abs(denom) > type(denom)(1e-30):
        lam = sigma / denom

        # Position corrections
        corr_i = lam * r_ij_old * inv_m_i
        corr_j = lam * r_ij_old * inv_m_j

        # Apply corrections atomically (required due to data dependencies)
        wp.atomic_sub(positions, i, corr_i)
        wp.atomic_add(positions, j, corr_j)


@wp.kernel
def _shake_iteration_out_kernel(
    positions: wp.array(dtype=Any),
    positions_old: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    bond_atom_i: wp.array(dtype=wp.int32),
    bond_atom_j: wp.array(dtype=wp.int32),
    bond_lengths_sq: wp.array(dtype=Any),
    position_corrections: wp.array(dtype=Any),
    max_error: wp.array(dtype=wp.float64),
):
    """Single SHAKE iteration computing corrections (non-mutating).

    Computes position corrections without applying them.

    Launch Grid
    -----------
    dim = [num_bonds]

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Current positions.
    positions_old : wp.array(dtype=wp.vec3f or wp.vec3d)
        Positions before unconstrained update.
    masses : wp.array
        Atomic masses.
    bond_atom_i : wp.array(dtype=wp.int32)
        First atom index for each bond.
    bond_atom_j : wp.array(dtype=wp.int32)
        Second atom index for each bond.
    bond_lengths_sq : wp.array
        Target bond length squared.
    position_corrections : wp.array(dtype=wp.vec3f or wp.vec3d)
        Output corrections for each atom. Shape (N,).
    max_error : wp.array(dtype=wp.float64)
        Maximum constraint error.
    """
    bond_idx = wp.tid()

    i = bond_atom_i[bond_idx]
    j = bond_atom_j[bond_idx]

    r_i = positions[i]
    r_j = positions[j]
    r_i_old = positions_old[i]
    r_j_old = positions_old[j]
    m_i = masses[i]
    m_j = masses[j]
    d_sq = bond_lengths_sq[bond_idx]

    r_ij = r_i - r_j
    r_ij_old = r_i_old - r_j_old

    r_sq = wp.dot(r_ij, r_ij)
    sigma = r_sq - d_sq

    wp.atomic_max(max_error, 0, wp.abs(wp.float64(sigma)))

    dot = wp.dot(r_ij, r_ij_old)

    inv_m_i = type(m_i)(1.0) / m_i
    inv_m_j = type(m_j)(1.0) / m_j

    denom = type(dot)(2.0) * (inv_m_i + inv_m_j) * dot
    if wp.abs(denom) > type(denom)(1e-30):
        lam = sigma / denom

        corr_i = -lam * r_ij_old * inv_m_i
        corr_j = lam * r_ij_old * inv_m_j

        wp.atomic_add(position_corrections, i, corr_i)
        wp.atomic_add(position_corrections, j, corr_j)


@wp.kernel
def _shake_iteration_out_tiled_kernel(
    positions: wp.array(dtype=Any),
    positions_old: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    bond_atom_i: wp.array(dtype=wp.int32),
    bond_atom_j: wp.array(dtype=wp.int32),
    bond_lengths_sq: wp.array(dtype=Any),
    position_corrections: wp.array(dtype=Any),
    max_error: wp.array(dtype=wp.float64),
):
    """Single SHAKE iteration with tile reductions (non-mutating).

    Launch Grid: dim = [num_bonds], block_dim = TILE_DIM
    """
    bond_idx = wp.tid()

    i = bond_atom_i[bond_idx]
    j = bond_atom_j[bond_idx]

    r_i = positions[i]
    r_j = positions[j]
    r_i_old = positions_old[i]
    r_j_old = positions_old[j]
    m_i = masses[i]
    m_j = masses[j]
    d_sq = bond_lengths_sq[bond_idx]

    r_ij = r_i - r_j
    r_ij_old = r_i_old - r_j_old

    r_sq = wp.dot(r_ij, r_ij)
    sigma = r_sq - d_sq

    # Compute local error
    local_error = wp.abs(wp.float64(sigma))

    # Tile reduction for max error
    t_error = wp.tile(local_error)
    max_tile_error = wp.tile_reduce(wp.max, t_error)
    block_max_error = max_tile_error[0]

    # Only first thread in block updates max error
    if bond_idx % TILE_DIM == 0:
        wp.atomic_max(max_error, 0, block_max_error)

    dot = wp.dot(r_ij, r_ij_old)

    inv_m_i = type(m_i)(1.0) / m_i
    inv_m_j = type(m_j)(1.0) / m_j

    denom = type(dot)(2.0) * (inv_m_i + inv_m_j) * dot
    if wp.abs(denom) > type(denom)(1e-30):
        lam = sigma / denom

        corr_i = -lam * r_ij_old * inv_m_i
        corr_j = lam * r_ij_old * inv_m_j

        # Atomic operations still required due to data dependencies
        wp.atomic_add(position_corrections, i, corr_i)
        wp.atomic_add(position_corrections, j, corr_j)


# ==============================================================================
# RATTLE Kernels
# ==============================================================================


@wp.kernel
def _rattle_iteration_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    bond_atom_i: wp.array(dtype=wp.int32),
    bond_atom_j: wp.array(dtype=wp.int32),
    max_error: wp.array(dtype=wp.float64),
):
    """Single RATTLE iteration for velocity constraints (in-place).

    Ensures velocities are perpendicular to bond vectors.

    Launch Grid
    -----------
    dim = [num_bonds]

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Current (constrained) positions.
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Velocities. MODIFIED in-place.
    masses : wp.array
        Atomic masses.
    bond_atom_i : wp.array(dtype=wp.int32)
        First atom index for each bond.
    bond_atom_j : wp.array(dtype=wp.int32)
        Second atom index for each bond.
    max_error : wp.array(dtype=wp.float64)
        Maximum constraint error.
    """
    bond_idx = wp.tid()

    i = bond_atom_i[bond_idx]
    j = bond_atom_j[bond_idx]

    r_i = positions[i]
    r_j = positions[j]
    v_i = velocities[i]
    v_j = velocities[j]
    m_i = masses[i]
    m_j = masses[j]

    # Bond vector
    r_ij = r_i - r_j

    # Relative velocity
    v_ij = v_i - v_j

    # Velocity constraint: v_ij · r_ij = 0
    kappa = wp.dot(v_ij, r_ij)

    # Track maximum error
    wp.atomic_max(max_error, 0, wp.abs(wp.float64(kappa)))

    # Bond length squared
    r_sq = wp.dot(r_ij, r_ij)

    # Inverse masses
    inv_m_i = type(m_i)(1.0) / m_i
    inv_m_j = type(m_j)(1.0) / m_j

    # Lagrange multiplier
    denom = (inv_m_i + inv_m_j) * r_sq
    if wp.abs(denom) > type(denom)(1e-30):
        mu = kappa / denom

        # Velocity corrections
        corr_i = -mu * r_ij * inv_m_i
        corr_j = mu * r_ij * inv_m_j

        # Apply corrections atomically
        wp.atomic_sub(velocities, i, corr_i)
        wp.atomic_add(velocities, j, corr_j)


@wp.kernel
def _rattle_iteration_out_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    bond_atom_i: wp.array(dtype=wp.int32),
    bond_atom_j: wp.array(dtype=wp.int32),
    velocity_corrections: wp.array(dtype=Any),
    max_error: wp.array(dtype=wp.float64),
):
    """Single RATTLE iteration computing corrections (non-mutating).

    Launch Grid
    -----------
    dim = [num_bonds]
    """
    bond_idx = wp.tid()

    i = bond_atom_i[bond_idx]
    j = bond_atom_j[bond_idx]

    r_i = positions[i]
    r_j = positions[j]
    v_i = velocities[i]
    v_j = velocities[j]
    m_i = masses[i]
    m_j = masses[j]

    r_ij = r_i - r_j
    v_ij = v_i - v_j

    kappa = wp.dot(v_ij, r_ij)

    wp.atomic_max(max_error, 0, wp.abs(wp.float64(kappa)))

    r_sq = wp.dot(r_ij, r_ij)

    inv_m_i = type(m_i)(1.0) / m_i
    inv_m_j = type(m_j)(1.0) / m_j

    denom = (inv_m_i + inv_m_j) * r_sq
    if wp.abs(denom) > type(denom)(1e-30):
        mu = kappa / denom

        corr_i = -mu * r_ij * inv_m_i
        corr_j = mu * r_ij * inv_m_j

        wp.atomic_add(velocity_corrections, i, corr_i)
        wp.atomic_add(velocity_corrections, j, corr_j)


@wp.kernel
def _rattle_iteration_tiled_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    bond_atom_i: wp.array(dtype=wp.int32),
    bond_atom_j: wp.array(dtype=wp.int32),
    max_error: wp.array(dtype=wp.float64),
):
    """Single RATTLE iteration with tile reductions for error tracking.

    Updates velocities to satisfy velocity constraints using atomic operations
    for velocity corrections. Uses tile reductions for max error computation.

    Launch Grid: dim = [num_bonds], block_dim = TILE_DIM

    Notes
    -----
    - Atomic operations on velocities still required due to data dependencies
    - Tile reductions reduce atomic contention for max_error tracking
    - Data dependencies (atoms in multiple bonds) limit parallelization
    """
    bond_idx = wp.tid()

    i = bond_atom_i[bond_idx]
    j = bond_atom_j[bond_idx]

    r_i = positions[i]
    r_j = positions[j]
    v_i = velocities[i]
    v_j = velocities[j]
    m_i = masses[i]
    m_j = masses[j]

    # Bond vector
    r_ij = r_i - r_j

    # Relative velocity
    v_ij = v_i - v_j

    # Velocity constraint: v_ij · r_ij = 0
    kappa = wp.dot(v_ij, r_ij)

    # Compute local error
    local_error = wp.abs(wp.float64(kappa))

    # Tile reduction for max error
    t_error = wp.tile(local_error)
    max_tile_error = wp.tile_reduce(wp.max, t_error)
    block_max_error = max_tile_error[0]

    # Only first thread in block updates max error
    if bond_idx % TILE_DIM == 0:
        wp.atomic_max(max_error, 0, block_max_error)

    # Bond length squared
    r_sq = wp.dot(r_ij, r_ij)

    # Inverse masses
    inv_m_i = type(m_i)(1.0) / m_i
    inv_m_j = type(m_j)(1.0) / m_j

    # Lagrange multiplier
    denom = (inv_m_i + inv_m_j) * r_sq
    if wp.abs(denom) > type(denom)(1e-30):
        mu = kappa / denom

        # Velocity corrections
        corr_i = -mu * r_ij * inv_m_i
        corr_j = mu * r_ij * inv_m_j

        # Apply corrections atomically (required due to data dependencies)
        wp.atomic_sub(velocities, i, corr_i)
        wp.atomic_add(velocities, j, corr_j)


@wp.kernel
def _rattle_iteration_out_tiled_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    bond_atom_i: wp.array(dtype=wp.int32),
    bond_atom_j: wp.array(dtype=wp.int32),
    velocity_corrections: wp.array(dtype=Any),
    max_error: wp.array(dtype=wp.float64),
):
    """Single RATTLE iteration with tile reductions (non-mutating).

    Launch Grid: dim = [num_bonds], block_dim = TILE_DIM
    """
    bond_idx = wp.tid()

    i = bond_atom_i[bond_idx]
    j = bond_atom_j[bond_idx]

    r_i = positions[i]
    r_j = positions[j]
    v_i = velocities[i]
    v_j = velocities[j]
    m_i = masses[i]
    m_j = masses[j]

    r_ij = r_i - r_j
    v_ij = v_i - v_j

    kappa = wp.dot(v_ij, r_ij)

    # Compute local error
    local_error = wp.abs(wp.float64(kappa))

    # Tile reduction for max error
    t_error = wp.tile(local_error)
    max_tile_error = wp.tile_reduce(wp.max, t_error)
    block_max_error = max_tile_error[0]

    # Only first thread in block updates max error
    if bond_idx % TILE_DIM == 0:
        wp.atomic_max(max_error, 0, block_max_error)

    r_sq = wp.dot(r_ij, r_ij)

    inv_m_i = type(m_i)(1.0) / m_i
    inv_m_j = type(m_j)(1.0) / m_j

    denom = (inv_m_i + inv_m_j) * r_sq
    if wp.abs(denom) > type(denom)(1e-30):
        mu = kappa / denom

        corr_i = -mu * r_ij * inv_m_i
        corr_j = mu * r_ij * inv_m_j

        # Atomic operations still required due to data dependencies
        wp.atomic_add(velocity_corrections, i, corr_i)
        wp.atomic_add(velocity_corrections, j, corr_j)


# ==============================================================================
# Kernel Overloads for Explicit Typing
# ==============================================================================

_T = [wp.float32, wp.float64]  # Scalar types
_V = [wp.vec3f, wp.vec3d]  # Vector types

_shake_iteration_kernel_overload = {}
_shake_iteration_out_kernel_overload = {}
_shake_iteration_tiled_kernel_overload = {}
_shake_iteration_out_tiled_kernel_overload = {}
_rattle_iteration_kernel_overload = {}
_rattle_iteration_out_kernel_overload = {}
_rattle_iteration_tiled_kernel_overload = {}
_rattle_iteration_out_tiled_kernel_overload = {}

for t, v in zip(_T, _V):
    _shake_iteration_kernel_overload[v] = wp.overload(
        _shake_iteration_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
            wp.array(dtype=wp.float64),
        ],
    )
    _shake_iteration_out_kernel_overload[v] = wp.overload(
        _shake_iteration_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
            wp.array(dtype=v),
            wp.array(dtype=wp.float64),
        ],
    )
    _rattle_iteration_kernel_overload[v] = wp.overload(
        _rattle_iteration_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.float64),
        ],
    )
    _rattle_iteration_out_kernel_overload[v] = wp.overload(
        _rattle_iteration_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
            wp.array(dtype=wp.float64),
        ],
    )
    _shake_iteration_tiled_kernel_overload[v] = wp.overload(
        _shake_iteration_tiled_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
            wp.array(dtype=wp.float64),
        ],
    )
    _shake_iteration_out_tiled_kernel_overload[v] = wp.overload(
        _shake_iteration_out_tiled_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
            wp.array(dtype=v),
            wp.array(dtype=wp.float64),
        ],
    )
    _rattle_iteration_tiled_kernel_overload[v] = wp.overload(
        _rattle_iteration_tiled_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.float64),
        ],
    )
    _rattle_iteration_out_tiled_kernel_overload[v] = wp.overload(
        _rattle_iteration_out_tiled_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
            wp.array(dtype=wp.float64),
        ],
    )


# ==============================================================================
# SHAKE Functional Interface
# ==============================================================================


def shake_iteration(
    positions: wp.array,
    positions_old: wp.array,
    masses: wp.array,
    bond_atom_i: wp.array,
    bond_atom_j: wp.array,
    bond_lengths_sq: wp.array,
    max_error: wp.array,
    device: str = None,
) -> wp.array:
    """
    Perform single SHAKE iteration (in-place).

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Current positions. Shape (N,). MODIFIED in-place.
    positions_old : wp.array(dtype=wp.vec3f or wp.vec3d)
        Positions before unconstrained update. Shape (N,).
    masses : wp.array
        Atomic masses. Shape (N,).
    bond_atom_i : wp.array(dtype=wp.int32)
        First atom index for each bond. Shape (M,).
    bond_atom_j : wp.array(dtype=wp.int32)
        Second atom index for each bond. Shape (M,).
    bond_lengths_sq : wp.array
        Target bond length squared. Shape (M,).
    max_error : wp.array(dtype=wp.float64)
        Output array to store max error. Shape (1,).
        Zeroed internally before each use.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array(dtype=wp.float64)
        Maximum constraint error :math:`|r^2_{ij} - d^2_{ij}|`. Shape (1,).
    """
    if device is None:
        device = positions.device

    max_error.zero_()
    num_bonds = bond_atom_i.shape[0]

    vec_dtype = positions.dtype
    wp.launch(
        _shake_iteration_tiled_kernel_overload[vec_dtype],
        dim=num_bonds,
        inputs=[
            positions,
            positions_old,
            masses,
            bond_atom_i,
            bond_atom_j,
            bond_lengths_sq,
            max_error,
        ],
        device=device,
        block_dim=TILE_DIM,
    )

    return max_error


def shake_constraints(
    positions: wp.array,
    positions_old: wp.array,
    masses: wp.array,
    bond_atom_i: wp.array,
    bond_atom_j: wp.array,
    bond_lengths_sq: wp.array,
    max_error: wp.array,
    num_iter: int = 10,
    device: str = None,
) -> wp.array:
    """
    Apply SHAKE constraints for a fixed number of iterations (in-place).

    This function runs a fixed number of SHAKE iterations without convergence
    checking during the loop. The final error is returned as a wp.array.
    The caller can check convergence by inspecting the error value.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Current positions. Shape (N,). MODIFIED in-place.
    positions_old : wp.array(dtype=wp.vec3f or wp.vec3d)
        Positions before unconstrained update. Shape (N,).
    masses : wp.array
        Atomic masses. Shape (N,).
    bond_atom_i : wp.array(dtype=wp.int32)
        First atom index for each bond. Shape (M,).
    bond_atom_j : wp.array(dtype=wp.int32)
        Second atom index for each bond. Shape (M,).
    bond_lengths_sq : wp.array
        Target bond length squared. Shape (M,).
    max_error : wp.array(dtype=wp.float64)
        Scratch array for max error tracking. Shape (1,).
        Caller must pre-allocate. Zeroed internally between iterations.
    num_iter : int, optional
        Number of iterations to run. Default 10.
        Typical values: 3-20 depending on constraint stiffness.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array(dtype=wp.float64)
        Final constraint error :math:`|r^2_{ij} - d^2_{ij}|`. Shape (1,).

    Example
    -------
    >>> # After unconstrained position update
    >>> max_error = wp.empty(1, dtype=wp.float64, device=device)
    >>> final_error = shake_constraints(
    ...     positions, positions_old, masses,
    ...     bond_i, bond_j, bond_lengths_sq, max_error,
    ...     num_iter=10
    ... )
    """
    if device is None:
        device = positions.device

    for _ in range(num_iter):
        max_error = shake_iteration(
            positions,
            positions_old,
            masses,
            bond_atom_i,
            bond_atom_j,
            bond_lengths_sq,
            max_error,
            device,
        )

    return max_error


def shake_iteration_out(
    positions: wp.array,
    positions_old: wp.array,
    masses: wp.array,
    bond_atom_i: wp.array,
    bond_atom_j: wp.array,
    bond_lengths_sq: wp.array,
    position_corrections: wp.array,
    max_error: wp.array,
    device: str = None,
) -> tuple[wp.array, wp.array]:
    """
    Compute SHAKE corrections without applying (non-mutating).

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Current positions. Shape (N,).
    positions_old : wp.array(dtype=wp.vec3f or wp.vec3d)
        Positions before unconstrained update. Shape (N,).
    masses : wp.array
        Atomic masses. Shape (N,).
    bond_atom_i : wp.array(dtype=wp.int32)
        First atom index for each bond. Shape (M,).
    bond_atom_j : wp.array(dtype=wp.int32)
        Second atom index for each bond. Shape (M,).
    bond_lengths_sq : wp.array
        Target bond length squared. Shape (M,).
    position_corrections : wp.array
        Output corrections. Shape (N,). Zeroed internally before each use.
    max_error : wp.array(dtype=wp.float64)
        Output max error. Shape (1,). Zeroed internally before each use.
    device : str, optional
        Warp device.

    Returns
    -------
    tuple[wp.array, wp.array]
        (position_corrections, max_error)
        max_error is shape (1,)
    """
    if device is None:
        device = positions.device

    position_corrections.zero_()
    max_error.zero_()
    num_bonds = bond_atom_i.shape[0]

    vec_dtype = positions.dtype
    wp.launch(
        _shake_iteration_out_tiled_kernel_overload[vec_dtype],
        dim=num_bonds,
        inputs=[
            positions,
            positions_old,
            masses,
            bond_atom_i,
            bond_atom_j,
            bond_lengths_sq,
            position_corrections,
            max_error,
        ],
        device=device,
        block_dim=TILE_DIM,
    )

    return position_corrections, max_error


def shake_constraints_out(
    positions: wp.array,
    positions_old: wp.array,
    masses: wp.array,
    bond_atom_i: wp.array,
    bond_atom_j: wp.array,
    bond_lengths_sq: wp.array,
    positions_out: wp.array,
    max_error: wp.array,
    num_iter: int = 10,
    device: str = None,
) -> tuple[wp.array, wp.array]:
    """
    Apply SHAKE constraints (non-mutating).

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Current positions. Shape (N,).
    positions_old : wp.array(dtype=wp.vec3f or wp.vec3d)
        Positions before unconstrained update. Shape (N,).
    masses : wp.array
        Atomic masses. Shape (N,).
    bond_atom_i : wp.array(dtype=wp.int32)
        First atom index for each bond. Shape (M,).
    bond_atom_j : wp.array(dtype=wp.int32)
        Second atom index for each bond. Shape (M,).
    bond_lengths_sq : wp.array
        Target bond length squared. Shape (M,).
    positions_out : wp.array
        Output positions. Shape (N,). Caller must pre-allocate.
        Caller must copy input positions into this array before calling
        (e.g., via ``wp.copy(positions_out, positions)``).
    max_error : wp.array(dtype=wp.float64)
        Scratch array for max error tracking. Shape (1,).
        Caller must pre-allocate. Zeroed internally between iterations.
    num_iter : int, optional
        Number of iterations to run. Default 10.
    device : str, optional
        Warp device.

    Returns
    -------
    tuple[wp.array, wp.array]
        (positions_out, final_error)
        final_error is shape (1,)
    """
    if device is None:
        device = positions.device

    error = shake_constraints(
        positions_out,
        positions_old,
        masses,
        bond_atom_i,
        bond_atom_j,
        bond_lengths_sq,
        max_error,
        num_iter,
        device,
    )

    return positions_out, error


# ==============================================================================
# RATTLE Functional Interface
# ==============================================================================


def rattle_iteration(
    positions: wp.array,
    velocities: wp.array,
    masses: wp.array,
    bond_atom_i: wp.array,
    bond_atom_j: wp.array,
    max_error: wp.array,
    device: str = None,
) -> wp.array:
    r"""
    Perform single RATTLE iteration (in-place).

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Current (constrained) positions. Shape (N,).
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Velocities. Shape (N,). MODIFIED in-place.
    masses : wp.array
        Atomic masses. Shape (N,).
    bond_atom_i : wp.array(dtype=wp.int32)
        First atom index for each bond. Shape (M,).
    bond_atom_j : wp.array(dtype=wp.int32)
        Second atom index for each bond. Shape (M,).
    max_error : wp.array(dtype=wp.float64)
        Output array to store max error. Shape (1,).
        Zeroed internally before each use.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array(dtype=wp.float64)
        Maximum velocity constraint error :math:`|v_{ij} \cdot r_{ij}|`. Shape (1,).
    """
    if device is None:
        device = positions.device

    max_error.zero_()
    num_bonds = bond_atom_i.shape[0]

    vec_dtype = positions.dtype
    wp.launch(
        _rattle_iteration_tiled_kernel_overload[vec_dtype],
        dim=num_bonds,
        inputs=[positions, velocities, masses, bond_atom_i, bond_atom_j, max_error],
        device=device,
        block_dim=TILE_DIM,
    )

    return max_error


def rattle_constraints(
    positions: wp.array,
    velocities: wp.array,
    masses: wp.array,
    bond_atom_i: wp.array,
    bond_atom_j: wp.array,
    max_error: wp.array,
    num_iter: int = 10,
    device: str = None,
) -> wp.array:
    r"""
    Apply RATTLE velocity constraints for a fixed number of iterations (in-place).

    This function runs a fixed number of RATTLE iterations without convergence
    checking during the loop. The final error is returned as a wp.array.
    The caller can check convergence by inspecting the error value.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Current (constrained) positions. Shape (N,).
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Velocities. Shape (N,). MODIFIED in-place.
    masses : wp.array
        Atomic masses. Shape (N,).
    bond_atom_i : wp.array(dtype=wp.int32)
        First atom index for each bond. Shape (M,).
    bond_atom_j : wp.array(dtype=wp.int32)
        Second atom index for each bond. Shape (M,).
    max_error : wp.array(dtype=wp.float64)
        Scratch array for max error tracking. Shape (1,).
        Caller must pre-allocate. Zeroed internally between iterations.
    num_iter : int, optional
        Number of iterations to run. Default 10.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array(dtype=wp.float64)
        Final constraint error :math:`|v_{ij} \cdot r_{ij}|`. Shape (1,).
    """
    if device is None:
        device = positions.device

    for _ in range(num_iter):
        max_error = rattle_iteration(
            positions, velocities, masses, bond_atom_i, bond_atom_j, max_error, device
        )

    return max_error


def rattle_iteration_out(
    positions: wp.array,
    velocities: wp.array,
    masses: wp.array,
    bond_atom_i: wp.array,
    bond_atom_j: wp.array,
    velocity_corrections: wp.array,
    max_error: wp.array,
    device: str = None,
) -> tuple[wp.array, wp.array]:
    """
    Compute RATTLE corrections without applying (non-mutating).

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Current (constrained) positions. Shape (N,).
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Velocities. Shape (N,).
    masses : wp.array
        Atomic masses. Shape (N,).
    bond_atom_i : wp.array(dtype=wp.int32)
        First atom index for each bond. Shape (M,).
    bond_atom_j : wp.array(dtype=wp.int32)
        Second atom index for each bond. Shape (M,).
    velocity_corrections : wp.array
        Output corrections. Shape (N,). Zeroed internally before each use.
    max_error : wp.array(dtype=wp.float64)
        Output max error. Shape (1,). Zeroed internally before each use.
    device : str, optional
        Warp device.

    Returns
    -------
    tuple[wp.array, wp.array]
        (velocity_corrections, max_error)
        max_error is shape (1,)
    """
    if device is None:
        device = positions.device

    velocity_corrections.zero_()
    max_error.zero_()
    num_bonds = bond_atom_i.shape[0]

    vec_dtype = positions.dtype
    wp.launch(
        _rattle_iteration_out_tiled_kernel_overload[vec_dtype],
        dim=num_bonds,
        inputs=[
            positions,
            velocities,
            masses,
            bond_atom_i,
            bond_atom_j,
            velocity_corrections,
            max_error,
        ],
        device=device,
        block_dim=TILE_DIM,
    )

    return velocity_corrections, max_error


def rattle_constraints_out(
    positions: wp.array,
    velocities: wp.array,
    masses: wp.array,
    bond_atom_i: wp.array,
    bond_atom_j: wp.array,
    velocities_out: wp.array,
    max_error: wp.array,
    num_iter: int = 10,
    device: str = None,
) -> tuple[wp.array, wp.array]:
    """
    Apply RATTLE velocity constraints (non-mutating).

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Current (constrained) positions. Shape (N,).
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Velocities. Shape (N,).
    masses : wp.array
        Atomic masses. Shape (N,).
    bond_atom_i : wp.array(dtype=wp.int32)
        First atom index for each bond. Shape (M,).
    bond_atom_j : wp.array(dtype=wp.int32)
        Second atom index for each bond. Shape (M,).
    velocities_out : wp.array
        Output velocities. Shape (N,). Caller must pre-allocate.
        Caller must copy input velocities into this array before calling
        (e.g., via ``wp.copy(velocities_out, velocities)``).
    max_error : wp.array(dtype=wp.float64)
        Scratch array for max error tracking. Shape (1,).
        Caller must pre-allocate. Zeroed internally between iterations.
    num_iter : int, optional
        Number of iterations to run. Default 10.
    device : str, optional
        Warp device.

    Returns
    -------
    tuple[wp.array, wp.array]
        (velocities_out, final_error)
        final_error is shape (1,)
    """
    if device is None:
        device = positions.device

    error = rattle_constraints(
        positions,
        velocities_out,
        masses,
        bond_atom_i,
        bond_atom_j,
        max_error,
        num_iter,
        device,
    )

    return velocities_out, error
