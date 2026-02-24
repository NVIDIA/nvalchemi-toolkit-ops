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
Velocity Verlet Integrator Kernels
==================================

GPU-accelerated Warp kernels for the velocity Verlet integrator,
providing time-reversible, symplectic integration for NVE molecular dynamics.

This module provides both mutating (in-place) and non-mutating versions
of each kernel for gradient tracking compatibility.

MATHEMATICAL FORMULATION
========================

The velocity Verlet algorithm is a second-order symplectic integrator:

Position update:

.. math::

    \\mathbf{r}(t + \\Delta t) = \\mathbf{r}(t) + \\mathbf{v}(t) \\Delta t
                                + \\frac{1}{2} \\mathbf{a}(t) \\Delta t^2

Velocity update:

.. math::

    \\mathbf{v}(t + \\Delta t) = \\mathbf{v}(t) + \\frac{1}{2}[\\mathbf{a}(t)
                                + \\mathbf{a}(t + \\Delta t)] \\Delta t

where :math:`\\mathbf{a} = \\mathbf{F}/m` is the acceleration.

TWO-PASS ALGORITHM
==================

**Pass 1 (position_update):**
    - Update positions to r(t+dt)
    - Update velocities to half-step v(t+dt/2) = v(t) + 0.5*a(t)*dt

**[User recalculates forces at new positions]**

**Pass 2 (velocity_finalize):**
    - Complete velocity update: v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt

USAGE EXAMPLE
=============

Mutating (in-place) version::

    for step in range(num_steps):
        # Pass 1: Update positions and half-step velocities
        velocity_verlet_position_update(
            positions, velocities, forces, masses, dt
        )

        # Recalculate forces at new positions
        forces = compute_forces(positions)

        # Pass 2: Complete velocity update
        velocity_verlet_velocity_finalize(
            velocities, forces, masses, dt
        )

Non-mutating version (for gradient tracking)::

    positions_out = wp.zeros_like(positions)
    velocities_out = wp.zeros_like(velocities)

    for step in range(num_steps):
        # Pass 1: Compute new positions and velocities
        positions_out, velocities_out = velocity_verlet_position_update_out(
            positions, velocities, forces, masses, dt,
            positions_out, velocities_out,
        )

        # Recalculate forces at new positions
        new_forces = compute_forces(positions_out)

        # Pass 2: Complete velocity update
        velocities_final = wp.zeros_like(velocities)
        velocities_final = velocity_verlet_velocity_finalize_out(
            velocities_out, new_forces, masses, dt,
            velocities_final,
        )

        positions, velocities, forces = positions_out, velocities_final, new_forces

BATCH MODE
==========

This module supports three execution modes for simulating multiple systems:

**Single System Mode** (default)::

    dt = wp.array([0.001], dtype=wp.float64, device="cuda:0")
    velocity_verlet_position_update(positions, velocities, forces, masses, dt)

**Batch Mode with batch_idx** (atomic operations)::

    # For systems with varying atom counts
    # batch_idx maps each atom to its system
    batch_idx = wp.array([0]*N0 + [1]*N1 + [2]*N2, dtype=wp.int32, device="cuda:0")
    dt = wp.array([dt0, dt1, dt2], dtype=wp.float64, device="cuda:0")  # Per-system timesteps

    velocity_verlet_position_update(
        positions, velocities, forces, masses, dt, batch_idx=batch_idx
    )

    # Launched with dim=num_atoms_total (parallel per-atom operations)

**Batch Mode with atom_ptr** (sequential per-system)::

    # For systems where each thread should process one complete system
    # atom_ptr defines atom ranges: system s owns atoms [atom_ptr[s], atom_ptr[s+1])
    atom_ptr = wp.array([0, N0, N0+N1, N0+N1+N2], dtype=wp.int32, device="cuda:0")
    dt = wp.array([dt0, dt1, dt2], dtype=wp.float64, device="cuda:0")

    velocity_verlet_position_update(
        positions, velocities, forces, masses, dt, atom_ptr=atom_ptr
    )

    # Launched with dim=num_systems (each thread processes one system sequentially)

**Choosing Between batch_idx and atom_ptr:**

- Use **batch_idx** when:
    - Systems have similar sizes
    - You want maximum parallelism (one thread per atom)
    - Memory access patterns are coalesced

- Use **atom_ptr** when:
    - Systems have very different sizes
    - You need per-system operations (reductions, etc.)
    - Each system needs independent sequential processing

REFERENCES
==========

- Swope et al. (1982). J. Chem. Phys. 76, 637 (Velocity Verlet)
- Verlet, L. (1967). Phys. Rev. 159, 98 (Original Verlet method)
"""

from __future__ import annotations

from typing import Any

import warp as wp

from nvalchemiops.dynamics.utils.launch_helpers import (
    build_family_dict,
    dispatch_family,
)
from nvalchemiops.dynamics.utils.shared_kernels import velocity_kick_families
from nvalchemiops.warp_dispatch import validate_out_array

__all__ = [
    # Mutating (in-place) APIs
    "velocity_verlet_position_update",
    "velocity_verlet_velocity_finalize",
    # Non-mutating (output) APIs
    "velocity_verlet_position_update_out",
    "velocity_verlet_velocity_finalize_out",
]


# ==============================================================================
# Mutating Kernels (in-place updates)
# ==============================================================================


@wp.kernel
def _velocity_verlet_position_update_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
):
    """Update positions and half-step velocities (in-place).

    Computes:
        r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        v_half = v(t) + 0.5*a(t)*dt

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    force = forces[atom_idx]
    mass = masses[atom_idx]
    dt_val = dt[0]

    inv_mass = type(mass)(1.0) / mass

    # Compute acceleration
    acc_x = force[0] * inv_mass
    acc_y = force[1] * inv_mass
    acc_z = force[2] * inv_mass

    # Position update: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
    half_dt_sq = type(dt_val)(0.5) * dt_val * dt_val
    new_pos_x = pos[0] + vel[0] * dt_val + acc_x * half_dt_sq
    new_pos_y = pos[1] + vel[1] * dt_val + acc_y * half_dt_sq
    new_pos_z = pos[2] + vel[2] * dt_val + acc_z * half_dt_sq

    # Half-step velocity: v_half = v(t) + 0.5*a(t)*dt
    half_dt = type(dt_val)(0.5) * dt_val
    half_vel_x = vel[0] + acc_x * half_dt
    half_vel_y = vel[1] + acc_y * half_dt
    half_vel_z = vel[2] + acc_z * half_dt

    # Write back
    positions[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities[atom_idx] = type(vel)(half_vel_x, half_vel_y, half_vel_z)


# ==============================================================================
# Non-Mutating Kernels (write to output arrays)
# ==============================================================================


@wp.kernel
def _velocity_verlet_position_update_out_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Update positions and half-step velocities (non-mutating).

    Computes:
        r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        v_half = v(t) + 0.5*a(t)*dt

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    force = forces[atom_idx]
    mass = masses[atom_idx]
    dt_val = dt[0]

    # Guard against division by zero: if mass is zero, set inv_mass to zero
    inv_mass = wp.where(mass > type(mass)(0.0), type(mass)(1.0) / mass, type(mass)(0.0))

    # Compute acceleration
    acc_x = force[0] * inv_mass
    acc_y = force[1] * inv_mass
    acc_z = force[2] * inv_mass

    # Position update: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
    half_dt_sq = type(dt_val)(0.5) * dt_val * dt_val
    new_pos_x = pos[0] + vel[0] * dt_val + acc_x * half_dt_sq
    new_pos_y = pos[1] + vel[1] * dt_val + acc_y * half_dt_sq
    new_pos_z = pos[2] + vel[2] * dt_val + acc_z * half_dt_sq

    # Half-step velocity: v_half = v(t) + 0.5*a(t)*dt
    half_dt = type(dt_val)(0.5) * dt_val
    half_vel_x = vel[0] + acc_x * half_dt
    half_vel_y = vel[1] + acc_y * half_dt
    half_vel_z = vel[2] + acc_z * half_dt

    # Write to output arrays
    positions_out[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities_out[atom_idx] = type(vel)(half_vel_x, half_vel_y, half_vel_z)


# ==============================================================================
# Batched Mutating Kernels
# ==============================================================================


@wp.kernel
def _batch_velocity_verlet_position_update_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
):
    """Update positions and half-step velocities for batched systems (in-place).

    Each atom uses the timestep for its corresponding system via batch_idx.

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()

    # Get per-system timestep via batch_idx
    system_id = batch_idx[atom_idx]
    dt_val = dt[system_id]

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    force = forces[atom_idx]
    mass = masses[atom_idx]

    # Guard against division by zero: if mass is zero, set inv_mass to zero
    if mass > type(mass)(0.0):
        inv_mass = type(mass)(1.0) / mass
    else:
        inv_mass = type(mass)(0.0)

    acc_x = force[0] * inv_mass
    acc_y = force[1] * inv_mass
    acc_z = force[2] * inv_mass

    half_dt_sq = type(dt_val)(0.5) * dt_val * dt_val
    new_pos_x = pos[0] + vel[0] * dt_val + acc_x * half_dt_sq
    new_pos_y = pos[1] + vel[1] * dt_val + acc_y * half_dt_sq
    new_pos_z = pos[2] + vel[2] * dt_val + acc_z * half_dt_sq

    half_dt = type(dt_val)(0.5) * dt_val
    half_vel_x = vel[0] + acc_x * half_dt
    half_vel_y = vel[1] + acc_y * half_dt
    half_vel_z = vel[2] + acc_z * half_dt

    positions[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities[atom_idx] = type(vel)(half_vel_x, half_vel_y, half_vel_z)


# ==============================================================================
# Batched Non-Mutating Kernels
# ==============================================================================


@wp.kernel
def _batch_velocity_verlet_position_update_out_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Update positions and half-step velocities for batched systems (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()

    system_id = batch_idx[atom_idx]
    dt_val = dt[system_id]

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    force = forces[atom_idx]
    mass = masses[atom_idx]

    # Guard against division by zero: if mass is zero, set inv_mass to zero
    if mass > type(mass)(0.0):
        inv_mass = type(mass)(1.0) / mass
    else:
        inv_mass = type(mass)(0.0)

    acc_x = force[0] * inv_mass
    acc_y = force[1] * inv_mass
    acc_z = force[2] * inv_mass

    half_dt_sq = type(dt_val)(0.5) * dt_val * dt_val
    new_pos_x = pos[0] + vel[0] * dt_val + acc_x * half_dt_sq
    new_pos_y = pos[1] + vel[1] * dt_val + acc_y * half_dt_sq
    new_pos_z = pos[2] + vel[2] * dt_val + acc_z * half_dt_sq

    half_dt = type(dt_val)(0.5) * dt_val
    half_vel_x = vel[0] + acc_x * half_dt
    half_vel_y = vel[1] + acc_y * half_dt
    half_vel_z = vel[2] + acc_z * half_dt

    positions_out[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities_out[atom_idx] = type(vel)(half_vel_x, half_vel_y, half_vel_z)


# ==============================================================================
# Pointer-Based (CSR) Mutating Kernels
# ==============================================================================


@wp.kernel
def _velocity_verlet_position_update_ptr_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
):
    """Update positions and half-step velocities using atom_ptr (in-place).

    Each thread processes one system's atoms sequentially.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (num_atoms_total,). MODIFIED in-place.
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (num_atoms_total,). MODIFIED in-place.
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (num_atoms_total,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (num_atoms_total,).
    atom_ptr : wp.array(dtype=wp.int32)
        CSR-style pointers. Shape (num_systems + 1,).
        System s owns atoms in range [atom_ptr[s], atom_ptr[s+1]).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep per system. Shape (num_systems,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]
    dt_val = dt[sys_id]

    half_dt = type(dt_val)(0.5) * dt_val
    half_dt_sq = type(dt_val)(0.5) * dt_val * dt_val

    for i in range(a0, a1):
        pos = positions[i]
        vel = velocities[i]
        force = forces[i]
        mass = masses[i]

        # Guard against division by zero: if mass is zero, set inv_mass to zero
        if mass > type(mass)(0.0):
            inv_mass = type(mass)(1.0) / mass
        else:
            inv_mass = type(mass)(0.0)

        # Compute acceleration
        acc_x = force[0] * inv_mass
        acc_y = force[1] * inv_mass
        acc_z = force[2] * inv_mass

        # Position update: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        new_pos_x = pos[0] + vel[0] * dt_val + acc_x * half_dt_sq
        new_pos_y = pos[1] + vel[1] * dt_val + acc_y * half_dt_sq
        new_pos_z = pos[2] + vel[2] * dt_val + acc_z * half_dt_sq

        # Half-step velocity: v_half = v(t) + 0.5*a(t)*dt
        half_vel_x = vel[0] + acc_x * half_dt
        half_vel_y = vel[1] + acc_y * half_dt
        half_vel_z = vel[2] + acc_z * half_dt

        positions[i] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
        velocities[i] = type(vel)(half_vel_x, half_vel_y, half_vel_z)


# ==============================================================================
# Pointer-Based (CSR) Non-Mutating Kernels
# ==============================================================================


@wp.kernel
def _velocity_verlet_position_update_ptr_out_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Update positions and half-step velocities using atom_ptr (non-mutating).

    Each thread processes one system's atoms sequentially.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (num_atoms_total,).
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (num_atoms_total,).
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (num_atoms_total,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (num_atoms_total,).
    atom_ptr : wp.array(dtype=wp.int32)
        CSR-style pointers. Shape (num_systems + 1,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep per system. Shape (num_systems,).
    positions_out : wp.array(dtype=wp.vec3f or wp.vec3d)
        Output positions. Shape (num_atoms_total,).
    velocities_out : wp.array(dtype=wp.vec3f or wp.vec3d)
        Output velocities. Shape (num_atoms_total,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]
    dt_val = dt[sys_id]

    half_dt = type(dt_val)(0.5) * dt_val
    half_dt_sq = type(dt_val)(0.5) * dt_val * dt_val

    for i in range(a0, a1):
        pos = positions[i]
        vel = velocities[i]
        force = forces[i]
        mass = masses[i]

        # Guard against division by zero: if mass is zero, set inv_mass to zero
        if mass > type(mass)(0.0):
            inv_mass = type(mass)(1.0) / mass
        else:
            inv_mass = type(mass)(0.0)

        # Compute acceleration
        acc_x = force[0] * inv_mass
        acc_y = force[1] * inv_mass
        acc_z = force[2] * inv_mass

        # Position update
        new_pos_x = pos[0] + vel[0] * dt_val + acc_x * half_dt_sq
        new_pos_y = pos[1] + vel[1] * dt_val + acc_y * half_dt_sq
        new_pos_z = pos[2] + vel[2] * dt_val + acc_z * half_dt_sq

        # Half-step velocity
        half_vel_x = vel[0] + acc_x * half_dt
        half_vel_y = vel[1] + acc_y * half_dt
        half_vel_z = vel[2] + acc_z * half_dt

        positions_out[i] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
        velocities_out[i] = type(vel)(half_vel_x, half_vel_y, half_vel_z)


# ==============================================================================
# Kernel Overloads via KernelFamily
# ==============================================================================

# Position update (inplace) -- keyed by vec_dtype
_position_update_families = build_family_dict(
    _velocity_verlet_position_update_kernel,
    lambda v, t: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=t),
    ],
    _batch_velocity_verlet_position_update_kernel,
    lambda v, t: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
    ],
    _velocity_verlet_position_update_ptr_kernel,
    lambda v, t: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
    ],
)

# Position update (out) -- keyed by vec_dtype
_position_update_out_families = build_family_dict(
    _velocity_verlet_position_update_out_kernel,
    lambda v, t: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=v),
    ],
    _batch_velocity_verlet_position_update_out_kernel,
    lambda v, t: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=v),
    ],
    _velocity_verlet_position_update_ptr_out_kernel,
    lambda v, t: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=v),
    ],
)


# ==============================================================================
# Functional Interface - Mutating
# ==============================================================================


def velocity_verlet_position_update(
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    dt: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> None:
    """
    Perform velocity Verlet position update step (in-place).

    Updates positions to r(t+dt) and velocities to half-step v(t+dt/2).
    After calling this function, recalculate forces at the new positions,
    then call velocity_verlet_velocity_finalize().

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,). MODIFIED in-place.
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single system, (B,) for batched.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from positions.

    See Also
    --------
    velocity_verlet_velocity_finalize : Complete the velocity update
    """
    dispatch_family(
        _position_update_families,
        positions,
        batch_idx=batch_idx,
        atom_ptr=atom_ptr,
        device=device,
        inputs_single=[positions, velocities, forces, masses, dt],
        inputs_batch=[positions, velocities, forces, masses, batch_idx, dt],
        inputs_ptr=[positions, velocities, forces, masses, atom_ptr, dt],
    )


def velocity_verlet_velocity_finalize(
    velocities: wp.array,
    forces_new: wp.array,
    masses: wp.array,
    dt: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> None:
    """
    Finalize velocity Verlet velocity update (in-place).

    Completes the velocity update using forces evaluated at the new positions:
    v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Half-step velocities. Shape (N,). MODIFIED in-place to full-step.
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces evaluated at new positions r(t+dt). Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from velocities.

    See Also
    --------
    velocity_verlet_position_update : First step of velocity Verlet
    """
    dispatch_family(
        velocity_kick_families,
        velocities,
        batch_idx=batch_idx,
        atom_ptr=atom_ptr,
        device=device,
        inputs_single=[velocities, forces_new, masses, dt, velocities],
        inputs_batch=[velocities, forces_new, masses, batch_idx, dt, velocities],
        inputs_ptr=[velocities, forces_new, masses, atom_ptr, dt, velocities],
    )


# ==============================================================================
# Functional Interface - Non-Mutating
# ==============================================================================


def velocity_verlet_position_update_out(
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    dt: wp.array,
    positions_out: wp.array,
    velocities_out: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> tuple[wp.array, wp.array]:
    """
    Perform velocity Verlet position update step (non-mutating).

    Writes new positions r(t+dt) and half-step velocities v(t+dt/2) to
    output arrays. Input arrays are NOT modified.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions at time t. Shape (N,).
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities at time t. Shape (N,).
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms at time t. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.
    positions_out : wp.array
        Pre-allocated output array for new positions.  Must match
        ``positions`` in shape, dtype, and device.
    velocities_out : wp.array
        Pre-allocated output array for half-step velocities.  Must match
        ``velocities`` in shape, dtype, and device.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from positions.

    Returns
    -------
    tuple[wp.array, wp.array]
        (positions_out, velocities_out) - New positions and half-step velocities.
    """
    validate_out_array(positions_out, positions, "positions_out")
    validate_out_array(velocities_out, velocities, "velocities_out")

    dispatch_family(
        _position_update_out_families,
        positions,
        batch_idx=batch_idx,
        atom_ptr=atom_ptr,
        device=device,
        inputs_single=[
            positions,
            velocities,
            forces,
            masses,
            dt,
            positions_out,
            velocities_out,
        ],
        inputs_batch=[
            positions,
            velocities,
            forces,
            masses,
            batch_idx,
            dt,
            positions_out,
            velocities_out,
        ],
        inputs_ptr=[
            positions,
            velocities,
            forces,
            masses,
            atom_ptr,
            dt,
            positions_out,
            velocities_out,
        ],
    )

    return positions_out, velocities_out


def velocity_verlet_velocity_finalize_out(
    velocities: wp.array,
    forces_new: wp.array,
    masses: wp.array,
    dt: wp.array,
    velocities_out: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Finalize velocity Verlet velocity update (non-mutating).

    Writes full-step velocities v(t+dt) to output array.
    Input arrays are NOT modified.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Half-step velocities v(t+dt/2). Shape (N,).
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at new positions r(t+dt). Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.
    velocities_out : wp.array
        Pre-allocated output array for full-step velocities.  Must match
        ``velocities`` in shape, dtype, and device.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from velocities.

    Returns
    -------
    wp.array
        Full-step velocities v(t+dt).
    """
    validate_out_array(velocities_out, velocities, "velocities_out")

    dispatch_family(
        velocity_kick_families,
        velocities,
        batch_idx=batch_idx,
        atom_ptr=atom_ptr,
        device=device,
        inputs_single=[velocities, forces_new, masses, dt, velocities_out],
        inputs_batch=[velocities, forces_new, masses, batch_idx, dt, velocities_out],
        inputs_ptr=[velocities, forces_new, masses, atom_ptr, dt, velocities_out],
    )

    return velocities_out
