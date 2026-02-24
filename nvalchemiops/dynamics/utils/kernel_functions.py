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

"""Shared Warp Functions for Dynamics Kernels.

This module provides reusable @wp.func functions for common computational
patterns in molecular dynamics integrators. These functions can be called
from any @wp.kernel to reduce code duplication and improve maintainability.

Categories
----------
- **Physics computations**: Acceleration, velocity updates, position updates
- **Utility functions**: Vector scaling, parameter lookup, atom range queries

All functions use dtype polymorphism via the `Any` type and `type()` casting
to work with both float32/vec3f and float64/vec3d precision.

Examples
--------
Basic usage in a kernel::

    from nvalchemiops.dynamics.utils.kernel_functions import (
        compute_acceleration_from_force,
        velocity_half_step_from_acceleration,
    )

    @wp.kernel
    def my_integrator_kernel(positions, velocities, forces, masses, dt):
        atom_idx = wp.tid()
        force = forces[atom_idx]
        mass = masses[atom_idx]
        vel = velocities[atom_idx]
        dt_val = dt[0]

        # Use shared functions
        acc = compute_acceleration_from_force(force, mass)
        vel_half = velocity_half_step_from_acceleration(vel, acc, dt_val)

        velocities[atom_idx] = vel_half
"""

from typing import Any

import warp as wp

__all__ = [
    # Physics functions
    "compute_acceleration_from_force",
    "velocity_half_step_from_acceleration",
    "position_update_from_velocity",
    "velocity_verlet_position_step",
    # Utility functions
    "scale_vector_by_scalar",
    # Algorithm-specific functions
    # FIRE optimizer
    "compute_vf_vv_ff",
    "fire_velocity_mixing",
    "clamp_displacement",
    "is_first_atom_of_system",
]


# ==============================================================================
# Physics Functions
# ==============================================================================


@wp.func
def compute_acceleration_from_force(
    force: Any,  # wp.vec3f or wp.vec3d
    mass: Any,  # wp.float32 or wp.float64
) -> Any:
    """Compute acceleration vector from force and mass.

    Computes Newton's second law: a = F/m

    Parameters
    ----------
    force : wp.vec3f or wp.vec3d
        Force vector acting on particle.
    mass : wp.float32 or wp.float64
        Particle mass.

    Returns
    -------
    acceleration : same type as force
        Acceleration vector (F/m).

    Examples
    --------
    In a kernel::

        force = forces[atom_idx]
        mass = masses[atom_idx]
        acc = compute_acceleration_from_force(force, mass)
    """
    inv_mass = type(force[0])(1.0) / type(force[0])(mass)
    return type(force)(
        force[0] * inv_mass,
        force[1] * inv_mass,
        force[2] * inv_mass,
    )


@wp.func
def velocity_half_step_from_acceleration(
    vel: Any,  # wp.vec3f or wp.vec3d
    acc: Any,  # wp.vec3f or wp.vec3d
    dt: Any,  # wp.float32 or wp.float64
) -> Any:
    """Compute half-step velocity update from acceleration.

    Computes v_half = v + 0.5 * a * dt

    This is the standard half-step velocity update used in velocity Verlet
    and related integrators.

    Parameters
    ----------
    vel : wp.vec3f or wp.vec3d
        Current velocity.
    acc : wp.vec3f or wp.vec3d
        Acceleration vector.
    dt : wp.float32 or wp.float64
        Timestep.

    Returns
    -------
    velocity_half : same type as vel
        Half-step velocity.

    Examples
    --------
    In a kernel::

        vel = velocities[atom_idx]
        acc = compute_acceleration_from_force(forces[atom_idx], masses[atom_idx])
        dt_val = dt[0]
        vel_half = velocity_half_step_from_acceleration(vel, acc, dt_val)
    """
    half_dt = type(dt)(0.5) * dt
    return type(vel)(
        vel[0] + type(vel[0])(acc[0]) * half_dt,
        vel[1] + type(vel[1])(acc[1]) * half_dt,
        vel[2] + type(vel[2])(acc[2]) * half_dt,
    )


@wp.func
def position_update_from_velocity(
    pos: Any,  # wp.vec3f or wp.vec3d
    vel: Any,  # wp.vec3f or wp.vec3d
    dt: Any,  # wp.float32 or wp.float64
) -> Any:
    """Compute position update from velocity.

    Computes r_new = r + v * dt

    This is the standard position update for Euler-like integration schemes.

    Parameters
    ----------
    pos : wp.vec3f or wp.vec3d
        Current position.
    vel : wp.vec3f or wp.vec3d
        Velocity vector.
    dt : wp.float32 or wp.float64
        Timestep.

    Returns
    -------
    position_new : same type as pos
        Updated position.

    Examples
    --------
    In a kernel::

        pos = positions[atom_idx]
        vel = velocities[atom_idx]
        dt_val = dt[0]
        pos_new = position_update_from_velocity(pos, vel, dt_val)
        positions[atom_idx] = pos_new
    """
    return type(pos)(
        pos[0] + type(pos[0])(vel[0]) * dt,
        pos[1] + type(pos[1])(vel[1]) * dt,
        pos[2] + type(pos[2])(vel[2]) * dt,
    )


@wp.func
def velocity_verlet_position_step(
    pos: Any,  # wp.vec3f or wp.vec3d
    vel: Any,  # wp.vec3f or wp.vec3d
    acc: Any,  # wp.vec3f or wp.vec3d
    dt: Any,  # wp.float32 or wp.float64
) -> Any:
    """Compute velocity Verlet position update.

    Computes r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2

    This is the standard position update for the velocity Verlet integrator,
    which includes both the velocity term and the acceleration term for
    improved accuracy.

    Parameters
    ----------
    pos : wp.vec3f or wp.vec3d
        Current position.
    vel : wp.vec3f or wp.vec3d
        Current velocity.
    acc : wp.vec3f or wp.vec3d
        Current acceleration.
    dt : wp.float32 or wp.float64
        Timestep.

    Returns
    -------
    position_new : same type as pos
        Updated position.

    Examples
    --------
    In a kernel::

        pos = positions[atom_idx]
        vel = velocities[atom_idx]
        acc = compute_acceleration_from_force(forces[atom_idx], masses[atom_idx])
        dt_val = dt[0]
        pos_new = velocity_verlet_position_step(pos, vel, acc, dt_val)
        positions[atom_idx] = pos_new
    """
    half_dt_sq = type(dt)(0.5) * dt * dt
    return type(pos)(
        pos[0] + type(pos[0])(vel[0]) * dt + type(pos[0])(acc[0]) * half_dt_sq,
        pos[1] + type(pos[1])(vel[1]) * dt + type(pos[1])(acc[1]) * half_dt_sq,
        pos[2] + type(pos[2])(vel[2]) * dt + type(pos[2])(acc[2]) * half_dt_sq,
    )


# ==============================================================================
# Utility Functions
# ==============================================================================


@wp.func
def scale_vector_by_scalar(
    vec: Any,  # wp.vec3f or wp.vec3d
    scale: Any,  # wp.float32 or wp.float64
) -> Any:
    """Scale a 3D vector by a scalar value.

    Computes v_scaled = v * s (component-wise multiplication)

    This is commonly used for velocity rescaling in thermostats and
    other operations requiring uniform scaling.

    Parameters
    ----------
    vec : wp.vec3f or wp.vec3d
        Input vector.
    scale : wp.float32 or wp.float64
        Scalar multiplier.

    Returns
    -------
    vector_scaled : same type as vec
        Scaled vector.

    Examples
    --------
    In a kernel (velocity rescaling)::

        vel = velocities[atom_idx]
        scale_factor = scale_factors[system_id]
        vel_scaled = scale_vector_by_scalar(vel, scale_factor)
        velocities[atom_idx] = vel_scaled
    """
    return type(vec)(
        vec[0] * type(vec[0])(scale),
        vec[1] * type(vec[1])(scale),
        vec[2] * type(vec[2])(scale),
    )


# ==============================================================================
# Algorithm-Specific Functions
# ==============================================================================


# ------------------------------------------------------------------------------
# FIRE Optimizer Functions
# ------------------------------------------------------------------------------


@wp.func
def compute_vf_vv_ff(
    velocity: Any,  # wp.vec3f or wp.vec3d
    force: Any,  # wp.vec3f or wp.vec3d
):
    """Compute triple dot product for FIRE optimizer diagnostics.

    Computes all three diagnostic quantities needed by FIRE:
    - v·f: power (force-velocity alignment), indicates uphill/downhill
    - v·v: velocity magnitude squared (kinetic energy proxy)
    - f·f: force magnitude squared

    These are used to compute the mixing ratio √(v·v / f·f) and detect
    uphill steps (v·f < 0).

    Parameters
    ----------
    velocity : wp.vec3f or wp.vec3d
        Velocity vector.
    force : wp.vec3f or wp.vec3d
        Force vector.

    Returns
    -------
    vf : float32 or float64
        Dot product v·f (power).
    vv : float32 or float64
        Dot product v·v (velocity magnitude squared).
    ff : float32 or float64
        Dot product f·f (force magnitude squared).

    Examples
    --------
    In a FIRE kernel::

        vel = velocities[atom_idx]
        force = forces[atom_idx]
        vf, vv, ff = compute_vf_vv_ff(vel, force)

        # Use for FIRE logic
        if vf > 0:
            # Downhill step - increase dt, mix velocities
            pass

    Notes
    -----
    This function consolidates three separate `wp.dot()` calls that appear
    together in 10+ FIRE kernels. It's particularly useful for accumulating
    per-system diagnostics in batched or RLE reduction kernels.

    See Also
    --------
    fire_velocity_mixing : Uses the vv/ff ratio for velocity updates
    """
    vf = wp.dot(velocity, force)
    vv = wp.dot(velocity, velocity)
    ff = wp.dot(force, force)
    return vf, vv, ff


@wp.func
def fire_velocity_mixing(
    velocity: Any,  # wp.vec3f or wp.vec3d
    force: Any,  # wp.vec3f or wp.vec3d
    alpha: Any,  # wp.float32 or wp.float64
    vv: Any,  # wp.float32 or wp.float64
    ff: Any,  # wp.float32 or wp.float64
) -> Any:
    """Perform FIRE velocity mixing with zero-safety.

    Computes the core FIRE velocity update:
        v_new = (1 - α)v + α·F·√(v·v / f·f)

    This mixes the current velocity with a force-based velocity scaled to
    match the current speed. The mixing parameter α controls the strength
    of the damping.

    The function includes safety for zero-force case (√(v·v / f·f) → 0).

    Parameters
    ----------
    velocity : wp.vec3f or wp.vec3d
        Current velocity vector.
    force : wp.vec3f or wp.vec3d
        Current force vector.
    alpha : wp.float32 or wp.float64
        FIRE mixing parameter (0 < α ≤ 1). Typically starts at 0.1-0.25.
    vv : wp.float32 or wp.float64
        Pre-computed v·v (velocity magnitude squared).
    ff : wp.float32 or wp.float64
        Pre-computed f·f (force magnitude squared).

    Returns
    -------
    velocity_new : same type as velocity
        Mixed velocity vector.

    Examples
    --------
    In a FIRE kernel::

        vel = velocities[atom_idx]
        force = forces[atom_idx]
        vf, vv, ff = compute_vf_vv_ff(vel, force)

        # Only mix if downhill
        if vf > 0:
            alpha_val = alpha[system_id]
            vel_mixed = fire_velocity_mixing(vel, force, alpha_val, vv, ff)
            velocities[atom_idx] = vel_mixed

    Notes
    -----
    - The ratio √(v·v / f·f) normalizes forces to have the same magnitude as velocities
    - If ff = 0 (no forces), returns original velocity (safe guard)
    - Used in both standard FIRE and FIRE2 variants
    - The mixing makes forces act like friction/damping rather than acceleration

    See Also
    --------
    compute_vf_vv_ff : Computes the required diagnostic quantities
    """
    zero = type(alpha)(0.0)
    one = type(alpha)(1.0)

    # Compute mixing ratio with zero-safety
    if ff > zero:
        ratio = wp.sqrt(vv / ff)
    else:
        ratio = zero

    # Mix: v_new = (1-α)v + α·F·ratio
    return (one - alpha) * velocity + (alpha * force * ratio)


@wp.func
def clamp_displacement(
    displacement: Any,  # wp.vec3f or wp.vec3d
    maxstep: Any,  # wp.float32 or wp.float64
) -> Any:
    """Clamp vector displacement to maximum magnitude.

    Limits the displacement vector to a maximum length without changing
    its direction. This prevents excessively large steps in optimization
    or integration.

    If ||dr|| ≤ maxstep: returns dr unchanged
    If ||dr|| > maxstep: returns dr·(maxstep / ||dr||)

    Parameters
    ----------
    displacement : wp.vec3f or wp.vec3d
        Proposed displacement vector (e.g., dt·v).
    maxstep : wp.float32 or wp.float64
        Maximum allowed displacement magnitude.

    Returns
    -------
    displacement_clamped : same type as displacement
        Clamped displacement vector.

    Examples
    --------
    In a FIRE kernel::

        dt_val = dt[system_id]
        maxstep_val = maxstep[system_id]
        vel = velocities[atom_idx]

        # Compute proposed displacement
        dr = dt_val * vel

        # Clamp to maximum step
        dr_clamped = clamp_displacement(dr, maxstep_val)

        # Apply position update
        positions[atom_idx] += dr_clamped

    Notes
    -----
    - Preserves displacement direction (unit vector unchanged)
    - Safe for zero-displacement case (returns zero vector)
    - Common in geometry optimization to prevent "overshooting"
    - FIRE uses this to limit atom movements per step

    Performance Note:
        Uses wp.length() which computes sqrt(x²+y²+z²). For very small
        displacements, the sqrt cost is minimal compared to safety benefit.

    See Also
    --------
    position_update_from_velocity : Basic position update without clamping
    """
    zero = type(maxstep)(0.0)
    one = type(maxstep)(1.0)

    dr_len = wp.length(displacement)

    # Avoid division by zero for null displacement
    if dr_len > zero:
        # Scale down if exceeds maximum
        scale = wp.min(one, maxstep / dr_len)
        return scale * displacement
    else:
        # Zero displacement - return unchanged
        return displacement


@wp.func
def is_first_atom_of_system(
    atom_idx: wp.int32,
    batch_idx: wp.array(dtype=wp.int32),
) -> wp.bool:
    """Check if atom is first in its batch_idx segment.

    Detects segment boundaries in a sorted batch_idx array. Used for
    race-condition-free writes to per-system state when each atom in
    a segment redundantly computes the same value.

    **CRITICAL**: Requires batch_idx to be sorted in non-decreasing order.

    Parameters
    ----------
    atom_idx : int32
        Atom index (typically wp.tid()).
    batch_idx : wp.array(dtype=int32)
        **Sorted** segment indices. Shape (N_atoms,).

    Returns
    -------
    is_first : bool
        True if this is the first atom of its system.

    Examples
    --------
    Race-free per-system parameter update::

        atom_idx = wp.tid()
        system_id = batch_idx[atom_idx]

        # All atoms in system redundantly compute new_dt
        new_dt = compute_new_dt(...)

        # Only first atom writes (no race condition)
        if is_first_atom_of_system(atom_idx, batch_idx):
            dt[system_id] = new_dt

    Notes
    -----
    - **Performance**: This pattern avoids expensive atomics or sync barriers
    - **Pattern**: Common in FIRE optimizer for dt/alpha/counter updates
    - **Requirement**: batch_idx MUST be sorted (e.g., from create_batch_idx())
    - **Alternative**: Use atom_ptr kernels with sequential loops

    Why This Works:
        When batch_idx is sorted, the first atom where batch_idx[i] != batch_idx[i-1]
        is guaranteed to be unique for that system. This atom can safely write
        per-system state without racing with other atoms.

    Common Usage Pattern:
        1. All atoms redundantly compute per-system value
        2. First atom per system writes result
        3. No atomic operations needed
        4. No race conditions (sorted batch_idx ensures uniqueness)

    """
    return atom_idx == 0 or batch_idx[atom_idx - 1] != batch_idx[atom_idx]
