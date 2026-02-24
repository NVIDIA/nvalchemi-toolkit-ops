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
Shared Warp Kernels for Dynamics
=================================

Common ``@wp.func`` device functions and thin ``@wp.kernel`` shells
reused across multiple integrators (velocity Verlet, Langevin BAOAB,
Nosé-Hoover chains, velocity rescaling).

Each operation is defined once as a generic ``@wp.func`` (not strictly
typed -- Warp infers concrete types at compile time) and wrapped by
three out-only kernel shells: ``single``, ``batch_idx``, ``atom_ptr``.

In-place dispatch is achieved by the host passing the same array as
both input and output.  This is safe because every kernel writes only
to the index that was read within the same thread (no cross-index
aliasing).

Autodiff is **not** required for these kernels.
"""

from typing import Any

import warp as wp

from nvalchemiops.dynamics.utils.launch_helpers import build_family_dict

# =============================================================================
# @wp.func -- shared physics computations
# =============================================================================


@wp.func
def velocity_half_kick(vel: Any, force: Any, mass: Any, dt_val: Any) -> Any:
    """Half-step velocity kick: ``v_new = v + 0.5 * (F / m) * dt``."""
    inv_mass = wp.where(mass > type(mass)(0.0), type(mass)(1.0) / mass, type(mass)(0.0))
    half_dt = type(dt_val)(0.5) * dt_val
    return vel + half_dt * force * inv_mass


@wp.func
def scale_velocity(vel: Any, scale: Any) -> Any:
    """Rescale velocity: ``v_new = v * scale``."""
    s = type(vel[0])(scale)
    return type(vel)(vel[0] * s, vel[1] * s, vel[2] * s)


@wp.func
def simple_position_update(pos: Any, vel: Any, dt_val: Any) -> Any:
    """Full-step position update: ``r_new = r + v * dt``."""
    return pos + vel * dt_val


# =============================================================================
# Velocity half-kick kernels (out-only)
# =============================================================================


@wp.kernel
def _velocity_kick_single_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Velocity half-kick, single-system mode.

    Launch Grid: dim = [num_atoms]
    """
    i = wp.tid()
    velocities_out[i] = velocity_half_kick(velocities[i], forces[i], masses[i], dt[0])


@wp.kernel
def _velocity_kick_batch_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Velocity half-kick, batch_idx mode.

    Launch Grid: dim = [num_atoms]
    """
    i = wp.tid()
    sys_id = batch_idx[i]
    velocities_out[i] = velocity_half_kick(
        velocities[i], forces[i], masses[i], dt[sys_id]
    )


@wp.kernel
def _velocity_kick_ptr_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Velocity half-kick, atom_ptr (CSR) mode.

    Launch Grid: dim = [num_systems]
    """
    sys_id = wp.tid()
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]
    dt_val = dt[sys_id]
    for i in range(a0, a1):
        velocities_out[i] = velocity_half_kick(
            velocities[i], forces[i], masses[i], dt_val
        )


# =============================================================================
# Velocity rescale kernels (out-only)
# =============================================================================


@wp.kernel
def _velocity_rescale_single_kernel(
    velocities: wp.array(dtype=Any),
    scale_factor: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Velocity rescale, single-system mode.

    Launch Grid: dim = [num_atoms]
    """
    i = wp.tid()
    velocities_out[i] = scale_velocity(velocities[i], scale_factor[0])


@wp.kernel
def _velocity_rescale_batch_kernel(
    velocities: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    scale_factor: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Velocity rescale, batch_idx mode.

    Launch Grid: dim = [num_atoms]
    """
    i = wp.tid()
    sys_id = batch_idx[i]
    velocities_out[i] = scale_velocity(velocities[i], scale_factor[sys_id])


@wp.kernel
def _velocity_rescale_ptr_kernel(
    velocities: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    scale_factor: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Velocity rescale, atom_ptr (CSR) mode.

    Launch Grid: dim = [num_systems]
    """
    sys_id = wp.tid()
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]
    s = scale_factor[sys_id]
    for i in range(a0, a1):
        velocities_out[i] = scale_velocity(velocities[i], s)


# =============================================================================
# Simple position update kernels (out-only)
# =============================================================================


@wp.kernel
def _position_update_single_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
):
    """Simple position update, single-system mode.

    Launch Grid: dim = [num_atoms]
    """
    i = wp.tid()
    positions_out[i] = simple_position_update(positions[i], velocities[i], dt[0])


@wp.kernel
def _position_update_batch_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
):
    """Simple position update, batch_idx mode.

    Launch Grid: dim = [num_atoms]
    """
    i = wp.tid()
    sys_id = batch_idx[i]
    positions_out[i] = simple_position_update(positions[i], velocities[i], dt[sys_id])


@wp.kernel
def _position_update_ptr_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
):
    """Simple position update, atom_ptr (CSR) mode.

    Launch Grid: dim = [num_systems]
    """
    sys_id = wp.tid()
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]
    dt_val = dt[sys_id]
    for i in range(a0, a1):
        positions_out[i] = simple_position_update(positions[i], velocities[i], dt_val)


# =============================================================================
# Pre-built KernelFamily dicts (with wp.overload registration)
# =============================================================================


velocity_kick_families = build_family_dict(
    _velocity_kick_single_kernel,
    lambda v, t: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.array(dtype=v),
    ],
    _velocity_kick_batch_kernel,
    lambda v, t: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
        wp.array(dtype=v),
    ],
    _velocity_kick_ptr_kernel,
    lambda v, t: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
        wp.array(dtype=v),
    ],
)

velocity_rescale_families = build_family_dict(
    _velocity_rescale_single_kernel,
    lambda v, t: [wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=v)],
    _velocity_rescale_batch_kernel,
    lambda v, t: [
        wp.array(dtype=v),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
        wp.array(dtype=v),
    ],
    _velocity_rescale_ptr_kernel,
    lambda v, t: [
        wp.array(dtype=v),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
        wp.array(dtype=v),
    ],
)

position_update_families = build_family_dict(
    _position_update_single_kernel,
    lambda v, t: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=v),
    ],
    _position_update_batch_kernel,
    lambda v, t: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
        wp.array(dtype=v),
    ],
    _position_update_ptr_kernel,
    lambda v, t: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
        wp.array(dtype=v),
    ],
)
