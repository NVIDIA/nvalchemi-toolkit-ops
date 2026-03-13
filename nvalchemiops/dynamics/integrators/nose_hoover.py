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
Nosé-Hoover Chain (NHC) Thermostat for NVT Ensemble.

This module implements the Nosé-Hoover chain thermostat following the
Martyna-Tobias-Klein (MTK) equations of motion with time-reversible
integration based on Tuckerman et al.

References
----------
- Martyna, Tobias, Klein. J Chem Phys, 101, 4177 (1994)
- Tuckerman et al. J Phys A: Math Gen, 39, 5629 (2006)

The Nosé-Hoover chain equations of motion:
    ṙᵢ = vᵢ
    v̇ᵢ = Fᵢ/mᵢ - η̇₁·vᵢ
    η̇₁ = (2·KE - Ndof·kT) / Q₁
    η̇ₖ = (Qₖ₋₁·η̇²ₖ₋₁ - kT) / Qₖ   for k > 1

Where:
    η   : thermostat chain positions (unitless)
    η̇   : thermostat chain velocities (1/time)
    Q   : thermostat chain masses (energy·time²)
    Ndof: degrees of freedom (typically 3N - 3)
    kT  : target temperature in energy units (k_B = 1)

BATCH MODE
==========

All functions in this module support three execution modes:

**Single System Mode**::

    # Simple position and velocity updates
    nhc_velocity_half_step(velocities, forces, masses, dt)
    nhc_position_update(positions, velocities, dt)

**Batch Mode with batch_idx** (atomic operations)::

    batch_idx = wp.array([0]*N0 + [1]*N1 + [2]*N2, dtype=wp.int32, device="cuda:0")
    dt = wp.array([dt0, dt1, dt2], dtype=wp.float64, device="cuda:0")

    nhc_velocity_half_step(velocities, forces, masses, dt, batch_idx=batch_idx)
    nhc_position_update(positions, velocities, dt, batch_idx=batch_idx)

**Batch Mode with atom_ptr** (sequential per-system)::

    atom_ptr = wp.array([0, N0, N0+N1, N0+N1+N2], dtype=wp.int32, device="cuda:0")

    nhc_velocity_half_step(velocities, forces, masses, dt, atom_ptr=atom_ptr)
    nhc_position_update(positions, velocities, dt, atom_ptr=atom_ptr)
"""

from __future__ import annotations

import os
from typing import Any

import warp as wp

from nvalchemiops.dynamics.utils.launch_helpers import dispatch_family
from nvalchemiops.dynamics.utils.shared_kernels import (
    position_update_families,
    velocity_kick_families,
)
from nvalchemiops.warp_dispatch import validate_out_array

__all__ = [
    # Mutating (in-place) APIs
    "nhc_thermostat_chain_update",
    "nhc_velocity_half_step",
    "nhc_position_update",
    "nhc_compute_chain_energy",
    # Non-mutating (output) APIs
    "nhc_thermostat_chain_update_out",
    "nhc_velocity_half_step_out",
    "nhc_position_update_out",
    # Utility functions
    "nhc_compute_masses",
]


# ==============================================================================
# Constants
# ==============================================================================

# Maximum supported chain length (typically 3-5 in practice)
MAX_CHAIN_LENGTH = 8

# Yoshida-Suzuki Integration Weights
# These weights provide 4th-order accurate, time-reversible integration
# for the thermostat chain propagation.

# 3-step Yoshida-Suzuki weights
_YS3_W0 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YS3_W1 = 1.0 - 2.0 * _YS3_W0
YOSHIDA_SUZUKI_3 = [_YS3_W0, _YS3_W1, _YS3_W0]

# 5-step Yoshida-Suzuki weights (higher accuracy)
_YS5_W0 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
_YS5_W1 = _YS5_W0
_YS5_W2 = 1.0 - 4.0 * _YS5_W0
YOSHIDA_SUZUKI_5 = [_YS5_W0, _YS5_W1, _YS5_W2, _YS5_W1, _YS5_W0]


# ==============================================================================
# Diagnostic Kernels
# ==============================================================================

# Tile block size for cooperative reductions
TILE_DIM = int(os.getenv("NVALCHEMIOPS_DYNAMICS_TILE_DIM", 256))


@wp.kernel
def _compute_2ke_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    ke2: wp.array(dtype=Any),
):
    """Compute 2*KE = sum(m * v^2) for thermostat forcing.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    v = velocities[atom_idx]
    m = masses[atom_idx]

    v_sq = wp.dot(v, v)

    wp.atomic_add(ke2, 0, type(ke2[0])(m * v_sq))


@wp.kernel
def _compute_2ke_tiled_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    ke2: wp.array(dtype=Any),
):
    """Compute 2*KE with tile reductions (single system).

    Launch Grid: dim = [num_atoms], block_dim = TILE_DIM
    """
    atom_idx = wp.tid()

    v = velocities[atom_idx]
    m = masses[atom_idx]

    v_sq = wp.dot(v, v)
    local_2ke = type(ke2[0])(m * v_sq)

    # Convert to tile for block-level reduction
    t = wp.tile(local_2ke)

    # Cooperative sum within block
    s = wp.tile_sum(t)

    # Extract scalar from tile sum
    sum_2ke = s[0]

    # Only first thread in block writes
    if atom_idx % TILE_DIM == 0:
        wp.atomic_add(ke2, 0, sum_2ke)


@wp.kernel(enable_backward=False)
def _batch_compute_2ke_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    ke2: wp.array(dtype=Any),
):
    """Compute 2*KE per system for batched simulations.

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    v = velocities[atom_idx]
    m = masses[atom_idx]

    v_sq = wp.dot(v, v)

    wp.atomic_add(ke2, system_id, type(ke2[system_id])(m * v_sq))


@wp.kernel
def _nhc_compute_masses_kernel(
    ndof: wp.array(dtype=wp.int32),
    target_temp: wp.array(dtype=Any),
    tau: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
):
    """Compute Nosé-Hoover chain masses (single system).

    Computes Q_k values for Nosé-Hoover chain:
        Q_0 = ndof * kT * tau^2
        Q_k = kT * tau^2 for k > 0

    Parameters
    ----------
    ndof : wp.array(dtype=wp.int32)
        Number of degrees of freedom. Shape (1,).
    target_temp : wp.array(dtype=Any)
        Target temperature (kT). Shape (1,).
    tau : wp.array(dtype=Any)
        Time constant. Shape (1,).
    masses : wp.array(dtype=Any)
        Chain masses output. Shape (chain_length,).

    Launch Grid
    -----------
    dim = [chain_length]
    """
    k = wp.tid()
    tau_val = tau[0]
    tau_sq = tau_val * tau_val
    if k == 0:
        masses[k] = type(tau_val)(ndof[0]) * target_temp[0] * tau_sq
    else:
        masses[k] = target_temp[0] * tau_sq


@wp.kernel
def _batch_nhc_compute_masses_kernel(
    ndof: wp.array(dtype=wp.int32),
    target_temp: wp.array(dtype=Any),
    tau: wp.array(dtype=Any),
    masses: wp.array2d(dtype=Any),
):
    """Compute Nosé-Hoover chain masses for batched simulations.

    Parameters
    ----------
    ndof : wp.array(dtype=wp.int32)
        Number of degrees of freedom per system. Shape (num_systems,).
    target_temp : wp.array(dtype=Any)
        Target temperature (kT) per system. Shape (num_systems,).
    tau : wp.array(dtype=Any)
        Time constant per system. Shape (num_systems,).
    masses : wp.array2d(dtype=Any)
        Chain masses output. Shape (num_systems, chain_length).

    Launch Grid
    -----------
    dim = [num_systems, chain_length]
    """
    sys_id, k = wp.tid()
    tau_val = tau[sys_id]
    tau_sq = tau_val * tau_val
    if k == 0:
        masses[sys_id, k] = type(tau_val)(ndof[sys_id]) * target_temp[sys_id] * tau_sq
    else:
        masses[sys_id, k] = target_temp[sys_id] * tau_sq


# ==============================================================================
# Chain Propagation Kernels (Pure Warp Implementation)
# ==============================================================================


@wp.kernel
def _nhc_chain_propagate_kernel(
    eta: wp.array2d(dtype=Any),
    eta_dot: wp.array2d(dtype=Any),
    eta_mass: wp.array2d(dtype=Any),
    ke2: wp.array(dtype=Any),
    target_temp: wp.array(dtype=Any),
    ndof: wp.array(dtype=Any),
    dt_chain: wp.array(dtype=Any),
    chain_length: int,
    vel_scale: wp.array(dtype=Any),
):
    """Propagate Nosé-Hoover chain for one Yoshida-Suzuki sub-step.

    This kernel implements the time-reversible Martyna-Tobias-Klein (MTK)
    integration scheme for Nosé-Hoover chains.

    Algorithm (for each system):
    1. Half-step position update: η_k += 0.5 * dt * η̇_k
    2. Backward sweep: Update η̇ from chain end to start with friction
    3. Compute velocity scale factor: exp(-0.5 * dt * η̇_0)
    4. Forward sweep: Update η̇ from start to chain end with new forces
    5. Half-step position update: η_k += 0.5 * dt * η̇_k

    Launch Grid
    -----------
    dim = [num_systems]

    Parameters
    ----------
    eta : wp.array2d(dtype=Any)
        Chain positions, shape (num_systems, chain_length). MODIFIED in-place.
    eta_dot : wp.array2d(dtype=Any)
        Chain velocities, shape (num_systems, chain_length). MODIFIED in-place.
    eta_mass : wp.array2d(dtype=Any)
        Chain masses, shape (num_systems, chain_length).
    ke2 : wp.array(dtype=Any)
        2*KE for each system, shape (num_systems,). MODIFIED to reflect scaled KE.
    target_temp : wp.array(dtype=wp.float64)
        Target temperature (kT), shape (num_systems,).
    ndof : wp.array(dtype=wp.float64)
        Degrees of freedom, shape (num_systems,).
    dt_chain : wp.array(dtype=wp.float64)
        Time step for this sub-step (weight * dt), shape (num_systems,).
    chain_length : int
        Number of thermostats in the chain.
    vel_scale : wp.array(dtype=wp.float64)
        Output velocity scale factors, shape (num_systems,). MODIFIED.
    """
    sys_id = wp.tid()

    kT = target_temp[sys_id]
    ndof_sys = ndof[sys_id]
    dt = dt_chain[sys_id]
    half_dt = type(dt)(0.5) * dt
    quarter_dt = type(dt)(0.25) * dt
    eighth_dt = type(dt)(0.125) * dt
    ke2_sys = ke2[sys_id]

    # Local copies for chain state (we'll write back at the end)
    # Using fixed-size local arrays for the chain
    eta_local = wp.vector(dtype=eta.dtype, length=MAX_CHAIN_LENGTH)
    eta_dot_local = wp.vector(dtype=eta_dot.dtype, length=MAX_CHAIN_LENGTH)
    eta_mass_local = wp.vector(dtype=eta_mass.dtype, length=MAX_CHAIN_LENGTH)

    # Load chain state
    for k in range(chain_length):
        eta_local[k] = eta[sys_id, k]
        eta_dot_local[k] = eta_dot[sys_id, k]
        eta_mass_local[k] = eta_mass[sys_id, k]

    # ========== Step 1: Half-step position update ==========
    for k in range(chain_length):
        eta_local[k] = eta_local[k] + half_dt * eta_dot_local[k]

    # ========== Step 2: Backward sweep (chain_length-1 down to 0) ==========

    # Update last thermostat (no friction from above)
    if chain_length > 1:
        G_last = (
            eta_mass_local[chain_length - 2]
            * eta_dot_local[chain_length - 2]
            * eta_dot_local[chain_length - 2]
            - kT
        ) / eta_mass_local[chain_length - 1]
        eta_dot_local[chain_length - 1] = (
            eta_dot_local[chain_length - 1] + quarter_dt * G_last
        )

    # Update intermediate thermostats (chain_length-2 down to 1)
    for k in range(chain_length - 2, 0, -1):
        G_k = (
            eta_mass_local[k - 1] * eta_dot_local[k - 1] * eta_dot_local[k - 1] - kT
        ) / eta_mass_local[k]
        # Apply friction from k+1
        scale = wp.exp(-eighth_dt * eta_dot_local[k + 1])
        eta_dot_local[k] = eta_dot_local[k] * scale
        eta_dot_local[k] = eta_dot_local[k] + quarter_dt * G_k
        eta_dot_local[k] = eta_dot_local[k] * scale

    # Update first thermostat (couples to particle KE)
    G_0 = (ke2_sys - ndof_sys * kT) / eta_mass_local[0]

    if chain_length > 1:
        scale = wp.exp(-eighth_dt * eta_dot_local[1])
        eta_dot_local[0] = eta_dot_local[0] * scale
        eta_dot_local[0] = eta_dot_local[0] + quarter_dt * G_0
        eta_dot_local[0] = eta_dot_local[0] * scale
    else:
        eta_dot_local[0] = eta_dot_local[0] + quarter_dt * G_0

    # ========== Step 3: Compute velocity scale factor ==========
    vel_scale_factor = wp.exp(-half_dt * eta_dot_local[0])
    vel_scale[sys_id] = vel_scale_factor

    # Update ke2 for the forward sweep
    ke2_sys = ke2_sys * vel_scale_factor * vel_scale_factor
    ke2[sys_id] = ke2_sys

    # ========== Step 4: Forward sweep (0 to chain_length-1) ==========

    # Update first thermostat with new force
    G_0_new = (ke2_sys - ndof_sys * kT) / eta_mass_local[0]

    if chain_length > 1:
        scale = wp.exp(-eighth_dt * eta_dot_local[1])
        eta_dot_local[0] = eta_dot_local[0] * scale
        eta_dot_local[0] = eta_dot_local[0] + quarter_dt * G_0_new
        eta_dot_local[0] = eta_dot_local[0] * scale
    else:
        eta_dot_local[0] = eta_dot_local[0] + quarter_dt * G_0_new

    # Update intermediate thermostats (1 to chain_length-2)
    for k in range(1, chain_length - 1):
        G_k = (
            eta_mass_local[k - 1] * eta_dot_local[k - 1] * eta_dot_local[k - 1] - kT
        ) / eta_mass_local[k]
        scale = wp.exp(-eighth_dt * eta_dot_local[k + 1])
        eta_dot_local[k] = eta_dot_local[k] * scale
        eta_dot_local[k] = eta_dot_local[k] + quarter_dt * G_k
        eta_dot_local[k] = eta_dot_local[k] * scale

    # Update last thermostat
    if chain_length > 1:
        G_last = (
            eta_mass_local[chain_length - 2]
            * eta_dot_local[chain_length - 2]
            * eta_dot_local[chain_length - 2]
            - kT
        ) / eta_mass_local[chain_length - 1]
        eta_dot_local[chain_length - 1] = (
            eta_dot_local[chain_length - 1] + quarter_dt * G_last
        )

    # ========== Step 5: Second half-step position update ==========
    for k in range(chain_length):
        eta_local[k] = eta_local[k] + half_dt * eta_dot_local[k]

    # ========== Write back chain state ==========
    for k in range(chain_length):
        eta[sys_id, k] = eta_local[k]
        eta_dot[sys_id, k] = eta_dot_local[k]


@wp.kernel
def _nhc_chain_propagate_single_kernel(
    eta: wp.array(dtype=Any),
    eta_dot: wp.array(dtype=Any),
    eta_mass: wp.array(dtype=Any),
    ke2: wp.array(dtype=Any),
    target_temp: wp.array(dtype=Any),
    ndof: wp.array(dtype=Any),
    dt_chain: wp.array(dtype=Any),
    chain_length: int,
    vel_scale: wp.array(dtype=Any),
):
    """Propagate Nosé-Hoover chain for single system (non-batched).

    Same algorithm as batched version but for 1D arrays.

    Parameters
    ----------
    eta : wp.array(dtype=Any)
        Chain positions, shape (chain_length,). MODIFIED in-place.
    eta_dot : wp.array(dtype=Any)
        Chain velocities, shape (chain_length,). MODIFIED in-place.
    eta_mass : wp.array(dtype=Any)
        Chain masses, shape (chain_length,).
    ke2 : wp.array(dtype=Any)
        2*KE for the system, shape (1,). MODIFIED to reflect scaled KE.
    target_temp : wp.array(dtype=Any)
        Target temperature (kT), shape (1,).
    ndof : wp.array(dtype=Any)
        Degrees of freedom, shape (1,).
    dt_chain : wp.array(dtype=Any)
        Time step for this sub-step (weight * dt), shape (1,).
    chain_length : int
        Number of thermostats in the chain.
    vel_scale : wp.array(dtype=Any)
        Output velocity scale factors, shape (1,). MODIFIED.

    Launch Grid
    -----------
    dim = [1]
    """
    kT = target_temp[0]
    ndof_sys = ndof[0]
    dt = dt_chain[0]
    half_dt = type(dt)(0.5) * dt
    quarter_dt = type(dt)(0.25) * dt
    eighth_dt = type(dt)(0.125) * dt
    ke2_sys = ke2[0]

    # Local copies for chain state
    eta_local = wp.vector(dtype=eta.dtype, length=MAX_CHAIN_LENGTH)
    eta_dot_local = wp.vector(dtype=eta_dot.dtype, length=MAX_CHAIN_LENGTH)
    eta_mass_local = wp.vector(dtype=eta_mass.dtype, length=MAX_CHAIN_LENGTH)

    # Load chain state
    for k in range(chain_length):
        eta_local[k] = eta[k]
        eta_dot_local[k] = eta_dot[k]
        eta_mass_local[k] = eta_mass[k]

    # ========== Step 1: Half-step position update ==========
    for k in range(chain_length):
        eta_local[k] = eta_local[k] + half_dt * eta_dot_local[k]

    # ========== Step 2: Backward sweep ==========
    if chain_length > 1:
        G_last = (
            eta_mass_local[chain_length - 2]
            * eta_dot_local[chain_length - 2]
            * eta_dot_local[chain_length - 2]
            - kT
        ) / eta_mass_local[chain_length - 1]
        eta_dot_local[chain_length - 1] = (
            eta_dot_local[chain_length - 1] + quarter_dt * G_last
        )

    for k in range(chain_length - 2, 0, -1):
        G_k = (
            eta_mass_local[k - 1] * eta_dot_local[k - 1] * eta_dot_local[k - 1] - kT
        ) / eta_mass_local[k]
        scale = wp.exp(-eighth_dt * eta_dot_local[k + 1])
        eta_dot_local[k] = eta_dot_local[k] * scale
        eta_dot_local[k] = eta_dot_local[k] + quarter_dt * G_k
        eta_dot_local[k] = eta_dot_local[k] * scale

    G_0 = (ke2_sys - ndof_sys * kT) / eta_mass_local[0]

    if chain_length > 1:
        scale = wp.exp(-eighth_dt * eta_dot_local[1])
        eta_dot_local[0] = eta_dot_local[0] * scale
        eta_dot_local[0] = eta_dot_local[0] + quarter_dt * G_0
        eta_dot_local[0] = eta_dot_local[0] * scale
    else:
        eta_dot_local[0] = eta_dot_local[0] + quarter_dt * G_0

    # ========== Step 3: Compute velocity scale factor ==========
    vel_scale_factor = wp.exp(-half_dt * eta_dot_local[0])
    vel_scale[0] = vel_scale_factor
    ke2_sys = ke2_sys * vel_scale_factor * vel_scale_factor
    ke2[0] = ke2_sys
    # ========== Step 4: Forward sweep ==========
    G_0_new = (ke2_sys - ndof_sys * kT) / eta_mass_local[0]
    if chain_length > 1:
        scale = wp.exp(-eighth_dt * eta_dot_local[1])
        eta_dot_local[0] = eta_dot_local[0] * scale
        eta_dot_local[0] = eta_dot_local[0] + quarter_dt * G_0_new
        eta_dot_local[0] = eta_dot_local[0] * scale
    else:
        eta_dot_local[0] = eta_dot_local[0] + quarter_dt * G_0_new
    for k in range(1, chain_length - 1):
        G_k = (
            eta_mass_local[k - 1] * eta_dot_local[k - 1] * eta_dot_local[k - 1] - kT
        ) / eta_mass_local[k]
        scale = wp.exp(-eighth_dt * eta_dot_local[k + 1])
        eta_dot_local[k] = eta_dot_local[k] * scale
        eta_dot_local[k] = eta_dot_local[k] + quarter_dt * G_k
        eta_dot_local[k] = eta_dot_local[k] * scale
    if chain_length > 1:
        G_last = (
            eta_mass_local[chain_length - 2]
            * eta_dot_local[chain_length - 2]
            * eta_dot_local[chain_length - 2]
            - kT
        ) / eta_mass_local[chain_length - 1]
        eta_dot_local[chain_length - 1] = (
            eta_dot_local[chain_length - 1] + quarter_dt * G_last
        )

    # ========== Step 5: Second half-step position update ==========
    for k in range(chain_length):
        eta_local[k] = eta_local[k] + half_dt * eta_dot_local[k]

    for k in range(chain_length):
        eta[k] = eta_local[k]
        eta_dot[k] = eta_dot_local[k]


# ==============================================================================
# Chain Energy Kernels
# ==============================================================================


@wp.kernel
def _nhc_compute_chain_energy_kernel(
    eta: wp.array(dtype=Any),
    eta_dot: wp.array(dtype=Any),
    eta_mass: wp.array(dtype=Any),
    target_temp: wp.array(dtype=Any),
    ndof: wp.array(dtype=Any),
    chain_length: int,
    ke_chain: wp.array(dtype=Any),
    pe_chain: wp.array(dtype=Any),
):
    """Compute NHC kinetic and potential energy for single system.

    KE_chain = sum_k 0.5 * Q_k * η̇_k²
    PE_chain = ndof * kT * η_0 + kT * sum_{k>0} η_k

    Parameters
    ----------
    eta : wp.array(dtype=Any)
        Chain positions, shape (chain_length,).
    eta_dot : wp.array(dtype=Any)
        Chain velocities, shape (chain_length,).
    eta_mass : wp.array(dtype=Any)
        Chain masses, shape (chain_length,).
    target_temp : wp.array(dtype=Any)
        Target temperature (kT), shape (1,).
    ndof : wp.array(dtype=Any)
        Degrees of freedom, shape (1,).
    chain_length : int
        Number of thermostats in the chain.
    ke_chain : wp.array(dtype=Any)
        Kinetic energy of the chain, shape (1,).
    pe_chain : wp.array(dtype=Any)
        Potential energy of the chain, shape (1,).

    Launch Grid
    -----------
    dim = [1]
    """
    kT = target_temp[0]
    ndof_sys = ndof[0]

    ke = type(eta[0])(0.0)
    pe = type(eta[0])(0.0)

    for k in range(chain_length):
        ke = ke + type(eta[0])(0.5) * eta_mass[k] * eta_dot[k] * eta_dot[k]
        if k == 0:
            pe = pe + type(eta[0])(ndof_sys) * type(eta[0])(kT) * eta[k]
        else:
            pe = pe + type(eta[0])(kT) * eta[k]

    ke_chain[0] = type(eta[0])(ke)
    pe_chain[0] = type(eta[0])(pe)


@wp.kernel
def _batch_nhc_compute_chain_energy_kernel(
    eta: wp.array2d(dtype=Any),
    eta_dot: wp.array2d(dtype=Any),
    eta_mass: wp.array2d(dtype=Any),
    target_temp: wp.array(dtype=Any),
    ndof: wp.array(dtype=Any),
    chain_length: int,
    ke_chain: wp.array(dtype=Any),
    pe_chain: wp.array(dtype=Any),
):
    """Compute NHC kinetic and potential energy for batched systems.

    Parameters
    ----------
    eta : wp.array2d(dtype=Any)
        Chain positions, shape (num_systems, chain_length).
    eta_dot : wp.array2d(dtype=Any)
        Chain velocities, shape (num_systems, chain_length).
    eta_mass : wp.array2d(dtype=Any)
        Chain masses, shape (num_systems, chain_length).
    target_temp : wp.array(dtype=Any)
        Target temperature (kT), shape (num_systems,).
    ndof : wp.array(dtype=Any)
        Degrees of freedom, shape (num_systems,).
    chain_length : int
        Number of thermostats in the chain.
    ke_chain : wp.array(dtype=Any)
        Kinetic energy of the chain, shape (num_systems,).
    pe_chain : wp.array(dtype=Any)
        Potential energy of the chain, shape (num_systems,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()

    # Use eta[sys_id, 0] to get a scalar for type inference (not eta[0] which returns a row)
    kT = type(eta[sys_id, 0])(target_temp[sys_id])
    ndof_sys = type(eta[sys_id, 0])(ndof[sys_id])

    ke = type(kT)(0.0)
    pe = type(kT)(0.0)

    for k in range(chain_length):
        ke = (
            ke
            + type(kT)(0.5)
            * eta_mass[sys_id, k]
            * eta_dot[sys_id, k]
            * eta_dot[sys_id, k]
        )
        if k == 0:
            pe = pe + ndof_sys * kT * eta[sys_id, k]
        else:
            pe = pe + kT * eta[sys_id, k]

    ke_chain[sys_id] = ke
    pe_chain[sys_id] = pe


# ==============================================================================
# Velocity Scaling Kernels
# ==============================================================================


@wp.kernel
def _scale_velocities_kernel(
    velocities: wp.array(dtype=Any),
    scale_factor: wp.array(dtype=Any),
):
    """Scale all velocities by a single factor.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    v = velocities[atom_idx]
    s = type(v[0])(scale_factor[0])

    new_vx = type(v[0])(v[0] * s)
    new_vy = type(v[1])(v[1] * s)
    new_vz = type(v[2])(v[2] * s)

    velocities[atom_idx] = type(v)(new_vx, new_vy, new_vz)


@wp.kernel
def _batch_scale_velocities_kernel(
    velocities: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    scale_factors: wp.array(dtype=Any),
):
    """Scale velocities with per-system factors.

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    v = velocities[atom_idx]
    s = type(v[0])(scale_factors[system_id])

    new_vx = type(v[0])(v[0] * s)
    new_vy = type(v[1])(v[1] * s)
    new_vz = type(v[2])(v[2] * s)

    velocities[atom_idx] = type(v)(new_vx, new_vy, new_vz)


@wp.kernel
def _scale_velocities_out_kernel(
    velocities: wp.array(dtype=Any),
    scale_factor: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Scale velocities to output array.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    v = velocities[atom_idx]
    s = type(v[0])(scale_factor[0])

    new_vx = type(v[0])(v[0] * s)
    new_vy = type(v[1])(v[1] * s)
    new_vz = type(v[2])(v[2] * s)

    velocities_out[atom_idx] = type(v)(new_vx, new_vy, new_vz)


@wp.kernel
def _batch_scale_velocities_out_kernel(
    velocities: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    scale_factors: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Scale velocities with per-system factors to output array.

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    v = velocities[atom_idx]
    s = type(v[0])(scale_factors[system_id])

    new_vx = type(v[0])(v[0] * s)
    new_vy = type(v[1])(v[1] * s)
    new_vz = type(v[2])(v[2] * s)

    velocities_out[atom_idx] = type(v)(new_vx, new_vy, new_vz)


# ==============================================================================
# Multiply Scale Factors Kernel
# ==============================================================================


@wp.kernel
def _multiply_scale_factors_kernel(
    total_scale: wp.array(dtype=Any),
    step_scale: wp.array(dtype=Any),
):
    """Multiply total scale factors by step scale factors.

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    total_scale[sys_id] = total_scale[sys_id] * step_scale[sys_id]


# Overloads for _multiply_scale_factors_kernel
_multiply_scale_factors_kernel_overload = {}
for t in [wp.float32, wp.float64]:
    _multiply_scale_factors_kernel_overload[t] = wp.overload(
        _multiply_scale_factors_kernel,
        [wp.array(dtype=t), wp.array(dtype=t)],
    )


# ==============================================================================
# Kernel Overloads for Explicit Typing
# ==============================================================================
# These overloads provide explicit type annotations for each kernel, avoiding
# Warp's type inference issues that can occur with dtype=Any parameters.

_T = [wp.float32, wp.float64]  # Scalar types
_V = [wp.vec3f, wp.vec3d]  # Vector types

# Diagnostic kernel overloads
_compute_2ke_kernel_overload = {}
_compute_2ke_tiled_kernel_overload = {}
_batch_compute_2ke_kernel_overload = {}

# Velocity scaling kernel overloads
_scale_velocities_kernel_overload = {}
_batch_scale_velocities_kernel_overload = {}
_scale_velocities_out_kernel_overload = {}
_batch_scale_velocities_out_kernel_overload = {}

# Compute masses kernel overloads (keyed by scalar type)
_nhc_compute_masses_kernel_overload = {}
_batch_nhc_compute_masses_kernel_overload = {}

for t in _T:
    _nhc_compute_masses_kernel_overload[t] = wp.overload(
        _nhc_compute_masses_kernel,
        [
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array(dtype=t),
        ],
    )
    _batch_nhc_compute_masses_kernel_overload[t] = wp.overload(
        _batch_nhc_compute_masses_kernel,
        [
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array2d(dtype=t),
        ],
    )

# Create overloads for all combinations of velocity type (v) and output type (t_out)
# The ke2 output needs to match the chain state dtype, not the velocity dtype
for v in _V:
    # Determine the scalar type for masses from the vector type
    t_mass = wp.float32 if v == wp.vec3f else wp.float64
    for t_out in _T:
        # Key by (velocity_type, output_type)
        _compute_2ke_kernel_overload[(v, t_out)] = wp.overload(
            _compute_2ke_kernel,
            [wp.array(dtype=v), wp.array(dtype=t_mass), wp.array(dtype=t_out)],
        )
        _compute_2ke_tiled_kernel_overload[(v, t_out)] = wp.overload(
            _compute_2ke_tiled_kernel,
            [wp.array(dtype=v), wp.array(dtype=t_mass), wp.array(dtype=t_out)],
        )
        _batch_compute_2ke_kernel_overload[(v, t_out)] = wp.overload(
            _batch_compute_2ke_kernel,
            [
                wp.array(dtype=v),
                wp.array(dtype=t_mass),
                wp.array(dtype=wp.int32),
                wp.array(dtype=t_out),
            ],
        )

# Create velocity scaling kernel overloads for all combinations of velocity type and scale factor dtype
for v in _V:
    for t_scale in _T:
        # Key by (velocity_type, scale_factor_dtype) to support mixed dtypes
        _scale_velocities_kernel_overload[(v, t_scale)] = wp.overload(
            _scale_velocities_kernel,
            [wp.array(dtype=v), wp.array(dtype=t_scale)],
        )
        _batch_scale_velocities_kernel_overload[(v, t_scale)] = wp.overload(
            _batch_scale_velocities_kernel,
            [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=t_scale)],
        )
        _scale_velocities_out_kernel_overload[(v, t_scale)] = wp.overload(
            _scale_velocities_out_kernel,
            [wp.array(dtype=v), wp.array(dtype=t_scale), wp.array(dtype=v)],
        )
        _batch_scale_velocities_out_kernel_overload[(v, t_scale)] = wp.overload(
            _batch_scale_velocities_out_kernel,
            [
                wp.array(dtype=v),
                wp.array(dtype=wp.int32),
                wp.array(dtype=t_scale),
                wp.array(dtype=v),
            ],
        )

# NHC chain propagation kernels - keyed by scalar type
_nhc_chain_propagate_single_kernel_overload = {}
for t in _T:
    _nhc_chain_propagate_single_kernel_overload[t] = wp.overload(
        _nhc_chain_propagate_single_kernel,
        [
            wp.array(dtype=t),  # eta
            wp.array(dtype=t),  # eta_dot
            wp.array(dtype=t),  # eta_mass
            wp.array(dtype=t),  # ke2
            wp.array(dtype=t),  # target_temp
            wp.array(dtype=t),  # ndof
            wp.array(dtype=t),  # dt_chain
            wp.int32,  # chain_length
            wp.array(dtype=t),  # vel_scale
        ],
    )

# NHC chain energy kernels - keyed by scalar type
_nhc_compute_chain_energy_kernel_overload = {}
_batch_nhc_compute_chain_energy_kernel_overload = {}
for t in _T:
    _nhc_compute_chain_energy_kernel_overload[t] = wp.overload(
        _nhc_compute_chain_energy_kernel,
        [
            wp.array(dtype=t),  # eta
            wp.array(dtype=t),  # eta_dot
            wp.array(dtype=t),  # eta_mass
            wp.array(dtype=t),  # target_temp
            wp.array(dtype=t),  # ndof
            wp.int32,  # chain_length
            wp.array(dtype=t),  # ke_chain
            wp.array(dtype=t),  # pe_chain
        ],
    )
    _batch_nhc_compute_chain_energy_kernel_overload[t] = wp.overload(
        _batch_nhc_compute_chain_energy_kernel,
        [
            wp.array2d(dtype=t),  # eta
            wp.array2d(dtype=t),  # eta_dot
            wp.array2d(dtype=t),  # eta_mass
            wp.array(dtype=t),  # target_temp
            wp.array(dtype=t),  # ndof
            wp.int32,  # chain_length
            wp.array(dtype=t),  # ke_chain
            wp.array(dtype=t),  # pe_chain
        ],
    )


# ==============================================================================
# Functional Interfaces
# ==============================================================================


def nhc_compute_masses(
    ndof: wp.array,
    target_temp: wp.array,
    tau: wp.array,
    chain_length: int,
    masses: wp.array,
    num_systems: int = 1,
    device: str = None,
    dtype=wp.float64,
) -> wp.array:
    """Compute Nosé-Hoover chain masses using GPU kernel.

    Computes Q_k values for Nosé-Hoover chain:
        Q_0 = ndof * kT * tau^2
        Q_k = kT * tau^2 for k > 0

    Parameters
    ----------
    ndof : wp.array(dtype=wp.int32)
        Number of degrees of freedom per system. Shape (1,) for single system,
        (num_systems,) for batched.
    target_temp : wp.array
        Target temperature (kT) per system. Shape (1,) for single system,
        (num_systems,) for batched.
    tau : wp.array
        Time constant per system. Shape (1,) for single system,
        (num_systems,) for batched.
    chain_length : int
        Number of thermostats in the chain.
    masses : wp.array
        Chain masses output. Caller must pre-allocate.
        Shape (chain_length,) for single system,
        (num_systems, chain_length) for batched.
    num_systems : int, optional
        Number of systems for batched mode. Default: 1.
    device : str, optional
        Warp device. If None, inferred from masses.
    dtype : dtype, optional
        Data type for the masses. Default: wp.float64.

    Returns
    -------
    wp.array
        Chain masses. Shape (chain_length,) for single system,
        (num_systems, chain_length) for batched.
    """
    if device is None:
        device = masses.device

    is_batched = masses.ndim == 2

    # Select overload based on dtype
    scalar_type = dtype

    if is_batched:
        wp.launch(
            _batch_nhc_compute_masses_kernel_overload[scalar_type],
            dim=(num_systems, chain_length),
            inputs=[ndof, target_temp, tau, masses],
            device=device,
        )
    else:
        wp.launch(
            _nhc_compute_masses_kernel_overload[scalar_type],
            dim=chain_length,
            inputs=[ndof, target_temp, tau, masses],
            device=device,
        )

    return masses


def nhc_thermostat_chain_update(
    velocities: wp.array,
    masses: wp.array,
    eta: wp.array,
    eta_dot: wp.array,
    eta_mass: wp.array,
    target_temp: wp.array,
    dt: wp.array,
    ndof: wp.array,
    ke2: wp.array,
    total_scale: wp.array,
    step_scale: wp.array,
    dt_chain: wp.array,
    nloops: int = 1,
    batch_idx: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> None:
    """
    Propagate Nosé-Hoover chain and scale velocities (in-place).

    Uses Yoshida-Suzuki factorization for time-reversible integration.
    All computations are performed on GPU using Warp kernels.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    eta : wp.array(dtype=wp.float64)
        Thermostat chain positions.
        Non-batched: Shape (chain_length,).
        Batched: Shape (num_systems, chain_length) as wp.array2d.
        MODIFIED in-place.
    eta_dot : wp.array(dtype=wp.float64)
        Thermostat chain velocities. Same shape as eta. MODIFIED in-place.
    eta_mass : wp.array(dtype=wp.float64)
        Thermostat chain masses. Same shape as eta.
    target_temp : wp.array(dtype=wp.float64)
        Target temperature (kT). Shape (1,) or (num_systems,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Time step. Shape (1,) or (num_systems,).
    ndof : wp.array(dtype=wp.float64)
        Degrees of freedom. Shape (1,) or (num_systems,).
    ke2 : wp.array
        Scratch array for 2*KE computation. Zeroed internally before each use.
        Shape (1,) for single system, (num_systems,) for batched.
    total_scale : wp.array
        Scratch array for accumulated velocity scale factor.
        Must be initialized to ones by caller (wp.ones).
        Shape (1,) for single system, (num_systems,) for batched.
    step_scale : wp.array
        Scratch array for per-step velocity scale factor.
        Shape (1,) for single system, (num_systems,) for batched.
    dt_chain : wp.array
        Scratch array for weighted time steps.
        Shape (1,) for single system, (num_systems,) for batched.
    nloops : int, optional
        Number of Yoshida-Suzuki integration sub-steps. Default: 1.
        Use nloops=3 or 5 for higher accuracy.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    num_systems : int, optional
        Number of systems for batched mode. Default: 1.
    device : str, optional
        Warp device. If None, inferred from velocities.
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]
    is_batched = batch_idx is not None

    # Get Yoshida-Suzuki weights
    if nloops == 1:
        weights = [1.0]
    elif nloops == 3:
        weights = YOSHIDA_SUZUKI_3
    elif nloops == 5:
        weights = YOSHIDA_SUZUKI_5
    else:
        # Simple equal weights for other values
        weights = [1.0 / nloops] * nloops

    # Determine chain length
    if is_batched:
        chain_length = eta.shape[1]
    else:
        chain_length = eta.shape[0]

    if chain_length > MAX_CHAIN_LENGTH:
        raise ValueError(
            f"Chain length {chain_length} exceeds maximum {MAX_CHAIN_LENGTH}"
        )

    # Compute 2*KE - ke2 is zeroed internally before each use
    vec_dtype = velocities.dtype
    chain_dtype = eta.dtype
    n_scale = num_systems if is_batched else 1
    ke2.zero_()
    if is_batched:
        wp.launch(
            _batch_compute_2ke_kernel_overload[(vec_dtype, chain_dtype)],
            dim=num_atoms,
            inputs=[velocities, masses, batch_idx, ke2],
            device=device,
        )
    else:
        wp.launch(
            _compute_2ke_kernel_overload[(vec_dtype, chain_dtype)],
            dim=num_atoms,
            inputs=[velocities, masses, ke2],
            device=device,
        )

    # Run Yoshida-Suzuki sub-steps
    for w in weights:
        # Compute weighted time step: dt_chain = w * dt
        if is_batched:
            # For batched case, we need to scale each system's dt by the weight
            _compute_weighted_dt(dt, dt_chain, w, num_systems, device)
        else:
            _compute_weighted_dt(dt, dt_chain, w, 1, device)
        # Propagate chain
        if is_batched:
            wp.launch(
                _nhc_chain_propagate_kernel,
                dim=num_systems,
                inputs=[
                    eta,
                    eta_dot,
                    eta_mass,
                    ke2,
                    target_temp,
                    ndof,
                    dt_chain,
                    chain_length,
                    step_scale,
                ],
                device=device,
            )
        else:
            wp.launch(
                _nhc_chain_propagate_single_kernel_overload[chain_dtype],
                dim=1,
                inputs=[
                    eta,
                    eta_dot,
                    eta_mass,
                    ke2,
                    target_temp,
                    ndof,
                    dt_chain,
                    chain_length,
                    step_scale,
                ],
                device=device,
            )

        # Accumulate total scale factor
        wp.launch(
            _multiply_scale_factors_kernel_overload[chain_dtype],
            dim=n_scale,
            inputs=[total_scale, step_scale],
            device=device,
        )
    # Scale velocities
    if is_batched:
        wp.launch(
            _batch_scale_velocities_kernel_overload[(vec_dtype, chain_dtype)],
            dim=num_atoms,
            inputs=[velocities, batch_idx, total_scale],
            device=device,
        )
    else:
        wp.launch(
            _scale_velocities_kernel_overload[(vec_dtype, chain_dtype)],
            dim=num_atoms,
            inputs=[velocities, total_scale],
            device=device,
        )


@wp.kernel
def _compute_weighted_dt_kernel(
    dt: wp.array(dtype=Any),
    dt_chain: wp.array(dtype=Any),
    weight: Any,
):
    """Compute weighted time step: dt_chain = weight * dt.

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    dt_chain[sys_id] = type(dt_chain[sys_id])(dt[sys_id]) * type(dt_chain[sys_id])(
        weight
    )


# Overloads for _compute_weighted_dt_kernel - support all combinations of dt and dt_chain dtypes
_compute_weighted_dt_kernel_overload = {}
for t_in in _T:
    for t_out in _T:
        _compute_weighted_dt_kernel_overload[(t_in, t_out)] = wp.overload(
            _compute_weighted_dt_kernel,
            [
                wp.array(dtype=t_in),
                wp.array(dtype=t_out),
                t_out,
            ],  # weight uses output type
        )


def _compute_weighted_dt(
    dt: wp.array,
    dt_chain: wp.array,
    weight: float,
    num_systems: int,
    device: str,
):
    """Helper to compute weighted dt using appropriate kernel overload."""
    dt_dtype = dt.dtype
    dt_chain_dtype = dt_chain.dtype
    weight_typed = dt_chain_dtype(weight)
    wp.launch(
        _compute_weighted_dt_kernel_overload[(dt_dtype, dt_chain_dtype)],
        dim=num_systems,
        inputs=[dt, dt_chain, weight_typed],
        device=device,
    )


def nhc_thermostat_chain_update_out(
    velocities: wp.array,
    masses: wp.array,
    eta: wp.array,
    eta_dot: wp.array,
    eta_mass: wp.array,
    target_temp: wp.array,
    dt: wp.array,
    ndof: wp.array,
    ke2: wp.array,
    total_scale: wp.array,
    step_scale: wp.array,
    dt_chain: wp.array,
    velocities_out: wp.array,
    eta_out: wp.array,
    eta_dot_out: wp.array,
    nloops: int = 1,
    batch_idx: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> tuple[wp.array, wp.array, wp.array]:
    """
    Propagate Nosé-Hoover chain and scale velocities (non-mutating).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    eta : wp.array(dtype=wp.float64)
        Thermostat chain positions. Shape (M,) or (B, M).
    eta_dot : wp.array(dtype=wp.float64)
        Thermostat chain velocities. Shape (M,) or (B, M).
    eta_mass : wp.array(dtype=wp.float64)
        Thermostat chain masses. Shape (M,) or (B, M).
    target_temp : wp.array(dtype=wp.float64)
        Target temperature (kT). Shape (1,) or (B,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Time step. Shape (1,) or (B,).
    ndof : wp.array(dtype=wp.float64)
        Degrees of freedom. Shape (1,) or (B,).
    ke2 : wp.array
        Scratch array for 2*KE computation. Zeroed internally before each use.
        Shape (1,) for single system, (num_systems,) for batched.
    total_scale : wp.array
        Scratch array for accumulated velocity scale factor.
        Must be initialized to ones by caller (wp.ones).
        Shape (1,) for single system, (num_systems,) for batched.
    step_scale : wp.array
        Scratch array for per-step velocity scale factor.
        Shape (1,) for single system, (num_systems,) for batched.
    dt_chain : wp.array
        Scratch array for weighted time steps.
        Shape (1,) for single system, (num_systems,) for batched.
    velocities_out : wp.array
        Output velocities. Must be pre-allocated with same shape/dtype/device as velocities.
    eta_out : wp.array
        Output eta. Must be pre-allocated with same shape/dtype/device as eta.
    eta_dot_out : wp.array
        Output eta_dot. Must be pre-allocated with same shape/dtype/device as eta_dot.
    nloops : int, optional
        Number of Yoshida-Suzuki integration sub-steps. Default: 1.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    num_systems : int, optional
        Number of systems for batched mode. Default: 1.
    device : str, optional
        Warp device. If None, inferred from velocities.

    Returns
    -------
    tuple[wp.array, wp.array, wp.array]
        (velocities_out, eta_out, eta_dot_out)
    """
    if device is None:
        device = velocities.device

    validate_out_array(velocities_out, velocities, "velocities_out")
    validate_out_array(eta_out, eta, "eta_out")
    validate_out_array(eta_dot_out, eta_dot, "eta_dot_out")

    # Copy inputs to outputs
    wp.copy(velocities_out, velocities)
    wp.copy(eta_out, eta)
    wp.copy(eta_dot_out, eta_dot)

    # Run in-place update on copies
    nhc_thermostat_chain_update(
        velocities_out,
        masses,
        eta_out,
        eta_dot_out,
        eta_mass,
        target_temp,
        dt,
        ndof,
        ke2,
        total_scale,
        step_scale,
        dt_chain,
        nloops=nloops,
        batch_idx=batch_idx,
        num_systems=num_systems,
        device=device,
    )

    return velocities_out, eta_out, eta_dot_out


def nhc_velocity_half_step(
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    dt: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> None:
    """
    Half-step velocity update (in-place).

    v += 0.5 * (F/m) * dt

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Time step. Shape (1,) or (B,).
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from velocities.
    """
    dispatch_family(
        velocity_kick_families,
        velocities,
        batch_idx=batch_idx,
        atom_ptr=atom_ptr,
        device=device,
        inputs_single=[velocities, forces, masses, dt, velocities],
        inputs_batch=[velocities, forces, masses, batch_idx, dt, velocities],
        inputs_ptr=[velocities, forces, masses, atom_ptr, dt, velocities],
    )


def nhc_velocity_half_step_out(
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    dt: wp.array,
    velocities_out: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Half-step velocity update (non-mutating).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Time step. Shape (1,) or (B,).
    velocities_out : wp.array
        Output velocities. Must be pre-allocated with same shape/dtype/device as velocities.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from velocities.

    Returns
    -------
    wp.array
        Updated velocities.
    """
    validate_out_array(velocities_out, velocities, "velocities_out")
    dispatch_family(
        velocity_kick_families,
        velocities,
        batch_idx=batch_idx,
        atom_ptr=atom_ptr,
        device=device,
        inputs_single=[velocities, forces, masses, dt, velocities_out],
        inputs_batch=[velocities, forces, masses, batch_idx, dt, velocities_out],
        inputs_ptr=[velocities, forces, masses, atom_ptr, dt, velocities_out],
    )
    return velocities_out


def nhc_position_update(
    positions: wp.array,
    velocities: wp.array,
    dt: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> None:
    """
    Full-step position update (in-place).

    r += v * dt

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,). MODIFIED in-place.
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Time step. Shape (1,) or (B,).
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from positions.
    """
    dispatch_family(
        position_update_families,
        positions,
        batch_idx=batch_idx,
        atom_ptr=atom_ptr,
        device=device,
        inputs_single=[positions, velocities, dt, positions],
        inputs_batch=[positions, velocities, batch_idx, dt, positions],
        inputs_ptr=[positions, velocities, atom_ptr, dt, positions],
    )


def nhc_position_update_out(
    positions: wp.array,
    velocities: wp.array,
    dt: wp.array,
    positions_out: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Full-step position update (non-mutating).

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,).
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Time step. Shape (1,) or (B,).
    positions_out : wp.array
        Output positions. Must be pre-allocated with same shape/dtype/device as positions.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from positions.

    Returns
    -------
    wp.array
        Updated positions.
    """
    validate_out_array(positions_out, positions, "positions_out")
    dispatch_family(
        position_update_families,
        positions,
        batch_idx=batch_idx,
        atom_ptr=atom_ptr,
        device=device,
        inputs_single=[positions, velocities, dt, positions_out],
        inputs_batch=[positions, velocities, batch_idx, dt, positions_out],
        inputs_ptr=[positions, velocities, atom_ptr, dt, positions_out],
    )
    return positions_out


def nhc_compute_chain_energy(
    eta: wp.array,
    eta_dot: wp.array,
    eta_mass: wp.array,
    target_temp: wp.array,
    ndof: wp.array,
    ke_chain: wp.array,
    pe_chain: wp.array,
    batch_idx: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> tuple[wp.array, wp.array]:
    """
    Compute Nosé-Hoover chain kinetic and potential energy.

    For conservation checks, the extended system Hamiltonian is:
        H_ext = KE_particles + PE + KE_chain + PE_chain

    where:
        KE_chain = sum_k 0.5 * Q_k * eta_dot_k^2
        PE_chain = ndof * kT * eta_0 + kT * sum_{k>0} eta_k

    Parameters
    ----------
    eta : wp.array(dtype=wp.float64)
        Thermostat chain positions. Shape (M,) or (B, M).
    eta_dot : wp.array(dtype=wp.float64)
        Thermostat chain velocities. Shape (M,) or (B, M).
    eta_mass : wp.array(dtype=wp.float64)
        Thermostat chain masses. Shape (M,) or (B, M).
    target_temp : wp.array(dtype=wp.float64)
        Target temperature (kT). Shape (1,) or (B,).
    ndof : wp.array(dtype=wp.float64)
        Degrees of freedom. Shape (1,) or (B,).
    ke_chain : wp.array
        Output kinetic energy of the chain.
        Shape (1,) for single system, (num_systems,) for batched.
    pe_chain : wp.array
        Output potential energy of the chain.
        Shape (1,) for single system, (num_systems,) for batched.
    batch_idx : wp.array(dtype=wp.int32), optional
        Not used directly, but included for API consistency.
    num_systems : int, optional
        Number of systems for batched mode. Default: 1.
    device : str, optional
        Warp device. If None, inferred from eta.

    Returns
    -------
    tuple[wp.array, wp.array]
        (ke_chain, pe_chain) each with shape (1,) or (B,).
    """
    if device is None:
        device = eta.device

    is_batched = num_systems > 1 or (batch_idx is not None)

    # Determine chain length
    if is_batched:
        chain_length = eta.shape[1]
    else:
        chain_length = eta.shape[0]

    chain_dtype = eta.dtype

    if is_batched:
        wp.launch(
            _batch_nhc_compute_chain_energy_kernel_overload[chain_dtype],
            dim=num_systems,
            inputs=[
                eta,
                eta_dot,
                eta_mass,
                target_temp,
                ndof,
                chain_length,
                ke_chain,
                pe_chain,
            ],
            device=device,
        )
    else:
        wp.launch(
            _nhc_compute_chain_energy_kernel_overload[chain_dtype],
            dim=1,
            inputs=[
                eta,
                eta_dot,
                eta_mass,
                target_temp,
                ndof,
                chain_length,
                ke_chain,
                pe_chain,
            ],
            device=device,
        )

    return ke_chain, pe_chain
