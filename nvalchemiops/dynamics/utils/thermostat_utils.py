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
Thermostat Utility Kernels
==========================

GPU-accelerated Warp kernels for temperature-related computations in
molecular dynamics simulations.

This module provides both mutating (in-place) and non-mutating versions
of each kernel for gradient tracking compatibility.

MATHEMATICAL FORMULATION
========================

Kinetic Energy:

.. math::

    KE = \\frac{1}{2} \\sum_i m_i |\\mathbf{v}_i|^2

Temperature (from equipartition theorem):

.. math::

    T = \\frac{2 \\cdot KE}{N_{DOF} \\cdot k_B}

Maxwell-Boltzmann Distribution:

.. math::

    v_i \\sim \\mathcal{N}\\left(0, \\sqrt{\\frac{k_B T}{m_i}}\\right)

BATCH MODE
==========

All functions in this module support three execution modes:

**Single System Mode**::

    ke = wp.empty(1, dtype=wp.float64, device="cuda:0")
    compute_kinetic_energy(velocities, masses, ke)
    temperature = wp.array([1.0], dtype=wp.float64, device="cuda:0")
    total_momentum = wp.empty(1, dtype=wp.vec3d, device="cuda:0")
    total_mass = wp.empty(1, dtype=wp.float64, device="cuda:0")
    com_velocities = wp.empty(1, dtype=wp.vec3d, device="cuda:0")
    initialize_velocities(
        velocities, masses, temperature, total_momentum, total_mass, com_velocities
    )

**Batch Mode with batch_idx** (atomic operations)::

    # Each atom tagged with its system ID
    batch_idx = wp.array([0]*N0 + [1]*N1 + [2]*N2, dtype=wp.int32, device="cuda:0")

    # Compute per-system kinetic energies
    ke = wp.empty(3, dtype=wp.float64, device="cuda:0")
    compute_kinetic_energy(
        velocities, masses, ke, batch_idx=batch_idx, num_systems=3
    )  # ke now has shape (3,)

    # Initialize with per-system temperatures
    temperature = wp.array([1.0, 1.5, 0.8], dtype=wp.float64, device="cuda:0")
    total_momentum = wp.empty(3, dtype=wp.vec3d, device="cuda:0")
    total_mass = wp.empty(3, dtype=wp.float64, device="cuda:0")
    com_velocities = wp.empty(3, dtype=wp.vec3d, device="cuda:0")
    initialize_velocities(
        velocities, masses, temperature,
        total_momentum, total_mass, com_velocities,
        batch_idx=batch_idx, num_systems=3
    )

**Batch Mode with atom_ptr** (sequential per-system)::

    # CSR-style pointers defining atom ranges
    atom_ptr = wp.array([0, N0, N0+N1, N0+N1+N2], dtype=wp.int32, device="cuda:0")

    # Same operations as batch_idx mode, but with atom_ptr
    ke = wp.empty(3, dtype=wp.float64, device="cuda:0")
    compute_kinetic_energy(
        velocities, masses, ke, atom_ptr=atom_ptr, num_systems=3
    )
"""

from __future__ import annotations

import os
from typing import Any

import warp as wp

__all__ = [
    # Mutating APIs
    "compute_kinetic_energy",
    "compute_temperature",
    "initialize_velocities",
    "remove_com_motion",
    # Non-mutating APIs
    "initialize_velocities_out",
    "remove_com_motion_out",
]


# ==============================================================================
# Kinetic Energy Kernels
# ==============================================================================

# Tile block size for cooperative reductions
TILE_DIM = int(os.getenv("NVALCHEMIOPS_DYNAMICS_TILE_DIM", 256))


@wp.kernel
def _compute_kinetic_energy_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    kinetic_energy: wp.array(dtype=Any),
):
    """Compute kinetic energy contribution from each atom.

    Accumulates KE = 0.5 * sum_i(m_i * v_i · v_i) using atomic adds.

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms,).
    kinetic_energy : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Pre-allocated output array.
        Shape (1,) for single system

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    vel = velocities[atom_idx]
    mass = masses[atom_idx]

    v_sq = wp.dot(vel, vel)
    ke_contribution = type(mass)(0.5) * mass * v_sq

    wp.atomic_add(kinetic_energy, 0, ke_contribution)


@wp.kernel
def _compute_kinetic_energy_tiled_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    kinetic_energy: wp.array(dtype=Any),
):
    """Compute kinetic energy with tile reductions (single system).

    Accumulates KE = 0.5 * sum_i(m_i * v_i · v_i) using block-level reductions.

    Launch Grid: dim = [num_atoms], block_dim = TILE_DIM
    """
    atom_idx = wp.tid()

    vel = velocities[atom_idx]
    mass = masses[atom_idx]

    v_sq = wp.dot(vel, vel)
    ke_contribution = type(mass)(0.5) * mass * v_sq

    # Convert to tile for block-level reduction
    t_ke = wp.tile(ke_contribution)

    # Cooperative sum within block
    s_ke = wp.tile_sum(t_ke)

    # Extract scalar from tile sum
    sum_ke = s_ke[0]

    # Only first thread in block writes
    if atom_idx % TILE_DIM == 0:
        wp.atomic_add(kinetic_energy, 0, sum_ke)


@wp.kernel
def _batch_compute_kinetic_energy_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    kinetic_energies: wp.array(dtype=Any),
):
    """Compute per-system kinetic energy for batched systems.

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms,).
    batch_idx : wp.array(dtype=wp.int32)
        System index for each atom. Shape (num_atoms,).
    kinetic_energies : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Pre-allocated output array.
        Shape (num_systems,).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    vel = velocities[atom_idx]
    mass = masses[atom_idx]

    v_sq = wp.dot(vel, vel)
    ke_contribution = type(mass)(0.5) * mass * v_sq

    wp.atomic_add(kinetic_energies, system_id, ke_contribution)


@wp.kernel
def _batch_compute_kinetic_energy_tiled_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    kinetic_energies: wp.array(dtype=Any),
):
    """Compute per-system kinetic energy with tile reductions (batched).

    Launch Grid: dim = [num_atoms_total], block_dim = TILE_DIM
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    vel = velocities[atom_idx]
    mass = masses[atom_idx]

    v_sq = wp.dot(vel, vel)
    ke_contribution = type(mass)(0.5) * mass * v_sq

    # Convert to tile for block-level reduction
    t_ke = wp.tile(ke_contribution)

    # Cooperative sum within block
    s_ke = wp.tile_sum(t_ke)

    # Extract scalar from tile sum
    sum_ke = s_ke[0]

    # Only first thread in block writes
    if atom_idx % TILE_DIM == 0:
        wp.atomic_add(kinetic_energies, system_id, sum_ke)


@wp.kernel
def _compute_kinetic_energy_ptr_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    kinetic_energies: wp.array(dtype=Any),
):
    """Compute per-system kinetic energy using atom_ptr (CSR format).

    Each thread processes one system's atoms sequentially.

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms_total,).
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms_total,).
    atom_ptr : wp.array(dtype=wp.int32)
        CSR-style pointers. Shape (num_systems + 1,).
        System s owns atoms in range [atom_ptr[s], atom_ptr[s+1]).
    kinetic_energies : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Pre-allocated output array. Shape (num_systems,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]

    ke_sum = type(kinetic_energies[0])(0.0)
    for i in range(a0, a1):
        vel = velocities[i]
        mass = masses[i]
        v_sq = wp.dot(vel, vel)
        ke_sum += type(mass)(0.5) * mass * v_sq

    kinetic_energies[sys_id] = ke_sum


# ==============================================================================
# COM Velocity Kernels
# ==============================================================================


@wp.kernel
def _compute_com_velocity_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    total_momentum: wp.array(dtype=Any),
    total_mass: wp.array(dtype=Any),
):
    """Compute center of mass momentum and total mass.

    COM velocity is computed after kernel as: v_COM = total_momentum / total_mass

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms,).
    total_momentum : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Total momentum. Shape (1,).
    total_mass : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Total mass. Shape (1,).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    vel = velocities[atom_idx]
    mass = masses[atom_idx]

    mom = mass * vel
    wp.atomic_add(total_momentum, 0, mom)
    wp.atomic_add(total_mass, 0, mass)


@wp.kernel
def _compute_com_velocity_tiled_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    total_momentum: wp.array(dtype=Any),
    total_mass: wp.array(dtype=Any),
):
    """Compute center of mass momentum and total mass with tile reductions (single system).

    Launch Grid: dim = [num_atoms], block_dim = TILE_DIM
    """
    atom_idx = wp.tid()

    vel = velocities[atom_idx]
    mass = masses[atom_idx]

    mom = mass * vel

    # Convert to tiles for block-level reduction
    t_mom_x = wp.tile(mom[0])
    t_mom_y = wp.tile(mom[1])
    t_mom_z = wp.tile(mom[2])
    t_mass = wp.tile(mass)

    # Cooperative sum within block
    s_mom_x = wp.tile_sum(t_mom_x)
    s_mom_y = wp.tile_sum(t_mom_y)
    s_mom_z = wp.tile_sum(t_mom_z)
    s_mass = wp.tile_sum(t_mass)

    # Extract scalar values from tile sums
    sum_mom_x = s_mom_x[0]
    sum_mom_y = s_mom_y[0]
    sum_mom_z = s_mom_z[0]
    sum_mass = s_mass[0]

    # Only first thread in block writes
    if atom_idx % TILE_DIM == 0:
        sum_mom = type(vel)(sum_mom_x, sum_mom_y, sum_mom_z)
        wp.atomic_add(total_momentum, 0, sum_mom)
        wp.atomic_add(total_mass, 0, sum_mass)


@wp.kernel
def _batch_compute_com_velocity_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    total_momentum: wp.array(dtype=Any),
    total_mass: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
):
    """Compute center of mass momentum and total mass.

    COM velocity is computed after kernel as: v_COM = total_momentum / total_mass

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms,).
    total_momentum : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Total momentum. Shape (1,).
    total_mass : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Total mass. Shape (1,).
    batch_idx : wp.array(dtype=wp.int32), e.g., wp.array(dtype=wp.int32)
        System index for each atom. Shape (num_atoms,).
    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    vel = velocities[atom_idx]
    mass = masses[atom_idx]

    mom = mass * vel
    wp.atomic_add(total_momentum, system_id, mom)
    wp.atomic_add(total_mass, system_id, mass)


@wp.kernel
def _batch_compute_com_velocity_tiled_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    total_momentum: wp.array(dtype=Any),
    total_mass: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
):
    """Compute center of mass momentum and total mass with tile reductions (batched).

    Launch Grid: dim = [num_atoms], block_dim = TILE_DIM
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    vel = velocities[atom_idx]
    mass = masses[atom_idx]

    mom = mass * vel

    # Convert to tiles for block-level reduction
    t_mom_x = wp.tile(mom[0])
    t_mom_y = wp.tile(mom[1])
    t_mom_z = wp.tile(mom[2])
    t_mass = wp.tile(mass)

    # Cooperative sum within block
    s_mom_x = wp.tile_sum(t_mom_x)
    s_mom_y = wp.tile_sum(t_mom_y)
    s_mom_z = wp.tile_sum(t_mom_z)
    s_mass = wp.tile_sum(t_mass)

    # Extract scalar values from tile sums
    sum_mom_x = s_mom_x[0]
    sum_mom_y = s_mom_y[0]
    sum_mom_z = s_mom_z[0]
    sum_mass = s_mass[0]

    # Only first thread in block writes
    if atom_idx % TILE_DIM == 0:
        sum_mom = type(vel)(sum_mom_x, sum_mom_y, sum_mom_z)
        wp.atomic_add(total_momentum, system_id, sum_mom)
        wp.atomic_add(total_mass, system_id, sum_mass)


@wp.kernel
def _compute_com_velocity_ptr_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    total_momentum: wp.array(dtype=Any),
    total_mass: wp.array(dtype=Any),
):
    """Compute center of mass momentum and total mass using atom_ptr (CSR format).

    Each thread processes one system's atoms sequentially.

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms_total,).
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms_total,).
    atom_ptr : wp.array(dtype=wp.int32)
        CSR-style pointers. Shape (num_systems + 1,).
    total_momentum : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Total momentum per system. Shape (num_systems,).
    total_mass : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Total mass per system. Shape (num_systems,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]

    mom = total_momentum[sys_id]
    tmass = total_mass[sys_id]

    mom_sum = type(mom)(type(mom[0])(0.0), type(mom[0])(0.0), type(mom[0])(0.0))
    mass_sum = type(tmass)(0.0)

    for i in range(a0, a1):
        vel = velocities[i]
        mass = masses[i]
        mom_sum += mass * vel
        mass_sum += mass

    total_momentum[sys_id] = mom_sum
    total_mass[sys_id] = mass_sum


@wp.kernel
def _remove_com_motion_kernel(
    velocities: wp.array(dtype=Any),
    com_velocity: wp.array(dtype=Any),
):
    """Remove center of mass velocity from all atoms (in-place).

    v_i = v_i - v_COM

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    com_velocity : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Center of mass velocity. Shape (1,).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    vel = velocities[atom_idx]

    velocities[atom_idx] = vel - com_velocity[0]


@wp.kernel
def _batch_remove_com_motion_kernel(
    velocities: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    com_velocity: wp.array(dtype=Any),
):
    """Remove center of mass velocity from all atoms (in-place).

    v_i = v_i - v_COM

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    batch_idx : wp.array(dtype=wp.int32), e.g., wp.array(dtype=wp.int32)
        System index for each atom. Shape (num_atoms,).
    com_velocity : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Center of mass velocity. Shape (num_systems,).
    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    vel = velocities[atom_idx]

    velocities[atom_idx] = vel - com_velocity[system_id]


@wp.kernel
def _remove_com_motion_ptr_kernel(
    velocities: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    com_velocity: wp.array(dtype=Any),
):
    """Remove center of mass velocity using atom_ptr (in-place).

    Each thread processes one system's atoms sequentially.

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms_total,). MODIFIED in-place.
    atom_ptr : wp.array(dtype=wp.int32)
        CSR-style pointers. Shape (num_systems + 1,).
    com_velocity : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Center of mass velocity per system. Shape (num_systems,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]
    v_com = com_velocity[sys_id]

    for i in range(a0, a1):
        vel = velocities[i]
        velocities[i] = vel - v_com


@wp.kernel
def _remove_com_motion_out_kernel(
    velocities: wp.array(dtype=Any),
    com_velocity: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Remove center of mass velocity from all atoms (non-mutating).

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    com_velocity : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Center of mass velocity. Shape (1,).
    velocities_out : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Output velocities. Shape (num_atoms,).
    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    vel = velocities[atom_idx]

    velocities_out[atom_idx] = vel - com_velocity[0]


@wp.kernel
def _batch_remove_com_motion_out_kernel(
    velocities: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    com_velocity: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Remove center of mass velocity from all atoms (non-mutating).

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    batch_idx : wp.array(dtype=wp.int32), e.g., wp.array(dtype=wp.int32)
        System index for each atom. Shape (num_atoms,).
    com_velocity : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Center of mass velocity. Shape (num_systems,).
    velocities_out : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Output velocities. Shape (num_atoms,).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    vel = velocities[atom_idx]

    velocities_out[atom_idx] = vel - com_velocity[system_id]


@wp.kernel
def _remove_com_motion_ptr_out_kernel(
    velocities: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    com_velocity: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Remove center of mass velocity using atom_ptr (non-mutating).

    Each thread processes one system's atoms sequentially.

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms_total,).
    atom_ptr : wp.array(dtype=wp.int32)
        CSR-style pointers. Shape (num_systems + 1,).
    com_velocity : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Center of mass velocity per system. Shape (num_systems,).
    velocities_out : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Output velocities. Shape (num_atoms_total,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]
    v_com = com_velocity[sys_id]

    for i in range(a0, a1):
        vel = velocities[i]
        velocities_out[i] = vel - v_com


# ==============================================================================
# Velocity Initialization Kernels
# ==============================================================================


@wp.kernel
def _initialize_velocities_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    random_seed: wp.uint64,
):
    """Initialize velocities from Maxwell-Boltzmann distribution (in-place).

    Each velocity component is drawn from N(0, sqrt(kT/m)).

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms,).
    temperature : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Temperature - k_B * T. Shape (1,).
    random_seed : wp.uint64
        Random seed.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    mass = masses[atom_idx]
    kT = type(mass)(temperature[0])

    # Standard deviation: sigma = sqrt(kT/m)
    sigma = wp.where(
        mass > type(mass)(0.0), wp.sqrt(type(mass)(kT) / mass), type(mass)(0.0)
    )

    # Initialize RNG state for this atom
    rng_state = wp.rand_init(int(random_seed), atom_idx)

    # Generate Gaussian-distributed velocities using wp.randn (N(0,1))
    vx = sigma * type(mass)(wp.randn(rng_state))
    vy = sigma * type(mass)(wp.randn(rng_state))
    vz = sigma * type(mass)(wp.randn(rng_state))

    vel = velocities[atom_idx]
    velocities[atom_idx] = type(vel)(vx, vy, vz)


@wp.kernel
def _batch_initialize_velocities_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    random_seed: wp.uint64,
    batch_idx: wp.array(dtype=wp.int32),
):
    """Initialize velocities from Maxwell-Boltzmann distribution (in-place, batched).

    Each velocity component is drawn from N(0, sqrt(kT/m)).

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms,).
    temperature : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Temperature - k_B * T. Shape (num_systems,).
    random_seed : wp.uint64
        Random seed.
    batch_idx : wp.array(dtype=wp.int32)
        System index for each atom. Shape (num_atoms,).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    mass = masses[atom_idx]
    kT = type(mass)(temperature[system_id])

    # Standard deviation: sigma = sqrt(kT/m)
    sigma = wp.sqrt(type(mass)(kT) / mass)

    # Initialize RNG state for this atom
    rng_state = wp.rand_init(int(random_seed), atom_idx)

    # Generate Gaussian-distributed velocities using wp.randn (N(0,1))
    vx = sigma * type(mass)(wp.randn(rng_state))
    vy = sigma * type(mass)(wp.randn(rng_state))
    vz = sigma * type(mass)(wp.randn(rng_state))

    vel = velocities[atom_idx]
    velocities[atom_idx] = type(vel)(vx, vy, vz)


@wp.kernel
def _initialize_velocities_ptr_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    random_seed: wp.uint64,
    atom_ptr: wp.array(dtype=wp.int32),
):
    """Initialize velocities from Maxwell-Boltzmann distribution (in-place, atom_ptr).

    Each thread processes one system's atoms sequentially.

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms_total,). MODIFIED in-place.
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms_total,).
    temperature : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Temperature - k_B * T. Shape (num_systems,).
    random_seed : wp.uint64
        Random seed.
    atom_ptr : wp.array(dtype=wp.int32)
        CSR-style pointers. Shape (num_systems + 1,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]
    kT = type(masses[a0])(temperature[sys_id])

    for i in range(a0, a1):
        mass = masses[i]
        sigma = wp.where(mass > type(mass)(0.0), wp.sqrt(kT / mass), type(mass)(0.0))

        # Use (random_seed + i) for per-atom variation
        rng_state = wp.rand_init(int(random_seed), i)

        vx = sigma * type(mass)(wp.randn(rng_state))
        vy = sigma * type(mass)(wp.randn(rng_state))
        vz = sigma * type(mass)(wp.randn(rng_state))

        vel = velocities[i]
        velocities[i] = type(vel)(vx, vy, vz)


@wp.kernel
def _initialize_velocities_out_kernel(
    masses: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    random_seed: wp.uint64,
    velocities_out: wp.array(dtype=Any),
):
    """Initialize velocities from Maxwell-Boltzmann distribution (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    mass = masses[atom_idx]
    kT = temperature[0]

    kT_typed = type(mass)(kT)
    sigma = wp.where(mass > type(mass)(0.0), wp.sqrt(kT_typed / mass), type(mass)(0.0))

    rng_state = wp.rand_init(int(random_seed), atom_idx)

    # Generate Gaussian-distributed velocities using wp.randn (N(0,1))
    vx = sigma * type(mass)(wp.randn(rng_state))
    vy = sigma * type(mass)(wp.randn(rng_state))
    vz = sigma * type(mass)(wp.randn(rng_state))

    vel_sample = velocities_out[atom_idx]
    velocities_out[atom_idx] = type(vel_sample)(vx, vy, vz)


@wp.kernel
def _batch_initialize_velocities_out_kernel(
    masses: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    random_seed: wp.uint64,
    velocities_out: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
):
    """Initialize velocities from Maxwell-Boltzmann distribution (non-mutating, batched).

    Parameters
    ----------
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms,).
    temperature : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Temperature. Shape (num_systems,).
    random_seed : wp.uint64
        Random seed.
    velocities_out : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Output velocities. Shape (num_atoms,).
    batch_idx : wp.array(dtype=wp.int32)
        System index for each atom. Shape (num_atoms,).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    mass = masses[atom_idx]
    kT = temperature[system_id]

    kT_typed = type(mass)(kT)
    sigma = wp.where(mass > type(mass)(0.0), wp.sqrt(kT_typed / mass), type(mass)(0.0))

    rng_state = wp.rand_init(int(random_seed), atom_idx)

    # Generate Gaussian-distributed velocities using wp.randn (N(0,1))
    vx = sigma * type(mass)(wp.randn(rng_state))
    vy = sigma * type(mass)(wp.randn(rng_state))
    vz = sigma * type(mass)(wp.randn(rng_state))

    vel_sample = velocities_out[atom_idx]
    velocities_out[atom_idx] = type(vel_sample)(vx, vy, vz)


@wp.kernel
def _initialize_velocities_ptr_out_kernel(
    masses: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    random_seed: wp.uint64,
    atom_ptr: wp.array(dtype=wp.int32),
    velocities_out: wp.array(dtype=Any),
):
    """Initialize velocities from Maxwell-Boltzmann distribution (non-mutating, atom_ptr).

    Each thread processes one system's atoms sequentially.

    Parameters
    ----------
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms_total,).
    temperature : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Temperature - k_B * T. Shape (num_systems,).
    random_seed : wp.uint64
        Random seed.
    atom_ptr : wp.array(dtype=wp.int32)
        CSR-style pointers. Shape (num_systems + 1,).
    velocities_out : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Output velocities. Shape (num_atoms_total,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]
    kT = temperature[sys_id]

    for i in range(a0, a1):
        mass = masses[i]
        kT_typed = type(mass)(kT)
        sigma = wp.where(
            mass > type(mass)(0.0), wp.sqrt(kT_typed / mass), type(mass)(0.0)
        )

        # Use (random_seed + i) for per-atom variation
        rng_state = wp.rand_init(int(random_seed), i)

        vx = sigma * type(mass)(wp.randn(rng_state))
        vy = sigma * type(mass)(wp.randn(rng_state))
        vz = sigma * type(mass)(wp.randn(rng_state))

        vel_sample = velocities_out[i]
        velocities_out[i] = type(vel_sample)(vx, vy, vz)


# ==============================================================================
# COM Velocity Division Kernels
# ==============================================================================


@wp.kernel
def _compute_com_from_momentum_kernel(
    total_momentum: wp.array(dtype=Any),
    total_mass: wp.array(dtype=Any),
    com_velocity: wp.array(dtype=Any),
):
    """Compute COM velocity from total momentum and mass (single system).

    v_COM = total_momentum / total_mass

    Parameters
    ----------
    total_momentum : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Total momentum. Shape (1,).
    total_mass : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Total mass. Shape (1,).
    com_velocity : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Center of mass velocity. Shape (1,).

    Launch Grid
    -----------
    dim = [1]
    """
    mass = total_mass[0]
    # Guard against division by zero: if total_mass is zero, set inv_mass to zero
    inv_mass = wp.where(mass > type(mass)(0.0), type(mass)(1.0) / mass, type(mass)(0.0))
    com_velocity[0] = total_momentum[0] * inv_mass


@wp.kernel
def _batch_compute_com_from_momentum_kernel(
    total_momentum: wp.array(dtype=Any),
    total_mass: wp.array(dtype=Any),
    com_velocities: wp.array(dtype=Any),
):
    """Compute COM velocity from total momentum and mass (batched).

    Parameters
    ----------
    total_momentum : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Total momentum. Shape (num_systems,).
    total_mass : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Total mass. Shape (num_systems,).
    com_velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Center of mass velocities. Shape (num_systems,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    mass = total_mass[sys_id]
    inv_mass = wp.where(mass > type(mass)(0.0), type(mass)(1.0) / mass, type(mass)(0.0))
    com_velocities[sys_id] = total_momentum[sys_id] * inv_mass


# ==============================================================================
# Temperature Computation Kernels
# ==============================================================================


@wp.kernel
def _compute_temperature_from_ke_kernel(
    kinetic_energies: wp.array(dtype=Any),
    num_atoms_per_system: wp.array(dtype=Any),
    temperatures: wp.array(dtype=Any),
):
    """Compute temperature from kinetic energy for batched systems: T = 2*KE / DOF

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    ke = kinetic_energies[sys_id]
    dof = 3 * num_atoms_per_system[sys_id] - 3
    temperatures[sys_id] = wp.where(
        type(ke)(dof) > type(ke)(0.0), type(ke)(2.0) * ke / type(ke)(dof), type(ke)(0.0)
    )


# ==============================================================================
# Kernel Overloads for Explicit Typing
# ==============================================================================

_T = [wp.float32, wp.float64]  # Scalar types
_V = [wp.vec3f, wp.vec3d]  # Vector types

# Kinetic energy kernel overloads
_compute_kinetic_energy_kernel_overload = {}
_compute_kinetic_energy_tiled_kernel_overload = {}
_batch_compute_kinetic_energy_kernel_overload = {}
_batch_compute_kinetic_energy_tiled_kernel_overload = {}
_compute_kinetic_energy_ptr_kernel_overload = {}

# Temperature kernel overloads
_compute_temperature_from_ke_kernel_overload = {}

# COM velocity kernel overloads
_compute_com_velocity_kernel_overload = {}
_compute_com_velocity_tiled_kernel_overload = {}
_batch_compute_com_velocity_kernel_overload = {}
_batch_compute_com_velocity_tiled_kernel_overload = {}
_compute_com_velocity_ptr_kernel_overload = {}

# Remove COM motion kernel overloads
_remove_com_motion_kernel_overload = {}
_batch_remove_com_motion_kernel_overload = {}
_remove_com_motion_ptr_kernel_overload = {}
_remove_com_motion_out_kernel_overload = {}
_batch_remove_com_motion_out_kernel_overload = {}
_remove_com_motion_ptr_out_kernel_overload = {}

# Initialize velocities kernel overloads
_initialize_velocities_kernel_overload = {}
_batch_initialize_velocities_kernel_overload = {}
_initialize_velocities_ptr_kernel_overload = {}
_initialize_velocities_out_kernel_overload = {}
_batch_initialize_velocities_out_kernel_overload = {}
_initialize_velocities_ptr_out_kernel_overload = {}

for t, v in zip(_T, _V):
    # Kinetic energy kernels (dtype agnostic - output matches input type)
    _compute_kinetic_energy_kernel_overload[v] = wp.overload(
        _compute_kinetic_energy_kernel,
        [wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=t)],
    )
    _compute_kinetic_energy_tiled_kernel_overload[v] = wp.overload(
        _compute_kinetic_energy_tiled_kernel,
        [wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=t)],
    )
    _batch_compute_kinetic_energy_kernel_overload[v] = wp.overload(
        _batch_compute_kinetic_energy_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
        ],
    )
    _batch_compute_kinetic_energy_tiled_kernel_overload[v] = wp.overload(
        _batch_compute_kinetic_energy_tiled_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
        ],
    )
    _compute_kinetic_energy_ptr_kernel_overload[v] = wp.overload(
        _compute_kinetic_energy_ptr_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
        ],
    )

    # Temperature kernels (keyed by scalar type)
    _compute_temperature_from_ke_kernel_overload[t] = wp.overload(
        _compute_temperature_from_ke_kernel,
        [wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )

    # COM velocity kernels (now using 1D vector arrays for momentum)
    _compute_com_velocity_kernel_overload[v] = wp.overload(
        _compute_com_velocity_kernel,
        [wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=v), wp.array(dtype=t)],
    )
    _compute_com_velocity_tiled_kernel_overload[v] = wp.overload(
        _compute_com_velocity_tiled_kernel,
        [wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=v), wp.array(dtype=t)],
    )
    _batch_compute_com_velocity_kernel_overload[v] = wp.overload(
        _batch_compute_com_velocity_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
        ],
    )
    _batch_compute_com_velocity_tiled_kernel_overload[v] = wp.overload(
        _batch_compute_com_velocity_tiled_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
        ],
    )
    _compute_com_velocity_ptr_kernel_overload[v] = wp.overload(
        _compute_com_velocity_ptr_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
            wp.array(dtype=t),
        ],
    )

    # Remove COM motion kernels (now using 1D vector arrays for com_velocity)
    _remove_com_motion_kernel_overload[v] = wp.overload(
        _remove_com_motion_kernel,
        [wp.array(dtype=v), wp.array(dtype=v)],
    )
    _batch_remove_com_motion_kernel_overload[v] = wp.overload(
        _batch_remove_com_motion_kernel,
        [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=v)],
    )
    _remove_com_motion_ptr_kernel_overload[v] = wp.overload(
        _remove_com_motion_ptr_kernel,
        [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=v)],
    )
    _remove_com_motion_out_kernel_overload[v] = wp.overload(
        _remove_com_motion_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=v)],
    )
    _batch_remove_com_motion_out_kernel_overload[v] = wp.overload(
        _batch_remove_com_motion_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
            wp.array(dtype=v),
        ],
    )
    _remove_com_motion_ptr_out_kernel_overload[v] = wp.overload(
        _remove_com_motion_ptr_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
            wp.array(dtype=v),
        ],
    )

    # Initialize velocities kernels (batch_idx moved to end)
    _initialize_velocities_kernel_overload[v] = wp.overload(
        _initialize_velocities_kernel,
        [wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=t), wp.uint64],
    )
    _batch_initialize_velocities_kernel_overload[v] = wp.overload(
        _batch_initialize_velocities_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.uint64,
            wp.array(dtype=wp.int32),
        ],
    )
    _initialize_velocities_ptr_kernel_overload[v] = wp.overload(
        _initialize_velocities_ptr_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.uint64,
            wp.array(dtype=wp.int32),
        ],
    )
    _initialize_velocities_out_kernel_overload[v] = wp.overload(
        _initialize_velocities_out_kernel,
        [wp.array(dtype=t), wp.array(dtype=t), wp.uint64, wp.array(dtype=v)],
    )
    _batch_initialize_velocities_out_kernel_overload[v] = wp.overload(
        _batch_initialize_velocities_out_kernel,
        [
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.uint64,
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
        ],
    )
    _initialize_velocities_ptr_out_kernel_overload[v] = wp.overload(
        _initialize_velocities_ptr_out_kernel,
        [
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.uint64,
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
        ],
    )


# ==============================================================================
# Functional Interface
# ==============================================================================


def compute_kinetic_energy(
    velocities: wp.array,
    masses: wp.array,
    kinetic_energy: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> wp.array:
    """
    Compute kinetic energy for single or batched MD systems.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    kinetic_energy : wp.array
        Output array. Same dtype as masses.
        Shape (1,) for single system, (B,) for batched.
        Zeroed internally before each use.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    num_systems : int, optional
        Number of systems for batched mode. Default 1.
    device : str, optional
        Warp device. If None, inferred from velocities.

    Returns
    -------
    wp.array
        Kinetic energy (same dtype as masses). Shape (1,) for single, (B,) for batched.

    Example
    -------
    Single system::

        import warp as wp
        import numpy as np

        velocities = wp.array(np.random.randn(100, 3), dtype=wp.vec3d, device="cuda:0")
        masses = wp.array(np.ones(100), dtype=wp.float64, device="cuda:0")

        ke = wp.empty(1, dtype=wp.float64, device="cuda:0")
        ke = compute_kinetic_energy(velocities, masses, ke)
        print(f"Kinetic energy: {ke.numpy()[0]}")

    Batched mode with batch_idx::

        # 3 systems with different atom counts
        batch_idx = wp.array([0]*30 + [1]*40 + [2]*30, dtype=wp.int32, device="cuda:0")
        ke = wp.empty(3, dtype=wp.float64, device="cuda:0")
        ke = compute_kinetic_energy(
            velocities, masses, ke, batch_idx=batch_idx, num_systems=3
        )
        # ke.shape = (3,), one KE per system

    Batched mode with atom_ptr::

        atom_ptr = wp.array([0, 30, 70, 100], dtype=wp.int32, device="cuda:0")
        ke = wp.empty(3, dtype=wp.float64, device="cuda:0")
        ke = compute_kinetic_energy(
            velocities, masses, ke, atom_ptr=atom_ptr, num_systems=3
        )

    See Also
    --------
    compute_temperature : Convert kinetic energy to temperature
    """
    if batch_idx is not None and atom_ptr is not None:
        raise ValueError("Provide batch_idx OR atom_ptr, not both")

    if device is None:
        device = velocities.device

    kinetic_energy.zero_()
    num_atoms = velocities.shape[0]

    vec_dtype = velocities.dtype

    if atom_ptr is not None:
        # Use atom_ptr mode (CSR) - launch with dim=num_systems
        num_systems_actual = atom_ptr.shape[0] - 1
        wp.launch(
            _compute_kinetic_energy_ptr_kernel_overload[vec_dtype],
            dim=num_systems_actual,
            inputs=[velocities, masses, atom_ptr, kinetic_energy],
            device=device,
        )
    elif batch_idx is not None:
        # Use batch_idx mode (no tiles - threads in block belong to different systems)
        wp.launch(
            _batch_compute_kinetic_energy_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, batch_idx, kinetic_energy],
            device=device,
        )
    else:
        # Single system with tiles - launch with dim=num_atoms
        wp.launch(
            _compute_kinetic_energy_tiled_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, kinetic_energy],
            device=device,
            block_dim=TILE_DIM,
        )

    return kinetic_energy


def compute_temperature(
    kinetic_energy: wp.array,
    temperature: wp.array,
    num_atoms_per_system: wp.array,
) -> wp.array:
    """
    Compute instantaneous temperature from kinetic energy.

    Temperature is computed as T = 2*KE / (DOF * k_B), where k_B = 1
    in natural units (so temperature is in energy units).

    Parameters
    ----------
    kinetic_energy : wp.array
        Pre-computed kinetic energy. Shape (1,) or (B,).
    temperature : wp.array
        Output temperature array. Temperature - k_B * T. Shape (1,) or (B,).
    num_atoms_per_system : wp.array
        Number of atoms (per system for batched).

    Returns
    -------
    wp.array
        Temperature in energy units (k_B*T). Shape (1,) or (B,).
    """

    # Compute temperature: T = 2*KE / DOF using Warp kernel
    wp.launch(
        _compute_temperature_from_ke_kernel_overload[kinetic_energy.dtype],
        dim=num_atoms_per_system.shape[0],
        inputs=[kinetic_energy, num_atoms_per_system, temperature],
        device=kinetic_energy.device,
    )

    return temperature


def initialize_velocities(
    velocities: wp.array,
    masses: wp.array,
    temperature: wp.array,
    total_momentum: wp.array,
    total_mass: wp.array,
    com_velocities: wp.array,
    random_seed: int = 42,
    remove_com: bool = True,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> None:
    """
    Initialize velocities from Maxwell-Boltzmann distribution (in-place).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    temperature : wp.array(dtype=wp.float32 or wp.float64)
        Target temperature (k_B*T in energy units). Shape (1,) or (B,).
    total_momentum : wp.array
        Scratch array for COM removal. Same vec dtype as velocities.
        Shape (B,) for batched, (1,) for single.
        Zeroed internally before each use.
        Only used when ``remove_com=True``.
    total_mass : wp.array
        Scratch array for COM removal. Same scalar dtype as masses.
        Shape (B,) for batched, (1,) for single.
        Zeroed internally before each use.
        Only used when ``remove_com=True``.
    com_velocities : wp.array
        Scratch array for COM removal. Same vec dtype as velocities.
        Shape (B,) for batched, (1,) for single.
        Only used when ``remove_com=True``.
    random_seed : int, optional
        Random seed for reproducibility. Default: 42.
    remove_com : bool, optional
        If True, remove center of mass motion after initialization.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    num_systems : int, optional
        Number of systems. Default 1.
    device : str, optional
        Warp device.

    See Also
    --------
    remove_com_motion : Remove center of mass motion
    compute_temperature : Compute instantaneous temperature
    """
    if batch_idx is not None and atom_ptr is not None:
        raise ValueError("Provide batch_idx OR atom_ptr, not both")

    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]
    vec_dtype = velocities.dtype

    if atom_ptr is not None:
        # Use atom_ptr mode - launch with dim=num_systems
        num_systems_actual = atom_ptr.shape[0] - 1
        wp.launch(
            _initialize_velocities_ptr_kernel_overload[vec_dtype],
            dim=num_systems_actual,
            inputs=[velocities, masses, temperature, wp.uint64(random_seed), atom_ptr],
            device=device,
        )
    elif batch_idx is not None:
        # Use batch_idx mode - launch with dim=num_atoms
        wp.launch(
            _batch_initialize_velocities_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, temperature, wp.uint64(random_seed), batch_idx],
            device=device,
        )
    else:
        # Single system - launch with dim=num_atoms
        wp.launch(
            _initialize_velocities_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, temperature, wp.uint64(random_seed)],
            device=device,
        )

    if remove_com:
        remove_com_motion(
            velocities,
            masses,
            total_momentum,
            total_mass,
            com_velocities,
            batch_idx=batch_idx,
            atom_ptr=atom_ptr,
            num_systems=num_systems,
            device=device,
        )


def initialize_velocities_out(
    masses: wp.array,
    temperature: wp.array,
    velocities_out: wp.array,
    total_momentum: wp.array,
    total_mass: wp.array,
    com_velocities: wp.array,
    random_seed: int = 42,
    remove_com: bool = True,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> wp.array:
    """
    Initialize velocities from Maxwell-Boltzmann distribution (non-mutating).

    Parameters
    ----------
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    temperature : wp.array(dtype=wp.float32 or wp.float64)
        Target temperature (k_B*T in energy units). Shape (1,) or (B,).
    velocities_out : wp.array
        Output array for velocities. Shape (N,). Caller must pre-allocate.
    total_momentum : wp.array
        Scratch array for COM removal. Same vec dtype as velocities.
        Shape (B,) for batched, (1,) for single.
        Zeroed internally before each use.
        Only used when ``remove_com=True``.
    total_mass : wp.array
        Scratch array for COM removal. Same scalar dtype as masses.
        Shape (B,) for batched, (1,) for single.
        Zeroed internally before each use.
        Only used when ``remove_com=True``.
    com_velocities : wp.array
        Scratch array for COM removal. Same vec dtype as velocities.
        Shape (B,) for batched, (1,) for single.
        Only used when ``remove_com=True``.
    random_seed : int, optional
        Random seed for reproducibility. Default: 42.
    remove_com : bool, optional
        If True, remove center of mass motion after initialization.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    num_systems : int, optional
        Number of systems. Default 1.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Initialized velocities.
    """
    if batch_idx is not None and atom_ptr is not None:
        raise ValueError("Provide batch_idx OR atom_ptr, not both")

    if device is None:
        device = masses.device

    num_atoms = masses.shape[0]

    # Determine correct dtypes based on masses
    scalar_dtype = masses.dtype
    if scalar_dtype == wp.float64:
        vec_dtype = wp.vec3d
    else:
        vec_dtype = wp.vec3f

    if atom_ptr is not None:
        # Use atom_ptr mode - launch with dim=num_systems
        num_systems_actual = atom_ptr.shape[0] - 1
        wp.launch(
            _initialize_velocities_ptr_out_kernel_overload[vec_dtype],
            dim=num_systems_actual,
            inputs=[
                masses,
                temperature,
                wp.uint64(random_seed),
                atom_ptr,
                velocities_out,
            ],
            device=device,
        )
    elif batch_idx is not None:
        # Use batch_idx mode - launch with dim=num_atoms
        wp.launch(
            _batch_initialize_velocities_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[
                masses,
                temperature,
                wp.uint64(random_seed),
                velocities_out,
                batch_idx,
            ],
            device=device,
        )
    else:
        # Single system - launch with dim=num_atoms
        wp.launch(
            _initialize_velocities_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[masses, temperature, wp.uint64(random_seed), velocities_out],
            device=device,
        )

    if remove_com:
        remove_com_motion(
            velocities_out,
            masses,
            total_momentum,
            total_mass,
            com_velocities,
            batch_idx=batch_idx,
            atom_ptr=atom_ptr,
            num_systems=num_systems,
            device=device,
        )

    return velocities_out


def remove_com_motion(
    velocities: wp.array,
    masses: wp.array,
    total_momentum: wp.array,
    total_mass: wp.array,
    com_velocities: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> None:
    """
    Remove center of mass velocity from the system (in-place).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    total_momentum : wp.array
        Scratch array for momentum accumulation. Same vec dtype as velocities.
        Shape (B,) for batched, (1,) for single.
        Zeroed internally before each use.
    total_mass : wp.array
        Scratch array for mass accumulation. Same scalar dtype as masses.
        Shape (B,) for batched, (1,) for single.
        Zeroed internally before each use.
    com_velocities : wp.array
        Scratch array for COM velocities. Same vec dtype as velocities.
        Shape (B,) for batched, (1,) for single.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    num_systems : int, optional
        Number of systems. Default 1.
    device : str, optional
        Warp device.
    """
    if batch_idx is not None and atom_ptr is not None:
        raise ValueError("Provide batch_idx OR atom_ptr, not both")

    if device is None:
        device = velocities.device

    total_momentum.zero_()
    total_mass.zero_()
    num_atoms = velocities.shape[0]

    vec_dtype = velocities.dtype

    if atom_ptr is not None:
        # Use atom_ptr mode - launch with dim=num_systems
        num_systems_actual = atom_ptr.shape[0] - 1

        wp.launch(
            _compute_com_velocity_ptr_kernel_overload[vec_dtype],
            dim=num_systems_actual,
            inputs=[velocities, masses, atom_ptr, total_momentum, total_mass],
            device=device,
        )

        # Compute COM velocity using Warp kernel (no numpy)
        wp.launch(
            _batch_compute_com_from_momentum_kernel,
            dim=num_systems_actual,
            inputs=[total_momentum, total_mass, com_velocities],
            device=device,
        )

        wp.launch(
            _remove_com_motion_ptr_kernel_overload[vec_dtype],
            dim=num_systems_actual,
            inputs=[velocities, atom_ptr, com_velocities],
            device=device,
        )
    elif batch_idx is not None:
        # Use batch_idx mode

        wp.launch(
            _batch_compute_com_velocity_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, total_momentum, total_mass, batch_idx],
            device=device,
        )

        # Compute COM velocity using Warp kernel (no numpy)
        wp.launch(
            _batch_compute_com_from_momentum_kernel,
            dim=num_systems,
            inputs=[total_momentum, total_mass, com_velocities],
            device=device,
        )

        wp.launch(
            _batch_remove_com_motion_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, batch_idx, com_velocities],
            device=device,
        )
    else:
        # Single system - launch with dim=num_atomm
        wp.launch(
            _compute_com_velocity_tiled_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, total_momentum, total_mass],
            device=device,
            block_dim=TILE_DIM,
        )

        # Compute COM velocity using Warp kernel (no numpy)
        wp.launch(
            _compute_com_from_momentum_kernel,
            dim=1,
            inputs=[total_momentum, total_mass, com_velocities],
            device=device,
        )

        wp.launch(
            _remove_com_motion_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, com_velocities],
            device=device,
        )


def remove_com_motion_out(
    velocities: wp.array,
    masses: wp.array,
    total_momentum: wp.array,
    total_mass: wp.array,
    com_velocities: wp.array,
    velocities_out: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> wp.array:
    """
    Remove center of mass velocity from the system (non-mutating).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    total_momentum : wp.array
        Scratch array for momentum accumulation. Same vec dtype as velocities.
        Shape (B,) for batched, (1,) for single.
        Zeroed internally before each use.
    total_mass : wp.array
        Scratch array for mass accumulation. Same scalar dtype as masses.
        Shape (B,) for batched, (1,) for single.
        Zeroed internally before each use.
    com_velocities : wp.array
        Scratch array for COM velocities. Same vec dtype as velocities.
        Shape (B,) for batched, (1,) for single.
    velocities_out : wp.array
        Output array. Shape (N,). Caller must pre-allocate.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    num_systems : int, optional
        Number of systems. Default 1.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Velocities with COM motion removed.
    """
    if batch_idx is not None and atom_ptr is not None:
        raise ValueError("Provide batch_idx OR atom_ptr, not both")

    if device is None:
        device = velocities.device

    total_momentum.zero_()
    total_mass.zero_()
    num_atoms = velocities.shape[0]

    vec_dtype = velocities.dtype

    if atom_ptr is not None:
        # Use atom_ptr mode - launch with dim=num_systems
        num_systems_actual = atom_ptr.shape[0] - 1

        wp.launch(
            _compute_com_velocity_ptr_kernel_overload[vec_dtype],
            dim=num_systems_actual,
            inputs=[velocities, masses, atom_ptr, total_momentum, total_mass],
            device=device,
        )

        # Compute COM velocity using Warp kernel (no numpy)
        wp.launch(
            _batch_compute_com_from_momentum_kernel,
            dim=num_systems_actual,
            inputs=[total_momentum, total_mass, com_velocities],
            device=device,
        )

        wp.launch(
            _remove_com_motion_ptr_out_kernel_overload[vec_dtype],
            dim=num_systems_actual,
            inputs=[velocities, atom_ptr, com_velocities, velocities_out],
            device=device,
        )
    elif batch_idx is not None:
        # Use batch_idx mode - launch with dim=num_atoms

        wp.launch(
            _batch_compute_com_velocity_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, total_momentum, total_mass, batch_idx],
            device=device,
        )

        # Compute COM velocity using Warp kernel (no numpy)
        wp.launch(
            _batch_compute_com_from_momentum_kernel,
            dim=num_systems,
            inputs=[total_momentum, total_mass, com_velocities],
            device=device,
        )

        wp.launch(
            _batch_remove_com_motion_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, batch_idx, com_velocities, velocities_out],
            device=device,
        )
    else:
        # Single system - launch with dim=num_atoms

        wp.launch(
            _compute_com_velocity_tiled_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, total_momentum, total_mass],
            device=device,
            block_dim=TILE_DIM,
        )

        # Compute COM velocity using Warp kernel (no numpy)
        wp.launch(
            _compute_com_from_momentum_kernel,
            dim=1,
            inputs=[total_momentum, total_mass, com_velocities],
            device=device,
        )

        wp.launch(
            _remove_com_motion_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, com_velocities, velocities_out],
            device=device,
        )

    return velocities_out
