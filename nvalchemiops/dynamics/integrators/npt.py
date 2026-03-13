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
NPT and NPH Integrators with Isotropic and Anisotropic Pressure Control.

This module implements the Martyna-Tobias-Klein (MTK) equations of motion
for NPT and NPH molecular dynamics simulations.

NPT (Isothermal-Isobaric)
-------------------------
Constant temperature and pressure via coupled Nosé-Hoover thermostat
and barostat. Samples the canonical NPT ensemble.

NPH (Isenthalpic-Isobaric)
--------------------------
Constant enthalpy and pressure without thermostat. The system evolves
on a constant enthalpy surface at fixed pressure.

Pressure Control Modes
----------------------
- Isotropic: Single pressure value, cell scales uniformly
- Anisotropic (orthorhombic): Independent x, y, z pressure control
- Anisotropic (triclinic): Full stress tensor control (9 components)

References
----------
- Martyna, Tobias, Klein, J. Chem. Phys. 101, 4177 (1994)
- Shinoda, Shiga, Mikami, Phys. Rev. B 69, 134103 (2004)
- LAMMPS fix_nh documentation

All kernels are dtype-agnostic and support both float32 and float64.
"""

from __future__ import annotations

import os
from typing import Any

import warp as wp

from nvalchemiops.dynamics.utils.cell_utils import (
    compute_cell_inverse,
    compute_cell_volume,
)
from nvalchemiops.dynamics.utils.launch_helpers import (
    KernelFamily,
    launch_family,
    resolve_execution_mode,
)
from nvalchemiops.dynamics.utils.thermostat_utils import compute_kinetic_energy
from nvalchemiops.warp_dispatch import validate_out_array

__all__ = [
    # Tensor types for pressure/virial
    "vec9f",
    "vec9d",
    "vec3f",
    "vec3d",
    # Pressure calculations
    "compute_pressure_tensor",
    "compute_scalar_pressure",
    # Barostat utilities
    "compute_barostat_mass",
    "compute_cell_kinetic_energy",
    "compute_barostat_potential_energy",
    # NPT integration steps - Mutating
    "npt_thermostat_half_step",
    "npt_barostat_half_step",  # Unified: dispatches based on target_pressures dtype
    "npt_velocity_half_step",  # Unified: dispatches based on mode parameter
    "npt_position_update",
    "npt_cell_update",
    # NPT integration steps - Non-mutating
    "npt_velocity_half_step_out",
    "npt_position_update_out",
    "npt_cell_update_out",
    # High-level NPT integration
    "run_npt_step",
    # NPH integration steps - Mutating
    "nph_barostat_half_step",  # Unified: dispatches based on target_pressures dtype
    "nph_velocity_half_step",  # Unified: dispatches based on mode parameter
    "nph_position_update",
    "nph_cell_update",
    # NPH integration steps - Non-mutating
    "nph_velocity_half_step_out",
    "nph_position_update_out",
    # High-level NPH integration
    "run_nph_step",
]


# ==============================================================================
# Tensor Types
# ==============================================================================

# 9-element vector types for symmetric tensor storage
# Order: [xx, xy, xz, yx, yy, yz, zx, zy, zz]
vec9f = wp.types.vector(length=9, dtype=wp.float32)
vec9d = wp.types.vector(length=9, dtype=wp.float64)

# 3-element vector types for anisotropic (orthorhombic) pressure
# Order: [Pxx, Pyy, Pzz]
vec3f = wp.vec3f
vec3d = wp.vec3d


# =============================================================================
# Shared @wp.func for NPT/NPH physics
# =============================================================================


@wp.func
def _npt_accel(f: Any, m: Any) -> Any:
    """Acceleration: F/m as vec3."""
    inv_mass = wp.where(m > type(m)(0.0), type(m)(1.0) / m, type(m)(0.0))
    return type(f)(f[0] * inv_mass, f[1] * inv_mass, f[2] * inv_mass)


@wp.func
def _drag_isotropic(h_dot: Any, V: Any, coupling: Any, eta_dot_1: Any, v: Any) -> Any:
    """Isotropic drag: (coupling * Tr(h_dot)/(3V) + eta_dot_1) * v."""
    trace_h_dot = h_dot[0, 0] + h_dot[1, 1] + h_dot[2, 2]
    eps_dot = trace_h_dot / (type(V)(3.0) * V)
    return (coupling * eps_dot + eta_dot_1) * v


@wp.func
def _drag_anisotropic(h_dot: Any, V: Any, coupling: Any, eta_dot_1: Any, v: Any) -> Any:
    """Anisotropic (diagonal) drag."""
    eps_dot_xx = h_dot[0, 0] / V
    eps_dot_yy = h_dot[1, 1] / V
    eps_dot_zz = h_dot[2, 2] / V
    return type(v)(
        (coupling * eps_dot_xx + eta_dot_1) * v[0],
        (coupling * eps_dot_yy + eta_dot_1) * v[1],
        (coupling * eps_dot_zz + eta_dot_1) * v[2],
    )


@wp.func
def _drag_triclinic(
    h_dot: Any, h_inv: Any, coupling: Any, eta_dot_1: Any, v: Any
) -> Any:
    """Triclinic (full tensor) drag: (coupling * h_dot @ h_inv + eta_dot_1 * I) @ v."""
    eps_dot = h_dot * h_inv
    drag_x = (
        coupling * (eps_dot[0, 0] * v[0] + eps_dot[0, 1] * v[1] + eps_dot[0, 2] * v[2])
        + eta_dot_1 * v[0]
    )
    drag_y = (
        coupling * (eps_dot[1, 0] * v[0] + eps_dot[1, 1] * v[1] + eps_dot[1, 2] * v[2])
        + eta_dot_1 * v[1]
    )
    drag_z = (
        coupling * (eps_dot[2, 0] * v[0] + eps_dot[2, 1] * v[1] + eps_dot[2, 2] * v[2])
        + eta_dot_1 * v[2]
    )
    return type(v)(drag_x, drag_y, drag_z)


@wp.func
def _npt_position_step(r: Any, v: Any, h_dot: Any, h_inv: Any, dt: Any) -> Any:
    """Position update: r + dt * (v + eps_dot @ r)."""
    eps_dot = wp.mul(h_dot, h_inv)
    eps_dot_r = wp.mul(eps_dot, r)
    return r + dt * (v + eps_dot_r)


# ==============================================================================
# Pressure Calculation Kernels
# ==============================================================================

# Tile block size for cooperative reductions
TILE_DIM = int(os.getenv("NVALCHEMIOPS_DYNAMICS_TILE_DIM", 256))


@wp.kernel
def _compute_kinetic_tensor_single_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    kinetic_tensors: wp.array2d(dtype=Any),
):
    """Compute kinetic contribution to pressure tensor (single system).

    P_kin[i,j] = sum_k m_k * v_k[i] * v_k[j]

    Launch Grid: dim = [num_atoms]
    Note: Uses array2d for atomic operations, converted to vec9 after.
    """
    atom_idx = wp.tid()
    v = velocities[atom_idx]
    m = masses[atom_idx]

    # Outer product: m * v ⊗ v
    wp.atomic_add(kinetic_tensors, 0, 0, m * v[0] * v[0])  # xx
    wp.atomic_add(kinetic_tensors, 0, 1, m * v[0] * v[1])  # xy
    wp.atomic_add(kinetic_tensors, 0, 2, m * v[0] * v[2])  # xz
    wp.atomic_add(kinetic_tensors, 0, 3, m * v[1] * v[0])  # yx
    wp.atomic_add(kinetic_tensors, 0, 4, m * v[1] * v[1])  # yy
    wp.atomic_add(kinetic_tensors, 0, 5, m * v[1] * v[2])  # yz
    wp.atomic_add(kinetic_tensors, 0, 6, m * v[2] * v[0])  # zx
    wp.atomic_add(kinetic_tensors, 0, 7, m * v[2] * v[1])  # zy
    wp.atomic_add(kinetic_tensors, 0, 8, m * v[2] * v[2])  # zz


@wp.kernel
def _compute_kinetic_tensor_single_tiled_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    kinetic_tensors: wp.array2d(dtype=Any),
):
    """Compute kinetic contribution to pressure tensor with tile reductions (single system).

    P_kin[i,j] = sum_k m_k * v_k[i] * v_k[j]

    Launch Grid: dim = [num_atoms], block_dim = TILE_DIM
    """
    atom_idx = wp.tid()
    v = velocities[atom_idx]
    m = masses[atom_idx]

    # Compute local outer product components
    p_xx = m * v[0] * v[0]
    p_xy = m * v[0] * v[1]
    p_xz = m * v[0] * v[2]
    p_yx = m * v[1] * v[0]
    p_yy = m * v[1] * v[1]
    p_yz = m * v[1] * v[2]
    p_zx = m * v[2] * v[0]
    p_zy = m * v[2] * v[1]
    p_zz = m * v[2] * v[2]

    # Convert to tiles for block-level reduction
    t_xx = wp.tile(p_xx)
    t_xy = wp.tile(p_xy)
    t_xz = wp.tile(p_xz)
    t_yx = wp.tile(p_yx)
    t_yy = wp.tile(p_yy)
    t_yz = wp.tile(p_yz)
    t_zx = wp.tile(p_zx)
    t_zy = wp.tile(p_zy)
    t_zz = wp.tile(p_zz)

    # Cooperative sum within block
    s_xx = wp.tile_sum(t_xx)
    s_xy = wp.tile_sum(t_xy)
    s_xz = wp.tile_sum(t_xz)
    s_yx = wp.tile_sum(t_yx)
    s_yy = wp.tile_sum(t_yy)
    s_yz = wp.tile_sum(t_yz)
    s_zx = wp.tile_sum(t_zx)
    s_zy = wp.tile_sum(t_zy)
    s_zz = wp.tile_sum(t_zz)

    # Extract scalar values from tile sums
    sum_xx = s_xx[0]
    sum_xy = s_xy[0]
    sum_xz = s_xz[0]
    sum_yx = s_yx[0]
    sum_yy = s_yy[0]
    sum_yz = s_yz[0]
    sum_zx = s_zx[0]
    sum_zy = s_zy[0]
    sum_zz = s_zz[0]

    # Only first thread in block writes (9 atomics per block)
    if atom_idx % TILE_DIM == 0:
        wp.atomic_add(kinetic_tensors, 0, 0, sum_xx)
        wp.atomic_add(kinetic_tensors, 0, 1, sum_xy)
        wp.atomic_add(kinetic_tensors, 0, 2, sum_xz)
        wp.atomic_add(kinetic_tensors, 0, 3, sum_yx)
        wp.atomic_add(kinetic_tensors, 0, 4, sum_yy)
        wp.atomic_add(kinetic_tensors, 0, 5, sum_yz)
        wp.atomic_add(kinetic_tensors, 0, 6, sum_zx)
        wp.atomic_add(kinetic_tensors, 0, 7, sum_zy)
        wp.atomic_add(kinetic_tensors, 0, 8, sum_zz)


@wp.kernel
def _compute_kinetic_tensor_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    kinetic_tensors: wp.array2d(dtype=Any),
):
    """Compute kinetic contribution to pressure tensor (batched).

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    v = velocities[atom_idx]
    m = masses[atom_idx]

    wp.atomic_add(kinetic_tensors, sys_id, 0, m * v[0] * v[0])
    wp.atomic_add(kinetic_tensors, sys_id, 1, m * v[0] * v[1])
    wp.atomic_add(kinetic_tensors, sys_id, 2, m * v[0] * v[2])
    wp.atomic_add(kinetic_tensors, sys_id, 3, m * v[1] * v[0])
    wp.atomic_add(kinetic_tensors, sys_id, 4, m * v[1] * v[1])
    wp.atomic_add(kinetic_tensors, sys_id, 5, m * v[1] * v[2])
    wp.atomic_add(kinetic_tensors, sys_id, 6, m * v[2] * v[0])
    wp.atomic_add(kinetic_tensors, sys_id, 7, m * v[2] * v[1])
    wp.atomic_add(kinetic_tensors, sys_id, 8, m * v[2] * v[2])


@wp.kernel
def _compute_kinetic_tensor_tiled_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    kinetic_tensors: wp.array2d(dtype=Any),
):
    """Compute kinetic contribution to pressure tensor with tile reductions (batched).

    Launch Grid: dim = [num_atoms], block_dim = TILE_DIM
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    v = velocities[atom_idx]
    m = masses[atom_idx]

    # Compute local outer product components
    p_xx = m * v[0] * v[0]
    p_xy = m * v[0] * v[1]
    p_xz = m * v[0] * v[2]
    p_yx = m * v[1] * v[0]
    p_yy = m * v[1] * v[1]
    p_yz = m * v[1] * v[2]
    p_zx = m * v[2] * v[0]
    p_zy = m * v[2] * v[1]
    p_zz = m * v[2] * v[2]

    # Convert to tiles for block-level reduction
    t_xx = wp.tile(p_xx)
    t_xy = wp.tile(p_xy)
    t_xz = wp.tile(p_xz)
    t_yx = wp.tile(p_yx)
    t_yy = wp.tile(p_yy)
    t_yz = wp.tile(p_yz)
    t_zx = wp.tile(p_zx)
    t_zy = wp.tile(p_zy)
    t_zz = wp.tile(p_zz)

    # Cooperative sum within block
    s_xx = wp.tile_sum(t_xx)
    s_xy = wp.tile_sum(t_xy)
    s_xz = wp.tile_sum(t_xz)
    s_yx = wp.tile_sum(t_yx)
    s_yy = wp.tile_sum(t_yy)
    s_yz = wp.tile_sum(t_yz)
    s_zx = wp.tile_sum(t_zx)
    s_zy = wp.tile_sum(t_zy)
    s_zz = wp.tile_sum(t_zz)

    # Extract scalar values from tile sums
    sum_xx = s_xx[0]
    sum_xy = s_xy[0]
    sum_xz = s_xz[0]
    sum_yx = s_yx[0]
    sum_yy = s_yy[0]
    sum_yz = s_yz[0]
    sum_zx = s_zx[0]
    sum_zy = s_zy[0]
    sum_zz = s_zz[0]

    # Only first thread in block writes (9 atomics per block per system)
    if atom_idx % TILE_DIM == 0:
        wp.atomic_add(kinetic_tensors, sys_id, 0, sum_xx)
        wp.atomic_add(kinetic_tensors, sys_id, 1, sum_xy)
        wp.atomic_add(kinetic_tensors, sys_id, 2, sum_xz)
        wp.atomic_add(kinetic_tensors, sys_id, 3, sum_yx)
        wp.atomic_add(kinetic_tensors, sys_id, 4, sum_yy)
        wp.atomic_add(kinetic_tensors, sys_id, 5, sum_yz)
        wp.atomic_add(kinetic_tensors, sys_id, 6, sum_zx)
        wp.atomic_add(kinetic_tensors, sys_id, 7, sum_zy)
        wp.atomic_add(kinetic_tensors, sys_id, 8, sum_zz)


@wp.kernel
def _finalize_pressure_tensor_kernel(
    kinetic_tensors: wp.array2d(dtype=Any),
    virial_tensors: wp.array(dtype=Any),
    volumes: wp.array(dtype=Any),
    pressure_tensors: wp.array(dtype=Any),
):
    """Finalize pressure tensor: P = (kinetic + virial) / V.

    Launch Grid: dim = [num_systems]
    """
    sys_id = wp.tid()
    V = volumes[sys_id]
    vir = virial_tensors[sys_id]

    # Compute each component explicitly to avoid type inference issues
    p0 = (kinetic_tensors[sys_id, 0] + vir[0]) / V
    p1 = (kinetic_tensors[sys_id, 1] + vir[1]) / V
    p2 = (kinetic_tensors[sys_id, 2] + vir[2]) / V
    p3 = (kinetic_tensors[sys_id, 3] + vir[3]) / V
    p4 = (kinetic_tensors[sys_id, 4] + vir[4]) / V
    p5 = (kinetic_tensors[sys_id, 5] + vir[5]) / V
    p6 = (kinetic_tensors[sys_id, 6] + vir[6]) / V
    p7 = (kinetic_tensors[sys_id, 7] + vir[7]) / V
    p8 = (kinetic_tensors[sys_id, 8] + vir[8]) / V

    pressure_tensors[sys_id] = type(vir)(p0, p1, p2, p3, p4, p5, p6, p7, p8)


@wp.kernel
def _compute_scalar_pressure_kernel(
    pressure_tensors: wp.array(dtype=Any),
    scalar_pressures: wp.array(dtype=Any),
):
    """Compute scalar pressure: P = (P_xx + P_yy + P_zz) / 3.

    Launch Grid: dim = [num_systems]
    """
    sys_id = wp.tid()
    P = pressure_tensors[sys_id]
    trace = P[0] + P[4] + P[8]
    scalar_pressures[sys_id] = trace / type(trace)(3.0)


# ==============================================================================
# Barostat Energy Kernels
# ==============================================================================


@wp.kernel
def _compute_cell_kinetic_energy_kernel(
    cell_velocities: wp.array(dtype=Any),
    cell_masses: wp.array(dtype=Any),
    kinetic_energy: wp.array(dtype=Any),
):
    """Compute cell kinetic energy: KE = 0.5 * W * ||ḣ||²_F.

    Launch Grid: dim = [num_systems]
    """
    sys_id = wp.tid()
    h_dot = cell_velocities[sys_id]
    W = cell_masses[sys_id]

    # Frobenius norm squared
    ke = type(W)(0.0)
    for i in range(3):
        for j in range(3):
            ke = ke + h_dot[i, j] * h_dot[i, j]

    kinetic_energy[sys_id] = type(W)(0.5) * W * ke


@wp.kernel
def _compute_barostat_potential_kernel(
    target_pressures: wp.array(dtype=Any),
    volumes: wp.array(dtype=Any),
    potential_energy: wp.array(dtype=Any),
):
    """Compute barostat potential: U = P_ext * V.

    Launch Grid: dim = [num_systems]
    """
    sys_id = wp.tid()
    P_ext = target_pressures[sys_id]
    V = volumes[sys_id]
    potential_energy[sys_id] = P_ext * V


# ==============================================================================
# NPT Velocity Update Kernels
# ==============================================================================


@wp.kernel
def _npt_velocity_half_step_single_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    cell_velocity: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    eta_dot: wp.array2d(dtype=Any),
    num_atoms: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """NPT isotropic velocity half-step, single system (out-only).

    v_new = v + dt/2 * (F/m - (1 + 1/N_f) * ε̇ * v - η̇₁ * v)

    For in-place: host passes same array as velocities and velocities_out.

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    v = velocities[atom_idx]
    m = masses[atom_idx]
    f = forces[atom_idx]
    h_dot = cell_velocity[0]
    eta_dot_1 = eta_dot[0, 0]

    V = volume[0]
    N_f = type(m)(3 * num_atoms[0])
    coupling = type(m)(1.0) + type(m)(1.0) / N_f
    dt_half = dt[0] * type(m)(0.5)
    accel = _npt_accel(f, m)
    drag = _drag_isotropic(h_dot, V, coupling, eta_dot_1, v)

    velocities_out[atom_idx] = v + dt_half * (accel - drag)


@wp.kernel
def _npt_velocity_half_step_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_velocities: wp.array(dtype=Any),
    volumes: wp.array(dtype=Any),
    eta_dots: wp.array2d(dtype=Any),
    num_atoms_per_system: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """NPT isotropic velocity half-step, batched (out-only).

    For in-place: host passes same array as velocities and velocities_out.

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    v = velocities[atom_idx]
    m = masses[atom_idx]
    f = forces[atom_idx]
    h_dot = cell_velocities[sys_id]
    eta_dot_1 = eta_dots[sys_id, 0]
    N = num_atoms_per_system[sys_id]

    V = volumes[sys_id]
    N_f = type(m)(3 * N)
    coupling = type(m)(1.0) + type(m)(1.0) / N_f
    dt_half = dt[sys_id] * type(m)(0.5)
    accel = _npt_accel(f, m)
    drag = _drag_isotropic(h_dot, V, coupling, eta_dot_1, v)

    velocities_out[atom_idx] = v + dt_half * (accel - drag)


# ==============================================================================
# NPH Velocity Update Kernels (no thermostat coupling)
# ==============================================================================


@wp.kernel
def _nph_velocity_half_step_single_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    cell_velocity: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    num_atoms: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """NPH isotropic velocity half-step, single system (out-only).

    v_new = v + dt/2 * (F/m - (1 + 1/N_f) * ε̇ * v)

    For in-place: host passes same array as velocities and velocities_out.

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    v = velocities[atom_idx]
    m = masses[atom_idx]
    f = forces[atom_idx]
    h_dot = cell_velocity[0]

    V = volume[0]
    N_f = type(m)(3 * num_atoms[0])
    coupling = type(m)(1.0) + type(m)(1.0) / N_f
    dt_half = dt[0] * type(m)(0.5)
    accel = _npt_accel(f, m)
    drag = _drag_isotropic(h_dot, V, coupling, type(m)(0.0), v)

    velocities_out[atom_idx] = v + dt_half * (accel - drag)


@wp.kernel
def _nph_velocity_half_step_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_velocities: wp.array(dtype=Any),
    volumes: wp.array(dtype=Any),
    num_atoms_per_system: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """NPH isotropic velocity half-step, batched (out-only).

    For in-place: host passes same array as velocities and velocities_out.

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    v = velocities[atom_idx]
    m = masses[atom_idx]
    f = forces[atom_idx]
    h_dot = cell_velocities[sys_id]
    N = num_atoms_per_system[sys_id]

    V = volumes[sys_id]
    N_f = type(m)(3 * N)
    coupling = type(m)(1.0) + type(m)(1.0) / N_f
    dt_half = dt[sys_id] * type(m)(0.5)
    accel = _npt_accel(f, m)
    drag = _drag_isotropic(h_dot, V, coupling, type(m)(0.0), v)

    velocities_out[atom_idx] = v + dt_half * (accel - drag)


# ==============================================================================
# Position Update Kernels (shared by NPT and NPH)
# ==============================================================================


@wp.kernel
def _position_update_single_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    cell_inv: wp.array(dtype=Any),
    cell_velocity: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
):
    """Position update for single system (out-only).

    r_new = r + dt * v + dt * ε̇ @ r

    For in-place: host passes same array as positions and positions_out.

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    r = positions[atom_idx]
    v = velocities[atom_idx]
    h_inv = cell_inv[0]
    h_dot = cell_velocity[0]

    positions_out[atom_idx] = _npt_position_step(r, v, h_dot, h_inv, dt[0])


@wp.kernel
def _position_update_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cells: wp.array(dtype=Any),
    cells_inv: wp.array(dtype=Any),
    cell_velocities: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
):
    """Position update, batched (out-only).

    For in-place: host passes same array as positions and positions_out.

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    r = positions[atom_idx]
    v = velocities[atom_idx]
    h_inv = cells_inv[sys_id]
    h_dot = cell_velocities[sys_id]

    positions_out[atom_idx] = _npt_position_step(r, v, h_dot, h_inv, dt[sys_id])


# ==============================================================================
# Cell Velocity Update Kernels (Barostat)
# ==============================================================================


@wp.kernel
def _npt_cell_velocity_update_kernel(
    cell_velocities: wp.array(dtype=Any),
    pressure_tensors: wp.array(dtype=Any),
    target_pressures: wp.array(dtype=Any),
    volumes: wp.array(dtype=Any),
    cell_masses: wp.array(dtype=Any),
    kinetic_energies: wp.array(dtype=Any),
    num_atoms_per_system: wp.array(dtype=wp.int32),
    eta_dots: wp.array2d(dtype=Any),
    dt: wp.array(dtype=Any),
):
    """
    NPT isotropic cell velocity update with Nosé-Hoover thermostat coupling.

    Algorithm
    ---------
    Updates the cell velocity matrix for isotropic pressure control. The cell
    velocity evolves according to the Martyna-Tobias-Klein (MTK) equations:

        ḧ = (V/W) * (P_inst - P_ext) - η̇₁ * ḣ

    where:
    - P_inst = Tr(P_tensor)/3 + 2*KE/(3N*V) is the instantaneous isotropic pressure
    - P_ext is the target external pressure (scalar)
    - V is the cell volume, W is the barostat mass
    - η̇₁ is the first thermostat chain velocity (provides drag)

    For isotropic mode, only diagonal cell velocity components are updated,
    and they are all set equal (uniform scaling).

    Launch Grid
    -----------
    dim = [num_systems]
        One thread per system in the batch.

    Parameters
    ----------
    cell_velocities : wp.array(dtype=mat33f/mat33d)
        Cell velocity matrices ḣ. Shape (B,). MODIFIED in-place.
    pressure_tensors : wp.array(dtype=vec9f/vec9d)
        Pressure tensors from virial. Components [xx,xy,xz,yx,yy,yz,zx,zy,zz].
    target_pressures : wp.array(dtype=float32/float64)
        Target scalar pressures. Shape (B,).
    volumes : wp.array(dtype=float32/float64)
        Cell volumes V. Shape (B,).
    cell_masses : wp.array(dtype=float32/float64)
        Barostat masses W. Shape (B,).
    kinetic_energies : wp.array(dtype=float32/float64)
        System kinetic energies. Shape (B,).
    num_atoms_per_system : wp.array(dtype=wp.int32)
        Atom counts per system. Shape (B,).
    eta_dots : wp.array2d(dtype=float32/float64)
        Thermostat chain velocities. Shape (B, chain_length).
    dt : wp.array(dtype=float32/float64)
        Time step per system. Shape (B,).

    Notes
    -----
    - This kernel assumes isotropic pressure control (uniform cell scaling).
    - Off-diagonal cell velocity components are set to zero.
    - The kinetic correction 2*KE/(3N*V) accounts for ideal gas contribution.
    """
    sys_id = wp.tid()
    h_dot = cell_velocities[sys_id]
    P_ext = target_pressures[sys_id]
    V = volumes[sys_id]
    W = cell_masses[sys_id]
    KE = kinetic_energies[sys_id]
    N = num_atoms_per_system[sys_id]
    eta_dot_1 = eta_dots[sys_id, 0]

    # Current pressure from tensor
    P = pressure_tensors[sys_id]
    P_current = (P[0] + P[4] + P[8]) / type(V)(3.0)

    # Cast KE to match V's type
    KE_typed = type(V)(KE)

    # Degrees of freedom contribution
    N_f = type(V)(3 * N)
    dof_term = type(V)(2.0) * KE_typed / N_f

    # Pressure difference
    P_diff = P_current + dof_term / V - P_ext

    # Cell velocity acceleration (isotropic)
    h_dot_accel = V * P_diff / W

    # Apply thermostat drag
    zero = type(V)(0.0)
    dt_half = dt[sys_id] * type(V)(0.5)
    h_dot_new_diag = h_dot[0, 0] + dt_half * (h_dot_accel - eta_dot_1 * h_dot[0, 0])

    cell_velocities[sys_id] = type(h_dot)(
        h_dot_new_diag,
        zero,
        zero,
        zero,
        h_dot_new_diag,
        zero,
        zero,
        zero,
        h_dot_new_diag,
    )


@wp.kernel
def _nph_cell_velocity_update_kernel(
    cell_velocities: wp.array(dtype=Any),
    pressure_tensors: wp.array(dtype=Any),
    target_pressures: wp.array(dtype=Any),
    volumes: wp.array(dtype=Any),
    cell_masses: wp.array(dtype=Any),
    kinetic_energies: wp.array(dtype=Any),
    num_atoms_per_system: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
):
    """
    NPH isotropic cell velocity update (no thermostat coupling).

    Algorithm
    ---------
    Updates cell velocity for constant enthalpy (NPH) dynamics. Without
    thermostat coupling, the equation simplifies to:

        ḧ = (V/W) * (P_inst - P_ext)

    The system evolves on a constant-enthalpy surface with temperature
    fluctuating naturally.

    Launch Grid
    -----------
    dim = [num_systems]

    Parameters
    ----------
    cell_velocities : wp.array(dtype=mat33f/mat33d)
        Cell velocity matrices. MODIFIED in-place.
    pressure_tensors : wp.array(dtype=vec9f/vec9d)
        Pressure tensors from virial.
    target_pressures : wp.array(dtype=float32/float64)
        Target scalar pressures.
    volumes, cell_masses, kinetic_energies : wp.array
        System properties.
    num_atoms_per_system : wp.array(dtype=wp.int32)
        Atom counts per system.
    dt : wp.array(dtype=float32/float64)
        Time step per system. Shape (B,).

    Notes
    -----
    - No thermostat drag term (η̇₁ * ḣ) compared to NPT.
    - Temperature is not controlled; enthalpy H = KE + PE + PV is conserved.
    """
    sys_id = wp.tid()
    h_dot = cell_velocities[sys_id]
    P_ext = target_pressures[sys_id]
    V = volumes[sys_id]
    W = cell_masses[sys_id]
    KE = kinetic_energies[sys_id]
    N = num_atoms_per_system[sys_id]

    # Current pressure from tensor
    P = pressure_tensors[sys_id]
    P_current = (P[0] + P[4] + P[8]) / type(V)(3.0)

    # Cast KE to match V's type
    KE_typed = type(V)(KE)

    # Degrees of freedom contribution
    N_f = type(V)(3 * N)
    dof_term = type(V)(2.0) * KE_typed / N_f

    # Pressure difference
    P_diff = P_current + dof_term / V - P_ext

    # Cell velocity acceleration (isotropic, no thermostat drag)
    h_dot_accel = V * P_diff / W

    zero = type(V)(0.0)
    dt_half = dt[sys_id] * type(V)(0.5)
    h_dot_new_diag = h_dot[0, 0] + dt_half * h_dot_accel

    cell_velocities[sys_id] = type(h_dot)(
        h_dot_new_diag,
        zero,
        zero,
        zero,
        h_dot_new_diag,
        zero,
        zero,
        zero,
        h_dot_new_diag,
    )


# ==============================================================================
# Anisotropic Cell Velocity Update Kernels
# ==============================================================================


@wp.kernel
def _npt_cell_velocity_update_aniso_kernel(
    cell_velocities: wp.array(dtype=Any),
    pressure_tensors: wp.array(dtype=Any),
    target_pressures: wp.array(dtype=Any),  # vec3: [Pxx, Pyy, Pzz]
    volumes: wp.array(dtype=Any),
    cell_masses: wp.array(dtype=Any),
    kinetic_energies: wp.array(dtype=Any),
    num_atoms_per_system: wp.array(dtype=wp.int32),
    eta_dots: wp.array2d(dtype=Any),
    dt: wp.array(dtype=Any),
):
    """NPT cell velocity update - anisotropic/orthorhombic.

    Independent pressure control for x, y, z axes.
    Cell remains orthorhombic (diagonal h_dot).

    ḧ_ii = V/W * (P_ii - P_ext_ii) - η̇₁ * ḣ_ii

    Launch Grid: dim = [num_systems]
    """
    sys_id = wp.tid()
    h_dot = cell_velocities[sys_id]
    P_ext = target_pressures[sys_id]  # vec3: [Pxx, Pyy, Pzz]
    V = volumes[sys_id]
    W = cell_masses[sys_id]
    KE = kinetic_energies[sys_id]
    N = num_atoms_per_system[sys_id]
    eta_dot_1 = eta_dots[sys_id, 0]

    # Current pressure tensor components
    P = pressure_tensors[sys_id]
    P_xx = P[0]
    P_yy = P[4]
    P_zz = P[8]

    KE_typed = type(V)(KE)
    N_f = type(V)(3 * N)
    dof_term = type(V)(2.0) * KE_typed / N_f
    dof_V = dof_term / V

    # Pressure differences for each axis
    P_diff_x = P_xx + dof_V - P_ext[0]
    P_diff_y = P_yy + dof_V - P_ext[1]
    P_diff_z = P_zz + dof_V - P_ext[2]

    # Cell velocity accelerations (independent per axis)
    h_dot_accel_x = V * P_diff_x / W
    h_dot_accel_y = V * P_diff_y / W
    h_dot_accel_z = V * P_diff_z / W

    # Apply thermostat drag per axis
    zero = type(V)(0.0)
    dt_half = dt[sys_id] * type(V)(0.5)
    h_dot_new_xx = h_dot[0, 0] + dt_half * (h_dot_accel_x - eta_dot_1 * h_dot[0, 0])
    h_dot_new_yy = h_dot[1, 1] + dt_half * (h_dot_accel_y - eta_dot_1 * h_dot[1, 1])
    h_dot_new_zz = h_dot[2, 2] + dt_half * (h_dot_accel_z - eta_dot_1 * h_dot[2, 2])

    cell_velocities[sys_id] = type(h_dot)(
        h_dot_new_xx, zero, zero, zero, h_dot_new_yy, zero, zero, zero, h_dot_new_zz
    )


@wp.kernel
def _nph_cell_velocity_update_aniso_kernel(
    cell_velocities: wp.array(dtype=Any),
    pressure_tensors: wp.array(dtype=Any),
    target_pressures: wp.array(dtype=Any),  # vec3: [Pxx, Pyy, Pzz]
    volumes: wp.array(dtype=Any),
    cell_masses: wp.array(dtype=Any),
    kinetic_energies: wp.array(dtype=Any),
    num_atoms_per_system: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
):
    """NPH cell velocity update - anisotropic/orthorhombic (no thermostat).

    ḧ_ii = V/W * (P_ii - P_ext_ii)

    Launch Grid: dim = [num_systems]
    """
    sys_id = wp.tid()
    h_dot = cell_velocities[sys_id]
    P_ext = target_pressures[sys_id]
    V = volumes[sys_id]
    W = cell_masses[sys_id]
    KE = kinetic_energies[sys_id]
    N = num_atoms_per_system[sys_id]

    P = pressure_tensors[sys_id]
    P_xx = P[0]
    P_yy = P[4]
    P_zz = P[8]

    KE_typed = type(V)(KE)
    N_f = type(V)(3 * N)
    dof_term = type(V)(2.0) * KE_typed / N_f
    dof_V = dof_term / V

    P_diff_x = P_xx + dof_V - P_ext[0]
    P_diff_y = P_yy + dof_V - P_ext[1]
    P_diff_z = P_zz + dof_V - P_ext[2]

    h_dot_accel_x = V * P_diff_x / W
    h_dot_accel_y = V * P_diff_y / W
    h_dot_accel_z = V * P_diff_z / W

    zero = type(V)(0.0)
    dt_half = dt[sys_id] * type(V)(0.5)
    h_dot_new_xx = h_dot[0, 0] + dt_half * h_dot_accel_x
    h_dot_new_yy = h_dot[1, 1] + dt_half * h_dot_accel_y
    h_dot_new_zz = h_dot[2, 2] + dt_half * h_dot_accel_z

    cell_velocities[sys_id] = type(h_dot)(
        h_dot_new_xx, zero, zero, zero, h_dot_new_yy, zero, zero, zero, h_dot_new_zz
    )


# ==============================================================================
# Triclinic Cell Velocity Update Kernels (Full Stress Tensor Control)
# ==============================================================================


@wp.kernel
def _npt_cell_velocity_update_triclinic_kernel(
    cell_velocities: wp.array(dtype=Any),
    pressure_tensors: wp.array(dtype=Any),
    target_pressures: wp.array(dtype=Any),  # vec9: full stress tensor
    volumes: wp.array(dtype=Any),
    cell_masses: wp.array(dtype=Any),
    kinetic_energies: wp.array(dtype=Any),
    num_atoms_per_system: wp.array(dtype=wp.int32),
    eta_dots: wp.array2d(dtype=Any),
    dt: wp.array(dtype=Any),
):
    """NPT cell velocity update - full triclinic.

    Full stress tensor control (all 9 components).
    ḧ_ij = V/W * (P_ij - P_ext_ij) - η̇₁ * ḣ_ij

    Launch Grid: dim = [num_systems]
    """
    sys_id = wp.tid()
    h_dot = cell_velocities[sys_id]
    P_ext = target_pressures[sys_id]  # vec9
    V = volumes[sys_id]
    W = cell_masses[sys_id]
    KE = kinetic_energies[sys_id]
    N = num_atoms_per_system[sys_id]
    eta_dot_1 = eta_dots[sys_id, 0]

    P = pressure_tensors[sys_id]

    KE_typed = type(V)(KE)
    N_f = type(V)(3 * N)
    dof_term = type(V)(2.0) * KE_typed / N_f
    dof_V = dof_term / V

    # For triclinic, dof correction only applies to diagonal
    # P_diff for each component
    P_diff_00 = P[0] + dof_V - P_ext[0]
    P_diff_01 = P[1] - P_ext[1]
    P_diff_02 = P[2] - P_ext[2]
    P_diff_10 = P[3] - P_ext[3]
    P_diff_11 = P[4] + dof_V - P_ext[4]
    P_diff_12 = P[5] - P_ext[5]
    P_diff_20 = P[6] - P_ext[6]
    P_diff_21 = P[7] - P_ext[7]
    P_diff_22 = P[8] + dof_V - P_ext[8]

    # Cell velocity accelerations
    V_W = V / W
    dt_half = dt[sys_id] * type(V)(0.5)
    h_dot_new_00 = h_dot[0, 0] + dt_half * (V_W * P_diff_00 - eta_dot_1 * h_dot[0, 0])
    h_dot_new_01 = h_dot[0, 1] + dt_half * (V_W * P_diff_01 - eta_dot_1 * h_dot[0, 1])
    h_dot_new_02 = h_dot[0, 2] + dt_half * (V_W * P_diff_02 - eta_dot_1 * h_dot[0, 2])
    h_dot_new_10 = h_dot[1, 0] + dt_half * (V_W * P_diff_10 - eta_dot_1 * h_dot[1, 0])
    h_dot_new_11 = h_dot[1, 1] + dt_half * (V_W * P_diff_11 - eta_dot_1 * h_dot[1, 1])
    h_dot_new_12 = h_dot[1, 2] + dt_half * (V_W * P_diff_12 - eta_dot_1 * h_dot[1, 2])
    h_dot_new_20 = h_dot[2, 0] + dt_half * (V_W * P_diff_20 - eta_dot_1 * h_dot[2, 0])
    h_dot_new_21 = h_dot[2, 1] + dt_half * (V_W * P_diff_21 - eta_dot_1 * h_dot[2, 1])
    h_dot_new_22 = h_dot[2, 2] + dt_half * (V_W * P_diff_22 - eta_dot_1 * h_dot[2, 2])

    cell_velocities[sys_id] = type(h_dot)(
        h_dot_new_00,
        h_dot_new_01,
        h_dot_new_02,
        h_dot_new_10,
        h_dot_new_11,
        h_dot_new_12,
        h_dot_new_20,
        h_dot_new_21,
        h_dot_new_22,
    )


@wp.kernel
def _nph_cell_velocity_update_triclinic_kernel(
    cell_velocities: wp.array(dtype=Any),
    pressure_tensors: wp.array(dtype=Any),
    target_pressures: wp.array(dtype=Any),  # vec9: full stress tensor
    volumes: wp.array(dtype=Any),
    cell_masses: wp.array(dtype=Any),
    kinetic_energies: wp.array(dtype=Any),
    num_atoms_per_system: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
):
    """NPH cell velocity update - full triclinic (no thermostat).

    ḧ_ij = V/W * (P_ij - P_ext_ij)

    Launch Grid: dim = [num_systems]
    """
    sys_id = wp.tid()
    h_dot = cell_velocities[sys_id]
    P_ext = target_pressures[sys_id]
    V = volumes[sys_id]
    W = cell_masses[sys_id]
    KE = kinetic_energies[sys_id]
    N = num_atoms_per_system[sys_id]

    P = pressure_tensors[sys_id]

    KE_typed = type(V)(KE)
    N_f = type(V)(3 * N)
    dof_term = type(V)(2.0) * KE_typed / N_f
    dof_V = dof_term / V

    P_diff_00 = P[0] + dof_V - P_ext[0]
    P_diff_01 = P[1] - P_ext[1]
    P_diff_02 = P[2] - P_ext[2]
    P_diff_10 = P[3] - P_ext[3]
    P_diff_11 = P[4] + dof_V - P_ext[4]
    P_diff_12 = P[5] - P_ext[5]
    P_diff_20 = P[6] - P_ext[6]
    P_diff_21 = P[7] - P_ext[7]
    P_diff_22 = P[8] + dof_V - P_ext[8]

    V_W = V / W
    dt_half = dt[sys_id] * type(V)(0.5)
    h_dot_new_00 = h_dot[0, 0] + dt_half * V_W * P_diff_00
    h_dot_new_01 = h_dot[0, 1] + dt_half * V_W * P_diff_01
    h_dot_new_02 = h_dot[0, 2] + dt_half * V_W * P_diff_02
    h_dot_new_10 = h_dot[1, 0] + dt_half * V_W * P_diff_10
    h_dot_new_11 = h_dot[1, 1] + dt_half * V_W * P_diff_11
    h_dot_new_12 = h_dot[1, 2] + dt_half * V_W * P_diff_12
    h_dot_new_20 = h_dot[2, 0] + dt_half * V_W * P_diff_20
    h_dot_new_21 = h_dot[2, 1] + dt_half * V_W * P_diff_21
    h_dot_new_22 = h_dot[2, 2] + dt_half * V_W * P_diff_22

    cell_velocities[sys_id] = type(h_dot)(
        h_dot_new_00,
        h_dot_new_01,
        h_dot_new_02,
        h_dot_new_10,
        h_dot_new_11,
        h_dot_new_12,
        h_dot_new_20,
        h_dot_new_21,
        h_dot_new_22,
    )


# ==============================================================================
# Anisotropic Velocity Update Kernels
# ==============================================================================


@wp.kernel
def _npt_velocity_half_step_aniso_single_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    cell_velocity: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    eta_dot: wp.array2d(dtype=Any),
    num_atoms: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """NPT anisotropic velocity half-step, single system (out-only).

    For in-place: host passes same array as velocities and velocities_out.

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    v = velocities[atom_idx]
    m = masses[atom_idx]
    f = forces[atom_idx]
    h_dot = cell_velocity[0]
    eta_dot_1 = eta_dot[0, 0]
    V = volume[0]

    N_f = type(m)(3 * num_atoms[0])
    coupling = type(m)(1.0) + type(m)(1.0) / N_f
    dt_half = dt[0] * type(m)(0.5)
    accel = _npt_accel(f, m)
    drag = _drag_anisotropic(h_dot, V, coupling, eta_dot_1, v)

    velocities_out[atom_idx] = v + dt_half * (accel - drag)


@wp.kernel
def _npt_velocity_half_step_aniso_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_velocities: wp.array(dtype=Any),
    volumes: wp.array(dtype=Any),
    eta_dots: wp.array2d(dtype=Any),
    num_atoms_per_system: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """NPT anisotropic velocity half-step, batched (out-only).

    For in-place: host passes same array as velocities and velocities_out.

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    v = velocities[atom_idx]
    m = masses[atom_idx]
    f = forces[atom_idx]
    h_dot = cell_velocities[sys_id]
    eta_dot_1 = eta_dots[sys_id, 0]
    N = num_atoms_per_system[sys_id]
    V = volumes[sys_id]

    N_f = type(m)(3 * N)
    coupling = type(m)(1.0) + type(m)(1.0) / N_f
    dt_half = dt[sys_id] * type(m)(0.5)
    accel = _npt_accel(f, m)
    drag = _drag_anisotropic(h_dot, V, coupling, eta_dot_1, v)

    velocities_out[atom_idx] = v + dt_half * (accel - drag)


# =============================================================================
# Triclinic NPT Velocity Kernels
# =============================================================================


@wp.kernel
def _npt_velocity_half_step_triclinic_single_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    cell_velocity: wp.array(dtype=Any),
    cell_inv: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    eta_dot: wp.array2d(dtype=Any),
    num_atoms: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """NPT triclinic velocity half-step, single system (out-only).

    For in-place: host passes same array as velocities and velocities_out.

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    v = velocities[atom_idx]
    m = masses[atom_idx]
    f = forces[atom_idx]
    h_dot = cell_velocity[0]
    h_inv = cell_inv[0]
    eta_dot_1 = eta_dot[0, 0]

    N_f = type(m)(3 * num_atoms[0])
    coupling = type(m)(1.0) + type(m)(1.0) / N_f
    dt_half = dt[0] * type(m)(0.5)
    accel = _npt_accel(f, m)
    drag = _drag_triclinic(h_dot, h_inv, coupling, eta_dot_1, v)

    velocities_out[atom_idx] = v + dt_half * (accel - drag)


@wp.kernel
def _npt_velocity_half_step_triclinic_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_velocities: wp.array(dtype=Any),
    cells_inv: wp.array(dtype=Any),
    volumes: wp.array(dtype=Any),
    eta_dots: wp.array2d(dtype=Any),
    num_atoms_per_system: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """NPT triclinic velocity half-step, batched (out-only).

    For in-place: host passes same array as velocities and velocities_out.

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    v = velocities[atom_idx]
    m = masses[atom_idx]
    f = forces[atom_idx]
    h_dot = cell_velocities[sys_id]
    h_inv = cells_inv[sys_id]
    eta_dot_1 = eta_dots[sys_id, 0]
    N = num_atoms_per_system[sys_id]

    N_f = type(m)(3 * N)
    coupling = type(m)(1.0) + type(m)(1.0) / N_f
    dt_half = dt[sys_id] * type(m)(0.5)
    accel = _npt_accel(f, m)
    drag = _drag_triclinic(h_dot, h_inv, coupling, eta_dot_1, v)

    velocities_out[atom_idx] = v + dt_half * (accel - drag)


# =============================================================================
# Triclinic NPH Velocity Kernels
# =============================================================================


@wp.kernel
def _nph_velocity_half_step_triclinic_single_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    cell_velocity: wp.array(dtype=Any),
    cell_inv: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    num_atoms: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """NPH triclinic velocity half-step, single system (out-only, no thermostat).

    For in-place: host passes same array as velocities and velocities_out.

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    v = velocities[atom_idx]
    m = masses[atom_idx]
    f = forces[atom_idx]
    h_dot = cell_velocity[0]
    h_inv = cell_inv[0]

    N_f = type(m)(3 * num_atoms[0])
    coupling = type(m)(1.0) + type(m)(1.0) / N_f
    dt_half = dt[0] * type(m)(0.5)
    accel = _npt_accel(f, m)
    drag = _drag_triclinic(h_dot, h_inv, coupling, type(m)(0.0), v)

    velocities_out[atom_idx] = v + dt_half * (accel - drag)


@wp.kernel
def _nph_velocity_half_step_triclinic_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_velocities: wp.array(dtype=Any),
    cells_inv: wp.array(dtype=Any),
    volumes: wp.array(dtype=Any),
    num_atoms_per_system: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """NPH triclinic velocity half-step, batched (out-only, no thermostat).

    For in-place: host passes same array as velocities and velocities_out.

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    v = velocities[atom_idx]
    m = masses[atom_idx]
    f = forces[atom_idx]
    h_dot = cell_velocities[sys_id]
    h_inv = cells_inv[sys_id]
    N = num_atoms_per_system[sys_id]

    N_f = type(m)(3 * N)
    coupling = type(m)(1.0) + type(m)(1.0) / N_f
    dt_half = dt[sys_id] * type(m)(0.5)
    accel = _npt_accel(f, m)
    drag = _drag_triclinic(h_dot, h_inv, coupling, type(m)(0.0), v)

    velocities_out[atom_idx] = v + dt_half * (accel - drag)


@wp.kernel
def _nph_velocity_half_step_aniso_single_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    cell_velocity: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    num_atoms: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """NPH anisotropic velocity half-step, single system (out-only, no thermostat).

    For in-place: host passes same array as velocities and velocities_out.

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    v = velocities[atom_idx]
    m = masses[atom_idx]
    f = forces[atom_idx]
    h_dot = cell_velocity[0]
    V = volume[0]

    N_f = type(m)(3 * num_atoms[0])
    coupling = type(m)(1.0) + type(m)(1.0) / N_f
    dt_half = dt[0] * type(m)(0.5)
    accel = _npt_accel(f, m)
    drag = _drag_anisotropic(h_dot, V, coupling, type(m)(0.0), v)

    velocities_out[atom_idx] = v + dt_half * (accel - drag)


@wp.kernel
def _nph_velocity_half_step_aniso_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_velocities: wp.array(dtype=Any),
    volumes: wp.array(dtype=Any),
    num_atoms_per_system: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """NPH anisotropic velocity half-step, batched (out-only, no thermostat).

    For in-place: host passes same array as velocities and velocities_out.

    Launch Grid: dim = [num_atoms]
    """
    atom_idx = wp.tid()
    sys_id = batch_idx[atom_idx]
    v = velocities[atom_idx]
    m = masses[atom_idx]
    f = forces[atom_idx]
    h_dot = cell_velocities[sys_id]
    N = num_atoms_per_system[sys_id]
    V = volumes[sys_id]

    N_f = type(m)(3 * N)
    coupling = type(m)(1.0) + type(m)(1.0) / N_f
    dt_half = dt[sys_id] * type(m)(0.5)
    accel = _npt_accel(f, m)
    drag = _drag_anisotropic(h_dot, V, coupling, type(m)(0.0), v)

    velocities_out[atom_idx] = v + dt_half * (accel - drag)


# ==============================================================================
# Cell Position Update Kernels
# ==============================================================================


@wp.kernel
def _cell_update_kernel(
    cells: wp.array(dtype=Any),
    cell_velocities: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    cells_out: wp.array(dtype=Any),
):
    """Update cell matrix: h_new = h + dt * ḣ (out-only).

    For in-place: host passes same array as cells and cells_out.

    Launch Grid: dim = [num_systems]
    """
    sys_id = wp.tid()
    h = cells[sys_id]
    h_dot = cell_velocities[sys_id]
    cells_out[sys_id] = h + dt[sys_id] * h_dot


# ==============================================================================
# NPT Thermostat Kernel
# ==============================================================================


@wp.kernel
def _npt_thermostat_chain_update_kernel(
    eta: wp.array2d(dtype=Any),
    eta_dot: wp.array2d(dtype=Any),
    kinetic_energies: wp.array(dtype=Any),
    target_temps: wp.array(dtype=Any),
    thermostat_masses: wp.array2d(dtype=Any),
    num_atoms_per_system: wp.array(dtype=wp.int32),
    chain_length: wp.int32,
    dt_chain: wp.array(dtype=Any),
):
    """Update Nosé-Hoover chain for NPT thermostat.

    Launch Grid: dim = [num_systems]
    """
    sys_id = wp.tid()
    KE = kinetic_energies[sys_id]
    T_target = target_temps[sys_id]
    N = num_atoms_per_system[sys_id]

    # Use eta_dot dtype as reference
    eta_dot_0 = eta_dot[sys_id, 0]

    N_f = type(eta_dot_0)(3 * N)
    kT = type(eta_dot_0)(T_target)
    KE_typed = type(eta_dot_0)(KE)

    # First thermostat driven by kinetic energy difference
    G1 = (type(eta_dot_0)(2.0) * KE_typed - N_f * kT) / thermostat_masses[sys_id, 0]

    dt_chain_sys = dt_chain[sys_id]
    dt_half = dt_chain_sys * type(eta_dot_0)(0.5)
    dt_quarter = dt_chain_sys * type(eta_dot_0)(0.25)

    # Last thermostat
    M_last = chain_length - 1
    eta_dot[sys_id, M_last] = (
        eta_dot[sys_id, M_last]
        + dt_quarter
        * (
            thermostat_masses[sys_id, M_last - 1]
            * eta_dot[sys_id, M_last - 1]
            * eta_dot[sys_id, M_last - 1]
            - kT
        )
        / thermostat_masses[sys_id, M_last]
    )

    # Middle thermostats (reverse order)
    for k in range(M_last - 1, 0, -1):
        eta_dot_k = eta_dot[sys_id, k]
        G_k = (
            thermostat_masses[sys_id, k - 1]
            * eta_dot[sys_id, k - 1]
            * eta_dot[sys_id, k - 1]
            - kT
        ) / thermostat_masses[sys_id, k]
        exp_factor = wp.exp(-dt_quarter * eta_dot[sys_id, k + 1])
        eta_dot[sys_id, k] = eta_dot_k * exp_factor + dt_quarter * G_k

    # First thermostat
    exp_factor = wp.exp(-dt_quarter * eta_dot[sys_id, 1])
    eta_dot[sys_id, 0] = eta_dot[sys_id, 0] * exp_factor + dt_quarter * G1

    # Update positions
    for k in range(chain_length):
        eta[sys_id, k] = eta[sys_id, k] + dt_half * eta_dot[sys_id, k]

    # Second half of velocity updates
    G1_new = (type(eta_dot_0)(2.0) * KE_typed - N_f * kT) / thermostat_masses[sys_id, 0]
    exp_factor = wp.exp(-dt_quarter * eta_dot[sys_id, 1])
    eta_dot[sys_id, 0] = eta_dot[sys_id, 0] * exp_factor + dt_quarter * G1_new

    for k in range(1, M_last):
        G_k = (
            thermostat_masses[sys_id, k - 1]
            * eta_dot[sys_id, k - 1]
            * eta_dot[sys_id, k - 1]
            - kT
        ) / thermostat_masses[sys_id, k]
        exp_factor = wp.exp(-dt_quarter * eta_dot[sys_id, k + 1])
        eta_dot[sys_id, k] = eta_dot[sys_id, k] * exp_factor + dt_quarter * G_k

    # Last thermostat
    G_last = (
        thermostat_masses[sys_id, M_last - 1]
        * eta_dot[sys_id, M_last - 1]
        * eta_dot[sys_id, M_last - 1]
        - kT
    ) / thermostat_masses[sys_id, M_last]
    eta_dot[sys_id, M_last] = eta_dot[sys_id, M_last] + dt_quarter * G_last


# ==============================================================================
# Functional Interfaces - Pressure Calculations
# ==============================================================================


def compute_pressure_tensor(
    velocities: wp.array,
    masses: wp.array,
    virial_tensors: wp.array,
    cells: wp.array,
    kinetic_tensors: wp.array,
    pressure_tensors: wp.array,
    volumes: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Compute full pressure tensor.

    P = (kinetic + virial) / V

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Particle velocities. Shape (N,).
    masses : wp.array
        Particle masses. Shape (N,).
    virial_tensors : wp.array(dtype=vec9f or vec9d)
        Virial tensor from forces. Shape (B,).
    cells : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell matrices. Shape (B,).
    kinetic_tensors : wp.array(dtype=scalar, ndim=2)
        Scratch array for kinetic tensor accumulation. Shape (B, 9).
        Zeroed internally before each use.
    pressure_tensors : wp.array(dtype=vec9f or vec9d)
        Output pressure tensor. Shape (B,).
    volumes : wp.array(dtype=scalar)
        Pre-computed cell volumes. Shape (B,).
        Caller must pre-compute via compute_cell_volume.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. If None, assumes single system.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array(dtype=vec9f or vec9d)
        Pressure tensor. Shape (B,).
    """
    if device is None:
        device = velocities.device

    kinetic_tensors.zero_()
    num_atoms = velocities.shape[0]
    num_systems = cells.shape[0]

    if batch_idx is None:
        kernel = _KINETIC_TENSOR_FAMILY.single
        inputs = [velocities, masses, kinetic_tensors]
    else:
        kernel = _KINETIC_TENSOR_FAMILY.batch_idx
        inputs = [velocities, masses, batch_idx, kinetic_tensors]
    wp.launch(kernel, dim=num_atoms, inputs=inputs, device=device, block_dim=TILE_DIM)

    wp.launch(
        _finalize_pressure_tensor_kernel,
        dim=num_systems,
        inputs=[kinetic_tensors, virial_tensors, volumes, pressure_tensors],
        device=device,
    )

    return pressure_tensors


def compute_scalar_pressure(
    pressure_tensors: wp.array,
    scalar_pressures: wp.array,
    device: str = None,
) -> wp.array:
    """
    Compute scalar pressure from pressure tensor.

    P_scalar = (P_xx + P_yy + P_zz) / 3

    Parameters
    ----------
    pressure_tensors : wp.array(dtype=vec9f or vec9d)
        Pressure tensor. Shape (B,).
    scalar_pressures : wp.array(dtype=scalar)
        Output scalar pressure. Shape (B,).
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Scalar pressure. Shape (B,).
    """
    if device is None:
        device = pressure_tensors.device

    num_systems = pressure_tensors.shape[0]

    wp.launch(
        _compute_scalar_pressure_kernel,
        dim=num_systems,
        inputs=[pressure_tensors, scalar_pressures],
        device=device,
    )

    return scalar_pressures


# ==============================================================================
# Array Broadcast Kernels (Pure Warp - no numpy)
# ==============================================================================


@wp.kernel
def _broadcast_scalar_f32_kernel(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
):
    """Broadcast scalar from src[0] to all elements of dst.

    Launch Grid: dim = [dst.shape[0]]
    """
    idx = wp.tid()
    dst[idx] = src[0]


@wp.kernel
def _broadcast_scalar_f64_kernel(
    src: wp.array(dtype=wp.float64),
    dst: wp.array(dtype=wp.float64),
):
    """Broadcast scalar from src[0] to all elements of dst.

    Launch Grid: dim = [dst.shape[0]]
    """
    idx = wp.tid()
    dst[idx] = src[0]


@wp.kernel
def _broadcast_scalar_i32_kernel(
    src: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.int32),
):
    """Broadcast scalar from src[0] to all elements of dst.

    Launch Grid: dim = [dst.shape[0]]
    """
    idx = wp.tid()
    dst[idx] = src[0]


# ==============================================================================
# Barostat Mass Kernel
# ==============================================================================


@wp.kernel
def _compute_barostat_mass_kernel(
    temperatures: wp.array(dtype=Any),
    tau_p: wp.array(dtype=Any),
    num_atoms: wp.array(dtype=wp.int32),
    masses_out: wp.array(dtype=Any),
):
    """
    Compute barostat masses for NPT/NPH simulations.

    Algorithm
    ---------
    The barostat mass is computed from the MTK equations:

        W = (N_f + d) * kT * τ_p²

    where N_f = 3N is degrees of freedom, d = 3 is dimensionality.

    Launch Grid
    -----------
    dim = [num_systems]

    Parameters
    ----------
    temperatures : wp.array(dtype=float32/float64)
        Target temperatures per system. Shape (B,).
    tau_p : wp.array(dtype=float32/float64)
        Pressure relaxation times per system. Shape (B,).
    num_atoms : wp.array(dtype=wp.int32)
        Atom counts per system. Shape (B,).
    masses_out : wp.array(dtype=float32/float64)
        Output barostat masses. Shape (B,).

    Notes
    -----
    - Larger W = slower pressure equilibration (more stable)
    - Smaller W = faster equilibration (may cause oscillations)
    """
    sys_id = wp.tid()
    T = temperatures[sys_id]
    tau = tau_p[sys_id]
    N = num_atoms[sys_id]

    # N_f = 3N degrees of freedom, d = 3 dimensionality
    N_f = type(T)(3 * N)
    d = type(T)(3.0)

    # W = (N_f + d) * kT * τ_p²
    masses_out[sys_id] = (N_f + d) * T * tau * tau


# ==============================================================================
# Kernel Family Tables
# ==============================================================================

# -- Kinetic tensor: single vs batched (tiled kernels, needs block_dim) ------
_KINETIC_TENSOR_FAMILY = KernelFamily(
    single=_compute_kinetic_tensor_single_tiled_kernel,
    batch_idx=_compute_kinetic_tensor_tiled_kernel,
)

# -- Position update: single vs batched (shared by NPT and NPH) -------------
_POSITION_UPDATE_FAMILY = KernelFamily(
    single=_position_update_single_kernel,
    batch_idx=_position_update_kernel,
)

# -- NPT velocity half-step: mode -> KernelFamily (single/batch) ------------
_NPT_VELOCITY_FAMILIES = {
    "isotropic": KernelFamily(
        single=_npt_velocity_half_step_single_kernel,
        batch_idx=_npt_velocity_half_step_kernel,
    ),
    "anisotropic": KernelFamily(
        single=_npt_velocity_half_step_aniso_single_kernel,
        batch_idx=_npt_velocity_half_step_aniso_kernel,
    ),
    "triclinic": KernelFamily(
        single=_npt_velocity_half_step_triclinic_single_kernel,
        batch_idx=_npt_velocity_half_step_triclinic_kernel,
    ),
}

# -- NPH velocity half-step: mode -> KernelFamily (single/batch) ------------
_NPH_VELOCITY_FAMILIES = {
    "isotropic": KernelFamily(
        single=_nph_velocity_half_step_single_kernel,
        batch_idx=_nph_velocity_half_step_kernel,
    ),
    "anisotropic": KernelFamily(
        single=_nph_velocity_half_step_aniso_single_kernel,
        batch_idx=_nph_velocity_half_step_aniso_kernel,
    ),
    "triclinic": KernelFamily(
        single=_nph_velocity_half_step_triclinic_single_kernel,
        batch_idx=_nph_velocity_half_step_triclinic_kernel,
    ),
}

# -- Barostat kernels: target_pressures.dtype -> kernel ----------------------
_NPT_BAROSTAT_KERNELS = {
    wp.float32: _npt_cell_velocity_update_kernel,
    wp.float64: _npt_cell_velocity_update_kernel,
    wp.vec3f: _npt_cell_velocity_update_aniso_kernel,
    wp.vec3d: _npt_cell_velocity_update_aniso_kernel,
    vec9f: _npt_cell_velocity_update_triclinic_kernel,
    vec9d: _npt_cell_velocity_update_triclinic_kernel,
}

_NPH_BAROSTAT_KERNELS = {
    wp.float32: _nph_cell_velocity_update_kernel,
    wp.float64: _nph_cell_velocity_update_kernel,
    wp.vec3f: _nph_cell_velocity_update_aniso_kernel,
    wp.vec3d: _nph_cell_velocity_update_aniso_kernel,
    vec9f: _nph_cell_velocity_update_triclinic_kernel,
    vec9d: _nph_cell_velocity_update_triclinic_kernel,
}

# ==============================================================================
# Functional Interfaces - Barostat Utilities
# ==============================================================================


def compute_barostat_mass(
    target_temperature: wp.array,
    tau_p: wp.array,
    num_atoms: wp.array,
    masses_out: wp.array,
    device: str = None,
) -> wp.array:
    """
    Compute barostat mass(es) for desired pressure fluctuation timescale.

    The barostat mass determines the inertia of cell volume/shape fluctuations
    in NPT/NPH simulations. It is computed from the Martyna-Tobias-Klein
    equations to give a characteristic pressure relaxation time τ_p:

    .. math::

        W = (N_f + d) \\cdot k_B T \\cdot \\tau_p^2

    where:
    - N_f = 3N is the number of degrees of freedom (N atoms in 3D)
    - d = 3 is the dimensionality (for 3D simulations)
    - k_B T is the thermal energy (in reduced units, k_B = 1)
    - τ_p is the pressure relaxation time

    Larger W gives slower pressure equilibration (more stable but slower to
    reach target pressure). Smaller W gives faster equilibration but may
    cause oscillations.

    Parameters
    ----------
    target_temperature : wp.array
        Per-system target temperatures. Shape (num_systems,).
        Caller must pre-broadcast if a single value applies to all systems.
    tau_p : wp.array
        Per-system pressure relaxation times. Shape (num_systems,).
        Typical values: 0.5-2.0 ps.
        Caller must pre-broadcast if a single value applies to all systems.
    num_atoms : wp.array(dtype=wp.int32)
        Per-system atom counts. Shape (num_systems,).
        Caller must pre-broadcast if a single value applies to all systems.
    masses_out : wp.array
        Output barostat masses W. Shape (num_systems,).
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Barostat mass(es) W. Shape (num_systems,).

    Examples
    --------
    Batched systems (using wp.arrays):

    >>> temps = wp.array([1.0, 2.0], dtype=wp.float64, device="cuda:0")
    >>> tau = wp.array([1.0, 1.0], dtype=wp.float64, device="cuda:0")
    >>> n_atoms = wp.array([100, 200], dtype=wp.int32, device="cuda:0")
    >>> W = wp.empty(2, dtype=wp.float64, device="cuda:0")
    >>> compute_barostat_mass(temps, tau, n_atoms, W)
    >>> print(W.numpy())  # [303.0, 1206.0]

    Notes
    -----
    - The formula assumes k_B = 1 (reduced units). Scale τ_p accordingly
      for real units.
    - For isotropic barostat, a single mass controls all cell dimensions.
    - For anisotropic/triclinic barostat, the same mass is typically used
      for all cell velocity components.
    - All input arrays must have the same length (num_systems). The caller
      is responsible for broadcasting scalar values to arrays before calling.

    References
    ----------
    .. [MTK1994] Martyna, Tobias, Klein, J. Chem. Phys. 101, 4177 (1994)
    .. [SSM2004] Shinoda, Shiga, Mikami, Phys. Rev. B 69, 134103 (2004)
    """
    if device is None:
        device = target_temperature.device

    num_systems = target_temperature.shape[0]

    # Launch kernel
    wp.launch(
        _compute_barostat_mass_kernel,
        dim=num_systems,
        inputs=[target_temperature, tau_p, num_atoms, masses_out],
        device=device,
    )

    return masses_out


def compute_cell_kinetic_energy(
    cell_velocities: wp.array,
    cell_masses: wp.array,
    kinetic_energy: wp.array,
    device: str = None,
) -> wp.array:
    """
    Compute kinetic energy of cell degrees of freedom.

    KE_cell = 0.5 * W * ||ḣ||²_F

    Parameters
    ----------
    cell_velocities : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell velocity matrices. Shape (B,).
    cell_masses : wp.array
        Barostat masses. Shape (B,).
    kinetic_energy : wp.array(dtype=scalar)
        Output cell kinetic energy. Shape (B,).
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Cell kinetic energy. Shape (B,).
    """
    if device is None:
        device = cell_velocities.device

    num_systems = cell_velocities.shape[0]

    wp.launch(
        _compute_cell_kinetic_energy_kernel,
        dim=num_systems,
        inputs=[cell_velocities, cell_masses, kinetic_energy],
        device=device,
    )

    return kinetic_energy


def compute_barostat_potential_energy(
    target_pressures: wp.array,
    volumes: wp.array,
    potential_energy: wp.array,
    device: str = None,
) -> wp.array:
    """
    Compute barostat potential energy: U = P_ext * V.

    Parameters
    ----------
    target_pressures : wp.array
        External/target pressures. Shape (B,).
    volumes : wp.array
        Cell volumes. Shape (B,).
    potential_energy : wp.array
        Output barostat potential energy. Shape (B,).
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Barostat potential energy. Shape (B,).
    """
    if device is None:
        device = target_pressures.device

    num_systems = target_pressures.shape[0]

    wp.launch(
        _compute_barostat_potential_kernel,
        dim=num_systems,
        inputs=[target_pressures, volumes, potential_energy],
        device=device,
    )

    return potential_energy


# ==============================================================================
# Functional Interfaces - NPT Integration
# ==============================================================================


def npt_thermostat_half_step(
    eta: wp.array,
    eta_dot: wp.array,
    kinetic_energy: wp.array,
    target_temperature: wp.array,
    thermostat_masses: wp.array,
    num_atoms_per_system: wp.array,
    chain_length: int,
    dt: wp.array,
    device: str = None,
) -> None:
    """
    Perform thermostat half-step for NPT.

    Parameters
    ----------
    eta : wp.array2d
        Thermostat positions. Shape (B, chain_length). MODIFIED in-place.
    eta_dot : wp.array2d
        Thermostat velocities. Shape (B, chain_length). MODIFIED in-place.
    kinetic_energy : wp.array
        Kinetic energy per system. Shape (B,).
    target_temperature : wp.array
        Target temperatures. Shape (B,).
    thermostat_masses : wp.array2d
        Thermostat masses. Shape (B, chain_length).
    num_atoms_per_system : wp.array(dtype=wp.int32)
        Number of atoms per system. Shape (B,).
    chain_length : int
        Number of thermostats in chain.
    dt : wp.array
        Full time step dt per system. Shape (B,). The half-step and quarter-step
        factors are applied internally.
    device : str, optional
        Warp device.
    """
    if device is None:
        device = eta.device

    num_systems = eta.shape[0]

    wp.launch(
        _npt_thermostat_chain_update_kernel,
        dim=num_systems,
        inputs=[
            eta,
            eta_dot,
            kinetic_energy,
            target_temperature,
            thermostat_masses,
            num_atoms_per_system,
            chain_length,
            dt,
        ],
        device=device,
    )


def npt_barostat_half_step(
    cell_velocities: wp.array,
    pressure_tensors: wp.array,
    target_pressures: wp.array,
    volumes: wp.array,
    cell_masses: wp.array,
    kinetic_energy: wp.array,
    num_atoms_per_system: wp.array,
    eta_dots: wp.array,
    dt: wp.array,
    device: str = None,
) -> None:
    """
    Perform barostat half-step for NPT ensemble (Martyna-Tobias-Klein equations).

    This function updates the cell velocity matrix ḣ based on the pressure
    difference between the instantaneous and target pressures, coupled with
    the Nosé-Hoover thermostat chain. The pressure control mode is automatically
    detected from the ``target_pressures`` array dtype:

    - **Isotropic** (scalar dtype): Uniform scaling in all directions
    - **Anisotropic/Orthorhombic** (vec3 dtype): Independent x, y, z control
    - **Triclinic** (vec9 dtype): Full stress tensor control

    Mathematical Formulation
    ------------------------
    The cell velocity follows the MTK equations of motion:

    **Isotropic mode:**

    .. math::

        \\ddot{h} = \\frac{V}{W}(P - P_{ext}) - \\dot{\\eta}_1 \\dot{h}

    where P is the scalar trace of the pressure tensor (P = Tr(P)/3).

    **Anisotropic mode:**

    .. math::

        \\ddot{h}_{ii} = \\frac{V}{W}(P_{ii} - P_{ext,ii}) - \\dot{\\eta}_1 \\dot{h}_{ii}

    for i ∈ {x, y, z}, with off-diagonal elements remaining zero.

    **Triclinic mode:**

    .. math::

        \\ddot{h}_{ij} = \\frac{V}{W}(P_{ij} - P_{ext,ij}) - \\dot{\\eta}_1 \\dot{h}_{ij}

    for all i, j ∈ {x, y, z}.

    The instantaneous pressure includes a kinetic correction:

    .. math::

        P_{ij}^{inst} = P_{ij}^{virial} + \\frac{2 KE}{N_f V} \\delta_{ij}

    Parameters
    ----------
    cell_velocities : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell velocity matrices ḣ. Shape (B,) where B is batch size.
        **MODIFIED in-place.**
    pressure_tensors : wp.array(dtype=vec9f or vec9d)
        Current pressure tensors from virial. Shape (B,).
        Components ordered as [xx, xy, xz, yx, yy, yz, zx, zy, zz].
    target_pressures : wp.array
        External/target pressure(s). The dtype determines the mode:

        - ``wp.float32`` or ``wp.float64``: Isotropic (scalar P). Shape (B,).
        - ``wp.vec3f`` or ``wp.vec3d``: Anisotropic [Pxx, Pyy, Pzz]. Shape (B,).
        - ``vec9f`` or ``vec9d``: Full stress tensor. Shape (B,).

    volumes : wp.array(dtype=scalar)
        Cell volumes V. Shape (B,).
    cell_masses : wp.array(dtype=scalar)
        Barostat masses W. Shape (B,). See :func:`compute_barostat_mass`.
    kinetic_energy : wp.array(dtype=scalar)
        System kinetic energies. Shape (B,).
    num_atoms_per_system : wp.array(dtype=wp.int32)
        Number of atoms per system. Shape (B,).
    eta_dots : wp.array2d(dtype=scalar)
        Thermostat chain velocities η̇. Shape (B, chain_length).
        Only eta_dots[:, 0] (first thermostat) couples to barostat.
    dt : wp.array(dtype=scalar)
        Full time step per system. Shape (B,). The half-step factor is applied
        internally.
    device : str, optional
        Warp device. Default: inferred from cell_velocities.

    Examples
    --------
    Isotropic pressure control (most common):

    >>> target_P = wp.array([1.0], dtype=wp.float32, device="cuda:0")
    >>> npt_barostat_half_step(
    ...     cell_velocities, pressure_tensors, target_P,
    ...     volumes, cell_masses, kinetic_energy,
    ...     num_atoms_per_system, eta_dots, dt=0.001
    ... )

    Anisotropic (orthorhombic) pressure control:

    >>> # Different pressures for x, y, z axes
    >>> target_P = wp.array([[1.0, 2.0, 1.5]], dtype=wp.vec3f, device="cuda:0")
    >>> npt_barostat_half_step(
    ...     cell_velocities, pressure_tensors, target_P,
    ...     volumes, cell_masses, kinetic_energy,
    ...     num_atoms_per_system, eta_dots, dt=0.001
    ... )

    Full triclinic stress control:

    >>> # Full 3x3 stress tensor (9 components)
    >>> target_stress = wp.array(
    ...     [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
    ...     dtype=vec9f, device="cuda:0"
    ... )
    >>> npt_barostat_half_step(
    ...     cell_velocities, pressure_tensors, target_stress,
    ...     volumes, cell_masses, kinetic_energy,
    ...     num_atoms_per_system, eta_dots, dt=0.001
    ... )

    See Also
    --------
    nph_barostat_half_step : Barostat without thermostat coupling.
    npt_velocity_half_step : Velocity update with barostat/thermostat coupling.
    compute_barostat_mass : Compute barostat mass from relaxation time.

    References
    ----------
    .. [MTK1994] Martyna, Tobias, Klein, J. Chem. Phys. 101, 4177 (1994)
    .. [SSM2004] Shinoda, Shiga, Mikami, Phys. Rev. B 69, 134103 (2004)
    """
    if device is None:
        device = cell_velocities.device

    num_systems = cell_velocities.shape[0]

    # Dispatch based on target_pressures dtype
    tp_dtype = target_pressures.dtype
    kernel = _NPT_BAROSTAT_KERNELS.get(tp_dtype)
    if kernel is None:
        raise ValueError(
            f"Unsupported target_pressures dtype: {tp_dtype}. "
            f"Expected scalar (float32/float64) for isotropic, "
            f"vec3 for anisotropic, or vec9 for triclinic."
        )
    wp.launch(
        kernel,
        dim=num_systems,
        inputs=[
            cell_velocities,
            pressure_tensors,
            target_pressures,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            eta_dots,
            dt,
        ],
        device=device,
    )


# Keep separate functions for backwards compatibility and explicit control
def npt_barostat_half_step_aniso(
    cell_velocities: wp.array,
    pressure_tensors: wp.array,
    target_pressures: wp.array,
    volumes: wp.array,
    cell_masses: wp.array,
    kinetic_energy: wp.array,
    num_atoms_per_system: wp.array,
    eta_dots: wp.array,
    dt: wp.array,
    device: str = None,
) -> None:
    """Anisotropic NPT barostat. See :func:`npt_barostat_half_step` with vec3 target."""
    if device is None:
        device = cell_velocities.device
    num_systems = cell_velocities.shape[0]
    wp.launch(
        _npt_cell_velocity_update_aniso_kernel,
        dim=num_systems,
        inputs=[
            cell_velocities,
            pressure_tensors,
            target_pressures,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            eta_dots,
            dt,
        ],
        device=device,
    )


def npt_barostat_half_step_triclinic(
    cell_velocities: wp.array,
    pressure_tensors: wp.array,
    target_pressures: wp.array,
    volumes: wp.array,
    cell_masses: wp.array,
    kinetic_energy: wp.array,
    num_atoms_per_system: wp.array,
    eta_dots: wp.array,
    dt: wp.array,
    device: str = None,
) -> None:
    """Triclinic NPT barostat. See :func:`npt_barostat_half_step` with vec9 target."""
    if device is None:
        device = cell_velocities.device
    num_systems = cell_velocities.shape[0]
    wp.launch(
        _npt_cell_velocity_update_triclinic_kernel,
        dim=num_systems,
        inputs=[
            cell_velocities,
            pressure_tensors,
            target_pressures,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            eta_dots,
            dt,
        ],
        device=device,
    )


def npt_velocity_half_step(
    velocities: wp.array,
    masses: wp.array,
    forces: wp.array,
    cell_velocities: wp.array,
    volumes: wp.array,
    eta_dots: wp.array,
    num_atoms: wp.array,
    dt: wp.array,
    batch_idx: wp.array = None,
    num_atoms_per_system: wp.array = None,
    cells_inv: wp.array = None,
    mode: str = "isotropic",
    device: str = None,
) -> None:
    """
    Perform half-step velocity update for NPT ensemble.

    Updates particle velocities accounting for:
    1. Forces from the potential energy surface
    2. Coupling to barostat (cell velocity / strain rate)
    3. Coupling to thermostat (Nosé-Hoover chain)

    Mathematical Formulation
    ------------------------
    The velocity equation of motion in NPT is:

    .. math::

        \\dot{v}_i = \\frac{F_i}{m_i} - \\left(\\gamma \\cdot \\dot{\\varepsilon} + \\dot{\\eta}_1\\right) v_i

    where:

    - F_i / m_i is the acceleration from forces
    - γ = 1 + 1/N_f is the coupling factor (N_f = 3N degrees of freedom)
    - ε̇ is the strain rate from cell velocity
    - η̇₁ is the first thermostat chain velocity

    **Isotropic mode** (default):
        Uses scalar strain rate ε̇ = Tr(ḣ)/V

    **Anisotropic mode**:
        Uses diagonal strain rates ε̇_ii = ḣ_ii/V for direction-dependent drag

    **Triclinic mode**:
        Uses full strain rate tensor ε̇ = ḣ @ h⁻¹ for full coupling.
        Requires ``cells_inv`` parameter.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Particle velocities. Shape (N,). **MODIFIED in-place.**
    masses : wp.array(dtype=scalar)
        Particle masses. Shape (N,).
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on particles. Shape (N,).
    cell_velocities : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell velocity matrices ḣ. Shape (B,).
    volumes : wp.array(dtype=scalar)
        Cell volumes. Shape (B,).
    eta_dots : wp.array2d(dtype=scalar)
        Thermostat chain velocities. Shape (B, chain_length).
    num_atoms : wp.array(dtype=wp.int32)
        Atom count for single-system mode. Shape (1,).
    dt : wp.array(dtype=scalar)
        Full time step per system. Shape (B,). The half-step factor is applied
        internally.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched simulations.
    num_atoms_per_system : wp.array(dtype=wp.int32), optional
        Number of atoms per system. Required for batched simulations.
    cells_inv : wp.array(dtype=wp.mat33f or wp.mat33d), optional
        Inverse cell matrices h⁻¹. Shape (B,). **Required for triclinic mode.**
    mode : str, optional
        Pressure control mode. One of:

        - ``"isotropic"`` (default): Uniform scalar strain rate coupling
        - ``"anisotropic"``: Diagonal strain rate coupling (orthorhombic)
        - ``"triclinic"``: Full tensor strain rate coupling (requires cells_inv)

    device : str, optional
        Warp device.

    Examples
    --------
    Single system (isotropic):

    >>> npt_velocity_half_step(
    ...     velocities, masses, forces, cell_velocities, volumes,
    ...     eta_dots, num_atoms=100, dt=0.001
    ... )

    Triclinic cell:

    >>> npt_velocity_half_step(
    ...     velocities, masses, forces, cell_velocities, volumes,
    ...     eta_dots, num_atoms=100, dt=0.001,
    ...     cells_inv=cells_inv, mode="triclinic"
    ... )

    See Also
    --------
    npt_barostat_half_step : Cell velocity update step.
    npt_position_update : Position update step.
    """
    # In-place: delegate to _out with velocities as both input and output
    npt_velocity_half_step_out(
        velocities,
        masses,
        forces,
        cell_velocities,
        volumes,
        eta_dots,
        num_atoms,
        dt,
        velocities_out=velocities,
        batch_idx=batch_idx,
        num_atoms_per_system=num_atoms_per_system,
        cells_inv=cells_inv,
        mode=mode,
        device=device,
        _skip_validation=True,
    )


def npt_velocity_half_step_out(
    velocities: wp.array,
    masses: wp.array,
    forces: wp.array,
    cell_velocities: wp.array,
    volumes: wp.array,
    eta_dots: wp.array,
    num_atoms: wp.array,
    dt: wp.array,
    velocities_out: wp.array,
    batch_idx: wp.array = None,
    num_atoms_per_system: wp.array = None,
    cells_inv: wp.array = None,
    mode: str = "isotropic",
    device: str = None,
    _skip_validation: bool = False,
) -> wp.array:
    """
    Perform half-step velocity update for NPT (non-mutating).

    Non-mutating version of :func:`npt_velocity_half_step` that returns
    a new array instead of modifying in-place.

    Parameters
    ----------
    velocities : wp.array
        Input velocities (not modified when velocities_out differs).
    masses, forces, cell_velocities, volumes, eta_dots : wp.array
        System state arrays.
    num_atoms : wp.array(dtype=wp.int32)
        Atom count for single-system mode. Shape (1,).
    dt : wp.array(dtype=scalar)
        Full time step per system. Shape (B,). The half-step factor is applied
        internally.
    velocities_out : wp.array
        Pre-allocated output array.
    batch_idx : wp.array, optional
        System indices for batched simulations.
    num_atoms_per_system : wp.array, optional
        Atom counts per system.
    cells_inv : wp.array, optional
        Inverse cell matrices. Required for triclinic mode.
    mode : str, optional
        Pressure control mode: "isotropic", "anisotropic", or "triclinic".
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Updated velocities.
    """
    if device is None:
        device = velocities.device

    if not _skip_validation:
        validate_out_array(velocities_out, velocities, "velocities_out")

    exec_mode = resolve_execution_mode(batch_idx, None)
    n_atoms = velocities.shape[0]

    if mode not in _NPT_VELOCITY_FAMILIES:
        raise ValueError(
            f"Unknown mode: '{mode}'. Expected 'isotropic', 'anisotropic', or 'triclinic'."
        )
    if mode == "triclinic" and cells_inv is None:
        raise ValueError("mode='triclinic' requires cells_inv parameter.")

    family = _NPT_VELOCITY_FAMILIES[mode]
    extra = [cells_inv] if mode == "triclinic" else []

    launch_family(
        family,
        mode=exec_mode,
        dim=n_atoms,
        inputs_single=[
            velocities,
            masses,
            forces,
            cell_velocities,
            *extra,
            volumes,
            eta_dots,
            num_atoms,
            dt,
            velocities_out,
        ],
        inputs_batch=[
            velocities,
            masses,
            forces,
            batch_idx,
            cell_velocities,
            *extra,
            volumes,
            eta_dots,
            num_atoms_per_system,
            dt,
            velocities_out,
        ],
        device=device,
    )
    return velocities_out


def npt_position_update(
    positions: wp.array,
    velocities: wp.array,
    cells: wp.array,
    cell_velocities: wp.array,
    dt: wp.array,
    cells_inv: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> None:
    """
    Update positions for NPT integration.

    Parameters
    ----------
    positions : wp.array
        Particle positions. MODIFIED in-place.
    velocities : wp.array
        Particle velocities.
    cells : wp.array
        Cell matrices.
    cell_velocities : wp.array
        Cell velocity matrices.
    dt : wp.array(dtype=scalar)
        Full time step per system. Shape (B,).
    cells_inv : wp.array
        Pre-computed cell inverses. Caller must pre-compute via
        ``compute_cell_inverse``.
    batch_idx : wp.array, optional
        System index for each atom.
    device : str, optional
        Warp device.
    """
    # In-place: delegate to _out with positions as both input and output
    npt_position_update_out(
        positions,
        velocities,
        cells,
        cell_velocities,
        dt,
        positions,
        cells_inv,
        batch_idx=batch_idx,
        device=device,
        _skip_validation=True,
    )


def npt_position_update_out(
    positions: wp.array,
    velocities: wp.array,
    cells: wp.array,
    cell_velocities: wp.array,
    dt: wp.array,
    positions_out: wp.array,
    cells_inv: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
    _skip_validation: bool = False,
) -> wp.array:
    """
    Update positions for NPT integration (non-mutating).

    Parameters
    ----------
    cells_inv : wp.array
        Pre-computed cell inverses. Caller must pre-compute via
        ``compute_cell_inverse``.

    Returns
    -------
    wp.array
        Updated positions.
    """
    if device is None:
        device = positions.device

    if not _skip_validation:
        validate_out_array(positions_out, positions, "positions_out")

    exec_mode = resolve_execution_mode(batch_idx, None)
    num_atoms = positions.shape[0]

    launch_family(
        _POSITION_UPDATE_FAMILY,
        mode=exec_mode,
        dim=num_atoms,
        inputs_single=[
            positions,
            velocities,
            cells,
            cells_inv,
            cell_velocities,
            dt,
            positions_out,
        ],
        inputs_batch=[
            positions,
            velocities,
            batch_idx,
            cells,
            cells_inv,
            cell_velocities,
            dt,
            positions_out,
        ],
        device=device,
    )
    return positions_out


def npt_cell_update(
    cells: wp.array,
    cell_velocities: wp.array,
    dt: wp.array,
    device: str = None,
) -> None:
    """
    Update cell matrices: h_new = h + dt * ḣ.

    Parameters
    ----------
    cells : wp.array
        Cell matrices. MODIFIED in-place.
    cell_velocities : wp.array
        Cell velocity matrices.
    dt : wp.array(dtype=scalar)
        Full time step per system. Shape (B,).
    device : str, optional
        Warp device.
    """
    npt_cell_update_out(
        cells,
        cell_velocities,
        dt,
        cells_out=cells,
        device=device,
        _skip_validation=True,
    )


def npt_cell_update_out(
    cells: wp.array,
    cell_velocities: wp.array,
    dt: wp.array,
    cells_out: wp.array,
    device: str = None,
    _skip_validation: bool = False,
) -> wp.array:
    """
    Update cell matrices (non-mutating).

    Returns
    -------
    wp.array
        Updated cell matrices.
    """
    if device is None:
        device = cells.device

    if not _skip_validation:
        validate_out_array(cells_out, cells, "cells_out")

    num_systems = cells.shape[0]

    wp.launch(
        _cell_update_kernel,
        dim=num_systems,
        inputs=[cells, cell_velocities, dt, cells_out],
        device=device,
    )

    return cells_out


def run_npt_step(
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    cells: wp.array,
    cell_velocities: wp.array,
    virial_tensors: wp.array,
    eta: wp.array,
    eta_dot: wp.array,
    thermostat_masses: wp.array,
    cell_masses: wp.array,
    target_temperature: wp.array,
    target_pressure: wp.array,
    num_atoms: wp.array,
    chain_length: int,
    dt: wp.array,
    pressure_tensors: wp.array,
    volumes: wp.array,
    kinetic_energy: wp.array,
    cells_inv: wp.array,
    kinetic_tensors: wp.array,
    num_atoms_per_system: wp.array,
    compute_forces_fn=None,
    batch_idx: wp.array = None,
    device: str = None,
) -> None:
    """
    Perform a complete NPT integration step.

    Integration order:
    1. Thermostat half-step
    2. Barostat half-step
    3. Velocity half-step
    4. Position update
    5. Cell update
    6. Recompute forces
    7. Velocity half-step
    8. Barostat half-step
    9. Thermostat half-step

    Parameters
    ----------
    positions, velocities, forces : wp.array
        Particle state arrays. MODIFIED in-place.
    masses : wp.array
        Particle masses.
    cells : wp.array
        Cell matrices. MODIFIED in-place.
    cell_velocities : wp.array
        Cell velocity matrices. MODIFIED in-place.
    virial_tensors : wp.array(dtype=vec9f or vec9d)
        Virial tensor. Updated by compute_forces_fn.
    eta, eta_dot : wp.array2d
        Thermostat state. MODIFIED in-place.
    thermostat_masses : wp.array2d
        Thermostat masses.
    cell_masses : wp.array
        Barostat masses.
    target_temperature, target_pressure : wp.array
        Target conditions.
    num_atoms : wp.array(dtype=wp.int32)
        Atom count for single-system mode. Shape (1,).
    chain_length : int
        Number of thermostats in chain.
    dt : wp.array(dtype=scalar)
        Full time step per system. Shape (B,).
    pressure_tensors : wp.array(dtype=vec9f or vec9d)
        Scratch array for pressure tensor. Shape (B,).
    volumes : wp.array(dtype=scalar)
        Scratch array for cell volumes. Shape (B,).
    kinetic_energy : wp.array(dtype=scalar)
        Scratch array for kinetic energy. Shape (B,).
        Zeroed internally before each use.
    cells_inv : wp.array(dtype=mat33)
        Scratch array for cell inverses. Shape (B,).
    kinetic_tensors : wp.array(dtype=scalar, ndim=2)
        Scratch array for kinetic tensor accumulation. Shape (B, 9).
        Zeroed internally before each use.
    num_atoms_per_system : wp.array(dtype=wp.int32)
        Number of atoms per system. Shape (B,).
    compute_forces_fn : callable, optional
        Force computation function.
    batch_idx : wp.array, optional
        System index for each atom.
    device : str, optional
        Warp device.
    """
    if device is None:
        device = positions.device

    compute_pressure_tensor(
        velocities,
        masses,
        virial_tensors,
        cells,
        kinetic_tensors,
        pressure_tensors,
        volumes,
        batch_idx=batch_idx,
        device=device,
    )

    # 1. Thermostat half-step
    npt_thermostat_half_step(
        eta,
        eta_dot,
        kinetic_energy,
        target_temperature,
        thermostat_masses,
        num_atoms_per_system,
        chain_length,
        dt,
        device=device,
    )

    # 2. Barostat half-step
    npt_barostat_half_step(
        cell_velocities,
        pressure_tensors,
        target_pressure,
        volumes,
        cell_masses,
        kinetic_energy,
        num_atoms_per_system,
        eta_dot,
        dt,
        device=device,
    )

    # 3. Velocity half-step
    npt_velocity_half_step(
        velocities,
        masses,
        forces,
        cell_velocities,
        volumes,
        eta_dot,
        num_atoms,
        dt,
        batch_idx=batch_idx,
        num_atoms_per_system=num_atoms_per_system,
        device=device,
    )

    # 4. Position update
    compute_cell_inverse(cells, cells_inv=cells_inv, device=device)
    npt_position_update(
        positions,
        velocities,
        cells,
        cell_velocities,
        dt,
        cells_inv=cells_inv,
        batch_idx=batch_idx,
        device=device,
    )

    # 5. Cell update
    npt_cell_update(cells, cell_velocities, dt, device=device)

    # 6. Recompute forces
    if compute_forces_fn is not None:
        compute_forces_fn(positions, cells, forces, virial_tensors)

    # Recompute pressure and volumes
    compute_cell_volume(cells, volumes=volumes, device=device)
    compute_kinetic_energy(
        velocities,
        masses,
        kinetic_energy=kinetic_energy,
        batch_idx=batch_idx,
        device=device,
    )
    compute_pressure_tensor(
        velocities,
        masses,
        virial_tensors,
        cells,
        kinetic_tensors,
        pressure_tensors,
        volumes,
        batch_idx=batch_idx,
        device=device,
    )

    # 7. Velocity half-step
    npt_velocity_half_step(
        velocities,
        masses,
        forces,
        cell_velocities,
        volumes,
        eta_dot,
        num_atoms,
        dt,
        batch_idx=batch_idx,
        num_atoms_per_system=num_atoms_per_system,
        device=device,
    )

    # 8. Barostat half-step
    npt_barostat_half_step(
        cell_velocities,
        pressure_tensors,
        target_pressure,
        volumes,
        cell_masses,
        kinetic_energy,
        num_atoms_per_system,
        eta_dot,
        dt,
        device=device,
    )

    # 9. Thermostat half-step
    npt_thermostat_half_step(
        eta,
        eta_dot,
        kinetic_energy,
        target_temperature,
        thermostat_masses,
        num_atoms_per_system,
        chain_length,
        dt,
        device=device,
    )


# ==============================================================================
# Functional Interfaces - NPH Integration
# ==============================================================================


def nph_barostat_half_step(
    cell_velocities: wp.array,
    pressure_tensors: wp.array,
    target_pressures: wp.array,
    volumes: wp.array,
    cell_masses: wp.array,
    kinetic_energy: wp.array,
    num_atoms_per_system: wp.array,
    dt: wp.array,
    device: str = None,
) -> None:
    """
    Perform barostat half-step for NPH ensemble (no thermostat coupling).

    NPH (isenthalpic-isobaric) simulations maintain constant pressure and
    enthalpy. Unlike NPT, there is no thermostat - the temperature evolves
    naturally on the constant-enthalpy surface.

    This function updates the cell velocity matrix ḣ based on the pressure
    difference. The pressure control mode is automatically detected from
    the ``target_pressures`` array dtype:

    - **Isotropic** (scalar dtype): Uniform scaling in all directions
    - **Anisotropic/Orthorhombic** (vec3 dtype): Independent x, y, z control
    - **Triclinic** (vec9 dtype): Full stress tensor control

    Mathematical Formulation
    ------------------------
    The cell velocity follows the MTK equations without thermostat:

    **Isotropic mode:**

    .. math::

        \\ddot{h} = \\frac{V}{W}(P - P_{ext})

    **Anisotropic mode:**

    .. math::

        \\ddot{h}_{ii} = \\frac{V}{W}(P_{ii} - P_{ext,ii})

    **Triclinic mode:**

    .. math::

        \\ddot{h}_{ij} = \\frac{V}{W}(P_{ij} - P_{ext,ij})

    Parameters
    ----------
    cell_velocities : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell velocity matrices ḣ. Shape (B,). **MODIFIED in-place.**
    pressure_tensors : wp.array(dtype=vec9f or vec9d)
        Current pressure tensors from virial. Shape (B,).
    target_pressures : wp.array
        External/target pressure(s). The dtype determines the mode:

        - ``wp.float32`` or ``wp.float64``: Isotropic. Shape (B,).
        - ``wp.vec3f`` or ``wp.vec3d``: Anisotropic [Pxx, Pyy, Pzz]. Shape (B,).
        - ``vec9f`` or ``vec9d``: Full stress tensor. Shape (B,).

    volumes : wp.array(dtype=scalar)
        Cell volumes V. Shape (B,).
    cell_masses : wp.array(dtype=scalar)
        Barostat masses W. Shape (B,).
    kinetic_energy : wp.array(dtype=scalar)
        System kinetic energies. Shape (B,).
    num_atoms_per_system : wp.array(dtype=wp.int32)
        Number of atoms per system. Shape (B,).
    dt : wp.array(dtype=scalar)
        Full time step per system. Shape (B,). The half-step factor is applied
        internally.
    device : str, optional
        Warp device.

    Examples
    --------
    Isotropic NPH:

    >>> target_P = wp.array([1.0], dtype=wp.float32, device="cuda:0")
    >>> nph_barostat_half_step(
    ...     cell_velocities, pressure_tensors, target_P,
    ...     volumes, cell_masses, kinetic_energy,
    ...     num_atoms_per_system, dt=0.001
    ... )

    Anisotropic NPH:

    >>> target_P = wp.array([[1.0, 2.0, 1.5]], dtype=wp.vec3f, device="cuda:0")
    >>> nph_barostat_half_step(
    ...     cell_velocities, pressure_tensors, target_P, ...
    ... )

    See Also
    --------
    npt_barostat_half_step : Barostat with thermostat coupling.
    run_nph_step : Complete NPH integration step.
    """
    if device is None:
        device = cell_velocities.device

    num_systems = cell_velocities.shape[0]

    # Dispatch based on target_pressures dtype
    tp_dtype = target_pressures.dtype
    kernel = _NPH_BAROSTAT_KERNELS.get(tp_dtype)
    if kernel is None:
        raise ValueError(
            f"Unsupported target_pressures dtype: {tp_dtype}. "
            f"Expected scalar for isotropic, vec3 for anisotropic, vec9 for triclinic."
        )
    wp.launch(
        kernel,
        dim=num_systems,
        inputs=[
            cell_velocities,
            pressure_tensors,
            target_pressures,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            dt,
        ],
        device=device,
    )


# Keep separate functions for backwards compatibility
def nph_barostat_half_step_aniso(
    cell_velocities: wp.array,
    pressure_tensors: wp.array,
    target_pressures: wp.array,
    volumes: wp.array,
    cell_masses: wp.array,
    kinetic_energy: wp.array,
    num_atoms_per_system: wp.array,
    dt: wp.array,
    device: str = None,
) -> None:
    """Anisotropic NPH barostat. See :func:`nph_barostat_half_step` with vec3 target."""
    if device is None:
        device = cell_velocities.device
    num_systems = cell_velocities.shape[0]
    wp.launch(
        _nph_cell_velocity_update_aniso_kernel,
        dim=num_systems,
        inputs=[
            cell_velocities,
            pressure_tensors,
            target_pressures,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            dt,
        ],
        device=device,
    )


def nph_barostat_half_step_triclinic(
    cell_velocities: wp.array,
    pressure_tensors: wp.array,
    target_pressures: wp.array,
    volumes: wp.array,
    cell_masses: wp.array,
    kinetic_energy: wp.array,
    num_atoms_per_system: wp.array,
    dt: wp.array,
    device: str = None,
) -> None:
    """Triclinic NPH barostat. See :func:`nph_barostat_half_step` with vec9 target."""
    if device is None:
        device = cell_velocities.device

    num_systems = cell_velocities.shape[0]

    wp.launch(
        _nph_cell_velocity_update_triclinic_kernel,
        dim=num_systems,
        inputs=[
            cell_velocities,
            pressure_tensors,
            target_pressures,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            dt,
        ],
        device=device,
    )


def nph_velocity_half_step(
    velocities: wp.array,
    masses: wp.array,
    forces: wp.array,
    cell_velocities: wp.array,
    volumes: wp.array,
    num_atoms: wp.array,
    dt: wp.array,
    batch_idx: wp.array = None,
    num_atoms_per_system: wp.array = None,
    cells_inv: wp.array = None,
    mode: str = "isotropic",
    device: str = None,
) -> None:
    """
    Perform half-step velocity update for NPH ensemble (no thermostat).

    Updates particle velocities accounting for:
    1. Forces from the potential energy surface
    2. Coupling to barostat (cell velocity / strain rate)

    Unlike NPT, there is no thermostat coupling - the temperature evolves
    naturally on the constant-enthalpy surface.

    Mathematical Formulation
    ------------------------
    The NPH velocity equation of motion:

    .. math::

        \\dot{v}_i = \\frac{F_i}{m_i} - \\gamma \\cdot \\dot{\\varepsilon} \\cdot v_i

    where γ = 1 + 1/N_f is the coupling factor.

    **Isotropic mode**: Uses scalar strain rate ε̇ = Tr(ḣ)/V

    **Anisotropic mode**: Uses diagonal strain rates ε̇_ii = ḣ_ii/V

    **Triclinic mode**: Uses full strain rate tensor ε̇ = ḣ @ h⁻¹

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Particle velocities. **MODIFIED in-place.**
    masses : wp.array(dtype=scalar)
        Particle masses.
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on particles.
    cell_velocities : wp.array(dtype=wp.mat33f or wp.mat33d)
        Cell velocity matrices.
    volumes : wp.array(dtype=scalar)
        Cell volumes.
    num_atoms : wp.array(dtype=wp.int32)
        Atom count for single-system mode. Shape (1,).
    dt : wp.array(dtype=scalar)
        Full time step per system. Shape (B,). The half-step factor is applied
        internally.
    batch_idx : wp.array, optional
        System index for each atom.
    num_atoms_per_system : wp.array, optional
        Number of atoms per system.
    cells_inv : wp.array, optional
        Inverse cell matrices. Required for triclinic mode.
    mode : str, optional
        Pressure control mode:

        - ``"isotropic"``: Uniform scalar strain rate coupling
        - ``"anisotropic"``: Diagonal strain rate coupling (orthorhombic)
        - ``"triclinic"``: Full tensor coupling (requires cells_inv)

    device : str, optional
        Warp device.

    See Also
    --------
    nph_barostat_half_step : Cell velocity update.
    npt_velocity_half_step : Velocity update with thermostat.
    """
    # In-place: delegate to _out with velocities as both input and output
    nph_velocity_half_step_out(
        velocities,
        masses,
        forces,
        cell_velocities,
        volumes,
        num_atoms,
        dt,
        velocities_out=velocities,
        batch_idx=batch_idx,
        num_atoms_per_system=num_atoms_per_system,
        cells_inv=cells_inv,
        mode=mode,
        device=device,
        _skip_validation=True,
    )


# NOTE: nph_velocity_half_step_aniso removed - use nph_velocity_half_step(mode="anisotropic")


def nph_velocity_half_step_out(
    velocities: wp.array,
    masses: wp.array,
    forces: wp.array,
    cell_velocities: wp.array,
    volumes: wp.array,
    num_atoms: wp.array,
    dt: wp.array,
    velocities_out: wp.array,
    batch_idx: wp.array = None,
    num_atoms_per_system: wp.array = None,
    cells_inv: wp.array = None,
    mode: str = "isotropic",
    device: str = None,
    _skip_validation: bool = False,
) -> wp.array:
    """
    Perform half-step velocity update for NPH (non-mutating).

    Non-mutating version of :func:`nph_velocity_half_step`.

    Parameters
    ----------
    velocities : wp.array
        Input velocities (not modified when velocities_out differs).
    masses, forces, cell_velocities, volumes : wp.array
        System state arrays.
    num_atoms : wp.array(dtype=wp.int32)
        Atom count for single-system mode. Shape (1,).
    dt : wp.array(dtype=scalar)
        Full time step per system. Shape (B,). The half-step factor is applied
        internally.
    velocities_out : wp.array
        Pre-allocated output array.
    batch_idx, num_atoms_per_system : wp.array, optional
        For batched simulations.
    cells_inv : wp.array, optional
        Inverse cell matrices. Required for triclinic mode.
    mode : str, optional
        Pressure control mode: "isotropic", "anisotropic", or "triclinic".
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Updated velocities.
    """
    if device is None:
        device = velocities.device

    if not _skip_validation:
        validate_out_array(velocities_out, velocities, "velocities_out")

    exec_mode = resolve_execution_mode(batch_idx, None)
    n_atoms = velocities.shape[0]

    if mode not in _NPH_VELOCITY_FAMILIES:
        raise ValueError(
            f"Unknown mode: '{mode}'. Expected 'isotropic', 'anisotropic', or 'triclinic'."
        )
    if mode == "triclinic" and cells_inv is None:
        raise ValueError("mode='triclinic' requires cells_inv parameter.")

    family = _NPH_VELOCITY_FAMILIES[mode]
    extra = [cells_inv] if mode == "triclinic" else []

    launch_family(
        family,
        mode=exec_mode,
        dim=n_atoms,
        inputs_single=[
            velocities,
            masses,
            forces,
            cell_velocities,
            *extra,
            volumes,
            num_atoms,
            dt,
            velocities_out,
        ],
        inputs_batch=[
            velocities,
            masses,
            forces,
            batch_idx,
            cell_velocities,
            *extra,
            volumes,
            num_atoms_per_system,
            dt,
            velocities_out,
        ],
        device=device,
    )
    return velocities_out


def nph_position_update(
    positions: wp.array,
    velocities: wp.array,
    cells: wp.array,
    cell_velocities: wp.array,
    dt: wp.array,
    cells_inv: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> None:
    """
    Update positions for NPH integration.

    Uses same kernel as NPT since position update is identical.
    """
    npt_position_update(
        positions,
        velocities,
        cells,
        cell_velocities,
        dt,
        cells_inv,
        batch_idx=batch_idx,
        device=device,
    )


def nph_position_update_out(
    positions: wp.array,
    velocities: wp.array,
    cells: wp.array,
    cell_velocities: wp.array,
    dt: wp.array,
    positions_out: wp.array,
    cells_inv: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Update positions for NPH integration (non-mutating).

    Returns
    -------
    wp.array
        Updated positions.
    """
    return npt_position_update_out(
        positions,
        velocities,
        cells,
        cell_velocities,
        dt,
        positions_out,
        cells_inv,
        batch_idx=batch_idx,
        device=device,
    )


def nph_cell_update(
    cells: wp.array,
    cell_velocities: wp.array,
    dt: wp.array,
    device: str = None,
) -> None:
    """
    Update cell matrices for NPH.

    Uses same kernel as NPT since cell update is identical.
    """
    npt_cell_update(cells, cell_velocities, dt, device=device)


def run_nph_step(
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    cells: wp.array,
    cell_velocities: wp.array,
    virial_tensors: wp.array,
    cell_masses: wp.array,
    target_pressure: wp.array,
    num_atoms: wp.array,
    dt: wp.array,
    pressure_tensors: wp.array,
    volumes: wp.array,
    kinetic_energy: wp.array,
    cells_inv: wp.array,
    kinetic_tensors: wp.array,
    num_atoms_per_system: wp.array,
    compute_forces_fn=None,
    batch_idx: wp.array = None,
    device: str = None,
) -> None:
    """
    Perform a complete NPH integration step.

    Integration order (no thermostat):
    1. Barostat half-step
    2. Velocity half-step
    3. Position update
    4. Cell update
    5. Recompute forces
    6. Velocity half-step
    7. Barostat half-step

    Parameters
    ----------
    positions, velocities, forces : wp.array
        Particle state arrays. MODIFIED in-place.
    masses : wp.array
        Particle masses.
    cells : wp.array
        Cell matrices. MODIFIED in-place.
    cell_velocities : wp.array
        Cell velocity matrices. MODIFIED in-place.
    virial_tensors : wp.array(dtype=vec9f or vec9d)
        Virial tensor. Updated by compute_forces_fn.
    cell_masses : wp.array
        Barostat masses.
    target_pressure : wp.array
        Target/external pressures.
    num_atoms : wp.array(dtype=wp.int32)
        Atom count for single-system mode. Shape (1,).
    dt : wp.array(dtype=scalar)
        Full time step per system. Shape (B,).
    pressure_tensors : wp.array(dtype=vec9f or vec9d)
        Scratch array for pressure tensor. Shape (B,).
    volumes : wp.array(dtype=scalar)
        Scratch array for cell volumes. Shape (B,).
    kinetic_energy : wp.array(dtype=scalar)
        Scratch array for kinetic energy. Shape (B,).
        Zeroed internally before each use.
    cells_inv : wp.array(dtype=mat33)
        Scratch array for cell inverses. Shape (B,).
    kinetic_tensors : wp.array(dtype=scalar, ndim=2)
        Scratch array for kinetic tensor accumulation. Shape (B, 9).
        Zeroed internally before each use.
    num_atoms_per_system : wp.array(dtype=wp.int32)
        Number of atoms per system. Shape (B,).
    compute_forces_fn : callable, optional
        Force computation function.
    batch_idx : wp.array, optional
        System index for each atom.
    device : str, optional
        Warp device.
    """
    if device is None:
        device = positions.device

    compute_pressure_tensor(
        velocities,
        masses,
        virial_tensors,
        cells,
        kinetic_tensors,
        pressure_tensors,
        volumes,
        batch_idx=batch_idx,
        device=device,
    )

    # 1. Barostat half-step
    nph_barostat_half_step(
        cell_velocities,
        pressure_tensors,
        target_pressure,
        volumes,
        cell_masses,
        kinetic_energy,
        num_atoms_per_system,
        dt,
        device=device,
    )

    # 2. Velocity half-step
    nph_velocity_half_step(
        velocities,
        masses,
        forces,
        cell_velocities,
        volumes,
        num_atoms,
        dt,
        batch_idx=batch_idx,
        num_atoms_per_system=num_atoms_per_system,
        device=device,
    )

    # 3. Position update
    compute_cell_inverse(cells, cells_inv=cells_inv, device=device)
    nph_position_update(
        positions,
        velocities,
        cells,
        cell_velocities,
        dt,
        cells_inv=cells_inv,
        batch_idx=batch_idx,
        device=device,
    )

    # 4. Cell update
    nph_cell_update(cells, cell_velocities, dt, device=device)

    # 5. Recompute forces
    if compute_forces_fn is not None:
        compute_forces_fn(positions, cells, forces, virial_tensors)

    # Recompute pressure and volumes
    compute_cell_volume(cells, volumes=volumes, device=device)
    compute_kinetic_energy(
        velocities,
        masses,
        kinetic_energy=kinetic_energy,
        batch_idx=batch_idx,
        device=device,
    )
    compute_pressure_tensor(
        velocities,
        masses,
        virial_tensors,
        cells,
        kinetic_tensors,
        pressure_tensors,
        volumes,
        batch_idx=batch_idx,
        device=device,
    )

    # 6. Velocity half-step
    nph_velocity_half_step(
        velocities,
        masses,
        forces,
        cell_velocities,
        volumes,
        num_atoms,
        dt,
        batch_idx=batch_idx,
        num_atoms_per_system=num_atoms_per_system,
        device=device,
    )

    # 7. Barostat half-step
    nph_barostat_half_step(
        cell_velocities,
        pressure_tensors,
        target_pressure,
        volumes,
        cell_masses,
        kinetic_energy,
        num_atoms_per_system,
        dt,
        device=device,
    )
