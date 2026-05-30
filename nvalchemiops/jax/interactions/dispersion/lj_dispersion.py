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

r"""JAX bindings for pairwise dispersion (:math:`r^{-6}`) interactions.

Wraps the framework-agnostic Warp kernels from
``nvalchemiops.interactions.dispersion.lj_real_kernels`` with JAX bindings,
mirroring ``nvalchemiops.jax.interactions.electrostatics.coulomb``.

The public API takes per-atom LJ :math:`\sigma`/:math:`\epsilon` and converts
internally to the geometric-rule dispersion charge
``b_i = 2 sqrt(eps_i) sigma_i**3``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from warp.jax_experimental import jax_kernel

from nvalchemiops.interactions.dispersion.lj_real_kernels import (
    _batch_lj_dispersion_energy_forces_kernel,
    _batch_lj_dispersion_energy_forces_matrix_kernel,
    _batch_lj_dispersion_energy_kernel,
    _batch_lj_dispersion_energy_matrix_kernel,
    _lj_dispersion_energy_forces_kernel,
    _lj_dispersion_energy_forces_matrix_kernel,
    _lj_dispersion_energy_kernel,
    _lj_dispersion_energy_matrix_kernel,
)

__all__ = [
    "lj_dispersion_energy",
    "lj_dispersion_forces",
    "lj_dispersion_energy_forces",
    "sigma_epsilon_to_dispersion_charge",
]

# --- Neighbor List (CSR) Format ---
_jax_lj_energy_list = jax_kernel(
    _lj_dispersion_energy_kernel,
    num_outputs=1,
    in_out_argnames=["energies"],
    enable_backward=False,
)
_jax_lj_energy_forces_list = jax_kernel(
    _lj_dispersion_energy_forces_kernel,
    num_outputs=2,
    in_out_argnames=["energies", "forces"],
    enable_backward=False,
)
_jax_batch_lj_energy_list = jax_kernel(
    _batch_lj_dispersion_energy_kernel,
    num_outputs=1,
    in_out_argnames=["energies"],
    enable_backward=False,
)
_jax_batch_lj_energy_forces_list = jax_kernel(
    _batch_lj_dispersion_energy_forces_kernel,
    num_outputs=2,
    in_out_argnames=["energies", "forces"],
    enable_backward=False,
)

# --- Neighbor Matrix Format ---
_jax_lj_energy_matrix = jax_kernel(
    _lj_dispersion_energy_matrix_kernel,
    num_outputs=1,
    in_out_argnames=["atomic_energies"],
    enable_backward=False,
)
_jax_lj_energy_forces_matrix = jax_kernel(
    _lj_dispersion_energy_forces_matrix_kernel,
    num_outputs=2,
    in_out_argnames=["atomic_energies", "atomic_forces"],
    enable_backward=False,
)
_jax_batch_lj_energy_matrix = jax_kernel(
    _batch_lj_dispersion_energy_matrix_kernel,
    num_outputs=1,
    in_out_argnames=["atomic_energies"],
    enable_backward=False,
)
_jax_batch_lj_energy_forces_matrix = jax_kernel(
    _batch_lj_dispersion_energy_forces_matrix_kernel,
    num_outputs=2,
    in_out_argnames=["atomic_energies", "atomic_forces"],
    enable_backward=False,
)


def sigma_epsilon_to_dispersion_charge(
    sigma: jax.Array, epsilon: jax.Array
) -> jax.Array:
    r"""Convert per-atom LJ :math:`\sigma, \epsilon` to dispersion charge.

    ``b_i = sqrt(C6_i) = 2 sqrt(eps_i) sigma_i**3`` (geometric rule).
    """
    return 2.0 * jnp.sqrt(epsilon.astype(jnp.float64)) * sigma.astype(jnp.float64) ** 3


def lj_dispersion_energy(
    positions: jax.Array,
    sigma: jax.Array,
    epsilon: jax.Array,
    cell: jax.Array,
    cutoff: float,
    alpha: float = 0.0,
    neighbor_list: jax.Array | None = None,
    neighbor_ptr: jax.Array | None = None,
    neighbor_shifts: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    fill_value: int | None = None,
    batch_idx: jax.Array | None = None,
) -> jax.Array:
    r"""Compute pairwise dispersion (:math:`r^{-6}`) energies. See the torch
    binding for full parameter semantics. ``alpha`` is the damping
    :math:`\beta` (0 for plain ``-C6/r^6``)."""
    use_list = neighbor_list is not None and neighbor_shifts is not None
    use_matrix = neighbor_matrix is not None and neighbor_matrix_shifts is not None
    if not use_list and not use_matrix:
        raise ValueError(
            "Must provide either neighbor_list/neighbor_shifts or "
            "neighbor_matrix/neighbor_matrix_shifts"
        )
    if use_list and use_matrix:
        raise ValueError("Cannot provide both neighbor list and neighbor matrix")

    original_dtype = positions.dtype
    positions_f64 = positions.astype(jnp.float64)
    b_f64 = sigma_epsilon_to_dispersion_charge(sigma, epsilon)
    cell_f64 = cell.astype(jnp.float64)
    if cell_f64.ndim == 2:
        cell_f64 = cell_f64[jnp.newaxis, :, :]

    num_atoms = positions_f64.shape[0]
    is_batched = batch_idx is not None
    energies = jnp.zeros(num_atoms, dtype=jnp.float64)

    if use_list:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr is required when using neighbor_list format")
        idx_j = neighbor_list[1].astype(jnp.int32)
        nptr = neighbor_ptr.astype(jnp.int32)
        nsh = neighbor_shifts.astype(jnp.int32)
        if is_batched:
            (energies,) = _jax_batch_lj_energy_list(
                positions_f64,
                b_f64,
                cell_f64,
                idx_j,
                nptr,
                nsh,
                batch_idx.astype(jnp.int32),
                float(cutoff),
                float(alpha),
                energies,
                launch_dims=(num_atoms,),
            )
        else:
            (energies,) = _jax_lj_energy_list(
                positions_f64,
                b_f64,
                cell_f64,
                idx_j,
                nptr,
                nsh,
                float(cutoff),
                float(alpha),
                energies,
                launch_dims=(num_atoms,),
            )
    else:
        nmat = neighbor_matrix.astype(jnp.int32)
        nmat_sh = neighbor_matrix_shifts.astype(jnp.int32)
        if fill_value is None:
            fill_value = num_atoms
        if is_batched:
            (energies,) = _jax_batch_lj_energy_matrix(
                positions_f64,
                b_f64,
                cell_f64,
                nmat,
                nmat_sh,
                batch_idx.astype(jnp.int32),
                float(cutoff),
                float(alpha),
                int(fill_value),
                energies,
                launch_dims=(num_atoms,),
            )
        else:
            (energies,) = _jax_lj_energy_matrix(
                positions_f64,
                b_f64,
                cell_f64,
                nmat,
                nmat_sh,
                float(cutoff),
                float(alpha),
                int(fill_value),
                energies,
                launch_dims=(num_atoms,),
            )
    return energies.astype(original_dtype)


def lj_dispersion_forces(
    positions: jax.Array,
    sigma: jax.Array,
    epsilon: jax.Array,
    cell: jax.Array,
    cutoff: float,
    alpha: float = 0.0,
    neighbor_list: jax.Array | None = None,
    neighbor_ptr: jax.Array | None = None,
    neighbor_shifts: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    fill_value: int | None = None,
    batch_idx: jax.Array | None = None,
) -> jax.Array:
    """Compute pairwise dispersion forces (convenience wrapper)."""
    _, forces = lj_dispersion_energy_forces(
        positions,
        sigma,
        epsilon,
        cell,
        cutoff,
        alpha,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        fill_value=fill_value,
        batch_idx=batch_idx,
    )
    return forces


def lj_dispersion_energy_forces(
    positions: jax.Array,
    sigma: jax.Array,
    epsilon: jax.Array,
    cell: jax.Array,
    cutoff: float,
    alpha: float = 0.0,
    neighbor_list: jax.Array | None = None,
    neighbor_ptr: jax.Array | None = None,
    neighbor_shifts: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    fill_value: int | None = None,
    batch_idx: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute pairwise dispersion energies and forces."""
    use_list = neighbor_list is not None and neighbor_shifts is not None
    use_matrix = neighbor_matrix is not None and neighbor_matrix_shifts is not None
    if not use_list and not use_matrix:
        raise ValueError(
            "Must provide either neighbor_list/neighbor_shifts or "
            "neighbor_matrix/neighbor_matrix_shifts"
        )
    if use_list and use_matrix:
        raise ValueError("Cannot provide both neighbor list and neighbor matrix")

    original_dtype = positions.dtype
    positions_f64 = positions.astype(jnp.float64)
    b_f64 = sigma_epsilon_to_dispersion_charge(sigma, epsilon)
    cell_f64 = cell.astype(jnp.float64)
    if cell_f64.ndim == 2:
        cell_f64 = cell_f64[jnp.newaxis, :, :]

    num_atoms = positions_f64.shape[0]
    is_batched = batch_idx is not None
    energies = jnp.zeros(num_atoms, dtype=jnp.float64)
    forces = jnp.zeros((num_atoms, 3), dtype=jnp.float64)

    if use_list:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr is required when using neighbor_list format")
        idx_j = neighbor_list[1].astype(jnp.int32)
        nptr = neighbor_ptr.astype(jnp.int32)
        nsh = neighbor_shifts.astype(jnp.int32)
        if is_batched:
            energies, forces = _jax_batch_lj_energy_forces_list(
                positions_f64,
                b_f64,
                cell_f64,
                idx_j,
                nptr,
                nsh,
                batch_idx.astype(jnp.int32),
                float(cutoff),
                float(alpha),
                energies,
                forces,
                launch_dims=(num_atoms,),
            )
        else:
            energies, forces = _jax_lj_energy_forces_list(
                positions_f64,
                b_f64,
                cell_f64,
                idx_j,
                nptr,
                nsh,
                float(cutoff),
                float(alpha),
                energies,
                forces,
                launch_dims=(num_atoms,),
            )
    else:
        nmat = neighbor_matrix.astype(jnp.int32)
        nmat_sh = neighbor_matrix_shifts.astype(jnp.int32)
        if fill_value is None:
            fill_value = num_atoms
        if is_batched:
            energies, forces = _jax_batch_lj_energy_forces_matrix(
                positions_f64,
                b_f64,
                cell_f64,
                nmat,
                nmat_sh,
                batch_idx.astype(jnp.int32),
                float(cutoff),
                float(alpha),
                int(fill_value),
                energies,
                forces,
                launch_dims=(num_atoms,),
            )
        else:
            energies, forces = _jax_lj_energy_forces_matrix(
                positions_f64,
                b_f64,
                cell_f64,
                nmat,
                nmat_sh,
                float(cutoff),
                float(alpha),
                int(fill_value),
                energies,
                forces,
                launch_dims=(num_atoms,),
            )
    return energies.astype(original_dtype), forces.astype(original_dtype)
