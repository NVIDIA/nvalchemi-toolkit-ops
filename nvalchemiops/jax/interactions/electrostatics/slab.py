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

"""JAX two-dimensional slab correction bindings."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nvalchemiops.interactions.electrostatics.slab_kernels import (
    _slab_correction_energy_forces_charge_grad_kernel_overload,
    _slab_correction_energy_forces_kernel_overload,
    _slab_correction_energy_kernel_overload,
    _slab_reduce_moments_kernel_overload,
)
from nvalchemiops.jax.interactions.electrostatics._utils import (
    _build_electrostatic_result,
    _make_jax_kernels,
    _normalize_dtype,
    _prepare_cell,
)

__all__ = ["compute_slab_correction"]


_jax_slab_reduce_moments = _make_jax_kernels(
    _slab_reduce_moments_kernel_overload,
    3,
    ["mz", "mz2", "qtotal"],
)

_jax_slab_correction_energy = _make_jax_kernels(
    _slab_correction_energy_kernel_overload,
    1,
    ["energy_out"],
)

_jax_slab_correction_energy_forces = _make_jax_kernels(
    _slab_correction_energy_forces_kernel_overload,
    3,
    ["energy_out", "forces", "virial"],
)

_jax_slab_correction_energy_forces_charge_grad = _make_jax_kernels(
    _slab_correction_energy_forces_charge_grad_kernel_overload,
    4,
    ["energy_out", "forces", "charge_grads", "virial"],
)


def _prepare_pbc_for_slab(pbc: jax.Array | None, num_systems: int) -> jax.Array:
    """Normalize and validate slab pbc as ``(B, 3)``."""
    if pbc is None:
        raise ValueError(
            "slab_correction=True requires an explicit `pbc` argument. "
            "Use a boolean array with shape (3,) for a single system or "
            "(B, 3) for batched systems."
        )

    pbc = jnp.asarray(pbc)
    if pbc.dtype != jnp.bool_:
        raise ValueError(f"pbc must be a bool array, got dtype={pbc.dtype}")

    if pbc.ndim == 1:
        if pbc.shape != (3,):
            raise ValueError(f"pbc must have shape (3,) or (B, 3), got {pbc.shape}")
        if num_systems != 1:
            raise ValueError(
                "batched slab correction requires pbc with shape (B, 3); "
                "shape (3,) is only valid for single-system calls"
            )
        return pbc[jnp.newaxis, :]

    if pbc.ndim != 2 or pbc.shape[1] != 3:
        raise ValueError(f"pbc must have shape (3,) or (B, 3), got {pbc.shape}")

    if pbc.shape[0] != num_systems:
        raise ValueError(
            f"pbc has {pbc.shape[0]} rows but cell describes {num_systems} systems"
        )

    return pbc


def compute_slab_correction(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
    batch_idx: jax.Array | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
) -> jax.Array | tuple[jax.Array, ...]:
    """Yeh-Berkowitz/Ballenegger slab correction for 2D periodic systems.

    Returns the standalone slab correction contribution for JAX electrostatics
    APIs. The caller can add the returned energy, force, charge-gradient, and
    virial terms to 3D-periodic Ewald or PME component outputs. This JAX binding
    is forward-only and exposes explicit outputs via flags.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    charges : jax.Array, shape (N,)
        Atomic charges.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrices.
    pbc : jax.Array, shape (3,) or (B, 3), dtype=bool
        Per-system periodic boundary conditions. True marks periodic directions
        and False marks the non-periodic slab direction. Systems whose pbc row
        is not slab-like contribute zero. A shape (3,) array is accepted only
        for single-system calls.
    batch_idx : jax.Array, shape (N,), dtype=int32, optional
        System index for each atom. Defaults to all zeros for a single system.
    compute_forces : bool, default=False
        If True, return per-atom slab forces.
    compute_charge_gradients : bool, default=False
        If True, return per-atom slab charge gradients dE_slab/dq_i.
    compute_virial : bool, default=False
        If True, return per-system slab virial tensors.

    Returns
    -------
    energies : jax.Array, shape (N,)
        Per-atom slab correction energy.
    forces : jax.Array, shape (N, 3), optional
        Per-atom slab force.
    charge_gradients : jax.Array, shape (N,), optional
        Per-atom slab charge gradient.
    virial : jax.Array, shape (B, 3, 3), optional
        Per-system slab virial tensor.
    """
    dtype = _normalize_dtype(positions.dtype)
    positions_cast = positions.astype(dtype)
    charges_cast = charges.astype(dtype)
    cell_cast, num_systems = _prepare_cell(cell.astype(dtype))
    pbc_cast = _prepare_pbc_for_slab(pbc, num_systems)
    num_atoms = positions_cast.shape[0]

    if batch_idx is None:
        batch_idx_i32 = jnp.zeros(num_atoms, dtype=jnp.int32)
    else:
        batch_idx_i32 = batch_idx.astype(jnp.int32)

    if num_atoms == 0:
        return _build_electrostatic_result(
            jnp.zeros((0,), dtype=jnp.float64),
            jnp.zeros((0, 3), dtype=dtype) if compute_forces else None,
            (jnp.zeros((0,), dtype=jnp.float64) if compute_charge_gradients else None),
            (jnp.zeros((num_systems, 3, 3), dtype=dtype) if compute_virial else None),
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    mz = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    mz2 = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    qtotal = jnp.zeros(num_systems, dtype=jnp.float64)
    mz, mz2, qtotal = _jax_slab_reduce_moments[dtype](
        positions_cast,
        charges_cast,
        batch_idx_i32,
        pbc_cast,
        cell_cast,
        mz,
        mz2,
        qtotal,
        launch_dims=(num_atoms,),
    )

    energy_in = jnp.zeros(num_atoms, dtype=jnp.float64)
    energy_out = jnp.zeros(num_atoms, dtype=jnp.float64)

    if compute_charge_gradients:
        forces = jnp.zeros((num_atoms, 3), dtype=dtype)
        charge_grads = jnp.zeros(num_atoms, dtype=jnp.float64)
        virial = jnp.zeros((num_systems, 3, 3), dtype=dtype)
        energy_out, forces, charge_grads, virial = (
            _jax_slab_correction_energy_forces_charge_grad[dtype](
                positions_cast,
                charges_cast,
                batch_idx_i32,
                pbc_cast,
                cell_cast,
                mz,
                mz2,
                qtotal,
                energy_in,
                energy_out,
                int(compute_virial),
                forces,
                charge_grads,
                virial,
                launch_dims=(num_atoms,),
            )
        )
        return _build_electrostatic_result(
            energy_out,
            forces,
            charge_grads,
            virial,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    if compute_forces or compute_virial:
        forces = jnp.zeros((num_atoms, 3), dtype=dtype)
        virial = jnp.zeros((num_systems, 3, 3), dtype=dtype)
        energy_out, forces, virial = _jax_slab_correction_energy_forces[dtype](
            positions_cast,
            charges_cast,
            batch_idx_i32,
            pbc_cast,
            cell_cast,
            mz,
            mz2,
            qtotal,
            energy_in,
            energy_out,
            int(compute_virial),
            forces,
            virial,
            launch_dims=(num_atoms,),
        )
        return _build_electrostatic_result(
            energy_out,
            forces,
            None,
            virial,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    (energy_out,) = _jax_slab_correction_energy[dtype](
        positions_cast,
        charges_cast,
        batch_idx_i32,
        pbc_cast,
        cell_cast,
        mz,
        mz2,
        qtotal,
        energy_in,
        energy_out,
        launch_dims=(num_atoms,),
    )
    return energy_out
