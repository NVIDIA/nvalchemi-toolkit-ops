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

"""Shared utilities for JAX electrostatics bindings."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel


def _normalize_dtype(dtype):
    """Normalize dtype for kernel dictionary lookup.

    Parameters
    ----------
    dtype : dtype-like
        Input dtype from a JAX array.

    Returns
    -------
    jnp.float32 or jnp.float64
        Normalized JAX dtype for kernel lookup.
    """
    if dtype == jnp.float32 or str(dtype) == "float32":
        return jnp.float32
    if dtype == jnp.float64 or str(dtype) == "float64":
        return jnp.float64
    raise ValueError(f"Unsupported dtype: {dtype}")


def _make_jax_kernels(
    wp_overload_dict: dict,
    num_outputs: int,
    in_out_argnames: list[str],
) -> dict:
    """Maps a ``jax`` data type to ``warp``.

    Parameters
    ----------
    wp_overload_dict : dict
        Warp kernel overload dictionary keyed by wp.float32/wp.float64.
    num_outputs : int
        Number of output arrays returned by the kernel.
    in_out_argnames : list of str
        Names of in-place output arguments.

    Returns
    -------
    dict
        Dictionary mapping jnp.float32/jnp.float64 to jax_kernel instances.
    """
    jax_to_wp = {jnp.float32: wp.float32, jnp.float64: wp.float64}
    return {
        jax_dtype: jax_kernel(
            wp_overload_dict[wp_dtype],
            num_outputs=num_outputs,
            in_out_argnames=in_out_argnames,
            enable_backward=False,
        )
        for jax_dtype, wp_dtype in jax_to_wp.items()
    }


def _prepare_cell(cell: jax.Array) -> tuple[jax.Array, int]:
    """Normalize a cell array to shape ``(B, 3, 3)``."""
    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    if cell.ndim != 3 or cell.shape[1:] != (3, 3):
        raise ValueError(f"cell must have shape (3, 3) or (B, 3, 3), got {cell.shape}")
    return cell, cell.shape[0]


def _build_electrostatic_result(
    energies: jax.Array,
    forces: jax.Array | None,
    charge_grads: jax.Array | None,
    virial: jax.Array | None,
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
) -> jax.Array | tuple[jax.Array, ...]:
    """Build an output tuple in electrostatics API order."""
    result = [energies]
    if compute_forces and forces is not None:
        result.append(forces)
    if compute_charge_gradients and charge_grads is not None:
        result.append(charge_grads)
    if compute_virial and virial is not None:
        result.append(virial)
    return tuple(result) if len(result) > 1 else result[0]


def _unpack_electrostatic_outputs(
    outputs: jax.Array | tuple[jax.Array, ...],
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, jax.Array | None]:
    """Unpack electrostatics outputs by flag combination without cursor logic."""
    output_tuple = outputs if isinstance(outputs, tuple) else (outputs,)

    if compute_forces and compute_charge_gradients and compute_virial:
        energies, forces, charge_grads, virial = output_tuple
    elif compute_forces and compute_charge_gradients:
        energies, forces, charge_grads = output_tuple
        virial = None
    elif compute_forces and compute_virial:
        energies, forces, virial = output_tuple
        charge_grads = None
    elif compute_charge_gradients and compute_virial:
        energies, charge_grads, virial = output_tuple
        forces = None
    elif compute_forces:
        energies, forces = output_tuple
        charge_grads = None
        virial = None
    elif compute_charge_gradients:
        energies, charge_grads = output_tuple
        forces = None
        virial = None
    elif compute_virial:
        energies, virial = output_tuple
        forces = None
        charge_grads = None
    else:
        (energies,) = output_tuple
        forces = None
        charge_grads = None
        virial = None

    return energies, forces, charge_grads, virial


def _combine_electrostatic_outputs(
    real_outputs: jax.Array | tuple[jax.Array, ...],
    reciprocal_outputs: jax.Array | tuple[jax.Array, ...],
    slab_outputs: jax.Array | tuple[jax.Array, ...] | None,
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
) -> jax.Array | tuple[jax.Array, ...]:
    """Combine real, reciprocal, and optional slab outputs by named fields."""
    real_energies, real_forces, real_charge_grads, real_virial = (
        _unpack_electrostatic_outputs(
            real_outputs,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )
    )
    (
        reciprocal_energies,
        reciprocal_forces,
        reciprocal_charge_grads,
        reciprocal_virial,
    ) = _unpack_electrostatic_outputs(
        reciprocal_outputs,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
    )

    energies = real_energies + reciprocal_energies
    forces = (
        real_forces + reciprocal_forces
        if compute_forces and real_forces is not None and reciprocal_forces is not None
        else None
    )
    charge_grads = (
        real_charge_grads + reciprocal_charge_grads
        if compute_charge_gradients
        and real_charge_grads is not None
        and reciprocal_charge_grads is not None
        else None
    )
    virial = (
        real_virial + reciprocal_virial
        if compute_virial and real_virial is not None and reciprocal_virial is not None
        else None
    )

    if slab_outputs is not None:
        slab_energies, slab_forces, slab_charge_grads, slab_virial = (
            _unpack_electrostatic_outputs(
                slab_outputs,
                compute_forces,
                compute_charge_gradients,
                compute_virial,
            )
        )
        energies = energies + slab_energies
        forces = (
            forces + slab_forces
            if compute_forces and forces is not None and slab_forces is not None
            else forces
        )
        charge_grads = (
            charge_grads + slab_charge_grads
            if compute_charge_gradients
            and charge_grads is not None
            and slab_charge_grads is not None
            else charge_grads
        )
        virial = (
            virial + slab_virial
            if compute_virial and virial is not None and slab_virial is not None
            else virial
        )

    return _build_electrostatic_result(
        energies,
        forces,
        charge_grads,
        virial,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
    )
