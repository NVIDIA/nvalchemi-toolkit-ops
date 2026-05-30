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

from typing import NamedTuple

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel


class ElectrostaticOutputs(NamedTuple):
    """Named electrostatic outputs used for internal composition."""

    energies: jax.Array
    forces: jax.Array | None = None
    charge_grads: jax.Array | None = None
    virial: jax.Array | None = None


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
    outputs: ElectrostaticOutputs,
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
) -> jax.Array | tuple[jax.Array, ...]:
    """Build an output tuple in electrostatics API order."""
    result = [outputs.energies]
    if compute_forces and outputs.forces is not None:
        result.append(outputs.forces)
    if compute_charge_gradients and outputs.charge_grads is not None:
        result.append(outputs.charge_grads)
    if compute_virial and outputs.virial is not None:
        result.append(outputs.virial)
    return tuple(result) if len(result) > 1 else result[0]


def _unpack_electrostatic_outputs(
    outputs: jax.Array | tuple[jax.Array, ...],
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
) -> ElectrostaticOutputs:
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

    return ElectrostaticOutputs(
        energies=energies,
        forces=forces,
        charge_grads=charge_grads,
        virial=virial,
    )


def _combine_electrostatic_outputs(
    real_outputs: jax.Array | tuple[jax.Array, ...],
    reciprocal_outputs: jax.Array | tuple[jax.Array, ...],
    slab_outputs: jax.Array | tuple[jax.Array, ...] | None,
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
) -> jax.Array | tuple[jax.Array, ...]:
    """Combine real, reciprocal, and optional slab outputs by named fields."""
    real = _unpack_electrostatic_outputs(
        real_outputs,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
    )
    reciprocal = _unpack_electrostatic_outputs(
        reciprocal_outputs,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
    )

    energies = real.energies + reciprocal.energies
    forces = (
        real.forces + reciprocal.forces
        if compute_forces and real.forces is not None and reciprocal.forces is not None
        else None
    )
    charge_grads = (
        real.charge_grads + reciprocal.charge_grads
        if compute_charge_gradients
        and real.charge_grads is not None
        and reciprocal.charge_grads is not None
        else None
    )
    virial = (
        real.virial + reciprocal.virial
        if compute_virial and real.virial is not None and reciprocal.virial is not None
        else None
    )
    combined = ElectrostaticOutputs(
        energies=energies,
        forces=forces,
        charge_grads=charge_grads,
        virial=virial,
    )

    if slab_outputs is not None:
        slab = _unpack_electrostatic_outputs(
            slab_outputs,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )
        combined = ElectrostaticOutputs(
            energies=combined.energies + slab.energies,
            forces=(
                combined.forces + slab.forces
                if compute_forces
                and combined.forces is not None
                and slab.forces is not None
                else combined.forces
            ),
            charge_grads=(
                combined.charge_grads + slab.charge_grads
                if compute_charge_gradients
                and combined.charge_grads is not None
                and slab.charge_grads is not None
                else combined.charge_grads
            ),
            virial=(
                combined.virial + slab.virial
                if compute_virial
                and combined.virial is not None
                and slab.virial is not None
                else combined.virial
            ),
        )

    return _build_electrostatic_result(
        combined,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
    )
