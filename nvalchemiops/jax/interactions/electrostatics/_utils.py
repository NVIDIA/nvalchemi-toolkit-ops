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


def _direct_output_deprecation_msg(fn: str) -> str:
    """Migration message for the deprecated direct-output flags on a full API."""
    return (
        f"The direct-output flags (compute_forces / compute_virial / "
        f"compute_charge_gradients / hybrid_forces) on {fn} are deprecated and "
        f"will be removed in a future release. Compute the energy and use "
        f"JAX autodiff on the energy instead:\n\n"
        f"    energy = {fn}(positions, charges, cell, ...).sum()\n"
        f"    # forces      = -dE/dR\n"
        f"    forces = -jax.grad(lambda pos: {fn}(pos, charges, cell, ...).sum())(positions)\n"
        f"    # row-vector displacement: positions_s = positions @ (I + strain)\n"
        f"    # virial = -dE/dstrain; stress = dE/dstrain / volume\n"
        f"    def energy_from_strain(strain):\n"
        f"        deform = jnp.eye(3, dtype=positions.dtype) + strain\n"
        f"        return {fn}(positions @ deform, charges, cell @ deform, ...).sum()\n"
        f"    grad_strain = jax.grad(energy_from_strain)(jnp.zeros((3, 3), dtype=positions.dtype))\n"
        f"    virial = -grad_strain\n"
        f"    # charge grad = dE/dq\n"
        f"    dE_dq = jax.grad(lambda chg: {fn}(positions, chg, cell, ...).sum())(charges)\n"
        f"    # hybrid q(R): keep charges = q(positions) in the graph and\n"
        f"    #             differentiate energy w.r.t. positions for the full\n"
        f"    #             dE/dR (including the dq/dR chain-rule term)."
    )


def _component_direct_output_deprecation_msg(fn: str, flags: tuple[str, ...]) -> str:
    """Migration message for deprecated training-style component outputs."""
    flag_text = " / ".join(flags)
    return (
        f"The component direct-output flag(s) {flag_text} on {fn} are deprecated "
        f"for differentiable training and will be removed in a future release. "
        f"Component compute_forces=True remains supported for no-autograd "
        f"MD/inference loops. For training, compute the energy and use "
        f"JAX autodiff on the energy instead."
    )
