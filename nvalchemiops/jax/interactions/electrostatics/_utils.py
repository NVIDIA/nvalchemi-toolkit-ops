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

from typing import Literal

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


_ENERGY_REDUCTIONS = ("atom", "system")


def _validate_energy_reduction(energy_reduction: Literal["atom", "system"]) -> None:
    """Validate the public ``energy_reduction`` keyword-only option.

    Parameters
    ----------
    energy_reduction : str
        Requested public energy layout. Must be a static Python string so
        that branching on it under ``jax.jit`` stays static (pass it via
        ``static_argnames`` when jitting).

    Raises
    ------
    ValueError
        If ``energy_reduction`` is not one of ``"atom"`` or ``"system"``.
    """
    if energy_reduction not in _ENERGY_REDUCTIONS:
        raise ValueError(
            f"energy_reduction must be one of {_ENERGY_REDUCTIONS}, "
            f"got {energy_reduction!r}"
        )


def _system_sum_from_atoms(
    values: jax.Array,
    batch_idx: jax.Array | None,
    num_systems: int,
) -> jax.Array:
    """Sum per-atom scalar values into one scalar per system.

    Uses a differentiable scatter-add so gradients flow back to ``values``
    unchanged; this is a uniform per-atom sum and does not weight atoms.
    """
    if batch_idx is None:
        return values.sum(keepdims=True)
    return (
        jnp.zeros((num_systems,), dtype=values.dtype)
        .at[batch_idx.astype(jnp.int32)]
        .add(values)
    )


def _per_system_atom_counts(
    batch_idx: jax.Array | None,
    num_systems: int,
    num_atoms: int,
) -> jax.Array:
    """Return per-system atom counts as float64."""
    if batch_idx is None:
        return jnp.full((num_systems,), float(num_atoms), dtype=jnp.float64)
    return (
        jnp.zeros((num_systems,), dtype=jnp.float64)
        .at[batch_idx.astype(jnp.int32)]
        .add(jnp.ones((num_atoms,), dtype=jnp.float64))
    )


def _distribute_system_values(
    system_values: jax.Array,
    batch_idx: jax.Array | None,
    num_atoms: int,
) -> jax.Array:
    """Distribute per-system values uniformly over each system's atoms."""
    if batch_idx is None:
        if num_atoms == 0:
            return jnp.zeros((0,), dtype=system_values.dtype)
        return jnp.full(
            (num_atoms,), system_values[0] / num_atoms, dtype=system_values.dtype
        )

    bidx = batch_idx.astype(jnp.int32)
    counts = _per_system_atom_counts(batch_idx, system_values.shape[0], num_atoms)
    return (system_values / jnp.maximum(counts, 1.0))[bidx]


def _apply_energy_reduction(
    result: jax.Array | tuple[jax.Array, ...],
    energy_reduction: Literal["atom", "system"],
    batch_idx: jax.Array | None,
    cell: jax.Array,
) -> jax.Array | tuple[jax.Array, ...]:
    """Apply the public ``energy_reduction`` layout to an atom-layout result.

    Called at the public wrapper boundary after existing custom-JVP,
    direct-output, or hybrid atom-layout operations have produced ``result``.
    Only the energy field changes: additional tuple entries (forces, charge
    gradients, virial) pass through unchanged. ``"system"`` reduction is a
    plain differentiable scatter-sum over atoms (uniform weights); it does
    not support nonuniform per-atom weighting.

    Parameters
    ----------
    result : jax.Array or tuple[jax.Array, ...]
        Atom-layout energy array, or a tuple whose first entry is the
        atom-layout energy array.
    energy_reduction : str
        Either ``"atom"`` (identity) or ``"system"`` (scatter-sum by
        ``batch_idx`` into shape ``(B,)``, or ``(1,)`` for a single system).
    batch_idx : jax.Array or None, shape (N,)
        System index per atom, or ``None`` for a single system.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrices, used only to infer the number of systems.

    Returns
    -------
    jax.Array or tuple[jax.Array, ...]
        ``result`` with its energy field reduced to the requested layout.
    """
    if energy_reduction == "atom":
        return result
    num_systems = 1 if cell.ndim == 2 else cell.shape[0]
    if isinstance(result, tuple):
        energies, *rest = result
        return (_system_sum_from_atoms(energies, batch_idx, num_systems), *rest)
    return _system_sum_from_atoms(result, batch_idx, num_systems)


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
