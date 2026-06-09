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

"""Private JAX electrostatics autograd helpers."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

__all__: list[str] = []


def _cell_3d(cell: jnp.ndarray) -> tuple[jnp.ndarray, bool]:
    """Promote ``cell`` to ``(S, 3, 3)`` and report whether it was 2-D."""
    if cell.ndim == 2:
        return cell[jnp.newaxis, :, :], True
    return cell, False


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 5))
def _inject_charge_grad(
    energy: jnp.ndarray,
    charges: jnp.ndarray,
    charge_grad: jnp.ndarray,
    has_batch_idx: bool,
    batch_idx: jnp.ndarray,
    num_systems: int,
) -> jnp.ndarray:
    """Attach analytical charge gradients to an energy tensor.

    The forward pass is an identity on ``energy``. The custom VJP routes the
    incoming energy cotangent to ``charges`` through the supplied analytical
    ``charge_grad`` while leaving the kernel-computed value unchanged.

    Parameters
    ----------
    energy : jnp.ndarray
        Energy tensor returned by the public API.
    charges : jnp.ndarray
        Charge array that should receive the injected gradient.
    charge_grad : jnp.ndarray
        Analytical per-atom ``dE/dq`` values.
    has_batch_idx : bool
        Whether ``batch_idx`` contains real per-atom system ids.
    batch_idx : jnp.ndarray
        Per-atom system ids, or a zero-filled sentinel for single-system calls.

    Returns
    -------
    jnp.ndarray
        The original ``energy`` value.
    """
    del charges, charge_grad, has_batch_idx, batch_idx
    return energy


def _inject_charge_grad_fwd(
    energy: jnp.ndarray,
    charges: jnp.ndarray,
    charge_grad: jnp.ndarray,
    has_batch_idx: bool,
    batch_idx: jnp.ndarray,
    num_systems: int,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    """Forward rule for :func:`_inject_charge_grad`."""
    del charges, num_systems
    return energy, (charge_grad, batch_idx)


def _inject_charge_grad_bwd(
    has_batch_idx: bool,
    num_systems: int,
    residuals: tuple[jnp.ndarray, jnp.ndarray],
    grad_energy: jnp.ndarray,
) -> tuple[jnp.ndarray, ...]:
    """Backward rule for :func:`_inject_charge_grad`."""
    charge_grad, batch_idx = residuals
    del has_batch_idx, num_systems
    atom_grad = grad_energy.reshape(charge_grad.shape).astype(jnp.float64)
    grad_charges = charge_grad * atom_grad
    return (
        grad_energy,
        grad_charges,
        jnp.zeros_like(charge_grad),
        None,
    )


_inject_charge_grad.defvjp(_inject_charge_grad_fwd, _inject_charge_grad_bwd)


def _cell_grad_from_strain_virial(
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    batch_idx: jnp.ndarray | None,
    grad_positions: jnp.ndarray,
    virial: jnp.ndarray,
    grad_system: jnp.ndarray,
) -> jnp.ndarray:
    """Convert a displacement virial into a cell VJP consistent with strain tests.

    The direct virial kernels report row-vector displacement
    ``W = -dE/dstrain``. A custom VJP must return gradients with respect to the
    actual function arguments ``positions`` and ``cell``. For the documented
    row-vector displacement
    recipe ``positions_s = positions @ (I + strain)`` and
    ``cell_s = cell @ (I + strain)``, choose ``grad_cell`` so that

    ``positions.T @ grad_positions + cell.T @ grad_cell == -W``.

    This gives JAX the same strain-first virial/stress contract as the Torch
    energy path without treating the direct virial as a literal ``dE/dcell``.

    Parameters
    ----------
    positions : jnp.ndarray, shape (N, 3)
        Atomic positions passed to the energy function.
    cell : jnp.ndarray, shape (3, 3) or (S, 3, 3)
        Cell argument passed to the energy function.
    batch_idx : jnp.ndarray or None
        System index for each atom. ``None`` means a single system.
    grad_positions : jnp.ndarray, shape (N, 3)
        VJP with respect to positions.
    virial : jnp.ndarray, shape (S, 3, 3)
        Direct strain virial from the explicit derivative kernel.
    grad_system : jnp.ndarray, shape (S,)
        Per-system cotangent used to scale ``virial``.

    Returns
    -------
    jnp.ndarray
        Cell cotangent with the same rank as the input ``cell``.
    """
    cell_batched, squeezed = _cell_3d(cell)
    num_systems = cell_batched.shape[0]

    pos_outer = positions[:, :, jnp.newaxis] * grad_positions[:, jnp.newaxis, :]
    if batch_idx is None:
        pos_term = pos_outer.sum(axis=0, keepdims=True)
    else:
        pos_term = (
            jnp.zeros(
                (num_systems, 3, 3),
                dtype=grad_positions.dtype,
            )
            .at[batch_idx.astype(jnp.int32)]
            .add(pos_outer)
        )

    scaled_virial = (
        virial.astype(grad_positions.dtype)
        * grad_system.astype(grad_positions.dtype)[:, jnp.newaxis, jnp.newaxis]
    )
    rhs = -scaled_virial - pos_term
    grad_cell = jnp.linalg.solve(jnp.swapaxes(cell_batched, -1, -2), rhs)
    if squeezed:
        return grad_cell[0]
    return grad_cell
