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

"""Shared utilities for electrostatics PyTorch bindings."""

from __future__ import annotations

from typing import NamedTuple

import torch

__all__ = [
    "ElectrostaticOutputs",
    "_InjectChargeGrad",
    "_build_electrostatic_result",
    "_combine_electrostatic_outputs",
    "_sum_charge_gradients",
    "_unpack_electrostatic_outputs",
]


class ElectrostaticOutputs(NamedTuple):
    """Named electrostatics outputs in public API order."""

    energies: torch.Tensor
    forces: torch.Tensor | None = None
    charge_grads: torch.Tensor | None = None
    virial: torch.Tensor | None = None


@torch.compiler.disable
def _sum_charge_gradients(
    real_space_charge_grads: torch.Tensor,
    reciprocal_charge_grads: torch.Tensor,
) -> torch.Tensor:
    """Sum electrostatic charge gradients eagerly on compiled paths."""
    return real_space_charge_grads + reciprocal_charge_grads


def _build_electrostatic_result(
    outputs: ElectrostaticOutputs,
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
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
    outputs: torch.Tensor | tuple[torch.Tensor, ...],
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

    return ElectrostaticOutputs(energies, forces, charge_grads, virial)


def _combine_electrostatic_outputs(
    real_outputs: torch.Tensor | tuple[torch.Tensor, ...],
    reciprocal_outputs: torch.Tensor | tuple[torch.Tensor, ...],
    slab_outputs: torch.Tensor | tuple[torch.Tensor, ...] | None,
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Combine real, reciprocal, and optional slab outputs by named fields."""
    real = _unpack_electrostatic_outputs(
        real_outputs, compute_forces, compute_charge_gradients, compute_virial
    )
    reciprocal = _unpack_electrostatic_outputs(
        reciprocal_outputs, compute_forces, compute_charge_gradients, compute_virial
    )

    energies = real.energies + reciprocal.energies
    forces = (
        real.forces + reciprocal.forces
        if compute_forces and real.forces is not None and reciprocal.forces is not None
        else None
    )

    if (
        compute_charge_gradients
        and real.charge_grads is not None
        and reciprocal.charge_grads is not None
    ):
        if torch.compiler.is_compiling():
            charge_grads = _sum_charge_gradients(
                real.charge_grads, reciprocal.charge_grads
            )
        else:
            charge_grads = real.charge_grads + reciprocal.charge_grads
    else:
        charge_grads = None

    virial = (
        real.virial + reciprocal.virial
        if compute_virial and real.virial is not None and reciprocal.virial is not None
        else None
    )

    if slab_outputs is not None:
        slab = _unpack_electrostatic_outputs(
            slab_outputs, compute_forces, compute_charge_gradients, compute_virial
        )
        energies = energies + slab.energies
        if compute_forces and forces is not None and slab.forces is not None:
            forces = forces + slab.forces
        if (
            compute_charge_gradients
            and charge_grads is not None
            and slab.charge_grads is not None
        ):
            charge_grads = charge_grads + slab.charge_grads
        if compute_virial and virial is not None and slab.virial is not None:
            virial = virial + slab.virial

    return _build_electrostatic_result(
        ElectrostaticOutputs(energies, forces, charge_grads, virial),
        compute_forces,
        compute_charge_gradients,
        compute_virial,
    )


class _InjectChargeGrad(torch.autograd.Function):
    """Inject analytical charge gradients into the autograd graph.

    A no-op in the forward pass (returns ``energy`` unchanged).  On backward,
    maps the per-system ``grad_energy`` to per-atom contributions using
    ``batch_idx`` and multiplies by the kernel-computed ``charge_grad``
    (dE/dq), so that ``energy.backward()`` propagates correct gradients
    through the charge pathway without a Warp backward tape.

    Parameters
    ----------
    energy : torch.Tensor
        Per-system energies, shape ``(S,)``.
    charges : torch.Tensor
        Charges with ``requires_grad=True``, shape ``(N,)``.
    charge_grad : torch.Tensor
        Analytical per-atom dE/dq from the forward kernel, shape ``(N,)``.
    batch_idx : torch.Tensor or None
        Per-atom system index, shape ``(N,)``.  ``None`` for single-system.
    """

    @staticmethod
    def forward(energy, charges, charge_grad, batch_idx):
        """Return energy unchanged."""
        return energy

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save charge_grad and batch_idx for backward."""
        _, _, charge_grad, batch_idx = inputs
        ctx.save_for_backward(charge_grad)
        ctx.batch_idx = batch_idx

    @staticmethod
    def backward(ctx, grad_energy):
        """Compute gradients for energy and charges."""
        (charge_grad,) = ctx.saved_tensors
        if ctx.batch_idx is not None:
            atom_grad = grad_energy.index_select(0, ctx.batch_idx)
        else:
            atom_grad = grad_energy.squeeze(0)
        return grad_energy, charge_grad * atom_grad, None, None
