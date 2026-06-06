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

"""Shared utilities for electrostatics PyTorch bindings.

This module owns the private autograd connector contracts:

* :class:`_InjectChargeGrad` -- a backward-compatible 4-argument shim that attaches
  analytical charge gradients to the public energy tensor.
"""

from __future__ import annotations

import torch

__all__ = [
    "_InjectCachedEvalGrad",
    "_InjectChargeGrad",
    "_build_electrostatic_result",
    "_combine_electrostatic_outputs",
    "_sum_charge_gradients",
    "_unpack_electrostatic_outputs",
]


@torch.compiler.disable
def _sum_charge_gradients(
    real_space_charge_grads: torch.Tensor,
    reciprocal_charge_grads: torch.Tensor,
) -> torch.Tensor:
    """Sum electrostatic charge gradients eagerly on compiled paths."""
    return real_space_charge_grads + reciprocal_charge_grads


def _build_electrostatic_result(
    energies: torch.Tensor,
    forces: torch.Tensor | None,
    charge_grads: torch.Tensor | None,
    virial: torch.Tensor | None,
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
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
    outputs: torch.Tensor | tuple[torch.Tensor, ...],
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
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
    real_outputs: torch.Tensor | tuple[torch.Tensor, ...],
    reciprocal_outputs: torch.Tensor | tuple[torch.Tensor, ...],
    slab_outputs: torch.Tensor | tuple[torch.Tensor, ...] | None,
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
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

    if (
        compute_charge_gradients
        and real_charge_grads is not None
        and reciprocal_charge_grads is not None
    ):
        if torch.compiler.is_compiling():
            charge_grads = _sum_charge_gradients(
                real_charge_grads, reciprocal_charge_grads
            )
        else:
            charge_grads = real_charge_grads + reciprocal_charge_grads
    else:
        charge_grads = None

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
        if compute_forces and forces is not None and slab_forces is not None:
            forces = forces + slab_forces
        if (
            compute_charge_gradients
            and charge_grads is not None
            and slab_charge_grads is not None
        ):
            charge_grads = charge_grads + slab_charge_grads
        if compute_virial and virial is not None and slab_virial is not None:
            virial = virial + slab_virial

    return _build_electrostatic_result(
        energies,
        forces,
        charge_grads,
        virial,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
    )


def _direct_output_deprecation_msg(fn: str) -> str:
    """Migration message for the deprecated direct-output flags on a FULL API."""
    return (
        f"The direct-output flags (compute_forces / compute_virial / "
        f"compute_charge_gradients / hybrid_forces) on {fn} are deprecated and "
        f"will be removed in a future release. Compute the energy and use "
        f"torch.autograd.grad on the energy instead:\n\n"
        f"    energy = {fn}(positions, charges, cell, ...).sum()\n"
        f"    # forces      = -dE/dR\n"
        f"    forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]\n"
        f"    # virial      = -dE/dstrain  (strain an (S,3,3) tensor; deform pos & cell by I+strain)\n"
        f"    virial = -torch.autograd.grad(energy, strain)[0]\n"
        f"    # charge grad = dE/dq\n"
        f"    dE_dq = torch.autograd.grad(energy, charges)[0]\n"
        f"    # hybrid q(R): keep charges = q(positions) in the graph and\n"
        f"    #             differentiate energy w.r.t. positions for the full\n"
        f"    #             dE/dR (including the dq/dR chain-rule term)."
    )


def _num_atoms_from_state(
    pos_grad_state: torch.Tensor | None,
    charge_grad_state: torch.Tensor | None,
    batch_idx: torch.Tensor | None,
) -> int:
    """Infer the atom count for per-atom cotangent broadcasting."""
    if charge_grad_state is not None:
        return int(charge_grad_state.shape[0])
    if pos_grad_state is not None:
        return int(pos_grad_state.shape[0])
    if batch_idx is not None:
        return int(batch_idx.shape[0])
    return 0


def _num_systems_from_state(
    grad_energy: torch.Tensor,
    cell_grad_state: torch.Tensor | None,
    batch_idx: torch.Tensor | None,
    num_atoms: int,
) -> int:
    """Infer the system count for per-system cotangent reduction."""
    if cell_grad_state is not None:
        return int(cell_grad_state.shape[0])
    if batch_idx is None:
        return 1
    if grad_energy.numel() != num_atoms:
        return int(grad_energy.numel())
    if batch_idx.numel() == 0:
        return 0
    return int(batch_idx.max().item()) + 1


def _energy_cotangents(
    grad_energy: torch.Tensor,
    batch_idx: torch.Tensor | None,
    num_atoms: int,
    num_systems: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-system and per-atom cotangents for injected derivatives."""
    grad = grad_energy.reshape(-1)
    if batch_idx is None:
        if num_systems > 1 and grad.numel() == num_systems:
            grad_system = grad
        elif num_systems > 1 and grad.numel() == 1:
            grad_system = grad.expand(num_systems)
        elif grad.numel() == num_atoms and num_atoms > 0:
            grad_system = grad.mean().reshape(1)
        elif grad.numel() == 0:
            grad_system = grad.new_zeros((1,))
        else:
            grad_system = grad.reshape(-1)[:1]
        atom_grad = grad_system[0].expand(num_atoms)
        return grad_system, atom_grad

    bidx = batch_idx
    if grad.numel() == num_systems:
        grad_system = grad
    elif grad.numel() == num_atoms:
        sums = grad.new_zeros((num_systems,))
        sums = sums.index_add(0, bidx, grad)
        counts = grad.new_zeros((num_systems,))
        counts = counts.index_add(0, bidx, torch.ones_like(grad))
        grad_system = sums / counts.clamp_min(1)
    elif grad.numel() == 1:
        grad_system = grad.expand(num_systems)
    else:
        raise RuntimeError(
            "Energy cotangent must be per-system, per-atom, or scalar for "
            "_InjectChargeGrad backward"
        )
    atom_grad = grad_system.index_select(0, bidx)
    return grad_system, atom_grad


def _is_uniform_cotangent(grad_energy: torch.Tensor) -> bool:
    """Return whether ``grad_energy`` is exactly uniform."""
    grad = grad_energy.reshape(-1)
    if grad.numel() <= 1:
        return True
    if grad.is_cuda:
        return False
    return bool(torch.all(grad == grad[0]).item())


class _InjectCachedEvalGrad(torch.autograd.Function):
    """Cut the eager graph for uniform first-order eval gradients.

    The forward is an identity over ``energy``. During ordinary first-order
    evaluation (``create_graph=False``), a uniform energy cotangent such as the
    one produced by ``energy.sum()`` can be served from direct derivative caches.
    During training / double-backward, or for non-uniform per-atom energy
    weights, the cotangent is passed through to the eager graph unchanged.
    """

    @staticmethod
    def forward(
        energy,
        positions,
        charges,
        cell,
        pos_grad_state,
        charge_grad_state,
        cell_grad_state,
        batch_idx,
    ):
        """Return energy unchanged."""
        return energy

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save detached direct-derivative caches for the eval branch."""
        (
            _energy,
            _positions,
            _charges,
            _cell,
            pos_grad_state,
            charge_grad_state,
            cell_grad_state,
            batch_idx,
        ) = inputs
        ctx.save_for_backward(pos_grad_state, charge_grad_state, cell_grad_state)
        ctx.batch_idx = batch_idx

    @staticmethod
    def backward(ctx, grad_energy):
        """Use cached derivatives for uniform eval, else pass to eager energy."""
        pos_grad_state, charge_grad_state, cell_grad_state = ctx.saved_tensors

        if torch.is_grad_enabled() or not _is_uniform_cotangent(grad_energy):
            return grad_energy, None, None, None, None, None, None, None

        num_atoms = _num_atoms_from_state(
            pos_grad_state, charge_grad_state, ctx.batch_idx
        )
        num_systems = _num_systems_from_state(
            grad_energy, cell_grad_state, ctx.batch_idx, num_atoms
        )
        grad_system, atom_grad = _energy_cotangents(
            grad_energy, ctx.batch_idx, num_atoms, num_systems
        )

        grad_positions = None
        if pos_grad_state is not None:
            grad_positions = pos_grad_state * atom_grad.unsqueeze(-1)

        grad_charges = None
        if charge_grad_state is not None:
            grad_charges = charge_grad_state * atom_grad

        grad_cell = None
        if cell_grad_state is not None:
            grad_cell = cell_grad_state * grad_system.view(-1, 1, 1)

        return None, grad_positions, grad_charges, grad_cell, None, None, None, None


class _InjectChargeGrad(torch.autograd.Function):
    """Backward-compatible 4-argument charge-gradient entry point.

    Per-system energy cotangents keep the historical path; public per-atom cotangents
    are reduced to per-system means before injection.

    Parameters
    ----------
    energy : torch.Tensor
        Energy tensor, either public per-atom ``(N,)`` or per-system ``(S,)``.
    charges : torch.Tensor
        Charges with ``requires_grad=True``, shape ``(N,)``.
    charge_grad : torch.Tensor
        Analytical per-atom ``dE/dq`` from the forward kernel, shape ``(N,)``.
    batch_idx : torch.Tensor or None
        Per-atom system index, shape ``(N,)``. ``None`` for single-system.
    """

    @staticmethod
    def forward(energy, charges, charge_grad, batch_idx):
        """Return energy unchanged."""
        return energy

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save detached charge-gradient state for backward."""
        _energy, _charges, charge_grad, batch_idx = inputs
        ctx.save_for_backward(charge_grad)
        ctx.batch_idx = batch_idx

    @staticmethod
    def backward(ctx, grad_energy):
        """Scale analytical ``dE/dq`` by the energy cotangent."""
        (charge_grad_state,) = ctx.saved_tensors
        num_atoms = int(charge_grad_state.shape[0])
        num_systems = _num_systems_from_state(
            grad_energy, None, ctx.batch_idx, num_atoms
        )
        _grad_system, atom_grad = _energy_cotangents(
            grad_energy, ctx.batch_idx, num_atoms, num_systems
        )
        grad_charges = charge_grad_state * atom_grad
        return grad_energy, grad_charges, None, None
