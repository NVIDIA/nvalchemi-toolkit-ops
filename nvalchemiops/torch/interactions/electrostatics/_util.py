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

from typing import Literal

import torch

_CotangentLayout = Literal["atom", "system"]

__all__ = [
    "_has_potentially_geometry_dependent_charges",
    "_InjectCachedEvalGrad",
    "_InjectCachedEvalGradWithFallback",
    "_InjectChargeGrad",
    "_reduce_atom_energy",
    "_validate_energy_reduction",
    "_build_electrostatic_result",
    "_combine_electrostatic_outputs",
    "_compiled_direct_output_deprecation_signal",
    "_detach_setup_tensor",
    "_sum_charge_gradients",
    "_unpack_electrostatic_outputs",
]


def _validate_energy_reduction(energy_reduction: str) -> str:
    """Validate and return the public electrostatics energy layout."""
    if energy_reduction not in {"atom", "system"}:
        raise ValueError(
            "energy_reduction must be either 'atom' or 'system', "
            f"got {energy_reduction!r}"
        )
    return energy_reduction


def _reduce_atom_energy(
    energy: torch.Tensor,
    batch_idx: torch.Tensor | None,
    num_systems: int,
) -> torch.Tensor:
    """Sum an atom-major energy tensor into an explicit system-major layout."""
    flat = energy.reshape(-1)
    result = flat.new_zeros((num_systems,))
    if flat.numel() == 0 or num_systems == 0:
        return result
    if batch_idx is None:
        result[0] = flat.sum()
        return result
    return result.index_add(0, batch_idx.to(device=flat.device, dtype=torch.long), flat)


def _system_cotangent_to_atoms(
    grad_system: torch.Tensor,
    batch_idx: torch.Tensor | None,
    num_atoms: int,
) -> torch.Tensor:
    """Broadcast an explicit system cotangent to atom-major layout."""
    grad = grad_system.reshape(-1)
    if num_atoms == 0:
        return grad.new_zeros((0,))
    if batch_idx is None:
        return grad[0].expand(num_atoms)
    return grad.index_select(0, batch_idx.to(device=grad.device, dtype=torch.long))


def _sum_charge_gradients(
    real_space_charge_grads: torch.Tensor,
    reciprocal_charge_grads: torch.Tensor,
) -> torch.Tensor:
    """Sum electrostatic charge gradients with traceable Torch arithmetic."""
    return real_space_charge_grads + reciprocal_charge_grads


def _has_potentially_geometry_dependent_charges(
    positions: torch.Tensor,
    charges: torch.Tensor,
) -> bool:
    """Return whether ``charges`` may carry a q(R) autograd path.

    A non-leaf charge tensor may depend on positions, e.g. charges = q(positions).
    Custom backwards must avoid differentiating a connected recompute with respect
    to both positions and charges and then returning both gradients, because
    PyTorch would apply dE/dq * dq/dR twice.

    Returns
    -------
    bool
        True when ``positions`` and ``charges`` both require gradients and
        ``charges`` is a non-leaf tensor that may depend on ``positions``.

    Notes
    -----
    Issue #115 is prevented by routing these workloads through
    :class:`_InjectChargeGrad`, eager electrostatics chains, or safe cached
    fallback paths that recompute independent partial derivatives before PyTorch
    applies the dE/dq * dq/dR chain term exactly once.
    """
    return bool(
        positions.requires_grad
        and charges.requires_grad
        and charges.grad_fn is not None
    )


def _detach_setup_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Detach optional setup/cache tensors from public autograd outputs."""
    return None if tensor is None else tensor.detach()


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
        charge_grads = _sum_charge_gradients(
            real_charge_grads,
            reciprocal_charge_grads,
        )
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
        f"    strain = torch.zeros(3, 3, dtype=positions.dtype, device=positions.device,\n"
        f"                         requires_grad=True)\n"
        f"    deformation = torch.eye(3, dtype=positions.dtype, device=positions.device) + strain\n"
        f"    positions_s = positions @ deformation\n"
        f"    cell_s = cell @ deformation\n"
        f"    energy = {fn}(positions_s, charges, cell_s, ...).sum()\n"
        f"    # forces      = -dE/dR\n"
        f"    forces = -torch.autograd.grad(energy, positions_s, create_graph=True)[0]\n"
        f"    # row-vector displacement: positions_s = positions @ (I + strain)\n"
        f"    # virial = -dE/dstrain; stress = dE/dstrain / volume\n"
        f"    grad_strain = torch.autograd.grad(energy, strain)[0]\n"
        f"    virial = -grad_strain\n"
        f"    # charge grad = dE/dq\n"
        f"    dE_dq = torch.autograd.grad(energy, charges)[0]\n"
        f"    # hybrid q(R): keep charges = q(positions) in the graph and\n"
        f"    #             differentiate energy w.r.t. positions for the full\n"
        f"    #             dE/dR (including the dq/dR chain-rule term)."
    )


def _compiled_direct_output_deprecation_signal(fn: str) -> None:
    """Emit a compile-safe migration signal for deprecated full-API direct outputs."""
    if torch.compiler.is_compiling():
        torch._dynamo.graph_break(_direct_output_deprecation_msg(fn))


def _component_direct_output_deprecation_msg(fn: str, flags: tuple[str, ...]) -> str:
    """Migration message for deprecated training-style component outputs."""
    flag_text = " / ".join(flags)
    return (
        f"The component direct-output flag(s) {flag_text} on {fn} are deprecated "
        f"for differentiable training and will be removed in a future release. "
        f"Component compute_forces=True remains supported for no-autograd "
        f"MD/inference loops. For training, compute the energy and use "
        f"torch.autograd.grad on the energy instead."
    )


def _sum_atom_values_by_system(
    values: torch.Tensor,
    batch_idx: torch.Tensor | None,
    num_systems: int,
) -> torch.Tensor:
    """Reduce atom-major values to per-system sums."""
    if num_systems == 1:
        return values.sum(dim=0, keepdim=True)

    if batch_idx is None:
        raise RuntimeError("batch_idx is required for multi-system atom reduction")

    result = values.new_zeros((num_systems, *values.shape[1:]))
    return result.index_add(0, batch_idx, values)


def _mean_atom_values_by_system(
    values: torch.Tensor,
    batch_idx: torch.Tensor | None,
    num_systems: int,
) -> torch.Tensor:
    """Reduce atom-major values to guarded per-system means."""
    if values.shape[0] == 0:
        return values.new_zeros((num_systems, *values.shape[1:]))
    if num_systems == 1:
        return values.mean(dim=0, keepdim=True)

    sums = _sum_atom_values_by_system(values, batch_idx, num_systems)
    counts = _sum_atom_values_by_system(
        values.new_ones((values.shape[0],)),
        batch_idx,
        num_systems,
    )
    count_shape = (num_systems, *([1] * (values.ndim - 1)))
    return sums / counts.clamp_min(1).reshape(count_shape)


def _broadcast_system_values_to_atoms(
    per_system: torch.Tensor,
    batch_idx: torch.Tensor | None,
    num_systems: int,
    num_atoms: int,
) -> torch.Tensor:
    """Broadcast per-system values to atom-major values."""
    if num_atoms == 0:
        return per_system.new_empty((0, *per_system.shape[1:]))
    if num_systems == 1:
        return per_system[0].expand((num_atoms, *per_system.shape[1:]))

    if batch_idx is None:
        raise RuntimeError("batch_idx is required for multi-system atom broadcast")

    return per_system.index_select(0, batch_idx)


def _distribute_system_mean_cotangent_to_atoms(
    per_system: torch.Tensor,
    batch_idx: torch.Tensor | None,
    num_systems: int,
    num_atoms: int,
) -> torch.Tensor:
    """Apply the adjoint of per-system mean reduction."""
    if num_atoms == 0:
        return per_system.new_empty((0, *per_system.shape[1:]))
    if num_systems == 1:
        return (
            per_system[0]
            .div(float(num_atoms))
            .expand((num_atoms, *per_system.shape[1:]))
            .clone()
        )

    counts = _sum_atom_values_by_system(
        per_system.new_ones((num_atoms,)),
        batch_idx,
        num_systems,
    )
    count_shape = (num_systems, *([1] * (per_system.ndim - 1)))
    scaled = per_system / counts.clamp_min(1).reshape(count_shape)
    return _broadcast_system_values_to_atoms(
        scaled,
        batch_idx,
        num_systems,
        num_atoms,
    )


def _energy_cotangents(
    grad_energy: torch.Tensor,
    batch_idx: torch.Tensor | None,
    num_atoms: int,
    num_systems: int,
    layout: _CotangentLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return explicit per-system and per-atom cotangents for injected derivatives."""
    grad = grad_energy.reshape(-1)
    if layout == "atom":
        if grad.numel() != num_atoms:
            raise RuntimeError(
                "Atom-major energy cotangent must have "
                f"{num_atoms} values, got {grad.numel()}"
            )
        atom_grad = grad
        if batch_idx is None:
            grad_system = (
                grad.mean().reshape(1) if num_atoms > 0 else grad.new_zeros((1,))
            )
            return grad_system, atom_grad

        bidx = batch_idx.to(device=grad.device, dtype=torch.long)
        grad_system = _mean_atom_values_by_system(grad, bidx, num_systems)
        return grad_system, atom_grad

    if layout != "system":
        raise ValueError(f"Unsupported energy cotangent layout {layout!r}")
    if grad.numel() != num_systems:
        raise RuntimeError(
            "System-major energy cotangent must have "
            f"{num_systems} values, got {grad.numel()}"
        )
    grad_system = grad
    atom_grad = _system_cotangent_to_atoms(grad_system, batch_idx, num_atoms)
    return grad_system, atom_grad


def _is_uniform_cotangent(grad_energy: torch.Tensor) -> bool:
    """Return whether ``grad_energy`` is exactly uniform."""
    if _is_sync_free_uniform_cotangent(grad_energy):
        return True
    grad = grad_energy.reshape(-1)
    if grad.numel() == 0:
        return True
    if not _can_inspect_cotangent_values(grad):
        return False
    return bool(torch.all(grad == grad[0]).item())


def _is_sync_free_uniform_cotangent(grad_energy: torch.Tensor) -> bool:
    """Return whether ``grad_energy`` is known uniform without reading values."""
    if grad_energy.numel() <= 1:
        return True
    return all(
        size <= 1 or stride == 0
        for size, stride in zip(grad_energy.shape, grad_energy.stride(), strict=True)
    )


def _can_inspect_cotangent_values(grad_energy: torch.Tensor) -> bool:
    """Whether eager code may perform an exact device-value uniformity check."""
    if not grad_energy.is_cuda:
        return True
    if torch.compiler.is_compiling():
        return False
    try:
        return not torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


def _is_per_system_uniform_cotangent(
    grad_energy: torch.Tensor,
    batch_idx: torch.Tensor | None,
    num_systems: int,
) -> bool:
    """Return whether an atom-major cotangent is uniform within each system."""
    grad = grad_energy.reshape(-1)
    if grad.numel() == 0:
        return True
    if batch_idx is not None and grad.numel() != batch_idx.numel():
        return False
    if _is_sync_free_uniform_cotangent(grad_energy):
        return True
    if not _can_inspect_cotangent_values(grad):
        return False
    if batch_idx is None:
        return bool(torch.all(grad == grad[0]).item())
    if grad.numel() != batch_idx.numel():
        return False

    idx = batch_idx.to(device=grad.device, dtype=torch.long)
    grad64 = grad.to(torch.float64)
    sys_min = torch.full(
        (num_systems,), float("inf"), dtype=torch.float64, device=grad.device
    ).scatter_reduce(0, idx, grad64, reduce="amin", include_self=False)
    sys_max = torch.full(
        (num_systems,), float("-inf"), dtype=torch.float64, device=grad.device
    ).scatter_reduce(0, idx, grad64, reduce="amax", include_self=False)
    return bool(
        torch.all(sys_min.index_select(0, idx) == sys_max.index_select(0, idx)).item()
    )


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
        energy_reduction="atom",
        num_systems=None,
    ):
        """Return energy in the requested public layout."""
        if energy_reduction == "system":
            count = int(cell.shape[0]) if num_systems is None else int(num_systems)
            return _reduce_atom_energy(energy, batch_idx, count)
        return energy

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save detached direct-derivative caches for the eval branch."""
        base_inputs = inputs[:8]
        (
            _energy,
            positions,
            _charges,
            cell,
            pos_grad_state,
            charge_grad_state,
            cell_grad_state,
            batch_idx,
        ) = base_inputs
        ctx.save_for_backward(pos_grad_state, charge_grad_state, cell_grad_state)
        ctx.batch_idx = batch_idx
        ctx.num_atoms = int(positions.shape[0])
        ctx.energy_reduction = inputs[8] if len(inputs) > 8 else "atom"
        ctx.num_systems = (
            int(inputs[9])
            if len(inputs) > 9 and inputs[9] is not None
            else (int(cell.shape[0]) if cell.dim() == 3 else 1)
        )
        ctx.num_inputs = len(inputs)

    @staticmethod
    def backward(ctx, grad_energy):
        """Use cached derivatives for uniform eval, else pass to eager energy."""
        pos_grad_state, charge_grad_state, cell_grad_state = ctx.saved_tensors
        num_atoms = ctx.num_atoms

        if ctx.energy_reduction == "system":
            grad_system, atom_grad = _energy_cotangents(
                grad_energy,
                ctx.batch_idx,
                num_atoms,
                ctx.num_systems,
                "system",
            )
            if torch.is_grad_enabled():
                result = (atom_grad, None, None, None, None, None, None, None)
                return result + (None,) * (ctx.num_inputs - len(result))
        elif torch.is_grad_enabled() or not _is_per_system_uniform_cotangent(
            grad_energy, ctx.batch_idx, ctx.num_systems
        ):
            result = (grad_energy, None, None, None, None, None, None, None)
            return result + (None,) * (ctx.num_inputs - len(result))

        if ctx.energy_reduction == "system":
            pass
        else:
            grad_system, atom_grad = _energy_cotangents(
                grad_energy,
                ctx.batch_idx,
                num_atoms,
                ctx.num_systems,
                "atom",
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

        result = (
            None,
            grad_positions,
            grad_charges,
            grad_cell,
            None,
            None,
            None,
            None,
        )
        return result + (None,) * (ctx.num_inputs - len(result))


class _InjectCachedEvalGradWithFallback(torch.autograd.Function):
    """Lazy variant of :class:`_InjectCachedEvalGrad`.

    The forward takes a detached/eval energy plus cached first derivatives. For
    ordinary uniform first-order losses it uses the caches. For create-graph or
    non-uniform weighted losses it calls ``fallback_fn(positions, charges, cell)``
    inside backward and differentiates that true energy graph.
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
        fallback_fn,
        energy_reduction="atom",
        num_systems=None,
        force_fallback=False,
        fallback_returns_system=False,
    ):
        """Return energy in the requested public layout."""
        if energy_reduction == "system":
            count = int(cell.shape[0]) if num_systems is None else int(num_systems)
            return _reduce_atom_energy(energy, batch_idx, count)
        return energy

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save inputs, caches, and the fallback callable."""
        base_inputs = inputs[:9]
        (
            _energy,
            positions,
            charges,
            cell,
            pos_grad_state,
            charge_grad_state,
            cell_grad_state,
            batch_idx,
            fallback_fn,
        ) = base_inputs
        ctx.save_for_backward(
            positions,
            charges,
            cell,
            pos_grad_state,
            charge_grad_state,
            cell_grad_state,
        )
        ctx.batch_idx = batch_idx
        ctx.fallback_fn = fallback_fn
        ctx.num_atoms = int(positions.shape[0])
        ctx.energy_reduction = inputs[9] if len(inputs) > 9 else "atom"
        ctx.num_systems = (
            int(inputs[10])
            if len(inputs) > 10 and inputs[10] is not None
            else (int(cell.shape[0]) if cell.dim() == 3 else 1)
        )
        ctx.force_fallback = bool(inputs[11]) if len(inputs) > 11 else False
        ctx.fallback_returns_system = bool(inputs[12]) if len(inputs) > 12 else False
        ctx.num_inputs = len(inputs)

    @staticmethod
    def backward(ctx, grad_energy):
        """Use caches for uniform eval, else lazily recompute the energy graph."""
        create_graph = torch.is_grad_enabled()
        (
            positions,
            charges,
            cell,
            pos_grad_state,
            charge_grad_state,
            cell_grad_state,
        ) = ctx.saved_tensors
        num_atoms = ctx.num_atoms

        def _pad(result):
            return result + (None,) * (ctx.num_inputs - len(result))

        def _public_fallback_energy(recomputed):
            if ctx.energy_reduction == "system" and not ctx.fallback_returns_system:
                return _reduce_atom_energy(recomputed, ctx.batch_idx, ctx.num_systems)
            return recomputed

        if ctx.energy_reduction == "system":
            grad_system, atom_grad = _energy_cotangents(
                grad_energy,
                ctx.batch_idx,
                num_atoms,
                ctx.num_systems,
                "system",
            )
            use_fallback = create_graph or ctx.force_fallback
        else:
            use_fallback = create_graph or not _is_per_system_uniform_cotangent(
                grad_energy, ctx.batch_idx, ctx.num_systems
            )

        if use_fallback:
            # q(R) routing:
            # - create_graph: return connected position/cell/alpha gradients only; do not
            #   also return grad_charges from the same connected recompute.
            # - first-order weighted loss: recompute position/cell/alpha partials with
            #   charges detached, then return dE/dq separately so PyTorch chains q(R) once.
            if _has_potentially_geometry_dependent_charges(positions, charges):
                if create_graph:
                    diff_inputs = []
                    diff_names = []
                    for name, tensor in (
                        ("positions", positions),
                        ("cell", cell),
                    ):
                        if tensor.requires_grad:
                            diff_inputs.append(tensor)
                            diff_names.append(name)
                    with torch.enable_grad():
                        recomputed = _public_fallback_energy(
                            ctx.fallback_fn(positions, charges, cell)
                        )
                        if diff_inputs:
                            diff_grads = torch.autograd.grad(
                                recomputed,
                                tuple(diff_inputs),
                                grad_outputs=grad_energy,
                                allow_unused=True,
                                create_graph=True,
                            )
                            grad_map = dict(zip(diff_names, diff_grads, strict=True))
                        else:
                            grad_map = {}
                    return _pad(
                        (
                            None,
                            grad_map.get("positions"),
                            None,
                            grad_map.get("cell"),
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                    )

                partial_inputs = []
                partial_names = []
                if positions.requires_grad:
                    partial_inputs.append(positions)
                    partial_names.append("positions")
                if cell.requires_grad:
                    partial_inputs.append(cell)
                    partial_names.append("cell")

                with torch.enable_grad():
                    partial_map = {}
                    if partial_inputs:
                        recomputed_partial = _public_fallback_energy(
                            ctx.fallback_fn(
                                positions,
                                charges.detach(),
                                cell,
                            )
                        )
                        partial_grads = torch.autograd.grad(
                            recomputed_partial,
                            tuple(partial_inputs),
                            grad_outputs=grad_energy,
                            allow_unused=True,
                            create_graph=create_graph,
                        )
                        partial_map = dict(
                            zip(partial_names, partial_grads, strict=True)
                        )

                    grad_charges = None
                    if charges.requires_grad:
                        recomputed_charge = _public_fallback_energy(
                            ctx.fallback_fn(positions, charges, cell)
                        )
                        (grad_charges,) = torch.autograd.grad(
                            recomputed_charge,
                            charges,
                            grad_outputs=grad_energy,
                            allow_unused=True,
                            create_graph=create_graph,
                        )
                return _pad(
                    (
                        None,
                        partial_map.get("positions"),
                        grad_charges,
                        partial_map.get("cell"),
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                )
            with torch.enable_grad():
                recomputed = _public_fallback_energy(
                    ctx.fallback_fn(positions, charges, cell)
                )
                diff_inputs = []
                diff_names = []
                for name, tensor in (
                    ("positions", positions),
                    ("charges", charges),
                    ("cell", cell),
                ):
                    if tensor.requires_grad:
                        diff_inputs.append(tensor)
                        diff_names.append(name)
                if not diff_inputs:
                    grad_map = {}
                else:
                    diff_grads = torch.autograd.grad(
                        recomputed,
                        tuple(diff_inputs),
                        grad_outputs=grad_energy,
                        allow_unused=True,
                        create_graph=create_graph,
                    )
                    grad_map = dict(zip(diff_names, diff_grads, strict=True))
            return _pad(
                (
                    None,
                    grad_map.get("positions"),
                    grad_map.get("charges"),
                    grad_map.get("cell"),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )

        if ctx.energy_reduction == "system":
            pass
        else:
            grad_system, atom_grad = _energy_cotangents(
                grad_energy,
                ctx.batch_idx,
                num_atoms,
                ctx.num_systems,
                "atom",
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

        return _pad(
            (
                None,
                grad_positions,
                grad_charges,
                grad_cell,
                None,
                None,
                None,
                None,
                None,
            )
        )


class _InjectChargeGrad(torch.autograd.Function):
    """Attach cached charge derivatives with an explicit energy layout.

    Uniform/per-system-uniform cotangents keep the historical direct-injection
    path. Non-uniform per-atom cotangents pass through to the input energy graph
    so weighted losses differentiate the real per-atom energy expression rather
    than a post-hoc average of cached total-energy charge gradients.

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
    def forward(
        energy,
        charges,
        charge_grad,
        batch_idx,
        energy_reduction="atom",
        num_systems=None,
        energy_is_system=False,
    ):
        """Return energy in the requested public layout."""
        if energy_reduction == "system" and not energy_is_system:
            count = 1 if num_systems is None else int(num_systems)
            return _reduce_atom_energy(energy, batch_idx, count)
        return energy

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save detached charge-gradient state for backward."""
        _energy, _charges, charge_grad, batch_idx = inputs[:4]
        ctx.save_for_backward(charge_grad)
        ctx.batch_idx = batch_idx
        ctx.energy_reduction = inputs[4] if len(inputs) > 4 else "atom"
        ctx.energy_is_system = bool(inputs[6]) if len(inputs) > 6 else False
        ctx.energy_is_system = ctx.energy_is_system or (
            _energy.numel() != charge_grad.shape[0]
        )
        ctx.output_layout = (
            "system"
            if ctx.energy_reduction == "system" or ctx.energy_is_system
            else "atom"
        )
        if len(inputs) > 5 and inputs[5] is not None:
            ctx.num_systems = int(inputs[5])
        elif ctx.energy_is_system:
            ctx.num_systems = int(_energy.numel())
        elif batch_idx is not None and batch_idx.numel() > 0:
            ctx.num_systems = int(batch_idx.max().item()) + 1
        else:
            ctx.num_systems = 1
        ctx.num_inputs = len(inputs)

    @staticmethod
    def backward(ctx, grad_energy):
        """Scale analytical ``dE/dq`` by the energy cotangent."""
        (charge_grad_state,) = ctx.saved_tensors
        num_atoms = int(charge_grad_state.shape[0])

        if ctx.output_layout == "system":
            grad_system, atom_grad = _energy_cotangents(
                grad_energy,
                ctx.batch_idx,
                num_atoms,
                ctx.num_systems,
                "system",
            )
            input_grad = grad_system if ctx.energy_is_system else atom_grad
        else:
            input_grad = grad_energy

        if torch.is_grad_enabled():
            result = (input_grad, None, None, None)
            return result + (None,) * (ctx.num_inputs - len(result))

        if ctx.output_layout == "system":
            pass
        else:
            if not _is_per_system_uniform_cotangent(
                grad_energy, ctx.batch_idx, ctx.num_systems
            ):
                result = (grad_energy, None, None, None)
                return result + (None,) * (ctx.num_inputs - len(result))
            _grad_system, atom_grad = _energy_cotangents(
                grad_energy,
                ctx.batch_idx,
                num_atoms,
                ctx.num_systems,
                "atom",
            )
        grad_charges = charge_grad_state * atom_grad
        result = (None, grad_charges, None, None)
        return result + (None,) * (ctx.num_inputs - len(result))
