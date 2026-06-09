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

"""Explicit ``ewald_real`` Torch autograd chain.

The chain registers forward, backward, and double-backward custom ops over the
factory kernels for single-system and batched calls. Forward emits per-atom
energies plus detached first-derivative caches when an input requires grad;
backward scales those caches by the per-system energy cotangent; double-backward
recomputes the pair Hessian-vector product from the forward inputs.

The chain owns position and charge derivatives. Literal ``dE/dcell`` is produced
through a Torch autograd path over periodic shifts, while direct-output virials
continue to use the legacy strain-virial kernels.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from nvalchemiops.interactions.electrostatics._factory_common import (
    _DISTANCE_EPSILON,
    _DerivState,
    _ewald_half_force_scale,
    _ewald_half_force_scale_deriv,
    get_backward_scale_kernel,
)
from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    REAL_SPACE_TILED_BLOCK_DIM,
    _ewald_real_space_force_magnitude,
)
from nvalchemiops.interactions.electrostatics.ewald_real_factory import (
    alloc_ewald_real_sentinels,
    get_ewald_real_kernel,
)
from nvalchemiops.torch._warp_op_helpers import (
    register_warp_op_chain,
)
from nvalchemiops.torch.interactions.electrostatics._util import (
    _is_per_system_uniform_cotangent,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = [
    "ewald_real_energy_single",
    "ewald_real_energy_batch",
    "register_ewald_real_ops",
    "real_space_cell_connect",
]

_REAL_SINGLE: dict[str, object] | None = None
_REAL_BATCH: dict[str, object] | None = None
_LITERAL_CELL_GRAD: dict[str, object] | None = None
_EWALD_REAL_OPS_REGISTERED = False


def _wp(tensor: torch.Tensor, dtype):
    """``wp.from_torch`` with shadow-gradient allocation disabled (chain owns bwd)."""
    return wp.from_torch(tensor.detach().contiguous(), dtype=dtype, requires_grad=False)


def _scoped_stream(device: torch.device):
    """Bind Warp's stream to PyTorch's current CUDA stream (graph-capture safe)."""
    if device.type != "cuda":
        from contextlib import nullcontext

        return nullcontext()
    return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device)))


def _atom_counts(batch_idx: torch.Tensor, num_systems: int) -> torch.Tensor:
    """Per-system atom counts (float64)."""
    counts = torch.zeros(num_systems, dtype=torch.float64, device=batch_idx.device)
    return counts.index_add(
        0,
        batch_idx,
        torch.ones(batch_idx.shape[0], dtype=torch.float64, device=batch_idx.device),
    )


def _per_system_cotangent(grad_energy_atom, batch_idx, num_systems, num_atoms):
    """Reduce a per-atom energy cotangent ``(N,)`` to per-system ``(S,)`` (mean)."""
    g = grad_energy_atom.reshape(-1).to(torch.float64)
    if batch_idx is None:
        if num_atoms == 0:
            return torch.zeros(1, dtype=torch.float64, device=g.device)
        return g.mean().reshape(1)
    sums = torch.zeros(num_systems, dtype=torch.float64, device=g.device)
    sums = sums.index_add(0, batch_idx, g)
    return sums / _atom_counts(batch_idx, num_systems).clamp_min(1.0)


def _distribute_to_atoms(per_system, batch_idx, num_systems, num_atoms):
    """Distribute a per-system ``dL/d(grad_energy)`` back to per-atom by ``1/count``."""
    if num_atoms == 0:
        return torch.zeros(0, dtype=torch.float64, device=per_system.device)
    if batch_idx is None:
        return (per_system / float(num_atoms)).expand(num_atoms).clone()
    counts = _atom_counts(batch_idx, num_systems)
    return (per_system / counts.clamp_min(1.0)).index_select(0, batch_idx)


def _cotangent_per_system_uniform(grad_energy_atom, batch_idx, num_systems):
    """Whether the per-atom energy cotangent is constant WITHIN each system.

    The cached-scale first backward serves ``grad_input = mean(cotangent) *
    dE_total/dinput`` per system, which equals the exact VJP only when the cotangent is
    uniform within a system (e.g. ``energy.sum()``). A non-uniform per-atom cotangent
    (a per-atom-energy-weighted loss) needs the weighted recompute below.
    """
    return _is_per_system_uniform_cotangent(
        grad_energy_atom,
        batch_idx,
        num_systems,
    )


def _neighbor_edges(
    use_matrix,
    idx_j,
    neighbor_ptr,
    neighbor_shifts,
    neighbor_matrix,
    neighbor_matrix_shifts,
    mask_value,
    num_atoms,
    device,
):
    """Directed real-space neighbor edges ``(edge_i, edge_j, unit_shifts)``.

    Handles both layouts the chain accepts: a 2D ``(2, E)`` edge list ``[edge_i; edge_j]``
    (batched neighbor list) and a flat CSR ``idx_j`` paired with ``neighbor_ptr``
    (single-system neighbor list); the matrix layout flattens via ``_matrix_to_edges``.
    """
    if use_matrix:
        return _matrix_to_edges(neighbor_matrix, neighbor_matrix_shifts, mask_value)
    if idx_j.dim() == 2:
        return idx_j[0].to(torch.long), idx_j[1].to(torch.long), neighbor_shifts
    counts = (neighbor_ptr[1:] - neighbor_ptr[:-1]).to(torch.long)
    edge_i = torch.repeat_interleave(
        torch.arange(num_atoms, device=device, dtype=torch.long), counts
    )
    return edge_i, idx_j.to(torch.long), neighbor_shifts


def _real_space_weighted_energy(
    positions, charges, cell, alpha, edge_i, edge_j, unit_shifts, batch_idx, w
):
    """Weighted real-space erfc energy ``sum_edges w[i] * 0.5 q_i q_j erfc(a r)/r``.

    Matches the per-pair convention of :func:`_real_space_dEdcell_analytic` (the
    FD-verified oracle). ``autograd.grad`` of this scalar gives the exact weighted VJP
    w.r.t. positions / charges / cell for an arbitrary per-atom cotangent ``w``.
    """
    scalar_alpha = alpha.reshape(-1).numel() == 1
    if batch_idx is None:
        periodic = unit_shifts.to(cell.dtype) @ cell[0]
        alpha_e = alpha.reshape(-1)[0].expand(edge_i.shape[0])
    else:
        sys_of_edge = batch_idx[edge_i].long()
        periodic = torch.bmm(
            unit_shifts.to(cell.dtype).unsqueeze(1), cell[sys_of_edge]
        ).squeeze(1)
        alpha_e = (
            alpha.reshape(-1)[0].expand(edge_i.shape[0])
            if scalar_alpha
            else alpha[sys_of_edge]
        )
    sep = (positions[edge_j] - positions[edge_i] + periodic).to(torch.float64)
    r = sep.norm(dim=1)
    qq = (charges[edge_i] * charges[edge_j]).to(torch.float64)
    a = alpha_e.to(torch.float64)
    e_pair = 0.5 * qq * torch.erfc(a * r) / r
    e_pair = torch.where(r > 1e-8, e_pair, torch.zeros_like(e_pair))
    return (w[edge_i].to(torch.float64) * e_pair).sum()


def _neighbor_args(
    use_matrix,
    idx_j,
    neighbor_ptr,
    neighbor_shifts,
    neighbor_matrix,
    neighbor_matrix_shifts,
    sentinels,
):
    """Six neighbor warp args in the frozen forward order (inactive => sentinel)."""
    if use_matrix:
        return (
            sentinels["idx_j"],
            sentinels["neighbor_ptr"],
            sentinels["unit_shifts"],
            _wp(neighbor_matrix, wp.int32),
            _wp(neighbor_matrix_shifts, wp.vec3i),
        )
    return (
        _wp(idx_j, wp.int32),
        _wp(neighbor_ptr, wp.int32),
        _wp(neighbor_shifts, wp.vec3i),
        sentinels["neighbor_matrix"],
        sentinels["unit_shifts_matrix"],
    )


def _is_empty(use_matrix, idx_j, neighbor_matrix) -> bool:
    """Whether there are zero edges to launch over."""
    if use_matrix:
        return neighbor_matrix.shape[0] == 0 or neighbor_matrix.shape[1] == 0
    return idx_j.shape[0] == 0


# ===========================================================================
# Forward / backward / double-backward implementations (shared single + batch)
# ===========================================================================


def _atom_cotangent(grad_energy_atom, batch_idx, num_systems, num_atoms):
    """Per-system cotangent (mean) broadcast back to per-atom ``(N,)`` f64.

    This is exactly the per-system ``grad_energy`` the old ``order="backward"``
    kernel scaled by, mapped onto atoms so the cached-derivative scale reproduces
    the kernel's first-backward values bit-for-bit.
    """
    g_sys = _per_system_cotangent(grad_energy_atom, batch_idx, num_systems, num_atoms)
    if batch_idx is None:
        return g_sys.expand(num_atoms)
    return g_sys.index_select(0, batch_idx)


def _forward_impl(
    positions,
    charges,
    cell,
    alpha,
    batch_idx,
    idx_j,
    neighbor_ptr,
    neighbor_shifts,
    neighbor_matrix,
    neighbor_matrix_shifts,
    mask_value,
    use_matrix,
    need_pos,
    need_charge,
    need_cell,
    need_virial,
):
    """Fused forward: energy + detached derivative caches / direct virial.

    When ``need_pos`` / ``need_charge`` are set, the single ``order="forward"``
    E_F / E_F_dQ kernel writes forces / charge-grad in the SAME pass as the energy;
    the caches are returned (detached) so the first backward is a pure scale instead
    of a second pair-loop. When ``need_cell`` and the matrix layout is active, the
    ``cell_literal`` forward variant additionally writes the per-atom literal
    ``dE/dcell`` block (output 3), so the cell backward is a pure scatter with no
    separate kernel launch (CSR keeps the edge-kernel path, so its output 3 is a
    zero-size placeholder). ``need_virial`` requests the legacy direct strain-virial
    output as another forward-only cache. Inactive caches are zero-size placeholders
    so the op's output arity is static. All-false ⇒ the ``_DerivState.E``
    energy-only kernel runs (inference cost unchanged).
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    batched = batch_idx is not None
    # The forward-fused literal dE/dcell cache is first-order value only. The
    # connector ignores it under create_graph, so it is safe to build even when
    # position/charge gradients are also active.
    use_cell_literal = bool(need_cell)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    # Caches: dE/dR (= -force) per atom, dE/dq per atom, literal dE/dcell per atom
    # (N,3,3). Zero-size when not needed so the op output arity stays static.
    dEdR = torch.zeros(
        num_atoms if need_pos else 0, 3, device=positions.device, dtype=input_dtype
    )
    dEdq = torch.zeros(
        num_atoms if need_charge else 0, device=positions.device, dtype=torch.float64
    )
    dedcell = torch.zeros(
        num_atoms if use_cell_literal else 0,
        3,
        3,
        device=positions.device,
        dtype=torch.float64,
    )
    num_systems = cell.shape[0]
    virial = torch.zeros(
        num_systems if need_virial else 0,
        3,
        3,
        device=positions.device,
        dtype=input_dtype,
    )
    if num_atoms == 0 or _is_empty(use_matrix, idx_j, neighbor_matrix):
        return energies, dEdR, dEdq, dedcell, virial

    if need_charge and not (need_pos or need_cell or need_virial):
        deriv_state = _DerivState.E_dQ
    elif need_charge:
        deriv_state = _DerivState.E_F_dQ
    elif need_pos or need_cell or need_virial:
        # The literal dE/dcell needs the per-pair force, so cell-only grad still
        # requests a force-bearing kernel (E_F).
        deriv_state = _DerivState.E_F
    else:
        deriv_state = _DerivState.E

    sentinels = alloc_ewald_real_sentinels(wp_scalar, device)
    nbr = _neighbor_args(
        use_matrix,
        idx_j,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        sentinels,
    )
    kernel = get_ewald_real_kernel(
        wp_scalar,
        batched=batched,
        neighbor_input="matrix" if use_matrix else "list",
        deriv_state=deriv_state,
        cell_grad=bool(need_virial),
        order="forward",
        tiled=use_matrix,
        cell_literal=use_cell_literal,
    )
    wp_batch = _wp(batch_idx, wp.int32) if batched else sentinels["batch_id"]
    # The forward kernel writes the physical force F only for force-bearing
    # specializations. The charge-only E_dQ specialization keeps atomic_forces on
    # the sentinel path and avoids force math/atomics.
    deriv_has_force = deriv_state in {_DerivState.E_F, _DerivState.E_F_dQ}
    forces = (
        torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
        if deriv_has_force
        else None
    )
    out_forces = _wp(forces, wp_vec) if deriv_has_force else sentinels["atomic_forces"]
    out_cg = _wp(dEdq, wp.float64) if need_charge else sentinels["charge_gradients"]
    out_virial = _wp(virial, wp_mat) if need_virial else sentinels["virial"]
    launch_inputs = [
        _wp(positions, wp_vec),
        _wp(charges, wp_scalar),
        _wp(cell, wp_mat),
        wp_batch,
        *nbr,
        int(mask_value),
        _wp(alpha, wp_scalar),
        _wp(energies, wp.float64),
        out_forces,
        out_cg,
        out_virial,
    ]
    if use_cell_literal:
        # The cell_literal kernel has one extra trailing output: the per-atom
        # literal dE/dcell block (mat33d). Only this launch supplies it; every
        # other launch uses the byte-identical 15-arg forward kernel.
        launch_inputs.append(_wp(dedcell, wp.mat33d))
    with _scoped_stream(positions.device):
        # The neighbor-matrix kernel is the cooperative-block (tiled) variant:
        # block_dim threads share each atom's row. CSR stays one-thread-per-atom.
        if use_matrix:
            wp.launch_tiled(
                kernel,
                dim=[num_atoms],
                inputs=launch_inputs,
                block_dim=REAL_SPACE_TILED_BLOCK_DIM,
                device=device,
            )
        else:
            wp.launch(kernel, dim=[num_atoms], inputs=launch_inputs, device=device)
    if need_pos:
        # dE/dR = -F (cache detached; pure value for the first-backward scale).
        dEdR = (-forces).detach()
    return energies, dEdR.detach(), dEdq.detach(), dedcell.detach(), virial.detach()


def _backward_impl(
    dEdR_cache,
    dEdq_cache,
    grad_energy_atom,
    positions,
    charges,
    cell,
    alpha,
    batch_idx,
    idx_j,
    neighbor_ptr,
    neighbor_shifts,
    neighbor_matrix,
    neighbor_matrix_shifts,
    mask_value,
    use_matrix,
    need_pos,
    need_charge,
    need_cell,
):
    """First backward = cheap scale of the detached forward caches (no pair loop).

    ``grad_positions = (per-system grad_energy)·dE/dR`` and
    ``grad_charges = (per-system grad_energy)·dE/dq`` -- numerically identical to the
    old ``order="backward"`` kernel output (same per-system mean reduction, same
    sign: the dE/dR cache already absorbed the ``-F`` negation). The forward inputs
    are still threaded through so the registered ``double_backward`` recomputes the
    Hessian from them (the caches are detached and never differentiated).
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    num_systems = cell.shape[0]

    grad_positions = torch.zeros(
        num_atoms, 3, device=positions.device, dtype=input_dtype
    )
    grad_charges = torch.zeros(
        num_atoms if need_charge else 0,
        device=positions.device,
        dtype=torch.float64,
    )
    grad_cell = torch.zeros(
        num_systems, 3, 3, device=positions.device, dtype=cell.dtype
    )
    if num_atoms == 0 or _is_empty(use_matrix, idx_j, neighbor_matrix):
        return grad_positions, grad_charges, grad_cell

    # Non-uniform per-atom cotangent: the cached dE_total/dinput (summed over pairs)
    # cannot be re-weighted post-hoc, so the per-system-mean scale path below is wrong.
    # Recompute the exact weighted VJP from the differentiable Torch pair energy. The
    # uniform path (the common training case, e.g. energy.sum()) keeps the fast scale.
    any_need = need_pos or need_charge or need_cell
    if any_need and not _cotangent_per_system_uniform(
        grad_energy_atom, batch_idx, num_systems
    ):
        edge_i, edge_j, unit_shifts = _neighbor_edges(
            use_matrix,
            idx_j,
            neighbor_ptr,
            neighbor_shifts,
            neighbor_matrix,
            neighbor_matrix_shifts,
            mask_value,
            num_atoms,
            positions.device,
        )
        # ``grad_cell`` is owned by the separate ``_RealCellGrad`` connector (whose
        # scatter already weights per-atom), so ``_backward_impl`` only supplies the
        # position/charge grads here -- leave ``grad_cell`` at zero as the scale path does.
        if edge_i.numel() > 0 and (need_pos or need_charge):
            with torch.inference_mode(False), torch.enable_grad():

                def _copy64(t):
                    return torch.empty_like(t, dtype=torch.float64).copy_(t).detach()

                def _leaf(t):
                    return _copy64(t).requires_grad_(True)

                p_leaf = _leaf(positions)
                q_leaf = _leaf(charges)
                c_buf = _copy64(cell)
                alpha_f = _copy64(alpha)
                w_f = _copy64(grad_energy_atom).reshape(-1)
                loss = _real_space_weighted_energy(
                    p_leaf,
                    q_leaf,
                    c_buf,
                    alpha_f,
                    edge_i,
                    edge_j,
                    unit_shifts,
                    batch_idx,
                    w_f,
                )
                gp, gq = torch.autograd.grad(loss, [p_leaf, q_leaf], allow_unused=True)
            if need_pos and gp is not None:
                grad_positions = gp.to(input_dtype)
            if need_charge and gq is not None:
                grad_charges = gq.to(torch.float64)
        return grad_positions, grad_charges, grad_cell

    scale_positions = need_pos and dEdR_cache.shape[0] == num_atoms
    scale_charges = need_charge and dEdq_cache.shape[0] == num_atoms
    if scale_positions or scale_charges:
        device = wp.device_from_torch(positions.device)
        wp_vec = get_wp_vec_dtype(input_dtype)
        grad_energy = _per_system_cotangent(
            grad_energy_atom, batch_idx, num_systems, num_atoms
        )
        sentinels = alloc_ewald_real_sentinels(get_wp_dtype(input_dtype), device)
        wp_batch = (
            _wp(batch_idx, wp.int32) if batch_idx is not None else sentinels["batch_id"]
        )
        kernel = get_backward_scale_kernel(
            get_wp_dtype(input_dtype),
            batched=batch_idx is not None,
            scale_positions=scale_positions,
            scale_charges=scale_charges,
        )
        with _scoped_stream(positions.device):
            wp.launch(
                kernel,
                dim=num_atoms,
                inputs=[
                    _wp(grad_energy, wp.float64),
                    wp_batch,
                    _wp(dEdR_cache, wp_vec),
                    _wp(dEdq_cache, wp.float64),
                    _wp(grad_positions, wp_vec),
                    _wp(grad_charges, wp.float64),
                    wp.int32(num_atoms),
                ],
                device=device,
            )
    return grad_positions, grad_charges, grad_cell


def _double_backward_impl(
    v_pos,
    v_charge,
    _v_cell_zero,
    dEdR_cache,
    dEdq_cache,
    grad_energy_atom,
    positions,
    charges,
    cell,
    alpha,
    batch_idx,
    idx_j,
    neighbor_ptr,
    neighbor_shifts,
    neighbor_matrix,
    neighbor_matrix_shifts,
    mask_value,
    use_matrix,
    need_pos,
    need_charge,
    need_cell,
):
    # ``dEdR_cache`` / ``dEdq_cache`` (the backward op's leading cache inputs) and
    # ``need_pos`` / ``need_charge`` are accepted for positional alignment but NOT
    # used: the second order is recomputed from the forward inputs (recompute mode),
    # the detached cache is first-order value only.
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    batched = batch_idx is not None
    num_systems = cell.shape[0]

    grad_grad_energy = torch.zeros(
        num_systems, device=positions.device, dtype=torch.float64
    )
    grad_positions = torch.zeros(
        num_atoms, 3, device=positions.device, dtype=input_dtype
    )
    grad_charges = torch.zeros(
        num_atoms if need_charge else 0,
        device=positions.device,
        dtype=torch.float64,
    )
    grad_cell = torch.zeros(
        num_systems, 3, 3, device=positions.device, dtype=input_dtype
    )

    if num_atoms == 0 or _is_empty(use_matrix, idx_j, neighbor_matrix):
        gge = _distribute_to_atoms(grad_grad_energy, batch_idx, num_systems, num_atoms)
        return gge, grad_positions, grad_charges, grad_cell

    grad_energy = _per_system_cotangent(
        grad_energy_atom, batch_idx, num_systems, num_atoms
    )

    sentinels = alloc_ewald_real_sentinels(wp_scalar, device)
    nbr = _neighbor_args(
        use_matrix,
        idx_j,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        sentinels,
    )
    deriv_state = _DerivState.E_F_dQ if need_charge else _DerivState.E_F
    kernel = get_ewald_real_kernel(
        wp_scalar,
        batched=batched,
        neighbor_input="matrix" if use_matrix else "list",
        deriv_state=deriv_state,
        cell_grad=bool(need_cell),
        order="double_backward",
        tiled=use_matrix,
    )
    wp_batch = _wp(batch_idx, wp.int32) if batched else sentinels["batch_id"]
    zero_v_cell = torch.zeros(
        num_systems, 3, 3, device=positions.device, dtype=input_dtype
    )
    launch_inputs = [
        _wp(v_pos, wp_vec),
        _wp(v_charge, wp.float64) if need_charge else sentinels["v_charge"],
        _wp(zero_v_cell, wp_mat) if need_cell else sentinels["v_cell"],
        _wp(grad_energy, wp.float64),
        _wp(positions, wp_vec),
        _wp(charges, wp_scalar),
        _wp(cell, wp_mat),
        wp_batch,
        *nbr,
        int(mask_value),
        _wp(alpha, wp_scalar),
        _wp(grad_grad_energy, wp.float64),
        _wp(grad_positions, wp_vec),
        _wp(grad_charges, wp.float64) if need_charge else sentinels["grad_charges"],
        _wp(grad_cell, wp_mat) if need_cell else sentinels["grad_cell"],
    ]
    with _scoped_stream(positions.device):
        if use_matrix:
            wp.launch_tiled(
                kernel,
                dim=[num_atoms],
                inputs=launch_inputs,
                block_dim=REAL_SPACE_TILED_BLOCK_DIM,
                device=device,
            )
        else:
            wp.launch(kernel, dim=[num_atoms], inputs=launch_inputs, device=device)
    gge = _distribute_to_atoms(grad_grad_energy, batch_idx, num_systems, num_atoms)
    return gge, grad_positions, grad_charges, grad_cell


# ===========================================================================
# Single-system chain launchers (batch_idx is None)
# ===========================================================================


def _real_forward_single(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
    use_matrix: bool,
    need_pos: bool,
    need_charge: bool,
    need_cell: bool,
    need_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _forward_impl(
        positions,
        charges,
        cell,
        alpha,
        None,
        idx_j,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
        use_matrix,
        need_pos,
        need_charge,
        need_cell,
        need_virial,
    )


def _real_backward_single(
    dEdR_cache: torch.Tensor,
    dEdq_cache: torch.Tensor,
    grad_energy: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
    use_matrix: bool,
    need_pos: bool,
    need_charge: bool,
    need_cell: bool,
    need_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _backward_impl(
        dEdR_cache,
        dEdq_cache,
        grad_energy,
        positions,
        charges,
        cell,
        alpha,
        None,
        idx_j,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
        use_matrix,
        need_pos,
        need_charge,
        need_cell,
    )


def _real_double_backward_single(
    v_pos: torch.Tensor,
    v_charge: torch.Tensor,
    v_cell_zero: torch.Tensor,
    dEdR_cache: torch.Tensor,
    dEdq_cache: torch.Tensor,
    grad_energy: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
    use_matrix: bool,
    need_pos: bool,
    need_charge: bool,
    need_cell: bool,
    need_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _double_backward_impl(
        v_pos,
        v_charge,
        v_cell_zero,
        dEdR_cache,
        dEdq_cache,
        grad_energy,
        positions,
        charges,
        cell,
        alpha,
        None,
        idx_j,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
        use_matrix,
        need_pos,
        need_charge,
        need_cell,
    )


# ===========================================================================
# Batched chain launchers (batch_idx provided)
# ===========================================================================


def _real_forward_batch(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
    use_matrix: bool,
    need_pos: bool,
    need_charge: bool,
    need_cell: bool,
    need_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _forward_impl(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        idx_j,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
        use_matrix,
        need_pos,
        need_charge,
        need_cell,
        need_virial,
    )


def _real_backward_batch(
    dEdR_cache: torch.Tensor,
    dEdq_cache: torch.Tensor,
    grad_energy: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
    use_matrix: bool,
    need_pos: bool,
    need_charge: bool,
    need_cell: bool,
    need_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _backward_impl(
        dEdR_cache,
        dEdq_cache,
        grad_energy,
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        idx_j,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
        use_matrix,
        need_pos,
        need_charge,
        need_cell,
    )


def _real_double_backward_batch(
    v_pos: torch.Tensor,
    v_charge: torch.Tensor,
    v_cell_zero: torch.Tensor,
    dEdR_cache: torch.Tensor,
    dEdq_cache: torch.Tensor,
    grad_energy: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
    use_matrix: bool,
    need_pos: bool,
    need_charge: bool,
    need_cell: bool,
    need_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _double_backward_impl(
        v_pos,
        v_charge,
        v_cell_zero,
        dEdR_cache,
        dEdq_cache,
        grad_energy,
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        idx_j,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
        use_matrix,
        need_pos,
        need_charge,
        need_cell,
    )


# ===========================================================================
# Register the chains
# ===========================================================================
#
# Single-system forward inputs (15): 0 positions, 1 charges, 2 cell, 3 alpha,
#   4 idx_j, 5 neighbor_ptr, 6 neighbor_shifts, 7 neighbor_matrix,
#   8 neighbor_matrix_shifts, 9 mask_value, 10 use_matrix, 11 need_pos, 12 need_charge,
#   13 need_cell, 14 need_virial.
# Forward OUTPUTS (5): 0 energy (N,), 1 dE/dR cache (N,3), 2 dE/dq cache (N,),
#   3 literal dE/dcell cache (N,3,3) | (0,3,3), 4 direct virial (S,3,3) | (0,3,3).
# Differentiable forward inputs: positions(0), charges(1), cell(2). The chain's
#   first-order cell gradient is deliberately zero; literal dE/dcell is owned by
#   _RealCellGrad. Registering cell here lets the chain's double-backward return
#   the missing d(grad_positions)/dcell term for stress losses. Only the energy
#   cotangent (output 0) drives the backward
#   (propagate_outputs=(0,)); outputs 1/2 are detached caches threaded to the
#   backward op via save_forward_outputs=(1, 2). Output 3 (dE/dcell) is NEITHER
#   propagated NOR saved -- it is returned to ewald.py and consumed by the
#   _RealCellGrad connector (its cotangent is dropped by the propagate_outputs slice).
# Backward op inputs (18): the 2 prepended caches (0 dE/dR, 1 dE/dq) + grad_energy(2)
#   + the 15 forward inputs (3..17). Double-backward differentiates grad_energy(2),
#   positions(3), charges(4), cell(5) -- shifted by the 2
#   cache slots; the cache slots and trailing need_* ints get None grads.


def _real_forward_fake(positions, *args):
    """Forward fake: ``(energy, dE/dR, dE/dq, dE/dcell, virial)``.

    The caches' presence is gated by the trailing ``need_pos`` / ``need_charge`` /
    ``need_cell`` / ``need_virial`` flags, matching the real launcher's output
    shapes.
    """
    need_pos, need_charge, need_cell, need_virial = (
        bool(args[-4]),
        bool(args[-3]),
        bool(args[-2]),
        bool(args[-1]),
    )
    use_cell_literal = need_cell
    n = positions.shape[0]
    energy = positions.new_empty(n, dtype=torch.float64)
    dEdR = positions.new_empty(n if need_pos else 0, 3, dtype=positions.dtype)
    dEdq = positions.new_empty(n if need_charge else 0, dtype=torch.float64)
    dedcell = positions.new_empty(
        n if use_cell_literal else 0, 3, 3, dtype=torch.float64
    )
    cell = args[1]
    num_systems = cell.shape[0]
    virial = positions.new_empty(num_systems if need_virial else 0, 3, 3)
    return energy, dEdR, dEdq, dedcell, virial


def _real_backward_fake(dEdR_cache, dEdq_cache, grad_energy, positions, charges, *args):
    """Backward fake: ``(grad_positions, grad_charges, zero grad_cell)``."""
    n = positions.shape[0]
    cell = args[0]
    return (
        positions.new_empty(n, 3, dtype=positions.dtype),
        positions.new_empty(n, dtype=torch.float64),
        cell.new_empty(cell.shape),
    )


def _real_double_backward_fake(
    v_pos,
    v_charge,
    v_cell_zero,
    dEdR_cache,
    dEdq_cache,
    grad_energy,
    positions,
    charges,
    *args,
):
    """Double-backward fake: ``(grad_grad_energy, grad_positions, grad_charges, grad_cell)``."""
    n = positions.shape[0]
    cell = args[0]
    return (
        positions.new_empty(n, dtype=torch.float64),
        positions.new_empty(n, 3, dtype=positions.dtype),
        positions.new_empty(n, dtype=torch.float64),
        cell.new_empty(cell.shape),
    )


def register_ewald_real_ops() -> None:
    """Register the Ewald real-space Torch custom-op chain once."""
    global _EWALD_REAL_OPS_REGISTERED, _LITERAL_CELL_GRAD, _REAL_BATCH, _REAL_SINGLE
    if _EWALD_REAL_OPS_REGISTERED:
        return

    _REAL_SINGLE = register_warp_op_chain(
        name="nvalchemiops::ewald_real_energy_single",
        forward=_real_forward_single,
        backward=_real_backward_single,
        double_backward=_real_double_backward_single,
        forward_fake=_real_forward_fake,
        backward_fake=_real_backward_fake,
        double_backward_fake=_real_double_backward_fake,
        forward_return_arity=5,
        propagate_outputs=(0,),
        save_forward_outputs=(1, 2),
        diff_input_positions=(0, 1, 2),
        n_forward_inputs=15,
        backward_return_arity=3,
        second_order_diff_positions=(2, 3, 4, 5),
        n_backward_inputs=18,
        double_backward_return_arity=4,
    )

    # Batched forward inputs (16): 0 positions, 1 charges, 2 cell, 3 alpha,
    #   4 batch_idx, 5 idx_j, 6 neighbor_ptr, 7 neighbor_shifts, 8 neighbor_matrix,
    #   9 neighbor_matrix_shifts, 10 mask_value, 11 use_matrix, 12 need_pos,
    #   13 need_charge, 14 need_cell, 15 need_virial.
    _REAL_BATCH = register_warp_op_chain(
        name="nvalchemiops::ewald_real_energy_batch",
        forward=_real_forward_batch,
        backward=_real_backward_batch,
        double_backward=_real_double_backward_batch,
        forward_fake=_real_forward_fake,
        backward_fake=_real_backward_fake,
        double_backward_fake=_real_double_backward_fake,
        forward_return_arity=5,
        propagate_outputs=(0,),
        save_forward_outputs=(1, 2),
        diff_input_positions=(0, 1, 2),
        n_forward_inputs=16,
        backward_return_arity=3,
        second_order_diff_positions=(2, 3, 4, 5),
        n_backward_inputs=19,
        double_backward_return_arity=4,
        batch_match=True,
    )

    _LITERAL_CELL_GRAD = register_warp_op_chain(
        name="nvalchemiops::ewald_real_literal_cell_grad",
        forward=_literal_cell_grad_forward,
        backward=_literal_cell_grad_backward,
        forward_fake=_literal_cell_grad_fake,
        backward_fake=_literal_cell_grad_backward_fake,
        diff_input_positions=(0, 1, 2, 12),
        n_forward_inputs=13,
        forward_return_arity=1,
        backward_return_arity=4,
    )

    _EWALD_REAL_OPS_REGISTERED = True


def ewald_real_energy_single(*args, **kwargs):
    """Call the registered single-system Ewald real-space energy op."""
    register_ewald_real_ops()
    if _REAL_SINGLE is None:
        raise RuntimeError("Ewald real single-system op registration failed")
    return _REAL_SINGLE["forward"](*args, **kwargs)


def ewald_real_energy_batch(*args, **kwargs):
    """Call the registered batched Ewald real-space energy op."""
    register_ewald_real_ops()
    if _REAL_BATCH is None:
        raise RuntimeError("Ewald real batched op registration failed")
    return _REAL_BATCH["forward"](*args, **kwargs)


# ===========================================================================
# Cell gradient via Torch-native periodic shift (literal dE/dcell)
# ===========================================================================


@wp.kernel
def _real_cell_grad_edge_kernel(
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    alpha: wp.array(dtype=wp.float64),
    edge_i: wp.array(dtype=wp.int32),
    edge_j: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    sys_of_atom: wp.array(dtype=wp.int32),
    dedcell_atom: wp.array(dtype=wp.mat33d),
) -> None:
    """One thread per directed edge; accumulate per-atom literal ``dE/dcell`` block.

    The atom ``i`` energy share contributes ``n (x) dE/dsep`` to ``dE/dcell`` with
    ``dE/dsep = -force`` and ``force = force_mag * sep`` (the per-pair force the
    forward kernel uses). Accumulating into a PER-ATOM output (``N`` slots, ~max-
    neighbors-way contention) instead of the per-system ``grad_cell`` (``S`` slots,
    catastrophic contention) is what keeps this fast. The per-system reduction and
    the upstream cotangent weighting are done cheaply in Torch afterwards.
    """
    e = wp.tid()
    i = edge_i[e]
    j = edge_j[e]
    isys = sys_of_atom[i]
    cell_t = wp.transpose(cell[isys])
    shift = unit_shifts[e]
    n = wp.vec3d(wp.float64(shift[0]), wp.float64(shift[1]), wp.float64(shift[2]))
    sep = positions[j] - positions[i] + cell_t * n
    r = wp.length(sep)
    if r > 1.0e-8:
        fm = _ewald_real_space_force_magnitude(charges[i], charges[j], r, alpha[isys])
        wp.atomic_add(dedcell_atom, i, -fm * wp.outer(n, sep))


def _real_cell_grad_via_kernel(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    edge_i: torch.Tensor,
    edge_j: torch.Tensor,
    unit_shifts: torch.Tensor,
    batch_idx: torch.Tensor | None,
    grad_energy_atom: torch.Tensor,
) -> torch.Tensor:
    """Fast literal ``dE/dcell`` via the edge-parallel Warp kernel (first-order path).

    Computes the same quantity as :func:`_real_space_dEdcell_analytic` but the
    per-pair ``n (x) dE/dsep`` blocks are summed in one Warp launch into a per-atom
    buffer (no ``(M, 3, 3)`` Torch intermediate / ``index_add``), then weighted by the
    upstream per-atom energy cotangent and reduced per system in Torch -- the fast
    stress/mixed backward path. Non-differentiable; used only when the
    backward is not itself being differentiated (no stress-loss double-backward).
    """
    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    if edge_i.shape[0] == 0:
        return torch.zeros(
            (num_systems, 3, 3), dtype=cell.dtype, device=positions.device
        )
    if batch_idx is None:
        sys_of_atom = torch.zeros(num_atoms, dtype=torch.int32, device=positions.device)
    else:
        sys_of_atom = batch_idx.to(torch.int32)
    device = wp.device_from_torch(positions.device)
    dedcell_atom = torch.zeros(
        (num_atoms, 3, 3), dtype=torch.float64, device=positions.device
    )

    def f64(t):
        return wp.from_torch(
            t.detach().to(torch.float64).contiguous(),
            dtype=wp.float64,
            requires_grad=False,
        )

    with _scoped_stream(positions.device):
        wp.launch(
            _real_cell_grad_edge_kernel,
            dim=edge_i.shape[0],
            inputs=[
                wp.from_torch(
                    positions.detach().to(torch.float64).contiguous(),
                    dtype=wp.vec3d,
                    requires_grad=False,
                ),
                f64(charges),
                wp.from_torch(
                    cell.detach().to(torch.float64).contiguous(),
                    dtype=wp.mat33d,
                    requires_grad=False,
                ),
                f64(alpha),
                wp.from_torch(
                    edge_i.to(torch.int32).contiguous(),
                    dtype=wp.int32,
                    requires_grad=False,
                ),
                wp.from_torch(
                    edge_j.to(torch.int32).contiguous(),
                    dtype=wp.int32,
                    requires_grad=False,
                ),
                wp.from_torch(
                    unit_shifts.to(torch.int32).contiguous(),
                    dtype=wp.vec3i,
                    requires_grad=False,
                ),
                wp.from_torch(
                    sys_of_atom.contiguous(), dtype=wp.int32, requires_grad=False
                ),
                wp.from_torch(dedcell_atom, dtype=wp.mat33d, requires_grad=False),
            ],
            device=device,
        )
    # Weight per-atom blocks by the upstream energy cotangent and reduce per system.
    weighted = grad_energy_atom.to(torch.float64).view(-1, 1, 1) * dedcell_atom
    grad_cell = torch.zeros(
        (num_systems, 3, 3), dtype=torch.float64, device=positions.device
    )
    grad_cell = grad_cell.index_add(0, sys_of_atom.to(torch.long), weighted)
    return grad_cell.to(cell.dtype)


def _real_space_dEdcell_analytic(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    edge_i: torch.Tensor,
    edge_j: torch.Tensor,
    unit_shifts: torch.Tensor,
    batch_idx: torch.Tensor | None,
    grad_energy_atom: torch.Tensor,
) -> torch.Tensor:
    """Literal ``dE/dcell`` (per system) from the closed-form per-pair derivative.

    ``E_pair = 1/2 q_i q_j erfc(alpha r)/r`` with ``r = |sep|`` and
    ``sep = pos_j - pos_i + unit_shifts @ cell``. Cell enters only through
    ``sep``, so ``dE/dcell = sum_edges w_i (n (x) dE/dsep)`` with ``n`` the integer
    shift and ``w_i = grad_energy_atom[i]`` the upstream per-atom energy cotangent.
    Expressed as ONE vectorized expression (no autograd over a recomputed energy),
    it stays fully differentiable, so under ``create_graph`` (stress-loss
    double-backward) Torch differentiates it for the ``dE/dR dcell`` /
    ``dE/dq dcell`` / ``dE/dcell^2`` cross terms. The first-order matrix path instead
    uses the forward-fused per-atom ``dedcell`` cache (a pure scatter); this analytic
    form is the differentiable double-backward path and the CSR edge kernel's oracle.
    """
    num_systems = cell.shape[0]
    if batch_idx is None:
        sys_of_edge = torch.zeros(
            edge_i.shape[0], dtype=torch.long, device=positions.device
        )
        periodic = unit_shifts.to(cell.dtype) @ cell[0]
        alpha_e = alpha[0].expand(edge_i.shape[0])
    else:
        sys_of_edge = batch_idx[edge_i].long()
        periodic = torch.bmm(
            unit_shifts.to(cell.dtype).unsqueeze(1), cell[sys_of_edge]
        ).squeeze(1)
        alpha_e = alpha[sys_of_edge]

    sep = (positions[edge_j] - positions[edge_i] + periodic).to(torch.float64)
    r = sep.norm(dim=1)
    qq = (charges[edge_i] * charges[edge_j]).to(torch.float64)
    a = alpha_e.to(torch.float64)
    erfc = torch.erfc(a * r)
    expt = torch.exp(-((a * r) ** 2))
    # d/dr [ 1/2 q_i q_j erfc(a r) / r ]
    d_e_dr = 0.5 * qq * (-2.0 * a / math.sqrt(math.pi) * expt / r - erfc / (r * r))
    coeff = d_e_dr / r  # dE/dsep = coeff * sep
    d_e_dsep = coeff.unsqueeze(1) * sep  # (M, 3)
    w = grad_energy_atom[edge_i].to(torch.float64)
    n = unit_shifts.to(torch.float64)  # (M, 3)
    # outer(n, dE/dsep)[p, q] = n_p * dE/dsep_q
    contrib = w.view(-1, 1, 1) * n.unsqueeze(2) * d_e_dsep.unsqueeze(1)  # (M, 3, 3)
    contrib = torch.where((r > 1e-8).view(-1, 1, 1), contrib, torch.zeros_like(contrib))
    grad_cell = torch.zeros(
        (num_systems, 3, 3), dtype=torch.float64, device=positions.device
    )
    grad_cell = grad_cell.index_add(0, sys_of_edge, contrib)
    return grad_cell.to(cell.dtype)


@wp.kernel
def _literal_cell_grad_edges_kernel(
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    alpha: wp.array(dtype=wp.float64),
    batch_id: wp.array(dtype=wp.int32),
    is_batched: wp.int32,
    edge_i: wp.array(dtype=wp.int32),
    edge_j: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    grad_energy_atom: wp.array(dtype=wp.float64),
    grad_cell: wp.array(dtype=wp.mat33d),
) -> None:
    """Edge-parallel literal ``dE/dcell`` for CSR stress double-backward."""
    e = wp.tid()
    i = edge_i[e]
    j = edge_j[e]
    isys = wp.int32(0)
    if is_batched != wp.int32(0):
        isys = batch_id[i]

    qi = charges[i]
    qj = charges[j]
    shift = unit_shifts[e]
    n_vec = wp.vec3d(wp.float64(shift[0]), wp.float64(shift[1]), wp.float64(shift[2]))
    sep = positions[j] - positions[i] + wp.transpose(cell[isys]) * n_vec
    r = wp.length(sep)
    if r > wp.float64(_DISTANCE_EPSILON):
        fm = qi * qj * _ewald_half_force_scale(r, alpha[isys])
        d_e_dsep = -fm * sep
        contrib = grad_energy_atom[i] * wp.outer(n_vec, d_e_dsep)
        wp.atomic_add(grad_cell, isys, contrib)


@wp.kernel
def _literal_cell_grad_matrix_tiled_kernel(
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    alpha: wp.array(dtype=wp.float64),
    batch_id: wp.array(dtype=wp.int32),
    is_batched: wp.int32,
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
    mask_value: wp.int32,
    grad_energy_atom: wp.array(dtype=wp.float64),
    grad_cell: wp.array(dtype=wp.mat33d),
) -> None:
    """Tiled neighbor-matrix literal ``dE/dcell``."""
    atom_i, lane = wp.tid()
    block_size = wp.block_dim()

    isys = wp.int32(0)
    if is_batched != wp.int32(0):
        isys = batch_id[atom_i]
    qi = charges[atom_i]
    pos_i = positions[atom_i]
    cell_t = wp.transpose(cell[isys])
    alpha_i = alpha[isys]
    w = grad_energy_atom[atom_i]

    acc = wp.mat33d()
    k = lane
    max_neighbors = neighbor_matrix.shape[1]
    while k < max_neighbors:
        j = neighbor_matrix[atom_i, k]
        if j != mask_value:
            shift = unit_shifts_matrix[atom_i, k]
            n_vec = wp.vec3d(
                wp.float64(shift[0]), wp.float64(shift[1]), wp.float64(shift[2])
            )
            sep = positions[j] - pos_i + cell_t * n_vec
            r = wp.length(sep)
            if r > wp.float64(_DISTANCE_EPSILON):
                fm = qi * charges[j] * _ewald_half_force_scale(r, alpha_i)
                acc += w * wp.outer(n_vec, -fm * sep)
        k += block_size

    acc_sum = wp.tile_sum(wp.tile(acc, preserve_type=True))
    if lane == 0:
        wp.atomic_add(grad_cell, isys, wp.tile_extract(acc_sum, 0))


@wp.kernel
def _literal_cell_grad_backward_edges_kernel(
    h_cell: wp.array(dtype=wp.mat33d),
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    alpha: wp.array(dtype=wp.float64),
    batch_id: wp.array(dtype=wp.int32),
    is_batched: wp.int32,
    edge_i: wp.array(dtype=wp.int32),
    edge_j: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    grad_energy_atom: wp.array(dtype=wp.float64),
    grad_positions: wp.array(dtype=wp.vec3d),
    grad_charges: wp.array(dtype=wp.float64),
    grad_cell: wp.array(dtype=wp.mat33d),
    grad_grad_energy_atom: wp.array(dtype=wp.float64),
) -> None:
    """Backward of the literal ``dE/dcell`` op for CSR edges."""
    e = wp.tid()
    i = edge_i[e]
    j = edge_j[e]
    isys = wp.int32(0)
    if is_batched != wp.int32(0):
        isys = batch_id[i]

    qi = charges[i]
    qj = charges[j]
    shift = unit_shifts[e]
    n_vec = wp.vec3d(wp.float64(shift[0]), wp.float64(shift[1]), wp.float64(shift[2]))
    sep = positions[j] - positions[i] + wp.transpose(cell[isys]) * n_vec
    r = wp.length(sep)
    if r > wp.float64(_DISTANCE_EPSILON):
        w = grad_energy_atom[i]
        m = h_cell[isys]
        p = wp.transpose(m) * n_vec
        pdot = wp.dot(p, sep)
        half_s = _ewald_half_force_scale(r, alpha[isys])
        half_ds = _ewald_half_force_scale_deriv(r, alpha[isys])
        fm = qi * qj * half_s
        coeff = qi * qj * half_ds / r
        grad_sep = -w * (fm * p + coeff * pdot * sep)

        wp.atomic_add(grad_positions, i, -grad_sep)
        wp.atomic_add(grad_positions, j, grad_sep)
        wp.atomic_add(grad_cell, isys, wp.outer(n_vec, grad_sep))
        wp.atomic_add(grad_charges, i, -w * qj * half_s * pdot)
        wp.atomic_add(grad_charges, j, -w * qi * half_s * pdot)
        wp.atomic_add(grad_grad_energy_atom, i, -fm * pdot)


@wp.kernel
def _literal_cell_grad_backward_matrix_tiled_kernel(
    h_cell: wp.array(dtype=wp.mat33d),
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    alpha: wp.array(dtype=wp.float64),
    batch_id: wp.array(dtype=wp.int32),
    is_batched: wp.int32,
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
    mask_value: wp.int32,
    grad_energy_atom: wp.array(dtype=wp.float64),
    grad_positions: wp.array(dtype=wp.vec3d),
    grad_charges: wp.array(dtype=wp.float64),
    grad_cell_atom: wp.array(dtype=wp.mat33d),
    grad_grad_energy_atom: wp.array(dtype=wp.float64),
) -> None:
    """Tiled backward of the literal ``dE/dcell`` op."""
    atom_i, lane = wp.tid()
    block_size = wp.block_dim()

    isys = wp.int32(0)
    if is_batched != wp.int32(0):
        isys = batch_id[atom_i]
    qi = charges[atom_i]
    pos_i = positions[atom_i]
    cell_t = wp.transpose(cell[isys])
    alpha_i = alpha[isys]
    w = grad_energy_atom[atom_i]
    m = h_cell[isys]

    gpos_i = wp.vec3d(0.0, 0.0, 0.0)
    gq_i = wp.float64(0.0)
    gcell_acc = wp.mat33d()
    gge_i = wp.float64(0.0)

    k = lane
    max_neighbors = neighbor_matrix.shape[1]
    while k < max_neighbors:
        j = neighbor_matrix[atom_i, k]
        if j != mask_value:
            qj = charges[j]
            shift = unit_shifts_matrix[atom_i, k]
            n_vec = wp.vec3d(
                wp.float64(shift[0]), wp.float64(shift[1]), wp.float64(shift[2])
            )
            sep = positions[j] - pos_i + cell_t * n_vec
            r = wp.length(sep)
            if r > wp.float64(_DISTANCE_EPSILON):
                p = wp.transpose(m) * n_vec
                pdot = wp.dot(p, sep)
                half_s = _ewald_half_force_scale(r, alpha_i)
                half_ds = _ewald_half_force_scale_deriv(r, alpha_i)
                fm = qi * qj * half_s
                coeff = qi * qj * half_ds / r
                grad_sep = -w * (fm * p + coeff * pdot * sep)

                gpos_i += -grad_sep
                wp.atomic_add(grad_positions, j, grad_sep)
                gcell_acc += wp.outer(n_vec, grad_sep)
                gq_i += -w * qj * half_s * pdot
                wp.atomic_add(grad_charges, j, -w * qi * half_s * pdot)
                gge_i += -fm * pdot
        k += block_size

    gpos_sum = wp.tile_sum(wp.tile(gpos_i, preserve_type=True))
    gq_sum = wp.tile_sum(wp.tile(gq_i))
    gcell_sum = wp.tile_sum(wp.tile(gcell_acc, preserve_type=True))
    gge_sum = wp.tile_sum(wp.tile(gge_i))
    if lane == 0:
        wp.atomic_add(grad_positions, atom_i, wp.tile_extract(gpos_sum, 0))
        wp.atomic_add(grad_charges, atom_i, wp.tile_extract(gq_sum, 0))
        grad_cell_atom[atom_i] = wp.tile_extract(gcell_sum, 0)
        wp.atomic_add(grad_grad_energy_atom, atom_i, wp.tile_extract(gge_sum, 0))


def _as_f64_tensor(t: torch.Tensor) -> torch.Tensor:
    """Detached contiguous f64 tensor for fixed-signature literal-cell kernels."""
    return t.detach().to(torch.float64).contiguous()


def _batch_arg(batch_idx: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return ``(batch_idx_int32, is_batched)`` for Warp kernels."""
    if batch_idx.numel() == 0:
        return batch_idx.to(torch.int32), 0
    return batch_idx.to(torch.int32).contiguous(), 1


def _literal_cell_grad_forward(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    edge_i: torch.Tensor,
    edge_j: torch.Tensor,
    unit_shifts: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
    use_matrix: bool,
    grad_energy_atom: torch.Tensor,
) -> torch.Tensor:
    """Warp-backed literal ``dE/dcell`` used only for differentiable stress."""
    device = wp.device_from_torch(positions.device)
    num_systems = cell.shape[0]
    grad_cell = torch.zeros(
        num_systems, 3, 3, dtype=torch.float64, device=positions.device
    )
    if positions.shape[0] == 0:
        return grad_cell.to(cell.dtype)

    batch_t, is_batched = _batch_arg(batch_idx)
    pos64 = _as_f64_tensor(positions)
    chg64 = _as_f64_tensor(charges)
    cell64 = _as_f64_tensor(cell)
    alpha64 = _as_f64_tensor(alpha)
    ge64 = grad_energy_atom.detach().to(torch.float64).contiguous()

    with _scoped_stream(positions.device):
        if use_matrix:
            if neighbor_matrix.numel() == 0 or neighbor_matrix.shape[1] == 0:
                return grad_cell.to(cell.dtype)
            wp.launch_tiled(
                _literal_cell_grad_matrix_tiled_kernel,
                dim=[positions.shape[0]],
                block_dim=REAL_SPACE_TILED_BLOCK_DIM,
                inputs=[
                    wp.from_torch(pos64, dtype=wp.vec3d, requires_grad=False),
                    wp.from_torch(chg64, dtype=wp.float64, requires_grad=False),
                    wp.from_torch(cell64, dtype=wp.mat33d, requires_grad=False),
                    wp.from_torch(alpha64, dtype=wp.float64, requires_grad=False),
                    wp.from_torch(batch_t, dtype=wp.int32, requires_grad=False),
                    int(is_batched),
                    wp.from_torch(
                        neighbor_matrix.to(torch.int32).contiguous(),
                        dtype=wp.int32,
                        requires_grad=False,
                    ),
                    wp.from_torch(
                        neighbor_matrix_shifts.to(torch.int32).contiguous(),
                        dtype=wp.vec3i,
                        requires_grad=False,
                    ),
                    int(mask_value),
                    wp.from_torch(ge64, dtype=wp.float64, requires_grad=False),
                    wp.from_torch(grad_cell, dtype=wp.mat33d, requires_grad=False),
                ],
                device=device,
            )
        else:
            if edge_i.numel() == 0:
                return grad_cell.to(cell.dtype)
            wp.launch(
                _literal_cell_grad_edges_kernel,
                dim=edge_i.shape[0],
                inputs=[
                    wp.from_torch(pos64, dtype=wp.vec3d, requires_grad=False),
                    wp.from_torch(chg64, dtype=wp.float64, requires_grad=False),
                    wp.from_torch(cell64, dtype=wp.mat33d, requires_grad=False),
                    wp.from_torch(alpha64, dtype=wp.float64, requires_grad=False),
                    wp.from_torch(batch_t, dtype=wp.int32, requires_grad=False),
                    int(is_batched),
                    wp.from_torch(
                        edge_i.to(torch.int32).contiguous(),
                        dtype=wp.int32,
                        requires_grad=False,
                    ),
                    wp.from_torch(
                        edge_j.to(torch.int32).contiguous(),
                        dtype=wp.int32,
                        requires_grad=False,
                    ),
                    wp.from_torch(
                        unit_shifts.to(torch.int32).contiguous(),
                        dtype=wp.vec3i,
                        requires_grad=False,
                    ),
                    wp.from_torch(ge64, dtype=wp.float64, requires_grad=False),
                    wp.from_torch(grad_cell, dtype=wp.mat33d, requires_grad=False),
                ],
                device=device,
            )
    return grad_cell.to(cell.dtype)


def _literal_cell_grad_backward(
    h_cell: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    edge_i: torch.Tensor,
    edge_j: torch.Tensor,
    unit_shifts: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
    use_matrix: bool,
    grad_energy_atom: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp HVP for the literal ``dE/dcell`` op."""
    device = wp.device_from_torch(positions.device)
    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    grad_positions = torch.zeros(
        num_atoms, 3, dtype=torch.float64, device=positions.device
    )
    grad_charges = torch.zeros(num_atoms, dtype=torch.float64, device=positions.device)
    grad_cell = torch.zeros(
        num_systems, 3, 3, dtype=torch.float64, device=positions.device
    )
    grad_grad_energy = torch.zeros(
        num_atoms, dtype=torch.float64, device=positions.device
    )
    if num_atoms == 0:
        return grad_positions, grad_charges, grad_cell, grad_grad_energy

    batch_t, is_batched = _batch_arg(batch_idx)
    pos64 = _as_f64_tensor(positions)
    chg64 = _as_f64_tensor(charges)
    cell64 = _as_f64_tensor(cell)
    alpha64 = _as_f64_tensor(alpha)
    h64 = h_cell.detach().to(torch.float64).contiguous()
    ge64 = grad_energy_atom.detach().to(torch.float64).contiguous()

    with _scoped_stream(positions.device):
        if use_matrix:
            if neighbor_matrix.numel() == 0 or neighbor_matrix.shape[1] == 0:
                return grad_positions, grad_charges, grad_cell, grad_grad_energy
            grad_cell_atom = torch.zeros(
                num_atoms, 3, 3, dtype=torch.float64, device=positions.device
            )
            wp.launch_tiled(
                _literal_cell_grad_backward_matrix_tiled_kernel,
                dim=[num_atoms],
                block_dim=REAL_SPACE_TILED_BLOCK_DIM,
                inputs=[
                    wp.from_torch(h64, dtype=wp.mat33d, requires_grad=False),
                    wp.from_torch(pos64, dtype=wp.vec3d, requires_grad=False),
                    wp.from_torch(chg64, dtype=wp.float64, requires_grad=False),
                    wp.from_torch(cell64, dtype=wp.mat33d, requires_grad=False),
                    wp.from_torch(alpha64, dtype=wp.float64, requires_grad=False),
                    wp.from_torch(batch_t, dtype=wp.int32, requires_grad=False),
                    int(is_batched),
                    wp.from_torch(
                        neighbor_matrix.to(torch.int32).contiguous(),
                        dtype=wp.int32,
                        requires_grad=False,
                    ),
                    wp.from_torch(
                        neighbor_matrix_shifts.to(torch.int32).contiguous(),
                        dtype=wp.vec3i,
                        requires_grad=False,
                    ),
                    int(mask_value),
                    wp.from_torch(ge64, dtype=wp.float64, requires_grad=False),
                    wp.from_torch(grad_positions, dtype=wp.vec3d, requires_grad=False),
                    wp.from_torch(grad_charges, dtype=wp.float64, requires_grad=False),
                    wp.from_torch(grad_cell_atom, dtype=wp.mat33d, requires_grad=False),
                    wp.from_torch(
                        grad_grad_energy, dtype=wp.float64, requires_grad=False
                    ),
                ],
                device=device,
            )
            if is_batched:
                grad_cell = torch.stack(
                    [
                        grad_cell_atom[batch_t == system_idx].sum(dim=0)
                        for system_idx in range(num_systems)
                    ],
                    dim=0,
                )
            else:
                grad_cell = grad_cell_atom.sum(dim=0, keepdim=True)
        else:
            if edge_i.numel() == 0:
                return grad_positions, grad_charges, grad_cell, grad_grad_energy
            wp.launch(
                _literal_cell_grad_backward_edges_kernel,
                dim=edge_i.shape[0],
                inputs=[
                    wp.from_torch(h64, dtype=wp.mat33d, requires_grad=False),
                    wp.from_torch(pos64, dtype=wp.vec3d, requires_grad=False),
                    wp.from_torch(chg64, dtype=wp.float64, requires_grad=False),
                    wp.from_torch(cell64, dtype=wp.mat33d, requires_grad=False),
                    wp.from_torch(alpha64, dtype=wp.float64, requires_grad=False),
                    wp.from_torch(batch_t, dtype=wp.int32, requires_grad=False),
                    int(is_batched),
                    wp.from_torch(
                        edge_i.to(torch.int32).contiguous(),
                        dtype=wp.int32,
                        requires_grad=False,
                    ),
                    wp.from_torch(
                        edge_j.to(torch.int32).contiguous(),
                        dtype=wp.int32,
                        requires_grad=False,
                    ),
                    wp.from_torch(
                        unit_shifts.to(torch.int32).contiguous(),
                        dtype=wp.vec3i,
                        requires_grad=False,
                    ),
                    wp.from_torch(ge64, dtype=wp.float64, requires_grad=False),
                    wp.from_torch(grad_positions, dtype=wp.vec3d, requires_grad=False),
                    wp.from_torch(grad_charges, dtype=wp.float64, requires_grad=False),
                    wp.from_torch(grad_cell, dtype=wp.mat33d, requires_grad=False),
                    wp.from_torch(
                        grad_grad_energy, dtype=wp.float64, requires_grad=False
                    ),
                ],
                device=device,
            )
    return grad_positions, grad_charges, grad_cell, grad_grad_energy


def _literal_cell_grad_fake(positions, charges, cell, *args):
    """Fake literal-cell forward output."""
    return cell.new_empty(cell.shape)


def _literal_cell_grad_backward_fake(h_cell, positions, charges, cell, *args):
    """Fake literal-cell backward outputs."""
    grad_energy_atom = args[-1]
    return (
        positions.new_empty(positions.shape, dtype=torch.float64),
        charges.new_empty(charges.shape, dtype=torch.float64),
        cell.new_empty(cell.shape, dtype=torch.float64),
        grad_energy_atom.new_empty(grad_energy_atom.shape, dtype=torch.float64),
    )


def _matrix_to_edges(
    neighbor_matrix: torch.Tensor, neighbor_matrix_shifts: torch.Tensor, mask_value: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten a neighbor matrix to ``(edge_i, edge_j, unit_shifts)``.

    Used by the CSR/edge-list double-backward path and as the Torch analytic
    oracle in tests; the matrix double-backward path keeps the matrix layout and
    launches the tiled literal-cell Warp op directly.
    """
    num_atoms, max_nbr = neighbor_matrix.shape
    device = neighbor_matrix.device
    rows = (
        torch.arange(num_atoms, device=device).unsqueeze(1).expand(num_atoms, max_nbr)
    )
    valid = neighbor_matrix != mask_value
    edge_i = rows[valid].to(torch.long)
    edge_j = neighbor_matrix[valid].to(torch.long)
    unit_shifts = neighbor_matrix_shifts[valid]
    return edge_i, edge_j, unit_shifts


def _scatter_dedcell_cache(
    dedcell: torch.Tensor,
    grad_energy_atom: torch.Tensor,
    batch_idx: torch.Tensor | None,
    num_systems: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reduce atom-major literal ``dE/dcell`` cache to per-system cell grads."""
    num_atoms = dedcell.shape[0]
    if batch_idx is None:
        sys_of_atom = torch.zeros(num_atoms, dtype=torch.long, device=dedcell.device)
    else:
        sys_of_atom = batch_idx.long()
    weighted = grad_energy_atom.to(torch.float64).view(-1, 1, 1) * dedcell.to(
        torch.float64
    )
    return (
        torch.zeros(
            (num_systems, 3, 3),
            dtype=torch.float64,
            device=dedcell.device,
        )
        .index_add(0, sys_of_atom, weighted)
        .to(dtype)
    )


class _CachedLiteralCellGrad(torch.autograd.Function):
    """Return cached literal ``dE/dcell`` and use the existing HVP in backward."""

    @staticmethod
    def forward(
        ctx,
        cached_grad_cell,
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        edge_i,
        edge_j,
        unit_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
        use_matrix,
        grad_energy_atom,
    ):
        ctx.save_for_backward(
            positions,
            charges,
            cell,
            alpha,
            batch_idx,
            edge_i,
            edge_j,
            unit_shifts,
            neighbor_matrix,
            neighbor_matrix_shifts,
            grad_energy_atom,
        )
        ctx.mask_value = mask_value
        ctx.use_matrix = use_matrix
        return cached_grad_cell

    @staticmethod
    def backward(ctx, h_cell):
        (
            positions,
            charges,
            cell,
            alpha,
            batch_idx,
            edge_i,
            edge_j,
            unit_shifts,
            neighbor_matrix,
            neighbor_matrix_shifts,
            grad_energy_atom,
        ) = ctx.saved_tensors
        grad_positions, grad_charges, grad_cell, grad_grad_energy = (
            _literal_cell_grad_backward(
                h_cell,
                positions,
                charges,
                cell,
                alpha,
                batch_idx,
                edge_i,
                edge_j,
                unit_shifts,
                neighbor_matrix,
                neighbor_matrix_shifts,
                int(ctx.mask_value),
                bool(ctx.use_matrix),
                grad_energy_atom,
            )
        )
        return (
            None,
            grad_positions,
            grad_charges,
            grad_cell,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_grad_energy,
        )


class _RealCellGrad(torch.autograd.Function):
    """Inject the literal real-space ``dE/dcell`` into the autograd graph.

    Forward is a value-zero pass-through ``(N,)`` added to the kernel energy, so the
    public energy *value* is unchanged. ``backward`` returns ``dE/dcell`` for the
    ``cell`` slot only (``positions`` / ``charges`` first-order gradients are owned by
    the Warp chain) via one of three paths:

    1. **matrix first-order** (``dedcell`` is the real ``(N,3,3)`` cache the
       ``cell_literal`` forward kernel fused): a pure scatter
       ``grad_cell = index_add(grad_energy_atom * dedcell, by system)`` -- no kernel
       launch, no edge list. Matrix and CSR/list layouts both provide this cache.
    2. **first-order fallback** (``dedcell`` is the ``(0,3,3)`` placeholder): the
       edge-parallel Warp kernel :func:`_real_cell_grad_via_kernel`.
    3. **double-backward** (``torch.is_grad_enabled()``, stress-loss
       ``create_graph``): the Warp-backed literal-cell op
       ``nvalchemiops::ewald_real_literal_cell_grad`` returns the same closed-form
       ``dE/dcell`` value and provides a custom HVP for the ``dE/dR dcell`` /
       ``dE/dq dcell`` / ``dE/dcell^2`` cross terms. The cached ``dedcell`` is
       detached, so it cannot serve this path.

    The edge list is NOT built eagerly for the matrix layout (it is ``num_atoms *
    max_nbr`` entries -- several ms at production neighbor counts). The raw
    ``neighbor_matrix`` / ``neighbor_matrix_shifts`` are saved so the rare
    ``create_graph`` branch can launch the tiled matrix HVP directly. The CSR layout
    passes its edges directly and an empty ``neighbor_matrix``.
    """

    @staticmethod
    def forward(
        positions,
        charges,
        cell,
        alpha,
        edge_i,
        edge_j,
        unit_shifts,
        batch_idx,
        dedcell,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
    ):
        return torch.zeros(
            positions.shape[0], dtype=torch.float64, device=positions.device
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            positions,
            charges,
            cell,
            alpha,
            edge_i,
            edge_j,
            unit_shifts,
            batch_idx,
            dedcell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            mask_value,
        ) = inputs
        ctx.save_for_backward(
            positions,
            charges,
            cell,
            alpha,
            edge_i,
            edge_j,
            unit_shifts,
            dedcell,
            neighbor_matrix,
            neighbor_matrix_shifts,
        )
        ctx.batch_idx = batch_idx
        ctx.mask_value = mask_value

    @staticmethod
    def backward(ctx, grad_energy_atom):
        (
            positions,
            charges,
            cell,
            alpha,
            edge_i,
            edge_j,
            unit_shifts,
            dedcell,
            neighbor_matrix,
            neighbor_matrix_shifts,
        ) = ctx.saved_tensors
        num_atoms = positions.shape[0]
        if torch.is_grad_enabled():
            # Path 3: stress-loss double-backward. dE/dcell must stay differentiable
            # so the dE/dR dcell / dE/dq dcell / dE/dcell^2 cross terms flow. Use
            # the Warp-backed literal-cell op instead of materializing the full
            # edge-level Torch analytic graph.
            if ctx.batch_idx is None:
                batch_idx = torch.zeros(0, dtype=torch.int32, device=positions.device)
            else:
                batch_idx = ctx.batch_idx
            if dedcell.shape[0] == num_atoms:
                grad_cell_cached = _scatter_dedcell_cache(
                    dedcell,
                    grad_energy_atom,
                    ctx.batch_idx,
                    cell.shape[0],
                    dtype=cell.dtype,
                )
                with torch.enable_grad():
                    grad_cell = _CachedLiteralCellGrad.apply(
                        grad_cell_cached,
                        positions,
                        charges,
                        cell,
                        alpha,
                        batch_idx,
                        edge_i,
                        edge_j,
                        unit_shifts,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        int(ctx.mask_value),
                        bool(
                            neighbor_matrix.numel() > 0 and neighbor_matrix.shape[1] > 0
                        ),
                        grad_energy_atom,
                    )
            else:
                with torch.enable_grad():
                    grad_cell = torch.ops.nvalchemiops.ewald_real_literal_cell_grad(
                        positions,
                        charges,
                        cell,
                        alpha,
                        batch_idx,
                        edge_i,
                        edge_j,
                        unit_shifts,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        int(ctx.mask_value),
                        bool(
                            neighbor_matrix.numel() > 0 and neighbor_matrix.shape[1] > 0
                        ),
                        grad_energy_atom,
                    )
        elif dedcell.shape[0] == num_atoms:
            # Path 1: matrix first-order. The forward kernel already fused the
            # per-atom literal dE/dcell block, so the backward is a pure scatter
            # (weight by the per-atom energy cotangent, reduce per system) -- no
            # kernel launch, no edge-list traversal.
            grad_cell = _scatter_dedcell_cache(
                dedcell,
                grad_energy_atom,
                ctx.batch_idx,
                cell.shape[0],
                dtype=cell.dtype,
            )
        else:
            # Path 2: first-order without a forward-fused per-atom cell cache.
            # CSR uses the edge kernel. Matrix layouts keep the raw neighbor
            # matrix so lazy/mode-selective cell-grad can use the tiled literal
            # op without materializing the full edge list.
            use_matrix = bool(
                neighbor_matrix.numel() > 0 and neighbor_matrix.shape[1] > 0
            )
            if use_matrix:
                if ctx.batch_idx is None:
                    batch_idx = torch.zeros(
                        0, dtype=torch.int32, device=positions.device
                    )
                else:
                    batch_idx = ctx.batch_idx
                grad_cell = torch.ops.nvalchemiops.ewald_real_literal_cell_grad(
                    positions,
                    charges,
                    cell,
                    alpha,
                    batch_idx,
                    edge_i,
                    edge_j,
                    unit_shifts,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    int(ctx.mask_value),
                    True,
                    grad_energy_atom,
                )
            else:
                grad_cell = _real_cell_grad_via_kernel(
                    positions,
                    charges,
                    cell,
                    alpha,
                    edge_i,
                    edge_j,
                    unit_shifts,
                    ctx.batch_idx,
                    grad_energy_atom,
                )
        # One grad slot per input: cell is index 2; the rest are non-diff.
        return (
            None,
            None,
            grad_cell,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def real_space_cell_connect(
    energy: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    edge_i: torch.Tensor,
    edge_j: torch.Tensor,
    unit_shifts: torch.Tensor,
    batch_idx: torch.Tensor | None,
    dedcell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
) -> torch.Tensor:
    """Add the value-zero cell-gradient connector to the kernel ``energy``.

    ``dedcell`` is the forward-fused per-atom literal ``dE/dcell`` cache (matrix
    layout) or a ``(0,3,3)`` placeholder (CSR / no fusion). For the matrix layout
    ``edge_i`` / ``edge_j`` / ``unit_shifts`` are passed empty and the raw
    ``neighbor_matrix`` / ``neighbor_matrix_shifts`` carry the connectivity for the
    deferred (double-backward-only) tiled HVP; CSR passes its edges directly and an
    empty ``neighbor_matrix``. The connector's backward selects the matching path.
    See :class:`_RealCellGrad`.
    """
    register_ewald_real_ops()
    return energy + _RealCellGrad.apply(
        positions,
        charges,
        cell,
        alpha,
        edge_i,
        edge_j,
        unit_shifts,
        batch_idx,
        dedcell,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
    )
