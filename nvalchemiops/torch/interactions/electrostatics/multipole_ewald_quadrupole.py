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
"""LMAX=2 (charges + dipoles + Cartesian quadrupoles) real-space Ewald torch wrappers.

Provides :func:`multipole_real_space_quadrupole_energy` (single + batched), a
``torch.library`` custom-op chain returning per-atom energies (``(N,)``
single-system, ``(N_total,)`` batched) that supports non-uniform per-atom
upstream gradients.

``quadrupoles`` must be Cartesian ``(N, 3, 3)`` and symmetric (both off-diagonal
triangles populated); the emitted gradient is the symmetric free-index partial
(:math:`\\partial E / \\partial Q_{\\alpha\\beta} = \\partial E / \\partial Q_{\\beta\\alpha}`).

Double-backward (``create_graph=True``) is supported for positions, charges,
dipoles, and quadrupoles. Cell double-backward is not supported.
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.interactions.electrostatics.multipole_ewald_cell_grad import (
    batch_multipole_real_space_quadrupole_csr_cell_grad,
    multipole_real_space_quadrupole_csr_cell_grad,
)
from nvalchemiops.interactions.electrostatics.multipole_ewald_kernels import (
    batch_multipole_real_space_quadrupole_csr_energy,
    batch_multipole_real_space_quadrupole_csr_energy_fused,
    multipole_real_space_quadrupole_csr_energy,
    multipole_real_space_quadrupole_csr_energy_fused,
)
from nvalchemiops.interactions.electrostatics.multipole_ewald_quadrupole_2nd_backward import (
    batch_multipole_real_space_quadrupole_csr_cell_grad_backward,
    batch_multipole_real_space_quadrupole_csr_energy_2nd_backward,
    multipole_real_space_quadrupole_csr_cell_grad_backward,
    multipole_real_space_quadrupole_csr_energy_2nd_backward,
)
from nvalchemiops.torch.types import (
    get_wp_dtype,
    get_wp_mat_dtype,
    get_wp_vec_dtype,
)

# ---------------------------------------------------------------------------
# l_max = 2 single-system real-space — torch.library.custom_op chain
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "nvalchemiops::multipole_real_space_quadrupole", mutates_args=()
)
def _rs_quadrupole_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    half_neighbor_list: bool,
) -> torch.Tensor:
    """Per-atom l_max=2 real-space energy (charges + dipoles + quadrupoles)."""
    wp_device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(positions.dtype)
    wp_vec = get_wp_vec_dtype(positions.dtype)
    wp_mat = get_wp_mat_dtype(positions.dtype)
    energies = torch.zeros(
        positions.shape[0], dtype=torch.float64, device=positions.device
    )
    multipole_real_space_quadrupole_csr_energy(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(energies, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return energies


@_rs_quadrupole_op.register_fake
def _(
    positions,
    charges,
    dipoles,
    quadrupoles,
    cell,
    sigma,
    alpha,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    half_neighbor_list,
):
    return positions.new_empty((positions.shape[0],), dtype=torch.float64)


@torch.library.custom_op(
    "nvalchemiops::multipole_real_space_quadrupole_backward", mutates_args=()
)
def _rs_quadrupole_backward_op(
    grad_energies: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    half_neighbor_list: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-order moment gradients (positions, charges, dipoles, quadrupoles)."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    grad_pos = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    grad_q = torch.zeros(num_atoms, dtype=input_dtype, device=device)
    grad_mu = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    grad_Q = torch.zeros((num_atoms, 3, 3), dtype=input_dtype, device=device)
    energies_scratch = torch.zeros(num_atoms, dtype=torch.float64, device=device)
    grad_energies_f64 = grad_energies.detach().to(torch.float64).contiguous()
    multipole_real_space_quadrupole_csr_energy_fused(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(grad_energies_f64, dtype=wp.float64),
        wp.from_torch(energies_scratch, dtype=wp.float64),
        wp.from_torch(grad_pos, dtype=wp_vec),
        wp.from_torch(grad_q, dtype=wp_scalar),
        wp.from_torch(grad_mu, dtype=wp_vec),
        wp.from_torch(grad_Q, dtype=wp_mat),
        with_pos_grad=True,
        with_charge_grad=True,
        with_dipole_grad=True,
        with_quad_grad=True,
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_pos, grad_q, grad_mu, grad_Q


@_rs_quadrupole_backward_op.register_fake
def _(
    grad_energies,
    positions,
    charges,
    dipoles,
    quadrupoles,
    cell,
    sigma,
    alpha,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    half_neighbor_list,
):
    return (
        torch.empty_like(positions),
        torch.empty_like(charges),
        torch.empty_like(dipoles),
        torch.empty_like(quadrupoles),
    )


@torch.library.custom_op(
    "nvalchemiops::multipole_real_space_quadrupole_cell_grad", mutates_args=()
)
def _rs_quadrupole_cell_grad_op(
    grad_energies: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    half_neighbor_list: bool,
) -> torch.Tensor:
    """Cell gradient (stress) for the l_max=2 real-space energy."""
    input_dtype = positions.dtype
    wp_device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    grad_cell = torch.zeros_like(cell.detach()).contiguous()
    grad_energies_f64 = grad_energies.detach().to(torch.float64).contiguous()
    multipole_real_space_quadrupole_csr_cell_grad(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(grad_energies_f64, dtype=wp.float64),
        wp.from_torch(grad_cell, dtype=wp_mat),
        device=str(wp_device),
        half_neighbor_list=half_neighbor_list,
    )
    return grad_cell


@_rs_quadrupole_cell_grad_op.register_fake
def _(
    grad_energies,
    positions,
    charges,
    dipoles,
    quadrupoles,
    cell,
    sigma,
    alpha,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    half_neighbor_list,
):
    return torch.empty_like(cell)


def _rs_quadrupole_cell_grad_setup(ctx, inputs, output):
    """Save inputs for the l=2 cell-grad double-backward (stress-loss)."""
    (
        grad_energies,
        positions,
        charges,
        dipoles,
        quadrupoles,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        half_neighbor_list,
    ) = inputs
    ctx.save_for_backward(
        grad_energies,
        positions,
        charges,
        dipoles,
        quadrupoles,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    )
    ctx.half_neighbor_list = half_neighbor_list


def _rs_quadrupole_cell_grad_backward(ctx, g_cell):
    """∂/∂{grad_energies, positions, charges, dipoles, quadrupoles, cell} of
    ⟨g_cell, dE/dcell⟩ (l=2 stress-loss)."""
    (
        grad_energies,
        positions,
        charges,
        dipoles,
        quadrupoles,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    ) = ctx.saved_tensors
    need = ctx.needs_input_grad
    if not any(need[:6]):
        return (None,) * 12
    device = positions.device
    wp_device = wp.device_from_torch(device)
    dtype = positions.dtype
    wp_scalar = get_wp_dtype(dtype)
    wp_vec = get_wp_vec_dtype(dtype)
    wp_mat = get_wp_mat_dtype(dtype)
    n = positions.shape[0]
    scale_val = 1.0 if ctx.half_neighbor_list else 0.5
    scale = torch.tensor([scale_val], dtype=torch.float64, device=device)
    grad_ge = torch.zeros(n, dtype=torch.float64, device=device)
    grad_positions = torch.zeros((n, 3), dtype=dtype, device=device)
    grad_charges = torch.zeros(n, dtype=dtype, device=device)
    grad_dipoles = torch.zeros((n, 3), dtype=dtype, device=device)
    grad_quadrupoles = torch.zeros((n, 3, 3), dtype=dtype, device=device)
    grad_cell_cc = torch.zeros((1, 3, 3), dtype=dtype, device=device)
    multipole_real_space_quadrupole_csr_cell_grad_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(scale, dtype=wp.float64),
        wp.from_torch(
            grad_energies.detach().to(torch.float64).contiguous(), dtype=wp.float64
        ),
        wp.from_torch(g_cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(grad_ge, dtype=wp.float64),
        wp.from_torch(grad_positions, dtype=wp_vec),
        wp.from_torch(grad_charges, dtype=wp_scalar),
        wp.from_torch(grad_dipoles, dtype=wp_vec),
        wp.from_torch(grad_quadrupoles, dtype=wp_mat),
        wp.from_torch(grad_cell_cc, dtype=wp_mat),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return (
        grad_ge.to(grad_energies.dtype) if need[0] else None,
        grad_positions if need[1] else None,
        grad_charges if need[2] else None,
        grad_dipoles if need[3] else None,
        grad_quadrupoles if need[4] else None,
        grad_cell_cc.reshape(cell.shape) if need[5] else None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "nvalchemiops::multipole_real_space_quadrupole_cell_grad",
    _rs_quadrupole_cell_grad_backward,
    setup_context=_rs_quadrupole_cell_grad_setup,
)


@torch.library.custom_op(
    "nvalchemiops::multipole_real_space_quadrupole_double_backward", mutates_args=()
)
def _rs_quadrupole_double_backward_op(
    gg_pos: torch.Tensor,
    gg_q: torch.Tensor,
    gg_mu: torch.Tensor,
    gg_Q: torch.Tensor,
    grad_energies: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    half_neighbor_list: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Second-order backward (``create_graph``) for the moment gradients."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    gg_ge_2nd = torch.zeros(num_atoms, dtype=torch.float64, device=device)
    gg_pos_2nd = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    gg_q_2nd = torch.zeros(num_atoms, dtype=input_dtype, device=device)
    gg_mu_2nd = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    gg_Q_2nd = torch.zeros((num_atoms, 3, 3), dtype=input_dtype, device=device)
    multipole_real_space_quadrupole_csr_energy_2nd_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(
            grad_energies.detach().to(torch.float64).contiguous(), dtype=wp.float64
        ),
        wp.from_torch(gg_pos.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(gg_q.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(gg_mu.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(gg_Q.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(gg_ge_2nd, dtype=wp.float64),
        wp.from_torch(gg_pos_2nd, dtype=wp_vec),
        wp.from_torch(gg_q_2nd, dtype=wp_scalar),
        wp.from_torch(gg_mu_2nd, dtype=wp_vec),
        wp.from_torch(gg_Q_2nd, dtype=wp_mat),
        device=str(wp_device),
        half_neighbor_list=half_neighbor_list,
    )
    return gg_ge_2nd, gg_pos_2nd, gg_q_2nd, gg_mu_2nd, gg_Q_2nd


@_rs_quadrupole_double_backward_op.register_fake
def _(
    gg_pos,
    gg_q,
    gg_mu,
    gg_Q,
    grad_energies,
    positions,
    charges,
    dipoles,
    quadrupoles,
    cell,
    sigma,
    alpha,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    half_neighbor_list,
):
    return (
        grad_energies.new_empty((positions.shape[0],), dtype=torch.float64),
        torch.empty_like(positions),
        torch.empty_like(charges),
        torch.empty_like(dipoles),
        torch.empty_like(quadrupoles),
    )


def _rs_quadrupole_backward_setup(ctx, inputs, output):
    """Save the forward inputs the moment-backward op needs for double-backward."""
    (
        grad_energies,
        positions,
        charges,
        dipoles,
        quadrupoles,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        half_neighbor_list,
    ) = inputs
    ctx.save_for_backward(
        grad_energies,
        positions,
        charges,
        dipoles,
        quadrupoles,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    )
    ctx.half_neighbor_list = half_neighbor_list


def _rs_quadrupole_backward_backward(ctx, gg_pos, gg_q, gg_mu, gg_Q):
    """Second-order backward: cotangents may be None -> substitute zeros."""
    (
        grad_energies,
        positions,
        charges,
        dipoles,
        quadrupoles,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    ) = ctx.saved_tensors
    gg_pos = gg_pos if gg_pos is not None else torch.zeros_like(positions)
    gg_q = gg_q if gg_q is not None else torch.zeros_like(charges)
    gg_mu = gg_mu if gg_mu is not None else torch.zeros_like(dipoles)
    gg_Q = gg_Q if gg_Q is not None else torch.zeros_like(quadrupoles)
    gg_ge_2nd, gg_pos_2nd, gg_q_2nd, gg_mu_2nd, gg_Q_2nd = (
        torch.ops.nvalchemiops.multipole_real_space_quadrupole_double_backward(
            gg_pos,
            gg_q,
            gg_mu,
            gg_Q,
            grad_energies,
            positions,
            charges,
            dipoles,
            quadrupoles,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            ctx.half_neighbor_list,
        )
    )
    # 12 inputs: grad_energies, positions, charges, dipoles, quadrupoles, cell,
    # sigma, alpha, idx_j, neighbor_ptr, unit_shifts, half_neighbor_list.
    return (
        gg_ge_2nd,
        gg_pos_2nd,
        gg_q_2nd,
        gg_mu_2nd,
        gg_Q_2nd,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _rs_quadrupole_setup(ctx, inputs, output):
    """Save forward inputs + the half-list flag for backward."""
    (
        positions,
        charges,
        dipoles,
        quadrupoles,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        half_neighbor_list,
    ) = inputs
    ctx.save_for_backward(
        positions,
        charges,
        dipoles,
        quadrupoles,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    )
    ctx.half_neighbor_list = half_neighbor_list


def _rs_quadrupole_backward(ctx, grad_energies):
    """Backward: fused moment-grad kernel + (only if needed) the cell-grad kernel."""
    (
        positions,
        charges,
        dipoles,
        quadrupoles,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    ) = ctx.saved_tensors
    need = ctx.needs_input_grad
    grad_pos, grad_q, grad_mu, grad_Q = (
        torch.ops.nvalchemiops.multipole_real_space_quadrupole_backward(
            grad_energies,
            positions,
            charges,
            dipoles,
            quadrupoles,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            ctx.half_neighbor_list,
        )
    )
    grad_cell = None
    if need[4]:
        grad_cell = torch.ops.nvalchemiops.multipole_real_space_quadrupole_cell_grad(
            grad_energies,
            positions,
            charges,
            dipoles,
            quadrupoles,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            ctx.half_neighbor_list,
        )
    # 11 inputs: positions, charges, dipoles, quadrupoles, cell, sigma, alpha,
    # idx_j, neighbor_ptr, unit_shifts, half_neighbor_list.
    return (
        grad_pos if need[0] else None,
        grad_q if need[1] else None,
        grad_mu if need[2] else None,
        grad_Q if need[3] else None,
        grad_cell,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "nvalchemiops::multipole_real_space_quadrupole_backward",
    _rs_quadrupole_backward_backward,
    setup_context=_rs_quadrupole_backward_setup,
)
torch.library.register_autograd(
    "nvalchemiops::multipole_real_space_quadrupole",
    _rs_quadrupole_backward,
    setup_context=_rs_quadrupole_setup,
)


def multipole_real_space_quadrupole_energy(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    cell: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None = None,
    half_neighbor_list: bool = False,
) -> torch.Tensor:
    """GTO-Ewald real-space LMAX=2 energy with autograd.

    Single-system; returns ``(N,)`` per-atom energies (``float64``). The caller
    owns the atom-global reduction (``.sum()`` for total energy). Non-uniform
    per-atom upstream gradients are supported.

    Backward returns gradients w.r.t. positions, charges, dipoles, and
    quadrupoles (:math:`\\partial E / \\partial Q` is the symmetric free-index
    partial). ``cell.requires_grad`` is handled via a separate cell-grad kernel
    launch. ``create_graph=True`` works for positions, charges, dipoles, and
    quadrupoles, but not when ``cell.requires_grad``.

    Single-system vs batched dispatch
    ---------------------------------
    Pass ``cell`` of shape ``(3, 3)`` / ``(1, 3, 3)`` (single) or
    ``(B, 3, 3)`` (batched) and set ``batch_idx`` to select the batched path.
    In batched mode ``sigma``/``alpha`` are per-system ``(B,)`` tensors and the
    return is per-atom ``(N_total,)`` (uniform with the single-system path; the
    caller ``scatter_add``s for per-system totals).

    Parameters
    ----------
    positions : torch.Tensor, shape ``(N, 3)``
    charges : torch.Tensor, shape ``(N,)``
    dipoles : torch.Tensor, shape ``(N, 3)``
        Cartesian.
    quadrupoles : torch.Tensor, shape ``(N, 3, 3)``
        Cartesian, symmetric.
    cell : torch.Tensor, shape ``(3, 3)`` / ``(1, 3, 3)`` or ``(B, 3, 3)``
    idx_j, neighbor_ptr, unit_shifts : torch.Tensor
        CSR neighbor list.
    sigma, alpha : torch.Tensor, shape ``(1,)`` or ``(B,)``
        GTO density-basis width :math:`\\sigma` and Ewald splitting parameter
        :math:`\\alpha` (per-system ``(B,)`` when batched).
    batch_idx : torch.Tensor, optional, shape ``(N_total,)``, int32
        Per-atom system index. Required when ``cell`` is ``(B, 3, 3)``; must be
        ``None`` for a single cell.
    half_neighbor_list : bool, default False
        Set when the CSR neighbor list stores each pair once.

    Returns
    -------
    torch.Tensor, ``float64``
        Per-atom real-space energy ``(N,)`` (single) or ``(N_total,)``
        (batched), before the :math:`F/(4\\pi)` Coulomb scale applied by the
        caller.
    """
    if batch_idx is not None:
        return _batch_multipole_real_space_quadrupole_energy(
            positions,
            charges,
            dipoles,
            quadrupoles,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            batch_idx,
            half_neighbor_list=half_neighbor_list,
        )
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"positions must be (N, 3), got {tuple(positions.shape)}")
    if cell.ndim == 2 and cell.shape == (3, 3):
        cell = cell.unsqueeze(0)
    if cell.ndim != 3 or cell.shape[-2:] != (3, 3) or cell.shape[0] != 1:
        raise ValueError(f"cell must be (3, 3) or (1, 3, 3); got {tuple(cell.shape)}")
    if dipoles.shape != (positions.shape[0], 3):
        raise ValueError(f"dipoles must be (N, 3); got {tuple(dipoles.shape)}")
    if quadrupoles.shape != (positions.shape[0], 3, 3):
        raise ValueError(
            f"quadrupoles must be (N, 3, 3); got {tuple(quadrupoles.shape)}"
        )
    return torch.ops.nvalchemiops.multipole_real_space_quadrupole(
        positions,
        charges,
        dipoles,
        quadrupoles,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        half_neighbor_list,
    )


# ---------------------------------------------------------------------------
# Batched l_max = 2 real-space — torch.library.custom_op chain
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_real_space_quadrupole", mutates_args=()
)
def _batch_rs_quadrupole_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    cells: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    batch_idx: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    half_neighbor_list: bool,
) -> torch.Tensor:
    """Per-system batched l_max=2 real-space energy ``(B,)`` (scatter_add)."""
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(positions.dtype)
    wp_vec = get_wp_vec_dtype(positions.dtype)
    wp_mat = get_wp_mat_dtype(positions.dtype)
    energies = torch.zeros(positions.shape[0], dtype=torch.float64, device=device)
    batch_multipole_real_space_quadrupole_csr_energy(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(batch_idx.detach().to(torch.int32).contiguous(), dtype=wp.int32),
        wp.from_torch(energies, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    # Per-atom return (the caller owns the atom-global reduction / scatter_add);
    # uniform with the l<=1 paths.
    return energies


@_batch_rs_quadrupole_op.register_fake
def _(
    positions,
    charges,
    dipoles,
    quadrupoles,
    cells,
    sigmas,
    alphas,
    batch_idx,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    half_neighbor_list,
):
    return positions.new_empty((positions.shape[0],), dtype=torch.float64)


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_real_space_quadrupole_backward", mutates_args=()
)
def _batch_rs_quadrupole_backward_op(
    grad_energies: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    cells: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    batch_idx: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    half_neighbor_list: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched l_max=2 moment gradients (grad_energies is per-atom)."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    grad_pos = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    grad_q = torch.zeros(num_atoms, dtype=input_dtype, device=device)
    grad_mu = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    grad_Q = torch.zeros((num_atoms, 3, 3), dtype=input_dtype, device=device)
    energies_scratch = torch.zeros(num_atoms, dtype=torch.float64, device=device)
    grad_energies_f64 = grad_energies.detach().to(torch.float64).contiguous()
    batch_multipole_real_space_quadrupole_csr_energy_fused(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(batch_idx.detach().to(torch.int32).contiguous(), dtype=wp.int32),
        wp.from_torch(grad_energies_f64, dtype=wp.float64),
        wp.from_torch(energies_scratch, dtype=wp.float64),
        wp.from_torch(grad_pos, dtype=wp_vec),
        wp.from_torch(grad_q, dtype=wp_scalar),
        wp.from_torch(grad_mu, dtype=wp_vec),
        wp.from_torch(grad_Q, dtype=wp_mat),
        with_pos_grad=True,
        with_charge_grad=True,
        with_dipole_grad=True,
        with_quad_grad=True,
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_pos, grad_q, grad_mu, grad_Q


@_batch_rs_quadrupole_backward_op.register_fake
def _(
    grad_energies,
    positions,
    charges,
    dipoles,
    quadrupoles,
    cells,
    sigmas,
    alphas,
    batch_idx,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    half_neighbor_list,
):
    return (
        torch.empty_like(positions),
        torch.empty_like(charges),
        torch.empty_like(dipoles),
        torch.empty_like(quadrupoles),
    )


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_real_space_quadrupole_cell_grad", mutates_args=()
)
def _batch_rs_quadrupole_cell_grad_op(
    grad_energies: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    cells: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    batch_idx: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    half_neighbor_list: bool,
) -> torch.Tensor:
    """Batched l_max=2 cell gradient (stress); grad_energies is per-atom."""
    input_dtype = positions.dtype
    wp_device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    grad_cells = torch.zeros_like(cells.detach()).contiguous()
    ge_f64 = grad_energies.detach().to(torch.float64).contiguous()
    # The cell-grad kernel reads sigma[0]/alpha[0]; all systems share them.
    batch_multipole_real_space_quadrupole_csr_cell_grad(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(batch_idx.detach().to(torch.int32).contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas[:1].detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas[:1].detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(ge_f64, dtype=wp.float64),
        wp.from_torch(grad_cells, dtype=wp_mat),
        device=str(wp_device),
        half_neighbor_list=half_neighbor_list,
    )
    return grad_cells


@_batch_rs_quadrupole_cell_grad_op.register_fake
def _(
    grad_energies,
    positions,
    charges,
    dipoles,
    quadrupoles,
    cells,
    sigmas,
    alphas,
    batch_idx,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    half_neighbor_list,
):
    return torch.empty_like(cells)


def _batch_rs_quadrupole_cell_grad_setup(ctx, inputs, output):
    """Save inputs for the batched l=2 cell-grad double-backward."""
    (
        grad_energies,
        positions,
        charges,
        dipoles,
        quadrupoles,
        cells,
        sigmas,
        alphas,
        batch_idx,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        half_neighbor_list,
    ) = inputs
    ctx.save_for_backward(
        grad_energies,
        positions,
        charges,
        dipoles,
        quadrupoles,
        cells,
        sigmas,
        alphas,
        batch_idx,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    )
    ctx.half_neighbor_list = half_neighbor_list


def _batch_rs_quadrupole_cell_grad_backward(ctx, g_cell):
    """Batched ∂/∂{grad_energies, positions, charges, dipoles, quadrupoles,
    cells} of ⟨g_cell, dE/dcell⟩ (l=2)."""
    (
        grad_energies,
        positions,
        charges,
        dipoles,
        quadrupoles,
        cells,
        sigmas,
        alphas,
        batch_idx,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    ) = ctx.saved_tensors
    need = ctx.needs_input_grad
    if not any(need[:6]):
        return (None,) * 13
    device = positions.device
    wp_device = wp.device_from_torch(device)
    dtype = positions.dtype
    wp_scalar = get_wp_dtype(dtype)
    wp_vec = get_wp_vec_dtype(dtype)
    wp_mat = get_wp_mat_dtype(dtype)
    n = positions.shape[0]
    scale = torch.tensor(
        [1.0 if ctx.half_neighbor_list else 0.5], dtype=torch.float64, device=device
    )
    grad_ge = torch.zeros(n, dtype=torch.float64, device=device)
    grad_positions = torch.zeros((n, 3), dtype=dtype, device=device)
    grad_charges = torch.zeros(n, dtype=dtype, device=device)
    grad_dipoles = torch.zeros((n, 3), dtype=dtype, device=device)
    grad_quadrupoles = torch.zeros((n, 3, 3), dtype=dtype, device=device)
    grad_cells = torch.zeros_like(cells)
    batch_multipole_real_space_quadrupole_csr_cell_grad_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(scale, dtype=wp.float64),
        wp.from_torch(
            grad_energies.detach().to(torch.float64).contiguous(), dtype=wp.float64
        ),
        wp.from_torch(g_cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(grad_ge, dtype=wp.float64),
        wp.from_torch(grad_positions, dtype=wp_vec),
        wp.from_torch(grad_charges, dtype=wp_scalar),
        wp.from_torch(grad_dipoles, dtype=wp_vec),
        wp.from_torch(grad_quadrupoles, dtype=wp_mat),
        wp.from_torch(grad_cells, dtype=wp_mat),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return (
        grad_ge.to(grad_energies.dtype) if need[0] else None,
        grad_positions if need[1] else None,
        grad_charges if need[2] else None,
        grad_dipoles if need[3] else None,
        grad_quadrupoles if need[4] else None,
        grad_cells if need[5] else None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "nvalchemiops::batch_multipole_real_space_quadrupole_cell_grad",
    _batch_rs_quadrupole_cell_grad_backward,
    setup_context=_batch_rs_quadrupole_cell_grad_setup,
)


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_real_space_quadrupole_double_backward",
    mutates_args=(),
)
def _batch_rs_quadrupole_double_backward_op(
    gg_pos: torch.Tensor,
    gg_q: torch.Tensor,
    gg_mu: torch.Tensor,
    gg_Q: torch.Tensor,
    grad_energies: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    cells: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    batch_idx: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    half_neighbor_list: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched l_max=2 second-order backward (``create_graph``)."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    gg_ge_2nd = torch.zeros(num_atoms, dtype=torch.float64, device=device)
    gg_pos_2nd = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    gg_q_2nd = torch.zeros(num_atoms, dtype=input_dtype, device=device)
    gg_mu_2nd = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    gg_Q_2nd = torch.zeros((num_atoms, 3, 3), dtype=input_dtype, device=device)
    batch_multipole_real_space_quadrupole_csr_energy_2nd_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(batch_idx.detach().to(torch.int32).contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(
            grad_energies.detach().to(torch.float64).contiguous(), dtype=wp.float64
        ),
        wp.from_torch(gg_pos.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(gg_q.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(gg_mu.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(gg_Q.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(gg_ge_2nd, dtype=wp.float64),
        wp.from_torch(gg_pos_2nd, dtype=wp_vec),
        wp.from_torch(gg_q_2nd, dtype=wp_scalar),
        wp.from_torch(gg_mu_2nd, dtype=wp_vec),
        wp.from_torch(gg_Q_2nd, dtype=wp_mat),
        device=str(wp_device),
        half_neighbor_list=half_neighbor_list,
    )
    return gg_ge_2nd, gg_pos_2nd, gg_q_2nd, gg_mu_2nd, gg_Q_2nd


@_batch_rs_quadrupole_double_backward_op.register_fake
def _(
    gg_pos,
    gg_q,
    gg_mu,
    gg_Q,
    grad_energies,
    positions,
    charges,
    dipoles,
    quadrupoles,
    cells,
    sigmas,
    alphas,
    batch_idx,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    half_neighbor_list,
):
    return (
        grad_energies.new_empty((positions.shape[0],), dtype=torch.float64),
        torch.empty_like(positions),
        torch.empty_like(charges),
        torch.empty_like(dipoles),
        torch.empty_like(quadrupoles),
    )


def _batch_rs_quadrupole_backward_setup(ctx, inputs, output):
    """Save inputs for the batched moment-backward double-backward."""
    (
        grad_energies,
        positions,
        charges,
        dipoles,
        quadrupoles,
        cells,
        sigmas,
        alphas,
        batch_idx,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        half_neighbor_list,
    ) = inputs
    ctx.save_for_backward(
        grad_energies,
        positions,
        charges,
        dipoles,
        quadrupoles,
        cells,
        sigmas,
        alphas,
        batch_idx,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    )
    ctx.half_neighbor_list = half_neighbor_list


def _batch_rs_quadrupole_backward_backward(ctx, gg_pos, gg_q, gg_mu, gg_Q):
    """Batched second-order backward; cotangents may be None -> zeros."""
    (
        grad_energies,
        positions,
        charges,
        dipoles,
        quadrupoles,
        cells,
        sigmas,
        alphas,
        batch_idx,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    ) = ctx.saved_tensors
    gg_pos = gg_pos if gg_pos is not None else torch.zeros_like(positions)
    gg_q = gg_q if gg_q is not None else torch.zeros_like(charges)
    gg_mu = gg_mu if gg_mu is not None else torch.zeros_like(dipoles)
    gg_Q = gg_Q if gg_Q is not None else torch.zeros_like(quadrupoles)
    gg_ge_2nd, gg_pos_2nd, gg_q_2nd, gg_mu_2nd, gg_Q_2nd = (
        torch.ops.nvalchemiops.batch_multipole_real_space_quadrupole_double_backward(
            gg_pos,
            gg_q,
            gg_mu,
            gg_Q,
            grad_energies,
            positions,
            charges,
            dipoles,
            quadrupoles,
            cells,
            sigmas,
            alphas,
            batch_idx,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            ctx.half_neighbor_list,
        )
    )
    # 13 inputs: grad_energies, positions, charges, dipoles, quadrupoles, cells,
    # sigmas, alphas, batch_idx, idx_j, neighbor_ptr, unit_shifts, half.
    return (
        gg_ge_2nd,
        gg_pos_2nd,
        gg_q_2nd,
        gg_mu_2nd,
        gg_Q_2nd,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _batch_rs_quadrupole_setup(ctx, inputs, output):
    """Save forward inputs + half-list flag for the batched l=2 backward."""
    (
        positions,
        charges,
        dipoles,
        quadrupoles,
        cells,
        sigmas,
        alphas,
        batch_idx,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        half_neighbor_list,
    ) = inputs
    ctx.save_for_backward(
        positions,
        charges,
        dipoles,
        quadrupoles,
        cells,
        sigmas,
        alphas,
        batch_idx,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    )
    ctx.half_neighbor_list = half_neighbor_list


def _batch_rs_quadrupole_backward(ctx, grad_per_atom):
    """Per-atom grad → moment-grad + gated cell-grad (forward is now per-atom)."""
    (
        positions,
        charges,
        dipoles,
        quadrupoles,
        cells,
        sigmas,
        alphas,
        batch_idx,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    ) = ctx.saved_tensors
    need = ctx.needs_input_grad
    ge_per_atom = grad_per_atom.contiguous()
    grad_pos, grad_q, grad_mu, grad_Q = (
        torch.ops.nvalchemiops.batch_multipole_real_space_quadrupole_backward(
            ge_per_atom,
            positions,
            charges,
            dipoles,
            quadrupoles,
            cells,
            sigmas,
            alphas,
            batch_idx,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            ctx.half_neighbor_list,
        )
    )
    grad_cells = None
    if need[4]:
        grad_cells = (
            torch.ops.nvalchemiops.batch_multipole_real_space_quadrupole_cell_grad(
                ge_per_atom,
                positions,
                charges,
                dipoles,
                quadrupoles,
                cells,
                sigmas,
                alphas,
                batch_idx,
                idx_j,
                neighbor_ptr,
                unit_shifts,
                ctx.half_neighbor_list,
            )
        )
    # 12 inputs: positions, charges, dipoles, quadrupoles, cells, sigmas, alphas,
    # batch_idx, idx_j, neighbor_ptr, unit_shifts, half_neighbor_list.
    return (
        grad_pos if need[0] else None,
        grad_q if need[1] else None,
        grad_mu if need[2] else None,
        grad_Q if need[3] else None,
        grad_cells,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "nvalchemiops::batch_multipole_real_space_quadrupole_backward",
    _batch_rs_quadrupole_backward_backward,
    setup_context=_batch_rs_quadrupole_backward_setup,
)
torch.library.register_autograd(
    "nvalchemiops::batch_multipole_real_space_quadrupole",
    _batch_rs_quadrupole_backward,
    setup_context=_batch_rs_quadrupole_setup,
)


def _batch_multipole_real_space_quadrupole_energy(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    cells: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    batch_idx: torch.Tensor,
    *,
    half_neighbor_list: bool = False,
) -> torch.Tensor:
    """Per-atom GTO-Ewald real-space LMAX=2 energy, batched (internal).

    Internal dispatch target reached via
    :func:`multipole_real_space_quadrupole_energy` with ``batch_idx=`` set.
    Returns per-atom :math:`(N_\\text{total},)` energies (uniform with the
    :math:`l_{max} \\le 1` paths; the caller ``scatter_add``s for per-system
    totals). Same conventions as
    :func:`multipole_real_space_quadrupole_energy`, with per-system
    ``cells (B, 3, 3)``, ``sigmas (B,)``, ``alphas (B,)``, and per-atom
    ``batch_idx (N_total,)``.

    Parameters
    ----------
    positions : torch.Tensor, shape ``(N_total, 3)``
    charges : torch.Tensor, shape ``(N_total,)``
    dipoles : torch.Tensor, shape ``(N_total, 3)``
        Cartesian.
    quadrupoles : torch.Tensor, shape ``(N_total, 3, 3)``
        Cartesian, symmetric.
    cells : torch.Tensor, shape ``(B, 3, 3)``
        Per-system unit cells.
    idx_j, neighbor_ptr, unit_shifts : torch.Tensor
        Flat CSR neighbor list covering all ``B`` systems.
    sigmas : torch.Tensor, shape ``(B,)``
        Per-system GTO density-basis width :math:`\\sigma`.
    alphas : torch.Tensor, shape ``(B,)``
        Per-system Ewald splitting parameter :math:`\\alpha`.
    batch_idx : torch.Tensor, shape ``(N_total,)``, int32
        Per-atom system index.
    half_neighbor_list : bool, default False
        Set when the CSR neighbor list stores each pair once.

    Returns
    -------
    torch.Tensor, shape ``(N_total,)``, ``float64``
        Per-atom real-space energy, before the :math:`F/(4\\pi)`
        Coulomb scale applied by the caller (which ``scatter_add``s for
        per-system totals).
    """
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"positions must be (N, 3), got {tuple(positions.shape)}")
    if cells.ndim != 3 or cells.shape[-2:] != (3, 3):
        raise ValueError(f"cells must be (B, 3, 3); got {tuple(cells.shape)}")
    return torch.ops.nvalchemiops.batch_multipole_real_space_quadrupole(
        positions,
        charges,
        dipoles,
        quadrupoles,
        cells,
        sigmas,
        alphas,
        batch_idx,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        half_neighbor_list,
    )


__all__ = [
    "multipole_real_space_quadrupole_energy",
]
