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

r"""
PyTorch bindings for the real-space multipole Ewald pipeline.

Every Warp kernel in
:mod:`nvalchemiops.interactions.electrostatics.multipole_ewald_kernels`
gets its own standalone :class:`torch.autograd.Function` wrapper, pairing a
forward kernel with its analytical-backward kernel. The user-facing
:func:`multipole_real_space_energy` composes those wrappers with plain torch
elementwise / reduction ops. The wrappers return per-atom energies :math:`(N,)`,
so the caller owns the atom-global reduction and an upstream distributed
utility can wrap that reduction with ``allreduce``.

Public autograd.Function classes
--------------------------------
* :class:`MultipoleRealSpaceFunction` — charges + dipoles (:math:`l_{max}=1`)
  forward, routing its backward through
  :class:`MultipoleRealSpaceBackwardFunction`.
* :class:`MultipoleRealSpaceBackwardFunction` — first-order backward with
  analytical second-order backward; supports ``create_graph=True`` force-loss
  training.
* :class:`MultipoleRealSpaceMonopoleFunction` — charges-only
  (:math:`l_{max}=0`) forward.
* :class:`MultipoleRealSpaceMonopoleBackwardFunction` — :math:`l_{max}=0`
  first-order backward with analytical second-order Warp kernel; supports
  ``create_graph=True`` force-loss training.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
        MultipoleSCFCache,
    )

from nvalchemiops.interactions.electrostatics.multipole_ewald_cell_grad import (
    batch_multipole_real_space_dipole_csr_cell_grad,
    batch_multipole_real_space_monopole_csr_cell_grad,
    multipole_real_space_dipole_csr_cell_grad,
    multipole_real_space_monopole_csr_cell_grad,
)
from nvalchemiops.interactions.electrostatics.multipole_ewald_kernels import (
    batch_multipole_real_space_dipole_csr_cell_grad_backward,
    batch_multipole_real_space_dipole_csr_energy,
    batch_multipole_real_space_dipole_csr_energy_2nd_backward,
    batch_multipole_real_space_dipole_csr_energy_backward,
    batch_multipole_real_space_dipole_csr_energy_fused,
    batch_multipole_real_space_monopole_csr_cell_grad_backward,
    batch_multipole_real_space_monopole_csr_energy,
    batch_multipole_real_space_monopole_csr_energy_2nd_backward,
    batch_multipole_real_space_monopole_csr_energy_backward,
    batch_multipole_real_space_monopole_csr_energy_fused,
    multipole_real_space_dipole_csr_cell_grad_backward,
    multipole_real_space_dipole_csr_energy,
    multipole_real_space_dipole_csr_energy_2nd_backward,
    multipole_real_space_dipole_csr_energy_backward,
    multipole_real_space_dipole_csr_energy_fused,
    multipole_real_space_monopole_csr_cell_grad_backward,
    multipole_real_space_monopole_csr_energy,
    multipole_real_space_monopole_csr_energy_2nd_backward,
    multipole_real_space_monopole_csr_energy_backward,
    multipole_real_space_monopole_csr_energy_fused,
)
from nvalchemiops.torch._warp_op_helpers import (
    register_warp_op_chain,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    split_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics._util import (
    _energy_cotangents,
)
from nvalchemiops.torch.math import FIELD_CONSTANT
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

# ---------------------------------------------------------------------------
# l_max = 0 first-order backward — standalone autograd.Function
# ---------------------------------------------------------------------------


class MultipoleRealSpaceMonopoleBackwardFunction(torch.autograd.Function):
    r"""Autograd wrapper around the :math:`l_{max}=0` first-order backward kernel.

    Separating the first-order backward into its own
    :class:`torch.autograd.Function` lets the outer
    :class:`MultipoleRealSpaceMonopoleFunction` compose it via ``.apply(...)``
    in its own ``backward``, so that
    ``torch.autograd.grad(energy, positions, create_graph=True)`` produces a
    differentiable ``grad_positions``. The second-order Warp kernel runs in this
    Function's own backward, enabling ``create_graph=True`` force-loss training.

    Forward outputs ``(grad_positions, grad_charges)``. Backward returns one
    gradient per forward input; non-differentiable slots (neighbor-list tensors,
    ``cell``, ``alpha``) are ``None``.
    """

    @staticmethod
    def forward(
        ctx,
        grad_energies: torch.Tensor,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        sigma: torch.Tensor,
        alpha: torch.Tensor,
        idx_j: torch.Tensor,
        neighbor_ptr: torch.Tensor,
        unit_shifts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the first-order backward kernel; save tensors for 2nd-order backward."""
        device = positions.device
        wp_device = wp.device_from_torch(device)
        input_dtype = positions.dtype

        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        num_atoms = positions.shape[0]

        grad_positions = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
        grad_charges = torch.zeros(charges.shape[0], dtype=input_dtype, device=device)
        multipole_real_space_monopole_csr_energy_backward(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
            wp.from_torch(grad_positions, dtype=wp_vec),
            wp.from_torch(grad_charges, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp_device),
        )

        ctx.save_for_backward(
            grad_energies,
            positions,
            charges,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        )
        return grad_positions, grad_charges

    @staticmethod
    def backward(
        ctx, gg_positions: torch.Tensor, gg_charges: torch.Tensor
    ):  # pragma: no cover
        """Second-order backward: produces (gg_ge_2nd, gg_pos_2nd, gg_chg_2nd)."""
        (
            grad_energies,
            positions,
            charges,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        ) = ctx.saved_tensors
        device = positions.device
        wp_device = wp.device_from_torch(device)
        input_dtype = positions.dtype

        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        num_atoms = positions.shape[0]

        gg_grad_energies_2nd = torch.zeros_like(grad_energies)
        gg_positions_2nd = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
        gg_charges_2nd = torch.zeros_like(charges)
        multipole_real_space_monopole_csr_energy_2nd_backward(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
            wp.from_torch(gg_positions.contiguous(), dtype=wp_vec),
            wp.from_torch(gg_charges.contiguous(), dtype=wp_scalar),
            wp.from_torch(gg_grad_energies_2nd, dtype=wp.float64),
            wp.from_torch(gg_positions_2nd, dtype=wp_vec),
            wp.from_torch(gg_charges_2nd, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp_device),
        )

        return (
            gg_grad_energies_2nd,
            gg_positions_2nd,
            gg_charges_2nd,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# l_max = 0 — fused energy + analytical gradients
# ---------------------------------------------------------------------------


class MultipoleRealSpaceMonopoleFusedScalarFunction(torch.autograd.Function):
    r"""Fused-kernel autograd.Function for the :math:`l_{max}=0` real-space path.

    Single-launch alternative to :class:`MultipoleRealSpaceMonopoleFunction` +
    :class:`MultipoleRealSpaceMonopoleBackwardFunction` for callers that want a
    scalar :math:`\text{float64}` total energy. Backward broadcasts the
    precomputed gradients (positions, charges, and — when
    ``cell.requires_grad`` — cell); the cell-grad kernel runs in forward.

    Notes
    -----
    * Double-backward is not supported.
    * Only ``positions``, ``charges``, and ``cell`` get analytical gradients;
      ``sigma``, ``alpha``, and neighbor-list tensors return ``None``.
    """

    @staticmethod
    def forward(
        ctx,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        sigma: torch.Tensor,
        alpha: torch.Tensor,
        idx_j: torch.Tensor,
        neighbor_ptr: torch.Tensor,
        unit_shifts: torch.Tensor,
        half_neighbor_list: bool = False,
    ) -> torch.Tensor:
        """Run the lmax=0 fused Warp kernel; stash analytical gradients in ctx."""
        device = positions.device
        wp_device = wp.device_from_torch(device)
        input_dtype = positions.dtype

        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        with_pos = bool(positions.requires_grad)
        with_q = bool(charges.requires_grad)
        with_cell = bool(cell.requires_grad)
        num_atoms = positions.shape[0]

        energies = torch.zeros(num_atoms, dtype=torch.float64, device=device)
        grad_positions = (
            torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
            if with_pos
            else torch.zeros((1, 3), dtype=input_dtype, device=device)
        )
        grad_charges = (
            torch.zeros(num_atoms, dtype=input_dtype, device=device)
            if with_q
            else torch.zeros(1, dtype=input_dtype, device=device)
        )

        multipole_real_space_monopole_csr_energy_fused(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(energies, dtype=wp.float64),
            wp.from_torch(grad_positions, dtype=wp_vec),
            wp.from_torch(grad_charges, dtype=wp_scalar),
            with_pos_grad=with_pos,
            with_charge_grad=with_q,
            wp_dtype=wp_scalar,
            device=str(wp_device),
        )

        energies_total = energies.sum()

        if with_cell:
            from nvalchemiops.interactions.electrostatics.multipole_ewald_cell_grad import (
                multipole_real_space_monopole_csr_cell_grad,
            )

            grad_cell_buf = torch.zeros_like(cell.detach()).contiguous()
            multipole_real_space_monopole_csr_cell_grad(
                wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
                wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
                wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
                wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
                wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
                wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
                wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
                wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
                wp.from_torch(grad_cell_buf, dtype=wp_mat),
                device=str(wp_device),
                half_neighbor_list=half_neighbor_list,
            )
            grad_cell_to_save = grad_cell_buf
        else:
            grad_cell_to_save = None

        # Save precomputed grads for the fast path and inputs for the
        # double-backward path. Plain forces broadcast the precomputed grads;
        # under ``create_graph`` pos/charge grads route through the on-tape
        # backward Function (2nd-order kernel) so force-loss training works.
        ctx.save_for_backward(
            grad_positions if with_pos else None,
            grad_charges if with_q else None,
            grad_cell_to_save,
            positions,
            charges,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        )
        ctx.with_cell = with_cell
        return energies_total

    @staticmethod
    def backward(ctx, grad_E: torch.Tensor):  # pragma: no cover
        """Broadcast precomputed grads for plain forces; route through the
        on-tape backward Function under ``create_graph``. Cell-grad is always a
        scalar broadcast."""
        (
            pre_grad_positions,
            pre_grad_charges,
            grad_cell,
            positions,
            charges,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        ) = ctx.saved_tensors
        need_pq = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        if need_pq and torch.is_grad_enabled():
            grad_energies = grad_E * torch.ones(
                positions.shape[0], dtype=torch.float64, device=positions.device
            )
            grad_positions, grad_charges = (
                MultipoleRealSpaceMonopoleBackwardFunction.apply(
                    grad_energies,
                    positions,
                    charges,
                    cell,
                    sigma,
                    alpha,
                    idx_j,
                    neighbor_ptr,
                    unit_shifts,
                )
            )
        else:
            grad_positions = (
                grad_E * pre_grad_positions if pre_grad_positions is not None else None
            )
            grad_charges = (
                grad_E * pre_grad_charges if pre_grad_charges is not None else None
            )
        return (
            grad_positions,
            grad_charges,
            grad_E * grad_cell if ctx.with_cell else None,
            None,  # sigma
            None,  # alpha
            None,  # idx_j
            None,  # neighbor_ptr
            None,  # unit_shifts
            None,  # half_neighbor_list
        )


# ---------------------------------------------------------------------------
# l_max = 0 (charges-only) torch.autograd.Function wrapper
# ---------------------------------------------------------------------------


class MultipoleRealSpaceMonopoleFunction(torch.autograd.Function):
    r"""Autograd-registered real-space multipole Ewald (charges-only).

    Wraps :func:`multipole_real_space_monopole_csr_energy` in its ``forward``
    and its analytical backward kernel
    :func:`multipole_real_space_monopole_csr_energy_backward` in its
    ``backward``. The output is per-atom energies :math:`(N,)` in
    :math:`\text{float64}`; the caller composes ``.sum()`` / ``scatter_add`` to
    get total energies.

    Backward produces analytical gradients w.r.t. ``(positions, charges)``;
    ``cell`` and ``alpha`` return ``None``.

    Usage
    -----

    .. code-block:: python

        energies = MultipoleRealSpaceMonopoleFunction.apply(
            positions, charges, cell, alpha,
            idx_j, neighbor_ptr, unit_shifts,
        )
    """

    @staticmethod
    def forward(
        ctx,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        sigma: torch.Tensor,
        alpha: torch.Tensor,
        idx_j: torch.Tensor,
        neighbor_ptr: torch.Tensor,
        unit_shifts: torch.Tensor,
    ) -> torch.Tensor:
        """Run the forward Warp kernel and save tensors for analytical backward."""
        device = positions.device
        wp_device = wp.device_from_torch(device)
        num_atoms = positions.shape[0]
        input_dtype = positions.dtype

        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        energies = torch.zeros(num_atoms, dtype=torch.float64, device=device)
        multipole_real_space_monopole_csr_energy(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
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

        ctx.save_for_backward(
            positions, charges, cell, sigma, alpha, idx_j, neighbor_ptr, unit_shifts
        )
        return energies

    @staticmethod
    def backward(ctx, grad_energies: torch.Tensor):  # pragma: no cover
        """Analytical backward via :class:`MultipoleRealSpaceMonopoleBackwardFunction`.

        Routing through the sub-Function is what enables double-backward:
        the sub-Function's own ``backward`` invokes the second-order Warp
        kernel, so ``torch.autograd.grad(..., create_graph=True)`` keeps
        ``grad_positions`` / ``grad_charges`` on the tape.
        """
        (
            positions,
            charges,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        ) = ctx.saved_tensors
        grad_positions, grad_charges = MultipoleRealSpaceMonopoleBackwardFunction.apply(
            grad_energies,
            positions,
            charges,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        )
        return grad_positions, grad_charges, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# General real-space multipole first-order backward — standalone autograd.Function
# ---------------------------------------------------------------------------


class MultipoleRealSpaceBackwardFunction(torch.autograd.Function):
    r"""Autograd wrapper around the :math:`l_{max}=1` first-order backward kernel.

    Mirrors :class:`MultipoleRealSpaceMonopoleBackwardFunction` for the
    charges + dipoles case. Separating into its own
    :class:`torch.autograd.Function` lets the outer
    :class:`MultipoleRealSpaceFunction` compose it via ``.apply(...)`` so that
    ``torch.autograd.grad(..., create_graph=True)`` produces differentiable
    forces via the analytical second-order kernel
    :func:`multipole_real_space_dipole_csr_energy_2nd_backward`.

    Forward outputs ``(grad_positions, grad_charges, grad_dipoles)``.
    """

    @staticmethod
    def forward(
        ctx,
        grad_energies: torch.Tensor,
        positions: torch.Tensor,
        charges: torch.Tensor,
        dipoles: torch.Tensor,
        cell: torch.Tensor,
        sigma: torch.Tensor,
        alpha: torch.Tensor,
        idx_j: torch.Tensor,
        neighbor_ptr: torch.Tensor,
        unit_shifts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the l_max=1 first-order backward Warp kernel."""
        device = positions.device
        wp_device = wp.device_from_torch(device)
        input_dtype = positions.dtype

        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        num_atoms = positions.shape[0]

        grad_positions = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
        grad_charges = torch.zeros(charges.shape[0], dtype=input_dtype, device=device)
        grad_dipoles = torch.zeros(
            (dipoles.shape[0], 3), dtype=input_dtype, device=device
        )
        multipole_real_space_dipole_csr_energy_backward(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
            wp.from_torch(grad_positions, dtype=wp_vec),
            wp.from_torch(grad_charges, dtype=wp_scalar),
            wp.from_torch(grad_dipoles, dtype=wp_vec),
            wp_dtype=wp_scalar,
            device=str(wp_device),
        )

        ctx.save_for_backward(
            grad_energies,
            positions,
            charges,
            dipoles,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        )
        return grad_positions, grad_charges, grad_dipoles

    @staticmethod
    def backward(  # pragma: no cover
        ctx,
        gg_positions: torch.Tensor,
        gg_charges: torch.Tensor,
        gg_dipoles: torch.Tensor,
    ):
        """Second-order backward via :func:`multipole_real_space_dipole_csr_energy_2nd_backward`."""
        (
            grad_energies,
            positions,
            charges,
            dipoles,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        ) = ctx.saved_tensors
        device = positions.device
        wp_device = wp.device_from_torch(device)
        input_dtype = positions.dtype

        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        num_atoms = positions.shape[0]

        gg_grad_energies_2nd = torch.zeros_like(grad_energies)
        gg_positions_2nd = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
        gg_charges_2nd = torch.zeros_like(charges)
        gg_dipoles_2nd = torch.zeros(
            (dipoles.shape[0], 3), dtype=input_dtype, device=device
        )
        multipole_real_space_dipole_csr_energy_2nd_backward(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
            wp.from_torch(gg_positions.contiguous(), dtype=wp_vec),
            wp.from_torch(gg_charges.contiguous(), dtype=wp_scalar),
            wp.from_torch(gg_dipoles.contiguous(), dtype=wp_vec),
            wp.from_torch(gg_grad_energies_2nd, dtype=wp.float64),
            wp.from_torch(gg_positions_2nd, dtype=wp_vec),
            wp.from_torch(gg_charges_2nd, dtype=wp_scalar),
            wp.from_torch(gg_dipoles_2nd, dtype=wp_vec),
            wp_dtype=wp_scalar,
            device=str(wp_device),
        )

        return (
            gg_grad_energies_2nd,
            gg_positions_2nd,
            gg_charges_2nd,
            gg_dipoles_2nd,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# l_max = 1 — fused energy + analytical gradients
# ---------------------------------------------------------------------------


class MultipoleRealSpaceDipoleFusedScalarFunction(torch.autograd.Function):
    r"""Fused-kernel autograd.Function for the :math:`l_{max}=1` real-space path.

    Single-launch alternative to :class:`MultipoleRealSpaceFunction` +
    :class:`MultipoleRealSpaceBackwardFunction` for callers that want a scalar
    total energy. When ``dipoles.requires_grad`` is set, the fused kernel emits
    analytical :math:`\partial E/\partial\boldsymbol{\mu}_i` per atom alongside
    the position and charge gradients.

    Returns
    -------
    torch.Tensor
        Scalar :math:`\text{float64}` total energy. Per-atom callers stay on the
        layered :class:`MultipoleRealSpaceFunction`.

    Notes
    -----
    * Double-backward not supported; use the layered Function pair if
      :math:`\partial^2 E` is needed.
    * Backward broadcasts the precomputed gradients in ``ctx``.
    """

    @staticmethod
    def forward(
        ctx,
        positions: torch.Tensor,
        charges: torch.Tensor,
        dipoles: torch.Tensor,
        cell: torch.Tensor,
        sigma: torch.Tensor,
        alpha: torch.Tensor,
        idx_j: torch.Tensor,
        neighbor_ptr: torch.Tensor,
        unit_shifts: torch.Tensor,
        half_neighbor_list: bool = False,
    ) -> torch.Tensor:
        """Run the lmax=1 fused Warp kernel; stash analytical gradients in ctx."""
        device = positions.device
        wp_device = wp.device_from_torch(device)
        input_dtype = positions.dtype

        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        with_pos = bool(positions.requires_grad)
        with_q = bool(charges.requires_grad)
        with_mu = bool(dipoles.requires_grad)
        with_cell = bool(cell.requires_grad)
        num_atoms = positions.shape[0]

        energies = torch.zeros(num_atoms, dtype=torch.float64, device=device)
        grad_positions = (
            torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
            if with_pos
            else torch.zeros((1, 3), dtype=input_dtype, device=device)
        )
        grad_charges = (
            torch.zeros(num_atoms, dtype=input_dtype, device=device)
            if with_q
            else torch.zeros(1, dtype=input_dtype, device=device)
        )
        grad_dipoles = (
            torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
            if with_mu
            else torch.zeros((1, 3), dtype=input_dtype, device=device)
        )

        multipole_real_space_dipole_csr_energy_fused(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(energies, dtype=wp.float64),
            wp.from_torch(grad_positions, dtype=wp_vec),
            wp.from_torch(grad_charges, dtype=wp_scalar),
            wp.from_torch(grad_dipoles, dtype=wp_vec),
            with_pos_grad=with_pos,
            with_charge_grad=with_q,
            with_dipole_grad=with_mu,
            wp_dtype=wp_scalar,
            device=str(wp_device),
        )

        energies_total = energies.sum()

        if with_cell:
            from nvalchemiops.interactions.electrostatics.multipole_ewald_cell_grad import (
                multipole_real_space_dipole_csr_cell_grad,
            )

            grad_cell_buf = torch.zeros_like(cell.detach()).contiguous()
            multipole_real_space_dipole_csr_cell_grad(
                wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
                wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
                wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
                wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
                wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
                wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
                wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
                wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
                wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
                wp.from_torch(grad_cell_buf, dtype=wp_mat),
                device=str(wp_device),
                half_neighbor_list=half_neighbor_list,
            )
            grad_cell_to_save = grad_cell_buf
        else:
            grad_cell_to_save = None

        # Save precomputed grads for the fast path and inputs for the
        # double-backward path. Plain forces broadcast the precomputed grads;
        # under ``create_graph`` pos/charge/dipole grads route through the
        # on-tape backward Function (2nd-order kernel) for force-loss training.
        ctx.save_for_backward(
            grad_positions if with_pos else None,
            grad_charges if with_q else None,
            grad_dipoles if with_mu else None,
            grad_cell_to_save,
            positions,
            charges,
            dipoles,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        )
        ctx.with_cell = with_cell
        return energies_total

    @staticmethod
    def backward(ctx, grad_E: torch.Tensor):  # pragma: no cover
        """Broadcast precomputed grads for plain forces; route through the
        on-tape backward Function under ``create_graph``."""
        (
            pre_grad_positions,
            pre_grad_charges,
            pre_grad_dipoles,
            grad_cell,
            positions,
            charges,
            dipoles,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        ) = ctx.saved_tensors
        need_pqm = (
            ctx.needs_input_grad[0]
            or ctx.needs_input_grad[1]
            or ctx.needs_input_grad[2]
        )
        if need_pqm and torch.is_grad_enabled():
            grad_energies = grad_E * torch.ones(
                positions.shape[0], dtype=torch.float64, device=positions.device
            )
            grad_positions, grad_charges, grad_dipoles = (
                MultipoleRealSpaceBackwardFunction.apply(
                    grad_energies,
                    positions,
                    charges,
                    dipoles,
                    cell,
                    sigma,
                    alpha,
                    idx_j,
                    neighbor_ptr,
                    unit_shifts,
                )
            )
        else:
            grad_positions = (
                grad_E * pre_grad_positions if pre_grad_positions is not None else None
            )
            grad_charges = (
                grad_E * pre_grad_charges if pre_grad_charges is not None else None
            )
            grad_dipoles = (
                grad_E * pre_grad_dipoles if pre_grad_dipoles is not None else None
            )
        return (
            grad_positions,
            grad_charges,
            grad_dipoles,
            grad_E * grad_cell if ctx.with_cell else None,
            None,  # sigma
            None,  # alpha
            None,  # idx_j
            None,  # neighbor_ptr
            None,  # unit_shifts
            None,  # half_neighbor_list
        )


# ---------------------------------------------------------------------------
# General real-space multipole forward (l_max = 1: charges + dipoles)
# ---------------------------------------------------------------------------


class MultipoleRealSpaceFunction(torch.autograd.Function):
    r"""Autograd-registered real-space multipole Ewald (charges + dipoles).

    Wraps :func:`multipole_real_space_dipole_csr_energy` forward and routes its
    backward through :class:`MultipoleRealSpaceBackwardFunction` (enabling
    ``create_graph=True`` force-loss training).

    Same shape as :class:`MultipoleRealSpaceMonopoleFunction` with an extra
    ``dipoles`` input and an extra ``grad_dipoles`` backward output. Returns
    per-atom energies :math:`(N,)` in :math:`\text{float64}`; the caller owns
    the atom-global reduction.
    """

    @staticmethod
    def forward(
        ctx,
        positions: torch.Tensor,
        charges: torch.Tensor,
        dipoles: torch.Tensor,
        cell: torch.Tensor,
        sigma: torch.Tensor,
        alpha: torch.Tensor,
        idx_j: torch.Tensor,
        neighbor_ptr: torch.Tensor,
        unit_shifts: torch.Tensor,
    ) -> torch.Tensor:
        """Run the l_max=1 forward Warp kernel; save tensors for the backward Function."""
        device = positions.device
        wp_device = wp.device_from_torch(device)
        num_atoms = positions.shape[0]
        input_dtype = positions.dtype

        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        energies = torch.zeros(num_atoms, dtype=torch.float64, device=device)
        multipole_real_space_dipole_csr_energy(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
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

        ctx.save_for_backward(
            positions,
            charges,
            dipoles,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        )
        return energies

    @staticmethod
    def backward(ctx, grad_energies: torch.Tensor):  # pragma: no cover
        """Route through :class:`MultipoleRealSpaceBackwardFunction`."""
        (
            positions,
            charges,
            dipoles,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        ) = ctx.saved_tensors
        grad_positions, grad_charges, grad_dipoles = (
            MultipoleRealSpaceBackwardFunction.apply(
                grad_energies,
                positions,
                charges,
                dipoles,
                cell,
                sigma,
                alpha,
                idx_j,
                neighbor_ptr,
                unit_shifts,
            )
        )
        return (
            grad_positions,
            grad_charges,
            grad_dipoles,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# l_max = 1 single-system real-space — torch.library.custom_op chain
# ---------------------------------------------------------------------------


def _real_space_dipole_forward(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
) -> torch.Tensor:
    """Run the l_max=1 real-space forward Warp kernel (charges + dipoles)."""
    wp_device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(positions.dtype)
    wp_vec = get_wp_vec_dtype(positions.dtype)
    wp_mat = get_wp_mat_dtype(positions.dtype)
    energies = torch.zeros(
        positions.shape[0], dtype=torch.float64, device=positions.device
    )
    multipole_real_space_dipole_csr_energy(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
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


def _real_space_dipole_forward_fake(  # pragma: no cover
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: per-atom energies ``(N,)`` in float64."""
    return positions.new_empty((positions.shape[0],), dtype=torch.float64)


def _real_space_dipole_backward(  # pragma: no cover
    grad_energies: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the l_max=1 first-order backward Warp kernel."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    grad_positions = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    grad_charges = torch.zeros(charges.shape[0], dtype=input_dtype, device=device)
    grad_dipoles = torch.zeros((dipoles.shape[0], 3), dtype=input_dtype, device=device)
    multipole_real_space_dipole_csr_energy_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_positions, dtype=wp_vec),
        wp.from_torch(grad_charges, dtype=wp_scalar),
        wp.from_torch(grad_dipoles, dtype=wp_vec),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_positions, grad_charges, grad_dipoles


def _real_space_dipole_double_backward(  # pragma: no cover
    gg_positions: torch.Tensor,
    gg_charges: torch.Tensor,
    gg_dipoles: torch.Tensor,
    grad_energies: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Second-order backward Warp kernel (``create_graph`` force-loss training)."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    gg_grad_energies_2nd = torch.zeros_like(grad_energies)
    gg_positions_2nd = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    gg_charges_2nd = torch.zeros_like(charges)
    gg_dipoles_2nd = torch.zeros(
        (dipoles.shape[0], 3), dtype=input_dtype, device=device
    )
    multipole_real_space_dipole_csr_energy_2nd_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp_vec),
        wp.from_torch(gg_charges.contiguous(), dtype=wp_scalar),
        wp.from_torch(gg_dipoles.contiguous(), dtype=wp_vec),
        wp.from_torch(gg_grad_energies_2nd, dtype=wp.float64),
        wp.from_torch(gg_positions_2nd, dtype=wp_vec),
        wp.from_torch(gg_charges_2nd, dtype=wp_scalar),
        wp.from_torch(gg_dipoles_2nd, dtype=wp_vec),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return gg_grad_energies_2nd, gg_positions_2nd, gg_charges_2nd, gg_dipoles_2nd


register_warp_op_chain(
    name="nvalchemiops::multipole_real_space_dipole",
    forward=_real_space_dipole_forward,
    backward=_real_space_dipole_backward,
    double_backward=_real_space_dipole_double_backward,
    diff_input_positions=(0, 1, 2),
    n_forward_inputs=9,
    second_order_diff_positions=(0, 1, 2, 3),
    n_backward_inputs=10,
    forward_fake=_real_space_dipole_forward_fake,
)


# ---------------------------------------------------------------------------
# l_max = 0 single-system real-space — torch.library.custom_op chain
# ---------------------------------------------------------------------------
# Charges-only analog of the l_max=1 chain above (no dipole channel). Same
# compile-friendliness rationale; routed from ``multipole_real_space_energy``.


def _real_space_monopole_forward(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
) -> torch.Tensor:
    """Run the l_max=0 real-space forward Warp kernel (charges only)."""
    wp_device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(positions.dtype)
    wp_vec = get_wp_vec_dtype(positions.dtype)
    wp_mat = get_wp_mat_dtype(positions.dtype)
    energies = torch.zeros(
        positions.shape[0], dtype=torch.float64, device=positions.device
    )
    multipole_real_space_monopole_csr_energy(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
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


def _real_space_monopole_forward_fake(  # pragma: no cover
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: per-atom energies ``(N,)`` in float64."""
    return positions.new_empty((positions.shape[0],), dtype=torch.float64)


def _real_space_monopole_backward(  # pragma: no cover
    grad_energies: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the l_max=0 first-order backward Warp kernel."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    grad_positions = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    grad_charges = torch.zeros(charges.shape[0], dtype=input_dtype, device=device)
    multipole_real_space_monopole_csr_energy_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_positions, dtype=wp_vec),
        wp.from_torch(grad_charges, dtype=wp_scalar),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_positions, grad_charges


def _real_space_monopole_double_backward(  # pragma: no cover
    gg_positions: torch.Tensor,
    gg_charges: torch.Tensor,
    grad_energies: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Second-order backward Warp kernel (``create_graph`` force-loss training)."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    gg_grad_energies_2nd = torch.zeros_like(grad_energies)
    gg_positions_2nd = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    gg_charges_2nd = torch.zeros_like(charges)
    multipole_real_space_monopole_csr_energy_2nd_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp_vec),
        wp.from_torch(gg_charges.contiguous(), dtype=wp_scalar),
        wp.from_torch(gg_grad_energies_2nd, dtype=wp.float64),
        wp.from_torch(gg_positions_2nd, dtype=wp_vec),
        wp.from_torch(gg_charges_2nd, dtype=wp_scalar),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return gg_grad_energies_2nd, gg_positions_2nd, gg_charges_2nd


register_warp_op_chain(
    name="nvalchemiops::multipole_real_space_monopole",
    forward=_real_space_monopole_forward,
    backward=_real_space_monopole_backward,
    double_backward=_real_space_monopole_double_backward,
    diff_input_positions=(0, 1),
    n_forward_inputs=8,
    second_order_diff_positions=(0, 1, 2),
    n_backward_inputs=9,
    forward_fake=_real_space_monopole_forward_fake,
)


# ---------------------------------------------------------------------------
# Fused-scalar real-space (l=0/1 single-system) — torch.library.custom_op chains
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "nvalchemiops::multipole_real_space_dipole_cell_grad", mutates_args=()
)
def _rs_dipole_cell_grad_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    half_neighbor_list: bool,
) -> torch.Tensor:
    """Unweighted total ``dE/dcell`` for the l=1 real-space energy (stress)."""
    wp_device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(positions.dtype)
    wp_vec = get_wp_vec_dtype(positions.dtype)
    wp_mat = get_wp_mat_dtype(positions.dtype)
    grad_cell = torch.zeros_like(cell.detach()).contiguous()
    multipole_real_space_dipole_csr_cell_grad(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(grad_cell, dtype=wp_mat),
        device=str(wp_device),
        half_neighbor_list=half_neighbor_list,
    )
    return grad_cell


@_rs_dipole_cell_grad_op.register_fake
def _(positions, charges, dipoles, cell, sigma, alpha, idx_j, nptr, shifts, half):
    return torch.empty_like(cell)


def _rs_dipole_cell_grad_setup(ctx, inputs, output):  # pragma: no cover
    """Save inputs for the l=1 cell-grad double-backward (stress-loss)."""
    (
        positions,
        charges,
        dipoles,
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
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    )
    ctx.half_neighbor_list = half_neighbor_list


def _rs_dipole_cell_grad_backward(ctx, g_cell):  # pragma: no cover
    """∂/∂{positions, charges, dipoles, cell} of ⟨g_cell, dE/dcell⟩ (l=1)."""
    (
        positions,
        charges,
        dipoles,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    ) = ctx.saved_tensors
    need = ctx.needs_input_grad
    if not (need[0] or need[1] or need[2] or need[3]):
        return (None,) * 10
    device = positions.device
    wp_device = wp.device_from_torch(device)
    dtype = positions.dtype
    wp_scalar = get_wp_dtype(dtype)
    wp_vec = get_wp_vec_dtype(dtype)
    wp_mat = get_wp_mat_dtype(dtype)
    n = positions.shape[0]
    scale_val = 1.0 if ctx.half_neighbor_list else 0.5
    scale = torch.tensor([scale_val], dtype=torch.float64, device=device)
    grad_positions = torch.zeros((n, 3), dtype=dtype, device=device)
    grad_charges = torch.zeros(n, dtype=dtype, device=device)
    grad_dipoles = torch.zeros((n, 3), dtype=dtype, device=device)
    grad_cell_cc = torch.zeros((1, 3, 3), dtype=dtype, device=device)
    multipole_real_space_dipole_csr_cell_grad_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(scale, dtype=wp.float64),
        wp.from_torch(g_cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(grad_positions, dtype=wp_vec),
        wp.from_torch(grad_charges, dtype=wp_scalar),
        wp.from_torch(grad_dipoles, dtype=wp_vec),
        wp.from_torch(grad_cell_cc, dtype=wp_mat),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return (
        grad_positions if need[0] else None,
        grad_charges if need[1] else None,
        grad_dipoles if need[2] else None,
        grad_cell_cc.reshape(cell.shape) if need[3] else None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "nvalchemiops::multipole_real_space_dipole_cell_grad",
    _rs_dipole_cell_grad_backward,
    setup_context=_rs_dipole_cell_grad_setup,
)


@torch.library.custom_op(
    "nvalchemiops::multipole_real_space_dipole_fused", mutates_args=()
)
def _rs_dipole_fused_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    half_neighbor_list: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One fused launch: scalar total energy + per-atom moment gradients."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    energies = torch.zeros(num_atoms, dtype=torch.float64, device=device)
    grad_positions = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    grad_charges = torch.zeros(num_atoms, dtype=input_dtype, device=device)
    grad_dipoles = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    multipole_real_space_dipole_csr_energy_fused(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(energies, dtype=wp.float64),
        wp.from_torch(grad_positions, dtype=wp_vec),
        wp.from_torch(grad_charges, dtype=wp_scalar),
        wp.from_torch(grad_dipoles, dtype=wp_vec),
        with_pos_grad=True,
        with_charge_grad=True,
        with_dipole_grad=True,
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return energies.sum(), grad_positions, grad_charges, grad_dipoles


@_rs_dipole_fused_op.register_fake
def _(positions, charges, dipoles, cell, sigma, alpha, idx_j, nptr, shifts, half):
    return (
        positions.new_empty((), dtype=torch.float64),
        torch.empty_like(positions),
        torch.empty_like(charges),
        torch.empty_like(dipoles),
    )


def _rs_dipole_fused_setup(ctx, inputs, output):  # pragma: no cover
    """Save inputs + the precomputed moment grads (returned as op outputs)."""
    (
        positions,
        charges,
        dipoles,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        half_neighbor_list,
    ) = inputs
    _energy, grad_positions, grad_charges, grad_dipoles = output
    ctx.save_for_backward(
        positions,
        charges,
        dipoles,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        grad_positions,
        grad_charges,
        grad_dipoles,
    )
    ctx.half_neighbor_list = half_neighbor_list


def _rs_dipole_fused_backward(ctx, grad_E, _gg_pos, _gg_q, _gg_mu):  # pragma: no cover
    """Broadcast precomputed grads (plain forces); on-tape op under create_graph."""
    (
        positions,
        charges,
        dipoles,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        pre_gp,
        pre_gq,
        pre_gmu,
    ) = ctx.saved_tensors
    need = ctx.needs_input_grad
    if (need[0] or need[1] or need[2]) and torch.is_grad_enabled():
        grad_energies = grad_E * torch.ones(
            positions.shape[0], dtype=torch.float64, device=positions.device
        )
        gp, gq, gmu = torch.ops.nvalchemiops.multipole_real_space_dipole_backward(
            grad_energies,
            positions,
            charges,
            dipoles,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        )
    else:
        gp, gq, gmu = grad_E * pre_gp, grad_E * pre_gq, grad_E * pre_gmu
    grad_cell = None
    if need[3]:
        grad_cell = (
            grad_E
            * torch.ops.nvalchemiops.multipole_real_space_dipole_cell_grad(
                positions,
                charges,
                dipoles,
                cell,
                sigma,
                alpha,
                idx_j,
                neighbor_ptr,
                unit_shifts,
                ctx.half_neighbor_list,
            )
        )
    return (
        gp if need[0] else None,
        gq if need[1] else None,
        gmu if need[2] else None,
        grad_cell,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "nvalchemiops::multipole_real_space_dipole_fused",
    _rs_dipole_fused_backward,
    setup_context=_rs_dipole_fused_setup,
)


class _AttachRealSpaceCellGradLmax1(torch.autograd.Function):
    """Attach the per-system-reduced cell gradient to a per-atom l<=1
    real-space energy.

    The per-atom :func:`multipole_real_space_energy` op (l=0/1) does not
    differentiate ``cell`` (it is detached in the kernel); the collective stress
    comes from the separate, twice-differentiable
    ``multipole_real_space_{monopole,dipole}_cell_grad`` op, which computes the
    UNWEIGHTED total ``dE/dcell``. This identity-forward Function injects that
    cell gradient into the per-atom energy's graph so that
    ``grad(E.sum(), cell)`` reproduces the collective stress (PR #96
    energy-autograd contract: the energy cotangent is per-system uniform under
    sum-reduction). ``grad_cell = grad_system * total_cell_grad`` where
    ``grad_system`` is the per-system-reduced energy cotangent
    (:func:`_energy_cotangents`). The cell-grad op is re-evaluated inside
    ``backward`` so its registered double-backward carries stress-loss
    (``d^2E/dcell d theta``). l=2 already routes cell-grad through its own
    per-atom-weighted op, so this covers l=0/1 only.

    Forward inputs: ``per_atom`` (the energy, slot 0, carries the
    position/charge/dipole graph) and ``cell`` (slot 1, the only slot this
    Function contributes a gradient to). The remaining tensors are saved for the
    backward cell-grad re-evaluation; this Function returns ``None`` for them at
    first order (their first-order grads flow through ``per_atom``).
    """

    @staticmethod
    def forward(
        per_atom,
        cell,
        positions,
        charges,
        dipoles,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
        num_systems,
        l_max,
        half_neighbor_list,
    ):
        """Identity over ``per_atom``."""
        return per_atom

    @staticmethod
    def setup_context(ctx, inputs, output):  # pragma: no cover
        """Save the inputs the backward re-evaluates the cell-grad op from."""
        (
            _per_atom,
            cell,
            positions,
            charges,
            dipoles,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
            num_systems,
            l_max,
            half_neighbor_list,
        ) = inputs
        ctx.save_for_backward(
            cell,
            positions,
            charges,
            dipoles,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx if batch_idx is not None else cell.new_zeros(0),
        )
        ctx.has_batch = batch_idx is not None
        ctx.num_systems = num_systems
        ctx.num_atoms = output.shape[0]
        ctx.l_max = l_max
        ctx.half_neighbor_list = half_neighbor_list

    @staticmethod
    def backward(ctx, grad_energy):  # pragma: no cover
        """Pass ``grad_energy`` through to ``per_atom``; inject ``grad_cell``."""
        (
            cell,
            positions,
            charges,
            dipoles,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx_saved,
        ) = ctx.saved_tensors
        grad_cell = None
        if ctx.needs_input_grad[1]:
            batch_idx = batch_idx_saved if ctx.has_batch else None
            grad_system, _ = _energy_cotangents(
                grad_energy, batch_idx, ctx.num_atoms, ctx.num_systems
            )
            if ctx.has_batch:
                if ctx.l_max == 1:
                    cg = torch.ops.nvalchemiops.batch_multipole_real_space_dipole_cell_grad(
                        positions,
                        charges,
                        dipoles,
                        cell,
                        sigma,
                        alpha,
                        idx_j,
                        neighbor_ptr,
                        unit_shifts,
                        batch_idx,
                        ctx.half_neighbor_list,
                    )
                else:
                    cg = torch.ops.nvalchemiops.batch_multipole_real_space_monopole_cell_grad(
                        positions,
                        charges,
                        cell,
                        sigma,
                        alpha,
                        idx_j,
                        neighbor_ptr,
                        unit_shifts,
                        batch_idx,
                        ctx.half_neighbor_list,
                    )
                grad_cell = grad_system.view(-1, 1, 1) * cg
            else:
                if ctx.l_max == 1:
                    cg = torch.ops.nvalchemiops.multipole_real_space_dipole_cell_grad(
                        positions,
                        charges,
                        dipoles,
                        cell,
                        sigma,
                        alpha,
                        idx_j,
                        neighbor_ptr,
                        unit_shifts,
                        ctx.half_neighbor_list,
                    )
                else:
                    cg = torch.ops.nvalchemiops.multipole_real_space_monopole_cell_grad(
                        positions,
                        charges,
                        cell,
                        sigma,
                        alpha,
                        idx_j,
                        neighbor_ptr,
                        unit_shifts,
                        ctx.half_neighbor_list,
                    )
                grad_cell = grad_system.reshape(()) * cg
        return (
            grad_energy,
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
            None,
            None,
            None,
        )


def multipole_real_space_energy_with_stress(
    positions: torch.Tensor,
    multipole_moments: torch.Tensor,
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
    r"""Per-atom real-space energy that also carries the cell gradient (stress).

    Same per-atom :math:`(N,)` / :math:`(N_\text{total},)` value and
    position/charge/dipole/quadrupole gradients as
    :func:`multipole_real_space_energy`, but additionally differentiable w.r.t.
    ``cell``: ``grad(E.sum(), cell)`` equals the collective stress (and
    ``d^2E/dcell d theta`` flows for stress-loss). This is the per-atom-energy
    composite's real-space half under the PR #96 energy-autograd contract,
    replacing the per-system-scalar ``*FusedScalarFunction`` path (which carried
    cell-grad only on a scalar). l=2 already routes cell-grad per-atom; for
    l<=1 the cell gradient is injected via :class:`_AttachRealSpaceCellGradLmax1`
    (the energy cotangent is per-system uniform under sum-reduction).
    """
    per_atom = multipole_real_space_energy(
        positions,
        multipole_moments,
        cell,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        sigma,
        alpha,
        batch_idx=batch_idx,
        half_neighbor_list=half_neighbor_list,
    )
    charges, dipoles_cart, _quadrupoles, l_max = split_multipole_moments(
        multipole_moments
    )
    if l_max == 2:
        # l=2 already differentiates cell through its per-atom-weighted cell-grad op.
        return per_atom
    num_systems = (
        int(cell.shape[0]) if (batch_idx is not None and cell.dim() == 3) else 1
    )
    dipoles_in = (
        dipoles_cart
        if dipoles_cart is not None
        else torch.zeros(
            (positions.shape[0], 3), dtype=positions.dtype, device=positions.device
        )
    )
    return _AttachRealSpaceCellGradLmax1.apply(
        per_atom,
        cell,
        positions,
        charges,
        dipoles_in,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
        num_systems,
        l_max,
        half_neighbor_list,
    )


@torch.library.custom_op(
    "nvalchemiops::multipole_real_space_monopole_cell_grad", mutates_args=()
)
def _rs_monopole_cell_grad_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    half_neighbor_list: bool,
) -> torch.Tensor:
    """Unweighted total ``dE/dcell`` for the l=0 real-space energy (stress)."""
    wp_device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(positions.dtype)
    wp_vec = get_wp_vec_dtype(positions.dtype)
    wp_mat = get_wp_mat_dtype(positions.dtype)
    grad_cell = torch.zeros_like(cell.detach()).contiguous()
    multipole_real_space_monopole_csr_cell_grad(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(grad_cell, dtype=wp_mat),
        device=str(wp_device),
        half_neighbor_list=half_neighbor_list,
    )
    return grad_cell


@_rs_monopole_cell_grad_op.register_fake
def _(positions, charges, cell, sigma, alpha, idx_j, nptr, shifts, half):
    return torch.empty_like(cell)


def _rs_monopole_cell_grad_setup(ctx, inputs, output):  # pragma: no cover
    """Save inputs for the cell-grad double-backward (stress-loss)."""
    (
        positions,
        charges,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        half_neighbor_list,
    ) = inputs
    ctx.save_for_backward(
        positions, charges, cell, sigma, alpha, idx_j, neighbor_ptr, unit_shifts
    )
    ctx.half_neighbor_list = half_neighbor_list


def _rs_monopole_cell_grad_backward(ctx, g_cell):  # pragma: no cover
    """∂/∂{positions, charges, cell} of ⟨g_cell, dE/dcell⟩ (stress-loss).

    Reuses the force-loss radial Hessian with the per-pair direction
    ``w = g_cellᵀ·n`` (n = unit_shifts); see the Warp kernel.
    """
    (
        positions,
        charges,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    ) = ctx.saved_tensors
    need = ctx.needs_input_grad
    if not (need[0] or need[1] or need[2]):
        return (None,) * 9
    device = positions.device
    wp_device = wp.device_from_torch(device)
    dtype = positions.dtype
    wp_scalar = get_wp_dtype(dtype)
    wp_vec = get_wp_vec_dtype(dtype)
    wp_mat = get_wp_mat_dtype(dtype)
    n = positions.shape[0]
    scale_val = 1.0 if ctx.half_neighbor_list else 0.5
    scale = torch.tensor([scale_val], dtype=torch.float64, device=device)
    grad_positions = torch.zeros((n, 3), dtype=dtype, device=device)
    grad_charges = torch.zeros(n, dtype=dtype, device=device)
    grad_cell_cc = torch.zeros((1, 3, 3), dtype=dtype, device=device)
    multipole_real_space_monopole_csr_cell_grad_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(scale, dtype=wp.float64),
        wp.from_torch(g_cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(grad_positions, dtype=wp_vec),
        wp.from_torch(grad_charges, dtype=wp_scalar),
        wp.from_torch(grad_cell_cc, dtype=wp_mat),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return (
        grad_positions if need[0] else None,
        grad_charges if need[1] else None,
        grad_cell_cc.reshape(cell.shape) if need[2] else None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "nvalchemiops::multipole_real_space_monopole_cell_grad",
    _rs_monopole_cell_grad_backward,
    setup_context=_rs_monopole_cell_grad_setup,
)


@torch.library.custom_op(
    "nvalchemiops::multipole_real_space_monopole_fused", mutates_args=()
)
def _rs_monopole_fused_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    sigma: torch.Tensor,
    alpha: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    half_neighbor_list: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One fused launch: scalar total energy + per-atom (position, charge) grads."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    energies = torch.zeros(num_atoms, dtype=torch.float64, device=device)
    grad_positions = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    grad_charges = torch.zeros(num_atoms, dtype=input_dtype, device=device)
    multipole_real_space_monopole_csr_energy_fused(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigma.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alpha.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(energies, dtype=wp.float64),
        wp.from_torch(grad_positions, dtype=wp_vec),
        wp.from_torch(grad_charges, dtype=wp_scalar),
        with_pos_grad=True,
        with_charge_grad=True,
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return energies.sum(), grad_positions, grad_charges


@_rs_monopole_fused_op.register_fake
def _(positions, charges, cell, sigma, alpha, idx_j, nptr, shifts, half):
    return (
        positions.new_empty((), dtype=torch.float64),
        torch.empty_like(positions),
        torch.empty_like(charges),
    )


def _rs_monopole_fused_setup(ctx, inputs, output):  # pragma: no cover
    """Save inputs + the precomputed (position, charge) grads."""
    (
        positions,
        charges,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        half_neighbor_list,
    ) = inputs
    _energy, grad_positions, grad_charges = output
    ctx.save_for_backward(
        positions,
        charges,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        grad_positions,
        grad_charges,
    )
    ctx.half_neighbor_list = half_neighbor_list


def _rs_monopole_fused_backward(ctx, grad_E, _gg_pos, _gg_q):  # pragma: no cover
    """Broadcast precomputed grads (plain forces); on-tape op under create_graph."""
    (
        positions,
        charges,
        cell,
        sigma,
        alpha,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        pre_gp,
        pre_gq,
    ) = ctx.saved_tensors
    need = ctx.needs_input_grad
    if (need[0] or need[1]) and torch.is_grad_enabled():
        grad_energies = grad_E * torch.ones(
            positions.shape[0], dtype=torch.float64, device=positions.device
        )
        gp, gq = torch.ops.nvalchemiops.multipole_real_space_monopole_backward(
            grad_energies,
            positions,
            charges,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        )
    else:
        gp, gq = grad_E * pre_gp, grad_E * pre_gq
    grad_cell = None
    if need[2]:
        grad_cell = (
            grad_E
            * torch.ops.nvalchemiops.multipole_real_space_monopole_cell_grad(
                positions,
                charges,
                cell,
                sigma,
                alpha,
                idx_j,
                neighbor_ptr,
                unit_shifts,
                ctx.half_neighbor_list,
            )
        )
    return (
        gp if need[0] else None,
        gq if need[1] else None,
        grad_cell,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "nvalchemiops::multipole_real_space_monopole_fused",
    _rs_monopole_fused_backward,
    setup_context=_rs_monopole_fused_setup,
)


# ---------------------------------------------------------------------------
# User-facing entry point
# ---------------------------------------------------------------------------


def multipole_real_space_energy(
    positions: torch.Tensor,
    multipole_moments: torch.Tensor,
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
    r"""GTO-Ewald real-space multipole energy.

    Unified entry point covering :math:`l_{max} \in \{0, 1, 2\}`. The trailing
    dim of ``multipole_moments`` selects the path:

    * :math:`(N, 1)` → :math:`l_{max}=0` (charges only)
    * :math:`(N, 4)` → :math:`l_{max}=1` (charges + dipoles, e3nn :math:`(y, z, x)` order)
    * :math:`(N, 9)` → :math:`l_{max}=2` (adds the quadrupole channels)

    Channel ``[:, 0]`` is the charge and ``[:, 1:4]`` is the dipole in e3nn
    :math:`(y, z, x)` spherical order; the wrapper permutes to Cartesian before
    calling the Warp kernel.

    Returns per-atom :math:`(N,)` :math:`\text{float64}` for all paths; the
    caller owns the atom-global reduction (``.sum()`` for total energy,
    ``scatter_add`` for per-system totals). Non-uniform per-atom backward
    weights are supported across all :math:`l_{max}`.

    Single-system vs batched dispatch
    ---------------------------------
    Mirrors :func:`multipole_ewald_summation`: pass ``cell`` of shape
    ``(3, 3)`` / ``(1, 3, 3)`` (single) or ``(B, 3, 3)`` (batched) and use
    ``batch_idx`` to select the batched path. In batched mode ``sigma`` and
    ``alpha`` are per-system ``(B,)`` tensors and the return is per-atom
    :math:`(N_\text{total},)` for all :math:`l_{max}`.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic positions (``(N_total, 3)`` flat across systems when batched).
    multipole_moments : torch.Tensor, shape (N, (l_max + 1)**2)
        Per-atom multipole moments in e3nn spherical layout.
    cell : torch.Tensor, shape (3, 3) / (1, 3, 3) or (B, 3, 3)
        Unit cell matrix (row vectors = lattice vectors); ``(B, 3, 3)`` when
        batched.
    idx_j, neighbor_ptr, unit_shifts : torch.Tensor
        CSR neighbor list (``int32`` / ``int32`` / ``vec3i``).
    sigma : torch.Tensor, shape (1,) or (B,)
        GTO density-basis width :math:`\sigma` (per-system ``(B,)`` when
        batched).
    alpha : torch.Tensor, shape (1,) or (B,)
        Ewald splitting parameter :math:`\alpha` (per-system ``(B,)`` when
        batched).
    batch_idx : torch.Tensor, optional, shape (N_total,), int32
        Per-atom system index (expected sorted). Required when ``cell`` is
        ``(B, 3, 3)``; must be ``None`` for a single cell.
    half_neighbor_list : bool, default False
        Forwarded to the :math:`l_{max}=2` path (no effect otherwise).

    Returns
    -------
    torch.Tensor, shape (N,), float64
        Per-atom real-space energy (single), or per-atom
        :math:`(N_\text{total},)` batched — for all :math:`l_{max}`.
    """
    if batch_idx is not None:
        return _batch_multipole_real_space_energy(
            positions,
            multipole_moments,
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
    if multipole_moments.ndim != 2 or multipole_moments.shape[0] != positions.shape[0]:
        raise ValueError(
            "multipole_moments must be (N, (l_max+1)^2) matching positions[0]; "
            f"got {tuple(multipole_moments.shape)}"
        )
    if cell.ndim == 2 and cell.shape == (3, 3):
        cell = cell.unsqueeze(0)
    if cell.ndim != 3 or cell.shape[-2:] != (3, 3):
        raise ValueError(f"cell must be (3, 3) or (1, 3, 3), got {tuple(cell.shape)}")
    if cell.shape[0] != 1:
        raise ValueError(
            f"single-system cell expected (1, 3, 3); got batch size {cell.shape[0]}. "
            "Pass batch_idx for B > 1."
        )

    charges, dipoles_cart, quadrupoles, l_max = split_multipole_moments(
        multipole_moments
    )
    if l_max == 2:
        # Local import keeps the l_max=2 second-order-backward kernel module
        # off the import path when the l_max=2 path is not exercised.
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
            multipole_real_space_quadrupole_energy,
        )

        return multipole_real_space_quadrupole_energy(
            positions,
            charges,
            dipoles_cart,
            quadrupoles,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            half_neighbor_list=half_neighbor_list,
        )

    if l_max == 1:
        return torch.ops.nvalchemiops.multipole_real_space_dipole(
            positions,
            charges,
            dipoles_cart,
            cell,
            sigma,
            alpha,
            idx_j,
            neighbor_ptr,
            unit_shifts,
        )
    return torch.ops.nvalchemiops.multipole_real_space_monopole(
        positions, charges, cell, sigma, alpha, idx_j, neighbor_ptr, unit_shifts
    )


# ===========================================================================
# Batched l_max=0 autograd.Function wrappers
# ===========================================================================


class BatchMultipoleRealSpaceMonopoleBackwardFunction(torch.autograd.Function):
    r"""Batched analog of :class:`MultipoleRealSpaceMonopoleBackwardFunction`.

    Forward runs
    :func:`batch_multipole_real_space_monopole_csr_energy_backward` (per-atom
    analytical :math:`\partial/\partial(\text{positions}, \text{charges})` with
    per-system ``cells[b]`` / ``alphas[b]`` lookup via ``batch_idx``); backward
    runs :func:`batch_multipole_real_space_monopole_csr_energy_2nd_backward` for
    the second-order pair Hessians. Supports ``create_graph=True`` force-loss
    training.
    """

    @staticmethod
    def forward(
        ctx,
        grad_energies: torch.Tensor,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cells: torch.Tensor,
        sigmas: torch.Tensor,
        alphas: torch.Tensor,
        idx_j: torch.Tensor,
        neighbor_ptr: torch.Tensor,
        unit_shifts: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch the batched first-order backward kernel; save tensors for the 2nd-backward."""
        device = positions.device
        wp_device = wp.device_from_torch(device)
        input_dtype = positions.dtype
        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        num_atoms = positions.shape[0]

        grad_positions = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
        grad_charges = torch.zeros(charges.shape[0], dtype=input_dtype, device=device)

        batch_multipole_real_space_monopole_csr_energy_backward(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
            wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
            wp.from_torch(grad_positions, dtype=wp_vec),
            wp.from_torch(grad_charges, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp_device),
        )

        ctx.save_for_backward(
            grad_energies,
            positions,
            charges,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        )
        return grad_positions, grad_charges

    @staticmethod
    def backward(
        ctx, gg_positions: torch.Tensor, gg_charges: torch.Tensor
    ):  # pragma: no cover
        """Second-order backward via :func:`batch_multipole_real_space_monopole_csr_energy_2nd_backward`."""
        (
            grad_energies,
            positions,
            charges,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        ) = ctx.saved_tensors
        device = positions.device
        wp_device = wp.device_from_torch(device)
        input_dtype = positions.dtype
        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        num_atoms = positions.shape[0]
        gg_grad_energies_2nd = torch.zeros_like(grad_energies)
        gg_positions_2nd = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
        gg_charges_2nd = torch.zeros_like(charges)

        batch_multipole_real_space_monopole_csr_energy_2nd_backward(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
            wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
            wp.from_torch(gg_positions.contiguous(), dtype=wp_vec),
            wp.from_torch(gg_charges.contiguous(), dtype=wp_scalar),
            wp.from_torch(gg_grad_energies_2nd, dtype=wp.float64),
            wp.from_torch(gg_positions_2nd, dtype=wp_vec),
            wp.from_torch(gg_charges_2nd, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp_device),
        )

        return (
            gg_grad_energies_2nd,
            gg_positions_2nd,
            gg_charges_2nd,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class BatchMultipoleRealSpaceMonopoleFusedScalarFunction(torch.autograd.Function):
    r"""Batched analog of :class:`MultipoleRealSpaceMonopoleFusedScalarFunction`.

    Single-launch fused energy + analytical-gradient kernel for the batched
    :math:`l_{max}=0` real-space path, returning per-system summed energies.

    Returns
    -------
    torch.Tensor
        Shape :math:`(B,)` :math:`\text{float64}` per-system total energies.

    Notes
    -----
    The precomputed gradient tensors assume an upstream gradient that is uniform
    within each system (the natural shape for ``E.sum().backward()`` or
    ``(weights * E).sum().backward()`` with per-system ``weights``). For
    non-uniform per-atom upstream gradients use
    :class:`BatchMultipoleRealSpaceMonopoleFunction`, which returns per-atom
    energies.
    """

    @staticmethod
    def forward(
        ctx,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cells: torch.Tensor,
        sigmas: torch.Tensor,
        alphas: torch.Tensor,
        idx_j: torch.Tensor,
        neighbor_ptr: torch.Tensor,
        unit_shifts: torch.Tensor,
        batch_idx: torch.Tensor,
        half_neighbor_list: bool = False,
    ) -> torch.Tensor:
        """Run the batched lmax=0 fused Warp kernel; stash per-atom gradients in ctx."""
        device = positions.device
        wp_device = wp.device_from_torch(device)
        input_dtype = positions.dtype
        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        with_pos = bool(positions.requires_grad)
        with_q = bool(charges.requires_grad)
        with_cell = bool(cells.requires_grad)
        num_atoms = positions.shape[0]
        num_systems = cells.shape[0]

        per_atom_energies = torch.zeros(num_atoms, dtype=torch.float64, device=device)
        grad_positions = (
            torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
            if with_pos
            else torch.zeros((1, 3), dtype=input_dtype, device=device)
        )
        grad_charges = (
            torch.zeros(num_atoms, dtype=input_dtype, device=device)
            if with_q
            else torch.zeros(1, dtype=input_dtype, device=device)
        )

        batch_multipole_real_space_monopole_csr_energy_fused(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
            wp.from_torch(per_atom_energies, dtype=wp.float64),
            wp.from_torch(grad_positions, dtype=wp_vec),
            wp.from_torch(grad_charges, dtype=wp_scalar),
            with_pos_grad=with_pos,
            with_charge_grad=with_q,
            wp_dtype=wp_scalar,
            device=str(wp_device),
        )

        per_system_energies = torch.zeros(
            num_systems, dtype=torch.float64, device=device
        )
        per_system_energies.scatter_add_(
            0, batch_idx.to(torch.int64), per_atom_energies
        )

        if with_cell:
            from nvalchemiops.interactions.electrostatics.multipole_ewald_cell_grad import (
                batch_multipole_real_space_monopole_csr_cell_grad,
            )

            grad_cell_buf = torch.zeros_like(cells.detach()).contiguous()
            batch_multipole_real_space_monopole_csr_cell_grad(
                wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
                wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
                wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
                wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
                wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
                wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
                wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
                wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
                wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
                wp.from_torch(grad_cell_buf, dtype=wp_mat),
                device=str(wp_device),
                half_neighbor_list=half_neighbor_list,
            )
            grad_cell_to_save = grad_cell_buf
        else:
            grad_cell_to_save = None

        # Save precomputed grads for the fast path and inputs for the
        # double-backward path. Plain forces broadcast the precomputed grads;
        # under ``create_graph`` pos/charge grads route through the on-tape
        # backward Function (2nd-order kernel) for force-loss training.
        ctx.save_for_backward(
            grad_positions if with_pos else None,
            grad_charges if with_q else None,
            grad_cell_to_save,
            positions,
            charges,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        )
        ctx.with_cell = with_cell
        return per_system_energies

    @staticmethod
    def backward(ctx, grad_E: torch.Tensor):  # pragma: no cover
        """Fast per-system precomputed-grad broadcast for plain forces; on-tape
        backward Function (double-backward capable) under ``create_graph``.

        ``grad_E`` has shape ``(B,)``; the per-atom cotangent is
        ``grad_E[batch_idx[i]]``.
        """
        (
            pre_grad_positions,
            pre_grad_charges,
            grad_cell,
            positions,
            charges,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        ) = ctx.saved_tensors
        need_pq = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        atom_weights = grad_E[batch_idx.to(torch.int64)]  # (N,)
        if need_pq and torch.is_grad_enabled():
            out_grad_positions, out_grad_charges = (
                BatchMultipoleRealSpaceMonopoleBackwardFunction.apply(
                    atom_weights.to(torch.float64),
                    positions,
                    charges,
                    cells,
                    sigmas,
                    alphas,
                    idx_j,
                    neighbor_ptr,
                    unit_shifts,
                    batch_idx,
                )
            )
        elif need_pq:
            out_grad_positions = (
                atom_weights.unsqueeze(-1).to(pre_grad_positions.dtype)
                * pre_grad_positions
                if pre_grad_positions is not None
                else None
            )
            out_grad_charges = (
                atom_weights.to(pre_grad_charges.dtype) * pre_grad_charges
                if pre_grad_charges is not None
                else None
            )
        else:
            out_grad_positions = out_grad_charges = None
        # Weight each system's (3, 3) cell-grad by its own grad_E[b].
        if ctx.with_cell:
            out_grad_cells = grad_E.view(-1, 1, 1).to(grad_cell.dtype) * grad_cell
        else:
            out_grad_cells = None
        return (
            out_grad_positions,
            out_grad_charges,
            out_grad_cells,  # cells
            None,  # sigmas
            None,  # alphas
            None,  # idx_j
            None,  # neighbor_ptr
            None,  # unit_shifts
            None,  # batch_idx
            None,  # half_neighbor_list
        )


class BatchMultipoleRealSpaceMonopoleFunction(torch.autograd.Function):
    r"""Batched l_max=0 GTO-Ewald multipole real-space (charges only)."""

    @staticmethod
    def forward(
        ctx,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cells: torch.Tensor,
        sigmas: torch.Tensor,
        alphas: torch.Tensor,
        idx_j: torch.Tensor,
        neighbor_ptr: torch.Tensor,
        unit_shifts: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Launch the batched forward Warp kernel; save tensors for backward."""
        device = positions.device
        wp_device = wp.device_from_torch(device)
        input_dtype = positions.dtype
        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        num_atoms = positions.shape[0]

        energies = torch.zeros(num_atoms, dtype=torch.float64, device=device)
        batch_multipole_real_space_monopole_csr_energy(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
            wp.from_torch(energies, dtype=wp.float64),
            wp_dtype=wp_scalar,
            device=str(wp_device),
        )

        ctx.save_for_backward(
            positions,
            charges,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        )
        return energies

    @staticmethod
    def backward(ctx, grad_energies: torch.Tensor):  # pragma: no cover
        """Route through :class:`BatchMultipoleRealSpaceMonopoleBackwardFunction`."""
        (
            positions,
            charges,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        ) = ctx.saved_tensors
        grad_positions, grad_charges = (
            BatchMultipoleRealSpaceMonopoleBackwardFunction.apply(
                grad_energies,
                positions,
                charges,
                cells,
                sigmas,
                alphas,
                idx_j,
                neighbor_ptr,
                unit_shifts,
                batch_idx,
            )
        )
        return (
            grad_positions,
            grad_charges,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Batched l_max = 0 / 1 real-space — torch.library.custom_op chains
# ---------------------------------------------------------------------------


def _batch_real_space_monopole_forward(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cells: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Batched l_max=0 real-space forward (charges only)."""
    wp_device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(positions.dtype)
    wp_vec = get_wp_vec_dtype(positions.dtype)
    wp_mat = get_wp_mat_dtype(positions.dtype)
    energies = torch.zeros(
        positions.shape[0], dtype=torch.float64, device=positions.device
    )
    batch_multipole_real_space_monopole_csr_energy(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
        wp.from_torch(energies, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return energies


def _batch_real_space_forward_fake(positions, *args):  # pragma: no cover
    """Shape/dtype metadata: per-atom energies ``(N_total,)`` in float64."""
    return positions.new_empty((positions.shape[0],), dtype=torch.float64)


def _batch_real_space_monopole_backward(  # pragma: no cover
    grad_energies: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cells: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched l_max=0 first-order backward."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    grad_positions = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    grad_charges = torch.zeros(charges.shape[0], dtype=input_dtype, device=device)
    batch_multipole_real_space_monopole_csr_energy_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
        wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_positions, dtype=wp_vec),
        wp.from_torch(grad_charges, dtype=wp_scalar),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_positions, grad_charges


def _batch_real_space_monopole_double_backward(  # pragma: no cover
    gg_positions: torch.Tensor,
    gg_charges: torch.Tensor,
    grad_energies: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    cells: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched l_max=0 second-order backward (``create_graph``)."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    gg_grad_energies_2nd = torch.zeros_like(grad_energies)
    gg_positions_2nd = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    gg_charges_2nd = torch.zeros_like(charges)
    batch_multipole_real_space_monopole_csr_energy_2nd_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
        wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp_vec),
        wp.from_torch(gg_charges.contiguous(), dtype=wp_scalar),
        wp.from_torch(gg_grad_energies_2nd, dtype=wp.float64),
        wp.from_torch(gg_positions_2nd, dtype=wp_vec),
        wp.from_torch(gg_charges_2nd, dtype=wp_scalar),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return gg_grad_energies_2nd, gg_positions_2nd, gg_charges_2nd


register_warp_op_chain(
    name="nvalchemiops::batch_multipole_real_space_monopole",
    forward=_batch_real_space_monopole_forward,
    backward=_batch_real_space_monopole_backward,
    double_backward=_batch_real_space_monopole_double_backward,
    diff_input_positions=(0, 1),
    n_forward_inputs=9,
    second_order_diff_positions=(0, 1, 2),
    n_backward_inputs=10,
    forward_fake=_batch_real_space_forward_fake,
    batch_match=True,
)


def _batch_real_space_dipole_forward(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    cells: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Batched l_max=1 real-space forward (charges + dipoles)."""
    wp_device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(positions.dtype)
    wp_vec = get_wp_vec_dtype(positions.dtype)
    wp_mat = get_wp_mat_dtype(positions.dtype)
    energies = torch.zeros(
        positions.shape[0], dtype=torch.float64, device=positions.device
    )
    batch_multipole_real_space_dipole_csr_energy(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
        wp.from_torch(energies, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return energies


def _batch_real_space_dipole_backward(  # pragma: no cover
    grad_energies: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    cells: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched l_max=1 first-order backward."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    grad_positions = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    grad_charges = torch.zeros(charges.shape[0], dtype=input_dtype, device=device)
    grad_dipoles = torch.zeros((dipoles.shape[0], 3), dtype=input_dtype, device=device)
    batch_multipole_real_space_dipole_csr_energy_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
        wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_positions, dtype=wp_vec),
        wp.from_torch(grad_charges, dtype=wp_scalar),
        wp.from_torch(grad_dipoles, dtype=wp_vec),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_positions, grad_charges, grad_dipoles


def _batch_real_space_dipole_double_backward(  # pragma: no cover
    gg_positions: torch.Tensor,
    gg_charges: torch.Tensor,
    gg_dipoles: torch.Tensor,
    grad_energies: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    cells: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched l_max=1 second-order backward (``create_graph``)."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    gg_grad_energies_2nd = torch.zeros_like(grad_energies)
    gg_positions_2nd = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    gg_charges_2nd = torch.zeros_like(charges)
    gg_dipoles_2nd = torch.zeros(
        (dipoles.shape[0], 3), dtype=input_dtype, device=device
    )
    batch_multipole_real_space_dipole_csr_energy_2nd_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
        wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp_vec),
        wp.from_torch(gg_charges.contiguous(), dtype=wp_scalar),
        wp.from_torch(gg_dipoles.contiguous(), dtype=wp_vec),
        wp.from_torch(gg_grad_energies_2nd, dtype=wp.float64),
        wp.from_torch(gg_positions_2nd, dtype=wp_vec),
        wp.from_torch(gg_charges_2nd, dtype=wp_scalar),
        wp.from_torch(gg_dipoles_2nd, dtype=wp_vec),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return gg_grad_energies_2nd, gg_positions_2nd, gg_charges_2nd, gg_dipoles_2nd


register_warp_op_chain(
    name="nvalchemiops::batch_multipole_real_space_dipole",
    forward=_batch_real_space_dipole_forward,
    backward=_batch_real_space_dipole_backward,
    double_backward=_batch_real_space_dipole_double_backward,
    diff_input_positions=(0, 1, 2),
    n_forward_inputs=10,
    second_order_diff_positions=(0, 1, 2, 3),
    n_backward_inputs=11,
    forward_fake=_batch_real_space_forward_fake,
    batch_match=True,
)


# ---------------------------------------------------------------------------
# Batched fused-scalar real-space (l=0/1) — torch.library.custom_op chains
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_real_space_dipole_cell_grad", mutates_args=()
)
def _batch_rs_dipole_cell_grad_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    cells: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
    half_neighbor_list: bool,
) -> torch.Tensor:
    """Per-system unweighted ``dE/dcell`` (B, 3, 3) for the batched l=1 energy."""
    wp_device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(positions.dtype)
    wp_vec = get_wp_vec_dtype(positions.dtype)
    wp_mat = get_wp_mat_dtype(positions.dtype)
    grad_cells = torch.zeros_like(cells.detach()).contiguous()
    batch_multipole_real_space_dipole_csr_cell_grad(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(batch_idx.detach().to(torch.int32).contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(grad_cells, dtype=wp_mat),
        device=str(wp_device),
        half_neighbor_list=half_neighbor_list,
    )
    return grad_cells


@_batch_rs_dipole_cell_grad_op.register_fake
def _(
    positions,
    charges,
    dipoles,
    cells,
    sigmas,
    alphas,
    idx_j,
    nptr,
    shifts,
    batch_idx,
    half,
):
    return torch.empty_like(cells)


def _batch_rs_dipole_cell_grad_setup(ctx, inputs, output):  # pragma: no cover
    """Save inputs for the batched l=1 cell-grad double-backward."""
    (
        positions,
        charges,
        dipoles,
        cells,
        sigmas,
        alphas,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
        half_neighbor_list,
    ) = inputs
    ctx.save_for_backward(
        positions,
        charges,
        dipoles,
        cells,
        sigmas,
        alphas,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
    )
    ctx.half_neighbor_list = half_neighbor_list


def _batch_rs_dipole_cell_grad_backward(ctx, g_cell):  # pragma: no cover
    """Batched ∂/∂{positions, charges, dipoles, cells} of ⟨g_cell, dE/dcell⟩."""
    (
        positions,
        charges,
        dipoles,
        cells,
        sigmas,
        alphas,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
    ) = ctx.saved_tensors
    need = ctx.needs_input_grad
    if not (need[0] or need[1] or need[2] or need[3]):
        return (None,) * 11
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
    grad_positions = torch.zeros((n, 3), dtype=dtype, device=device)
    grad_charges = torch.zeros(n, dtype=dtype, device=device)
    grad_dipoles = torch.zeros((n, 3), dtype=dtype, device=device)
    grad_cells = torch.zeros_like(cells)
    batch_multipole_real_space_dipole_csr_cell_grad_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(scale, dtype=wp.float64),
        wp.from_torch(g_cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(grad_positions, dtype=wp_vec),
        wp.from_torch(grad_charges, dtype=wp_scalar),
        wp.from_torch(grad_dipoles, dtype=wp_vec),
        wp.from_torch(grad_cells, dtype=wp_mat),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return (
        grad_positions if need[0] else None,
        grad_charges if need[1] else None,
        grad_dipoles if need[2] else None,
        grad_cells if need[3] else None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "nvalchemiops::batch_multipole_real_space_dipole_cell_grad",
    _batch_rs_dipole_cell_grad_backward,
    setup_context=_batch_rs_dipole_cell_grad_setup,
)


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_real_space_dipole_fused", mutates_args=()
)
def _batch_rs_dipole_fused_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    cells: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
    half_neighbor_list: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One fused launch: per-system energy (B,) + per-atom moment gradients."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    per_atom = torch.zeros(num_atoms, dtype=torch.float64, device=device)
    grad_positions = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    grad_charges = torch.zeros(num_atoms, dtype=input_dtype, device=device)
    grad_dipoles = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    batch_multipole_real_space_dipole_csr_energy_fused(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(batch_idx.detach().to(torch.int32).contiguous(), dtype=wp.int32),
        wp.from_torch(per_atom, dtype=wp.float64),
        wp.from_torch(grad_positions, dtype=wp_vec),
        wp.from_torch(grad_charges, dtype=wp_scalar),
        wp.from_torch(grad_dipoles, dtype=wp_vec),
        with_pos_grad=True,
        with_charge_grad=True,
        with_dipole_grad=True,
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    per_system = torch.zeros(cells.shape[0], dtype=torch.float64, device=device)
    per_system.scatter_add_(0, batch_idx.to(torch.int64), per_atom)
    return per_system, grad_positions, grad_charges, grad_dipoles


@_batch_rs_dipole_fused_op.register_fake
def _(
    positions,
    charges,
    dipoles,
    cells,
    sigmas,
    alphas,
    idx_j,
    nptr,
    shifts,
    batch_idx,
    half,
):
    return (
        cells.new_empty((cells.shape[0],), dtype=torch.float64),
        torch.empty_like(positions),
        torch.empty_like(charges),
        torch.empty_like(dipoles),
    )


def _batch_rs_dipole_fused_setup(ctx, inputs, output):  # pragma: no cover
    """Save inputs + precomputed moment grads (returned as op outputs)."""
    (
        positions,
        charges,
        dipoles,
        cells,
        sigmas,
        alphas,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
        half_neighbor_list,
    ) = inputs
    _e, grad_positions, grad_charges, grad_dipoles = output
    ctx.save_for_backward(
        positions,
        charges,
        dipoles,
        cells,
        sigmas,
        alphas,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
        grad_positions,
        grad_charges,
        grad_dipoles,
    )
    ctx.half_neighbor_list = half_neighbor_list


def _batch_rs_dipole_fused_backward(ctx, grad_E, _ggp, _ggq, _ggm):  # pragma: no cover
    """Per-system grad_E[batch_idx] broadcast (plain forces) / on-tape op (create_graph)."""
    (
        positions,
        charges,
        dipoles,
        cells,
        sigmas,
        alphas,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
        pre_gp,
        pre_gq,
        pre_gmu,
    ) = ctx.saved_tensors
    need = ctx.needs_input_grad
    atom_weights = grad_E[batch_idx.to(torch.int64)]
    if (need[0] or need[1] or need[2]) and torch.is_grad_enabled():
        gp, gq, gmu = torch.ops.nvalchemiops.batch_multipole_real_space_dipole_backward(
            atom_weights.to(torch.float64),
            positions,
            charges,
            dipoles,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        )
    else:
        w = atom_weights.to(pre_gp.dtype)
        gp = w.unsqueeze(-1) * pre_gp
        gq = w * pre_gq
        gmu = w.unsqueeze(-1) * pre_gmu
    grad_cells = None
    if need[3]:
        cg = torch.ops.nvalchemiops.batch_multipole_real_space_dipole_cell_grad(
            positions,
            charges,
            dipoles,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
            ctx.half_neighbor_list,
        )
        grad_cells = grad_E.view(-1, 1, 1).to(cg.dtype) * cg
    # 11 inputs: positions, charges, dipoles, cells, sigmas, alphas, idx_j,
    # neighbor_ptr, unit_shifts, batch_idx, half_neighbor_list.
    return (
        gp if need[0] else None,
        gq if need[1] else None,
        gmu if need[2] else None,
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
    "nvalchemiops::batch_multipole_real_space_dipole_fused",
    _batch_rs_dipole_fused_backward,
    setup_context=_batch_rs_dipole_fused_setup,
)


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_real_space_monopole_cell_grad", mutates_args=()
)
def _batch_rs_monopole_cell_grad_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cells: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
    half_neighbor_list: bool,
) -> torch.Tensor:
    """Per-system unweighted ``dE/dcell`` (B, 3, 3) for the batched l=0 energy."""
    wp_device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(positions.dtype)
    wp_vec = get_wp_vec_dtype(positions.dtype)
    wp_mat = get_wp_mat_dtype(positions.dtype)
    grad_cells = torch.zeros_like(cells.detach()).contiguous()
    batch_multipole_real_space_monopole_csr_cell_grad(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(batch_idx.detach().to(torch.int32).contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(grad_cells, dtype=wp_mat),
        device=str(wp_device),
        half_neighbor_list=half_neighbor_list,
    )
    return grad_cells


@_batch_rs_monopole_cell_grad_op.register_fake
def _(positions, charges, cells, sigmas, alphas, idx_j, nptr, shifts, batch_idx, half):
    return torch.empty_like(cells)


def _batch_rs_monopole_cell_grad_setup(ctx, inputs, output):  # pragma: no cover
    """Save inputs for the batched l=0 cell-grad double-backward."""
    (
        positions,
        charges,
        cells,
        sigmas,
        alphas,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
        half_neighbor_list,
    ) = inputs
    ctx.save_for_backward(
        positions,
        charges,
        cells,
        sigmas,
        alphas,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
    )
    ctx.half_neighbor_list = half_neighbor_list


def _batch_rs_monopole_cell_grad_backward(ctx, g_cell):  # pragma: no cover
    """Batched ∂/∂{positions, charges, cells} of ⟨g_cell, dE/dcell⟩."""
    (
        positions,
        charges,
        cells,
        sigmas,
        alphas,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
    ) = ctx.saved_tensors
    need = ctx.needs_input_grad
    if not (need[0] or need[1] or need[2]):
        return (None,) * 10
    device = positions.device
    wp_device = wp.device_from_torch(device)
    dtype = positions.dtype
    wp_scalar = get_wp_dtype(dtype)
    wp_vec = get_wp_vec_dtype(dtype)
    wp_mat = get_wp_mat_dtype(dtype)
    n = positions.shape[0]
    scale_val = 1.0 if ctx.half_neighbor_list else 0.5
    scale = torch.tensor([scale_val], dtype=torch.float64, device=device)
    grad_positions = torch.zeros((n, 3), dtype=dtype, device=device)
    grad_charges = torch.zeros(n, dtype=dtype, device=device)
    grad_cells = torch.zeros_like(cells)
    batch_multipole_real_space_monopole_csr_cell_grad_backward(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(scale, dtype=wp.float64),
        wp.from_torch(g_cell.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(grad_positions, dtype=wp_vec),
        wp.from_torch(grad_charges, dtype=wp_scalar),
        wp.from_torch(grad_cells, dtype=wp_mat),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return (
        grad_positions if need[0] else None,
        grad_charges if need[1] else None,
        grad_cells if need[2] else None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "nvalchemiops::batch_multipole_real_space_monopole_cell_grad",
    _batch_rs_monopole_cell_grad_backward,
    setup_context=_batch_rs_monopole_cell_grad_setup,
)


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_real_space_monopole_fused", mutates_args=()
)
def _batch_rs_monopole_fused_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cells: torch.Tensor,
    sigmas: torch.Tensor,
    alphas: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
    half_neighbor_list: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One fused launch: per-system energy (B,) + per-atom (position, charge) grads."""
    input_dtype = positions.dtype
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    num_atoms = positions.shape[0]
    per_atom = torch.zeros(num_atoms, dtype=torch.float64, device=device)
    grad_positions = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
    grad_charges = torch.zeros(num_atoms, dtype=input_dtype, device=device)
    batch_multipole_real_space_monopole_csr_energy_fused(
        wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
        wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
        wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
        wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
        wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(batch_idx.detach().to(torch.int32).contiguous(), dtype=wp.int32),
        wp.from_torch(per_atom, dtype=wp.float64),
        wp.from_torch(grad_positions, dtype=wp_vec),
        wp.from_torch(grad_charges, dtype=wp_scalar),
        with_pos_grad=True,
        with_charge_grad=True,
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    per_system = torch.zeros(cells.shape[0], dtype=torch.float64, device=device)
    per_system.scatter_add_(0, batch_idx.to(torch.int64), per_atom)
    return per_system, grad_positions, grad_charges


@_batch_rs_monopole_fused_op.register_fake
def _(positions, charges, cells, sigmas, alphas, idx_j, nptr, shifts, batch_idx, half):
    return (
        cells.new_empty((cells.shape[0],), dtype=torch.float64),
        torch.empty_like(positions),
        torch.empty_like(charges),
    )


def _batch_rs_monopole_fused_setup(ctx, inputs, output):  # pragma: no cover
    """Save inputs + precomputed (position, charge) grads."""
    (
        positions,
        charges,
        cells,
        sigmas,
        alphas,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
        half_neighbor_list,
    ) = inputs
    _e, grad_positions, grad_charges = output
    ctx.save_for_backward(
        positions,
        charges,
        cells,
        sigmas,
        alphas,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
        grad_positions,
        grad_charges,
    )
    ctx.half_neighbor_list = half_neighbor_list


def _batch_rs_monopole_fused_backward(ctx, grad_E, _ggp, _ggq):  # pragma: no cover
    """Per-system grad_E[batch_idx] broadcast (plain forces) / on-tape op (create_graph)."""
    (
        positions,
        charges,
        cells,
        sigmas,
        alphas,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
        pre_gp,
        pre_gq,
    ) = ctx.saved_tensors
    need = ctx.needs_input_grad
    atom_weights = grad_E[batch_idx.to(torch.int64)]
    if (need[0] or need[1]) and torch.is_grad_enabled():
        gp, gq = torch.ops.nvalchemiops.batch_multipole_real_space_monopole_backward(
            atom_weights.to(torch.float64),
            positions,
            charges,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        )
    else:
        w = atom_weights.to(pre_gp.dtype)
        gp = w.unsqueeze(-1) * pre_gp
        gq = w * pre_gq
    grad_cells = None
    if need[2]:
        cg = torch.ops.nvalchemiops.batch_multipole_real_space_monopole_cell_grad(
            positions,
            charges,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
            ctx.half_neighbor_list,
        )
        grad_cells = grad_E.view(-1, 1, 1).to(cg.dtype) * cg
    # 10 inputs: positions, charges, cells, sigmas, alphas, idx_j, neighbor_ptr,
    # unit_shifts, batch_idx, half_neighbor_list.
    return (
        gp if need[0] else None,
        gq if need[1] else None,
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
    "nvalchemiops::batch_multipole_real_space_monopole_fused",
    _batch_rs_monopole_fused_backward,
    setup_context=_batch_rs_monopole_fused_setup,
)


def _batch_multipole_real_space_energy(
    positions: torch.Tensor,
    multipole_moments: torch.Tensor,
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
    r"""Batched GTO-Ewald real-space multipole energy (internal dispatch target).

    Batched analog of :func:`multipole_real_space_energy`, reached via that
    function's ``batch_idx=`` path. The trailing dim of
    ``multipole_moments`` selects the :math:`l_{max}` path:
    :math:`(N_\text{total}, 1)` → charges only; :math:`(N_\text{total}, 4)` →
    charges + dipoles; :math:`(N_\text{total}, 9)` → adds quadrupoles. Supports
    ``create_graph=True`` force-loss training on the :math:`l_{max} \le 1`
    paths.

    Return shape:

    Per-atom :math:`(N_\text{total},)` float64 for all :math:`l_{max}`; the
    caller does ``scatter_add`` via ``batch_idx`` for per-system totals.

    Non-uniform per-atom backward weights are supported on all paths.

    Parameters
    ----------
    positions : torch.Tensor, shape (N_total, 3)
    multipole_moments : torch.Tensor, shape (N_total, (l_max+1)**2)
        Packed per-atom moments in e3nn spherical layout.
    cells : torch.Tensor, shape (B, 3, 3)
    idx_j, neighbor_ptr, unit_shifts : torch.Tensor
        Flat CSR neighbor list covering all ``B`` systems.
    sigmas : torch.Tensor, shape (B,)
        Per-system GTO density-basis width :math:`\sigma`.
    alphas : torch.Tensor, shape (B,)
        Per-system Ewald splitting parameter :math:`\alpha`.
    batch_idx : torch.Tensor, shape (N_total,), int32
        Per-atom system index.
    half_neighbor_list : bool, default False
        Forwarded to the :math:`l_{max}=2` path (no effect otherwise).

    Returns
    -------
    torch.Tensor
        Per-atom :math:`(N_\text{total},)` float64 for all :math:`l_{max}`.
    """
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(
            f"positions must be (N_total, 3), got {tuple(positions.shape)}"
        )
    if multipole_moments.ndim != 2 or multipole_moments.shape[0] != positions.shape[0]:
        raise ValueError(
            "multipole_moments must be (N_total, (l_max+1)^2) matching positions[0]; "
            f"got {tuple(multipole_moments.shape)}"
        )
    if cells.ndim != 3 or cells.shape[-2:] != (3, 3):
        raise ValueError(f"cells must be (B, 3, 3), got {tuple(cells.shape)}")
    if alphas.ndim != 1 or alphas.shape[0] != cells.shape[0]:
        raise ValueError(
            f"alphas must be (B={cells.shape[0]},), got {tuple(alphas.shape)}"
        )
    if sigmas.ndim != 1 or sigmas.shape[0] != cells.shape[0]:
        raise ValueError(
            f"sigmas must be (B={cells.shape[0]},), got {tuple(sigmas.shape)}"
        )
    if batch_idx.shape[0] != positions.shape[0]:
        raise ValueError(
            f"batch_idx must match N_total={positions.shape[0]}, "
            f"got {tuple(batch_idx.shape)}"
        )

    charges, dipoles_cart, quadrupoles, l_max = split_multipole_moments(
        multipole_moments
    )
    if l_max == 2:
        # Local import keeps the l_max=2 second-order-backward kernel module
        # off the import path when the l_max=2 path is not exercised.
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
            multipole_real_space_quadrupole_energy,
        )

        return multipole_real_space_quadrupole_energy(
            positions,
            charges,
            dipoles_cart,
            quadrupoles,
            cells,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigmas,
            alphas,
            batch_idx=batch_idx,
            half_neighbor_list=half_neighbor_list,
        )

    if l_max == 1:
        return torch.ops.nvalchemiops.batch_multipole_real_space_dipole(
            positions,
            charges,
            dipoles_cart,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        )
    return torch.ops.nvalchemiops.batch_multipole_real_space_monopole(
        positions,
        charges,
        cells,
        sigmas,
        alphas,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        batch_idx,
    )


# ===========================================================================
# Batched l_max = 1 (charges + dipoles) autograd.Function wrappers
# ===========================================================================


class BatchMultipoleRealSpaceBackwardFunction(torch.autograd.Function):
    r"""Batched analog of :class:`MultipoleRealSpaceBackwardFunction`.

    Forward runs the batched :math:`l_{max}=1` first-order backward kernel
    (charges + dipoles); backward runs
    :func:`batch_multipole_real_space_dipole_csr_energy_2nd_backward` for the
    second-order pair Hessians with per-system ``cells[b]`` / ``alphas[b]``
    lookup. Supports ``create_graph=True`` force-loss training.
    """

    @staticmethod
    def forward(
        ctx,
        grad_energies: torch.Tensor,
        positions: torch.Tensor,
        charges: torch.Tensor,
        dipoles: torch.Tensor,
        cells: torch.Tensor,
        sigmas: torch.Tensor,
        alphas: torch.Tensor,
        idx_j: torch.Tensor,
        neighbor_ptr: torch.Tensor,
        unit_shifts: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Launch the batched l_max=1 first-order backward kernel."""
        device = positions.device
        wp_device = wp.device_from_torch(device)
        input_dtype = positions.dtype
        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        num_atoms = positions.shape[0]

        grad_positions = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
        grad_charges = torch.zeros(charges.shape[0], dtype=input_dtype, device=device)
        grad_dipoles = torch.zeros(
            (dipoles.shape[0], 3), dtype=input_dtype, device=device
        )

        batch_multipole_real_space_dipole_csr_energy_backward(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
            wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
            wp.from_torch(grad_positions, dtype=wp_vec),
            wp.from_torch(grad_charges, dtype=wp_scalar),
            wp.from_torch(grad_dipoles, dtype=wp_vec),
            wp_dtype=wp_scalar,
            device=str(wp_device),
        )

        ctx.save_for_backward(
            grad_energies,
            positions,
            charges,
            dipoles,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        )
        return grad_positions, grad_charges, grad_dipoles

    @staticmethod
    def backward(  # pragma: no cover
        ctx,
        gg_positions: torch.Tensor,
        gg_charges: torch.Tensor,
        gg_dipoles: torch.Tensor,
    ):
        """Second-order backward via :func:`batch_multipole_real_space_dipole_csr_energy_2nd_backward`."""
        (
            grad_energies,
            positions,
            charges,
            dipoles,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        ) = ctx.saved_tensors
        device = positions.device
        wp_device = wp.device_from_torch(device)
        input_dtype = positions.dtype
        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        num_atoms = positions.shape[0]
        gg_grad_energies_2nd = torch.zeros_like(grad_energies)
        gg_positions_2nd = torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
        gg_charges_2nd = torch.zeros_like(charges)
        gg_dipoles_2nd = torch.zeros(
            (dipoles.shape[0], 3), dtype=input_dtype, device=device
        )

        batch_multipole_real_space_dipole_csr_energy_2nd_backward(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
            wp.from_torch(grad_energies.detach().contiguous(), dtype=wp.float64),
            wp.from_torch(gg_positions.contiguous(), dtype=wp_vec),
            wp.from_torch(gg_charges.contiguous(), dtype=wp_scalar),
            wp.from_torch(gg_dipoles.contiguous(), dtype=wp_vec),
            wp.from_torch(gg_grad_energies_2nd, dtype=wp.float64),
            wp.from_torch(gg_positions_2nd, dtype=wp_vec),
            wp.from_torch(gg_charges_2nd, dtype=wp_scalar),
            wp.from_torch(gg_dipoles_2nd, dtype=wp_vec),
            wp_dtype=wp_scalar,
            device=str(wp_device),
        )

        return (
            gg_grad_energies_2nd,
            gg_positions_2nd,
            gg_charges_2nd,
            gg_dipoles_2nd,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class BatchMultipoleRealSpaceDipoleFusedScalarFunction(torch.autograd.Function):
    r"""Batched analog of :class:`MultipoleRealSpaceDipoleFusedScalarFunction`.

    Returns a per-system scalar tensor :math:`(B,)`. Backward propagates
    per-system upstream weights (``grad_E[batch_idx[i]]``) over the precomputed
    per-atom gradient tensors for positions, charges, and dipoles.
    """

    @staticmethod
    def forward(
        ctx,
        positions: torch.Tensor,
        charges: torch.Tensor,
        dipoles: torch.Tensor,
        cells: torch.Tensor,
        sigmas: torch.Tensor,
        alphas: torch.Tensor,
        idx_j: torch.Tensor,
        neighbor_ptr: torch.Tensor,
        unit_shifts: torch.Tensor,
        batch_idx: torch.Tensor,
        half_neighbor_list: bool = False,
    ) -> torch.Tensor:
        """Run the batched lmax=1 fused Warp kernel; stash per-atom gradients in ctx."""
        device = positions.device
        wp_device = wp.device_from_torch(device)
        input_dtype = positions.dtype
        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        with_pos = bool(positions.requires_grad)
        with_q = bool(charges.requires_grad)
        with_mu = bool(dipoles.requires_grad)
        with_cell = bool(cells.requires_grad)
        num_atoms = positions.shape[0]
        num_systems = cells.shape[0]

        per_atom_energies = torch.zeros(num_atoms, dtype=torch.float64, device=device)
        grad_positions = (
            torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
            if with_pos
            else torch.zeros((1, 3), dtype=input_dtype, device=device)
        )
        grad_charges = (
            torch.zeros(num_atoms, dtype=input_dtype, device=device)
            if with_q
            else torch.zeros(1, dtype=input_dtype, device=device)
        )
        grad_dipoles = (
            torch.zeros((num_atoms, 3), dtype=input_dtype, device=device)
            if with_mu
            else torch.zeros((1, 3), dtype=input_dtype, device=device)
        )

        batch_multipole_real_space_dipole_csr_energy_fused(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
            wp.from_torch(per_atom_energies, dtype=wp.float64),
            wp.from_torch(grad_positions, dtype=wp_vec),
            wp.from_torch(grad_charges, dtype=wp_scalar),
            wp.from_torch(grad_dipoles, dtype=wp_vec),
            with_pos_grad=with_pos,
            with_charge_grad=with_q,
            with_dipole_grad=with_mu,
            wp_dtype=wp_scalar,
            device=str(wp_device),
        )

        per_system_energies = torch.zeros(
            num_systems, dtype=torch.float64, device=device
        )
        per_system_energies.scatter_add_(
            0, batch_idx.to(torch.int64), per_atom_energies
        )

        if with_cell:
            from nvalchemiops.interactions.electrostatics.multipole_ewald_cell_grad import (
                batch_multipole_real_space_dipole_csr_cell_grad,
            )

            grad_cell_buf = torch.zeros_like(cells.detach()).contiguous()
            batch_multipole_real_space_dipole_csr_cell_grad(
                wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
                wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
                wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
                wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
                wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
                wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
                wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
                wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
                wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
                wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
                wp.from_torch(grad_cell_buf, dtype=wp_mat),
                device=str(wp_device),
                half_neighbor_list=half_neighbor_list,
            )
            grad_cell_to_save = grad_cell_buf
        else:
            grad_cell_to_save = None

        # Save precomputed grads for the fast path and inputs for the
        # double-backward path. Plain forces broadcast the precomputed grads;
        # under ``create_graph`` pos/charge/dipole grads route through the
        # on-tape backward Function (2nd-order kernel) for force-loss training.
        ctx.save_for_backward(
            grad_positions if with_pos else None,
            grad_charges if with_q else None,
            grad_dipoles if with_mu else None,
            grad_cell_to_save,
            positions,
            charges,
            dipoles,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        )
        ctx.with_cell = with_cell
        return per_system_energies

    @staticmethod
    def backward(ctx, grad_E: torch.Tensor):  # pragma: no cover
        """Fast per-system precomputed-grad broadcast for plain forces; on-tape
        backward Function (double-backward capable) under ``create_graph``.
        ``grad_E`` is ``(B,)``; per-atom cotangent is ``grad_E[batch_idx[i]]``."""
        (
            pre_grad_positions,
            pre_grad_charges,
            pre_grad_dipoles,
            grad_cell,
            positions,
            charges,
            dipoles,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        ) = ctx.saved_tensors
        need_pqm = (
            ctx.needs_input_grad[0]
            or ctx.needs_input_grad[1]
            or ctx.needs_input_grad[2]
        )
        atom_weights = grad_E[batch_idx.to(torch.int64)]  # (N,)
        if need_pqm and torch.is_grad_enabled():
            out_grad_positions, out_grad_charges, out_grad_dipoles = (
                BatchMultipoleRealSpaceBackwardFunction.apply(
                    atom_weights.to(torch.float64),
                    positions,
                    charges,
                    dipoles,
                    cells,
                    sigmas,
                    alphas,
                    idx_j,
                    neighbor_ptr,
                    unit_shifts,
                    batch_idx,
                )
            )
        elif need_pqm:
            out_grad_positions = (
                atom_weights.unsqueeze(-1).to(pre_grad_positions.dtype)
                * pre_grad_positions
                if pre_grad_positions is not None
                else None
            )
            out_grad_charges = (
                atom_weights.to(pre_grad_charges.dtype) * pre_grad_charges
                if pre_grad_charges is not None
                else None
            )
            out_grad_dipoles = (
                atom_weights.unsqueeze(-1).to(pre_grad_dipoles.dtype) * pre_grad_dipoles
                if pre_grad_dipoles is not None
                else None
            )
        else:
            out_grad_positions = out_grad_charges = out_grad_dipoles = None
        if ctx.with_cell:
            out_grad_cells = grad_E.view(-1, 1, 1).to(grad_cell.dtype) * grad_cell
        else:
            out_grad_cells = None
        return (
            out_grad_positions,
            out_grad_charges,
            out_grad_dipoles,
            out_grad_cells,  # cells
            None,  # sigmas
            None,  # alphas
            None,  # idx_j
            None,  # neighbor_ptr
            None,  # unit_shifts
            None,  # batch_idx
            None,  # half_neighbor_list
        )


class BatchMultipoleRealSpaceFunction(torch.autograd.Function):
    r"""Batched l_max=1 GTO-Ewald multipole real-space (charges + dipoles)."""

    @staticmethod
    def forward(
        ctx,
        positions: torch.Tensor,
        charges: torch.Tensor,
        dipoles: torch.Tensor,
        cells: torch.Tensor,
        sigmas: torch.Tensor,
        alphas: torch.Tensor,
        idx_j: torch.Tensor,
        neighbor_ptr: torch.Tensor,
        unit_shifts: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Launch the batched l_max=1 forward Warp kernel."""
        device = positions.device
        wp_device = wp.device_from_torch(device)
        input_dtype = positions.dtype
        wp_scalar = get_wp_dtype(input_dtype)
        wp_vec = get_wp_vec_dtype(input_dtype)
        wp_mat = get_wp_mat_dtype(input_dtype)

        num_atoms = positions.shape[0]
        energies = torch.zeros(num_atoms, dtype=torch.float64, device=device)

        batch_multipole_real_space_dipole_csr_energy(
            wp.from_torch(positions.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
            wp.from_torch(cells.detach().contiguous(), dtype=wp_mat),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(neighbor_ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(unit_shifts.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigmas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(alphas.detach().contiguous(), dtype=wp_scalar),
            wp.from_torch(batch_idx.contiguous(), dtype=wp.int32),
            wp.from_torch(energies, dtype=wp.float64),
            wp_dtype=wp_scalar,
            device=str(wp_device),
        )

        ctx.save_for_backward(
            positions,
            charges,
            dipoles,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        )
        return energies

    @staticmethod
    def backward(ctx, grad_energies: torch.Tensor):  # pragma: no cover
        """Route through :class:`BatchMultipoleRealSpaceBackwardFunction`."""
        (
            positions,
            charges,
            dipoles,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        ) = ctx.saved_tensors
        gp, gc, gd = BatchMultipoleRealSpaceBackwardFunction.apply(
            grad_energies,
            positions,
            charges,
            dipoles,
            cells,
            sigmas,
            alphas,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
        )
        return (gp, gc, gd, None, None, None, None, None, None, None)


# ===========================================================================
# Composite: multipole_ewald_summation
# ===========================================================================


def _multipole_ewald_self_energy_per_atom(
    source_feats: torch.Tensor,
    sigma: float,
    alpha: float,
    quadrupoles: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Per-atom GTO-Ewald self-energy correction, shape :math:`(N,)`, float64.

    Subtracted from the reciprocal sum to remove the :math:`i=j`, :math:`n=0`
    image term. The caller can ``.sum()`` this for a single-system scalar or
    ``scatter_add`` into a :math:`(B,)` vector via ``batch_idx`` for the batched
    case — the formula is per-atom-local.

    .. math::

        E_\text{self}(l=0)_i &= \frac{F}{8\pi^{3/2}\,\sigma_c}\, q_i^2, \\
        E_\text{self}(l=1)_i &= \frac{F}{48\pi^{3/2}\,\sigma_c^3}\,
                                 |\boldsymbol{\mu}_i|^2, \\
        E_\text{self}(l=2)_i &= \frac{F}{320\pi^{3/2}\,\sigma_c^5}\,
                                 |Q_i|_F^2,

    with :math:`\sigma_c = \sqrt{\sigma^2 + 1/(4\alpha^2)}` and
    :math:`|Q|_F^2 = \sum_{\alpha\beta} Q_{\alpha\beta}^2` (Frobenius norm
    squared of the symmetric Cartesian quadrupole tensor, each off-diagonal
    entry counted twice).

    Parameters
    ----------
    quadrupoles : torch.Tensor, optional, shape (N, 3, 3)
        Cartesian symmetric quadrupole tensor. Adds the :math:`l=2` self-term
        when provided; ``None`` (default) is :math:`l_{max} \le 1` mode.
    """
    # Reuse the shared ``multipole_self_energy`` Warp op (the per-element
    # moment-square physics ``c0 q² + c1 |μ|² + c2 |Q|_F²``); the wrapper only
    # computes the GTO-Ewald overlap-constant coefficients and keeps the
    # per-atom output shape (the caller scatters / subtracts them).
    from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
        multipole_self_energy,
    )

    sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
    pi32 = math.pi**1.5
    device = source_feats.device

    def _coeff(value: float) -> torch.Tensor:
        return torch.tensor([value], dtype=torch.float64, device=device)

    c_self_q = _coeff(FIELD_CONSTANT / (8.0 * pi32 * sigma_c))
    c_self_mu = _coeff(FIELD_CONSTANT / (48.0 * pi32 * sigma_c**3))
    # l=2 Frobenius convention: |Q|_F² with the /320 normalization.
    c_self_q2 = _coeff(FIELD_CONSTANT / (320.0 * pi32 * sigma_c**5))

    charges = source_feats[..., 0].to(torch.float64)
    # |μ|² is invariant under the (y, z, x) ↔ (x, y, z) permutation, so the
    # e3nn dipole block feeds the Cartesian-dipole kernel slot unchanged.
    dipoles = (
        source_feats[..., 1:4].to(torch.float64)
        if source_feats.shape[-1] == 4
        else None
    )
    quads = quadrupoles.to(torch.float64) if quadrupoles is not None else None
    return multipole_self_energy(
        charges, dipoles, quads, c_self_q, c_self_mu, c_self_q2
    )


def multipole_ewald_summation(
    positions: torch.Tensor,
    multipole_moments: torch.Tensor,
    cell: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    *,
    sigma: float,
    alpha: float | None = None,
    k_cutoff: float | None = None,
    batch_idx: torch.Tensor | None = None,
    accuracy: float = 1e-6,
    cost_ratio: float = 1.0,
    half_neighbor_list: bool = False,
    cache: MultipoleSCFCache | None = None,
) -> torch.Tensor:
    r"""Full GTO-Ewald multipole electrostatic total energy.

    Composes the three canonical Ewald pieces:

    * **real-space**: GTO-Ewald-damped pair sum on a CSR neighbor list, via
      :func:`multipole_real_space_energy` or its batched analog, using
      :math:`T^{(0)}(r) = [\operatorname{erf}(r/(2\sigma)) -
      \operatorname{erf}(r/(2\sigma_c))] / r` with
      :math:`\sigma_c = \sqrt{\sigma^2 + 1/(4\alpha^2)}`.
    * **reciprocal-space**: GTO-smeared Fourier sum with Ewald damping
      :math:`\exp(-k^2/(4\alpha^2))/k^2`, via
      :func:`multipole_reciprocal_space_energy` or its batched analog
      (raw total, no self-subtract).
    * **self-energy correction**: the analytical per-atom term from
      :func:`_multipole_ewald_self_energy_per_atom`, subtracted to remove the
      :math:`i=j`, :math:`n=0` image that the reciprocal sum includes.

    The total is mathematically identical to
    :func:`multipole_electrostatic_energy` (direct k-space) for any
    :math:`(\sigma, \alpha)`.

    Single-system vs batched dispatch
    ---------------------------------
    Controlled by the optional ``batch_idx`` argument:

    * ``batch_idx=None`` (default) — single system. ``cell`` is :math:`(3, 3)`
      or :math:`(1, 3, 3)`; returns per-atom :math:`(N,)` float64.
    * ``batch_idx`` provided (shape :math:`(N_\text{total},)`) — B systems
      packed into flat per-atom tensors. ``cell`` must be :math:`(B, 3, 3)`;
      each atom's neighbors must live in the same system. Returns per-atom
      :math:`(N_\text{total},)` flat across systems.

    ``sigma`` and ``alpha`` are scalar floats in both modes (uniform across
    the batch).

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3) or (N_total, 3)
    multipole_moments : torch.Tensor, shape (N, (l_max+1)**2) matching positions
    cell : torch.Tensor, shape (3, 3), (1, 3, 3), or (B, 3, 3)
    idx_j, neighbor_ptr, unit_shifts : torch.Tensor
        Flat CSR neighbor list (``int32`` / ``int32`` / ``vec3i``).
    sigma : float
        GTO density-basis width.
    alpha : float, optional
        Ewald splitting parameter (positive). When ``None`` (default) it is
        auto-estimated from ``sigma`` and the system geometry via
        :func:`estimate_multipole_ewald_parameters` at the requested
        ``accuracy``. The caller must build the neighbor list with the matching
        real-space cutoff.
    k_cutoff : float, optional
        Maximum :math:`|k|` for the reciprocal sum. Auto-estimated from the
        Kolafa-Perram balance when ``None`` (single mode); batched mode always
        builds its own k-grid per system. Unnecessary when a ``cache`` is
        supplied (the cache already encodes the k-grid).
    batch_idx : torch.Tensor, optional
        :math:`(N_\text{total},)` int32. ``None`` selects single-system mode.
    accuracy : float, default 1e-6
        Target relative-energy accuracy used by the auto-estimator when
        ``alpha`` and/or ``k_cutoff`` are ``None``. Ignored if both are
        supplied.
    cost_ratio : float, default 1.0
        Hardware-empirical :math:`C_r / C_k` (per-real-space-pair vs per-k cost
        ratio) passed to :func:`estimate_multipole_ewald_parameters`. ``1.0``
        reproduces canonical Kolafa-Perram; higher values shift the optimum
        toward smaller real-space cutoff and larger reciprocal cutoff. Ignored
        if ``alpha`` and ``k_cutoff`` are supplied.
    half_neighbor_list : bool, default False
        Forwarded to the real-space pair sum. Set when the CSR neighbor list
        stores each pair once (only affects the :math:`l_{max}=2` path's
        cell-gradient kernel; no effect otherwise).
    cache : MultipoleSCFCache, optional
        Pre-built reciprocal cache (from :func:`prepare_multipole_scf_cache`).
        When given, the per-call reciprocal-cache rebuild (k-grid + GTO-Fourier
        ``phi_hat`` + per-k-factor tables) is skipped — the steady-state /
        MD path, analogous to passing precomputed ``k_squared`` to
        :func:`multipole_particle_mesh_ewald`. ``alpha`` is taken from the
        cache when not supplied, and ``k_cutoff`` becomes
        unnecessary. **Stress caveat:** a pre-built cache holds a fixed
        (detached) k-grid/volume, so ``grad(E, cell)`` does not flow through
        the reciprocal term — pass ``cache=None`` for stress / cell-gradient
        training. Forces (``grad(E, positions)``) are unaffected.

    Returns
    -------
    torch.Tensor
        Per-atom :math:`(N,)` :math:`\text{float64}` (single) or
        :math:`(N_\text{total},)` (batched, flat across systems) on
        ``positions.device``. Call ``.sum()`` for the total energy or
        ``torch.zeros(B).scatter_add(0, batch_idx, E)`` for per-system totals;
        forces/stress/charge-grads flow from ``grad(E.sum(), ...)``.
        Autograd-connected to ``positions`` and ``multipole_moments``.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    # A prebuilt reciprocal cache already encodes alpha + the k-grid, so the
    # estimator is unnecessary: take alpha from the cache when not given and
    # skip the Kolafa-Perram balance entirely (the cache supplies k_cutoff
    # implicitly). See the cache= note in the docstring for the stress caveat.
    if cache is not None and alpha is None:
        alpha = float(cache.alpha)

    # Auto-estimate alpha and/or k_cutoff from the Kolafa-Perram
    # balance if either is missing. We only run the estimator once per
    # call (single-system mode); batched callers that want per-system
    # parameters should pass them in explicitly.
    if cache is None and (alpha is None or k_cutoff is None):
        from nvalchemiops.torch.interactions.electrostatics.parameters import (
            estimate_multipole_ewald_parameters,
        )

        params = estimate_multipole_ewald_parameters(
            positions,
            cell,
            sigma=sigma,
            batch_idx=batch_idx,
            accuracy=accuracy,
            cost_ratio=cost_ratio,
        )
        if alpha is None:
            # The kernels take a single scalar alpha, so a batched estimate
            # must agree across systems.
            alpha_tensor = params.alpha
            if alpha_tensor.numel() == 1:
                alpha = float(alpha_tensor.item())
            else:
                alpha_min = float(alpha_tensor.min().item())
                alpha_max = float(alpha_tensor.max().item())
                if alpha_max - alpha_min > 1e-12 * max(alpha_max, 1.0):
                    raise ValueError(
                        "Auto-estimated alpha differs across batch systems "
                        f"({alpha_min} vs {alpha_max}). The current Ewald "
                        "kernel takes a single scalar alpha, so heterogeneous "
                        "batches must pre-compute per-system params and run "
                        "each system separately."
                    )
                alpha = alpha_min
        if k_cutoff is None:
            kcut_tensor = params.reciprocal_space_cutoff
            k_cutoff = float(kcut_tensor.max().item())

    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    is_batch = batch_idx is not None
    device = positions.device
    input_dtype = positions.dtype

    # Split packed e3nn moments into the Cartesian channels the kernels consume.
    # ``source_feats`` is the l<=1 e3nn block; ``quadrupoles`` is the Cartesian
    # traceless (N, 3, 3) l=2 channel (None for l<=1).
    charges, dipoles_cart, quadrupoles, l_max = split_multipole_moments(
        multipole_moments
    )
    source_feats = (
        multipole_moments[:, :4].contiguous()
        if l_max >= 1
        else multipole_moments[:, :1].contiguous()
    )

    # l_max=2 path: full Ewald total = real-space + direct-k reciprocal − self.
    if l_max == 2:
        from nvalchemiops.torch.interactions.electrostatics.multipole_electrostatics import (
            multipole_reciprocal_space_energy,
        )
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
            multipole_real_space_quadrupole_energy,
        )

        charges_l2, dipoles_cart_l2 = charges, dipoles_cart
        coulomb_scale = FIELD_CONSTANT / (4.0 * math.pi)

        if batch_idx is not None:
            if cell.ndim != 3 or cell.shape[-2:] != (3, 3):
                raise ValueError(
                    f"batched mode requires cell shape (B, 3, 3); got {tuple(cell.shape)}"
                )
            if cache is None and k_cutoff is None:
                raise ValueError(
                    "batched mode requires k_cutoff (batched reciprocal "
                    "builds its own k-grid per system) unless a prebuilt "
                    "cache= is supplied."
                )
            B = cell.shape[0]
            sigmas = torch.full((B,), sigma, dtype=input_dtype, device=device)
            alphas = torch.full((B,), alpha, dtype=input_dtype, device=device)
            # Per-atom (N_total,) for all three terms; caller owns the
            # reduction (PR #96 energy-autograd contract).
            e_real_atom = coulomb_scale * multipole_real_space_quadrupole_energy(
                positions,
                charges_l2,
                dipoles_cart_l2,
                quadrupoles,
                cell,
                idx_j,
                neighbor_ptr,
                unit_shifts,
                sigmas,
                alphas,
                batch_idx=batch_idx,
            )
            e_recip_atom = multipole_reciprocal_space_energy(
                positions,
                multipole_moments,
                cell,
                batch_idx=batch_idx,
                sigma=sigma,
                alpha=alpha,
                k_cutoff=k_cutoff,
                cache=cache,
            )
            atom_self = _multipole_ewald_self_energy_per_atom(
                source_feats,
                sigma,
                alpha,
                quadrupoles=quadrupoles,
            )
            return (e_real_atom + e_recip_atom - atom_self).to(torch.float64)

        if cell.ndim == 3:
            cell_l2 = cell
        else:
            cell_l2 = cell.unsqueeze(0)
        sigma_t = torch.tensor([sigma], dtype=input_dtype, device=device)
        alpha_t = torch.tensor([alpha], dtype=input_dtype, device=device)

        e_real = coulomb_scale * multipole_real_space_quadrupole_energy(
            positions,
            charges_l2,
            dipoles_cart_l2,
            quadrupoles,
            cell_l2,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma_t,
            alpha_t,
        )
        e_recip = multipole_reciprocal_space_energy(
            positions,
            multipole_moments,
            cell.reshape(3, 3) if cell.ndim == 3 else cell,
            sigma=sigma,
            alpha=alpha,
            k_cutoff=k_cutoff,
            cache=cache,
        )
        e_self = _multipole_ewald_self_energy_per_atom(
            source_feats,
            sigma,
            alpha,
            quadrupoles=quadrupoles,
        )
        # Per-atom (N,); caller owns the reduction.
        return (e_real + e_recip - e_self).to(torch.float64)

    # Delayed imports avoid a circular module graph.
    from nvalchemiops.torch.interactions.electrostatics.multipole_electrostatics import (
        multipole_reciprocal_space_energy,
    )

    # The real-space kernel returns T_0-based values without the F/(4π) Coulomb
    # prefactor that the reciprocal path bakes into its F/k² convention, so
    # scale the real-space energy by F/(4π) to align the two.
    coulomb_scale = FIELD_CONSTANT / (4.0 * math.pi)

    if is_batch:
        if cell.ndim != 3 or cell.shape[-2:] != (3, 3):
            raise ValueError(
                f"batched mode requires cell shape (B, 3, 3); got {tuple(cell.shape)}"
            )
        B = cell.shape[0]
        sigmas = torch.full((B,), sigma, dtype=input_dtype, device=device)
        alphas = torch.full((B,), alpha, dtype=input_dtype, device=device)

        if cache is None and k_cutoff is None:
            raise ValueError(
                "batched mode requires k_cutoff (batched reciprocal "
                "builds its own k-grid per system) unless a prebuilt cache= "
                "is supplied."
            )
        # Per-atom (N_total,) for all three terms; caller owns the reduction.
        e_real = coulomb_scale * multipole_real_space_energy_with_stress(
            positions,
            multipole_moments,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigmas,
            alphas,
            batch_idx=batch_idx,
        )
        e_recip = multipole_reciprocal_space_energy(
            positions,
            multipole_moments,
            cell,
            batch_idx=batch_idx,
            sigma=sigma,
            alpha=alpha,
            k_cutoff=k_cutoff,
            cache=cache,
        )
        atom_self = _multipole_ewald_self_energy_per_atom(source_feats, sigma, alpha)
        return (e_real + e_recip - atom_self).to(torch.float64)

    sigma_t = torch.tensor([sigma], dtype=input_dtype, device=device)
    alpha_t = torch.tensor([alpha], dtype=input_dtype, device=device)

    # Per-atom (N,) for all three terms; caller owns the reduction.
    e_real = coulomb_scale * multipole_real_space_energy_with_stress(
        positions,
        multipole_moments,
        cell,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        sigma_t,
        alpha_t,
    )
    e_recip = multipole_reciprocal_space_energy(
        positions,
        multipole_moments,
        cell,
        sigma=sigma,
        alpha=alpha,
        k_cutoff=k_cutoff,
        cache=cache,
    )
    e_self = _multipole_ewald_self_energy_per_atom(source_feats, sigma, alpha)
    return (e_real + e_recip - e_self).to(torch.float64)
