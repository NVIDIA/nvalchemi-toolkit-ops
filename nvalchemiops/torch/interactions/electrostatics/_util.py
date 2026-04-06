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

import torch

__all__ = ["_InjectChargeGrad"]


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
