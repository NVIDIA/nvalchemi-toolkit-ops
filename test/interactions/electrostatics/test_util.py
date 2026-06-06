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

"""Tests for private electrostatics Torch utility autograd shims."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
_util = pytest.importorskip("nvalchemiops.torch.interactions.electrostatics._util")
_InjectChargeGrad = _util._InjectChargeGrad

DT = torch.float64


def _legacy_charge_grad(grad_energy, charge_grad, batch_idx):
    """Reference: the historical charge-gradient injector backward math."""
    if batch_idx is not None:
        atom_grad = grad_energy.index_select(0, batch_idx)
    else:
        atom_grad = grad_energy.squeeze(0)
    return charge_grad * atom_grad


def test_charge_grad_single_system_bit_identical():
    """Single-system per-system cotangent matches the legacy charge path."""
    energy = torch.tensor([3.0], dtype=DT)
    charges = torch.tensor([1.0, -1.0, 0.5], dtype=DT, requires_grad=True)
    charge_grad = torch.tensor([0.2, -0.3, 0.1], dtype=DT)

    out = _InjectChargeGrad.apply(energy, charges, charge_grad, None)
    assert torch.equal(out, energy)
    grad_energy = torch.tensor([1.7], dtype=DT)
    out.backward(grad_energy)

    expected = _legacy_charge_grad(grad_energy, charge_grad, None)
    assert torch.equal(charges.grad, expected)


def test_charge_grad_batched_bit_identical():
    """Batched per-system cotangents are selected by ``batch_idx``."""
    energy = torch.tensor([3.0, 1.5], dtype=DT)
    charges = torch.tensor([1.0, -1.0, 0.5, 2.0], dtype=DT, requires_grad=True)
    charge_grad = torch.tensor([0.2, -0.3, 0.1, 0.4], dtype=DT)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

    out = _InjectChargeGrad.apply(energy, charges, charge_grad, batch_idx)
    grad_energy = torch.tensor([2.0, 5.0], dtype=DT)
    out.backward(grad_energy)

    expected = _legacy_charge_grad(grad_energy, charge_grad, batch_idx)
    assert torch.equal(charges.grad, expected)


def test_charge_grad_single_system_per_atom_cotangent_uses_mean():
    """Public per-atom energy cotangents reduce to one system scalar."""
    energy = torch.arange(3, dtype=DT)
    charges = torch.tensor([1.0, -1.0, 0.5], dtype=DT, requires_grad=True)
    charge_grad = torch.tensor([0.2, -0.3, 0.1], dtype=DT)

    out = _InjectChargeGrad.apply(energy, charges, charge_grad, None)
    grad_energy = torch.tensor([2.0, 4.0, 9.0], dtype=DT)
    out.backward(grad_energy)

    expected = charge_grad * grad_energy.mean()
    assert torch.equal(charges.grad, expected)


def test_charge_grad_batched_per_atom_cotangent_uses_system_mean():
    """Batched public per-atom cotangents reduce per system before injection."""
    energy = torch.arange(4, dtype=DT)
    charges = torch.tensor([1.0, -1.0, 0.5, 2.0], dtype=DT, requires_grad=True)
    charge_grad = torch.tensor([0.2, -0.3, 0.1, 0.4], dtype=DT)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

    out = _InjectChargeGrad.apply(energy, charges, charge_grad, batch_idx)
    grad_energy = torch.tensor([2.0, 4.0, 5.0, 7.0], dtype=DT)
    out.backward(grad_energy)

    atom_grad = torch.tensor([3.0, 3.0, 6.0, 6.0], dtype=DT)
    expected = charge_grad * atom_grad
    assert torch.equal(charges.grad, expected)
