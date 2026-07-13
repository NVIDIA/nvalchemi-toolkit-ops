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
_InjectCachedEvalGradWithFallback = _util._InjectCachedEvalGradWithFallback
_is_uniform_cotangent = _util._is_uniform_cotangent

from test.interactions.electrostatics._deriv_check import (  # noqa: E402
    finite_difference_jacobian,
)

DT = torch.float64


def _expected_charge_grad(grad_energy, charge_grad, batch_idx):
    """Reference charge-gradient injector backward math."""
    if batch_idx is not None:
        atom_grad = grad_energy.index_select(0, batch_idx)
    else:
        atom_grad = grad_energy.squeeze(0)
    return charge_grad * atom_grad


def test_charge_grad_single_system_bit_identical():
    """Single-system per-system cotangent matches the injector charge path."""
    energy = torch.tensor([3.0], dtype=DT)
    charges = torch.tensor([1.0, -1.0, 0.5], dtype=DT, requires_grad=True)
    charge_grad = torch.tensor([0.2, -0.3, 0.1], dtype=DT)

    out = _InjectChargeGrad.apply(energy, charges, charge_grad, None)
    assert torch.equal(out, energy)
    grad_energy = torch.tensor([1.7], dtype=DT)
    out.backward(grad_energy)

    expected = _expected_charge_grad(grad_energy, charge_grad, None)
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

    expected = _expected_charge_grad(grad_energy, charge_grad, batch_idx)
    assert torch.equal(charges.grad, expected)


def test_charge_grad_single_system_per_atom_cotangent_uses_mean():
    """Non-uniform per-atom cotangents pass through to the energy graph."""
    energy = torch.arange(3, dtype=DT, requires_grad=True)
    charges = torch.tensor([1.0, -1.0, 0.5], dtype=DT, requires_grad=True)
    charge_grad = torch.tensor([0.2, -0.3, 0.1], dtype=DT)

    out = _InjectChargeGrad.apply(energy, charges, charge_grad, None)
    grad_energy = torch.tensor([2.0, 4.0, 9.0], dtype=DT)
    out.backward(grad_energy)

    assert charges.grad is None
    assert torch.equal(energy.grad, grad_energy)


def test_charge_grad_batched_per_atom_cotangent_uses_system_mean():
    """Batched non-uniform per-atom cotangents use the energy graph."""
    energy = torch.arange(4, dtype=DT, requires_grad=True)
    charges = torch.tensor([1.0, -1.0, 0.5, 2.0], dtype=DT, requires_grad=True)
    charge_grad = torch.tensor([0.2, -0.3, 0.1, 0.4], dtype=DT)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

    out = _InjectChargeGrad.apply(energy, charges, charge_grad, batch_idx)
    grad_energy = torch.tensor([2.0, 4.0, 5.0, 7.0], dtype=DT)
    out.backward(grad_energy)

    assert charges.grad is None
    assert torch.equal(energy.grad, grad_energy)


def test_cached_eval_qR_nonuniform_fallback_uses_partial_derivatives():
    """q(R) weighted fallback returns partial dE/dR plus dE/dq for one chain term."""
    positions = torch.randn(3, 3, dtype=DT, requires_grad=True)
    theta = torch.randn(3, dtype=DT, requires_grad=True)
    charges = positions[:, 0].square() + theta
    cell = torch.eye(3, dtype=DT).unsqueeze(0).requires_grad_()
    energy = positions[:, 1].detach().clone().requires_grad_()
    pos_grad_state = torch.zeros_like(positions)
    charge_grad_state = torch.zeros_like(charges)
    cell_grad_state = torch.zeros_like(cell)
    grad_energy = torch.tensor([1.0, 2.0, 4.0], dtype=DT)

    def fallback_fn(p, q, _cell):
        return p[:, 1] * q + 0.5 * p[:, 2].square()

    out = _InjectCachedEvalGradWithFallback.apply(
        energy,
        positions,
        charges,
        cell,
        pos_grad_state,
        charge_grad_state,
        cell_grad_state,
        None,
        fallback_fn,
    )

    out.backward(grad_energy)

    expected_positions_grad = torch.stack(
        (
            grad_energy * 2.0 * positions.detach()[:, 0] * positions.detach()[:, 1],
            grad_energy * charges.detach(),
            grad_energy * positions.detach()[:, 2],
        ),
        dim=1,
    )
    torch.testing.assert_close(positions.grad, expected_positions_grad)

    expected_theta_grad = grad_energy * positions.detach()[:, 1]
    torch.testing.assert_close(theta.grad, expected_theta_grad)


def test_cached_eval_qR_create_graph_second_order():
    """create_graph q(R) fallback differentiates connected position gradients once."""
    positions = torch.randn(3, 3, dtype=DT, requires_grad=True)
    theta = torch.randn(3, dtype=DT, requires_grad=True)
    charges = positions[:, 0].square() + theta
    cell = torch.eye(3, dtype=DT).unsqueeze(0)
    energy = positions[:, 1].detach().clone()
    pos_grad_state = torch.zeros_like(positions)
    charge_grad_state = torch.zeros_like(charges)
    cell_grad_state = torch.zeros_like(cell)

    def fallback_fn(p, q, _cell):
        return p[:, 1] * q + 0.5 * p[:, 2].square()

    out = _InjectCachedEvalGradWithFallback.apply(
        energy,
        positions,
        charges,
        cell,
        pos_grad_state,
        charge_grad_state,
        cell_grad_state,
        None,
        fallback_fn,
    )

    grad_pos = torch.autograd.grad(out.sum(), positions, create_graph=True)[0]
    loss = grad_pos.pow(2).sum()
    (grad_theta_ad,) = torch.autograd.grad(loss, theta)

    def loss_of_theta(theta_in: torch.Tensor) -> torch.Tensor:
        p = positions.detach().clone().requires_grad_(True)
        q = p[:, 0].square() + theta_in
        e = fallback_fn(p, q, cell)
        g = torch.autograd.grad(e.sum(), p, create_graph=True)[0]
        return g.pow(2).sum()

    eps = 1e-6
    grad_theta_fd = finite_difference_jacobian(loss_of_theta, theta.detach(), eps=eps)

    torch.testing.assert_close(
        grad_theta_ad,
        grad_theta_fd,
        rtol=1e-5,
        atol=1e-7,
    )


def _available_devices():
    """Devices available for cotangent predicate tests."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", _available_devices())
def test_uniform_cotangent_accepts_expanded_scalar(device):
    """A ``sum``-style expanded scalar cotangent is uniform without a sync."""
    grad = torch.ones((), dtype=DT, device=device).expand(6)

    assert _is_uniform_cotangent(grad)


@pytest.mark.parametrize("device", _available_devices())
def test_uniform_cotangent_keeps_cuda_contiguous_constants_conservative(device):
    """Contiguous CUDA constants require value inspection, so stay on fallback."""
    grad = torch.ones(6, dtype=DT, device=device)

    expected = device == "cpu"
    assert _is_uniform_cotangent(grad) is expected


@pytest.mark.parametrize("device", _available_devices())
def test_ewald_uniform_predicates_accept_expanded_scalar(device):
    """Ewald real/reciprocal chains consume CUDA ``sum`` cotangents."""
    real_chain = pytest.importorskip(
        "nvalchemiops.torch.interactions.electrostatics._ewald_real_chain"
    )
    recip_chain = pytest.importorskip(
        "nvalchemiops.torch.interactions.electrostatics._ewald_recip_chain"
    )
    grad = torch.ones((), dtype=DT, device=device).expand(4)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

    assert real_chain._cotangent_per_system_uniform(grad, batch_idx, 2)
    assert recip_chain._cotangent_per_system_uniform(grad, batch_idx, 2)


@pytest.mark.parametrize("device", _available_devices())
def test_ewald_uniform_predicates_keep_cuda_per_system_constants_conservative(device):
    """Non-expanded CUDA constants stay exact by using the weighted fallback."""
    real_chain = pytest.importorskip(
        "nvalchemiops.torch.interactions.electrostatics._ewald_real_chain"
    )
    recip_chain = pytest.importorskip(
        "nvalchemiops.torch.interactions.electrostatics._ewald_recip_chain"
    )
    grad = torch.tensor([2.0, 2.0, 3.0, 3.0], dtype=DT, device=device)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

    expected = device == "cpu"
    assert real_chain._cotangent_per_system_uniform(grad, batch_idx, 2) is expected
    assert recip_chain._cotangent_per_system_uniform(grad, batch_idx, 2) is expected
