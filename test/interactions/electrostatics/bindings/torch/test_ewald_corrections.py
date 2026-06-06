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

"""Tests for the registered Ewald reciprocal correction chain."""

from __future__ import annotations

import math

import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics._ewald_corrections_chain import (
    ewald_energy_corrections,
    ewald_energy_corrections_batch,
)


def _requires_warp_device(device: str) -> torch.device:
    """Return a torch device, skipping unavailable Warp-backed targets."""
    if device == "cpu":
        pytest.skip("Ewald correction custom-op coverage is CUDA-focused")
    return torch.device(device)


def _single_reference(
    raw: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
) -> torch.Tensor:
    """Torch reference for single-system Ewald reciprocal corrections."""
    self_e = alpha[0] * charges.square() / math.sqrt(math.pi)
    bg = math.pi * charges * total_charge[0] / (2.0 * alpha[0].square() * volume[0])
    return raw - self_e - bg


def _batch_reference(
    raw: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    volumes: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
) -> torch.Tensor:
    """Torch reference for batched Ewald reciprocal corrections."""
    atom_alpha = alpha.index_select(0, batch_idx)
    atom_volume = volumes.index_select(0, batch_idx)
    atom_qtot = total_charges.index_select(0, batch_idx)
    self_e = atom_alpha * charges.square() / math.sqrt(math.pi)
    bg = math.pi * charges * atom_qtot / (2.0 * atom_alpha.square() * atom_volume)
    return raw - self_e - bg


class TestEwaldEnergyCorrections:
    """Validate forward, backward, and double-backward correction formulas."""

    def test_single_forward_matches_torch(self, device):
        """Single-system correction forward matches the Torch formula."""
        torch_device = _requires_warp_device(device)
        raw = torch.tensor([0.3, -0.2, 0.7], device=torch_device, dtype=torch.float64)
        charges = torch.tensor(
            [0.6, -0.4, 0.2], device=torch_device, dtype=torch.float64
        )
        volume = torch.tensor([31.0], device=torch_device, dtype=torch.float64)
        alpha = torch.tensor([0.35], device=torch_device, dtype=torch.float64)
        total_charge = charges.sum().reshape(1)

        actual = ewald_energy_corrections(raw, charges, volume, alpha, total_charge)
        expected = _single_reference(raw, charges, volume, alpha, total_charge)

        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_batch_forward_matches_torch(self, device):
        """Batched correction forward matches the Torch formula."""
        torch_device = _requires_warp_device(device)
        raw = torch.tensor(
            [0.3, -0.2, 0.7, 0.1, -0.4],
            device=torch_device,
            dtype=torch.float64,
        )
        charges = torch.tensor(
            [0.6, -0.4, 0.2, 0.5, -0.1],
            device=torch_device,
            dtype=torch.float64,
        )
        batch_idx = torch.tensor(
            [0, 0, 0, 1, 1], device=torch_device, dtype=torch.int32
        )
        volumes = torch.tensor([31.0, 29.0], device=torch_device, dtype=torch.float64)
        alpha = torch.tensor([0.35, 0.41], device=torch_device, dtype=torch.float64)
        total_charges = torch.zeros(2, device=torch_device, dtype=torch.float64)
        total_charges = total_charges.index_add(0, batch_idx.to(torch.long), charges)

        actual = ewald_energy_corrections_batch(
            raw,
            charges,
            batch_idx,
            volumes,
            alpha,
            total_charges,
        )
        expected = _batch_reference(
            raw, charges, batch_idx.to(torch.long), volumes, alpha, total_charges
        )

        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_single_gradcheck(self, device):
        """Single-system correction op passes gradcheck."""
        torch_device = _requires_warp_device(device)
        raw = torch.randn(
            4, device=torch_device, dtype=torch.float64, requires_grad=True
        )
        charges = torch.randn(
            4,
            device=torch_device,
            dtype=torch.float64,
            requires_grad=True,
        )
        volume = torch.tensor(
            [27.0],
            device=torch_device,
            dtype=torch.float64,
            requires_grad=True,
        )
        alpha = torch.tensor(
            [0.37],
            device=torch_device,
            dtype=torch.float64,
            requires_grad=True,
        )
        total_charge = charges.detach().sum().reshape(1).requires_grad_(True)

        assert torch.autograd.gradcheck(
            lambda *args: ewald_energy_corrections(*args).sum(),
            (raw, charges, volume, alpha, total_charge),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_single_gradgradcheck(self, device):
        """Single-system correction op passes gradgradcheck."""
        torch_device = _requires_warp_device(device)
        raw = torch.randn(
            4, device=torch_device, dtype=torch.float64, requires_grad=True
        )
        charges = torch.randn(
            4,
            device=torch_device,
            dtype=torch.float64,
            requires_grad=True,
        )
        volume = torch.tensor(
            [27.0],
            device=torch_device,
            dtype=torch.float64,
            requires_grad=True,
        )
        alpha = torch.tensor(
            [0.37],
            device=torch_device,
            dtype=torch.float64,
            requires_grad=True,
        )
        total_charge = charges.detach().sum().reshape(1).requires_grad_(True)

        assert torch.autograd.gradgradcheck(
            lambda *args: ewald_energy_corrections(*args).sum(),
            (raw, charges, volume, alpha, total_charge),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-5,
        )
