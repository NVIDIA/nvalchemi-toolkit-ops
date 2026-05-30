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

"""
Unit tests for pairwise dispersion (r^-6) calculations - PyTorch bindings.

Mirrors the structure of the Coulomb tests. Covers:
- Analytic two-atom plain -C6/r^6 energy and forces
- beta-damped real-space term vs closed form
- Newton's third law
- Energy<->force consistency (autograd and finite difference)
- Neighbor list vs neighbor matrix parity
- Batched vs looped-single parity
- float32 vs float64 agreement
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.dispersion import (
    lj_dispersion_energy,
    lj_dispersion_energy_forces,
    lj_dispersion_forces,
    sigma_epsilon_to_dispersion_charge,
)
from nvalchemiops.torch.neighbors import neighbor_list as neighbor_list_fn


@pytest.fixture(scope="module")
def device():
    """Available compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _c6(sigma: float, epsilon: float) -> float:
    """C6 = 4 * epsilon * sigma**6 for a single species."""
    return 4.0 * epsilon * sigma**6


def _screen(beta: float, r: float) -> float:
    """S(r) = (1 + x^2 + 0.5 x^4) exp(-x^2), x = beta r."""
    x2 = (beta * r) ** 2
    return (1.0 + x2 + 0.5 * x2 * x2) * math.exp(-x2)


def _two_atom(device, r=3.0, sigma=1.0, epsilon=1.0):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64, device=device
    )
    sig = torch.tensor([sigma, sigma], dtype=torch.float64, device=device)
    eps = torch.tensor([epsilon, epsilon], dtype=torch.float64, device=device)
    cell = torch.tensor(
        [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
        dtype=torch.float64,
        device=device,
    )
    # Symmetric neighbor list -> correct total energy/forces.
    nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
    nptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    nsh = torch.zeros((2, 3), dtype=torch.int32, device=device)
    return positions, sig, eps, cell, nl, nptr, nsh


def _random_system(device, n=12, box=9.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    positions = (torch.rand(n, 3, generator=g, dtype=torch.float64) * box).to(device)
    sigma = (0.8 + 0.6 * torch.rand(n, generator=g, dtype=torch.float64)).to(device)
    epsilon = (0.5 + 0.5 * torch.rand(n, generator=g, dtype=torch.float64)).to(device)
    cell = (torch.eye(3, dtype=torch.float64) * box)[None].to(device)
    return positions, sigma, epsilon, cell


class TestConversion:
    def test_sigma_epsilon_to_b(self, device):
        sigma = torch.tensor([1.0, 2.0], dtype=torch.float64, device=device)
        eps = torch.tensor([1.0, 0.25], dtype=torch.float64, device=device)
        b = sigma_epsilon_to_dispersion_charge(sigma, eps)
        # b = 2 sqrt(eps) sigma^3, and b^2 = C6 = 4 eps sigma^6
        expected = torch.tensor(
            [2.0 * 1.0 * 1.0, 2.0 * 0.5 * 8.0], dtype=torch.float64, device=device
        )
        assert torch.allclose(b, expected, rtol=1e-12)
        assert torch.allclose(b**2, 4.0 * eps * sigma**6, rtol=1e-12)


class TestPlainEnergy:
    def test_two_atom_energy(self, device):
        positions, sig, eps, cell, nl, nptr, nsh = _two_atom(device)
        E = lj_dispersion_energy(
            positions,
            sig,
            eps,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
        )
        expected = -_c6(1.0, 1.0) / 3.0**6
        assert torch.allclose(
            E.sum(),
            torch.tensor(expected, dtype=torch.float64, device=device),
            rtol=1e-10,
        )

    def test_attractive(self, device):
        positions, sig, eps, cell, nl, nptr, nsh = _two_atom(device)
        E = lj_dispersion_energy(
            positions,
            sig,
            eps,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
        )
        assert E.sum().item() < 0.0

    def test_r6_scaling(self, device):
        # Halving r increases |E| by 2^6 = 64.
        e_far = lj_dispersion_energy(
            *_two_atom(device, r=4.0)[:4],
            cutoff=20.0,
            alpha=0.0,
            neighbor_list=_two_atom(device, r=4.0)[4],
            neighbor_ptr=_two_atom(device, r=4.0)[5],
            neighbor_shifts=_two_atom(device, r=4.0)[6],
        ).sum()
        e_near = lj_dispersion_energy(
            *_two_atom(device, r=2.0)[:4],
            cutoff=20.0,
            alpha=0.0,
            neighbor_list=_two_atom(device, r=2.0)[4],
            neighbor_ptr=_two_atom(device, r=2.0)[5],
            neighbor_shifts=_two_atom(device, r=2.0)[6],
        ).sum()
        assert torch.allclose(e_near, 64.0 * e_far, rtol=1e-10)


class TestDampedEnergy:
    @pytest.mark.parametrize("beta", [0.2, 0.5, 1.0])
    def test_damped_vs_closed_form(self, device, beta):
        positions, sig, eps, cell, nl, nptr, nsh = _two_atom(device)
        E = lj_dispersion_energy(
            positions,
            sig,
            eps,
            cell,
            cutoff=10.0,
            alpha=beta,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
        )
        expected = -_c6(1.0, 1.0) * _screen(beta, 3.0) / 3.0**6
        assert torch.allclose(
            E.sum(),
            torch.tensor(expected, dtype=torch.float64, device=device),
            rtol=1e-10,
        )

    def test_beta_zero_matches_plain(self, device):
        positions, sig, eps, cell, nl, nptr, nsh = _two_atom(device)
        kw = dict(neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=nsh)
        e0 = lj_dispersion_energy(positions, sig, eps, cell, 10.0, 0.0, **kw).sum()
        eb = lj_dispersion_energy(positions, sig, eps, cell, 10.0, 1e-8, **kw).sum()
        assert torch.allclose(e0, eb, rtol=1e-6, atol=1e-12)


class TestForces:
    def test_plain_force_analytic(self, device):
        positions, sig, eps, cell, nl, nptr, nsh = _two_atom(device)
        _, F = lj_dispersion_energy_forces(
            positions,
            sig,
            eps,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
        )
        C6 = _c6(1.0, 1.0)
        r = 3.0
        # atom0: r_ij = r0 - r1 = (-3,0,0); F = -6 C6/r^8 * r_ij
        f0 = -6.0 * C6 / r**8 * np.array([-r, 0.0, 0.0])
        assert np.allclose(F[0].cpu().numpy(), f0, rtol=1e-9, atol=1e-12)
        # Newton's third law
        assert np.allclose(F.sum(0).cpu().numpy(), [0, 0, 0], atol=1e-12)

    @pytest.mark.parametrize("beta", [0.0, 0.4])
    def test_force_matches_autograd(self, device, beta):
        positions, sig, eps, cell = _random_system(device)
        nl, nptr, nsh = neighbor_list_fn(
            positions,
            cutoff=4.0,
            cell=cell,
            pbc=torch.tensor([True, True, True], device=device),
            return_neighbor_list=True,
        )
        pos_ag = positions.clone().requires_grad_(True)
        E, F = lj_dispersion_energy_forces(
            pos_ag,
            sig,
            eps,
            cell,
            cutoff=4.0,
            alpha=beta,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
        )
        E.sum().backward()
        F_ag = -pos_ag.grad
        assert torch.allclose(F, F_ag, rtol=1e-7, atol=1e-9)

    @pytest.mark.parametrize("beta", [0.0, 0.4])
    def test_force_finite_difference(self, device, beta):
        positions, sig, eps, cell = _random_system(device, n=6, box=8.0, seed=3)
        cutoff = 4.0
        pbc = torch.tensor([True, True, True], device=device)
        nl, nptr, nsh = neighbor_list_fn(
            positions, cutoff=cutoff, cell=cell, pbc=pbc, return_neighbor_list=True
        )

        def total_energy(p):
            return lj_dispersion_energy(
                p,
                sig,
                eps,
                cell,
                cutoff=cutoff,
                alpha=beta,
                neighbor_list=nl,
                neighbor_ptr=nptr,
                neighbor_shifts=nsh,
            ).sum()

        _, F = lj_dispersion_energy_forces(
            positions,
            sig,
            eps,
            cell,
            cutoff=cutoff,
            alpha=beta,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
        )
        h = 1e-6
        for atom in (0, 3):
            for d in range(3):
                pp = positions.clone()
                pp[atom, d] += h
                pm = positions.clone()
                pm[atom, d] -= h
                fd = -(total_energy(pp) - total_energy(pm)) / (2 * h)
                assert abs(fd.item() - F[atom, d].item()) < 1e-4

    def test_forces_helper_matches(self, device):
        positions, sig, eps, cell = _random_system(device)
        nl, nptr, nsh = neighbor_list_fn(
            positions,
            cutoff=4.0,
            cell=cell,
            pbc=torch.tensor([True, True, True], device=device),
            return_neighbor_list=True,
        )
        kw = dict(neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=nsh)
        F1 = lj_dispersion_forces(positions, sig, eps, cell, 4.0, 0.3, **kw)
        _, F2 = lj_dispersion_energy_forces(positions, sig, eps, cell, 4.0, 0.3, **kw)
        assert torch.allclose(F1, F2, rtol=1e-12)


class TestNeighborFormatParity:
    @pytest.mark.parametrize("beta", [0.0, 0.4])
    def test_list_matches_matrix(self, device, beta):
        positions, sig, eps, cell = _random_system(device)
        pbc = torch.tensor([True, True, True], device=device)
        cutoff = 4.0
        nl, nptr, nsh = neighbor_list_fn(
            positions, cutoff=cutoff, cell=cell, pbc=pbc, return_neighbor_list=True
        )
        nmat, _, nmat_sh = neighbor_list_fn(
            positions, cutoff=cutoff, cell=cell, pbc=pbc
        )

        E_list, F_list = lj_dispersion_energy_forces(
            positions,
            sig,
            eps,
            cell,
            cutoff=cutoff,
            alpha=beta,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
        )
        E_mat, F_mat = lj_dispersion_energy_forces(
            positions,
            sig,
            eps,
            cell,
            cutoff=cutoff,
            alpha=beta,
            neighbor_matrix=nmat,
            neighbor_matrix_shifts=nmat_sh,
            fill_value=positions.shape[0],
        )
        assert torch.allclose(E_list.sum(), E_mat.sum(), rtol=1e-9)
        assert torch.allclose(F_list, F_mat, rtol=1e-7, atol=1e-9)


class TestBatching:
    """Batched calculations using explicit neighbor lists (builder-independent)."""

    def _system(self, device, x=3.0):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [x, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        sig = torch.tensor([1.0, 1.2], dtype=torch.float64, device=device)
        eps = torch.tensor([1.0, 0.5], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]]],
            dtype=torch.float64,
            device=device,
        )
        nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        nptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        nsh = torch.zeros((2, 3), dtype=torch.int32, device=device)
        return positions, sig, eps, cell, nl, nptr, nsh

    @pytest.mark.parametrize("beta", [0.0, 0.4])
    def test_single_batch_matches_unbatched(self, device, beta):
        positions, sig, eps, cell, nl, nptr, nsh = self._system(device)
        kw = dict(neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=nsh)
        e_unb, f_unb = lj_dispersion_energy_forces(
            positions, sig, eps, cell, 10.0, beta, **kw
        )
        batch_idx = torch.zeros(2, dtype=torch.int32, device=device)
        e_b, f_b = lj_dispersion_energy_forces(
            positions, sig, eps, cell, 10.0, beta, batch_idx=batch_idx, **kw
        )
        assert torch.allclose(e_b, e_unb, rtol=1e-10)
        assert torch.allclose(f_b, f_unb, rtol=1e-10)

    @pytest.mark.parametrize("beta", [0.0, 0.4])
    def test_two_independent_batches(self, device, beta):
        # Two identical systems offset into a batch -> identical per-system results.
        positions, sig, eps, cell, _, _, _ = self._system(device)
        positions2 = torch.cat([positions, positions], dim=0)
        sigma2 = torch.cat([sig, sig], dim=0)
        eps2 = torch.cat([eps, eps], dim=0)
        cell2 = torch.cat([cell, cell], dim=0)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        # Pairs within each system only.
        nl = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        nptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        nsh = torch.zeros((4, 3), dtype=torch.int32, device=device)
        E, F = lj_dispersion_energy_forces(
            positions2,
            sigma2,
            eps2,
            cell2,
            10.0,
            beta,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
            batch_idx=batch_idx,
        )
        assert torch.allclose(E[0:2].sum(), E[2:4].sum(), rtol=1e-10)
        assert torch.allclose(F[0:2], F[2:4], rtol=1e-10)
        # Momentum conservation within each batch.
        zero3 = torch.zeros(3, dtype=F.dtype, device=device)
        assert torch.allclose(F[0:2].sum(0), zero3, atol=1e-12)
        assert torch.allclose(F[2:4].sum(0), zero3, atol=1e-12)


class TestPrecision:
    def test_float32_matches_float64(self, device):
        positions, sig, eps, cell = _random_system(device)
        nl, nptr, nsh = neighbor_list_fn(
            positions,
            cutoff=4.0,
            cell=cell,
            pbc=torch.tensor([True, True, True], device=device),
            return_neighbor_list=True,
        )
        kw = dict(neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=nsh)
        E64, F64 = lj_dispersion_energy_forces(
            positions, sig, eps, cell, 4.0, 0.3, **kw
        )
        E32, F32 = lj_dispersion_energy_forces(
            positions.float(), sig.float(), eps.float(), cell.float(), 4.0, 0.3, **kw
        )
        assert torch.allclose(E32.double().sum(), E64.sum(), rtol=1e-4, atol=1e-5)
        assert torch.allclose(F32.double(), F64, rtol=1e-3, atol=1e-4)
