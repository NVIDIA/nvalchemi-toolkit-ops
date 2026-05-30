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
Unit tests for dispersion PME (LJ-PME) - PyTorch bindings.

Covers:
- alpha(beta)-independence of the total dispersion-PME energy
- dispersion_pme total == brute-force direct r^-6 periodic lattice sum
- reciprocal/total forces: explicit == autograd
- reciprocal virial == finite-difference cell strain
- batched == single
- float32 ~ float64
- parameter estimation + auto-alpha path
"""

from __future__ import annotations

import pytest
import torch

from nvalchemiops.torch.interactions.dispersion import (
    dispersion_pme,
    dispersion_reciprocal_space,
    estimate_dispersion_pme_parameters,
)
from nvalchemiops.torch.neighbors import neighbor_list as neighbor_list_fn


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _system(device, n=16, box=12.0, seed=3):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = (torch.rand(n, 3, generator=g, dtype=torch.float64) * box).to(device)
    sigma = (0.9 + 0.4 * torch.rand(n, generator=g, dtype=torch.float64)).to(device)
    epsilon = (0.5 + 0.5 * torch.rand(n, generator=g, dtype=torch.float64)).to(device)
    cell = (torch.eye(3, dtype=torch.float64) * box)[None].to(device)
    return pos, sigma, epsilon, cell


def _nl(device, pos, cell, cutoff):
    pbc = torch.tensor([True, True, True], device=device)
    return neighbor_list_fn(
        pos, cutoff=cutoff, cell=cell, pbc=pbc, return_neighbor_list=True
    )


def _brute_force_dispersion(pos, sigma, epsilon, box, nmax=5):
    """Direct periodic sum of -C6_ij/r^6 (geometric C6) over image shells."""
    dev = pos.device
    b = 2.0 * torch.sqrt(epsilon) * sigma**3
    c6 = b[:, None] * b[None, :]
    n = pos.shape[0]
    rng = range(-nmax, nmax + 1)
    shifts = (
        torch.tensor(
            [[a, bb, c] for a in rng for bb in rng for c in rng],
            dtype=torch.float64,
            device=dev,
        )
        * box
    )
    total = 0.0
    zero = torch.zeros(3, dtype=torch.float64, device=dev)
    for s in shifts:
        d = pos[:, None, :] - pos[None, :, :] + s
        r2 = (d * d).sum(-1)
        if torch.allclose(s, zero):
            r2 = r2 + torch.eye(n, device=dev, dtype=torch.float64) * 1e30
        total += -0.5 * (c6 * r2.clamp(min=1e-30) ** (-3)).sum().item()
    return total


class TestAlphaIndependence:
    def test_total_energy_invariant_to_beta(self, device):
        pos, sigma, epsilon, cell = _system(device, n=40, box=16.0, seed=1)
        rc = 7.9
        nl, nptr, nsh = _nl(device, pos, cell, rc)
        kw = dict(
            mesh_dimensions=(80, 80, 80),
            spline_order=6,
            real_space_cutoff=rc,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
        )
        vals = [
            dispersion_pme(pos, sigma, epsilon, cell, alpha=beta, **kw).sum().item()
            for beta in (0.35, 0.40, 0.45, 0.50)
        ]
        spread = max(vals) - min(vals)
        assert spread < 2e-4, f"beta-dependence too large: {spread}, vals={vals}"


class TestPMEvsDirect:
    @pytest.mark.slow
    def test_pme_matches_brute_force(self, device):
        pos, sigma, epsilon, cell = _system(device, n=12, box=12.0, seed=2)
        box = 12.0
        ref = _brute_force_dispersion(pos, sigma, epsilon, box, nmax=6)
        rc = 5.95
        nl, nptr, nsh = _nl(device, pos, cell, rc)
        e = (
            dispersion_pme(
                pos,
                sigma,
                epsilon,
                cell,
                alpha=0.6,
                mesh_dimensions=(128, 128, 128),
                spline_order=6,
                real_space_cutoff=rc,
                neighbor_list=nl,
                neighbor_ptr=nptr,
                neighbor_shifts=nsh,
            )
            .sum()
            .item()
        )
        assert abs(e - ref) / abs(ref) < 1e-3, f"PME={e} ref={ref}"


class TestForces:
    def test_reciprocal_force_explicit_matches_autograd(self, device):
        pos, sigma, epsilon, cell = _system(device)
        mesh = (48, 48, 48)
        p = pos.clone().requires_grad_(True)
        e = dispersion_reciprocal_space(
            p, sigma, epsilon, cell, alpha=0.5, mesh_dimensions=mesh, spline_order=5
        )
        e.sum().backward()
        f_ag = -p.grad
        _, f_ex = dispersion_reciprocal_space(
            pos,
            sigma,
            epsilon,
            cell,
            alpha=0.5,
            mesh_dimensions=mesh,
            spline_order=5,
            compute_forces=True,
        )
        assert torch.allclose(f_ex, f_ag, atol=1e-10)

    def test_pme_force_explicit_matches_autograd(self, device):
        pos, sigma, epsilon, cell = _system(device)
        rc = 5.5
        mesh = (48, 48, 48)
        nl, nptr, nsh = _nl(device, pos, cell, rc)
        kw = dict(
            mesh_dimensions=mesh,
            spline_order=5,
            real_space_cutoff=rc,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
        )
        p = pos.clone().requires_grad_(True)
        e = dispersion_pme(p, sigma, epsilon, cell, alpha=0.5, **kw)
        e.sum().backward()
        f_ag = -p.grad
        _, f_ex = dispersion_pme(
            pos, sigma, epsilon, cell, alpha=0.5, compute_forces=True, **kw
        )
        assert torch.allclose(f_ex, f_ag, atol=1e-10)


class TestVirial:
    def test_reciprocal_virial_matches_finite_difference(self, device):
        pos, sigma, epsilon, cell = _system(device)
        mesh = (48, 48, 48)
        beta = 0.5
        _, virial = dispersion_reciprocal_space(
            pos,
            sigma,
            epsilon,
            cell,
            alpha=beta,
            mesh_dimensions=mesh,
            spline_order=5,
            compute_virial=True,
        )
        eye = torch.eye(3, dtype=torch.float64, device=device)
        h = 1e-5
        w_fd = torch.zeros(3, 3, dtype=torch.float64, device=device)

        def e_strained(strain):
            posp = pos @ (eye + strain).T
            cellp = cell @ (eye + strain).T
            return dispersion_reciprocal_space(
                posp,
                sigma,
                epsilon,
                cellp,
                alpha=beta,
                mesh_dimensions=mesh,
                spline_order=5,
            ).sum()

        for a in range(3):
            for c in range(3):
                sp = torch.zeros(3, 3, dtype=torch.float64, device=device)
                sp[a, c] = h
                w_fd[a, c] = -(e_strained(sp) - e_strained(-sp)) / (2 * h)
        assert torch.allclose(virial[0], w_fd, atol=1e-3, rtol=1e-3)

    def test_pme_virial_runs_and_matches_fd(self, device):
        pos, sigma, epsilon, cell = _system(device, n=12, box=12.0, seed=2)
        rc = 5.5
        mesh = (64, 64, 64)
        beta = 0.55
        nl, nptr, nsh = _nl(device, pos, cell, rc)
        kw = dict(
            mesh_dimensions=mesh,
            spline_order=6,
            real_space_cutoff=rc,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
        )
        _, virial = dispersion_pme(
            pos, sigma, epsilon, cell, alpha=beta, compute_virial=True, **kw
        )
        eye = torch.eye(3, dtype=torch.float64, device=device)
        h = 1e-5

        def e_strained(strain):
            posp = pos @ (eye + strain).T
            cellp = cell @ (eye + strain).T
            # neighbor shifts are integer cell multiples, invariant under strain
            return dispersion_pme(posp, sigma, epsilon, cellp, alpha=beta, **kw).sum()

        w_fd = torch.zeros(3, 3, dtype=torch.float64, device=device)
        for a in range(3):
            for c in range(3):
                sp = torch.zeros(3, 3, dtype=torch.float64, device=device)
                sp[a, c] = h
                w_fd[a, c] = -(e_strained(sp) - e_strained(-sp)) / (2 * h)
        assert torch.allclose(virial[0], w_fd, atol=1e-3, rtol=1e-3)


class TestBatching:
    def test_reciprocal_batched_matches_single(self, device):
        pos, sigma, epsilon, cell = _system(device)
        n = pos.shape[0]
        mesh = (48, 48, 48)
        beta = 0.5
        single = dispersion_reciprocal_space(
            pos, sigma, epsilon, cell, alpha=beta, mesh_dimensions=mesh, spline_order=5
        )
        bidx = torch.cat(
            [
                torch.zeros(n, dtype=torch.int32, device=device),
                torch.ones(n, dtype=torch.int32, device=device),
            ]
        )
        e_b = dispersion_reciprocal_space(
            torch.cat([pos, pos]),
            torch.cat([sigma, sigma]),
            torch.cat([epsilon, epsilon]),
            torch.cat([cell, cell]),
            alpha=beta,
            mesh_dimensions=mesh,
            spline_order=5,
            batch_idx=bidx,
        )
        assert torch.allclose(e_b[:n], single, atol=1e-10)
        assert torch.allclose(e_b[n:], single, atol=1e-10)


class TestPrecision:
    def test_float32_close_to_float64(self, device):
        pos, sigma, epsilon, cell = _system(device)
        mesh = (48, 48, 48)
        beta = 0.5
        e64 = dispersion_reciprocal_space(
            pos, sigma, epsilon, cell, alpha=beta, mesh_dimensions=mesh, spline_order=5
        ).sum()
        e32 = dispersion_reciprocal_space(
            pos.float(),
            sigma.float(),
            epsilon.float(),
            cell.float(),
            alpha=beta,
            mesh_dimensions=mesh,
            spline_order=5,
        ).sum()
        assert abs(e32.double().item() - e64.item()) / abs(e64.item()) < 1e-3


class TestParameters:
    def test_estimate_and_auto_alpha(self, device):
        pos, sigma, epsilon, cell = _system(device, n=30, box=14.0, seed=5)
        params = estimate_dispersion_pme_parameters(pos, cell, accuracy=1e-5)
        assert params.alpha.shape == (1,)
        assert float(params.alpha[0]) > 0
        assert len(params.mesh_dimensions) == 3
        rc = float(params.real_space_cutoff[0].item())
        nl, nptr, nsh = _nl(device, pos, cell, rc)
        # auto-alpha path runs end to end
        e = dispersion_pme(
            pos,
            sigma,
            epsilon,
            cell,
            alpha=None,
            accuracy=1e-5,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
        )
        assert torch.isfinite(e.sum())
