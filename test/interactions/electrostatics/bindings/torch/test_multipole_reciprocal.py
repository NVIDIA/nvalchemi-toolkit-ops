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

"""Tests for the multipole direct-k reciprocal-space energy (l = 0/1/2) and grads."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_electrostatic_energy,
    multipole_reciprocal_space_energy,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    cartesian_quadrupole_to_e3nn,
    dipole_cartesian_to_spherical,
    pack_charges_dipoles,
    pack_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
    FIELD_CONSTANT,
    multipole_ewald_summation,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
    multipole_real_space_quadrupole_energy,
)
from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
    multipole_pme_reciprocal_space,
)


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _system(
    *, n_atoms: int, box_len: float, device: str, seed: int, with_dipoles: bool
):
    rng = np.random.default_rng(seed)
    td = _torch_device(device)
    positions = torch.from_numpy(rng.uniform(0, box_len, size=(n_atoms, 3))).to(
        td, dtype=torch.float64
    )
    charges_np = rng.uniform(-1, 1, size=n_atoms)
    charges_np -= charges_np.mean()
    charges = torch.from_numpy(charges_np).to(td, dtype=torch.float64)
    dipoles = None
    if with_dipoles:
        dipoles = torch.from_numpy(0.3 * rng.standard_normal((n_atoms, 3))).to(
            td, dtype=torch.float64
        )
    cell = torch.eye(3, dtype=torch.float64, device=td) * box_len
    return positions, charges, dipoles, cell


class TestMultipoleReciprocalSpaceLimit:
    """At α→∞ the Gaussian damping factor exp(−k²/(4α²)) → 1 so
    reciprocal-space must converge to the direct-kspace Path B result.
    """

    @pytest.mark.parametrize("with_dipoles", [False, True])
    def test_large_alpha_matches_direct_kspace(self, device, with_dipoles):
        positions, charges, dipoles, cell = _system(
            n_atoms=6, box_len=5.0, device=device, seed=0, with_dipoles=with_dipoles
        )
        sigma, k_cutoff = 1.0, 3.0

        source_feats = pack_charges_dipoles(charges, dipoles)
        e_direct = multipole_electrostatic_energy(
            positions,
            source_feats,
            cell,
            sigma=sigma,
            k_cutoff=k_cutoff,
            include_self_interaction=True,
        )
        e_recip = multipole_reciprocal_space_energy(
            positions,
            source_feats,
            cell,
            sigma=sigma,
            alpha=100.0,
            k_cutoff=k_cutoff,
        )
        # The residual is O((k_max/α)²) from the exp series; with
        # k_cutoff=3 and α=100 it's ~(3/100)² ≈ 1e-3 magnitude.
        torch.testing.assert_close(e_recip, e_direct, rtol=2e-3, atol=2e-3)

    def test_small_alpha_damps_to_zero(self, device):
        """Heavy Gaussian damping (small α) drives the reciprocal sum toward zero."""
        positions, charges, dipoles, cell = _system(
            n_atoms=6, box_len=5.0, device=device, seed=0, with_dipoles=False
        )
        e = multipole_reciprocal_space_energy(
            positions,
            pack_charges_dipoles(charges, None),
            cell,
            sigma=1.0,
            alpha=0.05,
            k_cutoff=3.0,
        )
        assert abs(e.sum().item()) < 1e-10


class TestMultipoleReciprocalSpaceShapes:
    def test_output_scalar_float64(self, device):
        positions, charges, dipoles, cell = _system(
            n_atoms=5, box_len=5.0, device=device, seed=0, with_dipoles=True
        )
        e = multipole_reciprocal_space_energy(
            positions,
            pack_charges_dipoles(charges, dipoles),
            cell,
            sigma=1.0,
            alpha=0.4,
            k_cutoff=3.0,
        )
        assert e.shape == (5,)
        assert e.dtype == torch.float64
        assert e.device.type == _torch_device(device)
        assert torch.isfinite(e.sum())


class TestMultipoleReciprocalSpaceAutograd:
    def test_first_order_backward(self, device):
        positions_, charges_, dipoles_, cell = _system(
            n_atoms=6, box_len=5.0, device=device, seed=7, with_dipoles=True
        )
        positions = positions_.detach().clone().requires_grad_(True)
        source_feats = pack_charges_dipoles(
            charges_.detach().clone(), dipoles_.detach().clone()
        ).requires_grad_(True)
        e = multipole_reciprocal_space_energy(
            positions,
            source_feats,
            cell,
            sigma=1.0,
            alpha=0.4,
            k_cutoff=3.0,
        )
        e.sum().backward()
        for t in (positions, source_feats):
            assert t.grad is not None
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0

    def test_double_backward_force_loss(self, device):
        """create_graph=True force-loss path (reuses MultipoleRhoFunction's 2nd-order)."""
        positions_, charges_, dipoles_, cell = _system(
            n_atoms=5, box_len=5.0, device=device, seed=11, with_dipoles=True
        )
        positions = positions_.detach().clone().requires_grad_(True)
        source_feats = pack_charges_dipoles(
            charges_.detach().clone(), dipoles_.detach().clone()
        ).requires_grad_(True)
        e = multipole_reciprocal_space_energy(
            positions,
            source_feats,
            cell,
            sigma=1.0,
            alpha=0.4,
            k_cutoff=3.0,
        )
        (forces_neg,) = torch.autograd.grad(e.sum(), positions, create_graph=True)
        loss = (forces_neg**2).sum()
        loss.backward()
        for t in (positions, source_feats):
            assert t.grad is not None
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0


class TestBatchMultipoleReciprocalSpace:
    """Batched reciprocal-space entry — checks the sugar wrapper stays wired."""

    def test_matches_per_system_loop(self, device):
        rng = np.random.default_rng(0)
        td = _torch_device(device)
        n_per, L = 5, 4.0

        def _sys():
            pos = torch.from_numpy(rng.uniform(0, L, (n_per, 3))).to(td, torch.float64)
            chg_np = rng.uniform(-1, 1, n_per)
            chg_np -= chg_np.mean()
            chg = torch.from_numpy(chg_np).to(td, torch.float64)
            mu = torch.from_numpy(0.2 * rng.standard_normal((n_per, 3))).to(
                td, torch.float64
            )
            cell = torch.eye(3, dtype=torch.float64, device=td) * L
            return pos, chg, mu, cell

        systems = [_sys() for _ in range(3)]
        sigma, alpha, kcut = 1.0, 0.4, 3.0

        per_e = torch.stack(
            [
                multipole_reciprocal_space_energy(
                    p,
                    pack_charges_dipoles(c, m),
                    cell,
                    sigma=sigma,
                    alpha=alpha,
                    k_cutoff=kcut,
                ).sum()
                for (p, c, m, cell) in systems
            ]
        )
        pos_all = torch.cat([s[0] for s in systems])
        chg_all = torch.cat([s[1] for s in systems])
        mu_all = torch.cat([s[2] for s in systems])
        cells = torch.stack([s[3] for s in systems])
        bi = torch.cat(
            [torch.full((n_per,), i, dtype=torch.int32, device=td) for i in range(3)]
        )
        e_batch = multipole_reciprocal_space_energy(
            pos_all,
            pack_charges_dipoles(chg_all, mu_all),
            cells,
            batch_idx=bi,
            sigma=sigma,
            alpha=alpha,
            k_cutoff=kcut,
        )
        B = 3
        e_batch_per_sys = torch.zeros(
            B, dtype=torch.float64, device=e_batch.device
        ).scatter_add(0, bi.long(), e_batch)
        torch.testing.assert_close(e_batch_per_sys, per_e, rtol=0, atol=1e-14)

    def test_backward_runs(self, device):
        rng = np.random.default_rng(17)
        td = _torch_device(device)
        n_per = 4
        B = 2
        positions = (
            torch.from_numpy(rng.uniform(0, 5.0, (B * n_per, 3)))
            .to(td, torch.float64)
            .requires_grad_(True)
        )
        charges_np = rng.uniform(-1, 1, B * n_per)
        charges_np -= charges_np.mean()
        charges = torch.from_numpy(charges_np).to(td, torch.float64)
        dipoles = torch.from_numpy(0.2 * rng.standard_normal((B * n_per, 3))).to(
            td, torch.float64
        )
        source_feats = pack_charges_dipoles(charges, dipoles).requires_grad_(True)
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * 5.0 for _ in range(B)]
        )
        bi = torch.cat(
            [torch.full((n_per,), i, dtype=torch.int32, device=td) for i in range(B)]
        )
        e = multipole_reciprocal_space_energy(
            positions,
            source_feats,
            cells,
            batch_idx=bi,
            sigma=1.0,
            alpha=0.4,
            k_cutoff=3.0,
        )
        assert e.shape == (B * n_per,)
        e.sum().backward()
        for t in (positions, source_feats):
            assert t.grad is not None
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0


class TestMultipoleReciprocalSpaceValidation:
    def test_alpha_must_be_positive(self, device):
        positions, charges, _, cell = _system(
            n_atoms=4, box_len=5.0, device=device, seed=0, with_dipoles=False
        )
        with pytest.raises(ValueError, match="alpha must be positive"):
            multipole_reciprocal_space_energy(
                positions,
                pack_charges_dipoles(charges, None),
                cell,
                sigma=1.0,
                alpha=-0.1,
                k_cutoff=3.0,
            )

    def test_requires_k_grid_info(self, device):
        positions, charges, _, cell = _system(
            n_atoms=4, box_len=5.0, device=device, seed=0, with_dipoles=False
        )
        with pytest.raises(ValueError, match="k_cutoff"):
            multipole_reciprocal_space_energy(
                positions,
                pack_charges_dipoles(charges, None),
                cell,
                sigma=1.0,
                alpha=0.4,
            )


class TestMultipoleReciprocalSpaceDipoleFusedScalar:
    r"""Parity tests for ``MultipoleReciprocalSpaceDipoleFusedScalarFunction``.

    The fused Function uses the SAME forward kernels as
    :func:`multipole_scf_step_energy` / :func:`multipole_reciprocal_space_energy`
    and computes ``grad_rho`` from the same scalar-energy formula, so parity is
    bit-exact at fp64.
    """

    @pytest.fixture
    def gpu_system(self):
        if not torch.cuda.is_available():
            pytest.skip("Path A reciprocal kernels are GPU-only")
        positions, charges, dipoles, cell = _system(
            n_atoms=8, box_len=5.0, device="cuda:0", seed=0xCAFE, with_dipoles=True
        )
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
            prepare_multipole_scf_cache,
        )

        sigma, alpha = 1.0, 0.5
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=[sigma],
            k_cutoff=3.0,
            l_max=1,
            alpha=alpha,
            device="cuda:0",
        )
        return {
            "positions": positions,
            "charges": charges,
            "dipoles": dipoles,
            "cell": cell,
            "cache": cache,
        }

    def _run_reference(self, sd, *, with_pos, with_q, with_mu, include_self):
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
            multipole_scf_step_energy,
        )

        p = sd["positions"].clone().detach().requires_grad_(with_pos)
        q = sd["charges"].clone().detach().requires_grad_(with_q)
        mu = sd["dipoles"].clone().detach().requires_grad_(with_mu)
        sf = pack_charges_dipoles(q, mu)
        e = multipole_scf_step_energy(
            sd["cache"], p, sf, include_self_interaction=include_self
        )
        return p, q, mu, e

    def _run_fused(self, sd, *, with_pos, with_q, with_mu, include_self):
        from nvalchemiops.torch.interactions.electrostatics.multipole_autograd import (
            multipole_reciprocal_space_dipole_fused_scalar,
        )

        p = sd["positions"].clone().detach().requires_grad_(with_pos)
        q = sd["charges"].clone().detach().requires_grad_(with_q)
        mu = sd["dipoles"].clone().detach().requires_grad_(with_mu)
        sf = pack_charges_dipoles(q, mu)
        e = multipole_reciprocal_space_dipole_fused_scalar(
            p, sf, sd["cache"], include_self_interaction=include_self
        )
        return p, q, mu, e

    @pytest.mark.parametrize("include_self", [True, False])
    def test_forward_parity(self, gpu_system, include_self):
        """Bit-exact energy parity vs `multipole_scf_step_energy`."""
        _, _, _, e_ref = self._run_reference(
            gpu_system,
            with_pos=False,
            with_q=False,
            with_mu=False,
            include_self=include_self,
        )
        _, _, _, e_f = self._run_fused(
            gpu_system,
            with_pos=False,
            with_q=False,
            with_mu=False,
            include_self=include_self,
        )
        torch.testing.assert_close(e_f, e_ref, rtol=0, atol=0)

    @pytest.mark.parametrize("include_self", [True, False])
    def test_all_grads_parity(self, gpu_system, include_self):
        p_r, q_r, mu_r, e_r = self._run_reference(
            gpu_system,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=include_self,
        )
        e_r.sum().backward()
        p_f, q_f, mu_f, e_f = self._run_fused(
            gpu_system,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=include_self,
        )
        e_f.sum().backward()
        torch.testing.assert_close(p_f.grad, p_r.grad, rtol=0, atol=0)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=0)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=0)

    def test_per_input_gating_pos_only(self, gpu_system):
        p_f, q_f, mu_f, e_f = self._run_fused(
            gpu_system,
            with_pos=True,
            with_q=False,
            with_mu=False,
            include_self=True,
        )
        e_f.sum().backward()
        assert p_f.grad is not None
        assert q_f.grad is None
        assert mu_f.grad is None

    def test_no_grad_path(self, gpu_system):
        _, _, _, e_f = self._run_fused(
            gpu_system,
            with_pos=False,
            with_q=False,
            with_mu=False,
            include_self=True,
        )
        assert not e_f.requires_grad
        assert torch.isfinite(e_f).all()

    def test_weighted_backward(self, gpu_system):
        """Custom upstream scalar grad multiplier; gradients scale linearly.

        Tolerance is 1 ULP: the fused path multiplies by ``upstream`` after the
        backward kernels finish while the reference multiplies ``grad_rho``
        before, so atomic_add ordering gives fp64 ULP drift.
        """
        p_f, q_f, mu_f, e_f = self._run_fused(
            gpu_system,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=True,
        )
        upstream = torch.full_like(e_f, 2.5)
        e_f.backward(upstream)
        p_r, q_r, mu_r, e_r = self._run_reference(
            gpu_system,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=True,
        )
        e_r.backward(upstream)
        torch.testing.assert_close(p_f.grad, p_r.grad, rtol=1e-14, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=1e-14, atol=1e-15)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=1e-14, atol=1e-15)


class TestBatchMultipoleReciprocalSpaceDipoleFusedScalar:
    r"""Parity tests for the batched Path A reciprocal fused Function."""

    @pytest.fixture
    def gpu_batch(self):
        if not torch.cuda.is_available():
            pytest.skip("Path A reciprocal kernels are GPU-only")
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
            prepare_multipole_scf_cache,
        )

        sigma, alpha = 1.0, 0.5
        n_systems = 3
        n_per_sys = 6
        sub = []
        for s in range(n_systems):
            p, q, mu, c = _system(
                n_atoms=n_per_sys,
                box_len=5.0,
                device="cuda:0",
                seed=0x300 + s,
                with_dipoles=True,
            )
            sub.append({"positions": p, "charges": q, "dipoles": mu, "cell": c})

        td = "cuda:0"
        positions = torch.cat([s["positions"] for s in sub], dim=0)
        charges = torch.cat([s["charges"] for s in sub], dim=0)
        dipoles = torch.cat([s["dipoles"] for s in sub], dim=0)
        cells = torch.stack([s["cell"] for s in sub], dim=0)
        batch_idx = torch.cat(
            [
                torch.full((n_per_sys,), s_idx, dtype=torch.int32, device=td)
                for s_idx in range(n_systems)
            ]
        )

        cache = prepare_multipole_scf_cache(
            cells,
            sigma=sigma,
            receiver_sigmas=[sigma],
            k_cutoff=3.0,
            l_max=1,
            alpha=alpha,
            device=td,
        )
        return {
            "positions": positions,
            "charges": charges,
            "dipoles": dipoles,
            "batch_idx": batch_idx,
            "cache": cache,
            "n_systems": n_systems,
        }

    def _run_reference(self, b, *, with_pos, with_q, with_mu, include_self):
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
            multipole_scf_step_energy,
        )

        p = b["positions"].clone().detach().requires_grad_(with_pos)
        q = b["charges"].clone().detach().requires_grad_(with_q)
        mu = b["dipoles"].clone().detach().requires_grad_(with_mu)
        sf = pack_charges_dipoles(q, mu)
        e = multipole_scf_step_energy(
            b["cache"],
            p,
            sf,
            batch_idx=b["batch_idx"],
            include_self_interaction=include_self,
        )
        return p, q, mu, e

    def _run_fused(self, b, *, with_pos, with_q, with_mu, include_self):
        from nvalchemiops.torch.interactions.electrostatics.multipole_autograd_batch import (
            batch_multipole_reciprocal_space_dipole_fused_scalar,
        )

        p = b["positions"].clone().detach().requires_grad_(with_pos)
        q = b["charges"].clone().detach().requires_grad_(with_q)
        mu = b["dipoles"].clone().detach().requires_grad_(with_mu)
        sf = pack_charges_dipoles(q, mu)
        e = batch_multipole_reciprocal_space_dipole_fused_scalar(
            p,
            sf,
            b["batch_idx"],
            b["cache"],
            include_self_interaction=include_self,
        )
        return p, q, mu, e

    @pytest.mark.parametrize("include_self", [True, False])
    def test_forward_parity(self, gpu_batch, include_self):
        _, _, _, e_ref = self._run_reference(
            gpu_batch,
            with_pos=False,
            with_q=False,
            with_mu=False,
            include_self=include_self,
        )
        _, _, _, e_f = self._run_fused(
            gpu_batch,
            with_pos=False,
            with_q=False,
            with_mu=False,
            include_self=include_self,
        )
        torch.testing.assert_close(e_f, e_ref, rtol=0, atol=0)

    @pytest.mark.parametrize("include_self", [True, False])
    def test_all_grads_parity(self, gpu_batch, include_self):
        p_r, q_r, mu_r, e_r = self._run_reference(
            gpu_batch,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=include_self,
        )
        e_r.sum().backward()
        p_f, q_f, mu_f, e_f = self._run_fused(
            gpu_batch,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=include_self,
        )
        e_f.sum().backward()
        torch.testing.assert_close(p_f.grad, p_r.grad, rtol=0, atol=0)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=0)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=0)

    def test_weighted_backward(self, gpu_batch):
        """Per-system upstream-weight broadcast — non-uniform weights.

        Tolerance is 1 ULP — same atomic_add ordering divergence as the
        single-system weighted_backward test.
        """
        p_f, q_f, mu_f, e_f = self._run_fused(
            gpu_batch,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=False,
        )
        weights = torch.tensor([1.5, -0.5, 2.0], dtype=torch.float64, device=e_f.device)
        # broadcast per-system weights to per-atom via batch_idx
        atom_weights = weights[gpu_batch["batch_idx"].long()]
        (atom_weights * e_f).sum().backward()
        p_r, q_r, mu_r, e_r = self._run_reference(
            gpu_batch,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=False,
        )
        (atom_weights * e_r).sum().backward()
        torch.testing.assert_close(p_f.grad, p_r.grad, rtol=1e-14, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=1e-14, atol=1e-15)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=1e-14, atol=1e-15)


F = float(FIELD_CONSTANT)


def _fixture(seed=11, N=4):
    rng = np.random.default_rng(seed)
    L = 6.0
    cell_np = np.eye(3) * L
    pos = rng.uniform(0, L, size=(N, 3))
    q = rng.normal(size=N)
    q -= q.mean()
    mu = rng.normal(size=(N, 3))
    Qr = rng.normal(size=(N, 3, 3))
    Q = 0.5 * (Qr + Qr.transpose(0, 2, 1))
    return cell_np, pos, q, mu, Q, L


def _sf(q, mu):
    # e3nn packing [q, mu_y, mu_z, mu_x]
    return torch.tensor(
        np.concatenate([q[:, None], mu[:, [1, 2, 0]]], axis=1), dtype=torch.float64
    )


def _mm(q, mu, Q=None):
    """Packed e3nn ``multipole_moments``: l<=1 block (+ traceless l=2 from a
    Cartesian Q via the converter)."""
    from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
        cartesian_quadrupole_to_e3nn,
    )

    sf = _sf(q, mu)
    if Q is None:
        return sf
    Qt = Q if torch.is_tensor(Q) else torch.tensor(Q, dtype=torch.float64)
    return torch.cat([sf, cartesian_quadrupole_to_e3nn(Qt)], dim=-1)


def _half_kgrid(cell_np, kcut):
    """Canonical half k-set (origin + one of each ±k pair) + the full ±set."""
    G = 2.0 * np.pi * np.linalg.inv(cell_np).T
    nmax = int(np.ceil(kcut / np.linalg.norm(G, axis=1).min())) + 1
    half = []
    for a in range(-nmax, nmax + 1):
        for b in range(-nmax, nmax + 1):
            for c in range(-nmax, nmax + 1):
                k = a * G[0] + b * G[1] + c * G[2]
                if not (0 < np.linalg.norm(k) <= kcut):
                    continue
                if (a > 0) or (a == 0 and b > 0) or (a == 0 and b == 0 and c > 0):
                    half.append(k)
    half = np.array(half)
    return half


def _ref_recip(pos, q, mu, Q, ks_full, sigma, alpha, V):
    E = 0.0
    for k in ks_full:
        k2 = float(k @ k)
        e = np.exp(-1j * (pos @ k))
        rho = (q * e).sum() - 1j * ((mu @ k) * e).sum()
        rho -= 0.5 * (np.einsum("a,nab,b->n", k, Q, k) * e).sum()
        E += (
            (4.0 * math.pi / k2)
            * math.exp(-k2 * (0.25 / alpha**2 + sigma**2))
            * (rho * rho.conjugate()).real
        )
    return (F / (4.0 * math.pi)) * 0.5 / V * E


@pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8])
def test_reciprocal_quadrupole_parity_vs_reference(sigma):
    """Direct-k reciprocal energy matches the numpy reciprocal reference."""
    alpha, kcut = 0.45, 9.0
    cell_np, pos, q, mu, Q, L = _fixture()
    V = L**3
    half = _half_kgrid(cell_np, kcut)
    ks_prod = np.vstack([np.zeros((1, 3)), half])
    ks_full = np.vstack([half, -half])

    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
        prepare_multipole_scf_cache,
    )

    # Pin the kernel to the exact reference k-grid by pre-building a cache from
    # it (custom k-grids now live on the cache, not a k_vectors kwarg).
    cache = prepare_multipole_scf_cache(
        torch.tensor(cell_np),
        sigma=sigma,
        receiver_sigmas=[sigma],
        k_vectors=torch.tensor(ks_prod, dtype=torch.float64),
        l_max=2,
        alpha=alpha,
    )
    E = multipole_reciprocal_space_energy(
        torch.tensor(pos),
        _mm(q, mu, Q),
        torch.tensor(cell_np),
        sigma=sigma,
        alpha=alpha,
        cache=cache,
    )
    # Reference uses the SAME traceless Q the converter feeds the kernel.
    Q_tl = Q - np.eye(3)[None] * (np.trace(Q, axis1=1, axis2=2)[:, None, None] / 3.0)
    E_ref = _ref_recip(pos, q, mu, Q_tl, ks_full, sigma, alpha, V)
    assert abs(float(E.sum()) - E_ref) / abs(E_ref) < 1e-9


def test_reciprocal_quadrupole_grads_match_fd():
    """Autograd ∂E/∂{pos,q,μ,Q} match central finite differences."""
    sigma, alpha, kcut = 0.5, 0.45, 9.0
    cell_np, pos, q, mu, Q, L = _fixture()
    cell = torch.tensor(cell_np)
    kvec = torch.tensor(
        np.vstack([np.zeros((1, 3)), _half_kgrid(cell_np, kcut)]), dtype=torch.float64
    )
    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
        prepare_multipole_scf_cache,
    )

    # Position-independent cache built once from the fixed k-grid; reused across
    # every FD energy evaluation (positions enter per call via autograd).
    cache = prepare_multipole_scf_cache(
        cell,
        sigma=sigma,
        receiver_sigmas=[sigma],
        k_vectors=kvec,
        l_max=2,
        alpha=alpha,
    )

    def energy(pos_, q_, mu_, Q_):
        sf_ = torch.cat([q_[:, None], dipole_cartesian_to_spherical(mu_)], dim=1)
        mm_ = torch.cat([sf_, cartesian_quadrupole_to_e3nn(Q_)], dim=1)
        return multipole_reciprocal_space_energy(
            pos_,
            mm_,
            cell,
            sigma=sigma,
            alpha=alpha,
            cache=cache,
        )

    pos_t = torch.tensor(pos, requires_grad=True)
    q_t = torch.tensor(q, requires_grad=True)
    mu_t = torch.tensor(mu, requires_grad=True)
    Q_t = torch.tensor(Q, requires_grad=True)
    E = energy(pos_t, q_t, mu_t, Q_t)
    gpos, gq, gmu, gQ = torch.autograd.grad(E.sum(), [pos_t, q_t, mu_t, Q_t])

    h = 1e-6
    base = (torch.tensor(pos), torch.tensor(q), torch.tensor(mu), torch.tensor(Q))

    def fd(idx, shape):
        out = torch.zeros(shape, dtype=torch.float64)
        flat = out.view(-1)
        b0 = base[idx].clone()
        for i in range(flat.numel()):
            args_p = list(base)
            args_m = list(base)
            bp = b0.clone()
            bp.view(-1)[i] += h
            bm = b0.clone()
            bm.view(-1)[i] -= h
            args_p[idx] = bp
            args_m[idx] = bm
            flat[i] = (float(energy(*args_p).sum()) - float(energy(*args_m).sum())) / (
                2 * h
            )
        return out

    N = pos.shape[0]
    assert (gpos - fd(0, (N, 3))).abs().max() / gpos.abs().max() < 1e-5
    assert (gq - fd(1, (N,))).abs().max() / gq.abs().max() < 1e-5
    assert (gmu - fd(2, (N, 3))).abs().max() / gmu.abs().max() < 1e-5
    fd_Q_free = fd(3, (N, 3, 3))
    fd_Q = 0.5 * (fd_Q_free + fd_Q_free.transpose(-1, -2))  # kernel emits symmetric
    assert (gQ - fd_Q).abs().max() / fd_Q.abs().max() < 1e-5
    assert (gQ - gQ.transpose(-1, -2)).abs().max() < 1e-12


def test_reciprocal_quadrupole_none_unchanged():
    """quadrupoles=None reproduces the l<=1 reciprocal exactly (no regression)."""
    sigma, alpha, kcut = 0.5, 0.45, 9.0
    cell_np, pos, q, mu, Q, L = _fixture()
    kvec = torch.tensor(
        np.vstack([np.zeros((1, 3)), _half_kgrid(cell_np, kcut)]), dtype=torch.float64
    )
    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
        prepare_multipole_scf_cache,
    )

    # One l=2 cache off the shared grid serves both the l<=1 and Q=0 calls.
    cache = prepare_multipole_scf_cache(
        torch.tensor(cell_np),
        sigma=sigma,
        receiver_sigmas=[sigma],
        k_vectors=kvec,
        l_max=2,
        alpha=alpha,
    )
    args = dict(sigma=sigma, alpha=alpha, cache=cache)
    E_none = multipole_reciprocal_space_energy(
        torch.tensor(pos), _sf(q, mu), torch.tensor(cell_np), **args
    )
    E_q0 = multipole_reciprocal_space_energy(
        torch.tensor(pos),
        _mm(q, mu, np.zeros((pos.shape[0], 3, 3))),
        torch.tensor(cell_np),
        **args,
    )
    assert abs(float(E_none.sum()) - float(E_q0.sum())) < 1e-12


def _build_csr(pos, cell_np, cutoff):
    N = len(pos)
    Lv = np.diag(cell_np)
    ii, jj, sh = [], [], []
    for i in range(N):
        for j in range(N):
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    for sz in (-1, 0, 1):
                        if i == j and sx == sy == sz == 0:
                            continue
                        d = pos[j] - pos[i] + np.array([sx, sy, sz]) * Lv
                        if np.linalg.norm(d) < cutoff:
                            ii.append(i)
                            jj.append(j)
                            sh.append((sx, sy, sz))
    order = sorted(range(len(ii)), key=lambda k: (ii[k], jj[k]))
    ii = [ii[k] for k in order]
    jj = [jj[k] for k in order]
    sh = [sh[k] for k in order]
    ptr = np.zeros(N + 1, dtype=np.int32)
    for i in ii:
        ptr[i + 1] += 1
    ptr = np.cumsum(ptr).astype(np.int32)
    return (
        torch.tensor(jj, dtype=torch.int32),
        torch.tensor(ptr, dtype=torch.int32),
        torch.tensor(sh, dtype=torch.int32).reshape(-1, 3),
    )


def test_composite_quadrupole_matches_pme_composite():
    """Full direct-k Ewald composite (l=2) == PME composite to mesh accuracy.

    Both share the identical (FD-validated) real-space kernel + analytical
    self; they differ only in the reciprocal method.
    """
    sigma, alpha = 0.5, 0.45
    real_cut, kcut = 12.0, 12.0
    cell_np, pos, q, mu, Q, L = _fixture()
    # Detrace Q: the migrated Ewald path is e3nn-traceless; the PME oracle
    # (still Cartesian, trace-ful) must use the same traceless Q to agree.
    Q = Q - np.eye(3)[None] * (np.trace(Q, axis1=1, axis2=2)[:, None, None] / 3.0)
    idx_j, ptr, sh = _build_csr(pos, cell_np, real_cut)
    cell = torch.tensor(cell_np)
    sf = _sf(q, mu)
    pos_t = torch.tensor(pos)
    Q_t = torch.tensor(Q)

    E_dk = float(
        multipole_ewald_summation(
            pos_t,
            _mm(q, mu, Q),
            cell,
            idx_j,
            ptr,
            sh,
            sigma=sigma,
            alpha=alpha,
            k_cutoff=kcut,
        ).sum()
    )

    cs = F / (4.0 * math.pi)
    e_real = cs * float(
        multipole_real_space_quadrupole_energy(
            pos_t,
            sf[:, 0],
            sf[:, [3, 1, 2]].contiguous(),
            Q_t,
            cell.unsqueeze(0),
            idx_j,
            ptr,
            sh,
            torch.tensor([sigma], dtype=torch.float64),
            torch.tensor([alpha], dtype=torch.float64),
        ).sum()
    )
    # multipole_pme_reciprocal_space already returns recip - self - bg.
    e_recip_pme = float(
        multipole_pme_reciprocal_space(
            pos_t,
            torch.cat([sf, cartesian_quadrupole_to_e3nn(Q_t)], dim=-1),
            cell,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(48, 48, 48),
        ).sum()
    )
    E_pme = e_real + e_recip_pme
    assert abs(E_dk - E_pme) / abs(E_pme) < 2e-4


def test_composite_quadrupole_batched_matches_per_system():
    """Batched l=2 composite (2 systems) == single-system composite per system,
    and is autograd-connected across the flat moment tensors."""
    sigma, alpha, real_cut, kcut = 0.5, 0.45, 12.0, 12.0
    systems = [_fixture(11), _fixture(22)]

    # Per-system single composite.
    E_single = []
    for cell_np, pos, q, mu, Q, L in systems:
        idx_j, ptr, sh = _build_csr(pos, cell_np, real_cut)
        E = multipole_ewald_summation(
            torch.tensor(pos),
            _mm(q, mu, Q),
            torch.tensor(cell_np),
            idx_j,
            ptr,
            sh,
            sigma=sigma,
            alpha=alpha,
            k_cutoff=kcut,
        )
        E_single.append(float(E.sum()))

    # Concatenate into a flat batch with offset CSR + batch_idx.
    pos_all, q_all, mu_all, Q_all, cells, bidx = [], [], [], [], [], []
    idxj_all, ptr_all, sh_all = [], [0], []
    off, ptr_off = 0, 0
    for b, (cell_np, pos, q, mu, Q, L) in enumerate(systems):
        idx_j, ptr, sh = _build_csr(pos, cell_np, real_cut)
        n = pos.shape[0]
        pos_all.append(pos)
        q_all.append(q)
        mu_all.append(mu)
        Q_all.append(Q)
        cells.append(cell_np)
        bidx += [b] * n
        idxj_all.append(idx_j.numpy() + off)
        ptr_all += (ptr.numpy()[1:] + ptr_off).tolist()
        sh_all.append(sh.numpy())
        off += n
        ptr_off += len(idx_j)
    pos_all = np.concatenate(pos_all)
    q_all = np.concatenate(q_all)
    mu_all = np.concatenate(mu_all)
    Q_all = np.concatenate(Q_all)
    cells = np.stack(cells)
    sf = _sf(q_all, mu_all)
    idx_j = torch.tensor(np.concatenate(idxj_all), dtype=torch.int32)
    ptr = torch.tensor(ptr_all, dtype=torch.int32)
    sh = torch.tensor(np.concatenate(sh_all), dtype=torch.int32)
    bidx = torch.tensor(bidx, dtype=torch.int32)
    pos_t = torch.tensor(pos_all, requires_grad=True)
    Q_t = torch.tensor(Q_all, requires_grad=True)

    mm_b = torch.cat([sf, cartesian_quadrupole_to_e3nn(Q_t)], dim=-1)
    Eb = multipole_ewald_summation(
        pos_t,
        mm_b,
        torch.tensor(cells),
        idx_j,
        ptr,
        sh,
        sigma=sigma,
        alpha=alpha,
        k_cutoff=kcut,
        batch_idx=bidx,
    )
    assert Eb.shape == (off,)
    Eb_per_sys = torch.zeros(2, dtype=torch.float64).scatter_add(
        0, bidx.long(), Eb.cpu()
    )
    for i in range(2):
        assert abs(float(Eb_per_sys[i]) - E_single[i]) / abs(E_single[i]) < 1e-9
    gpos, gQ = torch.autograd.grad(Eb.sum(), [pos_t, Q_t])
    assert torch.isfinite(gpos).all() and torch.isfinite(gQ).all()


def test_pme_composite_quadrupole_matches_directk_and_batched():
    """multipole_particle_mesh_ewald(..., quadrupoles=Q) matches the direct-k
    Ewald composite (PME mesh accuracy) single-system; the batched l=2 path
    reproduces the single-system result bit-for-bit (B=1 system)."""
    from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
        multipole_particle_mesh_ewald,
    )

    sigma, alpha, real_cut, kcut = 0.5, 0.45, 12.0, 12.0
    cell_np, pos, q, mu, Q, L = _fixture()
    # Detrace Q so the traceless Ewald path matches the trace-ful PME oracle.
    Q = Q - np.eye(3)[None] * (np.trace(Q, axis1=1, axis2=2)[:, None, None] / 3.0)
    idx_j, ptr, sh = _build_csr(pos, cell_np, real_cut)
    cell = torch.tensor(cell_np)
    sf = _sf(q, mu)
    pos_t = torch.tensor(pos, requires_grad=True)
    Q_t = torch.tensor(Q, requires_grad=True)

    E_dk = float(
        multipole_ewald_summation(
            torch.tensor(pos),
            _mm(q, mu, Q),
            cell,
            idx_j,
            ptr,
            sh,
            sigma=sigma,
            alpha=alpha,
            k_cutoff=kcut,
        ).sum()
    )
    E_pme = multipole_particle_mesh_ewald(
        pos_t,
        torch.cat([sf, cartesian_quadrupole_to_e3nn(Q_t)], dim=-1),
        cell,
        idx_j,
        ptr,
        sh,
        sigma=sigma,
        alpha=alpha,
        mesh_dimensions=(48, 48, 48),
    )
    assert abs(float(E_pme.sum()) - E_dk) / abs(E_dk) < 2e-4
    gpos, gQ = torch.autograd.grad(E_pme.sum(), [pos_t, Q_t])
    assert torch.isfinite(gpos).all() and torch.isfinite(gQ).all()
    assert (gQ - gQ.transpose(-1, -2)).abs().max() < 1e-10

    # A single-system batch reproduces the single-system composite bit-for-bit
    # (energy + forces + ∂E/∂Q).
    bidx = torch.zeros(pos.shape[0], dtype=torch.int32)
    pos_b = torch.tensor(pos, requires_grad=True)
    Q_b = torch.tensor(Q, requires_grad=True)
    E_batch = multipole_particle_mesh_ewald(
        pos_b,
        torch.cat([sf, cartesian_quadrupole_to_e3nn(Q_b)], dim=-1),
        cell.reshape(1, 3, 3),
        idx_j,
        ptr,
        sh,
        sigma=sigma,
        alpha=alpha,
        mesh_dimensions=(48, 48, 48),
        batch_idx=bidx,
    )
    assert E_batch.shape == (pos.shape[0],)
    torch.testing.assert_close(
        E_batch.sum(), E_pme.detach().sum(), rtol=1e-9, atol=1e-9
    )
    gpos_b, gQ_b = torch.autograd.grad(E_batch.sum(), [pos_b, Q_b])
    torch.testing.assert_close(gpos_b, gpos, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(gQ_b, gQ, rtol=1e-8, atol=1e-8)


def test_composite_quadrupole_autograd_connected():
    """Composite l=2 is finite + autograd-connected on all moment channels."""
    sigma, alpha = 0.5, 0.45
    cell_np, pos, q, mu, Q, L = _fixture()
    idx_j, ptr, sh = _build_csr(pos, cell_np, 12.0)
    cell = torch.tensor(cell_np)
    pos_t = torch.tensor(pos, requires_grad=True)
    sf = _sf(q, mu).requires_grad_(True)
    Q_t = torch.tensor(Q, requires_grad=True)
    mm = torch.cat([sf, cartesian_quadrupole_to_e3nn(Q_t)], dim=-1)
    E = multipole_ewald_summation(
        pos_t,
        mm,
        cell,
        idx_j,
        ptr,
        sh,
        sigma=sigma,
        alpha=alpha,
        k_cutoff=12.0,
    )
    assert torch.isfinite(E).all()
    gpos, gsf, gQ = torch.autograd.grad(E.sum(), [pos_t, sf, Q_t])
    assert torch.isfinite(gpos).all()
    assert torch.isfinite(gsf).all()
    assert torch.isfinite(gQ).all()
    assert (gQ - gQ.transpose(-1, -2)).abs().max() < 1e-12


def _fd_cell_grad(energy_fn, cell_np, h=1e-6):
    """Central-difference ∂E/∂cell (3, 3)."""
    fd = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            cp = cell_np.copy()
            cp[a, b] += h
            cm = cell_np.copy()
            cm[a, b] -= h
            fd[a, b] = (float(energy_fn(cp)) - float(energy_fn(cm))) / (2 * h)
    return fd


@pytest.mark.parametrize("with_quad", [False, True])
def test_composite_stress_matches_fd(with_quad):
    """∂E/∂cell through the full composite Ewald (real + direct-k recip − self)
    matches FD — l≤1 and l=2, single-system."""
    sigma, alpha, real_cut, kcut = 0.5, 0.45, 9.0, 9.0
    cell_np, pos, q, mu, Q, L = _fixture(7)
    idx_j, ptr, sh = _build_csr(pos, cell_np, real_cut)
    mm = _mm(q, mu, Q if with_quad else None)
    kw = dict(sigma=sigma, alpha=alpha, k_cutoff=kcut)

    def energy(cell_t):
        return multipole_ewald_summation(
            torch.tensor(pos), mm, cell_t, idx_j, ptr, sh, **kw
        )

    cell = torch.tensor(cell_np, requires_grad=True)
    (gcell,) = torch.autograd.grad(energy(cell).sum(), [cell])
    fd = _fd_cell_grad(lambda c: energy(torch.tensor(c)).sum(), cell_np)
    rel = np.abs(gcell.numpy() - fd).max() / (np.abs(fd).max() + 1e-30)
    assert rel < 5e-5, f"stress rel vs FD = {rel:.3e}"


@pytest.mark.parametrize("with_quad", [False, True])
def test_batched_stress_matches_per_system(with_quad):
    """Batched composite stress equals the per-system single-system stress
    (l≤1 and l=2)."""
    sigma, alpha, real_cut, kcut = 0.5, 0.45, 9.0, 9.0
    systems = [_fixture(11), _fixture(22)]

    # per-system single stress
    single_g = []
    for cell_np, pos, q, mu, Q, L in systems:
        idx_j, ptr, sh = _build_csr(pos, cell_np, real_cut)
        kw = dict(sigma=sigma, alpha=alpha, k_cutoff=kcut)
        ct = torch.tensor(cell_np, requires_grad=True)
        mm = _mm(q, mu, Q if with_quad else None)
        E = multipole_ewald_summation(torch.tensor(pos), mm, ct, idx_j, ptr, sh, **kw)
        single_g.append(torch.autograd.grad(E.sum(), [ct])[0].numpy())

    # batched
    pos_all, q_all, mu_all, Q_all, cells, bidx = [], [], [], [], [], []
    idxj, ptr_all, sh_all, off, po = [], [0], [], 0, 0
    for b, (cell_np, pos, q, mu, Q, L) in enumerate(systems):
        ij, pt, s = _build_csr(pos, cell_np, real_cut)
        n = pos.shape[0]
        pos_all.append(pos)
        q_all.append(q)
        mu_all.append(mu)
        Q_all.append(Q)
        cells.append(cell_np)
        bidx += [b] * n
        idxj.append(ij.numpy() + off)
        ptr_all += (pt.numpy()[1:] + po).tolist()
        sh_all.append(s.numpy())
        off += n
        po += len(ij)
    mm = _mm(
        np.concatenate(q_all),
        np.concatenate(mu_all),
        np.concatenate(Q_all) if with_quad else None,
    )
    cells_t = torch.tensor(np.stack(cells), requires_grad=True)
    kw = dict(
        sigma=sigma,
        alpha=alpha,
        k_cutoff=kcut,
        batch_idx=torch.tensor(bidx, dtype=torch.int32),
    )
    Eb = multipole_ewald_summation(
        torch.tensor(np.concatenate(pos_all)),
        mm,
        cells_t,
        torch.tensor(np.concatenate(idxj), dtype=torch.int32),
        torch.tensor(ptr_all, dtype=torch.int32),
        torch.tensor(np.concatenate(sh_all), dtype=torch.int32),
        **kw,
    )
    (gb,) = torch.autograd.grad(Eb.sum(), [cells_t])
    for b in range(2):
        rel = np.abs(gb.numpy()[b] - single_g[b]).max() / (
            np.abs(single_g[b]).max() + 1e-30
        )
        assert rel < 1e-9, f"system {b}: batched stress rel = {rel:.3e}"


def test_reciprocal_quadrupole_pos_hvp_matches_fd():
    """create_graph=True pos-HVP through the l=2 reciprocal (force-loss with Q
    present) matches a central-difference HVP."""
    sigma, alpha, kcut = 0.5, 0.45, 8.0
    cell_np, pos, q, mu, Q, L = _fixture(seed=3)
    cell = torch.tensor(cell_np)
    mm = _mm(q, mu, Q)
    rng = np.random.default_rng(1)
    v = torch.tensor(rng.normal(size=pos.shape))

    def grad_pos(p, create):
        pt = torch.tensor(p, requires_grad=True)
        E = multipole_reciprocal_space_energy(
            pt, mm, cell, sigma=sigma, alpha=alpha, k_cutoff=kcut
        )
        return torch.autograd.grad(E.sum(), [pt], create_graph=create)[0]

    pt = torch.tensor(pos, requires_grad=True)
    E = multipole_reciprocal_space_energy(
        pt, mm, cell, sigma=sigma, alpha=alpha, k_cutoff=kcut
    )
    gp = torch.autograd.grad(E.sum(), [pt], create_graph=True)[0]
    hvp = torch.autograd.grad((gp * v).sum(), [pt])[0]

    h = 1e-6
    fd = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for d in range(3):
            pp = pos.copy()
            pp[i, d] += h
            pm = pos.copy()
            pm[i, d] -= h
            fd[i, d] = float(
                ((grad_pos(pp, False) - grad_pos(pm, False)) * v).sum()
            ) / (2 * h)
    rel = np.abs(hvp.detach().numpy() - fd).max() / (np.abs(fd).max() + 1e-30)
    assert rel < 1e-4, f"pos-HVP rel = {rel:.3e}"


def test_reciprocal_quadrupole_q_hvp_matches_fd():
    """create_graph=True Q-HVP (and mixed ∂²E/∂r∂Q) through the l=2 reciprocal
    match finite differences."""
    sigma, alpha, kcut = 0.5, 0.45, 8.0
    cell_np, pos, q, mu, Q, L = _fixture(seed=3)
    cell = torch.tensor(cell_np)
    sf = _sf(q, mu)
    rng = np.random.default_rng(2)
    vQr = rng.normal(size=Q.shape)
    vQ = torch.tensor(0.5 * (vQr + vQr.transpose(0, 2, 1)))

    def grad_Q(Qm):
        Qt = torch.tensor(Qm, requires_grad=True)
        E = multipole_reciprocal_space_energy(
            torch.tensor(pos),
            torch.cat([sf, cartesian_quadrupole_to_e3nn(Qt)], dim=-1),
            cell,
            sigma=sigma,
            alpha=alpha,
            k_cutoff=kcut,
        )
        return torch.autograd.grad(E.sum(), [Qt], create_graph=True)[0]

    Qt = torch.tensor(Q, requires_grad=True)
    E = multipole_reciprocal_space_energy(
        torch.tensor(pos),
        torch.cat([sf, cartesian_quadrupole_to_e3nn(Qt)], dim=-1),
        cell,
        sigma=sigma,
        alpha=alpha,
        k_cutoff=kcut,
    )
    gQ = torch.autograd.grad(E.sum(), [Qt], create_graph=True)[0]
    qhvp = torch.autograd.grad((gQ * vQ).sum(), [Qt])[0]

    h = 1e-6
    fd = np.zeros_like(Q)
    for i in range(Q.shape[0]):
        for a in range(3):
            for b in range(3):
                Qp = Q.copy()
                Qp[i, a, b] += h
                Qm = Q.copy()
                Qm[i, a, b] -= h
                fd[i, a, b] = float(((grad_Q(Qp) - grad_Q(Qm)) * vQ).sum()) / (2 * h)
    rel = np.abs(qhvp.detach().numpy() - fd).max() / (np.abs(fd).max() + 1e-30)
    assert rel < 1e-4, f"Q-HVP rel = {rel:.3e}"


def test_reciprocal_quadrupole_batched_hvp_matches_per_system():
    """Batched l=2 reciprocal pos-HVP + Q-HVP equal the per-system single-system
    HVPs (validates the batched K_a/K_b/K_c indexing). Same cell per system so
    the per-system k-grids align with the batched (padded) grid."""
    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
        prepare_multipole_scf_cache,
    )
    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
        multipole_scf_step_energy,
    )

    sigma, alpha, kcut, rs = 0.5, 0.45, 8.0, [0.8]
    L = 6.0
    cell_np = np.eye(3) * L
    rng = np.random.default_rng(7)
    Ns = [4, 3]

    def mk(n):
        p = rng.uniform(0, L, (n, 3))
        qq = rng.normal(size=n)
        qq -= qq.mean()
        m = rng.normal(size=(n, 3))
        Qr = rng.normal(size=(n, 3, 3))
        return (
            p,
            np.concatenate([qq[:, None], m[:, [1, 2, 0]]], 1),
            0.5 * (Qr + Qr.transpose(0, 2, 1)),
        )

    sysd = [mk(n) for n in Ns]
    pos_all = np.concatenate([s[0] for s in sysd])
    sf_all = np.concatenate([s[1] for s in sysd])
    Q_all = np.concatenate([s[2] for s in sysd])
    bidx = np.concatenate([[b] * Ns[b] for b in range(len(Ns))]).astype(np.int32)

    cacheb = prepare_multipole_scf_cache(
        torch.tensor(np.stack([cell_np] * len(Ns))),
        sigma=sigma,
        receiver_sigmas=rs,
        k_cutoff=kcut,
        l_max=2,
        alpha=alpha,
    )
    ptb = torch.tensor(pos_all, requires_grad=True)
    Qtb = torch.tensor(Q_all, requires_grad=True)
    Eb = multipole_scf_step_energy(
        cacheb,
        ptb,
        torch.tensor(sf_all),
        batch_idx=torch.tensor(bidx),
        include_self_interaction=True,
        quadrupoles=Qtb,
    )
    gpb, gQb = torch.autograd.grad(Eb.sum(), [ptb, Qtb], create_graph=True)
    vpos = torch.tensor(rng.normal(size=pos_all.shape))
    vQr = rng.normal(size=Q_all.shape)
    vQ = torch.tensor(0.5 * (vQr + vQr.transpose(0, 2, 1)))
    poshvp_b = torch.autograd.grad((gpb * vpos).sum(), [ptb], retain_graph=True)[0]
    qhvp_b = torch.autograd.grad((gQb * vQ).sum(), [Qtb], retain_graph=True)[0]

    off = 0
    for b, (p, sf, Q) in enumerate(sysd):
        n = Ns[b]
        sl = slice(off, off + n)
        off += n
        cache = prepare_multipole_scf_cache(
            torch.tensor(cell_np),
            sigma=sigma,
            receiver_sigmas=rs,
            k_cutoff=kcut,
            l_max=2,
            alpha=alpha,
        )
        pt = torch.tensor(p, requires_grad=True)
        Qt = torch.tensor(Q, requires_grad=True)
        E = multipole_scf_step_energy(
            cache, pt, torch.tensor(sf), include_self_interaction=True, quadrupoles=Qt
        )
        gp, gQ = torch.autograd.grad(E.sum(), [pt, Qt], create_graph=True)
        ph = torch.autograd.grad((gp * vpos[sl]).sum(), [pt], retain_graph=True)[0]
        qh = torch.autograd.grad((gQ * vQ[sl]).sum(), [Qt], retain_graph=True)[0]
        assert (ph - poshvp_b[sl].detach()).abs().max() < 1e-12
        assert (qh - qhvp_b[sl].detach()).abs().max() < 1e-12


# =============================================================================
# Stress-loss (create_graph through dE/dcell): the S5 regression column.
# =============================================================================
#
# Guards ∂²E/∂cell∂pos on the direct-k reciprocal path (single + batched,
# l=0/1/2) against a PURE double finite-difference of the ENERGY — the
# value-only oracle that exposed the original silent-wrong stress-loss (a
# self-consistent autograd stress can still disagree with the energy's true
# second derivative). "StressLoss" in the name auto-marks these slow.


def _recip_stress_dot(positions, mm, cell, g, *, batch_idx=None):
    """``⟨g, dE/dcell⟩`` with ``create_graph`` so it is differentiable in pos."""
    cell = cell.clone().requires_grad_(True)
    e = multipole_reciprocal_space_energy(
        positions,
        mm,
        cell,
        batch_idx=batch_idx,
        sigma=0.5,
        alpha=0.45,
        k_cutoff=9.0,
    )
    (stress,) = torch.autograd.grad(e.sum(), cell, create_graph=True)
    return (stress * g).sum()


class TestDirectKGatherPerAtom:
    r"""Per-atom direct-k reciprocal energy via the spread-transpose gather.

    ``multipole_rho_gather_t`` decomposes the collective ``E = scale·Σ_k 2 f_k
    |ρ|²`` into ``E_i = scale·m_i·(Sᵀ·2 f_k ρ)_i`` (``Sᵀ`` = the rho-assembly's
    own transpose). The decomposition is bit-identical to the collective energy
    and carries the same first- and (force-loss / stress) second-order grads.
    """

    @pytest.fixture
    def gpu_system(self):
        if not torch.cuda.is_available():
            pytest.skip("Path A reciprocal kernels are GPU-only")
        positions, charges, dipoles, cell = _system(
            n_atoms=8, box_len=5.0, device="cuda:0", seed=0xBEEF, with_dipoles=True
        )
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
            prepare_multipole_scf_cache,
        )

        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[1.0],
            k_cutoff=3.0,
            l_max=1,
            alpha=0.5,
            device="cuda:0",
        )
        return {
            "positions": positions,
            "charges": charges,
            "dipoles": dipoles,
            "cell": cell,
            "cache": cache,
        }

    @staticmethod
    def _peratom(cache, pos, q, dip):
        import nvalchemiops.torch.interactions.electrostatics.multipole_autograd  # noqa: F401
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
            _TWO_PI_6,
        )

        rho = torch.ops.nvalchemiops.multipole_rho(
            q, dip, pos, cache.source_phi_hat, cache.k_vectors, cache.volume
        )
        rho = rho * (cache.volume.detach() / cache.volume)
        phi_hat = (2.0 * cache.per_k_factor).unsqueeze(-1) * rho
        g = torch.ops.nvalchemiops.multipole_rho_gather_t(
            phi_hat, pos, cache.source_phi_hat, cache.k_vectors, cache.volume
        )
        # Restore the moment-grad's detached 1/V so the volume cell-grad is exact.
        g = g * (cache.volume.detach() / cache.volume)
        scale = 0.5 * cache.volume / _TWO_PI_6
        return scale * (q * g[:, 0] + (dip * g[:, [3, 1, 2]]).sum(-1))

    @staticmethod
    def _collective(cache, pos, q, dip):
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
            multipole_scf_step_energy,
        )

        return multipole_scf_step_energy(
            cache, pos, pack_charges_dipoles(q, dip), include_self_interaction=True
        )

    def test_value_parity(self, gpu_system):
        """Σ_i E_i bit-identical to the collective raw reciprocal energy."""
        sd = gpu_system
        e_i = self._peratom(sd["cache"], sd["positions"], sd["charges"], sd["dipoles"])
        e_coll = self._collective(
            sd["cache"], sd["positions"], sd["charges"], sd["dipoles"]
        )
        assert e_i.shape == (sd["positions"].shape[0],)
        assert e_coll.shape == (sd["positions"].shape[0],)
        torch.testing.assert_close(e_i.sum(), e_coll.sum(), rtol=1e-12, atol=1e-12)

    def test_force_loss_parity(self, gpu_system):
        """Force-loss grads (∂²E/∂r∂θ) match the collective path bit-for-bit."""
        sd = gpu_system
        wf = torch.randn_like(sd["positions"])

        def grads(efn):
            p = sd["positions"].clone().requires_grad_(True)
            q = sd["charges"].clone().requires_grad_(True)
            mu = sd["dipoles"].clone().requires_grad_(True)
            e = efn(sd["cache"], p, q, mu)
            gp = torch.autograd.grad(e.sum(), p, create_graph=True)[0]
            loss = (gp * wf).sum()
            return torch.autograd.grad(loss, (q, mu, p))

        for a, b in zip(grads(self._peratom), grads(self._collective)):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)

    def test_stress_parity(self, gpu_system):
        """∂E/∂cell matches the collective path bit-for-bit."""
        sd = gpu_system
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
            prepare_multipole_scf_cache,
        )

        def stress(efn):
            cell = sd["cell"].clone().requires_grad_(True)
            cache = prepare_multipole_scf_cache(
                cell,
                sigma=1.0,
                receiver_sigmas=[1.0],
                k_cutoff=3.0,
                l_max=1,
                alpha=0.5,
                device="cuda:0",
            )
            e = efn(cache, sd["positions"], sd["charges"], sd["dipoles"])
            return torch.autograd.grad(e.sum(), cell)[0]

        torch.testing.assert_close(
            stress(self._peratom), stress(self._collective), rtol=1e-9, atol=1e-9
        )


class TestDirectKGatherPerAtomLmax2:
    r"""Per-atom direct-k reciprocal energy at l=2 via the Q-channel gather.

    Adds the Cartesian-quadrupole channel ``multipole_rho_q_gather_t`` to the
    l<=1 ``multipole_rho_gather_t`` decomposition; ``Σ_i E_i`` and its
    force-loss / stress grads must match the collective
    ``multipole_scf_step_energy(..., quadrupoles=Q)`` bit-for-bit.
    """

    @pytest.fixture
    def gpu_system(self):
        if not torch.cuda.is_available():
            pytest.skip("Path A reciprocal kernels are GPU-only")
        positions, charges, dipoles, cell = _system(
            n_atoms=8, box_len=5.0, device="cuda:0", seed=0xC0DE, with_dipoles=True
        )
        rng = torch.Generator(device="cuda:0").manual_seed(0xC0DE)
        qf = torch.randn(8, 3, 3, dtype=torch.float64, device="cuda:0", generator=rng)
        quadrupoles = 0.5 * (qf + qf.mT)
        return {
            "positions": positions,
            "charges": charges,
            "dipoles": dipoles,
            "quadrupoles": quadrupoles,
            "cell": cell,
        }

    @staticmethod
    def _mkcache(cell):
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
            prepare_multipole_scf_cache,
        )

        return prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[1.0],
            k_cutoff=3.0,
            l_max=2,
            alpha=0.5,
            device="cuda:0",
        )

    @staticmethod
    def _peratom(cache, pos, q, dip, quad):
        import nvalchemiops.torch.interactions.electrostatics.multipole_autograd  # noqa: F401
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
            _TWO_PI_6,
        )

        rho = torch.ops.nvalchemiops.multipole_rho(
            q, dip, pos, cache.source_phi_hat, cache.k_vectors, cache.volume
        )
        rho = rho + torch.ops.nvalchemiops.multipole_rho_q(
            quad, pos, cache.source_coeff2, cache.k_vectors, cache.volume
        )
        rho = rho * (cache.volume.detach() / cache.volume)
        phi_hat = (2.0 * cache.per_k_factor).unsqueeze(-1) * rho
        g = torch.ops.nvalchemiops.multipole_rho_gather_t(
            phi_hat, pos, cache.source_phi_hat, cache.k_vectors, cache.volume
        )
        g_q = torch.ops.nvalchemiops.multipole_rho_q_gather_t(
            phi_hat, pos, cache.source_coeff2, cache.k_vectors, cache.volume
        )
        g = g * (cache.volume.detach() / cache.volume)
        g_q = g_q * (cache.volume.detach() / cache.volume)
        scale = 0.5 * cache.volume / _TWO_PI_6
        return scale * (
            q * g[:, 0] + (dip * g[:, [3, 1, 2]]).sum(-1) + (quad * g_q).sum((-1, -2))
        )

    @staticmethod
    def _collective(cache, pos, q, dip, quad):
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
            multipole_scf_step_energy,
        )

        return multipole_scf_step_energy(
            cache,
            pos,
            pack_multipole_moments(q, dip),
            quadrupoles=quad,
            include_self_interaction=True,
        )

    def test_value_parity(self, gpu_system):
        sd = gpu_system
        cache = self._mkcache(sd["cell"])
        e_i = self._peratom(
            cache, sd["positions"], sd["charges"], sd["dipoles"], sd["quadrupoles"]
        )
        e_coll = self._collective(
            cache, sd["positions"], sd["charges"], sd["dipoles"], sd["quadrupoles"]
        )
        assert e_i.shape == (sd["positions"].shape[0],)
        assert e_coll.shape == (sd["positions"].shape[0],)
        torch.testing.assert_close(e_i.sum(), e_coll.sum(), rtol=1e-12, atol=1e-12)

    def test_force_loss_parity(self, gpu_system):
        sd = gpu_system
        cache = self._mkcache(sd["cell"])
        wf = torch.randn_like(sd["positions"])

        def grads(efn):
            p = sd["positions"].clone().requires_grad_(True)
            q = sd["charges"].clone().requires_grad_(True)
            mu = sd["dipoles"].clone().requires_grad_(True)
            quad = sd["quadrupoles"].clone().requires_grad_(True)
            e = efn(cache, p, q, mu, quad)
            gp = torch.autograd.grad(e.sum(), p, create_graph=True)[0]
            loss = (gp * wf).sum()
            return torch.autograd.grad(loss, (q, mu, quad, p))

        for a, b in zip(grads(self._peratom), grads(self._collective)):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)

    def test_stress_parity(self, gpu_system):
        sd = gpu_system

        def stress(efn):
            cell = sd["cell"].clone().requires_grad_(True)
            cache = self._mkcache(cell)
            e = efn(
                cache, sd["positions"], sd["charges"], sd["dipoles"], sd["quadrupoles"]
            )
            return torch.autograd.grad(e.sum(), cell)[0]

        torch.testing.assert_close(
            stress(self._peratom), stress(self._collective), rtol=1e-9, atol=1e-9
        )


class TestDirectKGatherPerAtomBatched:
    r"""Batched per-atom direct-k reciprocal energy (l=2) via the batched gathers.

    ``batch_multipole_rho_gather_t`` + ``batch_multipole_rho_q_gather_t``
    decompose the collective batched ``multipole_scf_step_energy`` into per-atom
    ``(N_total,)`` energies; ``Σ`` per system and the force-loss / stress grads
    must match the collective path bit-for-bit.
    """

    @staticmethod
    def _build(device):
        import numpy as np

        sigma, alpha, kcut, rs = 0.5, 0.45, 8.0, [0.8]
        ell = 6.0
        rng = np.random.default_rng(7)
        ns = [4, 3]
        cell_np = np.eye(3) * ell

        def mk(n):
            p = rng.uniform(0, ell, (n, 3))
            qq = rng.normal(size=n)
            qq -= qq.mean()
            m = rng.normal(size=(n, 3))
            qr = rng.normal(size=(n, 3, 3))
            return p, qq, m, 0.5 * (qr + qr.transpose(0, 2, 1))

        sysd = [mk(n) for n in ns]
        pos = torch.tensor(np.concatenate([s[0] for s in sysd]), device=device)
        q = torch.tensor(np.concatenate([s[1] for s in sysd]), device=device)
        mu = torch.tensor(np.concatenate([s[2] for s in sysd]), device=device)
        quad = torch.tensor(np.concatenate([s[3] for s in sysd]), device=device)
        bidx = torch.tensor(
            np.concatenate([[b] * ns[b] for b in range(len(ns))]).astype(np.int32),
            device=device,
        )
        cells_np = np.stack([cell_np] * len(ns))
        return {
            "pos": pos,
            "q": q,
            "mu": mu,
            "quad": quad,
            "bidx": bidx,
            "cells_np": cells_np,
            "B": len(ns),
            "sigma": sigma,
            "alpha": alpha,
            "kcut": kcut,
            "rs": rs,
        }

    @staticmethod
    def _mkcache(sd, cells):
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
            prepare_multipole_scf_cache,
        )

        return prepare_multipole_scf_cache(
            cells,
            sigma=sd["sigma"],
            receiver_sigmas=sd["rs"],
            k_cutoff=sd["kcut"],
            l_max=2,
            alpha=sd["alpha"],
            device=cells.device,
        )

    @staticmethod
    def _peratom(sd, cache, pos, q, mu, quad):
        import nvalchemiops.torch.interactions.electrostatics.multipole_autograd  # noqa: F401
        import nvalchemiops.torch.interactions.electrostatics.multipole_autograd_batch  # noqa: F401,E501
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
            _TWO_PI_6,
        )

        bidx = sd["bidx"]
        bl = bidx.long()
        B = sd["B"]
        rho = torch.ops.nvalchemiops.batch_multipole_rho(
            q, mu, pos, cache.source_phi_hat, cache.k_vectors, cache.volume, bidx
        )
        rho = rho + torch.ops.nvalchemiops.batch_multipole_rho_q(
            quad, pos, cache.source_coeff2, cache.k_vectors, cache.volume, bidx
        )
        volf = cache.volume.detach() / cache.volume
        rho = rho * volf.reshape(B, 1, 1)
        phi_hat = (2.0 * cache.per_k_factor).unsqueeze(-1) * rho
        g = torch.ops.nvalchemiops.batch_multipole_rho_gather_t(
            phi_hat, pos, cache.source_phi_hat, cache.k_vectors, cache.volume, bidx
        )
        g_q = torch.ops.nvalchemiops.batch_multipole_rho_q_gather_t(
            phi_hat, pos, cache.source_coeff2, cache.k_vectors, cache.volume, bidx
        )
        g = g * volf.index_select(0, bl).reshape(-1, 1)
        g_q = g_q * volf.index_select(0, bl).reshape(-1, 1, 1)
        scale = (0.5 * cache.volume / _TWO_PI_6).index_select(0, bl)
        return scale * (
            q * g[:, 0] + (mu * g[:, [3, 1, 2]]).sum(-1) + (quad * g_q).sum((-1, -2))
        )

    @staticmethod
    def _collective(sd, cache, pos, q, mu, quad):
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
            multipole_scf_step_energy,
        )

        sf = torch.cat([q[:, None], mu[:, [1, 2, 0]]], 1)
        return multipole_scf_step_energy(
            cache,
            pos,
            sf,
            batch_idx=sd["bidx"],
            quadrupoles=quad,
            include_self_interaction=True,
        )

    @pytest.fixture
    def sd(self):
        if not torch.cuda.is_available():
            pytest.skip("Path A reciprocal kernels are GPU-only")
        return self._build("cuda:0")

    def test_value_parity(self, sd):
        cells = torch.tensor(sd["cells_np"], device="cuda:0")
        cache = self._mkcache(sd, cells)
        e_i = self._peratom(sd, cache, sd["pos"], sd["q"], sd["mu"], sd["quad"])
        e_coll = self._collective(sd, cache, sd["pos"], sd["q"], sd["mu"], sd["quad"])
        assert e_i.shape == (sd["pos"].shape[0],)
        assert e_coll.shape == (sd["pos"].shape[0],)
        per_sys_i = torch.zeros(
            sd["B"], dtype=torch.float64, device="cuda:0"
        ).scatter_add(0, sd["bidx"].long(), e_i)
        per_sys_coll = torch.zeros(
            sd["B"], dtype=torch.float64, device="cuda:0"
        ).scatter_add(0, sd["bidx"].long(), e_coll)
        torch.testing.assert_close(per_sys_i, per_sys_coll, rtol=1e-12, atol=1e-12)

    def test_force_loss_parity(self, sd):
        cells = torch.tensor(sd["cells_np"], device="cuda:0")
        cache = self._mkcache(sd, cells)
        wf = torch.randn_like(sd["pos"])

        def grads(efn):
            p = sd["pos"].clone().requires_grad_(True)
            q = sd["q"].clone().requires_grad_(True)
            mu = sd["mu"].clone().requires_grad_(True)
            quad = sd["quad"].clone().requires_grad_(True)
            e = efn(sd, cache, p, q, mu, quad)
            gp = torch.autograd.grad(e.sum(), p, create_graph=True)[0]
            return torch.autograd.grad((gp * wf).sum(), (q, mu, quad, p))

        for a, b in zip(grads(self._peratom), grads(self._collective)):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)

    def test_stress_parity(self, sd):
        def stress(efn):
            cells = torch.tensor(sd["cells_np"], device="cuda:0").requires_grad_(True)
            cache = self._mkcache(sd, cells)
            e = efn(sd, cache, sd["pos"], sd["q"], sd["mu"], sd["quad"])
            return torch.autograd.grad(e.sum(), cells)[0]

        torch.testing.assert_close(
            stress(self._peratom), stress(self._collective), rtol=1e-9, atol=1e-9
        )


class TestReciprocalStressLoss:
    """∂²E/∂cell∂pos on the direct-k reciprocal path matches a double-FD of E."""

    @pytest.mark.parametrize("level", [0, 1, 2])
    def test_stress_loss_fd_single(self, level):
        cell_np, pos, q, mu, Q, _ = _fixture()
        if level == 0:
            mu = np.zeros_like(mu)
        mm = _mm(q, mu, Q if level == 2 else None)
        cell0 = torch.tensor(cell_np)
        g = torch.tensor(
            np.random.default_rng(level).normal(size=(3, 3)), dtype=torch.float64
        )
        p = torch.tensor(pos, requires_grad=True)
        (gp,) = torch.autograd.grad(_recip_stress_dot(p, mm, cell0, g), p)
        assert gp.norm() > 1e-6  # not a silent zero

        # double finite-difference of the energy: d/dpos ( d/dcell ⟨g, E⟩ ).
        def stress_g_fd(p_):
            ec = 1e-5
            acc = 0.0
            for i in range(3):
                for j in range(3):
                    cp = cell0.clone()
                    cp[i, j] += ec
                    cm = cell0.clone()
                    cm[i, j] -= ec
                    ep = multipole_reciprocal_space_energy(
                        p_, mm, cp, sigma=0.5, alpha=0.45, k_cutoff=9.0
                    )
                    em = multipole_reciprocal_space_energy(
                        p_, mm, cm, sigma=0.5, alpha=0.45, k_cutoff=9.0
                    )
                    acc += (
                        float(g[i, j]) * (float(ep.sum()) - float(em.sum())) / (2 * ec)
                    )
            return acc

        eps = 1e-4
        fd = torch.zeros_like(p)
        for a in range(p.shape[0]):
            for d in range(3):
                pp = p.detach().clone()
                pp[a, d] += eps
                pm = p.detach().clone()
                pm[a, d] -= eps
                fd[a, d] = (stress_g_fd(pp) - stress_g_fd(pm)) / (2 * eps)
        assert (gp - fd).norm() / fd.norm() < 1e-4

    @pytest.mark.parametrize("level", [0, 1, 2])
    def test_stress_loss_fd_batched(self, level):
        syss = [_fixture(seed=s) for s in (3, 7)]
        cells = np.stack([s[0] for s in syss])
        pos = np.concatenate([s[1] for s in syss])
        q = np.concatenate([s[2] for s in syss])
        mu = np.concatenate([s[3] for s in syss])
        Q = np.concatenate([s[4] for s in syss])
        n = syss[0][1].shape[0]
        batch_idx = torch.tensor([0] * n + [1] * n, dtype=torch.int32)
        if level == 0:
            mu = np.zeros_like(mu)
        mm = _mm(q, mu, Q if level == 2 else None)
        cell0 = torch.tensor(cells)
        g = torch.tensor(
            np.random.default_rng(level + 5).normal(size=cells.shape),
            dtype=torch.float64,
        )
        p = torch.tensor(pos, requires_grad=True)
        (gp,) = torch.autograd.grad(
            _recip_stress_dot(p, mm, cell0, g, batch_idx=batch_idx), p
        )
        assert gp.norm() > 1e-6

        def stress_g_fd(p_):
            ec = 1e-5
            acc = 0.0
            for b in range(2):
                for i in range(3):
                    for j in range(3):
                        cp = cell0.clone()
                        cp[b, i, j] += ec
                        cm = cell0.clone()
                        cm[b, i, j] -= ec
                        ep = multipole_reciprocal_space_energy(
                            p_,
                            mm,
                            cp,
                            batch_idx=batch_idx,
                            sigma=0.5,
                            alpha=0.45,
                            k_cutoff=9.0,
                        ).sum()
                        em = multipole_reciprocal_space_energy(
                            p_,
                            mm,
                            cm,
                            batch_idx=batch_idx,
                            sigma=0.5,
                            alpha=0.45,
                            k_cutoff=9.0,
                        ).sum()
                        acc += float(g[b, i, j]) * (float(ep) - float(em)) / (2 * ec)
            return acc

        eps = 1e-4
        fd = torch.zeros_like(p)
        for a in range(p.shape[0]):
            for d in range(3):
                pp = p.detach().clone()
                pp[a, d] += eps
                pm = p.detach().clone()
                pm[a, d] -= eps
                fd[a, d] = (stress_g_fd(pp) - stress_g_fd(pm)) / (2 * eps)
        assert (gp - fd).norm() / fd.norm() < 1e-4


class TestEwaldCompositeStressLoss:
    """``multipole_ewald_summation`` stress-loss (real-space + reciprocal + self).

    The S5 acceptance test: ``create_graph`` through ``dE/dcell`` on the full
    composite (previously raised in the reciprocal half). Guarded vs a double-FD
    of the composite energy, l=0/1/2.
    """

    @pytest.mark.parametrize("level", [0, 1, 2])
    def test_composite_stress_loss_fd(self, level):
        cell_np, pos, q, mu, Q, _ = _fixture()
        if level == 0:
            mu = np.zeros_like(mu)
        mm = _mm(q, mu, Q if level == 2 else None)
        idx_j, ptr, shifts = _build_csr(pos, cell_np, 5.0)
        cell0 = torch.tensor(cell_np)
        g = torch.tensor(
            np.random.default_rng(level + 2).normal(size=(3, 3)), dtype=torch.float64
        )

        def energy(p_, cell_):
            return multipole_ewald_summation(
                p_,
                mm,
                cell_,
                idx_j,
                ptr,
                shifts,
                sigma=0.5,
                alpha=0.45,
                k_cutoff=9.0,
            )

        def stress_dot(p_):
            cell = cell0.clone().requires_grad_(True)
            (stress,) = torch.autograd.grad(
                energy(p_, cell).sum(), cell, create_graph=True
            )
            return (stress * g).sum()

        p = torch.tensor(pos, requires_grad=True)
        (gp,) = torch.autograd.grad(stress_dot(p), p)
        assert gp.norm() > 1e-6

        def stress_g_fd(p_):
            ec = 1e-5
            acc = 0.0
            for i in range(3):
                for j in range(3):
                    cp = cell0.clone()
                    cp[i, j] += ec
                    cm = cell0.clone()
                    cm[i, j] -= ec
                    acc += (
                        float(g[i, j])
                        * (float(energy(p_, cp).sum()) - float(energy(p_, cm).sum()))
                        / (2 * ec)
                    )
            return acc

        eps = 1e-4
        fd = torch.zeros_like(p)
        for a in range(p.shape[0]):
            for d in range(3):
                pp = p.detach().clone()
                pp[a, d] += eps
                pm = p.detach().clone()
                pm[a, d] -= eps
                fd[a, d] = (stress_g_fd(pp) - stress_g_fd(pm)) / (2 * eps)
        assert (gp - fd).norm() / fd.norm() < 1e-4
