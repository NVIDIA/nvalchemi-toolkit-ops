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

"""Tests for multipole electrostatic feature extraction and the SCF cache/step API."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    MultipoleSCFCache,
    multipole_electrostatic_energy,
    multipole_electrostatic_features,
    multipole_scf_step_energy,
    multipole_scf_step_features,
    prepare_multipole_scf_cache,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    cartesian_quadrupole_to_e3nn,
    pack_charges_dipoles,
    pack_multipole_moments,
    split_packed_for_kernels,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.torch.math import FIELD_CONSTANT, compute_overlap_constants
from nvalchemiops.torch.math.gto import NormMode, inv_cl


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _random_system(
    *,
    seed: int = 0,
    n_atoms: int = 8,
    box_len: float = 6.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    with_dipoles: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, box_len, size=(n_atoms, 3))
    charges = rng.uniform(-1.0, 1.0, n_atoms)
    charges -= charges.mean()
    dipoles = rng.standard_normal((n_atoms, 3)) * 0.3
    cell = np.eye(3) * box_len
    charges_t = torch.from_numpy(charges).to(device=device, dtype=dtype)
    out = {
        "positions": torch.from_numpy(positions).to(device=device, dtype=dtype),
        "charges": charges_t,
        "cell": torch.from_numpy(cell).to(device=device, dtype=dtype),
        "positions_np": positions,
        "charges_np": charges,
        "cell_np": cell,
    }
    if with_dipoles:
        dipoles_t = torch.from_numpy(dipoles).to(device=device, dtype=dtype)
        out["dipoles"] = dipoles_t
        out["dipoles_np"] = dipoles
        out["source_feats"] = pack_charges_dipoles(charges_t, dipoles_t)
    else:
        out["dipoles"] = None
        out["dipoles_np"] = np.zeros_like(dipoles)
        out["source_feats"] = pack_charges_dipoles(charges_t, None)
    return out


class TestBasics:
    def test_permuted_shape(self, device):
        td = _torch_device(device)
        sys = _random_system(seed=0, n_atoms=4, device=td)
        feats = multipole_electrostatic_features(
            sys["positions"],
            sys["source_feats"],
            sys["cell"],
            sigma=1.0,
            receiver_sigmas=[0.5, 1.0, 1.5],
            k_cutoff=4.0,
        )
        assert feats.shape == (4, 3 * 4)
        assert feats.dtype == torch.float64
        assert feats.device.type == td

    def test_permuted_shape_multi_sigma(self, device):
        td = _torch_device(device)
        sys = _random_system(seed=1, n_atoms=4, device=td)
        feats = multipole_electrostatic_features(
            sys["positions"],
            sys["source_feats"],
            sys["cell"],
            sigma=1.0,
            receiver_sigmas=[0.5, 1.0],
            k_cutoff=4.0,
        )
        assert feats.shape == (4, 2 * 4)

    def test_charges_only_branch(self, device):
        td = _torch_device(device)
        sys = _random_system(seed=2, n_atoms=5, device=td, with_dipoles=False)
        feats = multipole_electrostatic_features(
            sys["positions"],
            sys["source_feats"],
            sys["cell"],
            sigma=1.0,
            receiver_sigmas=[0.8, 1.2],
            k_cutoff=4.0,
        )
        assert feats.shape == (5, 2 * 4)


class TestValidation:
    def test_rejects_empty_receiver_sigmas(self):
        with pytest.raises(ValueError, match="receiver_sigmas"):
            multipole_electrostatic_features(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                receiver_sigmas=[],
                k_cutoff=4.0,
            )

    def test_rejects_non_positive_receiver_sigma(self):
        with pytest.raises(ValueError, match="receiver_sigmas"):
            multipole_electrostatic_features(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                receiver_sigmas=[0.5, -0.1],
                k_cutoff=4.0,
            )

    def test_missing_both_k_cutoff_and_k_vectors(self):
        with pytest.raises(ValueError, match="k_vectors"):
            multipole_electrostatic_features(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                receiver_sigmas=[1.0],
            )


class TestPhysicalInvariants:
    def test_translation_invariance_of_l0_channel(self, device):
        """l=0 per-atom features are translation-invariant (only cos²+sin² appears)."""
        td = _torch_device(device)
        sys = _random_system(seed=5, n_atoms=6, box_len=5.0, device=td)
        shift = torch.tensor([1.3, -0.7, 2.1], dtype=torch.float64, device=td)
        receiver_sigmas = [0.7, 1.3]
        n_sigma = len(receiver_sigmas)

        f_before = multipole_electrostatic_features(
            sys["positions"],
            sys["source_feats"],
            sys["cell"],
            sigma=1.0,
            receiver_sigmas=receiver_sigmas,
            k_cutoff=3.5,
        )
        f_after = multipole_electrostatic_features(
            sys["positions"] + shift,
            sys["source_feats"],
            sys["cell"],
            sigma=1.0,
            receiver_sigmas=receiver_sigmas,
            k_cutoff=3.5,
        )
        # Columns [0, n_sigma) are the translation-invariant l=0 channels.
        np.testing.assert_allclose(
            f_before[:, :n_sigma].detach().cpu().numpy(),
            f_after[:, :n_sigma].detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-13,
        )

    def test_zero_source_moments_give_zero_features(self, device):
        """All-zero charges and dipoles → features identically zero."""
        td = _torch_device(device)
        sys = _random_system(seed=9, n_atoms=4, device=td)
        zero_source = torch.zeros_like(sys["source_feats"])
        feats = multipole_electrostatic_features(
            sys["positions"],
            zero_source,
            sys["cell"],
            sigma=1.0,
            receiver_sigmas=[0.7, 1.3],
            k_cutoff=3.5,
        )
        np.testing.assert_array_equal(
            feats.detach().cpu().numpy(), np.zeros((4, 2 * 4))
        )


_SIGMA = 1.0


_RSIG = [1.0, 1.5]


_KCUT = 4.0


_BOX = 5.0


def _rand_moments(n, seed, td, *, l2=True):
    rng = np.random.default_rng(seed)
    ch = rng.standard_normal(n)
    ch -= ch.mean()
    charges = torch.from_numpy(ch).to(td, torch.float64)
    dip = torch.from_numpy(rng.standard_normal((n, 3))).to(td, torch.float64)
    if not l2:
        return pack_multipole_moments(charges, dip)
    A = rng.standard_normal((n, 3, 3))
    Q = 0.5 * (A + A.transpose(0, 2, 1))
    tr = np.einsum("nii->n", Q) / 3.0
    for d in range(3):
        Q[:, d, d] -= tr
    quad = torch.from_numpy(Q).to(td, torch.float64)
    return pack_multipole_moments(charges, dip, quad)


def _system(n, seed, td, *, l2=True):
    rng = np.random.default_rng(seed + 999)
    pos = torch.from_numpy(rng.uniform(0.0, _BOX, (n, 3))).to(td, torch.float64)
    cell = torch.eye(3, dtype=torch.float64, device=td) * _BOX
    mm = _rand_moments(n, seed, td, l2=l2)
    return pos, cell, mm


def _features(pos, mm, cell, *, feature_max_l):
    return multipole_electrostatic_features(
        pos,
        mm,
        cell,
        sigma=_SIGMA,
        receiver_sigmas=_RSIG,
        k_cutoff=_KCUT,
        feature_max_l=feature_max_l,
    )


class TestFeaturesQuadrupole:
    def test_shape(self, device):
        td = _torch_device(device)
        pos, cell, mm = _system(5, 0, td)
        f = _features(pos, mm, cell, feature_max_l=2)
        assert f.shape == (5, len(_RSIG) * 9)
        assert f.dtype == torch.float64

    def test_decoupling_l1_block_unchanged(self, device):
        """feature_max_l=2's l≤1 columns equal the feature_max_l=1 output."""
        td = _torch_device(device)
        pos, cell, mm = _system(6, 1, td)
        f1 = _features(pos, mm, cell, feature_max_l=1)
        f2 = _features(pos, mm, cell, feature_max_l=2)
        nsig = len(_RSIG)
        torch.testing.assert_close(f2[:, : nsig * 4], f1, rtol=0, atol=1e-12)

    def test_l2_block_matches_numpy_projection(self, device):
        """The 5 l=2 channels match an independent numpy projection + self-subtract."""
        td = _torch_device(device)
        pos, cell, mm = _system(5, 2, td)
        nsig = len(_RSIG)
        f2 = _features(pos, mm, cell, feature_max_l=2).detach().cpu().numpy()

        # Reconstruct ρ/V, then project the l=2 receiver block in numpy and lay
        # out per the (σ, m) permuted convention.
        from nvalchemiops.torch.interactions.electrostatics.multipole_autograd import (
            MultipoleRhoFunction,
            MultipoleRhoQFunction,
            _compute_structure_factor_table,
            _l2_receiver_block,
        )
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
            prepare_multipole_scf_cache,
        )

        cache = prepare_multipole_scf_cache(
            cell,
            sigma=_SIGMA,
            receiver_sigmas=_RSIG,
            k_cutoff=_KCUT,
            l_max=2,
            feature_max_l=2,
        )
        src_l1, quad_cart, _ = split_packed_for_kernels(mm)
        ch = src_l1[:, 0].contiguous()
        dip = src_l1[:, [3, 1, 2]].contiguous()
        rho = MultipoleRhoFunction.apply(
            ch, dip, pos, cache.source_phi_hat, cache.k_vectors, cache
        )
        rho = rho + MultipoleRhoQFunction.apply(
            quad_cart, pos, cache.source_coeff2, cache.k_vectors, cache
        )
        V = (cache.per_k_factor.unsqueeze(-1) * rho).detach().cpu().numpy()

        cos, sin = _compute_structure_factor_table(pos, cache)
        cosn, sinn = cos.detach().cpu().numpy(), sin.detach().cpu().numpy()
        recv2 = _l2_receiver_block(cache).detach().cpu().numpy()
        kfp = cache.k_factor_proj.detach().cpu().numpy()
        inv = 1.0 / (2 * np.pi) ** 3
        a = V[:, None, None, 0] * recv2[..., 0] + V[:, None, None, 1] * recv2[..., 1]
        b = V[:, None, None, 0] * recv2[..., 1] - V[:, None, None, 1] * recv2[..., 0]
        raw_l2 = (
            2.0
            * inv
            * (
                np.einsum("k,ksm,ki->ism", kfp, a, cosn)
                + np.einsum("k,ksm,ki->ism", kfp, b, sinn)
            )
        )
        src_e3nn_l2 = (
            cartesian_quadrupole_to_e3nn(quad_cart.to(torch.float64))
            .detach()
            .cpu()
            .numpy()
        )
        foc2 = cache.feature_overlap_l2.detach().cpu().numpy()
        feat_l2 = raw_l2 - src_e3nn_l2[:, None, :] * foc2[None, :, None]

        off = nsig * 4
        ref = np.empty((pos.shape[0], 5 * nsig))
        for s in range(nsig):
            ref[:, s * 5 : (s + 1) * 5] = feat_l2[:, s, :]
        np.testing.assert_allclose(f2[:, off:], ref, rtol=0, atol=1e-9)

    def test_grads_match_fd(self, device):
        td = _torch_device(device)
        pos, cell, mm = _system(4, 3, td)
        w = torch.sin(
            torch.arange(4 * len(_RSIG) * 9, device=td, dtype=torch.float64)
        ).reshape(4, len(_RSIG) * 9)

        def loss(p, m):
            return (_features(p, m, cell, feature_max_l=2) * w).sum()

        pl = pos.clone().requires_grad_(True)
        ml = mm.clone().requires_grad_(True)
        loss(pl, ml).backward()
        gp, gm = pl.grad.clone(), ml.grad.clone()

        eps = 1e-6
        fdp = torch.zeros_like(gp)
        for i in range(pos.shape[0]):
            for d in range(3):
                pp = pos.clone()
                pp[i, d] += eps
                pmn = pos.clone()
                pmn[i, d] -= eps
                fdp[i, d] = (loss(pp, mm) - loss(pmn, mm)) / (2 * eps)
        torch.testing.assert_close(gp, fdp, rtol=2e-5, atol=1e-7)

        fdm = torch.zeros_like(gm)
        for i in range(pos.shape[0]):
            for c in range(9):
                mp = mm.clone()
                mp[i, c] += eps
                mn = mm.clone()
                mn[i, c] -= eps
                fdm[i, c] = (loss(pos, mp) - loss(pos, mn)) / (2 * eps)
        torch.testing.assert_close(gm, fdm, rtol=2e-5, atol=1e-7)

    def test_batched_matches_single(self, device):
        td = _torch_device(device)
        pos0, cell, mm0 = _system(4, 4, td)
        pos1, _, mm1 = _system(6, 5, td)
        posb = torch.cat([pos0, pos1])
        mmb = torch.cat([mm0, mm1])
        bidx = torch.cat(
            [
                torch.zeros(4, dtype=torch.int32, device=td),
                torch.ones(6, dtype=torch.int32, device=td),
            ]
        )
        cellb = torch.stack([cell, cell])
        fb = multipole_electrostatic_features(
            posb,
            mmb,
            cellb,
            batch_idx=bidx,
            sigma=_SIGMA,
            receiver_sigmas=_RSIG,
            k_cutoff=_KCUT,
            feature_max_l=2,
        )
        f0 = _features(pos0, mm0, cell, feature_max_l=2)
        f1 = _features(pos1, mm1, cell, feature_max_l=2)
        torch.testing.assert_close(fb[:4], f0, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(fb[4:], f1, rtol=1e-10, atol=1e-12)

    def test_force_loss_create_graph(self, device):
        """create_graph=True force-loss grad w.r.t. moments must match FD."""
        td = _torch_device(device)
        pos, cell, mm = _system(4, 7, td)
        n = pos.shape[0]
        w = torch.sin(
            torch.arange(n * len(_RSIG) * 9, device=td, dtype=torch.float64)
        ).reshape(n, len(_RSIG) * 9)

        def floss(m):
            p = pos.clone().requires_grad_(True)
            f = _features(p, m, cell, feature_max_l=2)
            (forces,) = torch.autograd.grad((f * w).sum(), p, create_graph=True)
            return (forces**2).sum()

        ml = mm.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(floss(ml), ml)
        eps = 1e-6
        fd = torch.zeros_like(g)
        for i in range(n):
            for c in range(9):
                mp = mm.clone()
                mp[i, c] += eps
                mn = mm.clone()
                mn[i, c] -= eps
                fd[i, c] = (floss(mp).item() - floss(mn).item()) / (2 * eps)
        torch.testing.assert_close(g, fd, rtol=3e-5, atol=1e-6)

    def test_force_loss_create_graph_batched(self, device):
        """Batched analog of :meth:`test_force_loss_create_graph`."""
        td = _torch_device(device)
        pos0, cell, mm0 = _system(3, 8, td)
        pos1, _, mm1 = _system(4, 9, td)
        pos = torch.cat([pos0, pos1])
        mm = torch.cat([mm0, mm1])
        bidx = torch.cat(
            [
                torch.zeros(3, dtype=torch.int32, device=td),
                torch.ones(4, dtype=torch.int32, device=td),
            ]
        )
        cellb = torch.stack([cell, cell])
        n = pos.shape[0]
        w = torch.cos(
            torch.arange(n * len(_RSIG) * 9, device=td, dtype=torch.float64)
        ).reshape(n, len(_RSIG) * 9)

        def floss(m):
            p = pos.clone().requires_grad_(True)
            f = multipole_electrostatic_features(
                p,
                m,
                cellb,
                batch_idx=bidx,
                sigma=_SIGMA,
                receiver_sigmas=_RSIG,
                k_cutoff=_KCUT,
                feature_max_l=2,
            )
            (forces,) = torch.autograd.grad((f * w).sum(), p, create_graph=True)
            return (forces**2).sum()

        ml = mm.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(floss(ml), ml)
        eps = 1e-6
        fd = torch.zeros_like(g)
        for i in range(n):
            for c in range(9):
                mp = mm.clone()
                mp[i, c] += eps
                mn = mm.clone()
                mn[i, c] -= eps
                fd[i, c] = (floss(mp).item() - floss(mn).item()) / (2 * eps)
        torch.testing.assert_close(g, fd, rtol=3e-5, atol=1e-6)

    def test_source_l1_with_feature_l2_decoupled(self, device):
        """feature_max_l=2 works with an l≤1 source (no quadrupole)."""
        td = _torch_device(device)
        pos, cell, mm_l1 = _system(5, 6, td, l2=False)
        f = _features(pos, mm_l1, cell, feature_max_l=2)
        assert f.shape == (5, len(_RSIG) * 9)
        off = len(_RSIG) * 4
        assert f[:, off:].abs().max() > 1e-6


def _cell(*, box_len: float, device: str) -> torch.Tensor:
    return torch.eye(3, dtype=torch.float64, device=device) * box_len


class TestCacheShapes:
    def test_shapes_and_metadata(self, device):
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.6, 1.0, 1.4],
            k_cutoff=3.5,
        )
        assert isinstance(cache, MultipoleSCFCache)
        assert cache.n_sigma == 3
        assert cache.l_max == 1
        assert cache.sigma == 1.0
        assert cache.receiver_sigmas == (0.6, 1.0, 1.4)
        assert cache.density_normalize == NormMode.MULTIPOLES
        assert cache.feature_normalize == NormMode.RECEIVER

        n_k = cache.n_k
        assert cache.k_vectors.shape == (n_k, 3)
        assert cache.k_norm2.shape == (n_k,)
        assert cache.source_phi_hat.shape == (n_k, 4, 2)
        assert cache.receiver_phi_hat.shape == (n_k, 3, 4, 2)
        assert cache.per_k_factor.shape == (n_k,)
        assert cache.k_factor_proj.shape == (n_k,)
        # [l=0, l=1, l=2] source self-overlap constants.
        assert cache.source_overlap_constants.shape == (3,)
        assert cache.feature_overlap_constants.shape == (3, 2)
        # Default feature_max_l=1 → no l=2 receiver block / self constant.
        assert cache.feature_max_l == 1
        assert cache.feature_overlap_l2 is None
        assert cache.out_col_lut_natural.shape == (3, 4)
        assert cache.out_col_lut_permuted.shape == (3, 4)
        assert cache.volume.shape == ()
        assert cache.device.type == td
        # The cache must not hold position-dependent cos/sin tables.
        assert not hasattr(cache, "cosines")
        assert not hasattr(cache, "sines")

    def test_lmax_zero_feature_oc_has_zero_l1_column(self, device):
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.8],
            k_cutoff=3.0,
            l_max=0,
        )
        np.testing.assert_array_equal(
            cache.feature_overlap_constants[:, 1].detach().cpu().numpy(),
            np.zeros(1),
        )
        assert float(cache.source_overlap_constants[1]) == 0.0

    def test_origin_is_row_zero_of_k_vectors(self, device):
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[1.0],
            k_cutoff=3.0,
        )
        np.testing.assert_array_equal(
            cache.k_vectors[0].detach().cpu().numpy(), np.zeros(3)
        )


class TestCacheContents:
    """The cache must be content-equivalent to what the one-shot binding would compute."""

    def test_per_k_factor_matches_closed_form(self, device):
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.0
        )
        k_norm2 = cache.k_norm2.detach().cpu().numpy()
        expected = np.zeros_like(k_norm2)
        mask = k_norm2 > 0.0
        expected[mask] = FIELD_CONSTANT / k_norm2[mask]
        np.testing.assert_allclose(
            cache.per_k_factor.detach().cpu().numpy(),
            expected,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_k_factor_proj_convention(self, device):
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.0
        )
        kfp = cache.k_factor_proj.detach().cpu().numpy()
        assert kfp[0] == 0.5
        assert np.all(kfp[1:] == 1.0)

    def test_feature_overlap_constants_match_host_computation(self, device):
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        sigma = 1.0
        receiver_sigmas = [0.7, 1.3]
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            k_cutoff=3.0,
        )
        oc_np = compute_overlap_constants(
            max_L=1,
            sigma_source=sigma,
            sigmas_receive=receiver_sigmas,
            normalize_source=NormMode.MULTIPOLES,
            normalize_receive=NormMode.RECEIVER,
        )
        np.testing.assert_allclose(
            cache.feature_overlap_constants.detach().cpu().numpy(),
            oc_np,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_receiver_phi_at_origin_zero_for_l1(self, device):
        """Receiver φ̂ at k = 0 has zero l=1 block (Y_1^m ∝ k̂ · k vanishes)."""
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.8, 1.2],
            k_cutoff=3.0,
        )
        phi_origin_l1 = cache.receiver_phi_hat[0, :, 1:4, :].detach().cpu().numpy()
        np.testing.assert_array_equal(phi_origin_l1, np.zeros_like(phi_origin_l1))

    def test_source_phi_hat_l0_matches_closed_form_at_origin(self, device):
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        sigma = 1.0
        cache = prepare_multipole_scf_cache(
            cell, sigma=sigma, receiver_sigmas=[1.0], k_cutoff=3.0
        )
        expected = (
            inv_cl(sigma, 0, NormMode.MULTIPOLES)
            * 4.0
            * math.pi
            * math.sqrt(math.pi / 2.0)
            * sigma**3
            / math.sqrt(4.0 * math.pi)
        )
        assert cache.source_phi_hat[0, 0, 0].item() == pytest.approx(
            expected, rel=1e-14
        )
        assert cache.source_phi_hat[0, 0, 1].item() == 0.0


class TestCacheReuse:
    def test_k_vectors_passed_through(self, device):
        """Supplying pre-built k_vectors produces a cache using exactly those."""
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        k_cutoff = 3.5
        k_half = generate_k_vectors_ewald_summation(cell, k_cutoff)
        k = torch.cat([k_half.new_zeros((1, 3)), k_half], dim=0).to(torch.float64)

        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[1.0],
            k_vectors=k,
        )
        np.testing.assert_array_equal(
            cache.k_vectors.detach().cpu().numpy(), k.detach().cpu().numpy()
        )


class TestSCFCacheValidation:
    def test_rejects_empty_receiver_sigmas(self):
        with pytest.raises(ValueError, match="receiver_sigmas"):
            prepare_multipole_scf_cache(
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                receiver_sigmas=[],
                k_cutoff=3.0,
            )

    def test_rejects_missing_k_cutoff_and_k_vectors(self):
        with pytest.raises(ValueError, match="k_vectors"):
            prepare_multipole_scf_cache(
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                receiver_sigmas=[1.0],
            )

    def test_rejects_bad_l_max(self):
        # l_max=2 is valid; l_max=3 remains unsupported.
        with pytest.raises(ValueError, match="l_max"):
            prepare_multipole_scf_cache(
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                receiver_sigmas=[1.0],
                k_cutoff=3.0,
                l_max=3,
            )

    def test_rejects_bad_cell_shape(self):
        with pytest.raises(ValueError, match="cell"):
            prepare_multipole_scf_cache(
                torch.eye(4, dtype=torch.float64),
                sigma=1.0,
                receiver_sigmas=[1.0],
                k_cutoff=3.0,
            )


def _scf_step_random_system(
    *, seed: int, n_atoms: int, box_len: float, device: str, with_dipoles: bool = True
):
    rng = np.random.default_rng(seed)
    positions = torch.from_numpy(rng.uniform(0.0, box_len, size=(n_atoms, 3))).to(
        device=device, dtype=torch.float64
    )
    charges_np = rng.uniform(-1.0, 1.0, n_atoms)
    charges_np -= charges_np.mean()
    charges = torch.from_numpy(charges_np).to(device=device, dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64, device=device) * box_len
    dipoles = None
    if with_dipoles:
        dipoles_np = rng.standard_normal((n_atoms, 3)) * 0.3
        dipoles = torch.from_numpy(dipoles_np).to(device=device, dtype=torch.float64)
    source_feats = pack_charges_dipoles(charges, dipoles)
    return positions, charges, dipoles, cell, source_feats


class TestEnergyStep:
    def test_shape_and_dtype(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _scf_step_random_system(
            seed=0, n_atoms=5, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.5
        )
        e = multipole_scf_step_energy(cache, positions, source_feats)
        assert e.shape == (5,)
        assert e.dtype == torch.float64
        assert e.device.type == td

    @pytest.mark.parametrize("seed", [0, 17, 42])
    @pytest.mark.parametrize("with_dipoles", [True, False])
    def test_parity_with_one_shot_energy(self, device, seed, with_dipoles):
        """scf_step_energy matches multipole_electrostatic_energy to float64 ULP."""
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _scf_step_random_system(
            seed=seed, n_atoms=6, box_len=5.0, device=td, with_dipoles=with_dipoles
        )
        sigma = 1.0
        k_cutoff = 3.5
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=[1.0],
            k_cutoff=k_cutoff,
            l_max=1 if with_dipoles else 0,
        )
        e_scf = multipole_scf_step_energy(cache, positions, source_feats)
        e_one_shot = multipole_electrostatic_energy(
            positions,
            source_feats,
            cell,
            sigma=sigma,
            k_cutoff=k_cutoff,
        )
        np.testing.assert_allclose(
            float(e_scf.sum()), float(e_one_shot.sum()), rtol=1e-14, atol=1e-14
        )

    def test_charges_only_matches_zero_dipole_branch(self, device):
        td = _torch_device(device)
        positions, charges, _, cell, source_feats_none = _scf_step_random_system(
            seed=11, n_atoms=5, box_len=5.0, device=td, with_dipoles=False
        )
        zero_d = torch.zeros((5, 3), dtype=torch.float64, device=td)
        source_feats_zero = pack_charges_dipoles(charges, zero_d)
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.5
        )
        e_none = multipole_scf_step_energy(cache, positions, source_feats_none)
        e_zero = multipole_scf_step_energy(cache, positions, source_feats_zero)
        assert torch.allclose(e_none, e_zero, rtol=1e-14, atol=1e-14)

    def test_multi_step_same_cache(self, device):
        """Different moments through one cache give different energies, each matching the one-shot call."""
        td = _torch_device(device)
        positions, _, _, cell, source_feats_a = _scf_step_random_system(
            seed=23, n_atoms=5, box_len=5.0, device=td
        )
        _, _, _, _, source_feats_b = _scf_step_random_system(
            seed=31, n_atoms=5, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.5
        )
        e_a_scf = multipole_scf_step_energy(cache, positions, source_feats_a)
        e_b_scf = multipole_scf_step_energy(cache, positions, source_feats_b)
        assert float(e_a_scf.sum()) != float(e_b_scf.sum())

        e_a_one = multipole_electrostatic_energy(
            positions,
            source_feats_a,
            cell,
            sigma=1.0,
            k_cutoff=3.5,
        )
        e_b_one = multipole_electrostatic_energy(
            positions,
            source_feats_b,
            cell,
            sigma=1.0,
            k_cutoff=3.5,
        )
        np.testing.assert_allclose(
            float(e_a_scf.sum()), float(e_a_one.sum()), rtol=1e-14, atol=1e-14
        )
        np.testing.assert_allclose(
            float(e_b_scf.sum()), float(e_b_one.sum()), rtol=1e-14, atol=1e-14
        )

    def test_rejects_wrong_n_atoms(self, device):
        td = _torch_device(device)
        positions, _, _, cell, _ = _scf_step_random_system(
            seed=0, n_atoms=4, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], k_cutoff=3.5
        )
        wrong_source = torch.zeros((3, 1), dtype=torch.float64, device=td)
        with pytest.raises(ValueError):
            multipole_scf_step_energy(cache, positions, wrong_source)


class TestFeatureStep:
    def test_shape_permuted(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _scf_step_random_system(
            seed=0, n_atoms=5, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.7, 1.3],
            k_cutoff=3.5,
        )
        feats = multipole_scf_step_features(cache, positions, source_feats)
        assert feats.shape == (5, 2 * 4)
        assert feats.dtype == torch.float64

    def test_shape_permuted_multi_sigma(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _scf_step_random_system(
            seed=1, n_atoms=4, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.8, 1.2],
            k_cutoff=3.5,
        )
        feats = multipole_scf_step_features(cache, positions, source_feats)
        assert feats.shape == (4, 2 * 4)

    @pytest.mark.parametrize("seed", [0, 17, 42])
    @pytest.mark.parametrize("with_dipoles", [True, False])
    def test_parity_with_one_shot_features(self, device, seed, with_dipoles):
        """scf_step_features matches multipole_electrostatic_features to float64 ULP."""
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _scf_step_random_system(
            seed=seed, n_atoms=6, box_len=5.0, device=td, with_dipoles=with_dipoles
        )
        sigma = 1.0
        receiver_sigmas = [0.6, 1.0, 1.4]
        k_cutoff = 3.5
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            k_cutoff=k_cutoff,
            l_max=1 if with_dipoles else 0,
        )
        f_scf = multipole_scf_step_features(cache, positions, source_feats)
        f_one_shot = multipole_electrostatic_features(
            positions,
            source_feats,
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            k_cutoff=k_cutoff,
        )
        np.testing.assert_allclose(
            f_scf.detach().cpu().numpy(),
            f_one_shot.detach().cpu().numpy(),
            rtol=1e-14,
            atol=1e-14,
        )

    @pytest.mark.parametrize("mode_name", ["multipoles", "receiver", "none"])
    def test_parity_across_feature_norm_modes(self, device, mode_name):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _scf_step_random_system(
            seed=7, n_atoms=5, box_len=5.0, device=td
        )
        sigma = 1.0
        receiver_sigmas = [0.8, 1.2]
        k_cutoff = 3.5
        mode = NormMode[mode_name.upper()]
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            k_cutoff=k_cutoff,
            feature_normalize=mode,
        )
        f_scf = multipole_scf_step_features(cache, positions, source_feats)
        f_one_shot = multipole_electrostatic_features(
            positions,
            source_feats,
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            k_cutoff=k_cutoff,
            feature_normalize=mode,
        )
        np.testing.assert_allclose(
            f_scf.detach().cpu().numpy(),
            f_one_shot.detach().cpu().numpy(),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_include_self_interaction(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _scf_step_random_system(
            seed=19, n_atoms=4, box_len=5.0, device=td
        )
        sigma = 1.0
        receiver_sigmas = [1.0]
        k_cutoff = 3.5
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            k_cutoff=k_cutoff,
        )
        f_scf = multipole_scf_step_features(
            cache, positions, source_feats, include_self_interaction=True
        )
        f_one_shot = multipole_electrostatic_features(
            positions,
            source_feats,
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            k_cutoff=k_cutoff,
            include_self_interaction=True,
        )
        np.testing.assert_allclose(
            f_scf.detach().cpu().numpy(),
            f_one_shot.detach().cpu().numpy(),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_multi_step_same_cache(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats_a = _scf_step_random_system(
            seed=101, n_atoms=4, box_len=5.0, device=td
        )
        _, _, _, _, source_feats_b = _scf_step_random_system(
            seed=103, n_atoms=4, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.9, 1.1],
            k_cutoff=3.5,
        )
        f_a = multipole_scf_step_features(cache, positions, source_feats_a)
        f_b = multipole_scf_step_features(cache, positions, source_feats_b)
        assert float((f_a - f_b).abs().max()) > 1e-10
        f_a_one = multipole_electrostatic_features(
            positions,
            source_feats_a,
            cell,
            sigma=1.0,
            receiver_sigmas=[0.9, 1.1],
            k_cutoff=3.5,
        )
        np.testing.assert_allclose(
            f_a.detach().cpu().numpy(),
            f_a_one.detach().cpu().numpy(),
            rtol=1e-14,
            atol=1e-14,
        )
