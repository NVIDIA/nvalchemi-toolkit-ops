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

"""Packed ``multipole_moments`` layer + e3nn l=2 <-> Cartesian converter."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    cartesian_quadrupole_to_e3nn,
    dipole_cartesian_to_spherical,
    dipole_spherical_to_cartesian,
    e3nn_to_cartesian_quadrupole,
    infer_l_max,
    pack_multipole_moments,
    split_multipole_moments,
)


def _rand_traceless_Q(n, rng):
    Qr = rng.standard_normal((n, 3, 3))
    Q = 0.5 * (Qr + Qr.transpose(0, 2, 1))
    Q -= np.eye(3)[None] * (np.trace(Q, axis1=1, axis2=2)[:, None, None] / 3.0)
    return Q


class TestInferLMax:
    @pytest.mark.parametrize("last,expected", [(1, 0), (4, 1), (9, 2)])
    def test_supported(self, last, expected):
        assert infer_l_max(torch.zeros(5, last)) == expected

    @pytest.mark.parametrize("last", [2, 3, 5, 16])
    def test_unsupported_raises(self, last):
        with pytest.raises(ValueError):
            infer_l_max(torch.zeros(5, last))


class TestL2Converter:
    def test_traceless_output(self):
        rng = np.random.default_rng(0)
        f = torch.tensor(rng.standard_normal((32, 5)))
        Q = e3nn_to_cartesian_quadrupole(f)
        assert Q.shape == (32, 3, 3)
        tr = Q.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max().item()
        assert tr < 1e-12, f"|Tr Q|={tr:.2e}"
        # symmetric
        assert (Q - Q.transpose(-1, -2)).abs().max().item() < 1e-14

    def test_roundtrip_e3nn_cart_e3nn(self):
        rng = np.random.default_rng(1)
        f = torch.tensor(rng.standard_normal((64, 5)))
        f_rt = cartesian_quadrupole_to_e3nn(e3nn_to_cartesian_quadrupole(f))
        assert torch.allclose(f, f_rt, atol=1e-12)

    def test_roundtrip_cart_e3nn_cart_traceless(self):
        rng = np.random.default_rng(2)
        Q = torch.tensor(_rand_traceless_Q(64, rng))
        Q_rt = e3nn_to_cartesian_quadrupole(cartesian_quadrupole_to_e3nn(Q))
        assert torch.allclose(Q, Q_rt, atol=1e-12)

    def test_trace_is_dropped(self):
        rng = np.random.default_rng(3)
        Q = torch.tensor(_rand_traceless_Q(16, rng))
        Q_with_trace = Q + torch.eye(3) * 0.7  # add isotropic part
        # cart->e3nn->cart must recover the TRACELESS part only.
        Q_back = e3nn_to_cartesian_quadrupole(
            cartesian_quadrupole_to_e3nn(Q_with_trace)
        )
        assert torch.allclose(Q_back, Q, atol=1e-12)

    def test_matches_e3nn_044(self):
        """The hardcoded closed-form constants reproduce e3nn==0.4.4's
        'component' real-SH l=2 decomposition of ``k·Q·k``."""
        pytest.importorskip("e3nn")
        import torch.serialization

        torch.serialization.add_safe_globals([slice])  # e3nn 0.4.4 constants.pt
        from e3nn import o3

        rng = np.random.default_rng(4)
        v = rng.standard_normal((400, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        Q = _rand_traceless_Q(1, rng)[0]
        lhs = np.einsum("ki,ij,kj->k", v, Q, v)  # k·Q·k
        c = cartesian_quadrupole_to_e3nn(torch.tensor(Q[None]))[0].numpy()  # (5,)
        Y = o3.spherical_harmonics(
            2, torch.tensor(v), normalize=True, normalization="component"
        ).numpy()
        rhs = Y @ c
        rel = np.abs(lhs - rhs).max() / (np.abs(lhs).max() + 1e-30)
        assert rel < 1e-10, f"e3nn match rel={rel:.2e}"


class TestDipolePerm:
    def test_cart_to_e3nn_order(self):
        # e3nn l=1 order is (m=-1, m=0, m=+1) = (y, z, x).
        cart = torch.tensor([[1.0, 2.0, 3.0]])  # (x, y, z)
        sph = dipole_cartesian_to_spherical(cart)
        assert torch.allclose(sph, torch.tensor([[2.0, 3.0, 1.0]]))

    def test_roundtrip_cart_sph_cart(self):
        rng = np.random.default_rng(40)
        cart = torch.tensor(rng.standard_normal((32, 3)))
        rt = dipole_spherical_to_cartesian(dipole_cartesian_to_spherical(cart))
        assert torch.allclose(rt, cart, atol=1e-14)

    def test_roundtrip_sph_cart_sph(self):
        rng = np.random.default_rng(41)
        sph = torch.tensor(rng.standard_normal((32, 3)))
        rt = dipole_cartesian_to_spherical(dipole_spherical_to_cartesian(sph))
        assert torch.allclose(rt, sph, atol=1e-14)


class TestPackSplit:
    @pytest.mark.parametrize("l_max", [0, 1, 2])
    def test_pack_split_roundtrip(self, l_max):
        rng = np.random.default_rng(10 + l_max)
        n = 20
        q = torch.tensor(rng.standard_normal(n))
        d = torch.tensor(rng.standard_normal((n, 3))) if l_max >= 1 else None
        Q = torch.tensor(_rand_traceless_Q(n, rng)) if l_max >= 2 else None
        packed = pack_multipole_moments(q, d, Q)
        assert packed.shape == (n, (l_max + 1) ** 2)
        assert infer_l_max(packed) == l_max
        c2, d2, Q2, lm = split_multipole_moments(packed)
        assert lm == l_max
        assert torch.allclose(c2, q, atol=1e-12)
        if l_max >= 1:
            assert torch.allclose(d2, d, atol=1e-12)
        else:
            assert d2 is None
        if l_max >= 2:
            assert torch.allclose(Q2, Q, atol=1e-12)
        else:
            assert Q2 is None

    def test_pack_warns_on_traceful_q(self):
        rng = np.random.default_rng(20)
        q = torch.tensor(rng.standard_normal(8))
        d = torch.tensor(rng.standard_normal((8, 3)))
        Q = torch.tensor(_rand_traceless_Q(8, rng)) + torch.eye(3) * 0.5
        with pytest.warns(UserWarning, match="traceless"):
            pack_multipole_moments(q, d, Q)


class TestAutogradGraphable:
    def test_grad_lands_on_packed_5component_block(self):
        rng = np.random.default_rng(30)
        n = 16
        moments = torch.zeros(n, 9, dtype=torch.float64)
        moments[:, 4:9] = torch.tensor(rng.standard_normal((n, 5)))
        moments.requires_grad_(True)
        _, _, Q, _ = split_multipole_moments(moments)
        H = torch.tensor(rng.standard_normal((n, 3, 3)))
        H = 0.5 * (H + H.transpose(-1, -2))
        E = 0.5 * (Q * H).sum()
        (g,) = torch.autograd.grad(E, moments)
        assert g.shape == (n, 9)
        assert g[:, :4].abs().max() == 0.0  # charge/dipole untouched
        assert g[:, 4:9].abs().max() > 0.0  # l=2 block carries it

    @pytest.mark.slow
    def test_converter_compiles_fullgraph(self):
        rng = np.random.default_rng(31)
        f = torch.tensor(rng.standard_normal((16, 5)))

        def fn(x):
            return e3nn_to_cartesian_quadrupole(x).sum()

        torch._dynamo.reset()
        compiled = torch.compile(fn, fullgraph=True)
        assert torch.allclose(compiled(f), fn(f))  # fullgraph => no break
