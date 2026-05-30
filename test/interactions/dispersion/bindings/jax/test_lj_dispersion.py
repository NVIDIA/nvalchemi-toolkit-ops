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

"""JAX pairwise dispersion (r^-6) tests + torch<->JAX parity.

JAX runs on its default (GPU) device; torch parity references are computed on
CPU to avoid a warp stream/device conflict when both frameworks share the GPU
in one process.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("jax", reason="No JAX installed.")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from nvalchemiops.jax.interactions.dispersion import (  # noqa: E402
    lj_dispersion_energy,
    lj_dispersion_energy_forces,
    sigma_epsilon_to_dispersion_charge,
)


def _c6(sigma, epsilon):
    return 4.0 * epsilon * sigma**6


def _screen(beta, r):
    x2 = (beta * r) ** 2
    return (1.0 + x2 + 0.5 * x2 * x2) * math.exp(-x2)


def _nl_numpy(pos, box, rc):
    """Build a symmetric CSR neighbor list with numpy (minimum image)."""
    n = len(pos)
    src, dst, sh = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pos[i] - pos[j]
            s = np.round(d / box)
            if np.linalg.norm(d - s * box) < rc:
                src.append(i)
                dst.append(j)
                sh.append(s)
    order = np.argsort(src, kind="stable")
    src = np.array(src)[order]
    dst = np.array(dst)[order]
    sh = np.array(sh)[order].astype(np.int32)
    ptr = np.zeros(n + 1, dtype=np.int32)
    for i in src:
        ptr[i + 1] += 1
    ptr = np.cumsum(ptr).astype(np.int32)
    nl = np.stack([src, dst]).astype(np.int32)
    return jnp.array(nl), jnp.array(ptr), jnp.array(sh)


def _rand_system(n=12, box=12.0, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 3)) * box
    sigma = 0.9 + 0.4 * rng.random(n)
    epsilon = 0.5 + 0.5 * rng.random(n)
    cell = np.eye(3) * box
    return pos, sigma, epsilon, cell


class TestConversion:
    def test_b(self):
        b = sigma_epsilon_to_dispersion_charge(
            jnp.array([1.0, 2.0]), jnp.array([1.0, 0.25])
        )
        assert np.allclose(
            np.array(b) ** 2, 4.0 * np.array([1.0, 0.25]) * np.array([1.0, 2.0]) ** 6
        )


class TestPlain:
    def test_two_atom_energy(self):
        pos = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        sig = jnp.array([1.0, 1.0])
        eps = jnp.array([1.0, 1.0])
        cell = jnp.array([[[100.0, 0, 0], [0, 100.0, 0], [0, 0, 100.0]]])
        nl = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        nptr = jnp.array([0, 1, 2], dtype=jnp.int32)
        nsh = jnp.zeros((2, 3), dtype=jnp.int32)
        e = lj_dispersion_energy(
            pos,
            sig,
            eps,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
        )
        assert abs(float(e.sum()) - (-_c6(1.0, 1.0) / 3.0**6)) < 1e-10

    @pytest.mark.parametrize("beta", [0.2, 0.5, 1.0])
    def test_damped_closed_form(self, beta):
        pos = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        sig = jnp.array([1.0, 1.0])
        eps = jnp.array([1.0, 1.0])
        cell = jnp.array([[[100.0, 0, 0], [0, 100.0, 0], [0, 0, 100.0]]])
        nl = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        nptr = jnp.array([0, 1, 2], dtype=jnp.int32)
        nsh = jnp.zeros((2, 3), dtype=jnp.int32)
        e = lj_dispersion_energy(
            pos,
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
        assert abs(float(e.sum()) - expected) < 1e-10

    def test_newton_third_law(self):
        pos, sig, eps, cell = _rand_system()
        nl, nptr, nsh = _nl_numpy(pos, 12.0, 5.0)
        _, f = lj_dispersion_energy_forces(
            jnp.array(pos),
            jnp.array(sig),
            jnp.array(eps),
            jnp.array(cell)[None],
            cutoff=5.0,
            alpha=0.4,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
        )
        assert float(jnp.abs(f.sum(0)).max()) < 1e-9


class TestPrecision:
    def test_float32_close(self):
        pos, sig, eps, cell = _rand_system()
        nl, nptr, nsh = _nl_numpy(pos, 12.0, 5.0)
        kw = dict(neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=nsh)
        e64 = lj_dispersion_energy(
            jnp.array(pos),
            jnp.array(sig),
            jnp.array(eps),
            jnp.array(cell)[None],
            5.0,
            0.4,
            **kw,
        ).sum()
        e32 = lj_dispersion_energy(
            jnp.array(pos, dtype=jnp.float32),
            jnp.array(sig, dtype=jnp.float32),
            jnp.array(eps, dtype=jnp.float32),
            jnp.array(cell, dtype=jnp.float32)[None],
            5.0,
            0.4,
            **kw,
        ).sum()
        assert abs(float(e32) - float(e64)) / abs(float(e64)) < 1e-4


class TestTorchParity:
    """In-process torch <-> JAX parity.

    torch must run on the *same* device family as JAX: mixing a Warp CPU
    context (torch on CPU) with Warp's CUDA context (JAX on GPU) in one process
    triggers a CUDA illegal-memory-access. torch-GPU + JAX-GPU is stable.
    """

    @pytest.mark.parametrize("beta", [0.0, 0.4])
    def test_pairwise_parity(self, beta):
        torch = pytest.importorskip("torch")
        from nvalchemiops.torch.interactions.dispersion import (
            lj_dispersion_energy_forces as t_ef,
        )

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos, sig, eps, cell = _rand_system(n=14, box=12.0, seed=7)
        nl, nptr, nsh = _nl_numpy(pos, 12.0, 5.0)
        e_j, f_j = lj_dispersion_energy_forces(
            jnp.array(pos),
            jnp.array(sig),
            jnp.array(eps),
            jnp.array(cell)[None],
            cutoff=5.0,
            alpha=beta,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=nsh,
        )
        e_t, f_t = t_ef(
            torch.tensor(pos, device=dev),
            torch.tensor(sig, device=dev),
            torch.tensor(eps, device=dev),
            torch.tensor(cell, device=dev)[None],
            cutoff=5.0,
            alpha=beta,
            neighbor_list=torch.tensor(np.array(nl), device=dev),
            neighbor_ptr=torch.tensor(np.array(nptr), device=dev),
            neighbor_shifts=torch.tensor(np.array(nsh), device=dev),
        )
        assert abs(float(e_j.sum()) - float(e_t.sum())) < 1e-9
        assert np.abs(np.array(f_j) - f_t.cpu().numpy()).max() < 1e-9
