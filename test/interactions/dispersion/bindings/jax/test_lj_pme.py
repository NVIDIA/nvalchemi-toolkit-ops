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

"""JAX dispersion PME (LJ-PME) tests + torch<->JAX parity.

JAX runs on GPU; torch parity references are computed on CPU (warp on a
different device) to avoid an in-process GPU stream conflict.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax", reason="No JAX installed.")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from nvalchemiops.jax.interactions.dispersion import (  # noqa: E402
    dispersion_pme,
    dispersion_reciprocal_space,
    estimate_dispersion_pme_parameters,
)


def _rand_system(n=12, box=12.0, seed=2):
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 3)) * box
    sigma = 0.9 + 0.4 * rng.random(n)
    epsilon = 0.5 + 0.5 * rng.random(n)
    cell = np.eye(3) * box
    return pos, sigma, epsilon, cell


def _nl_numpy(pos, box, rc):
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
    return np.stack([src, dst]).astype(np.int32), ptr, sh


def _brute_force(pos, sigma, epsilon, box, nmax=6):
    b = 2.0 * np.sqrt(epsilon) * sigma**3
    c6 = b[:, None] * b[None, :]
    n = len(pos)
    rng = range(-nmax, nmax + 1)
    shifts = np.array([[a, bb, c] for a in rng for bb in rng for c in rng]) * box
    total = 0.0
    for s in shifts:
        d = pos[:, None, :] - pos[None, :, :] + s
        r2 = (d * d).sum(-1)
        if np.allclose(s, 0):
            r2 = r2 + np.eye(n) * 1e30
        total += -0.5 * (c6 * np.clip(r2, 1e-30, None) ** (-3)).sum()
    return total


class TestPMEvsDirect:
    @pytest.mark.slow
    def test_pme_matches_brute_force(self):
        pos, sigma, epsilon, cell = _rand_system(n=12, box=12.0, seed=2)
        ref = _brute_force(pos, sigma, epsilon, 12.0, nmax=6)
        rc = 5.95
        nl, nptr, nsh = _nl_numpy(pos, 12.0, rc)
        e = dispersion_pme(
            jnp.array(pos),
            jnp.array(sigma),
            jnp.array(epsilon),
            jnp.array(cell)[None],
            alpha=0.6,
            mesh_dimensions=(128, 128, 128),
            spline_order=6,
            real_space_cutoff=rc,
            neighbor_list=jnp.array(nl),
            neighbor_ptr=jnp.array(nptr),
            neighbor_shifts=jnp.array(nsh),
        )
        assert abs(float(e.sum()) - ref) / abs(ref) < 1e-3


class TestAlphaIndependence:
    def test_invariant_to_beta(self):
        pos, sigma, epsilon, cell = _rand_system(n=40, box=16.0, seed=1)
        rc = 7.9
        nl, nptr, nsh = _nl_numpy(pos, 16.0, rc)
        kw = dict(
            mesh_dimensions=(80, 80, 80),
            spline_order=6,
            real_space_cutoff=rc,
            neighbor_list=jnp.array(nl),
            neighbor_ptr=jnp.array(nptr),
            neighbor_shifts=jnp.array(nsh),
        )
        vals = [
            float(
                dispersion_pme(
                    jnp.array(pos),
                    jnp.array(sigma),
                    jnp.array(epsilon),
                    jnp.array(cell)[None],
                    alpha=beta,
                    **kw,
                ).sum()
            )
            for beta in (0.35, 0.40, 0.45, 0.50)
        ]
        assert max(vals) - min(vals) < 2e-4


class TestParameters:
    def test_estimate(self):
        pos, sigma, epsilon, cell = _rand_system(n=30, box=14.0, seed=5)
        params = estimate_dispersion_pme_parameters(
            jnp.array(pos), jnp.array(cell)[None], accuracy=1e-5
        )
        assert float(params.alpha[0]) > 0
        assert len(params.mesh_dimensions) == 3


class TestTorchParity:
    """In-process torch <-> JAX reciprocal parity.

    torch runs on the same device family as JAX (mixing a Warp CPU context with
    JAX's Warp CUDA context in one process triggers a CUDA illegal access).
    """

    def test_reciprocal_energy_force_virial_parity(self):
        torch = pytest.importorskip("torch")
        from nvalchemiops.torch.interactions.dispersion import (
            dispersion_reciprocal_space as t_rec,
        )

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos, sigma, epsilon, cell = _rand_system(n=16, box=12.0, seed=3)
        mesh = (48, 48, 48)
        beta = 0.5
        e_j, f_j, v_j = dispersion_reciprocal_space(
            jnp.array(pos),
            jnp.array(sigma),
            jnp.array(epsilon),
            jnp.array(cell)[None],
            alpha=beta,
            mesh_dimensions=mesh,
            spline_order=5,
            compute_forces=True,
            compute_virial=True,
        )
        e_t, f_t, v_t = t_rec(
            torch.tensor(pos, device=dev),
            torch.tensor(sigma, device=dev),
            torch.tensor(epsilon, device=dev),
            torch.tensor(cell, device=dev)[None],
            alpha=beta,
            mesh_dimensions=mesh,
            spline_order=5,
            compute_forces=True,
            compute_virial=True,
        )
        assert abs(float(e_j.sum()) - float(e_t.sum())) / abs(float(e_t.sum())) < 1e-9
        assert np.abs(np.array(f_j) - f_t.cpu().numpy()).max() < 1e-8
        assert np.abs(np.array(v_j) - v_t.cpu().numpy()).max() < 1e-8
