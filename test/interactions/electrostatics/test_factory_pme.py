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

"""Tests for PME reciprocal factory kernels.

Covers:

1. **Convolve double-backward** -- the factory-backed convolve double-backward
   kernels (single + batch) match a central-difference of the convolve backward
   outputs contracted with random cotangents on all four backward outputs (f64).
   Covers both the position terms (``dL/dmesh_fft`` / ``dL/dgrad_convolved``)
   and the cell/stress terms (``dL/dalpha`` / ``dL/dvolume`` /
   ``dL/dk_squared``), asserted FD-correct and genuinely nonzero.

2. **Stress-path second backward** -- end-to-end through the public PME reciprocal
   energy: caller-supplied ``volume=`` is pinned as static metadata, while the
   full ``d2E/dcell2`` second backward runs and is finite + nonzero (confirming
   the k_squared/volume -> cell propagation is live when volume is derived from
   ``cell``).

3. **Launcher parity** -- direct factory kernel launches and the public
   ``pme_convolve`` / ``pme_corrections`` wrapper launchers produce identical
   outputs for ``wp.float32`` + ``wp.float64`` x single/batch (forward +
   backward). These are wrapper-wiring checks; the finite-difference sections are
   the independent derivative oracle.

4. **Position-grad path + force loss** -- through the public
   ``pme_reciprocal_space`` autograd path: ``-grad(E.sum(), positions)`` matches
   an F3-style finite difference (~1e-9, f64), and a force-loss ``.backward()``
   runs without the "no autograd formula registered" error (exercising the
   convolve double-backward) and matches a nested finite difference.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.interactions.electrostatics import pme_kernels as pk
from nvalchemiops.interactions.electrostatics.pme_factory import (
    get_pme_kernel,
)

_DTYPES = [wp.float32, wp.float64]
_DTYPE_IDS = ["f32", "f64"]
_NPF = {wp.float32: np.float32, wp.float64: np.float64}
_VEC2 = {wp.float32: wp.vec2f, wp.float64: wp.vec2d}


# ==============================================================================
# Helpers
# ==============================================================================


def _wp(a, dtype, device):
    return wp.from_numpy(np.ascontiguousarray(a), dtype=dtype, device=device)


def _convolve_mesh_system(rng, *, nx=4, ny=5, nzr=3, batch=None, dtype=wp.float64):
    """Random convolve inputs. ``batch`` None -> single (3D meshes)."""
    npf = _NPF[dtype]
    shape = (nx, ny, nzr) if batch is None else (batch, nx, ny, nzr)
    ksq = rng.uniform(0.1, 3.0, size=shape).astype(npf)
    # zero the k=0 grid point(s)
    if batch is None:
        ksq[0, 0, 0] = 0.0
    else:
        ksq[:, 0, 0, 0] = 0.0
    bx = rng.uniform(0.5, 1.0, size=nx).astype(npf)
    by = rng.uniform(0.5, 1.0, size=ny).astype(npf)
    bz = rng.uniform(0.5, 1.0, size=nzr).astype(npf)
    nsys = 1 if batch is None else batch
    alpha = rng.uniform(0.3, 0.6, size=nsys).astype(npf)
    vol = rng.uniform(80.0, 120.0, size=nsys).astype(npf)
    m = rng.standard_normal((*shape, 2)).astype(npf)
    g = rng.standard_normal((*shape, 2)).astype(npf)
    return dict(ksq=ksq, bx=bx, by=by, bz=bz, alpha=alpha, vol=vol, m=m, g=g)


def _run_convolve_backward(sysd, *, batch, dtype, device):
    """Run the public convolve backward wrapper; return numpy outputs."""
    vec2 = _VEC2[dtype]
    grad_mesh = np.zeros_like(sysd["m"])
    nsys = sysd["alpha"].shape[0]
    ga = np.zeros(nsys, dtype=_NPF[dtype])
    gv = np.zeros(nsys, dtype=_NPF[dtype])
    gksq = np.zeros_like(sysd["ksq"])

    wp_grad_mesh = _wp(grad_mesh, vec2, device)
    wp_ga = _wp(ga, dtype, device)
    wp_gv = _wp(gv, dtype, device)
    wp_gksq = _wp(gksq, dtype, device)

    args = (
        _wp(sysd["m"], vec2, device),
        _wp(sysd["g"], vec2, device),
        _wp(sysd["ksq"], dtype, device),
        _wp(sysd["bx"], dtype, device),
        _wp(sysd["by"], dtype, device),
        _wp(sysd["bz"], dtype, device),
        _wp(sysd["alpha"], dtype, device),
        _wp(sysd["vol"], dtype, device),
        wp_grad_mesh,
        wp_ga,
        wp_gv,
        wp_gksq,
    )
    launch = pk.batch_pme_convolve_backward if batch else pk.pme_convolve_backward
    launch(*args, wp_dtype=dtype, device=device)
    wp.synchronize()
    return wp_grad_mesh.numpy(), wp_ga.numpy(), wp_gv.numpy(), wp_gksq.numpy()


def _run_convolve_double_backward(sysd, cot, *, batch, dtype, device):
    """Run the convolve double-backward; return numpy outputs."""
    vec2 = _VEC2[dtype]
    grad_mesh_out = np.zeros_like(sysd["m"])
    grad_grad_conv = np.zeros_like(sysd["g"])
    nsys = sysd["alpha"].shape[0]
    gksq_out = np.zeros_like(sysd["ksq"])
    ga_out = np.zeros(nsys, dtype=_NPF[dtype])
    gv_out = np.zeros(nsys, dtype=_NPF[dtype])

    wp_grad_mesh_out = _wp(grad_mesh_out, vec2, device)
    wp_ggc = _wp(grad_grad_conv, vec2, device)
    wp_gksq_out = _wp(gksq_out, dtype, device)
    wp_ga_out = _wp(ga_out, dtype, device)
    wp_gv_out = _wp(gv_out, dtype, device)

    args = (
        _wp(cot["h_m"], vec2, device),
        _wp(cot["h_a"], dtype, device),
        _wp(cot["h_v"], dtype, device),
        _wp(cot["h_ksq"], dtype, device),
        _wp(sysd["m"], vec2, device),
        _wp(sysd["g"], vec2, device),
        _wp(sysd["ksq"], dtype, device),
        _wp(sysd["bx"], dtype, device),
        _wp(sysd["by"], dtype, device),
        _wp(sysd["bz"], dtype, device),
        _wp(sysd["alpha"], dtype, device),
        _wp(sysd["vol"], dtype, device),
        wp_grad_mesh_out,
        wp_ggc,
        wp_gksq_out,
        wp_ga_out,
        wp_gv_out,
    )
    launch = (
        pk.batch_pme_convolve_double_backward
        if batch
        else pk.pme_convolve_double_backward
    )
    launch(*args, wp_dtype=dtype, device=device)
    wp.synchronize()
    return (
        wp_grad_mesh_out.numpy(),
        wp_ggc.numpy(),
        wp_gksq_out.numpy(),
        wp_ga_out.numpy(),
        wp_gv_out.numpy(),
    )


# ==============================================================================
# 1. Convolve double-backward vs finite-diff of the backward outputs
# ==============================================================================

_FD_EPS = 1e-6
_FD_TOL = 1e-6


class TestConvolveDoubleBackwardFD:
    @pytest.mark.parametrize("batch", [None, 2], ids=["single", "batch"])
    def test_dbwd_vs_fd(self, device, batch):
        rng = np.random.default_rng(0)
        sysd = _convolve_mesh_system(rng, batch=batch, dtype=wp.float64)
        is_batch = batch is not None
        nsys = sysd["alpha"].shape[0]
        cot = dict(
            h_m=rng.standard_normal(sysd["m"].shape),
            h_a=rng.standard_normal(nsys),
            h_v=rng.standard_normal(nsys),
            h_ksq=rng.standard_normal(sysd["ksq"].shape),
        )

        def L(m_=None, g_=None, alpha_=None, vol_=None, ksq_=None):
            sd = dict(sysd)
            if m_ is not None:
                sd["m"] = m_
            if g_ is not None:
                sd["g"] = g_
            if alpha_ is not None:
                sd["alpha"] = alpha_
            if vol_ is not None:
                sd["vol"] = vol_
            if ksq_ is not None:
                sd["ksq"] = ksq_
            gm, ga, gv, gksq = _run_convolve_backward(
                sd, batch=is_batch, dtype=wp.float64, device=device
            )
            return (
                (cot["h_m"] * gm).sum()
                + (cot["h_a"] * ga).sum()
                + (cot["h_v"] * gv).sum()
                + (cot["h_ksq"] * gksq).sum()
            )

        m0, g0 = sysd["m"], sysd["g"]
        a0, v0, ksq0 = sysd["alpha"], sysd["vol"], sysd["ksq"]

        def _fd_array(x0, setter):
            fd = np.zeros_like(x0)
            for idx in np.ndindex(*x0.shape):
                xp = x0.copy()
                xp[idx] += _FD_EPS
                xm = x0.copy()
                xm[idx] -= _FD_EPS
                fd[idx] = (setter(xp) - setter(xm)) / (2 * _FD_EPS)
            return fd

        fd_g = _fd_array(g0, lambda x: L(g_=x))
        fd_m = _fd_array(m0, lambda x: L(m_=x))
        fd_a = _fd_array(a0, lambda x: L(alpha_=x))
        fd_v = _fd_array(v0, lambda x: L(vol_=x))
        fd_ksq = _fd_array(ksq0, lambda x: L(ksq_=x))

        gm_out, ggc, gksq_out, ga_out, gv_out = _run_convolve_double_backward(
            sysd, cot, batch=is_batch, dtype=wp.float64, device=device
        )

        # dL/dgrad_convolved and dL/dmesh_fft match the FD of the backward outputs.
        max_g = np.abs(ggc - fd_g).max()
        max_m = np.abs(gm_out - fd_m).max()
        assert max_g < _FD_TOL, f"dL/dg FD max_abs={max_g:.3e}"
        assert max_m < _FD_TOL, f"dL/dm FD max_abs={max_m:.3e}"

        # Cell/stress second-order outputs (alpha / volume / k_squared) match FD.
        max_a = np.abs(ga_out - fd_a).max()
        max_v = np.abs(gv_out - fd_v).max()
        max_ksq = np.abs(gksq_out - fd_ksq).max()
        assert max_a < _FD_TOL, f"dL/dalpha FD max_abs={max_a:.3e}"
        assert max_v < _FD_TOL, f"dL/dvolume FD max_abs={max_v:.3e}"
        assert max_ksq < _FD_TOL, f"dL/dk_squared FD max_abs={max_ksq:.3e}"

        # The cell/stress terms are genuinely nonzero (not silently zeroed).
        assert np.abs(ga_out).max() > 1e-6
        assert np.abs(gv_out).max() > 1e-6
        assert np.abs(gksq_out).max() > 1e-6

    @pytest.mark.parametrize("batch", [None, 2], ids=["single", "batch"])
    def test_dbwd_force_loss_shortcut(self, device, batch):
        # With only h_grad_mesh nonzero (the pure force-loss cotangent set),
        # dL/dgrad_convolved == forward convolve applied to h_grad_mesh, and
        # dL/dmesh_fft == 0. This pins the linearity shortcut the chain relies on.
        rng = np.random.default_rng(1)
        sysd = _convolve_mesh_system(rng, batch=batch, dtype=wp.float64)
        is_batch = batch is not None
        nsys = sysd["alpha"].shape[0]
        cot = dict(
            h_m=rng.standard_normal(sysd["m"].shape),
            h_a=np.zeros(nsys),
            h_v=np.zeros(nsys),
            h_ksq=np.zeros_like(sysd["ksq"]),
        )
        gm_out, ggc, _, _, _ = _run_convolve_double_backward(
            sysd, cot, batch=is_batch, dtype=wp.float64, device=device
        )
        # dL/dg == forward convolve of h_m: run the forward convolve on h_m.
        sd = dict(sysd)
        sd["m"] = cot["h_m"]
        conv_hm = _run_convolve_forward(
            sd, batch=is_batch, dtype=wp.float64, device=device
        )
        assert np.abs(ggc - conv_hm).max() < 1e-10
        assert np.abs(gm_out).max() < 1e-12

    @pytest.mark.parametrize("batch", [None, 2], ids=["single", "batch"])
    def test_dbwd_f32_matches_f64(self, device, batch):
        # f32 correctness (looser): the f32 double-backward outputs match the f64
        # outputs cast down. FD in f32 is too noisy to be a useful oracle (roundoff
        # ~1e-1 relative), so the f64 result (FD-validated in test_dbwd_vs_fd) is
        # the reference. Covers all five outputs incl. the cell/stress terms.
        rng = np.random.default_rng(2)
        sysd = _convolve_mesh_system(rng, batch=batch, dtype=wp.float64)
        is_batch = batch is not None
        nsys = sysd["alpha"].shape[0]
        cot = dict(
            h_m=rng.standard_normal(sysd["m"].shape),
            h_a=rng.standard_normal(nsys),
            h_v=rng.standard_normal(nsys),
            h_ksq=rng.standard_normal(sysd["ksq"].shape),
        )
        ref = _run_convolve_double_backward(
            sysd, cot, batch=is_batch, dtype=wp.float64, device=device
        )
        sysd32 = {k: v.astype(np.float32) for k, v in sysd.items()}
        cot32 = {k: v.astype(np.float32) for k, v in cot.items()}
        got = _run_convolve_double_backward(
            sysd32, cot32, batch=is_batch, dtype=wp.float32, device=device
        )
        names = ["dL/dm", "dL/dg", "dL/dk_squared", "dL/dalpha", "dL/dvolume"]
        for name, r, g in zip(names, ref, got):
            np.testing.assert_allclose(
                g, r.astype(np.float32), rtol=1e-4, atol=1e-5, err_msg=name
            )


def _run_convolve_forward(sysd, *, batch, dtype, device):
    vec2 = _VEC2[dtype]
    out = np.zeros_like(sysd["m"])
    wp_out = _wp(out, vec2, device)
    args = (
        _wp(sysd["m"], vec2, device),
        _wp(sysd["ksq"], dtype, device),
        _wp(sysd["bx"], dtype, device),
        _wp(sysd["by"], dtype, device),
        _wp(sysd["bz"], dtype, device),
        _wp(sysd["alpha"], dtype, device),
        _wp(sysd["vol"], dtype, device),
        wp_out,
    )
    launch = pk.batch_pme_convolve if batch else pk.pme_convolve
    launch(*args, wp_dtype=dtype, device=device)
    wp.synchronize()
    return wp_out.numpy()


# ==============================================================================
# 3. Launcher parity (direct factory call vs public wrapper)
# ==============================================================================
#
# Single-system factory convolve kernels use the same 3d mesh rank as the public
# low-level launchers; batched factory convolve kernels use 4d meshes.


class TestFactoryConvolveParity:
    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batch", [None, 2], ids=["single", "batch"])
    def test_forward_parity(self, device, dtype, batch):
        rng = np.random.default_rng(20)
        sysd = _convolve_mesh_system(rng, batch=batch, dtype=dtype)
        is_batch = batch is not None
        ref = _run_convolve_forward(sysd, batch=is_batch, dtype=dtype, device=device)

        vec2 = _VEC2[dtype]
        kernel = get_pme_kernel(
            dtype,
            component="pme_convolve",
            batched=is_batch,
            order="forward",
        )
        mesh = sysd["m"]
        ksq = sysd["ksq"]
        out = np.zeros_like(mesh)
        wp_out = _wp(out, vec2, device)
        wp.launch(
            kernel,
            dim=mesh.shape[:-1],
            inputs=[
                _wp(mesh, vec2, device),
                _wp(ksq, dtype, device),
                _wp(sysd["bx"], dtype, device),
                _wp(sysd["by"], dtype, device),
                _wp(sysd["bz"], dtype, device),
                _wp(sysd["alpha"], dtype, device),
                _wp(sysd["vol"], dtype, device),
                wp_out,
            ],
            device=device,
        )
        wp.synchronize()
        got = wp_out.numpy()
        assert np.array_equal(got, ref)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batch", [None, 2], ids=["single", "batch"])
    def test_backward_parity(self, device, dtype, batch):
        rng = np.random.default_rng(21)
        sysd = _convolve_mesh_system(rng, batch=batch, dtype=dtype)
        is_batch = batch is not None
        gm_ref, ga_ref, gv_ref, gksq_ref = _run_convolve_backward(
            sysd, batch=is_batch, dtype=dtype, device=device
        )

        vec2 = _VEC2[dtype]
        kernel = get_pme_kernel(
            dtype,
            component="pme_convolve",
            batched=is_batch,
            order="backward",
        )
        nsys = sysd["alpha"].shape[0]
        mesh = sysd["m"]
        g = sysd["g"]
        ksq = sysd["ksq"]
        grad_mesh = np.zeros_like(mesh)
        gksq = np.zeros_like(ksq)
        ga = np.zeros(nsys, dtype=_NPF[dtype])
        gv = np.zeros(nsys, dtype=_NPF[dtype])
        wp_grad_mesh = _wp(grad_mesh, vec2, device)
        wp_gksq = _wp(gksq, dtype, device)
        wp_ga = _wp(ga, dtype, device)
        wp_gv = _wp(gv, dtype, device)
        wp.launch(
            kernel,
            dim=mesh.shape[:-1],
            inputs=[
                _wp(mesh, vec2, device),
                _wp(g, vec2, device),
                _wp(ksq, dtype, device),
                _wp(sysd["bx"], dtype, device),
                _wp(sysd["by"], dtype, device),
                _wp(sysd["bz"], dtype, device),
                _wp(sysd["alpha"], dtype, device),
                _wp(sysd["vol"], dtype, device),
                wp_grad_mesh,
                wp_ga,
                wp_gv,
                wp_gksq,
            ],
            device=device,
        )
        wp.synchronize()
        gm_got = wp_grad_mesh.numpy()
        gksq_got = wp_gksq.numpy()
        assert np.array_equal(gm_got, gm_ref)
        assert np.array_equal(gksq_got, gksq_ref)
        # Scalar grads (atomic reduction) match to f32/f64 round-off (order of
        # atomic adds may differ between hand-written and factory launches).
        rtol = 1e-12 if dtype == wp.float64 else 1e-5
        np.testing.assert_allclose(wp_ga.numpy(), ga_ref, rtol=rtol, atol=0.0)
        np.testing.assert_allclose(wp_gv.numpy(), gv_ref, rtol=rtol, atol=0.0)


def _run_corrections_forward(sysd, *, batch, charge_grad, dtype, device):
    """Hand-written corrections forward; return (corrected, charge_grad|None)."""
    n = sysd["raw"].shape[0]
    corrected = np.zeros(n, dtype=_NPF[dtype])
    wp_corr = _wp(corrected, dtype, device)
    if batch:
        if charge_grad:
            cg = np.zeros(n, dtype=_NPF[dtype])
            wp_cg = _wp(cg, dtype, device)
            pk.batch_pme_energy_corrections_with_charge_grad(
                _wp(sysd["raw"], dtype, device),
                _wp(sysd["q"], dtype, device),
                _wp(sysd["bidx"], wp.int32, device),
                _wp(sysd["vol"], dtype, device),
                _wp(sysd["alpha"], dtype, device),
                _wp(sysd["qtot"], dtype, device),
                wp_corr,
                wp_cg,
                wp_dtype=dtype,
                device=device,
            )
            wp.synchronize()
            return wp_corr.numpy(), wp_cg.numpy()
        pk.batch_pme_energy_corrections(
            _wp(sysd["raw"], dtype, device),
            _wp(sysd["q"], dtype, device),
            _wp(sysd["bidx"], wp.int32, device),
            _wp(sysd["vol"], dtype, device),
            _wp(sysd["alpha"], dtype, device),
            _wp(sysd["qtot"], dtype, device),
            wp_corr,
            wp_dtype=dtype,
            device=device,
        )
        wp.synchronize()
        return wp_corr.numpy(), None
    if charge_grad:
        cg = np.zeros(n, dtype=_NPF[dtype])
        wp_cg = _wp(cg, dtype, device)
        pk.pme_energy_corrections_with_charge_grad(
            _wp(sysd["raw"], dtype, device),
            _wp(sysd["q"], dtype, device),
            _wp(sysd["vol"], dtype, device),
            _wp(sysd["alpha"], dtype, device),
            _wp(sysd["qtot"], dtype, device),
            wp_corr,
            wp_cg,
            wp_dtype=dtype,
            device=device,
        )
        wp.synchronize()
        return wp_corr.numpy(), wp_cg.numpy()
    pk.pme_energy_corrections(
        _wp(sysd["raw"], dtype, device),
        _wp(sysd["q"], dtype, device),
        _wp(sysd["vol"], dtype, device),
        _wp(sysd["alpha"], dtype, device),
        _wp(sysd["qtot"], dtype, device),
        wp_corr,
        wp_dtype=dtype,
        device=device,
    )
    wp.synchronize()
    return wp_corr.numpy(), None


def _corrections_system(rng, *, batch, dtype):
    npf = _NPF[dtype]
    if batch is None:
        n = 7
        bidx = np.zeros(n, dtype=np.int32)
        nsys = 1
    else:
        per = 5
        n = per * batch
        bidx = np.repeat(np.arange(batch), per).astype(np.int32)
        nsys = batch
    raw = rng.standard_normal(n).astype(npf)
    q = rng.uniform(-1.0, 1.0, size=n).astype(npf)
    alpha = rng.uniform(0.3, 0.6, size=nsys).astype(npf)
    vol = rng.uniform(80.0, 120.0, size=nsys).astype(npf)
    qtot = np.zeros(nsys, dtype=npf)
    for s in range(nsys):
        qtot[s] = q[bidx == s].sum()
    return dict(raw=raw, q=q, bidx=bidx, alpha=alpha, vol=vol, qtot=qtot)


class TestFactoryCorrectionsParity:
    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batch", [None, 2], ids=["single", "batch"])
    @pytest.mark.parametrize("charge_grad", [False, True], ids=["E", "E_dQ"])
    def test_forward_parity(self, device, dtype, batch, charge_grad):
        rng = np.random.default_rng(30)
        sysd = _corrections_system(rng, batch=batch, dtype=dtype)
        is_batch = batch is not None
        ref_E, ref_cg = _run_corrections_forward(
            sysd, batch=is_batch, charge_grad=charge_grad, dtype=dtype, device=device
        )

        kernel = get_pme_kernel(
            dtype,
            component="pme_corrections",
            batched=is_batch,
            order="forward",
            charge_grad=charge_grad,
        )
        n = sysd["raw"].shape[0]
        corrected = np.zeros(n, dtype=_NPF[dtype])
        cg_out = np.zeros(n, dtype=_NPF[dtype])
        wp_corr = _wp(corrected, dtype, device)
        wp_cg = _wp(cg_out, dtype, device)
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                _wp(sysd["raw"], dtype, device),
                _wp(sysd["q"], dtype, device),
                _wp(sysd["bidx"], wp.int32, device),
                _wp(sysd["vol"], dtype, device),
                _wp(sysd["alpha"], dtype, device),
                _wp(sysd["qtot"], dtype, device),
                wp_corr,
                wp_cg,
            ],
            device=device,
        )
        wp.synchronize()
        assert np.array_equal(wp_corr.numpy(), ref_E)
        if charge_grad:
            assert np.array_equal(wp_cg.numpy(), ref_cg)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batch", [None, 2], ids=["single", "batch"])
    def test_backward_parity(self, device, dtype, batch):
        rng = np.random.default_rng(31)
        sysd = _corrections_system(rng, batch=batch, dtype=dtype)
        is_batch = batch is not None
        n = sysd["raw"].shape[0]
        nsys = sysd["alpha"].shape[0]
        grad_E = rng.standard_normal(n).astype(_NPF[dtype])

        # Hand-written reference.
        gr_ref = np.zeros(n, dtype=_NPF[dtype])
        gc_ref = np.zeros(n, dtype=_NPF[dtype])
        gv_ref = np.zeros(nsys, dtype=_NPF[dtype])
        ga_ref = np.zeros(nsys, dtype=_NPF[dtype])
        gq_ref = np.zeros(nsys, dtype=_NPF[dtype])
        wp_gr = _wp(gr_ref, dtype, device)
        wp_gc = _wp(gc_ref, dtype, device)
        wp_gv = _wp(gv_ref, dtype, device)
        wp_ga = _wp(ga_ref, dtype, device)
        wp_gq = _wp(gq_ref, dtype, device)
        if is_batch:
            pk.batch_pme_energy_corrections_backward(
                _wp(grad_E, dtype, device),
                _wp(sysd["raw"], dtype, device),
                _wp(sysd["q"], dtype, device),
                _wp(sysd["bidx"], wp.int32, device),
                _wp(sysd["vol"], dtype, device),
                _wp(sysd["alpha"], dtype, device),
                _wp(sysd["qtot"], dtype, device),
                wp_gr,
                wp_gc,
                wp_gv,
                wp_ga,
                wp_gq,
                wp_dtype=dtype,
                device=device,
            )
        else:
            pk.pme_energy_corrections_backward(
                _wp(grad_E, dtype, device),
                _wp(sysd["raw"], dtype, device),
                _wp(sysd["q"], dtype, device),
                _wp(sysd["vol"], dtype, device),
                _wp(sysd["alpha"], dtype, device),
                _wp(sysd["qtot"], dtype, device),
                wp_gr,
                wp_gc,
                wp_gv,
                wp_ga,
                wp_gq,
                wp_dtype=dtype,
                device=device,
            )
        wp.synchronize()

        # Factory.
        kernel = get_pme_kernel(
            dtype,
            component="pme_corrections",
            batched=is_batch,
            order="backward",
        )
        gr = np.zeros(n, dtype=_NPF[dtype])
        gc = np.zeros(n, dtype=_NPF[dtype])
        gv = np.zeros(nsys, dtype=_NPF[dtype])
        ga = np.zeros(nsys, dtype=_NPF[dtype])
        gq = np.zeros(nsys, dtype=_NPF[dtype])
        wp_gr2 = _wp(gr, dtype, device)
        wp_gc2 = _wp(gc, dtype, device)
        wp_gv2 = _wp(gv, dtype, device)
        wp_ga2 = _wp(ga, dtype, device)
        wp_gq2 = _wp(gq, dtype, device)
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                _wp(grad_E, dtype, device),
                _wp(sysd["raw"], dtype, device),
                _wp(sysd["q"], dtype, device),
                _wp(sysd["bidx"], wp.int32, device),
                _wp(sysd["vol"], dtype, device),
                _wp(sysd["alpha"], dtype, device),
                _wp(sysd["qtot"], dtype, device),
                wp_gr2,
                wp_gc2,
                wp_gv2,
                wp_ga2,
                wp_gq2,
            ],
            device=device,
        )
        wp.synchronize()
        # Per-atom grads bit-exact; scalar atomic reductions to round-off.
        assert np.array_equal(wp_gr2.numpy(), wp_gr.numpy())
        assert np.array_equal(wp_gc2.numpy(), wp_gc.numpy())
        rtol = 1e-12 if dtype == wp.float64 else 1e-5
        np.testing.assert_allclose(wp_gv2.numpy(), wp_gv.numpy(), rtol=rtol, atol=0.0)
        np.testing.assert_allclose(wp_ga2.numpy(), wp_ga.numpy(), rtol=rtol, atol=0.0)
        np.testing.assert_allclose(wp_gq2.numpy(), wp_gq.numpy(), rtol=rtol, atol=0.0)


# ==============================================================================
# 4. Factory double-backward parity (factory direct call vs public wrapper)
# ==============================================================================


class TestFactoryDoubleBackwardParity:
    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batch", [None, 2], ids=["single", "batch"])
    def test_convolve_dbwd_parity(self, device, dtype, batch):
        rng = np.random.default_rng(40)
        sysd = _convolve_mesh_system(rng, batch=batch, dtype=dtype)
        is_batch = batch is not None
        nsys = sysd["alpha"].shape[0]
        cot = dict(
            h_m=rng.standard_normal(sysd["m"].shape).astype(_NPF[dtype]),
            h_a=rng.standard_normal(nsys).astype(_NPF[dtype]),
            h_v=rng.standard_normal(nsys).astype(_NPF[dtype]),
            h_ksq=rng.standard_normal(sysd["ksq"].shape).astype(_NPF[dtype]),
        )
        gm_ref, ggc_ref, gksq_ref, ga_ref, gv_ref = _run_convolve_double_backward(
            sysd, cot, batch=is_batch, dtype=dtype, device=device
        )

        vec2 = _VEC2[dtype]
        kernel = get_pme_kernel(
            dtype,
            component="pme_convolve",
            batched=is_batch,
            order="double_backward",
        )
        h_m = cot["h_m"]
        h_ksq = cot["h_ksq"]
        mesh = sysd["m"]
        g = sysd["g"]
        ksq = sysd["ksq"]
        gm_out = np.zeros_like(mesh)
        ggc = np.zeros_like(g)
        gksq_out = np.zeros_like(ksq)
        ga_out = np.zeros(nsys, dtype=_NPF[dtype])
        gv_out = np.zeros(nsys, dtype=_NPF[dtype])
        wp_gm_out = _wp(gm_out, vec2, device)
        wp_ggc = _wp(ggc, vec2, device)
        wp_gksq_out = _wp(gksq_out, dtype, device)
        wp_ga_out = _wp(ga_out, dtype, device)
        wp_gv_out = _wp(gv_out, dtype, device)
        wp.launch(
            kernel,
            dim=mesh.shape[:-1],
            inputs=[
                _wp(h_m, vec2, device),
                _wp(cot["h_a"], dtype, device),
                _wp(cot["h_v"], dtype, device),
                _wp(h_ksq, dtype, device),
                _wp(mesh, vec2, device),
                _wp(g, vec2, device),
                _wp(ksq, dtype, device),
                _wp(sysd["bx"], dtype, device),
                _wp(sysd["by"], dtype, device),
                _wp(sysd["bz"], dtype, device),
                _wp(sysd["alpha"], dtype, device),
                _wp(sysd["vol"], dtype, device),
                wp_gm_out,
                wp_ggc,
                wp_gksq_out,
                wp_ga_out,
                wp_gv_out,
            ],
            device=device,
        )
        wp.synchronize()
        gm_got = wp_gm_out.numpy()
        ggc_got = wp_ggc.numpy()
        gksq_got = wp_gksq_out.numpy()
        # Elementwise outputs (incl. per-k dL/dk_squared) match between the
        # factory direct call and the public wrapper.
        assert np.array_equal(gm_got, gm_ref)
        assert np.array_equal(ggc_got, ggc_ref)
        assert np.array_equal(gksq_got, gksq_ref)
        # The cell/stress scalar grads (dL/dalpha, dL/dvolume) are atomic-summed
        # over k-points, so their reduction order is nondeterministic on GPU;
        # f32 matches to rtol rather than bit-exactly (same as the corrections
        # double-backward parity check).
        rtol = 1e-12 if dtype == wp.float64 else 1e-5
        np.testing.assert_allclose(wp_ga_out.numpy(), ga_ref, rtol=rtol, atol=0.0)
        np.testing.assert_allclose(wp_gv_out.numpy(), gv_ref, rtol=rtol, atol=0.0)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize("batch", [None, 2], ids=["single", "batch"])
    def test_corrections_dbwd_parity(self, device, dtype, batch):
        rng = np.random.default_rng(41)
        sysd = _corrections_system(rng, batch=batch, dtype=dtype)
        is_batch = batch is not None
        n = sysd["raw"].shape[0]
        nsys = sysd["alpha"].shape[0]
        grad_E = rng.standard_normal(n).astype(_NPF[dtype])
        h_raw = rng.standard_normal(n).astype(_NPF[dtype])
        h_chg = rng.standard_normal(n).astype(_NPF[dtype])
        h_vol = rng.standard_normal(nsys).astype(_NPF[dtype])
        h_alpha = rng.standard_normal(nsys).astype(_NPF[dtype])
        h_qtot = rng.standard_normal(nsys).astype(_NPF[dtype])

        def _alloc():
            return (
                np.zeros(n, dtype=_NPF[dtype]),  # gge
                np.zeros(n, dtype=_NPF[dtype]),  # graw
                np.zeros(n, dtype=_NPF[dtype]),  # gchg
                np.zeros(nsys, dtype=_NPF[dtype]),  # gvol
                np.zeros(nsys, dtype=_NPF[dtype]),  # galpha
                np.zeros(nsys, dtype=_NPF[dtype]),  # gqtot
            )

        # Hand-written reference.
        gge, graw, gchg, gvol, galpha, gqtot = _alloc()
        wp_gge = _wp(gge, dtype, device)
        wp_graw = _wp(graw, dtype, device)
        wp_gchg = _wp(gchg, dtype, device)
        wp_gvol = _wp(gvol, dtype, device)
        wp_galpha = _wp(galpha, dtype, device)
        wp_gqtot = _wp(gqtot, dtype, device)
        common_in = (
            _wp(h_raw, dtype, device),
            _wp(h_chg, dtype, device),
            _wp(h_vol, dtype, device),
            _wp(h_alpha, dtype, device),
            _wp(h_qtot, dtype, device),
            _wp(grad_E, dtype, device),
            _wp(sysd["raw"], dtype, device),
            _wp(sysd["q"], dtype, device),
        )
        if is_batch:
            pk.batch_pme_energy_corrections_double_backward(
                *common_in,
                _wp(sysd["bidx"], wp.int32, device),
                _wp(sysd["vol"], dtype, device),
                _wp(sysd["alpha"], dtype, device),
                _wp(sysd["qtot"], dtype, device),
                wp_gge,
                wp_graw,
                wp_gchg,
                wp_gvol,
                wp_galpha,
                wp_gqtot,
                wp_dtype=dtype,
                device=device,
            )
        else:
            pk.pme_energy_corrections_double_backward(
                *common_in,
                _wp(sysd["vol"], dtype, device),
                _wp(sysd["alpha"], dtype, device),
                _wp(sysd["qtot"], dtype, device),
                wp_gge,
                wp_graw,
                wp_gchg,
                wp_gvol,
                wp_galpha,
                wp_gqtot,
                wp_dtype=dtype,
                device=device,
            )
        wp.synchronize()

        # Factory.
        kernel = get_pme_kernel(
            dtype,
            component="pme_corrections",
            batched=is_batch,
            order="double_backward",
        )
        gge2, graw2, gchg2, gvol2, galpha2, gqtot2 = _alloc()
        wp_gge2 = _wp(gge2, dtype, device)
        wp_graw2 = _wp(graw2, dtype, device)
        wp_gchg2 = _wp(gchg2, dtype, device)
        wp_gvol2 = _wp(gvol2, dtype, device)
        wp_galpha2 = _wp(galpha2, dtype, device)
        wp_gqtot2 = _wp(gqtot2, dtype, device)
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                _wp(h_raw, dtype, device),
                _wp(h_chg, dtype, device),
                _wp(h_vol, dtype, device),
                _wp(h_alpha, dtype, device),
                _wp(h_qtot, dtype, device),
                _wp(grad_E, dtype, device),
                _wp(sysd["raw"], dtype, device),
                _wp(sysd["q"], dtype, device),
                _wp(sysd["bidx"], wp.int32, device),
                _wp(sysd["vol"], dtype, device),
                _wp(sysd["alpha"], dtype, device),
                _wp(sysd["qtot"], dtype, device),
                wp_gge2,
                wp_graw2,
                wp_gchg2,
                wp_gvol2,
                wp_galpha2,
                wp_gqtot2,
            ],
            device=device,
        )
        wp.synchronize()
        # Per-atom grads match exactly; scalar atomic reductions to round-off.
        assert np.array_equal(wp_gge2.numpy(), wp_gge.numpy())
        assert np.array_equal(wp_graw2.numpy(), wp_graw.numpy())
        assert np.array_equal(wp_gchg2.numpy(), wp_gchg.numpy())
        rtol = 1e-12 if dtype == wp.float64 else 1e-5
        np.testing.assert_allclose(
            wp_gvol2.numpy(), wp_gvol.numpy(), rtol=rtol, atol=0.0
        )
        np.testing.assert_allclose(
            wp_galpha2.numpy(), wp_galpha.numpy(), rtol=rtol, atol=0.0
        )
        np.testing.assert_allclose(
            wp_gqtot2.numpy(), wp_gqtot.numpy(), rtol=rtol, atol=0.0
        )


# ==============================================================================
# 2. Position-grad path + force-loss double-backward (public PME autograd path)
# ==============================================================================


def _pme_system(device, dtype=torch.float64):
    """Small neutral single-system for the reciprocal-space autograd path."""
    rng = np.random.default_rng(11)
    n = 6
    pos = torch.tensor(rng.uniform(0.3, 5.7, size=(n, 3)), dtype=dtype, device=device)
    q = torch.tensor([0.5, -0.5, 0.3, -0.3, 0.7, -0.7], dtype=dtype, device=device)
    q = q - q.mean()  # neutral
    cell = torch.eye(3, dtype=dtype, device=device) * 6.0
    return pos, q, cell


class TestPositionGradPath:
    """End-to-end PME reciprocal-space position-grad + force-loss second backward.

    The energy path is ``spline_spread -> rfftn -> pme_fused_convolve -> irfftn ->
    spline_gather -> corrections``; the convolve is the only Warp custom op in the
    chain to positions, so the force-loss second backward launches the convolve
    double-backward. Uses ``compute_forces=False`` so the path to positions is
    pure Torch autograd through the convolve (the analytic spline force returned by
    ``compute_forces=True`` would bypass the double-backward).
    """

    def _energy_fn(self, q, cell, device):
        from nvalchemiops.torch.interactions.electrostatics.pme import (
            pme_reciprocal_space,
        )

        def energy(p):
            return pme_reciprocal_space(
                p,
                q,
                cell,
                0.4,
                mesh_dimensions=(8, 8, 8),
                spline_order=4,
                compute_forces=False,
            ).sum()

        return energy

    def test_force_matches_fd(self, device):
        tdev = "cuda" if device == "cuda:0" else device
        pos, q, cell = _pme_system(tdev)
        pos = pos.detach().requires_grad_(True)
        energy = self._energy_fn(q, cell, tdev)

        gpos = torch.autograd.grad(energy(pos), pos, create_graph=True)[0]
        force = (-gpos).detach()

        eps = 1e-5
        fd = torch.zeros_like(pos)
        with torch.no_grad():
            for i in range(pos.shape[0]):
                for d in range(3):
                    pp = pos.detach().clone()
                    pp[i, d] += eps
                    pm = pos.detach().clone()
                    pm[i, d] -= eps
                    fd[i, d] = -(energy(pp) - energy(pm)) / (2 * eps)
        max_abs = float((force - fd).abs().max())
        assert max_abs < 1e-6, f"force FD max_abs={max_abs:.3e}"

    def test_force_loss_double_backward(self, device):
        # A force loss L = ||grad(E)||^2; its .backward()/grad exercises the
        # convolve double-backward. Must run (no "no autograd formula
        # registered") and match a nested finite difference.
        tdev = "cuda" if device == "cuda:0" else device
        pos, q, cell = _pme_system(tdev)
        pos = pos.detach().requires_grad_(True)
        energy = self._energy_fn(q, cell, tdev)

        gpos = torch.autograd.grad(energy(pos), pos, create_graph=True)[0]
        loss = (gpos**2).sum()
        g2 = torch.autograd.grad(loss, pos)[0]
        assert torch.isfinite(g2).all()
        assert float(g2.abs().max()) > 0.0

        # Nested FD reference: d/dR [ ||grad_R E||^2 ].
        def loss_fn(p):
            e = energy(p)
            g = torch.autograd.grad(e, p, create_graph=False)[0]
            return (g**2).sum()

        eps = 1e-5
        fd2 = torch.zeros_like(pos)
        for i in range(pos.shape[0]):
            for d in range(3):
                pp = pos.detach().clone()
                pp[i, d] += eps
                pp.requires_grad_(True)
                pm = pos.detach().clone()
                pm[i, d] -= eps
                pm.requires_grad_(True)
                fd2[i, d] = (loss_fn(pp) - loss_fn(pm)) / (2 * eps)
        max_abs = float((g2 - fd2).abs().max())
        assert max_abs < 1e-6, f"force-loss 2nd-bwd FD max_abs={max_abs:.3e}"


# ==============================================================================
# 3. Stress-path second backward (cell / volume) -- exercises the convolve
#    double-backward's cell/stress outputs (dL/dvolume, dL/dk_squared).
# ==============================================================================


class TestStressPathDoubleBackward:
    """End-to-end stress-loss-style second backward through the PME reciprocal
    energy w.r.t. the cell / volume.

    The cell flows into the convolve via ``k_squared`` (``generate_k_vectors_pme``)
    and ``volume`` (``det(cell)``), so a second backward w.r.t. the cell launches
    the convolve double-backward's cell/stress outputs (``dL/dk_squared`` per-k,
    ``dL/dvolume`` atomic-summed). These were previously deferred outputs;
    a nonzero, FD-correct result confirms nothing is silently zeroed on the stress
    path. The per-k math itself is pinned by ``TestConvolveDoubleBackwardFD``.
    """

    def test_supplied_volume_is_static_metadata(self, device):
        # Public PME treats caller-supplied volume as fixed setup metadata,
        # matching k_vectors / k_squared / cell_inv_t. Cell-derived volume
        # gradients are covered by test_full_cell_second_backward_nonzero and
        # low-level dL/dvolume correctness is pinned by TestConvolveDoubleBackwardFD.
        from nvalchemiops.torch.interactions.electrostatics.pme import (
            pme_reciprocal_space,
        )

        tdev = "cuda" if device == "cuda:0" else device
        pos, q, cell = _pme_system(tdev)
        v0 = float(torch.abs(torch.linalg.det(cell)))

        def energy_vol(v):
            return pme_reciprocal_space(
                pos,
                q,
                cell.detach(),
                0.4,
                mesh_dimensions=(8, 8, 8),
                spline_order=4,
                volume=v,
                compute_forces=False,
            ).sum()

        vol = torch.tensor([v0], dtype=torch.float64, device=tdev, requires_grad=True)
        energy = energy_vol(vol)
        assert not energy.requires_grad

    def test_full_cell_second_backward_nonzero(self, device):
        # Full cell second derivative: cell flows to E through spline, k_squared
        # AND volume. This is a runs+finite SMOKE test of the k_squared/volume ->
        # cell wiring: it confirms the convolve double-backward path is registered
        # (no "no autograd formula registered") and the result is finite + nonzero.
        # It does NOT validate the magnitude of the cell terms (the spline->cell
        # path alone makes g2 nonzero) -- the per-k correctness + nonzero-ness of
        # dL/dk_squared/dL/dvolume is pinned by TestConvolveDoubleBackwardFD.
        from nvalchemiops.torch.interactions.electrostatics.pme import (
            pme_reciprocal_space,
        )

        tdev = "cuda" if device == "cuda:0" else device
        pos, q, cell = _pme_system(tdev)

        def energy_cell(c):
            return pme_reciprocal_space(
                pos,
                q,
                c,
                0.4,
                mesh_dimensions=(8, 8, 8),
                spline_order=4,
                compute_forces=False,
            ).sum()

        c = cell.clone().requires_grad_(True)
        g = torch.autograd.grad(energy_cell(c), c, create_graph=True)[0]
        g2 = torch.autograd.grad(g.sum(), c)[0]
        assert torch.isfinite(g2).all()
        max_abs = float(g2.abs().max())
        assert max_abs > 1e-6, "cell second-order is silently zero"
