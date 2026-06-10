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

"""Tests for the forward-fused literal ``dE/dcell`` (``cell_literal``) path.

Two guarantees:

1. **Kernel parity** -- the ``cell_literal=True`` forward kernel's per-atom
   ``dedcell`` output, weighted by a per-atom energy cotangent and reduced per
   system, must equal (a) the FD-verified edge kernel ``_real_cell_grad_via_kernel``
   to round-off (identical Warp force-magnitude formula) and (b) the closed-form
   ``_real_space_dEdcell_analytic`` to within FD tolerance.
2. **End-to-end FD stress** -- the public ``ewald_real_space`` matrix path (which
   requests the ``cell_literal`` kernel when ``cell.requires_grad``) produces a
   stress (``-dE/dstrain``) matching central finite differences, for fixed and
   q(R) charges, single + batch.

Uses NONZERO neighbor-matrix shifts (a real periodic crystal) so the cell path is
actually exercised (with all-zero shifts ``dE/dcell = sum 0 (x) F = 0``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.interactions.electrostatics._factory_common import _DerivState
from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    REAL_SPACE_TILED_BLOCK_DIM,
)
from nvalchemiops.interactions.electrostatics.ewald_real_factory import (
    alloc_ewald_real_sentinels,
    get_ewald_real_kernel,
)
from nvalchemiops.torch.interactions.electrostatics._ewald_real_chain import (
    _real_cell_grad_via_kernel,
    _real_space_dEdcell_analytic,
)
from nvalchemiops.torch.interactions.electrostatics.ewald import ewald_real_space
from test.interactions.electrostatics._deriv_check import fixed_charge_system
from test.interactions.electrostatics.conftest import create_cscl_supercell

_MASK = 999
_DTYPES = [torch.float64, torch.float32]
_DTYPE_IDS = ["f64", "f32"]


def _csr_to_matrix(num_atoms, edge_i, edge_j, shifts):
    """Build a full (directed) neighbor matrix + integer shifts from edge lists."""
    counts = np.bincount(edge_i, minlength=num_atoms)
    max_nbr = int(counts.max())
    nm = np.full((num_atoms, max_nbr), _MASK, dtype=np.int64)
    nms = np.zeros((num_atoms, max_nbr, 3), dtype=np.int64)
    cursor = np.zeros(num_atoms, dtype=np.int64)
    for e in range(edge_i.shape[0]):
        i = int(edge_i[e])
        c = cursor[i]
        nm[i, c] = edge_j[e]
        nms[i, c] = shifts[e]
        cursor[i] = c + 1
    return nm, nms


def _matrix_system(dtype_t, device, batched, seed=3):
    """A CsCl supercell as a full neighbor matrix (nonzero shifts), single or batch."""
    sysd = fixed_charge_system(
        create_cscl_supercell,
        size=2,
        dtype=dtype_t,
        device=device,
        jitter=0.1,
        cutoff=5.0,
        seed=seed,
    )
    pos = sysd.positions.detach()
    chg = sysd.charges.detach()
    cell = sysd.cell.detach()
    alpha = sysd.alpha.detach()
    n = pos.shape[0]
    ei = sysd.neighbor_list[0].cpu().numpy()
    ej = sysd.neighbor_list[1].cpu().numpy()
    sh = sysd.neighbor_shifts.cpu().numpy()
    if batched:
        pos = torch.cat([pos, pos * 1.01], 0)
        chg = torch.cat([chg, -chg], 0)
        cell = torch.cat([cell, cell * 1.02], 0)
        alpha = torch.cat([alpha, alpha], 0)
        ei = np.concatenate([ei, ei + n])
        ej = np.concatenate([ej, ej + n])
        sh = np.concatenate([sh, sh], 0)
        batch_idx = torch.cat(
            [
                torch.zeros(n, dtype=torch.long, device=pos.device),
                torch.ones(n, dtype=torch.long, device=pos.device),
            ]
        )
        n_tot, nsys = 2 * n, 2
    else:
        batch_idx = None
        n_tot, nsys = n, 1
    nm_np, nms_np = _csr_to_matrix(n_tot, ei, ej, sh)
    assert (nms_np != 0).any(), "shifts all zero -- cell path untested"
    nm = torch.tensor(nm_np, dtype=torch.long, device=pos.device)
    nms = torch.tensor(nms_np, dtype=torch.long, device=pos.device)
    edge_i = torch.as_tensor(ei, dtype=torch.long, device=pos.device)
    edge_j = torch.as_tensor(ej, dtype=torch.long, device=pos.device)
    shifts = torch.as_tensor(sh, dtype=torch.long, device=pos.device)
    return dict(
        pos=pos,
        chg=chg,
        cell=cell,
        alpha=alpha,
        batch_idx=batch_idx,
        nm=nm,
        nms=nms,
        edge_i=edge_i,
        edge_j=edge_j,
        shifts=shifts,
        nsys=nsys,
        n_tot=n_tot,
    )


@pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
@pytest.mark.parametrize("dtype_t", _DTYPES, ids=_DTYPE_IDS)
def test_cell_literal_dedcell_matches_edge_kernel(device, dtype_t, batched):
    """cell_literal forward dedcell == FD-verified edge kernel (and ~= analytic)."""
    torch_device = torch.device(device)
    s = _matrix_system(dtype_t, torch_device, batched)
    wp_dtype = wp.float64 if dtype_t == torch.float64 else wp.float32
    wp_vec = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp_mat = wp.mat33d if wp_dtype == wp.float64 else wp.mat33f
    dev = wp.device_from_torch(torch_device)
    n_tot, nsys = s["n_tot"], s["nsys"]

    kernel = get_ewald_real_kernel(
        wp_dtype,
        batched=batched,
        neighbor_input="matrix",
        deriv_state=_DerivState.E_F,
        cell_grad=False,
        order="forward",
        tiled=True,
        cell_literal=True,
    )
    energies = wp.zeros(n_tot, dtype=wp.float64, device=dev)
    forces = wp.zeros(n_tot, dtype=wp_vec, device=dev)
    cg = wp.zeros(n_tot, dtype=wp.float64, device=dev)
    virial = wp.zeros(nsys, dtype=wp_mat, device=dev)
    dedcell = wp.zeros(n_tot, dtype=wp.mat33d, device=dev)

    def w(t, d):
        return wp.from_torch(t.detach().contiguous(), dtype=d, requires_grad=False)

    batch_w = (
        w(s["batch_idx"].to(torch.int32), wp.int32)
        if batched
        else wp.zeros(0, dtype=wp.int32, device=dev)
    )
    nm = wp.from_numpy(
        s["nm"].cpu().numpy().astype(np.int32), dtype=wp.int32, device=dev
    )
    nms = wp.from_numpy(
        s["nms"].cpu().numpy().astype(np.int32).reshape(n_tot, -1, 3),
        dtype=wp.vec3i,
        device=dev,
    )
    empty_i = wp.zeros(0, dtype=wp.int32, device=dev)
    empty_v = wp.zeros(0, dtype=wp.vec3i, device=dev)

    wp.launch_tiled(
        kernel,
        dim=[n_tot],
        inputs=[
            w(s["pos"], wp_vec),
            w(s["chg"], wp_dtype),
            w(s["cell"], wp_mat),
            batch_w,
            empty_i,
            empty_i,
            empty_v,
            nm,
            nms,
            wp.int32(_MASK),
            w(s["alpha"], wp_dtype),
            energies,
            forces,
            cg,
            virial,
            dedcell,
        ],
        block_dim=REAL_SPACE_TILED_BLOCK_DIM,
        device=dev,
    )
    wp.synchronize()
    dedcell_t = wp.to_torch(dedcell).to(torch.float64)

    gen = torch.Generator(device="cpu").manual_seed(7)
    ge = torch.randn(n_tot, generator=gen, dtype=torch.float64).to(torch_device)
    sys_of_atom = (
        s["batch_idx"]
        if batched
        else torch.zeros(n_tot, dtype=torch.long, device=torch_device)
    )
    gc_kernel = torch.zeros(
        (nsys, 3, 3), dtype=torch.float64, device=torch_device
    ).index_add(0, sys_of_atom, ge.view(-1, 1, 1) * dedcell_t)

    gc_edge = _real_cell_grad_via_kernel(
        s["pos"],
        s["chg"],
        s["cell"],
        s["alpha"],
        s["edge_i"],
        s["edge_j"],
        s["shifts"],
        s["batch_idx"],
        ge,
    ).to(torch.float64)
    gc_analytic = _real_space_dEdcell_analytic(
        s["pos"],
        s["chg"],
        s["cell"],
        s["alpha"],
        s["edge_i"],
        s["edge_j"],
        s["shifts"],
        s["batch_idx"],
        ge,
    ).to(torch.float64)

    scale = gc_analytic.abs().max().item() + 1e-30
    rel_edge = (gc_kernel - gc_edge).abs().max().item() / scale
    rel_an = (gc_kernel - gc_analytic).abs().max().item() / scale
    # Same Warp formula as the edge kernel => round-off; f32 looser.
    tol_edge = 1e-11 if dtype_t == torch.float64 else 1e-4
    assert rel_edge < tol_edge, f"vs edge kernel rel={rel_edge:.3e}"
    assert rel_an < 1e-5, f"vs analytic rel={rel_an:.3e}"


@pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
@pytest.mark.parametrize("dtype_t", _DTYPES, ids=_DTYPE_IDS)
def test_csr_cell_literal_dedcell_matches_edge_kernel(device, dtype_t, batched):
    """CSR/list cell_literal forward dedcell matches the edge-kernel oracle."""
    torch_device = torch.device(device)
    s = _matrix_system(dtype_t, torch_device, batched, seed=4)
    wp_dtype = wp.float64 if dtype_t == torch.float64 else wp.float32
    wp_vec = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp_mat = wp.mat33d if wp_dtype == wp.float64 else wp.mat33f
    dev = wp.device_from_torch(torch_device)
    n_tot, nsys = s["n_tot"], s["nsys"]

    edge_i_np = s["edge_i"].cpu().numpy()
    order = np.argsort(edge_i_np, kind="stable")
    edge_i_sorted = edge_i_np[order]
    edge_j = s["edge_j"].cpu().numpy()[order].astype(np.int32)
    shifts = s["shifts"].cpu().numpy()[order].astype(np.int32)
    counts = np.bincount(edge_i_sorted, minlength=n_tot).astype(np.int32)
    ptr = np.zeros(n_tot + 1, dtype=np.int32)
    ptr[1:] = np.cumsum(counts)

    kernel = get_ewald_real_kernel(
        wp_dtype,
        batched=batched,
        neighbor_input="list",
        deriv_state=_DerivState.E_F_dQ,
        cell_grad=False,
        order="forward",
        cell_literal=True,
    )
    sentinels = alloc_ewald_real_sentinels(wp_dtype, dev)
    energies = wp.zeros(n_tot, dtype=wp.float64, device=dev)
    forces = wp.zeros(n_tot, dtype=wp_vec, device=dev)
    cg = wp.zeros(n_tot, dtype=wp.float64, device=dev)
    virial = wp.zeros(nsys, dtype=wp_mat, device=dev)
    dedcell = wp.zeros(n_tot, dtype=wp.mat33d, device=dev)

    def w(t, d):
        return wp.from_torch(t.detach().contiguous(), dtype=d, requires_grad=False)

    batch_w = (
        w(s["batch_idx"].to(torch.int32), wp.int32)
        if batched
        else sentinels["batch_id"]
    )

    wp.launch(
        kernel,
        dim=n_tot,
        inputs=[
            w(s["pos"], wp_vec),
            w(s["chg"], wp_dtype),
            w(s["cell"], wp_mat),
            batch_w,
            wp.from_numpy(edge_j, dtype=wp.int32, device=dev),
            wp.from_numpy(ptr, dtype=wp.int32, device=dev),
            wp.from_numpy(shifts, dtype=wp.vec3i, device=dev),
            sentinels["neighbor_matrix"],
            sentinels["unit_shifts_matrix"],
            wp.int32(_MASK),
            w(s["alpha"], wp_dtype),
            energies,
            forces,
            cg,
            virial,
            dedcell,
        ],
        device=dev,
    )
    wp.synchronize()

    gen = torch.Generator(device="cpu").manual_seed(11)
    ge = torch.randn(n_tot, generator=gen, dtype=torch.float64).to(torch_device)
    sys_of_atom = (
        s["batch_idx"]
        if batched
        else torch.zeros(n_tot, dtype=torch.long, device=torch_device)
    )
    gc_kernel = torch.zeros(
        (nsys, 3, 3), dtype=torch.float64, device=torch_device
    ).index_add(
        0, sys_of_atom, ge.view(-1, 1, 1) * wp.to_torch(dedcell).to(torch.float64)
    )
    gc_edge = _real_cell_grad_via_kernel(
        s["pos"],
        s["chg"],
        s["cell"],
        s["alpha"],
        torch.as_tensor(edge_i_sorted, dtype=torch.long, device=torch_device),
        torch.as_tensor(edge_j, dtype=torch.long, device=torch_device),
        torch.as_tensor(shifts, dtype=torch.long, device=torch_device),
        s["batch_idx"],
        ge,
    ).to(torch.float64)
    scale = gc_edge.abs().max().item() + 1e-30
    rel_edge = (gc_kernel - gc_edge).abs().max().item() / scale
    tol_edge = 1e-11 if dtype_t == torch.float64 else 1e-4
    assert rel_edge < tol_edge, f"vs edge kernel rel={rel_edge:.3e}"


@pytest.mark.parametrize("qr", [False, True], ids=["fixed", "qR"])
@pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
@pytest.mark.parametrize("dtype_t", _DTYPES, ids=_DTYPE_IDS)
def test_cell_literal_fd_stress(device, dtype_t, batched, qr):
    """Public matrix-path stress (-dE/dstrain) matches central finite differences."""
    torch_device = torch.device(device)
    s = _matrix_system(dtype_t, torch_device, batched, seed=5)
    pos0, chg0, cell0, alpha = s["pos"], s["chg"], s["cell"], s["alpha"]
    nm, nms, batch_idx, nsys, n_tot = (
        s["nm"],
        s["nms"],
        s["batch_idx"],
        s["nsys"],
        s["n_tot"],
    )
    bidx_full = (
        batch_idx
        if batched
        else torch.zeros(n_tot, dtype=torch.long, device=torch_device)
    )

    def make_charges(pos_def):
        if not qr:
            return chg0
        raw = torch.sin(pos_def.sum(dim=1)) * 0.3
        if batched:
            counts = torch.bincount(bidx_full).to(raw.dtype)
            means = (
                torch.zeros(nsys, dtype=raw.dtype, device=raw.device).index_add(
                    0, bidx_full, raw
                )
                / counts
            )
            return raw - means.index_select(0, bidx_full)
        return raw - raw.mean()

    def energy_of_strain(strain):
        defo = torch.eye(3, dtype=pos0.dtype, device=torch_device) + strain
        pos_def = torch.einsum("nij,nj->ni", defo[bidx_full], pos0)
        cell_def = torch.einsum("sij,sjk->sik", cell0, defo.transpose(-1, -2))
        return ewald_real_space(
            positions=pos_def,
            charges=make_charges(pos_def),
            cell=cell_def,
            alpha=alpha,
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            mask_value=_MASK,
            batch_idx=batch_idx,
        ).sum()

    strain = torch.zeros(
        (nsys, 3, 3), dtype=pos0.dtype, device=torch_device, requires_grad=True
    )
    grad = torch.autograd.grad(energy_of_strain(strain), strain)[0]

    gen = torch.Generator(device="cpu").manual_seed(11)
    v = torch.randn((nsys, 3, 3), generator=gen, dtype=pos0.dtype).to(torch_device)
    v = v / v.norm()
    eps = 1e-6 if dtype_t == torch.float64 else 2e-3
    with torch.no_grad():
        fd = (
            energy_of_strain((strain + eps * v).detach())
            - energy_of_strain((strain - eps * v).detach())
        ) / (2 * eps)
    analytic = (grad * v).sum()
    rel = (analytic - fd).abs().item() / (fd.abs().item() + 1e-30)
    tol = 1e-5 if dtype_t == torch.float64 else 3e-2
    assert rel < tol, f"FD stress rel={rel:.3e} (analytic={analytic}, fd={fd})"


@pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
def test_cell_literal_cell_double_backward_gradgradcheck(device, batched):
    """Matrix ``d^2 E/dcell^2`` (path 3 lazy edge build) matches gradgradcheck.

    Under ``create_graph=True`` the cell backward takes :class:`_RealCellGrad`'s
    ``torch.is_grad_enabled()`` branch, which for the matrix layout builds the edge
    list lazily from the saved neighbor matrix (deferred from the forward) and runs
    the differentiable analytic ``dE/dcell``. ``gradgradcheck`` w.r.t. ``cell`` (with
    positions held constant) verifies this second derivative against central
    differences -- a sign/index error in the lazy-edge path 3 would fail it.

    Note: this isolates the ``cell`` second derivative. The full mixed pos-cell
    second derivative is a separate, layout-independent property of the factory chain
    (the detached ``dE/dR`` cache makes ``d(grad_pos)/d(cell)`` zero), unaffected by
    this fusion and out of scope here.
    """
    torch_device = torch.device(device)
    s = _matrix_system(torch.float64, torch_device, batched, seed=7)
    pos0, chg0, alpha = s["pos"], s["chg"], s["alpha"]
    nm, nms, batch_idx = s["nm"], s["nms"], s["batch_idx"]

    def energy_of_cell(cell):
        return ewald_real_space(
            positions=pos0,
            charges=chg0,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            mask_value=_MASK,
            batch_idx=batch_idx,
        ).sum()

    cell_in = s["cell"].clone().requires_grad_(True)
    assert torch.autograd.gradgradcheck(
        energy_of_cell, (cell_in,), eps=1e-6, atol=1e-4, rtol=1e-3
    )


@pytest.mark.parametrize("dtype_t", _DTYPES, ids=_DTYPE_IDS)
def test_cell_literal_strain_loss_double_backward_fd(device, dtype_t):
    """Stress-loss gradient w.r.t. strain matches finite differences."""
    torch_device = torch.device(device)
    s = _matrix_system(dtype_t, torch_device, batched=False, seed=9)
    pos0 = s["pos"].detach()
    chg = s["chg"].detach()
    cell0 = s["cell"].detach()
    alpha = s["alpha"].detach()
    nm = s["nm"]
    nms = s["nms"]
    atom_sys = torch.zeros(pos0.shape[0], dtype=torch.long, device=torch_device)

    def energy_of_strain(strain):
        eye = torch.eye(3, dtype=dtype_t, device=torch_device).unsqueeze(0)
        deform = eye + strain
        pos_def = torch.einsum("ni,nij->nj", pos0, deform[atom_sys])
        cell_def = torch.einsum("bij,bjk->bik", cell0, deform)
        return ewald_real_space(
            positions=pos_def,
            charges=chg,
            cell=cell_def,
            alpha=alpha,
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            mask_value=_MASK,
        ).sum()

    def loss_of_strain(strain):
        strain = strain.clone().requires_grad_(True)
        energy = energy_of_strain(strain)
        virial = -torch.autograd.grad(energy, strain, create_graph=True)[0]
        return virial.pow(2).sum()

    strain = torch.zeros(
        (1, 3, 3), dtype=dtype_t, device=torch_device, requires_grad=True
    )
    loss = loss_of_strain(strain)
    grad = torch.autograd.grad(loss, strain)[0]

    gen = torch.Generator(device="cpu").manual_seed(123)
    direction = torch.randn((1, 3, 3), generator=gen, dtype=dtype_t).to(torch_device)
    direction = direction / direction.norm()
    eps = 1e-5 if dtype_t == torch.float64 else 2e-3
    fd = (
        loss_of_strain((strain + eps * direction).detach())
        - loss_of_strain((strain - eps * direction).detach())
    ) / (2 * eps)
    analytic = (grad * direction).sum()
    rel = (analytic - fd).abs().item() / (fd.abs().item() + 1e-30)
    tol = 1e-5 if dtype_t == torch.float64 else 5e-3
    assert rel < tol, f"strain-loss DB rel={rel:.3e} analytic={analytic} fd={fd}"
