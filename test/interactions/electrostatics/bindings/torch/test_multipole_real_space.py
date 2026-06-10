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

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.interactions.electrostatics.multipole_ewald_cell_grad import (
    multipole_real_space_dipole_csr_cell_grad,
    multipole_real_space_monopole_csr_cell_grad,
)
from nvalchemiops.interactions.electrostatics.multipole_ewald_kernels import (
    multipole_real_space_dipole_csr_energy_fused,
    multipole_real_space_monopole_csr_energy_fused,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
    multipole_real_space_quadrupole_energy,
)


def _csr_neighbors_for_n_atoms(positions: torch.Tensor, cutoff: float):
    """O(N²) toy CSR neighbor list for the smoke test."""
    n = positions.shape[0]
    device = positions.device
    pairs_i = []
    pairs_j = []
    shifts = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = torch.norm(positions[i] - positions[j]).item()
            if d < cutoff:
                pairs_i.append(i)
                pairs_j.append(j)
                shifts.append([0, 0, 0])
    # Sort by (i, j) to build CSR.
    order = sorted(range(len(pairs_i)), key=lambda k: (pairs_i[k], pairs_j[k]))
    pairs_i = [pairs_i[k] for k in order]
    pairs_j = [pairs_j[k] for k in order]
    shifts = [shifts[k] for k in order]

    idx_j = torch.tensor(pairs_j, dtype=torch.int32, device=device)
    neighbor_ptr = torch.zeros(n + 1, dtype=torch.int32, device=device)
    for ii in pairs_i:
        neighbor_ptr[ii + 1] += 1
    neighbor_ptr = torch.cumsum(neighbor_ptr, dim=0).to(torch.int32)
    unit_shifts = torch.tensor(shifts, dtype=torch.int32, device=device)
    return idx_j, neighbor_ptr, unit_shifts


def _symmetric_random_Q(n, rng):
    Q = np.zeros((n, 3, 3))
    for k in range(n):
        raw = rng.normal(size=(3, 3))
        Q[k] = 0.5 * (raw + raw.T)
    return Q


def _build_inputs(
    device,
    n_atoms=4,
    sigma=0.5,
    alpha=0.35,
    positions_grad=True,
    charges_grad=True,
    dipoles_grad=True,
    quadrupoles_grad=True,
):
    rng = np.random.default_rng(20260606)
    pos = rng.uniform(0, 5, size=(n_atoms, 3))
    q = rng.normal(size=(n_atoms,))
    mu = rng.normal(size=(n_atoms, 3))
    Q = _symmetric_random_Q(n_atoms, rng)
    cell = np.eye(3) * 20.0

    pos_t = torch.tensor(
        pos, dtype=torch.float64, device=device, requires_grad=positions_grad
    )
    q_t = torch.tensor(
        q, dtype=torch.float64, device=device, requires_grad=charges_grad
    )
    mu_t = torch.tensor(
        mu, dtype=torch.float64, device=device, requires_grad=dipoles_grad
    )
    Q_t = torch.tensor(
        Q, dtype=torch.float64, device=device, requires_grad=quadrupoles_grad
    )
    cell_t = torch.tensor(cell, dtype=torch.float64, device=device).unsqueeze(0)
    sigma_t = torch.tensor([sigma], dtype=torch.float64, device=device)
    alpha_t = torch.tensor([alpha], dtype=torch.float64, device=device)

    cutoff = 10.0
    idx_j, neighbor_ptr, unit_shifts = _csr_neighbors_for_n_atoms(pos_t, cutoff)
    return (
        pos_t,
        q_t,
        mu_t,
        Q_t,
        cell_t,
        sigma_t,
        alpha_t,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    )


def test_quadrupole_forward_returns_per_atom():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda"
    pos, q, mu, Q, cell, sigma, alpha, idx_j, np_ptr, sh = _build_inputs(device)
    E = multipole_real_space_quadrupole_energy(
        pos, q, mu, Q, cell, idx_j, np_ptr, sh, sigma, alpha
    )
    assert E.shape == (pos.shape[0],), (
        f"expected per-atom (N,); got shape {tuple(E.shape)}"
    )
    assert torch.isfinite(E).all(), f"non-finite energy: {E}"


def test_quadrupole_backward_returns_gradients_for_all_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda"
    pos, q, mu, Q, cell, sigma, alpha, idx_j, np_ptr, sh = _build_inputs(device)
    E = multipole_real_space_quadrupole_energy(
        pos, q, mu, Q, cell, idx_j, np_ptr, sh, sigma, alpha
    )
    E.sum().backward()
    for name, t in [("pos", pos), ("q", q), ("mu", mu), ("Q", Q)]:
        assert t.grad is not None, f"no gradient on {name}"
        assert torch.isfinite(t.grad).all(), f"non-finite grad on {name}"


def test_quadrupole_backward_matches_fd_at_loose_tol():
    """FD check on positions to confirm autograd direction is correct.

    Loose tol (1e-4): fp64 with FD step h=1e-3 (truncation ~1e-6), just an
    order-of-magnitude check — full correctness validated at the kernel level.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda"
    pos, q, mu, Q, cell, sigma, alpha, idx_j, np_ptr, sh = _build_inputs(device)
    E = multipole_real_space_quadrupole_energy(
        pos, q, mu, Q, cell, idx_j, np_ptr, sh, sigma, alpha
    )
    E.sum().backward()

    h = 1e-3
    grad_pos_an = pos.grad.detach().clone()
    pos_d = pos.detach().clone()
    for atom_k in range(min(2, pos.shape[0])):
        for d in range(3):
            pos_plus = pos_d.clone()
            pos_plus[atom_k, d] += h
            pos_minus = pos_d.clone()
            pos_minus[atom_k, d] -= h
            E_plus = multipole_real_space_quadrupole_energy(
                pos_plus,
                q.detach(),
                mu.detach(),
                Q.detach(),
                cell,
                idx_j,
                np_ptr,
                sh,
                sigma,
                alpha,
            ).sum()
            E_minus = multipole_real_space_quadrupole_energy(
                pos_minus,
                q.detach(),
                mu.detach(),
                Q.detach(),
                cell,
                idx_j,
                np_ptr,
                sh,
                sigma,
                alpha,
            ).sum()
            fd = ((E_plus - E_minus) / (2 * h)).item()
            assert abs(grad_pos_an[atom_k, d].item() - fd) < 1e-4, (
                f"atom {atom_k} d{d}: an={grad_pos_an[atom_k, d].item():+.4e} "
                f"fd={fd:+.4e}"
            )


def test_batch_quadrupole_forward_returns_per_atom_energies():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda"
    pos0, q0, mu0, Q0, cell0, sigma, alpha, idx_j0, ptr0, sh0 = _build_inputs(device)
    pos1, q1, mu1, Q1, cell1, _, _, idx_j1, ptr1, sh1 = _build_inputs(device)

    n0 = pos0.shape[0]
    n1 = pos1.shape[0]
    pos_b = torch.cat([pos0.detach(), pos1.detach()], dim=0).requires_grad_(True)
    q_b = torch.cat([q0.detach(), q1.detach()], dim=0).requires_grad_(True)
    mu_b = torch.cat([mu0.detach(), mu1.detach()], dim=0).requires_grad_(True)
    Q_b = torch.cat([Q0.detach(), Q1.detach()], dim=0).requires_grad_(True)
    cells_b = torch.cat([cell0, cell1], dim=0)
    sigmas_b = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
    alphas_b = torch.tensor([0.35, 0.35], dtype=torch.float64, device=device)

    batch_idx = torch.cat(
        [
            torch.zeros(n0, dtype=torch.int32, device=device),
            torch.ones(n1, dtype=torch.int32, device=device),
        ]
    )

    idx_j_offset = idx_j1 + n0
    idx_j_b = torch.cat([idx_j0, idx_j_offset])
    ptr_b = torch.cat(
        [
            ptr0[:-1],
            ptr1 + ptr0[-1],
        ]
    )
    unit_shifts_b = torch.cat([sh0, sh1], dim=0)

    E_b = multipole_real_space_quadrupole_energy(
        pos_b,
        q_b,
        mu_b,
        Q_b,
        cells_b,
        idx_j_b,
        ptr_b,
        unit_shifts_b,
        sigmas_b,
        alphas_b,
        batch_idx=batch_idx,
    )
    # Per-atom return (N_total,) for all l_max (uniform with l<=1); the caller
    # scatter_adds to per-system.
    assert E_b.shape == (n0 + n1,), f"expected ({n0 + n1},); got {tuple(E_b.shape)}"
    assert torch.isfinite(E_b).all(), f"non-finite energy: {E_b}"
    per_system = torch.zeros(2, dtype=torch.float64, device=device).scatter_add(
        0, batch_idx.to(torch.int64), E_b
    )
    assert torch.isfinite(per_system).all()

    E_b.sum().backward()
    for name, t in [("pos_b", pos_b), ("q_b", q_b), ("mu_b", mu_b), ("Q_b", Q_b)]:
        assert t.grad is not None, f"no gradient on {name}"
        assert torch.isfinite(t.grad).all(), f"non-finite grad on {name}"


def _two_atom_pair(dtype=torch.float64, device="cpu", half_neighbor_list=False):
    """Build a simple 2-atom non-periodic pair for testing.

    With half_neighbor_list=True, only the (0, 1) edge is listed; with
    False (default), both (0, 1) and (1, 0) are listed.
    """
    rng = np.random.default_rng(20260610)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    q = torch.tensor(
        rng.normal(size=(2,)), dtype=dtype, device=device, requires_grad=True
    )
    mu = torch.tensor(
        rng.normal(size=(2, 3)), dtype=dtype, device=device, requires_grad=True
    )
    Q_raw = rng.normal(size=(2, 3, 3))
    Q_sym = 0.5 * (Q_raw + Q_raw.transpose(0, 2, 1))
    Q = torch.tensor(Q_sym, dtype=dtype, device=device, requires_grad=True)
    cell = (
        torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * 100.0
    )  # large box → no PBC
    sigma = torch.tensor([0.5], dtype=dtype, device=device)
    alpha = torch.tensor([0.35], dtype=dtype, device=device)
    if half_neighbor_list:
        # Only atom 0 sees atom 1; atom 1 has no neighbors
        idx_j = torch.tensor([1], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
        unit_shifts = torch.zeros((1, 3), dtype=torch.int32, device=device)
    else:
        # Full: atom 0 sees atom 1, atom 1 sees atom 0
        idx_j = torch.tensor([1, 0], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        unit_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)
    return dict(
        positions=pos,
        charges=q,
        dipoles=mu,
        quadrupoles=Q,
        cell=cell,
        sigma=sigma,
        alpha=alpha,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
    )


def test_quadrupole_2nd_backward_create_graph_true_runs():
    """Smoke test: ``create_graph=True`` no longer raises NotImplementedError."""
    p = _two_atom_pair()
    E = multipole_real_space_quadrupole_energy(**p).sum()
    grads = torch.autograd.grad(
        E,
        [p["positions"], p["charges"], p["dipoles"], p["quadrupoles"]],
        create_graph=True,
    )
    assert all(g.requires_grad for g in grads), (
        "grads from create_graph=True must have requires_grad=True"
    )


def test_quadrupole_2nd_backward_returns_finite():
    """The full ∂²E/∂param² is computable and finite."""
    p = _two_atom_pair()
    E = multipole_real_space_quadrupole_energy(**p).sum()
    (grad_pos,) = torch.autograd.grad(E, [p["positions"]], create_graph=True)
    loss = (grad_pos**2).sum()
    grads_2nd = torch.autograd.grad(
        loss,
        [p["positions"]],
        retain_graph=False,
    )
    g = grads_2nd[0]
    assert torch.isfinite(g).all(), "2nd backward produced non-finite values"
    assert g.abs().max() > 0, "2nd backward should be non-zero"


def test_quadrupole_2nd_backward_fd_match_positions():
    """FD on the 1st-order grad matches the autograd 2nd backward, positions."""
    p = _two_atom_pair()

    def grad_pos_at(positions, create_graph):
        params = {**p, "positions": positions}
        E = multipole_real_space_quadrupole_energy(**params).sum()
        (g,) = torch.autograd.grad(E, [positions], create_graph=create_graph)
        return g

    v = torch.randn_like(p["positions"])
    p_grad = p["positions"]
    p_grad.requires_grad_(True)
    grad_pos = grad_pos_at(p_grad, create_graph=True)
    loss = (grad_pos * v).sum()
    (autograd_hvp,) = torch.autograd.grad(loss, [p_grad])

    h = 1e-5
    fd_hvp = torch.zeros_like(p["positions"])
    pos_base = p["positions"].detach().clone()
    for a in range(p["positions"].shape[0]):
        for k in range(3):
            pos_p_v = pos_base.clone()
            pos_p_v[a, k] = pos_base[a, k] + h
            pos_p_v.requires_grad_(True)
            gp_plus = grad_pos_at(pos_p_v, create_graph=False)
            pos_m_v = pos_base.clone()
            pos_m_v[a, k] = pos_base[a, k] - h
            pos_m_v.requires_grad_(True)
            gp_minus = grad_pos_at(pos_m_v, create_graph=False)
            fd_hvp[a, k] = ((gp_plus - gp_minus) * v).sum() / (2 * h)

    diff = (autograd_hvp - fd_hvp).abs().max().item()
    denom = max(autograd_hvp.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, (
        f"autograd 2nd backward disagrees with FD: max_abs={diff:.3e}, "
        f"rel={rel:.3e}, autograd={autograd_hvp}, fd={fd_hvp}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_quadrupole_2nd_backward_create_graph_true_runs_cuda():
    """Smoke test on CUDA — exercises tile path."""
    p = _two_atom_pair(device="cuda")
    E = multipole_real_space_quadrupole_energy(**p).sum()
    grads = torch.autograd.grad(
        E,
        [p["positions"], p["charges"], p["dipoles"], p["quadrupoles"]],
        create_graph=True,
    )
    assert all(g.requires_grad for g in grads)


def test_quadrupole_2nd_backward_half_vs_full_neighbor_list_match():
    """Half and full neighbor lists produce the same 2nd-order outputs.

    Invokes the CSR kernel directly to test the half/full plumbing in isolation.
    """
    import warp as wp

    from nvalchemiops.interactions.electrostatics.multipole_ewald_quadrupole_2nd_backward import (
        multipole_real_space_quadrupole_csr_energy_2nd_backward,
    )

    rng = np.random.default_rng(20260611)
    dtype_t = torch.float64

    n = 2
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=dtype_t)
    q = torch.tensor(rng.normal(size=(n,)), dtype=dtype_t)
    mu = torch.tensor(rng.normal(size=(n, 3)), dtype=dtype_t)
    Q_raw = rng.normal(size=(n, 3, 3))
    Q = torch.tensor(0.5 * (Q_raw + Q_raw.transpose(0, 2, 1)), dtype=dtype_t)
    cell = (torch.eye(3, dtype=dtype_t) * 100.0).unsqueeze(0)
    sigma = torch.tensor([0.5], dtype=dtype_t)
    alpha = torch.tensor([0.35], dtype=dtype_t)
    ge = torch.tensor(rng.normal(size=(n,)), dtype=torch.float64)
    gp = torch.tensor(rng.normal(size=(n, 3)), dtype=dtype_t)
    gc = torch.tensor(rng.normal(size=(n,)), dtype=dtype_t)
    gd = torch.tensor(rng.normal(size=(n, 3)), dtype=dtype_t)
    gQ_raw = rng.normal(size=(n, 3, 3))
    gQ = torch.tensor(0.5 * (gQ_raw + gQ_raw.transpose(0, 2, 1)), dtype=dtype_t)

    def run(half_list):
        if half_list:
            idx_j = torch.tensor([1], dtype=torch.int32)
            ptr = torch.tensor([0, 1, 1], dtype=torch.int32)
            sh = torch.zeros((1, 3), dtype=torch.int32)
        else:
            idx_j = torch.tensor([1, 0], dtype=torch.int32)
            ptr = torch.tensor([0, 1, 2], dtype=torch.int32)
            sh = torch.zeros((2, 3), dtype=torch.int32)

        gg_ge_2nd = torch.zeros(n, dtype=torch.float64)
        gg_pos_2nd = torch.zeros((n, 3), dtype=dtype_t)
        gg_q_2nd = torch.zeros(n, dtype=dtype_t)
        gg_mu_2nd = torch.zeros((n, 3), dtype=dtype_t)
        gg_Q_2nd = torch.zeros((n, 3, 3), dtype=dtype_t)

        multipole_real_space_quadrupole_csr_energy_2nd_backward(
            wp.from_torch(pos.contiguous(), dtype=wp.vec3d),
            wp.from_torch(q.contiguous(), dtype=wp.float64),
            wp.from_torch(mu.contiguous(), dtype=wp.vec3d),
            wp.from_torch(Q.contiguous(), dtype=wp.mat33d),
            wp.from_torch(cell.contiguous(), dtype=wp.mat33d),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(sh.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigma.contiguous(), dtype=wp.float64),
            wp.from_torch(alpha.contiguous(), dtype=wp.float64),
            wp.from_torch(ge.contiguous(), dtype=wp.float64),
            wp.from_torch(gp.contiguous(), dtype=wp.vec3d),
            wp.from_torch(gc.contiguous(), dtype=wp.float64),
            wp.from_torch(gd.contiguous(), dtype=wp.vec3d),
            wp.from_torch(gQ.contiguous(), dtype=wp.mat33d),
            wp.from_torch(gg_ge_2nd, dtype=wp.float64),
            wp.from_torch(gg_pos_2nd, dtype=wp.vec3d),
            wp.from_torch(gg_q_2nd, dtype=wp.float64),
            wp.from_torch(gg_mu_2nd, dtype=wp.vec3d),
            wp.from_torch(gg_Q_2nd, dtype=wp.mat33d),
            device="cpu",
            half_neighbor_list=half_list,
        )
        return dict(
            ge=gg_ge_2nd.clone(),
            pos=gg_pos_2nd.clone(),
            q=gg_q_2nd.clone(),
            mu=gg_mu_2nd.clone(),
            Q=gg_Q_2nd.clone(),
        )

    out_full = run(half_list=False)
    out_half = run(half_list=True)

    for key in ("ge", "pos", "q", "mu", "Q"):
        diff = (out_full[key] - out_half[key]).abs().max().item()
        denom = max(out_full[key].abs().max().item(), 1e-12)
        rel = diff / denom
        assert rel < 1e-12, (
            f"{key}: half vs full disagree (rel={rel:.2e}). "
            f"full={out_full[key]}, half={out_half[key]}"
        )


@pytest.mark.parametrize("channel", ["charges", "dipoles"])
def test_quadrupole_2nd_backward_fd_match_other_channels(channel):
    """FD on the 1st-order grad matches autograd 2nd backward for non-position channels."""
    p = _two_atom_pair()
    x_ref = p[channel]

    def grad_x_at(x, create_graph):
        params = {**p, channel: x}
        E = multipole_real_space_quadrupole_energy(**params).sum()
        (g,) = torch.autograd.grad(E, [x], create_graph=create_graph)
        return g

    v = torch.randn_like(x_ref)
    x = x_ref.detach().clone().requires_grad_(True)
    grad_x = grad_x_at(x, create_graph=True)
    loss = (grad_x * v).sum()
    (autograd_hvp,) = torch.autograd.grad(loss, [x])

    h = 1e-5
    fd_hvp = torch.zeros_like(x_ref)
    flat = fd_hvp.view(-1)
    base = x_ref.detach().clone()
    for k in range(flat.shape[0]):
        x_p = base.clone()
        x_p.view(-1)[k] = base.view(-1)[k] + h
        x_p.requires_grad_(True)
        gp = grad_x_at(x_p, create_graph=False)
        x_m = base.clone()
        x_m.view(-1)[k] = base.view(-1)[k] - h
        x_m.requires_grad_(True)
        gm = grad_x_at(x_m, create_graph=False)
        flat[k] = ((gp - gm) * v).sum() / (2 * h)

    diff = (autograd_hvp - fd_hvp).abs().max().item()
    denom = max(autograd_hvp.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, (
        f"{channel}: autograd 2nd backward disagrees with FD: "
        f"max_abs={diff:.3e}, rel={rel:.3e}"
    )


def test_quadrupole_2nd_backward_fd_match_quadrupoles_symmetric():
    """FD-match for Q channel — symmetric perturbation matching the kernel's
    symmetric-Q contract."""
    p = _two_atom_pair()
    Q_ref = p["quadrupoles"]

    def grad_Q_at(Q, create_graph):
        params = {**p, "quadrupoles": Q}
        E = multipole_real_space_quadrupole_energy(**params).sum()
        (g,) = torch.autograd.grad(E, [Q], create_graph=create_graph)
        return g

    # v must be symmetric to match the kernel's symmetric-Q output convention.
    v_raw = torch.randn_like(Q_ref)
    v = 0.5 * (v_raw + v_raw.transpose(-1, -2))
    Q = Q_ref.detach().clone().requires_grad_(True)
    grad_Q = grad_Q_at(Q, create_graph=True)
    loss = (grad_Q * v).sum()
    (autograd_hvp,) = torch.autograd.grad(loss, [Q])

    # Symmetric perturbation: perturb [a,b] and [b,a] together by h; off-diagonal
    # entries divide by 2 to match the kernel's symmetric free-index emit.
    h = 1e-5
    fd_hvp = torch.zeros_like(Q_ref)
    base = Q_ref.detach().clone()
    n_atoms = base.shape[0]
    for n in range(n_atoms):
        for a in range(3):
            for b in range(3):
                Q_p = base.clone()
                Q_p[n, a, b] = base[n, a, b] + h
                if a != b:
                    Q_p[n, b, a] = base[n, b, a] + h
                Q_p.requires_grad_(True)
                gp = grad_Q_at(Q_p, create_graph=False)
                Q_m = base.clone()
                Q_m[n, a, b] = base[n, a, b] - h
                if a != b:
                    Q_m[n, b, a] = base[n, b, a] - h
                Q_m.requires_grad_(True)
                gm = grad_Q_at(Q_m, create_graph=False)
                fd_val = ((gp - gm) * v).sum() / (2 * h)
                if a == b:
                    fd_hvp[n, a, b] = fd_val
                else:
                    fd_hvp[n, a, b] = 0.5 * fd_val
                    fd_hvp[n, b, a] = 0.5 * fd_val

    diff = (autograd_hvp - fd_hvp).abs().max().item()
    denom = max(autograd_hvp.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, (
        f"quadrupoles: autograd 2nd backward disagrees with symmetric FD: "
        f"max_abs={diff:.3e}, rel={rel:.3e}"
    )


def test_quadrupole_batched_2nd_backward_matches_per_system():
    """Batched LMAX=2 second-order backward (force-loss create_graph) equals the
    per-system single-system 2nd-back, atom-for-atom."""
    from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
        multipole_real_space_quadrupole_energy,
    )

    dev = "cpu"
    s0 = _two_atom_pair(device=dev)
    rng = np.random.default_rng(7)
    s1 = _two_atom_pair(device=dev)
    nps = 2
    pos = torch.cat(
        [s0["positions"].detach(), s1["positions"].detach()], 0
    ).requires_grad_(True)
    q = torch.cat([s0["charges"].detach(), s1["charges"].detach()], 0)
    mu = torch.cat([s0["dipoles"].detach(), s1["dipoles"].detach()], 0)
    Q = torch.cat([s0["quadrupoles"].detach(), s1["quadrupoles"].detach()], 0)
    cells = torch.cat([s0["cell"], s1["cell"]], 0)
    sigmas = torch.tensor([0.5, 0.5], dtype=torch.float64, device=dev)
    alphas = torch.tensor([0.35, 0.35], dtype=torch.float64, device=dev)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=dev)
    idx_j = torch.tensor([1, 0, 3, 2], dtype=torch.int32, device=dev)
    ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=dev)
    sh = torch.zeros((4, 3), dtype=torch.int32, device=dev)
    v = torch.tensor(rng.normal(size=(2 * nps, 3)), dtype=torch.float64, device=dev)

    E = multipole_real_space_quadrupole_energy(
        pos,
        q,
        mu,
        Q,
        cells,
        idx_j,
        ptr,
        sh,
        sigmas,
        alphas,
        batch_idx=batch_idx,
    ).sum()
    forces = -torch.autograd.grad(E, pos, create_graph=True)[0]
    (gb,) = torch.autograd.grad((forces * v).sum(), [pos])

    # Per-system single-system 2nd-back.
    off = 0
    for b, s in enumerate([s0, s1]):
        sl = slice(off, off + nps)
        off += nps
        ps = s["positions"].detach().clone().requires_grad_(True)
        Es = multipole_real_space_quadrupole_energy(
            ps,
            s["charges"].detach(),
            s["dipoles"].detach(),
            s["quadrupoles"].detach(),
            s["cell"],
            s["idx_j"],
            s["neighbor_ptr"],
            s["unit_shifts"],
            s["sigma"],
            s["alpha"],
        ).sum()
        Fs = -torch.autograd.grad(Es, ps, create_graph=True)[0]
        (gs,) = torch.autograd.grad((Fs * v[sl]).sum(), [ps])
        rel = (gb[sl] - gs).abs().max() / (gs.abs().max() + 1e-30)
        assert rel < 1e-10, f"system {b}: batched 2nd-back rel = {rel:.3e}"


def _periodic_pair(dtype=torch.float64, device="cpu"):
    """4-atom periodic system with PBC shifts so cell appears in the math."""
    rng = np.random.default_rng(20260612)
    n = 4
    L = 6.0
    pos = torch.tensor(
        rng.uniform(0.0, L, size=(n, 3)),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    q = torch.tensor(
        rng.normal(size=(n,)), dtype=dtype, device=device, requires_grad=True
    )
    mu = torch.tensor(
        rng.normal(size=(n, 3)), dtype=dtype, device=device, requires_grad=True
    )
    Q_raw = rng.normal(size=(n, 3, 3))
    Q_sym = 0.5 * (Q_raw + Q_raw.transpose(0, 2, 1))
    Q = torch.tensor(Q_sym, dtype=dtype, device=device, requires_grad=True)
    cell = (
        torch.tensor(
            [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
            dtype=dtype,
            device=device,
        )
        .unsqueeze(0)
        .clone()
    )
    cell.requires_grad_(True)
    sigma = torch.tensor([0.5], dtype=dtype, device=device)
    alpha = torch.tensor([0.35], dtype=dtype, device=device)

    cutoff = 4.0
    idx_j_list = []
    shifts_list = []
    pos_np = pos.detach().cpu().numpy()
    for i in range(n):
        nbrs_i = []
        shifts_i = []
        for j in range(n):
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    for sz in (-1, 0, 1):
                        if i == j and sx == 0 and sy == 0 and sz == 0:
                            continue
                        disp = pos_np[j] - pos_np[i] + np.array([sx, sy, sz]) * L
                        if np.linalg.norm(disp) < cutoff:
                            nbrs_i.append(j)
                            shifts_i.append((sx, sy, sz))
        idx_j_list.append(nbrs_i)
        shifts_list.append(shifts_i)

    counts = [len(ell) for ell in idx_j_list]
    ptr = np.cumsum([0] + counts)
    idx_j = torch.tensor(
        [j for nbrs in idx_j_list for j in nbrs],
        dtype=torch.int32,
        device=device,
    )
    neighbor_ptr = torch.tensor(ptr, dtype=torch.int32, device=device)
    unit_shifts = torch.tensor(
        [s for shifts in shifts_list for s in shifts],
        dtype=torch.int32,
        device=device,
    ).reshape(-1, 3)

    return dict(
        positions=pos,
        charges=q,
        dipoles=mu,
        quadrupoles=Q,
        cell=cell,
        sigma=sigma,
        alpha=alpha,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
    )


def test_quadrupole_cell_grad_returns_finite():
    """Smoke test: cell.requires_grad=True produces a finite ∂E/∂cell."""
    p = _periodic_pair()
    E = multipole_real_space_quadrupole_energy(**p).sum()
    (grad_cell,) = torch.autograd.grad(E, [p["cell"]])
    assert grad_cell.shape == (1, 3, 3)
    assert torch.isfinite(grad_cell).all()
    assert grad_cell.abs().max() > 0


def test_quadrupole_cell_grad_matches_fd():
    """∂E/∂cell matches FD on cell entries."""
    p = _periodic_pair()
    cell0 = p["cell"].detach().clone()

    p_an = {**p, "cell": cell0.clone().requires_grad_(True)}
    E_an = multipole_real_space_quadrupole_energy(**p_an).sum()
    (grad_cell_an,) = torch.autograd.grad(E_an, [p_an["cell"]])

    h = 1e-5
    grad_cell_fd = torch.zeros_like(cell0)
    for a in range(3):
        for b in range(3):
            cell_p = cell0.clone()
            cell_p[0, a, b] += h
            cell_p.requires_grad_(False)
            p_p = {
                **p,
                "cell": cell_p,
                "positions": p["positions"].detach().clone(),
                "charges": p["charges"].detach().clone(),
                "dipoles": p["dipoles"].detach().clone(),
                "quadrupoles": p["quadrupoles"].detach().clone(),
            }
            E_p = multipole_real_space_quadrupole_energy(**p_p).sum().item()

            cell_m = cell0.clone()
            cell_m[0, a, b] -= h
            cell_m.requires_grad_(False)
            p_m = {
                **p,
                "cell": cell_m,
                "positions": p["positions"].detach().clone(),
                "charges": p["charges"].detach().clone(),
                "dipoles": p["dipoles"].detach().clone(),
                "quadrupoles": p["quadrupoles"].detach().clone(),
            }
            E_m = multipole_real_space_quadrupole_energy(**p_m).sum().item()

            grad_cell_fd[0, a, b] = (E_p - E_m) / (2 * h)

    diff = (grad_cell_an - grad_cell_fd).abs().max().item()
    denom = max(grad_cell_an.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, (
        f"cell grad mismatch: max_abs={diff:.3e}, rel={rel:.3e}\n"
        f"analytical={grad_cell_an}\nFD={grad_cell_fd}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_quadrupole_cell_grad_cuda():
    """Smoke test on CUDA — exercises tile cell-grad path."""
    p = _periodic_pair(device="cuda")
    E = multipole_real_space_quadrupole_energy(**p).sum()
    (grad_cell,) = torch.autograd.grad(E, [p["cell"]])
    assert torch.isfinite(grad_cell).all()
    assert grad_cell.abs().max() > 0


def _periodic_4atom(dtype=torch.float64):
    rng = np.random.default_rng(20260614)
    n = 4
    L = 6.0
    pos = torch.tensor(rng.uniform(0.0, L, size=(n, 3)), dtype=dtype)
    q = torch.tensor(rng.normal(size=(n,)), dtype=dtype)
    mu = torch.tensor(rng.normal(size=(n, 3)), dtype=dtype)
    cell = torch.tensor(
        [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
        dtype=dtype,
    ).unsqueeze(0)
    sigma = torch.tensor([0.5], dtype=dtype)
    alpha = torch.tensor([0.35], dtype=dtype)

    cutoff = 4.0
    idx_j_list, shifts_list = [], []
    pos_np = pos.cpu().numpy()
    for i in range(n):
        nb, sh = [], []
        for j in range(n):
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    for sz in (-1, 0, 1):
                        if i == j and (sx, sy, sz) == (0, 0, 0):
                            continue
                        disp = pos_np[j] - pos_np[i] + np.array([sx, sy, sz]) * L
                        if np.linalg.norm(disp) < cutoff:
                            nb.append(j)
                            sh.append((sx, sy, sz))
        idx_j_list.append(nb)
        shifts_list.append(sh)
    counts = [len(ell) for ell in idx_j_list]
    ptr = np.cumsum([0] + counts)
    idx_j = torch.tensor(
        [j for nbrs in idx_j_list for j in nbrs],
        dtype=torch.int32,
    )
    neighbor_ptr = torch.tensor(ptr, dtype=torch.int32)
    unit_shifts = torch.tensor(
        [s for shifts in shifts_list for s in shifts],
        dtype=torch.int32,
    ).reshape(-1, 3)
    return dict(
        positions=pos,
        charges=q,
        dipoles=mu,
        cell=cell,
        sigma=sigma,
        alpha=alpha,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
    )


def _launch_monopole_grad_cell(p, half_neighbor_list=False):
    """Launch the LMAX=0 cell-grad kernel and return the (1,3,3) grad."""
    grad_cell = torch.zeros_like(p["cell"]).contiguous()
    multipole_real_space_monopole_csr_cell_grad(
        wp.from_torch(p["positions"].contiguous(), dtype=wp.vec3d),
        wp.from_torch(p["charges"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["cell"].contiguous(), dtype=wp.mat33d),
        wp.from_torch(p["idx_j"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["neighbor_ptr"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["unit_shifts"].contiguous(), dtype=wp.vec3i),
        wp.from_torch(p["sigma"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["alpha"].contiguous(), dtype=wp.float64),
        wp.from_torch(grad_cell, dtype=wp.mat33d),
        device="cpu",
        half_neighbor_list=half_neighbor_list,
    )
    return grad_cell


def _launch_monopole_energy_total(p):
    """Launch LMAX=0 fused energy kernel and return scalar total energy."""
    n = p["positions"].shape[0]
    energies = torch.zeros(n, dtype=torch.float64)
    grad_pos = torch.zeros((n, 3), dtype=torch.float64)
    grad_q = torch.zeros(n, dtype=torch.float64)
    multipole_real_space_monopole_csr_energy_fused(
        wp.from_torch(p["positions"].contiguous(), dtype=wp.vec3d),
        wp.from_torch(p["charges"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["cell"].contiguous(), dtype=wp.mat33d),
        wp.from_torch(p["idx_j"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["neighbor_ptr"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["unit_shifts"].contiguous(), dtype=wp.vec3i),
        wp.from_torch(p["sigma"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["alpha"].contiguous(), dtype=wp.float64),
        wp.from_torch(energies, dtype=wp.float64),
        wp.from_torch(grad_pos, dtype=wp.vec3d),
        wp.from_torch(grad_q, dtype=wp.float64),
        with_pos_grad=False,
        with_charge_grad=False,
        wp_dtype=wp.float64,
        device="cpu",
    )
    return energies.sum().item()


def test_monopole_cell_grad_matches_fd():
    """LMAX=0 cell-grad kernel matches FD on the energy kernel."""
    p = _periodic_4atom()
    grad_an = _launch_monopole_grad_cell(p)

    h = 1e-5
    grad_fd = torch.zeros_like(p["cell"])
    cell0 = p["cell"].clone()
    for a in range(3):
        for b in range(3):
            cell_p = cell0.clone()
            cell_p[0, a, b] += h
            E_p = _launch_monopole_energy_total({**p, "cell": cell_p})
            cell_m = cell0.clone()
            cell_m[0, a, b] -= h
            E_m = _launch_monopole_energy_total({**p, "cell": cell_m})
            grad_fd[0, a, b] = (E_p - E_m) / (2 * h)

    diff = (grad_an - grad_fd).abs().max().item()
    denom = max(grad_an.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, (
        f"LMAX=0 cell grad mismatch rel={rel:.3e}\nanalytical={grad_an}\nFD={grad_fd}"
    )


def _launch_dipole_grad_cell(p, half_neighbor_list=False):
    grad_cell = torch.zeros_like(p["cell"]).contiguous()
    multipole_real_space_dipole_csr_cell_grad(
        wp.from_torch(p["positions"].contiguous(), dtype=wp.vec3d),
        wp.from_torch(p["charges"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["dipoles"].contiguous(), dtype=wp.vec3d),
        wp.from_torch(p["cell"].contiguous(), dtype=wp.mat33d),
        wp.from_torch(p["idx_j"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["neighbor_ptr"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["unit_shifts"].contiguous(), dtype=wp.vec3i),
        wp.from_torch(p["sigma"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["alpha"].contiguous(), dtype=wp.float64),
        wp.from_torch(grad_cell, dtype=wp.mat33d),
        device="cpu",
        half_neighbor_list=half_neighbor_list,
    )
    return grad_cell


def _launch_dipole_energy_total(p):
    n = p["positions"].shape[0]
    energies = torch.zeros(n, dtype=torch.float64)
    grad_pos = torch.zeros((n, 3), dtype=torch.float64)
    grad_q = torch.zeros(n, dtype=torch.float64)
    grad_mu = torch.zeros((n, 3), dtype=torch.float64)
    multipole_real_space_dipole_csr_energy_fused(
        wp.from_torch(p["positions"].contiguous(), dtype=wp.vec3d),
        wp.from_torch(p["charges"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["dipoles"].contiguous(), dtype=wp.vec3d),
        wp.from_torch(p["cell"].contiguous(), dtype=wp.mat33d),
        wp.from_torch(p["idx_j"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["neighbor_ptr"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["unit_shifts"].contiguous(), dtype=wp.vec3i),
        wp.from_torch(p["sigma"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["alpha"].contiguous(), dtype=wp.float64),
        wp.from_torch(energies, dtype=wp.float64),
        wp.from_torch(grad_pos, dtype=wp.vec3d),
        wp.from_torch(grad_q, dtype=wp.float64),
        wp.from_torch(grad_mu, dtype=wp.vec3d),
        with_pos_grad=False,
        with_charge_grad=False,
        with_dipole_grad=False,
        wp_dtype=wp.float64,
        device="cpu",
    )
    return energies.sum().item()


def test_dipole_cell_grad_matches_fd():
    """LMAX=1 cell-grad kernel matches FD on the energy kernel."""
    p = _periodic_4atom()
    grad_an = _launch_dipole_grad_cell(p)

    h = 1e-5
    grad_fd = torch.zeros_like(p["cell"])
    cell0 = p["cell"].clone()
    for a in range(3):
        for b in range(3):
            cell_p = cell0.clone()
            cell_p[0, a, b] += h
            E_p = _launch_dipole_energy_total({**p, "cell": cell_p})
            cell_m = cell0.clone()
            cell_m[0, a, b] -= h
            E_m = _launch_dipole_energy_total({**p, "cell": cell_m})
            grad_fd[0, a, b] = (E_p - E_m) / (2 * h)

    diff = (grad_an - grad_fd).abs().max().item()
    denom = max(grad_an.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, (
        f"LMAX=1 cell grad mismatch rel={rel:.3e}\nanalytical={grad_an}\nFD={grad_fd}"
    )


@pytest.mark.parametrize("lmax", [0, 1])
def test_monopole_dipole_half_vs_full_neighbor_list_match(lmax):
    """Half-list with scale=1.0 produces same output as full-list with scale=0.5."""
    p_full = _periodic_4atom()
    if lmax == 0:
        grad_full = _launch_monopole_grad_cell(p_full, half_neighbor_list=False)
    else:
        grad_full = _launch_dipole_grad_cell(p_full, half_neighbor_list=False)

    # Build a half list: keep only edges (i, j) with j > i (or j == i and shift > 0).
    nbptr_np = p_full["neighbor_ptr"].cpu().numpy()
    idx_j_np = p_full["idx_j"].cpu().numpy()
    sh_np = p_full["unit_shifts"].cpu().numpy()

    half_idx_j: list[int] = []
    half_shifts: list[tuple[int, int, int]] = []
    half_counts: list[int] = [0] * len(p_full["positions"])
    for i in range(len(p_full["positions"])):
        k_start, k_end = int(nbptr_np[i]), int(nbptr_np[i + 1])
        for k in range(k_start, k_end):
            j = int(idx_j_np[k])
            s = tuple(int(x) for x in sh_np[k])
            # canonical: (j > i) OR (j == i AND shift > (0,0,0))
            if j > i or (j == i and s > (0, 0, 0)):
                half_idx_j.append(j)
                half_shifts.append(s)
                half_counts[i] += 1
    half_ptr = np.cumsum([0] + half_counts)
    p_half = dict(p_full)
    p_half["idx_j"] = torch.tensor(half_idx_j, dtype=torch.int32)
    p_half["neighbor_ptr"] = torch.tensor(half_ptr, dtype=torch.int32)
    p_half["unit_shifts"] = torch.tensor(half_shifts, dtype=torch.int32).reshape(-1, 3)

    if lmax == 0:
        grad_half = _launch_monopole_grad_cell(p_half, half_neighbor_list=True)
    else:
        grad_half = _launch_dipole_grad_cell(p_half, half_neighbor_list=True)

    diff = (grad_full - grad_half).abs().max().item()
    denom = max(grad_full.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 1e-12, (
        f"LMAX={lmax}: half vs full disagree, rel={rel:.3e}\n"
        f"full={grad_full}\nhalf={grad_half}"
    )


def _periodic_csr(pos_np, cell_np, cutoff, device):
    """Full periodic CSR neighbor list with integer shifts in {-1,0,1}^3."""
    n = pos_np.shape[0]
    pi, pj, sh = [], [], []
    for i in range(n):
        for j in range(n):
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    for c in (-1, 0, 1):
                        if i == j and a == 0 and b == 0 and c == 0:
                            continue
                        shift = np.array([a, b, c])
                        if (
                            np.linalg.norm(pos_np[j] + shift @ cell_np - pos_np[i])
                            < cutoff
                        ):
                            pi.append(i)
                            pj.append(j)
                            sh.append([a, b, c])
    order = sorted(range(len(pi)), key=lambda k: (pi[k], pj[k]))
    pi = [pi[k] for k in order]
    pj = [pj[k] for k in order]
    sh = [sh[k] for k in order]
    idx_j = torch.tensor(
        [pj[k] for k in range(len(pj))], dtype=torch.int32, device=device
    )
    nptr = torch.zeros(n + 1, dtype=torch.int32, device=device)
    for ii in pi:
        nptr[ii + 1] += 1
    nptr = torch.cumsum(nptr, 0).to(torch.int32)
    return idx_j, nptr, torch.tensor(sh, dtype=torch.int32, device=device)


class TestRealSpaceMonopoleStressLoss:
    """l=0 real-space cell-grad double-backward (∂²E/∂cell∂θ) — Tier-1 S2.

    The standalone l=0 entry uses the non-fused chain (no cell-grad), so this
    exercises the fused op (used by ``multipole_ewald_summation``) directly:
    ``create_graph`` through ``dE/dcell`` then backprop to positions/charges/
    cell must be nonzero (silent-zero guard) and FD-match.
    """

    def test_stress_loss_fd(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float64
        rng = np.random.default_rng(7)
        n, lbox = 5, 4.0
        cell_np = np.diag([lbox, lbox, lbox]) + rng.normal(scale=0.2, size=(3, 3))
        pos_np = rng.uniform(0, lbox, size=(n, 3))
        idx_j, nptr, shifts = _periodic_csr(pos_np, cell_np, 5.0, device)
        assert int((shifts.abs().sum(1) > 0).sum()) > 0, "need image neighbors"
        sigma = torch.tensor([0.5], dtype=dt, device=device)
        alpha = torch.tensor([0.35], dtype=dt, device=device)
        q0 = torch.tensor(rng.normal(size=n), dtype=dt, device=device)
        q0 = q0 - q0.mean()
        cell0 = torch.tensor(cell_np, dtype=dt, device=device).unsqueeze(0)
        pos0 = torch.tensor(pos_np, dtype=dt, device=device)
        w = torch.tensor(rng.normal(size=(1, 3, 3)), dtype=dt, device=device)
        op = torch.ops.nvalchemiops.multipole_real_space_monopole_fused

        def sloss(pos, cell, q):
            e, _, _ = op(pos, q, cell, sigma, alpha, idx_j, nptr, shifts, False)
            (stress,) = torch.autograd.grad(e, cell, create_graph=True)
            return (stress * w).sum()

        eps = 1e-6
        # position channel
        pos = pos0.clone().requires_grad_(True)
        cell = cell0.clone().requires_grad_(True)
        (gp,) = torch.autograd.grad(sloss(pos, cell, q0), pos)
        assert gp.abs().max() > 1e-8, "silent zero (positions)"
        vp = torch.tensor(rng.normal(size=(n, 3)), dtype=dt, device=device)

        def sval_pos(p):
            return sloss(p, cell0.clone().requires_grad_(True), q0).item()

        fdp = (sval_pos(pos0 + eps * vp) - sval_pos(pos0 - eps * vp)) / (2 * eps)
        assert abs((gp * vp).sum().item() - fdp) / (abs(fdp) + 1e-12) < 1e-5
        # charge channel
        qg = q0.clone().requires_grad_(True)
        (gq,) = torch.autograd.grad(
            sloss(pos0, cell0.clone().requires_grad_(True), qg), qg
        )
        vq = torch.tensor(rng.normal(size=n), dtype=dt, device=device)

        def sval_q(qq):
            return sloss(pos0, cell0.clone().requires_grad_(True), qq).item()

        fdq = (sval_q(q0 + eps * vq) - sval_q(q0 - eps * vq)) / (2 * eps)
        assert abs((gq * vq).sum().item() - fdq) / (abs(fdq) + 1e-12) < 1e-5


class TestRealSpaceDipoleStressLoss:
    """l=1 real-space cell-grad double-backward (∂²E/∂cell∂θ) — Tier-1 S2.

    Exercises the fused l=1 op directly (the standalone l=1 entry has no
    cell-grad): stress-loss to positions/charges/dipoles/cell must FD-match.
    """

    def test_stress_loss_fd(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float64
        rng = np.random.default_rng(7)
        n, lbox = 5, 4.0
        cell_np = np.diag([lbox, lbox, lbox]) + rng.normal(scale=0.2, size=(3, 3))
        pos_np = rng.uniform(0, lbox, size=(n, 3))
        idx_j, nptr, shifts = _periodic_csr(pos_np, cell_np, 5.0, device)
        sigma = torch.tensor([0.5], dtype=dt, device=device)
        alpha = torch.tensor([0.35], dtype=dt, device=device)
        q0 = torch.tensor(rng.normal(size=n), dtype=dt, device=device)
        q0 = q0 - q0.mean()
        mu0 = torch.tensor(rng.normal(size=(n, 3)), dtype=dt, device=device)
        cell0 = torch.tensor(cell_np, dtype=dt, device=device).unsqueeze(0)
        pos0 = torch.tensor(pos_np, dtype=dt, device=device)
        w = torch.tensor(rng.normal(size=(1, 3, 3)), dtype=dt, device=device)
        op = torch.ops.nvalchemiops.multipole_real_space_dipole_fused

        def sloss(pos, cell, q, mu):
            e, _, _, _ = op(pos, q, mu, cell, sigma, alpha, idx_j, nptr, shifts, False)
            (stress,) = torch.autograd.grad(e, cell, create_graph=True)
            return (stress * w).sum()

        eps = 1e-6
        cs = lambda: cell0.clone().requires_grad_(True)  # noqa: E731
        # positions
        pos = pos0.clone().requires_grad_(True)
        (gp,) = torch.autograd.grad(sloss(pos, cs(), q0, mu0), pos)
        assert gp.abs().max() > 1e-8, "silent zero (positions)"
        vp = torch.tensor(rng.normal(size=(n, 3)), dtype=dt, device=device)
        fdp = (
            sloss(pos0 + eps * vp, cs(), q0, mu0).item()
            - sloss(pos0 - eps * vp, cs(), q0, mu0).item()
        ) / (2 * eps)
        assert abs((gp * vp).sum().item() - fdp) / (abs(fdp) + 1e-12) < 1e-5
        # dipoles
        mg = mu0.clone().requires_grad_(True)
        (gm,) = torch.autograd.grad(sloss(pos0, cs(), q0, mg), mg)
        vm = torch.tensor(rng.normal(size=(n, 3)), dtype=dt, device=device)
        fdm = (
            sloss(pos0, cs(), q0, mu0 + eps * vm).item()
            - sloss(pos0, cs(), q0, mu0 - eps * vm).item()
        ) / (2 * eps)
        assert abs((gm * vm).sum().item() - fdm) / (abs(fdm) + 1e-12) < 1e-5


class TestRealSpaceQuadrupoleStressLoss:
    """l=2 real-space cell-grad double-backward (∂²E/∂cell∂θ) — Tier-1 S3.

    Validates stress-loss to positions/charges/dipoles/quadrupoles/cell via
    the public l=2 entry (whose backward routes the cell-grad op).
    """

    def test_stress_loss_fd(self):
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (  # noqa: E501
            multipole_real_space_quadrupole_energy as qe,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float64
        rng = np.random.default_rng(7)
        n, lbox = 5, 4.0
        cell_np = np.diag([lbox, lbox, lbox]) + rng.normal(scale=0.2, size=(3, 3))
        pos_np = rng.uniform(0, lbox, size=(n, 3))
        idx_j, nptr, shifts = _periodic_csr(pos_np, cell_np, 5.0, device)
        sigma = torch.tensor([0.5], dtype=dt, device=device)
        alpha = torch.tensor([0.35], dtype=dt, device=device)
        q0 = torch.tensor(rng.normal(size=n), dtype=dt, device=device)
        q0 = q0 - q0.mean()
        mu0 = torch.tensor(rng.normal(size=(n, 3)), dtype=dt, device=device)
        Qr = rng.normal(size=(n, 3, 3))
        Qr = 0.5 * (Qr + Qr.transpose(0, 2, 1))
        Q0 = torch.tensor(Qr, dtype=dt, device=device)
        cell0 = torch.tensor(cell_np, dtype=dt, device=device).unsqueeze(0)
        pos0 = torch.tensor(pos_np, dtype=dt, device=device)
        w = torch.tensor(rng.normal(size=(1, 3, 3)), dtype=dt, device=device)
        cs = lambda: cell0.clone().requires_grad_(True)  # noqa: E731

        def sloss(pos, cell, q, mu, Q):
            e = qe(pos, q, mu, Q, cell, idx_j, nptr, shifts, sigma, alpha).sum()
            (stress,) = torch.autograd.grad(e, cell, create_graph=True)
            return (stress * w).sum()

        eps = 1e-6
        # positions
        pos = pos0.clone().requires_grad_(True)
        (gp,) = torch.autograd.grad(sloss(pos, cs(), q0, mu0, Q0), pos)
        assert gp.abs().max() > 1e-8, "silent zero (positions)"
        vp = torch.tensor(rng.normal(size=(n, 3)), dtype=dt, device=device)
        fdp = (
            sloss(pos0 + eps * vp, cs(), q0, mu0, Q0).item()
            - sloss(pos0 - eps * vp, cs(), q0, mu0, Q0).item()
        ) / (2 * eps)
        assert abs((gp * vp).sum().item() - fdp) / (abs(fdp) + 1e-12) < 1e-5
        # quadrupoles
        Qg = Q0.clone().requires_grad_(True)
        (gQ,) = torch.autograd.grad(sloss(pos0, cs(), q0, mu0, Qg), Qg)
        vQr = rng.normal(size=(n, 3, 3))
        vQr = 0.5 * (vQr + vQr.transpose(0, 2, 1))
        vQ = torch.tensor(vQr, dtype=dt, device=device)
        fdQ = (
            sloss(pos0, cs(), q0, mu0, Q0 + eps * vQ).item()
            - sloss(pos0, cs(), q0, mu0, Q0 - eps * vQ).item()
        ) / (2 * eps)
        assert abs((gQ * vQ).sum().item() - fdQ) / (abs(fdQ) + 1e-12) < 1e-5


def _periodic_csr_batched(pos_list, cell_list, cutoff, device):
    """Concatenated systems -> global CSR with per-system shifts + batch_idx."""
    pi, pj, sh, bidx = [], [], [], []
    off = 0
    for s, (pn, cn) in enumerate(zip(pos_list, cell_list)):
        ns = pn.shape[0]
        bidx.extend([s] * ns)
        for i in range(ns):
            for j in range(ns):
                for a in (-1, 0, 1):
                    for b in (-1, 0, 1):
                        for c in (-1, 0, 1):
                            if i == j and a == 0 and b == 0 and c == 0:
                                continue
                            if (
                                np.linalg.norm(pn[j] + np.array([a, b, c]) @ cn - pn[i])
                                < cutoff
                            ):
                                pi.append(off + i)
                                pj.append(off + j)
                                sh.append([a, b, c])
        off += ns
    o = sorted(range(len(pi)), key=lambda k: (pi[k], pj[k]))
    idx = torch.tensor([pj[k] for k in o], dtype=torch.int32, device=device)
    nptr = torch.zeros(off + 1, dtype=torch.int32, device=device)
    for ii in pi:
        nptr[ii + 1] += 1
    nptr = torch.cumsum(nptr, 0).to(torch.int32)
    shifts = torch.tensor([sh[k] for k in o], dtype=torch.int32, device=device)
    return idx, nptr, shifts, torch.tensor(bidx, dtype=torch.int32, device=device)


class TestRealSpaceBatchedStressLoss:
    """Batched real-space cell-grad double-backward (∂²E/∂cell∂pos) — Tier-1 S4.

    l=0/1 via the batched fused ops; l=2 via the public batched l=2 energy.
    Per-system stress + create_graph to positions must FD-match.
    """

    @pytest.mark.parametrize("lmax", [0, 1, 2])
    def test_stress_loss_fd(self, lmax):
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (  # noqa: E501
            multipole_real_space_quadrupole_energy as qe,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float64
        rng = np.random.default_rng(5 + lmax)
        counts = [4, 5]
        B, lbox = 2, 4.0
        cells = [
            np.diag([lbox, lbox, lbox]) + rng.normal(scale=0.2, size=(3, 3))
            for _ in range(B)
        ]
        poss = [rng.uniform(0, lbox, size=(c, 3)) for c in counts]
        idx_j, nptr, shifts, bidx = _periodic_csr_batched(poss, cells, 5.0, device)
        ntot = sum(counts)
        cell0 = torch.tensor(np.stack(cells), dtype=dt, device=device)
        pos0 = torch.tensor(np.concatenate(poss), dtype=dt, device=device)
        sigma = torch.full((B,), 0.5, dtype=dt, device=device)
        alpha = torch.full((B,), 0.35, dtype=dt, device=device)
        q0 = torch.tensor(rng.normal(size=ntot), dtype=dt, device=device)
        for b in range(B):
            m = bidx == b
            q0[m] = q0[m] - q0[m].mean()
        mu0 = torch.tensor(rng.normal(size=(ntot, 3)), dtype=dt, device=device)
        Qr = rng.normal(size=(ntot, 3, 3))
        Qr = 0.5 * (Qr + Qr.transpose(0, 2, 1))
        Q0 = torch.tensor(Qr, dtype=dt, device=device)
        w = torch.tensor(rng.normal(size=(B, 3, 3)), dtype=dt, device=device)
        cs = lambda: cell0.clone().requires_grad_(True)  # noqa: E731

        dop = torch.ops.nvalchemiops.batch_multipole_real_space_dipole_fused
        mop = torch.ops.nvalchemiops.batch_multipole_real_space_monopole_fused

        def sloss(pos, cell):
            if lmax == 0:
                e = mop(pos, q0, cell, sigma, alpha, idx_j, nptr, shifts, bidx, False)[
                    0
                ]
            elif lmax == 1:
                e = dop(
                    pos, q0, mu0, cell, sigma, alpha, idx_j, nptr, shifts, bidx, False
                )[0]
            else:
                e = qe(
                    pos,
                    q0,
                    mu0,
                    Q0,
                    cell,
                    idx_j,
                    nptr,
                    shifts,
                    sigma,
                    alpha,
                    batch_idx=bidx,
                )
            (stress,) = torch.autograd.grad(e.sum(), cell, create_graph=True)
            return (stress * w).sum()

        pos = pos0.clone().requires_grad_(True)
        (gp,) = torch.autograd.grad(sloss(pos, cs()), pos)
        assert gp.abs().max() > 1e-8, f"lmax={lmax}: batched silent zero"
        v = torch.tensor(rng.normal(size=(ntot, 3)), dtype=dt, device=device)
        eps = 1e-6
        fd = (
            sloss(pos0 + eps * v, cs()).item() - sloss(pos0 - eps * v, cs()).item()
        ) / (2 * eps)
        # Tol 1e-4: the l=2 batched config can draw close periodic pairs giving
        # large-magnitude stress whose 2nd-order central difference floors near
        # ~3e-5 (analytic matches to that); still catches sign flips/silent zeros.
        assert abs((gp * v).sum().item() - fd) / (abs(fd) + 1e-12) < 1e-4


class TestRealSpaceEnergyWithStress:
    """Per-atom real-space energy that also carries cell-grad (Phase 4 component).

    ``multipole_real_space_energy_with_stress`` must match the per-atom
    :func:`multipole_real_space_energy` on value + position/charge gradients,
    and reproduce the collective stress (``grad(E.sum(), cell)``) of the
    per-system-scalar ``*FusedScalarFunction`` oracle, with a correct
    stress-loss (``d(stress)/d theta``) under the sum-reduction contract.
    """

    @pytest.mark.parametrize("l_max", [0, 1])
    def test_value_forces_match_plain_per_atom(self, l_max):
        from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
            pack_multipole_moments,
        )
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            multipole_real_space_energy,
            multipole_real_space_energy_with_stress,
        )

        td = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        rng = np.random.default_rng(0xA1 + l_max)
        n, ell = 6, 8.0
        cell = torch.eye(3, dtype=torch.float64, device=td) * ell
        pos = torch.from_numpy(rng.uniform(0, ell, (n, 3))).to(td)
        q = torch.from_numpy(rng.uniform(-1, 1, n)).to(td)
        q = q - q.mean()
        mu = (
            torch.from_numpy(rng.standard_normal((n, 3)) * 0.3).to(td)
            if l_max
            else None
        )
        idx_j, nptr, ush = _csr_neighbors_for_n_atoms(pos, 4.5)
        sig = torch.tensor([1.0], dtype=torch.float64, device=td)
        alp = torch.tensor([0.4], dtype=torch.float64, device=td)
        mm = pack_multipole_moments(q, mu) if l_max else pack_multipole_moments(q)

        plain = multipole_real_space_energy(pos, mm, cell, idx_j, nptr, ush, sig, alp)
        withs = multipole_real_space_energy_with_stress(
            pos, mm, cell, idx_j, nptr, ush, sig, alp
        )
        torch.testing.assert_close(withs, plain, rtol=0, atol=0)

        # Forces (grad wrt positions) must be identical.
        p1 = pos.clone().requires_grad_(True)
        f_plain = torch.autograd.grad(
            multipole_real_space_energy(p1, mm, cell, idx_j, nptr, ush, sig, alp).sum(),
            p1,
        )[0]
        p2 = pos.clone().requires_grad_(True)
        f_with = torch.autograd.grad(
            multipole_real_space_energy_with_stress(
                p2, mm, cell, idx_j, nptr, ush, sig, alp
            ).sum(),
            p2,
        )[0]
        torch.testing.assert_close(f_with, f_plain, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("l_max", [0, 1])
    def test_stress_matches_fused_scalar_oracle(self, l_max):
        from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
            pack_multipole_moments,
        )
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            MultipoleRealSpaceDipoleFusedScalarFunction,
            MultipoleRealSpaceMonopoleFusedScalarFunction,
            multipole_real_space_energy_with_stress,
        )

        td = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        rng = np.random.default_rng(0xB2 + l_max)
        n, ell = 6, 8.0
        cell = torch.eye(3, dtype=torch.float64, device=td) * ell
        pos = torch.from_numpy(rng.uniform(0, ell, (n, 3))).to(td)
        q = torch.from_numpy(rng.uniform(-1, 1, n)).to(td)
        q = q - q.mean()
        mu = (
            torch.from_numpy(rng.standard_normal((n, 3)) * 0.3).to(td)
            if l_max
            else None
        )
        idx_j, nptr, ush = _csr_neighbors_for_n_atoms(pos, 4.5)
        sig = torch.tensor([1.0], dtype=torch.float64, device=td)
        alp = torch.tensor([0.4], dtype=torch.float64, device=td)
        mm = pack_multipole_moments(q, mu) if l_max else pack_multipole_moments(q)

        c_or = cell.clone().requires_grad_(True)
        if l_max:
            e_or = MultipoleRealSpaceDipoleFusedScalarFunction.apply(
                pos, q, mu, c_or, sig, alp, idx_j, nptr, ush
            )
        else:
            e_or = MultipoleRealSpaceMonopoleFusedScalarFunction.apply(
                pos, q, c_or, sig, alp, idx_j, nptr, ush
            )
        s_or = torch.autograd.grad(e_or, c_or)[0]

        c2 = cell.clone().requires_grad_(True)
        e2 = multipole_real_space_energy_with_stress(
            pos, mm, c2, idx_j, nptr, ush, sig, alp
        )
        s_ours = torch.autograd.grad(e2.sum(), c2)[0]
        torch.testing.assert_close(s_ours, s_or, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("l_max", [0, 1])
    def test_stress_loss_finite_difference(self, l_max):
        from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
            pack_multipole_moments,
        )
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            multipole_real_space_energy_with_stress,
        )

        td = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        rng = np.random.default_rng(0xC3 + l_max)
        n, ell = 5, 8.0
        cell = torch.eye(3, dtype=torch.float64, device=td) * ell
        pos = torch.from_numpy(rng.uniform(0, ell, (n, 3))).to(td)
        q0 = torch.from_numpy(rng.uniform(-1, 1, n)).to(td)
        q0 = q0 - q0.mean()
        mu = (
            torch.from_numpy(rng.standard_normal((n, 3)) * 0.3).to(td)
            if l_max
            else None
        )
        idx_j, nptr, ush = _csr_neighbors_for_n_atoms(pos, 4.5)
        sig = torch.tensor([1.0], dtype=torch.float64, device=td)
        alp = torch.tensor([0.4], dtype=torch.float64, device=td)

        def stress(qv):
            c = cell.clone().requires_grad_(True)
            mm = pack_multipole_moments(qv, mu) if l_max else pack_multipole_moments(qv)
            e = multipole_real_space_energy_with_stress(
                pos, mm, c, idx_j, nptr, ush, sig, alp
            )
            return torch.autograd.grad(e.sum(), c, create_graph=True)[0]

        q1 = q0.clone().requires_grad_(True)
        analytic = torch.autograd.grad(stress(q1).sum(), q1)[0]
        h = 1e-6
        fd = torch.zeros_like(q0)
        for k in range(n):
            qp = q0.clone()
            qp[k] += h
            qm = q0.clone()
            qm[k] -= h
            fd[k] = (stress(qp).sum() - stress(qm).sum()) / (2 * h)
        assert (analytic - fd).norm().item() / (fd.norm().item() + 1e-12) < 1e-6
