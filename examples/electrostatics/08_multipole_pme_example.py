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
Multipole Particle Mesh Ewald (charges + dipoles + quadrupoles)
================================================================

This example demonstrates the GTO-PME multipole electrostatics path for
periodic systems with charges (l_max=0), dipoles (l_max=1), and
quadrupoles (l_max=2). PME accelerates the reciprocal-space piece via
B-spline interpolation + FFT, dropping the asymptotic cost from
``O(N · N_k)`` (Ewald direct k-space) to ``O(M log M + N · p³)``.

Topics covered:

- l_max=0/1/2 via ``multipole_particle_mesh_ewald`` (single composite call).
- Stress tensor via the cell-gradient (PME has it wired end-to-end —
  real + reciprocal — for all l_max).
- Force-loss-style training (l_max=2 ``create_graph=True``) through the
  full PME composite.
- Stress-loss-style training (``create_graph=True`` through ``dE/dcell``,
  i.e. ∂²E/∂cell∂θ) — single-system AND batched.
- Auto-estimated parameters via ``estimate_multipole_pme_parameters``.
- Batched (multi-system) workflows, including batched l_max=2.

.. important::
    Scope of the public PME API:

    * ``multipole_particle_mesh_ewald`` (composite) — supports **l_max =
      0, 1, and 2** as a single call, **single-system and batched**. All
      per-atom moments go in **one** e3nn-packed ``multipole_moments``
      tensor: ``(N, 1)`` charge, ``(N, 4)`` charge+dipole, ``(N, 9)``
      charge+dipole+quadrupole (l=2 traceless). Build it with
      :func:`~nvalchemiops.torch.interactions.electrostatics.pack_multipole_moments`
      from Cartesian channels. Energy, forces, per-moment gradients, and
      the stress tensor all flow through autograd, single-system and batched.
    * The unified real-space-only entry point is
      ``multipole_real_space_energy``, which takes the same packed
      ``multipole_moments``, for when you need just the short-range half.
    * **Force-loss training:** ``create_graph=True`` through the *full* PME
      composite is wired at l_max=0/1/2, **single-system and batched**.

.. note::
    This script is intended as an API demonstration. Do not use it for
    performance benchmarking; refer to the ``benchmarks`` folder.
"""

# %%
# Setup and Imports
# -----------------

from __future__ import annotations

import numpy as np
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    pack_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics.parameters import (
    estimate_multipole_pme_parameters,
)
from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
    multipole_particle_mesh_ewald,
)
from nvalchemiops.torch.neighbors import neighbor_list

# %%
# Configure Device
# ----------------

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# %%
# Build a NaCl Periodic System
# ----------------------------
# We use a small NaCl-like supercell (16 atoms, charge ±1 by default) as
# the base fixture. Pass ``total_charge ≠ 0`` to bias the system — PME
# handles a uniform background-charge correction internally so a
# non-neutral cell is well-defined.


def make_periodic_system(
    n_cells: int = 2,
    L: float = 6.0,
    total_charge: float = 0.0,
    dtype=torch.float64,
):
    """Return ``(positions, charges, cell, pbc)`` for a NaCl-like rock-salt
    supercell.

    Parameters
    ----------
    n_cells : int
        Number of replications along each axis. ``n_cells**3 * 2`` atoms total.
    L : float
        Side length of the (cubic) supercell in Å.
    total_charge : float, default 0.0
        Net charge of the system. The Na/Cl ± unit charges are perturbed by
        ``total_charge / N`` per atom to reach the requested net charge.
        PME supports a non-neutral cell via the implicit uniform-background
        correction baked into ``multipole_particle_mesh_ewald``.
    """
    base = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    bcharges = np.array([1.0, -1.0])
    positions, charges = [], []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                off = np.array([i, j, k])
                for p, c in zip(base, bcharges):
                    positions.append((p + off) * (L / n_cells))
                    charges.append(c)
    positions = torch.tensor(np.array(positions), dtype=dtype, device=device)
    charges = torch.tensor(np.array(charges), dtype=dtype, device=device)
    if total_charge != 0.0:
        N = charges.shape[0]
        charges = charges + (total_charge - charges.sum()) / N
    cell = (torch.eye(3, dtype=dtype, device=device) * L).unsqueeze(0)  # (1, 3, 3)
    pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
    return positions, charges, cell, pbc


positions, charges, cell, pbc = make_periodic_system()
print(f"System: {len(positions)} atoms, cell side = {cell[0, 0, 0]:.2f} Å")
print(
    f"Total charge: {charges.sum().item():+.4f} (0.0 = neutral; tweak via `total_charge=`)"
)

# %%
# Build the Neighbor List
# -----------------------

cutoff = 12.0
nl_2d, neighbor_ptr, unit_shifts = neighbor_list(
    positions,
    cutoff=cutoff,
    cell=cell,
    pbc=pbc,
    return_neighbor_list=True,
)
idx_j = nl_2d[1].contiguous()
print(f"neighbor list: {idx_j.shape[0]} pairs, cutoff = {cutoff} Å")

# %%
# l_max=0: Charges Only — Energy + Forces via the Composite
# ---------------------------------------------------------
# ``multipole_particle_mesh_ewald`` mirrors ``multipole_ewald_summation``:
# one packed ``multipole_moments`` tensor (e3nn order), automatic
# parameter estimation, and batched dispatch via ``batch_idx``.

sigma = 0.5

mm_l0 = pack_multipole_moments(charges)  # (N, 1)
pos_g = positions.clone().requires_grad_(True)
E_l0 = multipole_particle_mesh_ewald(
    pos_g,
    mm_l0,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=None,  # auto-estimate
    mesh_dimensions=None,  # auto-estimate
    accuracy=1e-6,
)
# ``multipole_particle_mesh_ewald`` returns PER-ATOM energies, shape ``(N,)``.
# Call ``.sum()`` for the system total; use ``scatter_add`` by ``batch_idx``
# for per-system totals in batched mode (see the batched section below).
forces_l0 = -torch.autograd.grad(E_l0.sum(), pos_g)[0]
print(
    f"l_max=0 PME E (total) = {E_l0.sum().item():.6f},"
    f" per-atom shape {tuple(E_l0.shape)}"
)
print(f"l_max=0 PME max |F| = {torch.norm(forces_l0, dim=1).max().item():.4f}")

# %%
# l_max=1: Charges + Dipoles
# --------------------------
# At l_max=1 each atom carries both a scalar charge ``q_i`` and a
# **dipole moment** ``μ_i`` — a Cartesian vec3 with units (charge × length)
# describing the per-atom dipole displacement. Physically, ``μ_i`` is the
# first multipole expansion coefficient of the atomic charge density:
#
# .. math::
#     ρ_i(r) = q_i δ(r - r_i) - μ_i · ∇ δ(r - r_i) + …
#
# The energy gradient w.r.t. the dipole is the negative on-site electric
# field at that atom (i.e. ``∂E/∂μ_i = -E_field(r_i)`` in the conventional
# sign — what an induced-dipole model would set to zero at self-consistency).
#
# API convention: ``multipole_particle_mesh_ewald`` accepts the per-atom
# moments as a single packed ``multipole_moments`` tensor in **e3nn
# spherical layout**:
#
# * ``(N, 1)`` → charge only
# * ``(N, 4)`` → ``[q, μ_y, μ_z, μ_x]`` (charge + dipole in e3nn order)
# * ``(N, 9)`` → the l=1 block plus the five traceless l=2 channels
#
# Build it from **physical Cartesian** channels with
# ``pack_multipole_moments(charges, dipoles[, quadrupoles])`` — it handles
# the Cartesian→e3nn permutation (and the l=2 detrace) for you.

rng = np.random.default_rng(2026)
dipoles_cart = torch.tensor(
    0.1 * rng.normal(size=(len(positions), 3)),
    dtype=positions.dtype,
    device=device,
)
mm_l1 = pack_multipole_moments(charges, dipoles_cart)  # (N, 4)

# Autograd-enabled copies. ``mu_cart_g`` is a Cartesian (x, y, z) dipole
# leaf; ``pack_multipole_moments`` is differentiable, so ``grad_mu`` below
# is ``∂E/∂μ`` in Cartesian order (= -E_field of that atom).
pos_g = positions.clone().requires_grad_(True)
mu_cart_g = dipoles_cart.clone().requires_grad_(True)
mm_l1_grad = pack_multipole_moments(charges, mu_cart_g)

E_l1 = multipole_particle_mesh_ewald(
    pos_g,
    mm_l1_grad,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=0.35,
)
grad_pos, grad_mu = torch.autograd.grad(E_l1.sum(), [pos_g, mu_cart_g])
print(f"l_max=1 PME E (total) = {E_l1.sum().item():.6f}")
print(
    f"l_max=1 PME max |F|             = {torch.norm(-grad_pos, dim=1).max().item():.4f}"
)
print(f"l_max=1 PME max |∂E/∂μ| (= |E_field|) = {grad_mu.abs().max().item():.4f}")


# %%
# l_max=2: Single Composite Call
# ------------------------------
# .. note::
#     The full l_max=2 PME total (energy, forces, per-moment grads, **and the
#     stress tensor**) is a single composite call (real + reciprocal + self),
#     exactly like ``multipole_ewald_summation``. No manual
#     ``E_real + E_recip − E_self`` assembly needed.
#
#     **Batched l_max=2 PME** routes through the same composite with
#     ``batch_idx``; see the batched-workflow section below.
#
# Pack charges + dipoles + the **Cartesian symmetric** quadrupole into one
# ``(N, 9)`` ``multipole_moments`` tensor via ``pack_multipole_moments`` —
# it detraces Q (the l=2 channel is traceless) and packs to e3nn layout.

quadrupoles_cart = torch.tensor(
    0.05 * rng.normal(size=(len(positions), 3, 3)),
    dtype=positions.dtype,
    device=device,
)
# Symmetrize (kernel assumes symmetric Q), then detrace (l=2 is traceless,
# so ``pack_multipole_moments`` round-trips exactly without warning).
quadrupoles_cart = 0.5 * (quadrupoles_cart + quadrupoles_cart.transpose(-1, -2))
_trace = quadrupoles_cart.diagonal(dim1=-2, dim2=-1).sum(-1)
quadrupoles_cart = quadrupoles_cart - (_trace / 3.0)[:, None, None] * torch.eye(
    3, dtype=positions.dtype, device=device
)
mm_l2 = pack_multipole_moments(charges, dipoles_cart, quadrupoles_cart)  # (N, 9)

alpha = 0.35
mesh = (16, 16, 16)

pos_g = positions.clone().requires_grad_(True)
q_g = charges.clone().requires_grad_(True)
mu_g_cart = dipoles_cart.clone().requires_grad_(True)
Q_g = quadrupoles_cart.clone().requires_grad_(True)

# Single composite call. ``pack_multipole_moments`` is differentiable, so
# gradients flow back to the Cartesian charge / dipole / quadrupole leaves.
mm_l2_grad = pack_multipole_moments(q_g, mu_g_cart, Q_g)
E_l2 = multipole_particle_mesh_ewald(
    pos_g,
    mm_l2_grad,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=alpha,
    mesh_dimensions=mesh,
)
grads = torch.autograd.grad(E_l2.sum(), [pos_g, q_g, mu_g_cart, Q_g])
print(f"l_max=2 PME total E = {E_l2.sum().item():.6f}")
print(f"l_max=2 PME max |F|     = {torch.norm(-grads[0], dim=1).max().item():.4f}")
print(f"l_max=2 PME max |∂E/∂q| = {grads[1].abs().max().item():.4f}")
print(f"l_max=2 PME max |∂E/∂μ| = {grads[2].abs().max().item():.4f}")
print(f"l_max=2 PME max |∂E/∂Q| = {grads[3].abs().max().item():.4f}")


# %%
# Stress Tensor via the Cell Gradient
# -----------------------------------
# For periodic systems the stress tensor is
#
# .. math::
#     \\sigma_{ab} = \\frac{1}{V} \\sum_b c_{ab} \\frac{\\partial E}{\\partial c_{ab}}
#
# .. tip::
#     **PME has cell-grad wired end-to-end** — both the real-space pair sum
#     and the reciprocal-space spread/convolve/gather chain propagate
#     ``cell.requires_grad`` through autograd, giving a complete
#     Ewald-equivalent stress.
#
# Both l_max=1 and l_max=2 stress come straight from the composite — pass
# ``cell.requires_grad=True`` and read ``∂E/∂cell``.

# l_max=1 stress via composite PME — real + reciprocal cell-gradient both wired.
cell_g = cell.clone().requires_grad_(True)
E_for_stress = multipole_particle_mesh_ewald(
    positions,
    mm_l1,
    cell_g,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=alpha,
)
(grad_cell_l1,) = torch.autograd.grad(E_for_stress.sum(), [cell_g])
volume = float(torch.det(cell[0]))
stress_l1 = (cell_g[0].T @ grad_cell_l1[0]) / volume
print("l_max=1 PME stress (real + reciprocal cell-grad both wired):")
print(stress_l1)

# l_max=2 stress via the composite directly — real + reciprocal
# spread/convolve cell-gradient; the self-energy has no cell dependence.
cell_g = cell.clone().requires_grad_(True)
E_l2_for_stress = multipole_particle_mesh_ewald(
    positions,
    mm_l2,
    cell_g,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=alpha,
    mesh_dimensions=mesh,
)
(grad_cell_l2,) = torch.autograd.grad(E_l2_for_stress.sum(), [cell_g])
stress_l2 = (cell[0].T @ grad_cell_l2[0]) / volume
print("\nl_max=2 PME stress (composite; real + reciprocal cell-grad both wired):")
print(stress_l2)


# %%
# Force-Loss-Style Training (l_max=2 ``create_graph=True``)
# ---------------------------------------------------------
# ``create_graph=True`` through the **full single-system PME composite** at
# l_max=2 is wired end-to-end: the PME-spread double-backward (∇³/∇⁴ octupole
# kernels) plus the mesh-mesh convolve double-backward make the l_max=2
# ``multipole_particle_mesh_ewald`` composite twice-differentiable, so
# backprop through the forces (the force-loss Hessian-vector product) is
# correct.
#
# The batched path is wired too — see the batched force-loss section below.

pos_g = positions.clone().requires_grad_(True)
E_fl = multipole_particle_mesh_ewald(
    pos_g,
    mm_l2,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=alpha,
    mesh_dimensions=mesh,
)
# E_fl is per-atom (N,); sum to a scalar before differentiating.
forces = -torch.autograd.grad(E_fl.sum(), pos_g, create_graph=True)[0]
force_labels = torch.randn_like(forces)
force_loss = ((forces - force_labels) ** 2).mean()
(grad_pos_2nd,) = torch.autograd.grad(force_loss, [pos_g])
print(f"l_max=2 PME force-loss        = {force_loss.item():.4f}")
print(f"l_max=2 ∂loss/∂positions max  = {grad_pos_2nd.abs().max().item():.4f}")


# %%
# Stress-Loss Training (l_max=2 ``create_graph=True`` through ``dE/dcell``)
# ------------------------------------------------------------------------
# The stress is ``∂E/∂cell``, so backprop a stress-error loss needs the mixed
# second derivative ``∂²E/∂cell∂θ``. PME has the **cell** second-order wired
# end-to-end at l_max=0/1/2 (single-system and batched): the reciprocal cell
# coupling routes through ``fractionalize`` (cell autograd) feeding the spread
# double-backward, plus the convolve k²/volume double-backward. Take the stress
# with ``create_graph=True`` and backprop the loss to positions.

pos_g = positions.clone().requires_grad_(True)
cell_g = cell.clone().requires_grad_(True)
E_sl = multipole_particle_mesh_ewald(
    pos_g,
    mm_l2,
    cell_g,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=alpha,
    mesh_dimensions=mesh,
)
# Virial stress with create_graph=True (real + reciprocal cell second-order).
(grad_cell_sl,) = torch.autograd.grad(E_sl.sum(), cell_g, create_graph=True)
stress_sl = (cell[0].T @ grad_cell_sl[0]) / volume
stress_label = torch.randn_like(stress_sl)
stress_label = 0.5 * (stress_label + stress_label.T)
stress_loss = ((stress_sl - stress_label) ** 2).mean()
(grad_pos_stress,) = torch.autograd.grad(stress_loss, [pos_g])
print(f"l_max=2 PME stress-loss       = {stress_loss.item():.4f}")
print(
    f"l_max=2 ∂(stress-loss)/∂positions max = {grad_pos_stress.abs().max().item():.4f}"
)


# %%
# Auto-Estimated PME Parameters
# -----------------------------
# ``multipole_particle_mesh_ewald`` accepts ``alpha=None`` and
# ``mesh_dimensions=None``; the Kolafa-Perram-based estimator picks both
# from ``sigma`` + the system geometry at the requested accuracy. You can
# also call the estimator yourself first to log / cache the chosen
# values — that's what we do below.

est = estimate_multipole_pme_parameters(
    positions,
    cell,
    sigma,
    accuracy=1e-6,
)
# ``est`` is a MultipolePMEParameters dataclass with shape-(B,) tensors
# (B=1 in single-system mode) plus a tuple ``mesh_dimensions``.
alpha_auto = float(est.alpha.item())
rcut_auto = float(est.real_space_cutoff.item())
mesh_auto = est.mesh_dimensions
mesh_spacing_auto = est.mesh_spacing[0].tolist()
print("Kolafa-Perram PME balance at accuracy=1e-6:")
print(f"  α                 = {alpha_auto:.4f}  (1 / Å)")
print(f"  real-space cutoff = {rcut_auto:.4f}  Å")
print(f"  mesh dims (nx,ny,nz) = {mesh_auto}")
print(
    f"  mesh spacing (Å)  = [{mesh_spacing_auto[0]:.3f}, {mesh_spacing_auto[1]:.3f}, {mesh_spacing_auto[2]:.3f}]"
)

E_auto = multipole_particle_mesh_ewald(
    positions,
    mm_l1,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=alpha_auto,
    mesh_dimensions=mesh_auto,
)
print(f"auto-parameters PME (l_max=1): E = {E_auto.sum().item():.6f}")
# Equivalent shortcut — pass None and let the composite estimate internally:
E_auto_via_none = multipole_particle_mesh_ewald(
    positions,
    mm_l1,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=None,
    mesh_dimensions=None,
    accuracy=1e-6,
)
print(
    f"  (matches alpha=None / mesh_dimensions=None shortcut:"
    f" {E_auto_via_none.sum().item():.6f})"
)


# %%
# Batched Workflow (Multi-System PME)
# -----------------------------------
# ``batch_idx`` triggers the batched path; ``cell`` becomes ``(B, 3, 3)``
# and per-atom tensors are flat ``(N_total, ...)``.

p1, c1, cellb1, pbc1 = make_periodic_system()
p2, c2, cellb2, _ = make_periodic_system()
n_per_system = len(p1)

positions_batch = torch.cat([p1, p2], dim=0)
charges_batch = torch.cat([c1, c2], dim=0)
mm_l0_batch = pack_multipole_moments(charges_batch)  # (N_total, 1)
cells_batch = torch.cat([cellb1, cellb2], dim=0)
pbc_batch = torch.cat([pbc1, pbc1], dim=0)  # (2, 3)
batch_idx = torch.cat(
    [
        torch.zeros(n_per_system, dtype=torch.int32, device=device),
        torch.ones(n_per_system, dtype=torch.int32, device=device),
    ]
)

# ``neighbor_list`` is batch-aware: pass ``batch_idx`` (+ a (B, 3, 3) ``cell``
# and (B, 3) ``pbc``) and it returns a single flat CSR with global ``idx_j``
# and a system-spanning ``neighbor_ptr``.
nl_2d_batch, ptr_batch, sh_batch = neighbor_list(
    positions_batch,
    cutoff=cutoff,
    cell=cells_batch,
    pbc=pbc_batch,
    batch_idx=batch_idx,
    return_neighbor_list=True,
)
idx_j_batch = nl_2d_batch[1].contiguous()

E_batch_pme = multipole_particle_mesh_ewald(
    positions_batch,
    mm_l0_batch,
    cells_batch,
    idx_j_batch,
    ptr_batch,
    sh_batch,
    sigma=sigma,
    alpha=alpha,
    batch_idx=batch_idx,
)
# E_batch_pme is per-atom (N_total,). Reduce to per-system (B,) with scatter_add.
B = int(cells_batch.shape[0])
E_batch_pme_sys = torch.zeros(
    B, dtype=E_batch_pme.dtype, device=E_batch_pme.device
).scatter_add(0, batch_idx.long(), E_batch_pme)
print(f"Batched (B=2) l_max=0 PME per-system: {E_batch_pme_sys.tolist()}")

# Batched l_max=2 PME composite — same ``batch_idx`` path as the
# single-system call: one packed ``multipole_moments`` (N_total, 9) tensor.
dipoles_batch = torch.cat([dipoles_cart, dipoles_cart], dim=0)
quadrupoles_batch = torch.cat([quadrupoles_cart, quadrupoles_cart], dim=0)
mm_l1_batch = pack_multipole_moments(charges_batch, dipoles_batch)  # (N_total, 4)
mm_l2_batch = pack_multipole_moments(
    charges_batch, dipoles_batch, quadrupoles_batch
)  # (N_total, 9)

pos_b = positions_batch.clone().requires_grad_(True)
E_batch_l2 = multipole_particle_mesh_ewald(
    pos_b,
    mm_l2_batch,
    cells_batch,
    idx_j_batch,
    ptr_batch,
    sh_batch,
    sigma=sigma,
    alpha=alpha,
    batch_idx=batch_idx,
)
E_batch_l2_sys = torch.zeros(
    B, dtype=E_batch_l2.dtype, device=E_batch_l2.device
).scatter_add(0, batch_idx.long(), E_batch_l2)
print(f"Batched (B=2) l_max=2 PME per-system: {E_batch_l2_sys.tolist()}")
# Per-system forces flow from the batched composite (sum over all atoms).
(forces_batch_l2,) = torch.autograd.grad(E_batch_l2.sum(), [pos_b])
print(
    "Batched (B=2) l_max=2 PME max |F| per system: "
    f"{[float(torch.norm(forces_batch_l2[batch_idx == b], dim=1).max()) for b in (0, 1)]}"
)


# %%
# Batched Force-Loss Training (``create_graph=True``, l_max=1 and l_max=2)
# -----------------------------------------------------------------------
# Multi-system force-loss training is the most common end workflow, and it
# works through the **batched PME composite** at both l_max=1 and l_max=2:
# the batched spread double-backward (effective-moment reuse for l≤1 +
# batched ∇³/∇⁴ octupole kernels for l=2) and the batched convolve
# double-backward make the batched ``multipole_particle_mesh_ewald``
# composite (``batch_idx=...``) twice-differentiable. Take
# ``∂E/∂positions`` with ``create_graph=True``, form a per-system force-error
# loss, and backprop it — exactly as in the single-system section, with
# ``batch_idx`` set.

force_labels_batch = torch.randn(
    positions_batch.shape[0], 3, dtype=positions.dtype, device=device
)

for lmax_tag, mm_b in (("l_max=1", mm_l1_batch), ("l_max=2", mm_l2_batch)):
    pos_g_batch = positions_batch.clone().requires_grad_(True)
    E_fl = multipole_particle_mesh_ewald(
        pos_g_batch,
        mm_b,
        cells_batch,
        idx_j_batch,
        ptr_batch,
        sh_batch,
        sigma=sigma,
        alpha=alpha,
        batch_idx=batch_idx,
    )
    # ``E_fl`` is per-atom ``(N_total,)``; ``.sum()`` gives the per-atom forces.
    forces_batch = -torch.autograd.grad(E_fl.sum(), pos_g_batch, create_graph=True)[0]
    force_loss_batch = ((forces_batch - force_labels_batch) ** 2).mean()
    (grad_pos_2nd_batch,) = torch.autograd.grad(force_loss_batch, [pos_g_batch])
    print(
        f"Batched (B=2) {lmax_tag} PME force-loss = "
        f"{force_loss_batch.item():.4f}  "
        f"∂loss/∂positions max = {grad_pos_2nd_batch.abs().max().item():.4f}"
    )


# %%
# Batched Stress-Loss Training (``create_graph=True`` through ``dE/dcell``)
# ------------------------------------------------------------------------
# Stress-loss is batched too: a ``(B, 3, 3)`` ``cell`` carries grad and the
# batched PME composite second-order through the cell (fractionalize-fed spread
# double-backward + convolve k²/volume double-backward) lets a per-system
# stress-error loss backprop to positions. Matches the per-system single result.

vols_batch = torch.det(cells_batch)  # (B,)
for lmax_tag, mm_b in (("l_max=1", mm_l1_batch), ("l_max=2", mm_l2_batch)):
    pos_g_batch = positions_batch.clone().requires_grad_(True)
    cells_g_batch = cells_batch.clone().requires_grad_(True)
    E_sl = multipole_particle_mesh_ewald(
        pos_g_batch,
        mm_b,
        cells_g_batch,
        idx_j_batch,
        ptr_batch,
        sh_batch,
        sigma=sigma,
        alpha=alpha,
        batch_idx=batch_idx,
    )
    (grad_cells_batch,) = torch.autograd.grad(
        E_sl.sum(), cells_g_batch, create_graph=True
    )
    stress_batch = (
        torch.einsum("bca,bcd->bad", cells_batch, grad_cells_batch)
        / (vols_batch[:, None, None])
    )
    stress_labels_batch = torch.randn_like(stress_batch)
    stress_labels_batch = 0.5 * (
        stress_labels_batch + stress_labels_batch.transpose(-1, -2)
    )
    stress_loss_batch = ((stress_batch - stress_labels_batch) ** 2).mean()
    (grad_pos_stress_batch,) = torch.autograd.grad(stress_loss_batch, [pos_g_batch])
    print(
        f"Batched (B=2) {lmax_tag} PME stress-loss = "
        f"{stress_loss_batch.item():.4f}  "
        f"∂loss/∂positions max = {grad_pos_stress_batch.abs().max().item():.4f}"
    )


# %%
# Summary
# -------
# * ``multipole_particle_mesh_ewald`` is the composite PME entry point for
#   **l_max = 0, 1, and 2**, **single-system and batched** (single call; all
#   moments in one packed ``multipole_moments`` tensor). Positions,
#   ``multipole_moments`` (and the Cartesian leaves it was packed from), and
#   ``cell`` all carry ``requires_grad`` and gradients flow through autograd —
#   energy, forces, per-moment grads, and the **full stress tensor** (real +
#   reciprocal) at all l_max, including batched.
#   **Return shape:** per-atom ``(N,)`` single-system or ``(N_total,)``
#   batched — call ``.sum()`` for the system total, or ``scatter_add`` by
#   ``batch_idx`` for per-system totals; forces = ``-grad(E.sum(),
#   positions)``; stress = ``grad(E.sum(), cell)``.
# * The unified real-space-only entry point is
#   ``multipole_real_space_energy``, which takes the same packed
#   ``multipole_moments``, for when you need just the short-range half.
# * **Force-loss training (create_graph=True):** the full PME composite is
#   twice-differentiable at l_max=0/1/2, **single-system and batched**.
# * **Stress-loss training (create_graph=True through ``dE/dcell``):** the
#   mixed ``∂²E/∂cell∂θ`` is wired through the full composite at l_max=0/1/2,
#   **single-system and batched** — train energy + forces + stress together.
# * PME's O(N log M) scaling makes it preferable to direct-k-space Ewald
#   for N ≳ 1000 — see the ``benchmarks`` folder for wall-clock numbers.
