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
Multipole Ewald Summation (charges + dipoles + quadrupoles)
============================================================

This example demonstrates the GTO-Ewald multipole electrostatics path for
periodic systems with charges (l_max=0), dipoles (l_max=1), and
quadrupoles (l_max=2). The Ewald path uses a direct k-space sum for the
reciprocal piece.

Topics covered:

- l_max=0/1/2 energy + forces (atomic gradients via autograd).
- Stress tensor via the cell-gradient (full Ewald: real + reciprocal).
- Force-loss-style training (l_max=2 ``create_graph=True`` through the
  **full composite**, real + reciprocal) — single-system AND batched.
- Stress-loss-style training (``create_graph=True`` through ``dE/dcell``,
  i.e. ∂²E/∂cell∂θ) — single-system AND batched.
- Full-PBC Ewald with parameter auto-estimation.
- Batched (multi-system) workflows, incl. batch-aware ``neighbor_list``.

The Ewald energy is decomposed as

.. math::
    E_\\text{total} = E_\\text{real} + E_\\text{reciprocal} - E_\\text{self}

with the GTO-Ewald short-range kernel
``T^{(0)}(r) = (\\text{erfc}(br) - \\text{erfc}(ar)) / r`` where
``a = 1/(2σ)`` and ``b = 1/(2σ_c)`` and ``σ_c = √(σ² + 1/(4α²))``.

.. important::
    Scope of the public Ewald API (l_max = 0, 1, **and 2** are all
    end-to-end):

    * ``multipole_ewald_summation`` (composite) — real + reciprocal + self
      for **all l_max**. All per-atom moments go in one packed
      ``multipole_moments`` tensor ``(N, (l_max+1)**2)`` in e3nn spherical
      order; ``pack_multipole_moments(charges, dipoles, quadrupoles)`` builds
      it from Cartesian channels (l=2 is **traceless**, 5 components), or pass
      an equivariant model's irrep block straight in. Energy, forces
      (``∂E/∂positions``), per-moment gradients, the stress tensor
      (``cell.requires_grad``), and **force-loss training**
      (``create_graph=True``) all flow through autograd at l_max=2.
    * ``multipole_real_space_energy(positions, multipole_moments, ...)`` — the
      unified public real-space-only entry point; returns per-atom for all
      l_max.

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
    multipole_ewald_summation,
    pack_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics.parameters import (
    estimate_multipole_ewald_parameters,
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
# the base fixture. Pass ``total_charge ≠ 0`` to bias the system —
# Ewald handles a uniform background-charge correction internally so a
# non-neutral cell is well-defined.


def make_periodic_system(
    n_cells: int = 2,
    L: float = 6.0,
    total_charge: float = 0.0,
    dtype=torch.float64,
):
    """Return ``(positions, charges, cell_3x3, cell_b, pbc)`` for a NaCl-like
    rock-salt supercell.

    Parameters
    ----------
    n_cells : int
        Number of replications along each axis. ``n_cells**3 * 2`` atoms total.
    L : float
        Side length of the (cubic) supercell in Å.
    total_charge : float, default 0.0
        Net charge of the system. The Na/Cl ± unit charges are perturbed by
        ``total_charge / N`` per atom to reach the requested net charge — a
        simple uniform offset (other charge-distribution schemes are
        possible). Ewald supports a non-neutral cell via the implicit
        uniform-background correction baked into ``multipole_ewald_summation``.
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
    # Uniformly offset every atomic charge so Σ q_i == total_charge.
    if total_charge != 0.0:
        N = charges.shape[0]
        charges = charges + (total_charge - charges.sum()) / N
    # Single-system Ewald wants cell shape (3, 3); the neighbor list and the
    # l_max=2 helper want (1, 3, 3).
    cell_3x3 = torch.eye(3, dtype=dtype, device=device) * L
    cell_b = cell_3x3.unsqueeze(0)
    pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
    return positions, charges, cell_3x3, cell_b, pbc


positions, charges, cell, cell_b, pbc = make_periodic_system()
print(f"System: {len(positions)} atoms, cell side = {cell[0, 0]:.2f} Å")
print(
    f"Total charge: {charges.sum().item():+.4f} (0.0 = neutral; tweak via `total_charge=`)"
)

# %%
# Build the Neighbor List
# -----------------------
# ``multipole_ewald_summation`` accepts a CSR-style neighbor list:
# a flat ``idx_j`` (target atoms), a ``neighbor_ptr`` (CSR row pointer of
# shape ``(N+1,)``), and per-pair PBC ``unit_shifts``. The general
# ``neighbor_list`` returns the list as a ``(2, n_pairs)`` COO tensor; we
# take the second row as ``idx_j``.

cutoff = 12.0
nl_2d, neighbor_ptr, unit_shifts = neighbor_list(
    positions,
    cutoff=cutoff,
    cell=cell_b,
    pbc=pbc,
    return_neighbor_list=True,
)
idx_j = nl_2d[1].contiguous()  # target column = flat per-atom neighbors
print(f"neighbor list: {idx_j.shape[0]} pairs, cutoff = {cutoff} Å")

# %%
# l_max=0: Charges Only — Energy + Forces
# ---------------------------------------
# ``multipole_ewald_summation`` takes a single packed ``multipole_moments``
# tensor of shape ``(N, (l_max+1)**2)`` in e3nn spherical-harmonic order.
# Build it with ``pack_multipole_moments(charges, dipoles=None,
# quadrupoles=None)`` from Cartesian channels (or pass an e3nn irrep block
# straight from an equivariant model). For l_max=0 the shape is ``(N, 1)``.

sigma = 0.5
alpha = 0.35
k_cutoff = 2.0

moments_l0 = pack_multipole_moments(charges)  # (N, 1) — charges only
pos_g = positions.clone().requires_grad_(True)
E_l0 = multipole_ewald_summation(
    pos_g,
    moments_l0,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=alpha,
    k_cutoff=k_cutoff,
)
# ``multipole_ewald_summation`` returns PER-ATOM energies, shape ``(N,)``.
# Call ``.sum()`` for the system total; use ``scatter_add`` for per-system
# totals in batched mode (see the batched section below).
(grad_pos,) = torch.autograd.grad(E_l0.sum(), [pos_g])
forces_l0 = -grad_pos  # ∂E/∂r_i = -F_i

print(
    f"l_max=0  E (total) = {E_l0.sum().item():.6f}, per-atom shape {tuple(E_l0.shape)}"
)
print(f"l_max=0  max |F| = {torch.norm(forces_l0, dim=1).max().item():.4f}")

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
# ``pack_multipole_moments`` takes the dipole in **physical Cartesian
# ``(x, y, z)``** and writes the packed ``(N, 4)`` block ``[q, μ_y, μ_z, μ_x]``
# (e3nn order) for you — so you reason in Cartesian and the e3nn permutation
# is handled at the boundary. The pack is a differentiable gather, so a
# ``requires_grad`` Cartesian dipole gets a Cartesian ``∂E/∂μ`` back.

rng = np.random.default_rng(2026)
dipoles_cart = torch.tensor(
    0.1 * rng.normal(size=(len(positions), 3)),
    dtype=positions.dtype,
    device=device,
)
moments_l1 = pack_multipole_moments(charges, dipoles_cart)  # (N, 4)

# Autograd-enabled copy: ``mu_cart_g`` (Cartesian) carries gradient through
# the packed moments; ``grad_mu`` below is ``∂E/∂μ`` in Cartesian (= -E_field).
pos_g = positions.clone().requires_grad_(True)
mu_cart_g = dipoles_cart.clone().requires_grad_(True)
moments_l1_grad = pack_multipole_moments(charges, mu_cart_g)

E_l1 = multipole_ewald_summation(
    pos_g,
    moments_l1_grad,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=alpha,
    k_cutoff=k_cutoff,
)
grad_pos, grad_mu = torch.autograd.grad(E_l1.sum(), [pos_g, mu_cart_g])
print(f"l_max=1  E (total) = {E_l1.sum().item():.6f}")
print(f"l_max=1  max |F|             = {torch.norm(-grad_pos, dim=1).max().item():.4f}")
print(f"l_max=1  max |∂E/∂μ| (= |E_field|) = {grad_mu.abs().max().item():.4f}")

# %%
# l_max=2: Quadrupoles via the Full Composite
# --------------------------------------------
# l_max=2 adds the per-atom **quadrupole** as the ``(N, 9)`` packed block's
# trailing 5 e3nn components — ``pack_multipole_moments(charges, dipoles,
# quadrupoles)`` converts a Cartesian symmetric ``(N, 3, 3)`` Q into them.
# The l=2 channel is **traceless** (5 DOF, e3nn convention): the isotropic
# trace of Q is dropped on the way in, so detrace Q yourself if you care
# about the value (a non-zero trace triggers a warning). The composite then
# returns the full Ewald-equivalent l_max=2 total (real + direct-k
# reciprocal + self).
#
# .. note::
#     An equivariant model emits the l=2 irrep directly — pass its ``(N, 9)``
#     output straight in, no Cartesian round-trip needed.

quadrupoles_cart = torch.tensor(
    0.05 * rng.normal(size=(len(positions), 3, 3)),
    dtype=positions.dtype,
    device=device,
)
# Symmetrize + detrace (l=2 is traceless symmetric, 5 DOF):
quadrupoles_cart = 0.5 * (quadrupoles_cart + quadrupoles_cart.transpose(-1, -2))
quadrupoles_cart = quadrupoles_cart - (
    torch.diagonal(quadrupoles_cart, dim1=-2, dim2=-1).sum(-1)[:, None, None] / 3.0
) * torch.eye(3, dtype=positions.dtype, device=device)

pos_g = positions.clone().requires_grad_(True)
q_g = charges.clone().requires_grad_(True)
mu_g_cart = dipoles_cart.clone().requires_grad_(True)
Q_g = quadrupoles_cart.clone().requires_grad_(True)

# One packed (N, 9) tensor carries all three channels (autograd-connected).
moments_l2 = pack_multipole_moments(q_g, mu_g_cart, Q_g)
E_l2 = multipole_ewald_summation(
    pos_g,
    moments_l2,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=alpha,
    k_cutoff=k_cutoff,
)
grads = torch.autograd.grad(E_l2.sum(), [pos_g, q_g, mu_g_cart, Q_g])
print(f"l_max=2  E (full Ewald, total) = {E_l2.sum().item():.6f}")
print(f"l_max=2  max |F|       = {torch.norm(-grads[0], dim=1).max().item():.4f}")
print(f"l_max=2  max |∂E/∂q|   = {grads[1].abs().max().item():.4f}")
print(f"l_max=2  max |∂E/∂μ|   = {grads[2].abs().max().item():.4f}")
print(f"l_max=2  max |∂E/∂Q|   = {grads[3].abs().max().item():.4f}")

# %%
# Stress Tensor via the Cell Gradient
# -----------------------------------
# For periodic systems the stress tensor is
#
# .. math::
#     \\sigma_{ab} = \\frac{1}{V} \\sum_b c_{ab} \\frac{\\partial E}{\\partial c_{ab}}
#
# (here ``c`` is the cell matrix in row-vector form, so
# ``∂E/∂c[a, b]`` is the gradient of the energy w.r.t. lattice entry
# ``(a, b)``; the cell-grad kernels emit this directly).
#
# The full Ewald stress (real-space + direct-k reciprocal cell-gradient) is
# wired for all l_max — ``multipole_ewald_summation`` produces the complete
# Ewald-equivalent stress directly through autograd. At l_max=2 just pass the
# packed ``(N, 9)`` moments and the composite stress is complete.

# l_max=1 full stress via composite Ewald (real + reciprocal cell-gradient).
cell_g = cell.clone().requires_grad_(True)
E_for_stress = multipole_ewald_summation(
    positions,
    moments_l1,
    cell_g,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=alpha,
    k_cutoff=k_cutoff,
)
(grad_cell,) = torch.autograd.grad(E_for_stress.sum(), [cell_g])
volume = float(torch.det(cell))
stress_l1 = (cell.T @ grad_cell) / volume
print("l_max=1 stress tensor (full Ewald: real + reciprocal):")
print(stress_l1)

# l_max=2 full stress via the composite (real + direct-k reciprocal).
moments_l2_static = pack_multipole_moments(charges, dipoles_cart, quadrupoles_cart)
cell_g = cell.clone().requires_grad_(True)
E_l2_for_stress = multipole_ewald_summation(
    positions,
    moments_l2_static,
    cell_g,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=alpha,
    k_cutoff=k_cutoff,
)
(grad_cell_l2,) = torch.autograd.grad(E_l2_for_stress.sum(), [cell_g])
stress_l2 = (cell.T @ grad_cell_l2) / volume
print("\nl_max=2 stress tensor (full Ewald: real + reciprocal):")
print(stress_l2)

# %%
# Force-Loss-Style Training (l_max=2 ``create_graph=True``, full Ewald)
# --------------------------------------------------------------------
# ``create_graph=True`` works through the **full** l_max=2 Ewald composite —
# both the real-space and the direct-k-space reciprocal second-order
# backward are registered. So forces from the l_max=2
# ``multipole_ewald_summation`` are autograd-connected and you can backprop a
# force-error loss to positions or moments. This is the canonical MLIP
# force-loss pattern, and it includes the reciprocal contribution to
# ``∂²E/∂x∂y``.

pos_g = positions.clone().requires_grad_(True)
E = multipole_ewald_summation(
    pos_g,
    moments_l2_static,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=alpha,
    k_cutoff=k_cutoff,
)
# Forces — autograd-connected with create_graph=True (full Ewald: the
# graph runs both the real-space and reciprocal 2nd-order kernels).
# E is per-atom (N,); sum to a scalar before differentiating.
forces = -torch.autograd.grad(E.sum(), pos_g, create_graph=True)[0]

# Mock force-error loss against random "labels".
force_labels = torch.randn_like(forces)
force_loss = ((forces - force_labels) ** 2).mean()
# Backprop through the force computation itself (full Ewald 2nd-order).
(grad_pos_2nd,) = torch.autograd.grad(force_loss, [pos_g])
print(f"l_max=2 force-loss = {force_loss.item():.4f}")
print(f"l_max=2 ∂loss/∂positions max = {grad_pos_2nd.abs().max().item():.4f}")

# %%
# Stress-Loss Training (l_max=2 ``create_graph=True`` through ``dE/dcell``)
# ------------------------------------------------------------------------
# The cousin of force-loss: backprop a **stress**-error loss. The stress is
# itself a first derivative ``∂E/∂cell``, so training it to a target requires
# the *mixed* second derivative ``∂²E/∂cell∂θ`` (θ = positions / moments).
# Take the stress with ``create_graph=True`` and backprop the loss — this runs
# the full l_max=2 composite second-order backward through the **cell** on both
# the real-space and the direct-k reciprocal halves (the reciprocal cell↔
# {positions, moments} cross-terms are wired single-system AND batched).
#
# Pair this with the force-loss above and an energy term for the canonical
# MLIP training objective (energy + forces + stress).

pos_g = positions.clone().requires_grad_(True)
cell_g = cell.clone().requires_grad_(True)
E = multipole_ewald_summation(
    pos_g,
    moments_l2_static,
    cell_g,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=alpha,
    k_cutoff=k_cutoff,
)
# Virial stress with create_graph=True so it stays autograd-connected to the
# inputs (full Ewald: real + reciprocal cell second-order).
(grad_cell,) = torch.autograd.grad(E.sum(), cell_g, create_graph=True)
stress = (cell.T @ grad_cell) / volume
# Mock stress-error loss against a random symmetric target.
stress_label = torch.randn_like(stress)
stress_label = 0.5 * (stress_label + stress_label.T)
stress_loss = ((stress - stress_label) ** 2).mean()
# Backprop the stress-loss to positions (∂²E/∂cell∂positions).
(grad_pos_stress,) = torch.autograd.grad(stress_loss, [pos_g])
print(f"l_max=2 stress-loss = {stress_loss.item():.4f}")
print(
    f"l_max=2 ∂(stress-loss)/∂positions max = {grad_pos_stress.abs().max().item():.4f}"
)

# %%
# Auto-Estimated Ewald Parameters
# -------------------------------
# ``multipole_ewald_summation`` accepts ``alpha=None`` and
# ``k_cutoff=None``; the Kolafa-Perram balance auto-estimates them
# from ``sigma`` and the system geometry at the requested accuracy
# (relative energy error). You can also call the estimator yourself
# first when you want to log / log-scan / cache the values — that's
# what we do below.

est = estimate_multipole_ewald_parameters(
    positions,
    cell_b,
    sigma,
    accuracy=1e-6,
)
# ``est`` is a MultipoleEwaldParameters dataclass with shape-(B,) tensors.
alpha_auto = float(est.alpha.item())
rcut_auto = float(est.real_space_cutoff.item())
kcut_auto = float(est.reciprocal_space_cutoff.item())
print("Kolafa-Perram balance at accuracy=1e-6:")
print(f"  α                    = {alpha_auto:.4f}  (1 / Å)")
print(f"  real-space cutoff    = {rcut_auto:.4f}  Å")
print(f"  reciprocal cutoff |k|= {kcut_auto:.4f}  (1 / Å)")

E_auto = multipole_ewald_summation(
    positions,
    moments_l1,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=alpha_auto,
    k_cutoff=kcut_auto,
)
print(f"auto-parameters Ewald (l_max=1): E = {E_auto.sum().item():.6f}")
# Equivalent shortcut — pass None and let the wrapper estimate internally:
E_auto_via_none = multipole_ewald_summation(
    positions,
    moments_l1,
    cell,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigma=sigma,
    alpha=None,
    k_cutoff=None,
    accuracy=1e-6,
)
print(f"  (matches alpha/k_cutoff=None shortcut: {E_auto_via_none.sum().item():.6f})")

# %%
# Batched Workflow (Multi-System)
# -------------------------------
# Pack multiple systems into a single flat call by providing ``batch_idx``.
# All per-atom tensors are flat ``(N_total, ...)``; ``cell`` becomes
# ``(B, 3, 3)``. ``neighbor_list`` is batch-aware: pass the same
# ``batch_idx`` (plus a ``(B, 3, 3)`` ``cell`` and ``(B, 3)`` ``pbc``) and it
# runs the batched search and returns a single flat CSR — ``idx_j`` already
# carries global atom indices and ``neighbor_ptr`` spans all systems, with
# every atom's neighbors confined to its own system. No manual per-system
# stitching needed.

# Build two identical systems for a 2-batch demo.
p1, c1, _, cellb1, pbc1 = make_periodic_system()
p2, c2, _, cellb2, _ = make_periodic_system()
n_per_system = len(p1)

positions_batch = torch.cat([p1, p2], dim=0)
charges_batch = torch.cat([c1, c2], dim=0)
moments_l0_batch = pack_multipole_moments(charges_batch)  # (N_total, 1)
cells_batch = torch.cat([cellb1, cellb2], dim=0)  # (2, 3, 3)
pbc_batch = torch.cat([pbc1, pbc1], dim=0)  # (2, 3)
batch_idx = torch.cat(
    [
        torch.zeros(n_per_system, dtype=torch.int32, device=device),
        torch.ones(n_per_system, dtype=torch.int32, device=device),
    ]
)

# Batched neighbor list → flat CSR (idx_j is global, neighbor_ptr spans all
# systems). Passing the 3-D ``cell`` lets the method auto-select without a
# device-to-host sync on ``batch_idx``.
nl_2d_batch, ptr_batch, sh_batch = neighbor_list(
    positions_batch,
    cutoff=cutoff,
    cell=cells_batch,
    pbc=pbc_batch,
    batch_idx=batch_idx,
    return_neighbor_list=True,
)
idx_j_batch = nl_2d_batch[1].contiguous()

E_batch = multipole_ewald_summation(
    positions_batch,
    moments_l0_batch,
    cells_batch,
    idx_j_batch,
    ptr_batch,
    sh_batch,
    sigma=sigma,
    alpha=alpha,
    k_cutoff=k_cutoff,
    batch_idx=batch_idx,
)
# E_batch is per-atom (N_total,). Reduce to per-system (B,) with scatter_add.
B = int(cells_batch.shape[0])
E_batch_sys = torch.zeros(B, dtype=E_batch.dtype, device=E_batch.device).scatter_add(
    0, batch_idx.long(), E_batch
)
print(f"Batched (B=2) l_max=0 per-system energies: {E_batch_sys.tolist()}")

# The **full composite** (real + direct-k reciprocal + self) is batched at
# every l_max — the batched direct-k reciprocal supports charges, dipoles,
# and Cartesian quadrupoles. Just pass the packed ``multipole_moments``
# (1/4/9 columns) + ``batch_idx``; the result equals the per-system
# single-system composite to fp64 round-off.
dipoles_batch = torch.cat([dipoles_cart, dipoles_cart], dim=0)
quadrupoles_batch = torch.cat([quadrupoles_cart, quadrupoles_cart], dim=0)
moments_l1_batch = pack_multipole_moments(charges_batch, dipoles_batch)
moments_l2_batch = pack_multipole_moments(
    charges_batch, dipoles_batch, quadrupoles_batch
)

E_batch_l1 = multipole_ewald_summation(
    positions_batch,
    moments_l1_batch,
    cells_batch,
    idx_j_batch,
    ptr_batch,
    sh_batch,
    sigma=sigma,
    alpha=alpha,
    k_cutoff=k_cutoff,
    batch_idx=batch_idx,
)
E_batch_l1_sys = torch.zeros(
    B, dtype=E_batch_l1.dtype, device=E_batch_l1.device
).scatter_add(0, batch_idx.long(), E_batch_l1)
print(
    f"Batched (B=2) l_max=1 full-Ewald per-system energies: {E_batch_l1_sys.tolist()}"
)

E_batch_l2 = multipole_ewald_summation(
    positions_batch,
    moments_l2_batch,
    cells_batch,
    idx_j_batch,
    ptr_batch,
    sh_batch,
    sigma=sigma,
    alpha=alpha,
    k_cutoff=k_cutoff,
    batch_idx=batch_idx,
)
E_batch_l2_sys = torch.zeros(
    B, dtype=E_batch_l2.dtype, device=E_batch_l2.device
).scatter_add(0, batch_idx.long(), E_batch_l2)
print(
    f"Batched (B=2) l_max=2 full-Ewald per-system energies: {E_batch_l2_sys.tolist()}"
)

# %%
# Batched Force-Loss Training (``create_graph=True``, l_max=1 and l_max=2)
# -----------------------------------------------------------------------
# Multi-system force-loss training is the most common end workflow, and it
# works through the **batched full composite** at both l_max=1 and l_max=2:
# the real-space second-order backward and the direct-k reciprocal
# double-backward are both wired for the batched path. Take
# ``∂E/∂positions`` with ``create_graph=True``,
# form a per-system force-error loss, and backprop it to positions/moments —
# exactly as in the single-system section, but with ``batch_idx`` set. The
# batched 2nd-order matches the per-system single-system result atom-for-atom.

# Build a label tensor and run the batched force loss at l_max=1 and l_max=2.
force_labels_batch = torch.randn(
    positions_batch.shape[0], 3, dtype=positions.dtype, device=device
)

for lmax_tag, moments_b in (
    ("l_max=1", moments_l1_batch),
    ("l_max=2", moments_l2_batch),
):
    pos_g_batch = positions_batch.clone().requires_grad_(True)
    E_fl = multipole_ewald_summation(
        pos_g_batch,
        moments_b,
        cells_batch,
        idx_j_batch,
        ptr_batch,
        sh_batch,
        sigma=sigma,
        alpha=alpha,
        k_cutoff=k_cutoff,
        batch_idx=batch_idx,
    )
    # Forces for the whole batch (autograd-connected). ``E_fl`` is per-atom
    # ``(N_total,)``; ``.sum()`` gives ∂(Σ_i E_i)/∂r_i = the per-atom forces.
    forces_batch = -torch.autograd.grad(E_fl.sum(), pos_g_batch, create_graph=True)[0]
    force_loss_batch = ((forces_batch - force_labels_batch) ** 2).mean()
    (grad_pos_2nd_batch,) = torch.autograd.grad(force_loss_batch, [pos_g_batch])
    print(
        f"Batched (B=2) {lmax_tag} force-loss = {force_loss_batch.item():.4f}  "
        f"∂loss/∂positions max = {grad_pos_2nd_batch.abs().max().item():.4f}"
    )

# %%
# Batched Stress-Loss Training (``create_graph=True`` through ``dE/dcell``)
# ------------------------------------------------------------------------
# Stress-loss is batched too: a ``(B, 3, 3)`` ``cell`` carries grad, the
# per-system virial is autograd-connected, and the batched composite
# second-order backward (real-space + direct-k reciprocal cell cross-terms)
# lets a stress-error loss backprop to positions. Matches the per-system
# single-system result atom-for-atom.

vols_batch = torch.det(cells_batch)  # (B,)
for lmax_tag, moments_b in (
    ("l_max=1", moments_l1_batch),
    ("l_max=2", moments_l2_batch),
):
    pos_g_batch = positions_batch.clone().requires_grad_(True)
    cells_g_batch = cells_batch.clone().requires_grad_(True)
    E_sl = multipole_ewald_summation(
        pos_g_batch,
        moments_b,
        cells_g_batch,
        idx_j_batch,
        ptr_batch,
        sh_batch,
        sigma=sigma,
        alpha=alpha,
        k_cutoff=k_cutoff,
        batch_idx=batch_idx,
    )
    # Per-system virial stress (B, 3, 3), create_graph-connected.
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
        f"Batched (B=2) {lmax_tag} stress-loss = {stress_loss_batch.item():.4f}  "
        f"∂loss/∂positions max = {grad_pos_stress_batch.abs().max().item():.4f}"
    )

# %%
# Summary
# -------
# * ``multipole_ewald_summation`` is the composite Ewald entry point for
#   **l_max = 0, 1, and 2** — real + direct-k reciprocal + self, end-to-end.
#   All per-atom moments go in one packed ``multipole_moments`` tensor
#   ``(N, (l_max+1)**2)`` in e3nn order; build it from Cartesian channels with
#   ``pack_multipole_moments(charges, dipoles, quadrupoles)`` (l=2 is traceless,
#   5 components) or pass an equivariant model's irrep block directly.
#   **Return shape:** per-atom ``(N,)`` single-system or ``(N_total,)``
#   batched — call ``.sum()`` for the system total, or ``scatter_add`` by
#   ``batch_idx`` to get per-system totals; forces = ``-grad(E.sum(),
#   positions)``; stress = ``grad(E.sum(), cell)``.
# * Positions, the packed moments, and ``cell`` can all carry
#   ``requires_grad=True``; gradients flow through autograd — including the
#   **full Ewald stress tensor** (real-space + reciprocal cell-gradient,
#   l_max ≤ 2). A Cartesian-channel gradient is recovered by packing a
#   ``requires_grad`` Cartesian tensor (the pack is differentiable).
# * **Force-loss training** (``create_graph=True``) is wired through the
#   **full composite** at l_max=2 — **single-system AND batched** — via the
#   real-space second-order backward plus the direct-k reciprocal Q-channel
#   double-backward. The reciprocal contribution to ``∂²E/∂x∂y`` is included
#   (batched matches per-system atom-for-atom).
# * **Stress-loss training** (``create_graph=True`` through ``dE/dcell``, i.e.
#   the mixed ``∂²E/∂cell∂θ``) is likewise wired through the full composite at
#   l_max=0/1/2, **single-system AND batched** — real-space + direct-k
#   reciprocal cell cross-terms. Train energy + forces + stress together for
#   the canonical MLIP objective.
# * The unified real-space-only entry point is
#   ``multipole_real_space_energy(positions, multipole_moments, ...)``,
#   returning per-atom for all l_max.
