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

r"""
Multipole SCF Cache + Step (amortized fixed-cell workflow)
==========================================================

This example demonstrates the two-phase ``prepare_cache + scf_step`` pattern
for the **reciprocal-space ("k-space")** formulation of periodic multipole
electrostatics. Rather than the real-space + reciprocal split of
``multipole_ewald_summation``, this path evaluates the periodic Coulomb
interaction *directly* in reciprocal space — the GTO smearing of each multipole
makes the k-sum converge — so its whole geometry setup is cell-only and can be
built once and reused.

The reciprocal state for a periodic system — k-vectors, GTO Fourier factors
:math:`\hat\phi(\mathbf{k})`, per-k Coulomb factors, and overlap constants —
depends *only on the cell*, not on the atomic positions or per-atom moments.
``prepare_multipole_scf_cache`` builds that position-independent state **once**;
``multipole_scf_step_energy`` / ``multipole_scf_step_features`` then consume the
cache plus per-step ``(positions, source_feats)``, so the expensive geometry-only
work is amortized across many evaluations — MD steps at a fixed cell, SCF
iterations of an induced-moment model, or repeated feature extraction.

Throughout, ``l`` is the angular-momentum order of the multipole / feature
channels: ``l=0`` scalar (charges), ``l=1`` vector (dipoles), ``l=2`` rank-2
(quadrupoles).

Topics covered:

- Building a :class:`MultipoleSCFCache` once for a fixed cubic periodic cell.
- An MD-style loop that reuses the cache across many position updates, calling
  ``multipole_scf_step_energy`` each step.
- Feature extraction (``multipole_scf_step_features``) reusing the same cache.
- The cache is **cell-specific**: it must be rebuilt if the cell changes.
- A **batched** cache (one ``MultipoleSCFCache`` over a stack of cells) driven
  by ``batch_idx`` via the unified API.
- Autograd: the step energy is connected to ``positions`` and ``source_feats``.

.. important::
    The step functions take ``source_feats`` in the **e3nn-packed**
    ``(N, (l_max+1)**2)`` layout (``l_max=0`` → ``(N, 1)`` charges; ``l_max=1``
    → ``(N, 4)`` ``[q, mu_y, mu_z, mu_x]``), **not** the Cartesian packed
    ``multipole_moments``. For ``l_max <= 1`` both happen to coincide, so we use
    the public ``pack_multipole_moments`` helper to build the packed block from
    Cartesian channels. The Cartesian l=2 quadrupole is passed separately via
    the ``quadrupoles=(N, 3, 3)`` kwarg.

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
    multipole_scf_step_energy,
    multipole_scf_step_features,
    pack_multipole_moments,
    prepare_multipole_scf_cache,
)
from nvalchemiops.torch.math.gto import NormMode

# %%
# Configure Device
# ----------------

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

dtype = torch.float64

# %%
# Build a Small Cubic Periodic System
# ------------------------------------
# A handful of atoms with charges + dipoles in a cubic box. The cell is what
# the cache keys on; ``positions`` and ``source_feats`` flow per step.

rng = np.random.default_rng(2026)
n_atoms = 8
box_len = 6.0

cell = torch.eye(3, dtype=dtype, device=device) * box_len

positions0 = torch.tensor(
    rng.uniform(0.0, box_len, size=(n_atoms, 3)),
    dtype=dtype,
    device=device,
)

# Neutral charges + small random dipoles → an l_max=1 source.
charges = torch.tensor(rng.uniform(-1.0, 1.0, n_atoms), dtype=dtype, device=device)
charges = charges - charges.mean()  # enforce charge neutrality
dipoles_cart = torch.tensor(
    0.3 * rng.standard_normal((n_atoms, 3)), dtype=dtype, device=device
)

# ``source_feats`` is the e3nn-packed (N, 4) block [q, mu_y, mu_z, mu_x].
# For l_max <= 1 this matches ``pack_multipole_moments`` exactly.
source_feats = pack_multipole_moments(charges, dipoles_cart)  # (N, 4)
print(f"System: {n_atoms} atoms, cubic cell side = {box_len:.2f} Å")
print(f"source_feats shape (e3nn l_max=1 packed): {tuple(source_feats.shape)}")

# %%
# Build the Cache ONCE
# --------------------
# ``prepare_multipole_scf_cache`` runs the position-independent geometry
# kernels for this cell and returns an immutable bundle of device tensors. The
# ``sigma`` is the density-side Gaussian width; ``receiver_sigmas`` are the
# feature-side widths (only used by the feature step, but cheap to include).
# ``k_cutoff`` sets the reciprocal-space k-grid. Built once, reused below.

sigma = 1.0
receiver_sigmas = [0.8, 1.2]
k_cutoff = 3.5

cache = prepare_multipole_scf_cache(
    cell,
    sigma=sigma,
    receiver_sigmas=receiver_sigmas,
    k_cutoff=k_cutoff,
    l_max=1,
    feature_max_l=1,
    density_normalize=NormMode.MULTIPOLES,
    feature_normalize=NormMode.RECEIVER,
)
print(
    f"Cache built for this cell: n_k = {cache.n_k} k-vectors, "
    f"n_sigma = {cache.n_sigma}, l_max = {cache.l_max}"
)

# %%
# What the Cache Holds
# --------------------
# A :class:`MultipoleSCFCache` bundles the position-independent, cell-derived
# state of the reciprocal sum:
#
# - the **k-vectors** of the reciprocal grid (and the cell volume),
# - the **GTO Fourier factors** :math:`\hat\phi(\mathbf{k})` (source-density and
#   receiver bases),
# - the **per-k Coulomb factors** that weight each reciprocal mode, and
# - the **overlap constants** for the receiver-GTO projection.
#
# All of these depend only on the cell, so the per-step calls below only do the
# position- and moment-dependent work (structure factors + the moment
# contraction), not the geometry setup.

# %%
# MD-Style Loop Reusing the Cache
# -------------------------------
# Jitter the positions each "step" and recompute the periodic electrostatic
# energy via ``multipole_scf_step_energy(cache, positions, source_feats)``. The
# cache is built *outside* the loop — only the per-step moment-dependent
# arithmetic and the position-dependent structure factors are recomputed each
# step. The cell never changes, so the cache stays valid.

positions = positions0.clone()
# ``multipole_scf_step_energy`` returns PER-ATOM energies, shape ``(N,)``.
# Call ``.sum()`` for the system total; forces = ``-grad(E.sum(), positions)``.
print("MD-style energy trajectory (cache reused every step):")
for step in range(5):
    # Small random displacement, as an MD integrator would produce.
    positions = positions + 0.02 * torch.tensor(
        rng.standard_normal((n_atoms, 3)), dtype=dtype, device=device
    )
    energy = multipole_scf_step_energy(cache, positions, source_feats)
    print(f"  step {step}:  E = {energy.sum().item():+.6f}")

# %%
# Feature Extraction Reusing the Same Cache
# -----------------------------------------
# ``multipole_scf_step_features`` consumes the identical cache to produce
# atom-centered multipole features in the reference permuted flat layout
# ``(N_atoms, N_sigma * (feature_max_l + 1)**2)``. With two receiver sigmas and
# ``feature_max_l=1`` (4 e3nn components per σ), the width is ``2 * 4 = 8``.

features = multipole_scf_step_features(cache, positions, source_feats)
print(f"features shape: {tuple(features.shape)}  (N_atoms, N_sigma * 4)")
print(f"features dtype: {features.dtype}")

# %%
# The Cache Is Cell-Specific — Rebuild When the Cell Changes
# ----------------------------------------------------------
# Every tensor in the cache (k-vectors, :math:`\hat\phi`, per-k factors, volume)
# is derived from the cell. If the cell changes — e.g. an NPT barostat step or a
# different system — the cache MUST be rebuilt; reusing a stale cache would feed
# the wrong reciprocal-space state. (Within a fixed-cell NVT/NVE run or an SCF
# loop, one cache serves every step.)

cell_expanded = cell * 1.05  # a 5% isotropic cell expansion
cache_expanded = prepare_multipole_scf_cache(
    cell_expanded,
    sigma=sigma,
    receiver_sigmas=receiver_sigmas,
    k_cutoff=k_cutoff,
    l_max=1,
)
E_orig = multipole_scf_step_energy(cache, positions, source_feats)
E_expanded = multipole_scf_step_energy(cache_expanded, positions, source_feats)
print(f"E with original-cell cache  = {E_orig.sum().item():+.6f}")
print(f"E with expanded-cell cache  = {E_expanded.sum().item():+.6f}  (rebuilt cache)")

# %%
# Batched SCF Cache (Multiple Cells at Once)
# ------------------------------------------
# The unified API lets a single cache cover a **batch** of cells: pass a stacked
# ``(B, 3, 3)`` cell to ``prepare_multipole_scf_cache`` and a per-atom int32
# ``batch_idx`` (sorted so atoms group by system) to the step functions. All
# per-atom tensors are flat ``(N_total, ...)``; each atom is tied to its cell by
# ``batch_idx``. The flat per-system slices match the single-system calls.

cells_batch = torch.stack([cell, cell_expanded], dim=0)  # (B=2, 3, 3)
positions_batch = torch.cat([positions, positions], dim=0)
source_feats_batch = torch.cat([source_feats, source_feats], dim=0)
batch_idx = torch.cat(
    [
        torch.zeros(n_atoms, dtype=torch.int32, device=device),
        torch.ones(n_atoms, dtype=torch.int32, device=device),
    ]
)

cache_batch = prepare_multipole_scf_cache(
    cells_batch,
    sigma=sigma,
    receiver_sigmas=receiver_sigmas,
    k_cutoff=k_cutoff,
    l_max=1,
    feature_max_l=1,
)
E_batch = multipole_scf_step_energy(
    cache_batch, positions_batch, source_feats_batch, batch_idx=batch_idx
)
# E_batch is per-atom (N_total,). Reduce to per-system (B=2,) with scatter_add.
B = 2
E_batch_sys = torch.zeros(B, dtype=E_batch.dtype, device=E_batch.device).scatter_add(
    0, batch_idx.long(), E_batch
)
print(
    f"Batched (B=2) per-atom shape = {tuple(E_batch.shape)};"
    f" per-system after scatter_add: {tuple(E_batch_sys.shape)}"
)
print(f"  system 0 (original cell) E = {E_batch_sys[0].item():+.6f}")
print(f"  system 1 (expanded cell) E = {E_batch_sys[1].item():+.6f}")
print(
    "batched system-0 energy matches single-cell cache: "
    f"{torch.allclose(E_batch_sys[0], E_orig.sum())}"
)

# %%
# Autograd: Energy Is Connected to Positions and Moments
# ------------------------------------------------------
# The step energy is a normal autograd-connected scalar. Backprop to positions
# gives forces (``F = -dE/dr``); backprop to the Cartesian dipoles gives
# ``dE/dmu`` (the negative on-site field — what an induced-dipole SCF drives to
# zero at self-consistency). The pack is differentiable, so a ``requires_grad``
# Cartesian dipole recovers its Cartesian gradient.

pos_g = positions.clone().requires_grad_(True)
mu_g = dipoles_cart.clone().requires_grad_(True)
source_feats_g = pack_multipole_moments(charges, mu_g)

E = multipole_scf_step_energy(cache, pos_g, source_feats_g)
# E is per-atom (N,); differentiate the sum to get per-atom forces.
grad_pos, grad_mu = torch.autograd.grad(E.sum(), [pos_g, mu_g])
forces = -grad_pos
print(f"autograd energy  E (total) = {E.sum().item():+.6f}")
print(f"max |F| (= max |dE/dr|)    = {torch.norm(forces, dim=1).max().item():.4f}")
print(f"max |dE/dmu| (= |E_field|) = {grad_mu.abs().max().item():.4f}")

# %%
# Summary
# -------
# * ``prepare_multipole_scf_cache(cell, ...)`` builds the position-independent
#   reciprocal-space state **once** for a fixed cell.
# * ``multipole_scf_step_energy`` / ``multipole_scf_step_features`` consume that
#   cache plus per-step ``(positions, source_feats)`` — amortizing the
#   geometry-only work across MD steps, SCF iterations, or repeated feature
#   extraction.
# * ``source_feats`` is the **e3nn-packed** ``(N, (l_max+1)**2)`` block
#   (``[q, mu_y, mu_z, mu_x]`` at ``l_max=1``); l=2 quadrupoles go through the
#   separate ``quadrupoles=(N, 3, 3)`` kwarg.
# * The cache is **cell-specific** — rebuild it whenever the cell changes.
# * The step energy is autograd-connected to ``positions`` and ``source_feats``,
#   so forces and moment gradients flow through normally.
#   **Return shape:** per-atom ``(N,)`` single-system or ``(N_total,)`` batched
#   — call ``.sum()`` for the system total; forces = ``-grad(E.sum(),
#   positions)``; use ``scatter_add`` by ``batch_idx`` for per-system totals.
