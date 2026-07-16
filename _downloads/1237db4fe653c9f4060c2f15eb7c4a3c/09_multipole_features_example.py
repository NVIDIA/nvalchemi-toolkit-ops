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
Atom-Centered Multipole Electrostatic Features
==============================================

This example demonstrates ``multipole_electrostatic_features`` — a
reciprocal-space projection that turns the periodic electrostatic potential
of a set of source multipoles into LODE/ACE-style **per-atom feature
vectors**. Unlike the energy paths (Ewald / PME), this entry point needs
**no neighbor list**: every atom's descriptor is computed directly from the
k-space potential, projected onto a multi-σ basis of receiver Gaussian-type
orbitals (GTOs).

Throughout, ``l`` is the **angular-momentum order** of the spherical-harmonic
channel: ``l=0`` is a scalar (1 component), ``l=1`` a vector (3 components), and
``l=2`` a rank-2 tensor (5 components). ``feature_max_l`` is the **receiver cap**
on ``l`` — the highest angular order projected into the per-atom descriptor.

Topics covered:

- What atom-centered multipole features are and how to build them.
- ``feature_max_l=0`` (scalar, one channel per receiver σ) then
  ``feature_max_l=1`` (scalar + 3 vector channels per σ): output shapes
  and a few values.
- ``feature_max_l=2`` (adds the five l=2 receiver channels per σ).
- Richer source moments (charges → +dipoles → +quadrupoles) and how they
  change the features.
- Choosing ``feature_max_l`` independently of the source ``l_max``.
- Features are autograd-connected to ``positions`` and
  ``multipole_moments`` (a short ``.backward()`` demo).
- Batched (multi-system) extraction via the ``batch_idx`` argument.

The per-atom feature is

.. math::
    f_{i, \sigma_r, l, m} \;=\; \frac{2}{(2\pi)^3}
        \sum_{\mathbf{k}} w(\mathbf{k}) \cdot
        \text{Re}\!\left[V^{*}(\mathbf{k})\,
            \hat\phi_{l,m}^{\sigma_r}(\mathbf{k})\,
            e^{i\mathbf{k}\cdot\mathbf{r}_i}\right],

where :math:`V(\mathbf{k})` is the periodic potential assembled from the
source ``multipole_moments`` and :math:`\hat\phi_{l,m}^{\sigma_r}` is the
receiver-GTO Fourier coefficient at receiver width :math:`\sigma_r`.

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
    multipole_electrostatic_features,
    pack_multipole_moments,
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

# %%
# Build a Small Periodic System
# -----------------------------
# A handful of atoms in a cubic cell with assorted charges. The feature
# extractor takes a single ``(3, 3)`` ``cell`` (it is a per-system
# reciprocal-space projection) and packed per-atom ``multipole_moments``.


def make_system(dtype=torch.float64):
    """Return ``(positions, charges, cell)`` for a 6-atom cubic cell."""
    L = 6.0
    positions = torch.tensor(
        [
            [0.5, 0.5, 0.5],
            [2.5, 0.5, 0.5],
            [0.5, 2.5, 0.5],
            [0.5, 0.5, 2.5],
            [3.0, 3.0, 3.0],
            [4.5, 1.5, 2.0],
        ],
        dtype=dtype,
        device=device,
    )
    charges = torch.tensor(
        [1.0, -1.0, 0.5, -0.5, 0.8, -0.8],
        dtype=dtype,
        device=device,
    )
    # The feature path expects a single (3, 3) cell.
    cell = torch.eye(3, dtype=dtype, device=device) * L
    return positions, charges, cell


positions, charges, cell = make_system()
print(f"System: {len(positions)} atoms, cell side = {cell[0, 0]:.2f} Å")
print(f"Total charge: {charges.sum().item():+.4f}")

# %%
# The Receiver Basis and the k-Space Cutoff
# -----------------------------------------
# ``sigma`` is the **density-side** Gaussian width that smears each source
# multipole; ``receiver_sigmas`` is the multi-σ **receiver** basis the
# potential is projected onto. Each receiver width contributes one block of
# ``(feature_max_l + 1)**2`` channels to the per-atom feature vector, so the
# output width is ``len(receiver_sigmas) * (feature_max_l + 1)**2``.
# ``k_cutoff`` bounds the reciprocal sum (pass ``k_vectors`` instead to
# reuse a precomputed grid across calls for a fixed geometry).

sigma = 1.0
receiver_sigmas = [0.7, 1.0, 1.5]
k_cutoff = 4.0
n_sigma = len(receiver_sigmas)
print(f"density sigma = {sigma}, receiver_sigmas = {receiver_sigmas}")

# %%
# ``feature_max_l=0``: Scalar Features
# ------------------------------------
# With ``feature_max_l=0`` each atom gets one scalar (l=0) channel per
# receiver σ — shape ``(N, n_sigma * 1)``. These l=0 channels are
# translation-invariant. Source moments here are charges only, ``(N, 1)``.

moments_q = pack_multipole_moments(charges)  # (N, 1)
feats_fmax0 = multipole_electrostatic_features(
    positions,
    moments_q,
    cell,
    sigma=sigma,
    receiver_sigmas=receiver_sigmas,
    k_cutoff=k_cutoff,
    feature_max_l=0,
)
print(f"feature_max_l=0 features shape = {tuple(feats_fmax0.shape)}  (N, n_sigma * 1)")
print("first atom's l=0 features (one per receiver σ):")
print(feats_fmax0[0].tolist())

# %%
# ``feature_max_l=1``: Scalar + Vector Features
# ---------------------------------------------
# ``feature_max_l=1`` projects out the l=0 and l=1 receiver blocks: each σ
# now contributes ``(l+1)**2 = 4`` channels, so the output is
# ``(N, n_sigma * 4)``. The columns are laid out grouped by l-block
# (all σ for l=0, then all σ for l=1). The receiver cap is **independent**
# of the source ``l_max`` — here the source is still charges only.

feats_fmax1 = multipole_electrostatic_features(
    positions,
    moments_q,
    cell,
    sigma=sigma,
    receiver_sigmas=receiver_sigmas,
    k_cutoff=k_cutoff,
    feature_max_l=1,
)
print(f"feature_max_l=1 features shape = {tuple(feats_fmax1.shape)}  (N, n_sigma * 4)")
# The first n_sigma columns are the l=0 block and match the feature_max_l=0
# output exactly (the receiver l-blocks are decoupled).
print(
    "l=0 block matches feature_max_l=0 output: "
    f"{torch.allclose(feats_fmax1[:, :n_sigma], feats_fmax0)}"
)
print(f"first atom's full feature vector (len {feats_fmax1.shape[1]}):")
print(feats_fmax1[0].tolist())

# %%
# Richer Source Moments: Charges → Dipoles → Quadrupoles
# ------------------------------------------------------
# The *source* multipoles assemble the potential :math:`V(\mathbf{k})` that
# is projected. Enriching them (adding per-atom dipoles, then quadrupoles)
# changes the features even at a fixed receiver cap. Build the packed
# ``multipole_moments`` from physical **Cartesian** channels with
# ``pack_multipole_moments(charges, dipoles, quadrupoles)`` — it handles the
# Cartesian→e3nn permutation and the l=2 detrace for you.

rng = np.random.default_rng(2026)
dipoles_cart = torch.tensor(
    0.3 * rng.standard_normal((len(positions), 3)),
    dtype=positions.dtype,
    device=device,
)
# A clean physical axial (linear) quadrupole: diag(-1, -1, 2) is symmetric and
# traceless by construction, scaled per atom. ``pack_multipole_moments`` accepts
# any symmetric Cartesian (N, 3, 3) and drops a residual trace for you, so no
# manual symmetrize/detrace gymnastics are needed.
axial = torch.diag(
    torch.tensor([-1.0, -1.0, 2.0], dtype=positions.dtype, device=device)
)
strengths = torch.tensor(
    0.2 * rng.standard_normal(len(positions)), dtype=positions.dtype, device=device
)
quadrupoles_cart = strengths[:, None, None] * axial  # (N, 3, 3)

moments_qd = pack_multipole_moments(charges, dipoles_cart)  # (N, 4)
moments_qdq = pack_multipole_moments(charges, dipoles_cart, quadrupoles_cart)  # (N, 9)

# Keep the receiver cap at feature_max_l=1 — the source l_max may exceed it.
feats_qd = multipole_electrostatic_features(
    positions,
    moments_qd,
    cell,
    sigma=sigma,
    receiver_sigmas=receiver_sigmas,
    k_cutoff=k_cutoff,
    feature_max_l=1,
)
feats_qdq = multipole_electrostatic_features(
    positions,
    moments_qdq,
    cell,
    sigma=sigma,
    receiver_sigmas=receiver_sigmas,
    k_cutoff=k_cutoff,
    feature_max_l=1,
)
print("Effect of richer source moments (same receiver cap feature_max_l=1):")
print(f"  charges-only           ‖f‖ = {feats_fmax1.norm().item():.4f}")
print(f"  charges+dipoles        ‖f‖ = {feats_qd.norm().item():.4f}")
print(f"  charges+dipoles+quads  ‖f‖ = {feats_qdq.norm().item():.4f}")
print(f"  Δ‖f‖ from adding dipoles  = {(feats_qd - feats_fmax1).norm().item():.4f}")
print(f"  Δ‖f‖ from adding quads    = {(feats_qdq - feats_qd).norm().item():.4f}")

# %%
# ``feature_max_l=2``: l=2 Receiver Channels
# ------------------------------------------
# Raise the **receiver** cap to ``feature_max_l=2`` to project out the five
# l=2 channels too — ``(l+1)**2 = 9`` channels per σ, output
# ``(N, n_sigma * 9)``. The receiver cap is independent of the source: it
# works with a charges-only source as well, but a quadrupolar source feeds
# the l=2 channels more signal. The l≤1 columns are unchanged from the
# ``feature_max_l=1`` output (the l-blocks are decoupled).

feats_fmax2 = multipole_electrostatic_features(
    positions,
    moments_qdq,
    cell,
    sigma=sigma,
    receiver_sigmas=receiver_sigmas,
    k_cutoff=k_cutoff,
    feature_max_l=2,
)
print(f"feature_max_l=2 features shape = {tuple(feats_fmax2.shape)}  (N, n_sigma * 9)")
print(
    "l≤1 block matches feature_max_l=1 output: "
    f"{torch.allclose(feats_fmax2[:, : n_sigma * 4], feats_qdq)}"
)
# The trailing 5 * n_sigma columns are the new l=2 receiver channels.
print(
    f"first atom's l=2 receiver block max |value| = {feats_fmax2[0, n_sigma * 4 :].abs().max().item():.4f}"
)

# %%
# Choosing ``feature_max_l``
# --------------------------
# The source ``l_max`` (which physical multipoles you *supply* via
# ``multipole_moments``) and the receiver ``feature_max_l`` (the angular
# richness of the descriptor your *model consumes*) are chosen
# **independently**:
#
# - You can cap ``feature_max_l=1`` even with quadrupolar (l=2) sources — the
#   higher source moments still shape the l=0/l=1 receiver channels — if your
#   model only needs scalar + vector descriptors.
# - You can raise ``feature_max_l=2`` even for a charges-only source to obtain a
#   richer angular descriptor; the l=2 receiver channels capture the angular
#   structure of the surrounding charge density regardless of source order.
#
# Pick the source ``l_max`` from the physics (what multipoles your atoms carry)
# and ``feature_max_l`` from the model (how much angular detail it can use).
print("source l_max and receiver feature_max_l are chosen independently:")
print(f"  quadrupole source, feature_max_l=1 → width {feats_qdq.shape[1]} (l<=1)")
print(f"  quadrupole source, feature_max_l=2 → width {feats_fmax2.shape[1]} (l<=2)")
charges_only_fmax2 = multipole_electrostatic_features(
    positions,
    moments_q,
    cell,
    sigma=sigma,
    receiver_sigmas=receiver_sigmas,
    k_cutoff=k_cutoff,
    feature_max_l=2,
)
print(
    f"  charges-only source, feature_max_l=2 → width {charges_only_fmax2.shape[1]} "
    "(richer angular descriptor from a scalar source)"
)

# %%
# Normalization Modes
# -------------------
# ``density_normalize`` controls the source-basis normalization and
# ``feature_normalize`` the receiver-basis normalization. The defaults
# (``NormMode.MULTIPOLES`` for the density, ``NormMode.RECEIVER`` for the
# features) match the customer ``GTOElectrostaticFeatures`` convention.
# They are exposed for experimentation — here we just confirm the defaults
# round-trip and show how to pass them explicitly.

feats_default = multipole_electrostatic_features(
    positions,
    moments_qd,
    cell,
    sigma=sigma,
    receiver_sigmas=receiver_sigmas,
    k_cutoff=k_cutoff,
    feature_max_l=1,
    density_normalize=NormMode.MULTIPOLES,
    feature_normalize=NormMode.RECEIVER,
)
print(
    "explicit default NormModes reproduce the implicit defaults: "
    f"{torch.allclose(feats_default, feats_qd)}"
)

# %%
# Autograd: Features Are Differentiable
# -------------------------------------
# The features are autograd-connected to both ``positions`` and the packed
# ``multipole_moments`` (and, through the differentiable
# ``pack_multipole_moments`` gather, to the Cartesian charge / dipole /
# quadrupole leaves). Reduce the feature tensor to a scalar and ``.backward()``
# to obtain gradients — this is the path a feature-consuming model trains
# through.

pos_g = positions.clone().requires_grad_(True)
q_g = charges.clone().requires_grad_(True)
mu_g = dipoles_cart.clone().requires_grad_(True)
moments_g = pack_multipole_moments(q_g, mu_g)
cell_g = cell.clone().requires_grad_(True)

feats = multipole_electrostatic_features(
    pos_g,
    moments_g,
    cell_g,
    sigma=sigma,
    receiver_sigmas=receiver_sigmas,
    k_cutoff=k_cutoff,
    feature_max_l=1,
)
# A mock scalar readout (e.g. a linear head over the descriptor).
scalar = (feats**2).sum()
scalar.backward()
print(f"scalar readout = {scalar.item():.6f}")
print(f"‖∂scalar/∂positions‖ = {pos_g.grad.norm().item():.4f}")
print(f"‖∂scalar/∂charges‖   = {q_g.grad.norm().item():.4f}")
print(f"‖∂scalar/∂dipoles‖   = {mu_g.grad.norm().item():.4f}")
print(f"‖∂scalar/∂cell‖   = {cell_g.grad.norm().item():.4f}")

# %%
# Batched (Multi-System) Features
# -------------------------------
# The same ``multipole_electrostatic_features`` call extracts features for
# several systems at once via the ``batch_idx`` argument — there is no separate
# batched function. All per-atom tensors are flat ``(N_total, ...)``, the
# ``cell`` argument becomes a batched ``(B, 3, 3)`` stack, and ``batch_idx`` (an
# int32 per-atom system index, sorted so atoms group by system) ties each atom
# to its cell. The flat result equals the per-system single calls.

positions2, charges2, cell2 = make_system()
positions_batch = torch.cat([positions, positions2], dim=0)
moments_batch = pack_multipole_moments(
    torch.cat([charges, charges2], dim=0),
    torch.cat([dipoles_cart, dipoles_cart], dim=0),
)
cells_batch = torch.stack([cell, cell2], dim=0)  # (2, 3, 3)
batch_idx = torch.cat(
    [
        torch.zeros(len(positions), dtype=torch.int32, device=device),
        torch.ones(len(positions2), dtype=torch.int32, device=device),
    ]
)

feats_batch = multipole_electrostatic_features(
    positions_batch,
    moments_batch,
    cells_batch,
    batch_idx=batch_idx,
    sigma=sigma,
    receiver_sigmas=receiver_sigmas,
    k_cutoff=k_cutoff,
    feature_max_l=1,
)
print(
    f"Batched (B=2) features shape = {tuple(feats_batch.shape)}  (N_total, n_sigma * 4)"
)
# System 0's slice equals the single-system call on the same inputs.
print(
    "batched system-0 slice matches single-system call: "
    f"{torch.allclose(feats_batch[: len(positions)], feats_qd)}"
)

# %%
# Summary
# -------
# * ``multipole_electrostatic_features`` projects the periodic electrostatic
#   potential of a set of source multipoles onto a multi-σ receiver-GTO basis,
#   producing per-atom LODE/ACE-style descriptors — **no neighbor list
#   needed** (it is a reciprocal-space projection).
# * The output width is ``len(receiver_sigmas) * (feature_max_l + 1)**2``,
#   laid out grouped by receiver l-block; the receiver cap ``feature_max_l``
#   (0/1/2) is **independent** of the source ``l_max`` inferred from the
#   packed ``multipole_moments`` (1/4/9 columns).
# * Source moments go in one packed ``multipole_moments`` tensor; build it
#   from Cartesian channels with ``pack_multipole_moments(charges, dipoles,
#   quadrupoles)`` or pass an equivariant model's irrep block directly.
# * Features are autograd-connected to ``positions`` and
#   ``multipole_moments`` — backprop a model loss straight through.
# * Pass ``batch_idx`` (plus a batched ``(B, 3, 3)`` cell) to the same function
#   to run many systems in one flat call; the result matches the per-system
#   calls.
