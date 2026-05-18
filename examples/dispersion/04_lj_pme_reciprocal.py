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
LJ-PME Reciprocal-Space + Self-Energy (PR1 of LJ-PME)
======================================================

This example demonstrates the FFT-based reciprocal-space and self-energy
components of Lennard-Jones PME (LJ-PME) for the attractive dispersion
(:math:`r^{-6}`) interaction. With geometric combination rules
(:math:`C_{6,ij} = \\sqrt{C_{6,ii}\\,C_{6,jj}}`) the long-range sum
factorizes into a single FFT — the same cost as Coulomb PME.

The total LJ-PME energy (Wennberg et al., JCTC 2013) is

.. math::

    V_{\\text{total}} = V_{\\text{real}} + V_{\\text{recip}} - V_{\\text{self}}.

This example covers the *reciprocal-space* and *self-energy* terms only
(landed in PR1). The damped real-space term arrives in PR2; the unified
``lj_pme()`` top-level orchestrator in PR3.

Two backends are demonstrated side-by-side from the same script:

* **PyTorch** — eager evaluation through ``torch.compile``-friendly
  ``warp_custom_op`` wrappers.
* **JAX** — JIT-compatible ``jax_kernel`` wrappers (skipped if JAX is
  not installed).

In this example you will learn:

- Compute the reciprocal-space dispersion energy with
  ``pme_dispersion_reciprocal_space``
- Compute the self-energy correction with
  ``pme_dispersion_energy_corrections``
- Verify the isolated-atom limit: with no neighbors, V_recip → V_self
- Watch mesh convergence of V_recip at fixed β
- Run multiple independent systems in a single batched call
- Compose with ``jax.jit`` (JAX backend)

.. important::
    This script is intended as an API demonstration. Do not use it for
    performance benchmarking; refer to the ``benchmarks`` folder instead.
"""

# %%
# Setup and Imports
# -----------------

from __future__ import annotations

import numpy as np

# %%
# Build a Backend-Neutral System Description
# ------------------------------------------
# Generate positions / C6 / cell as NumPy arrays so the same data feeds
# both backends. We use argon FCC as a clean dispersion test bed: a
# single element, neutral, and dominated by London dispersion.


def create_argon_fcc(n_cells: int = 3, lattice_constant: float = 5.26):
    """Argon FCC supercell (NumPy arrays)."""
    fcc_basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )
    positions = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                offset = np.array([i, j, k])
                positions.extend(
                    (frac + offset) * lattice_constant for frac in fcc_basis
                )
    positions_np = np.array(positions, dtype=np.float64)
    c6_np = np.full(positions_np.shape[0], 60.0, dtype=np.float64)
    cell_np = (np.eye(3, dtype=np.float64) * lattice_constant * n_cells)[None, :, :]
    return positions_np, c6_np, cell_np


# A medium supercell for the bulk demonstration, plus a single-atom box
# for the isolated-atom convergence check.
pos_np, c6_np, cell_np = create_argon_fcc(n_cells=3)
beta_val = 0.35
mesh_dimensions = (32, 32, 32)
print(f"System: {pos_np.shape[0]}-atom argon FCC supercell")
print(f"  Cell length: {cell_np[0, 0, 0]:.2f} Å")
print(f"  C6 per atom: {c6_np[0]:.1f}  (illustrative units)")

# Isolated-atom test system: one atom at the center of a 20 Å cubic box.
pos_iso_np = np.array([[10.0, 10.0, 10.0]], dtype=np.float64)
c6_iso_np = np.array([60.0], dtype=np.float64)
cell_iso_np = (np.eye(3, dtype=np.float64) * 20.0)[None, :, :]

# Two independent systems for the batched demonstration.
pos_a, c6_a, cell_a = create_argon_fcc(n_cells=2, lattice_constant=5.20)
pos_b, c6_b, cell_b = create_argon_fcc(n_cells=2, lattice_constant=5.45)
pos_batch_np = np.concatenate([pos_a, pos_b], axis=0)
c6_batch_np = np.concatenate([c6_a, c6_b], axis=0)
cell_batch_np = np.concatenate([cell_a, cell_b], axis=0)
batch_idx_np = np.concatenate(
    [
        np.zeros(pos_a.shape[0], dtype=np.int32),
        np.ones(pos_b.shape[0], dtype=np.int32),
    ]
)
beta_batch_np = np.array([0.35, 0.40], dtype=np.float64)


# =============================================================================
# PyTorch Backend
# =============================================================================
print("\n" + "=" * 70)
print("PyTorch backend")
print("=" * 70)

try:
    import torch

    from nvalchemiops.torch.interactions.dispersion import (
        pme_dispersion_energy_corrections,
        pme_dispersion_reciprocal_space,
    )

    HAS_TORCH = True
except ImportError as exc:
    HAS_TORCH = False
    print(f"  PyTorch backend unavailable ({exc}); skipping.")

if HAS_TORCH:
    if torch.cuda.is_available():
        torch_device = torch.device("cuda:0")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        torch_device = torch.device("cpu")
        print("  Device: CPU")

    DTYPE = torch.float64

    def _t(arr, dtype=DTYPE):
        """numpy -> torch on the active device."""
        if isinstance(arr, np.ndarray) and arr.dtype.kind == "i":
            return torch.from_numpy(arr).to(torch_device)
        return torch.tensor(arr, dtype=dtype, device=torch_device)

    positions = _t(pos_np)
    c6 = _t(c6_np)
    cell = _t(cell_np)
    beta = torch.tensor([beta_val], dtype=DTYPE, device=torch_device)

    E_recip = pme_dispersion_reciprocal_space(
        positions=positions,
        c6_coefficients=c6,
        cell=cell,
        beta=beta,
        mesh_dimensions=mesh_dimensions,
        spline_order=4,
    )
    E_self = pme_dispersion_energy_corrections(c6, beta)
    print(f"\n  beta = {float(beta[0]):.3f} Å⁻¹  mesh = {mesh_dimensions}")
    print(f"    V_recip = {E_recip.item():+.6f}  (FFT-based long-range sum)")
    print(f"    V_self  = {E_self.item():+.6f}  (= -β⁶ ΣC6_ii / 12)")

    # Isolated-atom convergence
    pos_iso = _t(pos_iso_np)
    c6_iso = _t(c6_iso_np)
    cell_iso = _t(cell_iso_np)
    print("\n  Isolated-atom convergence (V_recip / V_self → 1):")
    print("      beta  | V_recip      | V_self       | (V_recip-V_self)/|V_self|")
    print("    " + "-" * 65)
    for bv in [0.2, 0.3, 0.5, 0.8, 1.0]:
        b_t = torch.tensor([bv], dtype=DTYPE, device=torch_device)
        er = pme_dispersion_reciprocal_space(
            pos_iso,
            c6_iso,
            cell_iso,
            b_t,
            mesh_dimensions=(64, 64, 64),
            spline_order=4,
        )
        es = pme_dispersion_energy_corrections(c6_iso, b_t)
        rel = (er.item() - es.item()) / abs(es.item())
        print(f"     {bv:.2f}  | {er.item():+11.4e} | {es.item():+11.4e} | {rel:+.2e}")

    # Mesh convergence
    print("\n  Mesh convergence (β=0.35, fixed argon supercell):")
    print("      mesh   | V_recip        | |ΔV_recip| vs finest")
    print("    " + "-" * 55)
    finest = None
    results = []
    for n in [16, 24, 32, 48, 64, 96]:
        er = pme_dispersion_reciprocal_space(
            positions,
            c6,
            cell,
            beta,
            mesh_dimensions=(n, n, n),
            spline_order=4,
        )
        results.append((n, er.item()))
        finest = er.item()
    for n, val in results:
        err = abs(val - finest)
        print(f"     {n:3d}³  | {val:+13.6e}  | {err:.2e}")

    # Batched evaluation
    pos_batch = _t(pos_batch_np)
    c6_batch = _t(c6_batch_np)
    cell_batch = _t(cell_batch_np)
    batch_idx = _t(batch_idx_np, dtype=torch.int32)
    beta_batch = _t(beta_batch_np)
    E_recip_batch = pme_dispersion_reciprocal_space(
        positions=pos_batch,
        c6_coefficients=c6_batch,
        cell=cell_batch,
        beta=beta_batch,
        mesh_dimensions=(32, 32, 32),
        spline_order=4,
        batch_idx=batch_idx,
    )
    E_self_batch = pme_dispersion_energy_corrections(
        c6_batch, beta_batch, batch_idx=batch_idx
    )
    print("\n  Batched evaluation (two argon supercells):")
    for s in range(2):
        print(
            f"    system {s}: V_recip = {E_recip_batch[s].item():+.4e}, "
            f"V_self = {E_self_batch[s].item():+.4e}"
        )


# =============================================================================
# JAX Backend
# =============================================================================
print("\n" + "=" * 70)
print("JAX backend")
print("=" * 70)

try:
    import jax
    import jax.numpy as jnp

    from nvalchemiops.jax.interactions.dispersion import (
        pme_dispersion_energy_corrections as jax_pme_dispersion_energy_corrections,
    )
    from nvalchemiops.jax.interactions.dispersion import (
        pme_dispersion_reciprocal_space as jax_pme_dispersion_reciprocal_space,
    )

    HAS_JAX = True
except ImportError as exc:
    HAS_JAX = False
    print(f"  JAX backend unavailable ({exc}); skipping.")

if HAS_JAX:
    print(f"  JAX devices: {jax.devices()}")

    def _j(arr, dtype=jnp.float64):
        """numpy -> jax."""
        if isinstance(arr, np.ndarray) and arr.dtype.kind == "i":
            return jnp.array(arr, dtype=jnp.int32)
        return jnp.array(arr, dtype=dtype)

    positions_j = _j(pos_np)
    c6_j = _j(c6_np)
    cell_j = _j(cell_np)
    beta_j = jnp.array([beta_val], dtype=jnp.float64)

    E_recip_j = jax_pme_dispersion_reciprocal_space(
        positions=positions_j,
        c6_coefficients=c6_j,
        cell=cell_j,
        beta=beta_j,
        mesh_dimensions=mesh_dimensions,
        spline_order=4,
    )
    E_self_j = jax_pme_dispersion_energy_corrections(c6_j, beta_j)
    print(f"\n  beta = {float(beta_j[0]):.3f} Å⁻¹  mesh = {mesh_dimensions}")
    print(f"    V_recip = {float(E_recip_j[0]):+.6f}")
    print(f"    V_self  = {float(E_self_j[0]):+.6f}")

    # JIT-compiled evaluation
    @jax.jit
    def lj_pme_recip_minus_self_jit(positions, c6, cell, beta):
        """JIT-compiled reciprocal + self-energy total for a single system."""
        e_recip = jax_pme_dispersion_reciprocal_space(
            positions=positions,
            c6_coefficients=c6,
            cell=cell,
            beta=beta,
            mesh_dimensions=mesh_dimensions,
            spline_order=4,
        )
        e_self = jax_pme_dispersion_energy_corrections(c6, beta)
        return e_recip - e_self

    e_pr1 = lj_pme_recip_minus_self_jit(positions_j, c6_j, cell_j, beta_j)
    print(f"    jit (V_recip - V_self) = {float(e_pr1[0]):+.6f}")

    # Isolated-atom convergence (JAX)
    pos_iso_j = _j(pos_iso_np)
    c6_iso_j = _j(c6_iso_np)
    cell_iso_j = _j(cell_iso_np)
    print("\n  Isolated-atom convergence (V_recip / V_self → 1):")
    print("      beta  | V_recip      | V_self       | (V_recip-V_self)/|V_self|")
    print("    " + "-" * 65)
    for bv in [0.2, 0.3, 0.5, 0.8, 1.0]:
        b_j = jnp.array([bv], dtype=jnp.float64)
        er = jax_pme_dispersion_reciprocal_space(
            pos_iso_j,
            c6_iso_j,
            cell_iso_j,
            b_j,
            mesh_dimensions=(64, 64, 64),
            spline_order=4,
        )
        es = jax_pme_dispersion_energy_corrections(c6_iso_j, b_j)
        rel = (float(er[0]) - float(es[0])) / abs(float(es[0]))
        print(
            f"     {bv:.2f}  | {float(er[0]):+11.4e} | {float(es[0]):+11.4e} | {rel:+.2e}"
        )

    # Batched evaluation (JAX)
    pos_batch_j = _j(pos_batch_np)
    c6_batch_j = _j(c6_batch_np)
    cell_batch_j = _j(cell_batch_np)
    batch_idx_j = _j(batch_idx_np, dtype=jnp.int32)
    beta_batch_j = _j(beta_batch_np)
    E_recip_batch_j = jax_pme_dispersion_reciprocal_space(
        positions=pos_batch_j,
        c6_coefficients=c6_batch_j,
        cell=cell_batch_j,
        beta=beta_batch_j,
        mesh_dimensions=(32, 32, 32),
        spline_order=4,
        batch_idx=batch_idx_j,
    )
    E_self_batch_j = jax_pme_dispersion_energy_corrections(
        c6_batch_j, beta_batch_j, batch_idx=batch_idx_j
    )
    print("\n  Batched evaluation (two argon supercells):")
    for s in range(2):
        print(
            f"    system {s}: V_recip = {float(E_recip_batch_j[s]):+.4e}, "
            f"V_self = {float(E_self_batch_j[s]):+.4e}"
        )


# %%
# Summary
# -------
# - ``pme_dispersion_reciprocal_space`` returns the FFT-based long-range
#   :math:`r^{-6}` lattice sum with B-spline spreading and gathering.
# - ``pme_dispersion_energy_corrections`` returns the closed-form
#   :math:`V_{\\text{self}} = -\\beta^6 \\sum_i C_{6,ii}/12`.
# - Both functions accept a ``batch_idx`` for evaluating many independent
#   systems in one call.
# - The PyTorch and JAX backends are interchangeable; the JAX path also
#   composes with ``jax.jit``.
# - The damped real-space term (``lj_pme_real_space``) and the unified
#   ``lj_pme()`` orchestrator land in subsequent PRs.
print("\nDone.")
