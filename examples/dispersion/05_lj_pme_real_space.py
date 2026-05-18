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
Real-Space LJ-PME (PR2): Damped Energy, Forces, and Virial
==========================================================

This example demonstrates the *real-space* component of Lennard-Jones PME
(LJ-PME) introduced in PR2. Combined with the reciprocal-space and
self-energy terms from PR1, this gives the total LJ-PME energy
(Wennberg et al., JCTC 2013):

.. math::

    V_{\\text{total}} = V_{\\text{real}} + V_{\\text{recip}} - V_{\\text{self}}.

The real-space term computes, for each neighbor pair (i, j) within a
cutoff,

.. math::

    V_{ij} = \\frac{C_{12,ij}}{r_{ij}^{12}}
             - \\frac{C_{6,ij} \\, g(\\beta r_{ij})}{r_{ij}^{6}},
    \\qquad
    g(x) = e^{-x^2}\\!\\left(1 + x^2 + \\tfrac{x^4}{2}\\right),

with geometric combination rules. The damping function :math:`g(\\beta r)`
is the Wennberg complement of the long-range PME term.

Two backends are demonstrated side-by-side from the same script:

* **PyTorch** — autograd-aware ``warp_custom_op`` wrappers.
* **JAX** — JIT-compatible ``jax_kernel`` wrappers (skipped if JAX is
  not installed).

In this example you will learn:

- Compute the damped real-space dispersion energy with
  ``lj_pme_real_space``
- Compute forces via the kernel and confirm they match autograd (Torch)
- Verify momentum conservation
- Compute the virial tensor (used for pressure / NPT)
- Combine ``V_real``, ``V_recip``, and ``V_self`` to recover the total
  LJ-PME energy
- See how the real- vs. reciprocal-space split is tuned by ``beta``
- Compose with ``jax.jit`` (JAX backend)

.. important::
    This script is intended as an API demonstration. Do not use it for
    performance benchmarking; refer to the ``benchmarks`` folder instead.
"""

# %%
# Setup and Imports
# -----------------

from __future__ import annotations

import math

import numpy as np

# %%
# Build a Backend-Neutral System Description
# ------------------------------------------
# A small argon supercell — neutral, single-element, dominated by
# dispersion, geometric combination is trivially exact.


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
    c12_np = np.full(positions_np.shape[0], 5.5e4, dtype=np.float64)
    cell_np = (np.eye(3, dtype=np.float64) * lattice_constant * n_cells)[None, :, :]
    return positions_np, c6_np, c12_np, cell_np


pos_np, c6_np, c12_np, cell_np = create_argon_fcc(n_cells=3)
cutoff = 9.0  # Å
beta_val = 0.35
mesh_dimensions = (32, 32, 32)
print(f"System: {pos_np.shape[0]}-atom argon FCC supercell")
print(f"  Cell length: {cell_np[0, 0, 0]:.2f} Å")
print(f"  Cutoff: {cutoff} Å,  beta: {beta_val} Å⁻¹")


# =============================================================================
# PyTorch Backend
# =============================================================================
print("\n" + "=" * 70)
print("PyTorch backend")
print("=" * 70)

try:
    import torch

    from nvalchemiops.torch.interactions.dispersion import (
        lj_pme_real_space,
        pme_dispersion_energy_corrections,
        pme_dispersion_reciprocal_space,
    )
    from nvalchemiops.torch.neighbors import neighbor_list as torch_nbr_list

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
    positions = torch.tensor(pos_np, dtype=DTYPE, device=torch_device)
    c6 = torch.tensor(c6_np, dtype=DTYPE, device=torch_device)
    c12 = torch.tensor(c12_np, dtype=DTYPE, device=torch_device)
    cell = torch.tensor(cell_np, dtype=DTYPE, device=torch_device)
    beta = torch.tensor([beta_val], dtype=DTYPE, device=torch_device)

    pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=torch_device)
    nbr_mat, num_nbrs, nbr_shifts = torch_nbr_list(
        positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=False
    )
    print(
        f"  Neighbor matrix: shape={tuple(nbr_mat.shape)},"
        f" mean neighbors/atom = {float(num_nbrs.float().mean()):.2f}"
    )

    # Energy + Forces + Virial
    energies, forces, virial = lj_pme_real_space(
        positions,
        c6,
        c12,
        cell,
        nbr_mat,
        nbr_shifts,
        beta=beta,
        cutoff=cutoff,
        num_neighbors=num_nbrs,
        compute_forces=True,
        compute_virial=True,
        half_neighbor_list=False,  # neighbor_list_fn returns a full list
    )
    V_real = energies.sum().item()
    F_max = torch.linalg.vector_norm(forces, dim=1).max().item()
    F_sum = forces.sum(dim=0).abs().max().item()
    print(f"\n  V_real           = {V_real:+.4f}")
    print(f"  Max force         = {F_max:.4e}")
    print(f"  |Σ F| (momentum)  = {F_sum:.2e}    (should be ~0)")
    print(f"  Virial trace tr(W)= {virial.diag().sum().item():+.4f}")

    # Autograd cross-check (Torch)
    positions_g = positions.clone().requires_grad_(True)
    e_g = lj_pme_real_space(
        positions_g,
        c6,
        c12,
        cell,
        nbr_mat,
        nbr_shifts,
        beta=beta,
        cutoff=cutoff,
        num_neighbors=num_nbrs,
        half_neighbor_list=False,
    )
    e_g.sum().backward()
    force_diff = (forces - (-positions_g.grad)).abs().max().item()
    print(
        f"  max |F_kernel - (-dE/dr)|         = {force_diff:.2e}  (autograd cross-check)"
    )

    # Total LJ-PME energy
    V_recip = pme_dispersion_reciprocal_space(
        positions, c6, cell, beta, mesh_dimensions=mesh_dimensions, spline_order=4
    ).item()
    V_self = pme_dispersion_energy_corrections(c6, beta).item()
    V_total = V_real + V_recip - V_self
    print(
        f"\n  LJ-PME total at β={beta_val}, cutoff={cutoff} Å, mesh={mesh_dimensions}:"
    )
    print(f"    V_real  = {V_real:+.4f}")
    print(f"    V_recip = {V_recip:+.4f}")
    print(f"    V_self  = {V_self:+.4f}")
    print(f"    V_total = {V_total:+.4f}")

    # Beta sweep (fixed parameters, not jointly tuned)
    print("\n  beta sweep with fixed cutoff and mesh=48³ (not jointly tuned):")
    print("      beta  | V_real      | V_recip     | V_self      | V_total")
    print("    " + "-" * 65)
    for bv in [0.25, 0.30, 0.35, 0.40]:
        b_t = torch.tensor([bv], dtype=DTYPE, device=torch_device)
        Vr = (
            lj_pme_real_space(
                positions,
                c6,
                c12,
                cell,
                nbr_mat,
                nbr_shifts,
                beta=b_t,
                cutoff=cutoff,
                num_neighbors=num_nbrs,
                half_neighbor_list=False,
            )
            .sum()
            .item()
        )
        Vk = pme_dispersion_reciprocal_space(
            positions, c6, cell, b_t, mesh_dimensions=(48, 48, 48), spline_order=4
        ).item()
        Vs = pme_dispersion_energy_corrections(c6, b_t).item()
        Vt = Vr + Vk - Vs
        print(
            f"     {bv:.2f}  | {Vr:+9.4f}   | {Vk:+9.4f}   | {Vs:+9.4f}   | {Vt:+9.4f}"
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
        lj_pme_real_space as jax_lj_pme_real_space,
    )
    from nvalchemiops.jax.interactions.dispersion import (
        pme_dispersion_energy_corrections as jax_pme_dispersion_energy_corrections,
    )
    from nvalchemiops.jax.interactions.dispersion import (
        pme_dispersion_reciprocal_space as jax_pme_dispersion_reciprocal_space,
    )
    from nvalchemiops.jax.neighbors import neighbor_list as jax_nbr_list

    HAS_JAX = True
except ImportError as exc:
    HAS_JAX = False
    print(f"  JAX backend unavailable ({exc}); skipping.")

if HAS_JAX:
    print(f"  JAX devices: {jax.devices()}")

    positions_j = jnp.array(pos_np, dtype=jnp.float64)
    c6_j = jnp.array(c6_np, dtype=jnp.float64)
    c12_j = jnp.array(c12_np, dtype=jnp.float64)
    cell_j = jnp.array(cell_np, dtype=jnp.float64)
    beta_j = jnp.array([beta_val], dtype=jnp.float64)

    pbc_j = jnp.array([[True, True, True]], dtype=jnp.bool_)
    nbr_mat_j, num_nbrs_j, nbr_shifts_j = jax_nbr_list(
        positions_j, cutoff, cell=cell_j, pbc=pbc_j, return_neighbor_list=False
    )
    print(
        f"  Neighbor matrix: shape={tuple(nbr_mat_j.shape)},"
        f" mean neighbors/atom = {float(num_nbrs_j.astype(jnp.float64).mean()):.2f}"
    )

    # Energy + Forces + Virial
    energies_j, forces_j, virial_j = jax_lj_pme_real_space(
        positions_j,
        c6_j,
        c12_j,
        cell_j,
        nbr_mat_j,
        nbr_shifts_j,
        beta=beta_j,
        cutoff=cutoff,
        num_neighbors=num_nbrs_j,
        compute_forces=True,
        compute_virial=True,
        half_neighbor_list=False,
    )
    V_real_j = float(energies_j.sum())
    F_max_j = float(jnp.linalg.norm(forces_j, axis=1).max())
    F_sum_j = float(jnp.abs(forces_j.sum(axis=0)).max())
    print(f"\n  V_real           = {V_real_j:+.4f}")
    print(f"  Max force         = {F_max_j:.4e}")
    print(f"  |Σ F| (momentum)  = {F_sum_j:.2e}")
    print(f"  Virial trace tr(W)= {float(virial_j.diagonal().sum()):+.4f}")

    # JIT-compiled total LJ-PME energy
    @jax.jit
    def lj_pme_total_energy_jit(positions, c6, c12, cell, nm, ns, nn, beta):
        """JIT-compiled V_real + V_recip - V_self for one system."""
        e_real = jax_lj_pme_real_space(
            positions,
            c6,
            c12,
            cell,
            nm,
            ns,
            beta=beta,
            cutoff=cutoff,
            num_neighbors=nn,
            half_neighbor_list=False,
        )
        e_recip = jax_pme_dispersion_reciprocal_space(
            positions, c6, cell, beta, mesh_dimensions=mesh_dimensions, spline_order=4
        )
        e_self = jax_pme_dispersion_energy_corrections(c6, beta)
        return e_real.sum() + e_recip[0] - e_self[0]

    V_total_j = float(
        lj_pme_total_energy_jit(
            positions_j,
            c6_j,
            c12_j,
            cell_j,
            nbr_mat_j,
            nbr_shifts_j,
            num_nbrs_j,
            beta_j,
        )
    )
    print(f"\n  jit V_total = V_real + V_recip - V_self = {V_total_j:+.4f}")


# %%
# Damping Function Behavior (backend-independent)
# -----------------------------------------------
# ``g(βr)`` smoothly transitions from 1 at r=0 to 0 at large r,
# truncating dispersion before the cutoff:

print("\nDamping function g(beta·r) = exp(-x²)(1 + x² + x⁴/2):")
print("    r (Å) | β·r   | g(β·r)")
print("  " + "-" * 40)
for r in [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]:
    x = beta_val * r
    g = math.exp(-(x * x)) * (1.0 + x * x + 0.5 * x * x * x * x)
    print(f"   {r:5.1f}  | {x:5.2f} | {g:.4f}")

# %%
# Summary
# -------
# - ``lj_pme_real_space`` computes the damped pair sum
#   :math:`C_{12}/r^{12} - C_6\\, g(\\beta r)/r^6` over neighbor pairs and
#   returns per-atom energies, plus optional forces and the 3×3 virial.
# - PyTorch: kernel forces match autograd derivatives to machine
#   precision.
# - JAX: composes with ``jax.jit`` for end-to-end JIT-compiled
#   evaluation.
# - Combined with ``pme_dispersion_reciprocal_space`` (PR1) and
#   ``pme_dispersion_energy_corrections`` (PR1), it produces the total
#   LJ-PME energy.
# - At a jointly-tuned (β, cutoff, mesh) the total is independent of β;
#   moving β alone shifts mass between the two sums and exposes the
#   per-term truncation error.
# - The top-level ``lj_pme()`` wrapper with automatic β/mesh estimation
#   arrives in PR3.
print("\nDone.")
