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
Unified LJ-PME (PR3): Top-level ``lj_pme()`` with Auto Parameter Estimation
===========================================================================

This example demonstrates the unified ``lj_pme()`` top-level
orchestrator that combines the three LJ-PME components from PR1 and PR2:

.. math::

    V_{\\text{total}} = V_{\\text{real}} + V_{\\text{recip}} - V_{\\text{self}},

with automatic estimation of the dispersion splitting parameter β and
the FFT mesh dimensions (Wennberg et al., JCTC 2013; GROMACS-style
matched-tail criterion). Pass an ``accuracy`` and a cutoff and the
estimator picks β and mesh that put both tails below that threshold.

In this example you will learn:

- Call ``lj_pme()`` with default parameter estimation
- Compute forces alongside the energy
- Inspect the parameters chosen by
  ``estimate_pme_dispersion_parameters``
- See β-balance: total energy stays consistent as β varies, provided
  cutoff and mesh are jointly tuned
- Compose with ``jax.jit`` (JAX backend)

Two backends are demonstrated side-by-side from the same script:

* **PyTorch** — autograd-aware wrappers
* **JAX** — JIT-compatible wrappers (skipped if JAX is not installed)

.. important::
    Demonstrates the API. Not a performance benchmark.
"""

# %%
# Setup and Imports
# -----------------

from __future__ import annotations

import numpy as np


def create_argon_fcc(n_cells: int = 3, lattice_constant: float = 5.26):
    """Argon FCC supercell as NumPy arrays."""
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
    cell_np = np.eye(3, dtype=np.float64) * lattice_constant * n_cells
    return positions_np, c6_np, c12_np, cell_np


pos_np, c6_np, c12_np, cell_np = create_argon_fcc(n_cells=3)
cutoff = 9.0  # Å
print(f"System: {pos_np.shape[0]}-atom argon FCC supercell")
print(f"  Cell length: {cell_np[0, 0]:.2f} Å")
print(f"  Real-space cutoff: {cutoff} Å")


# =============================================================================
# PyTorch Backend
# =============================================================================
print("\n" + "=" * 70)
print("PyTorch backend")
print("=" * 70)

try:
    import torch

    from nvalchemiops.torch.interactions.dispersion import (
        estimate_pme_dispersion_parameters,
        lj_pme,
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
    pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=torch_device)

    # %%
    # Parameter Estimation
    # --------------------
    # Pick the cutoff (real-space tail), and the estimator chooses β and
    # the mesh to match the same accuracy threshold on both sides.
    params = estimate_pme_dispersion_parameters(cell, cutoff=cutoff, accuracy=1e-3)
    print(
        f"\n  Auto-estimated parameters (accuracy=1e-3):\n"
        f"    cutoff = {params.cutoff} Å\n"
        f"    beta   = {float(params.beta[0]):.4f} Å⁻¹\n"
        f"    mesh   = {params.mesh_dimensions}\n"
        f"    mesh_spacing = {[f'{float(s):.2f}' for s in params.mesh_spacing[0]]} Å"
    )

    # %%
    # Energy via the Unified API
    # --------------------------
    # ``lj_pme()`` builds the matching neighbor list for the chosen
    # cutoff and runs the three LJ-PME components. With no β/cutoff/mesh
    # supplied it falls back to the joint estimator.

    nbr_mat, num_nbrs, nbr_shifts = torch_nbr_list(
        positions,
        params.cutoff,
        cell=cell.unsqueeze(0),
        pbc=pbc,
        return_neighbor_list=False,
    )

    E = lj_pme(
        positions,
        c6,
        c12,
        cell,
        nbr_mat,
        nbr_shifts,
        num_neighbors=num_nbrs,
    )
    print(f"\n  V_total (auto-parameters): {E.item():+.4f}")

    # %%
    # Energy + Forces
    # ---------------
    E, F = lj_pme(
        positions,
        c6,
        c12,
        cell,
        nbr_mat,
        nbr_shifts,
        num_neighbors=num_nbrs,
        compute_forces=True,
    )
    F_max = torch.linalg.vector_norm(F, dim=1).max().item()
    F_sum = F.sum(dim=0).abs().max().item()
    print(f"  V_total = {E.item():+.4f}")
    print(f"  Max |F| = {F_max:.4e}")
    print(f"  |Σ F|   = {F_sum:.2e}    (should be ~0)")

    # %%
    # β-balance with Jointly-Tuned Parameters
    # ----------------------------------------
    # Sweep cutoff; for each cutoff the estimator picks the matching β
    # and mesh. V_total should be approximately invariant (the residual
    # depends on the chosen accuracy and the system size).
    print("\n  β-balance sweep (accuracy=1e-5, joint estimator):")
    print("      cutoff | beta    | mesh         | V_total")
    print("    " + "-" * 55)
    for cut in [7.0, 9.0, 12.0]:
        params_i = estimate_pme_dispersion_parameters(cell, cutoff=cut, accuracy=1e-5)
        nm_i, nn_i, ns_i = torch_nbr_list(
            positions, cut, cell=cell.unsqueeze(0), pbc=pbc, return_neighbor_list=False
        )
        E_i = lj_pme(
            positions,
            c6,
            c12,
            cell,
            nm_i,
            ns_i,
            num_neighbors=nn_i,
            beta=params_i.beta,
            cutoff=params_i.cutoff,
            mesh_dimensions=params_i.mesh_dimensions,
            accuracy=1e-5,
        )
        print(
            f"      {cut:5.1f}  | {float(params_i.beta[0]):.4f}  | {str(params_i.mesh_dimensions):12s} | {E_i.item():+.4f}"
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
        estimate_pme_dispersion_parameters as jax_estimate_pme_dispersion_parameters,
    )
    from nvalchemiops.jax.interactions.dispersion import (
        lj_pme as jax_lj_pme,
    )
    from nvalchemiops.jax.neighbors import neighbor_list as jax_nbr_list

    HAS_JAX = True
except ImportError as exc:
    HAS_JAX = False
    print(f"  JAX backend unavailable ({exc}); skipping.")

if HAS_JAX:
    try:
        print(f"  JAX devices: {jax.devices()}")
    except (NameError, AttributeError):  # Exception:
        pass

    positions_j = jnp.array(pos_np, dtype=jnp.float64)
    c6_j = jnp.array(c6_np, dtype=jnp.float64)
    c12_j = jnp.array(c12_np, dtype=jnp.float64)
    cell_j = jnp.array(cell_np, dtype=jnp.float64)
    pbc_j = jnp.array([[True, True, True]], dtype=jnp.bool_)

    params_j = jax_estimate_pme_dispersion_parameters(
        cell_j, cutoff=cutoff, accuracy=1e-3
    )
    print(
        f"\n  Auto-estimated parameters (accuracy=1e-3):\n"
        f"    cutoff = {params_j.cutoff} Å\n"
        f"    beta   = {float(params_j.beta[0]):.4f} Å⁻¹\n"
        f"    mesh   = {params_j.mesh_dimensions}"
    )

    nbr_mat_j, num_nbrs_j, nbr_shifts_j = jax_nbr_list(
        positions_j,
        params_j.cutoff,
        cell=cell_j[None],
        pbc=pbc_j,
        return_neighbor_list=False,
    )

    E_j = jax_lj_pme(
        positions_j,
        c6_j,
        c12_j,
        cell_j,
        nbr_mat_j,
        nbr_shifts_j,
        num_neighbors=num_nbrs_j,
    )
    print(f"\n  V_total (auto-parameters): {float(E_j[0]):+.4f}")

    # JIT compilation
    @jax.jit
    def lj_pme_energy_jit(positions, c6, c12, cell, nm, ns, nn):
        return jax_lj_pme(
            positions,
            c6,
            c12,
            cell,
            nm,
            ns,
            num_neighbors=nn,
            beta=0.3723,
            cutoff=cutoff,
            mesh_dimensions=(16, 16, 16),
        )[0]

    E_jit = float(
        lj_pme_energy_jit(
            positions_j, c6_j, c12_j, cell_j, nbr_mat_j, nbr_shifts_j, num_nbrs_j
        )
    )
    print(f"  jit-compiled V_total:     {E_jit:+.4f}")


# %%
# Summary
# -------
# - ``lj_pme()`` is the recommended entry point: pass a system + cutoff,
#   get the total LJ-PME energy with optional forces.
# - β, mesh dimensions, and the real-space cutoff can be supplied
#   explicitly or auto-estimated jointly from ``accuracy``.
# - ``estimate_pme_dispersion_parameters()`` exposes the estimator so
#   users can pre-compute and reuse the parameters in MD loops.
# - The PyTorch path is autograd-aware; the JAX path composes with
#   ``jax.jit``.
print("\nDone.")
