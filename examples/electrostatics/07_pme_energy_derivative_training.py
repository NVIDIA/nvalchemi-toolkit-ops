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
Energy-Derivative Training Contract (Forces, Stress, Charge Gradients)
======================================================================

This example demonstrates the recommended way to obtain forces, virial/stress,
and charge gradients from PME (and Ewald) for machine-learning interatomic
potential (MLIP) training: **energy is the only differentiable output, and all
derivatives are taken from it with** ``torch.autograd.grad``.

The legacy ``compute_forces`` / ``compute_virial`` / ``compute_charge_gradients``
/ ``hybrid_forces`` flags on ``particle_mesh_ewald`` and ``ewald_summation`` are
deprecated and emit a ``DeprecationWarning``. They remain available for
compatibility in v0.4.0, so this example also checks that the autograd results
match the legacy direct outputs, which is the migration check.

In this example you will learn:

- Forces from energy autograd: ``F = -grad(E.sum(), positions)``
- Force-loss training (double-backward) with ``create_graph=True``
- Geometry-dependent charges ``q(R)``: the full force includes the
  ``dE/dq . dq/dR`` charge-model chain-rule term
- Strain-first virial/stress and stress-loss training
- Charge gradients from energy autograd: ``dE/dq = grad(E.sum(), charges)``

.. important::
    This script is intended as an API demonstration. Do not use this script
    for performance benchmarking; refer to the `benchmarks` folder instead.
"""

# %%
# Setup and Imports
# -----------------
# We use ``particle_mesh_ewald`` throughout; ``ewald_summation`` follows the
# exact same energy-derivative contract.

from __future__ import annotations

import warnings

import numpy as np
import torch

from nvalchemiops.torch.interactions.electrostatics import particle_mesh_ewald
from nvalchemiops.torch.neighbors import neighbor_list as neighbor_list_fn

# %%
# Configure Device
# ----------------

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA device")
    print(f"  {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

dtype = torch.float64

# %%
# Create a NaCl Crystal System
# ----------------------------
# A small NaCl rock-salt supercell (2x2x2 = 16 atoms) is enough to demonstrate
# every derivative path. Charges use the same floating dtype as the geometry.


def create_nacl_system(n_cells: int = 2, lattice_constant: float = 5.64):
    """Create a NaCl crystal supercell."""
    base_positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    base_charges = np.array([1.0, -1.0])

    positions = []
    charges = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                offset = np.array([i, j, k])
                for pos, charge in zip(base_positions, base_charges):
                    positions.append((pos + offset) * lattice_constant)
                    charges.append(charge)

    positions = torch.tensor(np.array(positions), dtype=dtype, device=device)
    charges = torch.tensor(np.array(charges), dtype=dtype, device=device)
    cell = torch.eye(3, dtype=dtype, device=device) * lattice_constant * n_cells
    cell = cell.unsqueeze(0)
    pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
    return positions, charges, cell, pbc


positions, charges, cell, pbc = create_nacl_system(n_cells=2)

# Rattle the perfect lattice so the per-atom forces are nonzero -- otherwise the
# centrosymmetric crystal gives zero forces and the autograd-vs-direct force
# check below would compare 0 to 0.
torch.manual_seed(0)
positions = positions + 0.05 * torch.randn_like(positions)

print(f"\nSystem: {len(positions)} atoms NaCl crystal (rattled)")

# Build the real-space neighbor list once; reuse it for every call below.
neighbor_list, neighbor_ptr, neighbor_shifts = neighbor_list_fn(
    positions, 8.0, cell=cell, pbc=pbc, return_neighbor_list=True
)

# Fixed PME parameters keep the example deterministic and fast.
pme_kwargs = dict(
    alpha=0.35,
    mesh_dimensions=(32, 32, 32),
    spline_order=4,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
)

# %%
# Forces From Energy Autograd
# ---------------------------
# With no deprecated flag, the call returns the per-atom energy only. The full
# force is the negative gradient of the total energy w.r.t. positions.

positions_f = positions.detach().requires_grad_(True)
energy = particle_mesh_ewald(positions_f, charges, cell, **pme_kwargs)
print(f"\nenergy shape: {tuple(energy.shape)}  (per-atom)")

forces = -torch.autograd.grad(energy.sum(), positions_f)[0]
print(f"forces shape: {tuple(forces.shape)}")
print(f"max force magnitude: {forces.norm(dim=1).max().item():.6f}")

# %%
# Migration check: the autograd force equals the legacy ``compute_forces=True``
# direct output (the deprecated flag still works, with a ``DeprecationWarning``).

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*compute_forces.*",
    )
    _, forces_direct = particle_mesh_ewald(
        positions, charges, cell, compute_forces=True, **pme_kwargs
    )

force_diff = (forces - forces_direct).abs().max().item()
print(f"max |autograd force - direct force|: {force_diff:.2e}")
if force_diff >= 1e-6:
    raise RuntimeError(f"autograd force does not match direct force: {force_diff:.2e}")

# %%
# Force-Loss Training (Double-Backward)
# -------------------------------------
# To train on a force loss, build the force with ``create_graph=True`` so the
# subsequent ``loss.backward()`` can differentiate through the force. We use a
# trivial linear charge model ``q(R)`` to stand in for a learned charge head.

torch.manual_seed(0)
weight = torch.randn(3, dtype=dtype, device=device, requires_grad=True)


def charge_model(pos: torch.Tensor) -> torch.Tensor:
    """Toy geometry-dependent charge model with enforced neutrality."""
    raw = pos @ weight
    return raw - raw.mean()


positions_q = positions.detach().requires_grad_(True)
charges_qr = charge_model(positions_q)  # kept in the graph -> full q(R) force

energy = particle_mesh_ewald(positions_q, charges_qr, cell, **pme_kwargs)
forces_qr = -torch.autograd.grad(energy.sum(), positions_q, create_graph=True)[0]

target_forces = torch.zeros_like(forces_qr)
force_loss = (forces_qr - target_forces).pow(2).sum()
force_loss.backward()  # differentiates through the force construction

print(f"\nq(R) force-loss backward OK; weight.grad shape: {tuple(weight.grad.shape)}")
if not torch.isfinite(weight.grad).all():
    raise RuntimeError("q(R) force-loss produced non-finite parameter gradients")

# %%
# Because ``charges = charge_model(positions)`` stays connected to ``positions``,
# the autograd force includes both the fixed-charge term and the
# ``dE/dq . dq/dR`` chain-rule term. The legacy ``compute_forces=True`` output is
# only the fixed-charge partial and does not include the charge-model term --
# this is the central reason direct force output on the full API is deprecated.

# %%
# Strain-First Virial and Stress
# ------------------------------
# ``strain`` is not a PME argument. Build a differentiable strain tensor, deform
# positions and cell by ``I + strain``, and let autograd map gradients back to
# strain. The virial is ``W = -dE/d(strain)`` and tensile-positive stress is
# ``dE/d(strain) / V``; see ``conventions.md`` for the project-wide sign.

num_systems = cell.shape[0]
positions_s = positions.detach().requires_grad_(True)
strain = torch.zeros(num_systems, 3, 3, device=device, dtype=dtype, requires_grad=True)
eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
deform = eye + strain

# Single system: every atom maps to system 0.
batch_idx = torch.zeros(positions_s.shape[0], dtype=torch.int32, device=device)
positions_def = torch.einsum("ni,nij->nj", positions_s, deform[batch_idx])
cell_def = torch.einsum("bij,bjk->bik", cell, deform)

energy = particle_mesh_ewald(positions_def, charges, cell_def, **pme_kwargs)
grad_strain = torch.autograd.grad(energy.sum(), strain)[0]
virial = -grad_strain
volume = torch.abs(torch.linalg.det(cell_def))
stress = grad_strain / volume[:, None, None]  # tensile-positive Cauchy

print(f"\nvirial shape: {tuple(virial.shape)}")
print(f"stress shape: {tuple(stress.shape)}")

# %%
# Migration check: the strain-first virial equals the legacy
# ``compute_virial=True`` direct virial (both are ``-dE/d(strain)``).

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*compute_virial.*",
    )
    _, virial_direct = particle_mesh_ewald(
        positions, charges, cell, compute_virial=True, **pme_kwargs
    )

virial_diff = (virial - virial_direct).abs().max().item()
print(f"max |strain-first virial - direct virial|: {virial_diff:.2e}")
if virial_diff >= 1e-5:
    raise RuntimeError(
        f"strain-first virial does not match direct virial: {virial_diff:.2e}"
    )

# %%
# Stress-Loss Training (Double-Backward)
# --------------------------------------
# Stress training uses the same strain-first recipe with ``create_graph=True``,
# so the stress loss back-propagates to model parameters.

weight_s = torch.randn(3, dtype=dtype, device=device, requires_grad=True)
positions_s = positions.detach().requires_grad_(True)
strain = torch.zeros(num_systems, 3, 3, device=device, dtype=dtype, requires_grad=True)
deform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) + strain
positions_def = torch.einsum("ni,nij->nj", positions_s, deform[batch_idx])
cell_def = torch.einsum("bij,bjk->bik", cell, deform)

charges_qr = positions_def @ weight_s
charges_qr = charges_qr - charges_qr.mean()

energy = particle_mesh_ewald(positions_def, charges_qr, cell_def, **pme_kwargs)
grad_strain = torch.autograd.grad(energy.sum(), strain, create_graph=True)[0]
virial = -grad_strain
volume = torch.abs(torch.linalg.det(cell_def))
stress = grad_strain / volume[:, None, None]

stress_loss = stress.pow(2).sum()
stress_loss.backward()

stress_grad_ok = bool(torch.isfinite(weight_s.grad).all())
print(f"\nstress-loss backward OK; weight_s.grad finite: {stress_grad_ok}")
if not stress_grad_ok:
    raise RuntimeError("stress-loss produced non-finite parameter gradients")

# %%
# Combined Energy + Force + Stress Loss (Performance Tip)
# -------------------------------------------------------
# When a single loss mixes energy, forces, AND stress, take the forces and the
# virial from **one** ``torch.autograd.grad`` call over both ``positions`` and
# ``strain`` -- NOT two separate calls. Each ``create_graph=True`` ``grad`` call
# builds its own first-derivative graph node, and ``loss.backward()`` then runs the
# (O(K*N)) reciprocal second-derivative once per node. Fusing the two into a single
# call avoids duplicate reciprocal double-backward work; the gradients are
# bit-identical to the two-call form.
weight_m = torch.randn(3, dtype=dtype, device=device, requires_grad=True)
positions_m = positions.detach().requires_grad_(True)
strain_m = torch.zeros(
    num_systems, 3, 3, device=device, dtype=dtype, requires_grad=True
)
deform_m = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) + strain_m
positions_md = torch.einsum("ni,nij->nj", positions_m, deform_m[batch_idx])
cell_md = torch.einsum("bij,bjk->bik", cell, deform_m)
charges_m = positions_md @ weight_m
charges_m = charges_m - charges_m.mean()

energy = particle_mesh_ewald(positions_md, charges_m, cell_md, **pme_kwargs)
# ONE combined grad call -> one double-backward (do this instead of separate
# ``grad(E, positions)`` and ``grad(E, strain)`` calls):
grad_pos, grad_strain = torch.autograd.grad(
    energy.sum(), (positions_m, strain_m), create_graph=True
)
# ``positions_m`` are the undeformed reference coordinates, so ``forces_m`` are
# reference-frame forces. Differentiate with respect to ``positions_md`` instead
# when training against deformed-coordinate force targets.
forces_m = -grad_pos
virial_m = -grad_strain
volume_m = torch.abs(torch.linalg.det(cell_md))
stress_m = grad_strain / volume_m[:, None, None]

mixed_loss = energy.sum() + forces_m.pow(2).sum() + stress_m.pow(2).sum()
mixed_loss.backward()
mixed_grad_ok = bool(torch.isfinite(weight_m.grad).all())
print(f"\nmixed E+F+stress backward OK; weight_m.grad finite: {mixed_grad_ok}")
if not mixed_grad_ok:
    raise RuntimeError("mixed-loss produced non-finite parameter gradients")

# %%
# Charge Gradients From Energy Autograd
# -------------------------------------
# ``dE/dq`` is an ordinary gradient of the energy w.r.t. charges.

charges_g = charges.detach().requires_grad_(True)
energy = particle_mesh_ewald(positions, charges_g, cell, **pme_kwargs)
charge_grad = torch.autograd.grad(energy.sum(), charges_g)[0]
print(f"\ncharge gradient shape: {tuple(charge_grad.shape)}")

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*compute_forces.*",
    )
    _, _, charge_grad_direct = particle_mesh_ewald(
        positions,
        charges,
        cell,
        compute_forces=True,
        compute_charge_gradients=True,
        **pme_kwargs,
    )

cg_diff = (charge_grad - charge_grad_direct).abs().max().item()
print(f"max |autograd dE/dq - direct dE/dq|: {cg_diff:.2e}")
if cg_diff >= 1e-6:
    raise RuntimeError(f"autograd dE/dq does not match direct dE/dq: {cg_diff:.2e}")

# %%
# Summary
# -------
# This example demonstrated the energy-derivative training contract:
#
# 1. **Forces** -- ``F = -torch.autograd.grad(E.sum(), positions)[0]``; with
#    ``create_graph=True`` for force-loss training.
# 2. **q(R) forces** -- keep ``charges = charge_model(positions)`` in the graph so
#    the full ``dE/dR`` includes the ``dE/dq . dq/dR`` charge-model term.
# 3. **Virial / stress** -- strain-first: deform positions and cell by
#    ``I + strain``, then ``virial = -grad(E.sum(), strain)`` and
#    ``stress = grad(E.sum(), strain) / volume[:, None, None]``.
# 4. **Charge gradients** -- ``dE/dq = torch.autograd.grad(E.sum(), charges)[0]``.
# 5. **Combined E + F + stress loss** -- take forces and virial from a single
#    ``torch.autograd.grad(E.sum(), (positions, strain), create_graph=True)`` call,
#    not two separate calls, so the reciprocal double-backward runs once.
#
# Each autograd result matched the corresponding (deprecated) direct kernel
# output, confirming the migration is numerically exact.

print("\nEnergy-derivative training contract example complete!")
