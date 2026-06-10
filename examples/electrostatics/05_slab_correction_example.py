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
2D Slab Correction for Ewald and PME
====================================

This example demonstrates how to apply the Yeh-Berkowitz / Ballenegger
two-dimensional slab correction to Ewald and Particle Mesh Ewald (PME)
electrostatics. The correction is used for slab-like systems with two periodic
directions and one non-periodic direction, such as interfaces with vacuum
padding.

In this example you will learn:

- How to run ``ewald_summation(..., slab_correction=True)``
- How to run ``particle_mesh_ewald(..., slab_correction=True)``
- How to pass slab periodicity with a boolean ``pbc`` tensor
- How to compute the standalone slab correction with ``compute_slab_correction``
- How the standalone correction equals the integrated Ewald energy/force delta
- How to derive slab forces from energy autograd for training
- How to compose total Ewald and PME component workflows with slab
- How triclinic slab cells use the normal to the periodic plane

.. important::
    This script is intended as an API demonstration. Do not use this script
    for performance benchmarking; refer to the `benchmarks` folder instead.
"""

# %%
# Setup and Imports
# -----------------
# The slab correction is available through the high-level Ewald and PME APIs,
# and as a standalone helper. The standalone helper is useful for debugging,
# validation, and adding the correction when composing total component workflows.

from __future__ import annotations

import warnings

import numpy as np
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    compute_slab_correction,
    ewald_real_space,
    ewald_reciprocal_space,
    ewald_summation,
    generate_k_vectors_ewald_summation,
    particle_mesh_ewald,
    pme_reciprocal_space,
)
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


def _legacy_direct_output_call(function, *args, **kwargs):
    """Call a deprecated direct-output path used for explicit migration checks."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="The direct-output flags.*",
        )
        return function(*args, **kwargs)


def _format_array(array: torch.Tensor) -> str:
    """Format small arrays consistently for gallery text output."""
    return np.array2string(
        array.detach().cpu().numpy(),
        precision=6,
        suppress_small=False,
    )


# %%
# Create a Small Slab System
# --------------------------
# We use a two-ion CsCl-like system in a cell with a long z direction. The
# long cell vector represents vacuum padding normal to the slab.
#
# ``pbc_slab`` marks x and y as periodic and z as non-periodic. Batched slab
# simulations should pass an explicit ``(B, 3)`` tensor so each system carries
# its own slab geometry.


def create_cscl_slab_system() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Create a small T/T/F slab system with vacuum along z."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        dtype=torch.float64,
        device=device,
    )
    charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
    cell = torch.diag(
        torch.tensor([10.0, 10.0, 30.0], dtype=torch.float64, device=device)
    ).unsqueeze(0)
    pbc_slab = torch.tensor([[True, True, False]], dtype=torch.bool, device=device)
    return positions, charges, cell, pbc_slab


positions, charges, cell, pbc_slab = create_cscl_slab_system()

print("Slab system:")
print(f"  Number of atoms: {positions.shape[0]}")
print(f"  Cell rows:\n{_format_array(cell[0])}")
print(f"  Slab pbc: {_format_array(pbc_slab[0])}")
print(f"  Total charge: {charges.sum().item():.1f}")

# %%
# Build the Real-Space Neighbor List
# ----------------------------------
# The neighbor list controls real-space periodic images. For this slab setup,
# use the same T/T/F periodicity and a cell with enough vacuum along z.

alpha = 0.3
real_space_cutoff = 5.0
k_cutoff = 2.5
mesh_dimensions = (16, 16, 16)
alpha_tensor = torch.tensor([alpha], dtype=torch.float64, device=device)

neighbor_list, neighbor_ptr, neighbor_shifts = neighbor_list_fn(
    positions,
    real_space_cutoff,
    cell=cell,
    pbc=pbc_slab,
    return_neighbor_list=True,
)

print("\nNeighbor list:")
print(f"  Number of neighbor entries: {neighbor_list.shape[1]}")
print(f"  Real-space cutoff: {real_space_cutoff:.1f} Å")

# %%
# Standard 3D Ewald (Legacy Direct-Output Baseline)
# -------------------------------------------------
# First compute the uncorrected 3D-periodic Ewald result. This is the quantity
# that will receive the slab correction. This section intentionally uses
# deprecated full-API direct-output flags only to build a compatibility baseline
# for the checks below.

energies_3d, forces_3d, charge_grads_3d, virial_3d = _legacy_direct_output_call(
    ewald_summation,
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha,
    k_cutoff=k_cutoff,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
    compute_charge_gradients=True,
    compute_virial=True,
)

print("\nStandard 3D Ewald:")
print(f"  Total energy: {energies_3d.sum().item(): .8f}")
print(f"  Max force magnitude: {forces_3d.norm(dim=1).max().item(): .8f}")
print(f"  Charge gradients: {_format_array(charge_grads_3d)}")
print(f"  Virial trace: {torch.trace(virial_3d[0]).item(): .8f}")

# %%
# Ewald with Slab Correction (Legacy Direct-Output Check)
# ------------------------------------------------------
# Set ``slab_correction=True`` and pass the slab periodicity. The output tuple
# follows the same ordering as ordinary Ewald: energies, forces, charge
# gradients, and virial when all optional quantities are requested. For training
# code, copy the energy-autograd section immediately below instead.

energies_slab, forces_slab, charge_grads_slab, virial_slab = _legacy_direct_output_call(
    ewald_summation,
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha,
    k_cutoff=k_cutoff,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
    compute_charge_gradients=True,
    compute_virial=True,
    pbc=pbc_slab,
    slab_correction=True,
)

print("\nEwald with slab correction:")
print(f"  Total energy: {energies_slab.sum().item(): .8f}")
print(f"  Energy delta: {(energies_slab - energies_3d).sum().item(): .8f}")
print(f"  Max force magnitude: {forces_slab.norm(dim=1).max().item(): .8f}")
print(f"  Charge gradients: {_format_array(charge_grads_slab)}")
print(f"  Virial trace: {torch.trace(virial_slab[0]).item(): .8f}")

# %%
# Energy-Autograd Slab Forces
# ---------------------------
# For differentiable training, call the full API without direct-output flags and
# derive forces from the returned per-atom energies. The direct-output calls
# above are legacy compatibility checks.

positions_ag = positions.detach().requires_grad_(True)
ewald_energy_ag = ewald_summation(
    positions=positions_ag,
    charges=charges,
    cell=cell,
    alpha=alpha,
    k_cutoff=k_cutoff,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    pbc=pbc_slab,
    slab_correction=True,
)
ewald_forces_ag = -torch.autograd.grad(ewald_energy_ag.sum(), positions_ag)[0]
print("\nEwald slab energy-autograd force check:")
print(
    f"  Max force delta vs legacy direct: {(ewald_forces_ag - forces_slab).abs().max().item():.2e}"
)

# %%
# Standalone Slab Correction
# --------------------------
# The same correction can be computed directly. The standalone result equals
# the difference between the slab-corrected and uncorrected Ewald outputs.

correction_energy, correction_forces, correction_charge_grads, correction_virial = (
    compute_slab_correction(
        positions=positions,
        charges=charges,
        cell=cell,
        pbc=pbc_slab,
        compute_forces=True,
        compute_charge_gradients=True,
        compute_virial=True,
    )
)

energy_delta_error = torch.max(
    torch.abs((energies_slab - energies_3d) - correction_energy)
)
force_delta_error = torch.max(torch.abs((forces_slab - forces_3d) - correction_forces))
charge_grad_delta_error = torch.max(
    torch.abs((charge_grads_slab - charge_grads_3d) - correction_charge_grads)
)
virial_delta_error = torch.max(torch.abs((virial_slab - virial_3d) - correction_virial))

print("\nStandalone correction:")
print(f"  Correction energy: {correction_energy.sum().item(): .8f}")
print(f"  Max energy delta error: {energy_delta_error.item():.2e}")
print(f"  Max force delta error: {force_delta_error.item():.2e}")
print(f"  Max charge-gradient delta error: {charge_grad_delta_error.item():.2e}")
print(f"  Max virial delta error: {virial_delta_error.item():.2e}")

# %%
# Total Ewald Component Composition
# ---------------------------------
# If you need the Ewald real-space, Ewald reciprocal-space, and slab terms
# separately, compute the three pieces explicitly and add matching outputs.

ewald_component_real_energy, ewald_component_real_forces = ewald_real_space(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha_tensor,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
)
ewald_k_vectors = generate_k_vectors_ewald_summation(cell.detach(), k_cutoff)
ewald_component_reciprocal_energy, ewald_component_reciprocal_forces = (
    ewald_reciprocal_space(
        positions=positions,
        charges=charges,
        cell=cell,
        k_vectors=ewald_k_vectors,
        alpha=alpha_tensor,
        compute_forces=True,
    )
)
ewald_component_slab_energy, ewald_component_slab_forces = compute_slab_correction(
    positions=positions,
    charges=charges,
    cell=cell,
    pbc=pbc_slab,
    compute_forces=True,
)

total_ewald_energy = (
    ewald_component_real_energy
    + ewald_component_reciprocal_energy
    + ewald_component_slab_energy
)
total_ewald_forces = (
    ewald_component_real_forces
    + ewald_component_reciprocal_forces
    + ewald_component_slab_forces
)

print("\nTotal Ewald composition from components:")
print(
    f"  Max energy error: {torch.max(torch.abs(total_ewald_energy - energies_slab)).item():.2e}"
)
print(
    f"  Max force error: {torch.max(torch.abs(total_ewald_forces - forces_slab)).item():.2e}"
)

# %%
# PME with Slab Correction (Legacy Direct-Output Check)
# ----------------------------------------------------
# Full PME accepts the same slab correction arguments as Ewald. The reciprocal
# PME component itself remains a 3D-periodic reciprocal-space calculation; the
# slab term is added by the high-level ``particle_mesh_ewald`` wrapper. This
# direct-output tuple is a legacy compatibility check; the following section is
# the training recipe.

pme_3d_energy, pme_3d_forces = _legacy_direct_output_call(
    particle_mesh_ewald,
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha,
    mesh_dimensions=mesh_dimensions,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
)

pme_slab_energy, pme_slab_forces = _legacy_direct_output_call(
    particle_mesh_ewald,
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha,
    mesh_dimensions=mesh_dimensions,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
    pbc=pbc_slab,
    slab_correction=True,
)

print("\nPME with slab correction:")
print(f"  Standard PME energy: {pme_3d_energy.sum().item(): .8f}")
print(f"  Slab PME energy: {pme_slab_energy.sum().item(): .8f}")
print(f"  Energy delta: {(pme_slab_energy - pme_3d_energy).sum().item(): .8f}")
print(f"  Max force magnitude: {pme_slab_forces.norm(dim=1).max().item(): .8f}")

# %%
# PME Energy-Autograd Slab Forces
# -------------------------------
# PME uses the same training recipe: omit direct-output flags and differentiate
# the energy. This is the path to copy into force-loss training code.

positions_pme_ag = positions.detach().requires_grad_(True)
pme_energy_ag = particle_mesh_ewald(
    positions=positions_pme_ag,
    charges=charges,
    cell=cell,
    alpha=alpha,
    mesh_dimensions=mesh_dimensions,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    pbc=pbc_slab,
    slab_correction=True,
)
pme_forces_ag = -torch.autograd.grad(pme_energy_ag.sum(), positions_pme_ag)[0]
print("\nPME slab energy-autograd force check:")
print(
    f"  Max force delta vs legacy direct: {(pme_forces_ag - pme_slab_forces).abs().max().item():.2e}"
)

# %%
# Total PME Component Composition
# -------------------------------
# If you need real-space, PME reciprocal-space, and slab terms separately,
# compute the three pieces explicitly and add matching outputs.

real_energy, real_forces = ewald_real_space(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha_tensor,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
)
reciprocal_energy, reciprocal_forces = pme_reciprocal_space(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=alpha,
    mesh_dimensions=mesh_dimensions,
    compute_forces=True,
)
slab_energy_correction, slab_forces_correction = compute_slab_correction(
    positions=positions,
    charges=charges,
    cell=cell,
    pbc=pbc_slab,
    compute_forces=True,
)

total_pme_energy = real_energy + reciprocal_energy + slab_energy_correction
total_pme_forces = real_forces + reciprocal_forces + slab_forces_correction

print("\nTotal PME composition from components:")
print(
    f"  Max energy error: {torch.max(torch.abs(total_pme_energy - pme_slab_energy)).item():.2e}"
)
print(
    f"  Max force error: {torch.max(torch.abs(total_pme_forces - pme_slab_forces)).item():.2e}"
)

# %%
# Triclinic Slab Cells
# --------------------
# Triclinic cells are also supported. The slab normal follows the plane spanned
# by the two periodic cell vectors; it is not locked to a Cartesian axis.
#
# Here we reuse the same positions and charges with a tilted cell and compute
# the standalone correction.

triclinic_cell = torch.tensor(
    [[[10.0, 0.0, 0.0], [1.5, 9.0, 0.8], [0.2, 0.4, 30.0]]],
    dtype=torch.float64,
    device=device,
)

triclinic_energy, triclinic_forces = compute_slab_correction(
    positions=positions,
    charges=charges,
    cell=triclinic_cell,
    pbc=pbc_slab,
    compute_forces=True,
)

print("\nTriclinic standalone correction:")
print(f"  Cell rows:\n{_format_array(triclinic_cell[0])}")
print(f"  Correction energy: {triclinic_energy.sum().item(): .8f}")
print(f"  Forces:\n{_format_array(triclinic_forces)}")

# %%
# Summary
# -------
# Use ``ewald_summation(..., slab_correction=True, pbc=pbc_slab)`` or
# ``particle_mesh_ewald(..., slab_correction=True, pbc=pbc_slab)`` when you want
# the correction included in the total energy. For training derivatives, copy
# the energy-autograd force sections above. Use ``compute_slab_correction``
# directly when you need the correction term alone or when composing
# ``ewald_real_space`` with ``ewald_reciprocal_space`` or
# ``pme_reciprocal_space``.
#
# For repeated molecular dynamics loops, keep ``pbc_slab`` as a contiguous
# ``(B, 3)`` tensor on the target device.
