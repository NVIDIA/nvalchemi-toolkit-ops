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
2D Slab Correction for Ewald Summation
======================================

This example demonstrates how to apply the Yeh-Berkowitz / Ballenegger
two-dimensional slab correction to Ewald electrostatics. The correction is used
for slab-like systems with two periodic directions and one non-periodic
direction, such as interfaces with vacuum padding.

In this example you will learn:

- How to run ``ewald_summation(..., slab_correction=True)``
- How to pass slab periodicity with a boolean ``pbc`` tensor
- How to compute the standalone slab correction with ``apply_slab_correction``
- How the standalone correction equals the integrated Ewald energy/force delta
- How triclinic slab cells use the normal to the periodic plane

.. important::
    This script is intended as an API demonstration. Do not use this script
    for performance benchmarking; refer to the `benchmarks` folder instead.
"""

# %%
# Setup and Imports
# -----------------
# The slab correction is available through the high-level Ewald API and as a
# standalone helper. The standalone helper is useful for debugging, validation,
# and adding the correction to precomputed 3D Ewald outputs.

from __future__ import annotations

import torch

from nvalchemiops.torch.interactions.electrostatics import (
    apply_slab_correction,
    ewald_summation,
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
print(f"  Cell rows:\n{cell[0].cpu().numpy()}")
print(f"  Slab pbc: {pbc_slab[0].cpu().numpy()}")
print(f"  Total charge: {charges.sum().item():.1f}")

# %%
# Build the Real-Space Neighbor List
# ----------------------------------
# The slab correction modifies the long-range Ewald result after computing a
# standard 3D Ewald sum in a cell with vacuum padding. The neighbor list controls
# the real-space periodic images used by that 3D Ewald calculation.
#
# For this demonstration we use full 3D periodicity for the neighbor list and a
# cutoff smaller than the vacuum gap, so no short-range neighbors cross the slab
# normal.

alpha = 0.3
real_space_cutoff = 5.0
k_cutoff = 2.5

pbc_neighbor = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
neighbor_list, neighbor_ptr, neighbor_shifts = neighbor_list_fn(
    positions,
    real_space_cutoff,
    cell=cell,
    pbc=pbc_neighbor,
    return_neighbor_list=True,
)

print("\nNeighbor list:")
print(f"  Number of neighbor entries: {neighbor_list.shape[1]}")
print(f"  Real-space cutoff: {real_space_cutoff:.1f} Å")

# %%
# Standard 3D Ewald
# -----------------
# First compute the uncorrected 3D-periodic Ewald result. This is the quantity
# that will receive the slab correction.

energies_3d, forces_3d, charge_grads_3d, virial_3d = ewald_summation(
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
print(f"  Charge gradients: {charge_grads_3d.cpu().numpy()}")
print(f"  Virial trace: {torch.trace(virial_3d[0]).item(): .8f}")

# %%
# Ewald with Slab Correction
# --------------------------
# Set ``slab_correction=True`` and pass the slab periodicity. The output tuple
# follows the same ordering as ordinary Ewald: energies, forces, charge
# gradients, and virial when all optional quantities are requested.

energies_slab, forces_slab, charge_grads_slab, virial_slab = ewald_summation(
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
print(f"  Charge gradients: {charge_grads_slab.cpu().numpy()}")
print(f"  Virial trace: {torch.trace(virial_slab[0]).item(): .8f}")

# %%
# Standalone Slab Correction
# --------------------------
# The same correction can be computed directly. The standalone result equals
# the difference between the slab-corrected and uncorrected Ewald outputs.

correction_energy, correction_forces, correction_charge_grads, correction_virial = (
    apply_slab_correction(
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

triclinic_energy, triclinic_forces = apply_slab_correction(
    positions=positions,
    charges=charges,
    cell=triclinic_cell,
    pbc=pbc_slab,
    compute_forces=True,
)

print("\nTriclinic standalone correction:")
print(f"  Cell rows:\n{triclinic_cell[0].cpu().numpy()}")
print(f"  Correction energy: {triclinic_energy.sum().item(): .8f}")
print(f"  Forces:\n{triclinic_forces.cpu().numpy()}")

# %%
# Summary
# -------
# Use ``ewald_summation(..., slab_correction=True, pbc=pbc_slab)`` when you want
# the correction included in the total Ewald outputs. Use
# ``apply_slab_correction`` directly when you need the correction term alone.
#
# For repeated molecular dynamics loops, keep ``pbc_slab`` as a contiguous
# ``(B, 3)`` tensor on the target device.
