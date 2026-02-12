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
Coulomb Electrostatic Interactions
==================================

This example demonstrates how to compute Coulomb electrostatic interactions
using GPU-accelerated kernels. The implementation supports both direct (undamped)
Coulomb interactions and damped interactions used as the real-space component
in Ewald/PME methods.

In this example you will learn:

- How to compute Coulomb energies and forces between charged particles
- The difference between undamped (1/r) and damped (erfc(αr)/r) interactions
- Using both neighbor list (COO) and neighbor matrix formats
- Computing forces via autograd vs explicit force kernels
- Batch evaluation for multiple systems
- Computing gradients with respect to charges

.. important::
    This script is intended as an API demonstration. Do not use this script
    for performance benchmarking; refer to the `benchmarks` folder instead.
"""

# %%
# Setup and Imports
# -----------------
# First, we import the necessary modules. The Coulomb API provides three main
# functions: ``coulomb_energy``, ``coulomb_forces``, and ``coulomb_energy_forces``.

from __future__ import annotations

import torch

from nvalchemiops.torch.interactions.electrostatics import (
    coulomb_energy,
    coulomb_energy_forces,
)

# %%
# Configure Device
# ----------------
# We use CUDA if available for GPU acceleration.

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# %%
# Basic Coulomb Calculation
# -------------------------
# We start with a simple two-atom system: a +1 and -1 charge separated by 2.0 Å.
# This demonstrates both undamped Coulomb (direct 1/r) and damped Coulomb
# (erfc(αr)/r used in Ewald/PME real-space).
#
# The undamped Coulomb energy for a pair is:
#
# .. math::
#     E = \frac{q_i q_j}{r}
#
# The damped Coulomb (Ewald real-space) energy is:
#
# .. math::
#     E = \frac{q_i q_j \, \mathrm{erfc}(\alpha r)}{r}

# Two charges: +1 and -1 separated by 2.0 Angstroms
positions = torch.tensor(
    [
        [0.0, 0.0, 0.0],  # Charge +1
        [2.0, 0.0, 0.0],  # Charge -1
    ],
    dtype=torch.float64,
    device=device,
)

charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)

# Large cell to avoid periodic images
cell = torch.tensor(
    [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
    dtype=torch.float64,
    device=device,
)

# Direct neighbor specification: pair (0, 1)
neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

print("System setup:")
print(f"  Positions: {positions.cpu().numpy()}")
print(f"  Charges: {charges.cpu().numpy()}")
print("  Distance: 2.0 Å")

# %%
# Undamped Coulomb
# ~~~~~~~~~~~~~~~~
# With α=0, we get the direct Coulomb interaction.

energies_undamped, forces_undamped = coulomb_energy_forces(
    positions,
    charges,
    cell,
    cutoff=10.0,
    alpha=0.0,  # Undamped
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
)

r = 2.0
expected_energy = 1.0 * (-1.0) / r  # = -0.5

print("\nUndamped Coulomb (α=0):")
print(
    f"  Total energy: {energies_undamped.sum().item():.6f} (expected: {expected_energy:.6f})"
)
print(f"  Forces on atom 0: {forces_undamped[0].cpu().numpy()}")
print(f"  Forces on atom 1: {forces_undamped[1].cpu().numpy()}")

# %%
# Damped Coulomb (Ewald Real-Space)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# With α > 0, the interaction is damped by erfc(αr). This is the short-range
# component used in Ewald/PME methods.

energies_damped, forces_damped = coulomb_energy_forces(
    positions,
    charges,
    cell,
    cutoff=10.0,
    alpha=0.3,  # Damped with α=0.3
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
)

print("\nDamped Coulomb (α=0.3):")
print(f"  Total energy: {energies_damped.sum().item():.6f}")
print(f"  Forces on atom 0: {forces_damped[0].cpu().numpy()}")
print(f"  Forces on atom 1: {forces_damped[1].cpu().numpy()}")

# %%
# Newton's Third Law Check
# ~~~~~~~~~~~~~~~~~~~~~~~~
# The sum of forces should be zero for an isolated system.

net_force = forces_undamped.sum(dim=0)
print("\nNewton's 3rd law check:")
print(f"  Net force: {net_force.cpu().numpy()}")
print(f"  Magnitude: {net_force.norm().item():.2e} (should be ~0)")

# %%
# Autograd vs Explicit Forces
# ---------------------------
# The Coulomb functions support automatic differentiation. Forces computed via
# autograd should match the explicit forces returned by the kernel.

positions_grad = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.73, 0.0],  # Equilateral triangle
    ],
    dtype=torch.float64,
    device=device,
    requires_grad=True,
)

charges_3 = torch.tensor([1.0, -0.5, -0.5], dtype=torch.float64, device=device)
cell_3 = torch.tensor(
    [[[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]]],
    dtype=torch.float64,
    device=device,
)

# All pairs as neighbors
neighbor_list_3 = torch.tensor(
    [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.int32, device=device
)
neighbor_ptr_3 = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=device)
neighbor_shifts_3 = torch.zeros((6, 3), dtype=torch.int32, device=device)

print("\nThree-atom triangular system:")
print(f"  Charges: {charges_3.cpu().numpy()}")

# %%
# Get explicit forces from the kernel:

_, explicit_forces = coulomb_energy_forces(
    positions_grad.detach(),
    charges_3,
    cell_3,
    cutoff=10.0,
    alpha=0.0,
    neighbor_list=neighbor_list_3,
    neighbor_ptr=neighbor_ptr_3,
    neighbor_shifts=neighbor_shifts_3,
)

print("\nExplicit Forces:")
for i, f in enumerate(explicit_forces):
    print(f"  Atom {i}: {f.cpu().numpy()}")

# %%
# Get autograd forces via backpropagation:

energies = coulomb_energy(
    positions_grad,
    charges_3,
    cell_3,
    cutoff=10.0,
    alpha=0.0,
    neighbor_list=neighbor_list_3,
    neighbor_ptr=neighbor_ptr_3,
    neighbor_shifts=neighbor_shifts_3,
)
total_energy = energies.sum()
total_energy.backward()
autograd_forces = -positions_grad.grad

print(f"\nTotal energy: {total_energy.item():.6f}")
print("\nAutograd Forces (-∂E/∂r):")
for i, f in enumerate(autograd_forces):
    print(f"  Atom {i}: {f.cpu().numpy()}")

force_diff = (autograd_forces - explicit_forces).abs().max().item()
print(f"\nMax force difference: {force_diff:.2e}")
print(f"Forces match: {'✓' if force_diff < 1e-8 else '✗'}")

# %%
# Neighbor List vs Neighbor Matrix Format
# ---------------------------------------
# The Coulomb functions support two neighbor formats:
#
# - **Neighbor List (COO)**: Shape ``(2, num_pairs)`` where each pair is listed once
# - **Neighbor Matrix**: Shape ``(N, max_neighbors)`` with padding
#
# Both formats produce identical results.

# Create a 4-atom system in a square
positions_4 = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ],
    dtype=torch.float64,
    device=device,
)
charges_4 = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64, device=device)
cell_4 = torch.tensor(
    [[[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]]],
    dtype=torch.float64,
    device=device,
)

print("\n4-atom square system:")
print(f"  Charges: {charges_4.cpu().numpy()}")

# %%
# Neighbor list format (COO):

neighbor_list_coo = torch.tensor(
    [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]],
    dtype=torch.int32,
    device=device,
)
neighbor_ptr_coo = torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32, device=device)
neighbor_shifts_coo = torch.zeros((12, 3), dtype=torch.int32, device=device)

energies_list, forces_list = coulomb_energy_forces(
    positions_4,
    charges_4,
    cell_4,
    cutoff=10.0,
    alpha=0.0,
    neighbor_list=neighbor_list_coo,
    neighbor_ptr=neighbor_ptr_coo,
    neighbor_shifts=neighbor_shifts_coo,
)

print(f"\nNeighbor List format: Total energy = {energies_list.sum().item():.6f}")

# %%
# Neighbor matrix format:

fill_value = 4
neighbor_matrix = torch.tensor(
    [
        [1, 2, 3, fill_value],
        [0, 2, 3, fill_value],
        [0, 1, 3, fill_value],
        [0, 1, 2, fill_value],
    ],
    dtype=torch.int32,
    device=device,
)
neighbor_matrix_shifts = torch.zeros((4, 4, 3), dtype=torch.int32, device=device)

energies_matrix, forces_matrix = coulomb_energy_forces(
    positions_4,
    charges_4,
    cell_4,
    cutoff=10.0,
    alpha=0.0,
    neighbor_matrix=neighbor_matrix,
    neighbor_matrix_shifts=neighbor_matrix_shifts,
    fill_value=fill_value,
)

print(f"Neighbor Matrix format: Total energy = {energies_matrix.sum().item():.6f}")

energy_diff = (energies_list.sum() - energies_matrix.sum()).abs().item()
force_diff = (forces_list - forces_matrix).abs().max().item()
print(f"\nEnergy difference: {energy_diff:.2e}")
print(f"Max force difference: {force_diff:.2e}")
print(f"Formats match: {'✓' if energy_diff < 1e-10 and force_diff < 1e-10 else '✗'}")

# %%
# Batch Evaluation
# ----------------
# Multiple independent systems can be evaluated simultaneously using the
# ``batch_idx`` parameter. Each atom is assigned to a system index.

# Create two identical 2-atom systems
system_positions = torch.tensor(
    [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=torch.float64, device=device
)
system_charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)

# Concatenate for batch (second system offset by 100 Å)
positions_batch = torch.cat([system_positions, system_positions + 100.0])
charges_batch = torch.cat([system_charges, system_charges])
batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

cell_batch = torch.tensor(
    [[[200.0, 0.0, 0.0], [0.0, 200.0, 0.0], [0.0, 0.0, 200.0]]],
    dtype=torch.float64,
    device=device,
)

# Neighbor pairs: (0,1) and (2,3)
nl_batch = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device)
ptr_batch = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
shifts_batch = torch.zeros((4, 3), dtype=torch.int32, device=device)

print("\nBatch Evaluation: Two identical 2-atom systems")

# %%
# Batch calculation:

energies_batch, _ = coulomb_energy_forces(
    positions_batch,
    charges_batch,
    cell_batch,
    cutoff=10.0,
    alpha=0.0,
    neighbor_list=nl_batch,
    neighbor_ptr=ptr_batch,
    neighbor_shifts=shifts_batch,
    batch_idx=batch_idx,
)

# Single system for comparison
single_nl = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
single_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
single_shifts = torch.zeros((1, 3), dtype=torch.int32, device=device)
energies_single, _ = coulomb_energy_forces(
    system_positions,
    system_charges,
    cell_batch,
    cutoff=10.0,
    alpha=0.0,
    neighbor_list=single_nl,
    neighbor_ptr=single_ptr,
    neighbor_shifts=single_shifts,
)

print(f"Single system energy: {energies_single.sum().item():.6f}")
print(f"Batch system 0 energy: {energies_batch[:2].sum().item():.6f}")
print(f"Batch system 1 energy: {energies_batch[2:].sum().item():.6f}")

# %%
# Charge Gradients
# ----------------
# The autograd support extends to charge gradients, enabling sensitivity analysis
# and optimization with respect to charges.

positions_q = torch.tensor(
    [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, device=device
)
charges_q = torch.tensor(
    [1.0, -1.0], dtype=torch.float64, device=device, requires_grad=True
)
cell_q = torch.tensor(
    [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
    dtype=torch.float64,
    device=device,
)
nl_q = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
ptr_q = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
shifts_q = torch.zeros((1, 3), dtype=torch.int32, device=device)

energies_q = coulomb_energy(
    positions_q,
    charges_q,
    cell_q,
    cutoff=10.0,
    alpha=0.0,
    neighbor_list=nl_q,
    neighbor_ptr=ptr_q,
    neighbor_shifts=shifts_q,
)
total_energy_q = energies_q.sum()
total_energy_q.backward()

print("\nCharge Gradients")
print(f"Two atoms at r=3.0 Å, charges: {charges_q.data.cpu().numpy()}")
print(f"Total energy: {total_energy_q.item():.6f}")

# %%
# Verify against analytical result:
#
# For E = q₀q₁/r:
#
# - ∂E/∂q₀ = q₁/r = -1/3 ≈ -0.333
# - ∂E/∂q₁ = q₀/r = 1/3 ≈ 0.333

print(f"\n∂E/∂q₀ = {charges_q.grad[0].item():.6f} (expected: {-1.0 / 3.0:.6f})")
print(f"∂E/∂q₁ = {charges_q.grad[1].item():.6f} (expected: {1.0 / 3.0:.6f})")

match_q0 = abs(charges_q.grad[0].item() - (-1.0 / 3.0)) < 1e-6
match_q1 = abs(charges_q.grad[1].item() - (1.0 / 3.0)) < 1e-6
print(f"Match: {'✓' if match_q0 and match_q1 else '✗'}")

# %%
# Summary
# -------
# This example demonstrated:
#
# 1. **Undamped Coulomb**: Direct 1/r interactions with ``alpha=0``
# 2. **Damped Coulomb**: Ewald real-space with ``alpha > 0``
# 3. **Autograd support**: Forces via ``-positions.grad``
# 4. **Neighbor formats**: Both COO list and matrix formats
# 5. **Batch evaluation**: Multiple systems with ``batch_idx``
# 6. **Charge gradients**: Sensitivity via ``charges.grad``
#
# For visualization of energy/force profiles as a function of distance,
# see the separate visualization scripts in the examples folder.

print("\nCoulomb example complete!")
