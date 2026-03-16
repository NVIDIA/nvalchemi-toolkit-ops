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
Ewald Summation for Long-Range Electrostatics
==============================================

This example demonstrates how to compute long-range electrostatic interactions
in periodic systems using the Ewald summation method. Ewald splits the slowly
converging Coulomb sum into rapidly converging real-space and reciprocal-space
components.

In this example you will learn:

- How to set up and run Ewald summation with automatic parameter estimation
- Using neighbor list and neighbor matrix formats
- Understanding convergence with accuracy-based parameter estimation
- Effect of the splitting parameter alpha
- Batch evaluation for multiple systems
- Computing charge gradients for ML potential training

The Ewald energy is decomposed as:

.. math::
    E_{\\text{total}} = E_{\\text{real}} + E_{\\text{reciprocal}} - E_{\\text{self}}

.. important::
    This script is intended as an API demonstration. Do not use this script
    for performance benchmarking; refer to the `benchmarks` folder instead.
"""

# %%
# Setup and Imports
# -----------------
# First, we import the necessary modules. The Ewald API provides unified functions
# that handle both single-system and batched calculations.

from __future__ import annotations

import numpy as np
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    estimate_ewald_parameters,
    ewald_real_space,
    ewald_reciprocal_space,
    ewald_summation,
    generate_k_vectors_ewald_summation,
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
# Create a NaCl Crystal System
# ----------------------------
# We define a helper function to create NaCl rock salt crystal supercells.
# NaCl has Na+ at (0,0,0) and Cl- at (0.5,0.5,0.5) in fractional coordinates.


def create_nacl_system(n_cells: int = 2, lattice_constant: float = 5.64):
    """Create a NaCl crystal supercell.

    Parameters
    ----------
    n_cells : int
        Number of unit cells in each direction.
    lattice_constant : float
        NaCl lattice constant in Angstroms.

    Returns
    -------
    positions, charges, cell, pbc : torch.Tensor
        System tensors.
    """
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

    positions = torch.tensor(np.array(positions), dtype=torch.float64, device=device)
    charges = torch.tensor(np.array(charges), dtype=torch.float64, device=device)
    cell = torch.eye(3, dtype=torch.float64, device=device) * lattice_constant * n_cells
    cell = cell.unsqueeze(0)
    pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)

    return positions, charges, cell, pbc


# %%
# Basic Usage with Automatic Parameters
# -------------------------------------
# The simplest way to use Ewald summation is with automatic parameter estimation.
# Given an accuracy tolerance, the API estimates optimal alpha, real-space cutoff,
# and reciprocal-space cutoff using the Kolafa-Perram formula.

# Create a small NaCl crystal (2×2×2 unit cells = 16 atoms)
positions, charges, cell, pbc = create_nacl_system(n_cells=2)

print(f"System: {len(positions)} atoms NaCl crystal")
print(f"Cell size: {cell[0, 0, 0]:.2f} Å")
print(f"Total charge: {charges.sum().item():.1f} (should be 0 for neutral)")

# %%
# Estimate optimal parameters for target accuracy:

params = estimate_ewald_parameters(positions.cpu(), cell.cpu(), accuracy=1e-6)

print("\nEstimated parameters (accuracy=1e-6):")
print(f"  alpha = {params.alpha.item():.4f}")
print(f"  real_space_cutoff = {params.real_space_cutoff.item():.2f} Å")
print(f"  reciprocal_space_cutoff = {params.reciprocal_space_cutoff.item():.2f} Å⁻¹")

# %%
# Build neighbor list and run Ewald summation:

neighbor_list, neighbor_ptr, neighbor_shifts = neighbor_list_fn(
    positions,
    params.real_space_cutoff.item(),
    cell=cell,
    pbc=pbc,
    return_neighbor_list=True,
)

energies, forces = ewald_summation(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    accuracy=1e-6,  # Parameters estimated automatically
    compute_forces=True,
)

total_energy = energies.sum().item()
print("\nEwald Summation Results:")
print(f"  Total energy: {total_energy:.6f}")
print(f"  Energy per atom: {total_energy / len(positions):.6f}")
print(f"  Max force magnitude: {torch.norm(forces, dim=1).max().item():.6f}")

# %%
# Neighbor List vs Neighbor Matrix Format
# ---------------------------------------
# The Ewald functions support two neighbor formats. Both produce identical results.
# We use the estimated parameters from above for consistency.

# Build both neighbor formats using the estimated real-space cutoff
neighbor_list, neighbor_ptr, neighbor_shifts = neighbor_list_fn(
    positions,
    params.real_space_cutoff.item(),
    cell=cell,
    pbc=pbc,
    return_neighbor_list=True,
)
neighbor_matrix, _, neighbor_matrix_shifts = neighbor_list_fn(
    positions,
    params.real_space_cutoff.item(),
    cell=cell,
    pbc=pbc,
    return_neighbor_list=False,
)

print("\nNeighbor format comparison (accuracy=1e-6):")
print(
    f"  Using alpha={params.alpha.item():.4f}, k_cutoff={params.reciprocal_space_cutoff.item():.2f}"
)

# %%
# Using neighbor list format:

energies_list = ewald_summation(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    accuracy=1e-6,  # Parameters estimated automatically
)

print(f"  List format energy: {energies_list.sum().item():.6f}")

# %%
# Using neighbor matrix format:

energies_matrix = ewald_summation(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_matrix=neighbor_matrix,
    neighbor_matrix_shifts=neighbor_matrix_shifts,
    accuracy=1e-6,  # Same accuracy for comparison
)

print(f"  Matrix format energy: {energies_matrix.sum().item():.6f}")
print(
    f"  Difference: {abs(energies_list.sum().item() - energies_matrix.sum().item()):.2e}"
)

# %%
# Convergence with Accuracy Parameter
# -----------------------------------
# The Ewald summation accuracy depends on the accuracy parameter, which controls
# both the real-space cutoff and the k-space cutoff. The parameter estimation
# uses the Kolafa-Perram formula to balance computational cost.

accuracies = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
energies_acc = []
params_list = []

print("\nConvergence with Accuracy Target:")
print("  Accuracy | alpha  | r_cutoff | k_cutoff | Num k-vectors | Energy")
print("  " + "-" * 75)

for acc in accuracies:
    # Estimate optimal parameters for this accuracy
    params_acc = estimate_ewald_parameters(positions, cell, accuracy=acc)
    params_list.append(params_acc)

    # Build neighbor list with appropriate cutoff
    nl_acc, nptr_acc, ns_acc = neighbor_list_fn(
        positions,
        params_acc.real_space_cutoff.item(),
        cell=cell,
        pbc=pbc,
        return_neighbor_list=True,
    )

    # Generate k-vectors for counting
    k_vectors = generate_k_vectors_ewald_summation(
        cell, params_acc.reciprocal_space_cutoff
    )
    num_kvec = k_vectors.shape[1]

    # Run Ewald summation with estimated parameters
    energy = ewald_summation(
        positions=positions,
        charges=charges,
        cell=cell,
        neighbor_list=nl_acc,
        neighbor_ptr=nptr_acc,
        neighbor_shifts=ns_acc,
        accuracy=acc,
    )
    total_e = energy.sum().item()
    energies_acc.append(total_e)

    print(
        f"   {acc:.0e}  |  {params_acc.alpha.item():.3f}  |   {params_acc.real_space_cutoff.item():5.2f}  |"
        f"   {params_acc.reciprocal_space_cutoff.item():5.2f}  |  {num_kvec:10d}   | {total_e:12.6f}"
    )

# %%
# Show convergence relative to highest accuracy:

ref_energy = energies_acc[-1]
print("\nRelative error from reference (accuracy=1e-7):")
for acc, e in zip(accuracies[:-1], energies_acc[:-1]):
    rel_err = abs((e - ref_energy) / ref_energy)
    print(f"  accuracy={acc:.0e}: {rel_err:.2e}")

# %%
# Real-Space and Reciprocal-Space Components
# ------------------------------------------
# The Ewald energy splits into real-space (short-range) and reciprocal-space
# (long-range) components. You can access these separately.

# Create a larger system for clearer demonstration
positions_4, charges_4, cell_4, pbc_4 = create_nacl_system(n_cells=4)

params_4 = estimate_ewald_parameters(positions_4, cell_4, accuracy=1e-4)
k_vectors_4 = generate_k_vectors_ewald_summation(
    cell_4, params_4.reciprocal_space_cutoff
)

neighbor_list_4, neighbor_ptr_4, neighbor_shifts_4 = neighbor_list_fn(
    positions_4,
    params_4.real_space_cutoff.item(),
    cell=cell_4,
    pbc=pbc_4,
    return_neighbor_list=True,
)

print(f"\nEnergy Components ({len(positions_4)} atoms):")

# %%
# Real-space component:

real_energy = ewald_real_space(
    positions=positions_4,
    charges=charges_4,
    cell=cell_4,
    alpha=params_4.alpha,
    neighbor_list=neighbor_list_4,
    neighbor_ptr=neighbor_ptr_4,
    neighbor_shifts=neighbor_shifts_4,
    compute_forces=False,
)

print(f"  Real-space: {real_energy.sum().item():.6f}")

# %%
# Reciprocal-space component:

recip_energy = ewald_reciprocal_space(
    positions=positions_4,
    charges=charges_4,
    cell=cell_4,
    k_vectors=k_vectors_4,
    alpha=params_4.alpha,
    compute_forces=False,
)

print(f"  Reciprocal-space: {recip_energy.sum().item():.6f}")
print(f"  Total: {(real_energy.sum() + recip_energy.sum()).item():.6f}")

# %%
# Charge Gradients for ML Potentials
# ----------------------------------
# The Ewald functions support computing analytical charge gradients (∂E/∂q_i),
# which are useful for training machine learning potentials that predict charges.
# The charge gradient represents the electrostatic potential at each atom.

print("\nCharge Gradients:")

# Compute real-space component with charge gradients
real_energies, real_forces, real_charge_grads = ewald_real_space(
    positions=positions_4,
    charges=charges_4,
    cell=cell_4,
    alpha=params_4.alpha,
    neighbor_list=neighbor_list_4,
    neighbor_ptr=neighbor_ptr_4,
    neighbor_shifts=neighbor_shifts_4,
    compute_forces=True,
    compute_charge_gradients=True,
)

print(f"  Real-space charge gradients shape: {real_charge_grads.shape}")
print(
    f"  Real-space charge gradients range: [{real_charge_grads.min().item():.4f}, {real_charge_grads.max().item():.4f}]"
)

# Compute reciprocal-space component with charge gradients
recip_energies, recip_forces, recip_charge_grads = ewald_reciprocal_space(
    positions=positions_4,
    charges=charges_4,
    cell=cell_4,
    k_vectors=k_vectors_4,
    alpha=params_4.alpha,
    compute_forces=True,
    compute_charge_gradients=True,
)

print(
    f"  Reciprocal-space charge gradients range: [{recip_charge_grads.min().item():.4f}, {recip_charge_grads.max().item():.4f}]"
)

# Total charge gradient is the sum of components
total_charge_grads = real_charge_grads + recip_charge_grads
print(
    f"  Total charge gradients range: [{total_charge_grads.min().item():.4f}, {total_charge_grads.max().item():.4f}]"
)

# %%
# Full Ewald with charge gradients in one call:
#
# Instead of calling the components separately, ``ewald_summation`` now supports
# ``compute_charge_gradients`` directly — matching ``particle_mesh_ewald``.

energies_full, forces_full, charge_grads_full = ewald_summation(
    positions=positions_4,
    charges=charges_4,
    cell=cell_4,
    neighbor_list=neighbor_list_4,
    neighbor_ptr=neighbor_ptr_4,
    neighbor_shifts=neighbor_shifts_4,
    compute_forces=True,
    compute_charge_gradients=True,
    accuracy=1e-4,
)

print(
    f"\n  Full Ewald charge gradients range: [{charge_grads_full.min().item():.4f}, {charge_grads_full.max().item():.4f}]"
)

# Verify unified call matches component-level results
component_diff = (charge_grads_full - total_charge_grads).abs().max().item()
print(f"  Unified vs component charge gradient max diff: {component_diff:.2e}")

# %%
# Verify charge gradients against autograd:

charges_4.requires_grad_(True)
energies_total = ewald_summation(
    positions=positions_4,
    charges=charges_4,
    cell=cell_4,
    neighbor_list=neighbor_list_4,
    neighbor_ptr=neighbor_ptr_4,
    neighbor_shifts=neighbor_shifts_4,
    accuracy=1e-4,
).sum()

energies_total.backward()
autograd_charge_grads = charges_4.grad.clone()
charges_4.requires_grad_(False)
charges_4.grad = None

# Compare explicit vs autograd charge gradients
charge_grad_diff = (charge_grads_full - autograd_charge_grads).abs().max().item()
print(f"  Explicit vs Autograd charge gradient max diff: {charge_grad_diff:.2e}")

# %%
# Batch Evaluation
# ----------------
# Multiple systems can be evaluated simultaneously using batch_idx.
# Each system can have different alpha values.

n_systems = 3
all_positions = []
all_charges = []
all_cells = []
all_pbc = []
batch_idx_list = []

print(f"\nBatch Evaluation: Creating {n_systems} systems...")

for i in range(n_systems):
    n_cells = i + 2  # 2×2×2, 3×3×3, 4×4×4
    pos, chrg, cell_i, pbc_i = create_nacl_system(n_cells=n_cells)
    batch_idx_list.extend([i] * len(pos))
    all_positions.append(pos)
    all_charges.append(chrg)
    all_cells.append(cell_i)
    all_pbc.append(pbc_i)
    print(f"  System {i}: {len(pos)} atoms ({n_cells}×{n_cells}×{n_cells})")

# %%
# Concatenate all systems:

positions_batch = torch.cat(all_positions, dim=0)
charges_batch = torch.cat(all_charges, dim=0)
cells_batch = torch.cat(all_cells, dim=0)
pbc_batch = torch.cat(all_pbc, dim=0)
batch_idx = torch.tensor(batch_idx_list, dtype=torch.int32, device=device)

# Estimate parameters for the batch with desired accuracy
params_batch = estimate_ewald_parameters(
    positions_batch, cells_batch, batch_idx=batch_idx, accuracy=1e-5
)

print(f"\nTotal atoms: {len(positions_batch)}")
print(f"Per-system alphas: {params_batch.alpha.tolist()}")
print(f"Real-space cutoff: {params_batch.real_space_cutoff.max().item():.2f} Å")
print(f"K-space cutoff: {params_batch.reciprocal_space_cutoff.max().item():.2f} Å⁻¹")

# %%
# Build batched neighbor list and run:

# Use the maximum real-space cutoff across all systems
real_cutoff_batch = params_batch.real_space_cutoff.max().item()
neighbor_matrix_batch, _, neighbor_matrix_shifts_batch = neighbor_list_fn(
    positions_batch,
    real_cutoff_batch,
    cell=cells_batch,
    pbc=pbc_batch,
    method="batch_naive",
    batch_idx=batch_idx,
    return_neighbor_list=False,
)

energies_batch, forces_batch = ewald_summation(
    positions=positions_batch,
    charges=charges_batch,
    cell=cells_batch,
    batch_idx=batch_idx,
    neighbor_matrix=neighbor_matrix_batch,
    neighbor_matrix_shifts=neighbor_matrix_shifts_batch,
    compute_forces=True,
    accuracy=1e-5,  # Parameters estimated automatically for batch
)

print("\nPer-system results:")
for i in range(n_systems):
    mask = batch_idx == i
    n_atoms = mask.sum().item()
    sys_energy = energies_batch[mask].sum().item()
    max_force = torch.norm(forces_batch[mask], dim=1).max().item()
    print(f"  System {i}: {n_atoms} atoms, E={sys_energy:.4f}, |F|_max={max_force:.4f}")

# %%
# Verify batch vs individual calculations:

print("\nVerification (individual calculations with same accuracy):")
for i in range(n_systems):
    mask = batch_idx == i
    pos_i = positions_batch[mask]
    chrg_i = charges_batch[mask]
    cell_i = cells_batch[i : i + 1]
    pbc_i = pbc_batch[i : i + 1]

    # Use same cutoff as batch for fair comparison
    nl_i, nptr_i, ns_i = neighbor_list_fn(
        pos_i, real_cutoff_batch, cell=cell_i, pbc=pbc_i, return_neighbor_list=True
    )

    e_i = ewald_summation(
        positions=pos_i,
        charges=chrg_i,
        cell=cell_i,
        neighbor_list=nl_i,
        neighbor_ptr=nptr_i,
        neighbor_shifts=ns_i,
        accuracy=1e-5,  # Same accuracy as batch
    )
    print(f"  System {i}: E={e_i.sum().item():.4f}")

# %%
# Summary
# -------
# This example demonstrated:
#
# 1. **Automatic parameter estimation** based on target accuracy using the
#    Kolafa-Perram formula via ``estimate_ewald_parameters``
# 2. **Neighbor format flexibility** with list and matrix formats
# 3. **Accuracy-based convergence** showing how the accuracy parameter
#    controls both real-space and k-space cutoffs
# 4. **Component access** for real-space and reciprocal-space energies
# 5. **Charge gradients** (∂E/∂q_i) for ML potential training
# 6. **Batch evaluation** for multiple systems with automatic per-system alpha
#
# Key equations implemented:
#
# - Real-space: :math:`E_{\\text{real}} = \\frac{1}{2} \\sum_{i \\neq j} q_i q_j \\frac{\\mathrm{erfc}(\\alpha r_{ij})}{r_{ij}}`
# - Reciprocal: :math:`E_{\\text{recip}} = \\frac{1}{2V} \\sum_{\\mathbf{k} \\neq 0} \\frac{4\\pi}{k^2} e^{-k^2/4\\alpha^2} |S(\\mathbf{k})|^2`
# - Self-energy: :math:`E_{\\text{self}} = \\frac{\\alpha}{\\sqrt{\\pi}} \\sum_i q_i^2`
# - Charge gradient: :math:`\\frac{\\partial E}{\\partial q_i} = \\phi_i` (electrostatic potential)

print("\nEwald summation example complete!")
