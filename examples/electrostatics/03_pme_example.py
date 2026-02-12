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
Particle Mesh Ewald (PME) for Long-Range Electrostatics
=======================================================

This example demonstrates how to compute long-range electrostatic interactions
using the Particle Mesh Ewald (PME) method. PME achieves O(N log N) scaling by
using FFT-based mesh interpolation for the reciprocal-space contribution.

In this example you will learn:

- How to set up and run PME with automatic parameter estimation
- Using neighbor list and neighbor matrix formats
- Understanding convergence with accuracy-based parameter estimation
- Effect of the splitting parameter alpha and accuracy
- Batch evaluation for multiple systems
- Comparison between PME and standard Ewald summation
- Computing charge gradients for ML potential training

PME accelerates the reciprocal-space sum using B-spline interpolation:

1. Spread charges to mesh using B-splines
2. FFT to reciprocal space
3. Multiply by Green's function
4. Inverse FFT to get potentials
5. Interpolate forces back to atom positions

.. important::
    This script is intended as an API demonstration. Do not use this script
    for performance benchmarking; refer to the `benchmarks` folder instead.
"""

# %%
# Setup and Imports
# -----------------
# First, we import the necessary modules. The PME API provides unified functions
# that handle both single-system and batched calculations.

from __future__ import annotations

import time

import numpy as np
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    estimate_ewald_parameters,
    estimate_pme_parameters,
    ewald_real_space,
    ewald_summation,
    particle_mesh_ewald,
    pme_reciprocal_space,
)
from nvalchemiops.torch.neighbors import neighbor_list as neighbor_list_fn

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
# Create a NaCl Crystal System
# ----------------------------
# We define a helper function to create NaCl rock salt crystal supercells.


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

    positions = torch.tensor(positions, dtype=torch.float64, device=device)
    charges = torch.tensor(charges, dtype=torch.float64, device=device)
    cell = torch.eye(3, dtype=torch.float64, device=device) * lattice_constant * n_cells
    cell = cell.unsqueeze(0)
    pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)

    return positions, charges, cell, pbc


# %%
# Basic Usage with Automatic Parameters
# -------------------------------------
# The simplest way to use PME is with automatic parameter estimation.
# Given an accuracy tolerance, the API estimates optimal alpha, mesh dimensions,
# and real-space cutoff.

# Create a NaCl crystal (3×3×3 unit cells = 54 atoms)
positions, charges, cell, pbc = create_nacl_system(n_cells=3)

print(f"System: {len(positions)} atoms NaCl crystal")
print(f"Cell size: {cell[0, 0, 0]:.2f} Å")
print(f"Total charge: {charges.sum().item():.1f} (should be 0 for neutral)")

# %%
# Estimate optimal PME parameters:

params = estimate_pme_parameters(positions.cpu(), cell.cpu(), accuracy=1e-6)

print("\nEstimated parameters (accuracy=1e-6):")
print(f"  alpha = {params.alpha.item():.4f}")
print(f"  mesh_dimensions = {params.mesh_dimensions}")
spacing = (
    params.mesh_spacing[0] if params.mesh_spacing.dim() == 2 else params.mesh_spacing
)
print(
    f"  mesh_spacing = ({spacing[0].item():.2f}, {spacing[1].item():.2f}, {spacing[2].item():.2f}) Å"
)
print(f"  real_space_cutoff = {params.real_space_cutoff.item():.2f} Å")

# %%
# Build neighbor list and run PME:

neighbor_list, neighbor_ptr, neighbor_shifts = neighbor_list_fn(
    positions,
    params.real_space_cutoff.item(),
    cell=cell,
    pbc=pbc,
    return_neighbor_list=True,
)

t0 = time.time()
energies, forces = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
    accuracy=1e-6,  # Parameters estimated automatically
)
t1 = time.time()

total_energy = energies.sum().item()
print("\nPME Results:")
print(f"  Total energy: {total_energy:.6f}")
print(f"  Energy per atom: {total_energy / len(positions):.6f}")
print(f"  Max force magnitude: {torch.norm(forces, dim=1).max().item():.6f}")
print(f"  Time: {(t1 - t0) * 1000:.2f} ms")

# %%
# Neighbor List vs Neighbor Matrix Format
# ---------------------------------------
# PME supports both neighbor formats, producing identical results.
# We use the estimated parameters from above for consistency.

# Build both formats using the estimated real-space cutoff
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
print(f"  Using alpha={params.alpha.item():.4f}, mesh_dims={params.mesh_dimensions}")

# %%
# Using neighbor list format:

t0 = time.time()
energies_list, forces_list = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    neighbor_shifts=neighbor_shifts,
    compute_forces=True,
    accuracy=1e-6,  # Parameters estimated automatically
)
t_list = (time.time() - t0) * 1000

print(f"  List format: E={energies_list.sum().item():.6f}, time={t_list:.2f} ms")

# %%
# Using neighbor matrix format:

t0 = time.time()
energies_matrix, forces_matrix = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_matrix=neighbor_matrix,
    neighbor_matrix_shifts=neighbor_matrix_shifts,
    compute_forces=True,
    accuracy=1e-6,  # Same accuracy for comparison
)
t_matrix = (time.time() - t0) * 1000

print(f"  Matrix format: E={energies_matrix.sum().item():.6f}, time={t_matrix:.2f} ms")

energy_diff = abs(energies_list.sum().item() - energies_matrix.sum().item())
force_diff = (forces_list - forces_matrix).abs().max().item()
print(f"\nEnergy difference: {energy_diff:.2e}")
print(f"Max force difference: {force_diff:.2e}")

# %%
# Convergence with Accuracy Parameter
# -----------------------------------
# The PME accuracy depends on the accuracy parameter, which controls both the
# mesh resolution and alpha parameter. The parameter estimation uses optimal
# formulas to balance computational cost between real and reciprocal space.

accuracies = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
results_acc = []

print("\nConvergence with Accuracy Target:")
print("  Accuracy | alpha  | mesh_dims    | r_cutoff | Energy       | Time")
print("  " + "-" * 75)

# %%
# Run PME with different accuracy targets:

for acc in accuracies:
    # Estimate optimal parameters for this accuracy
    params_acc = estimate_pme_parameters(positions, cell, accuracy=acc)

    # Build neighbor list with appropriate cutoff
    nl_acc, nptr_acc, ns_acc = neighbor_list_fn(
        positions,
        params_acc.real_space_cutoff.item(),
        cell=cell,
        pbc=pbc,
        return_neighbor_list=True,
    )

    t0 = time.time()
    energies_acc = particle_mesh_ewald(
        positions=positions,
        charges=charges,
        cell=cell,
        neighbor_list=nl_acc,
        neighbor_ptr=nptr_acc,
        neighbor_shifts=ns_acc,
        accuracy=acc,
    )
    t_elapsed = (time.time() - t0) * 1000

    total_e = energies_acc.sum().item()
    results_acc.append((acc, params_acc, total_e, t_elapsed))
    print(
        f"   {acc:.0e}  |  {params_acc.alpha.item():.3f}  | {str(params_acc.mesh_dimensions):12s} |"
        f"   {params_acc.real_space_cutoff.item():5.2f}  | {total_e:10.6f}  | {t_elapsed:.2f} ms"
    )

# %%
# Show convergence relative to highest accuracy:

ref_energy_acc = results_acc[-1][2]
print("\nRelative error from reference (accuracy=1e-7):")
for acc, _, e, _ in results_acc[:-1]:
    rel_err = abs((e - ref_energy_acc) / ref_energy_acc)
    print(f"  accuracy={acc:.0e}: {rel_err:.2e}")

# %%
# Effect of Accuracy on PME Parameters
# ------------------------------------
# The accuracy parameter controls how alpha and mesh dimensions are chosen.
# Lower accuracy targets require larger meshes and cutoffs.

positions_2, charges_2, cell_2, pbc_2 = create_nacl_system(n_cells=2)
accuracies = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

print("\nAccuracy Effect on PME Parameters:")
print("  Accuracy | alpha | mesh_dims | real_cutoff | Total Energy")
print("  " + "-" * 65)

# %%
# Sweep through accuracy values:

for accuracy in accuracies:
    params_acc = estimate_pme_parameters(
        positions_2.cpu(), cell_2.cpu(), accuracy=accuracy
    )

    nl_acc, nptr_acc, ns_acc = neighbor_list_fn(
        positions_2,
        params_acc.real_space_cutoff.item(),
        cell=cell_2,
        pbc=pbc_2,
        return_neighbor_list=True,
    )

    energies_acc = particle_mesh_ewald(
        positions=positions_2,
        charges=charges_2,
        cell=cell_2,
        alpha=params_acc.alpha.item(),
        mesh_dimensions=tuple(params_acc.mesh_dimensions),
        neighbor_list=nl_acc,
        neighbor_ptr=nptr_acc,
        neighbor_shifts=ns_acc,
    )

    print(
        f"   {accuracy:.0e}  |  {params_acc.alpha.item():.2f}  | {str(params_acc.mesh_dimensions):9s}"
        f"  |    {params_acc.real_space_cutoff.item():5.2f}   | {energies_acc.sum().item():.6f}"
    )

# %%
# Accessing Real-Space and Reciprocal-Space Components
# ----------------------------------------------------
# You can compute the components separately if needed.

params_comp = estimate_pme_parameters(positions, cell, accuracy=1e-4)

nl_comp, nptr_comp, ns_comp = neighbor_list_fn(
    positions,
    params_comp.real_space_cutoff.item(),
    cell=cell,
    pbc=pbc,
    return_neighbor_list=True,
)

print("\nEnergy Components:")

# %%
# Real-space component (uses same kernel as Ewald):

real_energy = ewald_real_space(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=params_comp.alpha,
    neighbor_list=nl_comp,
    neighbor_ptr=nptr_comp,
    neighbor_shifts=ns_comp,
)

print(f"  Real-space: {real_energy.sum().item():.6f}")

# %%
# PME reciprocal-space component (FFT-based):

recip_energy = pme_reciprocal_space(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=params_comp.alpha,
    mesh_dimensions=tuple(params_comp.mesh_dimensions),
)

print(f"  Reciprocal-space (PME): {recip_energy.sum().item():.6f}")
print(f"  Total: {(real_energy.sum() + recip_energy.sum()).item():.6f}")

# %%
# Charge Gradients for ML Potentials
# ----------------------------------
# PME supports computing analytical charge gradients (∂E/∂q_i), which are useful
# for training machine learning potentials that predict atomic partial charges.
# The charge gradient represents the electrostatic potential at each atom.

print("\nCharge Gradients:")

# Compute PME reciprocal-space with charge gradients
recip_energies, recip_forces, recip_charge_grads = pme_reciprocal_space(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=params_comp.alpha,
    mesh_dimensions=tuple(params_comp.mesh_dimensions),
    compute_forces=True,
    compute_charge_gradients=True,
)

print(f"  PME reciprocal charge gradients shape: {recip_charge_grads.shape}")
print(
    f"  PME reciprocal charge gradients range: [{recip_charge_grads.min().item():.4f}, {recip_charge_grads.max().item():.4f}]"
)

# Compute real-space with charge gradients
real_energies, real_forces, real_charge_grads = ewald_real_space(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=params_comp.alpha,
    neighbor_list=nl_comp,
    neighbor_ptr=nptr_comp,
    neighbor_shifts=ns_comp,
    compute_forces=True,
    compute_charge_gradients=True,
)

print(
    f"  Real-space charge gradients range: [{real_charge_grads.min().item():.4f}, {real_charge_grads.max().item():.4f}]"
)

# Total charge gradient is the sum of components
total_charge_grads = real_charge_grads + recip_charge_grads
print(
    f"  Total charge gradients range: [{total_charge_grads.min().item():.4f}, {total_charge_grads.max().item():.4f}]"
)

# %%
# Full PME with charge gradients in one call:

energies_full, forces_full, charge_grads_full = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=nl_comp,
    neighbor_ptr=nptr_comp,
    neighbor_shifts=ns_comp,
    compute_forces=True,
    compute_charge_gradients=True,
    accuracy=1e-4,
)

print(
    f"\n  Full PME charge gradients range: [{charge_grads_full.min().item():.4f}, {charge_grads_full.max().item():.4f}]"
)

# %%
# Verify charge gradients against autograd:

charges.requires_grad_(True)
energies_total = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=nl_comp,
    neighbor_ptr=nptr_comp,
    neighbor_shifts=ns_comp,
    accuracy=1e-4,
).sum()

energies_total.backward()
autograd_charge_grads = charges.grad.clone()
charges.requires_grad_(False)
charges.grad = None

# Compare explicit vs autograd charge gradients
charge_grad_diff = (charge_grads_full - autograd_charge_grads).abs().max().item()
print(f"  Explicit vs Autograd charge gradient max diff: {charge_grad_diff:.2e}")

# %%
# Batch Evaluation
# ----------------
# Multiple systems can be evaluated simultaneously using batch_idx.

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
params_batch = estimate_pme_parameters(
    positions_batch, cells_batch, batch_idx=batch_idx, accuracy=1e-5
)

print(f"\nTotal atoms: {len(positions_batch)}")
print(f"Per-system alphas: {params_batch.alpha.tolist()}")
print(f"Mesh dimensions: {params_batch.mesh_dimensions}")
print(f"Real-space cutoff: {params_batch.real_space_cutoff.max().item():.2f} Å")

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

t0 = time.time()
energies_batch, forces_batch = particle_mesh_ewald(
    positions=positions_batch,
    charges=charges_batch,
    cell=cells_batch,
    batch_idx=batch_idx,
    neighbor_matrix=neighbor_matrix_batch,
    neighbor_matrix_shifts=neighbor_matrix_shifts_batch,
    compute_forces=True,
    accuracy=1e-5,  # Parameters estimated automatically for batch
)
t_batch = (time.time() - t0) * 1000

print(f"\nBatch evaluation time: {t_batch:.2f} ms")
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

    e_i = particle_mesh_ewald(
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
# PME vs Standard Ewald Comparison
# --------------------------------
# PME becomes more efficient than standard Ewald for larger systems due to
# its O(N log N) scaling vs O(N²) for explicit k-vector summation.
# Both methods use the same accuracy target for fair comparison.


system_sizes = [2, 3, 4]
accuracy_cmp = 1e-5

print(f"\nPME vs Ewald Performance (accuracy={accuracy_cmp:.0e}):")
print("  N_cells | N_atoms | Ewald (ms) | PME (ms)  | Energy diff")
print("  " + "-" * 60)

# %%
# Compare timing and accuracy:

for n_cells in system_sizes:
    pos_cmp, chrg_cmp, cell_cmp, pbc_cmp = create_nacl_system(n_cells=n_cells)

    # Estimate parameters for both methods
    ewald_params = estimate_ewald_parameters(pos_cmp, cell_cmp, accuracy=accuracy_cmp)
    pme_params = estimate_pme_parameters(pos_cmp, cell_cmp, accuracy=accuracy_cmp)

    # Use the larger cutoff to ensure both methods have same neighbors
    real_cutoff_cmp = max(
        ewald_params.real_space_cutoff.item(),
        pme_params.real_space_cutoff.item(),
    )

    nl_cmp, nptr_cmp, ns_cmp = neighbor_list_fn(
        pos_cmp, real_cutoff_cmp, cell=cell_cmp, pbc=pbc_cmp, return_neighbor_list=True
    )

    # Standard Ewald with automatic parameter estimation
    t0 = time.time()
    energies_ewald = ewald_summation(
        positions=pos_cmp,
        charges=chrg_cmp,
        cell=cell_cmp,
        neighbor_list=nl_cmp,
        neighbor_ptr=nptr_cmp,
        neighbor_shifts=ns_cmp,
        accuracy=accuracy_cmp,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_ewald = (time.time() - t0) * 1000

    # PME with automatic parameter estimation
    t0 = time.time()
    energies_pme = particle_mesh_ewald(
        positions=pos_cmp,
        charges=chrg_cmp,
        cell=cell_cmp,
        neighbor_list=nl_cmp,
        neighbor_ptr=nptr_cmp,
        neighbor_shifts=ns_cmp,
        accuracy=accuracy_cmp,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_pme = (time.time() - t0) * 1000

    e_diff = abs(energies_ewald.sum().item() - energies_pme.sum().item())
    print(
        f"    {n_cells}     | {len(pos_cmp):5d}   | {t_ewald:9.2f}  | {t_pme:8.2f}  | {e_diff:.2e}"
    )

print("\nNote: PME becomes increasingly efficient for larger systems.")

# %%
# Summary
# -------
# This example demonstrated:
#
# 1. **Automatic parameter estimation** for alpha and mesh dimensions using
#    ``estimate_pme_parameters`` with target accuracy
# 2. **Neighbor format flexibility** with list and matrix formats
# 3. **Accuracy-based convergence** showing how the accuracy parameter
#    controls both mesh resolution and real-space cutoff
# 4. **Accuracy-parameter relationships** for PME
# 5. **Component access** for real-space and reciprocal-space
# 6. **Charge gradients** (∂E/∂q_i) for ML potential training
# 7. **Batch evaluation** for multiple systems with automatic per-system alpha
# 8. **PME vs Ewald** performance comparison with same accuracy
#
# Key PME steps:
#
# - Charge spreading: :math:`Q(\\mathbf{x}) = \\sum_i q_i M_p(\\mathbf{x} - \\mathbf{r}_i)`
# - FFT convolution: :math:`\\tilde{\\Phi}(\\mathbf{k}) = G(\\mathbf{k}) \\tilde{Q}(\\mathbf{k})`
# - Force interpolation from mesh gradients
# - Charge gradient: :math:`\\frac{\\partial E}{\\partial q_i} = \\phi_i` (electrostatic potential)

print("\nPME example complete!")
