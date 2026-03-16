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
Particle Mesh Ewald (PME) with JAX
==================================

This example demonstrates how to compute long-range electrostatic interactions
using the Particle Mesh Ewald (PME) method with the JAX backend. PME achieves
O(N log N) scaling through FFT-based mesh interpolation.

In this example you will learn:

- How to set up and run PME with automatic parameter estimation in JAX
- Using neighbor list (COO) and neighbor matrix formats
- Accessing real-space and reciprocal-space components separately
- Computing charge gradients for ML potential training
- ``jax.jit`` compilation of the full neighbor list + PME pipeline

.. important::
    This script is intended as an API demonstration. Do not use this script
    for performance benchmarking; refer to the `benchmarks` folder instead.
"""

# %%
# Setup and Imports
# -----------------
# Import JAX and the nvalchemiops electrostatics API.

from __future__ import annotations

import sys
import time

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print(
        "This example requires JAX. Install with: pip install 'nvalchemi-toolkit-ops[jax]'"
    )
    sys.exit(0)

import numpy as np

try:
    from nvalchemiops.jax.interactions.electrostatics import (
        estimate_pme_parameters,
        ewald_real_space,
        particle_mesh_ewald,
        pme_reciprocal_space,
    )
    from nvalchemiops.jax.neighbors import neighbor_list
    from nvalchemiops.jax.neighbors.naive import naive_neighbor_list
    from nvalchemiops.jax.neighbors.neighbor_utils import compute_naive_num_shifts
except Exception as exc:
    print(
        f"JAX/Warp backend unavailable ({exc}). This example requires a CUDA-backed runtime."
    )
    sys.exit(0)

# %%
# Check Device
# ------------

print("=" * 70)
print("JAX PME ELECTROSTATICS EXAMPLE")
print("=" * 70)

devices = jax.devices()
print(f"\nJAX devices: {devices}")
print(f"Default device: {jax.default_backend()}")

# %%
# Create a NaCl Crystal System
# ----------------------------
# We define a helper function to create NaCl rock salt crystal supercells.


def create_nacl_system(n_cells: int = 3, lattice_constant: float = 5.64):
    """Create a NaCl crystal supercell.

    Parameters
    ----------
    n_cells : int
        Number of unit cells in each direction.
    lattice_constant : float
        NaCl lattice constant in Angstroms.

    Returns
    -------
    positions, charges, cell, pbc : jax.Array
        System arrays with float64 dtype for electrostatics.
    """
    base_positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    base_charges = np.array([1.0, -1.0])

    positions_list = []
    charges_list = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                offset = np.array([i, j, k])
                for pos, charge in zip(base_positions, base_charges):
                    positions_list.append((pos + offset) * lattice_constant)
                    charges_list.append(charge)

    # Convert to JAX arrays with float64 for electrostatics accuracy
    positions = jnp.array(positions_list, dtype=jnp.float64)
    charges = jnp.array(charges_list, dtype=jnp.float64)
    cell = jnp.eye(3, dtype=jnp.float64) * lattice_constant * n_cells
    cell = cell[None, ...]  # Add batch dimension: (1, 3, 3)
    pbc = jnp.array([[True, True, True]])

    return positions, charges, cell, pbc


# %%
# Basic PME with Automatic Parameters
# -----------------------------------
# The simplest way to use PME is with automatic parameter estimation.

print("\n" + "=" * 70)
print("BASIC PME WITH AUTOMATIC PARAMETERS")
print("=" * 70)

# Create a NaCl crystal (3×3×3 unit cells = 54 atoms)
positions, charges, cell, pbc = create_nacl_system(n_cells=3)

print(f"\nSystem: {len(positions)} atoms NaCl crystal")
print(f"Cell size: {float(cell[0, 0, 0]):.2f} Å")
print(f"Total charge: {float(charges.sum()):.1f} (should be 0 for neutral)")

# %%
# Estimate optimal PME parameters:

params = estimate_pme_parameters(positions, cell, accuracy=1e-6)

print("\nEstimated parameters (accuracy=1e-6):")
print(f"  alpha = {float(params.alpha[0]):.4f}")
print(f"  mesh_dimensions = {params.mesh_dimensions}")
print(
    f"  mesh_spacing = ({float(params.mesh_spacing[0, 0]):.2f}, "
    f"{float(params.mesh_spacing[0, 1]):.2f}, {float(params.mesh_spacing[0, 2]):.2f}) Å"
)
print(f"  real_space_cutoff = {float(params.real_space_cutoff[0]):.2f} Å")

# %%
# Build neighbor list and run PME:

cutoff = float(params.real_space_cutoff[0])
nl, nptr, ns = neighbor_list(
    positions,
    cutoff,
    cell=cell,
    pbc=pbc,
    return_neighbor_list=True,
)

energies, forces = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=nl,
    neighbor_ptr=nptr,
    neighbor_shifts=ns,
    compute_forces=True,
    accuracy=1e-6,
)

total_energy = float(energies.sum())
max_force = float(jnp.linalg.norm(forces, axis=1).max())

print("\nPME Results:")
print(f"  Total energy: {total_energy:.6f}")
print(f"  Energy per atom: {total_energy / len(positions):.6f}")
print(f"  Max force magnitude: {max_force:.6f}")

# %%
# Neighbor Matrix vs COO Format Comparison
# ----------------------------------------
# PME supports both neighbor formats, producing identical results.

print("\n" + "=" * 70)
print("NEIGHBOR FORMAT COMPARISON")
print("=" * 70)

# Build both formats using the estimated real-space cutoff
# COO format (neighbor list)
nl_coo, nptr_coo, ns_coo = neighbor_list(
    positions,
    cutoff,
    cell=cell,
    pbc=pbc,
    return_neighbor_list=True,
)

# Dense format (neighbor matrix)
nm_dense, num_dense, ns_dense = neighbor_list(
    positions,
    cutoff,
    cell=cell,
    pbc=pbc,
    return_neighbor_list=False,
)

print(f"\nUsing alpha={float(params.alpha[0]):.4f}, mesh_dims={params.mesh_dimensions}")

# %%
# Using neighbor list (COO) format:

energies_coo, forces_coo = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=nl_coo,
    neighbor_ptr=nptr_coo,
    neighbor_shifts=ns_coo,
    compute_forces=True,
    accuracy=1e-6,
)

print(f"  COO format: E={float(energies_coo.sum()):.6f}")

# %%
# Using neighbor matrix (dense) format:

energies_dense, forces_dense = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_matrix=nm_dense,
    neighbor_matrix_shifts=ns_dense,
    compute_forces=True,
    accuracy=1e-6,
)

print(f"  Dense format: E={float(energies_dense.sum()):.6f}")

# Compare results
energy_diff = abs(float(energies_coo.sum()) - float(energies_dense.sum()))
force_diff = float(jnp.abs(forces_coo - forces_dense).max())

print(f"\nEnergy difference: {energy_diff:.2e}")
print(f"Max force difference: {force_diff:.2e}")

# %%
# Real-Space and Reciprocal-Space Components
# ------------------------------------------
# You can compute the components separately if needed.

print("\n" + "=" * 70)
print("ENERGY COMPONENTS")
print("=" * 70)

# Use lower accuracy for this demo to speed up parameter estimation
params_comp = estimate_pme_parameters(positions, cell, accuracy=1e-4)
cutoff_comp = float(params_comp.real_space_cutoff[0])

nl_comp, nptr_comp, ns_comp = neighbor_list(
    positions,
    cutoff_comp,
    cell=cell,
    pbc=pbc,
    return_neighbor_list=True,
)

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

print(f"\n  Real-space: {float(real_energy.sum()):.6f}")

# %%
# PME reciprocal-space component (FFT-based):

recip_energy = pme_reciprocal_space(
    positions=positions,
    charges=charges,
    cell=cell,
    alpha=params_comp.alpha,
    mesh_dimensions=params_comp.mesh_dimensions,
)

print(f"  Reciprocal-space (PME): {float(recip_energy.sum()):.6f}")
print(f"  Total (sum): {float(real_energy.sum() + recip_energy.sum()):.6f}")

# %%
# Compare with full PME:

full_pme_energy = particle_mesh_ewald(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_list=nl_comp,
    neighbor_ptr=nptr_comp,
    neighbor_shifts=ns_comp,
    accuracy=1e-4,
)

print(f"  Full PME: {float(full_pme_energy.sum()):.6f}")

component_diff = abs(
    float(real_energy.sum() + recip_energy.sum()) - float(full_pme_energy.sum())
)
print(f"\n  Component sum vs full PME difference: {component_diff:.2e}")

# %%
# Charge Gradients for ML Potentials
# ----------------------------------
# PME supports computing analytical charge gradients (∂E/∂q_i), which are useful
# for training machine learning potentials that predict atomic partial charges.

print("\n" + "=" * 70)
print("CHARGE GRADIENTS")
print("=" * 70)

# Compute PME with charge gradients
energies_cg, forces_cg, charge_grads = particle_mesh_ewald(
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

print(f"\n  Charge gradients shape: {charge_grads.shape}")
print(
    f"  Charge gradients range: [{float(charge_grads.min()):.4f}, "
    f"{float(charge_grads.max()):.4f}]"
)
print(f"  Charge gradients mean: {float(charge_grads.mean()):.4f}")

# The charge gradient represents dE/dq for each atom
# For neutral systems, the sum should be close to zero due to symmetry
print(f"  Sum of charge gradients: {float(charge_grads.sum()):.4e}")

# %%
# Verify by checking gradient symmetry for Na+ and Cl- ions:

na_grads = charge_grads[charges > 0]  # Na+ ions
cl_grads = charge_grads[charges < 0]  # Cl- ions

print(f"\n  Na+ charge gradients mean: {float(na_grads.mean()):.4f}")
print(f"  Cl- charge gradients mean: {float(cl_grads.mean()):.4f}")

# %%
# JIT Compilation
# ---------------
# Demonstrate combining the neighbor list build and PME calculation into a
# single ``jax.jit``-compiled function. This allows JAX to fuse the entire
# pipeline into one optimized computation.
#
# For JIT compatibility:
#
# - ``max_neighbors`` must be specified (static array shapes)
# - ``mesh_dimensions`` must be a concrete tuple (static FFT sizes)
# - ``alpha`` can be a traced JAX array
# - ``compute_forces`` and other boolean flags must be static
# - Parameter estimation (``estimate_pme_parameters``) should happen **outside**
#   the jitted function since it determines array shapes
# - Periodic shift metadata (``shift_range``, ``num_shifts_per_system``,
#   ``max_shifts_per_system``) must be pre-computed outside jit using
#   ``compute_naive_num_shifts``, since the launch dimensions must be concrete

print("\n" + "=" * 70)
print("JIT COMPILATION")
print("=" * 70)

# First, estimate parameters outside jit (determines static shapes)
jit_positions, jit_charges, jit_cell, jit_pbc = create_nacl_system(n_cells=3)
jit_params = estimate_pme_parameters(jit_positions, jit_cell, accuracy=1e-5)
jit_cutoff = float(jit_params.real_space_cutoff[0])
jit_mesh_dims = tuple(int(x) for x in jit_params.mesh_dimensions)
jit_alpha = jit_params.alpha

# Pre-compute shift metadata outside jit (launch sizes must be concrete)
shift_range, num_shifts_per_system, max_shifts_per_system = compute_naive_num_shifts(
    jit_cell, jit_cutoff, jit_pbc
)


# Define a function that builds neighbors and computes PME
# We will compare the performance of the jitted and non-jitted versions.
def compute_pme_energy_forces(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
    alpha: jax.Array,
    shift_range: jax.Array = shift_range,
    num_shifts_per_system: jax.Array = num_shifts_per_system,
    cutoff: float = jit_cutoff,
    max_neighbors: int = 128,
    max_shifts_per_system: int = max_shifts_per_system,
    mesh_dimensions: tuple[int, int, int] = jit_mesh_dims,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled neighbor list + PME pipeline."""
    # Build neighbor matrix inside jit (max_neighbors must be static,
    # shift metadata pre-computed outside jit)
    neighbor_matrix, _, neighbor_matrix_shifts = naive_neighbor_list(
        positions,
        cutoff,
        cell=cell,
        pbc=pbc,
        max_neighbors=max_neighbors,
        shift_range_per_dimension=shift_range,
        num_shifts_per_system=num_shifts_per_system,
        max_shifts_per_system=max_shifts_per_system,
    )

    # Compute PME (mesh_dimensions is static, alpha is traced)
    energies, forces = particle_mesh_ewald(
        positions=positions,
        charges=charges,
        cell=cell,
        alpha=alpha,
        mesh_dimensions=mesh_dimensions,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        compute_forces=True,
    )

    return energies, forces


jit_compute_pme_energy_forces = jax.jit(compute_pme_energy_forces)

# %%
# Run the non-jitted function:
energies, forces = compute_pme_energy_forces(
    jit_positions, jit_charges, jit_cell, jit_pbc, jit_alpha
)
total_energy = float(energies.sum())
max_force = float(jnp.linalg.norm(forces, axis=1).max())
print(f"  Non-jitted total energy: {total_energy:.6f}")
print(f"  Non-jitted max force: {max_force:.6f}")

# Calculate Performance
# Warmup measurements
for _ in range(10):
    energies, forces = compute_pme_energy_forces(
        jit_positions, jit_charges, jit_cell, jit_pbc, jit_alpha
    )
energies.block_until_ready()
forces.block_until_ready()

# Timed measurements
start_time = time.time()
for _ in range(50):
    energies, forces = compute_pme_energy_forces(
        jit_positions, jit_charges, jit_cell, jit_pbc, jit_alpha
    )
energies.block_until_ready()
forces.block_until_ready()
total_time = time.time() - start_time
print(f"  Non-jitted average time per call: {total_time / 50:.6f} seconds")

# %%
# Run the jitted function:

print("\nCompiling and running jitted PME pipeline...")
jit_energies, jit_forces = jit_compute_pme_energy_forces(
    jit_positions, jit_charges, jit_cell, jit_pbc, jit_alpha
)

jit_total_energy = float(jit_energies.sum())
jit_max_force = float(jnp.linalg.norm(jit_forces, axis=1).max())

print(f"  JIT total energy: {jit_total_energy:.6f}")
print(f"  JIT max force: {jit_max_force:.6f}")

# Calculate Performance
# Warmup measurements
for _ in range(10):
    jit_energies, jit_forces = jit_compute_pme_energy_forces(
        jit_positions, jit_charges, jit_cell, jit_pbc, jit_alpha
    )
jit_energies.block_until_ready()
jit_forces.block_until_ready()

# Timed measurements
start_time = time.time()
for _ in range(50):
    jit_energies, jit_forces = jit_compute_pme_energy_forces(
        jit_positions, jit_charges, jit_cell, jit_pbc, jit_alpha
    )
jit_energies.block_until_ready()
jit_forces.block_until_ready()
total_time = time.time() - start_time
print(f"  JIT average time per call: {total_time / 50:.6f} seconds")

# Compare with non-jitted result (note: may differ slightly due to different
# accuracy settings or neighbor list truncation from max_neighbors)
energy_diff_jit = abs(jit_total_energy - total_energy)
print(f"  Difference vs non-jitted (different accuracy): {energy_diff_jit:.2e}")


# %%
# Summary
# -------
# This example demonstrated:
#
# 1. **Automatic parameter estimation** for alpha and mesh dimensions using
#    ``estimate_pme_parameters`` with target accuracy
# 2. **Neighbor format flexibility** with COO (list) and dense (matrix) formats
# 3. **Component access** for real-space and reciprocal-space separately
# 4. **Charge gradients** (∂E/∂q_i) for ML potential training
# 5. **JIT compilation** of the full neighbor list + PME pipeline
#
# Key JAX-specific patterns:
#
# - Use ``jnp.float64`` for electrostatics calculations
# - Cell shape is ``(1, 3, 3)`` with batch dimension
# - Use ``float()`` to extract scalar values from JAX arrays for printing
# - Parameters from ``estimate_pme_parameters`` are JAX arrays
# - For ``jax.jit``: estimate parameters outside, pass ``max_neighbors``
#   and ``mesh_dimensions`` as static values

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nKey takeaways:")
print("  - Use estimate_pme_parameters() for automatic parameter selection")
print("  - Both COO and dense neighbor formats produce identical results")
print("  - Real and reciprocal components can be computed separately")
print("  - Charge gradients are available for ML potential training")
print("  - Use jax.jit to fuse neighbor list + PME into one compiled function")
print("\nJAX PME example completed successfully!")
