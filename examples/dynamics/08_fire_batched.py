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
Batched FIRE Optimization with LJ Clusters
==========================================

This example demonstrates **batched geometry optimization** using the FIRE
optimizer with two different batching strategies:

1. **batch_idx mode**: Each atom is tagged with a system index
   - Convenient for heterogeneous systems with different atom counts
   - Uses atomic accumulation (vf, vv, ff arrays must be zeroed each step)

2. **atom_ptr mode (CSR)**: Atom ranges defined by CSR-style pointers
   - More efficient for homogeneous batches
   - No cross-thread synchronization needed
   - Each system processed by a single thread

Both modes optimize multiple independent LJ clusters in parallel, with
per-system FIRE parameters (dt, alpha, counters) that adapt independently.

We use realistic Lennard-Jones argon clusters with neighbor list management,
demonstrating a complete batched optimization workflow.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from _dynamics_utils import (
    DEFAULT_CUTOFF,
    DEFAULT_SKIN,
    EPSILON_AR,
    MASS_AR,
    SIGMA_AR,
    BatchedMDSystem,
    create_random_box_cluster,
    mass_amu_to_internal,
)

from nvalchemiops.batch_utils import create_atom_ptr, create_batch_idx
from nvalchemiops.dynamics.optimizers import fire_step
from nvalchemiops.segment_ops import (
    segmented_max_norm,
    segmented_sum,
)

# ==============================================================================
# Main Example
# ==============================================================================

wp.init()
device = "cuda:0" if wp.is_cuda_available() else "cpu"
print(f"Using device: {device}")

# %%
# Create Batch of LJ Clusters
# ---------------------------
#
# We create multiple LJ argon clusters with different sizes.
# Each cluster is in its own periodic box (isolated cluster approach).

num_systems = 4
atom_counts = [16, 24, 20, 32]  # Different sizes per system
total_atoms = sum(atom_counts)

# Box size large enough to avoid self-interaction
box_L = 30.0  # Å
min_dist = 0.9 * SIGMA_AR  # Minimum distance for initial placement

print("\nBatch setup:")
print(f"  Number of systems: {num_systems}")
print(f"  Atoms per system: {atom_counts}")
print(f"  Total atoms: {total_atoms}")
print(f"  LJ parameters: ε = {EPSILON_AR:.4f} eV, σ = {SIGMA_AR:.2f} Å")

# Generate initial positions for all systems
all_positions = []
all_cells = []
for sys_id, count in enumerate(atom_counts):
    pos = create_random_box_cluster(count, box_L, min_dist, seed=42 + sys_id)
    all_positions.append(pos)
    all_cells.append(np.eye(3, dtype=np.float64) * box_L)

# Concatenate into batched arrays
positions_np = np.concatenate(all_positions, axis=0)
cells_np = np.stack(all_cells, axis=0)
batch_idx_np = np.concatenate(
    [np.full(count, sys_id, dtype=np.int32) for sys_id, count in enumerate(atom_counts)]
)

# Create masses (converted to internal units)
masses_np = mass_amu_to_internal(np.full(total_atoms, MASS_AR, dtype=np.float64))

# Create Warp arrays
positions = wp.array(positions_np, dtype=wp.vec3d, device=device)
velocities = wp.zeros(total_atoms, dtype=wp.vec3d, device=device)
masses = wp.array(masses_np, dtype=wp.float64, device=device)

# Create batching arrays using nvalchemiops.batch_utils
atom_counts_wp = wp.array(np.array(atom_counts, dtype=np.int32), device=device)
atom_ptr = wp.zeros(num_systems + 1, dtype=wp.int32, device=device)
create_atom_ptr(atom_counts_wp, atom_ptr)
batch_idx = wp.zeros(total_atoms, dtype=wp.int32, device=device)
create_batch_idx(atom_ptr, batch_idx)

print(f"  batch_idx shape: {batch_idx.shape}")
print(f"  atom_ptr shape: {atom_ptr.shape}")
print(f"  atom_ptr values: {atom_ptr.numpy()}")

# Create batched MD system (using BatchedMDSystem)
lj_system = BatchedMDSystem(
    positions=positions_np,
    cells=cells_np,
    batch_idx=batch_idx_np,
    num_systems=num_systems,
    masses=masses_np,
    epsilon=EPSILON_AR,
    sigma=SIGMA_AR,
    cutoff=DEFAULT_CUTOFF,
    skin=DEFAULT_SKIN,
    switch_width=1.0,  # Smooth cutoff for optimization
    device=device,
)

# %%
# FIRE Parameters (per-system)
# ----------------------------
#
# Each system has its own FIRE parameters that adapt independently.

dt0 = 1.0
dt_max = 10.0
dt_min = 1e-3
alpha0 = 0.1
f_inc = 1.1
f_dec = 0.5
f_alpha = 0.99
n_min = 5
maxstep = 0.2 * SIGMA_AR  # Max step in Å

# Per-system arrays (shape (B,) for batched mode)
dt = wp.array([dt0] * num_systems, dtype=wp.float64, device=device)
alpha = wp.array([alpha0] * num_systems, dtype=wp.float64, device=device)
alpha_start = wp.array([alpha0] * num_systems, dtype=wp.float64, device=device)
f_alpha_arr = wp.array([f_alpha] * num_systems, dtype=wp.float64, device=device)
dt_min_arr = wp.array([dt_min] * num_systems, dtype=wp.float64, device=device)
dt_max_arr = wp.array([dt_max] * num_systems, dtype=wp.float64, device=device)
maxstep_arr = wp.array([maxstep] * num_systems, dtype=wp.float64, device=device)
n_steps_positive = wp.zeros(num_systems, dtype=wp.int32, device=device)
n_min_arr = wp.array([n_min] * num_systems, dtype=wp.int32, device=device)
f_dec_arr = wp.array([f_dec] * num_systems, dtype=wp.float64, device=device)
f_inc_arr = wp.array([f_inc] * num_systems, dtype=wp.float64, device=device)

# Accumulators for batch_idx mode (shape (B,))
vf = wp.zeros(num_systems, dtype=wp.float64, device=device)
vv = wp.zeros(num_systems, dtype=wp.float64, device=device)
ff = wp.zeros(num_systems, dtype=wp.float64, device=device)
uphill_flag = wp.zeros(num_systems, dtype=wp.int32, device=device)


# %%
# Method 1: batch_idx Optimization
# --------------------------------
#
# In batch_idx mode, each atom is tagged with its system index.
# The kernel uses atomic operations to accumulate vf, vv, ff per system.

print("\n" + "=" * 80)
print("METHOD 1: batch_idx BATCHING")
print("=" * 80)

# Reset state for batch_idx run
positions_bidx = wp.array(positions_np.copy(), dtype=wp.vec3d, device=device)
velocities_bidx = wp.zeros(total_atoms, dtype=wp.vec3d, device=device)
dt_bidx = wp.array([dt0] * num_systems, dtype=wp.float64, device=device)
alpha_bidx = wp.array([alpha0] * num_systems, dtype=wp.float64, device=device)
n_steps_pos_bidx = wp.zeros(num_systems, dtype=wp.int32, device=device)

# Update LJ system positions
wp.copy(lj_system.wp_positions, positions_bidx)

max_steps = 2000
force_tol = 1e-3  # eV/Å
log_interval = 200
check_interval = 50

# History
bidx_energy_hist = []
bidx_maxf_hist = []

print(f"\nRunning batch_idx optimization ({max_steps} max steps)...")
print(f"Force tolerance: {force_tol:.1e} eV/Å")
print("-" * 70)
print(f"{'Step':>6} {'Total E':>14} {'max|F|':>12} {'Converged':>12}")
print("-" * 70)

for step in range(max_steps):
    # Update positions in LJ system and compute forces
    wp.copy(lj_system.wp_positions, positions_bidx)
    energies = lj_system.compute_forces()

    # Zero accumulators before each step (required for batch_idx mode)
    vf.zero_()
    vv.zero_()
    ff.zero_()

    # FIRE step with batch_idx
    fire_step(
        positions=positions_bidx,
        velocities=velocities_bidx,
        forces=lj_system.wp_forces,
        masses=masses,
        alpha=alpha_bidx,
        dt=dt_bidx,
        alpha_start=alpha_start,
        f_alpha=f_alpha_arr,
        dt_min=dt_min_arr,
        dt_max=dt_max_arr,
        maxstep=maxstep_arr,
        n_steps_positive=n_steps_pos_bidx,
        n_min=n_min_arr,
        f_dec=f_dec_arr,
        f_inc=f_inc_arr,
        uphill_flag=uphill_flag,
        vf=vf,
        vv=vv,
        ff=ff,
        batch_idx=batch_idx,
        device=device,
    )

    # Check convergence at intervals
    if step % check_interval == 0 or step == max_steps - 1:
        # Use GPU-accelerated segmented ops for reductions
        system_energies = wp.zeros(num_systems, dtype=wp.float64, device=device)
        segmented_sum(energies, batch_idx, system_energies)
        max_forces = wp.zeros(num_systems, dtype=wp.float64, device=device)
        segmented_max_norm(lj_system.wp_forces, batch_idx, max_forces)

        # Sync and convert to numpy for logging
        wp.synchronize()
        system_energies_np = system_energies.numpy()
        max_forces_np = max_forces.numpy()

        total_energy = system_energies_np.sum()
        global_max_f = max_forces_np.max()
        num_converged = (max_forces_np < force_tol).sum()

        bidx_energy_hist.append(total_energy)
        bidx_maxf_hist.append(global_max_f)

        if step % log_interval == 0 or step == max_steps - 1:
            print(
                f"{step:>6d} {total_energy:>14.6f} {global_max_f:>12.2e} {num_converged:>8d}/{num_systems}"
            )

        if num_converged == num_systems:
            print(f"\nAll systems converged at step {step}!")
            break

print(f"\nFinal per-system energies (batch_idx): {system_energies_np}")
print(f"Final max forces per system: {max_forces_np}")


# %%
# Method 2: atom_ptr Optimization
# -------------------------------
#
# In atom_ptr (CSR) mode, atom ranges are defined by pointers.
# Each system is processed by a single thread (no cross-thread sync needed).
# Note: vf, vv, ff are NOT used in this mode.

print("\n" + "=" * 80)
print("METHOD 2: atom_ptr (CSR) BATCHING")
print("=" * 80)

# Reset state for atom_ptr run
positions_ptr = wp.array(positions_np.copy(), dtype=wp.vec3d, device=device)
velocities_ptr = wp.zeros(total_atoms, dtype=wp.vec3d, device=device)
dt_ptr = wp.array([dt0] * num_systems, dtype=wp.float64, device=device)
alpha_ptr = wp.array([alpha0] * num_systems, dtype=wp.float64, device=device)
n_steps_pos_ptr = wp.zeros(num_systems, dtype=wp.int32, device=device)

# History
ptr_energy_hist = []
ptr_maxf_hist = []

print(f"\nRunning atom_ptr optimization ({max_steps} max steps)...")
print(f"Force tolerance: {force_tol:.1e} eV/Å")
print("-" * 70)
print(f"{'Step':>6} {'Total E':>14} {'max|F|':>12} {'Converged':>12}")
print("-" * 70)

for step in range(max_steps):
    # Update positions in LJ system and compute forces
    wp.copy(lj_system.wp_positions, positions_ptr)
    energies = lj_system.compute_forces()

    # FIRE step with atom_ptr (NO accumulators needed!)
    fire_step(
        positions=positions_ptr,
        velocities=velocities_ptr,
        forces=lj_system.wp_forces,
        masses=masses,
        alpha=alpha_ptr,
        dt=dt_ptr,
        alpha_start=alpha_start,
        f_alpha=f_alpha_arr,
        dt_min=dt_min_arr,
        dt_max=dt_max_arr,
        maxstep=maxstep_arr,
        n_steps_positive=n_steps_pos_ptr,
        n_min=n_min_arr,
        f_dec=f_dec_arr,
        f_inc=f_inc_arr,
        uphill_flag=uphill_flag,
        atom_ptr=atom_ptr,  # Use atom_ptr instead of batch_idx
        device=device,
    )

    # Check convergence at intervals
    if step % check_interval == 0 or step == max_steps - 1:
        # Use GPU-accelerated segmented ops for reductions
        system_energies = wp.zeros(num_systems, dtype=wp.float64, device=device)
        segmented_sum(energies, batch_idx, system_energies)
        max_forces = wp.zeros(num_systems, dtype=wp.float64, device=device)
        segmented_max_norm(lj_system.wp_forces, batch_idx, max_forces)

        # Sync and convert to numpy for logging
        wp.synchronize()
        system_energies_np = system_energies.numpy()
        max_forces_np = max_forces.numpy()

        total_energy = system_energies_np.sum()
        global_max_f = max_forces_np.max()
        num_converged = (max_forces_np < force_tol).sum()

        ptr_energy_hist.append(total_energy)
        ptr_maxf_hist.append(global_max_f)

        if step % log_interval == 0 or step == max_steps - 1:
            print(
                f"{step:>6d} {total_energy:>14.6f} {global_max_f:>12.2e} {num_converged:>8d}/{num_systems}"
            )

        if num_converged == num_systems:
            print(f"\nAll systems converged at step {step}!")
            break

print(f"\nFinal per-system energies (atom_ptr): {system_energies_np}")
print(f"Final max forces per system: {max_forces_np}")


# %%
# Compare Results
# ---------------

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

wp.synchronize()
pos_bidx_np = positions_bidx.numpy()
pos_ptr_np = positions_ptr.numpy()

# Check that both methods converged to similar positions
# (Note: may differ slightly due to different convergence paths)
max_diff = np.max(np.abs(pos_bidx_np - pos_ptr_np))
print(f"\nMax position difference between methods: {max_diff:.2e} Å")

# Final energies comparison
print(f"\nFinal total energy (batch_idx): {bidx_energy_hist[-1]:.6f} eV")
print(f"Final total energy (atom_ptr):  {ptr_energy_hist[-1]:.6f} eV")
print(f"Energy difference: {abs(bidx_energy_hist[-1] - ptr_energy_hist[-1]):.2e} eV")


# %%
# Plot Convergence
# ----------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# Energy convergence
steps_bidx = np.arange(len(bidx_energy_hist)) * check_interval
steps_ptr = np.arange(len(ptr_energy_hist)) * check_interval

axes[0].plot(steps_bidx, bidx_energy_hist, "b-", lw=2, label="batch_idx")
axes[0].plot(steps_ptr, ptr_energy_hist, "r--", lw=2, label="atom_ptr")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Total Energy (eV)")
axes[0].set_title("Energy Convergence")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Force convergence
axes[1].semilogy(steps_bidx, bidx_maxf_hist, "b-", lw=2, label="batch_idx")
axes[1].semilogy(steps_ptr, ptr_maxf_hist, "r--", lw=2, label="atom_ptr")
axes[1].axhline(
    force_tol, color="k", ls="--", lw=1, label=f"tolerance ({force_tol:.0e})"
)
axes[1].set_xlabel("Step")
axes[1].set_ylabel("max|F| (eV/Å)")
axes[1].set_title("Force Convergence")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle("Batched FIRE: LJ Cluster Optimization", fontsize=14)
plt.show()


# %%
# Summary
# -------
#
# **When to use batch_idx:**
# - Heterogeneous batches (different atom counts per system)
# - When you already have per-atom system tags from your data pipeline
# - Simple setup: just create the index array
#
# **When to use atom_ptr (CSR):**
# - Homogeneous or semi-homogeneous batches
# - Maximum performance (no atomic operations)
# - When atoms are naturally stored contiguously per system
#
# Both methods give equivalent optimization results but may have different
# performance characteristics depending on batch size and system heterogeneity.

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
batch_idx mode:
  - Each atom tagged with system index
  - Uses atomic accumulation (vf, vv, ff must be zeroed each step)
  - Good for heterogeneous systems
  - Simple to set up

atom_ptr (CSR) mode:
  - Atom ranges defined by CSR pointers
  - No accumulator arrays needed (cleaner API)
  - Each system processed by single thread
  - More efficient for large batches

Both methods support:
  - Per-system FIRE parameters that adapt independently
  - Downhill check (optional) for energy-based rollback
  - Compatible with any force computation (LJ, NN potentials, etc.)
""")
