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
Langevin Dynamics with Lennard-Jones Potential
==============================================

This example demonstrates GPU-accelerated molecular dynamics using the
Langevin (BAOAB) thermostat with a Lennard-Jones potential.

We simulate liquid argon at 94.4 K (near the triple point) to demonstrate:

1. Building an FCC lattice system
2. Computing neighbor lists with periodic boundaries
3. Using the Lennard-Jones potential for energy and forces
4. Running Langevin dynamics in the NVT ensemble
5. Monitoring thermodynamic properties

Physical System: Liquid Argon
-----------------------------
- LJ parameters: ε = 0.0104 eV, σ = 3.40 Å
- Temperature: 94.4 K (near triple point)
- Density: ~1.4 g/cm³

The BAOAB integrator uses the splitting:
B (velocity half-step) → A (position half-step) → O (Ornstein-Uhlenbeck) →
A (position half-step) → B (velocity half-step with new forces)

This provides excellent configurational sampling for the NVT ensemble.
"""

# %%
# Imports
# -------
# We import utilities from nvalchemiops for the Lennard-Jones potential,
# neighbor list construction, and MD integrators.

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

# Local utilities for this example
from _dynamics_utils import (
    DEFAULT_CUTOFF,
    DEFAULT_SKIN,
    EPSILON_AR,
    SIGMA_AR,
    MDSystem,
    create_fcc_argon,
    run_langevin_baoab,
)

from nvalchemiops.dynamics.utils import compute_kinetic_energy

# %%
# System setup
# ------------

print("=" * 95)
print("LANGEVIN DYNAMICS WITH LENNARD-JONES POTENTIAL")
print("=" * 95)
print()

device = "cuda:0" if wp.is_cuda_available() else "cpu"
print(f"Using device: {device}")

# Create FCC argon system (4x4x4 unit cells = 256 atoms)
print("\n--- Creating FCC Argon System ---")
positions, cell = create_fcc_argon(num_unit_cells=4, a=5.26)
print(f"Created {len(positions)} atoms in {cell[0, 0]:.2f} Å³ box")

print("\n--- Initializing MD System ---")
system = MDSystem(
    positions=positions,
    cell=cell,
    epsilon=EPSILON_AR,
    sigma=SIGMA_AR,
    cutoff=DEFAULT_CUTOFF,
    skin=DEFAULT_SKIN,
    device=device,
    dtype=np.float64,
)

print("\n--- Setting Initial Temperature ---")
system.initialize_temperature(temperature=94.4, seed=42)

print("\n--- Initial Energy Calculation ---")
wp_energies = system.compute_forces()
pe = float(wp_energies.numpy().sum())
ke_arr = wp.zeros(1, dtype=wp.float64, device=device)
compute_kinetic_energy(
    velocities=system.wp_velocities,
    masses=system.wp_masses,
    kinetic_energy=ke_arr,
    device=device,
)
ke = float(ke_arr.numpy()[0])
print(f"  Kinetic Energy:   {ke:>12.4f} eV")
print(f"  Potential Energy: {pe:>12.4f} eV")
print(f"  Total Energy:     {ke + pe:>12.4f} eV")
print(f"  Neighbors:        {system.neighbor_manager.total_neighbors()}")

# %%
# Langevin dynamics (BAOAB)
# ------------------------

print("\n--- Equilibration (500 steps) ---")
_eq_stats = run_langevin_baoab(
    system=system,
    num_steps=500,
    dt_fs=1.0,
    temperature_K=94.4,
    friction_per_fs=0.01,
    log_interval=100,
    seed=42,
)

print("\n--- Production Run (2000 steps) ---")
prod_stats = run_langevin_baoab(
    system=system,
    num_steps=2000,
    dt_fs=1.0,
    temperature_K=94.4,
    friction_per_fs=0.01,
    log_interval=200,
    seed=1000,
)

# %%
# Analysis
# --------

print("\n--- Analysis ---")
temps = np.array([s.temperature for s in prod_stats])
total_energies = np.array([s.total_energy for s in prod_stats])
steps = np.array([s.step for s in prod_stats])

print(f"  Mean Temperature:    {temps.mean():.2f} ± {temps.std():.2f} K")
print("  Target Temperature:  94.4 K")
print(f"  Temperature Error:   {abs(temps.mean() - 94.4) / 94.4 * 100:.2f}%")
print()
print(f"  Mean Total Energy:   {total_energies.mean():.4f} eV")
print(f"  Energy Fluctuation:  {total_energies.std():.4f} eV")

# %%
# Plot
# ---------------------------------------------------

fig, ax = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True, constrained_layout=True)
ax[0].plot(steps, temps, lw=1.5)
ax[0].axhline(94.4, color="k", ls="--", lw=1.0, label="target")
ax[0].set_ylabel("Temperature (K)")
ax[0].legend(frameon=False, loc="best")

ax[1].plot(steps, total_energies, lw=1.5)
ax[1].set_xlabel("Step")
ax[1].set_ylabel("Total Energy (eV)")
fig.suptitle("Langevin (BAOAB): Temperature and Total Energy")

print("\n" + "=" * 95)
print("SIMULATION COMPLETE")
print("=" * 95)
