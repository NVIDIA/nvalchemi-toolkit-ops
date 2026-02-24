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
Velocity Verlet Dynamics with Lennard-Jones Potential
=====================================================

This example demonstrates GPU-accelerated molecular dynamics using the
Velocity Verlet integrator with a Lennard-Jones (LJ) potential.

We simulate argon in the **NVE** ensemble (no thermostat) to demonstrate:

1. Building an FCC lattice system
2. Computing neighbor lists with periodic boundaries
3. Using the LJ potential for energy and forces
4. Running Velocity Verlet dynamics
5. Monitoring energy conservation and basic stability diagnostics

Notes
-----
Velocity Verlet is stable and time-reversible, but for LJ systems it is still
sensitive to:
- timestep size,
- neighbor list rebuild logic,
- force discontinuities at the cutoff (consider using `switch_width > 0`).
"""

# %%
# Imports
# -------

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from _dynamics_utils import (
    DEFAULT_CUTOFF,
    DEFAULT_SKIN,
    EPSILON_AR,
    SIGMA_AR,
    MDSystem,
    create_fcc_argon,
    run_velocity_verlet,
)

# %%
# System setup
# ------------

print("=" * 95)
print("VELOCITY VERLET DYNAMICS WITH LENNARD-JONES POTENTIAL (NVE)")
print("=" * 95)
print()

device = "cuda:0" if wp.is_cuda_available() else "cpu"
print(f"Using device: {device}")

print("\n--- Creating FCC Argon System ---")
positions, cell = create_fcc_argon(num_unit_cells=4, a=5.26)  # 256 atoms
print(f"Created {len(positions)} atoms in {cell[0, 0]:.2f} Å³ box")

print("\n--- Initializing MD System ---")
system = MDSystem(
    positions=positions,
    cell=cell,
    epsilon=EPSILON_AR,
    sigma=SIGMA_AR,
    cutoff=DEFAULT_CUTOFF,
    skin=DEFAULT_SKIN,
    # NVE can benefit from smooth cutoffs; keep hard cutoff by default for now.
    switch_width=0.0,
    device=device,
    dtype=np.float64,
)

print("\n--- Setting Initial Temperature ---")
system.initialize_temperature(temperature=94.4, seed=42)

# %%
# Velocity Verlet (NVE)
# ---------------------

print("\n--- NVE Run (3000 steps) ---")
stats = run_velocity_verlet(
    system=system,
    num_steps=3000,
    dt_fs=1.0,
    log_interval=200,
)

# %%
# Analysis
# --------

print("\n--- Analysis ---")
temps = np.array([s.temperature for s in stats])
total_energies = np.array([s.total_energy for s in stats])
steps = np.array([s.step for s in stats])

e0 = total_energies[0]
drift = total_energies - e0

print(f"  Mean Temperature:    {temps.mean():.2f} ± {temps.std():.2f} K")
print(f"  Mean Total Energy:   {total_energies.mean():.4f} eV")
print(f"  Energy Fluctuation:  {total_energies.std():.4f} eV")
print(
    f"  Energy Drift (last): {drift[-1]:.6f} eV  (relative: {drift[-1] / e0 * 100:.3e}%)"
)

fig, ax = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True, constrained_layout=True)
ax[0].plot(steps, temps, lw=1.5)
ax[0].set_ylabel("Temperature (K)")
ax[1].plot(steps, drift, lw=1.5)
ax[1].axhline(0.0, color="k", ls="--", lw=1.0)
ax[1].set_xlabel("Step")
ax[1].set_ylabel(r"$E(t) - E(0)$ (eV)")
fig.suptitle("Velocity Verlet (NVE): Temperature and Energy Drift")

print("\n" + "=" * 95)
print("SIMULATION COMPLETE")
print("=" * 95)
