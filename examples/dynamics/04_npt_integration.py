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
NPT Dynamics (MTK + Nosé-Hoover Chain) with Lennard-Jones Potential
==================================================================

This example demonstrates isothermal-isobaric (NPT) dynamics using the
Martyna-Tobias-Klein (MTK) barostat coupled with a Nosé-Hoover chain (NHC)
thermostat, driven by the Lennard-Jones (LJ) potential.
"""

# %%
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
    pressure_atm_to_ev_per_a3,
    run_npt_mtk,
)

# %%
print("=" * 95)
print("NPT (MTK + NHC) DYNAMICS WITH LENNARD-JONES POTENTIAL")
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
    switch_width=0.0,
    device=device,
    dtype=np.float64,
)

print("\n--- Setting Initial Temperature ---")
system.initialize_temperature(temperature=94.4, seed=42)

# %%
print("\n--- NPT Run (3000 steps) ---")
print(f"Pressure units sanity: 1 atm = {pressure_atm_to_ev_per_a3(1.0):.6e} eV/Å³")
stats = run_npt_mtk(
    system=system,
    num_steps=3000,
    dt_fs=1.0,
    target_temperature_K=94.4,
    target_pressure_atm=1.0,
    tdamp_fs=500.0,
    pdamp_fs=5000.0,
    chain_length=3,
    log_interval=200,
)

# %%
print("\n--- Analysis ---")
temps = np.array([s.temperature for s in stats])
pressures_atm = np.array([s.pressure for s in stats])
volumes = np.array([s.volume for s in stats])
steps = np.array([s.step for s in stats])

print(f"  Mean Temperature: {temps.mean():.2f} ± {temps.std():.2f} K")
print("  Target Temperature: 94.4 K")
print()
print(f"  Mean Pressure:    {pressures_atm.mean():.3f} ± {pressures_atm.std():.3f} atm")
print("  Target Pressure:  1.0 atm")
print()
print(f"  Mean Volume:      {volumes.mean():.2f} ± {volumes.std():.2f} Å³")

fig, ax = plt.subplots(3, 1, figsize=(7.0, 6.5), sharex=True, constrained_layout=True)
ax[0].plot(steps, temps, lw=1.5)
ax[0].axhline(94.4, color="k", ls="--", lw=1.0, label="target")
ax[0].set_ylabel("Temperature (K)")
ax[0].legend(frameon=False, loc="best")
ax[1].plot(steps, pressures_atm, lw=1.5)
ax[1].axhline(1.0, color="k", ls="--", lw=1.0, label="target")
ax[1].set_ylabel("Pressure (atm)")
ax[1].legend(frameon=False, loc="best")
ax[2].plot(steps, volumes, lw=1.5)
ax[2].set_xlabel("Step")
ax[2].set_ylabel(r"Volume ($\AA^3$)")
fig.suptitle("NPT (MTK + NHC): Temperature, Pressure, and Volume")

print("\n" + "=" * 95)
print("SIMULATION COMPLETE")
print("=" * 95)
