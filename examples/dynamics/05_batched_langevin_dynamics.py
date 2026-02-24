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
Batched Langevin Dynamics (BAOAB) with Lennard-Jones Potential
=============================================================

This example demonstrates **batched** molecular dynamics: multiple independent
systems are packed into a single set of arrays, and we integrate all systems
in one go on the GPU.

Why batching matters
-------------------
Many workflows (sampling, optimization, hyperparameter sweeps) involve running
many small systems. Batching amortizes kernel launch overhead and improves GPU
utilization.

In this example we:
- create two independent FCC argon systems (256 atoms each),
- assign each system a different target temperature,
- run a batched Langevin (BAOAB) trajectory,
- plot per-system temperature and total energy vs step.
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
    BatchedMDSystem,
    create_fcc_argon,
    run_batched_langevin_baoab,
)

# %%
print("=" * 95)
print("BATCHED LANGEVIN (BAOAB) DYNAMICS WITH LENNARD-JONES POTENTIAL")
print("=" * 95)
print()

device = "cuda:0" if wp.is_cuda_available() else "cpu"
print(f"Using device: {device}")

num_systems = 2
positions_0, cell_0 = create_fcc_argon(num_unit_cells=4, a=5.26)  # 256 atoms
positions_1, cell_1 = create_fcc_argon(
    num_unit_cells=4, a=5.26
)  # identical second system

positions = np.concatenate([positions_0, positions_1], axis=0)
batch_idx = np.concatenate(
    [
        np.zeros(len(positions_0), dtype=np.int32),
        np.ones(len(positions_1), dtype=np.int32),
    ],
    axis=0,
)
cells = np.stack([cell_0, cell_1], axis=0)

system = BatchedMDSystem(
    positions=positions,
    cells=cells,
    batch_idx=batch_idx,
    num_systems=num_systems,
    epsilon=EPSILON_AR,
    sigma=SIGMA_AR,
    cutoff=DEFAULT_CUTOFF,
    skin=DEFAULT_SKIN,
    switch_width=0.0,
    device=device,
    dtype=np.float64,
)

temperatures = np.array([94.4, 150.0], dtype=np.float64)
frictions = np.array([0.01, 0.01], dtype=np.float64)
system.initialize_temperature(temperatures, seed=42)

history = run_batched_langevin_baoab(
    system=system,
    num_steps=2000,
    dt_fs=1.0,
    temperatures_K=temperatures,
    frictions_per_fs=frictions,
    log_interval=100,
    seed=123,
)

# %%
# Plot per-system traces

fig, ax = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True, constrained_layout=True)

for sys_id, stats in history.items():
    steps = np.array([s.step for s in stats])
    temps = np.array([s.temperature for s in stats])
    energies = np.array([s.total_energy for s in stats])
    ax[0].plot(
        steps,
        temps,
        lw=1.5,
        label=f"system {sys_id} (target {temperatures[sys_id]:.1f} K)",
    )
    ax[1].plot(steps, energies, lw=1.5, label=f"system {sys_id}")

ax[0].set_ylabel("Temperature (K)")
ax[0].legend(frameon=False, loc="best")
ax[1].set_xlabel("Step")
ax[1].set_ylabel("Total Energy (eV)")
ax[1].legend(frameon=False, loc="best")
fig.suptitle("Batched Langevin (BAOAB): Per-system Temperature and Total Energy")
