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
FIRE2 Geometry Optimization (LJ Cluster)
=========================================

This example demonstrates geometry optimization with the **FIRE2** optimizer
(Guenole et al., 2020) using:

- The **package LJ implementation** (neighbor-list accelerated)
- The **package FIRE2 kernels** (:func:`nvalchemiops.dynamics.optimizers.fire2_step`)
- The shared example utilities in :mod:`examples.dynamics._dynamics_utils`

Compared to FIRE (``06_fire_optimization.py``), FIRE2:

- Assumes unit mass (no ``masses`` parameter)
- Requires ``batch_idx`` even for single-system mode
- Uses fewer per-system state arrays (``alpha``, ``dt``, ``nsteps_inc``)
- Hyperparameters are Python scalars (``delaystep``, ``dtgrow``, etc.)

We optimize the same small Lennard-Jones cluster and plot convergence.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp
from _dynamics_utils import MDSystem, create_random_cluster

from nvalchemiops.dynamics.optimizers import fire2_step

wp.init()

device = "cuda:0" if wp.is_cuda_available() else "cpu"
print(f"Using device: {device}")

# %%
# Create a Lennard-Jones Cluster
# ------------------------------

num_atoms = 32
epsilon = 0.0104  # eV (argon-like)
sigma = 3.40  # Å
cutoff = 2.5 * sigma
skin = 0.5
box_L = 80.0  # Å (large to avoid self-interaction across PBC)

cell = np.eye(3, dtype=np.float64) * box_L
initial_positions = create_random_cluster(
    num_atoms=num_atoms,
    radius=12.0,
    min_dist=0.9 * sigma,
    center=np.array([0.5 * box_L, 0.5 * box_L, 0.5 * box_L]),
    seed=42,
)

system = MDSystem(
    positions=initial_positions,
    cell=cell,
    masses=np.full(num_atoms, 39.948, dtype=np.float64),  # amu (argon)
    epsilon=epsilon,
    sigma=sigma,
    cutoff=cutoff,
    skin=skin,
    switch_width=0.0,
    device=device,
    dtype=np.float64,
)

# %%
# FIRE2 Optimization Loop
# -----------------------
#
# FIRE2 uses a simpler state than FIRE: just ``alpha``, ``dt``, ``nsteps_inc``
# per system, plus 4 scratch buffers. Hyperparameters are Python scalars.
# ``batch_idx`` is always required (all zeros for single system).

max_steps = 3000
force_tolerance = 1e-3  # eV/Å (max force)

wp_dtype = system.wp_dtype

# Per-system state arrays (shape (1,) for single system)
alpha = wp.array([0.09], dtype=wp_dtype, device=device)
dt = wp.array([0.005], dtype=wp_dtype, device=device)
nsteps_inc = wp.zeros(1, dtype=wp.int32, device=device)

# Scratch buffers (shape (1,) for single system)
vf = wp.zeros(1, dtype=wp_dtype, device=device)
v_sumsq = wp.zeros(1, dtype=wp_dtype, device=device)
f_sumsq = wp.zeros(1, dtype=wp_dtype, device=device)
max_norm = wp.zeros(1, dtype=wp_dtype, device=device)

# batch_idx: all zeros for single system
batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=device)

# Velocities (FIRE2 uses unit mass, so velocities are just momenta)
velocities = wp.zeros(num_atoms, dtype=system.wp_vec_dtype, device=device)

# History for plotting
energy_hist: list[float] = []
maxf_hist: list[float] = []
dt_hist: list[float] = []
alpha_hist: list[float] = []

print("\n" + "=" * 95)
print("FIRE2 GEOMETRY OPTIMIZATION (LJ cluster)")
print("=" * 95)
print(f"  atoms: {num_atoms}, cutoff={cutoff:.2f} Å, box={box_L:.1f} Å")
print(f"  max_steps={max_steps}, force_tol={force_tolerance:.2e} eV/Å")
print("  FIRE2 defaults: delaystep=60, dtgrow=1.05, alpha0=0.09, maxstep=0.1")

log_interval = 100
check_interval = 50

for step in range(max_steps):
    # Compute forces at current positions
    energies = system.compute_forces()

    # FIRE2 step: updates positions, velocities, alpha, dt, nsteps_inc in-place
    fire2_step(
        positions=system.wp_positions,
        velocities=velocities,
        forces=system.wp_forces,
        batch_idx=batch_idx,
        alpha=alpha,
        dt=dt,
        nsteps_inc=nsteps_inc,
        vf=vf,
        v_sumsq=v_sumsq,
        f_sumsq=f_sumsq,
        max_norm=max_norm,
        device=device,
    )

    # Logging / stopping criteria (host read only at intervals)
    if step % check_interval == 0 or step == max_steps - 1:
        pe = float(energies.numpy().sum())
        fmax = float(
            torch.linalg.norm(wp.to_torch(system.wp_forces), dim=1).max().item()
        )

        energy_hist.append(pe)
        maxf_hist.append(fmax)
        dt_hist.append(float(dt.numpy()[0]))
        alpha_hist.append(float(alpha.numpy()[0]))

        if step % log_interval == 0 or step == max_steps - 1:
            print(
                f"step={step:5d}  PE={pe:12.6f} eV  max|F|={fmax:10.3e} eV/Å  "
                f"dt={dt_hist[-1]:8.5f}  alpha={alpha_hist[-1]:7.4f}  "
                f"n+={int(nsteps_inc.numpy()[0]):3d}"
            )

        if fmax < force_tolerance:
            print(f"\nConverged at step {step} (max|F|={fmax:.3e} eV/Å).")
            break

# %%
# Plot convergence

steps = np.arange(len(energy_hist))

fig, ax = plt.subplots(2, 1, figsize=(7.0, 5.5), sharex=True, constrained_layout=True)
ax[0].plot(steps, energy_hist, lw=1.5)
ax[0].set_ylabel("Potential Energy (eV)")
ax[0].set_title("FIRE2 Optimization Convergence")

ax[1].semilogy(steps, maxf_hist, lw=1.5)
ax[1].axhline(force_tolerance, color="k", ls="--", lw=1.0, label="tolerance")
ax[1].set_xlabel("Log point index")
ax[1].set_ylabel(r"max$|F|$ (eV/$\AA$)")
ax[1].legend(frameon=False, loc="best")

# %%
# Visualize initial vs final geometry (XY projection)

pos0 = initial_positions
pos1 = wp.to_torch(system.wp_positions).cpu().numpy()

fig2, ax2 = plt.subplots(
    1, 2, figsize=(8.0, 3.5), sharex=True, sharey=True, constrained_layout=True
)
ax2[0].scatter(pos0[:, 0], pos0[:, 1], s=20)
ax2[0].set_title("Initial (XY)")
ax2[0].set_xlabel("x (Å)")
ax2[0].set_ylabel("y (Å)")
ax2[1].scatter(pos1[:, 0], pos1[:, 1], s=20)
ax2[1].set_title("Optimized (XY)")
ax2[1].set_xlabel("x (Å)")

plt.show()
