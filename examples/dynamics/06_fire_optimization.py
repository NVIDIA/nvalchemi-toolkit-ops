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
FIRE Geometry Optimization (LJ Cluster)
======================================

This example demonstrates geometry optimization with the FIRE optimizer using:
- The **package LJ implementation** (neighbor-list accelerated)
- The **package FIRE kernels** (:mod:`nvalchemiops.dynamics.optimizers`)
- The shared example utilities in :mod:`examples.dynamics._dynamics_utils`

We optimize a small Lennard-Jones cluster (an isolated cluster inside a large box)
and plot convergence (energy and max force).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from _dynamics_utils import MDSystem, create_random_cluster

from nvalchemiops.dynamics.optimizers import fire_step

wp.init()

device = "cuda:0" if wp.is_cuda_available() else "cpu"
print(f"Using device: {device}")

# %%
# Create a Lennard-Jones Cluster
# ------------------------------
#
# We use an "isolated cluster in a large box" approach (PBC far away),
# which works well with the existing neighbor-list machinery.

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
# FIRE Optimization Loop
# ----------------------
#
# We use the unified ``fire_step`` API that handles:
# - Velocity mixing (FIRE algorithm)
# - Adaptive timestep (dt)
# - Adaptive mixing parameter (alpha)
# - MD-like position update with maxstep capping

max_steps = 3000
force_tolerance = 1e-3  # eV/Å (max force)

# FIRE parameters
dt0 = 1.0
dt_max = 10.0
dt_min = 1e-3
alpha0 = 0.1
f_inc = 1.1
f_dec = 0.5
f_alpha = 0.99
n_min = 5
maxstep = 0.2 * sigma  # Å (max displacement per iteration)

# Device-side FIRE state arrays (shape = (1,) for single system)
wp_dtype = system.wp_dtype
dt_wp = wp.array([dt0], dtype=wp_dtype, device=device)
alpha_wp = wp.array([alpha0], dtype=wp_dtype, device=device)
alpha_start_wp = wp.array([alpha0], dtype=wp_dtype, device=device)
f_alpha_wp = wp.array([f_alpha], dtype=wp_dtype, device=device)
dt_min_wp = wp.array([dt_min], dtype=wp_dtype, device=device)
dt_max_wp = wp.array([dt_max], dtype=wp_dtype, device=device)
maxstep_wp = wp.array([maxstep], dtype=wp_dtype, device=device)
n_steps_positive_wp = wp.zeros(1, dtype=wp.int32, device=device)
n_min_wp = wp.array([n_min], dtype=wp.int32, device=device)
f_dec_wp = wp.array([f_dec], dtype=wp_dtype, device=device)
f_inc_wp = wp.array([f_inc], dtype=wp_dtype, device=device)
uphill_flag_wp = wp.zeros(1, dtype=wp.int32, device=device)

# Accumulators for diagnostic scalars (vf, vv, ff) - must be zeroed before each step
vf_wp = wp.zeros(1, dtype=wp_dtype, device=device)
vv_wp = wp.zeros(1, dtype=wp_dtype, device=device)
ff_wp = wp.zeros(1, dtype=wp_dtype, device=device)

# History for plotting
energy_hist: list[float] = []
maxf_hist: list[float] = []
dt_hist: list[float] = []
alpha_hist: list[float] = []

print("\n" + "=" * 95)
print("FIRE GEOMETRY OPTIMIZATION (LJ cluster)")
print("=" * 95)
print(f"  atoms: {num_atoms}, cutoff={cutoff:.2f} Å, box={box_L:.1f} Å")
print(f"  max_steps={max_steps}, force_tol={force_tolerance:.2e} eV/Å")
print(f"  dt0={dt0}, dt_max={dt_max}, alpha0={alpha0}, n_min={n_min}")
print(f"  dt_min={dt_min}, maxstep={maxstep:.3f} Å")

log_interval = 100
check_interval = 50

for step in range(max_steps):
    # Compute forces at current positions
    energies = system.compute_forces()

    # Zero the accumulators before each FIRE step
    vf_wp.zero_()
    vv_wp.zero_()
    ff_wp.zero_()

    # Single FIRE step using the unified API
    # This performs: diagnostic computation, velocity mixing, parameter update, MD step
    fire_step(
        positions=system.wp_positions,
        velocities=system.wp_velocities,
        forces=system.wp_forces,
        masses=system.wp_masses,
        alpha=alpha_wp,
        dt=dt_wp,
        alpha_start=alpha_start_wp,
        f_alpha=f_alpha_wp,
        dt_min=dt_min_wp,
        dt_max=dt_max_wp,
        maxstep=maxstep_wp,
        n_steps_positive=n_steps_positive_wp,
        n_min=n_min_wp,
        f_dec=f_dec_wp,
        f_inc=f_inc_wp,
        uphill_flag=uphill_flag_wp,
        vf=vf_wp,
        vv=vv_wp,
        ff=ff_wp,
    )

    # Logging / stopping criteria (host read only at intervals)
    if step % check_interval == 0 or step == max_steps - 1:
        pe = float(energies.numpy().sum())
        fmax = float(np.linalg.norm(system.wp_forces.numpy(), axis=1).max())

        energy_hist.append(pe)
        maxf_hist.append(fmax)
        dt_hist.append(float(dt_wp.numpy()[0]))
        alpha_hist.append(float(alpha_wp.numpy()[0]))

        if step % log_interval == 0 or step == max_steps - 1:
            print(
                f"step={step:5d}  PE={pe:12.6f} eV  max|F|={fmax:10.3e} eV/Å  "
                f"dt={dt_hist[-1]:6.3f}  alpha={alpha_hist[-1]:7.4f}  "
                f"n+={int(n_steps_positive_wp.numpy()[0]):3d}"
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
ax[0].set_title("FIRE Optimization Convergence")

ax[1].semilogy(steps, maxf_hist, lw=1.5)
ax[1].axhline(force_tolerance, color="k", ls="--", lw=1.0, label="tolerance")
ax[1].set_xlabel("Log point index")
ax[1].set_ylabel(r"max$|F|$ (eV/$\AA$)")
ax[1].legend(frameon=False, loc="best")

# %%
# Visualize initial vs final geometry (XY projection)

pos0 = initial_positions
pos1 = system.wp_positions.numpy()

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
