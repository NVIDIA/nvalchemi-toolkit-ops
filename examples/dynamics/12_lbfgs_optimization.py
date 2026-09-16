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
L-BFGS Geometry Optimization (LJ Cluster)
==========================================

This example demonstrates geometry optimization with the **L-BFGS** optimizer
using:

- The **package LJ implementation** (neighbor-list accelerated)
- The **package L-BFGS kernels**
  (:func:`nvalchemiops.dynamics.optimizers.lbfgs_step`)
- The shared example utilities in :mod:`examples.dynamics._dynamics_utils`

L-BFGS is a quasi-Newton method: it builds an implicit approximation to the
inverse Hessian from the last few position and gradient differences, and picks
a step length along that direction with a line search. Compared with FIRE2
(``09_fire2_optimization.py``) it:

- Reaches a given force tolerance in **far fewer force evaluations**, which is
  the cost that dominates relaxation with a machine-learned potential
- Needs no timestep to tune; the line search chooses the step length
- Costs more memory: ``2 * history_size`` vectors of stored history
- Reports progress through a per-system ``status`` array rather than requiring
  the caller to test the forces

To make that first point concrete, this example relaxes the *same* cluster with
both optimizers and compares how many force evaluations each needed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp
from _dynamics_utils import MDSystem, create_random_cluster

from nvalchemiops.dynamics.optimizers import (
    LBFGS_CONVERGED,
    LBFGS_LS_FAILED,
    LBFGS_NEED_EVAL,
    fire2_step,
    lbfgs_reduce_energy,
    lbfgs_step,
)

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


def make_system() -> MDSystem:
    """A fresh system at the same starting geometry, for a fair comparison."""
    return MDSystem(
        positions=initial_positions.copy(),
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


system = make_system()
wp_dtype = system.wp_dtype
wp_vec_dtype = system.wp_vec_dtype

# %%
# Allocate the L-BFGS Buffers
# ---------------------------
#
# Every array the optimizer touches is yours: the package allocates nothing,
# initializes nothing, and keeps no hidden state between calls. That means no
# allocation happens inside the loop, and it also means the three buffers whose
# starting values are *not* zero are your responsibility -- see below.
# ``history_size`` is the memory knob: the two history buffers dominate, and
# 3 to 7 is the usual range.
#
# Note the precision split. Coordinates may be single or double precision, but
# **every per-system scalar is float64 regardless**. The line search compares a
# difference of *total* energies, and at ``E ~ -1e4 eV`` a single-precision
# accumulator would be coarser than the energy differences being tested.

history_size = 6

# Per-degree-of-freedom arrays, at the coordinate precision.
x_base = wp.zeros(num_atoms, dtype=wp_vec_dtype, device=device)
force_base = wp.zeros(num_atoms, dtype=wp_vec_dtype, device=device)
direction = wp.zeros(num_atoms, dtype=wp_vec_dtype, device=device)
s_history = wp.zeros((history_size, num_atoms), dtype=wp_vec_dtype, device=device)
y_history = wp.zeros((history_size, num_atoms), dtype=wp_vec_dtype, device=device)


# Per-history-slot and per-system scalars, always float64.
def f64(*shape) -> wp.array:
    """Allocate a zeroed float64 array."""
    return wp.zeros(
        shape if len(shape) > 1 else shape[0], dtype=wp.float64, device=device
    )


ys = f64(history_size, 1)
yy = f64(history_size, 1)
alpha_hist = f64(history_size, 1)
beta_hist = f64(history_size, 1)
ss = f64(1)
f_base = f64(1)
gg = f64(1)
gd = f64(1)
fmax = f64(1)
frms_sq = f64(1)
smax = f64(1)
d0 = f64(1)
dmax = f64(1)
dquad = f64(1)
alpha_step = f64(1)

# Per-system integer control state.
status = wp.zeros(1, dtype=wp.int32, device=device)
iteration = wp.zeros(1, dtype=wp.int32, device=device)
end = wp.zeros(1, dtype=wp.int32, device=device)
n_loop = wp.zeros(1, dtype=wp.int32, device=device)
ls_trials = wp.zeros(1, dtype=wp.int32, device=device)
history_count = wp.zeros(1, dtype=wp.int32, device=device)

lbfgs_state = dict(
    x_base=x_base,
    force_base=force_base,
    direction=direction,
    s_history=s_history,
    y_history=y_history,
    ys=ys,
    yy=yy,
    alpha_hist=alpha_hist,
    beta_hist=beta_hist,
    ss=ss,
    f_base=f_base,
    gg=gg,
    gd=gd,
    fmax=fmax,
    frms_sq=frms_sq,
    smax=smax,
    d0=d0,
    dmax=dmax,
    dquad=dquad,
    alpha_step=alpha_step,
    status=status,
    iteration=iteration,
    end=end,
    n_loop=n_loop,
    ls_trials=ls_trials,
    history_count=history_count,
)

# Everything above starts at zero, which is already correct for all but three
# buffers. Set those three explicitly. To restart a relaxation later, zero the
# buffers again and repeat exactly these three lines.
alpha_step.fill_(1.0)  # the line-search step length for a fresh direction
iteration.fill_(-1)  # the "never evaluated yet" marker
status.fill_(LBFGS_NEED_EVAL)  # numerically zero, but say it out loud

# Batching metadata: all zeros for a single system.
batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=device)
n_particles = wp.array([num_atoms], dtype=wp.int32, device=device)

# Per-system total energy, which the line search consumes.
energy = wp.zeros(1, dtype=wp.float64, device=device)

# %%
# L-BFGS Optimization Loop
# ------------------------
#
# You own the loop, exactly as with FIRE2, but the contract is different:
# **each call consumes exactly one energy/force evaluation**, and you stop when
# ``status`` says so rather than by testing the forces yourself.
#
# ``lbfgs_reduce_energy`` sums the per-atom energies the model returns into the
# per-system totals the optimizer needs, accumulating in float64. Use it rather
# than summing yourself in single precision.

max_evals = 500
force_tolerance = 1e-3  # eV/Å, on the largest per-atom force
maxstep = 0.2  # Å, the furthest any atom may move in one step

energy_hist: list[float] = []
maxf_hist: list[float] = []
alpha_hist_log: list[float] = []

print("\n" + "=" * 95)
print("L-BFGS GEOMETRY OPTIMIZATION (LJ cluster)")
print("=" * 95)
print(f"  atoms: {num_atoms}, cutoff={cutoff:.2f} Å, box={box_L:.1f} Å")
print(f"  history_size={history_size}, force_tol={force_tolerance:.2e} eV/Å")
print(f"  maxstep={maxstep} Å")

log_interval = 10
lbfgs_evals = 0

for step in range(max_evals):
    # One energy/force evaluation per step. This is the expensive part with a
    # real potential, and the reason L-BFGS is worth its extra bookkeeping.
    energies = system.compute_forces()
    lbfgs_reduce_energy(energies, batch_idx, energy)
    lbfgs_evals += 1

    lbfgs_step(
        positions=system.wp_positions,
        forces=system.wp_forces,
        energy=energy,
        batch_idx=batch_idx,
        n_particles=n_particles,
        force_tol=force_tolerance,
        maxstep=maxstep,
        **lbfgs_state,
    )

    # `fmax` is computed on the device every call, so logging it is free of an
    # extra reduction; reading it back is the only synchronization.
    pe = float(energies.numpy().sum())
    current_fmax = float(fmax.numpy()[0])
    energy_hist.append(pe)
    maxf_hist.append(current_fmax)
    alpha_hist_log.append(float(alpha_step.numpy()[0]))

    if step % log_interval == 0:
        print(
            f"eval={step:4d}  PE={pe:12.6f} eV  max|F|={current_fmax:10.3e} eV/Å  "
            f"alpha={alpha_hist_log[-1]:9.3e}  "
            f"iter={int(iteration.numpy()[0]):3d}  "
            f"hist={int(history_count.numpy()[0]):2d}"
        )

    state_now = int(status.numpy()[0])
    if state_now != LBFGS_NEED_EVAL:
        break

# %%
# Reading ``status``
# ------------------
#
# ``status`` is the only value you need to inspect. ``LBFGS_LS_FAILED`` is not
# a convergence claim: it means the line search could not make progress even
# from a steepest-descent direction, and the positions have been restored to
# the last accepted point.
#
# Note which force array to trust afterwards. ``forces`` is an input the
# optimizer only reads, so after a rollback it still holds the forces at the
# *rejected* trial and no longer matches ``positions``. ``force_base`` is
# written alongside ``x_base`` at every accepted point, so it is the one that
# describes the geometry actually handed back.

final_status = int(status.numpy()[0])
status_name = {
    LBFGS_NEED_EVAL: "NEED_EVAL (ran out of evaluations)",
    LBFGS_CONVERGED: "CONVERGED",
    LBFGS_LS_FAILED: "LS_FAILED (line search stalled)",
}[final_status]
print(f"\nFinished after {lbfgs_evals} force evaluations: {status_name}")
print(f"  final max|F| = {maxf_hist[-1]:.3e} eV/Å")
print(f"  final PE     = {energy_hist[-1]:.6f} eV")

lbfgs_positions = wp.to_torch(system.wp_positions).cpu().numpy().copy()

# %%
# Compare Against FIRE2
# ---------------------
#
# The same cluster, the same starting geometry and the same force tolerance,
# relaxed with FIRE2. What matters is the number of force evaluations: with a
# machine-learned potential the model call dominates the optimizer's own
# kernel time by orders of magnitude.
#
# The FIRE2 settings below were **tuned for this system** by sweeping the
# timestep and step cap, and are the best found. Comparing against an untuned
# baseline would overstate the result: FIRE2 is sensitive to its timestep, and
# the much smaller ``dt`` used in ``09_fire2_optimization.py`` does not converge
# this cluster within a few thousand evaluations. L-BFGS needs no such tuning,
# which is a practical advantage in its own right.

fire2_system = make_system()
velocities = wp.zeros(num_atoms, dtype=wp_vec_dtype, device=device)
alpha = wp.array([0.09], dtype=wp_dtype, device=device)
dt = wp.array([1.0], dtype=wp_dtype, device=device)
nsteps_inc = wp.zeros(1, dtype=wp.int32, device=device)
vf = wp.zeros(1, dtype=wp_dtype, device=device)
v_sumsq = wp.zeros(1, dtype=wp_dtype, device=device)
f_sumsq = wp.zeros(1, dtype=wp_dtype, device=device)
max_norm = wp.zeros(1, dtype=wp_dtype, device=device)

fire2_evals = 0
fire2_maxf_hist: list[float] = []
fire2_converged = False

for step in range(3000):
    fire2_energies = fire2_system.compute_forces()
    fire2_evals += 1
    current = float(
        torch.linalg.norm(wp.to_torch(fire2_system.wp_forces), dim=1).max().item()
    )
    fire2_maxf_hist.append(current)
    if current < force_tolerance:
        fire2_converged = True
        break

    fire2_step(
        positions=fire2_system.wp_positions,
        velocities=velocities,
        forces=fire2_system.wp_forces,
        batch_idx=batch_idx,
        alpha=alpha,
        dt=dt,
        nsteps_inc=nsteps_inc,
        vf=vf,
        v_sumsq=v_sumsq,
        f_sumsq=f_sumsq,
        max_norm=max_norm,
        maxstep=0.2,
        tmax=5.0,
        delaystep=20,
        dtgrow=1.1,
        dtshrink=0.5,
    )

print("\n" + "=" * 95)
print("FORCE EVALUATIONS TO CONVERGENCE")
print("=" * 95)
print(f"  L-BFGS : {lbfgs_evals:5d}  ({status_name})")
print(
    f"  FIRE2  : {fire2_evals:5d}  "
    f"({'converged, tuned settings' if fire2_converged else 'hit the evaluation cap'})"
)
if fire2_converged and final_status == LBFGS_CONVERGED:
    print(f"  ratio  : {lbfgs_evals / fire2_evals:.3f} (lower is better for L-BFGS)")
else:
    print("  ratio  : not comparable, one optimizer did not converge")

# %%
# Plot convergence
# ----------------

fig, ax = plt.subplots(2, 1, figsize=(7.0, 5.5), constrained_layout=True)

ax[0].semilogy(np.arange(len(maxf_hist)), maxf_hist, lw=1.5, label="L-BFGS")
ax[0].semilogy(
    np.arange(len(fire2_maxf_hist)), fire2_maxf_hist, lw=1.0, alpha=0.8, label="FIRE2"
)
ax[0].axhline(force_tolerance, color="k", ls="--", lw=1.0, label="tolerance")
ax[0].set_xlabel("Force evaluations")
ax[0].set_ylabel(r"max$|F|$ (eV/$\AA$)")
ax[0].set_title("Convergence per force evaluation")
ax[0].legend(frameon=False, loc="best")

ax[1].plot(np.arange(len(energy_hist)), energy_hist, lw=1.5)
ax[1].set_xlabel("Force evaluations")
ax[1].set_ylabel("Potential Energy (eV)")
ax[1].set_title("L-BFGS energy")

# %%
# Visualize initial vs final geometry (XY projection)

fig2, ax2 = plt.subplots(
    1, 2, figsize=(8.0, 3.5), sharex=True, sharey=True, constrained_layout=True
)
ax2[0].scatter(initial_positions[:, 0], initial_positions[:, 1], s=20)
ax2[0].set_title("Initial (XY)")
ax2[0].set_xlabel("x (Å)")
ax2[0].set_ylabel("y (Å)")
ax2[1].scatter(lbfgs_positions[:, 0], lbfgs_positions[:, 1], s=20)
ax2[1].set_title("L-BFGS optimized (XY)")
ax2[1].set_xlabel("x (Å)")

plt.show()
