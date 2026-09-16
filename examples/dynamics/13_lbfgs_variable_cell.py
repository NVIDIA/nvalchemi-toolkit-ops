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
Variable-Cell L-BFGS Optimization (FCC Argon)
==============================================

Joint relaxation of atomic coordinates and the simulation cell with the
**L-BFGS** optimizer, using the PyTorch binding
:func:`nvalchemiops.torch.lbfgs.lbfgs_step_coord_cell`.

Compared with the FIRE2 variable-cell example (``11_fire2_variable_cell.py``),
the L-BFGS path is a single call and takes the **stress** directly:

- You pass ``stress``; the optimizer converts it to a cell driving force
  internally, so there is no separate ``stress_to_cell_force`` step
- The same ``stress`` is used for the stress convergence criterion, which is
  compared against a real stress rather than against a packed cell force
- There is no timestep or cell mass to tune

How the coupling works
----------------------
Positions and cell are mapped into one packed coordinate vector, so the
quasi-Newton recursion couples them with no special handling. The coordinates
follow ASE's ``UnitCellFilter`` convention: with a reference cell ``H0`` fixed
at the start and lattice vectors held as columns, the deformation gradient is
``Phi = H H0^-1``, atoms are stored as ``u = Phi^-1 r`` and the cell as
``kappa * Phi``.

Two consequences are worth knowing:

- ``lbfgs_set_reference_cell`` must be called **once**, before the first step.
  Re-referencing mid-run invalidates every stored curvature pair.
- The optimizer buffers must be sized for ``num_atoms + 2 * num_systems``
  degrees of freedom, because the cell contributes two packed entries per
  system.

Every buffer is caller-owned. Nothing is allocated or initialized for you, so
this example shows the full allocation, including how to build the extended
topology arrays with the generic batch utilities -- which is also what makes
ragged batches (systems with different atom counts) expressible.

Convergence is always evaluated on the **Cartesian** forces and the stress, so
``force_tol`` keeps its meaning as a force per atom however far the cell
deforms.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp
from _dynamics_utils import (
    EPSILON_AR,
    SIGMA_AR,
    MDSystem,
    create_fcc_lattice,
    pressure_ev_per_a3_to_gpa,
    pressure_gpa_to_ev_per_a3,
    virial_to_stress,
)

from nvalchemiops.batch_utils import atom_ptr_to_batch_idx
from nvalchemiops.dynamics.utils import align_cell, wrap_positions_to_cell
from nvalchemiops.dynamics.utils.cell_filter import extend_atom_ptr
from nvalchemiops.torch.lbfgs import (
    LBFGS_CONVERGED,
    LBFGS_LS_FAILED,
    LBFGS_NEED_EVAL,
    lbfgs_cell_kappa,
    lbfgs_reduce_energy,
    lbfgs_set_reference_cell,
    lbfgs_step_coord_cell,
)

CUTOFF = 2.5 * SIGMA_AR  # ~8.5 Å

wp.init()
device = "cuda:0" if wp.is_cuda_available() else "cpu"
torch_device = torch.device(device)
print(f"Using device: {device}")

# %%
# Create Initial System
# ---------------------
#
# An FCC argon lattice, deliberately expanded from its equilibrium spacing so
# there is something for the cell to relax.

n_cells = 3  # 3x3x3 = 108 atoms
a_initial = 5.5  # Å (equilibrium is near 5.26 Å)

positions_np, cell_np = create_fcc_lattice(n_cells, a_initial)
num_atoms = len(positions_np)
num_systems = 1

target_pressure_gpa = 0.01
target_pressure = pressure_gpa_to_ev_per_a3(target_pressure_gpa)

print(f"System: {num_atoms} atoms in {n_cells}³ FCC lattice")
print(f"Initial lattice constant: {a_initial:.3f} Å")
print(f"Target external pressure: {target_pressure_gpa:.3f} GPa")

positions = wp.array(positions_np, dtype=wp.vec3d, device=device)
cell = wp.array(cell_np.reshape(1, 3, 3), dtype=wp.mat33d, device=device)

md_system = MDSystem(
    positions=positions_np,
    cell=cell_np,
    epsilon=EPSILON_AR,
    sigma=SIGMA_AR,
    cutoff=CUTOFF,
    skin=0.5,
    switch_width=1.0,  # smooth cutoff, so the energy is differentiable
    device=device,
)

# %%
# Align the Cell, Once
# --------------------
#
# The optimizer keeps the cell lower-triangular, which is what stops it
# drifting into a rigid rotation. Align before capturing the reference cell:
# re-aligning mid-run would redefine the chart and invalidate the history.

transform = wp.empty(1, dtype=wp.mat33d, device=device)
positions, cell = align_cell(positions, cell, transform=transform, device=device)
wp.copy(md_system.wp_positions, positions)
md_system.update_cell(cell)

# Wrap once, before the first step. Wrapping *during* the relaxation would be a
# mistake here: it teleports atoms across the periodic boundary, and the stored
# curvature pairs relate displacements to gradient changes, so a discontinuous
# jump in the coordinates corrupts the history. FIRE2 tolerates per-step
# wrapping because it carries no such history.
wrap_positions_to_cell(
    positions=md_system.wp_positions,
    cells=md_system.wp_cell,
    cells_inv=md_system.wp_cell_inv,
    device=device,
)
wp.synchronize()
print(f"\nAligned cell:\n{cell.numpy()[0]}")

# %%
# Allocate the Buffers
# --------------------
#
# Two groups, both yours: the 26 optimizer buffers, identical to the
# coordinate-only path, and the 14 variable-cell buffers that carry the
# coordinate chart and its scratch space.

# Work directly on the system's own arrays, so the optimizer's in-place writes
# are immediately visible to the force engine. Only the cell needs an explicit
# hand-back, because its inverse and the neighbor list depend on it.
positions_t = wp.to_torch(md_system.wp_positions)
cell_t = wp.to_torch(cell).reshape(num_systems, 3, 3)
batch_idx = torch.zeros(num_atoms, dtype=torch.int32, device=torch_device)
n_particles = torch.full(
    (num_systems,), num_atoms, dtype=torch.int32, device=torch_device
)

dtype = torch.float64
history_size = 6
# The cell contributes two packed entries per system, so every optimizer buffer
# is sized for num_atoms + 2 * num_systems degrees of freedom, not num_atoms.
num_dofs = num_atoms + 2 * num_systems


def zeros(*shape, dt=dtype) -> torch.Tensor:
    """Allocate a zeroed tensor on the run device."""
    return torch.zeros(shape, dtype=dt, device=torch_device)


f64, i32 = torch.float64, torch.int32

# Per-degree-of-freedom arrays, at the coordinate precision.
x_base = zeros(num_dofs, 3)
force_base = zeros(num_dofs, 3)
direction = zeros(num_dofs, 3)
s_history = zeros(history_size, num_dofs, 3)
y_history = zeros(history_size, num_dofs, 3)

# Per-history-slot and per-system scalars. These stay float64 whatever
# precision the coordinates use: the line search compares a difference of
# *total* energies, which single precision cannot resolve near convergence.
ys = zeros(history_size, num_systems, dt=f64)
yy = zeros(history_size, num_systems, dt=f64)
alpha_hist = zeros(history_size, num_systems, dt=f64)
beta_hist = zeros(history_size, num_systems, dt=f64)
ss = zeros(num_systems, dt=f64)
f_base = zeros(num_systems, dt=f64)
gg = zeros(num_systems, dt=f64)
gd = zeros(num_systems, dt=f64)
fmax = zeros(num_systems, dt=f64)
frms_sq = zeros(num_systems, dt=f64)
smax = zeros(num_systems, dt=f64)
d0 = zeros(num_systems, dt=f64)
dmax = zeros(num_systems, dt=f64)
dquad = zeros(num_systems, dt=f64)
alpha_step = zeros(num_systems, dt=f64)

# Per-system integer control state.
status = zeros(num_systems, dt=i32)
iteration = zeros(num_systems, dt=i32)
end = zeros(num_systems, dt=i32)
n_loop = zeros(num_systems, dt=i32)
ls_trials = zeros(num_systems, dt=i32)
history_count = zeros(num_systems, dt=i32)

# Three buffers do not start at zero. Setting them is the whole of
# initialization; to restart later, zero everything and repeat these lines.
alpha_step.fill_(1.0)  # the line-search step length for a fresh direction
iteration.fill_(-1)  # the "never evaluated yet" marker
status.fill_(LBFGS_NEED_EVAL)  # numerically zero, but say it out loud

# The buffers are passed positionally, in this exact order, to every step.
optimizer_buffers = (
    x_base, force_base, direction, s_history, y_history,
    ys, yy, alpha_hist, beta_hist, ss, f_base, gg, gd,
    fmax, frms_sq, smax, d0, dmax, dquad, alpha_step,
    status, iteration, end, n_loop, ls_trials, history_count,
)  # fmt: skip

# %%
# Build the Extended Topology
# ---------------------------
#
# The packed layout interleaves each system's atoms with its two cell entries,
# so it needs its own CSR pointers and its own per-entry system index. These
# come from the generic batch utilities rather than from a bespoke allocator,
# which is what lets ragged batches work: build ``atom_ptr`` for whatever atom
# counts you actually have, and the extension follows.

atom_ptr = wp.array(
    np.arange(num_systems + 1, dtype=np.int32) * num_atoms,
    dtype=wp.int32,
    device=device,
)
ext_atom_ptr = torch.zeros(num_systems + 1, dtype=i32, device=torch_device)
ext_batch_idx = torch.zeros(num_dofs, dtype=i32, device=torch_device)
extend_atom_ptr(atom_ptr, wp.from_torch(ext_atom_ptr, dtype=wp.int32), device=device)
atom_ptr_to_batch_idx(
    wp.from_torch(ext_atom_ptr, dtype=wp.int32),
    wp.from_torch(ext_batch_idx, dtype=wp.int32),
)

# The chart itself, plus scratch the step reuses every call.
ref_cell = zeros(num_systems, 3, 3)
ref_cell_inv = zeros(num_systems, 3, 3)
kappa = zeros(num_systems)
phi = zeros(num_systems, 3, 3)
phi_inv = zeros(num_systems, 3, 3)
d_phi = zeros(num_systems, 3, 3)
cell_dof_a = zeros(num_systems, 3)
cell_dof_b = zeros(num_systems, 3)
cell_force_a = zeros(num_systems, 3)
cell_force_b = zeros(num_systems, 3)
ext_positions = zeros(num_dofs, 3)
ext_forces = zeros(num_dofs, 3)

# Capture the reference cell that defines the chart. Once, before stepping.
lbfgs_set_reference_cell(cell_t, ref_cell, ref_cell_inv)

# ``kappa`` scales the cell coordinate against the atomic ones. It depends only
# on topology, so it is computed once. The default here puts the cell and the
# atoms on a comparable footing; raise it to make the cell move less per step.
lbfgs_cell_kappa(n_particles, kappa, cell_force_scale=1.0 / num_atoms)

cell_buffers = (
    ref_cell, ref_cell_inv, kappa, ext_batch_idx, ext_atom_ptr,
    phi, phi_inv, d_phi, cell_dof_a, cell_dof_b,
    cell_force_a, cell_force_b, ext_positions, ext_forces,
)  # fmt: skip

energy_t = torch.zeros(num_systems, dtype=f64, device=torch_device)

print(f"\nOptimizer degrees of freedom: {num_atoms} atoms + {2 * num_systems} cell")
# %%
# Optimization Loop
# -----------------
#
# One energy/force/stress evaluation per call, and ``status`` says when
# to stop. Both tolerances are physical: ``force_tol`` is eV/Å on the largest
# per-atom force, ``stress_tol`` is a stress.

max_evals = 200
force_tol = 1e-4  # eV/Å
stress_tol = pressure_gpa_to_ev_per_a3(0.03)  # 0.03 GPa
maxstep = 0.2  # Å

energy_hist: list[float] = []
max_force_hist: list[float] = []
volume_hist: list[float] = []
pressure_hist: list[float] = []

print("\n" + "=" * 95)
print("VARIABLE-CELL L-BFGS OPTIMIZATION (FCC argon)")
print("=" * 95)
print(f"{'eval':>6} {'PE (eV)':>14} {'max|F|':>11} {'|stress| GPa':>13} {'a (Å)':>9}")

n_evals = 0
for step in range(max_evals):
    energies, forces, virial = md_system.compute_forces_virial()
    stress = virial_to_stress(virial, md_system.wp_cell, target_pressure, device)
    n_evals += 1

    # Sum per-atom energies into the per-system total, in float64.
    lbfgs_reduce_energy(wp.to_torch(energies), batch_idx, energy_t)

    lbfgs_step_coord_cell(
        positions_t,
        cell_t,
        wp.to_torch(forces),
        wp.to_torch(stress).reshape(num_systems, 3, 3),
        energy_t,
        batch_idx,
        n_particles,
        *optimizer_buffers,
        *cell_buffers,
        force_tol=force_tol,
        stress_tol=stress_tol,
        maxstep=maxstep,
    )

    # Positions were written in place; the cell needs handing back so its
    # inverse is recomputed and the neighbor list is rebuilt.
    md_system.update_cell(cell)

    converged = int(status.item()) != LBFGS_NEED_EVAL
    if step < 20 or step % 25 == 0 or converged:
        volume = float(np.linalg.det(cell_t.detach().cpu().numpy()[0]))
        stress_np = stress.numpy()[0]
        stress_gpa = pressure_ev_per_a3_to_gpa(0.5 * (stress_np + stress_np.T))
        stress_residual = float(np.linalg.svd(stress_gpa, compute_uv=False).max())
        pe = float(energies.numpy().sum())
        fmax_now = float(fmax.item())

        energy_hist.append(pe)
        max_force_hist.append(fmax_now)
        volume_hist.append(volume)
        pressure_hist.append(stress_residual)
        print(
            f"{step:>6d} {pe:>14.6f} {fmax_now:>11.3e} {stress_residual:>13.3e} "
            f"{(volume / (n_cells**3)) ** (1 / 3):>9.4f}"
        )

    if converged:
        break

# %%
# Result
# ------

final_status = int(status.item())
status_name = {
    LBFGS_NEED_EVAL: "NEED_EVAL (ran out of evaluations)",
    LBFGS_CONVERGED: "CONVERGED",
    LBFGS_LS_FAILED: "LS_FAILED (line search stalled)",
}[final_status]

final_volume = float(np.linalg.det(cell_t.detach().cpu().numpy()[0]))
final_a = (final_volume / (n_cells**3)) ** (1 / 3)
print(f"\nFinished after {n_evals} evaluations: {status_name}")
print(f"  lattice constant: {a_initial:.4f} Å -> {final_a:.4f} Å")
print(f"  final max|F|    : {float(fmax.item()):.3e} eV/Å")
print(f"  final volume    : {final_volume:.2f} Å³")
print(f"  final |stress|  : {pressure_hist[-1]:.4f} GPa")
print("  (textbook FCC argon equilibrium is near 5.26 Å)")

# %%
# Plot convergence
# ----------------

points = np.arange(len(energy_hist))
fig, ax = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=True, constrained_layout=True)

ax[0].plot(points, energy_hist, lw=1.5)
ax[0].set_ylabel("Potential Energy (eV)")
ax[0].set_title("Variable-Cell L-BFGS Convergence")

ax[1].semilogy(points, max_force_hist, lw=1.5, label=r"max$|F|$")
ax[1].axhline(force_tol, color="k", ls="--", lw=1.0, label="force tolerance")
ax[1].set_ylabel(r"max$|F|$ (eV/$\AA$)")
ax[1].legend(frameon=False, loc="best")

ax[2].plot(points, [(v / n_cells**3) ** (1 / 3) for v in volume_hist], lw=1.5)
ax[2].set_xlabel("Log point index")
ax[2].set_ylabel("Lattice constant (Å)")

plt.show()
