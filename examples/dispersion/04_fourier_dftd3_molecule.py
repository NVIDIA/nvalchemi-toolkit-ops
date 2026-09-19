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
FourierD3: Dispersion Without a Real-Space Cutoff
==================================================

The DFT-D3 dispersion correction decays as :math:`1/r^6`. Summed over three dimensions that
leaves a truncation error decaying only as :math:`1/r^3`, so converging it in real space needs
a neighbour list far larger than a machine-learned force field's own, and building that list
comes to dominate the simulation step.

FourierD3 removes the cutoff instead of enlarging it: the dispersion sum is evaluated on a
particle mesh, and the only real-space cutoff left is the short coordination-number list the
force field already builds.

In this example you will learn:

- How to decompose the D3 reference tables for the species in your system
- How to evaluate the correction with :func:`fourier_dftd3`
- How its energy compares with the real-space :func:`dftd3` at several cutoffs

.. important::
    This script is intended as an API demonstration. Do not use this script
    for performance benchmarking; refer to the `benchmarks` folder instead.
"""

# %%
# Setup
# -----
#
# The D3 reference parameters are in atomic units, so every length here is in Bohr.

import numpy as np
import torch

from nvalchemiops.torch.interactions.dispersion import (
    FourierD3Parameters,
    dftd3,
    fourier_dftd3,
)

BOHR_TO_ANGSTROM = 0.529177210544
HARTREE_TO_EV = 27.211386245981

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("FourierD3 requires a CUDA device. Skipping.")
    raise SystemExit(0)

# %%
# A periodic cell
# ---------------
#
# A simple cubic arrangement of carbon and hydrogen, large enough that the dispersion tail
# reaches well beyond any reasonable neighbour list.

rng = np.random.default_rng(0)
box = 20.0  # Bohr
n_atoms = 64
positions = torch.tensor(
    rng.uniform(0.0, box, (n_atoms, 3)), dtype=torch.float64, device=device
)
numbers = torch.tensor(rng.choice([1, 6], n_atoms), dtype=torch.int32, device=device)
cell = torch.eye(3, dtype=torch.float64, device=device) * box

# %%
# Reference parameters
# --------------------
#
# ``FourierD3Parameters`` decomposes Grimme's reference tensor into separable factors, once,
# for the species present. That decomposition is what lets an environment-dependent pair
# coefficient be evaluated on a mesh at all.
#
# It depends only on the reference tables, the species and the requested tolerance --- not on
# the damping parameters --- so one instance serves every functional.
#
# The tables below stand in for the real ones. To use the published values, see
# ``examples/dispersion/utils.py``, which downloads and parses them.

max_z = 10
n_ref = 5
rcov = torch.zeros(max_z, dtype=torch.float64, device=device)
rcov[[1, 6]] = torch.tensor([0.60, 1.20], dtype=torch.float64, device=device)
r4r2 = torch.zeros(max_z, dtype=torch.float64, device=device)
r4r2[[1, 6]] = torch.tensor([1.00, 1.40], dtype=torch.float64, device=device)

c6ab = torch.zeros(max_z, max_z, n_ref, n_ref, dtype=torch.float64, device=device)
cn_ref = torch.zeros_like(c6ab)
used = {1: 2, 6: 5}
factors = {z: rng.normal(size=(n, 3)) for z, n in used.items()}
for z_i, n_i in used.items():
    for z_j, n_j in used.items():
        block = factors[z_i] @ factors[z_j].T + 20.0
        c6ab[z_i, z_j, :n_i, :n_j] = torch.tensor(block, device=device)
        for p in range(n_i):
            cn_ref[z_i, z_j, p, :n_j] = float(np.linspace(0.0, 3.5, n_i)[p])

params = FourierD3Parameters.from_tables(
    rcov, r4r2, c6ab, cn_ref, species=[1, 6], device=device
)
print(f"species covered      : {params.n_species}")
print(f"decomposition rank   : {params.rank}")
print(f"mesh channels needed : {params.n_species * params.rank}")
print(f"reconstruction error : {params.max_relative_error:.2e}")

# %%
# The neighbour list
# ------------------
#
# Only the coordination numbers need one, so the cutoff is short. It must match ``r_cut``
# exactly: the counting function is built to reach zero there.

r_cut = 6.0 / BOHR_TO_ANGSTROM  # 6 Angstrom, the usual MLFF cutoff, in Bohr

positions_np = positions.cpu().numpy()
cell_np = cell.cpu().numpy()


def build_neighbour_list(cutoff):
    """A directed CSR list holding both orientations of every pair within ``cutoff``.

    Written out here so the example depends on nothing but NumPy; in practice use
    ``nvalchemiops.torch.neighbors.neighbor_list``.
    """
    reach = int(np.ceil(cutoff / box)) + 1
    offsets = np.arange(-reach, reach + 1)
    lattice = np.stack(
        np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1
    ).reshape(-1, 3)

    sources, targets, shifts = [], [], []
    for translation in lattice:
        delta = (
            positions_np[None, :, :] + translation @ cell_np - positions_np[:, None, :]
        )
        distance = np.linalg.norm(delta, axis=-1)
        for i in range(n_atoms):
            for j in range(n_atoms):
                if (i == j and not translation.any()) or distance[i, j] >= cutoff:
                    continue
                sources.append(i)
                targets.append(j)
                shifts.append(translation)

    order = np.argsort(sources, kind="stable")
    sources = np.asarray(sources)[order]
    targets = np.asarray(targets)[order]
    shifts = np.asarray(shifts)[order]
    pointer = np.zeros(n_atoms + 1, dtype=np.int32)
    for source in sources:
        pointer[source + 1] += 1
    pointer = np.cumsum(pointer).astype(np.int32)

    def to_device(array, dtype=torch.int32):
        return torch.tensor(array, dtype=dtype, device=device)

    return (
        torch.stack([to_device(sources), to_device(targets)]),
        to_device(pointer),
        to_device(shifts),
    )


neighbor_list, neighbor_ptr, unit_shifts = build_neighbour_list(r_cut)
print(
    f"\nneighbour cutoff : {r_cut:.3f} Bohr ({r_cut * BOHR_TO_ANGSTROM:.1f} Angstrom)"
)
print(f"directed edges   : {neighbor_list.shape[1]}")

# %%
# Evaluating the correction
# -------------------------
#
# ``cell`` and ``r_cut`` are both required. Exactly one of ``mesh_dimensions`` and
# ``mesh_spacing`` must be given; there is no accuracy-based default.

energy, forces, virial = fourier_dftd3(
    positions,
    numbers,
    a1=0.4289,
    a2=4.4407,
    s8=0.7875,  # PBE-D3(BJ)
    fd3_params=params,
    cell=cell,
    r_cut=r_cut,
    mesh_dimensions=(32, 32, 32),
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    unit_shifts=unit_shifts,
    compute_virial=True,
)

print(f"\nenergy       : {energy.item():.8f} Hartree")
print(f"             : {energy.item() * HARTREE_TO_EV:.6f} eV")
print(f"max |force|  : {forces.abs().max().item():.3e} Hartree/Bohr")
print(f"virial trace : {virial[0].diagonal().sum().item():.6e} Hartree")

# %%
# Mesh convergence
# ----------------
#
# The mesh is the only accuracy knob on the dispersion sum. Refining it converges the energy;
# there is no cutoff to enlarge.

print("\n mesh      energy (Hartree)      change")
previous = None
for size in (16, 24, 32, 48):
    value = fourier_dftd3(
        positions,
        numbers,
        a1=0.4289,
        a2=4.4407,
        s8=0.7875,
        fd3_params=params,
        cell=cell,
        r_cut=r_cut,
        mesh_dimensions=(size, size, size),
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
    )[0].item()
    change = "" if previous is None else f"{abs(value - previous) / abs(value):.2e}"
    print(f" {size:3d}^3    {value:+.12f}    {change:>9}")
    previous = value

# %%
# Against the real-space sum
# --------------------------
#
# ``dftd3`` computes the same damped dispersion sum, truncated at a cutoff you choose. For a
# :math:`1/r^6` interaction in three dimensions the neglected tail falls off only as
# :math:`1/r^3`, so the energy approaches its converged value slowly while the neighbour list
# grows as the cutoff cubed.
#
# Enlarging the cutoff below should walk ``dftd3`` toward the mesh result. The two do not meet
# exactly: FourierD3 uses a modified coordination-number function that reaches zero at the
# list cutoff, which shifts the environment-dependent coefficients slightly.

reference = fourier_dftd3(
    positions,
    numbers,
    a1=0.4289,
    a2=4.4407,
    s8=0.7875,
    fd3_params=params,
    cell=cell,
    r_cut=r_cut,
    mesh_dimensions=(48, 48, 48),
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    unit_shifts=unit_shifts,
)[0].item()

print(f"\nFourierD3, no dispersion cutoff : {reference:+.10f} Hartree")
print(f"\n{'cutoff (A)':>10} {'dftd3 (Hartree)':>18} {'short by':>10}")
for cutoff_angstrom in (6.0, 8.0, 10.0, 12.0, 15.0):
    cutoff = cutoff_angstrom / BOHR_TO_ANGSTROM
    targets, pointer, images = build_neighbour_list(cutoff)
    value = dftd3(
        positions,
        numbers,
        a1=0.4289,
        a2=4.4407,
        s8=0.7875,
        covalent_radii=rcov,
        r4r2=r4r2,
        c6_reference=c6ab,
        coord_num_ref=cn_ref,
        cell=cell,
        neighbor_list=targets,
        neighbor_ptr=pointer,
        unit_shifts=images,
    )[0].item()
    print(
        f"{cutoff_angstrom:10.1f} {value:+18.10f} "
        f"{abs(value - reference) / abs(reference):9.1%}"
    )

# %%
# The first row is the 6 Angstrom cutoff a machine-learned force field typically supplies ---
# the same list FourierD3 is given here --- and it is already several percent short. The error
# falls slowly from there: 15 Angstrom, two and a half times the range and a much larger
# neighbour list, is still short by a few parts in a thousand. FourierD3 reaches the
# untruncated value from the 6 Angstrom list directly.

# %%
# Summary
# -------
#
# - ``FourierD3Parameters.from_tables`` decomposes the reference tensor once, per species set,
#   independently of the functional.
# - ``fourier_dftd3`` needs a periodic cell and a coordination-number list, and the list's
#   cutoff must equal ``r_cut``.
# - Accuracy is controlled by the mesh rather than by a dispersion cutoff, so the cost of a
#   converged correction does not grow with the interaction range.
#
# For open boundary conditions, or for small molecules where the truncation error does not
# matter, :func:`nvalchemiops.torch.interactions.dispersion.dftd3` remains the right choice.
