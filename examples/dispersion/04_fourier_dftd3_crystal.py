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
FourierD3: Periodic Particle-Mesh Dispersion
============================================

The DFT-D3 dispersion correction decays as :math:`1/r^6`. Summed over three dimensions that
leaves a truncation error decaying only as :math:`1/r^3`, so converging it in real space needs
a neighbour list far larger than a machine-learned force field's own, and building that list
comes to dominate the simulation step.

FourierD3 evaluates the dispersion sum on a particle mesh instead, so there is no pair cutoff
on the dispersion at all. It still needs one real-space cutoff --- the short
coordination-number list the force field already builds --- but that radius is fixed by the
coordination-number model rather than by how far the dispersion reaches.

FourierD3 is periodic by construction: the mesh sum runs over the infinite lattice, so there
is no open-boundary path.

In this example you will learn:

- How to turn the published DFT-D3 tables into :class:`FourierD3Parameters`
- How to evaluate the correction on a CsCl crystal with :func:`fourier_dftd3`
- How to check that your mesh is fine enough

.. important::
    This script is intended as an API demonstration. Do not use this script
    for performance benchmarking; refer to the `benchmarks` folder instead.
"""

# %%
# Setup
# -----
#
# The D3 reference parameters are in atomic units, so every length here is in Bohr.

import os
from pathlib import Path

import torch

from nvalchemiops.torch import neighbors
from nvalchemiops.torch.interactions.dispersion import (
    FourierD3Parameters,
    fourier_dftd3,
)

BOHR_TO_ANGSTROM = 0.529177210544
HARTREE_TO_EV = 27.211386245981

# The Torch binding runs on either device; CPU is far slower but agrees to round-off.
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# Reference parameters
# --------------------
#
# The published Grimme tables, the same ones the real-space
# :func:`~nvalchemiops.torch.interactions.dispersion.dftd3` uses. ``utils.py`` downloads and
# parses them on first run and caches the result.

param_file = (
    Path(os.path.expanduser("~")) / ".cache" / "nvalchemiops" / "dftd3_parameters.pt"
)
if param_file.exists():
    tables = torch.load(param_file, weights_only=True)
else:
    from utils import extract_dftd3_parameters, save_dftd3_parameters

    tables = extract_dftd3_parameters()
    save_dftd3_parameters(tables)

# %%
# A CsCl crystal
# --------------
#
# Caesium chloride is simple cubic with Cs at the corner and Cl at the body centre. The
# supercell below is wide enough that the coordination-number cutoff fits inside it.

LATTICE_A = 4.119 / BOHR_TO_ANGSTROM  # Bohr
REPEATS = 4

basis_fractional = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64)
basis_numbers = torch.tensor([55, 17], dtype=torch.int32)  # Cs, Cl

offsets = torch.cartesian_prod(*[torch.arange(REPEATS, dtype=torch.float64)] * 3)
fractional = ((offsets[:, None, :] + basis_fractional[None, :, :]) / REPEATS).reshape(
    -1, 3
)
cell = torch.eye(3, dtype=torch.float64) * (LATTICE_A * REPEATS)

positions = (fractional @ cell).to(device)
numbers = basis_numbers.repeat(REPEATS**3).to(device)
cell = cell.to(device)

print(f"atoms      : {positions.shape[0]}")
print(f"cell edge  : {float(cell[0, 0]) * BOHR_TO_ANGSTROM:.2f} Angstrom")

# %%
# Selecting the species
# ---------------------
#
# ``FourierD3Parameters`` decomposes Grimme's reference tensor into separable factors, once,
# for the species actually present. That decomposition is what lets an environment-dependent
# pair coefficient be evaluated on a mesh at all, and its cost grows with the number of
# distinct elements --- so pass only the ones in your system.
#
# It depends on the tables, the species and the requested tolerance, but not on the damping
# parameters, so one instance serves every functional.

species = sorted(set(numbers.tolist()))
params = FourierD3Parameters.from_tables(
    tables["rcov"].to(device=device, dtype=torch.float64),
    tables["r4r2"].to(device=device, dtype=torch.float64),
    tables["c6ab"].to(device=device, dtype=torch.float64),
    tables["cn_ref"].to(device=device, dtype=torch.float64),
    species=species,
    device=device,
    dtype=torch.float64,
)
print(f"\nspecies              : {species}")
print(f"decomposition rank   : {params.rank}")
print(f"mesh channels needed : {params.n_species * params.rank}")
print(f"reconstruction error : {params.max_relative_error:.2e}")

# %%
# The coordination-number neighbour list
# --------------------------------------
#
# Only the coordination numbers need a list, so the cutoff is short. It must equal ``cutoff``
# exactly: the counting function is built to reach zero there.

cutoff = 6.0 / BOHR_TO_ANGSTROM  # 6 Angstrom, a typical MLFF cutoff, in Bohr

pbc = torch.tensor([True, True, True], device=device)
neighbor_list, neighbor_ptr, unit_shifts = neighbors.neighbor_list(
    positions,
    cutoff=cutoff,
    cell=cell,
    pbc=pbc,
    return_neighbor_list=True,
)
print(f"\nneighbour cutoff : {cutoff * BOHR_TO_ANGSTROM:.1f} Angstrom")
print(f"directed edges   : {neighbor_list.shape[1]}")

# %%
# Evaluating the correction
# -------------------------
#
# ``cell`` and ``cutoff`` are both required. Exactly one of ``mesh_dimensions`` and
# ``mesh_spacing`` must be given; there is no accuracy-based default. Forces and the virial
# are returned directly rather than obtained by differentiating the energy.

DAMPING = dict(a1=0.4289, a2=4.4407, s8=0.7875)  # PBE-D3(BJ)

energy, forces, virial = fourier_dftd3(
    positions,
    numbers,
    **DAMPING,
    fourier_d3_params=params,
    cell=cell,
    cutoff=cutoff,
    mesh_dimensions=(32, 32, 32),
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    unit_shifts=unit_shifts,
    compute_virial=True,
)

n_atoms = positions.shape[0]
print(f"\nenergy         : {energy.item():.8f} Hartree")
print(f"per atom       : {energy.item() / n_atoms * HARTREE_TO_EV * 1e3:.4f} meV")
# Every site in an ideal CsCl lattice sits at a centre of symmetry, so the forces cancel.
# A value at round-off here is a check on the gather, not a null result.
print(
    f"max |force|    : {forces.abs().max().item():.3e} Hartree/Bohr (zero by symmetry)"
)
print(f"virial trace   : {virial[0].diagonal().sum().item():.6e} Hartree")

# %%
# Checking the mesh
# -----------------
#
# Refining the mesh with everything else fixed shows how sensitive the answer is to it. The
# change between successive meshes is an indication, not a certified error bar: it says the
# energy has stopped moving, not that it has stopped moving toward the right value.
#
# The mesh is also not the only control. Spline order and the decomposition tolerance
# (``tol`` in :meth:`FourierD3Parameters.from_tables`) both shift the result, and energy
# converging says nothing on its own about forces or the virial, which converge more slowly.

print(f"\n{'mesh':>6} {'energy/atom (meV)':>20} {'|change|':>12}")
previous = None
for size in (16, 32, 48):
    value = fourier_dftd3(
        positions,
        numbers,
        **DAMPING,
        fourier_d3_params=params,
        cell=cell,
        cutoff=cutoff,
        mesh_dimensions=(size, size, size),
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
    )[0].item()
    per_atom = value / n_atoms * HARTREE_TO_EV * 1e3
    change = "" if previous is None else f"{abs(per_atom - previous):.2e}"
    print(f"{size:3d}^3  {per_atom:20.10f} {change:>12}")
    previous = per_atom

# %%
# Summary
# -------
#
# - ``FourierD3Parameters.from_tables`` decomposes the published reference tensor once, for
#   the species present, independently of the functional.
# - ``fourier_dftd3`` needs a periodic cell and a coordination-number list, and the list's
#   cutoff must equal ``cutoff``.
# - Refine the mesh to check sensitivity; also vary the spline order and the decomposition
#   tolerance before trusting a number, and check forces as well as energy.
#
# FourierD3 applies no cutoff to the dispersion sum, so its cost does not grow as the
# interaction range does. It is not a drop-in replacement for the real-space
# :func:`~nvalchemiops.torch.interactions.dispersion.dftd3`: it uses a modified
# coordination-number function, so the two models differ. See the dispersion user guide for
# that distinction, and prefer ``dftd3`` for open boundary conditions.
