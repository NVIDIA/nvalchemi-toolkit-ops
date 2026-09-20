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
JAX FourierD3: Periodic Particle-Mesh Dispersion Under ``jax.jit``
==================================================================

FourierD3 evaluates the DFT-D3(BJ) correction on a particle mesh, so the dispersion sum
carries no pair cutoff. The one real-space cutoff that remains is the short
coordination-number list. The JAX binding is the counterpart of
:func:`nvalchemiops.torch.interactions.dispersion.fourier_dftd3`, and the argument list is the
same minus ``device``.

Eager JAX dispatches each pass separately, and that overhead dominates a correction this
cheap. Wrapping the call in :func:`jax.jit` is what makes it worth using, so that is what
this example concentrates on.

In this example you will learn:

- How to build ``FourierD3Parameters`` for the JAX API
- How to evaluate the correction on a CsCl crystal
- How to ``jax.jit`` the call, and which arguments have to be static to do so

For mesh sensitivity, see ``04_fourier_dftd3_crystal.py``; the behaviour is identical here.

.. important::
    This script is intended as an API demonstration. Do not use this script
    for performance benchmarking; refer to the `benchmarks` folder instead.
"""

# %%
# Setup
# -----
#
# The D3 reference parameters are in atomic units, so every length here is in Bohr. Double
# precision has to be enabled before the first array is created.

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print(
        "This example requires JAX. Install with: pip install 'nvalchemi-toolkit-ops[jax]'"
    )
    raise SystemExit(0) from None

jax.config.update("jax_enable_x64", True)

from nvalchemiops.jax import neighbors  # noqa: E402
from nvalchemiops.jax.interactions.dispersion import (  # noqa: E402
    FourierD3Parameters,
    fourier_dftd3,
)

BOHR_TO_ANGSTROM = 0.529177210544
HARTREE_TO_EV = 27.211386245981

if jax.default_backend() == "cpu":
    print("This example needs a GPU backend to be worth running. Skipping.")
    raise SystemExit(0)

# %%
# Reference parameters
# --------------------
#
# The published Grimme tables, cached by ``utils.py`` on first run. The decomposition of the
# reference tensor happens once on the host, for the species present, and does not depend on
# the damping parameters --- one instance serves every functional.

param_file = (
    Path(os.path.expanduser("~")) / ".cache" / "nvalchemiops" / "dftd3_parameters.pt"
)
# The tables ship as a Torch checkpoint, so reading them needs Torch even on the JAX path.
# This is the provenance route ``03_jax_dftd3_molecule.py`` uses too; only the parameters
# pass through Torch, never the evaluation.
import torch  # noqa: E402

if param_file.exists():
    torch_tables = torch.load(param_file, weights_only=True)
else:
    from utils import extract_dftd3_parameters, save_dftd3_parameters

    torch_tables = extract_dftd3_parameters()
    save_dftd3_parameters(torch_tables)
tables = {name: value.numpy() for name, value in torch_tables.items()}

# %%
# A CsCl crystal
# --------------
#
# Caesium chloride is simple cubic with Cs at the corner and Cl at the body centre.
# FourierD3 is periodic by construction: there is no open-boundary path, because the mesh sum
# is over the infinite lattice.

LATTICE_A = 4.119 / BOHR_TO_ANGSTROM  # Bohr
REPEATS = 4

basis_fractional = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
basis_numbers = np.array([55, 17], dtype=np.int32)  # Cs, Cl

offsets = np.stack(np.meshgrid(*[np.arange(REPEATS)] * 3, indexing="ij"), axis=-1)
offsets = offsets.reshape(-1, 3).astype(float)
fractional = ((offsets[:, None, :] + basis_fractional[None, :, :]) / REPEATS).reshape(
    -1, 3
)
cell_np = np.eye(3) * (LATTICE_A * REPEATS)

positions = jnp.asarray(fractional @ cell_np)
numbers = jnp.asarray(np.tile(basis_numbers, REPEATS**3), dtype=jnp.int32)
cell = jnp.asarray(cell_np)
n_atoms = int(positions.shape[0])

species = sorted(set(np.asarray(numbers).tolist()))
params = FourierD3Parameters.from_tables(
    tables["rcov"].astype(np.float64),
    tables["r4r2"].astype(np.float64),
    tables["c6ab"].astype(np.float64),
    tables["cn_ref"].astype(np.float64),
    species=species,
)
print(f"atoms                : {n_atoms}")
print(f"cell edge            : {float(cell[0, 0]) * BOHR_TO_ANGSTROM:.2f} Angstrom")
print(f"species              : {species}")
print(f"decomposition rank   : {params.rank}")
print(f"mesh channels needed : {params.n_species * params.rank}")

# %%
# The coordination-number list
# ----------------------------
#
# Only the coordination numbers need a neighbour list, so its cutoff is short. It must equal
# ``cutoff``: the counting function reaches zero exactly there, and a list built to a
# different radius would silently truncate it.

cutoff = 6.0 / BOHR_TO_ANGSTROM  # 6 Angstrom, a typical MLFF cutoff, in Bohr

pbc = jnp.array([True, True, True])
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
# ``cell`` and ``cutoff`` are both required, and exactly one of ``mesh_dimensions`` and
# ``mesh_spacing`` must be given --- there is no accuracy-based default.
#
# Energy has shape ``(num_systems,)``, forces ``(n_atoms, 3)`` and the virial
# ``(num_systems, 3, 3)``. Forces and the virial are returned explicitly; ``jax.grad`` of the
# energy is not the route to them here.

damping = dict(a1=0.4289, a2=4.4407, s8=0.7875)  # PBE-D3(BJ)
common = dict(
    fourier_d3_params=params,
    cell=cell,
    cutoff=cutoff,
    mesh_dimensions=(32, 32, 32),
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    unit_shifts=unit_shifts,
)

energy, forces, virial = fourier_dftd3(
    positions, numbers, **damping, compute_virial=True, **common
)

print(f"\nenergy       : {float(energy[0]):.8f} Hartree")
print(f"per atom     : {float(energy[0]) / n_atoms * HARTREE_TO_EV * 1e3:.4f} meV")
print(
    f"max |force|  : {float(jnp.abs(forces).max()):.3e} Hartree/Bohr (zero by symmetry)"
)
print(f"virial trace : {float(jnp.trace(virial[0])):.6e} Hartree")

# %%
# Compiling the call
# ------------------
#
# Everything that changes the shape of the work --- the damping constants, ``cutoff``, the
# mesh and the spline order --- has to be static.
#
# The wrapper below closes over the cell and the parameters as well, so it is reusable for a
# **fixed-cell** trajectory whose input shapes stay the same: positions and numbers are
# traced arguments, and the neighbour list may change content but not length without
# retracing. Under variable-cell dynamics the closed-over cell would go stale silently, so
# pass ``cell`` as an argument there instead --- it is a traced array like any other.

jitted = jax.jit(
    lambda pos, num, box, nl, ptr, sh: fourier_dftd3(
        pos,
        num,
        **damping,
        fourier_d3_params=params,
        cell=box,
        cutoff=cutoff,
        mesh_dimensions=(32, 32, 32),
        neighbor_list=nl,
        neighbor_ptr=ptr,
        unit_shifts=sh,
    )
)

compiled_energy, compiled_forces = jitted(
    positions, numbers, cell, neighbor_list, neighbor_ptr, unit_shifts
)
jax.block_until_ready(compiled_energy)

print(f"\neager energy    : {float(energy[0]):.12f} Hartree")
print(f"compiled energy : {float(compiled_energy[0]):.12f} Hartree")
print(f"force agreement : {float(jnp.abs(compiled_forces - forces).max()):.2e}")

# %%
# Summary
# -------
#
# - ``FourierD3Parameters.from_tables`` decomposes the published reference tensor once per
#   species set, on the host, independently of the functional.
# - ``fourier_dftd3`` is periodic only, needs a coordination-number list whose cutoff equals
#   ``cutoff``, and takes exactly one of ``mesh_dimensions`` or ``mesh_spacing``.
# - Under ``jax.jit`` the shape-determining arguments must be static. Anything left in the
#   closure is frozen at trace time, so pass ``cell`` as an argument if it changes; one
#   compiled step then serves a trajectory whose input shapes are stable.
# - Forces and the virial are returned explicitly, not obtained by differentiation.
#
# For open boundary conditions, or for small systems where the truncation error does not
# matter, :func:`nvalchemiops.jax.interactions.dispersion.dftd3` remains the right choice.
