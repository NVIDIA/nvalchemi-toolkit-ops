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
JAX FourierD3: Particle-Mesh Dispersion Under ``jax.jit``
=========================================================

FourierD3 evaluates the DFT-D3(BJ) correction on a particle mesh, so the dispersion sum
carries no real-space cutoff and the only neighbour list left is the short one the
coordination numbers need. The JAX binding is the counterpart of
:func:`nvalchemiops.torch.interactions.dispersion.fourier_dftd3`, and the argument list is the
same minus ``device``.

Eager JAX dispatches each of the nine passes separately, and that overhead dominates a
correction this cheap. Wrapping the call in :func:`jax.jit` is what makes it worth using, so
that is what this example concentrates on.

In this example you will learn:

- How to build ``FourierD3Parameters`` for the JAX API
- How to evaluate the correction, and what the returned virial means
- How to ``jax.jit`` the call, and which arguments have to be static to do so

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

# Runs on either backend -- the block-per-atom passes narrow to one thread per atom where
# there are no block launches -- but CPU is far too slow to be worth demonstrating.
if jax.default_backend() == "cpu":
    print("This example needs a GPU backend to be worth running. Skipping.")
    raise SystemExit(0)

# %%
# A periodic cell
# ---------------
#
# A cubic box of carbon and hydrogen. FourierD3 is periodic by construction: there is no
# open-boundary path, because the mesh sum is over the infinite lattice.

rng = np.random.default_rng(0)
box = 20.0  # Bohr
n_atoms = 64
positions = jnp.asarray(rng.uniform(0.0, box, (n_atoms, 3)))
numbers = jnp.asarray(rng.choice([1, 6], n_atoms), dtype=jnp.int32)
cell = jnp.eye(3) * box

# %%
# Reference parameters
# --------------------
#
# The decomposition of Grimme's reference tensor runs once on the host, for the species
# present, and is independent of the damping parameters --- one instance serves every
# functional. The tables below stand in for the published ones; ``examples/dispersion/utils.py``
# downloads and parses the real values.

max_z = 10
n_ref = 5
rcov = np.zeros(max_z)
rcov[[1, 6]] = [0.60, 1.20]
r4r2 = np.zeros(max_z)
r4r2[[1, 6]] = [1.00, 1.40]

c6ab = np.zeros((max_z, max_z, n_ref, n_ref))
cn_ref = np.zeros_like(c6ab)
used = {1: 2, 6: 5}
factors = {z: rng.normal(size=(n, 3)) for z, n in used.items()}
for z_i, n_i in used.items():
    for z_j, n_j in used.items():
        c6ab[z_i, z_j, :n_i, :n_j] = factors[z_i] @ factors[z_j].T + 20.0
        for p in range(n_i):
            cn_ref[z_i, z_j, p, :n_j] = float(np.linspace(0.0, 3.5, n_i)[p])

params = FourierD3Parameters.from_tables(rcov, r4r2, c6ab, cn_ref, species=[1, 6])
print(f"species covered      : {params.n_species}")
print(f"decomposition rank   : {params.rank}")
print(f"mesh channels needed : {params.n_species * params.rank}")

# %%
# The coordination-number list
# ----------------------------
#
# Only the coordination numbers need a neighbour list, so its cutoff is short. It must equal
# ``r_cut``: the counting function is built to reach zero exactly there, and a list built to a
# different radius would silently truncate it.

r_cut = 6.0 / BOHR_TO_ANGSTROM  # 6 Angstrom, the usual MLFF cutoff, in Bohr

pbc = jnp.array([True, True, True])
neighbor_list, neighbor_ptr, unit_shifts = neighbors.neighbor_list(
    positions,
    cutoff=r_cut,
    cell=cell,
    pbc=pbc,
    return_neighbor_list=True,
)
print(
    f"\nneighbour cutoff : {r_cut:.3f} Bohr ({r_cut * BOHR_TO_ANGSTROM:.1f} Angstrom)"
)
print(f"directed edges   : {neighbor_list.shape[1]}")

# %%
# Evaluating the correction
# -------------------------
#
# ``cell`` and ``r_cut`` are both required, and exactly one of ``mesh_dimensions`` and
# ``mesh_spacing`` must be given --- there is no accuracy-based default, because the right
# mesh depends on the cell and on how much error you are willing to accept.
#
# Forces are returned rather than differentiated out. The kernels carry hand-derived
# adjoints and are built with ``enable_backward=False``, which is what keeps the call
# compilable; ``jax.grad`` of the energy is therefore not the route to forces here.

damping = dict(a1=0.4289, a2=4.4407, s8=0.7875)  # PBE-D3(BJ)
common = dict(
    fd3_params=params,
    cell=cell,
    r_cut=r_cut,
    mesh_dimensions=(32, 32, 32),
    neighbor_list=neighbor_list,
    neighbor_ptr=neighbor_ptr,
    unit_shifts=unit_shifts,
)

energy, forces, virial = fourier_dftd3(
    positions, numbers, **damping, compute_virial=True, **common
)

print(f"\nenergy       : {float(energy[0]):.8f} Hartree")
print(f"             : {float(energy[0]) * HARTREE_TO_EV:.6f} eV")
print(f"max |force|  : {float(jnp.abs(forces).max()):.3e} Hartree/Bohr")
print(f"virial trace : {float(jnp.trace(virial[0])):.6e} Hartree")

# %%
# Compiling the call
# ------------------
#
# Everything that changes the shape of the work --- the damping constants, ``r_cut``, the mesh
# and the spline order --- has to be static, because the Warp kernels are specialised on them.
# The arrays stay traced, so a compiled step can be reused across a trajectory as long as the
# neighbour list keeps its length.

jitted = jax.jit(
    lambda pos, num, nl, ptr, sh: fourier_dftd3(
        pos,
        num,
        **damping,
        fd3_params=params,
        cell=cell,
        r_cut=r_cut,
        mesh_dimensions=(32, 32, 32),
        neighbor_list=nl,
        neighbor_ptr=ptr,
        unit_shifts=sh,
    )
)

compiled_energy, compiled_forces = jitted(
    positions, numbers, neighbor_list, neighbor_ptr, unit_shifts
)
jax.block_until_ready(compiled_energy)

print(f"\neager energy    : {float(energy[0]):.12f} Hartree")
print(f"compiled energy : {float(compiled_energy[0]):.12f} Hartree")
print(f"force agreement : {float(jnp.abs(compiled_forces - forces).max()):.2e}")

# %%
# Mesh convergence
# ----------------
#
# The mesh is the only accuracy knob on the dispersion sum. Refining it converges the energy;
# there is no cutoff to enlarge, and no neighbour list that grows while you do it.

print("\n mesh      energy (Hartree)      change")
previous = None
for size in (16, 24, 32, 48):
    value = float(
        fourier_dftd3(
            positions,
            numbers,
            **damping,
            fd3_params=params,
            cell=cell,
            r_cut=r_cut,
            mesh_dimensions=(size, size, size),
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            unit_shifts=unit_shifts,
        )[0][0]
    )
    change = "" if previous is None else f"{abs(value - previous) / abs(value):.2e}"
    print(f" {size:3d}^3    {value:+.12f}    {change:>9}")
    previous = value

# %%
# Summary
# -------
#
# - ``FourierD3Parameters.from_tables`` decomposes the reference tensor once per species set,
#   on the host, independently of the functional.
# - ``fourier_dftd3`` is periodic only, needs a coordination-number list whose cutoff equals
#   ``r_cut``, and takes exactly one of ``mesh_dimensions`` or ``mesh_spacing``.
# - Under ``jax.jit`` the shape-determining arguments must be static; the arrays stay traced,
#   so one compiled step serves a whole trajectory at fixed neighbour-list length.
# - Forces and the virial are returned directly rather than obtained by differentiation.
#
# For open boundary conditions, or for small systems where the truncation error does not
# matter, :func:`nvalchemiops.jax.interactions.dispersion.dftd3` remains the right choice.
