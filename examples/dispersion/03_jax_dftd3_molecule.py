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
JAX DFT-D3 Dispersion Correction for a Molecule
================================================

This example demonstrates how to compute the DFT-D3 dispersion energy
and forces for a single molecular system using the JAX API with
GPU-accelerated Warp kernels.

The DFT-D3 method provides London dispersion corrections to standard DFT
calculations, which is essential for accurately modeling non-covalent interactions.
This implementation uses environment-dependent C6 coefficients and includes
Becke-Johnson damping (D3-BJ).

In this example you will learn:

- How to load DFT-D3 parameters and convert them for the JAX API
- Loading molecular coordinates from an XYZ file into JAX arrays
- Computing neighbor lists for non-periodic systems using the JAX API
- Calculating dispersion energies and forces with the JAX DFT-D3 function
- ``jax.jit`` compilation of the full neighbor list + DFT-D3 pipeline

.. important::
    This script is intended as an API demonstration. Do not use this script
    for performance benchmarking; refer to the `benchmarks` folder instead.
"""

# %%
# Setup and Parameter Loading
# ----------------------------
# First, we need to import the necessary modules and load the DFT-D3 parameters.
# The parameters contain element-specific C6 coefficients and radii that are
# used in the dispersion energy calculation.

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

# Import JAX DFT-D3 API
from nvalchemiops.jax.interactions.dispersion import D3Parameters, dftd3
from nvalchemiops.jax.neighbors import neighbor_list
from nvalchemiops.jax.neighbors.naive import naive_neighbor_list

# Unit conversion constants (CODATA 2022)
BOHR_TO_ANGSTROM = 0.529177210544
HARTREE_TO_EV = 27.211386245981
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# Check for cached parameters, download if needed
param_file = (
    Path(os.path.expanduser("~")) / ".cache" / "nvalchemiops" / "dftd3_parameters.pt"
)
if not param_file.exists():
    print("Downloading DFT-D3 parameters...")
    from utils import extract_dftd3_parameters, save_dftd3_parameters

    params_torch = extract_dftd3_parameters()
    save_dftd3_parameters(params_torch)
else:
    params_torch = torch.load(param_file, weights_only=True)
    print("Loaded cached DFT-D3 parameters")

# Convert PyTorch tensors to JAX arrays
d3_params = D3Parameters(
    rcov=jnp.array(params_torch["rcov"].numpy(), dtype=jnp.float32),
    r4r2=jnp.array(params_torch["r4r2"].numpy(), dtype=jnp.float32),
    c6ab=jnp.array(params_torch["c6ab"].numpy(), dtype=jnp.float32),
    cn_ref=jnp.array(params_torch["cn_ref"].numpy(), dtype=jnp.float32),
)

print(f"Loaded D3 parameters for elements 1-{d3_params.max_z}")

# %%
# Load Molecular Structure
# ------------------------
# We'll load a molecular dimer from an XYZ file. This is a simple text format
# where the first line contains the number of atoms, the second line is a
# comment, and subsequent lines contain: element symbol, x, y, z coordinates.

# Get the directory containing this script
script_dir = Path(__file__).parent
xyz_file = script_dir / "dimer.xyz"

with open(xyz_file) as f:
    lines = f.readlines()
    num_atoms = int(lines[0])

    coords_angstrom = np.zeros((num_atoms, 3), dtype=np.float32)
    atomic_numbers_np = np.zeros(num_atoms, dtype=np.int32)

    for i, line in enumerate(lines[2:]):
        parts = line.split()
        symbol = parts[0]

        # Map element symbols to atomic numbers
        atomic_number = 6 if symbol == "C" else 1  # Carbon or Hydrogen
        atomic_numbers_np[i] = atomic_number

        # Store coordinates (in Angstrom)
        coords_angstrom[i, 0] = float(parts[1])
        coords_angstrom[i, 1] = float(parts[2])
        coords_angstrom[i, 2] = float(parts[3])

# Convert to JAX arrays
coords_angstrom_jax = jnp.array(coords_angstrom)
numbers = jnp.array(atomic_numbers_np, dtype=jnp.int32)

# Convert coordinates to Bohr for DFT-D3 calculation
positions_bohr = coords_angstrom_jax * ANGSTROM_TO_BOHR

print(f"Loaded molecule with {num_atoms} atoms")
print(f"Coordinates shape: {positions_bohr.shape}")

# %%
# Compute Neighbor List
# ---------------------
# The DFT-D3 calculation requires knowing which atoms are within interaction
# range of each other. We use the GPU-accelerated neighbor list from nvalchemiops.
#
# For a non-periodic (molecular) system, we create a large cubic cell and set
# periodic boundary conditions (PBC) to False.

# For a non-periodic (molecular) system, we simply compute pairwise distances
# without periodic boundary conditions.

# Cutoff of 20 Angstrom in Bohr
cutoff_bohr = 20.0 * ANGSTROM_TO_BOHR

# Compute neighbor list using naive method (better for small non-periodic systems)
# The cell_list method requires cell/pbc even for non-periodic systems
neighbor_matrix, num_neighbors_per_atom = neighbor_list(
    positions_bohr,
    cutoff=cutoff_bohr,
    method="naive",
    max_neighbors=64,
)

print(f"Neighbor matrix shape: {neighbor_matrix.shape}")
print(f"Average neighbors per atom: {float(jnp.mean(num_neighbors_per_atom)):.1f}")

# %%
# Calculate Dispersion Energy and Forces
# ---------------------------------------
# Now we can compute the DFT-D3 dispersion correction. The function returns:
#
# - energy: total dispersion energy [num_systems] in Hartree
# - forces: atomic forces [num_atoms, 3] in Hartree/Bohr
# - coord_num: coordination numbers [num_atoms] (dimensionless)
#
# We use PBE0 functional parameters:
# - s6 = 1.0 (C6 term coefficient, standard for D3-BJ)
# - s8 = 1.2177 (C8 term coefficient, PBE0-specific)
# - a1 = 0.4145 (BJ damping parameter, PBE0-specific)
# - a2 = 4.8593 (BJ damping radius, PBE0-specific)

energy, forces, coord_num = dftd3(
    positions=positions_bohr,
    numbers=numbers,
    a1=0.4145,
    a2=4.8593,
    s8=1.2177,
    s6=1.0,
    d3_params=d3_params,
    neighbor_matrix=neighbor_matrix,
    fill_value=num_atoms,
)

# %%
# Results
# -------
# Convert outputs to conventional units for display:
# - Energy: Hartree -> eV
# - Forces: Hartree/Bohr -> eV/Angstrom

# Convert energy to eV
energy_ev = float(energy[0]) * HARTREE_TO_EV

# Convert forces to eV/Angstrom
forces_ev_angstrom = forces * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
max_force = float(jnp.max(jnp.linalg.norm(forces_ev_angstrom, axis=1)))

print(f"\nDispersion Energy: {energy_ev:.6f} eV")
print(f"Energy per atom: {energy_ev / num_atoms:.6f} eV")
print(f"Maximum force magnitude: {max_force:.6f} eV/Angstrom")
print(f"\nCoordination numbers: {np.array(coord_num)}")

# %%
# JIT Compilation
# ---------------
# Demonstrate combining the neighbor list build and DFT-D3 calculation into a
# single ``jax.jit``-compiled function. This fuses the entire pipeline into one
# optimized computation.
#
# For JIT compatibility:
#
# - ``max_neighbors`` must be specified (static array shapes)
# - Functional parameters (``a1``, ``a2``, ``s8``, etc.) must be **static
#   literals** inside the jitted function (required by Warp FFI kernels)
# - ``D3Parameters`` should be constructed inside the jitted function from
#   traced arrays
# - ``fill_value`` and ``num_systems`` should be static

print("\n" + "=" * 70)
print("JIT COMPILATION")
print("=" * 70)


# Define a jitted function that builds neighbors and computes DFT-D3
@jax.jit
def compute_d3_energy_forces(
    positions: jax.Array,
    numbers: jax.Array,
    rcov: jax.Array,
    r4r2: jax.Array,
    c6ab: jax.Array,
    cn_ref: jax.Array,
    cutoff: float = cutoff_bohr,
    max_neighbors: int = 64,
    fill_value: int = num_atoms,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JIT-compiled neighbor list + DFT-D3 pipeline."""
    # Build neighbor matrix inside jit (max_neighbors must be static)
    nbmat, _ = naive_neighbor_list(positions, cutoff, max_neighbors=max_neighbors)

    # Construct D3Parameters inside jit from traced arrays
    params = D3Parameters(rcov=rcov, r4r2=r4r2, c6ab=c6ab, cn_ref=cn_ref)

    # Compute DFT-D3 with PBE0 parameters as static literals
    energy, forces, coord_num = dftd3(
        positions=positions,
        numbers=numbers,
        a1=0.4145,
        a2=4.8593,
        s8=1.2177,
        s6=1.0,
        d3_params=params,
        neighbor_matrix=nbmat,
        fill_value=fill_value,
    )

    return energy, forces, coord_num


# %%
# Run the jitted function:

print("\nCompiling and running jitted DFT-D3 pipeline...")
jit_energy, jit_forces, jit_cn = compute_d3_energy_forces(
    positions_bohr,
    numbers,
    d3_params.rcov,
    d3_params.r4r2,
    d3_params.c6ab,
    d3_params.cn_ref,
)

jit_energy_ev = float(jit_energy[0]) * HARTREE_TO_EV
jit_forces_ev = jit_forces * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
jit_max_force = float(jnp.max(jnp.linalg.norm(jit_forces_ev, axis=1)))

print(f"  JIT dispersion energy: {jit_energy_ev:.6f} eV")
print(f"  JIT max force: {jit_max_force:.6f} eV/Angstrom")

# Compare with non-jitted result
energy_diff = abs(jit_energy_ev - energy_ev)
print(f"  Energy difference vs non-jitted: {energy_diff:.2e} eV")

# Second call should be fast (already compiled)
print("\nRunning jitted function again (should reuse compiled code)...")
jit_energy_2, jit_forces_2, _ = compute_d3_energy_forces(
    positions_bohr,
    numbers,
    d3_params.rcov,
    d3_params.r4r2,
    d3_params.c6ab,
    d3_params.cn_ref,
)
print(
    f"  JIT dispersion energy (2nd call): {float(jit_energy_2[0]) * HARTREE_TO_EV:.6f} eV"
)

# %%
# Summary
# -------
# This example demonstrated:
#
# 1. **Parameter loading** from cached DFT-D3 reference data (Grimme group)
# 2. **Molecular structure** loading from XYZ files into JAX arrays
# 3. **Neighbor list** construction for non-periodic systems
# 4. **DFT-D3 energy and forces** with PBE0 functional parameters
# 5. **Unit conversions** between atomic (Bohr/Hartree) and conventional
#    (Angstrom/eV) units
# 6. **JIT compilation** of the full neighbor list + DFT-D3 pipeline
#
# Key JAX-specific patterns:
#
# - Load PyTorch parameters and convert to JAX arrays via ``jnp.array``
# - Construct ``D3Parameters`` from JAX arrays
# - For ``jax.jit``: use static literals for functional parameters
#   (``a1``, ``a2``, ``s8``), specify ``max_neighbors`` for static shapes,
#   and construct ``D3Parameters`` inside the jitted function

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nKey takeaways:")
print("  - DFT-D3 works in atomic units (Bohr, Hartree) internally")
print("  - Convert Angstrom -> Bohr for positions, Hartree -> eV for energy")
print("  - D3Parameters holds element-specific reference data")
print("  - PBE0 parameters: a1=0.4145, a2=4.8593, s8=1.2177, s6=1.0")
print("  - Use jax.jit to fuse neighbor list + DFT-D3 into one compiled function")
print("\nJAX DFT-D3 example completed successfully!")
