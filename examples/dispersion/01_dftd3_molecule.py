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
DFT-D3 Dispersion Correction for a Molecule
============================================

This example demonstrates how to compute the DFT-D3 dispersion energy
and forces for a single molecular system using GPU-accelerated kernels.

The DFT-D3 method provides London dispersion corrections to standard DFT
calculations, which is essential for accurately modeling non-covalent interactions.
This implementation uses environment-dependent C6 coefficients and includes
Becke-Johnson damping (D3-BJ).

In this example you will learn:

- How to set up DFT-D3 parameters for a specific functional (PBE0)
- Loading molecular coordinates from an XYZ file
- Computing neighbor lists for non-periodic systems
- Calculating dispersion energies and forces on the GPU

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

import torch

# Import utilities for parameter generation and example DFTD3 module
from utils import (
    DFTD3,
    extract_dftd3_parameters,
    save_dftd3_parameters,
)

from nvalchemiops.torch.neighbors import neighbor_list

# Check for cached parameters, download if needed
# This step downloads ~500 KB of reference data from the Grimme group
param_file = (
    Path(os.path.expanduser("~")) / ".cache" / "nvalchemiops" / "dftd3_parameters.pt"
)
if not param_file.exists():
    print("Downloading DFT-D3 parameters...")
    params = extract_dftd3_parameters()
    save_dftd3_parameters(params)
else:
    params = torch.load(param_file, weights_only=True)
    print("Loaded cached DFT-D3 parameters")

# %%
# Configure Device and Initialize D3 Module
# ------------------------------------------
# We'll use GPU if available for faster computation. The DFTD3 module is
# initialized with functional-specific parameters. Here we use PBE0 parameters:
#
# - s6: scales the C6/R^6 term (always 1.0 for D3-BJ)
# - s8: scales the C8/R^8 term (functional-specific)
# - a1, a2: Becke-Johnson damping parameters (functional-specific)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
print(f"Using device: {device}")

# Initialize with PBE0 parameters
d3_module = DFTD3(
    s6=1.0,  # C6 term coefficient
    s8=1.2177,  # C8 term coefficient (PBE0)
    a1=0.4145,  # BJ damping parameter (PBE0)
    a2=4.8593,  # BJ damping radius (PBE0)
    units="conventional",  # Coordinates in Angstrom, energy in eV
)

# %%
# Load Molecular Structure
# ------------------------
# We'll load a molecular dimer from an XYZ file. This is a simple text format
# where the first line contains the number of atoms, the second line is a
# comment, and subsequent lines contain: element symbol, x, y, z coordinates.

with open("./dimer.xyz") as f:
    lines = f.readlines()
    num_atoms = int(lines[0])
    coords = torch.zeros(num_atoms, 3, device=device, dtype=dtype)
    atomic_numbers = torch.zeros(num_atoms, device=device, dtype=torch.int32)

    for i, line in enumerate(lines[2:]):
        parts = line.split()
        symbol = parts[0]

        # Map element symbols to atomic numbers
        atomic_number = 6 if symbol == "C" else 1  # Carbon or Hydrogen
        atomic_numbers[i] = atomic_number

        # Store coordinates (in Angstrom)
        coords[i, 0] = float(parts[1])
        coords[i, 1] = float(parts[2])
        coords[i, 2] = float(parts[3])

print(f"Loaded molecule with {num_atoms} atoms")
print(f"Coordinates shape: {coords.shape}")

# %%
# Compute Neighbor List
# ---------------------
# The DFT-D3 calculation requires knowing which atoms are within interaction
# range of each other. We use the GPU-accelerated neighbor list from nvalchemiops.
#
# For a non-periodic (molecular) system, we create a large cubic cell and set
# periodic boundary conditions (PBC) to False.

# Large cell to contain the molecule (30 Angstrom box)
cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * 30.0
pbc = torch.tensor([False, False, False], device=device, dtype=torch.bool)

# Compute neighbor list with 20 Angstrom cutoff
# Returns a neighbor matrix (num_atoms x max_neighbors) with padding
neighbor_matrix, num_neighbors_per_atom, neighbor_matrix_shifts = neighbor_list(
    coords,
    cutoff=20.0,  # Interaction cutoff in Angstrom
    cell=cell,
    pbc=pbc,
    method="cell_list",  # O(N) cell list algorithm
    max_neighbors=64,  # Maximum neighbors per atom
)

print(f"Neighbor matrix shape: {neighbor_matrix.shape}")
print(f"Average neighbors per atom: {num_neighbors_per_atom.float().mean():.1f}")

# %%
# Calculate Dispersion Energy and Forces
# ---------------------------------------
# Now we can compute the DFT-D3 dispersion correction. The module returns:
#
# - energies: dispersion energy contribution per atom (eV)
# - forces: dispersion forces on each atom (eV/Angstrom)
# - coord_num: coordination numbers (used internally for C6 calculation)

energies, forces, coord_num = d3_module(
    positions=coords,
    numbers=atomic_numbers,
    neighbor_matrix=neighbor_matrix,
)

# %%
# Results
# -------
# The total dispersion energy is the sum of atomic contributions.
# Forces point in the direction that would lower the energy.

total_energy = energies.sum().item()
max_force = forces.norm(dim=1).max().item()

print(f"\nDispersion Energy: {total_energy:.6f} eV")
print(f"Energy per atom: {total_energy / num_atoms:.6f} eV")
print(f"Maximum force magnitude: {max_force:.6f} eV/Angstrom")
print(f"\nCoordination numbers: {coord_num.cpu().numpy()}")
