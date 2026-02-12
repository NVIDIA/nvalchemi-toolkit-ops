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
DFT-D3 Dispersion Correction for Batched Crystals
==================================================

This example demonstrates GPU-accelerated DFT-D3 calculations for batches
of periodic crystal structures. Batch processing is essential for high-throughput
computational materials screening and allows efficient parallelization across
multiple structures simultaneously.

We'll load a crystal structure from the Crystallography Open Database (COD),
replicate it to create a batch, and compute dispersion energies and forces for
all structures at once using GPU acceleration.

Key concepts covered:

- Loading crystal structures with periodic boundary conditions
- Converting between fractional and Cartesian coordinates
- Batch processing multiple structures efficiently
- Using batch-aware neighbor lists
- Working with atomic units (Bohr, Hartree) internally

.. note::
   This example assumes ``01_dftd3_molecule.py`` has been run first to cache
   the DFT-D3 parameters.

.. important::
    This script is intended as an API demonstration. Do not use this script
    for performance benchmarking; refer to the `benchmarks` folder instead.
"""

# %%
# Setup and Configuration
# -----------------------
# First, we'll import the necessary modules and set up our batch parameters.
# The low-level D3 kernels use atomic units internally (Bohr for distances,
# Hartree for energies), so we'll need conversion constants.

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import warp as wp
from utils import (
    BOHR_TO_ANGSTROM,  # Conversion: 1 Bohr = 0.529177 Angstrom
    HARTREE_TO_EV,  # Conversion: 1 Hartree = 27.2114 eV
    extract_dftd3_parameters,
    load_d3_parameters,
    save_dftd3_parameters,
)

from nvalchemiops.torch.interactions.dispersion import dftd3
from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.neighbor_utils import estimate_max_neighbors

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
BATCH_SIZE = 8  # Number of structures to process simultaneously
CUTOFF = 20.0  # Interaction cutoff in Angstrom

print(f"Running on device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE} structures")
print(f"Cutoff radius: {CUTOFF} Angstrom")

# %%
# Load DFT-D3 Parameters
# ----------------------
# Load the cached DFT-D3 parameters that contain element-specific C6 coefficients
# and covalent radii. These were downloaded in example 01.

cache_dir = Path(os.path.expanduser("~")) / ".cache" / "nvalchemiops"
param_file = cache_dir / "dftd3_parameters.pt"

if not param_file.exists():
    print("Generating DFT-D3 parameters...")
    params = extract_dftd3_parameters()
    save_dftd3_parameters(params)
else:
    print("Using cached DFT-D3 parameters")

d3_params = load_d3_parameters(param_file)

# Set random seed for reproducibility
torch.manual_seed(42)
torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)

# %%
# Utility Function: Lattice Parameters to Matrix
# -----------------------------------------------
# Crystals are often described by lattice parameters (lengths and angles).
# We need to convert these to a matrix representation for calculations.


def lattice_params_to_matrix(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> torch.Tensor:
    """
    Convert lattice parameters to a 3x3 lattice matrix.

    Parameters
    ----------
    a, b, c : float
        Lattice lengths in Angstroms
    alpha, beta, gamma : float
        Lattice angles in degrees

    Returns
    -------
    torch.Tensor, shape (3, 3)
        Lattice matrix where each row is a lattice vector
    """
    # Convert angles to radians
    alpha_rad = math.radians(alpha)
    beta_rad = math.radians(beta)
    gamma_rad = math.radians(gamma)

    # Compute lattice vectors using standard crystallographic convention
    ax = a
    ay = 0.0
    az = 0.0

    bx = b * math.cos(gamma_rad)
    by = b * math.sin(gamma_rad)
    bz = 0.0

    cx = c * math.cos(beta_rad)
    cy = (
        c
        * (math.cos(alpha_rad) - math.cos(beta_rad) * math.cos(gamma_rad))
        / math.sin(gamma_rad)
    )
    cz = math.sqrt(c**2 - cx**2 - cy**2)

    lattice_matrix = torch.tensor([[ax, ay, az], [bx, by, bz], [cx, cy, cz]])
    return lattice_matrix


# %%
# Load Crystal Structure from COD
# --------------------------------
# We'll load a crystal structure from a JSON file downloaded from the
# Crystallography Open Database (COD). The structure contains:
#
# - Lattice parameters (a, b, c, alpha, beta, gamma)
# - Atomic positions in fractional coordinates
# - Element symbols and atomic numbers

with open("./4300813.json") as read_file:
    data = json.load(read_file)

num_atoms = len(data["atoms"])
print("\nLoaded crystal structure COD ID 4300813")
print(f"Number of atoms in unit cell: {num_atoms}")
print("Lattice parameters:")
print(f"  a = {data['unit_cell']['a']:.3f} Angstrom")
print(f"  b = {data['unit_cell']['b']:.3f} Angstrom")
print(f"  c = {data['unit_cell']['c']:.3f} Angstrom")
print(f"  alpha = {data['unit_cell']['alpha']:.1f}°")
print(f"  beta = {data['unit_cell']['beta']:.1f}°")
print(f"  gamma = {data['unit_cell']['gamma']:.1f}°")

# Convert lattice parameters to matrix
cell = lattice_params_to_matrix(
    data["unit_cell"]["a"],
    data["unit_cell"]["b"],
    data["unit_cell"]["c"],
    data["unit_cell"]["alpha"],
    data["unit_cell"]["beta"],
    data["unit_cell"]["gamma"],
).unsqueeze(0)  # Add batch dimension

# %%
# Extract Atomic Positions and Numbers
# -------------------------------------
# Crystal structures are typically given in fractional coordinates (relative to
# lattice vectors). We'll convert these to Cartesian coordinates for calculations.

fractional = torch.zeros((num_atoms, 3), device=DEVICE, dtype=DTYPE)
numbers = torch.zeros((num_atoms), device=DEVICE, dtype=torch.int32)
atomic_number_mapping = data["atomic_numbers"]

for i, atom in enumerate(data["atoms"]):
    fractional[i, :] = torch.tensor(
        [atom["x"], atom["y"], atom["z"]], device=DEVICE, dtype=DTYPE
    )
    numbers[i] = atomic_number_mapping[atom["element"]]

# Set periodic boundary conditions (all three directions)
pbc = torch.tensor([True, True, True], device=DEVICE, dtype=torch.bool).unsqueeze(0)

# Convert fractional to Cartesian coordinates: r_cart = fractional @ cell^T
cartesian = fractional @ cell[0].T

print("Converted fractional to Cartesian coordinates")
print(f"Positions shape: {cartesian.shape}")

# %%
# Create Batch of Structures
# ---------------------------
# To demonstrate batch processing, we'll replicate the crystal structure
# multiple times. In practice, each structure in the batch could be:
#
# - Different crystal polymorphs
# - Structures at different temperatures
# - Molecular dynamics snapshots
# - High-throughput screening candidates
#
# For this example, we'll use the same structure replicated with small
# random perturbations.

# Replicate positions for batch
positions_batch = cartesian.repeat(BATCH_SIZE, 1)
# Add small random noise to distinguish structures (optional)
positions_batch = torch.normal(positions_batch, 0.00).to(DEVICE)

# Convert from Angstrom to Bohr (atomic units) for DFT-D3 kernels
positions_bohr = positions_batch / BOHR_TO_ANGSTROM

# Replicate atomic numbers
numbers_batch = numbers.repeat(BATCH_SIZE)

print(f"\nCreated batch of {BATCH_SIZE} structures")
print(f"Total atoms in batch: {len(numbers_batch)}")
print(f"Positions shape: {positions_batch.shape}")

# %%
# Set Up Batch Indexing
# ---------------------
# For batch processing, we need to tell the code which atoms belong to which
# structure. We use two tensors for this:
#
# - ``ptr``: Pointer array marking structure boundaries [0, N, 2N, ..., BATCH_SIZE*N]
# - ``batch_idx``: Per-atom array indicating structure membership [0,0,...,1,1,...,2,2,...]

ptr = torch.arange(
    0, (BATCH_SIZE + 1) * num_atoms, num_atoms, device=DEVICE, dtype=torch.int32
)

batch_idx = torch.repeat_interleave(
    torch.arange(BATCH_SIZE, dtype=torch.int32, device=DEVICE), num_atoms
)

# Replicate cell and PBC for each structure in batch
# Convert cell to Bohr for consistency with positions
cell_batch = (cell / BOHR_TO_ANGSTROM).repeat(BATCH_SIZE, 1, 1)
pbc_batch = pbc.repeat(BATCH_SIZE, 1)

print("Batch indexing:")
print(f"  ptr shape: {ptr.shape}")
print(f"  batch_idx shape: {batch_idx.shape}")

# %%
# Compute Batch Neighbor Lists
# -----------------------------
# The neighbor list efficiently identifies which atoms are within interaction
# range. For batches, we use the batch-aware ``batch_cell_list`` method which
# processes all structures simultaneously while respecting batch boundaries.
#
# Key features:
#
# - O(N) scaling per structure using spatial cell lists
# - Handles periodic boundary conditions automatically
# - Returns shifts for atoms interacting across periodic boundaries

# Estimate the number of neighbors per atom
# For this sparse crystal structure (COD 4300813), we need to adjust the estimation
# parameters based on the actual atomic density within the cutoff sphere.
#
# The crystal has a relatively low atomic density. To avoid over-allocation,
# we compute a more accurate atomic_density estimate:
#   - Cutoff sphere volume: (4/3) * π * (20 Å)³ ≈ 33,510 Ų
#   - Unit cell volume can be computed from lattice parameters
#   - For this structure: ~279 actual neighbors suggests density ≈ 0.001 atoms/ų
#
# Strategy: Use a conservative estimate based on material type rather than
# generic defaults. For molecular crystals with organic/light elements,
# atomic_density ≈ 0.001-0.003 atoms/ų is typical in Bohr units.
cutoff_bohr = CUTOFF / BOHR_TO_ANGSTROM
max_neighbors = estimate_max_neighbors(
    cutoff_bohr,
    atomic_density=0.0015,  # Adjusted for sparse molecular crystal
    safety_factor=1.5,  # Modest safety margin for variations
)

# Build neighbor list (in Bohr to match positions)
# Note that we can choose to either use the `max_neighbors` value, or
# if you know the exact number of neighbors per atom for a given cutoff,
# you can use that value instead for optimal performance
neighbor_matrix, num_neighbors_per_atom, neighbor_matrix_shifts = neighbor_list(
    positions_bohr,
    cutoff_bohr,
    cell_batch,
    pbc_batch,
    batch_idx=batch_idx,
    batch_ptr=ptr,
    max_neighbors=max_neighbors,  # can specify an integer
    method="batch_cell_list",  # Batch-aware O(N) algorithm
)

print("\nNeighbor list computed:")
print(f"  Neighbor matrix shape: {neighbor_matrix.shape}")
print(f"  Average neighbors per atom: {num_neighbors_per_atom.float().mean():.1f}")
print(f"  Max neighbors: {num_neighbors_per_atom.max()}")
print(f"  Estimated max neighbors: {max_neighbors}")
if num_neighbors_per_atom.max() > max_neighbors:
    print(
        f"  WARNING: Actual max neighbors ({num_neighbors_per_atom.max()})"
        f" exceeds estimated max neighbors ({max_neighbors})"
    )
elif num_neighbors_per_atom.max() < max_neighbors * 0.5:
    print(
        f"  WARNING: Actual max neighbors ({num_neighbors_per_atom.max()})"
        f" is less than 50% of estimated max neighbors ({max_neighbors})"
    )


# %%
# Calculate DFT-D3 Dispersion
# ----------------------------
# Now we can compute the DFT-D3 dispersion energy and forces for all structures
# in the batch simultaneously. The computation uses GPU-accelerated Warp kernels.
#
# We use PBE0 functional parameters:
#
# - ``s6 = 1.0``: Always 1.0 for D3-BJ damping
# - ``s8 = 1.2177``: PBE0-specific C8 scaling
# - ``a1 = 0.4145``: PBE0-specific BJ damping parameter
# - ``a2 = 4.8593``: PBE0-specific BJ damping radius
#
# .. note::
#    All inputs to the low-level kernels must be in atomic units (Bohr for
#    distances, Hartree for energies). We'll convert back to eV/Angstrom
#    for output.

energies, forces, coord_num = dftd3(
    positions=positions_bohr,
    numbers=numbers_batch,
    neighbor_matrix=neighbor_matrix,
    d3_params=d3_params,
    a1=0.4145,  # BJ damping (PBE0)
    a2=4.8593,  # BJ damping (PBE0)
    s6=1.0,  # C6 coefficient
    s8=1.2177,  # C8 coefficient (PBE0)
    cell=cell_batch,
    neighbor_matrix_shifts=neighbor_matrix_shifts,
    batch_idx=batch_idx,
    device=positions_bohr.device.type,
)
wp.synchronize()  # Ensure GPU computation completes

print("\nDFT-D3 calculation completed")
print(f"Computed energies and forces for {BATCH_SIZE} structures")

# %%
# Results and Analysis
# --------------------
# Convert results from atomic units (Hartree, Bohr) to conventional units
# (eV, Angstrom) for easier interpretation.

energies_eV = energies * HARTREE_TO_EV
forces_eV_Ang = forces * HARTREE_TO_EV / BOHR_TO_ANGSTROM

print("\n" + "=" * 60)
print("DISPERSION ENERGY RESULTS")
print("=" * 60)
print("\nPer-structure dispersion energies (eV):")
for i, energy in enumerate(energies_eV):
    print(f"  Structure {i}: {energy:.6f} eV")

print("\nEnergy statistics:")
print(f"  Mean: {energies_eV.mean():.6f} eV")
print(f"  Std:  {energies_eV.std():.6f} eV")
print(f"  Min:  {energies_eV.min():.6f} eV")
print(f"  Max:  {energies_eV.max():.6f} eV")

print("\n" + "=" * 60)
print("FORCE RESULTS (first 5 atoms)")
print("=" * 60)
print("\nForces (eV/Angstrom):")
print(forces_eV_Ang[:5])

max_force_magnitude = forces_eV_Ang.norm(dim=1).max()
print(f"\nMaximum force magnitude: {max_force_magnitude:.6f} eV/Angstrom")
