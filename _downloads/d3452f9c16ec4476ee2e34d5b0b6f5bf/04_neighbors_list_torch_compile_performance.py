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
Torch.compile Performance Benefits Example
==========================================

This example demonstrates the significant performance benefits of using torch.compile
with nvalchemiops neighbor list routines in a realistic molecular dynamics simulation.
We'll cover:

- torch.compile integration with build_cell_list and query_cell_list
- Lennard-Jones molecular dynamics simulation using neighbor lists
- Performance comparison between compiled and uncompiled versions
- Speedup analysis and compilation overhead considerations

The build_cell_list and query_cell_list functions are torch.compile compatible,
which means they can be used within compiled functions. This example shows how
to integrate them into an MD workflow for maximum performance.

Note that torch.compile does not necessarily speedup the neighbor list routines itself,
since they are written with CUDA/Warp kernels, but they show that these routines can
be used within compiled functions without incurring graph breaks.

Note that this example is not a comprehensive performance analysis of the neighbor list routines,
but rather a demonstration of how to use them within compiled functions.
"""

import time

import numpy as np
import torch
from system_utils import create_bulk_structure

from nvalchemiops.torch.neighbors.cell_list import (
    build_cell_list,
    cell_list,
    estimate_cell_list_sizes,
    query_cell_list,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
    estimate_max_neighbors,
)

# %%
# Set up the computation device and check torch.compile availability
# ==================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")
print(f"PyTorch version: {torch.__version__}")

# %%
# System parameters and Lennard-Jones potential setup
# ===================================================

print("\n" + "=" * 70)
print("SYSTEM SETUP")
print("=" * 70)

# System parameters
num_atoms = 512  # Medium size system for meaningful timing
box_size = 15.0
temperature = 1.0  # Reduced units
dt = 0.001  # Time step
cutoff = 2.5  # LJ cutoff
skin_distance = 0.5
total_cutoff = cutoff + skin_distance

# Lennard-Jones parameters (argon-like, reduced units)
lj_epsilon = 1.0  # Energy scale
lj_sigma = 1.0  # Length scale

print(f"\nSystem: {num_atoms} atoms in {box_size}³ box")
print(f"LJ parameters: ε={lj_epsilon}, σ={lj_sigma}")
print(f"Cutoff: {cutoff} Å, Skin: {skin_distance} Å")
print(f"Time step: {dt}, Temperature: {temperature}")

# %%
# Create initial system configuration
# ===================================

print("\n" + "=" * 70)
print("INITIAL CONFIGURATION")
print("=" * 70)


# Create FCC lattice at appropriate density
def create_fcc_lattice(
    num_atoms: int, box_size: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create FCC lattice at appropriate density."""
    structure = create_bulk_structure("Al", "fcc", a=4.05, cubic=True)
    # Create supercell to get enough atoms
    n_repeat = int(np.ceil((num_atoms / len(structure)) ** (1 / 3)))
    structure.make_supercell([n_repeat, n_repeat, n_repeat])

    positions = torch.tensor(
        structure.cart_coords[:num_atoms], dtype=dtype, device=device
    )
    # Scale to desired box size
    scale = box_size / structure.lattice.abc[0]
    positions = positions * scale

    cell = (torch.eye(3, dtype=dtype, device=device) * box_size).unsqueeze(0)
    pbc = torch.tensor([True, True, True], device=device)
    return positions, cell, pbc


# Initialize positions and velocities
positions, cell, pbc = create_fcc_lattice(num_atoms, box_size)
velocities = torch.randn_like(positions)

# Remove center-of-mass velocity and scale to target temperature
velocities = velocities - velocities.mean(dim=0)
current_temp = (velocities**2).sum() / (3 * num_atoms)
velocities = velocities * torch.sqrt(temperature / current_temp)

print(f"Initial density: {num_atoms / box_size**3:.4f}")
print(f"Initial temperature: {(velocities**2).sum().item() / (3 * num_atoms):.3f}")

# %%
# Define Lennard-Jones force computation
# ======================================


def compute_lj_forces_matrix(
    positions: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    cell: torch.Tensor,
    epsilon: float = 1.0,
    sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Lennard-Jones forces using neighbor matrix format.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic positions
    neighbor_matrix : torch.Tensor, shape (N, max_neighbors)
        Neighbor atom indices (-1 for invalid)
    neighbor_matrix_shifts : torch.Tensor, shape (N, max_neighbors, 3)
        Unit cell shifts for each neighbor
    num_neighbors : torch.Tensor, shape (N,)
        Number of valid neighbors per atom
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix
    epsilon : float
        LJ energy parameter
    sigma : float
        LJ length parameter

    Returns
    -------
    forces : torch.Tensor, shape (N, 3)
        Atomic forces
    potential : torch.Tensor, scalar
        Total potential energy
    """
    # Create mask for valid neighbors
    mask = neighbor_matrix >= 0

    # Compute Cartesian shifts
    cartesian_shifts = neighbor_matrix_shifts.float() @ cell.squeeze(0)

    # Vectorized force computation
    pos_i = positions.unsqueeze(1)
    pos_j = positions[neighbor_matrix]

    # Distance vectors with PBC
    dr = pos_j - pos_i + cartesian_shifts

    # Compute distances
    r2 = (dr**2).sum(dim=-1)  # (N, max_neighbors)
    r = torch.sqrt(r2.clamp(min=1e-10))  # Avoid division by zero

    # LJ force calculation
    sigma_over_r = sigma / r
    sigma_over_r6 = sigma_over_r**6
    sigma_over_r12 = sigma_over_r6**2

    # Apply mask to only compute for valid neighbors
    u_pair = torch.where(
        mask, 4 * epsilon * (sigma_over_r12 - sigma_over_r6), torch.zeros_like(r)
    )

    force_mag = torch.where(
        mask,
        24 * epsilon / r2 * (sigma_over_r6 - 2 * sigma_over_r12),
        torch.zeros_like(r),
    )

    # Force vectors
    force_vectors = force_mag.unsqueeze(-1) * dr  # (N, max_neighbors, 3)

    # Sum forces on each atom
    forces = force_vectors.sum(dim=1)

    # Total potential energy
    potential = u_pair.sum() * 0.5  # Factor of 0.5 to avoid double counting

    return forces, potential


# %%
# Setup Uncompiled MD Step
# ========================


def md_step_uncompiled(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Perform one MD time step without compilation."""

    # Get neighbor matrix
    neighbor_matrix, num_neighbors, neighbor_shifts = cell_list(
        positions, cutoff, cell, pbc, fill_value=-1
    )

    # Compute forces
    forces, potential = compute_lj_forces_matrix(
        positions,
        neighbor_matrix,
        neighbor_shifts,
        num_neighbors,
        cell,
        lj_epsilon,
        lj_sigma,
    )

    # Velocity Verlet integration
    # Half-step velocity update
    velocities = velocities + 0.5 * dt * forces

    # Position update
    positions = positions + dt * velocities

    # Apply PBC
    box_size = cell[0, 0, 0]
    positions = positions % box_size

    # Recompute forces at new positions
    neighbor_matrix, num_neighbors, neighbor_shifts = cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_shifts,
        num_neighbors=num_neighbors,
        fill_value=-1,
    )

    forces, potential = compute_lj_forces_matrix(
        positions,
        neighbor_matrix,
        neighbor_shifts,
        num_neighbors,
        cell,
        lj_epsilon,
        lj_sigma,
    )

    # Half-step velocity update
    velocities = velocities + 0.5 * dt * forces

    # Calculate kinetic energy
    kinetic = 0.5 * (velocities**2).sum()

    return positions, velocities, potential.item(), kinetic.item()


# %%
# Setup Compiled MD Step with Low-Level API
# =========================================


def create_compiled_md_step():
    """Create a compiled MD step function using build_cell_list + query_cell_list."""

    # Pre-allocate all tensors (required for compilation)
    max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(
        cell, pbc, cutoff
    )

    cell_list_cache = allocate_cell_list(
        total_atoms=num_atoms,
        max_total_cells=max_total_cells,
        neighbor_search_radius=neighbor_search_radius,
        device=device,
    )

    max_neighbors = estimate_max_neighbors(cutoff)

    neighbor_matrix = torch.full(
        (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
    )
    neighbor_shifts = torch.zeros(
        (num_atoms, max_neighbors, 3), dtype=torch.int32, device=device
    )
    num_neighbors_arr = torch.zeros(num_atoms, dtype=torch.int32, device=device)

    @torch.compile(mode="default", fullgraph=True)
    def compiled_md_step(
        positions: torch.Tensor,
        velocities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compiled MD step using low-level API."""

        # Build cell list
        build_cell_list(positions, cutoff, cell, pbc, *cell_list_cache)

        # Query neighbors
        neighbor_matrix.fill_(-1)
        neighbor_shifts.fill_(0)
        num_neighbors_arr.fill_(0)
        query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            *cell_list_cache,
            neighbor_matrix,
            neighbor_shifts,
            num_neighbors_arr,
        )

        # Compute forces
        forces, potential = compute_lj_forces_matrix(
            positions,
            neighbor_matrix,
            neighbor_shifts,
            num_neighbors_arr,
            cell,
            lj_epsilon,
            lj_sigma,
        )

        # Velocity Verlet integration
        velocities = velocities + 0.5 * dt * forces
        positions = positions + dt * velocities

        # Apply PBC
        box_size = cell[0, 0, 0]
        positions = positions % box_size

        # Rebuild and recompute forces
        build_cell_list(positions, cutoff, cell, pbc, *cell_list_cache)
        neighbor_matrix.fill_(-1)
        neighbor_shifts.fill_(0)
        num_neighbors_arr.fill_(0)

        query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            *cell_list_cache,
            neighbor_matrix,
            neighbor_shifts,
            num_neighbors_arr,
        )

        forces, potential = compute_lj_forces_matrix(
            positions,
            neighbor_matrix,
            neighbor_shifts,
            num_neighbors_arr,
            cell,
            lj_epsilon,
            lj_sigma,
        )

        velocities = velocities + 0.5 * dt * forces

        kinetic = 0.5 * (velocities**2).sum()

        return positions, velocities, potential, kinetic

    return compiled_md_step


print("\n" + "=" * 70)
print("CREATING COMPILED MD STEP")
print("=" * 70)

print("\nCompiling MD step function...")
compiled_md_step = create_compiled_md_step()
print("Compilation complete!")

# %%
# Run short simulations and compare performance
# =============================================

print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

n_steps = 50
n_warmup = 5

# Warmup compiled version
print(f"\nWarming up compiled version ({n_warmup} steps)...")
pos_comp = positions.clone()
vel_comp = velocities.clone()
for _ in range(n_warmup):
    pos_comp, vel_comp, _, _ = compiled_md_step(pos_comp, vel_comp)

# Benchmark uncompiled version
print(f"\nBenchmarking uncompiled version ({n_steps} steps)...")
pos_uncomp = positions.clone()
vel_uncomp = velocities.clone()

torch.cuda.synchronize() if device.type == "cuda" else None
start_time = time.time()

for step in range(n_steps):
    pos_uncomp, vel_uncomp, pot_uncomp, kin_uncomp = md_step_uncompiled(
        pos_uncomp, vel_uncomp, cell, pbc, cutoff, dt
    )

torch.cuda.synchronize() if device.type == "cuda" else None
uncompiled_time = time.time() - start_time

# Benchmark compiled version
print(f"Benchmarking compiled version ({n_steps} steps)...")
pos_comp = positions.clone()
vel_comp = velocities.clone()

torch.cuda.synchronize() if device.type == "cuda" else None
start_time = time.time()

for step in range(n_steps):
    pos_comp, vel_comp, pot_comp, kin_comp = compiled_md_step(pos_comp, vel_comp)

torch.cuda.synchronize() if device.type == "cuda" else None
compiled_time = time.time() - start_time

# Results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print("\nPerformance:")
print(
    f"  Uncompiled: {uncompiled_time:.4f} s ({uncompiled_time / n_steps * 1000:.2f} ms/step)"
)
print(
    f"  Compiled:   {compiled_time:.4f} s ({compiled_time / n_steps * 1000:.2f} ms/step)"
)
print(f"  Speedup:    {uncompiled_time / compiled_time:.2f}x")

print(f"\nEnergies after {n_steps} steps:")
print(f"  Potential: {pot_comp:.4f}")
print(f"  Kinetic:   {kin_comp:.4f}")
print(f"  Total:     {pot_comp + kin_comp:.4f}")

# %%
print("\nExample completed successfully!")
