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
Neighbor List Rebuild Detection Example
=======================================

This example demonstrates how to use rebuild detection functions in nvalchemiops
to efficiently determine when neighbor lists need to be reconstructed during
molecular dynamics simulations. We'll cover:

- cell_list_needs_rebuild: Detect when atoms move between spatial cells
- neighbor_list_needs_rebuild: Detect when atoms exceed skin distance
- Skin distance approach for efficient neighbor list caching
- Integration with build_cell_list + query_cell_list for MD workflows

Rebuild detection is crucial for MD performance - neighbor lists are expensive to
compute but only need updating when atoms have moved significantly. Smart rebuild
detection can improve simulation performance by 2-10x.
"""

import numpy as np
import torch

from nvalchemiops.torch.neighbors.cell_list import (
    build_cell_list,
    estimate_cell_list_sizes,
    query_cell_list,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
    estimate_max_neighbors,
)
from nvalchemiops.torch.neighbors.rebuild_detection import (
    cell_list_needs_rebuild,
    neighbor_list_needs_rebuild,
)

# %%
# Set up the computation device and simulation parameters
# =======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

# Simulation parameters
num_atoms = 128
box_size = 10.0
cutoff = 2.5
skin_distance = 0.5  # Buffer distance to avoid frequent rebuilds
total_cutoff = cutoff + skin_distance

print("\nSimulation Parameters:")
print(f"  System: {num_atoms} atoms in {box_size}³ box")
print(f"  Neighbor cutoff: {cutoff} Å")
print(f"  Skin distance: {skin_distance} Å")
print(f"  Total cutoff (neighbor + skin): {total_cutoff} Å")

# %%
# Create initial system configuration
# ===================================

print("\n" + "=" * 70)
print("INITIAL SYSTEM SETUP")
print("=" * 70)

# Create simple cubic lattice
n_side = int(np.ceil(num_atoms ** (1 / 3)))
lattice_spacing = box_size / n_side

# Generate lattice positions
d = (torch.arange(n_side, dtype=dtype, device=device) + 0.5) * lattice_spacing
di, dj, dk = torch.meshgrid(d, d, d, indexing="ij")
positions_lattice = torch.stack([di.flatten(), dj.flatten(), dk.flatten()], dim=1)
initial_positions = positions_lattice[:num_atoms].clone()

# System setup
cell = (torch.eye(3, dtype=dtype, device=device) * box_size).unsqueeze(0)
pbc = torch.tensor([True, True, True], device=device)

print(f"Created lattice with spacing {lattice_spacing:.3f} Å")
print(
    f"Initial position range: {initial_positions.min().item():.3f} to {initial_positions.max().item():.3f}"
)

# %%
# Build initial neighbor list with skin distance
# ===============================================

print("\n" + "=" * 70)
print("BUILDING INITIAL NEIGHBOR LIST")
print("=" * 70)

# Estimate memory requirements
max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(
    cell, pbc, total_cutoff
)

print("Memory estimates:")
print(f"  Max cells: {max_total_cells}")
print(f"  Neighbor search radius: {neighbor_search_radius}")

# Allocate cell list cache
cell_list_cache = allocate_cell_list(
    total_atoms=num_atoms,
    max_total_cells=max_total_cells,
    neighbor_search_radius=neighbor_search_radius,
    device=device,
)

(
    cells_per_dimension,
    neighbor_search_radius,
    atom_periodic_shifts,
    atom_to_cell_mapping,
    atoms_per_cell_count,
    cell_atom_start_indices,
    cell_atom_list,
) = cell_list_cache

# Build cell list with total_cutoff (including skin)
build_cell_list(initial_positions, total_cutoff, cell, pbc, *cell_list_cache)

print("\nBuilt cell list:")
print(f"  Cells per dimension: {cells_per_dimension.tolist()}")
print(f"  Neighbor search radius: {neighbor_search_radius.tolist()}")

# Query to get initial neighbors (using actual cutoff, not total)
max_neighbors = estimate_max_neighbors(total_cutoff)
neighbor_matrix = torch.full(
    (num_atoms, max_neighbors), -1, dtype=torch.int32, device=device
)
neighbor_shifts = torch.zeros(
    (num_atoms, max_neighbors, 3), dtype=torch.int32, device=device
)
num_neighbors_arr = torch.zeros(num_atoms, dtype=torch.int32, device=device)

query_cell_list(
    initial_positions,
    cutoff,
    cell,
    pbc,
    *cell_list_cache,
    neighbor_matrix,
    neighbor_shifts,
    num_neighbors_arr,
)

print(f"\nInitial neighbor list (cutoff={cutoff}):")
print(f"  Total pairs: {num_neighbors_arr.sum()}")
print(f"  Avg neighbors per atom: {num_neighbors_arr.float().mean():.2f}")

# Save reference for rebuild detection
reference_positions = initial_positions.clone()
reference_atom_to_cell_mapping = atom_to_cell_mapping.clone()

# %%
# Simulate atomic motion and test rebuild detection
# =================================================

print("\n" + "=" * 70)
print("SIMULATING ATOMIC MOTION")
print("=" * 70)

# Simulate a sequence of small displacements
n_steps = 20
displacement_per_step = 0.15  # Small displacement per step
rebuild_count = 0

print(f"\nSimulating {n_steps} MD steps:")
print(f"  Displacement per step: {displacement_per_step} Å")
print(f"  Skin distance: {skin_distance} Å")
print()

old_positions = reference_positions.clone()
for step in range(n_steps):
    # Apply random small displacement
    displacement = (
        torch.rand(num_atoms, 3, device=device, dtype=dtype) - 0.5
    ) * displacement_per_step
    current_positions = old_positions + displacement

    # Apply periodic boundary conditions
    current_positions = current_positions % box_size

    # Check if cell list needs rebuild (atoms moved between cells)
    cell_rebuild_needed = cell_list_needs_rebuild(
        current_positions=current_positions,
        atom_to_cell_mapping=reference_atom_to_cell_mapping,
        cells_per_dimension=cells_per_dimension,
        cell=cell,
        pbc=pbc,
    )

    # Check if neighbor list needs rebuild (exceeded skin distance)
    neighbor_rebuild_needed = neighbor_list_needs_rebuild(
        reference_positions,
        current_positions,
        skin_distance,
    )

    # Calculate max atomic displacement for reference
    displacements = current_positions - reference_positions
    # Account for PBC
    displacements = displacements - torch.round(displacements / box_size) * box_size
    max_displacement = torch.norm(displacements, dim=1).max().item()

    status = ""
    if cell_rebuild_needed.item() or neighbor_rebuild_needed.item():
        # Rebuild!
        rebuild_count += 1
        status = "REBUILD"

        # Rebuild cell list
        build_cell_list(current_positions, total_cutoff, cell, pbc, *cell_list_cache)

        # Update reference
        reference_positions = current_positions.clone()
        reference_atom_to_cell_mapping = atom_to_cell_mapping.clone()

    print(
        f"Step {step:2d}: max_disp={max_displacement:.4f} Å  "
        f"cell_rebuild={cell_rebuild_needed.item()}, "
        f"neighbor_rebuild={neighbor_rebuild_needed.item()}  {status}"
    )

    # Query neighbors (always use actual cutoff, not total_cutoff)
    query_cell_list(
        current_positions,
        cutoff,
        cell,
        pbc,
        *cell_list_cache,
        neighbor_matrix,
        neighbor_shifts,
        num_neighbors_arr,
    )
    old_positions = current_positions.clone()

print("\nRebuild Statistics:")
print(f"  Total rebuilds: {rebuild_count} / {n_steps} steps")
print(f"  Rebuild rate: {rebuild_count / n_steps * 100:.1f}%")
print(f"  Performance gain: ~{n_steps / max(1, rebuild_count):.1f}x")

# %%
# Demonstrate large atomic motion causing rebuild
# ===============================================

print("\n" + "=" * 70)
print("LARGE DISPLACEMENT TEST")
print("=" * 70)

# Reset to initial configuration
current_positions = initial_positions.clone()
reference_positions = initial_positions.clone()

# Build fresh cell list
build_cell_list(current_positions, total_cutoff, cell, pbc, *cell_list_cache)
reference_atom_to_cell_mapping = atom_to_cell_mapping.clone()

print("\nTesting with increasing displacements:")

for displacement_magnitude in [0.1, 0.3, 0.5, 0.7, 1.0]:
    # Apply displacement to a few atoms
    displaced_positions = reference_positions.clone()
    displaced_positions[:10] += displacement_magnitude

    # Check rebuild need
    cell_rebuild = cell_list_needs_rebuild(
        current_positions=displaced_positions,
        atom_to_cell_mapping=reference_atom_to_cell_mapping,
        cells_per_dimension=cells_per_dimension,
        cell=cell,
        pbc=pbc,
    )

    neighbor_rebuild = neighbor_list_needs_rebuild(
        reference_positions,
        displaced_positions,
        skin_distance,
    )

    rebuild_status = "YES" if (cell_rebuild.item() or neighbor_rebuild.item()) else "NO"
    print(
        f"  Displacement {displacement_magnitude:.1f} Å: "
        f"cell={cell_rebuild.item()}, neighbor={neighbor_rebuild.item()}  "
        f"-> Rebuild: {rebuild_status}"
    )

print("\nExample completed successfully!")
