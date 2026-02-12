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
Batch Neighbor List Example
============================

This example demonstrates how to use the batch neighbor list functions in nvalchemiops
with multiple molecular and crystalline systems.
We'll cover:

- batch_cell_list: Batch O(N) processing with spatial cell lists
- batch_naive_neighbor_list: Batch O(N²) processing for small systems
- Using batch_idx to identify which system each atom belongs to
- Processing heterogeneous batches with different sizes and parameters
- Comparing batch vs single-system processing

Batch processing allows efficient computation of neighbor lists for multiple systems
simultaneously, which is essential for high-throughput molecular screening and
ensemble simulations.
"""

import numpy as np
import torch
from system_utils import create_bulk_structure, create_molecule_structure

from nvalchemiops.torch.neighbors.batch_cell_list import batch_cell_list
from nvalchemiops.torch.neighbors.batch_naive import batch_naive_neighbor_list
from nvalchemiops.torch.neighbors.cell_list import cell_list

# %%
# Set up the computation device
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

# %%
# Create multiple systems
# =======================
# We'll create a diverse set of molecular and crystalline systems

print("\n" + "=" * 70)
print("CREATING SYSTEMS")
print("=" * 70)

# Create molecular systems
water = create_molecule_structure("H2O", box_size=15.0)
co2 = create_molecule_structure("CO2", box_size=12.0)
methane = create_molecule_structure("CH4", box_size=10.0)

# Create a small crystalline system
fcc_al = create_bulk_structure("Al", "fcc", a=4.05, cubic=True)
# Create 2x2x2 supercell
fcc_al.make_supercell([2, 2, 2])

# Collect all systems
systems = [water, co2, methane, fcc_al]
system_names = ["H2O", "CO2", "CH4", "Al-fcc(2x2x2)"]

print(f"\nCreated {len(systems)} systems:")
for name, system in zip(system_names, systems):
    lattice_abc = system.lattice.abc
    print(
        f"  {name}: {len(system)} atoms, cell: [{lattice_abc[0]:.2f}, {lattice_abc[1]:.2f}, {lattice_abc[2]:.2f}]"
    )

# %%
# Convert systems to batch format
# ================================
# Combine all systems into the batch format required by nvalchemiops

print("\n" + "=" * 70)
print("CONVERTING TO BATCH FORMAT")
print("=" * 70)

# Extract positions, cells, and PBC from all systems
all_positions = []
all_cells = []
all_pbc = []
batch_indices = []

for sys_idx, system in enumerate(systems):
    all_positions.append(system.cart_coords)
    all_cells.append(system.lattice.matrix)
    all_pbc.append(
        np.array([True, True, True])
    )  # pymatgen structures are always periodic
    # Create batch_idx: which system does each atom belong to
    batch_indices.extend([sys_idx] * len(system))

# Convert to torch tensors
positions = torch.tensor(np.vstack(all_positions), dtype=dtype, device=device)
cells = torch.tensor(np.array(all_cells), dtype=dtype, device=device).reshape(-1, 3, 3)
pbc = torch.tensor(np.array(all_pbc), device=device).reshape(-1, 3)
batch_idx = torch.tensor(batch_indices, dtype=torch.int32, device=device)

# Define single cutoff for all systems
cutoff = 5.0

print("\nBatch configuration:")
print(f"  Total atoms: {positions.shape[0]}")
print(f"  Number of systems: {len(systems)}")
print(f"  batch_idx shape: {batch_idx.shape}")
print(f"  Cutoff: {cutoff} Å")

# Show batch_idx distribution
atom_counts = [len(system) for system in systems]
print(f"\n  Atoms per system: {atom_counts}")

for sys_idx, (name, count) in enumerate(zip(system_names, atom_counts)):
    mask = batch_idx == sys_idx
    print(f"    System {sys_idx} ({name}): {mask.sum()} atoms (batch_idx={sys_idx})")

# %%
# Method 1: Batch Cell List Algorithm (O(N))
# ==========================================
# Process all systems simultaneously with cell list algorithm

print("\n" + "=" * 70)
print("METHOD 1: BATCH CELL LIST (O(N))")
print("=" * 70)

# Return neighbor matrix format (default)
neighbor_matrix_batch, num_neighbors_batch, shifts_batch = batch_cell_list(
    positions, cutoff, cells, pbc, batch_idx
)

print(f"\nReturned neighbor matrix: {neighbor_matrix_batch.shape}")
print(f"  Total neighbor pairs: {num_neighbors_batch.sum()}")
print(f"  Average neighbors per atom: {num_neighbors_batch.float().mean():.2f}")

# Or return neighbor list (COO) format
neighbor_list_batch, neighbor_ptr_batch, shifts_coo = batch_cell_list(
    positions, cutoff, cells, pbc, batch_idx, return_neighbor_list=True
)

print(f"\nReturned neighbor list (COO): {neighbor_list_batch.shape}")
print(f"  Total pairs: {neighbor_list_batch.shape[1]}")
print(f"  Neighbor ptr shape: {neighbor_ptr_batch.shape}")

# Analyze results per system
print("\nPairs per system:")
start_idx = 0
for sys_idx, (name, count) in enumerate(zip(system_names, atom_counts)):
    end_idx = start_idx + count
    system_num_neighbors = num_neighbors_batch[start_idx:end_idx].sum().item()
    avg_neighbors = system_num_neighbors / count if count > 0 else 0

    print(f"  {name}: {system_num_neighbors} pairs, {avg_neighbors:.1f} neighbors/atom")
    start_idx = end_idx

# %%
# Method 2: Batch Naive Algorithm (O(N²))
# =======================================
# For comparison, use naive algorithm on batch of small systems

print("\n" + "=" * 70)
print("METHOD 2: BATCH NAIVE ALGORITHM (O(N²))")
print("=" * 70)

# Create batch of small systems for naive algorithm demo
small_systems = [water, co2, methane]  # Exclude larger Al crystal
small_system_names = ["H2O", "CO2", "CH4"]

# Convert to batch format
small_positions_list = [
    torch.tensor(s.cart_coords, dtype=dtype, device=device) for s in small_systems
]
small_positions = torch.cat(small_positions_list)

small_cells = torch.stack(
    [torch.tensor(s.lattice.matrix, dtype=dtype, device=device) for s in small_systems]
)

small_pbc = torch.stack(
    [torch.tensor([True, True, True], device=device) for s in small_systems]
)

# Create batch_idx
small_batch_idx = torch.cat(
    [
        torch.full((len(s),), i, dtype=torch.int32, device=device)
        for i, s in enumerate(small_systems)
    ]
)

print(f"Small systems batch: {small_positions.shape[0]} total atoms")

# Batch naive neighbor list
neighbor_matrix_naive, num_neighbors_naive, shifts_naive = batch_naive_neighbor_list(
    small_positions,
    cutoff,
    batch_idx=small_batch_idx,
    cell=small_cells,
    pbc=small_pbc,
)

print(f"Returned neighbor matrix: {neighbor_matrix_naive.shape}")
print(f"Total neighbor pairs: {num_neighbors_naive.sum()}")

# Compare with batch cell list on same systems
neighbor_matrix_cell, num_neighbors_cell, _ = batch_cell_list(
    small_positions, cutoff, small_cells, small_pbc, small_batch_idx
)

print("\nVerification (naive vs cell list):")
print(f"  Naive total pairs: {num_neighbors_naive.sum()}")
print(f"  Cell list total pairs: {num_neighbors_cell.sum()}")
print(f"  Results match: {torch.equal(num_neighbors_naive, num_neighbors_cell)}")

# %%
# Extract individual system results from batch
# ============================================

print("\n" + "=" * 70)
print("EXTRACTING INDIVIDUAL SYSTEM RESULTS")
print("=" * 70)


def extract_system_neighbors(system_idx, neighbor_list, batch_idx):
    """Extract neighbor list for a specific system from batch results (COO format)."""
    source_atoms = neighbor_list[0]
    target_atoms = neighbor_list[1]

    # Get atom range for this system
    system_mask = batch_idx == system_idx
    system_atom_indices = torch.where(system_mask)[0]
    first_atom = system_atom_indices[0].item()
    last_atom = system_atom_indices[-1].item()

    # Find pairs where source atom belongs to this system
    pair_mask = (source_atoms >= first_atom) & (source_atoms <= last_atom)

    # Extract and adjust indices to be local to the system
    system_source = source_atoms[pair_mask] - first_atom
    system_target = target_atoms[pair_mask] - first_atom

    return system_source, system_target, pair_mask


# Analyze each system individually
print("\nPer-system analysis:")
for sys_idx, (system, name) in enumerate(zip(systems, system_names)):
    sys_source, sys_target, pair_mask = extract_system_neighbors(
        sys_idx, neighbor_list_batch, batch_idx
    )

    n_atoms = len(system)
    n_pairs = len(sys_source)
    avg_neighbors = n_pairs / n_atoms if n_atoms > 0 else 0

    print(f"\n{name}:")
    print(f"  Atoms: {n_atoms}")
    print(f"  Neighbor pairs: {n_pairs}")
    print(f"  Avg neighbors per atom: {avg_neighbors:.2f}")

    if n_pairs > 0:
        # Show first few pairs
        print("  Sample pairs: ", end="")
        for i in range(min(3, n_pairs)):
            print(f"({sys_source[i]}->{sys_target[i]})", end=" ")
        print()

# %%
# Compare batch vs single-system processing
# =========================================

print("\n" + "=" * 70)
print("BATCH VS SINGLE-SYSTEM COMPARISON")
print("=" * 70)

# Process each system individually and compare with batch results
print("\nVerifying batch results against single-system calculations:\n")

for sys_idx, (system, name) in enumerate(zip(systems, system_names)):
    # Convert system to tensors
    sys_positions = torch.tensor(system.cart_coords, dtype=dtype, device=device)
    sys_cell = torch.tensor(
        system.lattice.matrix, dtype=dtype, device=device
    ).unsqueeze(0)
    sys_pbc = torch.tensor([True, True, True], device=device)

    # Calculate single system neighbor list
    _, num_neighbors_single, _ = cell_list(sys_positions, cutoff, sys_cell, sys_pbc)
    single_total = num_neighbors_single.sum().item()

    # Extract from batch results
    system_mask = batch_idx == sys_idx
    batch_total = num_neighbors_batch[system_mask].sum().item()

    # Compare
    match_status = "✓" if single_total == batch_total else "✗"
    print(
        f"{match_status} {name:15s}: single={single_total:4d}, batch={batch_total:4d}"
    )

# %%
# Demonstrate heterogeneous batch parameters
# ==========================================
# Show that each system can have different properties

print("\n" + "=" * 70)
print("HETEROGENEOUS BATCH PARAMETERS")
print("=" * 70)

print("\nBatch supports different parameters per system:")
print(f"  System sizes: {atom_counts}")
print("  Unit cells (box sizes):")
for idx, (name, system) in enumerate(zip(system_names, systems)):
    cell_size = system.lattice.abc[0]
    print(f"    {name}: {cell_size:.2f} Å")

print("  PBC settings:")
for idx, (name, system) in enumerate(zip(system_names, systems)):
    pbc_str = "TTT"  # pymatgen structures are always periodic
    print(f"    {name}: [{pbc_str}]")

print(f"\n  Single cutoff used for all: {cutoff} Å")
print("  (Note: Currently all systems share the same cutoff)")

# %%
print("\nExample completed successfully!")
