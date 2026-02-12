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
Simple Neighbor List Example
=============================

This example demonstrates how to use the basic neighbor list functions in nvalchemiops
on random systems. We'll cover:

1. cell_list: O(N) algorithm using spatial cell lists (best for large systems)
2. naive_neighbor_list: O(N²) algorithm (best for small systems)
3. neighbor_list: Unified wrapper with automatic method selection (RECOMMENDED)
4. Comparison between algorithms
5. Neighbor matrix vs neighbor list (COO) formats
6. build_cell_list + query_cell_list: Lower-level API with caching

The neighbor list construction efficiently finds all atom pairs within a cutoff distance,
which is essential for molecular simulations and materials science calculations.
"""

import torch

from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.cell_list import (
    build_cell_list,
    cell_list,
    estimate_cell_list_sizes,
    query_cell_list,
)
from nvalchemiops.torch.neighbors.naive import naive_neighbor_list
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
    estimate_max_neighbors,
    get_neighbor_list_from_neighbor_matrix,
)

# %%
# Set up the computation device
# =============================
# Choose device (CPU or GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

# %%
# Create random systems of different sizes
# ========================================
# We'll compare performance on small vs large systems

print("\n" + "=" * 70)
print("CREATING TEST SYSTEMS")
print("=" * 70)

# Small system: good for naive algorithm
small_num_atoms = 100
small_box_size = 10.0

# Large system: requires cell list algorithm
large_num_atoms = 30_000
large_box_size = 100.0

cutoff = 5.0

torch.manual_seed(42)

# Small system
small_positions = (
    torch.rand(small_num_atoms, 3, dtype=dtype, device=device) * small_box_size
)
small_cell = (torch.eye(3, dtype=dtype, device=device) * small_box_size).unsqueeze(0)

# Large system
large_positions = (
    torch.rand(large_num_atoms, 3, dtype=dtype, device=device) * large_box_size
)
large_cell = (torch.eye(3, dtype=dtype, device=device) * large_box_size).unsqueeze(0)

# Periodic boundary conditions (all directions periodic)
pbc = torch.tensor([True, True, True], device=device).unsqueeze(0)

print(f"\nSmall system: {small_num_atoms} atoms in {small_box_size}³ box")
print(f"Large system: {large_num_atoms} atoms in {large_box_size}³ box")
print(f"Cutoff distance: {cutoff} Å")

# %%
# Method 1: Cell List Algorithm (O(N) - best for large systems)
# =============================================================
# The cell list algorithm uses spatial decomposition for efficient neighbor finding

print("\n" + "=" * 70)
print("METHOD 1: CELL LIST ALGORITHM (O(N))")
print("=" * 70)

# On large system (where cell list excels)
print("\n--- Large System (30,000 atoms) ---")

# Return neighbor matrix format (default)
neighbor_matrix, num_neighbors, shifts = cell_list(
    large_positions,
    cutoff,
    large_cell,
    pbc,
)

print(f"Returned neighbor matrix: shape {neighbor_matrix.shape}")
print(f"  neighbor_matrix: (num_atoms, max_neighbors) = {neighbor_matrix.shape}")
print(f"  num_neighbors: (num_atoms,) = {num_neighbors.shape}")
print(f"  shifts: (num_atoms, max_neighbors, 3) = {shifts.shape}")
print(f"Total neighbor pairs: {num_neighbors.sum()}")
print(f"Average neighbors per atom: {num_neighbors.float().mean():.2f}")

# Convert neighbor matrix to neighbor list (COO) format for GNNs
neighbor_list_coo, neighbor_list_ptr, shifts_coo = (
    get_neighbor_list_from_neighbor_matrix(
        neighbor_matrix, num_neighbors, shifts, fill_value=large_num_atoms
    )
)  # fill_value is the number of atoms in the system

source_atoms = neighbor_list_coo[0]
target_atoms = neighbor_list_coo[1]
print(f"\nReturned neighbor list (COO format): shape {neighbor_list_coo.shape}")
print(f"  neighbor_list: (2, num_pairs) = {neighbor_list_coo.shape}")
print(f"  source_atoms: {source_atoms.shape}")
print(f"  target_atoms: {target_atoms.shape}")
print(f"  shifts: (num_pairs, 3) = {shifts_coo.shape}")
print(f"Total pairs (COO): {neighbor_list_coo.shape[1]}")

# Show some example neighbors
if neighbor_list_coo.shape[1] > 0:
    print("\nFirst 5 neighbor pairs:")
    for idx in range(min(5, neighbor_list_coo.shape[1])):
        src, tgt = source_atoms[idx].item(), target_atoms[idx].item()
        shift = shifts_coo[idx]
        print(f"  Atom {src} -> Atom {tgt}, shift: {shift.tolist()}")

# %%
# Method 2: Naive Algorithm (O(N²) - best for small systems)
# ==========================================================
# The naive algorithm computes all pairwise distances

print("\n" + "=" * 70)
print("METHOD 2: NAIVE ALGORITHM (O(N²))")
print("=" * 70)

# On small system (where naive algorithm is competitive)
print("\n--- Small System (100 atoms) ---")

# Return neighbor matrix format
neighbor_matrix_naive, num_neighbors_naive, shifts_naive = naive_neighbor_list(
    small_positions, cutoff, cell=small_cell, pbc=pbc
)

print(f"Returned neighbor matrix: shape {neighbor_matrix_naive.shape}")
print(f"Total neighbor pairs: {num_neighbors_naive.sum()}")
print(f"Average neighbors per atom: {num_neighbors_naive.float().mean():.2f}")

# Or return neighbor list (COO) format
neighbor_list_naive, neighbor_ptr_naive, shifts_naive_coo = naive_neighbor_list(
    small_positions, cutoff, cell=small_cell, pbc=pbc, return_neighbor_list=True
)

print(f"\nReturned neighbor list (COO format): shape {neighbor_list_naive.shape}")
print(f"Total pairs (COO): {neighbor_list_naive.shape[1]}")
print(f"Neighbor ptr shape: {neighbor_ptr_naive.shape}")

# %%
# Comparing Cell List vs Naive Algorithm
# ======================================
# Compare performance and verify results match

print("\n" + "=" * 70)
print("ALGORITHM COMPARISON")
print("=" * 70)

print("\n--- Small System (100 atoms) ---")

# Cell list on small system
cell_start = torch.cuda.Event(enable_timing=True)
cell_end = torch.cuda.Event(enable_timing=True)
# Warmup
for _ in range(10):
    _, num_neighbors_cell_small, _ = cell_list(small_positions, cutoff, small_cell, pbc)
torch.cuda.synchronize()
cell_start.record()
for _ in range(10):
    _, num_neighbors_cell_small, _ = cell_list(small_positions, cutoff, small_cell, pbc)
cell_end.record()
torch.cuda.synchronize()
cell_time_small = cell_start.elapsed_time(cell_end) / 10.0

# Naive on small system
naive_start = torch.cuda.Event(enable_timing=True)
naive_end = torch.cuda.Event(enable_timing=True)
for _ in range(10):
    _, num_neighbors_naive_small, _ = naive_neighbor_list(
        small_positions, cutoff, cell=small_cell, pbc=pbc
    )
torch.cuda.synchronize()
naive_start.record()
for _ in range(10):
    _, num_neighbors_naive_small, _ = naive_neighbor_list(
        small_positions, cutoff, cell=small_cell, pbc=pbc
    )
naive_end.record()
torch.cuda.synchronize()
naive_time_small = naive_start.elapsed_time(naive_end) / 10.0

print(f"Cell list:  {cell_time_small} ms, {num_neighbors_cell_small.sum()} pairs")
print(f"Naive:      {naive_time_small} ms, {num_neighbors_naive_small.sum()} pairs")
print(
    f"Results match: {torch.equal(num_neighbors_cell_small, num_neighbors_naive_small)}"
)

print("\n--- Large System (30,000 atoms) ---")

# Cell list on large system
cell_start = torch.cuda.Event(enable_timing=True)
cell_end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
cell_start.record()
for _ in range(10):
    _, num_neighbors_cell_large, _ = cell_list(large_positions, cutoff, large_cell, pbc)
cell_end.record()
torch.cuda.synchronize()
cell_time_large = cell_start.elapsed_time(cell_end) / 10.0

# Naive on large system (will be slower)
naive_start = torch.cuda.Event(enable_timing=True)
naive_end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
naive_start.record()
for _ in range(10):
    _, num_neighbors_naive_large, _ = naive_neighbor_list(
        large_positions, cutoff, cell=large_cell, pbc=pbc
    )
naive_end.record()
torch.cuda.synchronize()
naive_time_large = naive_start.elapsed_time(naive_end) / 10.0

print(f"Cell list:  {cell_time_large} ms, {num_neighbors_cell_large.sum()} pairs")
print(f"Naive:      {naive_time_large} ms, {num_neighbors_naive_large.sum()} pairs")
print(
    f"Results match: {torch.equal(num_neighbors_cell_large, num_neighbors_naive_large)}"
)
print(f"\nSpeedup (cell list vs naive): {naive_time_large / cell_time_large:.1f}x")

# %%
# Method 3: Unified neighbor_list Wrapper (Recommended)
# =====================================================
# The neighbor_list() wrapper provides a unified API that automatically
# selects the best algorithm based on system size and parameters

print("\n" + "=" * 70)
print("METHOD 3: UNIFIED neighbor_list() WRAPPER (RECOMMENDED)")
print("=" * 70)

print("\n--- Automatic Method Selection ---")
# The wrapper automatically chooses the best algorithm:
# - Small systems (< 5000 atoms): naive algorithm
# - Large systems (>= 5000 atoms): cell_list algorithm
# - If cutoff2 is provided: dual cutoff algorithms
# - If batch_idx/batch_ptr is provided: batch algorithms

# Small system - automatically uses naive algorithm
print("\nSmall system (auto-selects naive):")
nm_auto_small, num_auto_small, shifts_auto_small = neighbor_list(
    small_positions, cutoff, cell=small_cell, pbc=pbc
)
print(f"  Total pairs: {num_auto_small.sum()}")
print("  Method selected: naive (auto)")

# Large system - automatically uses cell_list algorithm
print("\nLarge system (auto-selects cell_list):")
nm_auto_large, num_auto_large, shifts_auto_large = neighbor_list(
    large_positions, cutoff, cell=large_cell, pbc=pbc
)
print(f"  Total pairs: {num_auto_large.sum()}")
print("  Method selected: cell_list (auto)")

print("\n--- Explicit Method Selection ---")
# You can also explicitly specify the method
nm_explicit, num_explicit, shifts_explicit = neighbor_list(
    small_positions, cutoff, cell=small_cell, pbc=pbc, method="cell_list"
)
print(f"Explicitly using cell_list on small system: {num_explicit.sum()} pairs")

print("\n--- Passing Arguments to Underlying Methods ---")
# Pass kwargs to the underlying algorithm (e.g., max_neighbors)
nm_kwargs, num_kwargs, shifts_kwargs = neighbor_list(
    small_positions,
    cutoff,
    cell=small_cell,
    pbc=pbc,
    method="naive",
    max_neighbors=50,  # Passed to naive_neighbor_list
    half_fill=True,  # Also passed through
)
print(f"Using kwargs (max_neighbors=50, half_fill=True): {num_kwargs.sum()} pairs")
print(f"Neighbor matrix shape: {nm_kwargs.shape}")

print("\n--- Return Format Options ---")
# Get neighbor list (COO) format directly
nlist_coo, neighbor_ptr_coo, shifts_coo = neighbor_list(
    small_positions, cutoff, cell=small_cell, pbc=pbc, return_neighbor_list=True
)
print(f"Neighbor list (COO) format: {nlist_coo.shape}")
print(f"  Source atoms: {nlist_coo[0].shape}")
print(f"  Target atoms: {nlist_coo[1].shape}")
print(f"  Neighbor ptr: {neighbor_ptr_coo.shape}")

print("\n--- Benefits of the Wrapper ---")
print("✓ Unified API for all neighbor list methods")
print("✓ Automatic algorithm selection based on system size")
print("✓ Easy to switch between methods for testing")
print("✓ Consistent return values across all methods")
print("✓ Pass-through of method-specific kwargs")

# %%
# Method 4: Low-level Cell List API with Caching
# ==============================================
# For advanced users: separate build and query phases for efficient caching

print("\n" + "=" * 70)
print("METHOD 4: LOW-LEVEL CELL LIST API (BUILD + QUERY)")
print("=" * 70)

# This approach is useful when you want to:
# 1. Build cell list once, query multiple times with different cutoffs
# 2. Cache cell list for MD simulations with rebuild detection
# 3. Have fine control over memory allocation

positions = large_positions
cell_tensor = large_cell

print("\n--- Step 1: Estimate Memory Requirements ---")
max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(
    cell_tensor, pbc, cutoff
)
print(f"Estimated max cells: {max_total_cells}")
print(f"Neighbor search radius: {neighbor_search_radius}")

print("\n--- Step 2: Allocate Memory ---")
cell_list_cache = allocate_cell_list(
    total_atoms=positions.shape[0],
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

print(f"Allocated cell list cache with {len(cell_list_cache)} components")

print("\n--- Step 3: Build Cell List ---")
build_cell_list(positions, cutoff, cell_tensor, pbc, *cell_list_cache)
print(f"Built cell list with {cells_per_dimension.tolist()} cells per dimension")

print("\n--- Step 4: Query Neighbors ---")
# Allocate output tensors
max_neighbors = estimate_max_neighbors(cutoff, safety_factor=20.0)
neighbor_matrix_query = torch.full(
    (positions.shape[0], max_neighbors), -1, dtype=torch.int32, device=device
)
neighbor_shifts_query = torch.zeros(
    (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
)
num_neighbors_query = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)

query_cell_list(
    positions,
    cutoff,
    cell_tensor,
    pbc,
    *cell_list_cache,
    neighbor_matrix_query,
    neighbor_shifts_query,
    num_neighbors_query,
)

print(f"Query found {num_neighbors_query.sum()} neighbor pairs")
print(f"Average neighbors per atom: {num_neighbors_query.float().mean():.2f}")

# %%
# Understanding Half-Fill Mode
# ============================
# Control whether to store both (i,j) and (j,i) pairs

print("\n" + "=" * 70)
print("HALF-FILL MODE")
print("=" * 70)

positions = small_positions
cell_tensor = small_cell

# Full neighbor list: both (i,j) and (j,i) stored
print("\n--- Full Neighbor List (half_fill=False) ---")
neighbor_matrix_full, num_neighbors_full, _ = cell_list(
    positions, cutoff, cell_tensor, pbc, half_fill=False
)
print(f"Total pairs: {num_neighbors_full.sum()}")

# Half-fill: only (i,j) where i < j (or with non-zero periodic shift)
print("\n--- Half-Fill Neighbor List (half_fill=True) ---")
neighbor_matrix_half, num_neighbors_half, _ = cell_list(
    positions, cutoff, cell_tensor, pbc, half_fill=True
)
print(f"Total pairs: {num_neighbors_half.sum()}")
print(f"Reduction: {num_neighbors_half.sum() / num_neighbors_full.sum():.1%} of full")

# %%
# Validate neighbor distances
# ===========================

print("\n" + "=" * 70)
print("DISTANCE VALIDATION")
print("=" * 70)

# Use neighbor list format for easy iteration
neighbor_list, neighbor_ptr, shifts = cell_list(
    small_positions, cutoff, small_cell, pbc, return_neighbor_list=True
)

if neighbor_list.shape[1] > 0:
    # Calculate distances for first 10 pairs
    n_check = min(10, neighbor_list.shape[1])
    source_atoms = neighbor_list[0, :n_check]
    target_atoms = neighbor_list[1, :n_check]
    pair_shifts = shifts[:n_check]

    # Compute Cartesian shifts
    cartesian_shifts = pair_shifts.float() @ small_cell.squeeze(0)

    # Calculate distances
    pos_i = small_positions[source_atoms]
    pos_j = small_positions[target_atoms]
    distances = torch.norm(pos_j - pos_i + cartesian_shifts, dim=1)

    print(f"\nFirst {n_check} neighbor distances:")
    for idx in range(n_check):
        src = source_atoms[idx].item()
        tgt = target_atoms[idx].item()
        dist = distances[idx].item()
        print(f"  Pair {idx}: atom {src} -> {tgt}, distance = {dist:.4f} Å")

    max_distance = distances.max().item()
    print(f"\nMax distance in pairs: {max_distance:.4f} Å")
    print(f"Cutoff distance: {cutoff:.4f} Å")
    if max_distance <= cutoff + 1e-5:
        print("✓ All neighbor distances are within the cutoff")
    else:
        print("✗ Some distances are outside the cutoff")

# %%
print("\nExample completed successfully!")
