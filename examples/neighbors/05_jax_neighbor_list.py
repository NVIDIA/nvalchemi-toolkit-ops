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
JAX Neighbor List Example
=========================

This example demonstrates how to use the JAX neighbor list API in nvalchemiops
for computing neighbor lists in periodic systems.

In this example you will learn:

- How to use the unified ``neighbor_list()`` API with JAX arrays
- Matrix format vs COO (list) format outputs
- Comparing ``naive_neighbor_list`` and ``cell_list`` algorithms
- Using ``half_fill`` mode for symmetric neighbor lists
- Validating neighbor distances are within cutoff
- ``jax.jit`` compilation of the neighbor matrix

.. important::

    This example is for educational purposes. Do not use it for performance
    benchmarking, as the code includes print statements and small system sizes
    that are not representative of production workloads.
"""

import jax
import jax.numpy as jnp

from nvalchemiops.jax.neighbors import neighbor_list
from nvalchemiops.jax.neighbors.cell_list import cell_list
from nvalchemiops.jax.neighbors.naive import naive_neighbor_list

# %%
# Setup
# =====
# JAX handles device placement automatically. We'll create a random periodic
# system to demonstrate the neighbor list API.

print("=" * 70)
print("JAX NEIGHBOR LIST EXAMPLE")
print("=" * 70)

# System parameters
num_atoms = 200
box_size = 15.0
cutoff = 5.0

# Create random atomic positions using JAX random
key = jax.random.PRNGKey(42)
positions = jax.random.uniform(key, (num_atoms, 3), dtype=jnp.float32) * box_size

# Create a cubic periodic cell: (1, 3, 3) shape
cell = jnp.eye(3, dtype=jnp.float32)[None, ...] * box_size

# Enable periodic boundary conditions in all directions: (1, 3) shape
pbc = jnp.array([[True, True, True]])

print("\nSystem configuration:")
print(f"  Number of atoms: {num_atoms}")
print(f"  Box size: {box_size} Å")
print(f"  Cutoff distance: {cutoff} Å")
print(f"  Positions shape: {positions.shape}")
print(f"  Cell shape: {cell.shape}")
print(f"  PBC shape: {pbc.shape}")

# %%
# Unified API - Matrix Format (default)
# =====================================
# The ``neighbor_list()`` function automatically selects the best algorithm
# based on system size. For small systems (< 5000 atoms), it uses the naive
# O(N²) algorithm. For larger systems, it uses the cell list O(N) algorithm.

print("\n" + "=" * 70)
print("UNIFIED API - MATRIX FORMAT")
print("=" * 70)

# Call the unified API (returns matrix format by default)
neighbor_matrix, num_neighbors, shifts = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc
)

print("\nReturned neighbor matrix format:")
print(f"  neighbor_matrix shape: {neighbor_matrix.shape}")
print(f"  num_neighbors shape: {num_neighbors.shape}")
print(f"  shifts shape: {shifts.shape}")
print("\nStatistics:")
print(f"  Total neighbor pairs: {int(num_neighbors.sum())}")
print(f"  Average neighbors per atom: {float(num_neighbors.mean()):.2f}")
print(f"  Max neighbors for any atom: {int(num_neighbors.max())}")
print(f"  Min neighbors for any atom: {int(num_neighbors.min())}")

# Show first few neighbors of atom 0
print("\nFirst 5 neighbors of atom 0:")
for i in range(min(5, int(num_neighbors[0]))):
    neighbor_idx = int(neighbor_matrix[0, i])
    shift = shifts[0, i].tolist()
    print(f"  Neighbor {i}: atom {neighbor_idx}, shift {shift}")

# %%
# Unified API - COO Format
# ========================
# The COO (coordinate) format is often preferred for graph neural networks.
# Set ``return_neighbor_list=True`` to get this format.

print("\n" + "=" * 70)
print("UNIFIED API - COO FORMAT")
print("=" * 70)

# Get neighbor list in COO format
neighbor_list_coo, neighbor_ptr, shifts_coo = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=True
)

print("\nReturned COO format:")
print(f"  neighbor_list shape: {neighbor_list_coo.shape} (2 x num_pairs)")
print(f"  neighbor_ptr shape: {neighbor_ptr.shape} (CSR pointers)")
print(f"  shifts shape: {shifts_coo.shape}")

source_atoms = neighbor_list_coo[0]
target_atoms = neighbor_list_coo[1]

print("\nStatistics:")
print(f"  Total pairs: {neighbor_list_coo.shape[1]}")
print(f"  Source atoms range: [{int(source_atoms.min())}, {int(source_atoms.max())}]")
print(f"  Target atoms range: [{int(target_atoms.min())}, {int(target_atoms.max())}]")

# Show first few pairs
print("\nFirst 5 neighbor pairs:")
for i in range(min(5, neighbor_list_coo.shape[1])):
    src = int(source_atoms[i])
    tgt = int(target_atoms[i])
    shift = shifts_coo[i].tolist()
    print(f"  Pair {i}: atom {src} -> atom {tgt}, shift {shift}")

# %%
# Algorithm Comparison
# ====================
# The nvalchemiops library provides two main algorithms:
# - ``naive_neighbor_list``: O(N²) - best for small systems
# - ``cell_list``: O(N) - best for large systems
#
# Both should produce identical results.

print("\n" + "=" * 70)
print("ALGORITHM COMPARISON")
print("=" * 70)

# Direct call to naive algorithm
nm_naive, num_naive, shifts_naive = naive_neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc
)

# Direct call to cell list algorithm
nm_cell, num_cell, shifts_cell = cell_list(positions, cutoff, cell=cell, pbc=pbc)

print("\nNaive algorithm (O(N²)):")
print(f"  Total pairs: {int(num_naive.sum())}")
print(f"  Average neighbors: {float(num_naive.mean()):.2f}")

print("\nCell list algorithm (O(N)):")
print(f"  Total pairs: {int(num_cell.sum())}")
print(f"  Average neighbors: {float(num_cell.mean()):.2f}")

# Verify they find the same number of pairs per atom
pairs_match = jnp.allclose(num_naive, num_cell)
print(f"\nResults match: {pairs_match}")
#
# %%
# Distance Validation
# ===================
# Let's verify that all neighbor pairs are actually within the cutoff distance.

print("\n" + "=" * 70)
print("DISTANCE VALIDATION")
print("=" * 70)

# Get neighbor list in COO format for easy distance computation
nlist, nptr, nshifts = naive_neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=True
)

if nlist.shape[1] > 0:
    # Extract source and target positions
    src_idx = nlist[0]
    tgt_idx = nlist[1]

    pos_src = positions[src_idx]
    pos_tgt = positions[tgt_idx]

    # Compute Cartesian shift from lattice shift
    # shifts are in lattice coordinates, multiply by cell vectors
    cell_squeezed = cell.squeeze(0)  # (3, 3)
    cartesian_shifts = jnp.einsum(
        "ij,jk->ik", nshifts.astype(jnp.float32), cell_squeezed
    )

    # Compute distances: r_j - r_i + shift
    diff = pos_tgt - pos_src + cartesian_shifts
    distances = jnp.linalg.norm(diff, axis=1)

    print(f"\nComputed distances for {len(distances)} neighbor pairs:")
    print(f"  Min distance: {float(distances.min()):.4f} Å")
    print(f"  Max distance: {float(distances.max()):.4f} Å")
    print(f"  Mean distance: {float(distances.mean()):.4f} Å")
    print(f"  Cutoff: {cutoff} Å")

    # Check if all distances are within cutoff (with small tolerance)
    within_cutoff = jnp.all(distances <= cutoff + 1e-5)
    print(f"\n  All distances within cutoff: {within_cutoff}")

    # Show distribution of first 10 distances
    print("\nFirst 10 neighbor distances:")
    for i in range(min(10, len(distances))):
        src = int(src_idx[i])
        tgt = int(tgt_idx[i])
        dist = float(distances[i])
        print(f"  Atom {src} -> {tgt}: {dist:.4f} Å")

else:
    print("\nNo neighbor pairs found (empty system or cutoff too small)")

# %%
# JIT compilation
# ===============
# Demonstrate usage of `jax.jit` to include neighborhood computation

print("\n" + "=" * 70)
print("JIT compilation example")
print("=" * 70)


@jax.jit
def run_compute_loop(
    positions,
    cell,
    pbc,
    max_neighbors: int = 128,
    max_total_cells: int = 16,
    cutoff: float = 6.0,
    max_num_atoms: int = 200,
) -> jax.Array:
    """Example of encapsulating a compute loop"""
    num_loops = 100
    all_neighbors = jnp.zeros(
        (num_loops, max_num_atoms, max_neighbors), dtype=positions.dtype
    )
    # generate some random positions
    key = jax.random.PRNGKey(64)
    for i in range(num_loops):
        new_positions = (
            jax.random.normal(key, (max_num_atoms, 3), dtype=positions.dtype)
            + positions
        )
        # for JIT compilation, max_neighbors and total cells **must** be specified to
        # accommoate for static array shapes
        neighbor_matrix, neighbor_ptr, neighbor_matrix_shifts = cell_list(
            new_positions,
            cutoff,
            cell * 1.5,
            pbc,
            max_neighbors=max_neighbors,
            max_total_cells=max_total_cells,
        )
        # in this example we don't do any additional computation
        # other than neighborhoods; include your computation logic
        # within this scope
        all_neighbors = all_neighbors.at[i].set(neighbor_matrix)
    return all_neighbors


# run the compute loop N times
num_loops = 100

print(f"\nRun neighbor computation loop {num_loops} times.")
all_neighbors = run_compute_loop(positions, cell, pbc)
print(f"Rturned neighbor matrix shape: {all_neighbors.shape}")

# %%
# Summary
# =======
# This example demonstrated the JAX neighbor list API in nvalchemiops:
#
# - **Unified API**: ``neighbor_list()`` automatically selects the best algorithm
# - **Matrix format**: Dense (N, max_neighbors) format for neighbor indices
# - **COO format**: Sparse (2, num_pairs) format for graph neural networks
# - **Algorithm choice**: O(N²) naive vs O(N) cell list for different system sizes
# - **Half-fill mode**: Store only unique pairs to save memory
# - **Distance validation**: Verify all pairs are within cutoff

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nKey takeaways:")
print("  - Use neighbor_list() for automatic algorithm selection")
print("  - Use return_neighbor_list=True for COO format (GNNs)")
print("  - Use half_fill=True to store only unique pairs")
print("  - naive_neighbor_list: O(N²), best for < 5000 atoms")
print("  - cell_list: O(N), best for >= 5000 atoms")
print("\nExample completed successfully!")
