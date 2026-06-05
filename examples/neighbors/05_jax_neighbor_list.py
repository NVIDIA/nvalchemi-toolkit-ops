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
- Estimating dispatch cost with ``estimate_neighbor_list_costs`` /
  ``suggest_neighbor_list_method``
- Evaluating an inline Warp ``pair_fn`` (per-pair energy and force)

.. important::

    This example is for educational purposes. Do not use it for performance
    benchmarking, as the code includes print statements and small system sizes
    that are not representative of production workloads.
"""

import sys

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print(
        "This example requires JAX. Install with: pip install 'nvalchemi-toolkit-ops[jax]'"
    )
    sys.exit(0)

try:
    import warp as wp

    from nvalchemiops.jax.neighbors import (
        estimate_neighbor_list_costs,
        neighbor_list,
        suggest_neighbor_list_method,
    )
except Exception as exc:
    print(
        f"JAX/Warp backend unavailable ({exc}). This example requires a CUDA-backed runtime."
    )
    sys.exit(0)
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
# The ``neighbor_list()`` function provides a consistent entry point for JAX
# neighbor-list construction while preserving the same matrix-format outputs
# used by the direct algorithms.

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
# The nvalchemiops library provides direct access to two main algorithms:
# - ``naive_neighbor_list``: O(N²) all-pairs distance checks
# - ``cell_list``: spatial decomposition for larger systems
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
print(f"Returned neighbor matrix shape: {all_neighbors.shape}")

# %%
# Cost-model dispatch
# ===================
# ``estimate_neighbor_list_costs`` and ``suggest_neighbor_list_method`` expose the
# geometry cost model that ``neighbor_list(method=None)`` uses internally. Call them
# once on per-system geometry (``batch_ptr``, ``cell``, ``pbc``, ``cutoff``) and pass
# the returned name as an explicit ``method=`` so repeated builds skip the
# auto-dispatch host read. They synchronize on the host (a small selector kernel runs
# on the device and its result is read back), so call them outside ``jax.jit``.

print("\n" + "=" * 70)
print("COST-MODEL DISPATCH")
print("=" * 70)

batch_ptr = jnp.array([0, num_atoms], dtype=jnp.int32)
cost_report = estimate_neighbor_list_costs(batch_ptr, cell, pbc, cutoff)
print("\nFeasible strategies (cheapest first):")
for strategy, cost in cost_report:
    print(f"  {strategy:24s} estimated cost (arbitrary units): {cost:.3g}")

suggested_method = suggest_neighbor_list_method(batch_ptr, cell, pbc, cutoff)
print(f"\nSuggested method: {suggested_method}")

# Reuse the suggestion as an explicit ``method=`` on the unified entry point.
nm_suggested, num_suggested, _ = neighbor_list(
    positions, cutoff, cell=cell, pbc=pbc, method=suggested_method
)
print(f"Total pairs via suggested method: {int(num_suggested.sum())}")

# %%
# Inline pair potentials with ``pair_fn``
# =======================================
# A Warp ``pair_fn`` evaluates a pairwise potential as neighbors are enumerated,
# returning per-pair energy and force in the same pass (no second loop over the
# list). On JAX, ``pair_fn`` is exposed through the direct algorithm bindings
# (``naive_neighbor_list`` / ``cell_list``), not the unified ``neighbor_list``. The
# ``pair_energies`` / ``pair_forces`` buffers are auto-allocated, appended to the
# return tuple, and forward-only (use ``return_distances`` / ``return_vectors`` for
# differentiable geometry).

print("\n" + "=" * 70)
print("INLINE PAIR POTENTIALS (pair_fn)")
print("=" * 70)


@wp.func
def lj_pair_fn(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    """Lennard-Jones per-pair energy and force from per-atom (epsilon, sigma)."""
    epsilon = wp.sqrt(pair_params[i, 0] * pair_params[j, 0])
    sigma = 0.5 * (pair_params[i, 1] + pair_params[j, 1])
    sr = sigma / distance
    sr2 = sr * sr
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    energy = 4.0 * epsilon * (sr12 - sr6)
    force = (24.0 * epsilon * (sr6 - 2.0 * sr12) / (distance * distance)) * r_ij
    return energy, force


# Per-atom (epsilon, sigma) table, shape (num_atoms, 2), float32.
pair_params = jnp.stack(
    [
        jnp.full((num_atoms,), 0.0104, dtype=jnp.float32),  # epsilon
        jnp.full((num_atoms,), 3.40, dtype=jnp.float32),  # sigma
    ],
    axis=1,
)

# pair_fn returns auto-allocated energy/force buffers appended after the matrix
# outputs: (neighbor_matrix, num_neighbors, shifts, pair_energies, pair_forces).
nm_pair, num_pair, _shifts_pair, pair_energies, pair_forces = naive_neighbor_list(
    positions,
    cutoff,
    cell=cell,
    pbc=pbc,
    max_neighbors=128,
    pair_fn=lj_pair_fn,
    pair_params=pair_params,
)

print("\nPair-output buffers (matrix-aligned with the neighbor matrix):")
print(f"  pair_energies shape: {pair_energies.shape}")
print(f"  pair_forces shape:   {pair_forces.shape}")
print("  Auto-allocated, returned, and forward-only.")

# %%
# Summary
# =======
# This example demonstrated the JAX neighbor list API in nvalchemiops:
#
# - **Unified API**: ``neighbor_list()`` provides a single entry point
# - **Matrix format**: Dense (N, max_neighbors) format for neighbor indices
# - **COO format**: Sparse (2, num_pairs) format for graph neural networks
# - **Algorithm choice**: Direct naive and cell-list calls for comparison
# - **Half-fill mode**: Store only unique pairs to save memory
# - **Distance validation**: Verify all pairs are within cutoff
# - **Cost-model dispatch**: ``estimate``/``suggest`` helpers pick a method
# - **Inline pair_fn**: Per-pair energy/force via the direct algorithm bindings

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nKey takeaways:")
print("  - Use neighbor_list() as the unified JAX entry point")
print("  - Use return_neighbor_list=True for COO format (GNNs)")
print("  - Use half_fill=True to store only unique pairs")
print("  - naive_neighbor_list performs O(N²) all-pairs checks")
print("  - cell_list uses spatial decomposition")
print("  - suggest_neighbor_list_method picks a method from geometry")
print("  - pair_fn evaluates a pairwise potential during enumeration")
print("\nExample completed successfully!")
