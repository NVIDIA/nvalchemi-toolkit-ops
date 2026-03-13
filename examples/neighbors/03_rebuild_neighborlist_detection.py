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

- ``cell_list_needs_rebuild``: Detect when atoms move between spatial cells
- ``neighbor_list_needs_rebuild``: Detect when atoms exceed skin distance
- ``batch_neighbor_list_needs_rebuild`` / ``batch_cell_list_needs_rebuild``:
  Batch variants producing per-system GPU-side rebuild flags
- Selective skip in batch neighbor list APIs using ``rebuild_flags``:
  only rebuild systems that actually need it, with no CPU-GPU sync

Rebuild detection is crucial for MD performance — neighbor lists are expensive to
compute but only need updating when atoms have moved significantly. The batch
variants enable per-system rebuild decisions with a single GPU kernel, while the
selective skip avoids unnecessary neighbor recomputation for stable systems.
"""

import numpy as np
import torch

from nvalchemiops.torch.neighbors.batch_cell_list import (
    batch_build_cell_list,
    batch_cell_list,
    batch_query_cell_list,
    estimate_batch_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.batch_naive import batch_naive_neighbor_list
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
    batch_cell_list_needs_rebuild,
    batch_neighbor_list_needs_rebuild,
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

print("\nSingle-system section completed.")

# %%
# Batch Rebuild Detection
# =======================
#
# When simulating many systems at once, the batch variants
# ``batch_neighbor_list_needs_rebuild`` and ``batch_cell_list_needs_rebuild``
# return a per-system boolean tensor entirely on the GPU — no CPU-GPU sync.
#
# Each flag independently reports whether that system needs rebuilding.

print("\n" + "=" * 70)
print("BATCH REBUILD DETECTION")
print("=" * 70)

# Set up a batch of systems with different atom counts
batch_sizes = [32, 48, 40]
batch_size = sum(batch_sizes)
num_systems_batch = len(batch_sizes)
batch_box_size = 5.0
batch_cutoff = 1.5
batch_skin = 0.4
batch_total_cutoff = batch_cutoff + batch_skin

# Create per-atom batch index and batch pointer
batch_idx = torch.repeat_interleave(
    torch.arange(num_systems_batch, dtype=torch.int32, device=device),
    torch.tensor(batch_sizes, dtype=torch.int32, device=device),
)
ptr_vals = [0] + [sum(batch_sizes[: i + 1]) for i in range(num_systems_batch)]
batch_ptr = torch.tensor(ptr_vals, dtype=torch.int32, device=device)

# Per-system cells and PBCs
batch_cell = (torch.eye(3, dtype=dtype, device=device) * batch_box_size).unsqueeze(0)
batch_cell = batch_cell.expand(num_systems_batch, -1, -1).contiguous()
batch_pbc = torch.zeros(num_systems_batch, 3, dtype=torch.bool, device=device)

# Random initial positions for each system
torch.manual_seed(1234)
batch_positions = torch.rand(batch_size, 3, dtype=dtype, device=device) * batch_box_size

print(f"\nBatch of {num_systems_batch} systems, {batch_sizes} atoms each")
print(f"  Cutoff: {batch_cutoff}, skin: {batch_skin}")

# Build initial batch neighbor lists
batch_max_neighbors = estimate_max_neighbors(batch_total_cutoff)
batch_nm, batch_nn = batch_naive_neighbor_list(
    positions=batch_positions,
    cutoff=batch_total_cutoff,
    batch_idx=batch_idx,
    batch_ptr=batch_ptr,
    max_neighbors=batch_max_neighbors,
)
print(f"\nInitial batch neighbor list built (max_neighbors={batch_max_neighbors})")
for s in range(num_systems_batch):
    sys_mask = batch_idx == s
    avg_nn = batch_nn[sys_mask].float().mean().item()
    print(f"  System {s}: avg {avg_nn:.1f} neighbors")

# %%
# Check rebuild flags after small and large displacements
# -------------------------------------------------------
#
# Move only atoms in system 1 beyond the skin distance threshold.
# Only system 1's flag should be True.

reference_batch_positions = batch_positions.clone()

# Simulate a step where system 1 atoms move significantly
current_batch_positions = batch_positions.clone()
sys1_start = batch_ptr[1].item()
sys1_end = batch_ptr[2].item()
# Move system 1 atoms by 2 × skin distance
current_batch_positions[sys1_start:sys1_end] += batch_skin * 2.0

rebuild_flags = batch_neighbor_list_needs_rebuild(
    reference_positions=reference_batch_positions,
    current_positions=current_batch_positions,
    batch_idx=batch_idx,
    skin_distance_threshold=batch_skin,
)

print(f"\nAfter moving system 1 atoms by {batch_skin * 2.0:.2f} Å:")
print(f"  rebuild_flags device: {rebuild_flags.device}  (stays on GPU, no CPU sync)")
for s in range(num_systems_batch):
    print(f"  System {s} needs rebuild: {rebuild_flags[s].item()}")

if rebuild_flags[0].item() or not rebuild_flags[1].item() or rebuild_flags[2].item():
    raise RuntimeError(
        "Unexpected rebuild flags: expected only system 1 to need rebuild"
    )

# %%
# GPU-Side Selective Skip in Batch Neighbor APIs
# ===============================================
#
# Now we use ``rebuild_flags`` directly in ``batch_naive_neighbor_list``.
# Only system 1 is recomputed; systems 0 and 2 skip the kernel entirely
# on the GPU — their neighbor data is preserved from the previous build.
#
# To make the neighbor-count change clearly visible we use three small
# hand-crafted systems:
#
# - **System 0 / 2** (stable): 4 atoms in a tight 0.4 Å grid → every pair is
#   within the 1.0 Å cutoff → each atom has 3 neighbors.
# - **System 1** (displaced): same tight cluster initially.  After the "MD
#   step" the atoms are spread to a 3.0 Å grid — all inter-atom distances
#   exceed the cutoff → every atom drops to 0 neighbors.
#
# This avoids any CPU-GPU sync and minimizes wasted GPU work.

print("\n" + "=" * 70)
print("GPU-SIDE SELECTIVE SKIP IN BATCH NEIGHBOR APIS")
print("=" * 70)

# --- Build controlled mini-systems -------------------------------------------
sk_cutoff = 1.0  # short cutoff so spacing > cutoff → 0 neighbors
sk_max_neighbors = 10
sk_n_atoms = 4  # atoms per system

# Tight cluster positions (spacing 0.4 < 1.0 → fully connected)
tight_offsets = torch.tensor(
    [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.4, 0.4, 0.0]],
    dtype=dtype,
    device=device,
)

# Sparse cluster (spacing 3.0 > 1.0 → no neighbors)
sparse_offsets = torch.tensor(
    [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [3.0, 3.0, 0.0]],
    dtype=dtype,
    device=device,
)

sk_positions_initial = torch.cat(
    [tight_offsets, tight_offsets + 10.0, tight_offsets + 20.0], dim=0
)
# After "MD step": system 1 atoms spread apart, systems 0 and 2 unchanged
sk_positions_after = torch.cat(
    [tight_offsets, sparse_offsets + 10.0, tight_offsets + 20.0], dim=0
)

sk_n_total = sk_n_atoms * 3
sk_batch_idx = torch.repeat_interleave(
    torch.arange(3, dtype=torch.int32, device=device),
    sk_n_atoms,
)
sk_batch_ptr = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device=device)

# Build initial neighbor list (all three systems are tight clusters)
sk_nm = torch.full(
    (sk_n_total, sk_max_neighbors), sk_n_total, dtype=torch.int32, device=device
)
sk_nn = torch.zeros(sk_n_total, dtype=torch.int32, device=device)
batch_naive_neighbor_list(
    positions=sk_positions_initial,
    cutoff=sk_cutoff,
    batch_idx=sk_batch_idx,
    batch_ptr=sk_batch_ptr,
    max_neighbors=sk_max_neighbors,
    neighbor_matrix=sk_nm,
    num_neighbors=sk_nn,
)

print("\nInitial state (all systems: tight 0.4 Å cluster, cutoff=1.0 Å):")
for s in range(3):
    mask = sk_batch_idx == s
    print(f"  System {s}: avg {sk_nn[mask].float().mean().item():.1f} neighbors/atom")

# Detect which systems need rebuilding (only system 1 moved)
sk_ref_positions = sk_positions_initial.clone()
sk_rebuild_flags = batch_neighbor_list_needs_rebuild(
    reference_positions=sk_ref_positions,
    current_positions=sk_positions_after,
    batch_idx=sk_batch_idx,
    skin_distance_threshold=0.1,  # tight threshold: any move > 0.1 triggers flag
)

print("\nrebuild_flags after spreading system 1 atoms to 3.0 Å spacing:")
for s in range(3):
    print(f"  System {s}: {sk_rebuild_flags[s].item()}")

# Selective rebuild: only system 1 is recomputed on the GPU
batch_naive_neighbor_list(
    positions=sk_positions_after,
    cutoff=sk_cutoff,
    batch_idx=sk_batch_idx,
    batch_ptr=sk_batch_ptr,
    max_neighbors=sk_max_neighbors,
    neighbor_matrix=sk_nm,  # in-place: non-rebuilt systems preserved
    num_neighbors=sk_nn,
    rebuild_flags=sk_rebuild_flags,
)

print("\nAfter selective rebuild (GPU kernel skipped for systems 0 and 2):")
for s in range(3):
    mask = sk_batch_idx == s
    rebuilt = sk_rebuild_flags[s].item()
    print(
        f"  System {s}: avg {sk_nn[mask].float().mean().item():.1f} neighbors/atom"
        f"  (rebuilt={rebuilt})"
    )

# System 1 should now show 0 neighbors (atoms spread beyond cutoff)
if sk_nn[sk_batch_idx == 1].sum().item() != 0:
    raise RuntimeError("System 1 neighbors should be 0 after spreading atoms apart")
# Systems 0 and 2 should still have 3 neighbors/atom (fully connected cluster)
for s in (0, 2):
    if sk_nn[sk_batch_idx == s].float().mean().item() != float(sk_n_atoms - 1):
        raise RuntimeError(f"System {s} neighbor counts should be unchanged")

print(
    "\nVerified:"
    "\n  System 1 rebuilt → neighbor count dropped from 3 to 0 (atoms spread beyond cutoff)"
    "\n  Systems 0 and 2 skipped → neighbor count unchanged at 3"
)

# %%
# Batch Cell List with Selective Skip
# ====================================
#
# The same pattern works with the O(N) cell list algorithm.
# ``batch_cell_list_needs_rebuild`` detects when atoms cross cell boundaries,
# while ``batch_neighbor_list_needs_rebuild`` uses skin distance.
# Either method produces ``rebuild_flags`` that can be fed directly into
# ``batch_cell_list`` / ``batch_query_cell_list`` to skip non-rebuilt systems.
#
# We reuse the same three mini-systems from above so the neighbor-count
# change is equally clear.

print("\n" + "=" * 70)
print("BATCH CELL LIST WITH SELECTIVE SKIP")
print("=" * 70)

# Use a large periodic box so cell list can be built
cl_box = 30.0
cl_cell = (torch.eye(3, dtype=dtype, device=device) * cl_box).unsqueeze(0)
cl_cell = cl_cell.expand(3, -1, -1).contiguous()
cl_pbc = torch.ones(3, 3, dtype=torch.bool, device=device)

# Build initial cell list and neighbor matrix (all systems: tight cluster)
cl_nm, cl_nn, cl_shifts = batch_cell_list(
    positions=sk_positions_initial,
    cutoff=sk_cutoff,
    cell=cl_cell,
    pbc=cl_pbc,
    batch_idx=sk_batch_idx,
    max_neighbors=sk_max_neighbors,
)

print("\nInitial state (all systems: tight 0.4 Å cluster, cutoff=1.0 Å):")
for s in range(3):
    mask = sk_batch_idx == s
    print(f"  System {s}: avg {cl_nn[mask].float().mean().item():.1f} neighbors/atom")

# Estimate and allocate cell list data structures
max_total_cells_cl, neighbor_search_radius_cl = estimate_batch_cell_list_sizes(
    cl_cell, cl_pbc, cutoff=sk_cutoff
)
cl_cache = allocate_cell_list(
    sk_n_total, max_total_cells_cl, neighbor_search_radius_cl, device
)

# Build cell list at reference positions and save atom-to-cell mapping
batch_build_cell_list(
    sk_positions_initial,
    sk_cutoff,
    cl_cell,
    cl_pbc,
    sk_batch_idx,
    *cl_cache,
)
ref_cl_atom_to_cell_mapping = cl_cache[3].clone()

# Detect rebuild by cell boundary crossing (system 1 atoms move by 3 Å)
cl_rebuild_flags = batch_cell_list_needs_rebuild(
    current_positions=sk_positions_after,
    atom_to_cell_mapping=ref_cl_atom_to_cell_mapping,
    batch_idx=sk_batch_idx,
    cells_per_dimension=cl_cache[0],
    cell=cl_cell,
    pbc=cl_pbc,
)

print("\nbatch_cell_list_needs_rebuild flags (system 1 atoms moved 3.0 Å):")
for s in range(3):
    print(f"  System {s}: {cl_rebuild_flags[s].item()}")

# Rebuild the full cell list with new positions before selective query
batch_build_cell_list(
    sk_positions_after,
    sk_cutoff,
    cl_cell,
    cl_pbc,
    sk_batch_idx,
    *cl_cache,
)

# Selective query: only recompute neighbors for flagged systems
cl_nm_sel = cl_nm.clone()
cl_nn_sel = cl_nn.clone()
cl_shifts_sel = cl_shifts.clone()

batch_query_cell_list(
    positions=sk_positions_after,
    cell=cl_cell,
    pbc=cl_pbc,
    cutoff=sk_cutoff,
    batch_idx=sk_batch_idx,
    cells_per_dimension=cl_cache[0],
    neighbor_search_radius=cl_cache[1],
    atom_periodic_shifts=cl_cache[2],
    atom_to_cell_mapping=cl_cache[3],
    atoms_per_cell_count=cl_cache[4],
    cell_atom_start_indices=cl_cache[5],
    cell_atom_list=cl_cache[6],
    neighbor_matrix=cl_nm_sel,
    neighbor_matrix_shifts=cl_shifts_sel,
    num_neighbors=cl_nn_sel,
    half_fill=False,
    rebuild_flags=cl_rebuild_flags,
)

print("\nAfter selective batch_query_cell_list:")
for s in range(3):
    mask = sk_batch_idx == s
    rebuilt = cl_rebuild_flags[s].item()
    print(
        f"  System {s}: avg {cl_nn_sel[mask].float().mean().item():.1f} neighbors/atom"
        f"  (rebuilt={rebuilt})"
    )

# System 1 → 0 neighbors; systems 0 and 2 unchanged at 3
if cl_nn_sel[sk_batch_idx == 1].sum().item() != 0:
    raise RuntimeError(
        "System 1 (cell list) neighbors should be 0 after spreading atoms apart"
    )
for s in (0, 2):
    if cl_nn_sel[sk_batch_idx == s].float().mean().item() != float(sk_n_atoms - 1):
        raise RuntimeError(
            f"System {s} (cell list) neighbor counts should be unchanged"
        )

print(
    "\nVerified:"
    "\n  System 1 rebuilt → neighbor count dropped from 3 to 0 (atoms spread beyond cutoff)"
    "\n  Systems 0 and 2 skipped → neighbor count unchanged at 3"
)

print("\nExample completed successfully!")
print(
    "\nKey takeaways:"
    "\n  - batch_*_needs_rebuild returns GPU-resident per-system bool tensor (no CPU sync)"
    "\n  - Pass rebuild_flags to batch_naive_neighbor_list / batch_cell_list / batch_query_cell_list"
    "\n  - Non-rebuilt systems return immediately from GPU kernel — zero extra GPU work"
    "\n  - Pre-allocate neighbor_matrix and num_neighbors and pass them in to enable in-place update"
)
