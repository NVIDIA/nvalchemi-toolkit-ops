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
Targeted Lennard-Jones Pair Outputs
===================================

This example demonstrates three neighbor-list features used together:

1. ``target_indices`` for compact source rows
2. Per-neighbor vectors and distances
3. An inline Warp ``pair_fn`` that computes Lennard-Jones energies and forces

The system is a 4x4x4 FCC Argon box using the same parameters as the dynamics
examples: epsilon = 0.0104 eV, sigma = 3.40 Å, lattice constant = 5.26 Å, and
cutoff = 2.5 * sigma.
"""

import torch
import warp as wp

from nvalchemiops.torch.neighbors import estimate_neighbor_list_costs, neighbor_list
from nvalchemiops.torch.neighbors.neighbor_utils import estimate_max_neighbors

# %%
# System setup
# ============

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

EPSILON_AR = 0.0104
SIGMA_AR = 3.40
LATTICE_A_AR = 5.26
CUTOFF = 2.5 * SIGMA_AR
NUM_UNIT_CELLS = 4

print("=" * 70)
print("TARGETED LENNARD-JONES PAIR OUTPUTS")
print("=" * 70)
print(f"Using device: {device}")
print(f"Using dtype: {dtype}")


def create_fcc_argon_box(
    num_unit_cells: int,
    lattice_constant: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create an FCC Argon box with periodic boundary conditions."""
    basis = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        dtype=dtype,
        device=device,
    )
    unit_axis = torch.arange(num_unit_cells, dtype=dtype, device=device)
    grid = torch.stack(
        torch.meshgrid(unit_axis, unit_axis, unit_axis, indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    positions = (grid[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    positions = positions * lattice_constant

    box_length = float(num_unit_cells) * lattice_constant
    cell = (torch.eye(3, dtype=dtype, device=device) * box_length).unsqueeze(0)
    pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
    return positions, cell, pbc


positions, cell, pbc = create_fcc_argon_box(
    NUM_UNIT_CELLS,
    LATTICE_A_AR,
    dtype=dtype,
    device=device,
)
num_atoms = positions.shape[0]
target_indices = torch.arange(0, num_atoms, 4, dtype=torch.int32, device=device)
batch_ptr = torch.tensor([0, num_atoms], dtype=torch.int32, device=device)

argon_density = 4.0 / (LATTICE_A_AR**3)
max_neighbors = estimate_max_neighbors(
    CUTOFF,
    atomic_density=argon_density,
    safety_factor=1.5,
)

print(f"\nFCC Argon box: {num_atoms} atoms")
print(f"Box length: {cell[0, 0, 0].item():.2f} Å")
print(f"LJ parameters: epsilon={EPSILON_AR:.4f} eV, sigma={SIGMA_AR:.2f} Å")
print(f"Cutoff: {CUTOFF:.2f} Å")
print(f"Targeted source rows: {target_indices.numel()} of {num_atoms}")
print(f"Estimated max_neighbors: {max_neighbors}")


# %%
# Dispatch estimate
# =================


def _dispatch_report(
    label: str,
    batch_ptr: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    **kwargs,
) -> str:
    """Print sorted Torch neighbor-list strategy costs and return the cheapest."""
    report = estimate_neighbor_list_costs(batch_ptr, cell, pbc, cutoff, **kwargs)
    print(f"\n{label}")
    for strategy, cost in report:
        print(f"  {strategy:24s} estimated cost (arbitrary units): {cost:.3g}")
    return report[0][0]


selected_strategy = _dispatch_report(
    "Torch dispatch estimate for targeted LJ outputs:",
    batch_ptr,
    cell,
    pbc,
    CUTOFF,
    target_indices=target_indices,
    return_vectors=True,
    return_distances=True,
    use_pair_fn=True,
    positions_dtype=positions.dtype,
)
print(f"Selected strategy: {selected_strategy}")


# %%
# Lennard-Jones pair function
# ===========================


@wp.func
def lj_pair_fn(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    """Compute Lennard-Jones pair energy and force from per-atom parameters."""
    epsilon = wp.sqrt(pair_params[i, 0] * pair_params[j, 0])
    sigma = 0.5 * (pair_params[i, 1] + pair_params[j, 1])
    sr = sigma / distance
    sr2 = sr * sr
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    energy = 4.0 * epsilon * (sr12 - sr6)
    force = (24.0 * epsilon * (sr6 - 2.0 * sr12) / (distance * distance)) * r_ij
    return energy, force


pair_params = torch.empty((num_atoms, 2), dtype=dtype, device=device)
pair_params[:, 0] = EPSILON_AR
pair_params[:, 1] = SIGMA_AR

num_targets = target_indices.numel()
fill_value = num_atoms
neighbor_matrix = torch.full(
    (num_targets, max_neighbors),
    fill_value,
    dtype=torch.int32,
    device=device,
)
neighbor_shifts = torch.zeros(
    (num_targets, max_neighbors, 3),
    dtype=torch.int32,
    device=device,
)
neighbor_counts = torch.zeros(num_targets, dtype=torch.int32, device=device)
neighbor_vectors = torch.zeros(
    (num_targets, max_neighbors, 3),
    dtype=dtype,
    device=device,
)
neighbor_distances = torch.zeros(
    (num_targets, max_neighbors),
    dtype=dtype,
    device=device,
)
pair_energies = torch.zeros(
    (num_targets, max_neighbors),
    dtype=dtype,
    device=device,
)
pair_forces = torch.zeros(
    (num_targets, max_neighbors, 3),
    dtype=dtype,
    device=device,
)

(
    neighbor_matrix,
    neighbor_counts,
    neighbor_shifts,
    neighbor_distances,
    neighbor_vectors,
    pair_energies,
    pair_forces,
) = neighbor_list(
    positions,
    CUTOFF,
    cell=cell,
    pbc=pbc,
    method=selected_strategy,
    target_indices=target_indices,
    fill_value=fill_value,
    max_neighbors=max_neighbors,
    neighbor_matrix=neighbor_matrix,
    neighbor_matrix_shifts=neighbor_shifts,
    num_neighbors=neighbor_counts,
    return_vectors=True,
    return_distances=True,
    pair_fn=lj_pair_fn,
    pair_params=pair_params,
    neighbor_vectors=neighbor_vectors,
    neighbor_distances=neighbor_distances,
    pair_energies=pair_energies,
    pair_forces=pair_forces,
)


# %%
# Validate pair outputs
# =====================

neighbor_slots = torch.arange(max_neighbors, device=device)
valid_slots = neighbor_slots[None, :] < neighbor_counts[:, None]

source_atoms = target_indices[:, None].expand_as(neighbor_matrix)[valid_slots].long()
target_atoms = neighbor_matrix[valid_slots].long()
active_distances = neighbor_distances[valid_slots]
active_vectors = neighbor_vectors[valid_slots]

source_params = pair_params[source_atoms]
target_params = pair_params[target_atoms]
epsilon = torch.sqrt(source_params[:, 0] * target_params[:, 0])
sigma = 0.5 * (source_params[:, 1] + target_params[:, 1])
sr = sigma / active_distances
sr2 = sr * sr
sr6 = sr2 * sr2 * sr2
sr12 = sr6 * sr6
reference_energies = 4.0 * epsilon * (sr12 - sr6)
reference_forces = (
    24.0
    * epsilon[:, None]
    * (sr6 - 2.0 * sr12)[:, None]
    / active_distances[:, None].pow(2)
    * active_vectors
)

torch.testing.assert_close(
    pair_energies[valid_slots],
    reference_energies,
    rtol=5.0e-4,
    atol=5.0e-6,
)
torch.testing.assert_close(
    pair_forces[valid_slots],
    reference_forces,
    rtol=5.0e-4,
    atol=5.0e-6,
)

print("\nTargeted pair-output statistics:")
print(f"  Compact output rows: {num_targets}")
print(f"  Active targeted pairs: {int(valid_slots.sum().item())}")
print(f"  Average targeted neighbors: {neighbor_counts.float().mean().item():.2f}")
print(
    f"  Targeted directed LJ energy sum: {pair_energies[valid_slots].sum().item():.6f} eV"
)
print("  LJ pair energies and forces match the Torch reference")

print("\nFirst targeted LJ pairs:")
sample_count = min(5, source_atoms.numel())
for pair_index in range(sample_count):
    source = int(source_atoms[pair_index].item())
    target = int(target_atoms[pair_index].item())
    distance = float(active_distances[pair_index].item())
    energy = float(pair_energies[valid_slots][pair_index].item())
    force_norm = float(pair_forces[valid_slots][pair_index].norm().item())
    print(
        f"  Atom {source} -> {target}: "
        f"distance={distance:.4f} Å, energy={energy:.6f} eV, "
        f"|force|={force_norm:.6f} eV/Å"
    )

print("\nExample completed successfully!")
