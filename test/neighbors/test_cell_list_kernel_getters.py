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

"""Smoke tests for the cell-list neighbor-matrix kernel getter."""

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.cell_list import (
    PAIR_CENTRIC_MAX_LINEAR_LAUNCH,
    build_cell_list,
    compute_batch_pair_centric_n_outer,
    get_cell_list_cells_per_system_kernel,
    get_query_cell_list_kernel,
    is_pair_centric_launch_safe,
    pair_centric_launch_size,
    query_cell_list,
)


@wp.func
def _factory_pair_fn_clf(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    """Tiny pair function returning ``(p_i + p_j + distance, -r_ij)``."""
    energy = pair_params[i, 0] + pair_params[j, 0] + distance
    force = -r_ij
    return energy, force


@wp.func
def _factory_pair_fn_alt_clf(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    """Alternate pair function used to exercise the cache-key path."""
    energy = pair_params[i, 0] - pair_params[j, 0] + distance
    force = r_ij
    return energy, force


@wp.func
def _lj_pair_fn_clf(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    """Lennard-Jones pair function with Lorentz-Berthelot mixing.

    Reads ``(epsilon, sigma)`` from ``pair_params[i]`` and ``pair_params[j]``.
    Returns energy ``U_ij`` and the Cartesian force on atom ``i`` due to
    atom ``j``.
    """
    eps_i = pair_params[i, 0]
    sigma_i = pair_params[i, 1]
    eps_j = pair_params[j, 0]
    sigma_j = pair_params[j, 1]
    eps = wp.sqrt(eps_i * eps_j)
    sigma = 0.5 * (sigma_i + sigma_j)
    inv_r = 1.0 / distance
    sr = sigma * inv_r
    s6 = sr * sr * sr * sr * sr * sr
    s12 = s6 * s6
    energy = 4.0 * eps * (s12 - s6)
    force = -(24.0 * eps * inv_r * inv_r * (2.0 * s12 - s6)) * r_ij
    return energy, force


def _lj_reference(r_ij_np: np.ndarray, eps_i, sigma_i, eps_j, sigma_j):
    """NumPy reference for the LJ pair function (matches ``_lj_pair_fn_clf``)."""
    distance = float(np.linalg.norm(r_ij_np))
    eps = float(np.sqrt(eps_i * eps_j))
    sigma = 0.5 * (sigma_i + sigma_j)
    inv_r = 1.0 / distance
    sr = sigma * inv_r
    s6 = sr**6
    s12 = s6**2
    energy = 4.0 * eps * (s12 - s6)
    force = -(24.0 * eps * inv_r * inv_r * (2.0 * s12 - s6)) * r_ij_np
    return energy, force


def _skip_missing_cuda(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test parameter")


def test_cells_per_system_kernel_getter_is_public_and_stable():
    """The JAX binding uses a public getter instead of a private kernel import."""
    kernel1 = get_cell_list_cells_per_system_kernel()
    kernel2 = get_cell_list_cells_per_system_kernel()

    assert kernel1 is kernel2


def test_cell_list_query_kernel_uses_pair_fn_object_cache_key():
    """The cache key uses the Warp function object, not ``id(pair_fn)``."""
    kernel1 = get_query_cell_list_kernel(
        wp.float32,
        pair_fn=_factory_pair_fn_clf,
    )
    kernel2 = get_query_cell_list_kernel(
        wp.float32,
        pair_fn=_factory_pair_fn_clf,
    )
    kernel3 = get_query_cell_list_kernel(
        wp.float32,
        pair_fn=_factory_pair_fn_alt_clf,
    )
    assert kernel1 is kernel2
    assert kernel1 is not kernel3


def test_pair_centric_launch_guard_rejects_large_batched_shape():
    """The guard rejects the OMat24 :60 cutoff-15 pair-centric launch shape."""
    n_outer = compute_batch_pair_centric_n_outer((23, 25, 23), half_fill=False)
    launch_size = pair_centric_launch_size(3840, n_outer, 64)

    assert launch_size == 27_687_075_840
    assert launch_size > PAIR_CENTRIC_MAX_LINEAR_LAUNCH
    assert not is_pair_centric_launch_safe(3840, n_outer, 64)


def test_pair_centric_launch_guard_accepts_small_batched_shape():
    """The guard accepts a small pair-centric launch shape."""
    n_outer = compute_batch_pair_centric_n_outer((1, 1, 1), half_fill=False)

    assert pair_centric_launch_size(64, n_outer, 64) == 110_592
    assert is_pair_centric_launch_safe(64, n_outer, 64)


def _build_single_smoke_state(device, positions_np, box, cutoff, max_neighbors):
    """Build the single-system cell-list state used by the smoke tests.

    Allocates every output / scratch buffer (with shapes the kernel
    expects), runs :func:`build_cell_list`, and returns the dict of
    ``wp.array`` handles plus the underlying torch tensors so callers
    can inspect the outputs.
    """
    n_atoms = positions_np.shape[0]
    positions_t = torch.from_numpy(positions_np).to(device)
    cell_t = (torch.eye(3, dtype=torch.float32, device=device) * box).unsqueeze(0)
    pbc_t = torch.zeros(3, dtype=torch.bool, device=device)
    cells_per_dimension = torch.zeros(3, dtype=torch.int32, device=device)
    atom_periodic_shifts = torch.zeros((n_atoms, 3), dtype=torch.int32, device=device)
    atom_to_cell_mapping = torch.zeros((n_atoms, 3), dtype=torch.int32, device=device)
    max_total_cells = 64
    atoms_per_cell_count = torch.zeros(
        max_total_cells, dtype=torch.int32, device=device
    )
    cell_atom_start_indices = torch.zeros(
        max_total_cells, dtype=torch.int32, device=device
    )
    cell_atom_list = torch.zeros(n_atoms, dtype=torch.int32, device=device)
    neighbor_search_radius = torch.zeros(3, dtype=torch.int32, device=device)
    sorted_positions = torch.zeros_like(positions_t)
    sorted_shifts = torch.zeros_like(atom_periodic_shifts)
    neighbor_matrix = torch.full(
        (n_atoms, max_neighbors), -1, dtype=torch.int32, device=device
    )
    neighbor_matrix_shifts = torch.zeros(
        (n_atoms, max_neighbors, 3), dtype=torch.int32, device=device
    )
    num_neighbors = torch.zeros(n_atoms, dtype=torch.int32, device=device)
    rebuild_flags = torch.ones(1, dtype=torch.bool, device=device)

    build_cell_list(
        wp.from_torch(positions_t, dtype=wp.vec3f),
        wp.from_torch(cell_t, dtype=wp.mat33f),
        wp.from_torch(pbc_t, dtype=wp.bool),
        cutoff,
        wp.from_torch(cells_per_dimension, dtype=wp.int32),
        wp.from_torch(atom_periodic_shifts, dtype=wp.vec3i),
        wp.from_torch(atom_to_cell_mapping, dtype=wp.vec3i),
        wp.from_torch(atoms_per_cell_count, dtype=wp.int32),
        wp.from_torch(cell_atom_start_indices, dtype=wp.int32),
        wp.from_torch(cell_atom_list, dtype=wp.int32),
        wp.float32,
        device,
    )
    cpd = cells_per_dimension.cpu().tolist()
    nsr = [1 if cpd[d] > 1 else 0 for d in range(3)]
    neighbor_search_radius.copy_(torch.tensor(nsr, dtype=torch.int32, device=device))
    return {
        "positions_t": positions_t,
        "cell_t": cell_t,
        "pbc_t": pbc_t,
        "cells_per_dimension": cells_per_dimension,
        "atom_periodic_shifts": atom_periodic_shifts,
        "atom_to_cell_mapping": atom_to_cell_mapping,
        "atoms_per_cell_count": atoms_per_cell_count,
        "cell_atom_start_indices": cell_atom_start_indices,
        "cell_atom_list": cell_atom_list,
        "neighbor_search_radius": neighbor_search_radius,
        "sorted_positions": sorted_positions,
        "sorted_shifts": sorted_shifts,
        "neighbor_matrix": neighbor_matrix,
        "neighbor_matrix_shifts": neighbor_matrix_shifts,
        "num_neighbors": num_neighbors,
        "rebuild_flags": rebuild_flags,
    }


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cell_list_pair_fn_outputs(device):
    """Optional cell-list path runs and writes non-zero pair energies."""
    _skip_missing_cuda(device)

    # Three atoms on a line in a generous box; cutoff covers atom-1 only
    # from atom-0's perspective.  PBC=False so the shift collapses to 0.
    positions_np = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    box = 4.0
    cutoff = 0.75
    n_atoms = positions_np.shape[0]
    max_neighbors = 4

    positions_t = torch.from_numpy(positions_np).to(device)
    cell_t = (torch.eye(3, dtype=torch.float32, device=device) * box).unsqueeze(0)
    pbc_t = torch.zeros(3, dtype=torch.bool, device=device)
    params_t = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32, device=device)

    # Cell-list build scratch.
    cells_per_dimension = torch.zeros(3, dtype=torch.int32, device=device)
    atom_periodic_shifts = torch.zeros((n_atoms, 3), dtype=torch.int32, device=device)
    atom_to_cell_mapping = torch.zeros((n_atoms, 3), dtype=torch.int32, device=device)
    max_total_cells = 64
    atoms_per_cell_count = torch.zeros(
        max_total_cells, dtype=torch.int32, device=device
    )
    cell_atom_start_indices = torch.zeros(
        max_total_cells, dtype=torch.int32, device=device
    )
    cell_atom_list = torch.zeros(n_atoms, dtype=torch.int32, device=device)
    neighbor_search_radius = torch.zeros(3, dtype=torch.int32, device=device)

    # Query outputs / scratch.
    sorted_positions = torch.zeros_like(positions_t)
    sorted_shifts = torch.zeros_like(atom_periodic_shifts)
    neighbor_matrix = torch.full(
        (n_atoms, max_neighbors), -1, dtype=torch.int32, device=device
    )
    neighbor_matrix_shifts = torch.zeros(
        (n_atoms, max_neighbors, 3), dtype=torch.int32, device=device
    )
    num_neighbors = torch.zeros(n_atoms, dtype=torch.int32, device=device)
    rebuild_flags = torch.ones(1, dtype=torch.bool, device=device)

    # Optional per-neighbor outputs.
    pair_energies = torch.zeros(
        (n_atoms, max_neighbors), dtype=torch.float32, device=device
    )
    pair_forces = torch.zeros(
        (n_atoms, max_neighbors, 3), dtype=torch.float32, device=device
    )
    neighbor_vectors = torch.zeros(
        (n_atoms, max_neighbors, 3), dtype=torch.float32, device=device
    )
    neighbor_distances = torch.zeros(
        (n_atoms, max_neighbors), dtype=torch.float32, device=device
    )

    # First-pass: estimate cell-list sizes ahead of build.  ``build_cell_list``
    # below initialises cells_per_dimension itself so we just need a generous
    # ``atoms_per_cell_count`` buffer.
    positions_wp = wp.from_torch(positions_t, dtype=wp.vec3f)
    cell_wp = wp.from_torch(cell_t, dtype=wp.mat33f)
    pbc_wp = wp.from_torch(pbc_t, dtype=wp.bool)

    build_cell_list(
        positions_wp,
        cell_wp,
        pbc_wp,
        cutoff,
        wp.from_torch(cells_per_dimension, dtype=wp.int32),
        wp.from_torch(atom_periodic_shifts, dtype=wp.vec3i),
        wp.from_torch(atom_to_cell_mapping, dtype=wp.vec3i),
        wp.from_torch(atoms_per_cell_count, dtype=wp.int32),
        wp.from_torch(cell_atom_start_indices, dtype=wp.int32),
        wp.from_torch(cell_atom_list, dtype=wp.int32),
        wp.float32,
        device,
    )

    # neighbor_search_radius needs to be computed (matching the higher-level
    # wrapper).  For this smoke test we cheat: with cutoff < cell-side the
    # search radius collapses to 1 along each axis where >1 cell exists.
    cpd = cells_per_dimension.cpu().tolist()
    nsr = [1 if cpd[d] > 1 else 0 for d in range(3)]
    neighbor_search_radius.copy_(torch.tensor(nsr, dtype=torch.int32, device=device))

    query_cell_list(
        positions_wp,
        cell_wp,
        pbc_wp,
        cutoff,
        wp.from_torch(cells_per_dimension, dtype=wp.int32),
        wp.from_torch(neighbor_search_radius, dtype=wp.int32),
        wp.from_torch(atom_periodic_shifts, dtype=wp.vec3i),
        wp.from_torch(atom_to_cell_mapping, dtype=wp.vec3i),
        wp.from_torch(atoms_per_cell_count, dtype=wp.int32),
        wp.from_torch(cell_atom_start_indices, dtype=wp.int32),
        wp.from_torch(cell_atom_list, dtype=wp.int32),
        wp.from_torch(neighbor_matrix, dtype=wp.int32),
        wp.from_torch(neighbor_matrix_shifts, dtype=wp.vec3i),
        wp.from_torch(num_neighbors, dtype=wp.int32),
        wp.float32,
        device,
        half_fill=False,
        strategy="atom_centric",
        pair_fn=_factory_pair_fn_clf,
        pair_params=wp.from_torch(params_t, dtype=wp.float32),
        pair_energies=wp.from_torch(pair_energies, dtype=wp.float32),
        pair_forces=wp.from_torch(pair_forces, dtype=wp.vec3f),
        return_vectors=True,
        neighbor_vectors=wp.from_torch(neighbor_vectors, dtype=wp.vec3f),
        return_distances=True,
        neighbor_distances=wp.from_torch(neighbor_distances, dtype=wp.float32),
        sorted_positions=wp.from_torch(sorted_positions, dtype=wp.vec3f),
        sorted_atom_periodic_shifts=wp.from_torch(sorted_shifts, dtype=wp.vec3i),
        rebuild_flags=wp.from_torch(rebuild_flags, dtype=wp.bool),
    )

    nn = num_neighbors.cpu().tolist()
    assert nn[0] >= 1
    nm0 = neighbor_matrix[0, : nn[0]].cpu().tolist()
    assert 1 in nm0
    slot = nm0.index(1)
    # r_ij = positions[1] - positions[0] = (0.5, 0, 0); distance = 0.5.
    expected_energy = 1.0 + 2.0 + 0.5
    assert pair_energies[0, slot].item() == pytest.approx(expected_energy, rel=1e-5)
    # Force is -r_ij = -(0.5, 0, 0).
    assert pair_forces[0, slot].cpu().tolist() == pytest.approx([-0.5, 0.0, 0.0])
    # At least one non-zero pair energy was written.
    assert pair_energies.abs().max().item() > 0.0


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cell_list_partial_target_indices(device):
    """``target_indices`` restricts the central-atom iteration."""
    _skip_missing_cuda(device)

    positions_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    box = 4.0
    cutoff = 0.6
    max_neighbors = 4

    state = _build_single_smoke_state(device, positions_np, box, cutoff, max_neighbors)
    # Only atom 0 should be probed; compact row 0 corresponds to atom id 0.
    target_indices = torch.tensor([0], dtype=torch.int32, device=device)
    neighbor_matrix = torch.full(
        (1, max_neighbors), -1, dtype=torch.int32, device=device
    )
    neighbor_matrix_shifts = torch.zeros(
        (1, max_neighbors, 3), dtype=torch.int32, device=device
    )
    num_neighbors = torch.zeros(1, dtype=torch.int32, device=device)

    query_cell_list(
        wp.from_torch(state["positions_t"], dtype=wp.vec3f),
        wp.from_torch(state["cell_t"], dtype=wp.mat33f),
        wp.from_torch(state["pbc_t"], dtype=wp.bool),
        cutoff,
        wp.from_torch(state["cells_per_dimension"], dtype=wp.int32),
        wp.from_torch(state["neighbor_search_radius"], dtype=wp.int32),
        wp.from_torch(state["atom_periodic_shifts"], dtype=wp.vec3i),
        wp.from_torch(state["atom_to_cell_mapping"], dtype=wp.vec3i),
        wp.from_torch(state["atoms_per_cell_count"], dtype=wp.int32),
        wp.from_torch(state["cell_atom_start_indices"], dtype=wp.int32),
        wp.from_torch(state["cell_atom_list"], dtype=wp.int32),
        wp.from_torch(neighbor_matrix, dtype=wp.int32),
        wp.from_torch(neighbor_matrix_shifts, dtype=wp.vec3i),
        wp.from_torch(num_neighbors, dtype=wp.int32),
        wp.float32,
        device,
        half_fill=False,
        strategy="atom_centric",
        target_indices=wp.from_torch(target_indices, dtype=wp.int32),
        sorted_positions=wp.from_torch(state["sorted_positions"], dtype=wp.vec3f),
        sorted_atom_periodic_shifts=wp.from_torch(
            state["sorted_shifts"], dtype=wp.vec3i
        ),
        rebuild_flags=wp.from_torch(state["rebuild_flags"], dtype=wp.bool),
    )

    nn = num_neighbors.cpu().tolist()
    assert nn == [2]
    found = set(neighbor_matrix[0, : nn[0]].cpu().tolist())
    assert found == {1, 2}


@pytest.mark.parametrize("device", ["cuda:0"])
def test_pair_centric_supports_pair_fn(device):
    """``strategy='pair_centric'`` writes pair-function outputs directly."""
    _skip_missing_cuda(device)

    positions_np = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    box = 4.0
    cutoff = 0.75
    max_neighbors = 4
    state = _build_single_smoke_state(device, positions_np, box, cutoff, max_neighbors)
    params_t = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32, device=device)
    n_atoms = positions_np.shape[0]
    pair_energies = torch.zeros(
        (n_atoms, max_neighbors), dtype=torch.float32, device=device
    )
    pair_forces = torch.zeros(
        (n_atoms, max_neighbors, 3), dtype=torch.float32, device=device
    )
    neighbor_vectors = torch.zeros(
        (n_atoms, max_neighbors, 3), dtype=torch.float32, device=device
    )
    neighbor_distances = torch.zeros(
        (n_atoms, max_neighbors), dtype=torch.float32, device=device
    )

    query_cell_list(
        wp.from_torch(state["positions_t"], dtype=wp.vec3f),
        wp.from_torch(state["cell_t"], dtype=wp.mat33f),
        wp.from_torch(state["pbc_t"], dtype=wp.bool),
        cutoff,
        wp.from_torch(state["cells_per_dimension"], dtype=wp.int32),
        wp.from_torch(state["neighbor_search_radius"], dtype=wp.int32),
        wp.from_torch(state["atom_periodic_shifts"], dtype=wp.vec3i),
        wp.from_torch(state["atom_to_cell_mapping"], dtype=wp.vec3i),
        wp.from_torch(state["atoms_per_cell_count"], dtype=wp.int32),
        wp.from_torch(state["cell_atom_start_indices"], dtype=wp.int32),
        wp.from_torch(state["cell_atom_list"], dtype=wp.int32),
        wp.from_torch(state["neighbor_matrix"], dtype=wp.int32),
        wp.from_torch(state["neighbor_matrix_shifts"], dtype=wp.vec3i),
        wp.from_torch(state["num_neighbors"], dtype=wp.int32),
        wp.float32,
        device,
        half_fill=False,
        strategy="pair_centric",
        n_outer=compute_batch_pair_centric_n_outer(
            tuple(state["neighbor_search_radius"].cpu().tolist()),
            half_fill=False,
        ),
        pair_fn=_factory_pair_fn_clf,
        pair_params=wp.from_torch(params_t, dtype=wp.float32),
        pair_energies=wp.from_torch(pair_energies, dtype=wp.float32),
        pair_forces=wp.from_torch(pair_forces, dtype=wp.vec3f),
        return_vectors=True,
        neighbor_vectors=wp.from_torch(neighbor_vectors, dtype=wp.vec3f),
        return_distances=True,
        neighbor_distances=wp.from_torch(neighbor_distances, dtype=wp.float32),
        sorted_positions=wp.from_torch(state["sorted_positions"], dtype=wp.vec3f),
        sorted_atom_periodic_shifts=wp.from_torch(
            state["sorted_shifts"], dtype=wp.vec3i
        ),
        rebuild_flags=wp.from_torch(state["rebuild_flags"], dtype=wp.bool),
    )
    assert pair_energies.abs().max().item() > 0.0
    assert neighbor_vectors.abs().max().item() > 0.0
    assert neighbor_distances.max().item() > 0.0


@pytest.mark.parametrize("device", ["cuda:0"])
def test_pair_centric_partial_outputs_are_compact(device):
    """Pair-centric ``target_indices`` uses compact rows with optional outputs."""
    _skip_missing_cuda(device)

    positions_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    box = 4.0
    cutoff = 0.6
    max_neighbors = 4
    state = _build_single_smoke_state(device, positions_np, box, cutoff, max_neighbors)

    target_indices = torch.tensor([0], dtype=torch.int32, device=device)
    params_t = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32, device=device
    )
    neighbor_matrix = torch.full(
        (1, max_neighbors), -1, dtype=torch.int32, device=device
    )
    neighbor_matrix_shifts = torch.zeros(
        (1, max_neighbors, 3), dtype=torch.int32, device=device
    )
    num_neighbors = torch.zeros(1, dtype=torch.int32, device=device)
    pair_energies = torch.zeros((1, max_neighbors), dtype=torch.float32, device=device)
    pair_forces = torch.zeros((1, max_neighbors, 3), dtype=torch.float32, device=device)
    neighbor_vectors = torch.zeros(
        (1, max_neighbors, 3), dtype=torch.float32, device=device
    )
    neighbor_distances = torch.zeros(
        (1, max_neighbors), dtype=torch.float32, device=device
    )

    query_cell_list(
        wp.from_torch(state["positions_t"], dtype=wp.vec3f),
        wp.from_torch(state["cell_t"], dtype=wp.mat33f),
        wp.from_torch(state["pbc_t"], dtype=wp.bool),
        cutoff,
        wp.from_torch(state["cells_per_dimension"], dtype=wp.int32),
        wp.from_torch(state["neighbor_search_radius"], dtype=wp.int32),
        wp.from_torch(state["atom_periodic_shifts"], dtype=wp.vec3i),
        wp.from_torch(state["atom_to_cell_mapping"], dtype=wp.vec3i),
        wp.from_torch(state["atoms_per_cell_count"], dtype=wp.int32),
        wp.from_torch(state["cell_atom_start_indices"], dtype=wp.int32),
        wp.from_torch(state["cell_atom_list"], dtype=wp.int32),
        wp.from_torch(neighbor_matrix, dtype=wp.int32),
        wp.from_torch(neighbor_matrix_shifts, dtype=wp.vec3i),
        wp.from_torch(num_neighbors, dtype=wp.int32),
        wp.float32,
        device,
        half_fill=False,
        strategy="pair_centric",
        n_outer=compute_batch_pair_centric_n_outer(
            tuple(state["neighbor_search_radius"].cpu().tolist()),
            half_fill=False,
        ),
        target_indices=wp.from_torch(target_indices, dtype=wp.int32),
        pair_fn=_factory_pair_fn_clf,
        pair_params=wp.from_torch(params_t, dtype=wp.float32),
        pair_energies=wp.from_torch(pair_energies, dtype=wp.float32),
        pair_forces=wp.from_torch(pair_forces, dtype=wp.vec3f),
        return_vectors=True,
        neighbor_vectors=wp.from_torch(neighbor_vectors, dtype=wp.vec3f),
        return_distances=True,
        neighbor_distances=wp.from_torch(neighbor_distances, dtype=wp.float32),
        sorted_positions=wp.from_torch(state["sorted_positions"], dtype=wp.vec3f),
        sorted_atom_periodic_shifts=wp.from_torch(
            state["sorted_shifts"], dtype=wp.vec3i
        ),
        rebuild_flags=wp.from_torch(state["rebuild_flags"], dtype=wp.bool),
    )

    nn = int(num_neighbors.cpu().item())
    assert nn == 2
    found = set(neighbor_matrix[0, :nn].cpu().tolist())
    assert found == {1, 2}
    assert pair_energies[0, :nn].abs().min().item() > 0.0
    assert neighbor_vectors[0, :nn].abs().max().item() > 0.0
    assert neighbor_distances[0, :nn].cpu().tolist() == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cell_list_lj_pair_fn(device):
    """Cell-list ``pair_fn`` computes Lennard-Jones energy + force matching NumPy."""
    _skip_missing_cuda(device)

    positions_np = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.5, 0.0], [0.4, 0.4, 0.6]],
        dtype=np.float32,
    )
    # (epsilon, sigma) per atom.
    params_np = np.array(
        [[1.0, 1.0], [1.2, 0.9], [0.8, 1.1], [1.0, 1.05]],
        dtype=np.float32,
    )
    cutoff = 2.0
    box = 8.0
    n_atoms = positions_np.shape[0]
    max_neighbors = 8

    state = _build_single_smoke_state(device, positions_np, box, cutoff, max_neighbors)
    params_t = torch.from_numpy(params_np).to(device)
    pair_energies = torch.zeros(
        (n_atoms, max_neighbors), dtype=torch.float32, device=device
    )
    pair_forces = torch.zeros(
        (n_atoms, max_neighbors, 3), dtype=torch.float32, device=device
    )

    query_cell_list(
        wp.from_torch(state["positions_t"], dtype=wp.vec3f),
        wp.from_torch(state["cell_t"], dtype=wp.mat33f),
        wp.from_torch(state["pbc_t"], dtype=wp.bool),
        cutoff,
        wp.from_torch(state["cells_per_dimension"], dtype=wp.int32),
        wp.from_torch(state["neighbor_search_radius"], dtype=wp.int32),
        wp.from_torch(state["atom_periodic_shifts"], dtype=wp.vec3i),
        wp.from_torch(state["atom_to_cell_mapping"], dtype=wp.vec3i),
        wp.from_torch(state["atoms_per_cell_count"], dtype=wp.int32),
        wp.from_torch(state["cell_atom_start_indices"], dtype=wp.int32),
        wp.from_torch(state["cell_atom_list"], dtype=wp.int32),
        wp.from_torch(state["neighbor_matrix"], dtype=wp.int32),
        wp.from_torch(state["neighbor_matrix_shifts"], dtype=wp.vec3i),
        wp.from_torch(state["num_neighbors"], dtype=wp.int32),
        wp.float32,
        device,
        half_fill=False,
        strategy="atom_centric",
        pair_fn=_lj_pair_fn_clf,
        pair_params=wp.from_torch(params_t, dtype=wp.float32),
        pair_energies=wp.from_torch(pair_energies, dtype=wp.float32),
        pair_forces=wp.from_torch(pair_forces, dtype=wp.vec3f),
        sorted_positions=wp.from_torch(state["sorted_positions"], dtype=wp.vec3f),
        sorted_atom_periodic_shifts=wp.from_torch(
            state["sorted_shifts"], dtype=wp.vec3i
        ),
        rebuild_flags=wp.from_torch(state["rebuild_flags"], dtype=wp.bool),
    )

    nm_cpu = state["neighbor_matrix"].cpu().numpy()
    nn_cpu = state["num_neighbors"].cpu().numpy()
    pe_cpu = pair_energies.cpu().numpy()
    pf_cpu = pair_forces.cpu().numpy()
    cutoff_sq = cutoff * cutoff

    checked_pairs = 0
    for i in range(n_atoms):
        for slot in range(int(nn_cpu[i])):
            j = int(nm_cpu[i, slot])
            assert j >= 0
            r_ij = positions_np[j] - positions_np[i]
            r2 = float(np.dot(r_ij, r_ij))
            assert r2 < cutoff_sq, f"pair ({i},{j}) exceeds cutoff"
            ref_energy, ref_force = _lj_reference(
                r_ij,
                params_np[i, 0],
                params_np[i, 1],
                params_np[j, 0],
                params_np[j, 1],
            )
            assert pe_cpu[i, slot] == pytest.approx(ref_energy, rel=1e-5, abs=1e-5)
            np.testing.assert_allclose(pf_cpu[i, slot], ref_force, rtol=1e-5, atol=1e-5)
            checked_pairs += 1
    assert checked_pairs > 0


def test_root_namespace_hides_low_level_helpers():
    """Root neighbors namespace should not expose low-level helper imports."""
    import nvalchemiops.neighbors as neighbors

    for name in (
        "get_query_cell_list_kernel",
        "get_naive_neighbor_matrix_kernel",
        "select_cell_list_strategy",
        "compute_batch_pair_centric_n_outer",
        "fill_neighbor_matrix_tail",
        "wrap_positions_single",
    ):
        assert not hasattr(neighbors, name)
