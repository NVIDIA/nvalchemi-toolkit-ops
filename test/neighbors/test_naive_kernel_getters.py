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

"""Tests for the public naive kernel kernel getter helper."""

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.naive import (
    get_naive_neighbor_matrix_kernel,
    naive_neighbor_matrix,
    naive_neighbor_matrix_pbc,
)
from nvalchemiops.neighbors.naive.launchers import _reject_pair_fn_for_dual_cutoff
from nvalchemiops.torch.neighbors.neighbor_utils import compute_naive_num_shifts


@wp.func
def _factory_pair_fn(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    energy = pair_params[i, 0] + pair_params[j, 0] + distance
    force = -r_ij
    return energy, force


@wp.func
def _factory_pair_fn_alt(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    energy = pair_params[i, 0] - pair_params[j, 0] + distance
    force = r_ij
    return energy, force


@wp.func
def _lj_pair_fn(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    """Lennard-Jones pair function with Lorentz-Berthelot mixing.

    Reads ``(epsilon, sigma)`` from ``pair_params[i]`` and ``pair_params[j]``.
    Returns energy ``U_ij`` and the Cartesian force on atom ``i`` due to atom
    ``j``: ``F_i = -(24 eps / r^2)(2 (sigma/r)^12 - (sigma/r)^6) * r_ij``,
    where ``r_ij = positions[j] - positions[i]``.
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
    """NumPy reference for the LJ pair function (matches `_lj_pair_fn`)."""
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
    """Skip CUDA cases when the local test host has no CUDA device."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test parameter")


def test_single_cutoff_kernel_uses_pair_fn_object_cache_key():
    """The cache key uses the Warp function object, not id(pair_fn)."""
    kernel1 = get_naive_neighbor_matrix_kernel(
        wp.float32,
        pair_fn=_factory_pair_fn,
    )
    kernel2 = get_naive_neighbor_matrix_kernel(
        wp.float32,
        pair_fn=_factory_pair_fn,
    )
    kernel3 = get_naive_neighbor_matrix_kernel(
        wp.float32,
        pair_fn=_factory_pair_fn_alt,
    )

    assert kernel1 is kernel2
    assert kernel1 is not kernel3


def test_dual_cutoff_allows_empty_pair_params_sentinel():
    """Dual-cutoff validation treats ``(0, 0)`` pair params as inactive."""
    _reject_pair_fn_for_dual_cutoff(
        None,
        wp.empty((0, 0), dtype=wp.float32, device="cpu"),
    )


def test_dual_cutoff_rejects_active_pair_params_without_pair_fn():
    """Dual-cutoff validation rejects active pair params without ``pair_fn``."""
    with pytest.raises(ValueError, match="pair_params is only valid"):
        _reject_pair_fn_for_dual_cutoff(
            None,
            wp.empty((1, 0), dtype=wp.float32, device="cpu"),
        )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_naive_pair_fn_outputs(device):
    """No-PBC pair_fn writes per-slot energies and row forces."""
    _skip_missing_cuda(device)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    pair_params = torch.tensor(
        [[1.0], [2.0], [3.0]], dtype=torch.float32, device=device
    )
    neighbor_matrix = torch.full((3, 4), -1, dtype=torch.int32, device=device)
    num_neighbors = torch.zeros(3, dtype=torch.int32, device=device)
    pair_energies = torch.zeros((3, 4), dtype=torch.float32, device=device)
    pair_forces = torch.zeros((3, 4, 3), dtype=torch.float32, device=device)

    naive_neighbor_matrix(
        wp.from_torch(positions, dtype=wp.vec3f),
        0.75,
        wp.from_torch(neighbor_matrix, dtype=wp.int32),
        wp.from_torch(num_neighbors, dtype=wp.int32),
        wp.float32,
        device,
        pair_fn=_factory_pair_fn,
        pair_params=wp.from_torch(pair_params, dtype=wp.float32),
        pair_energies=wp.from_torch(pair_energies, dtype=wp.float32),
        pair_forces=wp.from_torch(pair_forces, dtype=wp.vec3f),
    )

    assert num_neighbors[0].item() == 1
    assert neighbor_matrix[0, 0].item() == 1
    assert pair_energies[0, 0].item() == pytest.approx(3.5)
    assert pair_forces[0, 0].detach().cpu().tolist() == pytest.approx(
        [-0.5, -0.0, -0.0]
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_naive_pbc_partial_pair_fn_outputs(device):
    """PBC partial rows support pair_fn and do not require symmetric target rows."""
    _skip_missing_cuda(device)
    positions = torch.tensor(
        [[0.1, 0.1, 0.1], [1.9, 0.1, 0.1]],
        dtype=torch.float32,
        device=device,
    )
    cell = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0) * 2.0
    pbc = torch.ones((1, 3), dtype=torch.bool, device=device)
    shift_range, num_shifts, _ = compute_naive_num_shifts(cell, 0.3, pbc)
    target_indices = torch.tensor([0], dtype=torch.int32, device=device)
    pair_params = torch.tensor([[1.0], [2.0]], dtype=torch.float32, device=device)
    neighbor_matrix = torch.full((1, 8), -1, dtype=torch.int32, device=device)
    neighbor_matrix_shifts = torch.zeros((1, 8, 3), dtype=torch.int32, device=device)
    num_neighbors = torch.zeros(1, dtype=torch.int32, device=device)
    pair_energies = torch.zeros((1, 8), dtype=torch.float32, device=device)
    pair_forces = torch.zeros((1, 8, 3), dtype=torch.float32, device=device)

    naive_neighbor_matrix_pbc(
        wp.from_torch(positions, dtype=wp.vec3f),
        0.3,
        wp.from_torch(cell, dtype=wp.mat33f),
        wp.from_torch(shift_range, dtype=wp.vec3i),
        int(num_shifts[0].item()),
        wp.from_torch(neighbor_matrix, dtype=wp.int32),
        wp.from_torch(neighbor_matrix_shifts, dtype=wp.vec3i),
        wp.from_torch(num_neighbors, dtype=wp.int32),
        wp.float32,
        device,
        target_indices=wp.from_torch(target_indices, dtype=wp.int32),
        pair_fn=_factory_pair_fn,
        pair_params=wp.from_torch(pair_params, dtype=wp.float32),
        pair_energies=wp.from_torch(pair_energies, dtype=wp.float32),
        pair_forces=wp.from_torch(pair_forces, dtype=wp.vec3f),
    )

    assert num_neighbors[0].item() >= 1
    assert 1 in neighbor_matrix[0, : num_neighbors[0].item()].detach().cpu().tolist()
    first = (neighbor_matrix[0, : num_neighbors[0].item()] == 1).nonzero()[0].item()
    assert pair_energies[0, first].item() == pytest.approx(3.2, abs=1.0e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_naive_lj_pair_fn(device):
    """No-PBC pair_fn computes Lennard-Jones energy + force matching NumPy reference.

    Atoms have heterogeneous (epsilon, sigma) per-atom parameters so the
    Lorentz-Berthelot mixing path inside ``_lj_pair_fn`` is exercised on
    every pair.
    """
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
    n_atoms = positions_np.shape[0]
    max_neighbors = 8

    positions = torch.from_numpy(positions_np).to(device)
    pair_params = torch.from_numpy(params_np).to(device)
    neighbor_matrix = torch.full(
        (n_atoms, max_neighbors), -1, dtype=torch.int32, device=device
    )
    num_neighbors = torch.zeros(n_atoms, dtype=torch.int32, device=device)
    pair_energies = torch.zeros(
        (n_atoms, max_neighbors), dtype=torch.float32, device=device
    )
    pair_forces = torch.zeros(
        (n_atoms, max_neighbors, 3), dtype=torch.float32, device=device
    )

    naive_neighbor_matrix(
        wp.from_torch(positions, dtype=wp.vec3f),
        cutoff,
        wp.from_torch(neighbor_matrix, dtype=wp.int32),
        wp.from_torch(num_neighbors, dtype=wp.int32),
        wp.float32,
        device,
        pair_fn=_lj_pair_fn,
        pair_params=wp.from_torch(pair_params, dtype=wp.float32),
        pair_energies=wp.from_torch(pair_energies, dtype=wp.float32),
        pair_forces=wp.from_torch(pair_forces, dtype=wp.vec3f),
    )

    nm_cpu = neighbor_matrix.detach().cpu().numpy()
    nn_cpu = num_neighbors.detach().cpu().numpy()
    pe_cpu = pair_energies.detach().cpu().numpy()
    pf_cpu = pair_forces.detach().cpu().numpy()
    cutoff_sq = cutoff * cutoff

    # Verify every recorded pair's energy and force match the NumPy reference.
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
    # Sanity-check that the dataset actually produced pairs to verify.
    assert checked_pairs > 0


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_naive_pbc_lj_pair_fn(device):
    """PBC partial-row pair_fn computes Lennard-Jones outputs across periodic images."""
    _skip_missing_cuda(device)
    # Two atoms near opposite faces of a small periodic box so the nearest
    # neighbour crosses a periodic image rather than the direct distance.
    positions_np = np.array(
        [[0.1, 0.5, 0.5], [1.9, 0.5, 0.5]],
        dtype=np.float32,
    )
    params_np = np.array([[1.0, 1.0], [1.5, 0.9]], dtype=np.float32)
    box = 2.0
    cutoff = 0.5

    positions = torch.from_numpy(positions_np).to(device)
    cell = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0) * box
    pbc = torch.ones((1, 3), dtype=torch.bool, device=device)
    shift_range, num_shifts, _ = compute_naive_num_shifts(cell, cutoff, pbc)
    target_indices = torch.tensor([0], dtype=torch.int32, device=device)
    pair_params = torch.from_numpy(params_np).to(device)
    max_neighbors = 8
    neighbor_matrix = torch.full(
        (1, max_neighbors), -1, dtype=torch.int32, device=device
    )
    neighbor_matrix_shifts = torch.zeros(
        (1, max_neighbors, 3), dtype=torch.int32, device=device
    )
    num_neighbors = torch.zeros(1, dtype=torch.int32, device=device)
    pair_energies = torch.zeros((1, max_neighbors), dtype=torch.float32, device=device)
    pair_forces = torch.zeros((1, max_neighbors, 3), dtype=torch.float32, device=device)

    naive_neighbor_matrix_pbc(
        wp.from_torch(positions, dtype=wp.vec3f),
        cutoff,
        wp.from_torch(cell, dtype=wp.mat33f),
        wp.from_torch(shift_range, dtype=wp.vec3i),
        int(num_shifts[0].item()),
        wp.from_torch(neighbor_matrix, dtype=wp.int32),
        wp.from_torch(neighbor_matrix_shifts, dtype=wp.vec3i),
        wp.from_torch(num_neighbors, dtype=wp.int32),
        wp.float32,
        device,
        target_indices=wp.from_torch(target_indices, dtype=wp.int32),
        pair_fn=_lj_pair_fn,
        pair_params=wp.from_torch(pair_params, dtype=wp.float32),
        pair_energies=wp.from_torch(pair_energies, dtype=wp.float32),
        pair_forces=wp.from_torch(pair_forces, dtype=wp.vec3f),
    )

    nn = int(num_neighbors[0].item())
    assert nn >= 1
    nm_cpu = neighbor_matrix.detach().cpu().numpy()[0, :nn]
    nms_cpu = neighbor_matrix_shifts.detach().cpu().numpy()[0, :nn]
    pe_cpu = pair_energies.detach().cpu().numpy()[0, :nn]
    pf_cpu = pair_forces.detach().cpu().numpy()[0, :nn]
    cutoff_sq = cutoff * cutoff

    for slot in range(nn):
        j = int(nm_cpu[slot])
        shift = nms_cpu[slot].astype(np.float32)
        # PBC partial-row convention: r_ij is the displacement from atom 0 to
        # the periodic image of atom j shifted by ``shift``.
        r_ij = (positions_np[j] + shift * box) - positions_np[0]
        r2 = float(np.dot(r_ij, r_ij))
        assert r2 < cutoff_sq, (
            f"slot {slot}: pair ({0},{j},shift={shift}) exceeds cutoff"
        )
        ref_energy, ref_force = _lj_reference(
            r_ij,
            params_np[0, 0],
            params_np[0, 1],
            params_np[j, 0],
            params_np[j, 1],
        )
        assert pe_cpu[slot] == pytest.approx(ref_energy, rel=1e-4, abs=1e-5)
        np.testing.assert_allclose(pf_cpu[slot], ref_force, rtol=1e-4, atol=1e-5)
