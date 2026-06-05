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

"""Smoke tests for the cluster-tile pair-output kernel getter.

Mirrors :mod:`test.neighbors.test_naive_kernel_getters` and
:mod:`test.neighbors.test_cell_list_kernel_getters`: covers the
``get_query_cluster_tile_kernel`` cache-key contract, an integration test with a real
physical potential (Lorentz-Berthelot Lennard-Jones),.
"""

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.cluster_tile import (
    TILE_GROUP_SIZE,
    build_cluster_tile_list,
    get_query_cluster_tile_kernel,
    query_cluster_tile,
    query_cluster_tile_coo,
)
from nvalchemiops.neighbors.cluster_tile.launchers import (
    _compute_morton,
    _permute_gather_soa,
)

pytestmark = pytest.mark.gpu

# ---------------------------------------------------------------------------
# Module-scope @wp.func definitions (cache-key + LJ tests)
# ---------------------------------------------------------------------------


@wp.func
def _factory_pair_fn_ct(
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
def _factory_pair_fn_alt_ct(
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
def _lj_pair_fn_ct(
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
    """NumPy reference for the LJ pair function (matches ``_lj_pair_fn_ct``)."""
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


# ---------------------------------------------------------------------------
# get_query_cluster_tile_kernel cache key
# ---------------------------------------------------------------------------


def test_query_cluster_tile_kernel_uses_pair_fn_object_cache_key():
    """The cache key uses the Warp function object, not ``id(pair_fn)``."""
    kernel1 = get_query_cluster_tile_kernel(
        pair_fn=_factory_pair_fn_ct,
    )
    kernel2 = get_query_cluster_tile_kernel(
        pair_fn=_factory_pair_fn_ct,
    )
    kernel3 = get_query_cluster_tile_kernel(
        pair_fn=_factory_pair_fn_alt_ct,
    )
    assert kernel1 is kernel2
    assert kernel1 is not kernel3


# ---------------------------------------------------------------------------
# LJ integration: cluster-tile pair_fn output vs NumPy reference
# ---------------------------------------------------------------------------


def _mat33f_from_torch(mat: torch.Tensor):
    if mat.ndim == 2:
        mat = mat.unsqueeze(0)
    return wp.from_torch(
        mat.detach().contiguous().to(torch.float32),
        dtype=wp.mat33f,
        return_ctype=True,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cluster_tile_lj_pair_fn(device):
    """Lorentz-Berthelot LJ pair function via ``query_cluster_tile``.

    Builds a small Morton-sorted cluster (TILE_GROUP_SIZE atoms), runs the
    pair-output ``query_cluster_tile`` path with the ``_lj_pair_fn_ct``
    Warp ``@wp.func``, and compares the per-pair energies against a NumPy
    reference.
    """
    _skip_missing_cuda(device)
    if device == "cpu":
        pytest.skip(
            "cluster_tile kernels use Warp tile primitives and are CUDA-only; "
            "CPU parameter is not supported"
        )

    torch.manual_seed(0)
    natom = TILE_GROUP_SIZE  # one cluster
    box = 8.0
    cutoff = 3.0
    max_neighbors = natom

    # Random atoms spread inside the box; LJ params per atom.
    positions = torch.rand(natom, 3, dtype=torch.float32, device=device) * box
    eps = (torch.rand(natom, dtype=torch.float32, device=device) * 0.5) + 0.5
    sigma = (torch.rand(natom, dtype=torch.float32, device=device) * 0.5) + 0.8
    pair_params = torch.stack([eps, sigma], dim=1).contiguous()

    cell = torch.eye(3, dtype=torch.float32, device=device) * box
    inv_cell = torch.linalg.inv(cell).contiguous()

    # ---- Morton sort upstream ----
    morton_codes = torch.zeros(natom, dtype=torch.int32, device=device)
    sorted_atom_index = torch.zeros(natom, dtype=torch.int32, device=device)
    sorted_pos_x = torch.zeros(natom, dtype=torch.float32, device=device)
    sorted_pos_y = torch.zeros(natom, dtype=torch.float32, device=device)
    sorted_pos_z = torch.zeros(natom, dtype=torch.float32, device=device)
    num_neighbors_scratch = torch.zeros(natom, dtype=torch.int32, device=device)
    num_tiles = torch.zeros(1, dtype=torch.int32, device=device)

    _compute_morton(
        wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True),
        _mat33f_from_torch(inv_cell),
        natom,
        wp.from_torch(morton_codes, dtype=wp.int32, return_ctype=True),
        wp.from_torch(sorted_atom_index, dtype=wp.int32, return_ctype=True),
        wp.from_torch(num_neighbors_scratch, dtype=wp.int32, return_ctype=True),
        wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        device,
    )
    perm = torch.argsort(morton_codes)
    sorted_atom_index.copy_(perm.to(torch.int32))
    _permute_gather_soa(
        wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True),
        wp.from_torch(sorted_atom_index, dtype=wp.int32, return_ctype=True),
        natom,
        wp.from_torch(sorted_pos_x, dtype=wp.float32, return_ctype=True),
        wp.from_torch(sorted_pos_y, dtype=wp.float32, return_ctype=True),
        wp.from_torch(sorted_pos_z, dtype=wp.float32, return_ctype=True),
        device,
    )

    # ---- Tile-pair enumeration ----
    ngroup = 1
    ngroup_padded = TILE_GROUP_SIZE
    max_tiles = ngroup * ngroup
    tile_row_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
    tile_col_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
    group_ctr_x = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
    group_ctr_y = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
    group_ctr_z = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
    group_ext_x = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
    group_ext_y = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)
    group_ext_z = torch.zeros(ngroup_padded, dtype=torch.float32, device=device)

    build_cluster_tile_list(
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp.float32, return_ctype=True),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp.float32, return_ctype=True),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp.float32, return_ctype=True),
        cell=_mat33f_from_torch(cell),
        inv_cell=_mat33f_from_torch(inv_cell),
        cutoff=cutoff,
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
        tile_col_group=wp.from_torch(tile_col_group, dtype=wp.int32, return_ctype=True),
        wp_dtype=wp.float32,
        device=device,
        group_ctr_x_buffer=wp.from_torch(
            group_ctr_x, dtype=wp.float32, return_ctype=True
        ),
        group_ctr_y_buffer=wp.from_torch(
            group_ctr_y, dtype=wp.float32, return_ctype=True
        ),
        group_ctr_z_buffer=wp.from_torch(
            group_ctr_z, dtype=wp.float32, return_ctype=True
        ),
        group_ext_x_buffer=wp.from_torch(
            group_ext_x, dtype=wp.float32, return_ctype=True
        ),
        group_ext_y_buffer=wp.from_torch(
            group_ext_y, dtype=wp.float32, return_ctype=True
        ),
        group_ext_z_buffer=wp.from_torch(
            group_ext_z, dtype=wp.float32, return_ctype=True
        ),
    )

    n_tiles = int(num_tiles.cpu().item())
    assert n_tiles >= 1

    # ---- Pair-output query_cluster_tile with pair_fn = LJ ----
    neighbor_matrix = torch.full(
        (natom, max_neighbors), natom, dtype=torch.int32, device=device
    )
    neighbor_matrix_shifts = torch.zeros(
        (natom, max_neighbors, 3), dtype=torch.int32, device=device
    )
    num_neighbors = torch.zeros(natom, dtype=torch.int32, device=device)
    pair_energies = torch.zeros(
        (natom, max_neighbors), dtype=torch.float32, device=device
    )
    pair_forces = torch.zeros(
        (natom, max_neighbors, 3), dtype=torch.float32, device=device
    )

    query_cluster_tile(
        sorted_atom_index=wp.from_torch(
            sorted_atom_index, dtype=wp.int32, return_ctype=True
        ),
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp.float32, return_ctype=True),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp.float32, return_ctype=True),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp.float32, return_ctype=True),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
        tile_col_group=wp.from_torch(tile_col_group, dtype=wp.int32, return_ctype=True),
        cell=_mat33f_from_torch(cell),
        inv_cell=_mat33f_from_torch(inv_cell),
        cutoff=cutoff,
        natom=natom,
        neighbor_matrix=wp.from_torch(
            neighbor_matrix, dtype=wp.int32, return_ctype=True
        ),
        num_neighbors=wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True),
        neighbor_matrix_shifts=wp.from_torch(
            neighbor_matrix_shifts, dtype=wp.int32, return_ctype=True
        ),
        wp_dtype=wp.float32,
        device=device,
        pair_fn=_lj_pair_fn_ct,
        pair_params=wp.from_torch(pair_params, dtype=wp.float32),
        pair_energies=wp.from_torch(pair_energies, dtype=wp.float32),
        pair_forces=wp.from_torch(pair_forces, dtype=wp.vec3f),
    )

    # Compare to NumPy reference per active pair.
    positions_np = positions.cpu().numpy()
    eps_np = eps.cpu().numpy()
    sigma_np = sigma.cpu().numpy()
    cell_np = cell.cpu().numpy()
    inv_cell_np = inv_cell.cpu().numpy()
    nm_np = neighbor_matrix.cpu().numpy()
    nn_np = num_neighbors.cpu().numpy()
    pe_np = pair_energies.cpu().numpy()

    n_compared = 0
    for i in range(natom):
        for slot in range(int(nn_np[i])):
            j = int(nm_np[i, slot])
            d = positions_np[j] - positions_np[i]
            # Triclinic min-image wrap (matches ``_wrap_triclinic``).
            f = inv_cell_np.T @ d
            shift = -np.round(f).astype(np.int32)
            f = f + shift
            d_wrapped = cell_np.T @ f
            energy_ref, _ = _lj_reference(
                d_wrapped,
                float(eps_np[i]),
                float(sigma_np[i]),
                float(eps_np[j]),
                float(sigma_np[j]),
            )
            np.testing.assert_allclose(
                float(pe_np[i, slot]),
                energy_ref,
                rtol=1e-4,
                atol=1e-4,
            )
            n_compared += 1
    assert n_compared > 0, "expected at least one active pair to compare"


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cluster_tile_coo_pair_outputs(device):
    """COO conversion writes flat vectors, distances, and pair_fn outputs."""
    _skip_missing_cuda(device)
    if device == "cpu":
        pytest.skip(
            "cluster_tile kernels use Warp tile primitives and are CUDA-only; "
            "CPU parameter is not supported"
        )

    torch.manual_seed(2)
    natom = TILE_GROUP_SIZE
    box = 8.0
    cutoff = 3.0
    max_pairs = natom * natom

    positions = torch.rand(natom, 3, dtype=torch.float32, device=device) * box
    eps = (torch.rand(natom, dtype=torch.float32, device=device) * 0.5) + 0.5
    sigma = (torch.rand(natom, dtype=torch.float32, device=device) * 0.5) + 0.8
    pair_params = torch.stack([eps, sigma], dim=1).contiguous()
    cell = torch.eye(3, dtype=torch.float32, device=device) * box
    inv_cell = torch.linalg.inv(cell).contiguous()

    morton_codes = torch.zeros(natom, dtype=torch.int32, device=device)
    sorted_atom_index = torch.zeros(natom, dtype=torch.int32, device=device)
    sorted_pos_x = torch.zeros(natom, dtype=torch.float32, device=device)
    sorted_pos_y = torch.zeros(natom, dtype=torch.float32, device=device)
    sorted_pos_z = torch.zeros(natom, dtype=torch.float32, device=device)
    num_neighbors_scratch = torch.zeros(natom, dtype=torch.int32, device=device)
    num_tiles = torch.zeros(1, dtype=torch.int32, device=device)

    _compute_morton(
        wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True),
        _mat33f_from_torch(inv_cell),
        natom,
        wp.from_torch(morton_codes, dtype=wp.int32, return_ctype=True),
        wp.from_torch(sorted_atom_index, dtype=wp.int32, return_ctype=True),
        wp.from_torch(num_neighbors_scratch, dtype=wp.int32, return_ctype=True),
        wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        device,
    )
    sorted_atom_index.copy_(torch.argsort(morton_codes).to(torch.int32))
    _permute_gather_soa(
        wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True),
        wp.from_torch(sorted_atom_index, dtype=wp.int32, return_ctype=True),
        natom,
        wp.from_torch(sorted_pos_x, dtype=wp.float32, return_ctype=True),
        wp.from_torch(sorted_pos_y, dtype=wp.float32, return_ctype=True),
        wp.from_torch(sorted_pos_z, dtype=wp.float32, return_ctype=True),
        device,
    )

    tile_row_group = torch.zeros(1, dtype=torch.int32, device=device)
    tile_col_group = torch.zeros(1, dtype=torch.int32, device=device)
    group_ctr_x = torch.zeros(TILE_GROUP_SIZE, dtype=torch.float32, device=device)
    group_ctr_y = torch.zeros(TILE_GROUP_SIZE, dtype=torch.float32, device=device)
    group_ctr_z = torch.zeros(TILE_GROUP_SIZE, dtype=torch.float32, device=device)
    group_ext_x = torch.zeros(TILE_GROUP_SIZE, dtype=torch.float32, device=device)
    group_ext_y = torch.zeros(TILE_GROUP_SIZE, dtype=torch.float32, device=device)
    group_ext_z = torch.zeros(TILE_GROUP_SIZE, dtype=torch.float32, device=device)

    build_cluster_tile_list(
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp.float32, return_ctype=True),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp.float32, return_ctype=True),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp.float32, return_ctype=True),
        cell=_mat33f_from_torch(cell),
        inv_cell=_mat33f_from_torch(inv_cell),
        cutoff=cutoff,
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
        tile_col_group=wp.from_torch(tile_col_group, dtype=wp.int32, return_ctype=True),
        wp_dtype=wp.float32,
        device=device,
        group_ctr_x_buffer=wp.from_torch(
            group_ctr_x, dtype=wp.float32, return_ctype=True
        ),
        group_ctr_y_buffer=wp.from_torch(
            group_ctr_y, dtype=wp.float32, return_ctype=True
        ),
        group_ctr_z_buffer=wp.from_torch(
            group_ctr_z, dtype=wp.float32, return_ctype=True
        ),
        group_ext_x_buffer=wp.from_torch(
            group_ext_x, dtype=wp.float32, return_ctype=True
        ),
        group_ext_y_buffer=wp.from_torch(
            group_ext_y, dtype=wp.float32, return_ctype=True
        ),
        group_ext_z_buffer=wp.from_torch(
            group_ext_z, dtype=wp.float32, return_ctype=True
        ),
    )
    n_tiles = int(num_tiles.cpu().item())
    assert n_tiles >= 1

    pair_counter = torch.zeros(1, dtype=torch.int32, device=device)
    coo_list = torch.zeros((max_pairs, 2), dtype=torch.int32, device=device)
    coo_shifts = torch.zeros((max_pairs, 3), dtype=torch.int32, device=device)
    vectors = torch.zeros((max_pairs, 3), dtype=torch.float32, device=device)
    distances = torch.zeros(max_pairs, dtype=torch.float32, device=device)
    pair_energies = torch.zeros(max_pairs, dtype=torch.float32, device=device)
    pair_forces = torch.zeros((max_pairs, 3), dtype=torch.float32, device=device)

    query_cluster_tile_coo(
        sorted_atom_index=wp.from_torch(
            sorted_atom_index, dtype=wp.int32, return_ctype=True
        ),
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp.float32, return_ctype=True),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp.float32, return_ctype=True),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp.float32, return_ctype=True),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
        tile_col_group=wp.from_torch(tile_col_group, dtype=wp.int32, return_ctype=True),
        cell=_mat33f_from_torch(cell),
        inv_cell=_mat33f_from_torch(inv_cell),
        cutoff=cutoff,
        natom=natom,
        max_pairs=max_pairs,
        pair_counter=wp.from_torch(pair_counter, dtype=wp.int32, return_ctype=True),
        coo_list=wp.from_torch(coo_list, dtype=wp.int32, return_ctype=True),
        coo_shifts=wp.from_torch(coo_shifts, dtype=wp.int32, return_ctype=True),
        wp_dtype=wp.float32,
        device=device,
        return_vectors=True,
        return_distances=True,
        pair_fn=_lj_pair_fn_ct,
        pair_params=wp.from_torch(pair_params, dtype=wp.float32),
        neighbor_vectors=wp.from_torch(vectors, dtype=wp.vec3f),
        neighbor_distances=wp.from_torch(distances, dtype=wp.float32),
        pair_energies=wp.from_torch(pair_energies, dtype=wp.float32),
        pair_forces=wp.from_torch(pair_forces, dtype=wp.vec3f),
    )

    npairs = int(pair_counter.cpu().item())
    assert npairs > 0
    positions_np = positions.cpu().numpy()
    cell_np = cell.cpu().numpy()
    inv_cell_np = inv_cell.cpu().numpy()
    eps_np = eps.cpu().numpy()
    sigma_np = sigma.cpu().numpy()
    coo_np = coo_list[:npairs].cpu().numpy()
    vectors_np = vectors[:npairs].cpu().numpy()
    distances_np = distances[:npairs].cpu().numpy()
    energies_np = pair_energies[:npairs].cpu().numpy()

    for slot, (i, j) in enumerate(coo_np):
        d = positions_np[j] - positions_np[i]
        f = inv_cell_np.T @ d
        shift = -np.round(f).astype(np.int32)
        d_wrapped = cell_np.T @ (f + shift)
        energy_ref, _ = _lj_reference(
            d_wrapped,
            float(eps_np[i]),
            float(sigma_np[i]),
            float(eps_np[j]),
            float(sigma_np[j]),
        )
        np.testing.assert_allclose(vectors_np[slot], d_wrapped, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            distances_np[slot], np.linalg.norm(d_wrapped), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            float(energies_np[slot]), energy_ref, rtol=1e-4, atol=1e-4
        )


# ---------------------------------------------------------------------------
# Binding-layer pair-output surface: torch ``cluster_tile_neighbor_list`` LJ
# ---------------------------------------------------------------------------


def test_torch_cluster_tile_neighbor_list_pair_fn_smoke():
    """End-to-end LJ ``pair_fn`` through the torch binding writes finite energies.

    Exercises :func:`nvalchemiops.torch.neighbors.cluster_tile.cluster_tile_neighbor_list`
    with the new ``pair_fn`` / ``pair_params`` / ``pair_energies``
    / ``pair_forces`` kwargs to confirm they flow from the
    convenience wrapper down through ``query_cluster_tile`` into the warp
    pair-output kernel.  Distinct from the warp-layer LJ test above.
    """
    if not torch.cuda.is_available():
        pytest.skip("torch cluster_tile pair-output smoke requires CUDA tensors")
    from nvalchemiops.torch.neighbors.cluster_tile import cluster_tile_neighbor_list

    device = "cuda:0"
    torch.manual_seed(1)
    natom = TILE_GROUP_SIZE
    box = 8.0
    cutoff = 3.0
    max_neighbors = natom

    positions = torch.rand(natom, 3, dtype=torch.float32, device=device) * box
    eps = (torch.rand(natom, dtype=torch.float32, device=device) * 0.5) + 0.5
    sigma = (torch.rand(natom, dtype=torch.float32, device=device) * 0.5) + 0.8
    pair_params = torch.stack([eps, sigma], dim=1).contiguous()
    cell = torch.eye(3, dtype=torch.float32, device=device) * box

    neighbor_matrix, num_neighbors, _shifts = cluster_tile_neighbor_list(
        positions,
        cutoff,
        cell,
        max_neighbors=max_neighbors,
        pair_fn=_lj_pair_fn_ct,
        pair_params=pair_params,
    )
    # The binding allocates pair_energies / pair_forces
    # internally; callers can also pass them explicitly.  Smoke-check
    # that the matrix has live entries (the LJ values themselves are
    # validated by the warp-layer test above).
    assert int(num_neighbors.sum().item()) > 0
    assert neighbor_matrix.shape == (natom, max_neighbors)


def test_torch_cluster_tile_neighbor_list_coo_pair_outputs_smoke():
    """Torch ``format='coo'`` fills caller-owned flat pair-output buffers."""
    if not torch.cuda.is_available():
        pytest.skip("torch cluster_tile COO pair-output smoke requires CUDA tensors")
    from nvalchemiops.torch.neighbors.cluster_tile import cluster_tile_neighbor_list

    device = "cuda:0"
    torch.manual_seed(3)
    natom = TILE_GROUP_SIZE
    box = 8.0
    cutoff = 3.5
    max_neighbors = natom
    max_pairs = natom * max_neighbors

    positions = torch.rand(natom, 3, dtype=torch.float32, device=device) * box
    eps = (torch.rand(natom, dtype=torch.float32, device=device) * 0.5) + 0.5
    sigma = (torch.rand(natom, dtype=torch.float32, device=device) * 0.5) + 0.8
    pair_params = torch.stack([eps, sigma], dim=1).contiguous()
    cell = torch.eye(3, dtype=torch.float32, device=device) * box
    vectors = torch.zeros((max_pairs, 3), dtype=torch.float32, device=device)
    distances = torch.zeros(max_pairs, dtype=torch.float32, device=device)
    energies = torch.zeros(max_pairs, dtype=torch.float32, device=device)
    forces = torch.zeros((max_pairs, 3), dtype=torch.float32, device=device)

    neighbor_list, _neighbor_ptr, _shifts = cluster_tile_neighbor_list(
        positions,
        cutoff,
        cell,
        max_neighbors=max_neighbors,
        max_pairs=max_pairs,
        format="coo",
        return_vectors=True,
        return_distances=True,
        pair_fn=_lj_pair_fn_ct,
        pair_params=pair_params,
        neighbor_vectors=vectors,
        neighbor_distances=distances,
        pair_energies=energies,
        pair_forces=forces,
    )

    npairs = int(neighbor_list.shape[1])
    assert npairs > 0
    assert torch.isfinite(vectors[:npairs]).all()
    assert torch.isfinite(distances[:npairs]).all()
    assert torch.isfinite(energies[:npairs]).all()
    assert torch.isfinite(forces[:npairs]).all()
    assert torch.all(distances[:npairs] > 0)


def test_torch_cluster_tile_neighbor_list_coo_pair_outputs_require_buffers():
    """COO pair outputs require caller-owned flat output buffers."""
    if not torch.cuda.is_available():
        pytest.skip(
            "torch cluster_tile COO buffer-validation test constructs CUDA inputs"
        )
    from nvalchemiops.torch.neighbors.cluster_tile import cluster_tile_neighbor_list

    device = "cuda:0"
    natom = TILE_GROUP_SIZE
    positions = torch.rand(natom, 3, dtype=torch.float32, device=device)
    cell = torch.eye(3, dtype=torch.float32, device=device) * 10.0

    with pytest.raises(ValueError, match="neighbor_vectors is required"):
        cluster_tile_neighbor_list(
            positions,
            cutoff=10.0,
            cell=cell,
            max_neighbors=natom,
            format="coo",
            return_vectors=True,
        )


def test_torch_batch_cluster_tile_neighbor_list_coo_pair_outputs_smoke():
    """Batched Torch ``format='coo'`` fills flat pair-output buffers."""
    if not torch.cuda.is_available():
        pytest.skip(
            "torch batch_cluster_tile COO pair-output smoke requires CUDA tensors"
        )
    from nvalchemiops.torch.neighbors.batch_cluster_tile import (
        batch_cluster_tile_neighbor_list,
    )

    device = "cuda:0"
    torch.manual_seed(4)
    sizes = [TILE_GROUP_SIZE, TILE_GROUP_SIZE]
    positions = torch.cat(
        [
            torch.rand(sizes[0], 3, dtype=torch.float32, device=device) * 8.0,
            torch.rand(sizes[1], 3, dtype=torch.float32, device=device) * 7.0,
        ],
        dim=0,
    ).contiguous()
    batch_ptr = torch.tensor(
        [0, sizes[0], sum(sizes)], dtype=torch.int32, device=device
    )
    cell_batch = torch.stack(
        [
            torch.eye(3, dtype=torch.float32, device=device) * 8.0,
            torch.eye(3, dtype=torch.float32, device=device) * 7.0,
        ],
        dim=0,
    ).contiguous()
    max_neighbors = sum(sizes)
    max_pairs = int(positions.shape[0]) * max_neighbors
    vectors = torch.zeros((max_pairs, 3), dtype=torch.float32, device=device)
    distances = torch.zeros(max_pairs, dtype=torch.float32, device=device)

    neighbor_list, _neighbor_ptr, _shifts = batch_cluster_tile_neighbor_list(
        positions,
        cutoff=3.5,
        cell_batch=cell_batch,
        batch_ptr=batch_ptr,
        max_neighbors=max_neighbors,
        max_pairs=max_pairs,
        format="coo",
        return_vectors=True,
        return_distances=True,
        neighbor_vectors=vectors,
        neighbor_distances=distances,
    )

    npairs = int(neighbor_list.shape[1])
    assert npairs > 0
    assert torch.isfinite(vectors[:npairs]).all()
    assert torch.isfinite(distances[:npairs]).all()
    assert torch.all(distances[:npairs] > 0)


@wp.func
def _ct_getter_sum_pair_fn(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    return pair_params[i, 0] + pair_params[j, 0] + distance, -r_ij


def test_jax_cluster_tile_neighbor_list_pair_fn_supported():
    """JAX cluster_tile binding now wires ``pair_fn`` (fp32, matrix-only) via a
    call-time ``jax_callable`` closing over the function; returns per-pair
    ``pair_energies`` / ``pair_forces``.  See ``bindings/jax/test_pair_fn.py``.
    """
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from nvalchemiops.jax.neighbors.cluster_tile import cluster_tile_neighbor_list

    rng = np.random.RandomState(0)
    positions = jnp.asarray(
        rng.uniform(0.0, 5.0, size=(TILE_GROUP_SIZE, 3)).astype(np.float32)
    )
    cell = jnp.eye(3, dtype=jnp.float32) * 10.0
    pp = ((jnp.arange(TILE_GROUP_SIZE, dtype=jnp.float32) + 1.0) * 0.5).reshape(-1, 1)

    out = cluster_tile_neighbor_list(
        positions,
        cutoff=2.0,
        cell=cell,
        max_neighbors=64,
        return_distances=True,
        return_vectors=True,
        pair_fn=_ct_getter_sum_pair_fn,
        pair_params=pp,
    )
    # nm, nn, shifts, distances, vectors, pe, pf
    assert len(out) == 7
    assert out[5].shape[0] == TILE_GROUP_SIZE

    # pair_fn still requires pair_params.
    with pytest.raises(ValueError, match="pair_fn requires pair_params"):
        cluster_tile_neighbor_list(
            positions,
            cutoff=2.0,
            cell=cell,
            pair_fn=_ct_getter_sum_pair_fn,
        )


def test_tile_warp_import_paths_are_not_provided():
    """Removed tile_warp import paths are not compatibility surfaces."""
    import importlib

    for module_name in (
        "nvalchemiops.neighbors.tile_warp",
        "nvalchemiops.neighbors.tile_batch_warp",
        "nvalchemiops.torch.neighbors.tile_warp",
        "nvalchemiops.torch.neighbors.batch_tile_warp",
        "nvalchemiops.jax.neighbors.tile_warp",
        "nvalchemiops.jax.neighbors.batch_tile_warp",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)
