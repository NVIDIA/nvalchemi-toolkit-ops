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

"""End-to-end ``pair_fn`` tests for the high-level Torch cell-list bindings.

Regression coverage for the case where the Torch wrapper converted pair-output
buffers with ``return_ctype=True`` and handed those CTYPE structs to
``_prepare_pair_output_args``, which dereferences ``pair_params.dtype`` and
crashed with ``AttributeError: 'array_t' object has no attribute 'dtype'``
*before* launching the kernel.  These exercise the high-level
``cell_list`` / ``batch_cell_list`` entry points (the level a user calls);
``test/neighbors/test_pair_outputs.py`` only covers the ``output_args`` helpers
in isolation and never exercised this conversion path.
"""

import pytest
import torch
import warp as wp

from nvalchemiops.torch.neighbors.batch_cell_list import batch_cell_list
from nvalchemiops.torch.neighbors.batch_naive import batch_naive_neighbor_list
from nvalchemiops.torch.neighbors.cell_list import cell_list
from nvalchemiops.torch.neighbors.naive import naive_neighbor_list

from ...test_utils import create_simple_cubic_system


@wp.func
def _sum_pair_fn(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    """Analytic pair function: ``energy = p_i + p_j + distance``, ``force = -r_ij``.

    Chosen so the result can be cross-checked exactly against the kernel's own
    ``neighbor_vectors`` / ``neighbor_distances`` outputs.
    """
    energy = pair_params[i, 0] + pair_params[j, 0] + distance
    force = -r_ij
    return energy, force


def _alloc_pair_buffers(n_atoms: int, max_neighbors: int, device: str):
    """Allocate the output + scratch buffers consumed by the pair path."""
    nm = torch.full((n_atoms, max_neighbors), -1, dtype=torch.int32, device=device)
    nms = torch.zeros((n_atoms, max_neighbors, 3), dtype=torch.int32, device=device)
    nn = torch.zeros((n_atoms,), dtype=torch.int32, device=device)
    nv = torch.zeros((n_atoms, max_neighbors, 3), dtype=torch.float32, device=device)
    nd = torch.zeros((n_atoms, max_neighbors), dtype=torch.float32, device=device)
    pe = torch.zeros((n_atoms, max_neighbors), dtype=torch.float32, device=device)
    pf = torch.zeros((n_atoms, max_neighbors, 3), dtype=torch.float32, device=device)
    # Per-atom parameter (num_params == 1); distinct per atom.
    pp = (
        (torch.arange(n_atoms, dtype=torch.float32, device=device) + 1.0) * 0.5
    ).reshape(n_atoms, 1)
    return nm, nms, nn, nv, nd, pe, pf, pp


def _check_pair_outputs(nm, nn, nv, nd, pe, pf, pp):
    """Verify pair_energies/pair_forces match ``_sum_pair_fn`` on filled slots."""
    nm = nm.cpu()
    nn = nn.cpu()
    nv = nv.cpu()
    nd = nd.cpu()
    pe = pe.cpu()
    pf = pf.cpu()
    pp = pp.cpu()
    checked = 0
    n_atoms = nm.shape[0]
    for i in range(n_atoms):
        for slot in range(int(nn[i])):
            j = int(nm[i, slot])
            assert 0 <= j < n_atoms
            expected_energy = float(pp[i, 0]) + float(pp[j, 0]) + float(nd[i, slot])
            assert pe[i, slot] == pytest.approx(expected_energy, rel=1e-5, abs=1e-5)
            expected_force = -nv[i, slot]
            assert torch.allclose(pf[i, slot], expected_force, rtol=1e-5, atol=1e-5)
            checked += 1
    assert checked > 0, "no neighbor pairs were found; test exercised nothing"


def test_cell_list_pair_fn_runs_and_matches(device):
    """High-level single-system ``cell_list`` with ``pair_fn`` (regression)."""
    dtype = torch.float32
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=dtype, device=device
    )
    pbc = pbc.reshape(3)
    cutoff = 1.1
    max_neighbors = 16
    nm, nms, nn, nv, nd, pe, pf, pp = _alloc_pair_buffers(8, max_neighbors, device)

    cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        neighbor_matrix=nm,
        neighbor_matrix_shifts=nms,
        num_neighbors=nn,
        return_vectors=True,
        return_distances=True,
        pair_fn=_sum_pair_fn,
        pair_params=pp,
        neighbor_vectors=nv,
        neighbor_distances=nd,
        pair_energies=pe,
        pair_forces=pf,
    )

    _check_pair_outputs(nm, nn, nv, nd, pe, pf, pp)


def test_cell_list_pair_fn_coo_outputs_aligned(device):
    """COO format returns ``pair_fn`` energies/forces COO-packed and aligned
    with the neighbor list; the in-place matrix buffers are left untouched."""
    dtype = torch.float32
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=dtype, device=device
    )
    pbc = pbc.reshape(3)
    cutoff = 1.1
    max_neighbors = 16
    nm, nms, nn, nv, nd, pe, pf, pp = _alloc_pair_buffers(8, max_neighbors, device)

    nl, _nptr, _nl_shifts, d_coo, v_coo, pe_coo, pf_coo = cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        neighbor_matrix=nm,
        neighbor_matrix_shifts=nms,
        num_neighbors=nn,
        return_neighbor_list=True,
        return_vectors=True,
        return_distances=True,
        pair_fn=_sum_pair_fn,
        pair_params=pp,
        neighbor_vectors=nv,
        neighbor_distances=nd,
        pair_energies=pe,
        pair_forces=pf,
    )
    num_pairs = nl.shape[1]
    assert pe_coo.shape == (num_pairs,)
    assert pf_coo.shape == (num_pairs, 3)
    # COO energies/forces match ``_sum_pair_fn`` evaluated on the COO pairs.
    i_idx, j_idx = nl[0].long(), nl[1].long()
    expected_e = pp[i_idx, 0] + pp[j_idx, 0] + d_coo
    assert torch.allclose(pe_coo, expected_e, rtol=1e-5, atol=1e-5)
    assert torch.allclose(pf_coo, -v_coo, rtol=1e-5, atol=1e-5)
    # The caller's in-place buffers keep their matrix layout.
    assert pe.shape == (8, max_neighbors)


def test_naive_pair_fn_optional_buffers_and_returned(device):
    """Torch naive with ``pair_fn``: energy/force buffers are optional
    (auto-allocated) and returned, matching ``_sum_pair_fn``."""
    dtype = torch.float32
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=dtype, device=device
    )
    pbc = pbc.reshape(3)
    cutoff = 1.1
    max_neighbors = 16
    pp = ((torch.arange(8, dtype=dtype, device=device) + 1.0) * 0.5).reshape(8, 1)
    # Pass neither pair_energies nor pair_forces: they are auto-allocated.
    nm, nn, _shifts, nd, nv, pe, pf = naive_neighbor_list(
        positions,
        cutoff,
        cell,
        pbc,
        max_neighbors=max_neighbors,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn,
        pair_params=pp,
    )
    assert pe.shape == (8, max_neighbors)
    assert pf.shape == (8, max_neighbors, 3)
    _check_pair_outputs(nm, nn, nv, nd, pe, pf, pp)


def test_naive_pair_fn_coo_outputs_aligned(device):
    """Torch naive ``pair_fn`` energies/forces are COO-packed and aligned with
    the neighbor list in COO mode."""
    dtype = torch.float32
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=dtype, device=device
    )
    pbc = pbc.reshape(3)
    pp = ((torch.arange(8, dtype=dtype, device=device) + 1.0) * 0.5).reshape(8, 1)
    nl, _nptr, _nl_shifts, d_coo, _v_coo, pe_coo, pf_coo = naive_neighbor_list(
        positions,
        1.1,
        cell,
        pbc,
        max_neighbors=16,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn,
        pair_params=pp,
    )
    num_pairs = nl.shape[1]
    assert pe_coo.shape == (num_pairs,)
    assert pf_coo.shape == (num_pairs, 3)
    i_idx, j_idx = nl[0].long(), nl[1].long()
    expected_e = pp[i_idx, 0] + pp[j_idx, 0] + d_coo
    assert torch.allclose(pe_coo, expected_e, rtol=1e-5, atol=1e-5)


def test_batch_cell_list_pair_fn_runs_and_matches(device):
    """High-level batched ``batch_cell_list`` with ``pair_fn`` (regression).

    This is the exact path from the reported crash:
    ``batch_cell_list -> _batch_query_cell_list_optional ->
    batch_query_cell_list_pair_centric_sorted -> _prepare_pair_output_args``.
    """
    dtype = torch.float32
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=dtype, device=device
    )
    pbc = pbc.reshape(1, 3)
    batch_idx = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)
    cutoff = 1.1
    max_neighbors = 16
    nm, nms, nn, nv, nd, pe, pf, pp = _alloc_pair_buffers(8, max_neighbors, device)

    batch_cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        batch_idx,
        neighbor_matrix=nm,
        neighbor_matrix_shifts=nms,
        num_neighbors=nn,
        return_vectors=True,
        return_distances=True,
        pair_fn=_sum_pair_fn,
        pair_params=pp,
        neighbor_vectors=nv,
        neighbor_distances=nd,
        pair_energies=pe,
        pair_forces=pf,
    )

    _check_pair_outputs(nm, nn, nv, nd, pe, pf, pp)


def test_batch_naive_pair_fn_optional_buffers_and_returned(device):
    """Torch batch_naive with ``pair_fn``: energy/force buffers are optional
    (auto-allocated) and returned, matching ``_sum_pair_fn``."""
    dtype = torch.float32
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=dtype, device=device
    )
    pbc = pbc.reshape(1, 3)
    batch_idx = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)
    batch_ptr = torch.tensor([0, positions.shape[0]], dtype=torch.int32, device=device)
    max_neighbors = 16
    pp = ((torch.arange(8, dtype=dtype, device=device) + 1.0) * 0.5).reshape(8, 1)
    nm, nn, _shifts, nd, nv, pe, pf = batch_naive_neighbor_list(
        positions,
        1.1,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        pbc=pbc,
        cell=cell,
        max_neighbors=max_neighbors,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn,
        pair_params=pp,
    )
    assert pe.shape == (8, max_neighbors)
    assert pf.shape == (8, max_neighbors, 3)
    _check_pair_outputs(nm, nn, nv, nd, pe, pf, pp)
