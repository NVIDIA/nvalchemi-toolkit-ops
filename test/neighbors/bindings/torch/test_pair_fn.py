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

from nvalchemiops.torch.neighbors import compile_pair_fn
from nvalchemiops.torch.neighbors.batch_cell_list import (
    batch_cell_list,
    batch_query_cell_list,
    estimate_batch_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.batch_naive import batch_naive_neighbor_list
from nvalchemiops.torch.neighbors.cell_list import (
    allocate_query_sort_scratch,
    cell_list,
    estimate_cell_list_sizes,
    query_cell_list,
)
from nvalchemiops.torch.neighbors.naive import naive_neighbor_list
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
    compute_naive_num_shifts,
)

from ...test_utils import create_simple_cubic_system

_PAIR_PARAMS_REQUIRED = "pair_params is required when pair_fn is provided"


def _missing_pair_params_cpu_fixtures():
    """Small CPU tensors for missing-``pair_params`` validation tests."""
    positions = _two_cluster_positions("cpu")
    cell = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3) * 3.0
    pbc = torch.tensor([[True, True, True]], dtype=torch.bool)
    batch_idx = torch.zeros(positions.shape[0], dtype=torch.int32)
    batch_ptr = torch.tensor([0, positions.shape[0]], dtype=torch.int32)
    return positions, cell, pbc, batch_idx, batch_ptr


def _alloc_query_neighbor_buffers(n_atoms: int, max_neighbors: int, device: str):
    """Allocate minimal neighbor-matrix buffers for query-wrapper tests."""
    neighbor_matrix = torch.full(
        (n_atoms, max_neighbors),
        n_atoms,
        dtype=torch.int32,
        device=device,
    )
    neighbor_matrix_shifts = torch.zeros(
        (n_atoms, max_neighbors, 3),
        dtype=torch.int32,
        device=device,
    )
    num_neighbors = torch.zeros((n_atoms,), dtype=torch.int32, device=device)
    return neighbor_matrix, neighbor_matrix_shifts, num_neighbors


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


def _alloc_target_pair_buffers(
    n_atoms: int,
    n_rows: int,
    max_neighbors: int,
    device: str,
):
    """Allocate compact-row outputs and full per-atom pair parameters."""
    nm = torch.full((n_rows, max_neighbors), -1, dtype=torch.int32, device=device)
    nms = torch.zeros((n_rows, max_neighbors, 3), dtype=torch.int32, device=device)
    nn = torch.zeros((n_rows,), dtype=torch.int32, device=device)
    nv = torch.zeros((n_rows, max_neighbors, 3), dtype=torch.float32, device=device)
    nd = torch.zeros((n_rows, max_neighbors), dtype=torch.float32, device=device)
    pe = torch.zeros((n_rows, max_neighbors), dtype=torch.float32, device=device)
    pf = torch.zeros((n_rows, max_neighbors, 3), dtype=torch.float32, device=device)
    pp = (
        (torch.arange(n_atoms, dtype=torch.float32, device=device) + 1.0) * 0.5
    ).reshape(n_atoms, 1)
    return nm, nms, nn, nv, nd, pe, pf, pp


def _skip_without_cuda(device: str) -> None:
    """Skip a test device parameter unless it is a CUDA device."""
    if not str(device).startswith("cuda"):
        pytest.skip("CUDA is required for torch.compile fullgraph Warp check")


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


def _check_pair_outputs_from_geometry(positions, cell, nm, nn, shifts, pe, pf, pp):
    """Verify pair outputs without returned distance/vector buffers."""
    positions = positions.detach().cpu()
    cell = cell.detach().cpu()
    nm = nm.cpu()
    nn = nn.cpu()
    shifts = shifts.cpu()
    pe = pe.cpu()
    pf = pf.cpu()
    pp = pp.cpu()
    checked = 0
    for i in range(nm.shape[0]):
        for slot in range(int(nn[i])):
            j = int(nm[i, slot])
            assert 0 <= j < positions.shape[0]
            system = 0 if cell.shape[0] == 1 else i
            dr = (
                positions[j]
                - positions[i]
                + shifts[i, slot].to(positions.dtype) @ cell[system]
            )
            distance = dr.norm()
            expected_energy = float(pp[i, 0]) + float(pp[j, 0]) + float(distance)
            assert pe[i, slot] == pytest.approx(expected_energy, rel=1e-5, abs=1e-5)
            assert torch.allclose(pf[i, slot], -dr, rtol=1e-5, atol=1e-5)
            checked += 1
    assert checked > 0, "no neighbor pairs were found; test exercised nothing"


def _check_target_pair_outputs(nm, nn, nv, nd, pe, pf, pp, target_indices):
    """Verify pair outputs when compact rows map through target_indices."""
    nm = nm.cpu()
    nn = nn.cpu()
    nv = nv.cpu()
    nd = nd.cpu()
    pe = pe.cpu()
    pf = pf.cpu()
    pp = pp.cpu()
    target_indices = target_indices.cpu()
    checked = 0
    for row in range(nm.shape[0]):
        atom_i = int(target_indices[row])
        for slot in range(int(nn[row])):
            atom_j = int(nm[row, slot])
            assert 0 <= atom_j < pp.shape[0]
            expected_energy = (
                float(pp[atom_i, 0]) + float(pp[atom_j, 0]) + float(nd[row, slot])
            )
            assert pe[row, slot] == pytest.approx(expected_energy, rel=1e-5, abs=1e-5)
            assert torch.allclose(pf[row, slot], -nv[row, slot], rtol=1e-5, atol=1e-5)
            checked += 1
    assert checked > 0, "no neighbor pairs were found; test exercised nothing"


def _compiled_pair_fn(name: str):
    """Create a pre-specialized pair function with a unique op-name prefix."""
    return compile_pair_fn(_sum_pair_fn, name=name)


def _two_cluster_positions(device: str):
    """Return positions with two obvious neighbor pairs."""
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def _single_system_pbc(device: str):
    """Small periodic system whose shape metadata can be precomputed."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.45, 0.0, 0.0],
            [2.8, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    cell = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3) * 3.0
    pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
    return positions, cell, pbc


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


def test_cell_list_pair_fn_without_geometry_outputs(device):
    """``pair_fn`` works without public distance/vector buffers."""
    dtype = torch.float32
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=dtype, device=device
    )
    pbc = pbc.reshape(3)
    pp = ((torch.arange(8, dtype=dtype, device=device) + 1.0) * 0.5).reshape(8, 1)

    nm, nn, shifts, pe, pf = cell_list(
        positions,
        1.1,
        cell,
        pbc,
        max_neighbors=16,
        pair_fn=_sum_pair_fn,
        pair_params=pp,
    )

    _check_pair_outputs_from_geometry(positions, cell, nm, nn, shifts, pe, pf, pp)


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


def test_naive_pair_fn_target_indices_compact_rows(device):
    """Torch naive ``target_indices + pair_fn`` uses compact source rows."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    target_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    pp = ((torch.arange(4, dtype=torch.float32, device=device) + 1.0) * 0.5).reshape(
        4, 1
    )

    nm, nn, nd, nv, pe, pf = naive_neighbor_list(
        positions,
        0.75,
        max_neighbors=4,
        target_indices=target_indices,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn,
        pair_params=pp,
    )

    assert pe.shape == (2, 4)
    _check_target_pair_outputs(nm, nn, nv, nd, pe, pf, pp, target_indices)


def test_naive_pair_fn_target_indices_fullgraph_rejected(device):
    """Torch fullgraph rejects eager-only Python ``pair_fn`` routes clearly."""
    _skip_without_cuda(device)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    target_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    pp = ((torch.arange(4, dtype=torch.float32, device=device) + 1.0) * 0.5).reshape(
        4, 1
    )
    nm = torch.full((2, 4), 4, dtype=torch.int32, device=device)
    nn = torch.zeros((2,), dtype=torch.int32, device=device)
    nd = torch.zeros((2, 4), dtype=torch.float32, device=device)
    nv = torch.zeros((2, 4, 3), dtype=torch.float32, device=device)
    pe = torch.zeros((2, 4), dtype=torch.float32, device=device)
    pf = torch.zeros((2, 4, 3), dtype=torch.float32, device=device)

    @torch.compile(fullgraph=True)
    def run(positions, nm, nn, nd, nv, pe, pf):
        return naive_neighbor_list(
            positions,
            0.75,
            max_neighbors=4,
            neighbor_matrix=nm,
            num_neighbors=nn,
            target_indices=target_indices,
            return_distances=True,
            return_vectors=True,
            neighbor_distances=nd,
            neighbor_vectors=nv,
            pair_fn=_sum_pair_fn,
            pair_params=pp,
            pair_energies=pe,
            pair_forces=pf,
        )

    with pytest.raises(Exception, match="eager-only"):
        run(positions, nm, nn, nd, nv, pe, pf)


def test_compiled_pair_fn_naive_fullgraph_matrix(device):
    """Pre-specialized ``pair_fn`` works under fullgraph for naive matrix output."""
    _skip_without_cuda(device)
    positions = _two_cluster_positions(device)
    max_neighbors = 4
    cpf = _compiled_pair_fn("naive_fullgraph")
    nm, _nms, nn, nv, nd, pe, pf, pp = _alloc_pair_buffers(
        positions.shape[0], max_neighbors, device
    )

    @torch.compile(fullgraph=True)
    def run(positions, nm, nn, nv, nd, pe, pf):
        return naive_neighbor_list(
            positions,
            0.75,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm,
            num_neighbors=nn,
            return_distances=True,
            return_vectors=True,
            neighbor_vectors=nv,
            neighbor_distances=nd,
            pair_fn=cpf,
            pair_params=pp,
            pair_energies=pe,
            pair_forces=pf,
        )

    nm_out, nn_out, nd_out, nv_out, pe_out, pf_out = run(
        positions, nm, nn, nv, nd, pe, pf
    )
    assert nm_out is nm
    _check_pair_outputs(nm_out, nn_out, nv_out, nd_out, pe_out, pf_out, pp)


def test_compiled_pair_fn_naive_target_indices_fullgraph_matrix(device):
    """Compiled naive fullgraph preserves compact target rows."""
    _skip_without_cuda(device)
    positions = _two_cluster_positions(device)
    target_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    max_neighbors = 4
    cpf = _compiled_pair_fn("naive_target_fullgraph")
    nm, _nms, nn, nv, nd, pe, pf, pp = _alloc_pair_buffers(
        target_indices.shape[0], max_neighbors, device
    )
    pp = ((torch.arange(4, dtype=torch.float32, device=device) + 1.0) * 0.5).reshape(
        4, 1
    )

    @torch.compile(fullgraph=True)
    def run(positions, nm, nn, nv, nd, pe, pf):
        return naive_neighbor_list(
            positions,
            0.75,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm,
            num_neighbors=nn,
            target_indices=target_indices,
            return_distances=True,
            return_vectors=True,
            neighbor_vectors=nv,
            neighbor_distances=nd,
            pair_fn=cpf,
            pair_params=pp,
            pair_energies=pe,
            pair_forces=pf,
        )

    nm_out, nn_out, nd_out, nv_out, pe_out, pf_out = run(
        positions, nm, nn, nv, nd, pe, pf
    )
    _check_target_pair_outputs(
        nm_out, nn_out, nv_out, nd_out, pe_out, pf_out, pp, target_indices
    )


def test_compiled_pair_fn_naive_pbc_fullgraph_matrix(device):
    """Compiled naive fullgraph supports precomputed PBC shift metadata."""
    _skip_without_cuda(device)
    positions, cell, pbc = _single_system_pbc(device)
    cutoff = 0.75
    max_neighbors = 8
    cpf = _compiled_pair_fn("naive_pbc_fullgraph")
    nm, nms, nn, nv, nd, pe, pf, pp = _alloc_pair_buffers(
        positions.shape[0], max_neighbors, device
    )
    shift_range, num_shifts, max_shifts = compute_naive_num_shifts(cell, cutoff, pbc)

    @torch.compile(fullgraph=True)
    def run(positions, nm, nms, nn, nv, nd, pe, pf):
        return naive_neighbor_list(
            positions,
            cutoff,
            cell,
            pbc.reshape(3),
            max_neighbors=max_neighbors,
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            num_neighbors=nn,
            shift_range_per_dimension=shift_range,
            num_shifts_per_system=num_shifts,
            max_shifts_per_system=max_shifts,
            return_distances=True,
            return_vectors=True,
            neighbor_vectors=nv,
            neighbor_distances=nd,
            pair_fn=cpf,
            pair_params=pp,
            pair_energies=pe,
            pair_forces=pf,
        )

    nm_out, nn_out, _shifts, nd_out, nv_out, pe_out, pf_out = run(
        positions, nm, nms, nn, nv, nd, pe, pf
    )
    _check_pair_outputs(nm_out, nn_out, nv_out, nd_out, pe_out, pf_out, pp)


def test_compiled_pair_fn_batch_naive_fullgraph_matrix(device):
    """Pre-specialized ``pair_fn`` works for batch naive fullgraph matrix output."""
    _skip_without_cuda(device)
    positions = torch.cat(
        [
            _two_cluster_positions(device),
            torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32, device=device),
        ],
        dim=0,
    )
    batch_idx = torch.tensor([0, 0, 0, 0, 1], dtype=torch.int32, device=device)
    batch_ptr = torch.tensor([0, 4, 5], dtype=torch.int32, device=device)
    max_neighbors = 4
    cpf = _compiled_pair_fn("batch_naive_fullgraph")
    nm, _nms, nn, nv, nd, pe, pf, pp = _alloc_pair_buffers(
        positions.shape[0], max_neighbors, device
    )

    @torch.compile(fullgraph=True)
    def run(positions, nm, nn, nv, nd, pe, pf):
        return batch_naive_neighbor_list(
            positions,
            0.75,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm,
            num_neighbors=nn,
            return_distances=True,
            return_vectors=True,
            neighbor_vectors=nv,
            neighbor_distances=nd,
            pair_fn=cpf,
            pair_params=pp,
            pair_energies=pe,
            pair_forces=pf,
        )

    nm_out, nn_out, nd_out, nv_out, pe_out, pf_out = run(
        positions, nm, nn, nv, nd, pe, pf
    )
    _check_pair_outputs(
        nm_out[:4],
        nn_out[:4],
        nv_out[:4],
        nd_out[:4],
        pe_out[:4],
        pf_out[:4],
        pp[:4],
    )


def test_compiled_pair_fn_batch_naive_target_indices_fullgraph_matrix(device):
    """Compiled batch naive fullgraph preserves compact target rows."""
    _skip_without_cuda(device)
    positions = torch.cat(
        [
            _two_cluster_positions(device),
            torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32, device=device),
        ],
        dim=0,
    )
    batch_idx = torch.tensor([0, 0, 0, 0, 1], dtype=torch.int32, device=device)
    batch_ptr = torch.tensor([0, 4, 5], dtype=torch.int32, device=device)
    target_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    max_neighbors = 4
    cpf = _compiled_pair_fn("batch_naive_target_fullgraph")
    nm, _nms, nn, nv, nd, pe, pf, _pp = _alloc_pair_buffers(
        target_indices.shape[0], max_neighbors, device
    )
    pp = ((torch.arange(5, dtype=torch.float32, device=device) + 1.0) * 0.5).reshape(
        5, 1
    )

    @torch.compile(fullgraph=True)
    def run(positions, nm, nn, nv, nd, pe, pf):
        return batch_naive_neighbor_list(
            positions,
            0.75,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm,
            num_neighbors=nn,
            target_indices=target_indices,
            return_distances=True,
            return_vectors=True,
            neighbor_vectors=nv,
            neighbor_distances=nd,
            pair_fn=cpf,
            pair_params=pp,
            pair_energies=pe,
            pair_forces=pf,
        )

    nm_out, nn_out, nd_out, nv_out, pe_out, pf_out = run(
        positions, nm, nn, nv, nd, pe, pf
    )
    _check_target_pair_outputs(
        nm_out, nn_out, nv_out, nd_out, pe_out, pf_out, pp, target_indices
    )


def test_compiled_pair_fn_batch_naive_pbc_fullgraph_matrix(device):
    """Compiled batch naive fullgraph supports precomputed PBC metadata."""
    _skip_without_cuda(device)
    positions, cell, pbc = _single_system_pbc(device)
    batch_idx = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)
    batch_ptr = torch.tensor([0, positions.shape[0]], dtype=torch.int32, device=device)
    cutoff = 0.75
    max_neighbors = 8
    cpf = _compiled_pair_fn("batch_naive_pbc_fullgraph")
    nm, nms, nn, nv, nd, pe, pf, pp = _alloc_pair_buffers(
        positions.shape[0], max_neighbors, device
    )
    shift_range, num_shifts, max_shifts = compute_naive_num_shifts(cell, cutoff, pbc)

    @torch.compile(fullgraph=True)
    def run(positions, nm, nms, nn, nv, nd, pe, pf):
        return batch_naive_neighbor_list(
            positions,
            cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cell=cell,
            pbc=pbc,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            num_neighbors=nn,
            shift_range_per_dimension=shift_range,
            num_shifts_per_system=num_shifts,
            max_shifts_per_system=max_shifts,
            max_atoms_per_system=positions.shape[0],
            return_distances=True,
            return_vectors=True,
            neighbor_vectors=nv,
            neighbor_distances=nd,
            pair_fn=cpf,
            pair_params=pp,
            pair_energies=pe,
            pair_forces=pf,
        )

    nm_out, nn_out, _shifts, nd_out, nv_out, pe_out, pf_out = run(
        positions, nm, nms, nn, nv, nd, pe, pf
    )
    _check_pair_outputs(nm_out, nn_out, nv_out, nd_out, pe_out, pf_out, pp)


def test_compiled_pair_fn_cell_list_fullgraph_matrix(device):
    """Compiled ``pair_fn`` works under fullgraph for cell-list matrix output."""
    _skip_without_cuda(device)
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=torch.float32, device=device
    )
    pbc = pbc.reshape(3)
    cutoff = 1.1
    max_neighbors = 16
    cpf = _compiled_pair_fn("cell_list_fullgraph")
    nm, nms, nn, nv, nd, pe, pf, pp = _alloc_pair_buffers(8, max_neighbors, device)
    max_cells, radius = estimate_cell_list_sizes(
        cell,
        pbc,
        cutoff,
        min_cells_per_dimension=4,
    )
    (
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    ) = allocate_cell_list(positions.shape[0], max_cells, radius, torch.device(device))
    sorted_positions, sorted_shifts = allocate_query_sort_scratch(
        positions.shape[0], dtype=positions.dtype, device=device
    )

    @torch.compile(fullgraph=True)
    def run(positions, nm, nms, nn, nv, nd, pe, pf):
        return cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            num_neighbors=nn,
            cells_per_dimension=cells_per_dimension,
            neighbor_search_radius=neighbor_search_radius,
            atom_periodic_shifts=atom_periodic_shifts,
            atom_to_cell_mapping=atom_to_cell_mapping,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            strategy="atom_centric",
            atom_centric_path="direct",
            sorted_positions=sorted_positions,
            sorted_shifts=sorted_shifts,
            return_distances=True,
            return_vectors=True,
            neighbor_vectors=nv,
            neighbor_distances=nd,
            pair_fn=cpf,
            pair_params=pp,
            pair_energies=pe,
            pair_forces=pf,
        )

    nm_out, nn_out, _shifts, nd_out, nv_out, pe_out, pf_out = run(
        positions, nm, nms, nn, nv, nd, pe, pf
    )
    _check_pair_outputs(nm_out, nn_out, nv_out, nd_out, pe_out, pf_out, pp)


def test_compiled_pair_fn_cell_list_target_indices_fullgraph_matrix(device):
    """Compiled cell-list fullgraph preserves compact target rows."""
    _skip_without_cuda(device)
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=torch.float32, device=device
    )
    pbc = pbc.reshape(3)
    target_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    cutoff = 1.1
    max_neighbors = 16
    cpf = _compiled_pair_fn("cell_list_target_fullgraph")
    nm, nms, nn, nv, nd, pe, pf, pp = _alloc_target_pair_buffers(
        positions.shape[0], target_indices.shape[0], max_neighbors, device
    )
    max_cells, radius = estimate_cell_list_sizes(
        cell,
        pbc,
        cutoff,
        min_cells_per_dimension=4,
    )
    (
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    ) = allocate_cell_list(positions.shape[0], max_cells, radius, torch.device(device))
    sorted_positions, sorted_shifts = allocate_query_sort_scratch(
        positions.shape[0], dtype=positions.dtype, device=device
    )

    @torch.compile(fullgraph=True)
    def run(positions, nm, nms, nn, nv, nd, pe, pf):
        return cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            num_neighbors=nn,
            cells_per_dimension=cells_per_dimension,
            neighbor_search_radius=neighbor_search_radius,
            atom_periodic_shifts=atom_periodic_shifts,
            atom_to_cell_mapping=atom_to_cell_mapping,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            strategy="atom_centric",
            atom_centric_path="direct",
            sorted_positions=sorted_positions,
            sorted_shifts=sorted_shifts,
            target_indices=target_indices,
            return_distances=True,
            return_vectors=True,
            neighbor_vectors=nv,
            neighbor_distances=nd,
            pair_fn=cpf,
            pair_params=pp,
            pair_energies=pe,
            pair_forces=pf,
        )

    nm_out, nn_out, _shifts, nd_out, nv_out, pe_out, pf_out = run(
        positions, nm, nms, nn, nv, nd, pe, pf
    )
    assert nm_out.shape[0] == int(target_indices.shape[0])
    _check_target_pair_outputs(
        nm_out, nn_out, nv_out, nd_out, pe_out, pf_out, pp, target_indices
    )


def test_compiled_pair_fn_batch_cell_list_fullgraph_matrix(device):
    """Compiled ``pair_fn`` works under fullgraph for batch cell-list matrix output."""
    _skip_without_cuda(device)
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=torch.float32, device=device
    )
    pbc = pbc.reshape(1, 3)
    batch_idx = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)
    cutoff = 1.1
    max_neighbors = 16
    cpf = _compiled_pair_fn("batch_cell_list_fullgraph")
    nm, nms, nn, nv, nd, pe, pf, pp = _alloc_pair_buffers(8, max_neighbors, device)
    max_cells, radius = estimate_batch_cell_list_sizes(
        cell,
        pbc,
        cutoff,
        min_cells_per_dimension=4,
    )
    (
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    ) = allocate_cell_list(positions.shape[0], max_cells, radius, torch.device(device))

    @torch.compile(fullgraph=True)
    def run(positions, nm, nms, nn, nv, nd, pe, pf):
        return batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            num_neighbors=nn,
            cells_per_dimension=cells_per_dimension,
            neighbor_search_radius=neighbor_search_radius,
            atom_periodic_shifts=atom_periodic_shifts,
            atom_to_cell_mapping=atom_to_cell_mapping,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            strategy="atom_centric",
            atom_centric_path="direct",
            return_distances=True,
            return_vectors=True,
            neighbor_vectors=nv,
            neighbor_distances=nd,
            pair_fn=cpf,
            pair_params=pp,
            pair_energies=pe,
            pair_forces=pf,
        )

    nm_out, nn_out, _shifts, nd_out, nv_out, pe_out, pf_out = run(
        positions, nm, nms, nn, nv, nd, pe, pf
    )
    _check_pair_outputs(nm_out, nn_out, nv_out, nd_out, pe_out, pf_out, pp)


def test_compiled_pair_fn_batch_cell_list_target_indices_fullgraph_matrix(device):
    """Compiled batch cell-list fullgraph preserves compact target rows."""
    _skip_without_cuda(device)
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=torch.float32, device=device
    )
    cell = cell.unsqueeze(0)
    pbc = pbc.reshape(1, 3)
    batch_idx = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)
    target_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    cutoff = 1.1
    max_neighbors = 16
    cpf = _compiled_pair_fn("batch_cell_list_target_fullgraph")
    nm, nms, nn, nv, nd, pe, pf, pp = _alloc_target_pair_buffers(
        positions.shape[0], target_indices.shape[0], max_neighbors, device
    )
    max_cells, radius = estimate_batch_cell_list_sizes(
        cell,
        pbc,
        cutoff,
        min_cells_per_dimension=4,
    )
    (
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    ) = allocate_cell_list(positions.shape[0], max_cells, radius, torch.device(device))

    @torch.compile(fullgraph=True)
    def run(positions, nm, nms, nn, nv, nd, pe, pf):
        return batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            num_neighbors=nn,
            cells_per_dimension=cells_per_dimension,
            neighbor_search_radius=neighbor_search_radius,
            atom_periodic_shifts=atom_periodic_shifts,
            atom_to_cell_mapping=atom_to_cell_mapping,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            strategy="atom_centric",
            atom_centric_path="direct",
            target_indices=target_indices,
            return_distances=True,
            return_vectors=True,
            neighbor_vectors=nv,
            neighbor_distances=nd,
            pair_fn=cpf,
            pair_params=pp,
            pair_energies=pe,
            pair_forces=pf,
        )

    nm_out, nn_out, _shifts, nd_out, nv_out, pe_out, pf_out = run(
        positions, nm, nms, nn, nv, nd, pe, pf
    )
    assert nm_out.shape[0] == int(target_indices.shape[0])
    _check_target_pair_outputs(
        nm_out, nn_out, nv_out, nd_out, pe_out, pf_out, pp, target_indices
    )


def test_compiled_pair_fn_fullgraph_rejects_coo_output(device):
    """Compiled pair functions are matrix-only under fullgraph."""
    _skip_without_cuda(device)
    positions = _two_cluster_positions(device)
    max_neighbors = 4
    cpf = _compiled_pair_fn("coo_rejected")
    nm, _nms, nn, nv, nd, pe, pf, pp = _alloc_pair_buffers(
        positions.shape[0], max_neighbors, device
    )

    @torch.compile(fullgraph=True)
    def run(positions, nm, nn, nv, nd, pe, pf):
        return naive_neighbor_list(
            positions,
            0.75,
            max_neighbors=max_neighbors,
            return_neighbor_list=True,
            neighbor_matrix=nm,
            num_neighbors=nn,
            return_distances=True,
            return_vectors=True,
            neighbor_vectors=nv,
            neighbor_distances=nd,
            pair_fn=cpf,
            pair_params=pp,
            pair_energies=pe,
            pair_forces=pf,
        )

    with pytest.raises(Exception, match="matrix neighbor-list output only"):
        run(positions, nm, nn, nv, nd, pe, pf)


def test_compiled_pair_fn_fullgraph_requires_fixed_buffers(device):
    """Compiled pair functions fail early when fixed buffers are omitted."""
    _skip_without_cuda(device)
    positions = _two_cluster_positions(device)
    cpf = _compiled_pair_fn("missing_buffers")
    pp = ((torch.arange(4, dtype=torch.float32, device=device) + 1.0) * 0.5).reshape(
        4, 1
    )

    @torch.compile(fullgraph=True)
    def run(positions):
        return naive_neighbor_list(
            positions,
            0.75,
            max_neighbors=4,
            return_distances=True,
            return_vectors=True,
            pair_fn=cpf,
            pair_params=pp,
        )

    with pytest.raises(Exception, match="fixed-shape caller-provided buffers"):
        run(positions)


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


def test_batch_cell_list_pair_fn_without_geometry_outputs(device):
    """Batched ``pair_fn`` works without public geometry buffers."""
    dtype = torch.float32
    positions, cell, pbc = create_simple_cubic_system(
        num_atoms=8, cell_size=2.0, dtype=dtype, device=device
    )
    pbc = pbc.reshape(1, 3)
    batch_idx = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)
    pp = ((torch.arange(8, dtype=dtype, device=device) + 1.0) * 0.5).reshape(8, 1)

    nm, nn, shifts, pe, pf = batch_cell_list(
        positions,
        1.1,
        cell,
        pbc,
        batch_idx,
        max_neighbors=16,
        pair_fn=_sum_pair_fn,
        pair_params=pp,
    )

    _check_pair_outputs_from_geometry(positions, cell, nm, nn, shifts, pe, pf, pp)


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


def test_batch_naive_pair_fn_target_indices_compact_rows(device):
    """Torch batch_naive ``target_indices + pair_fn`` uses compact source rows."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    batch_idx = torch.tensor([0, 0, 0, 0, 1], dtype=torch.int32, device=device)
    batch_ptr = torch.tensor([0, 4, 5], dtype=torch.int32, device=device)
    target_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    pp = ((torch.arange(5, dtype=torch.float32, device=device) + 1.0) * 0.5).reshape(
        5, 1
    )

    nm, nn, nd, nv, pe, pf = batch_naive_neighbor_list(
        positions,
        0.75,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        max_neighbors=4,
        target_indices=target_indices,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn,
        pair_params=pp,
    )

    assert pe.shape == (2, 4)
    assert pf.shape == (2, 4, 3)
    _check_target_pair_outputs(nm, nn, nv, nd, pe, pf, pp, target_indices)


def test_compiled_naive_missing_pair_params_raises():
    """Compiled naive rejects omitted ``pair_params`` before custom-op dispatch."""
    positions, _cell, _pbc, _batch_idx, _batch_ptr = _missing_pair_params_cpu_fixtures()
    cpf = _compiled_pair_fn("missing_pair_params_naive")
    with pytest.raises(ValueError, match=_PAIR_PARAMS_REQUIRED):
        naive_neighbor_list(positions, 0.75, max_neighbors=4, pair_fn=cpf)


def test_compiled_batch_naive_missing_pair_params_raises():
    """Compiled batch naive rejects omitted ``pair_params`` before dispatch."""
    positions, _cell, _pbc, batch_idx, batch_ptr = _missing_pair_params_cpu_fixtures()
    cpf = _compiled_pair_fn("missing_pair_params_batch_naive")
    with pytest.raises(ValueError, match=_PAIR_PARAMS_REQUIRED):
        batch_naive_neighbor_list(
            positions,
            0.75,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=4,
            pair_fn=cpf,
        )


def test_compiled_cell_list_missing_pair_params_raises():
    """Compiled cell list rejects omitted ``pair_params`` before dispatch."""
    positions, cell, pbc, _batch_idx, _batch_ptr = _missing_pair_params_cpu_fixtures()
    cpf = _compiled_pair_fn("missing_pair_params_cell_list")
    with pytest.raises(ValueError, match=_PAIR_PARAMS_REQUIRED):
        cell_list(
            positions,
            0.75,
            cell,
            pbc.reshape(3),
            max_neighbors=4,
            pair_fn=cpf,
        )


def test_compiled_batch_cell_list_missing_pair_params_raises():
    """Compiled batch cell list rejects omitted ``pair_params`` before dispatch."""
    positions, cell, pbc, batch_idx, _batch_ptr = _missing_pair_params_cpu_fixtures()
    cpf = _compiled_pair_fn("missing_pair_params_batch_cell_list")
    with pytest.raises(ValueError, match=_PAIR_PARAMS_REQUIRED):
        batch_cell_list(
            positions,
            0.75,
            cell,
            pbc,
            batch_idx,
            max_neighbors=4,
            pair_fn=cpf,
        )


def test_compiled_query_cell_list_missing_pair_params_raises():
    """Compiled query wrapper rejects omitted ``pair_params`` before dispatch."""
    positions, cell, pbc, _batch_idx, _batch_ptr = _missing_pair_params_cpu_fixtures()
    max_neighbors = 4
    max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(
        cell,
        pbc.reshape(3),
        0.75,
    )
    cell_list_cache = allocate_cell_list(
        positions.shape[0],
        max_total_cells,
        neighbor_search_radius,
        positions.device,
    )
    neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
        _alloc_query_neighbor_buffers(
            positions.shape[0],
            max_neighbors,
            positions.device,
        )
    )
    cpf = _compiled_pair_fn("missing_pair_params_query_cell_list")
    with pytest.raises(ValueError, match=_PAIR_PARAMS_REQUIRED):
        query_cell_list(
            positions,
            0.75,
            cell,
            pbc.reshape(3),
            *cell_list_cache,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            pair_fn=cpf,
        )


def test_compiled_batch_query_cell_list_missing_pair_params_raises():
    """Compiled batch query wrapper rejects omitted ``pair_params`` before dispatch."""
    positions, cell, pbc, batch_idx, _batch_ptr = _missing_pair_params_cpu_fixtures()
    max_neighbors = 4
    max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
        cell,
        pbc,
        0.75,
    )
    cell_list_cache = allocate_cell_list(
        positions.shape[0],
        max_total_cells,
        neighbor_search_radius,
        positions.device,
    )
    neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
        _alloc_query_neighbor_buffers(
            positions.shape[0],
            max_neighbors,
            positions.device,
        )
    )
    cpf = _compiled_pair_fn("missing_pair_params_batch_query_cell_list")
    with pytest.raises(ValueError, match=_PAIR_PARAMS_REQUIRED):
        batch_query_cell_list(
            positions,
            cell,
            pbc,
            0.75,
            batch_idx,
            *cell_list_cache,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            pair_fn=cpf,
        )


def test_naive_pair_fn_missing_pair_params_raises():
    """Raw ``pair_fn`` also rejects omitted ``pair_params`` at the torch wrapper."""
    positions, _cell, _pbc, _batch_idx, _batch_ptr = _missing_pair_params_cpu_fixtures()
    with pytest.raises(ValueError, match=_PAIR_PARAMS_REQUIRED):
        naive_neighbor_list(
            positions,
            0.75,
            max_neighbors=4,
            pair_fn=_sum_pair_fn,
        )


def test_compiled_naive_fullgraph_missing_pair_params_diagnostic(device):
    """Fullgraph keeps the specific missing-buffer diagnostic for ``pair_params``."""
    _skip_without_cuda(device)
    positions = _two_cluster_positions(device)
    max_neighbors = 4
    cpf = _compiled_pair_fn("missing_pair_params_naive_fullgraph")
    nm, _nms, nn, nv, nd, pe, pf, _pp = _alloc_pair_buffers(
        positions.shape[0], max_neighbors, device
    )

    @torch.compile(fullgraph=True)
    def run(positions, nm, nn, nv, nd, pe, pf):
        return naive_neighbor_list(
            positions,
            0.75,
            max_neighbors=max_neighbors,
            neighbor_matrix=nm,
            num_neighbors=nn,
            return_distances=True,
            return_vectors=True,
            neighbor_vectors=nv,
            neighbor_distances=nd,
            pair_fn=cpf,
            pair_energies=pe,
            pair_forces=pf,
        )

    with pytest.raises(
        Exception,
        match=r"CompiledPairFn under torch\.compile\(fullgraph=True\).*missing .*pair_params",
    ):
        run(positions, nm, nn, nv, nd, pe, pf)
