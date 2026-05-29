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

"""Preallocated / preconverted setup helpers for neighbor-list benchmarks.

Each ``_setup_*`` function takes the problem inputs, allocates all state
and output tensors up front, preconverts them into ``wp.array`` handles,
and returns a ``step()`` closure that runs a single neighbor-list build
using only raw warp launchers.  The timed region therefore contains
zero ``wp.from_torch`` calls, zero allocations, and zero CPU-GPU syncs.

Any per-call integer that the launcher requires (``num_tiles``,
``total_cells``) is derived from a constant-geometry "prime" call during
setup and cached — valid because the benchmark uses static positions.
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.neighbors.batch_cell_list import (
    batch_build_cell_list as wp_batch_build_cell_list,
)
from nvalchemiops.neighbors.batch_cell_list import (
    batch_query_cell_list as wp_batch_query_cell_list,
)
from nvalchemiops.neighbors.batch_naive import (
    batch_naive_neighbor_matrix as wp_batch_naive_tiled,
)
from nvalchemiops.neighbors.batch_naive import (
    batch_naive_neighbor_matrix_pbc as wp_batch_naive_pbc_tiled,
)
from nvalchemiops.neighbors.cell_list import (
    build_cell_list as wp_build_cell_list,
)
from nvalchemiops.neighbors.cell_list import (
    compute_batch_pair_centric_n_outer,
    get_cell_list_gather_kernel,
    query_cell_list_pair_centric_sorted,
)
from nvalchemiops.neighbors.cell_list import (
    query_cell_list as wp_query_cell_list,
)

# Cluster-tile warp launchers (renamed from the pre-refactor ``tile_warp`` /
# ``tile_batch_warp`` modules).  Aliased to the historical benchmark names so
# the timed pipelines below stay close to the original speed-of-light layout.
from nvalchemiops.neighbors.cluster_tile.launchers import (
    _compute_morton as wp_compute_morton,
)
from nvalchemiops.neighbors.cluster_tile.launchers import (
    _permute_gather_soa as wp_permute_gather_soa,
)
from nvalchemiops.neighbors.cluster_tile.launchers import (
    _tile_sort_pairs as wp_tile_sort_pairs,
)
from nvalchemiops.neighbors.cluster_tile.launchers import (
    batch_build_cluster_tile_list as wp_build_batch_tile_neighbor_list,
)
from nvalchemiops.neighbors.cluster_tile.launchers import (
    batch_query_cluster_tile as wp_batch_tile_to_matrix,
)
from nvalchemiops.neighbors.cluster_tile.launchers import (
    build_cluster_tile_list as wp_build_tile_neighbor_list,
)
from nvalchemiops.neighbors.cluster_tile.launchers import (
    query_cluster_tile as wp_tile_to_matrix,
)
from nvalchemiops.neighbors.naive import (
    naive_neighbor_matrix as wp_naive_tiled,
)
from nvalchemiops.neighbors.naive import (
    naive_neighbor_matrix_pbc as wp_naive_pbc_tiled,
)
from nvalchemiops.neighbors.neighbor_utils import (
    fill_neighbor_matrix_tail as wp_fill_neighbor_matrix_tail,
)
from nvalchemiops.torch.neighbors.batch_cell_list import (
    estimate_batch_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.batch_cluster_tile import (
    _batched_morton_sort_padded,
)
from nvalchemiops.torch.neighbors.batch_cluster_tile import (
    allocate_batch_cluster_tile_list as allocate_batch_tile_neighbor_list,
)
from nvalchemiops.torch.neighbors.cell_list import (
    estimate_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.cluster_tile import (
    _cell_invcell_from_cell,
    _mat33f_from_torch,
)
from nvalchemiops.torch.neighbors.cluster_tile import (
    allocate_cluster_tile_list as allocate_tile_neighbor_list,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
)

TILE_GROUP_SIZE = 32


@wp.kernel(enable_backward=False)
def _zero_int32_kernel(arr: wp.array(dtype=wp.int32), n: wp.int32):
    """Coalesced zero of ``arr[:n]``.  One thread per element."""
    i = wp.tid()
    if i < n:
        arr[i] = 0


def _pad_positions_soa(positions, device):
    """Pad (N, 3) positions to (ngroup*TILE_GROUP_SIZE,) SoA x/y/z."""
    n = positions.shape[0]
    ngroup = (n + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    n_padded = ngroup * TILE_GROUP_SIZE
    pos_x = torch.zeros(n_padded, dtype=torch.float32, device=device)
    pos_y = torch.zeros(n_padded, dtype=torch.float32, device=device)
    pos_z = torch.zeros(n_padded, dtype=torch.float32, device=device)
    pos_x[:n].copy_(positions[:, 0])
    pos_y[:n].copy_(positions[:, 1])
    pos_z[:n].copy_(positions[:, 2])
    return pos_x, pos_y, pos_z, n, ngroup


def _setup_cluster_tile_single(positions, cutoff, cell, n, max_nb, device):
    """Production cluster-tile pipeline (no graph capture).

    Pipeline (per step):
      1. ``compute_morton``         — Morton codes + sorted_atom_index=i + nn.zero +
         num_tiles[0]=0 in one fused launch.
      2. ``wp.utils.radix_sort_pairs`` — CUB radix sort on (morton, sorted_atom_index).
      3. ``permute_gather_soa``     — fused gather into SoA tiles.
      4. ``build_tile_neighbor_list`` — rank2group bbox + group2tile.
      5. ``tile_to_matrix``         — conversion (always-write shifts).
      6. ``fill_neighbor_matrix_tail`` — coalesced sentinel fill of unused
         columns.  Lets the caller skip ``nm.fill_(n)`` + ``nms.zero_()``.
    """
    wp_device = str(device)
    cell_mat, inv_cell_mat = _cell_invcell_from_cell(cell)
    cell_mat = cell_mat.to(torch.float32).contiguous()
    inv_cell_mat = inv_cell_mat.to(torch.float32).contiguous()
    wp_cell = _mat33f_from_torch(cell_mat)
    wp_inv_cell = _mat33f_from_torch(inv_cell_mat)
    wp_positions_vec3 = wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True)

    # Radix-sort workspaces — wp.utils.radix_sort_pairs needs 2*n keys + values.
    radix_keys = torch.empty(2 * n, dtype=torch.int32, device=device)
    radix_indices = torch.empty(2 * n, dtype=torch.int32, device=device)
    wp_radix_keys = wp.from_torch(radix_keys, dtype=wp.int32)
    wp_radix_indices = wp.from_torch(radix_indices, dtype=wp.int32)

    (
        sorted_atom_index,
        morton_codes,
        spx,
        spy,
        spz,
        gcx,
        gcy,
        gcz,
        gex,
        gey,
        gez,
        num_tiles,
        tile_row_group,
        tile_col_group,
    ) = allocate_tile_neighbor_list(n, device)
    nm = torch.empty((n, max_nb), dtype=torch.int32, device=device)
    nn = torch.empty(n, dtype=torch.int32, device=device)
    nms = torch.empty((n, max_nb, 3), dtype=torch.int32, device=device)

    wp_morton_codes_arr = wp.from_torch(morton_codes, dtype=wp.int32)
    wp_sorted_atom_index_arr = wp.from_torch(sorted_atom_index, dtype=wp.int32)
    wp_sorted_atom_index = wp.from_torch(
        sorted_atom_index, dtype=wp.int32, return_ctype=True
    )
    wp_spx = wp.from_torch(spx, dtype=wp.float32, return_ctype=True)
    wp_spy = wp.from_torch(spy, dtype=wp.float32, return_ctype=True)
    wp_spz = wp.from_torch(spz, dtype=wp.float32, return_ctype=True)
    wp_gcx = wp.from_torch(gcx, dtype=wp.float32, return_ctype=True)
    wp_gcy = wp.from_torch(gcy, dtype=wp.float32, return_ctype=True)
    wp_gcz = wp.from_torch(gcz, dtype=wp.float32, return_ctype=True)
    wp_gex = wp.from_torch(gex, dtype=wp.float32, return_ctype=True)
    wp_gey = wp.from_torch(gey, dtype=wp.float32, return_ctype=True)
    wp_gez = wp.from_torch(gez, dtype=wp.float32, return_ctype=True)
    wp_num_tiles = wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True)
    wp_tile_row_group = wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True)
    wp_tile_col_group = wp.from_torch(tile_col_group, dtype=wp.int32, return_ctype=True)
    wp_nm = wp.from_torch(nm, dtype=wp.int32, return_ctype=True)
    wp_nn = wp.from_torch(nn, dtype=wp.int32, return_ctype=True)
    wp_nms = wp.from_torch(nms, dtype=wp.int32, return_ctype=True)

    def _build():
        wp_compute_morton(
            wp_positions_vec3,
            wp_inv_cell,
            int(n),
            wp_morton_codes_arr,
            wp_sorted_atom_index_arr,
            wp_nn,
            wp_num_tiles,
            wp_device,
        )
        # CUB radix sort needs 2*n-sized buffers; copy morton/sorted_atom_index in.
        radix_keys[:n].copy_(morton_codes)
        radix_indices[:n].copy_(sorted_atom_index)
        wp.utils.radix_sort_pairs(wp_radix_keys, wp_radix_indices, int(n))
        sorted_atom_index.copy_(radix_indices[:n])
        wp_permute_gather_soa(
            wp_positions_vec3,
            wp_sorted_atom_index,
            n,
            wp_spx,
            wp_spy,
            wp_spz,
            wp_device,
        )
        wp_build_tile_neighbor_list(
            sorted_pos_x=wp_spx,
            sorted_pos_y=wp_spy,
            sorted_pos_z=wp_spz,
            cell=wp_cell,
            inv_cell=wp_inv_cell,
            cutoff=float(cutoff),
            num_tiles=wp_num_tiles,
            tile_row_group=wp_tile_row_group,
            tile_col_group=wp_tile_col_group,
            wp_dtype=wp.float32,
            device=wp_device,
            group_ctr_x_buffer=wp_gcx,
            group_ctr_y_buffer=wp_gcy,
            group_ctr_z_buffer=wp_gcz,
            group_ext_x_buffer=wp_gex,
            group_ext_y_buffer=wp_gey,
            group_ext_z_buffer=wp_gez,
        )

    _build()

    def step():
        _build()
        wp_tile_to_matrix(
            sorted_atom_index=wp_sorted_atom_index,
            sorted_pos_x=wp_spx,
            sorted_pos_y=wp_spy,
            sorted_pos_z=wp_spz,
            num_tiles=wp_num_tiles,
            tile_row_group=wp_tile_row_group,
            tile_col_group=wp_tile_col_group,
            cell=wp_cell,
            inv_cell=wp_inv_cell,
            cutoff=float(cutoff),
            natom=int(n),
            neighbor_matrix=wp_nm,
            num_neighbors=wp_nn,
            neighbor_matrix_shifts=wp_nms,
            wp_dtype=wp.float32,
            device=wp_device,
        )
        wp_fill_neighbor_matrix_tail(
            wp_nn,
            int(n),
            int(max_nb),
            int(n),
            wp_nm,
            wp_device,
        )

    return step


def _setup_cluster_tile_graph_single(
    positions,
    cutoff,
    cell,
    n,
    max_nb,
    device,
):
    """Pure-Warp cluster-tile with Warp graph capture.

    Uses owned ``wp.array`` storage for every input + scratch and
    captures ``step()`` into a Warp graph.  The query reads ``num_tiles[0]``
    on-device, so the launch dimension is the allocated tile capacity and no
    host-side tile count is baked into the graph; re-capture is still required
    if the atom count or cutoff changes.  Every ``wp.array`` touched by the
    captured graph must be held by the returned closure (``_keepalive``);
    otherwise replay hits ``cudaErrorIllegalAddress``.
    """
    wp_device_obj = wp.get_device(str(device))
    wp_device = str(device)

    # Owned wp.array (not wp.from_torch) — graph capture is unstable
    # against the torch-allocator-backed view.
    positions_np = positions.contiguous().detach().cpu().numpy()
    positions_wp = wp.array(positions_np, dtype=wp.vec3f, device=wp_device_obj)
    cell_mat_t, inv_cell_mat_t = _cell_invcell_from_cell(cell)
    wp_cell = _mat33f_from_torch(cell_mat_t.to(torch.float32))
    wp_inv_cell = _mat33f_from_torch(inv_cell_mat_t.to(torch.float32))

    # --- All scratch allocated as wp.array — no torch in step(). ---
    # Radix sort needs 2*n keys + 2*n values.
    morton_codes = wp.empty(2 * n, dtype=wp.int32, device=wp_device_obj)
    sorted_atom_index = wp.empty(2 * n, dtype=wp.int32, device=wp_device_obj)
    spx = wp.empty(n, dtype=wp.float32, device=wp_device_obj)
    spy = wp.empty(n, dtype=wp.float32, device=wp_device_obj)
    spz = wp.empty(n, dtype=wp.float32, device=wp_device_obj)
    gcx = wp.empty(n // TILE_GROUP_SIZE, dtype=wp.float32, device=wp_device_obj)
    gcy = wp.empty(n // TILE_GROUP_SIZE, dtype=wp.float32, device=wp_device_obj)
    gcz = wp.empty(n // TILE_GROUP_SIZE, dtype=wp.float32, device=wp_device_obj)
    gex = wp.empty(n // TILE_GROUP_SIZE, dtype=wp.float32, device=wp_device_obj)
    gey = wp.empty(n // TILE_GROUP_SIZE, dtype=wp.float32, device=wp_device_obj)
    gez = wp.empty(n // TILE_GROUP_SIZE, dtype=wp.float32, device=wp_device_obj)
    num_tiles = wp.zeros(1, dtype=wp.int32, device=wp_device_obj)

    nclusters = n // TILE_GROUP_SIZE
    max_tiles = nclusters * (nclusters + 1) // 2
    tile_row_group = wp.empty(max_tiles, dtype=wp.int32, device=wp_device_obj)
    tile_col_group = wp.empty(max_tiles, dtype=wp.int32, device=wp_device_obj)

    nm = wp.empty((n, max_nb), dtype=wp.int32, device=wp_device_obj)
    nn_arr = wp.zeros(n, dtype=wp.int32, device=wp_device_obj)
    nms = wp.empty((n, max_nb, 3), dtype=wp.int32, device=wp_device_obj)

    cutoff_f = float(cutoff)

    # Decide which sort path to use.  wp.tile_sort has specializations for
    # N ∈ {1024, 2048}: faster than CUB at 1024, marginal at 2048; above that
    # the bitonic sort is too slow and CUB wins.  Either way, the sort lives
    # INSIDE the captured graph — CUB's `wp.utils.radix_sort_pairs` survives
    # capture + sync-between-replays at every size we tested (previously
    # diagnosed as a CUB issue was actually a missing-keepalive bug).
    def _do_morton_and_sort():
        wp_compute_morton(
            positions_wp,
            wp_inv_cell,
            n,
            morton_codes,
            sorted_atom_index,
            nn_arr,
            num_tiles,
            wp_device,
        )
        # ``_tile_sort_pairs`` launches a tile-sort specialization for the
        # supported fixed sizes (1024 / 2048) and returns False otherwise, so
        # we fall back to CUB radix sort. Either path lives inside the captured
        # graph.
        if not wp_tile_sort_pairs(morton_codes, sorted_atom_index, n, wp_device):
            wp.utils.radix_sort_pairs(morton_codes, sorted_atom_index, n)

    def _run_pipeline():
        """Whole pipeline (morton+sort → gather → build → to_matrix →
        fill_tail) captured into one graph; step() is a single replay.

        The query launches over the allocated tile-buffer capacity and the
        kernel reads ``num_tiles[0]`` on-device, so no host-side tile count is
        needed (unlike the pre-refactor ``tile_to_matrix`` which took
        ``n_tiles``).
        """
        _do_morton_and_sort()
        wp_permute_gather_soa(
            positions_wp, sorted_atom_index, n, spx, spy, spz, wp_device
        )
        wp_build_tile_neighbor_list(
            sorted_pos_x=spx,
            sorted_pos_y=spy,
            sorted_pos_z=spz,
            cell=wp_cell,
            inv_cell=wp_inv_cell,
            cutoff=cutoff_f,
            num_tiles=num_tiles,
            tile_row_group=tile_row_group,
            tile_col_group=tile_col_group,
            wp_dtype=wp.float32,
            device=wp_device,
            group_ctr_x_buffer=gcx,
            group_ctr_y_buffer=gcy,
            group_ctr_z_buffer=gcz,
            group_ext_x_buffer=gex,
            group_ext_y_buffer=gey,
            group_ext_z_buffer=gez,
        )
        wp_tile_to_matrix(
            sorted_atom_index=sorted_atom_index,
            sorted_pos_x=spx,
            sorted_pos_y=spy,
            sorted_pos_z=spz,
            num_tiles=num_tiles,
            tile_row_group=tile_row_group,
            tile_col_group=tile_col_group,
            cell=wp_cell,
            inv_cell=wp_inv_cell,
            cutoff=cutoff_f,
            natom=n,
            neighbor_matrix=nm,
            num_neighbors=nn_arr,
            neighbor_matrix_shifts=nms,
            wp_dtype=wp.float32,
            device=wp_device,
        )
        wp_fill_neighbor_matrix_tail(nn_arr, n, max_nb, n, nm, wp_device)

    # Warm up the full pipeline (incl. sort) so any one-shot JIT compile
    # happens before capture.
    for _ in range(5):
        _run_pipeline()
    wp.synchronize()

    # Capture the whole pipeline (morton+sort → gather → build →
    # to_matrix → fill_tail) into a single Warp graph.  step() is one
    # ``wp.capture_launch`` per call.
    stream = wp.get_stream(wp_device_obj)
    wp.capture_begin(wp_device_obj, stream)
    _run_pipeline()
    graph = wp.capture_end(wp_device_obj, stream)

    # Keep references to every wp.array that the captured graph points into.
    # The CUDA graph records raw device pointers; if any source wp.array is
    # GC'd after setup returns, the pointer is dangling and the next replay
    # hits ``cudaErrorIllegalAddress``.  Bundle them onto ``step`` so the
    # closure owns them.
    _keepalive = (
        positions_wp,
        morton_codes,
        sorted_atom_index,
        spx,
        spy,
        spz,
        gcx,
        gcy,
        gcz,
        gex,
        gey,
        gez,
        num_tiles,
        tile_row_group,
        tile_col_group,
        nm,
        nn_arr,
        nms,
    )

    def step(_keep=_keepalive):
        wp.capture_launch(graph)

    return step


def _setup_cell_list_pair_centric_single(
    positions,
    cutoff,
    cell,
    pbc,
    n,
    max_nb,
    device,
    block_dim=64,
    skip_prefill=False,
    graph_safe=False,
):
    """Standalone pair-centric cell-list query timer.

    Calls ``query_cell_list_pair_centric_sorted`` (per-emit atomic).

    Mirrors the production torch-wrapper dispatch path for the
    ``_should_dispatch_pair_centric`` branch, but calls the raw Warp
    launchers directly so the timed region measures *only* the kernels —
    no ``torch.library.custom_op`` machinery, no dispatch heuristic.

    Pipeline inside ``step()``:
      1. ``wp_build_cell_list`` (binning)
      2. exclusive cumsum on ``atoms_per_cell_count`` → ``cell_atom_start_indices``
      3. ``gather_fused`` to produce sorted-by-cell positions + shifts
      4. ``query_cell_list_pair_centric_sorted`` (the two pair-centric kernels)

    Everything else — the half-shell offsets, sort/gather scratch
    tensors, all ``wp.from_torch`` conversions — is preallocated outside
    the timed region.
    """
    wp_device = str(device)
    cell3 = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pbc_sq = pbc.squeeze(0) if pbc.ndim == 2 else pbc
    max_total_cells, nsr = estimate_cell_list_sizes(cell3, pbc_sq, cutoff)

    # n_outer is the only host-side dependency on the per-axis radius; the
    # kernel decodes (dx, dy, dz) on-the-fly via the shared shift-index
    # decoders.  Half-shell at radius R: n_outer = R*(2R+1)² + R*(2R+1) + R.
    if torch.is_tensor(nsr):
        Rx, Ry, Rz = int(nsr[0].item()), int(nsr[1].item()), int(nsr[2].item())
    else:
        Rx, Ry, Rz = int(nsr[0]), int(nsr[1]), int(nsr[2])
    n_outer = compute_batch_pair_centric_n_outer((Rx, Ry, Rz), True)

    # Cell-list scratch tensors (build outputs).
    cpd = torch.empty(3, dtype=torch.int32, device=device)
    aps = torch.empty((n, 3), dtype=torch.int32, device=device)
    atc = torch.empty((n, 3), dtype=torch.int32, device=device)
    apcc = torch.zeros(max_total_cells, dtype=torch.int32, device=device)
    casi = torch.zeros(max_total_cells, dtype=torch.int32, device=device)
    cal = torch.empty(n, dtype=torch.int32, device=device)

    # Pair-centric inputs (sorted-by-cell SoA-like vec3 arrays).
    sorted_positions = torch.empty((n, 3), dtype=torch.float32, device=device)
    sorted_shifts = torch.empty((n, 3), dtype=torch.int32, device=device)

    # Outputs.
    nm = torch.empty((n, max_nb), dtype=torch.int32, device=device)
    nn = torch.empty(n, dtype=torch.int32, device=device)
    nms = torch.empty((n, max_nb, 3), dtype=torch.int32, device=device)

    # Preconvert handles — all heap-allocated; reused every step.
    wp_positions = wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True)
    wp_cell = wp.from_torch(cell3, dtype=wp.mat33f, return_ctype=True)
    wp_pbc = wp.from_torch(pbc_sq, dtype=wp.bool, return_ctype=True)
    wp_cpd = wp.from_torch(cpd, dtype=wp.int32, return_ctype=True)
    wp_aps = wp.from_torch(aps, dtype=wp.vec3i, return_ctype=True)
    wp_atc = wp.from_torch(atc, dtype=wp.vec3i, return_ctype=True)
    wp_apcc_full = wp.from_torch(apcc, dtype=wp.int32)
    wp_casi_full = wp.from_torch(casi, dtype=wp.int32)
    wp_cal = wp.from_torch(cal, dtype=wp.int32, return_ctype=True)
    wp_apcc_c = wp.from_torch(apcc, dtype=wp.int32, return_ctype=True)
    wp_casi_c = wp.from_torch(casi, dtype=wp.int32, return_ctype=True)
    wp_sorted_pos = wp.from_torch(sorted_positions, dtype=wp.vec3f, return_ctype=True)
    wp_sorted_shifts = wp.from_torch(sorted_shifts, dtype=wp.vec3i, return_ctype=True)
    wp_nsr = wp.from_torch(nsr, dtype=wp.int32, return_ctype=True)
    wp_nm = wp.from_torch(nm, dtype=wp.int32, return_ctype=True)
    wp_nms = wp.from_torch(nms, dtype=wp.vec3i, return_ctype=True)
    wp_nn = wp.from_torch(nn, dtype=wp.int32, return_ctype=True)
    # Extra wp.array views (not ctype) for the graph-safe path: needed by
    # `wp.utils.array_scan` and the warp zero-kernel which take wp.array
    # inputs (the ctypes views can't be passed to wp.utils.array_scan).
    wp_nn_arr = wp.from_torch(nn, dtype=wp.int32)
    wp_apcc_arr = wp_apcc_full
    wp_casi_arr = wp_casi_full

    def step():
        if not skip_prefill:
            nm.fill_(n)
            nms.zero_()
        if graph_safe:
            # Warp launches only — no torch caching-allocator interaction
            # in the captured region.
            wp.launch(
                _zero_int32_kernel,
                dim=int(n),
                inputs=[wp_nn_arr, int(n)],
                device=wp_device,
            )
            wp.launch(
                _zero_int32_kernel,
                dim=int(max_total_cells),
                inputs=[wp_apcc_arr, int(max_total_cells)],
                device=wp_device,
            )
        else:
            nn.zero_()
            apcc.zero_()
        wp_build_cell_list(
            positions=wp_positions,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=cutoff,
            cells_per_dimension=wp_cpd,
            atom_periodic_shifts=wp_aps,
            atom_to_cell_mapping=wp_atc,
            atoms_per_cell_count=wp_apcc_full,
            cell_atom_start_indices=wp_casi_full,
            cell_atom_list=wp_cal,
            wp_dtype=wp.float32,
            device=wp_device,
        )
        if max_total_cells > 1:
            if graph_safe:
                wp.utils.array_scan(wp_apcc_arr, wp_casi_arr, inclusive=False)
            else:
                # casi[0] stays 0 from initial torch.zeros alloc; cumsum
                # writes casi[1:].
                torch.cumsum(apcc[:-1], dim=0, out=casi[1:])
        wp.launch(
            get_cell_list_gather_kernel(wp.float32),
            dim=n,
            inputs=[
                wp_positions,
                wp_aps,
                wp_cal,
                wp_sorted_pos,
                wp_sorted_shifts,
            ],
            device=wp_device,
        )
        query_cell_list_pair_centric_sorted(
            sorted_positions=wp_sorted_pos,
            sorted_atom_periodic_shifts=wp_sorted_shifts,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=float(cutoff),
            cells_per_dimension=wp_cpd,
            neighbor_search_radius=wp_nsr,
            atoms_per_cell_count=wp_apcc_c,
            cell_atom_start_indices=wp_casi_c,
            cell_atom_list=wp_cal,
            neighbor_matrix=wp_nm,
            neighbor_matrix_shifts=wp_nms,
            num_neighbors=wp_nn,
            wp_dtype=wp.float32,
            device=wp_device,
            n_outer=n_outer,
            block_dim=block_dim,
            half_fill=True,
            rebuild_flags=None,
        )
        if skip_prefill:
            # Coalesced tail fill: write sentinel `n` to nm[i, nn[i]..max_nb-1].
            # nms tail is left with stale data — safe because consumers gate
            # on nm sentinel and never index nms past nn[i].
            wp_fill_neighbor_matrix_tail(
                wp_nn,
                int(n),
                int(max_nb),
                int(n),
                wp_nm,
                wp_device,
            )

    return step


def _setup_cell_list_single(
    positions,
    cutoff,
    cell,
    pbc,
    n,
    max_nb,
    device,
    skip_prefill=False,
    graph_safe=False,
):
    wp_device = str(device)
    cell3 = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pbc_sq = pbc.squeeze(0)
    max_total_cells, nsr = estimate_cell_list_sizes(cell3, pbc_sq, cutoff)

    cpd = torch.empty(3, dtype=torch.int32, device=device)
    aps = torch.empty((n, 3), dtype=torch.int32, device=device)
    atc = torch.empty((n, 3), dtype=torch.int32, device=device)
    apcc = torch.zeros(max_total_cells, dtype=torch.int32, device=device)
    casi = torch.zeros(max_total_cells, dtype=torch.int32, device=device)
    cal = torch.empty(n, dtype=torch.int32, device=device)
    nm = torch.empty((n, max_nb), dtype=torch.int32, device=device)
    nn = torch.empty(n, dtype=torch.int32, device=device)
    nms = torch.empty((n, max_nb, 3), dtype=torch.int32, device=device)

    # Caller-allocated sort scratch + always-True rebuild flag for the
    # warp-level query_cell_list (no hidden state inside the launcher).
    sorted_pos = torch.empty((n, 3), dtype=torch.float32, device=device)
    sorted_shifts = torch.empty((n, 3), dtype=torch.int32, device=device)
    always_true = torch.ones(1, dtype=torch.bool, device=device)

    wp_positions = wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True)
    wp_cell = wp.from_torch(cell3, dtype=wp.mat33f, return_ctype=True)
    wp_pbc = wp.from_torch(pbc_sq, dtype=wp.bool, return_ctype=True)
    wp_cpd = wp.from_torch(cpd, dtype=wp.int32, return_ctype=True)
    wp_nsr = wp.from_torch(nsr, dtype=wp.int32, return_ctype=True)
    wp_aps = wp.from_torch(aps, dtype=wp.vec3i, return_ctype=True)
    wp_atc = wp.from_torch(atc, dtype=wp.vec3i, return_ctype=True)
    wp_apcc_full = wp.from_torch(apcc, dtype=wp.int32)
    wp_casi_full = wp.from_torch(casi, dtype=wp.int32)
    wp_cal = wp.from_torch(cal, dtype=wp.int32, return_ctype=True)
    wp_apcc_c = wp.from_torch(apcc, dtype=wp.int32, return_ctype=True)
    wp_casi_c = wp.from_torch(casi, dtype=wp.int32, return_ctype=True)
    wp_nm = wp.from_torch(nm, dtype=wp.int32, return_ctype=True)
    wp_nms = wp.from_torch(nms, dtype=wp.vec3i, return_ctype=True)
    wp_nn = wp.from_torch(nn, dtype=wp.int32, return_ctype=True)
    wp_sorted_pos = wp.from_torch(sorted_pos, dtype=wp.vec3f, return_ctype=True)
    wp_sorted_shifts = wp.from_torch(sorted_shifts, dtype=wp.vec3i, return_ctype=True)
    wp_always_true = wp.from_torch(always_true, dtype=wp.bool, return_ctype=True)
    # Extra wp.array views for graph-safe path (zero kernel + scan inputs).
    wp_nn_arr = wp.from_torch(nn, dtype=wp.int32)
    wp_apcc_arr = wp_apcc_full
    wp_casi_arr = wp_casi_full

    def step():
        if not skip_prefill:
            nm.fill_(n)
            nms.zero_()
        if graph_safe:
            wp.launch(
                _zero_int32_kernel,
                dim=int(n),
                inputs=[wp_nn_arr, int(n)],
                device=wp_device,
            )
            wp.launch(
                _zero_int32_kernel,
                dim=int(max_total_cells),
                inputs=[wp_apcc_arr, int(max_total_cells)],
                device=wp_device,
            )
        else:
            nn.zero_()
            apcc.zero_()
        wp_build_cell_list(
            positions=wp_positions,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=cutoff,
            cells_per_dimension=wp_cpd,
            atom_periodic_shifts=wp_aps,
            atom_to_cell_mapping=wp_atc,
            atoms_per_cell_count=wp_apcc_full,
            cell_atom_start_indices=wp_casi_full,
            cell_atom_list=wp_cal,
            wp_dtype=wp.float32,
            device=wp_device,
        )
        if max_total_cells > 1:
            if graph_safe:
                wp.utils.array_scan(wp_apcc_arr, wp_casi_arr, inclusive=False)
            else:
                torch.cumsum(apcc[:-1], dim=0, out=casi[1:])
        wp_query_cell_list(
            positions=wp_positions,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=cutoff,
            cells_per_dimension=wp_cpd,
            neighbor_search_radius=wp_nsr,
            atom_periodic_shifts=wp_aps,
            atom_to_cell_mapping=wp_atc,
            atoms_per_cell_count=wp_apcc_c,
            cell_atom_start_indices=wp_casi_c,
            cell_atom_list=wp_cal,
            sorted_positions=wp_sorted_pos,
            sorted_atom_periodic_shifts=wp_sorted_shifts,
            neighbor_matrix=wp_nm,
            neighbor_matrix_shifts=wp_nms,
            num_neighbors=wp_nn,
            rebuild_flags=wp_always_true,
            wp_dtype=wp.float32,
            device=wp_device,
            half_fill=True,
        )
        if skip_prefill:
            wp_fill_neighbor_matrix_tail(
                wp_nn,
                int(n),
                int(max_nb),
                int(n),
                wp_nm,
                wp_device,
            )

    return step


def _wrap_step_in_torch_graph(step, device, warmup=5):
    """Wrap an existing ``step()`` closure in a Warp graph capture.

    Mixed torch + warp pipelines require all kernels to land on a single
    stream during capture.  We force both libraries onto one side stream
    (``torch.cuda.stream(...)`` + ``wp.ScopedStream(...)``) and then call
    ``wp.capture_begin/end`` on that stream — same pattern as the
    cluster-tile graph variant.  ``torch.cuda.graph`` is avoided because
    it routes torch ops through the caching allocator's graph mempool in
    a way that conflicts with Warp's own stream-capture state on the
    same stream.

    Returns a step closure that does one ``wp.capture_launch`` per call.
    """
    wp_device_obj = wp.get_device(str(device))
    side_stream = torch.cuda.Stream(device=device)
    side_stream.wait_stream(torch.cuda.current_stream(device))
    wp_side_stream = wp.stream_from_torch(side_stream)

    with torch.cuda.stream(side_stream), wp.ScopedStream(wp_side_stream):
        for _ in range(warmup):
            step()
    torch.cuda.current_stream(device).wait_stream(side_stream)
    torch.cuda.synchronize(device)

    with torch.cuda.stream(side_stream), wp.ScopedStream(wp_side_stream):
        wp.capture_begin(wp_device_obj, wp_side_stream)
        step()
        graph = wp.capture_end(wp_device_obj, wp_side_stream)

    def graphed_step(_keep=(step, graph, side_stream, wp_side_stream)):
        wp.capture_launch(graph)

    return graphed_step


def _setup_cell_list_pair_centric_graph_single(
    positions,
    cutoff,
    cell,
    pbc,
    n,
    max_nb,
    device,
    block_dim=64,
):
    """Pair-centric cell-list with ``skip_prefill=True`` wrapped in a
    ``torch.cuda.graph``.  Same kernels as :func:`_setup_cell_list_pair_centric_single`
    (skip-prefill variant) — graph capture absorbs the per-step CPU
    launch overhead.
    """
    step = _setup_cell_list_pair_centric_single(
        positions,
        cutoff,
        cell,
        pbc,
        n,
        max_nb,
        device,
        block_dim=block_dim,
        skip_prefill=True,
        graph_safe=True,
    )
    return _wrap_step_in_torch_graph(step, device)


def _setup_cell_list_atom_centric_graph_single(
    positions,
    cutoff,
    cell,
    pbc,
    n,
    max_nb,
    device,
):
    """Atom-centric cell-list with ``skip_prefill=True`` wrapped in a
    ``torch.cuda.graph``.  Same kernels as
    :func:`_setup_cell_list_single` (skip-prefill variant) — graph capture
    absorbs the per-step CPU launch overhead.
    """
    step = _setup_cell_list_single(
        positions,
        cutoff,
        cell,
        pbc,
        n,
        max_nb,
        device,
        skip_prefill=True,
        graph_safe=True,
    )
    return _wrap_step_in_torch_graph(step, device)


def _setup_naive_single(positions, cutoff, cell, pbc, n, max_nb, use_pbc, device):
    """Brute-force naive (tiled).  No-PBC uses the non-PBC launcher; PBC uses
    the PBC launcher with the default cubic shift range [-1, 0, 1]^3."""
    wp_device = str(device)

    nm = torch.empty((n, max_nb), dtype=torch.int32, device=device)
    nn = torch.empty(n, dtype=torch.int32, device=device)

    wp_positions = wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True)
    wp_nm = wp.from_torch(nm, dtype=wp.int32, return_ctype=True)
    wp_nn = wp.from_torch(nn, dtype=wp.int32, return_ctype=True)

    if not use_pbc:

        def step():
            nm.fill_(n)
            nn.zero_()
            wp_naive_tiled(
                positions=wp_positions,
                cutoff=cutoff,
                neighbor_matrix=wp_nm,
                num_neighbors=wp_nn,
                wp_dtype=wp.float32,
                device=wp_device,
                half_fill=True,
                rebuild_flags=None,
            )

        return step

    cell3 = cell if cell.ndim == 3 else cell.unsqueeze(0)
    shift_range = torch.tensor([[1, 1, 1]], dtype=torch.int32, device=device)
    num_shifts = 27  # (2*1+1)^3
    nms = torch.empty((n, max_nb, 3), dtype=torch.int32, device=device)

    wp_cell = wp.from_torch(cell3, dtype=wp.mat33f, return_ctype=True)
    wp_shift_range = wp.from_torch(shift_range, dtype=wp.vec3i, return_ctype=True)
    wp_nms = wp.from_torch(nms, dtype=wp.vec3i, return_ctype=True)

    def step():
        nm.fill_(n)
        nn.zero_()
        nms.zero_()
        wp_naive_pbc_tiled(
            positions=wp_positions,
            cutoff=cutoff,
            cell=wp_cell,
            shift_range=wp_shift_range,
            num_shifts=num_shifts,
            neighbor_matrix=wp_nm,
            neighbor_matrix_shifts=wp_nms,
            num_neighbors=wp_nn,
            wp_dtype=wp.float32,
            device=wp_device,
            half_fill=True,
            rebuild_flags=None,
            wrap_positions=False,
        )

    return step


def _setup_cluster_tile_batch(
    positions, cutoff, cell_batch, batch_ptr, n, max_nb, device
):
    wp_device = str(device)
    num_systems = cell_batch.shape[0]
    inv_cell_batch = torch.linalg.inv(cell_batch).contiguous()
    batch_idx = torch.repeat_interleave(
        torch.arange(num_systems, dtype=torch.int32, device=device),
        (batch_ptr[1:] - batch_ptr[:-1]),
    )

    (
        sorted_atom_index,
        sort_inv,
        spx,
        spy,
        spz,
        batch_idx_sorted,
        batch_ptr_padded,
        group_system,
        group_ptr,
        gcx,
        gcy,
        gcz,
        gex,
        gey,
        gez,
        num_tiles,
        tile_row_group,
        tile_col_group,
        tile_system,
    ) = allocate_batch_tile_neighbor_list(batch_ptr, device)
    nm = torch.empty((n, max_nb), dtype=torch.int32, device=device)
    nn = torch.empty(n, dtype=torch.int32, device=device)
    nms = torch.empty((n, max_nb, 3), dtype=torch.int32, device=device)

    wp_sorted_atom_index = wp.from_torch(
        sorted_atom_index, dtype=wp.int32, return_ctype=True
    )
    wp_spx = wp.from_torch(spx, dtype=wp.float32, return_ctype=True)
    wp_spy = wp.from_torch(spy, dtype=wp.float32, return_ctype=True)
    wp_spz = wp.from_torch(spz, dtype=wp.float32, return_ctype=True)
    wp_group_system = wp.from_torch(group_system, dtype=wp.int32, return_ctype=True)
    wp_group_ptr = wp.from_torch(group_ptr, dtype=wp.int32, return_ctype=True)
    wp_cell_batch = wp.from_torch(
        cell_batch.contiguous(), dtype=wp.mat33f, return_ctype=True
    )
    wp_inv_cell_batch = wp.from_torch(
        inv_cell_batch, dtype=wp.mat33f, return_ctype=True
    )
    wp_gcx = wp.from_torch(gcx, dtype=wp.float32, return_ctype=True)
    wp_gcy = wp.from_torch(gcy, dtype=wp.float32, return_ctype=True)
    wp_gcz = wp.from_torch(gcz, dtype=wp.float32, return_ctype=True)
    wp_gex = wp.from_torch(gex, dtype=wp.float32, return_ctype=True)
    wp_gey = wp.from_torch(gey, dtype=wp.float32, return_ctype=True)
    wp_gez = wp.from_torch(gez, dtype=wp.float32, return_ctype=True)
    wp_num_tiles = wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True)
    wp_tile_row_group = wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True)
    wp_tile_col_group = wp.from_torch(tile_col_group, dtype=wp.int32, return_ctype=True)
    wp_tile_system = wp.from_torch(tile_system, dtype=wp.int32, return_ctype=True)
    wp_nm = wp.from_torch(nm, dtype=wp.int32, return_ctype=True)
    wp_nn = wp.from_torch(nn, dtype=wp.int32, return_ctype=True)
    wp_nms = wp.from_torch(nms, dtype=wp.int32, return_ctype=True)

    def _build():
        nm.fill_(n)
        nn.zero_()
        nms.zero_()
        num_tiles.zero_()
        gcx.zero_()
        gcy.zero_()
        gcz.zero_()
        gex.zero_()
        gey.zero_()
        gez.zero_()
        _batched_morton_sort_padded(
            positions,
            batch_idx,
            batch_ptr,
            inv_cell_batch,
            sorted_atom_index,
            sort_inv,
            spx,
            spy,
            spz,
            batch_idx_sorted,
            batch_ptr_padded,
        )
        group_ptr.copy_(
            (batch_ptr_padded // TILE_GROUP_SIZE).to(torch.int32).contiguous()
        )
        group_system.copy_(
            batch_idx_sorted[::TILE_GROUP_SIZE].to(torch.int32).contiguous()
        )
        wp_build_batch_tile_neighbor_list(
            sorted_pos_x=wp_spx,
            sorted_pos_y=wp_spy,
            sorted_pos_z=wp_spz,
            group_system=wp_group_system,
            group_ptr=wp_group_ptr,
            cell_batch=wp_cell_batch,
            inv_cell_batch=wp_inv_cell_batch,
            cutoff=float(cutoff),
            num_tiles=wp_num_tiles,
            tile_row_group=wp_tile_row_group,
            tile_col_group=wp_tile_col_group,
            tile_system=wp_tile_system,
            wp_dtype=wp.float32,
            device=wp_device,
            group_ctr_x_buffer=wp_gcx,
            group_ctr_y_buffer=wp_gcy,
            group_ctr_z_buffer=wp_gcz,
            group_ext_x_buffer=wp_gex,
            group_ext_y_buffer=wp_gey,
            group_ext_z_buffer=wp_gez,
        )

    _build()

    def step():
        _build()
        wp_batch_tile_to_matrix(
            sorted_atom_index=wp_sorted_atom_index,
            sorted_pos_x=wp_spx,
            sorted_pos_y=wp_spy,
            sorted_pos_z=wp_spz,
            cell_batch=wp_cell_batch,
            inv_cell_batch=wp_inv_cell_batch,
            num_tiles=wp_num_tiles,
            tile_row_group=wp_tile_row_group,
            tile_col_group=wp_tile_col_group,
            tile_system=wp_tile_system,
            cutoff=float(cutoff),
            natom=int(n),
            neighbor_matrix=wp_nm,
            num_neighbors=wp_nn,
            neighbor_matrix_shifts=wp_nms,
            wp_dtype=wp.float32,
            device=wp_device,
        )

    return step


def _setup_cell_list_batch(
    positions, cutoff, cell_batch, pbc, batch_idx, n, max_nb, device
):
    wp_device = str(device)
    num_systems = cell_batch.shape[0]
    max_total_cells, nsr = estimate_batch_cell_list_sizes(cell_batch, pbc, cutoff)

    cpd = torch.empty((num_systems, 3), dtype=torch.int32, device=device)
    aps_t = torch.empty((n, 3), dtype=torch.int32, device=device)
    atc = torch.empty((n, 3), dtype=torch.int32, device=device)
    apcc = torch.zeros(max_total_cells, dtype=torch.int32, device=device)
    casi = torch.zeros(max_total_cells, dtype=torch.int32, device=device)
    cal = torch.empty(n, dtype=torch.int32, device=device)
    cell_offsets = torch.zeros(num_systems, dtype=torch.int32, device=device)
    cells_per_system_scratch = torch.zeros(
        num_systems, dtype=torch.int32, device=device
    )
    nm = torch.empty((n, max_nb), dtype=torch.int32, device=device)
    nn = torch.empty(n, dtype=torch.int32, device=device)
    nms = torch.empty((n, max_nb, 3), dtype=torch.int32, device=device)

    wp_positions = wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True)
    wp_cell = wp.from_torch(cell_batch, dtype=wp.mat33f, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
    wp_batch_idx = wp.from_torch(
        batch_idx.to(torch.int32), dtype=wp.int32, return_ctype=True
    )
    wp_cpd = wp.from_torch(cpd, dtype=wp.vec3i, return_ctype=True)
    wp_nsr = wp.from_torch(nsr, dtype=wp.vec3i, return_ctype=True)
    wp_aps_t = wp.from_torch(aps_t, dtype=wp.vec3i, return_ctype=True)
    wp_atc = wp.from_torch(atc, dtype=wp.vec3i, return_ctype=True)
    wp_apcc_full = wp.from_torch(apcc, dtype=wp.int32)
    wp_casi_full = wp.from_torch(casi, dtype=wp.int32)
    wp_cal = wp.from_torch(cal, dtype=wp.int32, return_ctype=True)
    wp_cell_offsets_full = wp.from_torch(cell_offsets, dtype=wp.int32)
    wp_cell_offsets_c = wp.from_torch(cell_offsets, dtype=wp.int32, return_ctype=True)
    wp_cells_per_system_scratch = wp.from_torch(
        cells_per_system_scratch, dtype=wp.int32
    )
    wp_apcc_c = wp.from_torch(apcc, dtype=wp.int32, return_ctype=True)
    wp_casi_c = wp.from_torch(casi, dtype=wp.int32, return_ctype=True)
    wp_nm = wp.from_torch(nm, dtype=wp.int32, return_ctype=True)
    wp_nms = wp.from_torch(nms, dtype=wp.vec3i, return_ctype=True)
    wp_nn = wp.from_torch(nn, dtype=wp.int32, return_ctype=True)
    cells_per_system_query = torch.zeros(num_systems, dtype=torch.int32, device=device)

    def _do_build():
        nm.fill_(n)
        nn.zero_()
        nms.zero_()
        apcc.zero_()
        wp_batch_build_cell_list(
            positions=wp_positions,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=cutoff,
            batch_idx=wp_batch_idx,
            cells_per_dimension=wp_cpd,
            cell_offsets=wp_cell_offsets_full,
            cells_per_system=wp_cells_per_system_scratch,
            atom_periodic_shifts=wp_aps_t,
            atom_to_cell_mapping=wp_atc,
            atoms_per_cell_count=wp_apcc_full,
            cell_atom_start_indices=wp_casi_full,
            cell_atom_list=wp_cal,
            wp_dtype=wp.float32,
            device=wp_device,
        )
        cells_per_system_query.copy_(cpd.prod(dim=1).to(torch.int32))
        cell_offsets.zero_()
        if num_systems > 1:
            torch.cumsum(cells_per_system_query[:-1], dim=0, out=cell_offsets[1:])

    _do_build()

    def step():
        _do_build()
        wp_batch_query_cell_list(
            positions=wp_positions,
            cell=wp_cell,
            pbc=wp_pbc,
            cutoff=cutoff,
            batch_idx=wp_batch_idx,
            cells_per_dimension=wp_cpd,
            neighbor_search_radius=wp_nsr,
            cell_offsets=wp_cell_offsets_c,
            atom_periodic_shifts=wp_aps_t,
            atom_to_cell_mapping=wp_atc,
            atoms_per_cell_count=wp_apcc_c,
            cell_atom_start_indices=wp_casi_c,
            cell_atom_list=wp_cal,
            neighbor_matrix=wp_nm,
            neighbor_matrix_shifts=wp_nms,
            num_neighbors=wp_nn,
            wp_dtype=wp.float32,
            device=wp_device,
            half_fill=True,
        )

    return step


def _setup_batch_naive(
    positions,
    cutoff,
    cell_batch,
    pbc,
    batch_idx,
    batch_ptr,
    n,
    max_nb,
    use_pbc,
    device,
):
    """Batched brute-force naive (tiled) via raw warp launcher."""
    wp_device = str(device)
    max_atoms_per_system = int((batch_ptr[1:] - batch_ptr[:-1]).max().item())

    nm = torch.empty((n, max_nb), dtype=torch.int32, device=device)
    nn = torch.empty(n, dtype=torch.int32, device=device)

    wp_positions = wp.from_torch(positions, dtype=wp.vec3f, return_ctype=True)
    wp_batch_idx = wp.from_torch(
        batch_idx.to(torch.int32),
        dtype=wp.int32,
        return_ctype=True,
    )
    wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32, return_ctype=True)
    wp_nm = wp.from_torch(nm, dtype=wp.int32, return_ctype=True)
    wp_nn = wp.from_torch(nn, dtype=wp.int32, return_ctype=True)

    if not use_pbc:

        def step():
            nm.fill_(n)
            nn.zero_()
            wp_batch_naive_tiled(
                positions=wp_positions,
                cutoff=cutoff,
                batch_idx=wp_batch_idx,
                batch_ptr=wp_batch_ptr,
                neighbor_matrix=wp_nm,
                num_neighbors=wp_nn,
                wp_dtype=wp.float32,
                device=wp_device,
                half_fill=True,
                rebuild_flags=None,
            )

        return step

    # PBC: precompute shift_range + num_shifts (static for the benchmark).
    cell3 = cell_batch if cell_batch.ndim == 3 else cell_batch.unsqueeze(0)
    pbc2 = pbc if pbc.ndim == 2 else pbc.unsqueeze(0)
    shift_range, num_shifts_per_system, max_shifts_per_system = (
        compute_naive_num_shifts(cell3, cutoff, pbc2)
    )
    nms = torch.empty((n, max_nb, 3), dtype=torch.int32, device=device)

    wp_cell = wp.from_torch(cell3, dtype=wp.mat33f, return_ctype=True)
    wp_shift_range = wp.from_torch(shift_range, dtype=wp.vec3i, return_ctype=True)
    wp_num_shifts = wp.from_torch(
        num_shifts_per_system,
        dtype=wp.int32,
        return_ctype=True,
    )
    wp_nms = wp.from_torch(nms, dtype=wp.vec3i, return_ctype=True)

    def step():
        # Note: the raw launcher internally wp.zeros(total_atoms, vec3i) for
        # per_atom_cell_offsets when wrap_positions=False.  That is an
        # unavoidable per-call allocation inside the library (~n*12 bytes);
        # everything else is preconverted.
        nm.fill_(n)
        nn.zero_()
        nms.zero_()
        wp_batch_naive_pbc_tiled(
            positions=wp_positions,
            cell=wp_cell,
            cutoff=cutoff,
            batch_ptr=wp_batch_ptr,
            batch_idx=wp_batch_idx,
            shift_range=wp_shift_range,
            num_shifts_arr=wp_num_shifts,
            max_shifts_per_system=int(max_shifts_per_system),
            neighbor_matrix=wp_nm,
            neighbor_matrix_shifts=wp_nms,
            num_neighbors=wp_nn,
            wp_dtype=wp.float32,
            device=wp_device,
            max_atoms_per_system=max_atoms_per_system,
            half_fill=True,
            rebuild_flags=None,
            wrap_positions=False,
            sort_positions=False,
        )

    return step
