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

"""Warp-facing launchers for naive neighbor lists."""

from __future__ import annotations

from typing import NamedTuple

import warp as wp

from nvalchemiops.neighbors.naive.dispatch import (
    _has_naive_pair_outputs,
    _is_cpu_device,
    _pbc_mode_from_wrap,
)
from nvalchemiops.neighbors.naive.kernels import (
    BLOCK_DIM,
    get_naive_neighbor_matrix_dual_cutoff_kernel,
    get_naive_neighbor_matrix_kernel,
)
from nvalchemiops.neighbors.neighbor_utils import (
    DTYPE_INFO_ALL,
    compute_inv_cells,
    resolve_buffer_alias,
    selective_zero_num_neighbors,
    wrap_positions_batch,
    wrap_positions_single,
)
from nvalchemiops.neighbors.neighbor_utils import (
    empty_sentinel as _empty_sentinel,
)
from nvalchemiops.neighbors.output_args import (
    _is_empty_pair_params,
    _prepare_pair_output_args,
)

__all__ = [
    "get_naive_neighbor_matrix_kernel",
    "get_naive_neighbor_matrix_dual_cutoff_kernel",
    "naive_neighbor_matrix",
    "naive_neighbor_matrix_pbc",
    "batch_naive_neighbor_matrix",
    "batch_naive_neighbor_matrix_pbc",
    "naive_neighbor_matrix_dual_cutoff",
    "naive_neighbor_matrix_pbc_dual_cutoff",
    "batch_naive_neighbor_matrix_dual_cutoff",
    "batch_naive_neighbor_matrix_pbc_dual_cutoff",
]

_SUPPORTED_DTYPES = (wp.float16, wp.float32, wp.float64)
_DTYPE_INFO: dict[type, tuple[type, type]] = {
    dtype: DTYPE_INFO_ALL[dtype] for dtype in _SUPPORTED_DTYPES
}


class _ScalarSentinels(NamedTuple):
    """Zero-size placeholders for inactive scalar-kernel inputs."""

    offsets: wp.array
    cell: wp.array
    shift_range: wp.array
    num_shifts: wp.array
    batch_idx: wp.array
    batch_ptr: wp.array
    target_indices: wp.array
    neighbor_matrix: wp.array
    neighbor_matrix_shifts: wp.array
    num_neighbors: wp.array
    neighbor_vectors: wp.array
    neighbor_distances: wp.array
    pair_params: wp.array
    pair_energies: wp.array
    pair_forces: wp.array
    rebuild_flags: wp.array


def _reject_pair_fn_for_dual_cutoff(
    pair_fn: wp.Function | None, pair_params: wp.array | None
) -> None:
    """Reject active pair_fn/pair_params kwargs in dual-cutoff launchers."""
    if pair_fn is not None:
        raise ValueError("pair_fn is not supported with dual_cutoff=True")
    if pair_params is not None and not _is_empty_pair_params(pair_params):
        raise ValueError("pair_params is only valid when pair_fn is provided")


def _wrap_pbc_positions(
    positions: wp.array,
    cell: wp.array,
    inv_cell: wp.array,
    positions_wrapped: wp.array,
    per_atom_cell_offsets: wp.array,
    wp_dtype: type,
    device: str,
    *,
    batched: bool,
    batch_idx: wp.array | None,
) -> None:
    """Launch the shared position-wrapping helper for one batching mode."""
    if batched:
        if batch_idx is None:
            raise ValueError("batch_idx is required for batched PBC wrapping")
        wrap_positions_batch(
            positions,
            cell,
            inv_cell,
            batch_idx,
            positions_wrapped,
            per_atom_cell_offsets,
            wp_dtype,
            device,
        )
        return
    wrap_positions_single(
        positions,
        cell,
        inv_cell,
        positions_wrapped,
        per_atom_cell_offsets,
        wp_dtype,
        device,
    )


def _prepare_pbc_positions(
    positions: wp.array,
    cell: wp.array,
    batch_idx: wp.array | None,
    wp_dtype: type,
    device: str,
    *,
    batched: bool,
    wrap_positions: bool,
    positions_wrapped_buffer: wp.array | None = None,
    per_atom_cell_offsets_buffer: wp.array | None = None,
    inv_cell_buffer: wp.array | None = None,
):
    """Prepare wrapped or prewrapped positions and per-atom cell offsets.

    Caller-supplied scratch buffers are used when provided.  If any are
    absent the launcher allocates a fresh buffer for the call.
    """
    if not wrap_positions:
        return positions, _empty_sentinel(1, wp.vec3i, device)

    total_atoms = positions.shape[0]
    vec_dtype, mat_dtype = _DTYPE_INFO[wp_dtype]
    if inv_cell_buffer is None:
        inv_cell_buffer = wp.empty((cell.shape[0],), dtype=mat_dtype, device=device)
    if positions_wrapped_buffer is None:
        positions_wrapped_buffer = wp.empty(
            (total_atoms,), dtype=vec_dtype, device=device
        )
    if per_atom_cell_offsets_buffer is None:
        per_atom_cell_offsets_buffer = wp.empty(
            (total_atoms,), dtype=wp.vec3i, device=device
        )
    compute_inv_cells(cell, inv_cell_buffer, wp_dtype, device)
    _wrap_pbc_positions(
        positions,
        cell,
        inv_cell_buffer,
        positions_wrapped_buffer,
        per_atom_cell_offsets_buffer,
        wp_dtype,
        device,
        batched=batched,
        batch_idx=batch_idx,
    )
    return positions_wrapped_buffer, per_atom_cell_offsets_buffer


def _scalar_sentinels(wp_dtype: type, device: str) -> _ScalarSentinels:
    """Return zero-size placeholders for inactive scalar-kernel inputs."""
    vec_dtype, mat_dtype = _DTYPE_INFO[wp_dtype]
    return _ScalarSentinels(
        offsets=_empty_sentinel(1, wp.vec3i, device),
        cell=_empty_sentinel(1, mat_dtype, device),
        shift_range=_empty_sentinel(1, wp.vec3i, device),
        num_shifts=_empty_sentinel(1, wp.int32, device),
        batch_idx=_empty_sentinel(1, wp.int32, device),
        batch_ptr=_empty_sentinel(1, wp.int32, device),
        target_indices=_empty_sentinel(1, wp.int32, device),
        neighbor_matrix=_empty_sentinel(2, wp.int32, device),
        neighbor_matrix_shifts=_empty_sentinel(2, wp.vec3i, device),
        num_neighbors=_empty_sentinel(1, wp.int32, device),
        neighbor_vectors=_empty_sentinel(2, vec_dtype, device),
        neighbor_distances=_empty_sentinel(2, wp_dtype, device),
        pair_params=_empty_sentinel(2, wp_dtype, device),
        pair_energies=_empty_sentinel(2, wp_dtype, device),
        pair_forces=_empty_sentinel(2, vec_dtype, device),
        rebuild_flags=_empty_sentinel(1, wp.bool, device),
    )


def _launch_naive_neighbor_matrix_no_pbc(
    positions: wp.array,
    cutoff: float,
    neighbor_matrix: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    *,
    batched: bool,
    batch_idx: wp.array | None = None,
    batch_ptr: wp.array | None = None,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    target_indices: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
    native_strategy: str = "auto",
) -> None:
    """Launch the single-cutoff no-PBC naive neighbor-matrix path."""
    if batched and (batch_idx is None or batch_ptr is None):
        raise ValueError("batch_idx and batch_ptr are required for batched launch")

    has_pair_outputs = _has_naive_pair_outputs(
        target_indices,
        return_vectors,
        return_distances,
        pair_fn,
        pair_params,
        neighbor_vectors,
        neighbor_distances,
        pair_energies,
        pair_forces,
    )
    partial = target_indices is not None
    # Dispatch scalar vs tile-cooperative.  Pair-output and partial
    # (target_indices) paths have only a scalar specialization; CPU always
    # uses scalar (Warp forces block_dim=1).  The single-system path tiles on
    # CUDA unconditionally, while the batched path applies the adaptive
    # ``use_tiled`` heuristic: the tile-cooperative kernel wins for
    # few-large-systems but the scalar thread-local-counter kernel wins for
    # many-small-systems, so dispatch on the atoms-per-system density.
    if native_strategy not in {"auto", "scalar", "tile"}:
        raise ValueError(
            "native_strategy must be 'auto' | 'scalar' | 'tile', "
            f"got {native_strategy!r}",
        )
    if native_strategy == "scalar":
        strategy = "scalar"
    elif native_strategy == "tile":
        if has_pair_outputs or _is_cpu_device(device):
            raise ValueError(
                "native_strategy='tile' requires CUDA and no pair-output or "
                "target_indices path",
            )
        strategy = "tile"
    elif has_pair_outputs or _is_cpu_device(device):
        strategy = "scalar"
    elif batched:
        total_atoms = positions.shape[0]
        if total_atoms < 2048:
            use_tiled = False
        else:
            num_systems = batch_ptr.shape[0] - 1
            use_tiled = total_atoms >= 256 * num_systems
            if use_tiled and total_atoms > 12288:
                use_tiled = total_atoms >= 512 * num_systems
        strategy = "tile" if use_tiled else "scalar"
    else:
        strategy = "tile"

    (
        empty_offsets,
        empty_cell,
        empty_shift_range,
        empty_num_shifts,
        empty_batch_idx,
        empty_batch_ptr,
        empty_target_indices,
        empty_matrix,
        empty_shifts,
        empty_num_neighbors,
        empty_vectors,
        empty_distances,
        empty_pair_params,
        empty_energies,
        empty_forces,
        empty_rebuild_flags,
    ) = _scalar_sentinels(wp_dtype, device)
    batch_idx_arg = batch_idx if batch_idx is not None else empty_batch_idx
    batch_ptr_arg = batch_ptr if batch_ptr is not None else empty_batch_ptr
    target_indices_arg = (
        target_indices if target_indices is not None else empty_target_indices
    )
    rebuild_flags_arg = (
        rebuild_flags if rebuild_flags is not None else empty_rebuild_flags
    )
    cutoff_sq = wp_dtype(cutoff * cutoff)

    if strategy == "tile":
        wp.launch_tiled(
            kernel=get_naive_neighbor_matrix_kernel(
                wp_dtype,
                pbc_mode="none",
                batched=batched,
                half_fill=half_fill,
                selective=rebuild_flags is not None,
                strategy="tile",
            ),
            dim=[1, positions.shape[0]],
            inputs=[
                positions,
                empty_offsets,
                cutoff_sq,
                empty_cell,
                empty_shift_range,
                empty_num_shifts,
                batch_idx_arg,
                batch_ptr_arg,
                neighbor_matrix,
                empty_shifts,
                num_neighbors,
                rebuild_flags_arg,
            ],
            block_dim=BLOCK_DIM,
            device=device,
        )
        return

    if has_pair_outputs:
        (
            neighbor_vectors_arg,
            neighbor_distances_arg,
            pair_params_arg,
            pair_energies_arg,
            pair_forces_arg,
        ) = _prepare_pair_output_args(
            wp_dtype,
            device,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
            pair_params=pair_params,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
            neighbor_vectors_name="neighbor_vectors",
            neighbor_distances_name="neighbor_distances",
            pair_energies_name="pair_energies",
            pair_forces_name="pair_forces",
        )
        dim = int(target_indices.shape[0]) if partial else int(positions.shape[0])
    else:
        neighbor_vectors_arg = empty_vectors
        neighbor_distances_arg = empty_distances
        pair_params_arg = empty_pair_params
        pair_energies_arg = empty_energies
        pair_forces_arg = empty_forces
        dim = int(positions.shape[0])

    wp.launch(
        kernel=get_naive_neighbor_matrix_kernel(
            wp_dtype,
            pbc_mode="none",
            batched=batched,
            half_fill=half_fill,
            selective=rebuild_flags is not None,
            partial=partial,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
        ),
        dim=(1, 1, dim),
        inputs=[
            positions,
            empty_offsets,
            cutoff_sq,
            wp_dtype(0.0),
            empty_cell,
            empty_shift_range,
            empty_num_shifts,
            batch_idx_arg,
            batch_ptr_arg,
            target_indices_arg,
            neighbor_matrix,
            empty_shifts,
            num_neighbors,
            empty_matrix,
            empty_shifts,
            empty_num_neighbors,
            neighbor_vectors_arg,
            neighbor_distances_arg,
            pair_params_arg,
            pair_energies_arg,
            pair_forces_arg,
            rebuild_flags_arg,
        ],
        device=device,
    )


def _launch_naive_neighbor_matrix_pbc(
    positions: wp.array,
    cutoff: float,
    cell: wp.array,
    shift_range: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    *,
    batched: bool,
    num_shifts: int | None = None,
    batch_ptr: wp.array | None = None,
    batch_idx: wp.array | None = None,
    num_shifts_arr: wp.array | None = None,
    max_shifts_per_system: int | None = None,
    max_atoms_per_system: int | None = None,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    wrap_positions: bool = True,
    target_indices: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
    positions_wrapped_buffer: wp.array | None = None,
    per_atom_cell_offsets_buffer: wp.array | None = None,
    inv_cell_buffer: wp.array | None = None,
    native_strategy: str = "auto",
) -> None:
    """Launch the single-cutoff PBC naive neighbor-matrix path."""
    if batched:
        if batch_ptr is None or batch_idx is None or num_shifts_arr is None:
            raise ValueError(
                "batch_ptr, batch_idx, and num_shifts_arr are required for batched PBC"
            )
        if max_shifts_per_system is None or max_atoms_per_system is None:
            raise ValueError(
                "max_shifts_per_system and max_atoms_per_system are required for batched PBC"
            )
    elif num_shifts is None:
        raise ValueError("num_shifts is required for single-system PBC")

    has_pair_outputs = _has_naive_pair_outputs(
        target_indices,
        return_vectors,
        return_distances,
        pair_fn,
        pair_params,
        neighbor_vectors,
        neighbor_distances,
        pair_energies,
        pair_forces,
    )
    partial = target_indices is not None
    pbc_mode = _pbc_mode_from_wrap(wrap_positions)
    if native_strategy not in {"auto", "scalar", "tile"}:
        raise ValueError(
            "native_strategy must be 'auto' | 'scalar' | 'tile', "
            f"got {native_strategy!r}",
        )
    can_tile = (
        not has_pair_outputs
        and not _is_cpu_device(device)
        and (not batched or wrap_positions)
    )
    if native_strategy == "tile":
        if not can_tile:
            raise ValueError(
                "native_strategy='tile' requires CUDA, no pair-output or "
                "target_indices path, and wrap_positions=True for batched PBC",
            )
        strategy = "tile"
    elif native_strategy == "scalar":
        strategy = "scalar"
    elif can_tile:
        strategy = "tile"
    else:
        strategy = "scalar"

    positions_work, per_atom_cell_offsets = _prepare_pbc_positions(
        positions,
        cell,
        batch_idx,
        wp_dtype,
        device,
        batched=batched,
        wrap_positions=wrap_positions,
        positions_wrapped_buffer=positions_wrapped_buffer,
        per_atom_cell_offsets_buffer=per_atom_cell_offsets_buffer,
        inv_cell_buffer=inv_cell_buffer,
    )
    (
        _empty_offsets,
        _empty_cell,
        _empty_shift_range,
        empty_num_shifts,
        empty_batch_idx,
        empty_batch_ptr,
        empty_target_indices,
        empty_matrix,
        empty_shifts,
        empty_num_neighbors,
        empty_vectors,
        empty_distances,
        empty_pair_params,
        empty_energies,
        empty_forces,
        empty_rebuild_flags,
    ) = _scalar_sentinels(wp_dtype, device)
    batch_idx_arg = batch_idx if batch_idx is not None else empty_batch_idx
    batch_ptr_arg = batch_ptr if batch_ptr is not None else empty_batch_ptr
    target_indices_arg = (
        target_indices if target_indices is not None else empty_target_indices
    )
    rebuild_flags_arg = (
        rebuild_flags if rebuild_flags is not None else empty_rebuild_flags
    )
    cutoff_sq = wp_dtype(cutoff * cutoff)

    if batched:
        launch_dim = (
            cell.shape[0],
            int(max_shifts_per_system),
            int(max_atoms_per_system),
        )
        num_shifts_arg = num_shifts_arr
        tile_dim = [int(max_shifts_per_system), positions.shape[0]]
    else:
        launch_dim = (1, int(num_shifts), positions.shape[0])
        num_shifts_arg = empty_num_shifts
        tile_dim = [int(num_shifts), positions.shape[0]]

    if strategy == "tile":
        wp.launch_tiled(
            kernel=get_naive_neighbor_matrix_kernel(
                wp_dtype,
                pbc_mode=pbc_mode.value,
                batched=batched,
                half_fill=half_fill,
                selective=rebuild_flags is not None,
                strategy="tile",
            ),
            dim=tile_dim,
            inputs=[
                positions_work,
                per_atom_cell_offsets,
                cutoff_sq,
                cell,
                shift_range,
                num_shifts_arg,
                batch_idx_arg,
                batch_ptr_arg,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                rebuild_flags_arg,
            ],
            block_dim=BLOCK_DIM,
            device=device,
        )
        return

    if has_pair_outputs:
        active_shift_dim = int(max_shifts_per_system) if batched else int(num_shifts)
        if partial and not half_fill:
            active_shift_dim = 2 * active_shift_dim - 1
        if partial:
            launch_dim = (1, active_shift_dim, int(target_indices.shape[0]))
        elif batched:
            launch_dim = (cell.shape[0], active_shift_dim, int(max_atoms_per_system))
        else:
            launch_dim = (1, active_shift_dim, positions.shape[0])
        (
            neighbor_vectors_arg,
            neighbor_distances_arg,
            pair_params_arg,
            pair_energies_arg,
            pair_forces_arg,
        ) = _prepare_pair_output_args(
            wp_dtype,
            device,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
            pair_params=pair_params,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
            neighbor_vectors_name="neighbor_vectors",
            neighbor_distances_name="neighbor_distances",
            pair_energies_name="pair_energies",
            pair_forces_name="pair_forces",
        )
    else:
        neighbor_vectors_arg = empty_vectors
        neighbor_distances_arg = empty_distances
        pair_params_arg = empty_pair_params
        pair_energies_arg = empty_energies
        pair_forces_arg = empty_forces

    wp.launch(
        kernel=get_naive_neighbor_matrix_kernel(
            wp_dtype,
            pbc_mode=pbc_mode.value,
            batched=batched,
            half_fill=half_fill,
            selective=rebuild_flags is not None,
            partial=partial,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
        ),
        dim=launch_dim,
        inputs=[
            positions_work,
            per_atom_cell_offsets,
            cutoff_sq,
            wp_dtype(0.0),
            cell,
            shift_range,
            num_shifts_arg,
            batch_idx_arg,
            batch_ptr_arg,
            target_indices_arg,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            empty_matrix,
            empty_shifts,
            empty_num_neighbors,
            neighbor_vectors_arg,
            neighbor_distances_arg,
            pair_params_arg,
            pair_energies_arg,
            pair_forces_arg,
            rebuild_flags_arg,
        ],
        device=device,
    )


def _launch_naive_neighbor_matrix_dual_cutoff(
    positions: wp.array,
    cutoff1: float,
    cutoff2: float,
    neighbor_matrix1: wp.array,
    num_neighbors1: wp.array,
    neighbor_matrix2: wp.array,
    num_neighbors2: wp.array,
    wp_dtype: type,
    device: str,
    *,
    batched: bool,
    batch_idx: wp.array | None = None,
    batch_ptr: wp.array | None = None,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
) -> None:
    """Launch dual-cutoff no-PBC kernels."""
    kernel = get_naive_neighbor_matrix_dual_cutoff_kernel(
        wp_dtype,
        pbc_mode="none",
        batched=batched,
        selective=rebuild_flags is not None,
    )
    (
        empty_offsets,
        empty_cell,
        empty_shift_range,
        empty_num_shifts,
        empty_batch_idx,
        empty_batch_ptr,
        empty_target_indices,
        _empty_matrix,
        empty_shifts,
        _empty_num_neighbors,
        empty_vectors,
        empty_distances,
        empty_pair_params,
        empty_energies,
        empty_forces,
        empty_rebuild_flags,
    ) = _scalar_sentinels(wp_dtype, device)
    if batched and (batch_idx is None or batch_ptr is None):
        raise ValueError("batch_idx and batch_ptr are required for batched dual cutoff")
    batch_idx_arg = batch_idx if batch_idx is not None else empty_batch_idx
    batch_ptr_arg = batch_ptr if batch_ptr is not None else empty_batch_ptr
    rebuild_flags_arg = (
        rebuild_flags if rebuild_flags is not None else empty_rebuild_flags
    )
    wp.launch(
        kernel=kernel,
        dim=(1, 1, positions.shape[0]),
        inputs=[
            positions,
            empty_offsets,
            wp_dtype(cutoff1 * cutoff1),
            wp_dtype(cutoff2 * cutoff2),
            empty_cell,
            empty_shift_range,
            empty_num_shifts,
            batch_idx_arg,
            batch_ptr_arg,
            empty_target_indices,
            neighbor_matrix1,
            empty_shifts,
            num_neighbors1,
            neighbor_matrix2,
            empty_shifts,
            num_neighbors2,
            empty_vectors,
            empty_distances,
            empty_pair_params,
            empty_energies,
            empty_forces,
            rebuild_flags_arg,
        ],
        device=device,
    )


def _launch_naive_neighbor_matrix_pbc_dual_cutoff(
    positions: wp.array,
    cutoff1: float,
    cutoff2: float,
    cell: wp.array,
    shift_range: wp.array,
    neighbor_matrix1: wp.array,
    neighbor_matrix2: wp.array,
    neighbor_matrix_shifts1: wp.array,
    neighbor_matrix_shifts2: wp.array,
    num_neighbors1: wp.array,
    num_neighbors2: wp.array,
    wp_dtype: type,
    device: str,
    *,
    batched: bool,
    num_shifts: int | None = None,
    batch_ptr: wp.array | None = None,
    batch_idx: wp.array | None = None,
    num_shifts_arr: wp.array | None = None,
    max_shifts_per_system: int | None = None,
    max_atoms_per_system: int | None = None,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    wrap_positions: bool = True,
    positions_wrapped_buffer: wp.array | None = None,
    per_atom_cell_offsets_buffer: wp.array | None = None,
    inv_cell_buffer: wp.array | None = None,
) -> None:
    """Launch dual-cutoff PBC kernels."""
    pbc_mode = _pbc_mode_from_wrap(wrap_positions)
    positions_work, per_atom_cell_offsets = _prepare_pbc_positions(
        positions,
        cell,
        batch_idx,
        wp_dtype,
        device,
        batched=batched,
        wrap_positions=wrap_positions,
        positions_wrapped_buffer=positions_wrapped_buffer,
        per_atom_cell_offsets_buffer=per_atom_cell_offsets_buffer,
        inv_cell_buffer=inv_cell_buffer,
    )
    kernel = get_naive_neighbor_matrix_dual_cutoff_kernel(
        wp_dtype,
        pbc_mode=pbc_mode.value,
        batched=batched,
        selective=rebuild_flags is not None,
    )
    (
        _empty_offsets,
        _empty_cell,
        _empty_shift_range,
        empty_num_shifts,
        empty_batch_idx,
        empty_batch_ptr,
        empty_target_indices,
        _empty_matrix,
        _empty_shifts,
        _empty_num_neighbors,
        empty_vectors,
        empty_distances,
        empty_pair_params,
        empty_energies,
        empty_forces,
        empty_rebuild_flags,
    ) = _scalar_sentinels(wp_dtype, device)
    batch_idx_arg = batch_idx if batch_idx is not None else empty_batch_idx
    batch_ptr_arg = batch_ptr if batch_ptr is not None else empty_batch_ptr
    rebuild_flags_arg = (
        rebuild_flags if rebuild_flags is not None else empty_rebuild_flags
    )

    if batched:
        if batch_ptr is None or batch_idx is None or num_shifts_arr is None:
            raise ValueError(
                "batch_ptr, batch_idx, and num_shifts_arr are required for batched PBC"
            )
        if max_shifts_per_system is None or max_atoms_per_system is None:
            raise ValueError(
                "max_shifts_per_system and max_atoms_per_system are required for batched PBC"
            )
        launch_dim = (cell.shape[0], max_shifts_per_system, max_atoms_per_system)
        num_shifts_arg = num_shifts_arr
    else:
        if num_shifts is None:
            raise ValueError("num_shifts is required for single-system PBC")
        launch_dim = (1, num_shifts, positions.shape[0])
        num_shifts_arg = empty_num_shifts

    wp.launch(
        kernel=kernel,
        dim=launch_dim,
        inputs=[
            positions_work,
            per_atom_cell_offsets,
            wp_dtype(cutoff1 * cutoff1),
            wp_dtype(cutoff2 * cutoff2),
            cell,
            shift_range,
            num_shifts_arg,
            batch_idx_arg,
            batch_ptr_arg,
            empty_target_indices,
            neighbor_matrix1,
            neighbor_matrix_shifts1,
            num_neighbors1,
            neighbor_matrix2,
            neighbor_matrix_shifts2,
            num_neighbors2,
            empty_vectors,
            empty_distances,
            empty_pair_params,
            empty_energies,
            empty_forces,
            rebuild_flags_arg,
        ],
        device=device,
    )


def naive_neighbor_matrix(
    positions: wp.array,
    cutoff: float,
    neighbor_matrix: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    target_indices: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
    native_strategy: str = "auto",
) -> None:
    """Core warp launcher for naive neighbor matrix construction (no PBC).

    Computes pairwise distances and fills the neighbor matrix with atom indices
    within the cutoff distance.  Internally dispatches between two kernels
    based on ``device``: a scalar SIMT kernel on CPU and a tile-cooperative
    kernel on CUDA (``wp.launch_tiled`` with ``block_dim = BLOCK_DIM``
    cooperatively sweeping the j-loop).  Both produce identical pair sets;
    per-row ordering within ``neighbor_matrix`` may differ.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Must be pre-allocated. Entries are filled with atom indices.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
        Must be pre-allocated. Updated in-place with actual neighbor counts.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically.
    rebuild_flags : wp.array, shape (1,), dtype=wp.bool, optional
        When provided, the kernel checks this flag on the GPU and skips work
        when False (no CPU-GPU sync).
    target_indices : wp.array, shape (M,), dtype=wp.int32, optional
        Unique, in-bounds global atom indices restricting which atoms act as
        sources (rows) in the output.  Output rows correspond to
        ``target_indices`` in order.  When omitted, all atoms are sources.
    return_vectors : bool, default=False
        If True, write per-pair displacement vectors into ``neighbor_vectors``.
        Requires ``neighbor_vectors`` to be supplied.
    return_distances : bool, default=False
        If True, write per-pair Euclidean distances into ``neighbor_distances``.
        Requires ``neighbor_distances`` to be supplied.
    pair_fn : wp.Function, optional
        Module-scope ``@wp.func`` with signature
        ``pair_fn(r_ij, distance, pair_params, i, j) -> (energy, force)``.  When
        provided, the kernel evaluates this callback per accepted pair and
        writes per-pair energies/forces into ``pair_energies`` / ``pair_forces``.
        ``pair_fn`` is keyed into the kernel cache by its function-object
        identity, so callers must use module-scope singleton ``@wp.func``
        objects (not lambdas or nested defs).
    pair_params : wp.array, shape (num_atoms, num_parameters), dtype=positions.dtype, optional
        Per-atom parameter table passed to ``pair_fn``.  ``pair_params[i]``
        is the parameter row of length ``num_parameters`` belonging to atom
        ``i`` and ``pair_fn`` may read any ``pair_params[j]`` row it needs
        (e.g. for Lorentz-Berthelot mixing in the Lennard-Jones pair
        potential).  Required when ``pair_fn`` is provided.
    neighbor_vectors : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*, optional
        OUTPUT: Required if ``return_vectors=True``.  Stores the displacement
        vector ``positions[j] - positions[i]`` for each recorded pair.
    neighbor_distances : wp.array, shape (rows, max_neighbors), dtype=wp.float*, optional
        OUTPUT: Required if ``return_distances=True``.  Stores the Euclidean
        distance for each recorded pair.
    pair_energies : wp.array, shape (rows, max_neighbors), dtype=wp.float*, optional
        OUTPUT: Required if ``pair_fn`` is provided.  Stores the per-pair
        energy returned by ``pair_fn``.
    pair_forces : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*, optional
        OUTPUT: Required if ``pair_fn`` is provided.  Stores the per-pair
        force returned by ``pair_fn``.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - The CUDA path uses ``wp.launch_tiled(block_dim=BLOCK_DIM)``; Warp forces
      ``block_dim = 1`` on CPU which would silently break the lane-cooperative
      partitioning, so CPU callers take the scalar path.
    - The tile-cooperative path is taken only for the default
      call.  When any of ``target_indices`` / ``return_vectors`` /
      ``return_distances`` / ``pair_fn`` is supplied, the scalar factory
      kernel is used regardless of device (no tile variant for these axes).

    See Also
    --------
    naive_neighbor_matrix_pbc : Version with periodic boundary conditions
    batch_naive_neighbor_matrix : Batched (multi-system) variant
    get_naive_neighbor_matrix_kernel : Low-level single-cutoff kernel accessor
    """
    if not _has_naive_pair_outputs(
        target_indices,
        return_vectors,
        return_distances,
        pair_fn,
        pair_params,
        neighbor_vectors,
        neighbor_distances,
        pair_energies,
        pair_forces,
    ):
        _launch_naive_neighbor_matrix_no_pbc(
            positions,
            cutoff,
            neighbor_matrix,
            num_neighbors,
            wp_dtype,
            device,
            batched=False,
            half_fill=half_fill,
            rebuild_flags=rebuild_flags,
        )
        return
    _launch_naive_neighbor_matrix_no_pbc(
        positions,
        cutoff,
        neighbor_matrix,
        num_neighbors,
        wp_dtype,
        device,
        batched=False,
        half_fill=half_fill,
        rebuild_flags=rebuild_flags,
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
        native_strategy=native_strategy,
    )


def batch_naive_neighbor_matrix(
    positions: wp.array,
    cutoff: float,
    batch_idx: wp.array,
    batch_ptr: wp.array,
    neighbor_matrix: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    target_indices: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
    native_strategy: str = "auto",
) -> None:
    """Core warp launcher for batched naive neighbor matrix construction (no PBC).

    Computes pairwise distances and fills the neighbor matrix for multiple
    systems in a batch using pure warp operations.  No periodic boundary
    conditions are applied.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated Cartesian coordinates for all systems.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store relationships where i < j to avoid double counting.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool, optional
        Per-system rebuild flags.  If provided, only systems where
        ``rebuild_flags[i]`` is True are processed; others are skipped on the
        GPU without CPU sync.  Per-system counters are reset via
        :func:`selective_zero_num_neighbors` internally.
    target_indices : wp.array, shape (M,), dtype=wp.int32, optional
        Unique, in-bounds global atom indices restricting which atoms act as
        sources.  In batched mode each target searches only atoms in its own
        system (the system is resolved via ``batch_idx``).  Output rows follow
        ``target_indices``.
    return_vectors : bool, default=False
        If True, write per-pair displacement vectors into ``neighbor_vectors``.
    return_distances : bool, default=False
        If True, write per-pair Euclidean distances into
        ``neighbor_distances``.
    pair_fn : wp.Function, optional
        Module-scope ``@wp.func`` with signature
        ``pair_fn(r_ij, distance, pair_params, i, j) -> (energy, force)``.  Required
        to be a module-scope singleton; keyed into the kernel cache by object
        identity.
    pair_params : wp.array, shape (num_atoms, num_parameters), dtype=positions.dtype, optional
        Per-atom parameter table passed to ``pair_fn``.  ``pair_params[i]``
        is the parameter row of length ``num_parameters`` belonging to atom
        ``i`` and ``pair_fn`` may read any ``pair_params[j]`` row it needs
        (e.g. for Lorentz-Berthelot mixing in the Lennard-Jones pair
        potential).
    neighbor_vectors : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*, optional
        OUTPUT: Required if ``return_vectors=True``.
    neighbor_distances : wp.array, shape (rows, max_neighbors), dtype=wp.float*, optional
        OUTPUT: Required if ``return_distances=True``.
    pair_energies : wp.array, shape (rows, max_neighbors), dtype=wp.float*, optional
        OUTPUT: Required if ``pair_fn`` is provided.
    pair_forces : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*, optional
        OUTPUT: Required if ``pair_fn`` is provided.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - Default calls dispatch internally:

      * On CPU, always use the scalar kernel (Warp forces ``block_dim=1`` on CPU).
      * On CUDA, use the tile-cooperative kernel when the adaptive
        ``use_tiled`` heuristic favours it (``total_atoms >= 2048`` and
        ``total_atoms >= 256 * num_systems``, with a tighter
        ``>= 512 * num_systems`` threshold above 12 288 atoms); otherwise fall
        back to the scalar kernel.
    - When any of the pair-output kwargs is supplied, the scalar factory
      kernel is used (no tile variant for the pair-output kwargs).

    See Also
    --------
    batch_naive_neighbor_matrix_pbc : Version with periodic boundary conditions
    naive_neighbor_matrix : Single-system variant
    get_naive_neighbor_matrix_kernel : Low-level single-cutoff kernel accessor
    """
    if not _has_naive_pair_outputs(
        target_indices,
        return_vectors,
        return_distances,
        pair_fn,
        pair_params,
        neighbor_vectors,
        neighbor_distances,
        pair_energies,
        pair_forces,
    ):
        if rebuild_flags is not None:
            selective_zero_num_neighbors(
                num_neighbors, batch_idx, rebuild_flags, device
            )
        _launch_naive_neighbor_matrix_no_pbc(
            positions,
            cutoff,
            neighbor_matrix,
            num_neighbors,
            wp_dtype,
            device,
            batched=True,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            half_fill=half_fill,
            rebuild_flags=rebuild_flags,
        )
        return
    if rebuild_flags is not None:
        selective_zero_num_neighbors(num_neighbors, batch_idx, rebuild_flags, device)
    _launch_naive_neighbor_matrix_no_pbc(
        positions,
        cutoff,
        neighbor_matrix,
        num_neighbors,
        wp_dtype,
        device,
        batched=True,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        half_fill=half_fill,
        rebuild_flags=rebuild_flags,
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
        native_strategy=native_strategy,
    )


def naive_neighbor_matrix_pbc(
    positions: wp.array,
    cutoff: float,
    cell: wp.array,
    shift_range: wp.array,
    num_shifts: int,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    wrap_positions: bool = True,
    target_indices: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
    positions_wrapped_buffer: wp.array | None = None,
    per_atom_cell_offsets_buffer: wp.array | None = None,
    inv_cell_buffer: wp.array | None = None,
    native_strategy: str = "auto",
    # Deprecated kwarg aliases:
    positions_wrapped: wp.array | None = None,
    per_atom_cell_offsets: wp.array | None = None,
    inv_cell: wp.array | None = None,
) -> None:
    """Core warp launcher for naive neighbor matrix construction with PBC.

    Computes neighbor relationships between atoms across periodic boundaries.
    Internally dispatches between two kernel families based on ``device``: a
    scalar SIMT kernel on CPU and a tile-cooperative kernel on CUDA
    (``wp.launch_tiled`` with ``block_dim = BLOCK_DIM``).  Both produce
    identical pair sets and shift vectors; per-row ordering inside
    ``neighbor_matrix`` may differ.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Cell matrix defining lattice vectors in Cartesian coordinates.
    shift_range : wp.array, shape (1, 3), dtype=wp.vec3i
        Shift range per dimension for the single system.
    num_shifts : int
        Number of periodic shifts for the single system.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors, 3), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store relationships where i < j.
    rebuild_flags : wp.array, shape (1,), dtype=wp.bool, optional
        When provided, the kernel checks this flag on the GPU and skips
        work when False (no CPU-GPU sync).
    wrap_positions : bool, default=True
        If True, wrap input positions into the primary cell before
        neighbor search.  When False the positions are assumed to be already
        wrapped (e.g. by a preceding integration step).
    target_indices : wp.array, shape (M,), dtype=wp.int32, optional
        Unique, in-bounds global atom indices restricting which atoms act as
        sources.  Output rows follow ``target_indices``.
    return_vectors : bool, default=False
        If True, write per-pair displacement vectors (including the periodic
        shift contribution) into ``neighbor_vectors``.
    return_distances : bool, default=False
        If True, write per-pair Euclidean distances into
        ``neighbor_distances``.
    pair_fn : wp.Function, optional
        Module-scope ``@wp.func`` with signature
        ``pair_fn(r_ij, distance, pair_params, i, j) -> (energy, force)``.  Keyed
        by object identity.
    pair_params : wp.array, shape (num_atoms, num_parameters), dtype=positions.dtype, optional
        Per-atom parameter table passed to ``pair_fn``.  ``pair_params[i]``
        is the parameter row of length ``num_parameters`` belonging to atom
        ``i`` and ``pair_fn`` may read any ``pair_params[j]`` row it needs
        (e.g. for Lorentz-Berthelot mixing in the Lennard-Jones pair
        potential).
    neighbor_vectors : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*, optional
        OUTPUT: Required if ``return_vectors=True``.
    neighbor_distances : wp.array, shape (rows, max_neighbors), dtype=wp.float*, optional
        OUTPUT: Required if ``return_distances=True``.
    pair_energies : wp.array, shape (rows, max_neighbors), dtype=wp.float*, optional
        OUTPUT: Required if ``pair_fn`` is provided.
    pair_forces : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*, optional
        OUTPUT: Required if ``pair_fn`` is provided.
    positions_wrapped_buffer : wp.array, shape (total_atoms,), dtype=wp.vec3*, optional
        Caller-supplied scratch buffer for wrapped positions (only used
        when ``wrap_positions=True``).  Treated as scratch — overwritten on
        every call.  When omitted the launcher allocates a fresh buffer
        for the call.
    per_atom_cell_offsets_buffer : wp.array, shape (total_atoms,), dtype=wp.vec3i, optional
        Caller-supplied scratch buffer for per-atom cell offsets
        (only used when ``wrap_positions=True``).
    inv_cell_buffer : wp.array, shape (num_systems,), dtype=wp.mat33*, optional
        Caller-supplied scratch buffer for inverse cell matrices
        (only used when ``wrap_positions=True``).
    positions_wrapped, per_atom_cell_offsets, inv_cell : deprecated
        Deprecated aliases of the ``*_buffer`` kwargs above.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - When ``wrap_positions`` is True, positions are wrapped into the primary
      cell in a preprocessing step before the neighbor search kernel.
    - The scratch buffers used for the wrap step
      (``positions_wrapped_buffer``, ``per_atom_cell_offsets_buffer``,
      ``inv_cell_buffer``) may be supplied by the caller to eliminate
      per-call allocation; their contents are overwritten on every call.
      When omitted the launcher allocates a fresh buffer for the call.
    - The CUDA path uses ``wp.launch_tiled(block_dim=BLOCK_DIM)``; CPU is
      forced to ``block_dim = 1`` by Warp, so CPU callers take the scalar path.
    - When any of the pair-output kwargs is supplied, the scalar factory
      kernel is used (no tile variant for the pair-output kwargs).

    See Also
    --------
    naive_neighbor_matrix : Version without periodic boundary conditions
    batch_naive_neighbor_matrix_pbc : Batched (multi-system) variant
    get_naive_neighbor_matrix_kernel : Low-level single-cutoff kernel accessor
    """
    positions_wrapped_buffer = resolve_buffer_alias(
        "positions_wrapped_buffer",
        positions_wrapped_buffer,
        "positions_wrapped",
        positions_wrapped,
    )
    per_atom_cell_offsets_buffer = resolve_buffer_alias(
        "per_atom_cell_offsets_buffer",
        per_atom_cell_offsets_buffer,
        "per_atom_cell_offsets",
        per_atom_cell_offsets,
    )
    inv_cell_buffer = resolve_buffer_alias(
        "inv_cell_buffer",
        inv_cell_buffer,
        "inv_cell",
        inv_cell,
    )
    if not _has_naive_pair_outputs(
        target_indices,
        return_vectors,
        return_distances,
        pair_fn,
        pair_params,
        neighbor_vectors,
        neighbor_distances,
        pair_energies,
        pair_forces,
    ):
        _launch_naive_neighbor_matrix_pbc(
            positions,
            cutoff,
            cell,
            shift_range,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            wp_dtype,
            device,
            batched=False,
            num_shifts=num_shifts,
            half_fill=half_fill,
            rebuild_flags=rebuild_flags,
            wrap_positions=wrap_positions,
            positions_wrapped_buffer=positions_wrapped_buffer,
            per_atom_cell_offsets_buffer=per_atom_cell_offsets_buffer,
            inv_cell_buffer=inv_cell_buffer,
        )
        return
    _launch_naive_neighbor_matrix_pbc(
        positions,
        cutoff,
        cell,
        shift_range,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        wp_dtype,
        device,
        num_shifts=num_shifts,
        half_fill=half_fill,
        rebuild_flags=rebuild_flags,
        wrap_positions=wrap_positions,
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
        batched=False,
        positions_wrapped_buffer=positions_wrapped_buffer,
        per_atom_cell_offsets_buffer=per_atom_cell_offsets_buffer,
        inv_cell_buffer=inv_cell_buffer,
        native_strategy=native_strategy,
    )


def batch_naive_neighbor_matrix_pbc(
    positions: wp.array,
    cell: wp.array,
    cutoff: float,
    batch_ptr: wp.array,
    batch_idx: wp.array,
    shift_range: wp.array,
    num_shifts_arr: wp.array,
    max_shifts_per_system: int,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    max_atoms_per_system: int,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    wrap_positions: bool = True,
    target_indices: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
    positions_wrapped_buffer: wp.array | None = None,
    per_atom_cell_offsets_buffer: wp.array | None = None,
    inv_cell_buffer: wp.array | None = None,
    native_strategy: str = "auto",
    # Deprecated kwarg aliases:
    positions_wrapped: wp.array | None = None,
    per_atom_cell_offsets: wp.array | None = None,
    inv_cell: wp.array | None = None,
) -> None:
    """Core warp launcher for batched naive neighbor matrix construction with PBC.

    Computes neighbor relationships between atoms across periodic boundaries
    for multiple systems in a batch using pure warp operations.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated Cartesian coordinates for all systems.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Cell matrices for each system.
    cutoff : float
        Cutoff distance for neighbor detection.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Shift range per dimension per system.
    num_shifts_arr : wp.array, shape (num_systems,), dtype=wp.int32
        Number of shifts per system.
    max_shifts_per_system : int
        Maximum per-system shift count (launch dimension).
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors, 3), dtype=wp.vec3i
        OUTPUT: Shift vectors for each neighbor.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors per atom.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    max_atoms_per_system : int
        Maximum number of atoms in any single system.
    half_fill : bool, default=False
        If True, only store half of the neighbor relationships.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool, optional
        Per-system rebuild flags.
    wrap_positions : bool, default=True
        If True, wrap input positions into the primary cell.
    target_indices : wp.array, shape (M,), dtype=wp.int32, optional
        Unique, in-bounds global atom indices restricting which atoms act as
        sources.  In batched mode each target searches only atoms in its own
        system (resolved via ``batch_idx``).  Output rows follow
        ``target_indices``.
    return_vectors : bool, default=False
        If True, write per-pair displacement vectors (including the periodic
        shift contribution) into ``neighbor_vectors``.
    return_distances : bool, default=False
        If True, write per-pair Euclidean distances into
        ``neighbor_distances``.
    pair_fn : wp.Function, optional
        Module-scope ``@wp.func`` with signature
        ``pair_fn(r_ij, distance, pair_params, i, j) -> (energy, force)``.  Keyed
        by object identity.
    pair_params : wp.array, shape (num_atoms, num_parameters), dtype=positions.dtype, optional
        Per-atom parameter table passed to ``pair_fn``.  ``pair_params[i]``
        is the parameter row of length ``num_parameters`` belonging to atom
        ``i`` and ``pair_fn`` may read any ``pair_params[j]`` row it needs
        (e.g. for Lorentz-Berthelot mixing in the Lennard-Jones pair
        potential).
    neighbor_vectors : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*, optional
        OUTPUT: Required if ``return_vectors=True``.
    neighbor_distances : wp.array, shape (rows, max_neighbors), dtype=wp.float*, optional
        OUTPUT: Required if ``return_distances=True``.
    pair_energies : wp.array, shape (rows, max_neighbors), dtype=wp.float*, optional
        OUTPUT: Required if ``pair_fn`` is provided.
    pair_forces : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*, optional
        OUTPUT: Required if ``pair_fn`` is provided.
    positions_wrapped_buffer : wp.array, shape (total_atoms,), dtype=wp.vec3*, optional
        Caller-supplied scratch for wrapped positions (used when
        ``wrap_positions=True``).  Optional — the launcher allocates when
        omitted.
    per_atom_cell_offsets_buffer : wp.array, shape (total_atoms,), dtype=wp.vec3i, optional
        Caller-supplied scratch for per-atom cell offsets.
    inv_cell_buffer : wp.array, shape (num_systems,), dtype=wp.mat33*, optional
        Caller-supplied scratch for inverse cell matrices.
    positions_wrapped, per_atom_cell_offsets, inv_cell : deprecated
        Deprecated aliases of the ``*_buffer`` kwargs above.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - When ``wrap_positions`` is True, positions are wrapped into the primary
      cell in a preprocessing step before the neighbor search kernel.
    - The scratch buffers used for the wrap step may be supplied by the
      caller (``positions_wrapped_buffer``, ``per_atom_cell_offsets_buffer``,
      ``inv_cell_buffer``) to eliminate per-call allocation; when omitted the
      launcher allocates fresh per call (batched callers do not share the
      single-system cache).
    - Default calls dispatch internally:

      * On CPU, use the scalar 3D-launch kernels.
      * On CUDA with ``wrap_positions=True``, use the tile-cooperative kernel
        when the adaptive ``use_tiled`` heuristic favours it
        (``256 <= avg_atoms_per_system < 6144`` and either
        ``avg_atoms_per_system >= 2048`` or ``total_atoms <= 8192``);
        otherwise fall back to the scalar 3D-launch kernel.
      * When ``wrap_positions=False`` the prewrapped scalar kernels are used
        on both devices (no tiled prewrapped variant).
    - When any of the pair-output kwargs is supplied, the scalar factory
      kernel is used (no tile variant for the pair-output kwargs).

    See Also
    --------
    batch_naive_neighbor_matrix : Version without periodic boundary conditions
    naive_neighbor_matrix_pbc : Single-system variant
    get_naive_neighbor_matrix_kernel : Low-level single-cutoff kernel accessor
    """
    positions_wrapped_buffer = resolve_buffer_alias(
        "positions_wrapped_buffer",
        positions_wrapped_buffer,
        "positions_wrapped",
        positions_wrapped,
    )
    per_atom_cell_offsets_buffer = resolve_buffer_alias(
        "per_atom_cell_offsets_buffer",
        per_atom_cell_offsets_buffer,
        "per_atom_cell_offsets",
        per_atom_cell_offsets,
    )
    inv_cell_buffer = resolve_buffer_alias(
        "inv_cell_buffer",
        inv_cell_buffer,
        "inv_cell",
        inv_cell,
    )
    if not _has_naive_pair_outputs(
        target_indices,
        return_vectors,
        return_distances,
        pair_fn,
        pair_params,
        neighbor_vectors,
        neighbor_distances,
        pair_energies,
        pair_forces,
    ):
        if rebuild_flags is not None:
            selective_zero_num_neighbors(
                num_neighbors, batch_idx, rebuild_flags, device
            )
        _launch_naive_neighbor_matrix_pbc(
            positions,
            cutoff,
            cell,
            shift_range,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            wp_dtype,
            device,
            batched=True,
            batch_ptr=batch_ptr,
            batch_idx=batch_idx,
            num_shifts_arr=num_shifts_arr,
            max_shifts_per_system=max_shifts_per_system,
            max_atoms_per_system=max_atoms_per_system,
            half_fill=half_fill,
            rebuild_flags=rebuild_flags,
            wrap_positions=wrap_positions,
            positions_wrapped_buffer=positions_wrapped_buffer,
            per_atom_cell_offsets_buffer=per_atom_cell_offsets_buffer,
            inv_cell_buffer=inv_cell_buffer,
        )
        return
    if rebuild_flags is not None:
        selective_zero_num_neighbors(num_neighbors, batch_idx, rebuild_flags, device)
    _launch_naive_neighbor_matrix_pbc(
        positions,
        cutoff,
        cell,
        shift_range,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        wp_dtype,
        device,
        batch_ptr=batch_ptr,
        batch_idx=batch_idx,
        num_shifts_arr=num_shifts_arr,
        max_shifts_per_system=max_shifts_per_system,
        max_atoms_per_system=max_atoms_per_system,
        half_fill=half_fill,
        rebuild_flags=rebuild_flags,
        wrap_positions=wrap_positions,
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
        batched=True,
    )


def naive_neighbor_matrix_dual_cutoff(
    positions: wp.array,
    cutoff1: float,
    cutoff2: float,
    neighbor_matrix1: wp.array,
    num_neighbors1: wp.array,
    neighbor_matrix2: wp.array,
    num_neighbors2: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
) -> None:
    """Core warp launcher for naive dual cutoff neighbor matrix construction (no PBC).

    Computes pairwise distances and fills two neighbor matrices with atom
    indices within different cutoff distances using pure warp operations.  No
    periodic boundary conditions are applied.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cutoff1 : float
        First cutoff distance (typically smaller).
    cutoff2 : float
        Second cutoff distance (typically larger).
    neighbor_matrix1 : wp.array, shape (total_atoms, max_neighbors1), dtype=wp.int32
        OUTPUT: First neighbor matrix to be filled.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom within cutoff1.
    neighbor_matrix2 : wp.array, shape (total_atoms, max_neighbors2), dtype=wp.int32
        OUTPUT: Second neighbor matrix to be filled.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom within cutoff2.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store relationships where i < j to avoid double counting.
    rebuild_flags : wp.array, shape (1,), dtype=wp.bool, optional
        Per-system rebuild flags.  If provided, only rebuilds when
        ``rebuild_flags[0]`` is True; otherwise skips on the GPU without
        CPU sync.
    pair_fn : Any, optional
        Not supported in dual-cutoff mode; raises ``ValueError`` if provided.
        Kept in the signature so callers can pass through generic kwargs.
    pair_params : wp.array, optional
        Not supported in dual-cutoff mode; raises ``ValueError`` if provided.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - Dual-cutoff mode does not support pair outputs or
      target-row restriction; ``pair_fn`` / ``pair_params`` raise
      ``ValueError`` if provided, and ``target_indices`` /
      ``return_vectors`` / ``return_distances`` are absent from the
      signature.

    See Also
    --------
    naive_neighbor_matrix_pbc_dual_cutoff : Version with periodic boundary conditions
    batch_naive_neighbor_matrix_dual_cutoff : Batched (multi-system) variant
    get_naive_neighbor_matrix_dual_cutoff_kernel : Low-level dual-cutoff kernel accessor
    """
    _reject_pair_fn_for_dual_cutoff(pair_fn, pair_params)
    _launch_naive_neighbor_matrix_dual_cutoff(
        positions,
        cutoff1,
        cutoff2,
        neighbor_matrix1,
        num_neighbors1,
        neighbor_matrix2,
        num_neighbors2,
        wp_dtype,
        device,
        batched=False,
        half_fill=half_fill,
        rebuild_flags=rebuild_flags,
    )


def naive_neighbor_matrix_pbc_dual_cutoff(
    positions: wp.array,
    cutoff1: float,
    cutoff2: float,
    cell: wp.array,
    shift_range: wp.array,
    num_shifts: int,
    neighbor_matrix1: wp.array,
    neighbor_matrix2: wp.array,
    neighbor_matrix_shifts1: wp.array,
    neighbor_matrix_shifts2: wp.array,
    num_neighbors1: wp.array,
    num_neighbors2: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    wrap_positions: bool = True,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    positions_wrapped_buffer: wp.array | None = None,
    per_atom_cell_offsets_buffer: wp.array | None = None,
    inv_cell_buffer: wp.array | None = None,
    # Deprecated kwarg aliases:
    positions_wrapped: wp.array | None = None,
    per_atom_cell_offsets: wp.array | None = None,
    inv_cell: wp.array | None = None,
) -> None:
    """Core warp launcher for naive dual cutoff neighbor matrix construction with PBC.

    Computes neighbor relationships between atoms across periodic boundaries
    for two different cutoff distances using pure warp operations.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cutoff1 : float
        First cutoff distance (typically smaller).
    cutoff2 : float
        Second cutoff distance (typically larger).
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Cell matrix defining lattice vectors in Cartesian coordinates.
    shift_range : wp.array, shape (1, 3), dtype=wp.vec3i
        Shift range per dimension for the single system.
    num_shifts : int
        Number of periodic shifts for the single system.
    neighbor_matrix1 : wp.array, shape (total_atoms, max_neighbors1), dtype=wp.int32
        OUTPUT: First neighbor matrix to be filled.
    neighbor_matrix2 : wp.array, shape (total_atoms, max_neighbors2), dtype=wp.int32
        OUTPUT: Second neighbor matrix to be filled.
    neighbor_matrix_shifts1 : wp.array, shape (total_atoms, max_neighbors1, 3), dtype=wp.vec3i
        OUTPUT: Shift vectors for first neighbor matrix.
    neighbor_matrix_shifts2 : wp.array, shape (total_atoms, max_neighbors2, 3), dtype=wp.vec3i
        OUTPUT: Shift vectors for second neighbor matrix.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom within cutoff1.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom within cutoff2.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store relationships where i < j to avoid double counting.
    rebuild_flags : wp.array, shape (1,), dtype=wp.bool, optional
        Per-system rebuild flags.  If provided, only rebuilds when
        ``rebuild_flags[0]`` is True; otherwise skips on the GPU without
        CPU sync.
    wrap_positions : bool, default=True
        If True, wrap input positions into the primary cell before neighbor
        search.  Set to False when positions are already wrapped (e.g. by a
        preceding integration step) to save two GPU kernel launches per call.
    pair_fn : Any, optional
        Not supported in dual-cutoff mode; raises ``ValueError`` if provided.
    pair_params : wp.array, optional
        Not supported in dual-cutoff mode; raises ``ValueError`` if provided.
    positions_wrapped_buffer : wp.array, shape (total_atoms,), dtype=wp.vec3*, optional
        Caller-supplied scratch for wrapped positions (used when
        ``wrap_positions=True``).  Optional — the launcher allocates when
        omitted.
    per_atom_cell_offsets_buffer : wp.array, shape (total_atoms,), dtype=wp.vec3i, optional
        Caller-supplied scratch for per-atom cell offsets.
    inv_cell_buffer : wp.array, shape (num_systems,), dtype=wp.mat33*, optional
        Caller-supplied scratch for inverse cell matrices.
    positions_wrapped, per_atom_cell_offsets, inv_cell : deprecated
        Deprecated aliases of the ``*_buffer`` kwargs above.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - When ``wrap_positions`` is True, positions are wrapped into the primary
      cell in a preprocessing step before the neighbor search kernel.
    - The scratch buffers used for the wrap step
      (``positions_wrapped_buffer``, ``per_atom_cell_offsets_buffer``,
      ``inv_cell_buffer``) may be supplied by the caller to eliminate
      per-call allocation; their contents are overwritten on every call.
      When omitted the launcher allocates a fresh buffer for the call.
    - Dual-cutoff mode does not support pair outputs or
      target-row restriction; ``pair_fn`` / ``pair_params`` raise
      ``ValueError`` if provided.

    See Also
    --------
    naive_neighbor_matrix_dual_cutoff : Version without periodic boundary conditions
    batch_naive_neighbor_matrix_pbc_dual_cutoff : Batched (multi-system) variant
    get_naive_neighbor_matrix_dual_cutoff_kernel : Low-level dual-cutoff kernel accessor
    """
    _reject_pair_fn_for_dual_cutoff(pair_fn, pair_params)
    positions_wrapped_buffer = resolve_buffer_alias(
        "positions_wrapped_buffer",
        positions_wrapped_buffer,
        "positions_wrapped",
        positions_wrapped,
    )
    per_atom_cell_offsets_buffer = resolve_buffer_alias(
        "per_atom_cell_offsets_buffer",
        per_atom_cell_offsets_buffer,
        "per_atom_cell_offsets",
        per_atom_cell_offsets,
    )
    inv_cell_buffer = resolve_buffer_alias(
        "inv_cell_buffer",
        inv_cell_buffer,
        "inv_cell",
        inv_cell,
    )
    _launch_naive_neighbor_matrix_pbc_dual_cutoff(
        positions,
        cutoff1,
        cutoff2,
        cell,
        shift_range,
        neighbor_matrix1,
        neighbor_matrix2,
        neighbor_matrix_shifts1,
        neighbor_matrix_shifts2,
        num_neighbors1,
        num_neighbors2,
        wp_dtype,
        device,
        batched=False,
        num_shifts=num_shifts,
        half_fill=half_fill,
        rebuild_flags=rebuild_flags,
        wrap_positions=wrap_positions,
        positions_wrapped_buffer=positions_wrapped_buffer,
        per_atom_cell_offsets_buffer=per_atom_cell_offsets_buffer,
        inv_cell_buffer=inv_cell_buffer,
    )


def batch_naive_neighbor_matrix_dual_cutoff(
    positions: wp.array,
    cutoff1: float,
    cutoff2: float,
    batch_idx: wp.array,
    batch_ptr: wp.array,
    neighbor_matrix1: wp.array,
    num_neighbors1: wp.array,
    neighbor_matrix2: wp.array,
    num_neighbors2: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
) -> None:
    """Core warp launcher for batched naive dual cutoff neighbor matrix construction (no PBC).

    Computes pairwise distances and fills two neighbor matrices with atom
    indices within different cutoff distances for multiple systems in a
    batch.  No periodic boundary conditions are applied.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated Cartesian coordinates for all systems.
    cutoff1 : float
        First cutoff distance (typically smaller).
    cutoff2 : float
        Second cutoff distance (typically larger).
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    neighbor_matrix1 : wp.array, shape (total_atoms, max_neighbors1), dtype=wp.int32
        OUTPUT: First neighbor matrix.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff1.
    neighbor_matrix2 : wp.array, shape (total_atoms, max_neighbors2), dtype=wp.int32
        OUTPUT: Second neighbor matrix.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff2.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store relationships where i < j.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool, optional
        Per-system rebuild flags.  If provided, only systems where
        ``rebuild_flags[i]`` is True are processed; others are skipped on the
        GPU without CPU sync.  Per-system counters are reset via
        :func:`selective_zero_num_neighbors` for both ``num_neighbors1`` and
        ``num_neighbors2`` internally.
    pair_fn : Any, optional
        Not supported in dual-cutoff mode; raises ``ValueError`` if provided.
    pair_params : wp.array, optional
        Not supported in dual-cutoff mode; raises ``ValueError`` if provided.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - Dual-cutoff mode does not support pair outputs or
      target-row restriction; ``pair_fn`` / ``pair_params`` raise
      ``ValueError`` if provided.

    See Also
    --------
    batch_naive_neighbor_matrix_pbc_dual_cutoff : Version with PBC
    naive_neighbor_matrix_dual_cutoff : Single-system variant
    get_naive_neighbor_matrix_dual_cutoff_kernel : Low-level dual-cutoff kernel accessor
    """
    _reject_pair_fn_for_dual_cutoff(pair_fn, pair_params)
    if rebuild_flags is not None:
        selective_zero_num_neighbors(num_neighbors1, batch_idx, rebuild_flags, device)
        selective_zero_num_neighbors(num_neighbors2, batch_idx, rebuild_flags, device)
    _launch_naive_neighbor_matrix_dual_cutoff(
        positions,
        cutoff1,
        cutoff2,
        neighbor_matrix1,
        num_neighbors1,
        neighbor_matrix2,
        num_neighbors2,
        wp_dtype,
        device,
        batched=True,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        half_fill=half_fill,
        rebuild_flags=rebuild_flags,
    )


def batch_naive_neighbor_matrix_pbc_dual_cutoff(
    positions: wp.array,
    cell: wp.array,
    cutoff1: float,
    cutoff2: float,
    batch_ptr: wp.array,
    batch_idx: wp.array,
    shift_range: wp.array,
    num_shifts_arr: wp.array,
    max_shifts_per_system: int,
    neighbor_matrix1: wp.array,
    neighbor_matrix2: wp.array,
    neighbor_matrix_shifts1: wp.array,
    neighbor_matrix_shifts2: wp.array,
    num_neighbors1: wp.array,
    num_neighbors2: wp.array,
    wp_dtype: type,
    device: str,
    max_atoms_per_system: int,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    wrap_positions: bool = True,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    positions_wrapped_buffer: wp.array | None = None,
    per_atom_cell_offsets_buffer: wp.array | None = None,
    inv_cell_buffer: wp.array | None = None,
    # Deprecated kwarg aliases:
    positions_wrapped: wp.array | None = None,
    per_atom_cell_offsets: wp.array | None = None,
    inv_cell: wp.array | None = None,
) -> None:
    """Core warp launcher for batched naive dual cutoff neighbor matrix construction with PBC.

    Computes neighbor relationships between atoms across periodic boundaries
    for two different cutoff distances and multiple systems in a batch.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated Cartesian coordinates for all systems.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Cell matrices for each system in the batch.
    cutoff1 : float
        First cutoff distance (typically smaller).
    cutoff2 : float
        Second cutoff distance (typically larger).
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.  Required for the position-wrapping
        preprocessing step that maps atoms to their system's cell.
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Shift range per dimension per system.
    num_shifts_arr : wp.array, shape (num_systems,), dtype=wp.int32
        Number of shifts per system.
    max_shifts_per_system : int
        Maximum per-system shift count (launch dimension).
    neighbor_matrix1 : wp.array, shape (total_atoms, max_neighbors1), dtype=wp.int32
        OUTPUT: First neighbor matrix.
    neighbor_matrix2 : wp.array, shape (total_atoms, max_neighbors2), dtype=wp.int32
        OUTPUT: Second neighbor matrix.
    neighbor_matrix_shifts1 : wp.array, shape (total_atoms, max_neighbors1, 3), dtype=wp.vec3i
        OUTPUT: Shift vectors for first neighbor matrix.
    neighbor_matrix_shifts2 : wp.array, shape (total_atoms, max_neighbors2, 3), dtype=wp.vec3i
        OUTPUT: Shift vectors for second neighbor matrix.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff1.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff2.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    max_atoms_per_system : int
        Maximum number of atoms in any single system.
    half_fill : bool, default=False
        If True, only store half of the neighbor relationships.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool, optional
        Per-system rebuild flags.  If provided, only systems where
        ``rebuild_flags[i]`` is True are processed; others are skipped on the
        GPU without CPU sync.  Per-system counters are reset via
        :func:`selective_zero_num_neighbors` for both ``num_neighbors1`` and
        ``num_neighbors2`` internally.
    wrap_positions : bool, default=True
        If True, wrap input positions into the primary cell before
        neighbor search.  Set to False when positions are already wrapped
        (e.g. by a preceding integration step) to save two GPU kernel
        launches per call.
    pair_fn : Any, optional
        Not supported in dual-cutoff mode; raises ``ValueError`` if provided.
    pair_params : wp.array, optional
        Not supported in dual-cutoff mode; raises ``ValueError`` if provided.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - When ``wrap_positions`` is True, positions are wrapped into the primary
      cell in a preprocessing step before the neighbor search kernel.
    - Dual-cutoff mode does not support pair outputs or
      target-row restriction; ``pair_fn`` / ``pair_params`` raise
      ``ValueError`` if provided.

    See Also
    --------
    batch_naive_neighbor_matrix_dual_cutoff : Version without PBC
    naive_neighbor_matrix_pbc_dual_cutoff : Single-system variant
    get_naive_neighbor_matrix_dual_cutoff_kernel : Low-level dual-cutoff kernel accessor
    """
    _reject_pair_fn_for_dual_cutoff(pair_fn, pair_params)
    positions_wrapped_buffer = resolve_buffer_alias(
        "positions_wrapped_buffer",
        positions_wrapped_buffer,
        "positions_wrapped",
        positions_wrapped,
    )
    per_atom_cell_offsets_buffer = resolve_buffer_alias(
        "per_atom_cell_offsets_buffer",
        per_atom_cell_offsets_buffer,
        "per_atom_cell_offsets",
        per_atom_cell_offsets,
    )
    inv_cell_buffer = resolve_buffer_alias(
        "inv_cell_buffer",
        inv_cell_buffer,
        "inv_cell",
        inv_cell,
    )
    if rebuild_flags is not None:
        selective_zero_num_neighbors(num_neighbors1, batch_idx, rebuild_flags, device)
        selective_zero_num_neighbors(num_neighbors2, batch_idx, rebuild_flags, device)
    _launch_naive_neighbor_matrix_pbc_dual_cutoff(
        positions,
        cutoff1,
        cutoff2,
        cell,
        shift_range,
        neighbor_matrix1,
        neighbor_matrix2,
        neighbor_matrix_shifts1,
        neighbor_matrix_shifts2,
        num_neighbors1,
        num_neighbors2,
        wp_dtype,
        device,
        batched=True,
        batch_ptr=batch_ptr,
        batch_idx=batch_idx,
        num_shifts_arr=num_shifts_arr,
        max_shifts_per_system=max_shifts_per_system,
        max_atoms_per_system=max_atoms_per_system,
        half_fill=half_fill,
        rebuild_flags=rebuild_flags,
        wrap_positions=wrap_positions,
        positions_wrapped_buffer=positions_wrapped_buffer,
        per_atom_cell_offsets_buffer=per_atom_cell_offsets_buffer,
        inv_cell_buffer=inv_cell_buffer,
    )
