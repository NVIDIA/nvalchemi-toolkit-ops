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

"""Shared validation and buffer handling for Torch compact naive paths."""

from __future__ import annotations

import torch

from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors.neighbor_utils import (
    get_neighbor_list_from_neighbor_matrix as _get_neighbor_list_from_neighbor_matrix,
)

__all__: list[str] = []


def _validate_partial_request(
    positions: torch.Tensor,
    target_indices: torch.Tensor,
    rebuild_flags: torch.Tensor | None,
    *,
    strategy: str,
    has_geometry_or_pair_outputs: bool,
) -> None:
    """Validate compact naive request metadata before output preparation.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Input positions defining the valid atom-index range and device.
    target_indices : torch.Tensor, shape (R,), dtype=torch.int32
        Requested compact source rows.
    rebuild_flags : torch.Tensor or None
        Selective-rebuild flags, which are unsupported for compact rows.
    strategy : {"auto", "scalar", "tile"}
        Requested execution strategy.
    has_geometry_or_pair_outputs : bool
        Whether the request asks for distances, vectors, or pair-callback output.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If row indices have invalid rank, dtype, device, or eager values, or if
        CPU execution explicitly requests the CUDA tile strategy.
    NotImplementedError
        If selective rebuild is requested for compact rows, or tile execution
        is requested with geometry or pair-callback outputs.

    Notes
    -----
    This check does not allocate or reset output buffers. Eager calls check that
    indices are in bounds; value inspection is skipped while compiling.
    """
    if target_indices.ndim != 1 or target_indices.dtype != torch.int32:
        raise ValueError("target_indices must be a rank-one int32 tensor.")
    if target_indices.device != positions.device:
        raise ValueError("target_indices must be on the same device as positions.")
    if not torch.compiler.is_compiling() and bool(
        torch.any((target_indices < 0) | (target_indices >= positions.shape[0]))
    ):
        raise ValueError("target_indices must contain in-bounds atom indices.")
    if rebuild_flags is not None:
        raise NotImplementedError(
            "Partial neighbor lists do not support rebuild_flags",
        )
    if strategy == "tile" and positions.device.type == "cpu":
        raise ValueError(
            "strategy='tile' requires CUDA; use strategy='scalar' or 'auto' on CPU.",
        )
    if strategy == "tile" and has_geometry_or_pair_outputs:
        raise NotImplementedError(
            "strategy='tile' supports topology-only target_indices; "
            "geometry and pair-function outputs require strategy='scalar'.",
        )


def _validate_partial_output(
    name: str,
    tensor: torch.Tensor | None,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> None:
    """Validate an optional compact-row output buffer."""
    if tensor is None:
        return
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; got {tuple(tensor.shape)}.",
        )
    if tensor.dtype != expected_dtype:
        raise ValueError(f"{name} dtype must be {expected_dtype}; got {tensor.dtype}.")
    if tensor.device != expected_device:
        raise ValueError(
            f"{name} must be on the same device as positions "
            f"({expected_device}); got {tensor.device}.",
        )


def _prepare_partial_outputs(
    positions: torch.Tensor,
    target_indices: torch.Tensor,
    cutoff: float,
    *,
    pbc_enabled: bool,
    max_neighbors: int | None,
    fill_value: int | None,
    neighbor_matrix: torch.Tensor | None,
    num_neighbors: torch.Tensor | None,
    neighbor_matrix_shifts: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:
    """Validate, allocate, and reset compact topology output buffers.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Input positions defining output device and default padding value.
    target_indices : torch.Tensor, shape (R,)
        Requested compact source rows.
    cutoff : float
        Cutoff used to estimate capacity when no buffer or capacity is supplied.
    pbc_enabled : bool
        Whether periodic shift rows are required.
    max_neighbors : int or None
        Row capacity, inferred from a supplied matrix or the cutoff when absent.
    fill_value : int or None
        Matrix padding value; defaults to the total atom count.
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts : torch.Tensor or None
        Optional caller-owned topology output buffers.

    Returns
    -------
    neighbor_matrix : torch.Tensor, shape (R, M), dtype=int32
        Validated or newly allocated matrix, reset to ``fill_value``.
    num_neighbors : torch.Tensor, shape (R,), dtype=int32
        Validated or newly allocated row counts, reset to zero.
    neighbor_matrix_shifts : torch.Tensor or None
        Validated or newly allocated zeroed shifts for PBC; ``None`` otherwise.
    max_neighbors : int
        Resolved matrix row capacity.
    num_rows : int
        Number of compact rows, including repeated target indices.

    Raises
    ------
    ValueError
        If a supplied matrix, count, or applicable PBC shift buffer has an
        incompatible shape, dtype, or device.

    Notes
    -----
    All applicable caller buffers are validated before any is reset. Matrix
    padding and counts are then initialized in place; PBC shift buffers are
    zeroed in place, while non-PBC calls return no shift buffer.
    """
    num_rows = int(target_indices.shape[0])
    if max_neighbors is None and neighbor_matrix is not None:
        max_neighbors = int(neighbor_matrix.shape[1])
    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(cutoff)
    if fill_value is None:
        fill_value = int(positions.shape[0])

    _validate_partial_output(
        "neighbor_matrix",
        neighbor_matrix,
        (num_rows, max_neighbors),
        torch.int32,
        positions.device,
    )
    _validate_partial_output(
        "num_neighbors",
        num_neighbors,
        (num_rows,),
        torch.int32,
        positions.device,
    )
    if pbc_enabled:
        _validate_partial_output(
            "neighbor_matrix_shifts",
            neighbor_matrix_shifts,
            (num_rows, max_neighbors, 3),
            torch.int32,
            positions.device,
        )

    if neighbor_matrix is None:
        neighbor_matrix = torch.full(
            (num_rows, max_neighbors),
            fill_value,
            dtype=torch.int32,
            device=positions.device,
        )
    else:
        neighbor_matrix.fill_(fill_value)
    if num_neighbors is None:
        num_neighbors = torch.zeros(
            num_rows,
            dtype=torch.int32,
            device=positions.device,
        )
    else:
        num_neighbors.zero_()
    if pbc_enabled:
        if neighbor_matrix_shifts is None:
            neighbor_matrix_shifts = torch.zeros(
                (num_rows, max_neighbors, 3),
                dtype=torch.int32,
                device=positions.device,
            )
        else:
            neighbor_matrix_shifts.zero_()
    else:
        neighbor_matrix_shifts = None
    return (
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
        max_neighbors,
        num_rows,
    )


def _pack_partial_outputs(
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor | None,
    *,
    fill_value: int,
    return_neighbor_list: bool,
) -> tuple[torch.Tensor, ...]:
    """Pack compact matrix outputs in the public matrix or COO layout.

    Parameters
    ----------
    neighbor_matrix : torch.Tensor, shape (R, M)
        Compact-row neighbor indices.
    num_neighbors : torch.Tensor, shape (R,)
        Raw neighbor count for each compact row.
    neighbor_matrix_shifts : torch.Tensor or None, shape (R, M, 3)
        Periodic shifts for each stored pair, when PBC is enabled.
    fill_value : int
        Padding value excluded from COO conversion.
    return_neighbor_list : bool
        Whether to convert the matrix to the public COO representation.

    Returns
    -------
    tuple of torch.Tensor
        Matrix format returns ``(matrix, counts)`` and includes shifts as a
        third value when present. COO format returns ``(indices, pointer)``
        and likewise appends shifts for PBC.

    Notes
    -----
    Compact row order is preserved. COO source indices are compact row numbers,
    not the original atom indices stored in ``target_indices``.
    """
    if return_neighbor_list:
        if neighbor_matrix_shifts is None:
            neighbor_list, neighbor_ptr = _get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                fill_value=fill_value,
            )
            return neighbor_list, neighbor_ptr
        neighbor_list, neighbor_ptr, neighbor_list_shifts = (
            _get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                neighbor_shift_matrix=neighbor_matrix_shifts,
                fill_value=fill_value,
            )
        )
        return neighbor_list, neighbor_ptr, neighbor_list_shifts
    if neighbor_matrix_shifts is None:
        return neighbor_matrix, num_neighbors
    return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
