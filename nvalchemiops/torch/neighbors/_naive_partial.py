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
    """Validate a compact naive-neighbor request without touching outputs."""
    if target_indices.ndim != 1 or target_indices.dtype != torch.int32:
        raise ValueError("target_indices must be a rank-one int32 tensor.")
    if target_indices.device != positions.device:
        raise ValueError("target_indices must be on the same device as positions.")
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
    """Validate, allocate, and reset compact topology output buffers."""
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
    """Pack compact matrix outputs into the public matrix or COO contract."""
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
