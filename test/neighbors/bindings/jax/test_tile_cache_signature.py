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

"""Check cached JAX tile callables against scalar neighbor-list results."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list
from nvalchemiops.jax.neighbors.naive import naive_neighbor_list
from nvalchemiops.jax.neighbors.neighbor_utils import compute_naive_num_shifts

from .conftest import requires_gpu

pytestmark = requires_gpu

_CUTOFF = 1.0
_MAX_NEIGHBORS = 16
_FILL_VALUE = 6


def _single_inputs(pbc_mode: str):
    """Build a small single-system case with both direct and image pairs."""
    positions = jnp.array(
        [
            [0.2, 0.2, 0.2],
            [9.6, 0.2, 0.2],
            [4.0, 4.0, 4.0],
            [4.8, 4.0, 4.0],
        ],
        dtype=jnp.float32,
    )
    if pbc_mode == "none":
        return positions, {}

    cell = jnp.eye(3, dtype=positions.dtype)[None, :, :] * 10.0
    pbc = jnp.ones((1, 3), dtype=jnp.bool_)
    shifts, num_shifts, max_shifts = compute_naive_num_shifts(cell, _CUTOFF, pbc)
    return positions, {
        "cell": cell,
        "pbc": pbc,
        "shift_range_per_dimension": shifts,
        "num_shifts_per_system": num_shifts,
        "max_shifts_per_system": int(max_shifts),
        "wrap_positions": pbc_mode == "wrapped",
    }


def _batch_inputs(pbc_mode: str):
    """Build two independent systems with direct and periodic image pairs."""
    positions = jnp.array(
        [
            [0.2, 0.2, 0.2],
            [9.6, 0.2, 0.2],
            [4.0, 4.0, 4.0],
            [0.2, 2.2, 0.2],
            [9.6, 2.2, 0.2],
            [4.0, 6.0, 4.0],
        ],
        dtype=jnp.float32,
    )
    batch_idx = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
    batch_ptr = jnp.array([0, 3, 6], dtype=jnp.int32)
    if pbc_mode == "none":
        return positions, batch_idx, batch_ptr, {}

    cell = jnp.stack([jnp.eye(3, dtype=positions.dtype) * 10.0] * 2)
    pbc = jnp.ones((2, 3), dtype=jnp.bool_)
    shifts, num_shifts, max_shifts = compute_naive_num_shifts(cell, _CUTOFF, pbc)
    return (
        positions,
        batch_idx,
        batch_ptr,
        {
            "cell": cell,
            "pbc": pbc,
            "shift_range_per_dimension": shifts,
            "num_shifts_per_system": num_shifts,
            "max_shifts_per_system": int(max_shifts),
            "max_atoms_per_system": 3,
            "wrap_positions": pbc_mode == "wrapped",
        },
    )


def _same_active_pairs(tile_result, scalar_result, has_pbc: bool) -> None:
    """Compare counts and active neighbor/image multisets row by row."""
    tile_matrix, tile_counts = tile_result[:2]
    scalar_matrix, scalar_counts = scalar_result[:2]
    np.testing.assert_array_equal(np.asarray(tile_counts), np.asarray(scalar_counts))

    tile_shifts = np.asarray(tile_result[2]) if has_pbc else None
    scalar_shifts = np.asarray(scalar_result[2]) if has_pbc else None
    tile_matrix = np.asarray(tile_matrix)
    scalar_matrix = np.asarray(scalar_matrix)
    for row, count in enumerate(np.asarray(scalar_counts)):
        count = int(count)
        scalar_pairs = []
        tile_pairs = []
        for col in range(count):
            scalar_pair = (int(scalar_matrix[row, col]),)
            tile_pair = (int(tile_matrix[row, col]),)
            if has_pbc:
                scalar_pair += (tuple(int(v) for v in scalar_shifts[row, col]),)
                tile_pair += (tuple(int(v) for v in tile_shifts[row, col]),)
            scalar_pairs.append(scalar_pair)
            tile_pairs.append(tile_pair)
        assert sorted(tile_pairs) == sorted(scalar_pairs)


@pytest.mark.gpu
@pytest.mark.parametrize("execution", ["eager", "jit"])
@pytest.mark.parametrize(
    ("pbc_mode", "targeted"),
    [
        ("none", False),
        ("none", True),
        ("wrapped", False),
        ("wrapped", True),
        ("prewrapped", False),
        ("prewrapped", True),
    ],
)
def test_single_tile_cache_matches_scalar(execution, pbc_mode, targeted):
    """Cached single tile callbacks preserve full and compact topology."""
    positions, pbc_kwargs = _single_inputs(pbc_mode)
    targets = jnp.array([2, 0], dtype=jnp.int32) if targeted else None
    rows = int(targets.shape[0]) if targeted else positions.shape[0]
    matrix = jnp.full((rows, _MAX_NEIGHBORS), _FILL_VALUE, dtype=jnp.int32)
    counts = jnp.zeros((rows,), dtype=jnp.int32)
    shifts = (
        jnp.zeros((rows, _MAX_NEIGHBORS, 3), dtype=jnp.int32)
        if pbc_mode != "none"
        else None
    )

    def call(pos, nm, nn, nms, target_indices, strategy):
        return naive_neighbor_list(
            pos,
            _CUTOFF,
            max_neighbors=_MAX_NEIGHBORS,
            fill_value=_FILL_VALUE,
            neighbor_matrix=nm,
            num_neighbors=nn,
            neighbor_matrix_shifts=nms,
            target_indices=target_indices,
            strategy=strategy,
            **pbc_kwargs,
        )

    scalar = call(positions, matrix, counts, shifts, targets, "scalar")
    if execution == "jit":
        compiled_call = jax.jit(
            lambda pos, nm, nn, nms, target_indices: call(
                pos, nm, nn, nms, target_indices if targeted else None, "tile"
            )
        )
        tile = compiled_call(positions, matrix, counts, shifts, targets)
    else:
        tile = call(positions, matrix, counts, shifts, targets, "tile")

    _same_active_pairs(tile, scalar, has_pbc=pbc_mode != "none")


@pytest.mark.gpu
@pytest.mark.parametrize("execution", ["eager", "jit"])
@pytest.mark.parametrize(
    ("pbc_mode", "targeted"),
    [
        ("none", False),
        ("none", True),
        ("wrapped", False),
        ("wrapped", True),
        ("prewrapped", True),
    ],
)
def test_batch_tile_cache_matches_scalar(execution, pbc_mode, targeted):
    """Cached batched tile callbacks preserve counts, pairs, and images."""
    positions, batch_idx, batch_ptr, pbc_kwargs = _batch_inputs(pbc_mode)
    targets = jnp.array([4, 0], dtype=jnp.int32) if targeted else None
    rows = int(targets.shape[0]) if targeted else positions.shape[0]
    matrix = jnp.full((rows, _MAX_NEIGHBORS), _FILL_VALUE, dtype=jnp.int32)
    counts = jnp.zeros((rows,), dtype=jnp.int32)
    shifts = (
        jnp.zeros((rows, _MAX_NEIGHBORS, 3), dtype=jnp.int32)
        if pbc_mode != "none"
        else None
    )

    def call(pos, nm, nn, nms, target_indices, strategy):
        return batch_naive_neighbor_list(
            pos,
            _CUTOFF,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=_MAX_NEIGHBORS,
            fill_value=_FILL_VALUE,
            neighbor_matrix=nm,
            num_neighbors=nn,
            neighbor_matrix_shifts=nms,
            target_indices=target_indices,
            strategy=strategy,
            **pbc_kwargs,
        )

    scalar = call(positions, matrix, counts, shifts, targets, "scalar")
    if execution == "jit":
        compiled_call = jax.jit(
            lambda pos, nm, nn, nms, target_indices: call(
                pos, nm, nn, nms, target_indices if targeted else None, "tile"
            )
        )
        tile = compiled_call(positions, matrix, counts, shifts, targets)
    else:
        tile = call(positions, matrix, counts, shifts, targets, "tile")

    _same_active_pairs(tile, scalar, has_pbc=pbc_mode != "none")
