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

"""CPU-safe cell geometry validation tests for JAX atom-centric builds."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list
from nvalchemiops.jax.neighbors.cell_list import cell_list


@pytest.mark.parametrize(
    ("cell", "message"),
    [
        (
            jnp.array(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=jnp.float32,
            ),
            "Cell with volume == 0.0",
        ),
        (jnp.eye(3, dtype=jnp.float32).at[0, 0].set(jnp.nan), "finite values"),
        (jnp.eye(3, dtype=jnp.float32).at[0, 0].set(jnp.inf), "finite values"),
    ],
)
def test_atom_centric_cell_list_rejects_invalid_cell(cell, message):
    """Single-system atom-centric builds reject invalid eager cell geometry."""
    positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)

    with pytest.raises(RuntimeError, match=message):
        cell_list(
            positions,
            cutoff=1.0,
            cell=cell,
            pbc=jnp.ones(3, dtype=jnp.bool_),
            max_neighbors=8,
            strategy="atom_centric",
        )


@pytest.mark.parametrize(
    ("invalid_cell", "message"),
    [
        (
            jnp.array(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=jnp.float32,
            ),
            "Cells with volume == 0.0",
        ),
        (jnp.eye(3, dtype=jnp.float32).at[0, 0].set(jnp.nan), "finite values"),
        (jnp.eye(3, dtype=jnp.float32).at[0, 0].set(jnp.inf), "finite values"),
    ],
)
def test_atom_centric_batch_cell_list_rejects_invalid_cell(
    invalid_cell,
    message,
):
    """Batched atom-centric builds reject an invalid eager cell in any system."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    cell = jnp.stack((jnp.eye(3, dtype=jnp.float32), invalid_cell))

    with pytest.raises(RuntimeError, match=message):
        batch_cell_list(
            positions,
            cutoff=1.0,
            cell=cell,
            pbc=jnp.ones((2, 3), dtype=jnp.bool_),
            batch_idx=jnp.array([0, 1], dtype=jnp.int32),
            batch_ptr=jnp.array([0, 1, 2], dtype=jnp.int32),
            max_neighbors=8,
            strategy="atom_centric",
        )
