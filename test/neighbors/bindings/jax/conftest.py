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

"""Pytest configuration for JAX neighbor list tests."""

from __future__ import annotations

from importlib import import_module

import pytest

pytest.importorskip("jax", reason="No JAX installed.")

import jax
import jax.numpy as jnp
import numpy as np

# Enable float64 support
jax.config.update("jax_enable_x64", True)


requires_gpu = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()),
    reason="JAX Warp bindings require GPU",
)

try:
    _ = import_module("vesin")
    VESIN_AVAILABLE = True
except ModuleNotFoundError:
    VESIN_AVAILABLE = False

requires_vesin = pytest.mark.skipif(
    not VESIN_AVAILABLE, reason="`vesin` required for consistency checks."
)


def create_simple_cubic_system_jax(
    num_atoms: int = 8,
    cell_size: float = 2.0,
    dtype=jnp.float32,
):
    """Create a simple cubic system with JAX arrays."""
    n_side = int(round(num_atoms ** (1 / 3)))
    if n_side**3 != num_atoms:
        n_side = int(np.ceil(num_atoms ** (1 / 3)))

    coords = []
    spacing = cell_size / n_side
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                if len(coords) < num_atoms:
                    coords.append([i * spacing, j * spacing, k * spacing])

    positions = jnp.array(coords[:num_atoms], dtype=dtype)
    cell = (jnp.eye(3, dtype=dtype) * cell_size).reshape(1, 3, 3)
    pbc = jnp.array([[True, True, True]])

    return positions, cell, pbc


def create_batch_idx_and_ptr_jax(atoms_per_system: list[int]):
    """Create batch_idx and batch_ptr arrays for JAX."""
    total_atoms = sum(atoms_per_system)
    batch_idx = jnp.zeros(total_atoms, dtype=jnp.int32)
    batch_ptr_list = [0]

    start = 0
    for i, n in enumerate(atoms_per_system):
        batch_idx = batch_idx.at[start : start + n].set(i)
        start += n
        batch_ptr_list.append(start)

    batch_ptr = jnp.array(batch_ptr_list, dtype=jnp.int32)

    return batch_idx, batch_ptr
