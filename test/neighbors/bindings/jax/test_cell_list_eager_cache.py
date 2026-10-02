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

"""Real eager calls reuse compiled cell kernels while reading current arrays."""

import cProfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors.batch_cell_list import batch_cell_list
from nvalchemiops.jax.neighbors.cell_list import cell_list

from .conftest import requires_gpu

pytestmark = requires_gpu


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_warmed_cell_calls_reuse_compilation_with_new_positions(compact, batched):
    """Repeated shapes reuse code and changed coordinates change the neighbors."""
    first = jnp.array([[0, 0, 0], [0.4, 0, 0], [2, 0, 0]] * 2, jnp.float32)
    second = jnp.array([[2, 0, 0], [0, 0, 0], [0.4, 0, 0]] * 2, jnp.float32)
    kwargs = dict(
        cutoff=0.75,
        cell=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32) * 8, (2, 3, 3)),
        pbc=jnp.ones((2, 3), dtype=jnp.bool_),
        batch_idx=jnp.array([0, 0, 0, 1, 1, 1], jnp.int32),
        batch_ptr=jnp.array([0, 3, 6], jnp.int32),
        max_neighbors=16,
        strategy="atom_centric",
        return_neighbor_list=compact,
    )
    call = batch_cell_list if batched else cell_list
    num_systems = 2 if batched else 1
    if not batched:
        first, second = first[:3], second[:3]
        kwargs["cell"] = kwargs["cell"][0]
        kwargs["pbc"] = kwargs["pbc"][0]
        del kwargs["batch_idx"], kwargs["batch_ptr"]
    for positions in (first, second):
        jax.block_until_ready(call(positions, **kwargs))

    profiler = cProfile.Profile()
    with profiler:
        results = [call(positions, **kwargs) for positions in (first, second)]
        jax.block_until_ready(results)
    compilations = sum(
        entry.callcount
        for entry in profiler.getstats()
        if getattr(entry.code, "co_name", "") == "backend_compile_and_load"
    )
    assert compilations == 0, f"Warmed cell calls compiled {compilations} times"
    for result, expected in zip(
        results, ([1, 1, 0] * num_systems, [0, 1, 1] * num_systems), strict=True
    ):
        counts = np.diff(np.asarray(result[1])) if compact else np.asarray(result[1])
        np.testing.assert_array_equal(counts, expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("batched", [False, True])
def test_explicit_capacity_atom_calls_keep_cells_on_device(dtype, batched):
    """Explicit cell capacity avoids host reads of fresh eager GPU cells."""
    num_systems = 2 if batched else 1
    positions = jnp.array([[0, 0, 0], [0.4, 0, 0], [2, 0, 0]] * num_systems, dtype)
    kwargs = dict(
        cutoff=0.75,
        pbc=jnp.ones((num_systems, 3), dtype=jnp.bool_),
        max_neighbors=16,
        max_total_cells=128 * num_systems,
        strategy="atom_centric",
    )
    if batched:
        kwargs["batch_idx"] = jnp.array([0, 0, 0, 1, 1, 1], jnp.int32)
        kwargs["batch_ptr"] = jnp.array([0, 3, 6], jnp.int32)
    call = batch_cell_list if batched else cell_list
    warm_cell = jnp.broadcast_to(jnp.eye(3, dtype=dtype) * 8, (num_systems, 3, 3))
    jax.block_until_ready(call(positions, cell=warm_cell, **kwargs))

    for length in (9, 10):
        # A fresh device result avoids a cached host copy from the warmup.
        cell = warm_cell.at[:, 0, 0].set(length)
        with jax.transfer_guard_device_to_host("disallow_explicit"):
            result = call(positions, cell=cell, **kwargs)
            jax.block_until_ready(result)
        np.testing.assert_array_equal(np.asarray(result[1]), [1, 1, 0] * num_systems)
        matrix = np.asarray(result[0])
        shifts = np.asarray(result[2])
        for system in range(num_systems):
            offset = system * 3
            assert matrix[offset, 0] == offset + 1
            assert matrix[offset + 1, 0] == offset
            np.testing.assert_array_equal(shifts[offset : offset + 2, 0], 0)
