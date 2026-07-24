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

"""Backend-independent JAX dispatch validation tests."""

from __future__ import annotations

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list
from nvalchemiops.jax.neighbors.naive import naive_neighbor_list


def test_jitted_cpu_trace_preserves_cpu_dispatch():
    """A default-CPU jit trace is recognized as CPU before tile lowering."""
    script = """
import jax
import jax.numpy as jnp

from nvalchemiops.jax.neighbors._dispatch import _is_jax_cpu_array


@jax.jit
def classify(positions):
    return _is_jax_cpu_array(positions)


assert bool(classify(jnp.zeros((2, 3), dtype=jnp.float32)))
"""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("batched", [False, True])
def test_cpu_tile_rejects_before_launch(batched):
    """Explicit CPU tile requests fail before any native launch."""
    cpu = jax.local_devices(backend="cpu")[0]
    positions = jax.device_put(jnp.zeros((4, 3), dtype=jnp.float32), cpu)
    targets = jax.device_put(jnp.array([0, 2], dtype=jnp.int32), cpu)
    call = batch_naive_neighbor_list if batched else naive_neighbor_list
    kwargs = (
        {
            "batch_idx": jax.device_put(jnp.zeros((4,), dtype=jnp.int32), cpu),
            "batch_ptr": jax.device_put(jnp.array([0, 4], dtype=jnp.int32), cpu),
        }
        if batched
        else {}
    )

    with pytest.raises(ValueError, match="requires CUDA"):
        call(
            positions,
            1.0,
            max_neighbors=8,
            target_indices=targets,
            strategy="tile",
            **kwargs,
        )


def test_single_target_indices_validate_before_dispatch():
    """Single-system target indices use the same ABI validation as batch."""
    cpu = jax.local_devices(backend="cpu")[0]
    with pytest.raises(ValueError, match="rank-one int32"):
        naive_neighbor_list(
            jax.device_put(jnp.zeros((4, 3), dtype=jnp.float32), cpu),
            1.0,
            max_neighbors=8,
            target_indices=jax.device_put(jnp.array([0, 2], dtype=jnp.int64), cpu),
            strategy="scalar",
        )
