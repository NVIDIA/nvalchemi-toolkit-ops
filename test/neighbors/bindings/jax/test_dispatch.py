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
import warp as wp

from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list
from nvalchemiops.jax.neighbors.naive import naive_neighbor_list


@wp.func
def _sum_pair_fn(
    r_ij: wp.vec3f,
    distance: wp.float32,
    pair_params: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
):
    """Return simple energy and force pair outputs."""
    return pair_params[i, 0] + pair_params[j, 0] + distance, -r_ij


def test_jitted_cpu_trace_preserves_cpu_dispatch():
    """A traced placement is unknown without consulting the default backend."""
    script = """
import jax
import jax.numpy as jnp
import warp as wp

from nvalchemiops.jax.neighbors._dispatch import _jax_array_device_kind
from nvalchemiops.neighbors.naive.dispatch import (
    _NaiveWorkload,
    _resolve_naive_strategy,
)

observed = []
jax.default_backend = lambda: (_ for _ in ()).throw(AssertionError("backend queried"))

@jax.jit
def classify(positions):
    kind = _jax_array_device_kind(positions)
    strategy = _resolve_naive_strategy(
        "auto",
        _NaiveWorkload(
            device_kind=kind,
            wp_dtype=wp.float32,
            num_atoms=2,
            num_systems=1,
            partial=True,
            batched=False,
            pbc=False,
            wrap_positions=True,
            geometry_outputs=False,
        ),
    )
    observed.append((kind, strategy))
    return positions


classify(jnp.zeros((2, 3), dtype=jnp.float32)).block_until_ready()
assert observed == [("unknown", "scalar")]
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


def test_jitted_large_partial_auto_cpu_uses_scalar_output():
    """A traced CPU partial auto call stays on the scalar-compatible path."""
    script = """
import jax
import jax.numpy as jnp

from nvalchemiops.jax.neighbors.naive import naive_neighbor_list

targets = jnp.array([0, 511], dtype=jnp.int32)

@jax.jit
def run(positions):
    return naive_neighbor_list(
        positions,
        0.0,
        max_neighbors=4,
        target_indices=targets,
        strategy="auto",
    )

positions = jnp.arange(1024 * 3, dtype=jnp.float32).reshape(1024, 3)
neighbor_matrix, num_neighbors = run(positions)
neighbor_matrix.block_until_ready()
assert neighbor_matrix.shape == (2, 4)
assert num_neighbors.shape == (2,)
assert (num_neighbors == 0).all()
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


def test_jitted_gpu_partial_auto_does_not_query_default_backend():
    """Traced CUDA partial auto remains scalar without backend inference."""
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("Mixed CPU/GPU JAX backends are required.")
    script = """
import jax
import jax.numpy as jnp

from nvalchemiops.jax.neighbors.naive import naive_neighbor_list

gpu = jax.local_devices(backend="gpu")[0]
targets = jax.device_put(jnp.array([0, 1023], dtype=jnp.int32), gpu)
positions = jax.device_put(
    jnp.arange(1024 * 3, dtype=jnp.float32).reshape(1024, 3),
    gpu,
)

def run(pos):
    return naive_neighbor_list(
        pos,
        0.0,
        max_neighbors=4,
        target_indices=targets,
        strategy="auto",
    )

jax.default_backend = lambda: (_ for _ in ()).throw(AssertionError("backend queried"))
neighbor_matrix, num_neighbors = jax.jit(run, backend="gpu")(positions)
neighbor_matrix.block_until_ready()
assert neighbor_matrix.shape == (2, 4)
assert num_neighbors.shape == (2,)
assert (num_neighbors == 0).all()
"""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cuda,cpu"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_jitted_cpu_tile_rejects_before_launch():
    """An explicit tiled CPU request fails while JAX traces the call."""
    script = """
import jax
import jax.numpy as jnp

from nvalchemiops.jax.neighbors.naive import naive_neighbor_list

targets = jnp.array([0, 2], dtype=jnp.int32)

@jax.jit
def run(positions):
    return naive_neighbor_list(
        positions,
        1.0,
        max_neighbors=8,
        target_indices=targets,
        strategy="tile",
    )

try:
    run(jnp.zeros((4, 3), dtype=jnp.float32))
except jax.errors.JaxRuntimeError as exc:
    assert "No FFI handler registered" in str(exc)
    assert "Host" in str(exc)
else:
    raise AssertionError("expected CPU-targeted tile lowering to fail")
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


@pytest.mark.parametrize(
    ("graph_mode", "strategy"),
    [("none", "tile"), ("warp", "auto")],
)
@pytest.mark.parametrize("target_backend", ["gpu", "cpu"])
def test_jitted_tile_uses_target_backend_not_process_default(
    graph_mode,
    strategy,
    target_backend,
):
    """JIT tile routing ignores the process default backend."""
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("Mixed CPU/GPU JAX backends are required.")
    script = f"""
import numpy as np

import jax
import jax.numpy as jnp
import warp as wp

from nvalchemiops.jax.neighbors.naive import naive_neighbor_list

gpu = jax.local_devices(backend="gpu")[0]
targets = jax.device_put(jnp.array([2, 0], dtype=jnp.int32), gpu)
positions = jax.device_put(
    jnp.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        dtype=jnp.float32,
    ),
    gpu,
)

def run_no_graph(pos):
    return naive_neighbor_list(
        pos,
        0.75,
        max_neighbors=4,
        target_indices=targets,
        strategy="{strategy}",
        graph_mode="{graph_mode}",
    )

def run_warp(pos, matrix, counts):
    return naive_neighbor_list(
        pos,
        0.75,
        max_neighbors=4,
        target_indices=targets,
        neighbor_matrix=matrix,
        num_neighbors=counts,
        return_neighbor_list=False,
        strategy="{strategy}",
        graph_mode="{graph_mode}",
    )

jax.default_backend = lambda: "cpu"

if "{target_backend}" == "gpu":
    scalar = naive_neighbor_list(
        positions,
        0.75,
        max_neighbors=4,
        target_indices=targets,
        strategy="scalar",
    )
    if "{graph_mode}" == "none":
        tiled = jax.jit(run_no_graph, backend="gpu")(positions)
    else:
        matrix = jax.device_put(jnp.full((2, 4), 4, dtype=jnp.int32), gpu)
        counts = jax.device_put(jnp.zeros((2,), dtype=jnp.int32), gpu)
        tiled = jax.jit(run_warp, backend="gpu", donate_argnums=(1, 2))(
            positions,
            matrix,
            counts,
        )
    for scalar_array, tiled_array in zip(scalar, tiled):
        np.testing.assert_array_equal(np.asarray(scalar_array), np.asarray(tiled_array))
else:
    if "{graph_mode}" == "none":
        call = jax.jit(run_no_graph, backend="cpu")
        args = (jnp.zeros((4, 3), dtype=jnp.float32),)
    else:
        call = jax.jit(run_warp, backend="cpu", donate_argnums=(1, 2))
        args = (
            jnp.zeros((4, 3), dtype=jnp.float32),
            jnp.full((2, 4), 4, dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
        )
    launch_tiled_called = False
    real_launch_tiled = wp.launch_tiled
    def checked_launch_tiled(*args, **kwargs):
        global launch_tiled_called
        launch_tiled_called = True
        return real_launch_tiled(*args, **kwargs)
    wp.launch_tiled = checked_launch_tiled
    try:
        call(*args)
    except jax.errors.JaxRuntimeError as exc:
        assert "No FFI handler registered" in str(exc)
        assert "Host" in str(exc)
    else:
        raise AssertionError("expected CPU-targeted tile lowering to fail")
    finally:
        wp.launch_tiled = real_launch_tiled
    assert not launch_tiled_called
"""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cuda,cpu"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_jitted_cpu_partial_geometry_empty_routes_avoid_custom_calls():
    """Empty partial geometry routes return API-shaped outputs without Warp calls."""
    script = """
import jax
import jax.numpy as jnp

from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list
from nvalchemiops.jax.neighbors.naive import naive_neighbor_list
from test.neighbors.bindings.jax.test_dispatch import _sum_pair_fn

positions = jnp.zeros((2, 3), dtype=jnp.float32)
cell = jnp.eye(3, dtype=jnp.float32)[None, :, :]
pbc = jnp.ones((1, 3), dtype=jnp.bool_)
targets = jnp.array([0], dtype=jnp.int32)
shift_range = jnp.zeros((1, 3), dtype=jnp.int32)
num_shifts = jnp.ones((1,), dtype=jnp.int32)

@jax.jit
def single(nm, nn, shifts, distances, vectors):
    return naive_neighbor_list(
        positions,
        0.0,
        cell=cell,
        pbc=pbc,
        max_neighbors=4,
        target_indices=targets,
        neighbor_matrix=nm,
        num_neighbors=nn,
        neighbor_matrix_shifts=shifts,
        neighbor_distances=distances,
        neighbor_vectors=vectors,
        return_distances=True,
        return_vectors=True,
        pair_fn=_sum_pair_fn,
        pair_params=jnp.ones((2, 1), dtype=jnp.float32),
        shift_range_per_dimension=shift_range,
        num_shifts_per_system=num_shifts,
        max_shifts_per_system=1,
    )

out = single(
    jnp.full((1, 4), 99, dtype=jnp.int32),
    jnp.full((1,), 99, dtype=jnp.int32),
    jnp.full((1, 4, 3), 99, dtype=jnp.int32),
    jnp.full((1, 4), 99.0, dtype=jnp.float32),
    jnp.full((1, 4, 3), 99.0, dtype=jnp.float32),
)
nm, nn, shifts, distances, vectors, energies, forces = out
assert (nm == 2).all() and (nn == 0).all() and (shifts == 0).all()
assert (distances == 0).all() and (vectors == 0).all()
assert energies.shape == (1, 4) and forces.shape == (1, 4, 3)
assert (energies == 0).all() and (forces == 0).all()

@jax.jit
def batch():
    return batch_naive_neighbor_list(
        positions,
        1.0,
        batch_idx=jnp.zeros((2,), dtype=jnp.int32),
        batch_ptr=jnp.array([0, 2], dtype=jnp.int32),
        cell=cell,
        pbc=pbc,
        max_neighbors=4,
        target_indices=jnp.empty((0,), dtype=jnp.int32),
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
        shift_range_per_dimension=shift_range,
        num_shifts_per_system=num_shifts,
        max_shifts_per_system=1,
    )

neighbor_list, neighbor_ptr, neighbor_shifts, distances, vectors = batch()
assert neighbor_list.shape == (2, 0)
assert neighbor_ptr.shape == (1,) and (neighbor_ptr == 0).all()
assert neighbor_shifts.shape == (0, 3)
assert distances.shape == (0,) and vectors.shape == (0, 3)
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
