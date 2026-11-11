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

"""Complete-system sharding for the local JAX segment_mm operation."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from nvalchemiops.jax.segment_mm import segment_mm  # noqa: E402

from .test_segment_mm import _assert_close, _reference  # noqa: E402

pytestmark = pytest.mark.gpu
P = jax.sharding.PartitionSpec


@pytest.fixture(params=[1, 2], ids=["one_gpu", "two_gpus"])
def mesh(request):
    """Exercise manual-axis typing on one GPU and execution on two GPUs."""
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if len(devices) < request.param:
        pytest.skip(f"Requires {request.param} CUDA devices")
    selected = devices[: request.param]
    if (
        jax.process_count() == 1
        and len({device.device_kind for device in selected}) > 1
    ):
        pytest.skip("Mixed GPU architectures require separate JAX processes")
    return jax.sharding.Mesh(np.asarray(selected), ("systems",))


def _inputs(mesh, dtype):
    """Give every GPU distinct complete systems with equal static capacities."""
    rng = np.random.default_rng(729)
    count = mesh.size
    a = jnp.asarray(rng.normal(size=(count, 9, 3)) * 0.2, dtype)
    b = jnp.asarray(rng.normal(size=(count, 3, 3, 3)) * 0.2, dtype)
    g = jnp.asarray(rng.normal(size=(count, 9, 3)) * 0.2, dtype)
    return a, b, g


def _map(mesh, function, specs=(P("systems"),) * 3):
    """Keep validity checking enabled for the supported manual SPMD boundary."""
    return jax.shard_map(
        function,
        mesh=mesh,
        in_specs=specs,
        out_specs=P("systems"),
        check_vma=True,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_runtime_lengths_and_vjp(mesh, dtype):
    """One executable keeps each shard's outputs and gradients independent."""
    a, b, g = _inputs(mesh, dtype)
    distributed = _map(mesh, lambda a, b, ll: segment_mm(a[0], b[0], ll[0])[None, ...])

    def evaluate(operation, a, b, lengths, g):
        y, pullback = jax.vjp(lambda x, w: operation(x, w, lengths), a, b)
        return y, *pullback(g)

    lengths = jnp.asarray([[3, 0, 2], [0, 4, 3]][: mesh.size], jnp.int32)
    executable = (
        jax.jit(lambda a, b, ll, g: evaluate(distributed, a, b, ll, g))
        .lower(a, b, lengths, g)
        .compile()
    )
    for values in (
        [[3, 0, 2], [0, 4, 3]],
        [[0, 0, 0], [2, 0, 7]],
        [[0, 2, 4], [0, 0, 0]],
    ):
        lengths = jnp.asarray(values[: mesh.size], jnp.int32)
        actual = executable(a, b, lengths, g)
        expected = evaluate(jax.vmap(_reference), a, b, lengths, g)
        for value, ref in zip(actual, expected, strict=True):
            _assert_close(value, ref)
            assert value.sharding.is_equivalent_to(
                jax.sharding.NamedSharding(mesh, P("systems")), value.ndim
            )
        for rank, local_lengths in enumerate(values[: mesh.size]):
            used = sum(local_lengths)
            np.testing.assert_array_equal(np.asarray(actual[0])[rank, used:], 0)
            np.testing.assert_array_equal(np.asarray(actual[1])[rank, used:], 0)
            for group, length in enumerate(local_lengths):
                if not length:
                    np.testing.assert_array_equal(np.asarray(actual[2])[rank, group], 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_scan_and_jvp_over_vjp(mesh, dtype):
    """Scan carries retain shard variation through nonlinear second derivatives."""
    a, b, _ = _inputs(mesh, dtype)
    lengths = jnp.asarray(
        [[[3, 0, 2], [0, 4, 3]], [[0, 2, 4], [1, 0, 3]]][: mesh.size],
        jnp.int32,
    )

    def scan(operation, a, b, lengths):
        def step(carry, ll):
            y = operation(carry, b, ll)
            return jnp.tanh(y), None

        return jax.lax.scan(step, a, lengths)[0]

    distributed = _map(
        mesh, lambda a, b, ll: scan(segment_mm, a[0], b[0], ll[0])[None, ...]
    )
    reference = jax.vmap(lambda a, b, ll: scan(_reference, a, b, ll))

    def evaluate(operation, a, b, lengths):
        def loss(a, b):
            y = operation(a, b, lengths).astype(jnp.float32)
            return jnp.sum(jnp.sin(y) + 0.25 * y**2)

        return jax.jvp(
            jax.grad(loss, argnums=(0, 1)),
            (a, b),
            (jnp.ones_like(a) * 0.125, jnp.ones_like(b) * 0.125),
        )

    actual = jax.jit(lambda a, b, ll: evaluate(distributed, a, b, ll))(a, b, lengths)
    expected = evaluate(reference, a, b, lengths)
    for value, ref in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        _assert_close(value, ref)


def test_lengths_only_vary_and_replicated_operand_gradients(mesh):
    """Runtime lengths alone vary; JAX must sum gradients of replicated operands."""
    a, b, g = (value[0] for value in _inputs(mesh, jnp.float32))
    lengths = jnp.asarray([[3, 0, 2], [0, 4, 3]][: mesh.size], jnp.int32)
    distributed = _map(
        mesh,
        lambda a, b, ll: segment_mm(a, b, ll[0])[None, ...],
        (P(), P(), P("systems")),
    )
    reference = jax.vmap(lambda ll: _reference(a, b, ll))
    _assert_close(jax.jit(distributed)(a, b, lengths), reference(lengths))

    def loss(operation, a, b):
        return jnp.sum(operation(a, b, lengths) * g)

    actual = jax.jit(jax.grad(lambda a, b: loss(distributed, a, b), (0, 1)))(a, b)
    expected = jax.grad(
        lambda a, b: jnp.sum(jax.vmap(lambda ll: _reference(a, b, ll))(lengths) * g),
        (0, 1),
    )(a, b)
    for value, ref in zip(actual, expected, strict=True):
        _assert_close(value, ref)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_gradients_inside_shard_map(mesh, dtype):
    """Each mapped body can differentiate its own local system loss."""
    a, b, _ = _inputs(mesh, dtype)
    lengths = jnp.asarray([[3, 0, 2], [0, 4, 3]][: mesh.size], jnp.int32)

    def evaluate(operation, a, b, lengths):
        def loss(a, b):
            y = operation(a, b, lengths).astype(jnp.float32)
            return jnp.sum(jnp.sin(y) + 0.25 * y**2)

        return jax.jvp(
            jax.grad(loss, argnums=(0, 1)),
            (a, b),
            (jnp.ones_like(a) * 0.125, jnp.ones_like(b) * 0.125),
        )

    def local(a, b, lengths):
        result = evaluate(segment_mm, a[0], b[0], lengths[0])
        return jax.tree.map(lambda x: x[None, ...], result)

    actual = jax.jit(_map(mesh, local))(a, b, lengths)
    expected = jax.vmap(lambda a, b, ll: evaluate(_reference, a, b, ll))(a, b, lengths)
    for value, ref in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        _assert_close(value, ref)


@pytest.mark.parametrize("axis_type", ["reduced", "unreduced"])
def test_unsupported_manual_axis_types(mesh, axis_type):
    """Reject partial or reduced operands before varying-axis promotion."""

    def local(a, b, lengths):
        a = a[0]
        if axis_type == "reduced":
            b = jax.lax.pcast(b, "systems", to="reduced")
        else:
            a = jax.lax.pcast(a, "systems", to="unreduced")
        return segment_mm(a, b, lengths[0])[None, ...]

    distributed = _map(mesh, local, (P("systems"), P(), P("systems")))
    inputs = (
        jax.ShapeDtypeStruct((mesh.size, 9, 3), jnp.float32),
        jax.ShapeDtypeStruct((3, 3, 3), jnp.float32),
        jax.ShapeDtypeStruct((mesh.size, 3), jnp.int32),
    )
    with pytest.raises(ValueError, match="ordinary local operands"):
        jax.make_jaxpr(distributed)(*inputs)
