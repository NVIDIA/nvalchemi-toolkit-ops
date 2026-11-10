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

"""Opt-in JAX length checks preserve device execution and differentiation."""

from functools import partial

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from nvalchemiops.jax.segment_mm import segment_mm  # noqa: E402
from test.bindings.jax import test_segment_mm as _segment_mm_tests  # noqa: E402
from test.bindings.jax.test_segment_mm import (  # noqa: E402
    _assert_close,
    _inputs,
    _reference,
)

execution_device = _segment_mm_tests.execution_device


class TestSegmentMMValidation:
    """Checked calls return device error state for valid and invalid lengths."""

    def test_checked_keyword_reports_invalid_runtime_lengths(self):
        """A compiled checked call detects a negative runtime segment length."""
        a = jnp.ones((4, 2), dtype=jnp.float32)
        b = jnp.ones((2, 2, 3), dtype=jnp.float32)
        lengths = jnp.array([-1, 2], dtype=jnp.int32)
        checked = jax.jit(partial(segment_mm, validate=True))
        error, out = checked(a, b, lengths)
        assert out.shape == (4, 3)
        assert error.get() is not None

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_runtime_checks_do_not_overflow_int32(self, dtype):
        """One executable accepts changed valid lengths and catches wrapped sums."""
        a, b, _ = _inputs(dtype)
        checked = jax.jit(partial(segment_mm, validate=True))
        cases = (
            ((3, 0, 9, 2), True),
            ((0, 19, 0, 0), True),
            ((0, 0, 0, 0), True),
            ((-1, 2, 0, 0), False),
            ((5, 5, 5, 5), False),
            ((-2147483648, 1, 0, 0), False),
            ((2147483647, 2147483647, 2, 0), False),
            ((2147483647, 2147483647, 5, 0), False),
            ((0, 2, 0, 3), True),
        )
        with jax.enable_x64(False):
            for values, valid in cases:
                lengths = jnp.array(values, dtype=jnp.int32)
                error, out = checked(a, b, lengths)
                assert (error.get() is None) == valid
                if valid:
                    _assert_close(out, _reference(a, b, lengths))

    @pytest.mark.parametrize(
        "values,capacity,valid",
        [
            ((), 0, True),
            ((), 7, True),
            ((0, 0), 0, True),
            ((0, 0), 7, True),
            ((0, 1), 0, False),
            ((-1, 0), 0, False),
        ],
    )
    def test_empty_and_zero_capacity(self, values, capacity, valid):
        """Empty segments are valid; nonzero lengths cannot fit zero capacity."""
        a = jnp.ones((capacity, 2), dtype=jnp.float32)
        b = jnp.ones((len(values), 2, 3), dtype=jnp.float32)
        lengths = jnp.array(values, dtype=jnp.int32)
        checked = jax.jit(partial(segment_mm, validate=True))
        error, out = checked(a, b, lengths)
        assert (error.get() is None) == valid
        assert out.shape == (capacity, 3)
        if valid:
            np.testing.assert_array_equal(np.asarray(out), np.zeros(out.shape))

    @pytest.mark.parametrize("explicit_false", [False, True])
    def test_default_needs_no_checkify_transform(self, explicit_false):
        """Unchecked eager and compiled calls retain the ordinary array result."""
        a, b, _ = _inputs()
        lengths = jnp.array([3, 0, 9, 2], dtype=jnp.int32)
        operation = (
            partial(segment_mm, validate=False) if explicit_false else segment_mm
        )
        expected = _reference(a, b, lengths)
        _assert_close(operation(a, b, lengths), expected)
        _assert_close(jax.jit(operation)(a, b, lengths), expected)

    @pytest.mark.parametrize("validate", [False, True])
    def test_metadata_validation_is_always_enabled(self, validate):
        """Opting out of runtime checks does not disable channel validation."""
        a = jnp.ones((4, 2), dtype=jnp.float32)
        b = jnp.ones((2, 3, 2), dtype=jnp.float32)
        lengths = jnp.array([1, 1], dtype=jnp.int32)
        with pytest.raises(ValueError, match="channel dimensions.*match"):
            segment_mm(a, b, lengths, validate=validate)

    @pytest.mark.parametrize("validate", [None, 0, 1, "yes"])
    def test_validate_requires_python_bool(self, validate):
        """Ambiguous validation settings fail instead of selecting a truthy branch."""
        a, b, _ = _inputs()
        lengths = jnp.array([3, 0, 9, 2], dtype=jnp.int32)
        with pytest.raises(ValueError, match="validate.*bool"):
            segment_mm(a, b, lengths, validate=validate)

    def test_validate_must_be_static_under_jit(self):
        """A traced validation flag is rejected before any value conversion."""
        a, b, _ = _inputs()
        lengths = jnp.array([3, 0, 9, 2], dtype=jnp.int32)
        with pytest.raises(ValueError, match="validate.*bool"):
            jax.jit(segment_mm)(a, b, lengths, validate=True)

    def test_scan_checks_changed_lengths(self):
        """Checks inside scan preserve valid products and report a later bad step."""
        a, b, _ = _inputs()
        lengths = jnp.array([[3, 0, 9, 2], [0, 19, 0, 0]], dtype=jnp.int32)

        def run(a, b, lengths):
            def step(carry, current):
                return carry, segment_mm(a, b, current, validate=True)

            return jax.lax.scan(step, None, lengths)[1]

        checked = jax.jit(run)
        error, out = checked(a, b, lengths)
        assert error.get() is None
        expected = jnp.stack([_reference(a, b, current) for current in lengths])
        _assert_close(out, expected)
        error, _ = checked(a, b, lengths.at[1, 0].set(-1))
        assert error.get() is not None

    def test_vmap_preserves_checks_and_outputs(self):
        """Batching checked products retains both output values and error state."""
        a, b, _ = _inputs()
        lengths = jnp.array([[3, 0, 9, 2], [0, 19, 0, 0]], dtype=jnp.int32)
        batched = jax.vmap(partial(segment_mm, validate=True), in_axes=(None, None, 0))
        checked = jax.jit(batched)
        error, out = checked(a, b, lengths)
        assert error.get() is None
        expected = jnp.stack([_reference(a, b, current) for current in lengths])
        _assert_close(out, expected)
        error, _ = checked(a, b, lengths.at[1, 0].set(-1))
        assert error.get() is not None

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_jvp_over_vjp_preserves_checks(self, dtype):
        """Checked second derivatives match dense products and retain length errors."""
        a, b, _ = _inputs(dtype)
        tangent = (jnp.full_like(a, 0.125), jnp.full_like(b, 0.125))
        lengths = jnp.array([3, 0, 9, 2], dtype=jnp.int32)

        def evaluate(a, b, lengths):
            def energy(a, b):
                error, out = segment_mm(a, b, lengths, validate=True)
                out = out.astype(jnp.float32)
                return jnp.sum(jnp.sin(out) + 0.1 * out**2), error

            (grads, error), (second, _) = jax.jvp(
                jax.grad(energy, (0, 1), has_aux=True), (a, b), tangent
            )
            return error, (grads, second)

        def reference_energy(a, b):
            out = _reference(a, b, lengths).astype(jnp.float32)
            return jnp.sum(jnp.sin(out) + 0.1 * out**2)

        checked = jax.jit(evaluate)
        error, actual = checked(a, b, lengths)
        assert error.get() is None
        expected = jax.jvp(jax.grad(reference_energy, (0, 1)), (a, b), tangent)
        for value, reference in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            _assert_close(value, reference)
        error, _ = checked(a, b, lengths.at[0].set(-1))
        assert error.get() is not None

    def test_checked_shard_map(self, execution_device):
        """A local shard returns valid products and propagates its length error."""
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        a, b, _ = _inputs()
        lengths = jnp.array([[3, 0, 9, 2]], dtype=jnp.int32)
        mesh = Mesh(np.asarray([execution_device]), ("systems",))

        def local(a, b, ll):
            error, out = segment_mm(a[0], b[0], ll[0], validate=True)
            return jax.tree.map(lambda value: value[None], error), out[None]

        mapped = jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("systems"), P("systems"), P("systems")),
            out_specs=(P("systems"), P("systems")),
        )
        checked = jax.jit(mapped)
        error, out = checked(a[None], b[None], lengths)
        assert error.get() is None
        _assert_close(out[0], _reference(a, b, lengths[0]))
        error, _ = checked(a[None], b[None], lengths.at[0, 0].set(-1))
        assert error.get() is not None

    @pytest.mark.parametrize("compiled", [False, True])
    def test_checked_placement_overrides_default_device(
        self, execution_device, compiled
    ):
        """Outputs and every error leaf follow committed inputs on another device."""
        alternatives = [
            device
            for device in (*jax.devices("cpu"), *jax.devices())
            if device != execution_device
        ]
        if not alternatives:
            pytest.skip("Requires another device")
        target = alternatives[0]
        a, b, _ = _inputs()
        lengths = jnp.array([3, 0, 9, 2], dtype=jnp.int32)
        a, b, lengths = jax.device_put((a, b, lengths), target)
        checked = partial(segment_mm, validate=True)
        if compiled:
            checked = jax.jit(checked)
        error, out = checked(a, b, lengths)
        assert out.devices() == {target}
        assert all(leaf.devices() == {target} for leaf in jax.tree.leaves(error))
        assert error.get() is None
        _assert_close(out, _reference(a, b, lengths))

    @pytest.mark.parametrize("compiled", [False, True])
    def test_checked_execution_has_no_host_transfers(self, execution_device, compiled):
        """Warmed checked calls keep outputs and deferred errors on the device."""
        a, b, _ = _inputs()
        lengths = jnp.array([3, 0, 9, 2], dtype=jnp.int32)
        checked = partial(segment_mm, validate=True)
        if compiled:
            checked = jax.jit(checked)
        jax.block_until_ready(checked(a, b, lengths))
        for first, valid in ((2, True), (-1, False)):
            # Device-produced lengths have no cached host copy to hide a read.
            current = lengths.at[0].set(first)
            with jax.transfer_guard_device_to_host("disallow_explicit"):
                error, out = checked(a, b, current)
                assert out.devices() == {execution_device}
                assert all(
                    leaf.devices() == {execution_device}
                    for leaf in jax.tree.leaves(error)
                )
            # Error observation is an explicit caller choice after execution.
            assert (error.get() is None) == valid
            if valid:
                _assert_close(out, _reference(a, b, current))
