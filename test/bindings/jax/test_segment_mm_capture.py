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

"""Warp-owned JAX graph replay with prefix scans and split weight gradients."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from nvalchemiops.jax.segment_mm import segment_mm  # noqa: E402

from .test_segment_mm import _assert_close  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not any(device.platform == "gpu" for device in jax.devices()),
        reason="Requires CUDA JAX",
    ),
]


def _length_cases(groups):
    """Change scanned offsets and clear then refill the split-gradient workspace."""
    if groups == 1:
        return ((1007,), (0,), (33,), (1019,))
    return (
        (2049,) + (0,) * 63 + (31,) + (0,) * 63 + (1025,),
        (0,) * groups,
        (0,) * 64 + (769,) + (0,) * 63 + (33,),
        (512,) + (0,) * 126 + (1023, 17),
    )


def _dense_reference(a, b, g, lengths):
    """Evaluate dense FP64 products only on active slices, then round once."""
    dtype = np.asarray(a).dtype
    a, b, g = (np.asarray(value, dtype=np.float64) for value in (a, b, g))
    out = np.zeros((a.shape[0], b.shape[2]), dtype=np.float64)
    da, db = np.zeros_like(a), np.zeros_like(b)
    start = 0
    for group, length in enumerate(lengths):
        stop = start + length
        aa, gg = a[start:stop], g[start:stop]
        out[start:stop] = aa @ b[group]
        da[start:stop] = gg @ b[group].T
        db[group] = aa.T @ gg
        start = stop
    return tuple(value.astype(dtype) for value in (out, da, db))


class TestSegmentMMCapture:
    """Reuse one compiled graph while all input storage and runtime lengths change."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    @pytest.mark.parametrize("capacity,groups", [(1024, 1), (40000, 129)])
    def test_split_gradients_and_prefix_scan(self, dtype, capacity, groups):
        """Fresh array addresses cannot reuse stale captured scans or dB partials."""
        rng = np.random.default_rng(712)
        cases = _length_cases(groups)

        def inputs(values):
            a = rng.normal(0, 0.2, (capacity, 7)).astype(np.float32)
            b = rng.normal(0, 0.2, (groups, 7, 5)).astype(np.float32)
            g = rng.normal(0, 0.2, (capacity, 5)).astype(np.float32)
            a[sum(values) :] = np.nan
            g[sum(values) :] = np.nan
            return (
                jnp.asarray(a, dtype),
                jnp.asarray(b, dtype),
                jnp.asarray(values, jnp.int32),
                jnp.asarray(g, dtype),
            )

        def evaluate(a, b, lengths, g):
            out, pullback = jax.vjp(lambda a, b: segment_mm(a, b, lengths), a, b)
            return out, *pullback(g)

        initial = inputs(cases[0])
        # Calling the compiled executable cannot retrace or compile for new values.
        executable = jax.jit(evaluate).lower(*initial).compile()
        jax.block_until_ready(executable(*initial))
        # Retaining earlier inputs prevents their live device storage being reused.
        retained = [initial]
        for values in cases:
            current = inputs(values)
            retained.append(current)
            actual = executable(*current)
            a, b, _, g = current
            expected = _dense_reference(a, b, g, values)
            for value, reference in zip(actual, expected, strict=True):
                _assert_close(value, reference)
                assert np.isfinite(np.asarray(value)).all()
            used = sum(values)
            np.testing.assert_array_equal(np.asarray(actual[0])[used:], 0)
            np.testing.assert_array_equal(np.asarray(actual[1])[used:], 0)
            np.testing.assert_array_equal(
                np.asarray(actual[2])[np.asarray(values) == 0], 0
            )
            if used:
                assert np.abs(expected[2].astype(np.float32)).max() > 0.1
