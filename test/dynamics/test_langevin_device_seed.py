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

"""
Unit tests for the device-side step offset of the Langevin half step.

``random_seed`` stays the ``wp.uint64`` launch argument; an optional, read-only
``step_offset`` (``wp.array(dtype=wp.int32)``) is added to it when the kernel runs
(``None`` or an empty array: offset 0). The caller advances the counter.

Tests cover:
- ``step_offset=None`` and an empty offset equal the plain int-seed op bit for bit
- eager offsets 0..4 equal the seeds ``s .. s + 4`` (in-place and ``_out`` APIs;
  single, batch_idx and atom_ptr modes); the op never writes the offset
- seeds of ``2**31`` and above work
- a malformed offset is rejected before any launch
- a half step captured in a CUDA graph draws fresh noise on every replay when the
  caller advances the counter inside the graph
"""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.integrators import (
    langevin_baoab_half_step,
    langevin_baoab_half_step_out,
)

DEVICES = ["cuda:0"]

DTYPE_CONFIGS = [
    pytest.param(wp.vec3f, wp.float32, np.float32, id="float32"),
    pytest.param(wp.vec3d, wp.float64, np.float64, id="float64"),
]

MODES = ["single", "batch_idx", "atom_ptr"]
N_ATOMS = 96
N_SYSTEMS = 3


def _make_state(device, vec, scl, np_dtype, mode):
    rng = np.random.default_rng(0)
    n_sys = 1 if mode == "single" else N_SYSTEMS
    state = {
        "positions": rng.standard_normal((N_ATOMS, 3)).astype(np_dtype),
        "velocities": rng.standard_normal((N_ATOMS, 3)).astype(np_dtype),
        "forces": rng.standard_normal((N_ATOMS, 3)).astype(np_dtype),
    }
    arrays = {k: wp.array(v, dtype=vec, device=device) for k, v in state.items()}
    arrays["masses"] = wp.array(
        np.full(N_ATOMS, 2.0, np_dtype), dtype=scl, device=device
    )
    arrays["dt"] = wp.array(np.full(n_sys, 0.01, np_dtype), dtype=scl, device=device)
    arrays["temperature"] = wp.array(
        np.full(n_sys, 1.2, np_dtype), dtype=scl, device=device
    )
    arrays["friction"] = wp.array(
        np.full(n_sys, 0.7, np_dtype), dtype=scl, device=device
    )
    kw = {}
    if mode == "batch_idx":
        kw["batch_idx"] = wp.array(
            np.repeat(np.arange(N_SYSTEMS), N_ATOMS // N_SYSTEMS),
            dtype=wp.int32,
            device=device,
        )
    elif mode == "atom_ptr":
        kw["atom_ptr"] = wp.array(
            np.arange(N_SYSTEMS + 1) * (N_ATOMS // N_SYSTEMS),
            dtype=wp.int32,
            device=device,
        )
    return arrays, kw


def _half_step(arrays, kw, seed, step_offset=None):
    langevin_baoab_half_step(
        arrays["positions"],
        arrays["velocities"],
        arrays["forces"],
        arrays["masses"],
        arrays["dt"],
        arrays["temperature"],
        arrays["friction"],
        seed,
        step_offset=step_offset,
        **kw,
    )


@wp.kernel
def _bump(counter: wp.array(dtype=wp.int32)):
    """Stand-in for the caller's ``step_offset.add_(1)``."""
    counter[0] = counter[0] + 1


def _offset(value, device):
    return wp.array([value], dtype=wp.int32, device=device)


def _out_step(base, kw, seed, step_offset, device, vec):
    pos_out = wp.zeros(N_ATOMS, dtype=vec, device=device)
    vel_out = wp.zeros(N_ATOMS, dtype=vec, device=device)
    langevin_baoab_half_step_out(
        base["positions"],
        base["velocities"],
        base["forces"],
        base["masses"],
        base["dt"],
        base["temperature"],
        base["friction"],
        seed,
        pos_out,
        vel_out,
        step_offset=step_offset,
        **kw,
    )
    return pos_out.numpy(), vel_out.numpy()


class TestLangevinStepOffset:
    """``step_offset`` against the Python ``int`` seed."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_none_equals_int_seed(
        self, device, mode, dtype_vec, dtype_scalar, np_dtype
    ):
        """step_offset=None is bitwise the int-seed op (same call, same noise)."""
        a, kw = _make_state(device, dtype_vec, dtype_scalar, np_dtype, mode)
        b, _ = _make_state(device, dtype_vec, dtype_scalar, np_dtype, mode)
        _half_step(a, kw, 5)
        _half_step(b, kw, 5, step_offset=None)
        for name in ("positions", "velocities"):
            np.testing.assert_array_equal(a[name].numpy(), b[name].numpy())

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_empty_offset_is_zero(
        self, device, mode, dtype_vec, dtype_scalar, np_dtype
    ):
        """An empty step_offset array is offset 0 (a defined default, not an out-of-bounds read)."""
        a, kw = _make_state(device, dtype_vec, dtype_scalar, np_dtype, mode)
        b, _ = _make_state(device, dtype_vec, dtype_scalar, np_dtype, mode)
        _half_step(a, kw, 5)
        _half_step(b, kw, 5, step_offset=wp.zeros(0, dtype=wp.int32, device=device))
        for name in ("positions", "velocities"):
            np.testing.assert_array_equal(a[name].numpy(), b[name].numpy())

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    @pytest.mark.parametrize("base_seed", [9, 2**31 + 7, 2**40 + 3])
    def test_offset_equals_shifted_seed(
        self, device, mode, dtype_vec, dtype_scalar, np_dtype, base_seed
    ):
        """Offsets 0..4 with seed s equal the int seeds s .. s + 4, bitwise; the op leaves the offset alone."""
        ref, kw = _make_state(device, dtype_vec, dtype_scalar, np_dtype, mode)
        dev, _ = _make_state(device, dtype_vec, dtype_scalar, np_dtype, mode)
        for k in range(5):
            off = _offset(k, device)
            _half_step(ref, kw, base_seed + k)
            _half_step(dev, kw, base_seed, step_offset=off)
            assert off.numpy()[0] == k
        for name in ("positions", "velocities"):
            np.testing.assert_array_equal(ref[name].numpy(), dev[name].numpy())

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_out_api_matches_shifted_seed(
        self, device, mode, dtype_vec, dtype_scalar, np_dtype
    ):
        """The non-mutating API takes the offset too (inputs untouched, outputs bitwise equal to the shifted int seed)."""
        base, kw = _make_state(device, dtype_vec, dtype_scalar, np_dtype, mode)
        want = _out_step(base, kw, 80, None, device, dtype_vec)
        got = _out_step(base, kw, 77, _offset(3, device), device, dtype_vec)
        np.testing.assert_array_equal(want[0], got[0])
        np.testing.assert_array_equal(want[1], got[1])
        none = _out_step(base, kw, 80, None, device, dtype_vec)
        np.testing.assert_array_equal(want[1], none[1])


class TestLangevinStepOffsetValidation:
    """A malformed step_offset is rejected on the host, before any launch."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("api", ["inplace", "out"])
    @pytest.mark.parametrize(
        "make_offset,exc",
        [
            pytest.param(
                lambda d: wp.zeros((1, 1), dtype=wp.int32, device=d),
                ValueError,
                id="ndim2",
            ),
            pytest.param(
                lambda d: wp.zeros(1, dtype=wp.uint64, device=d), TypeError, id="dtype"
            ),
            pytest.param(
                lambda d: wp.zeros(1, dtype=wp.int32, device="cpu"),
                ValueError,
                id="device",
            ),
        ],
    )
    def test_bad_offset_raises(self, device, api, make_offset, exc):
        arrays, kw = _make_state(device, wp.vec3f, wp.float32, np.float32, "single")
        off = make_offset(device)
        before = arrays["velocities"].numpy().copy()
        with pytest.raises(exc):
            if api == "inplace":
                _half_step(arrays, kw, 1, step_offset=off)
            else:
                _out_step(arrays, kw, 1, off, device, wp.vec3f)
        np.testing.assert_array_equal(before, arrays["velocities"].numpy())


class TestLangevinStepOffsetGraph:
    """The step offset under CUDA-graph capture and replay."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_replay_draws_fresh_noise_and_matches_eager(
        self, device, mode, dtype_vec, dtype_scalar, np_dtype
    ):
        """k replays equal k eager calls with seeds s, s+1, ...; the noise differs from replay to replay."""
        n_replays = 5
        ref, kw = _make_state(device, dtype_vec, dtype_scalar, np_dtype, mode)
        cap, _ = _make_state(device, dtype_vec, dtype_scalar, np_dtype, mode)
        offset = _offset(0, device)
        # warm up (module load / overload build) on a scratch state
        scratch, _ = _make_state(device, dtype_vec, dtype_scalar, np_dtype, mode)
        _half_step(scratch, kw, 900, step_offset=offset)
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            # capture records the launches without running them; the caller's counter advance is part of the graph
            _half_step(cap, kw, 900, step_offset=offset)
            wp.launch(_bump, dim=1, inputs=[offset], device=device)
        v_prev = cap["velocities"].numpy().copy()
        increments = []
        for k in range(n_replays):
            _half_step(ref, kw, 900 + k)
            wp.capture_launch(capture.graph)
            v_now = cap["velocities"].numpy()
            increments.append(v_now - v_prev)
            v_prev = v_now.copy()
            np.testing.assert_array_equal(ref["velocities"].numpy(), v_now)
            np.testing.assert_array_equal(
                ref["positions"].numpy(), cap["positions"].numpy()
            )
        assert offset.numpy()[0] == n_replays
        # fresh noise on every replay: no two velocity increments are equal
        for i in range(n_replays):
            for j in range(i + 1, n_replays):
                assert not np.array_equal(increments[i], increments[j])
