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
Unit tests for the device-side RNG seed of the Langevin half step.

Tests cover:
- a ``wp.array(dtype=wp.int32)`` seed reproduces the Python ``int`` seed bit for
  bit (in-place and ``_out`` APIs; single, batch_idx and atom_ptr modes)
- the device seed is advanced by one per call
- a half step captured in a CUDA graph draws fresh noise on every replay and
  follows the eager ``random_seed + k`` sequence bitwise
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


def _half_step(arrays, kw, seed):
    langevin_baoab_half_step(
        arrays["positions"],
        arrays["velocities"],
        arrays["forces"],
        arrays["masses"],
        arrays["dt"],
        arrays["temperature"],
        arrays["friction"],
        seed,
        **kw,
    )


def _dev_seed(value, device):
    return wp.array([value], dtype=wp.int32, device=device)


class TestLangevinDeviceSeed:
    """Device-side seed against the Python ``int`` seed."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_matches_int_seed_and_advances(
        self, device, mode, dtype_vec, dtype_scalar, np_dtype
    ):
        """seed[0] == s equals random_seed=s bitwise, and seed[0] is s + 1 afterwards, over a sequence of steps."""
        ref, kw = _make_state(device, dtype_vec, dtype_scalar, np_dtype, mode)
        dev, _ = _make_state(device, dtype_vec, dtype_scalar, np_dtype, mode)
        seed = _dev_seed(1234, device)
        for k in range(4):
            _half_step(ref, kw, 1234 + k)
            _half_step(dev, kw, seed)
            assert seed.numpy()[0] == 1234 + k + 1
        for name in ("positions", "velocities"):
            np.testing.assert_array_equal(ref[name].numpy(), dev[name].numpy())

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_out_api_matches_int_seed(
        self, device, mode, dtype_vec, dtype_scalar, np_dtype
    ):
        """The non-mutating API takes the device seed too (inputs untouched, outputs bitwise equal)."""
        base, kw = _make_state(device, dtype_vec, dtype_scalar, np_dtype, mode)
        seed = _dev_seed(77, device)
        outs = []
        for s in (77, seed):
            pos_out = wp.zeros(N_ATOMS, dtype=dtype_vec, device=device)
            vel_out = wp.zeros(N_ATOMS, dtype=dtype_vec, device=device)
            langevin_baoab_half_step_out(
                base["positions"],
                base["velocities"],
                base["forces"],
                base["masses"],
                base["dt"],
                base["temperature"],
                base["friction"],
                s,
                pos_out,
                vel_out,
                **kw,
            )
            outs.append((pos_out.numpy(), vel_out.numpy()))
        np.testing.assert_array_equal(outs[0][0], outs[1][0])
        np.testing.assert_array_equal(outs[0][1], outs[1][1])
        assert seed.numpy()[0] == 78

    @pytest.mark.parametrize("device", DEVICES)
    def test_int_seed_unchanged(self, device):
        """A Python int seed stays a launch constant: same seed, same noise."""
        a, kw = _make_state(device, wp.vec3f, wp.float32, np.float32, "batch_idx")
        b, _ = _make_state(device, wp.vec3f, wp.float32, np.float32, "batch_idx")
        _half_step(a, kw, 5)
        _half_step(b, kw, 5)
        np.testing.assert_array_equal(a["velocities"].numpy(), b["velocities"].numpy())


class TestLangevinDeviceSeedGraph:
    """The device seed under CUDA-graph capture and replay."""

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
        seed = _dev_seed(900, device)
        # warm up (module load / overload build) on a scratch state, then restore the seed
        scratch, _ = _make_state(device, dtype_vec, dtype_scalar, np_dtype, mode)
        _half_step(scratch, kw, seed)
        seed.fill_(900)
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            # capture records the launches without running them
            _half_step(cap, kw, seed)
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
        assert seed.numpy()[0] == 900 + n_replays
        # fresh noise on every replay: no two velocity increments are equal
        for i in range(n_replays):
            for j in range(i + 1, n_replays):
                assert not np.array_equal(increments[i], increments[j])
