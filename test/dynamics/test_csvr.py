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
Tests for the CSVR (stochastic velocity rescaling) thermostat.

Tests cover:
- csvr_thermostat (in-place) and csvr_thermostat_out (non-mutating)
- Exact conditional moments of the kinetic-energy update
- Single, batch_idx, and atom_ptr modes and their parity
- Float32 and float64 support
- Equilibration and the conserved effective energy
"""

from typing import Any

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.integrators.csvr import (
    csvr_thermostat,
    csvr_thermostat_out,
)
from nvalchemiops.dynamics.integrators.velocity_verlet import (
    velocity_verlet_position_update,
    velocity_verlet_velocity_finalize,
)
from nvalchemiops.dynamics.utils.thermostat_utils import compute_kinetic_energy

# ==============================================================================
# Test Configuration
# ==============================================================================

DEVICES = ["cuda:0"]

DTYPE_CONFIGS = [
    pytest.param(wp.vec3f, wp.float32, np.float32, id="float32"),
    pytest.param(wp.vec3d, wp.float64, np.float64, id="float64"),
]

BATCH_MODES = [
    pytest.param("batch_idx", id="batch_idx"),
    pytest.param("atom_ptr", id="atom_ptr"),
]


# ==============================================================================
# Helpers
# ==============================================================================


@wp.kernel
def _harmonic_forces_kernel(
    positions: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
):
    """Unit-spring harmonic forces F = -r."""
    i = wp.tid()
    forces[i] = -positions[i]


def compute_harmonic_forces(positions: wp.array, forces: wp.array):
    """Compute unit-spring harmonic forces F = -r."""
    wp.launch(
        _harmonic_forces_kernel,
        dim=positions.shape[0],
        inputs=[positions, forces],
        device=positions.device,
    )


def _params(dtype_scalar, device, num_systems=1, kT=1.0, tau=0.5, dt=0.1, ndof=30.0):
    """Per-system CSVR inputs: target_temp, tau, dt, ndof, and ke/scale scratch."""

    def arr(x):
        values = np.broadcast_to(np.asarray(x, dtype=np.float64), (num_systems,))
        return wp.array(values.copy(), dtype=dtype_scalar, device=device)

    return (
        arr(kT),
        arr(tau),
        arr(dt),
        arr(ndof),
        wp.empty(num_systems, dtype=dtype_scalar, device=device),
        wp.empty(num_systems, dtype=dtype_scalar, device=device),
    )


def _batch_kwargs(mode, sizes, device):
    """``batch_idx`` or ``atom_ptr`` keyword for systems with the given atom counts."""
    if mode == "batch_idx":
        idx = np.repeat(np.arange(len(sizes), dtype=np.int32), sizes)
        return {"batch_idx": wp.array(idx, dtype=wp.int32, device=device)}
    ptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
    return {"atom_ptr": wp.array(ptr, dtype=wp.int32, device=device)}


def _random_system(num_atoms, np_dtype, dtype_vec, dtype_scalar, device, seed=0):
    """Random velocities and masses as numpy and warp arrays."""
    rng = np.random.default_rng(seed)
    vel_np = rng.standard_normal((num_atoms, 3)).astype(np_dtype)
    masses_np = rng.uniform(1.0, 2.0, num_atoms).astype(np_dtype)
    velocities = wp.array(vel_np, dtype=dtype_vec, device=device)
    masses = wp.array(masses_np, dtype=dtype_scalar, device=device)
    return vel_np, masses_np, velocities, masses


def _ke_np(vel_np, masses_np, sizes):
    """Per-system kinetic energy 0.5 * sum(m * v^2)."""
    e = 0.5 * masses_np * np.sum(vel_np.astype(np.float64) ** 2, axis=1)
    return np.add.reduceat(e, np.concatenate([[0], np.cumsum(sizes)[:-1]]))


def _rtol(np_dtype):
    """Comparison tolerance for the given precision."""
    return 1e-5 if np_dtype == np.float32 else 1e-10


def run_csvr_nvt_step(
    positions,
    velocities,
    forces,
    masses,
    target_temp,
    tau,
    dt,
    ndof,
    ke,
    scale,
    seed,
    batch_idx=None,
):
    """One CSVR-NVT step: thermostat, VV position update, forces, VV finalize."""
    csvr_thermostat(
        velocities,
        masses,
        target_temp,
        tau,
        dt,
        ndof,
        seed,
        ke,
        scale,
        batch_idx=batch_idx,
    )
    velocity_verlet_position_update(
        positions, velocities, forces, masses, dt, batch_idx=batch_idx
    )
    compute_harmonic_forces(positions, forces)
    velocity_verlet_velocity_finalize(
        velocities, forces, masses, dt, batch_idx=batch_idx
    )


# ==============================================================================
# Single System API Tests
# ==============================================================================


class TestCSVRAPI:
    """Test the public API on a single system."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """In-place call fills ke and scale and rescales velocities by scale."""
        vel_np, masses_np, velocities, masses = _random_system(
            50, np_dtype, dtype_vec, dtype_scalar, device
        )
        target_temp, tau, dt, ndof, ke, scale = _params(dtype_scalar, device)

        csvr_thermostat(velocities, masses, target_temp, tau, dt, ndof, 42, ke, scale)
        wp.synchronize_device(device)

        np.testing.assert_allclose(
            ke.numpy(), _ke_np(vel_np, masses_np, [50]), rtol=_rtol(np_dtype)
        )
        s = scale.numpy()[0]
        assert np.isfinite(s) and s != 1.0
        np.testing.assert_allclose(velocities.numpy(), vel_np * s, rtol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_out_preserves_input(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Non-mutating call leaves inputs intact and matches the in-place call."""
        vel_np, _, velocities, masses = _random_system(
            50, np_dtype, dtype_vec, dtype_scalar, device
        )
        target_temp, tau, dt, ndof, ke, scale = _params(dtype_scalar, device)
        velocities_out = wp.empty_like(velocities)

        result = csvr_thermostat_out(
            velocities,
            masses,
            target_temp,
            tau,
            dt,
            ndof,
            42,
            ke,
            scale,
            velocities_out,
        )
        wp.synchronize_device(device)

        assert result is velocities_out
        np.testing.assert_array_equal(velocities.numpy(), vel_np)
        np.testing.assert_allclose(result.numpy(), vel_np * scale.numpy()[0], rtol=1e-6)

        csvr_thermostat(velocities, masses, target_temp, tau, dt, ndof, 42, ke, scale)
        np.testing.assert_array_equal(velocities.numpy(), result.numpy())

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_device_inference(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Both functions infer the device from velocities."""
        _, _, velocities, masses = _random_system(
            20, np_dtype, dtype_vec, dtype_scalar, device
        )
        target_temp, tau, dt, ndof, ke, scale = _params(dtype_scalar, device)
        velocities_out = wp.empty_like(velocities)

        csvr_thermostat(
            velocities, masses, target_temp, tau, dt, ndof, 1, ke, scale, device=None
        )
        csvr_thermostat_out(
            velocities,
            masses,
            target_temp,
            tau,
            dt,
            ndof,
            2,
            ke,
            scale,
            velocities_out,
            device=None,
        )
        wp.synchronize_device(device)
        assert np.all(np.isfinite(velocities_out.numpy()))

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_ke_false_matches_default(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """A caller-supplied ke gives bit-identical results to the internal one."""
        vel_np, _, velocities, masses = _random_system(
            50, np_dtype, dtype_vec, dtype_scalar, device
        )
        target_temp, tau, dt, ndof, ke, scale = _params(dtype_scalar, device)
        csvr_thermostat(velocities, masses, target_temp, tau, dt, ndof, 9, ke, scale)

        velocities_2 = wp.array(vel_np, dtype=dtype_vec, device=device)
        ke_2 = wp.empty_like(ke)
        scale_2 = wp.empty_like(scale)
        compute_kinetic_energy(velocities_2, masses, ke_2)
        csvr_thermostat(
            velocities_2,
            masses,
            target_temp,
            tau,
            dt,
            ndof,
            9,
            ke_2,
            scale_2,
            compute_ke=False,
        )
        wp.synchronize_device(device)

        np.testing.assert_array_equal(scale_2.numpy(), scale.numpy())
        np.testing.assert_array_equal(velocities_2.numpy(), velocities.numpy())

    @pytest.mark.parametrize("device", DEVICES)
    def test_out_requires_velocities_out(self, device):
        """csvr_thermostat_out raises TypeError without velocities_out."""
        velocities = wp.zeros(10, dtype=wp.vec3f, device=device)
        masses = wp.ones(10, dtype=wp.float32, device=device)
        target_temp, tau, dt, ndof, ke, scale = _params(wp.float32, device)

        with pytest.raises(TypeError):
            csvr_thermostat_out(
                velocities, masses, target_temp, tau, dt, ndof, 0, ke, scale
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_out_validates_shape(self, device):
        """csvr_thermostat_out rejects a mismatched velocities_out."""
        velocities = wp.zeros(10, dtype=wp.vec3f, device=device)
        masses = wp.ones(10, dtype=wp.float32, device=device)
        target_temp, tau, dt, ndof, ke, scale = _params(wp.float32, device)

        with pytest.raises(ValueError, match="velocities_out shape mismatch"):
            csvr_thermostat_out(
                velocities,
                masses,
                target_temp,
                tau,
                dt,
                ndof,
                0,
                ke,
                scale,
                wp.zeros(5, dtype=wp.vec3f, device=device),
            )


# ==============================================================================
# Distribution Tests
# ==============================================================================


class TestCSVRDistribution:
    """Check the kinetic-energy update against its exact conditional moments.

    For a given K, the CSVR update K' = scale^2 K has

    E[K'] = c K + (1 - c) K_bar,
    Var[K'] = 2 (1 - c)^2 K_bar^2 / ndof + 4 c (1 - c) K K_bar / ndof,

    with c = exp(-dt / tau) and K_bar = ndof kT / 2. Many one-atom systems
    in a single batched launch provide independent draws.
    """

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    @pytest.mark.parametrize("c", [0.0, 0.5, 0.99])
    @pytest.mark.parametrize("ndof", [1, 2, 3, 300])
    def test_conditional_moments(
        self, device, dtype_vec, dtype_scalar, np_dtype, c, ndof
    ):
        """Sample mean and variance of K' agree with the exact values within 5 sigma."""
        num_samples, kT, dt_val = 20000, 1.3, 0.1
        tau_val = 0.0 if c == 0.0 else -dt_val / np.log(c)
        target_temp, tau, dt, ndof_arr, ke, scale = _params(
            dtype_scalar, device, num_samples, kT, tau_val, dt_val, ndof
        )
        ke.fill_(0.7 * ndof * kT)  # away from K_bar so relaxation is visible
        velocities = wp.zeros(num_samples, dtype=dtype_vec, device=device)
        masses = wp.ones(num_samples, dtype=dtype_scalar, device=device)
        batch_idx = wp.array(
            np.arange(num_samples, dtype=np.int32), dtype=wp.int32, device=device
        )

        csvr_thermostat(
            velocities,
            masses,
            target_temp,
            tau,
            dt,
            ndof_arr,
            2024,
            ke,
            scale,
            batch_idx=batch_idx,
            compute_ke=False,
        )
        wp.synchronize_device(device)

        k = float(ke.numpy()[0])
        k_new = scale.numpy().astype(np.float64) ** 2 * k
        k_bar = 0.5 * ndof * kT
        expected_mean = c * k + (1.0 - c) * k_bar
        expected_var = (
            2.0 * (1.0 - c) ** 2 * k_bar**2 + 4.0 * c * (1.0 - c) * k * k_bar
        ) / ndof

        se_mean = k_new.std() / np.sqrt(num_samples)
        se_var = ((k_new - k_new.mean()) ** 2).std() / np.sqrt(num_samples)
        assert abs(k_new.mean() - expected_mean) < 5 * se_mean
        assert abs(k_new.var() - expected_var) < 5 * se_var


# ==============================================================================
# Reference Behaviour Tests
# ==============================================================================


class TestCSVRReference:
    """Test limits, guards, and seeding."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_tau_inf_is_noop(self, device, dtype_vec, dtype_scalar, np_dtype):
        """tau = inf leaves velocities unchanged."""
        vel_np, _, velocities, masses = _random_system(
            30, np_dtype, dtype_vec, dtype_scalar, device
        )
        target_temp, tau, dt, ndof, ke, scale = _params(
            dtype_scalar, device, tau=np.inf
        )

        csvr_thermostat(velocities, masses, target_temp, tau, dt, ndof, 3, ke, scale)
        wp.synchronize_device(device)

        assert scale.numpy()[0] == 1.0
        np.testing.assert_array_equal(velocities.numpy(), vel_np)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_zero_ke_and_ndof_guard(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Systems with zero kinetic energy or zero ndof get scale = 1, no NaN."""
        vel_np, _, velocities, masses = _random_system(
            30, np_dtype, dtype_vec, dtype_scalar, device
        )
        vel_np[:10] = 0.0
        velocities = wp.array(vel_np, dtype=dtype_vec, device=device)
        target_temp, tau, dt, ndof, ke, scale = _params(
            dtype_scalar, device, num_systems=3, ndof=[30.0, 0.0, 30.0]
        )

        csvr_thermostat(
            velocities,
            masses,
            target_temp,
            tau,
            dt,
            ndof,
            5,
            ke,
            scale,
            **_batch_kwargs("batch_idx", [10, 10, 10], device),
        )
        wp.synchronize_device(device)

        s = scale.numpy()
        assert s[0] == 1.0 and s[1] == 1.0 and s[2] != 1.0
        assert np.all(np.isfinite(velocities.numpy()))
        np.testing.assert_array_equal(velocities.numpy()[:20], vel_np[:20])

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_seed_reproducibility(self, device, dtype_vec, dtype_scalar, np_dtype):
        """The same seed reproduces the scale; a different seed changes it."""
        _, _, velocities, masses = _random_system(
            30, np_dtype, dtype_vec, dtype_scalar, device
        )
        target_temp, tau, dt, ndof, ke, scale = _params(dtype_scalar, device)
        compute_kinetic_energy(velocities, masses, ke)

        def sample(seed):
            vel = wp.clone(velocities)
            csvr_thermostat(
                vel,
                masses,
                target_temp,
                tau,
                dt,
                ndof,
                seed,
                ke,
                scale,
                compute_ke=False,
            )
            return scale.numpy()[0]

        assert sample(11) == sample(11)
        assert sample(11) != sample(12)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_scale_depends_only_on_ke_and_seed(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Different velocities with the same supplied ke and seed give the same scale.

        This lets every rank of a distributed run apply the same factor.
        """
        target_temp, tau, dt, ndof, ke, scale = _params(dtype_scalar, device)
        ke.fill_(12.0)
        scales = []
        for seed in (0, 1):
            _, _, velocities, masses = _random_system(
                20, np_dtype, dtype_vec, dtype_scalar, device, seed=seed
            )
            csvr_thermostat(
                velocities,
                masses,
                target_temp,
                tau,
                dt,
                ndof,
                77,
                ke,
                scale,
                compute_ke=False,
            )
            scales.append(scale.numpy()[0])
        assert scales[0] == scales[1]


# ==============================================================================
# Parity Tests
# ==============================================================================


class TestCSVRParity:
    """Single-system mode matches system 0 of batched runs exactly."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    @pytest.mark.parametrize("mode", BATCH_MODES)
    @pytest.mark.parametrize("use_out", [False, True], ids=["inplace", "out"])
    def test_system0_matches_single(
        self, device, dtype_vec, dtype_scalar, np_dtype, mode, use_out
    ):
        """With the same ke and seed, batched system 0 equals the single-system run."""
        sizes = [20, 20, 20]
        vel_np, _, _, masses = _random_system(
            60, np_dtype, dtype_vec, dtype_scalar, device
        )
        ke_np = np.array([30.0, 25.0, 40.0])

        def run(vel_np, ke_np, **batch):
            n = len(ke_np)
            target_temp, tau, dt, ndof, _, scale = _params(dtype_scalar, device, n)
            ke = wp.array(ke_np, dtype=dtype_scalar, device=device)
            vel = wp.array(vel_np, dtype=dtype_vec, device=device)
            args = (vel, masses, target_temp, tau, dt, ndof, 7, ke, scale)
            if use_out:
                vel = csvr_thermostat_out(
                    *args, wp.empty_like(vel), compute_ke=False, **batch
                )
            else:
                csvr_thermostat(*args, compute_ke=False, **batch)
            return vel.numpy(), scale.numpy()

        vel_single, scale_single = run(vel_np[:20], ke_np[:1])
        vel_batch, scale_batch = run(
            vel_np, ke_np, **_batch_kwargs(mode, sizes, device)
        )

        assert scale_batch[0] == scale_single[0]
        np.testing.assert_array_equal(vel_batch[:20], vel_single)
        for s, start in enumerate([0, 20, 40]):
            np.testing.assert_allclose(
                vel_batch[start : start + 20],
                vel_np[start : start + 20] * scale_batch[s],
                rtol=1e-6,
            )


# ==============================================================================
# Batched Tests
# ==============================================================================


class TestCSVRBatched:
    """Test batch_idx and atom_ptr modes."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    @pytest.mark.parametrize("mode", BATCH_MODES)
    @pytest.mark.parametrize("use_out", [False, True], ids=["inplace", "out"])
    def test_variable_system_sizes(
        self, device, dtype_vec, dtype_scalar, np_dtype, mode, use_out
    ):
        """Per-system ke and scale are applied to systems of different sizes."""
        sizes = [5, 17, 1, 9]
        vel_np, masses_np, velocities, masses = _random_system(
            sum(sizes), np_dtype, dtype_vec, dtype_scalar, device
        )
        target_temp, tau, dt, ndof, ke, scale = _params(
            dtype_scalar, device, len(sizes), ndof=[3.0 * n for n in sizes]
        )
        args = (velocities, masses, target_temp, tau, dt, ndof, 21, ke, scale)
        batch = _batch_kwargs(mode, sizes, device)

        if use_out:
            result = csvr_thermostat_out(*args, wp.empty_like(velocities), **batch)
            np.testing.assert_array_equal(velocities.numpy(), vel_np)
        else:
            csvr_thermostat(*args, **batch)
            result = velocities
        wp.synchronize_device(device)

        np.testing.assert_allclose(
            ke.numpy(), _ke_np(vel_np, masses_np, sizes), rtol=_rtol(np_dtype)
        )
        expected = vel_np * np.repeat(scale.numpy(), sizes)[:, None]
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    @pytest.mark.parametrize("mode", BATCH_MODES)
    def test_per_system_parameters(
        self, device, dtype_vec, dtype_scalar, np_dtype, mode
    ):
        """Per-system target_temp, tau, and ndof are respected.

        With tau = 0, K' = kT/2 * chi^2(ndof), which for ndof = 3000 is
        within a few percent of ndof * kT / 2.
        """
        sizes = [10, 10, 10]
        ndof_val = 3000.0
        vel_np, _, velocities, masses = _random_system(
            30, np_dtype, dtype_vec, dtype_scalar, device
        )
        kT = np.array([0.5, 4.0, 1.0])
        target_temp, tau, dt, ndof, ke, scale = _params(
            dtype_scalar, device, 3, kT=kT, tau=[0.0, 0.0, np.inf], ndof=ndof_val
        )

        csvr_thermostat(
            velocities,
            masses,
            target_temp,
            tau,
            dt,
            ndof,
            8,
            ke,
            scale,
            **_batch_kwargs(mode, sizes, device),
        )
        wp.synchronize_device(device)

        k_new = scale.numpy().astype(np.float64) ** 2 * ke.numpy()
        np.testing.assert_allclose(k_new[:2], 0.5 * ndof_val * kT[:2], rtol=0.1)
        assert scale.numpy()[2] == 1.0
        np.testing.assert_array_equal(velocities.numpy()[20:], vel_np[20:])


class TestCSVRAtomPtr:
    """atom_ptr-specific tests."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_vs_batch_idx_equivalence(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """atom_ptr and batch_idx produce the same result."""
        sizes = [12, 7, 21]
        vel_np, _, _, masses = _random_system(
            sum(sizes), np_dtype, dtype_vec, dtype_scalar, device
        )
        results = []
        for mode in ("batch_idx", "atom_ptr"):
            velocities = wp.array(vel_np, dtype=dtype_vec, device=device)
            target_temp, tau, dt, ndof, ke, scale = _params(dtype_scalar, device, 3)
            csvr_thermostat(
                velocities,
                masses,
                target_temp,
                tau,
                dt,
                ndof,
                4,
                ke,
                scale,
                **_batch_kwargs(mode, sizes, device),
            )
            results.append((velocities.numpy(), scale.numpy()))

        np.testing.assert_allclose(results[0][1], results[1][1], rtol=_rtol(np_dtype))
        np.testing.assert_allclose(results[0][0], results[1][0], rtol=_rtol(np_dtype))

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("compute_ke", [True, False])
    def test_atom_ptr_mutual_exclusivity(self, device, compute_ke):
        """Passing both batch_idx and atom_ptr raises ValueError."""
        velocities = wp.zeros(4, dtype=wp.vec3f, device=device)
        masses = wp.ones(4, dtype=wp.float32, device=device)
        target_temp, tau, dt, ndof, ke, scale = _params(wp.float32, device, 2)
        batch = {
            **_batch_kwargs("batch_idx", [2, 2], device),
            **_batch_kwargs("atom_ptr", [2, 2], device),
        }

        with pytest.raises(ValueError, match="batch_idx OR atom_ptr"):
            csvr_thermostat(
                velocities,
                masses,
                target_temp,
                tau,
                dt,
                ndof,
                0,
                ke,
                scale,
                compute_ke=compute_ke,
                **batch,
            )


# ==============================================================================
# Physics Tests
# ==============================================================================


class TestCSVRPhysics:
    """NVT dynamics of harmonic oscillators."""

    @staticmethod
    def _sample_temperatures(sizes, kT, num_steps, device, sample_from, seed=0):
        """Run CSVR-NVT from a cold start and sample per-system temperatures."""
        num_atoms = sum(sizes)
        rng = np.random.default_rng(seed)
        positions = wp.array(
            rng.standard_normal((num_atoms, 3)), dtype=wp.vec3d, device=device
        )
        velocities = wp.array(
            0.1 * rng.standard_normal((num_atoms, 3)), dtype=wp.vec3d, device=device
        )
        masses_np = np.ones(num_atoms)
        masses = wp.array(masses_np, dtype=wp.float64, device=device)
        forces = wp.empty_like(positions)
        compute_harmonic_forces(positions, forces)
        ndof = [3.0 * n for n in sizes]
        target_temp, tau, dt, ndof_arr, ke, scale = _params(
            wp.float64, device, len(sizes), kT=kT, tau=0.1, dt=0.01, ndof=ndof
        )
        batch_idx = _batch_kwargs("batch_idx", sizes, device)["batch_idx"]

        temps = []
        for step in range(num_steps):
            run_csvr_nvt_step(
                positions,
                velocities,
                forces,
                masses,
                target_temp,
                tau,
                dt,
                ndof_arr,
                ke,
                scale,
                1000 + step,
                batch_idx=batch_idx,
            )
            if step >= sample_from and step % 20 == 0:
                ke_sys = _ke_np(velocities.numpy(), masses_np, sizes)
                temps.append(2.0 * ke_sys / np.array(ndof))
        return np.array(temps)

    @pytest.mark.parametrize("device", DEVICES)
    def test_temperature_equilibration_harmonic(self, device):
        """CSVR heats a cold system to the target with canonical fluctuations."""
        num_atoms, kT = 100, 1.0
        temps = self._sample_temperatures([num_atoms], kT, 4000, device, 1000)[:, 0]

        assert abs(temps.mean() - kT) < 0.1
        expected_std = kT * np.sqrt(2.0 / (3 * num_atoms))
        assert 0.5 * expected_std < temps.std() < 1.5 * expected_std

    @pytest.mark.parametrize("device", DEVICES)
    def test_batched_different_target_temperatures(self, device):
        """Each system equilibrates to its own target temperature."""
        kT = np.array([0.5, 1.0, 2.0])
        temps = self._sample_temperatures([64, 64, 64], kT, 3000, device, 1000)

        np.testing.assert_allclose(temps.mean(axis=0), kT, rtol=0.1)

    @pytest.mark.parametrize("device", DEVICES)
    def test_effective_energy_conserved(self, device):
        """KE + PE + energy given to the bath stays constant up to integration error."""
        num_atoms = 50
        rng = np.random.default_rng(42)
        positions = wp.array(
            0.5 * rng.standard_normal((num_atoms, 3)), dtype=wp.vec3d, device=device
        )
        velocities = wp.array(
            rng.standard_normal((num_atoms, 3)), dtype=wp.vec3d, device=device
        )
        masses = wp.ones(num_atoms, dtype=wp.float64, device=device)
        forces = wp.empty_like(positions)
        compute_harmonic_forces(positions, forces)
        target_temp, tau, dt, ndof, ke, scale = _params(
            wp.float64, device, kT=2.0, tau=0.05, dt=0.005, ndof=3.0 * num_atoms
        )

        def energy():
            vel = velocities.numpy()
            pos = positions.numpy()
            return 0.5 * np.sum(vel**2) + 0.5 * np.sum(pos**2)

        initial = energy()
        bath = 0.0
        drift = []
        for step in range(400):
            run_csvr_nvt_step(
                positions,
                velocities,
                forces,
                masses,
                target_temp,
                tau,
                dt,
                ndof,
                ke,
                scale,
                step,
            )
            bath += ke.numpy()[0] * (1.0 - scale.numpy()[0] ** 2)
            drift.append(energy() + bath - initial)

        assert abs(bath) > 1.0  # the thermostat did exchange energy
        assert np.max(np.abs(drift)) < 1e-3 * initial
