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

"""Framework-agnostic tests for the dispersion PME Warp kernels."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp
from scipy.special import erfc as np_erfc

from nvalchemiops.interactions.dispersion.pme_dispersion_kernels import (
    batch_pme_dispersion_green_structure_factor,
    batch_pme_dispersion_self_energy,
    lj_pme_real_space_energy,
    lj_pme_real_space_energy_forces,
    lj_pme_real_space_energy_forces_virial,
    pme_dispersion_green_structure_factor,
    pme_dispersion_self_energy,
)

PI = math.pi
SQRT_PI = math.sqrt(PI)
PI_3_2 = PI * SQRT_PI


def _np_dtype(wp_dtype):
    return np.float64 if wp_dtype == wp.float64 else np.float32


def _rtol(wp_dtype):
    return 1e-6 if wp_dtype == wp.float64 else 1e-4


def _scalar(value, device, wp_dtype):
    return wp.array([_np_dtype(wp_dtype)(value)], dtype=wp_dtype, device=device)


def _f_reference(x):
    """NumPy reference for f(x) = (1/3)[(1-2x²)e^(-x²) + 2x³√π·erfc(x)]."""
    x_sq = x * x
    return (1.0 / 3.0) * (
        (1.0 - 2.0 * x_sq) * np.exp(-x_sq) + 2.0 * x_sq * x * SQRT_PI * np_erfc(x)
    )


###########################################################################################
########################### Helper f(x) reference values ##################################
###########################################################################################


class TestDispersionFReference:
    """f(x) at known points — pure NumPy reference values used elsewhere."""

    def test_f_at_zero(self):
        """f(0) = 1/3 (erfc(0)=1, exp(0)=1, (1-0)*1 + 0 = 1)."""
        assert _f_reference(0.0) == pytest.approx(1.0 / 3.0, rel=1e-12)

    def test_f_decays_at_infinity(self):
        """f(x) → 0 as x → ∞."""
        assert _f_reference(10.0) == pytest.approx(0.0, abs=1e-10)

    def test_f_is_monotonically_decreasing(self):
        """f is monotonically decreasing on [0, ∞)."""
        xs = np.linspace(0.0, 5.0, 50)
        vals = _f_reference(xs)
        assert np.all(np.diff(vals) < 0.0)


###########################################################################################
########################### Green's Function Kernel #######################################
###########################################################################################


class TestDispersionGreenStructureFactor:
    """Tests for _pme_dispersion_green_structure_factor_kernel."""

    def test_shapes(self, device, wp_dtype):
        mesh_nx, mesh_ny, mesh_nz = 8, 8, 8
        nz_rfft = mesh_nz // 2 + 1

        k_squared = wp.zeros((mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device)
        miller_x = wp.zeros(mesh_nx, dtype=wp_dtype, device=device)
        miller_y = wp.zeros(mesh_ny, dtype=wp_dtype, device=device)
        miller_z = wp.zeros(nz_rfft, dtype=wp_dtype, device=device)
        beta = _scalar(0.3, device, wp_dtype)
        volume = _scalar(1000.0, device, wp_dtype)

        green_function = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        structure_factor_sq = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )

        pme_dispersion_green_structure_factor(
            k_squared=k_squared,
            miller_x=miller_x,
            miller_y=miller_y,
            miller_z=miller_z,
            beta=beta,
            volume=volume,
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=4,
            green_function=green_function,
            structure_factor_sq=structure_factor_sq,
            wp_dtype=wp_dtype,
            device=device,
        )
        assert green_function.shape == (mesh_nx, mesh_ny, nz_rfft)
        assert structure_factor_sq.shape == (mesh_nx, mesh_ny, nz_rfft)

    def test_green_k0_is_zero(self, device, wp_dtype):
        """G(k=0) is explicitly set to zero (m=0 mode excluded)."""
        mesh_nx, mesh_ny, mesh_nz = 4, 4, 4
        nz_rfft = mesh_nz // 2 + 1
        np_dtype = _np_dtype(wp_dtype)

        # All k_sq nonzero except (0,0,0)
        k_sq_np = np.ones((mesh_nx, mesh_ny, nz_rfft), dtype=np_dtype)
        k_sq_np[0, 0, 0] = 0.0
        k_squared = wp.array(k_sq_np, dtype=wp_dtype, device=device)

        miller_x = wp.zeros(mesh_nx, dtype=wp_dtype, device=device)
        miller_y = wp.zeros(mesh_ny, dtype=wp_dtype, device=device)
        miller_z = wp.zeros(nz_rfft, dtype=wp_dtype, device=device)
        beta = _scalar(0.3, device, wp_dtype)
        volume = _scalar(1000.0, device, wp_dtype)

        green_function = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        structure_factor_sq = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )

        pme_dispersion_green_structure_factor(
            k_squared=k_squared,
            miller_x=miller_x,
            miller_y=miller_y,
            miller_z=miller_z,
            beta=beta,
            volume=volume,
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=4,
            green_function=green_function,
            structure_factor_sq=structure_factor_sq,
            wp_dtype=wp_dtype,
            device=device,
        )
        assert green_function.numpy()[0, 0, 0] == 0.0

    def test_green_formula(self, device, wp_dtype):
        """G(k) = -(π^(3/2) β³ / (2V)) · f(|k|/(2β)) at non-zero k."""
        mesh_nx, mesh_ny, mesh_nz = 4, 4, 4
        nz_rfft = mesh_nz // 2 + 1
        np_dtype = _np_dtype(wp_dtype)
        beta_val = 0.5
        volume_val = 100.0

        k_sq_val = 4.0
        k_sq_np = np.full((mesh_nx, mesh_ny, nz_rfft), k_sq_val, dtype=np_dtype)
        k_sq_np[0, 0, 0] = 0.0
        k_squared = wp.array(k_sq_np, dtype=wp_dtype, device=device)

        miller_x = wp.zeros(mesh_nx, dtype=wp_dtype, device=device)
        miller_y = wp.zeros(mesh_ny, dtype=wp_dtype, device=device)
        miller_z = wp.zeros(nz_rfft, dtype=wp_dtype, device=device)
        beta = _scalar(beta_val, device, wp_dtype)
        volume = _scalar(volume_val, device, wp_dtype)

        green_function = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        structure_factor_sq = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )

        pme_dispersion_green_structure_factor(
            k_squared=k_squared,
            miller_x=miller_x,
            miller_y=miller_y,
            miller_z=miller_z,
            beta=beta,
            volume=volume,
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=4,
            green_function=green_function,
            structure_factor_sq=structure_factor_sq,
            wp_dtype=wp_dtype,
            device=device,
        )

        # Expected: G(k) = -(π^(3/2)·β³/(2V))·f(|k|/(2β))
        x = math.sqrt(k_sq_val) / (2.0 * beta_val)
        f_val = _f_reference(np.array(x, dtype=np.float64))
        expected = -PI_3_2 * (beta_val**3) * float(f_val) / (2.0 * volume_val)

        green_np = green_function.numpy()
        # All non-k=0 points share the same k_squared, so all should match expected
        for ix in range(mesh_nx):
            for iy in range(mesh_ny):
                for iz in range(nz_rfft):
                    if ix == 0 and iy == 0 and iz == 0:
                        assert green_np[ix, iy, iz] == 0.0
                    else:
                        assert green_np[ix, iy, iz] == pytest.approx(
                            expected, rel=_rtol(wp_dtype)
                        )

    def test_green_is_negative_for_nonzero_k(self, device, wp_dtype):
        """G(k) < 0 for k != 0 (attractive dispersion convention)."""
        mesh_nx, mesh_ny, mesh_nz = 8, 8, 8
        nz_rfft = mesh_nz // 2 + 1
        np_dtype = _np_dtype(wp_dtype)

        rng = np.random.default_rng(0)
        k_sq_np = rng.uniform(0.1, 10.0, size=(mesh_nx, mesh_ny, nz_rfft)).astype(
            np_dtype
        )
        k_sq_np[0, 0, 0] = 0.0
        k_squared = wp.array(k_sq_np, dtype=wp_dtype, device=device)

        miller_x = wp.zeros(mesh_nx, dtype=wp_dtype, device=device)
        miller_y = wp.zeros(mesh_ny, dtype=wp_dtype, device=device)
        miller_z = wp.zeros(nz_rfft, dtype=wp_dtype, device=device)
        beta = _scalar(0.5, device, wp_dtype)
        volume = _scalar(125.0, device, wp_dtype)

        green_function = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        structure_factor_sq = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )

        pme_dispersion_green_structure_factor(
            k_squared=k_squared,
            miller_x=miller_x,
            miller_y=miller_y,
            miller_z=miller_z,
            beta=beta,
            volume=volume,
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=4,
            green_function=green_function,
            structure_factor_sq=structure_factor_sq,
            wp_dtype=wp_dtype,
            device=device,
        )

        green_np = green_function.numpy()
        # k=0 exactly zero, all others strictly negative
        assert green_np[0, 0, 0] == 0.0
        assert np.all(green_np[1:] < 0.0) or np.all(green_np.reshape(-1)[1:] < 0.0)

    def test_batched_green_matches_single(self, device, wp_dtype):
        """Batched kernel result matches stacked single-system result."""
        mesh_nx, mesh_ny, mesh_nz = 8, 8, 8
        nz_rfft = mesh_nz // 2 + 1
        np_dtype = _np_dtype(wp_dtype)
        num_systems = 3

        rng = np.random.default_rng(42)
        k_sq_single = rng.uniform(0.1, 10.0, size=(mesh_nx, mesh_ny, nz_rfft)).astype(
            np_dtype
        )
        k_sq_single[0, 0, 0] = 0.0
        k_sq_batch_np = np.broadcast_to(
            k_sq_single, (num_systems, mesh_nx, mesh_ny, nz_rfft)
        ).copy()

        miller_x = wp.zeros(mesh_nx, dtype=wp_dtype, device=device)
        miller_y = wp.zeros(mesh_ny, dtype=wp_dtype, device=device)
        miller_z = wp.zeros(nz_rfft, dtype=wp_dtype, device=device)

        betas = np.array([0.3, 0.4, 0.5], dtype=np_dtype)
        volumes = np.array([100.0, 150.0, 200.0], dtype=np_dtype)

        # Batched run
        beta_batch = wp.array(betas, dtype=wp_dtype, device=device)
        volumes_batch = wp.array(volumes, dtype=wp_dtype, device=device)
        k_sq_batch = wp.array(k_sq_batch_np, dtype=wp_dtype, device=device)
        green_batch = wp.zeros(
            (num_systems, mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        sf_batch = wp.zeros((mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device)
        batch_pme_dispersion_green_structure_factor(
            k_squared=k_sq_batch,
            miller_x=miller_x,
            miller_y=miller_y,
            miller_z=miller_z,
            beta=beta_batch,
            volumes=volumes_batch,
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=4,
            green_function=green_batch,
            structure_factor_sq=sf_batch,
            wp_dtype=wp_dtype,
            device=device,
        )

        # Single runs
        for s in range(num_systems):
            beta_s = _scalar(float(betas[s]), device, wp_dtype)
            volume_s = _scalar(float(volumes[s]), device, wp_dtype)
            k_sq_s = wp.array(k_sq_single, dtype=wp_dtype, device=device)
            green_s = wp.zeros(
                (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
            )
            sf_s = wp.zeros((mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device)
            pme_dispersion_green_structure_factor(
                k_squared=k_sq_s,
                miller_x=miller_x,
                miller_y=miller_y,
                miller_z=miller_z,
                beta=beta_s,
                volume=volume_s,
                mesh_nx=mesh_nx,
                mesh_ny=mesh_ny,
                mesh_nz=mesh_nz,
                spline_order=4,
                green_function=green_s,
                structure_factor_sq=sf_s,
                wp_dtype=wp_dtype,
                device=device,
            )
            np.testing.assert_allclose(
                green_batch.numpy()[s],
                green_s.numpy(),
                rtol=_rtol(wp_dtype),
            )


###########################################################################################
########################### Self-Energy Kernel ############################################
###########################################################################################


class TestDispersionSelfEnergy:
    """Tests for _pme_dispersion_self_energy_kernel."""

    def test_per_atom_formula(self, device, wp_dtype):
        """Per-atom value equals -β⁶ · C6_ii / 12."""
        np_dtype = _np_dtype(wp_dtype)
        c6_np = np.array([1.0, 2.5, 0.5, 4.0], dtype=np_dtype)
        beta_val = 0.4

        c6 = wp.array(c6_np, dtype=wp_dtype, device=device)
        beta = _scalar(beta_val, device, wp_dtype)
        out = wp.zeros(len(c6_np), dtype=wp_dtype, device=device)

        pme_dispersion_self_energy(
            c6_coefficients=c6,
            beta=beta,
            energy_correction=out,
            wp_dtype=wp_dtype,
            device=device,
        )

        expected = -(beta_val**6) * c6_np / 12.0
        np.testing.assert_allclose(out.numpy(), expected, rtol=_rtol(wp_dtype))

    def test_zero_c6_gives_zero(self, device, wp_dtype):
        """Atoms with C6=0 have zero self-energy contribution."""
        np_dtype = _np_dtype(wp_dtype)
        c6_np = np.zeros(5, dtype=np_dtype)
        c6 = wp.array(c6_np, dtype=wp_dtype, device=device)
        beta = _scalar(0.3, device, wp_dtype)
        out = wp.zeros(5, dtype=wp_dtype, device=device)

        pme_dispersion_self_energy(
            c6_coefficients=c6,
            beta=beta,
            energy_correction=out,
            wp_dtype=wp_dtype,
            device=device,
        )
        np.testing.assert_array_equal(out.numpy(), np.zeros(5, dtype=np_dtype))

    def test_sign_is_negative_for_positive_c6(self, device, wp_dtype):
        """Self-energy is negative when C6 > 0 (attractive dispersion)."""
        np_dtype = _np_dtype(wp_dtype)
        c6_np = np.array([1.0, 2.0, 3.0], dtype=np_dtype)
        c6 = wp.array(c6_np, dtype=wp_dtype, device=device)
        beta = _scalar(0.5, device, wp_dtype)
        out = wp.zeros(3, dtype=wp_dtype, device=device)

        pme_dispersion_self_energy(
            c6_coefficients=c6,
            beta=beta,
            energy_correction=out,
            wp_dtype=wp_dtype,
            device=device,
        )
        assert np.all(out.numpy() < 0.0)

    def test_batched_self_energy_matches_per_system(self, device, wp_dtype):
        """Batched kernel respects per-system beta via batch_idx."""
        np_dtype = _np_dtype(wp_dtype)
        c6_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np_dtype)
        batch_idx_np = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        beta_np = np.array([0.3, 0.5], dtype=np_dtype)

        c6 = wp.array(c6_np, dtype=wp_dtype, device=device)
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)
        beta = wp.array(beta_np, dtype=wp_dtype, device=device)
        out = wp.zeros(5, dtype=wp_dtype, device=device)

        batch_pme_dispersion_self_energy(
            c6_coefficients=c6,
            batch_idx=batch_idx,
            beta=beta,
            energy_correction=out,
            wp_dtype=wp_dtype,
            device=device,
        )

        expected = np.empty_like(c6_np)
        for i in range(len(c6_np)):
            b = beta_np[batch_idx_np[i]]
            expected[i] = -(b**6) * c6_np[i] / 12.0
        np.testing.assert_allclose(out.numpy(), expected, rtol=_rtol(wp_dtype))


###########################################################################################
########################### Real-Space LJ-PME Kernels (PR2) ###############################
###########################################################################################


def _lj_pme_pair_energy(r, c6, c12, beta):
    """Reference V(r) = C12/r^12 - C6 * g(beta*r) / r^6 with g = exp(-x^2)(1+x^2+x^4/2)."""
    x_sq = (beta * r) ** 2
    g = np.exp(-x_sq) * (1.0 + x_sq + 0.5 * x_sq * x_sq)
    return c12 / r**12 - c6 * g / r**6


def _make_pair_inputs(r, c6_pair, c12_pair, beta_val, cutoff, device, wp_dtype):
    """Construct a 2-atom system with one half-neighbor pair separated by ``r``."""
    np_dtype = _np_dtype(wp_dtype)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    mat_dtype = wp.mat33d if wp_dtype == wp.float64 else wp.mat33f

    positions_np = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=np_dtype)
    positions = wp.array(positions_np, dtype=vec_dtype, device=device)

    cell_np = (np.eye(3, dtype=np_dtype) * 100.0).reshape(1, 3, 3)
    cell = wp.array(cell_np, dtype=mat_dtype, device=device)

    c6 = wp.array(
        np.array([c6_pair, c6_pair], dtype=np_dtype), dtype=wp_dtype, device=device
    )
    c12 = wp.array(
        np.array([c12_pair, c12_pair], dtype=np_dtype), dtype=wp_dtype, device=device
    )

    nbr_mat_np = np.array([[1], [2]], dtype=np.int32)  # half list: 0->1, 1->padding
    nbr_mat = wp.array(nbr_mat_np, dtype=wp.int32, device=device)
    nbr_shifts = wp.array(
        np.zeros((2, 1, 3), dtype=np.int32), dtype=wp.vec3i, device=device
    )
    num_nbrs = wp.array(np.array([1, 0], dtype=np.int32), dtype=wp.int32, device=device)

    beta = _scalar(beta_val, device, wp_dtype)
    cutoff_arr = _scalar(cutoff, device, wp_dtype)
    return {
        "positions": positions,
        "c6": c6,
        "c12": c12,
        "cell": cell,
        "nbr_mat": nbr_mat,
        "nbr_shifts": nbr_shifts,
        "num_nbrs": num_nbrs,
        "beta": beta,
        "cutoff": cutoff_arr,
        "mask_value": 2,
    }


class TestLJPMERealSpaceEnergy:
    """Pair energy correctness for the real-space LJ-PME kernel."""

    @pytest.mark.parametrize("r", [3.0, 4.0, 5.5])
    def test_pair_energy_matches_analytical(self, device, wp_dtype, r):
        c6, c12, beta_val, cutoff = 60.0, 5e4, 0.35, 10.0
        inp = _make_pair_inputs(r, c6, c12, beta_val, cutoff, device, wp_dtype)
        energies = wp.zeros(2, dtype=wp_dtype, device=device)
        lj_pme_real_space_energy(
            positions=inp["positions"],
            c6_coefficients=inp["c6"],
            c12_coefficients=inp["c12"],
            cell=inp["cell"],
            neighbor_matrix=inp["nbr_mat"],
            neighbor_matrix_shifts=inp["nbr_shifts"],
            num_neighbors=inp["num_nbrs"],
            beta=inp["beta"],
            cutoff=inp["cutoff"],
            mask_value=inp["mask_value"],
            atomic_energies=energies,
            wp_dtype=wp_dtype,
            device=device,
        )
        total = float(energies.numpy().sum())
        expected = _lj_pme_pair_energy(r, c6, c12, beta_val)
        assert total == pytest.approx(expected, rel=_rtol(wp_dtype))

    def test_zero_beta_recovers_full_lj(self, device, wp_dtype):
        """With beta=0 the damping g(0)=1 and we recover C12/r^12 - C6/r^6."""
        r, c6, c12 = 3.5, 60.0, 5e4
        # Note: kernel uses beta>0 in production; use a tiny beta to approximate.
        inp = _make_pair_inputs(r, c6, c12, 1e-8, 10.0, device, wp_dtype)
        energies = wp.zeros(2, dtype=wp_dtype, device=device)
        lj_pme_real_space_energy(
            positions=inp["positions"],
            c6_coefficients=inp["c6"],
            c12_coefficients=inp["c12"],
            cell=inp["cell"],
            neighbor_matrix=inp["nbr_mat"],
            neighbor_matrix_shifts=inp["nbr_shifts"],
            num_neighbors=inp["num_nbrs"],
            beta=inp["beta"],
            cutoff=inp["cutoff"],
            mask_value=inp["mask_value"],
            atomic_energies=energies,
            wp_dtype=wp_dtype,
            device=device,
        )
        total = float(energies.numpy().sum())
        bare_lj = c12 / r**12 - c6 / r**6
        assert total == pytest.approx(bare_lj, rel=_rtol(wp_dtype))

    def test_pair_beyond_cutoff_is_zero(self, device, wp_dtype):
        """Pairs with r > cutoff contribute zero."""
        r, c6, c12, beta_val, cutoff = 6.0, 60.0, 5e4, 0.35, 5.0
        inp = _make_pair_inputs(r, c6, c12, beta_val, cutoff, device, wp_dtype)
        energies = wp.zeros(2, dtype=wp_dtype, device=device)
        lj_pme_real_space_energy(
            positions=inp["positions"],
            c6_coefficients=inp["c6"],
            c12_coefficients=inp["c12"],
            cell=inp["cell"],
            neighbor_matrix=inp["nbr_mat"],
            neighbor_matrix_shifts=inp["nbr_shifts"],
            num_neighbors=inp["num_nbrs"],
            beta=inp["beta"],
            cutoff=inp["cutoff"],
            mask_value=inp["mask_value"],
            atomic_energies=energies,
            wp_dtype=wp_dtype,
            device=device,
        )
        assert float(energies.numpy().sum()) == 0.0

    def test_energy_split_equally_per_pair(self, device, wp_dtype):
        """Half-list: each atom of the pair receives V/2."""
        r, c6, c12, beta_val = 3.5, 60.0, 5e4, 0.35
        inp = _make_pair_inputs(r, c6, c12, beta_val, 10.0, device, wp_dtype)
        energies = wp.zeros(2, dtype=wp_dtype, device=device)
        lj_pme_real_space_energy(
            positions=inp["positions"],
            c6_coefficients=inp["c6"],
            c12_coefficients=inp["c12"],
            cell=inp["cell"],
            neighbor_matrix=inp["nbr_mat"],
            neighbor_matrix_shifts=inp["nbr_shifts"],
            num_neighbors=inp["num_nbrs"],
            beta=inp["beta"],
            cutoff=inp["cutoff"],
            mask_value=inp["mask_value"],
            atomic_energies=energies,
            wp_dtype=wp_dtype,
            device=device,
        )
        e_np = energies.numpy()
        assert e_np[0] == pytest.approx(e_np[1], rel=_rtol(wp_dtype))


class TestLJPMERealSpaceForces:
    """Force correctness for the real-space LJ-PME kernel."""

    def test_forces_satisfy_newton_third_law(self, device, wp_dtype):
        r, c6, c12, beta_val = 3.5, 60.0, 5e4, 0.35
        inp = _make_pair_inputs(r, c6, c12, beta_val, 10.0, device, wp_dtype)
        vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
        energies = wp.zeros(2, dtype=wp_dtype, device=device)
        forces = wp.zeros(2, dtype=vec_dtype, device=device)
        lj_pme_real_space_energy_forces(
            positions=inp["positions"],
            c6_coefficients=inp["c6"],
            c12_coefficients=inp["c12"],
            cell=inp["cell"],
            neighbor_matrix=inp["nbr_mat"],
            neighbor_matrix_shifts=inp["nbr_shifts"],
            num_neighbors=inp["num_nbrs"],
            beta=inp["beta"],
            cutoff=inp["cutoff"],
            mask_value=inp["mask_value"],
            atomic_energies=energies,
            atomic_forces=forces,
            wp_dtype=wp_dtype,
            device=device,
        )
        f = forces.numpy()
        # Forces are equal and opposite.
        np.testing.assert_allclose(
            f.sum(axis=0), 0.0, atol=1e-12 if wp_dtype == wp.float64 else 1e-5
        )
        np.testing.assert_allclose(f[1], -f[0], rtol=_rtol(wp_dtype))

    @pytest.mark.parametrize("r", [2.8, 3.5, 4.5])
    def test_forces_match_finite_difference(self, device, wp_dtype, r):
        """Compare against -dV/dr_i via central differences (float64 only)."""
        if wp_dtype == wp.float32:
            pytest.skip("FD test requires float64 precision")
        c6, c12, beta_val, cutoff = 60.0, 5e4, 0.35, 10.0

        def energy_at(r_val):
            inp = _make_pair_inputs(r_val, c6, c12, beta_val, cutoff, device, wp_dtype)
            e = wp.zeros(2, dtype=wp_dtype, device=device)
            lj_pme_real_space_energy(
                positions=inp["positions"],
                c6_coefficients=inp["c6"],
                c12_coefficients=inp["c12"],
                cell=inp["cell"],
                neighbor_matrix=inp["nbr_mat"],
                neighbor_matrix_shifts=inp["nbr_shifts"],
                num_neighbors=inp["num_nbrs"],
                beta=inp["beta"],
                cutoff=inp["cutoff"],
                mask_value=inp["mask_value"],
                atomic_energies=e,
                wp_dtype=wp_dtype,
                device=device,
            )
            return float(e.numpy().sum())

        h = 1e-5
        dV_dr = (energy_at(r + h) - energy_at(r - h)) / (2 * h)

        # Force on atom 1 (at +x) is F_1 = -dV/dr * r_hat_1, where r_hat_1 points
        # along +x (from atom 0 to atom 1 is +x; from atom 1 to atom 0 is -x).
        # With r_ij = r_i - r_j and F_i = (force_mag/r) * r_ij, atom 1 sees
        # r_ij = (-r, 0, 0) so F_1.x is negative when force_mag_over_r is positive.
        inp = _make_pair_inputs(r, c6, c12, beta_val, cutoff, device, wp_dtype)
        vec_dtype = wp.vec3d
        e_ = wp.zeros(2, dtype=wp_dtype, device=device)
        f_ = wp.zeros(2, dtype=vec_dtype, device=device)
        lj_pme_real_space_energy_forces(
            positions=inp["positions"],
            c6_coefficients=inp["c6"],
            c12_coefficients=inp["c12"],
            cell=inp["cell"],
            neighbor_matrix=inp["nbr_mat"],
            neighbor_matrix_shifts=inp["nbr_shifts"],
            num_neighbors=inp["num_nbrs"],
            beta=inp["beta"],
            cutoff=inp["cutoff"],
            mask_value=inp["mask_value"],
            atomic_energies=e_,
            atomic_forces=f_,
            wp_dtype=wp_dtype,
            device=device,
        )
        f_np = f_.numpy()
        # Atom 0 is at origin, atom 1 at (r, 0, 0). Pulling atom 1 outward by dr
        # increases r by dr, so F_1.x = -dV/dr.
        assert f_np[1, 0] == pytest.approx(-dV_dr, rel=1e-5)
        assert f_np[0, 0] == pytest.approx(dV_dr, rel=1e-5)


class TestLJPMERealSpaceVirial:
    """Virial correctness for the real-space LJ-PME kernel."""

    def test_virial_matches_pair_outer_product(self, device, wp_dtype):
        """W_xx = r_ij,x * F_ij,x for a single pair along x-axis."""
        r, c6, c12, beta_val = 3.5, 60.0, 5e4, 0.35
        inp = _make_pair_inputs(r, c6, c12, beta_val, 10.0, device, wp_dtype)
        vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
        energies = wp.zeros(2, dtype=wp_dtype, device=device)
        forces = wp.zeros(2, dtype=vec_dtype, device=device)
        virial = wp.zeros(9, dtype=wp_dtype, device=device)
        lj_pme_real_space_energy_forces_virial(
            positions=inp["positions"],
            c6_coefficients=inp["c6"],
            c12_coefficients=inp["c12"],
            cell=inp["cell"],
            neighbor_matrix=inp["nbr_mat"],
            neighbor_matrix_shifts=inp["nbr_shifts"],
            num_neighbors=inp["num_nbrs"],
            beta=inp["beta"],
            cutoff=inp["cutoff"],
            mask_value=inp["mask_value"],
            atomic_energies=energies,
            atomic_forces=forces,
            virial=virial,
            wp_dtype=wp_dtype,
            device=device,
        )
        vir = virial.numpy().reshape(3, 3)
        f_np = forces.numpy()
        # Single pair along +x: r_ij_0 (= r_i - r_j with i=0, j=1) = (-r, 0, 0).
        # F_ij is the force on i from this pair (= F_0 = -F_1).
        # W_ab = r_ij,a * F_ij,b. So W_xx = (-r) * F_0_x = -r * F_0_x.
        expected_xx = -r * f_np[0, 0]
        assert vir[0, 0] == pytest.approx(expected_xx, rel=_rtol(wp_dtype))
        # Off-diagonal entries are exactly zero (single colinear pair).
        for a in range(3):
            for b in range(3):
                if a == 0 and b == 0:
                    continue
                assert vir[a, b] == pytest.approx(
                    0.0, abs=1e-12 if wp_dtype == wp.float64 else 1e-5
                )
