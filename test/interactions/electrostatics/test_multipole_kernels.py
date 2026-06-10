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

"""Direct tests for the multipole direct-k and Ewald Warp kernels."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    ewald_real_space_energy,
)
from nvalchemiops.interactions.electrostatics.multipole_direct_kspace_kernels import (
    apply_per_k_factor,
    assemble_rho_k_dipole,
    build_structure_factor_table,
    compute_energy_product_per_k,
    eval_gto_fourier_dipole,
    eval_receiver_gto_fourier_dipole,
    project_features_dipole,
)
from nvalchemiops.interactions.electrostatics.multipole_ewald_kernels import (
    multipole_real_space_dipole_csr_energy,
    multipole_real_space_monopole_csr_energy,
)
from nvalchemiops.torch.math.gto import NormMode, inv_cl


class TestStructureFactorTable:
    """Tests for :func:`build_structure_factor_table`."""

    def _launch(self, k_vectors_np, positions_np, device, wp_dtype=wp.float64):
        np_scalar = np.float64 if wp_dtype == wp.float64 else np.float32
        vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
        k_vecs = wp.from_numpy(
            k_vectors_np.astype(np.float64), dtype=wp.vec3d, device=device
        )
        positions = wp.from_numpy(
            positions_np.astype(np_scalar), dtype=vec_dtype, device=device
        )
        n_k, n_atoms = k_vectors_np.shape[0], positions_np.shape[0]
        cos_tab = wp.zeros((n_k, n_atoms), dtype=wp.float64, device=device)
        sin_tab = wp.zeros((n_k, n_atoms), dtype=wp.float64, device=device)
        build_structure_factor_table(
            k_vecs, positions, cos_tab, sin_tab, wp_dtype=wp_dtype, device=device
        )
        return cos_tab.numpy(), sin_tab.numpy()

    def test_shapes(self, device):
        k = np.random.default_rng(0).standard_normal((7, 3))
        r = np.random.default_rng(1).standard_normal((5, 3))
        cos_tab, sin_tab = self._launch(k, r, device)
        assert cos_tab.shape == (7, 5)
        assert sin_tab.shape == (7, 5)

    def test_k_equals_zero(self, device):
        """For any k = 0 row, cos = 1 and sin = 0 for every atom."""
        k = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64)
        r = np.random.default_rng(7).standard_normal((4, 3))
        cos_tab, sin_tab = self._launch(k, r, device)
        np.testing.assert_array_equal(cos_tab[0], np.ones(4))
        np.testing.assert_array_equal(sin_tab[0], np.zeros(4))

    def test_atom_at_origin(self, device):
        """For any atom at r = 0, cos = 1 and sin = 0 for every k."""
        k = np.random.default_rng(11).standard_normal((6, 3)) * 2.0
        r = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        cos_tab, sin_tab = self._launch(k, r, device)
        np.testing.assert_array_equal(cos_tab[:, 0], np.ones(6))
        np.testing.assert_array_equal(sin_tab[:, 0], np.zeros(6))

    def test_matches_analytic_cos_sin(self, device):
        """Compare against direct numpy ``cos(k·r)``, ``sin(k·r)``."""
        rng = np.random.default_rng(23)
        k = rng.standard_normal((10, 3)) * 1.5
        r = rng.standard_normal((8, 3)) * 2.0
        cos_tab, sin_tab = self._launch(k, r, device)
        kr = k @ r.T  # (N_k, N_atoms)
        np.testing.assert_allclose(cos_tab, np.cos(kr), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(sin_tab, np.sin(kr), rtol=1e-14, atol=1e-14)

    def test_float32_positions(self, device):
        """Positions in float32 still give float64 output within 1 ULP of float64."""
        rng = np.random.default_rng(29)
        k = rng.standard_normal((6, 3))
        r = rng.standard_normal((4, 3))
        cos_f64, sin_f64 = self._launch(k, r, device, wp_dtype=wp.float64)
        cos_f32, sin_f32 = self._launch(k, r, device, wp_dtype=wp.float32)
        # Dot product accumulated in float64 keeps float32 positions close.
        np.testing.assert_allclose(cos_f32, cos_f64, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(sin_f32, sin_f64, rtol=0.0, atol=1e-6)


class TestGTOFourierDipole:
    """Tests for :func:`eval_gto_fourier_dipole`."""

    def _launch(self, k_vectors_np, sigma, mode, device):
        k_vecs = wp.from_numpy(
            k_vectors_np.astype(np.float64), dtype=wp.vec3d, device=device
        )
        k_norm2 = wp.from_numpy(
            (k_vectors_np**2).sum(axis=-1).astype(np.float64),
            dtype=wp.float64,
            device=device,
        )
        icl0 = inv_cl(sigma, 0, mode)
        icl1 = inv_cl(sigma, 1, mode)
        out = wp.zeros((k_vectors_np.shape[0], 4, 2), dtype=wp.float64, device=device)
        eval_gto_fourier_dipole(k_vecs, k_norm2, sigma, icl0, icl1, out, device=device)
        return out.numpy()

    def test_shape(self, device):
        k = np.random.default_rng(0).standard_normal((5, 3))
        out = self._launch(k, sigma=1.0, mode=NormMode.MULTIPOLES, device=device)
        assert out.shape == (5, 4, 2)

    def test_l0_is_real_l1_is_imaginary(self, device):
        """For l = 0 the imaginary part must be zero everywhere; for l = 1 the real part must be zero."""
        rng = np.random.default_rng(3)
        k = rng.standard_normal((12, 3)) * 2.0
        out = self._launch(k, sigma=0.8, mode=NormMode.MULTIPOLES, device=device)
        # l = 0 (index 0): imag == 0
        np.testing.assert_array_equal(out[:, 0, 1], np.zeros(12))
        # l = 1 (indices 1, 2, 3): real == 0
        np.testing.assert_array_equal(out[:, 1:4, 0], np.zeros((12, 3)))

    def test_k_equals_zero(self, device):
        r"""At k = 0 the l = 1 block vanishes; l = 0 equals the DC coefficient."""
        k = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        sigma = 1.0
        out = self._launch(k, sigma, NormMode.MULTIPOLES, device=device)
        # l = 1 block is zero (k^1 factor vanishes).
        np.testing.assert_array_equal(out[0, 1:4], np.zeros((3, 2)))
        icl0 = inv_cl(sigma, 0, NormMode.MULTIPOLES)
        expected_dc = (
            icl0
            * 4.0
            * math.pi
            * math.sqrt(math.pi / 2.0)
            * sigma**3
            * (1.0 / math.sqrt(4.0 * math.pi))
        )
        assert out[0, 0, 0] == pytest.approx(expected_dc, rel=1e-14)
        assert out[0, 0, 1] == 0.0

    def test_l0_is_rotation_invariant(self, device):
        """φ̂_{0,0} depends only on |k|, so rotating the k-vectors leaves it unchanged."""
        rng = np.random.default_rng(41)
        k = rng.standard_normal((8, 3))
        theta = 0.7
        R = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated_k = k @ R.T
        out_before = self._launch(k, 1.1, NormMode.MULTIPOLES, device=device)
        out_after = self._launch(rotated_k, 1.1, NormMode.MULTIPOLES, device=device)
        np.testing.assert_allclose(
            out_before[:, 0, 0], out_after[:, 0, 0], rtol=1e-14, atol=1e-14
        )

    def test_gaussian_decay(self, device):
        """φ̂ must decay as exp(-k²σ²/2) — halving σ increases the k² scale."""
        k = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        out_sigma1 = self._launch(k, 1.0, NormMode.MULTIPOLES, device=device)
        out_sigma2 = self._launch(k, 2.0, NormMode.MULTIPOLES, device=device)
        # Larger σ decays much faster in k, so the k=1/k=2 ratio is bigger.
        r1 = out_sigma1[0, 0, 0] / out_sigma1[1, 0, 0]
        r2 = out_sigma2[0, 0, 0] / out_sigma2[1, 0, 0]
        assert r2 > r1 * 10.0


def _build_inputs_for_rho(
    positions_np,
    charges_np,
    dipoles_np,
    k_vectors_np,
    sigma,
    device,
    wp_dtype=wp.float64,
    mode=NormMode.MULTIPOLES,
):
    """Upload a full set of arrays and evaluate cos/sin + φ̂ once for a test."""
    np_scalar = np.float64 if wp_dtype == wp.float64 else np.float32
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f

    k_vecs = wp.from_numpy(
        k_vectors_np.astype(np.float64), dtype=wp.vec3d, device=device
    )
    k_norm2_np = (k_vectors_np**2).sum(axis=-1)
    k_norm2 = wp.from_numpy(
        k_norm2_np.astype(np.float64), dtype=wp.float64, device=device
    )
    pos = wp.from_numpy(positions_np.astype(np_scalar), dtype=vec_dtype, device=device)
    charges = wp.from_numpy(charges_np.astype(np_scalar), dtype=wp_dtype, device=device)
    dipoles = wp.from_numpy(
        dipoles_np.astype(np_scalar), dtype=vec_dtype, device=device
    )

    n_k = k_vectors_np.shape[0]
    n_atoms = positions_np.shape[0]
    cos_tab = wp.zeros((n_k, n_atoms), dtype=wp.float64, device=device)
    sin_tab = wp.zeros((n_k, n_atoms), dtype=wp.float64, device=device)
    build_structure_factor_table(
        k_vecs, pos, cos_tab, sin_tab, wp_dtype=wp_dtype, device=device
    )

    gto_f = wp.zeros((n_k, 4, 2), dtype=wp.float64, device=device)
    icl0 = inv_cl(sigma, 0, mode)
    icl1 = inv_cl(sigma, 1, mode)
    eval_gto_fourier_dipole(k_vecs, k_norm2, sigma, icl0, icl1, gto_f, device=device)

    return {
        "charges": charges,
        "dipoles": dipoles,
        "cosines": cos_tab,
        "sines": sin_tab,
        "gto_fourier": gto_f,
        "k_norm2_np": k_norm2_np,
    }


class TestAssembleRhoKDipole:
    """Tests for :func:`assemble_rho_k_dipole`."""

    def test_shape(self, device):
        rng = np.random.default_rng(0)
        positions = rng.standard_normal((4, 3))
        charges = rng.uniform(-1.0, 1.0, 4)
        dipoles = rng.standard_normal((4, 3)) * 0.3
        k_vecs = rng.standard_normal((6, 3))
        inputs = _build_inputs_for_rho(
            positions, charges, dipoles, k_vecs, sigma=1.0, device=device
        )
        rho = wp.zeros((6, 2), dtype=wp.float64, device=device)
        assemble_rho_k_dipole(
            inputs["charges"],
            inputs["dipoles"],
            inputs["cosines"],
            inputs["sines"],
            inputs["gto_fourier"],
            volume=100.0,
            rho=rho,
            wp_dtype=wp.float64,
            device=device,
        )
        assert rho.shape == (6, 2)

    def test_zero_moments_give_zero_density(self, device):
        """All-zero charges and dipoles → ρ(k) = 0 at every k."""
        rng = np.random.default_rng(1)
        positions = rng.standard_normal((5, 3))
        charges = np.zeros(5, dtype=np.float64)
        dipoles = np.zeros((5, 3), dtype=np.float64)
        k_vecs = rng.standard_normal((4, 3))
        inputs = _build_inputs_for_rho(
            positions, charges, dipoles, k_vecs, sigma=1.0, device=device
        )
        rho = wp.zeros((4, 2), dtype=wp.float64, device=device)
        assemble_rho_k_dipole(
            inputs["charges"],
            inputs["dipoles"],
            inputs["cosines"],
            inputs["sines"],
            inputs["gto_fourier"],
            volume=100.0,
            rho=rho,
            wp_dtype=wp.float64,
            device=device,
        )
        np.testing.assert_array_equal(rho.numpy(), np.zeros((4, 2)))

    def test_neutral_system_zero_at_origin(self, device):
        r"""A net-neutral charge-only system gives :math:`\rho(k=0) = 0`."""
        rng = np.random.default_rng(2)
        positions = rng.standard_normal((5, 3))
        charges = rng.uniform(-1.0, 1.0, 5)
        charges -= charges.mean()  # net neutral
        dipoles = np.zeros((5, 3), dtype=np.float64)
        k_vecs = np.vstack(
            [np.zeros((1, 3), dtype=np.float64), rng.standard_normal((3, 3))]
        )
        inputs = _build_inputs_for_rho(
            positions, charges, dipoles, k_vecs, sigma=1.0, device=device
        )
        rho = wp.zeros((4, 2), dtype=wp.float64, device=device)
        assemble_rho_k_dipole(
            inputs["charges"],
            inputs["dipoles"],
            inputs["cosines"],
            inputs["sines"],
            inputs["gto_fourier"],
            volume=100.0,
            rho=rho,
            wp_dtype=wp.float64,
            device=device,
        )
        np.testing.assert_allclose(rho.numpy()[0], np.zeros(2), rtol=0.0, atol=1e-14)

    def test_float32_inputs(self, device):
        """Float32 charges / dipoles give float64 output close to float64 inputs."""
        rng = np.random.default_rng(91)
        positions = rng.standard_normal((4, 3))
        charges = rng.uniform(-1.0, 1.0, 4).astype(np.float32)
        dipoles = rng.standard_normal((4, 3)).astype(np.float32) * 0.3
        k_vecs = rng.standard_normal((3, 3))
        inputs_f64 = _build_inputs_for_rho(
            positions,
            charges.astype(np.float64),
            dipoles.astype(np.float64),
            k_vecs,
            sigma=1.0,
            device=device,
            wp_dtype=wp.float64,
        )
        inputs_f32 = _build_inputs_for_rho(
            positions,
            charges,
            dipoles,
            k_vecs,
            sigma=1.0,
            device=device,
            wp_dtype=wp.float32,
        )
        rho_f64 = wp.zeros((3, 2), dtype=wp.float64, device=device)
        rho_f32 = wp.zeros((3, 2), dtype=wp.float64, device=device)
        assemble_rho_k_dipole(
            inputs_f64["charges"],
            inputs_f64["dipoles"],
            inputs_f64["cosines"],
            inputs_f64["sines"],
            inputs_f64["gto_fourier"],
            volume=100.0,
            rho=rho_f64,
            wp_dtype=wp.float64,
            device=device,
        )
        assemble_rho_k_dipole(
            inputs_f32["charges"],
            inputs_f32["dipoles"],
            inputs_f32["cosines"],
            inputs_f32["sines"],
            inputs_f32["gto_fourier"],
            volume=100.0,
            rho=rho_f32,
            wp_dtype=wp.float32,
            device=device,
        )
        np.testing.assert_allclose(
            rho_f32.numpy(), rho_f64.numpy(), rtol=1e-6, atol=1e-6
        )


class TestApplyPerKFactor:
    """Tests for :func:`apply_per_k_factor`."""

    def test_shape_and_elementwise_multiply(self, device):
        rng = np.random.default_rng(3)
        n_k = 7
        rho_np = rng.standard_normal((n_k, 2))
        factor_np = rng.uniform(0.5, 2.0, n_k)
        rho = wp.from_numpy(rho_np, dtype=wp.float64, device=device)
        factor = wp.from_numpy(factor_np, dtype=wp.float64, device=device)
        pot = wp.zeros((n_k, 2), dtype=wp.float64, device=device)
        apply_per_k_factor(rho, factor, pot, device=device)
        np.testing.assert_allclose(
            pot.numpy(), rho_np * factor_np[:, None], rtol=0.0, atol=1e-15
        )

    def test_k0_factor_zero_gives_zero_potential(self, device):
        """Setting ``per_k_factor[0] = 0`` zeros the k=0 potential regardless of ρ."""
        rng = np.random.default_rng(11)
        rho_np = rng.standard_normal((5, 2))
        factor_np = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        rho = wp.from_numpy(rho_np, dtype=wp.float64, device=device)
        factor = wp.from_numpy(factor_np, dtype=wp.float64, device=device)
        pot = wp.zeros((5, 2), dtype=wp.float64, device=device)
        apply_per_k_factor(rho, factor, pot, device=device)
        np.testing.assert_array_equal(pot.numpy()[0], np.zeros(2))


class TestEnergyProductPerK:
    """Tests for :func:`compute_energy_product_per_k`."""

    def test_shape(self, device):
        rho = wp.from_numpy(
            np.random.default_rng(0).standard_normal((6, 2)),
            dtype=wp.float64,
            device=device,
        )
        pot = wp.from_numpy(
            np.random.default_rng(1).standard_normal((6, 2)),
            dtype=wp.float64,
            device=device,
        )
        per_k = wp.zeros(6, dtype=wp.float64, device=device)
        compute_energy_product_per_k(rho, pot, per_k, device=device)
        assert per_k.shape == (6,)

    def test_analytic_dot_product(self, device):
        """per_k[k] = 2 * (rho_r · V_r + rho_i · V_i) element by element."""
        rng = np.random.default_rng(3)
        rho_np = rng.standard_normal((8, 2))
        pot_np = rng.standard_normal((8, 2))
        rho = wp.from_numpy(rho_np, dtype=wp.float64, device=device)
        pot = wp.from_numpy(pot_np, dtype=wp.float64, device=device)
        per_k = wp.zeros(8, dtype=wp.float64, device=device)
        compute_energy_product_per_k(rho, pot, per_k, device=device)
        expected = 2.0 * (rho_np * pot_np).sum(axis=-1)
        np.testing.assert_allclose(per_k.numpy(), expected, rtol=1e-14, atol=1e-15)

    def test_zero_potential_gives_zero(self, device):
        """V = 0 → energy = 0 regardless of ρ (important for the k=0 convention)."""
        rng = np.random.default_rng(5)
        rho = wp.from_numpy(
            rng.standard_normal((4, 2)), dtype=wp.float64, device=device
        )
        pot = wp.zeros((4, 2), dtype=wp.float64, device=device)
        per_k = wp.zeros(4, dtype=wp.float64, device=device)
        compute_energy_product_per_k(rho, pot, per_k, device=device)
        np.testing.assert_array_equal(per_k.numpy(), np.zeros(4))


class TestReceiverGTOFourierDipole:
    """Tests for :func:`eval_receiver_gto_fourier_dipole`."""

    def _launch(self, k_vectors_np, sigmas, mode, device):
        k_vecs = wp.from_numpy(
            k_vectors_np.astype(np.float64), dtype=wp.vec3d, device=device
        )
        k_norm2 = wp.from_numpy(
            (k_vectors_np**2).sum(axis=-1).astype(np.float64),
            dtype=wp.float64,
            device=device,
        )
        sigmas_np = np.asarray(sigmas, dtype=np.float64)
        sig_wp = wp.from_numpy(sigmas_np, dtype=wp.float64, device=device)
        inv_cl_np = np.empty((len(sigmas_np), 2), dtype=np.float64)
        for i, s in enumerate(sigmas_np):
            inv_cl_np[i, 0] = inv_cl(float(s), 0, mode)
            inv_cl_np[i, 1] = inv_cl(float(s), 1, mode)
        inv_cl_wp = wp.from_numpy(inv_cl_np, dtype=wp.float64, device=device)
        out = wp.zeros(
            (k_vectors_np.shape[0], len(sigmas_np), 4, 2),
            dtype=wp.float64,
            device=device,
        )
        eval_receiver_gto_fourier_dipole(
            k_vecs, k_norm2, sig_wp, inv_cl_wp, out, device=device
        )
        return out.numpy()

    def test_shape(self, device):
        k = np.random.default_rng(0).standard_normal((7, 3))
        out = self._launch(k, [0.5, 1.0, 1.5], NormMode.RECEIVER, device=device)
        assert out.shape == (7, 3, 4, 2)

    def test_matches_source_kernel_when_single_sigma(self, device):
        r"""With one sigma, the receiver kernel output must match the source-side kernel."""
        rng = np.random.default_rng(5)
        k = rng.standard_normal((6, 3))
        sigma = 1.1
        mode = NormMode.MULTIPOLES
        out_multi = self._launch(k, [sigma], mode, device=device)

        k_vecs = wp.from_numpy(k.astype(np.float64), dtype=wp.vec3d, device=device)
        k_norm2 = wp.from_numpy(
            (k**2).sum(axis=-1).astype(np.float64), dtype=wp.float64, device=device
        )
        out_single_wp = wp.zeros((6, 4, 2), dtype=wp.float64, device=device)
        eval_gto_fourier_dipole(
            k_vecs,
            k_norm2,
            sigma,
            inv_cl(sigma, 0, mode),
            inv_cl(sigma, 1, mode),
            out_single_wp,
            device=device,
        )
        out_single = out_single_wp.numpy()
        # Identical arithmetic, identical float64 output.
        np.testing.assert_array_equal(out_multi[:, 0], out_single)


def _reference_project_to_features(
    potential: np.ndarray,
    feature_basis_fs: np.ndarray,
    cosines: np.ndarray,
    sines: np.ndarray,
    k_factor_proj: np.ndarray,
) -> np.ndarray:
    r"""Numpy port of the reference ``project_to_features_batch``."""
    n_k, n_sigma, m_dim = feature_basis_fs.shape[:3]
    sm_dim = n_sigma * m_dim
    proj_r = feature_basis_fs[..., 0].reshape(n_k, sm_dim)
    proj_i = feature_basis_fs[..., 1].reshape(n_k, sm_dim)
    v_r = potential[:, 0:1]
    v_i = potential[:, 1:2]
    A = v_r * proj_r + v_i * proj_i
    B = v_r * proj_i - v_i * proj_r
    A = A * k_factor_proj[:, None]
    B = B * k_factor_proj[:, None]
    proj_cos = A.T @ cosines
    proj_sin = B.T @ sines
    proj_total = 2.0 * (proj_cos + proj_sin)
    features = proj_total.T.reshape(cosines.shape[1], n_sigma, m_dim)
    return features / (2.0 * math.pi) ** 3


def _identity_out_col_lut(n_sigma: int) -> np.ndarray:
    """LUT that writes features in the natural flat layout ``s * 4 + lm``."""
    lut = np.empty((n_sigma, 4), dtype=np.int32)
    for s in range(n_sigma):
        for lm in range(4):
            lut[s, lm] = s * 4 + lm
    return lut


def _permuted_out_col_lut(n_sigma: int) -> np.ndarray:
    """LUT matching the reference output permutation for max_l=1."""
    lut = np.empty((n_sigma, 4), dtype=np.int32)
    for s in range(n_sigma):
        # l=0 block: first n_sigma slots, one per σ.
        lut[s, 0] = s
        # l=1 block: n_sigma slots consumed above, then 3 entries per σ.
        for m in range(3):
            lut[s, 1 + m] = n_sigma + s * 3 + m
    return lut


class TestProjectFeaturesDipole:
    """Tests for :func:`project_features_dipole`."""

    def _launch(
        self,
        potential_np,
        phi_np,
        cos_np,
        sin_np,
        k_factor_proj_np,
        device,
        *,
        source_feats_lm_np=None,
        overlap_constants_np=None,
        subtract_self=False,
        out_col_lut_np=None,
    ):
        """Launch with defaults — returns 3-D natural features unless overrides are passed."""
        n_atoms = cos_np.shape[1]
        n_sigma = phi_np.shape[1]
        if source_feats_lm_np is None:
            source_feats_lm_np = np.zeros((n_atoms, 4), dtype=np.float64)
        if overlap_constants_np is None:
            overlap_constants_np = np.zeros((n_sigma, 2), dtype=np.float64)
        if out_col_lut_np is None:
            out_col_lut_np = _identity_out_col_lut(n_sigma)

        potential = wp.from_numpy(potential_np, dtype=wp.float64, device=device)
        phi = wp.from_numpy(phi_np, dtype=wp.float64, device=device)
        cos_tab = wp.from_numpy(cos_np, dtype=wp.float64, device=device)
        sin_tab = wp.from_numpy(sin_np, dtype=wp.float64, device=device)
        kfp = wp.from_numpy(k_factor_proj_np, dtype=wp.float64, device=device)
        src_lm = wp.from_numpy(source_feats_lm_np, dtype=wp.float64, device=device)
        oc = wp.from_numpy(overlap_constants_np, dtype=wp.float64, device=device)
        lut = wp.from_numpy(out_col_lut_np, dtype=wp.int32, device=device)
        features = wp.zeros((n_atoms, n_sigma * 4), dtype=wp.float64, device=device)
        project_features_dipole(
            potential,
            phi,
            cos_tab,
            sin_tab,
            kfp,
            src_lm,
            oc,
            subtract_self,
            lut,
            features,
            device=device,
        )
        # Identity LUT → reshape to natural (N_atoms, N_σ, 4); else return raw 2-D.
        if np.array_equal(out_col_lut_np, _identity_out_col_lut(n_sigma)):
            return features.numpy().reshape(n_atoms, n_sigma, 4)
        return features.numpy()

    def test_shape(self, device):
        rng = np.random.default_rng(0)
        n_k, n_sigma, n_atoms = 5, 2, 4
        potential = rng.standard_normal((n_k, 2))
        phi = rng.standard_normal((n_k, n_sigma, 4, 2))
        cos_t = rng.standard_normal((n_k, n_atoms))
        sin_t = rng.standard_normal((n_k, n_atoms))
        kfp = np.ones(n_k)
        out = self._launch(potential, phi, cos_t, sin_t, kfp, device=device)
        assert out.shape == (n_atoms, n_sigma, 4)

    def test_zero_potential_gives_zero_features(self, device):
        """V(k) = 0 at every k → features = 0 regardless of receiver basis."""
        rng = np.random.default_rng(1)
        n_k, n_sigma, n_atoms = 4, 2, 3
        potential = np.zeros((n_k, 2))
        phi = rng.standard_normal((n_k, n_sigma, 4, 2))
        cos_t = rng.standard_normal((n_k, n_atoms))
        sin_t = rng.standard_normal((n_k, n_atoms))
        kfp = np.ones(n_k)
        out = self._launch(potential, phi, cos_t, sin_t, kfp, device=device)
        np.testing.assert_array_equal(out, np.zeros((n_atoms, n_sigma, 4)))

    def test_k0_factor_zero_suppresses_k0_contribution(self, device):
        r"""``k_factor_proj[0] = 0`` should zero out any ``k=0`` contribution."""
        rng = np.random.default_rng(2)
        n_k, n_sigma, n_atoms = 3, 2, 3
        potential = np.zeros((n_k, 2))
        potential[0] = [1.0, 0.5]  # non-zero only at k=0
        phi = rng.standard_normal((n_k, n_sigma, 4, 2))
        cos_t = rng.standard_normal((n_k, n_atoms))
        sin_t = rng.standard_normal((n_k, n_atoms))

        kfp_zero = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        kfp_half = np.array([0.5, 1.0, 1.0], dtype=np.float64)

        out_zero = self._launch(potential, phi, cos_t, sin_t, kfp_zero, device=device)
        out_half = self._launch(potential, phi, cos_t, sin_t, kfp_half, device=device)

        # With kfp[0]=0, features are zero. With kfp[0]=0.5, they aren't.
        np.testing.assert_array_equal(out_zero, np.zeros((n_atoms, n_sigma, 4)))
        assert np.any(out_half != 0.0)

    def test_matches_inlined_reference(self, device):
        """Parity with the numpy port of ``project_to_features_batch``."""
        rng = np.random.default_rng(42)
        n_k, n_sigma, n_atoms = 9, 3, 5
        potential = rng.standard_normal((n_k, 2))
        phi = rng.standard_normal((n_k, n_sigma, 4, 2))
        cos_t = rng.standard_normal((n_k, n_atoms))
        sin_t = rng.standard_normal((n_k, n_atoms))
        kfp = np.ones(n_k)
        kfp[0] = 0.5  # the reference convention at the origin

        ours = self._launch(potential, phi, cos_t, sin_t, kfp, device=device)
        expected = _reference_project_to_features(
            potential=potential,
            feature_basis_fs=phi,
            cosines=cos_t,
            sines=sin_t,
            k_factor_proj=kfp,
        )
        # Atomic-free kernel vs matmul: float64 summation-order drift < 1e-13.
        np.testing.assert_allclose(ours, expected, rtol=1e-13, atol=1e-14)

    def test_self_interaction_subtract_equals_torch_subtract(self, device):
        """Kernel-fused self-interaction subtract matches explicit torch subtraction after the fact."""
        rng = np.random.default_rng(101)
        n_k, n_sigma, n_atoms = 7, 2, 4
        potential = rng.standard_normal((n_k, 2))
        phi = rng.standard_normal((n_k, n_sigma, 4, 2))
        cos_t = rng.standard_normal((n_k, n_atoms))
        sin_t = rng.standard_normal((n_k, n_atoms))
        kfp = np.ones(n_k)
        src_lm = rng.standard_normal((n_atoms, 4))
        oc = rng.standard_normal((n_sigma, 2))

        without = self._launch(potential, phi, cos_t, sin_t, kfp, device=device)
        with_self = self._launch(
            potential,
            phi,
            cos_t,
            sin_t,
            kfp,
            device=device,
            source_feats_lm_np=src_lm,
            overlap_constants_np=oc,
            subtract_self=True,
        )

        # oc[s, 0] for lm=0; oc[s, 1] broadcast to lm=1, 2, 3.
        oc_per_lm = np.empty((n_sigma, 4), dtype=np.float64)
        oc_per_lm[:, 0] = oc[:, 0]
        oc_per_lm[:, 1:4] = oc[:, 1:2]  # broadcast
        self_corr = src_lm[:, None, :] * oc_per_lm[None, :, :]  # (N, N_σ, 4)
        expected = without - self_corr
        np.testing.assert_allclose(with_self, expected, rtol=1e-14, atol=1e-14)

    def test_subtract_self_false_ignores_source_feats_and_oc(self, device):
        """With ``subtract_self=False``, garbage in source_feats_lm / oc must not affect the output."""
        rng = np.random.default_rng(103)
        n_k, n_sigma, n_atoms = 5, 2, 3
        potential = rng.standard_normal((n_k, 2))
        phi = rng.standard_normal((n_k, n_sigma, 4, 2))
        cos_t = rng.standard_normal((n_k, n_atoms))
        sin_t = rng.standard_normal((n_k, n_atoms))
        kfp = np.ones(n_k)

        baseline = self._launch(potential, phi, cos_t, sin_t, kfp, device=device)
        garbage_src = rng.standard_normal((n_atoms, 4)) * 1e6
        garbage_oc = rng.standard_normal((n_sigma, 2)) * 1e6
        contaminated = self._launch(
            potential,
            phi,
            cos_t,
            sin_t,
            kfp,
            device=device,
            source_feats_lm_np=garbage_src,
            overlap_constants_np=garbage_oc,
            subtract_self=False,
        )
        np.testing.assert_array_equal(baseline, contaminated)

    def test_permuted_lut_writes_permuted_layout(self, device):
        r"""The permuted LUT changes only the column index, not the per-value result."""
        rng = np.random.default_rng(113)
        n_k, n_sigma, n_atoms = 6, 3, 4
        potential = rng.standard_normal((n_k, 2))
        phi = rng.standard_normal((n_k, n_sigma, 4, 2))
        cos_t = rng.standard_normal((n_k, n_atoms))
        sin_t = rng.standard_normal((n_k, n_atoms))
        kfp = np.ones(n_k)

        natural = self._launch(potential, phi, cos_t, sin_t, kfp, device=device)
        lut = _permuted_out_col_lut(n_sigma)
        permuted_flat = self._launch(
            potential,
            phi,
            cos_t,
            sin_t,
            kfp,
            device=device,
            out_col_lut_np=lut,
        )
        assert permuted_flat.shape == (n_atoms, n_sigma * 4)
        for s in range(n_sigma):
            # l=0 column for σ=s lives at flat index s.
            np.testing.assert_array_equal(permuted_flat[:, s], natural[:, s, 0])
            # l=1 block for σ=s: 3 consecutive slots starting at n_sigma + s * 3.
            base = n_sigma + s * 3
            np.testing.assert_array_equal(
                permuted_flat[:, base : base + 3], natural[:, s, 1:4]
            )


def _kvecs_and_norms_np(rng, n_k: int):
    kv = rng.normal(size=(n_k, 3)).astype(np.float64)
    kv[0] = 0.0  # origin row — covers the k=0 special case
    k2 = (kv * kv).sum(axis=-1)
    return kv, k2


def _fd_kernel_output(call, *arrays, eps: float = 1e-6):
    """FD of scalar ``L(arg_i) = (go * call(arg_i)).sum()`` w.r.t. one arg."""


class TestSourcePhiHatBackward:
    """Backward of :func:`eval_gto_fourier_dipole`."""

    def test_k1_vs_fd(self):
        from nvalchemiops.interactions.electrostatics.multipole_direct_kspace_kernels import (  # noqa: E501
            source_phi_hat_backward_dipole,
        )

        rng = np.random.default_rng(11)
        n_k = 6
        kv_np, k2_np = _kvecs_and_norms_np(rng, n_k)
        sigma = 0.7
        icl0, icl1 = 0.9, 0.3
        go = rng.normal(size=(n_k, 4, 2))

        wp_dev = wp.get_device("cuda:0")
        kv_wp = wp.array(kv_np, dtype=wp.vec3d, device=wp_dev)
        k2_wp = wp.array(k2_np, dtype=wp.float64, device=wp_dev)

        gk_vec = wp.empty(n_k, dtype=wp.vec3d, device=wp_dev)
        gk_n2 = wp.empty(n_k, dtype=wp.float64, device=wp_dev)
        source_phi_hat_backward_dipole(
            wp.array(go, dtype=wp.float64, device=wp_dev),
            kv_wp,
            k2_wp,
            sigma,
            icl0,
            icl1,
            gk_vec,
            gk_n2,
            device=str(wp_dev),
        )
        an_kv = gk_vec.numpy().reshape(n_k, 3)
        an_k2 = gk_n2.numpy()

        # FD via the forward kernel.
        def fwd(kv, k2):
            out = wp.empty((n_k, 4, 2), dtype=wp.float64, device=wp_dev)
            eval_gto_fourier_dipole(
                wp.array(kv, dtype=wp.vec3d, device=wp_dev),
                wp.array(k2, dtype=wp.float64, device=wp_dev),
                sigma,
                icl0,
                icl1,
                out,
                device=str(wp_dev),
            )
            return out.numpy()

        fd_kv = np.zeros_like(kv_np)
        eps = 1e-5
        for i in range(n_k):
            for d in range(3):
                p = kv_np.copy()
                p[i, d] += eps
                m = kv_np.copy()
                m[i, d] -= eps
                fd_kv[i, d] = (go * (fwd(p, k2_np) - fwd(m, k2_np))).sum() / (2 * eps)
        fd_k2 = np.zeros_like(k2_np)
        for i in range(n_k):
            p = k2_np.copy()
            p[i] += eps
            m = k2_np.copy()
            m[i] -= eps
            fd_k2[i] = (go * (fwd(kv_np, p) - fwd(kv_np, m))).sum() / (2 * eps)

        np.testing.assert_allclose(an_kv, fd_kv, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(an_k2, fd_k2, atol=1e-6, rtol=1e-6)


class TestReceiverPhiHatBackward:
    """Backward of :func:`eval_receiver_gto_fourier_dipole`."""

    def test_k2_vs_fd(self):
        from nvalchemiops.interactions.electrostatics.multipole_direct_kspace_kernels import (  # noqa: E501
            receiver_phi_hat_backward_dipole,
        )

        rng = np.random.default_rng(17)
        n_k, n_sigma = 6, 3
        kv_np, k2_np = _kvecs_and_norms_np(rng, n_k)
        sigmas = np.array([0.5, 0.7, 1.0])
        icl = rng.uniform(0.2, 1.0, size=(n_sigma, 2))
        go = rng.normal(size=(n_k, n_sigma, 4, 2))

        wp_dev = wp.get_device("cuda:0")
        kv_wp = wp.array(kv_np, dtype=wp.vec3d, device=wp_dev)
        k2_wp = wp.array(k2_np, dtype=wp.float64, device=wp_dev)
        sigmas_wp = wp.array(sigmas, dtype=wp.float64, device=wp_dev)
        icl_wp = wp.array(icl, dtype=wp.float64, device=wp_dev)
        gk_vec = wp.empty(n_k, dtype=wp.vec3d, device=wp_dev)
        gk_n2 = wp.empty(n_k, dtype=wp.float64, device=wp_dev)
        receiver_phi_hat_backward_dipole(
            wp.array(go, dtype=wp.float64, device=wp_dev),
            kv_wp,
            k2_wp,
            sigmas_wp,
            icl_wp,
            gk_vec,
            gk_n2,
            device=str(wp_dev),
        )
        an_kv = gk_vec.numpy().reshape(n_k, 3)
        an_k2 = gk_n2.numpy()

        def fwd(kv, k2):
            out = wp.empty((n_k, n_sigma, 4, 2), dtype=wp.float64, device=wp_dev)
            eval_receiver_gto_fourier_dipole(
                wp.array(kv, dtype=wp.vec3d, device=wp_dev),
                wp.array(k2, dtype=wp.float64, device=wp_dev),
                sigmas_wp,
                icl_wp,
                out,
                device=str(wp_dev),
            )
            return out.numpy()

        eps = 1e-5
        fd_kv = np.zeros_like(kv_np)
        for i in range(n_k):
            for d in range(3):
                p = kv_np.copy()
                p[i, d] += eps
                m = kv_np.copy()
                m[i, d] -= eps
                fd_kv[i, d] = (go * (fwd(p, k2_np) - fwd(m, k2_np))).sum() / (2 * eps)
        fd_k2 = np.zeros_like(k2_np)
        for i in range(n_k):
            p = k2_np.copy()
            p[i] += eps
            m = k2_np.copy()
            m[i] -= eps
            fd_k2[i] = (go * (fwd(kv_np, p) - fwd(kv_np, m))).sum() / (2 * eps)

        np.testing.assert_allclose(an_kv, fd_kv, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(an_k2, fd_k2, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _random_system(
    num_atoms: int,
    box_size: float,
    rng: np.random.Generator,
) -> dict:
    """Return a dict of numpy arrays describing a simple charge-neutral system."""
    positions = rng.uniform(0.0, box_size, size=(num_atoms, 3)).astype(np.float64)
    charges = rng.uniform(-1.0, 1.0, size=num_atoms).astype(np.float64)
    charges -= charges.mean()
    # Cell far bigger than the cutoff so no periodic images enter the half list.
    L = max(box_size * 3.0, 100.0)
    cell = np.array([[[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]], dtype=np.float64)
    # Half neighbor list (each pair once): atom i's neighbors are j > i.
    idx_j: list[int] = []
    neighbor_ptr: list[int] = [0]
    unit_shifts: list[list[int]] = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            idx_j.append(j)
            unit_shifts.append([0, 0, 0])
        neighbor_ptr.append(len(idx_j))
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "idx_j": np.asarray(idx_j, dtype=np.int32),
        "neighbor_ptr": np.asarray(neighbor_ptr, dtype=np.int32),
        "unit_shifts": np.asarray(unit_shifts, dtype=np.int32),
        "num_atoms": num_atoms,
    }


def _triclinic_system(rng: np.random.Generator) -> dict:
    """6-atom system in a non-orthogonal triclinic cell (no periodic shifts used)."""
    positions = rng.uniform(0.0, 5.0, size=(6, 3)).astype(np.float64)
    charges = rng.uniform(-1.0, 1.0, size=6).astype(np.float64)
    charges -= charges.mean()
    cell = np.array(
        [
            [
                [10.0, 0.0, 0.0],
                [2.5, 9.5, 0.0],
                [1.3, 1.1, 11.0],
            ]
        ],
        dtype=np.float64,
    )
    idx_j: list[int] = []
    neighbor_ptr: list[int] = [0]
    unit_shifts: list[list[int]] = []
    for i in range(6):
        for j in range(i + 1, 6):
            idx_j.append(j)
            unit_shifts.append([0, 0, 0])
        neighbor_ptr.append(len(idx_j))
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "idx_j": np.asarray(idx_j, dtype=np.int32),
        "neighbor_ptr": np.asarray(neighbor_ptr, dtype=np.int32),
        "unit_shifts": np.asarray(unit_shifts, dtype=np.int32),
        "num_atoms": 6,
    }


def _to_warp(system: dict, device: str, wp_dtype: type) -> dict:
    """Upload a numpy-backed system dict to Warp arrays on ``device`` at ``wp_dtype``."""
    np_scalar = np.float64 if wp_dtype == wp.float64 else np.float32
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    mat_dtype = wp.mat33d if wp_dtype == wp.float64 else wp.mat33f
    return {
        "positions": wp.from_numpy(
            system["positions"].astype(np_scalar), dtype=vec_dtype, device=device
        ),
        "charges": wp.from_numpy(
            system["charges"].astype(np_scalar), dtype=wp_dtype, device=device
        ),
        "cell": wp.from_numpy(
            system["cell"].astype(np_scalar), dtype=mat_dtype, device=device
        ),
        "idx_j": wp.from_numpy(system["idx_j"], dtype=wp.int32, device=device),
        "neighbor_ptr": wp.from_numpy(
            system["neighbor_ptr"], dtype=wp.int32, device=device
        ),
        "unit_shifts": wp.from_numpy(
            system["unit_shifts"], dtype=wp.vec3i, device=device
        ),
    }


_COLLAPSE_SIGMA = 0.5


def _sigma(device: str, wp_dtype: type = wp.float64, value: float = 0.0):
    r"""GTO width ``σ`` array for the real-space launchers.

    At ``σ → 0`` the GTO charge ``T^(0)`` reduces bit-for-bit to the legacy
    monopole Ewald ``erfc(α r)/r`` form, so the default ``σ = 0`` keeps the
    monopole-collapse / closed-form assertions valid.
    """
    np_scalar = np.float64 if wp_dtype == wp.float64 else np.float32
    return wp.from_numpy(
        np.array([value], dtype=np_scalar), dtype=wp_dtype, device=device
    )


def _launch_both(
    system: dict,
    device: str,
    alpha_value: float,
    wp_dtype: type = wp.float64,
    sigma_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run both kernels on the same inputs; return per-atom energies (ref, new).

    ``ref`` is the legacy monopole ``ewald_real_space_energy`` (σ-free); ``new``
    is the GTO ``multipole_real_space_monopole_csr_energy`` at ``sigma_value``
    (default ``σ = 0`` → bit-identical to ``ref``).
    """
    inputs = _to_warp(system, device, wp_dtype)
    alpha = wp.from_numpy(
        np.array(
            [alpha_value],
            dtype=np.float64 if wp_dtype == wp.float64 else np.float32,
        ),
        dtype=wp_dtype,
        device=device,
    )
    num_atoms = system["num_atoms"]
    ref = wp.zeros(num_atoms, dtype=wp.float64, device=device)
    new = wp.zeros(num_atoms, dtype=wp.float64, device=device)

    ewald_real_space_energy(
        inputs["positions"],
        inputs["charges"],
        inputs["cell"],
        inputs["idx_j"],
        inputs["neighbor_ptr"],
        inputs["unit_shifts"],
        alpha,
        ref,
        wp_dtype=wp_dtype,
        device=device,
    )
    multipole_real_space_monopole_csr_energy(
        inputs["positions"],
        inputs["charges"],
        inputs["cell"],
        inputs["idx_j"],
        inputs["neighbor_ptr"],
        inputs["unit_shifts"],
        _sigma(device, wp_dtype, sigma_value),
        alpha,
        new,
        wp_dtype=wp_dtype,
        device=device,
    )
    return ref.numpy(), new.numpy()


class TestMonopoleCollapse:
    r"""l_max=0 branch must reproduce :func:`ewald_real_space_energy` exactly."""

    def test_two_atoms_closed_form(self, device):
        """Two opposite charges at r=3, alpha=0.3 — closed-form plus parity."""
        system = {
            "positions": np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64),
            "charges": np.array([1.0, -1.0], dtype=np.float64),
            "cell": np.array(
                [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
                dtype=np.float64,
            ),
            "idx_j": np.array([1], dtype=np.int32),
            "neighbor_ptr": np.array([0, 1, 1], dtype=np.int32),
            "unit_shifts": np.array([[0, 0, 0]], dtype=np.int32),
            "num_atoms": 2,
        }
        alpha = 0.3
        ref, new = _launch_both(system, device, alpha)
        np.testing.assert_allclose(new, ref, rtol=0.0, atol=1e-15)
        expected = 0.5 * 1.0 * -1.0 * math.erfc(alpha * 3.0) / 3.0
        assert float(new.sum()) == pytest.approx(expected, abs=1e-7)

    @pytest.mark.parametrize("seed", [0, 7, 42, 123])
    @pytest.mark.parametrize("num_atoms", [3, 10, 25])
    def test_random_systems_float64(self, device, seed, num_atoms):
        """Random orthogonal systems, float64 — expect kernel-to-kernel parity at 1 ULP."""
        rng = np.random.default_rng(seed)
        system = _random_system(num_atoms, box_size=5.0, rng=rng)
        alpha = 0.4
        ref, new = _launch_both(system, device, alpha, wp_dtype=wp.float64)
        np.testing.assert_allclose(new, ref, rtol=0.0, atol=1e-15)

    def test_triclinic_cell_float64(self, device):
        """Triclinic (non-orthogonal) cell: per-edge periodic shift is zero here,
        but the cell transpose in the kernel still has off-diagonal entries. This
        catches any mismatch in how ``cell[0]`` is applied."""
        rng = np.random.default_rng(19)
        system = _triclinic_system(rng)
        alpha = 0.35
        ref, new = _launch_both(system, device, alpha, wp_dtype=wp.float64)
        np.testing.assert_allclose(new, ref, rtol=0.0, atol=1e-15)

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_random_systems_float32(self, device, seed):
        """Float32 inputs (float64 accumulators). Both kernels agree to 1 ULP float64."""
        rng = np.random.default_rng(seed)
        system = _random_system(10, box_size=5.0, rng=rng)
        alpha = 0.5
        ref, new = _launch_both(system, device, alpha, wp_dtype=wp.float32)
        np.testing.assert_allclose(new, ref, rtol=1e-13, atol=1e-15)

    def test_nonzero_unit_shifts(self, device):
        """Pairs with non-zero periodic shifts: verify that cell · shift is
        plumbed identically through both kernels."""
        # Two atoms at (0,0,0) and (1,0,0) in a 5x5x5 cell.
        # Use a periodic shift of (1, 0, 0) so the effective separation is 6.0.
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        charges = np.array([1.0, -1.0], dtype=np.float64)
        cell = np.array(
            [[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]], dtype=np.float64
        )
        system = {
            "positions": positions,
            "charges": charges,
            "cell": cell,
            "idx_j": np.array([1], dtype=np.int32),
            "neighbor_ptr": np.array([0, 1, 1], dtype=np.int32),
            "unit_shifts": np.array([[1, 0, 0]], dtype=np.int32),
            "num_atoms": 2,
        }
        alpha = 0.4
        ref, new = _launch_both(system, device, alpha)
        np.testing.assert_allclose(new, ref, rtol=0.0, atol=1e-15)
        # Effective separation |(1,0,0) + (5,0,0)| = 6.
        expected = 0.5 * 1.0 * -1.0 * math.erfc(alpha * 6.0) / 6.0
        assert float(new.sum()) == pytest.approx(expected, abs=1e-7)


class TestT0InteractionTensor:
    """Indirect tests for ``damped_coulomb_T0`` via the public kernel."""

    def test_matches_analytical_erfc_over_r(self, device):
        """Varying r at fixed alpha traces out erfc(alpha*r)/r exactly."""
        distances = [0.5, 1.0, 2.5, 4.0, 10.0]
        for r in distances:
            system = {
                "positions": np.array(
                    [[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=np.float64
                ),
                "charges": np.array([1.0, 1.0], dtype=np.float64),  # self-repulsion
                "cell": np.array(
                    [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
                    dtype=np.float64,
                ),
                "idx_j": np.array([1], dtype=np.int32),
                "neighbor_ptr": np.array([0, 1, 1], dtype=np.int32),
                "unit_shifts": np.array([[0, 0, 0]], dtype=np.int32),
                "num_atoms": 2,
            }
            alpha = 0.25
            _, new = _launch_both(system, device, alpha)
            # wp_erfc's ~1e-7 absolute error dominates at large r → use abs tol.
            expected = 0.5 * math.erfc(alpha * r) / r
            assert float(new.sum()) == pytest.approx(expected, abs=1e-7)


def _launch_dipole(
    *,
    positions: np.ndarray,
    charges: np.ndarray,
    dipoles: np.ndarray,
    cell: np.ndarray,
    idx_j: np.ndarray,
    neighbor_ptr: np.ndarray,
    unit_shifts: np.ndarray,
    alpha_value: float,
    device: str,
    wp_dtype: type = wp.float64,
    sigma_value: float = 0.0,
) -> np.ndarray:
    """Run the l_max=1 kernel on numpy-backed inputs and return per-atom energies.

    ``σ = 0`` (default) makes the charge T^(0) collapse bit-exactly to the legacy
    monopole kernel. Dipole T-tensors have a removable ``1/σ`` singularity at
    ``σ = 0``, so dipole-physics callers pass a small ``σ > 0`` (at ``α → 0`` the
    bare-Coulomb formulas hold for any such ``σ``).
    """
    np_scalar = np.float64 if wp_dtype == wp.float64 else np.float32
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    mat_dtype = wp.mat33d if wp_dtype == wp.float64 else wp.mat33f
    positions_wp = wp.from_numpy(
        positions.astype(np_scalar), dtype=vec_dtype, device=device
    )
    charges_wp = wp.from_numpy(charges.astype(np_scalar), dtype=wp_dtype, device=device)
    dipoles_wp = wp.from_numpy(
        dipoles.astype(np_scalar), dtype=vec_dtype, device=device
    )
    cell_wp = wp.from_numpy(cell.astype(np_scalar), dtype=mat_dtype, device=device)
    idx_j_wp = wp.from_numpy(idx_j, dtype=wp.int32, device=device)
    neighbor_ptr_wp = wp.from_numpy(neighbor_ptr, dtype=wp.int32, device=device)
    unit_shifts_wp = wp.from_numpy(unit_shifts, dtype=wp.vec3i, device=device)
    alpha_wp = wp.from_numpy(
        np.array([alpha_value], dtype=np_scalar),
        dtype=wp_dtype,
        device=device,
    )
    out = wp.zeros(positions.shape[0], dtype=wp.float64, device=device)
    multipole_real_space_dipole_csr_energy(
        positions_wp,
        charges_wp,
        dipoles_wp,
        cell_wp,
        idx_j_wp,
        neighbor_ptr_wp,
        unit_shifts_wp,
        wp.from_numpy(
            np.array([sigma_value], dtype=np_scalar), dtype=wp_dtype, device=device
        ),
        alpha_wp,
        out,
        wp_dtype=wp_dtype,
        device=device,
    )
    return out.numpy()


def _two_atom_pair_system(
    *,
    distance: float,
    charges: tuple[float, float] = (0.0, 0.0),
    dipole_i: tuple[float, float, float] = (0.0, 0.0, 0.0),
    dipole_j: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    """Two atoms along ``+x`` with the given charges and dipoles."""
    return {
        "positions": np.array(
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=np.float64
        ),
        "charges": np.array(charges, dtype=np.float64),
        "dipoles": np.array([dipole_i, dipole_j], dtype=np.float64),
        "cell": np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        ),
        "idx_j": np.array([1], dtype=np.int32),
        "neighbor_ptr": np.array([0, 1, 1], dtype=np.int32),
        "unit_shifts": np.array([[0, 0, 0]], dtype=np.int32),
    }


class TestDipoleMonopoleCollapse:
    """l_max=1 kernel with ``dipoles = 0`` must reproduce the l_max=0 kernel
    (which in turn matches the legacy monopole Ewald kernel, per
    :class:`TestMonopoleCollapse`)."""

    @pytest.mark.parametrize("seed", [0, 7, 42, 123])
    @pytest.mark.parametrize("num_atoms", [3, 10, 25])
    def test_zero_dipoles_matches_monopole(self, device, seed, num_atoms):
        rng = np.random.default_rng(seed)
        system = _random_system(num_atoms, box_size=5.0, rng=rng)
        alpha = 0.4
        # Run l_max=0.
        inputs = _to_warp(system, device, wp.float64)
        alpha_wp = wp.from_numpy(
            np.array([alpha], dtype=np.float64), dtype=wp.float64, device=device
        )
        monopole_energies = wp.zeros(num_atoms, dtype=wp.float64, device=device)
        multipole_real_space_monopole_csr_energy(
            inputs["positions"],
            inputs["charges"],
            inputs["cell"],
            inputs["idx_j"],
            inputs["neighbor_ptr"],
            inputs["unit_shifts"],
            _sigma(device, wp.float64, _COLLAPSE_SIGMA),
            alpha_wp,
            monopole_energies,
            wp_dtype=wp.float64,
            device=device,
        )
        # Run l_max=1 with zero dipoles.
        dipole_energies = _launch_dipole(
            positions=system["positions"],
            charges=system["charges"],
            dipoles=np.zeros((num_atoms, 3), dtype=np.float64),
            cell=system["cell"],
            idx_j=system["idx_j"],
            neighbor_ptr=system["neighbor_ptr"],
            unit_shifts=system["unit_shifts"],
            alpha_value=alpha,
            device=device,
            sigma_value=_COLLAPSE_SIGMA,
        )
        np.testing.assert_allclose(
            dipole_energies, monopole_energies.numpy(), rtol=0.0, atol=1e-15
        )

    def test_triclinic_cell_zero_dipoles(self, device):
        rng = np.random.default_rng(19)
        system = _triclinic_system(rng)
        alpha = 0.35
        num_atoms = system["num_atoms"]
        inputs = _to_warp(system, device, wp.float64)
        alpha_wp = wp.from_numpy(
            np.array([alpha], dtype=np.float64), dtype=wp.float64, device=device
        )
        monopole_energies = wp.zeros(num_atoms, dtype=wp.float64, device=device)
        multipole_real_space_monopole_csr_energy(
            inputs["positions"],
            inputs["charges"],
            inputs["cell"],
            inputs["idx_j"],
            inputs["neighbor_ptr"],
            inputs["unit_shifts"],
            _sigma(device, wp.float64, _COLLAPSE_SIGMA),
            alpha_wp,
            monopole_energies,
            wp_dtype=wp.float64,
            device=device,
        )
        dipole_energies = _launch_dipole(
            positions=system["positions"],
            charges=system["charges"],
            dipoles=np.zeros((num_atoms, 3), dtype=np.float64),
            cell=system["cell"],
            idx_j=system["idx_j"],
            neighbor_ptr=system["neighbor_ptr"],
            unit_shifts=system["unit_shifts"],
            alpha_value=alpha,
            device=device,
            sigma_value=_COLLAPSE_SIGMA,
        )
        np.testing.assert_allclose(
            dipole_energies, monopole_energies.numpy(), rtol=0.0, atol=1e-15
        )


class TestDipoleChargeDipole:
    """Analytical checks for charge-dipole pair interactions in the ``α → 0`` limit."""

    SMALL_ALPHA = 1e-3
    TOL_ABS = 1e-7
    # Small σ>0 avoids the removable 1/σ dipole-tensor singularity at σ=0.
    SMALL_SIGMA = 1e-3

    def test_dipole_at_j_aligned_with_r(self, device):
        """Charge +q at i, dipole (μ, 0, 0) at j, r_ij along +x.

        Physical formula: E = μ · ∇V_q(r_j), V_q(r) = q/|r - r_i|.
        Gradient at r_j = (d, 0, 0) is (-q/d^2, 0, 0). So E = -q·μ/d^2.
        Kernel returns 0.5 * this (half-list prefactor).
        """
        d = 5.0
        q, mu = 1.2, 0.7
        system = _two_atom_pair_system(
            distance=d, charges=(q, 0.0), dipole_j=(mu, 0.0, 0.0)
        )
        energies = _launch_dipole(
            **system,
            alpha_value=self.SMALL_ALPHA,
            device=device,
            sigma_value=self.SMALL_SIGMA,
        )
        expected = 0.5 * (-q * mu / d**2)
        assert float(energies.sum()) == pytest.approx(expected, abs=self.TOL_ABS)

    def test_charge_at_j_dipole_at_i(self, device):
        """Dipole (μ, 0, 0) at i, charge +q at j, r_ij along +x.

        Standard formula: E = +q·μ/d^2 (positive side of dipole faces charge).
        """
        d = 5.0
        q, mu = 1.2, 0.7
        system = _two_atom_pair_system(
            distance=d, charges=(0.0, q), dipole_i=(mu, 0.0, 0.0)
        )
        energies = _launch_dipole(
            **system,
            alpha_value=self.SMALL_ALPHA,
            device=device,
            sigma_value=self.SMALL_SIGMA,
        )
        expected = 0.5 * (q * mu / d**2)
        assert float(energies.sum()) == pytest.approx(expected, abs=self.TOL_ABS)

    def test_dipole_perpendicular_to_r(self, device):
        """Charge at i, dipole at j perpendicular to r_ij — energy should be 0."""
        d = 4.0
        q, mu = 0.9, 0.5
        system = _two_atom_pair_system(
            distance=d, charges=(q, 0.0), dipole_j=(0.0, mu, 0.0)
        )
        energies = _launch_dipole(
            **system,
            alpha_value=self.SMALL_ALPHA,
            device=device,
            sigma_value=self.SMALL_SIGMA,
        )
        assert float(energies.sum()) == pytest.approx(0.0, abs=self.TOL_ABS)


class TestDipoleDipoleDipole:
    """Analytical checks for dipole-dipole pair interactions.

    Standard formula: ``E_dd = (μ_i · μ_j)/r^3 - 3(μ_i · r̂)(μ_j · r̂)/r^3``.
    With the kernel's half-list 0.5 prefactor, expected = 0.5 * E_dd.
    """

    SMALL_ALPHA = 1e-3
    TOL_ABS = 1e-7
    # Small σ>0 avoids the removable 1/σ dipole-tensor singularity at σ=0.
    SMALL_SIGMA = 1e-3

    def test_parallel_along_r(self, device):
        """Both dipoles (μ, 0, 0), r along +x → attractive, -2μ²/d³."""
        d = 5.0
        mu = 1.3
        system = _two_atom_pair_system(
            distance=d, dipole_i=(mu, 0.0, 0.0), dipole_j=(mu, 0.0, 0.0)
        )
        energies = _launch_dipole(
            **system,
            alpha_value=self.SMALL_ALPHA,
            device=device,
            sigma_value=self.SMALL_SIGMA,
        )
        expected = 0.5 * (-2.0 * mu**2 / d**3)
        assert float(energies.sum()) == pytest.approx(expected, abs=self.TOL_ABS)

    def test_antiparallel_along_r(self, device):
        """Opposing dipoles (μ, 0, 0) and (-μ, 0, 0), r along +x → repulsive, +2μ²/d³."""
        d = 5.0
        mu = 1.3
        system = _two_atom_pair_system(
            distance=d, dipole_i=(mu, 0.0, 0.0), dipole_j=(-mu, 0.0, 0.0)
        )
        energies = _launch_dipole(
            **system,
            alpha_value=self.SMALL_ALPHA,
            device=device,
            sigma_value=self.SMALL_SIGMA,
        )
        expected = 0.5 * (2.0 * mu**2 / d**3)
        assert float(energies.sum()) == pytest.approx(expected, abs=self.TOL_ABS)

    def test_perpendicular_dipoles(self, device):
        """Dipoles along different transverse axes — zero energy."""
        d = 4.0
        mu = 1.1
        system = _two_atom_pair_system(
            distance=d, dipole_i=(0.0, mu, 0.0), dipole_j=(0.0, 0.0, mu)
        )
        energies = _launch_dipole(
            **system,
            alpha_value=self.SMALL_ALPHA,
            device=device,
            sigma_value=self.SMALL_SIGMA,
        )
        assert float(energies.sum()) == pytest.approx(0.0, abs=self.TOL_ABS)

    def test_parallel_perpendicular_to_r(self, device):
        """Both dipoles (0, μ, 0), r along +x.

        E_dd = μ² /r³ - 0 = +μ²/r³ (repulsive for two side-by-side parallel dipoles).
        """
        d = 5.0
        mu = 1.0
        system = _two_atom_pair_system(
            distance=d, dipole_i=(0.0, mu, 0.0), dipole_j=(0.0, mu, 0.0)
        )
        energies = _launch_dipole(
            **system,
            alpha_value=self.SMALL_ALPHA,
            device=device,
            sigma_value=self.SMALL_SIGMA,
        )
        expected = 0.5 * (mu**2 / d**3)
        assert float(energies.sum()) == pytest.approx(expected, abs=self.TOL_ABS)


class TestDipolePeriodicShifts:
    """Non-zero image shifts with dipoles."""

    def test_shift_preserves_collapse(self, device):
        """With shift (1,0,0) and a 5x5x5 cell → effective separation 6.
        Zero dipoles → matches l_max=0 kernel."""
        system = {
            "positions": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
            "charges": np.array([1.0, -1.0], dtype=np.float64),
            "dipoles": np.zeros((2, 3), dtype=np.float64),
            "cell": np.array(
                [[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]],
                dtype=np.float64,
            ),
            "idx_j": np.array([1], dtype=np.int32),
            "neighbor_ptr": np.array([0, 1, 1], dtype=np.int32),
            "unit_shifts": np.array([[1, 0, 0]], dtype=np.int32),
        }
        alpha = 0.4
        # Zero dipoles → the l=1 kernel reproduces the l=0 kernel at a common σ>0.
        dipole = _launch_dipole(
            **system, alpha_value=alpha, device=device, sigma_value=_COLLAPSE_SIGMA
        )

        ref_system = {k: v for k, v in system.items() if k != "dipoles"}
        ref_system["num_atoms"] = 2
        _, monopole = _launch_both(
            ref_system, device, alpha, sigma_value=_COLLAPSE_SIGMA
        )
        np.testing.assert_allclose(dipole, monopole, rtol=0.0, atol=1e-15)
