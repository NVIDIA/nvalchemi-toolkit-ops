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

"""JAX binding tests for dispersion PME (LJ-PME reciprocal-space + self-energy).

Covers the same surface as the torch binding tests: self-energy formula,
isolated-atom convergence, mesh convergence, batched vs single parity,
and float32 vs float64 parity.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="No JAX installed.")

# import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nvalchemiops.jax.interactions.dispersion import (  # noqa: E402
    lj_pme_real_space,
    pme_dispersion_energy_corrections,
    pme_dispersion_reciprocal_space,
)


@pytest.mark.gpu
class TestPMEDispersionEnergyCorrectionsJAX:
    """V_self = -β⁶ Σ C6_ii / 12 via the JAX binding."""

    def test_single_system_total(self, device):  # noqa: ARG002
        c6 = jnp.array([1.0, 2.5, 0.5], dtype=jnp.float64)
        beta = jnp.array([0.4], dtype=jnp.float64)
        out = pme_dispersion_energy_corrections(c6, beta)
        expected = -(0.4**6) * (1.0 + 2.5 + 0.5) / 12.0
        assert out.shape == (1,)
        assert float(out[0]) == pytest.approx(expected, rel=1e-12)

    def test_batched_per_system_sum(self, device):  # noqa: ARG002
        c6 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
        batch_idx = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
        beta = jnp.array([0.3, 0.5], dtype=jnp.float64)
        out = pme_dispersion_energy_corrections(c6, beta, batch_idx=batch_idx)
        exp_0 = -(0.3**6) * (1.0 + 2.0) / 12.0
        exp_1 = -(0.5**6) * (3.0 + 4.0 + 5.0) / 12.0
        assert out.shape == (2,)
        assert float(out[0]) == pytest.approx(exp_0, rel=1e-12)
        assert float(out[1]) == pytest.approx(exp_1, rel=1e-12)


@pytest.mark.gpu
class TestPMEDispersionReciprocalSpaceJAX:
    """Reciprocal-space LJ-PME energy via the JAX binding."""

    def test_isolated_atom_converges_to_self_energy(self, device):  # noqa: ARG002
        """Single atom: V_recip / V_self → 1 as β grows."""
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float64)
        c6 = jnp.array([1.0], dtype=jnp.float64)
        cell = jnp.eye(3, dtype=jnp.float64) * 20.0

        rel_residuals = []
        for beta_val in [0.3, 0.5, 0.8]:
            beta = jnp.array([beta_val], dtype=jnp.float64)
            E_recip = pme_dispersion_reciprocal_space(
                positions,
                c6,
                cell,
                beta,
                mesh_dimensions=(64, 64, 64),
                spline_order=4,
            )
            E_self = pme_dispersion_energy_corrections(c6, beta)
            rel_residuals.append(
                abs(float(E_recip[0]) - float(E_self[0])) / abs(float(E_self[0]))
            )

        assert rel_residuals[0] > rel_residuals[1] > rel_residuals[2]
        assert rel_residuals[2] < 5e-3

    def test_mesh_convergence(self, device):  # noqa: ARG002
        positions = jnp.array([[2.0, 3.0, 4.0], [7.0, 6.0, 5.0]], dtype=jnp.float64)
        c6 = jnp.array([1.0, 1.0], dtype=jnp.float64)
        cell = jnp.eye(3, dtype=jnp.float64) * 10.0
        beta = jnp.array([0.5], dtype=jnp.float64)

        E_64 = float(
            pme_dispersion_reciprocal_space(
                positions, c6, cell, beta, mesh_dimensions=(64, 64, 64)
            )[0]
        )
        E_96 = float(
            pme_dispersion_reciprocal_space(
                positions, c6, cell, beta, mesh_dimensions=(96, 96, 96)
            )[0]
        )
        E_128 = float(
            pme_dispersion_reciprocal_space(
                positions, c6, cell, beta, mesh_dimensions=(128, 128, 128)
            )[0]
        )
        err_64 = abs(E_64 - E_128)
        err_96 = abs(E_96 - E_128)
        assert err_96 < err_64
        assert err_96 < 1e-4 * abs(E_128) + 1e-12

    def test_batched_matches_stacked_singles(self, device):  # noqa: ARG002
        positions = jnp.array(
            [
                [0.5, 0.5, 0.5],
                [3.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=jnp.float64,
        )
        c6 = jnp.array([1.0, 1.0, 2.0, 0.5], dtype=jnp.float64)
        cell_single = jnp.eye(3, dtype=jnp.float64) * 10.0
        cell_batch = jnp.stack([cell_single, cell_single])
        beta_batch = jnp.array([0.4, 0.5], dtype=jnp.float64)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        mesh = (32, 32, 32)

        E_batch = pme_dispersion_reciprocal_space(
            positions,
            c6,
            cell_batch,
            beta_batch,
            mesh_dimensions=mesh,
            batch_idx=batch_idx,
        )
        assert E_batch.shape == (2,)

        E_0 = pme_dispersion_reciprocal_space(
            positions[:2],
            c6[:2],
            cell_single,
            jnp.array([float(beta_batch[0])], dtype=jnp.float64),
            mesh_dimensions=mesh,
        )
        E_1 = pme_dispersion_reciprocal_space(
            positions[2:],
            c6[2:],
            cell_single,
            jnp.array([float(beta_batch[1])], dtype=jnp.float64),
            mesh_dimensions=mesh,
        )
        assert float(E_batch[0]) == pytest.approx(float(E_0[0]), rel=1e-10)
        assert float(E_batch[1]) == pytest.approx(float(E_1[0]), rel=1e-10)

    def test_dtype_parity(self, device):  # noqa: ARG002
        positions_f64 = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float64)
        c6_f64 = jnp.array([1.5, 0.8], dtype=jnp.float64)
        cell_f64 = jnp.eye(3, dtype=jnp.float64) * 10.0
        beta_f64 = jnp.array([0.5], dtype=jnp.float64)

        E64 = pme_dispersion_reciprocal_space(
            positions_f64, c6_f64, cell_f64, beta_f64, mesh_dimensions=(32, 32, 32)
        )
        E32 = pme_dispersion_reciprocal_space(
            positions_f64.astype(jnp.float32),
            c6_f64.astype(jnp.float32),
            cell_f64.astype(jnp.float32),
            beta_f64.astype(jnp.float32),
            mesh_dimensions=(32, 32, 32),
        )
        assert E64.dtype == jnp.float64
        assert E32.dtype == jnp.float32
        assert float(E32[0]) == pytest.approx(float(E64[0]), rel=1e-3)

    def test_empty_system_returns_zero(self, device):  # noqa: ARG002
        positions = jnp.zeros((0, 3), dtype=jnp.float64)
        c6 = jnp.zeros((0,), dtype=jnp.float64)
        cell = jnp.eye(3, dtype=jnp.float64) * 10.0
        beta = jnp.array([0.4], dtype=jnp.float64)
        out = pme_dispersion_reciprocal_space(
            positions, c6, cell, beta, mesh_dimensions=(16, 16, 16)
        )
        assert out.shape == (1,)
        assert float(out[0]) == 0.0


@pytest.mark.gpu
class TestLJPMERealSpaceJAX:
    """JAX binding tests for real-space LJ-PME (PR2)."""

    @staticmethod
    def _two_atom_pair(r=3.5):
        positions = jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=jnp.float64)
        c6 = jnp.array([60.0, 60.0], dtype=jnp.float64)
        c12 = jnp.array([5e4, 5e4], dtype=jnp.float64)
        cell = jnp.eye(3, dtype=jnp.float64) * 100.0
        nbr_mat = jnp.array([[1], [2]], dtype=jnp.int32)
        nbr_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)
        num_nbrs = jnp.array([1, 0], dtype=jnp.int32)
        return positions, c6, c12, cell, nbr_mat, nbr_shifts, num_nbrs

    def test_pair_energy_matches_analytical(self, device):  # noqa: ARG002
        import math

        positions, c6, c12, cell, nbr_mat, nbr_shifts, num_nbrs = self._two_atom_pair()
        beta = jnp.array([0.35], dtype=jnp.float64)
        e = lj_pme_real_space(
            positions,
            c6,
            c12,
            cell,
            nbr_mat,
            nbr_shifts,
            beta=beta,
            cutoff=10.0,
            num_neighbors=num_nbrs,
            mask_value=2,
        )
        r = 3.5
        x_sq = (0.35 * r) ** 2
        g = math.exp(-x_sq) * (1 + x_sq + 0.5 * x_sq * x_sq)
        expected = 5e4 / r**12 - 60.0 * g / r**6
        assert float(e.sum()) == pytest.approx(expected, rel=1e-12)

    def test_forces_finite_difference(self, device):  # noqa: ARG002
        positions, c6, c12, cell, nbr_mat, nbr_shifts, num_nbrs = self._two_atom_pair()
        beta = jnp.array([0.35], dtype=jnp.float64)

        _, f_kernel = lj_pme_real_space(
            positions,
            c6,
            c12,
            cell,
            nbr_mat,
            nbr_shifts,
            beta=beta,
            cutoff=10.0,
            num_neighbors=num_nbrs,
            mask_value=2,
            compute_forces=True,
        )

        # FD over a 1D move of atom 1 along x.
        def energy_at(r_val):
            pos = jnp.array([[0.0, 0.0, 0.0], [r_val, 0.0, 0.0]], dtype=jnp.float64)
            e = lj_pme_real_space(
                pos,
                c6,
                c12,
                cell,
                nbr_mat,
                nbr_shifts,
                beta=beta,
                cutoff=10.0,
                num_neighbors=num_nbrs,
                mask_value=2,
            )
            return float(e.sum())

        h = 1e-5
        dE_dr = (energy_at(3.5 + h) - energy_at(3.5 - h)) / (2 * h)
        assert float(f_kernel[1, 0]) == pytest.approx(-dE_dr, rel=1e-5)
        assert float(f_kernel[0, 0]) == pytest.approx(dE_dr, rel=1e-5)

    def test_momentum_conservation(self, device):  # noqa: ARG002
        positions, c6, c12, cell, nbr_mat, nbr_shifts, num_nbrs = self._two_atom_pair()
        beta = jnp.array([0.35], dtype=jnp.float64)
        _, f = lj_pme_real_space(
            positions,
            c6,
            c12,
            cell,
            nbr_mat,
            nbr_shifts,
            beta=beta,
            cutoff=10.0,
            num_neighbors=num_nbrs,
            mask_value=2,
            compute_forces=True,
        )
        import numpy as np

        np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-12)

    def test_zero_beyond_cutoff(self, device):  # noqa: ARG002
        positions, c6, c12, cell, nbr_mat, nbr_shifts, num_nbrs = self._two_atom_pair(
            r=6.0
        )
        beta = jnp.array([0.35], dtype=jnp.float64)
        e = lj_pme_real_space(
            positions,
            c6,
            c12,
            cell,
            nbr_mat,
            nbr_shifts,
            beta=beta,
            cutoff=5.0,
            num_neighbors=num_nbrs,
            mask_value=2,
        )
        assert float(e.sum()) == 0.0

    def test_virial_finite_difference(self, device):  # noqa: ARG002
        positions, c6, c12, cell, nbr_mat, nbr_shifts, num_nbrs = self._two_atom_pair()
        beta = jnp.array([0.35], dtype=jnp.float64)

        def energy_at_strain(eps):
            I_plus_eps = jnp.eye(3, dtype=jnp.float64) + eps
            pos_p = positions @ I_plus_eps.T
            cell_p = cell @ I_plus_eps.T
            e = lj_pme_real_space(
                pos_p,
                c6,
                c12,
                cell_p,
                nbr_mat,
                nbr_shifts,
                beta=beta,
                cutoff=10.0,
                num_neighbors=num_nbrs,
                mask_value=2,
            )
            return float(e.sum())

        import numpy as np

        h = 1e-5
        fd = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                eps_plus = jnp.zeros((3, 3), dtype=jnp.float64).at[a, b].set(h)
                eps_minus = jnp.zeros((3, 3), dtype=jnp.float64).at[a, b].set(-h)
                fd[a, b] = (
                    energy_at_strain(eps_plus) - energy_at_strain(eps_minus)
                ) / (2 * h)

        _, _, virial = lj_pme_real_space(
            positions,
            c6,
            c12,
            cell,
            nbr_mat,
            nbr_shifts,
            beta=beta,
            cutoff=10.0,
            num_neighbors=num_nbrs,
            mask_value=2,
            compute_forces=True,
            compute_virial=True,
        )
        np.testing.assert_allclose(np.asarray(virial), -fd, rtol=1e-5, atol=1e-8)
