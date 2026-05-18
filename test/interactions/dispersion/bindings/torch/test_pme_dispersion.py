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

"""PyTorch binding tests for dispersion PME (LJ-PME reciprocal-space + self-energy).

Covers:
- Self-energy formula correctness.
- Isolated-atom test: V_recip - V_self → 0 in the continuum/large-β limit.
- Mesh convergence of V_recip.
- Batched vs stacked-single parity.
- Float32 vs float64 parity within tolerance.
"""

from __future__ import annotations

import pytest
import torch
import warp as wp

from nvalchemiops.torch.interactions.dispersion import (
    lj_pme_real_space,
    pme_dispersion_energy_corrections,
    pme_dispersion_reciprocal_space,
)


@pytest.fixture(scope="module")
def device():
    if not wp.is_cuda_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.mark.gpu
class TestPMEDispersionEnergyCorrections:
    """V_self = -β⁶ Σ C6_ii / 12 via the torch binding."""

    def test_single_system_total(self, device):
        c6 = torch.tensor([1.0, 2.5, 0.5], dtype=torch.float64, device=device)
        beta = torch.tensor([0.4], dtype=torch.float64, device=device)
        out = pme_dispersion_energy_corrections(c6, beta)
        expected = -(0.4**6) * (1.0 + 2.5 + 0.5) / 12.0
        assert out.shape == (1,)
        assert out.item() == pytest.approx(expected, rel=1e-12)

    def test_batched_per_system_sum(self, device):
        c6 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32, device=device)
        beta = torch.tensor([0.3, 0.5], dtype=torch.float64, device=device)
        out = pme_dispersion_energy_corrections(c6, beta, batch_idx=batch_idx)
        exp_0 = -(0.3**6) * (1.0 + 2.0) / 12.0
        exp_1 = -(0.5**6) * (3.0 + 4.0 + 5.0) / 12.0
        assert out.shape == (2,)
        assert out[0].item() == pytest.approx(exp_0, rel=1e-12)
        assert out[1].item() == pytest.approx(exp_1, rel=1e-12)

    def test_dtype_parity(self, device):
        c6_f32 = torch.tensor([1.0, 2.5], dtype=torch.float32, device=device)
        c6_f64 = c6_f32.to(torch.float64)
        beta_f32 = torch.tensor([0.4], dtype=torch.float32, device=device)
        beta_f64 = beta_f32.to(torch.float64)
        out_f32 = pme_dispersion_energy_corrections(c6_f32, beta_f32)
        out_f64 = pme_dispersion_energy_corrections(c6_f64, beta_f64)
        assert out_f32.dtype == torch.float32
        assert out_f64.dtype == torch.float64
        assert out_f32.item() == pytest.approx(out_f64.item(), rel=1e-5)


@pytest.mark.gpu
class TestPMEDispersionReciprocalSpace:
    """Reciprocal-space LJ-PME energy via the torch binding."""

    def test_isolated_atom_converges_to_self_energy(self, device):
        """Single atom: V_recip / V_self → 1 as β grows (continuum limit).

        For large β the dispersion long-range complement is sharply peaked
        near r=0 so periodic-image contributions become negligible, and the
        FFT-evaluated lattice sum should converge to the i=j limit
        captured by V_self. We track *relative* error since |V_self| grows
        as β^6 so absolute residuals are not directly comparable.
        """
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64, device=device)
        c6 = torch.tensor([1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 20.0

        rel_residuals = []
        for beta_val in [0.3, 0.5, 0.8]:
            beta = torch.tensor([beta_val], dtype=torch.float64, device=device)
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
                abs(E_recip.item() - E_self.item()) / abs(E_self.item())
            )

        # Relative residual decreases monotonically as β increases.
        assert rel_residuals[0] > rel_residuals[1] > rel_residuals[2]
        # At β=0.8 the relative residual is below 0.5%.
        assert rel_residuals[2] < 5e-3

    def test_mesh_convergence(self, device):
        """V_recip converges as the mesh becomes finer (fixed β, fixed atoms)."""
        positions = torch.tensor(
            [[2.0, 3.0, 4.0], [7.0, 6.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        c6 = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0
        beta = torch.tensor([0.5], dtype=torch.float64, device=device)

        E_64 = pme_dispersion_reciprocal_space(
            positions, c6, cell, beta, mesh_dimensions=(64, 64, 64)
        ).item()
        E_96 = pme_dispersion_reciprocal_space(
            positions, c6, cell, beta, mesh_dimensions=(96, 96, 96)
        ).item()
        E_128 = pme_dispersion_reciprocal_space(
            positions, c6, cell, beta, mesh_dimensions=(128, 128, 128)
        ).item()

        # Errors decrease as mesh refines (relative to the finest result).
        err_64 = abs(E_64 - E_128)
        err_96 = abs(E_96 - E_128)
        assert err_96 < err_64
        # Between 96 and 128 we're already well-converged.
        assert err_96 < 1e-4 * abs(E_128) + 1e-12

    def test_batched_matches_stacked_singles(self, device):
        """Batched evaluation matches stacked single-system evaluations."""
        # Two independent two-atom systems in identical 10 Å cubic boxes.
        positions = torch.tensor(
            [
                [0.5, 0.5, 0.5],
                [3.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        c6 = torch.tensor([1.0, 1.0, 2.0, 0.5], dtype=torch.float64, device=device)
        cell_single = torch.eye(3, dtype=torch.float64, device=device) * 10.0
        cell_batch = torch.stack([cell_single, cell_single])
        beta_batch = torch.tensor([0.4, 0.5], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
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
            torch.tensor([beta_batch[0].item()], dtype=torch.float64, device=device),
            mesh_dimensions=mesh,
        )
        E_1 = pme_dispersion_reciprocal_space(
            positions[2:],
            c6[2:],
            cell_single,
            torch.tensor([beta_batch[1].item()], dtype=torch.float64, device=device),
            mesh_dimensions=mesh,
        )
        assert E_batch[0].item() == pytest.approx(E_0.item(), rel=1e-10)
        assert E_batch[1].item() == pytest.approx(E_1.item(), rel=1e-10)

    def test_dtype_parity(self, device):
        """float32 and float64 results agree within float32 tolerance."""
        positions_f64 = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64, device=device
        )
        c6_f64 = torch.tensor([1.5, 0.8], dtype=torch.float64, device=device)
        cell_f64 = torch.eye(3, dtype=torch.float64, device=device) * 10.0
        beta_f64 = torch.tensor([0.5], dtype=torch.float64, device=device)

        E64 = pme_dispersion_reciprocal_space(
            positions_f64, c6_f64, cell_f64, beta_f64, mesh_dimensions=(32, 32, 32)
        )
        E32 = pme_dispersion_reciprocal_space(
            positions_f64.to(torch.float32),
            c6_f64.to(torch.float32),
            cell_f64.to(torch.float32),
            beta_f64.to(torch.float32),
            mesh_dimensions=(32, 32, 32),
        )
        assert E64.dtype == torch.float64
        assert E32.dtype == torch.float32
        assert E32.item() == pytest.approx(E64.item(), rel=2e-4)

    def test_empty_system_returns_zero(self, device):
        positions = torch.zeros((0, 3), dtype=torch.float64, device=device)
        c6 = torch.zeros((0,), dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0
        beta = torch.tensor([0.4], dtype=torch.float64, device=device)
        out = pme_dispersion_reciprocal_space(
            positions, c6, cell, beta, mesh_dimensions=(16, 16, 16)
        )
        assert out.shape == (1,)
        assert out.item() == 0.0

    def test_translation_invariance(self, device):
        """Reciprocal energy is invariant under uniform shifts (modulo PBC).

        Shifting all atoms by the same vector leaves |ρ(k)| unchanged in
        principle; on the B-spline mesh this holds up to discretization
        error, which decreases with mesh resolution and spline order.
        """
        positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=torch.float64,
            device=device,
        )
        c6 = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0
        beta = torch.tensor([0.4], dtype=torch.float64, device=device)
        mesh = (64, 64, 64)

        E0 = pme_dispersion_reciprocal_space(
            positions, c6, cell, beta, mesh_dimensions=mesh
        )
        shifted = positions + torch.tensor(
            [0.7, 1.3, 2.1], dtype=torch.float64, device=device
        )
        E1 = pme_dispersion_reciprocal_space(
            shifted, c6, cell, beta, mesh_dimensions=mesh
        )
        # Discretization-limited; expect agreement at the 1e-5 level.
        assert E1.item() == pytest.approx(E0.item(), rel=1e-5)


@pytest.mark.gpu
class TestLJPMERealSpace:
    """Real-space LJ-PME binding tests (PR2)."""

    @staticmethod
    def _two_atom_pair(device, r=3.5):
        """Two atoms separated by ``r`` along x with a half neighbor list."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        c6 = torch.tensor([60.0, 60.0], dtype=torch.float64, device=device)
        c12 = torch.tensor([5e4, 5e4], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 100.0
        nbr_mat = torch.tensor([[1], [2]], dtype=torch.int32, device=device)
        nbr_shifts = torch.zeros((2, 1, 3), dtype=torch.int32, device=device)
        num_nbrs = torch.tensor([1, 0], dtype=torch.int32, device=device)
        return positions, c6, c12, cell, nbr_mat, nbr_shifts, num_nbrs

    def test_pair_energy_matches_analytical(self, device):
        import math

        positions, c6, c12, cell, nbr_mat, nbr_shifts, num_nbrs = self._two_atom_pair(
            device
        )
        beta = torch.tensor([0.35], dtype=torch.float64, device=device)
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
        assert e.sum().item() == pytest.approx(expected, rel=1e-12)

    def test_forces_match_autograd(self, device):
        positions, c6, c12, cell, nbr_mat, nbr_shifts, num_nbrs = self._two_atom_pair(
            device
        )
        beta = torch.tensor([0.35], dtype=torch.float64, device=device)

        # Forces from kernel
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

        # Forces from autograd: F = -dE/d positions
        positions_g = positions.clone().requires_grad_(True)
        e = lj_pme_real_space(
            positions_g,
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
        e.sum().backward()
        f_autograd = -positions_g.grad

        torch.testing.assert_close(f_kernel, f_autograd, rtol=1e-10, atol=1e-12)

    def test_momentum_conservation(self, device):
        positions, c6, c12, cell, nbr_mat, nbr_shifts, num_nbrs = self._two_atom_pair(
            device
        )
        beta = torch.tensor([0.35], dtype=torch.float64, device=device)
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
        torch.testing.assert_close(
            f.sum(0), torch.zeros(3, dtype=f.dtype, device=f.device), atol=1e-12, rtol=0
        )

    def test_virial_finite_difference(self, device):
        """Compare kernel virial against FD over uniform strain (float64).

        Apply x'_i = (I + eps) x_i and cell' = (I + eps) cell. The
        derivative dE/d(eps_ab) at eps=0 equals the negative virial:
        ``W_ab = -dE/d(eps_ab)``. The Warp kernel returns
        :math:`W = \\sum_{i<j} r_{ij} F_{ij}`, which equals
        :math:`+dE/d(eps)` (sign convention W = -dE/deps means the
        kernel's positive accumulator measures the "stress" form).
        Here we just check that the kernel result agrees with FD up to
        the expected sign.
        """
        positions, c6, c12, cell, nbr_mat, nbr_shifts, num_nbrs = self._two_atom_pair(
            device
        )
        beta = torch.tensor([0.35], dtype=torch.float64, device=device)

        def energy_at_strain(eps):
            I_plus_eps = torch.eye(3, dtype=torch.float64, device=device) + eps
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

        h = 1e-5
        fd = torch.zeros(3, 3, dtype=torch.float64, device=device)
        for a in range(3):
            for b in range(3):
                eps_plus = torch.zeros(3, 3, dtype=torch.float64, device=device)
                eps_plus[a, b] = h
                eps_minus = torch.zeros(3, 3, dtype=torch.float64, device=device)
                eps_minus[a, b] = -h
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
        # Kernel uses W_ab = sum r_ij,a F_ij,b ; dE/deps = -W (so eq with -fd).
        torch.testing.assert_close(virial, -fd, rtol=1e-5, atol=1e-8)

    def test_zero_beyond_cutoff(self, device):
        positions, c6, c12, cell, nbr_mat, nbr_shifts, num_nbrs = self._two_atom_pair(
            device, r=6.0
        )
        beta = torch.tensor([0.35], dtype=torch.float64, device=device)
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
        assert e.sum().item() == 0.0

    def test_dtype_parity(self, device):
        positions_f64, c6_f64, c12_f64, cell_f64, nbr_mat, nbr_shifts, num_nbrs = (
            self._two_atom_pair(device)
        )
        beta_f64 = torch.tensor([0.35], dtype=torch.float64, device=device)
        e64, f64 = lj_pme_real_space(
            positions_f64,
            c6_f64,
            c12_f64,
            cell_f64,
            nbr_mat,
            nbr_shifts,
            beta=beta_f64,
            cutoff=10.0,
            num_neighbors=num_nbrs,
            mask_value=2,
            compute_forces=True,
        )
        e32, f32 = lj_pme_real_space(
            positions_f64.to(torch.float32),
            c6_f64.to(torch.float32),
            c12_f64.to(torch.float32),
            cell_f64.to(torch.float32),
            nbr_mat,
            nbr_shifts,
            beta=beta_f64.to(torch.float32),
            cutoff=10.0,
            num_neighbors=num_nbrs,
            mask_value=2,
            compute_forces=True,
        )
        assert e64.dtype == torch.float64
        assert e32.dtype == torch.float32
        assert e32.sum().item() == pytest.approx(e64.sum().item(), rel=1e-5)
        torch.testing.assert_close(f32.to(torch.float64), f64, rtol=1e-4, atol=1e-6)
