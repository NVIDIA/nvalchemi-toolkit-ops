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

"""PyTorch integration tests for the top-level ``lj_pme`` orchestrator (PR3).

Covers parameter estimation, end-to-end energy and forces, translation
invariance, batch parity, dtype parity, and force consistency with
autograd.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.torch.interactions.dispersion import (
    estimate_pme_dispersion_parameters,
    lj_pme,
)
from nvalchemiops.torch.interactions.dispersion.parameters import (
    solve_dispersion_beta,
)
from nvalchemiops.torch.neighbors import neighbor_list as nbr_list_fn


@pytest.fixture(scope="module")
def device():
    if not wp.is_cuda_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


"""
def _create_argon_fcc(n_cells: int, lattice_constant: float, device):
    fcc = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )
    positions = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for f in fcc:
                    positions.append((f + np.array([i, j, k])) * lattice_constant)
    positions = torch.tensor(np.array(positions), dtype=torch.float64, device=device)
    c6 = torch.full((positions.shape[0],), 60.0, dtype=torch.float64, device=device)
    c12 = torch.full((positions.shape[0],), 5.5e4, dtype=torch.float64, device=device)
    cell = torch.eye(3, dtype=torch.float64, device=device) * lattice_constant * n_cells
    return positions, c6, c12, cell
"""


def _create_argon_fcc(n_cells: int, lattice_constant: float, device):
    """Argon FCC supercell as torch tensors."""
    fcc = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )

    # 1. Create a 3D grid of cell indices (i, j, k) with shape (n_cells, n_cells, n_cells, 3)
    grid = np.stack(
        np.meshgrid(range(n_cells), range(n_cells), range(n_cells), indexing="ij"),
        axis=-1,
    )

    # 2. Reshape to exploit NumPy broadcasting:
    #    cells shape:   (N_cells_total, 1, 3) where N_cells_total = n_cells^3
    #    fcc shape:     (1, 4, 3)
    cells = grid.reshape(-1, 1, 3)
    fcc_basis = fcc.reshape(1, -1, 3)

    # 3. Add them together (broadcasting creates shape (N_cells_total, 4, 3))
    #    Then flatten the dimensions into (-1, 3) and scale by lattice_constant
    positions_np = (cells + fcc_basis).reshape(-1, 3) * lattice_constant

    # 4. Convert directly to PyTorch tensors on the target device
    positions = torch.tensor(positions_np, dtype=torch.float64, device=device)

    c6 = torch.full((positions.shape[0],), 60.0, dtype=torch.float64, device=device)
    c12 = torch.full((positions.shape[0],), 5.5e4, dtype=torch.float64, device=device)
    cell = torch.eye(3, dtype=torch.float64, device=device) * lattice_constant * n_cells

    return positions, c6, c12, cell


def _two_atom_system(r: float, device, box: float = 20.0):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64, device=device
    )
    c6 = torch.tensor([60.0, 60.0], dtype=torch.float64, device=device)
    c12 = torch.tensor([5.5e4, 5.5e4], dtype=torch.float64, device=device)
    cell = torch.eye(3, dtype=torch.float64, device=device) * box
    return positions, c6, c12, cell


@pytest.mark.gpu
class TestParameterEstimator:
    """Tests for ``estimate_pme_dispersion_parameters``."""

    def test_solve_dispersion_beta_satisfies_target(self):
        for cutoff in [7.0, 9.0, 12.0]:
            for tol in [1e-3, 1e-5]:
                b = solve_dispersion_beta(cutoff, tol)
                x = b * cutoff
                g = math.exp(-x * x) * (1.0 + x * x + 0.5 * x * x * x * x)
                # g(β·r_c) lands within ~10% of the target threshold
                assert g == pytest.approx(tol, rel=0.1)

    def test_estimator_returns_consistent_parameters(self, device):
        cell = torch.eye(3, dtype=torch.float64, device=device) * 20.0
        params = estimate_pme_dispersion_parameters(cell, cutoff=9.0, accuracy=1e-3)
        assert params.cutoff == 9.0
        assert params.beta.shape == (1,)
        assert params.mesh_dimensions[0] >= 1
        assert params.mesh_spacing.shape == (1, 3)
        assert params.accuracy == 1e-3

    def test_estimator_scales_mesh_with_cell(self, device):
        cell_small = torch.eye(3, dtype=torch.float64, device=device) * 10.0
        cell_large = torch.eye(3, dtype=torch.float64, device=device) * 30.0
        p_small = estimate_pme_dispersion_parameters(cell_small, cutoff=9.0)
        p_large = estimate_pme_dispersion_parameters(cell_large, cutoff=9.0)
        assert p_large.mesh_dimensions[0] > p_small.mesh_dimensions[0]


@pytest.mark.gpu
class TestLJPMEOrchestrator:
    """Top-level ``lj_pme()`` API tests."""

    def test_auto_parameter_energy(self, device):
        """Default-parameter call returns a finite, negative energy."""
        positions, c6, c12, cell = _create_argon_fcc(3, 5.26, device)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        cutoff = 9.0
        nm, nn, ns = nbr_list_fn(
            positions,
            cutoff,
            cell=cell.unsqueeze(0),
            pbc=pbc,
            return_neighbor_list=False,
        )
        E = lj_pme(positions, c6, c12, cell, nm, ns, num_neighbors=nn)
        assert E.shape == (1,)
        assert E.item() < 0.0
        assert math.isfinite(E.item())

    def test_two_atom_matches_pair_formula(self, device):
        """Two-atom result matches the bare V_pair = C12/r^12 - C6/r^6."""
        r = 3.5
        positions, c6, c12, cell = _two_atom_system(r, device)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        cutoff = 12.0
        nm, nn, ns = nbr_list_fn(
            positions,
            cutoff,
            cell=cell.unsqueeze(0),
            pbc=pbc,
            return_neighbor_list=False,
        )
        # Use a tight matched-tail accuracy and a fine mesh to converge.
        beta = solve_dispersion_beta(cutoff, 1e-6)
        E = lj_pme(
            positions,
            c6,
            c12,
            cell,
            nm,
            ns,
            num_neighbors=nn,
            beta=beta,
            cutoff=cutoff,
            mesh_dimensions=(128, 128, 128),
        )
        ref = 5.5e4 / r**12 - 60.0 / r**6
        # Looser tolerance because matched-tail criterion is approximate;
        # the answer should be within ~10% of the reference.
        assert E.item() == pytest.approx(ref, rel=0.1)

    def test_forces_match_real_space_plus_reciprocal(self, device):
        """End-to-end forces equal the sum of the real- and reciprocal-space pieces."""
        from nvalchemiops.torch.interactions.dispersion import (
            lj_pme_real_space,
            pme_dispersion_reciprocal_space,
        )

        positions, c6, c12, cell = _create_argon_fcc(2, 5.26, device)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        cutoff = 8.0
        beta = solve_dispersion_beta(cutoff, 1e-4)
        nm, nn, ns = nbr_list_fn(
            positions,
            cutoff,
            cell=cell.unsqueeze(0),
            pbc=pbc,
            return_neighbor_list=False,
        )

        _, F_total = lj_pme(
            positions,
            c6,
            c12,
            cell,
            nm,
            ns,
            num_neighbors=nn,
            beta=beta,
            cutoff=cutoff,
            mesh_dimensions=(32, 32, 32),
            compute_forces=True,
        )

        beta_t = torch.tensor([beta], dtype=torch.float64, device=device)
        _, F_real = lj_pme_real_space(
            positions,
            c6,
            c12,
            cell,
            nm,
            ns,
            beta=beta_t,
            cutoff=cutoff,
            num_neighbors=nn,
            compute_forces=True,
            half_neighbor_list=False,
        )
        _, F_recip = pme_dispersion_reciprocal_space(
            positions,
            c6,
            cell,
            beta_t,
            mesh_dimensions=(32, 32, 32),
            compute_forces=True,
        )
        torch.testing.assert_close(F_total, F_real + F_recip, rtol=1e-10, atol=1e-12)

    def test_momentum_conservation(self, device):
        """Σ F = 0 for any system (Newton's 3rd law)."""
        positions, c6, c12, cell = _create_argon_fcc(2, 5.26, device)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        nm, nn, ns = nbr_list_fn(
            positions, 9.0, cell=cell.unsqueeze(0), pbc=pbc, return_neighbor_list=False
        )
        _, F = lj_pme(
            positions, c6, c12, cell, nm, ns, num_neighbors=nn, compute_forces=True
        )
        torch.testing.assert_close(
            F.sum(dim=0),
            torch.zeros(3, dtype=F.dtype, device=F.device),
            atol=1e-9,
            rtol=0,
        )

    def test_translation_invariance(self, device):
        """Total energy is invariant under uniform translation of all atoms (modulo PBC)."""
        positions, c6, c12, cell = _create_argon_fcc(2, 5.26, device)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        nm, nn, ns = nbr_list_fn(
            positions, 9.0, cell=cell.unsqueeze(0), pbc=pbc, return_neighbor_list=False
        )
        E0 = lj_pme(
            positions,
            c6,
            c12,
            cell,
            nm,
            ns,
            num_neighbors=nn,
            mesh_dimensions=(48, 48, 48),
        )
        shift = torch.tensor([0.7, 1.3, 2.1], dtype=torch.float64, device=device)
        nm2, nn2, ns2 = nbr_list_fn(
            positions + shift,
            9.0,
            cell=cell.unsqueeze(0),
            pbc=pbc,
            return_neighbor_list=False,
        )
        E1 = lj_pme(
            positions + shift,
            c6,
            c12,
            cell,
            nm2,
            ns2,
            num_neighbors=nn2,
            mesh_dimensions=(48, 48, 48),
        )
        # Discretization-limited; agreement at the 1e-5 level is expected.
        assert E1.item() == pytest.approx(E0.item(), rel=1e-5)

    def test_dtype_parity(self, device):
        """float32 and float64 agree within float32 tolerance."""
        positions, c6, c12, cell = _create_argon_fcc(2, 5.26, device)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        nm, nn, ns = nbr_list_fn(
            positions, 9.0, cell=cell.unsqueeze(0), pbc=pbc, return_neighbor_list=False
        )
        E64 = lj_pme(positions, c6, c12, cell, nm, ns, num_neighbors=nn)
        E32 = lj_pme(
            positions.to(torch.float32),
            c6.to(torch.float32),
            c12.to(torch.float32),
            cell.to(torch.float32),
            nm,
            ns,
            num_neighbors=nn,
        )
        assert E64.dtype == torch.float64
        assert E32.dtype == torch.float32
        assert E32.item() == pytest.approx(E64.item(), rel=1e-4)

    @pytest.mark.xfail(
        reason="lj_pme_real_space currently single-system only; "
        "batched cells require a per-atom-cell kernel (future PR)."
    )
    def test_batched_matches_stacked_singles(self, device):
        """Batched lj_pme matches concatenated single-system calls."""
        pos_a, c6_a, c12_a, cell_a = _create_argon_fcc(2, 5.20, device)
        pos_b, c6_b, c12_b, cell_b = _create_argon_fcc(2, 5.45, device)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        cutoff = 9.0
        mesh = (32, 32, 32)

        # Batched
        pos_batch = torch.cat([pos_a, pos_b], dim=0)
        c6_batch = torch.cat([c6_a, c6_b], dim=0)
        c12_batch = torch.cat([c12_a, c12_b], dim=0)
        cell_batch = torch.stack([cell_a, cell_b])
        batch_idx = torch.cat(
            [
                torch.zeros(pos_a.shape[0], dtype=torch.int32, device=device),
                torch.ones(pos_b.shape[0], dtype=torch.int32, device=device),
            ]
        )
        # Build a single neighbor matrix from per-system batched cell-list.
        from nvalchemiops.torch.neighbors import batch_cell_list

        nm, nn, ns = batch_cell_list(
            pos_batch, cutoff, cell_batch, pbc, batch_idx=batch_idx, max_neighbors=128
        )
        beta = solve_dispersion_beta(cutoff, 1e-3)
        E_batch = lj_pme(
            pos_batch,
            c6_batch,
            c12_batch,
            cell_batch,
            nm,
            ns,
            num_neighbors=nn,
            batch_idx=batch_idx,
            beta=beta,
            cutoff=cutoff,
            mesh_dimensions=mesh,
        )
        assert E_batch.shape == (2,)

        # Single systems
        nm_a, nn_a, ns_a = nbr_list_fn(
            pos_a, cutoff, cell=cell_a.unsqueeze(0), pbc=pbc, return_neighbor_list=False
        )
        nm_b, nn_b, ns_b = nbr_list_fn(
            pos_b, cutoff, cell=cell_b.unsqueeze(0), pbc=pbc, return_neighbor_list=False
        )
        E_a = lj_pme(
            pos_a,
            c6_a,
            c12_a,
            cell_a,
            nm_a,
            ns_a,
            num_neighbors=nn_a,
            beta=beta,
            cutoff=cutoff,
            mesh_dimensions=mesh,
        )
        E_b = lj_pme(
            pos_b,
            c6_b,
            c12_b,
            cell_b,
            nm_b,
            ns_b,
            num_neighbors=nn_b,
            beta=beta,
            cutoff=cutoff,
            mesh_dimensions=mesh,
        )
        assert E_batch[0].item() == pytest.approx(E_a.item(), rel=1e-9)
        assert E_batch[1].item() == pytest.approx(E_b.item(), rel=1e-9)

    def test_explicit_beta_overrides_estimator(self, device):
        """Explicit beta is used (estimator is bypassed)."""
        positions, c6, c12, cell = _create_argon_fcc(2, 5.26, device)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        nm, nn, ns = nbr_list_fn(
            positions, 9.0, cell=cell.unsqueeze(0), pbc=pbc, return_neighbor_list=False
        )
        E_auto = lj_pme(positions, c6, c12, cell, nm, ns, num_neighbors=nn)
        E_explicit = lj_pme(
            positions,
            c6,
            c12,
            cell,
            nm,
            ns,
            num_neighbors=nn,
            beta=0.4,
            cutoff=9.0,
            mesh_dimensions=(32, 32, 32),
        )
        # Different parameters → different result.
        assert E_auto.item() != E_explicit.item()
