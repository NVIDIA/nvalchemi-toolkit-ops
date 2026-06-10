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

"""Validation / error-contract coverage for the multipole electrostatics torch
wrappers.

These exercise the torch-level guard branches (bad shapes, non-positive
parameters, cache/batch mismatches) that the numerical parity + FD suites never
hit because they always pass valid inputs. Kept separate so the contract is
explicit and the wrappers' validation lines are covered.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    pack_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_electrostatics import (
    multipole_electrostatic_energy,
    multipole_reciprocal_space_energy,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
    multipole_ewald_summation,
    multipole_real_space_energy,
)


def _empty_csr(n_atoms: int):
    """A valid-but-empty CSR neighbor list (no pairs) for guard-path tests."""
    idx_j = torch.zeros(0, dtype=torch.int32)
    neighbor_ptr = torch.zeros(n_atoms + 1, dtype=torch.int32)
    unit_shifts = torch.zeros((0, 3), dtype=torch.int32)
    return idx_j, neighbor_ptr, unit_shifts


def _cell(scale: float = 6.0) -> torch.Tensor:
    return torch.eye(3, dtype=torch.float64) * scale


def _positions(n: int = 2) -> torch.Tensor:
    return torch.zeros((n, 3), dtype=torch.float64)


class TestEwaldSummationValidation:
    """Guard branches in the ``multipole_ewald_summation`` composite."""

    def test_sigma_nonpositive(self):
        pos = _positions()
        mm = pack_multipole_moments(torch.zeros(2, dtype=torch.float64))
        idx_j, ptr, sh = _empty_csr(2)
        with pytest.raises(ValueError, match="sigma must be positive"):
            multipole_ewald_summation(
                pos, mm, _cell(), idx_j, ptr, sh, sigma=-1.0, alpha=0.3, k_cutoff=5.0
            )

    def test_alpha_nonpositive(self):
        # alpha + k_cutoff both given -> estimator skipped -> reach the alpha>0 guard.
        pos = _positions()
        mm = pack_multipole_moments(torch.zeros(2, dtype=torch.float64))
        idx_j, ptr, sh = _empty_csr(2)
        with pytest.raises(ValueError, match="alpha must be positive"):
            multipole_ewald_summation(
                pos, mm, _cell(), idx_j, ptr, sh, sigma=1.0, alpha=-1.0, k_cutoff=5.0
            )

    def test_l2_batched_requires_3d_cell(self):
        # l=2 (9-channel) + batch_idx but a single (3, 3) cell -> raise.
        pos = _positions()
        mm = pack_multipole_moments(
            torch.zeros(2, dtype=torch.float64),
            torch.zeros((2, 3), dtype=torch.float64),
            torch.zeros((2, 3, 3), dtype=torch.float64),
        )
        idx_j, ptr, sh = _empty_csr(2)
        batch_idx = torch.zeros(2, dtype=torch.int32)
        with pytest.raises(ValueError, match=r"\(B, 3, 3\)"):
            multipole_ewald_summation(
                pos,
                mm,
                _cell(),
                idx_j,
                ptr,
                sh,
                sigma=1.0,
                alpha=0.3,
                k_cutoff=5.0,
                batch_idx=batch_idx,
            )

    def test_l1_batched_requires_3d_cell(self):
        # l<=1 (4-channel) + batch_idx but a single (3, 3) cell -> raise.
        pos = _positions()
        mm = pack_multipole_moments(
            torch.zeros(2, dtype=torch.float64),
            torch.zeros((2, 3), dtype=torch.float64),
        )
        idx_j, ptr, sh = _empty_csr(2)
        batch_idx = torch.zeros(2, dtype=torch.int32)
        with pytest.raises(ValueError, match=r"\(B, 3, 3\)"):
            multipole_ewald_summation(
                pos,
                mm,
                _cell(),
                idx_j,
                ptr,
                sh,
                sigma=1.0,
                alpha=0.3,
                k_cutoff=5.0,
                batch_idx=batch_idx,
            )


class TestRealSpaceValidation:
    """Guard branches in ``multipole_real_space_energy``."""

    def test_moments_must_be_2d(self):
        pos = _positions()
        idx_j, ptr, sh = _empty_csr(2)
        bad_mm = torch.zeros(2, dtype=torch.float64)  # 1-D, not (N, C)
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_real_space_energy(
                pos, bad_mm, _cell(), idx_j, ptr, sh, sigma=1.0, alpha=0.3
            )


class TestElectrostaticEnergyValidation:
    """Guard branches in the Path-B ``multipole_electrostatic_energy``."""

    def test_batched_rejects_k_vectors(self):
        pos = _positions()
        mm = pack_multipole_moments(torch.zeros(2, dtype=torch.float64))
        batch_idx = torch.zeros(2, dtype=torch.int32)
        cell = _cell().unsqueeze(0)
        with pytest.raises(ValueError, match="k_vectors is not supported for batched"):
            multipole_electrostatic_energy(
                pos,
                mm,
                cell,
                sigma=1.0,
                batch_idx=batch_idx,
                k_vectors=torch.zeros((4, 3), dtype=torch.float64),
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SCF-cache build runs Warp geometry kernels"
)
class TestReciprocalCacheValidation:
    """``cache=`` structural checks in ``multipole_reciprocal_space_energy``."""

    def _single_l0_cache(self, device):
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
            prepare_multipole_scf_cache,
        )

        return prepare_multipole_scf_cache(
            _cell().to(device),
            sigma=1.0,
            receiver_sigmas=[1.0],
            k_cutoff=5.0,
            l_max=0,
            alpha=0.3,
            device=device,
        )

    def test_cache_batch_mismatch(self):
        device = "cuda"
        cache = self._single_l0_cache(device)  # single-system cache
        pos = _positions().to(device)
        mm = pack_multipole_moments(torch.zeros(2, dtype=torch.float64, device=device))
        cell = _cell().unsqueeze(0).to(device)  # (1, 3, 3) so the cell check passes
        batch_idx = torch.zeros(2, dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="cache.is_batched"):
            multipole_reciprocal_space_energy(
                pos, mm, cell, sigma=1.0, alpha=0.3, batch_idx=batch_idx, cache=cache
            )

    def test_cache_lmax_too_low(self):
        device = "cuda"
        cache = self._single_l0_cache(device)  # l_max=0 cache
        pos = _positions().to(device)
        mm = pack_multipole_moments(  # l=2 moments
            torch.zeros(2, dtype=torch.float64, device=device),
            torch.zeros((2, 3), dtype=torch.float64, device=device),
            torch.zeros((2, 3, 3), dtype=torch.float64, device=device),
        )
        with pytest.raises(ValueError, match="below the moment order"):
            multipole_reciprocal_space_energy(
                pos, mm, _cell().to(device), sigma=1.0, alpha=0.3, cache=cache
            )


class TestPmeResolveHelpers:
    """The precomputed-input resolve helpers in ``pme_multipole`` (the MD
    steady-state fast path callers never hit in the parity suite)."""

    MESH = (4, 4, 4)
    RFFT = (4, 4, 3)  # (nx, ny, nz // 2 + 1)

    def test_k_squared_precomputed_ok(self):
        from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
            _resolve_pme_k_squared,
        )

        ks = torch.ones(self.RFFT, dtype=torch.float64)
        out = _resolve_pme_k_squared(_cell(), self.MESH, torch.float64, ks)
        assert tuple(out.shape) == self.RFFT

    def test_k_squared_bad_shape(self):
        from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
            _resolve_pme_k_squared,
        )

        with pytest.raises(ValueError, match="k_squared shape"):
            _resolve_pme_k_squared(
                _cell(), self.MESH, torch.float64, torch.ones(self.MESH)
            )

    def test_batch_k_squared_precomputed_ok_and_bad(self):
        from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
            _resolve_batch_pme_k_squared,
        )

        cells = _cell().unsqueeze(0).repeat(2, 1, 1)
        good = torch.ones((2, *self.RFFT), dtype=torch.float64)
        out = _resolve_batch_pme_k_squared(cells, self.MESH, torch.float64, good)
        assert tuple(out.shape) == (2, *self.RFFT)
        with pytest.raises(ValueError, match="k_squared shape"):
            _resolve_batch_pme_k_squared(
                cells, self.MESH, torch.float64, torch.ones((2, *self.MESH))
            )

    def test_moduli_precomputed(self):
        from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
            _resolve_pme_moduli,
        )

        triplet = tuple(torch.ones(n, dtype=torch.float64) for n in self.MESH)
        bx, by, bz = _resolve_pme_moduli(
            self.MESH, 4, torch.float64, torch.device("cpu"), triplet
        )
        assert bx.shape[0] == 4 and bz.shape[0] == 4

    def test_cell_inv_t_precomputed_2d_and_3d(self):
        from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
            _resolve_cell_inv_t,
        )

        out2d = _resolve_cell_inv_t(_cell(), torch.eye(3, dtype=torch.float64))
        assert out2d.shape == (1, 3, 3)  # 2-D input is unsqueezed
        out3d = _resolve_cell_inv_t(_cell(), torch.eye(3, dtype=torch.float64)[None])
        assert out3d.shape == (1, 3, 3)

    def test_batch_cell_inv_t_precomputed_and_bad(self):
        from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
            _resolve_batch_cell_inv_t,
        )

        cells = _cell().unsqueeze(0).repeat(2, 1, 1)
        out = _resolve_batch_cell_inv_t(cells, cells.clone())
        assert out.shape == (2, 3, 3)
        with pytest.raises(ValueError, match=r"cell_inv_t must be shape \(B, 3, 3\)"):
            _resolve_batch_cell_inv_t(cells, torch.eye(3, dtype=torch.float64))
