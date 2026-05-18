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

"""JAX integration tests for ``lj_pme`` (PR3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("jax", reason="No JAX installed.")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nvalchemiops.jax.interactions.dispersion import (  # noqa: E402
    estimate_pme_dispersion_parameters,
    lj_pme,
)
from nvalchemiops.jax.interactions.dispersion.parameters import (  # noqa: E402
    solve_dispersion_beta,
)
from nvalchemiops.jax.neighbors import neighbor_list as nbr_list_fn  # noqa: E402


def _create_argon_fcc(n_cells: int, lattice_constant: float):
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
    #    Then flatten the first two dimensions into (-1, 3) and scale by lattice_constant
    positions_np = (cells + fcc_basis).reshape(-1, 3) * lattice_constant

    # 4. Convert directly to JAX arrays
    positions = jnp.array(positions_np, dtype=jnp.float64)

    c6 = jnp.full((positions.shape[0],), 60.0, dtype=jnp.float64)
    c12 = jnp.full((positions.shape[0],), 5.5e4, dtype=jnp.float64)
    cell = jnp.eye(3, dtype=jnp.float64) * lattice_constant * n_cells

    return positions, c6, c12, cell


def _two_atom_system(r: float, box: float = 20.0):
    positions = jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=jnp.float64)
    c6 = jnp.array([60.0, 60.0], dtype=jnp.float64)
    c12 = jnp.array([5.5e4, 5.5e4], dtype=jnp.float64)
    cell = jnp.eye(3, dtype=jnp.float64) * box
    return positions, c6, c12, cell


@pytest.mark.gpu
class TestParameterEstimatorJAX:
    def test_solve_dispersion_beta(self):
        for cutoff in [7.0, 9.0, 12.0]:
            for tol in [1e-3, 1e-5]:
                b = solve_dispersion_beta(cutoff, tol)
                x = b * cutoff
                g = math.exp(-x * x) * (1.0 + x * x + 0.5 * x * x * x * x)
                assert g == pytest.approx(tol, rel=0.1)

    def test_estimator_returns_dataclass(self, device):  # noqa: ARG002
        cell = jnp.eye(3, dtype=jnp.float64) * 20.0
        params = estimate_pme_dispersion_parameters(cell, cutoff=9.0, accuracy=1e-3)
        assert params.cutoff == 9.0
        assert params.beta.shape == (1,)
        assert params.accuracy == 1e-3


@pytest.mark.gpu
class TestLJPMEOrchestratorJAX:
    def test_auto_parameter_energy(self, device):  # noqa: ARG002
        positions, c6, c12, cell = _create_argon_fcc(3, 5.26)
        pbc = jnp.array([[True, True, True]], dtype=jnp.bool_)
        nm, nn, ns = nbr_list_fn(
            positions, 9.0, cell=cell[None], pbc=pbc, return_neighbor_list=False
        )
        E = lj_pme(positions, c6, c12, cell, nm, ns, num_neighbors=nn)
        assert E.shape == (1,)
        assert float(E[0]) < 0.0
        assert math.isfinite(float(E[0]))

    def test_two_atom_matches_pair_formula(self, device):  # noqa: ARG002
        r = 3.5
        positions, c6, c12, cell = _two_atom_system(r)
        pbc = jnp.array([[True, True, True]], dtype=jnp.bool_)
        cutoff = 12.0
        nm, nn, ns = nbr_list_fn(
            positions, cutoff, cell=cell[None], pbc=pbc, return_neighbor_list=False
        )
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
        assert float(E[0]) == pytest.approx(ref, rel=0.1)

    def test_momentum_conservation(self, device):  # noqa: ARG002
        positions, c6, c12, cell = _create_argon_fcc(2, 5.26)
        pbc = jnp.array([[True, True, True]], dtype=jnp.bool_)
        nm, nn, ns = nbr_list_fn(
            positions, 9.0, cell=cell[None], pbc=pbc, return_neighbor_list=False
        )
        _, F = lj_pme(
            positions, c6, c12, cell, nm, ns, num_neighbors=nn, compute_forces=True
        )
        np.testing.assert_allclose(np.asarray(F.sum(axis=0)), 0.0, atol=1e-9)

    def test_translation_invariance(self, device):  # noqa: ARG002
        positions, c6, c12, cell = _create_argon_fcc(2, 5.26)
        pbc = jnp.array([[True, True, True]], dtype=jnp.bool_)
        nm, nn, ns = nbr_list_fn(
            positions, 9.0, cell=cell[None], pbc=pbc, return_neighbor_list=False
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
        shift = jnp.array([0.7, 1.3, 2.1], dtype=jnp.float64)
        shifted = positions + shift
        nm2, nn2, ns2 = nbr_list_fn(
            shifted, 9.0, cell=cell[None], pbc=pbc, return_neighbor_list=False
        )
        E1 = lj_pme(
            shifted,
            c6,
            c12,
            cell,
            nm2,
            ns2,
            num_neighbors=nn2,
            mesh_dimensions=(48, 48, 48),
        )
        assert float(E1[0]) == pytest.approx(float(E0[0]), rel=1e-5)

    def test_dtype_parity(self, device):  # noqa: ARG002
        positions, c6, c12, cell = _create_argon_fcc(2, 5.26)
        pbc = jnp.array([[True, True, True]], dtype=jnp.bool_)
        nm, nn, ns = nbr_list_fn(
            positions, 9.0, cell=cell[None], pbc=pbc, return_neighbor_list=False
        )
        E64 = lj_pme(positions, c6, c12, cell, nm, ns, num_neighbors=nn)
        E32 = lj_pme(
            positions.astype(jnp.float32),
            c6.astype(jnp.float32),
            c12.astype(jnp.float32),
            cell.astype(jnp.float32),
            nm,
            ns,
            num_neighbors=nn,
        )
        assert E64.dtype == jnp.float64
        assert E32.dtype == jnp.float32
        assert float(E32[0]) == pytest.approx(float(E64[0]), rel=1e-3)

    def test_jit_compose(self, device):  # noqa: ARG002
        """``lj_pme`` composes with ``jax.jit``."""
        positions, c6, c12, cell = _create_argon_fcc(2, 5.26)
        pbc = jnp.array([[True, True, True]], dtype=jnp.bool_)
        nm, nn, ns = nbr_list_fn(
            positions, 9.0, cell=cell[None], pbc=pbc, return_neighbor_list=False
        )

        @jax.jit
        def f(pos):
            return lj_pme(
                pos,
                c6,
                c12,
                cell,
                nm,
                ns,
                num_neighbors=nn,
                beta=0.4,
                cutoff=9.0,
                mesh_dimensions=(32, 32, 32),
            )[0]

        E_jit = float(f(positions))
        E_eager = float(
            lj_pme(
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
            )[0]
        )
        assert E_jit == pytest.approx(E_eager, rel=1e-12)
