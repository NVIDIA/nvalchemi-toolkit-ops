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

"""JAX binding tests for FourierD3.

The Warp layer is covered against a NumPy reference and a direct lattice sum elsewhere. These
tests check the binding: the transforms, the ``jax_kernel`` plumbing, both neighbour formats,
validation, and tracing under ``jax.jit``.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="No JAX installed.")
jnp = jax.numpy

from nvalchemiops.jax.interactions.dispersion import (  # noqa: E402
    FourierD3Parameters,
    fourier_dftd3,
)
from test.interactions.dispersion.test_fourier_dftd3 import (  # noqa: E402
    _neighbour_list,
    _reference_tables,
    _to_dense,
)

DAMPING = dict(a1=0.4289, a2=4.4407, s8=0.7875, s6=1.0)
R_CUT = 4.0
MESH = (32, 32, 32)


@pytest.fixture()
def device():
    """GPU device fixture.

    ``jax_kernel`` wrappers are CUDA-only, a Warp JAX FFI limitation, so these tests do not
    run on CPU.
    """
    try:
        if len(jax.devices("gpu")) == 0:
            pytest.skip("No CUDA device available.")
    except RuntimeError:
        pytest.skip("No CUDA device available.")
    return "gpu"


@pytest.fixture(scope="module")
def system():
    """A small periodic cell with its neighbour list in both formats."""
    rng = np.random.default_rng(0)
    c6ab, cn_ref, species = _reference_tables()
    max_z = c6ab.shape[0]
    rcov = np.zeros(max_z)
    rcov[[1, 6, 8]] = [0.6, 1.2, 1.1]
    r4r2 = np.zeros(max_z)
    r4r2[[1, 6, 8]] = [1.0, 1.4, 1.2]

    n_atoms, box = 8, 9.0
    positions = rng.uniform(0.0, box, (n_atoms, 3))
    numbers = rng.choice(species, n_atoms)
    cell = np.eye(3) * box
    targets, pointer, shifts, _ = _neighbour_list(positions, cell, R_CUT)
    sources = np.repeat(np.arange(n_atoms), np.diff(pointer))
    matrix, matrix_shifts = _to_dense(targets, pointer, shifts, n_atoms)

    return {
        "positions": jnp.asarray(positions),
        "numbers": jnp.asarray(numbers, dtype=jnp.int32),
        "cell": jnp.asarray(cell),
        "params": FourierD3Parameters.from_tables(rcov, r4r2, c6ab, cn_ref, species),
        "neighbor_list": jnp.asarray(np.stack([sources, targets]), dtype=jnp.int32),
        "neighbor_ptr": jnp.asarray(pointer, dtype=jnp.int32),
        "unit_shifts": jnp.asarray(shifts, dtype=jnp.int32),
        "neighbor_matrix": jnp.asarray(matrix, dtype=jnp.int32),
        "neighbor_matrix_shifts": jnp.asarray(matrix_shifts, dtype=jnp.int32),
        "numpy": {
            "positions": positions,
            "numbers": numbers,
            "cell": cell,
            "rcov": rcov,
            "r4r2": r4r2,
            "targets": targets,
            "pointer": pointer,
            "shifts": shifts,
            "species": species,
            "c6ab": c6ab,
            "cn_ref": cn_ref,
            "n_atoms": n_atoms,
        },
    }


def _evaluate(system, **kwargs):
    """Call the public API with the CSR neighbour list unless told otherwise."""
    arguments = dict(
        fd3_params=system["params"],
        cell=system["cell"],
        r_cut=R_CUT,
        mesh_dimensions=MESH,
        neighbor_list=system["neighbor_list"],
        neighbor_ptr=system["neighbor_ptr"],
        unit_shifts=system["unit_shifts"],
        **DAMPING,
    )
    arguments.update(kwargs)
    return fourier_dftd3(system["positions"], system["numbers"], **arguments)


@pytest.mark.gpu
class TestAgreementWithWarpLayer:
    """The binding must reproduce what the Warp layer already validated."""

    def test_matches_the_warp_pipeline(self, device, system):
        """Energy and forces agree with the NumPy-driven harness."""
        from nvalchemiops.interactions.dispersion._c6_decomposition import (
            decompose_c6_reference,
        )
        from test.interactions.dispersion._fourier_harness import fourier_d3_energy

        raw = system["numpy"]
        decomposition = decompose_c6_reference(
            raw["c6ab"], raw["cn_ref"], raw["species"]
        )
        reference = fourier_d3_energy(
            raw["positions"],
            raw["numbers"],
            decomposition.species_map[raw["numbers"]],
            np.zeros(raw["n_atoms"], dtype=np.int32),
            raw["cell"][None],
            raw["rcov"],
            decomposition,
            raw["r4r2"][decomposition.species],
            raw["targets"],
            raw["pointer"],
            raw["shifts"] @ raw["cell"],
            R_CUT,
            MESH,
            (DAMPING["s6"], DAMPING["s8"], DAMPING["a1"], DAMPING["a2"]),
        )
        energy, forces = _evaluate(system)
        np.testing.assert_allclose(np.asarray(energy), reference["energy"], rtol=1e-12)
        np.testing.assert_allclose(
            np.asarray(forces),
            reference["forces"],
            atol=1e-11 * np.abs(reference["forces"]).max(),
        )

    def test_forces_match_finite_differences(self, device, system):
        """The returned forces are the gradient of the returned energy."""
        base = system["positions"]
        analytic = np.asarray(_evaluate(system)[1])
        step = 1e-5
        numerical = np.zeros_like(analytic)
        for atom in range(base.shape[0]):
            for axis in range(3):
                for sign in (1.0, -1.0):
                    moved = base.at[atom, axis].add(sign * step)
                    energy = _evaluate({**system, "positions": moved})[0]
                    numerical[atom, axis] -= sign * float(energy[0]) / (2.0 * step)
        np.testing.assert_allclose(
            analytic, numerical, atol=1e-6 * np.abs(numerical).max()
        )

    def test_virial_is_returned_and_symmetric(self, device, system):
        """The virial is available and symmetric."""
        energy, forces, virial = _evaluate(system, compute_virial=True)
        virial = np.asarray(virial)[0]
        assert np.abs(virial).max() > 0.0
        np.testing.assert_allclose(virial, virial.T, atol=1e-12 * np.abs(virial).max())


@pytest.mark.gpu
class TestNeighbourFormats:
    """Both neighbour representations, and the validation around them."""

    def test_dense_and_csr_agree(self, device, system):
        """The two formats give the same answer."""
        csr = _evaluate(system)
        dense = _evaluate(
            system,
            neighbor_list=None,
            neighbor_ptr=None,
            unit_shifts=None,
            neighbor_matrix=system["neighbor_matrix"],
            neighbor_matrix_shifts=system["neighbor_matrix_shifts"],
        )
        np.testing.assert_allclose(np.asarray(csr[0]), np.asarray(dense[0]), rtol=1e-12)
        np.testing.assert_allclose(
            np.asarray(csr[1]),
            np.asarray(dense[1]),
            atol=1e-11 * float(jnp.abs(csr[1]).max()),
        )

    def test_rejects_both_formats(self, device, system):
        """Supplying both neighbour formats is an error."""
        with pytest.raises(ValueError, match="Cannot provide both"):
            _evaluate(system, neighbor_matrix=system["neighbor_matrix"])

    def test_rejects_neither_format(self, device, system):
        """Supplying no neighbour format is an error."""
        with pytest.raises(ValueError, match="Must provide either"):
            _evaluate(system, neighbor_list=None, neighbor_ptr=None, unit_shifts=None)


@pytest.mark.gpu
class TestJit:
    """Tracing behaviour."""

    def test_matches_eager_under_jit(self, device, system):
        """Compilation does not change the result."""
        eager = _evaluate(system)

        def evaluate(positions):
            return fourier_dftd3(
                positions,
                system["numbers"],
                fd3_params=system["params"],
                cell=system["cell"],
                r_cut=R_CUT,
                mesh_dimensions=MESH,
                neighbor_list=system["neighbor_list"],
                neighbor_ptr=system["neighbor_ptr"],
                unit_shifts=system["unit_shifts"],
                **DAMPING,
            )

        traced = jax.jit(evaluate)(system["positions"])
        np.testing.assert_allclose(
            np.asarray(traced[0]), np.asarray(eager[0]), rtol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(traced[1]),
            np.asarray(eager[1]),
            atol=1e-12 * float(jnp.abs(eager[1]).max()),
        )

    def test_mesh_spacing_is_rejected_while_tracing(self, device, system):
        """``mesh_spacing`` reads cell lengths, so it cannot be used inside ``jax.jit``.

        Failing with an explanation beats silently baking in whatever the tracer produced.
        """

        def evaluate(cell):
            return fourier_dftd3(
                system["positions"],
                system["numbers"],
                fd3_params=system["params"],
                cell=cell,
                r_cut=R_CUT,
                mesh_dimensions=None,
                mesh_spacing=0.3,
                neighbor_list=system["neighbor_list"],
                neighbor_ptr=system["neighbor_ptr"],
                unit_shifts=system["unit_shifts"],
                **DAMPING,
            )

        with pytest.raises(ValueError, match="not possible inside jax.jit"):
            jax.jit(evaluate)(system["cell"])
