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
    positions = arguments.pop("positions", system["positions"])
    return fourier_dftd3(positions, system["numbers"], **arguments)


def _single(box, seed):
    """One periodic cell, as plain NumPy, with its neighbour list in both formats."""
    rng = np.random.default_rng(seed)
    c6ab, cn_ref, species = _reference_tables()
    max_z = c6ab.shape[0]
    rcov = np.zeros(max_z)
    rcov[[1, 6, 8]] = [0.6, 1.2, 1.1]
    r4r2 = np.zeros(max_z)
    r4r2[[1, 6, 8]] = [1.0, 1.4, 1.2]
    n_atoms = 8
    positions = rng.uniform(0.0, box, (n_atoms, 3))
    numbers = rng.choice(species, n_atoms)
    cell = np.eye(3) * box
    targets, pointer, shifts, _ = _neighbour_list(positions, cell, R_CUT)
    matrix, matrix_shifts = _to_dense(targets, pointer, shifts, n_atoms)
    return dict(
        positions=positions,
        numbers=numbers,
        cell=cell,
        matrix=matrix,
        matrix_shifts=matrix_shifts,
        n_atoms=n_atoms,
        params=FourierD3Parameters.from_tables(rcov, r4r2, c6ab, cn_ref, species),
    )


def _dense_call(
    parts, cells, matrix, matrix_shifts, batch_idx, num_systems, fill_value
):
    """Evaluate in the dense neighbour format."""
    return fourier_dftd3(
        jnp.asarray(parts["positions"]),
        jnp.asarray(parts["numbers"], dtype=jnp.int32),
        **DAMPING,
        fd3_params=parts["params"],
        cell=jnp.asarray(cells),
        r_cut=R_CUT,
        mesh_dimensions=MESH,
        neighbor_matrix=jnp.asarray(matrix, dtype=jnp.int32),
        neighbor_matrix_shifts=jnp.asarray(matrix_shifts, dtype=jnp.int32),
        fill_value=fill_value,
        batch_idx=None
        if batch_idx is None
        else jnp.asarray(batch_idx, dtype=jnp.int32),
        num_systems=num_systems,
    )


@pytest.mark.gpu
class TestBatching:
    """Several systems in one call, with different cells.

    The binding builds the Cartesian shifts itself, so this is not covered by the Warp-layer
    batching tests. The boxes are small enough relative to ``R_CUT`` that both systems have
    periodic neighbours well inside it; a system whose image shifts are all zero cannot
    detect which cell they were multiplied by.
    """

    def test_dense_batch_keeps_each_systems_cell(self, device):
        """A system's periodic images must be built from its own lattice."""
        systems = [_single(5.0, 0), _single(7.0, 1)]
        counts = [s["n_atoms"] for s in systems]
        total = int(sum(counts))
        offsets = np.cumsum([0] + counts[:-1]).astype(np.int64)
        width = max(s["matrix"].shape[1] for s in systems)

        rows, row_shifts = [], []
        for system, offset in zip(systems, offsets):
            own = system["matrix"]
            row = np.full((system["n_atoms"], width), total, dtype=np.int32)
            shift = np.zeros((system["n_atoms"], width, 3), dtype=np.int32)
            padded = own >= system["n_atoms"]
            row[:, : own.shape[1]] = np.where(padded, total, own + int(offset))
            shift[:, : own.shape[1]] = system["matrix_shifts"]
            rows.append(row)
            row_shifts.append(shift)

        batch = dict(
            positions=np.concatenate([s["positions"] for s in systems]),
            numbers=np.concatenate([s["numbers"] for s in systems]),
            params=systems[0]["params"],
        )
        batch_idx = np.concatenate(
            [np.full(count, index) for index, count in enumerate(counts)]
        )
        together = _dense_call(
            batch,
            np.stack([s["cell"] for s in systems]),
            np.concatenate(rows),
            np.concatenate(row_shifts),
            batch_idx,
            len(systems),
            total,
        )

        start = 0
        for index, system in enumerate(systems):
            alone = _dense_call(
                system,
                system["cell"][None],
                system["matrix"],
                system["matrix_shifts"],
                None,
                None,
                system["n_atoms"],
            )
            stop = start + system["n_atoms"]
            np.testing.assert_allclose(
                float(together[0][index]), float(alone[0][0]), rtol=1e-11
            )
            np.testing.assert_allclose(
                np.asarray(together[1][start:stop]),
                np.asarray(alone[1]),
                atol=1e-11 * float(jnp.abs(alone[1]).max()),
            )
            start = stop


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

    def test_rejects_a_half_filled_list(self, device, system):
        """Half of the coordination contributions would simply go missing."""
        numpy = system["numpy"]
        sources = np.repeat(np.arange(numpy["n_atoms"]), np.diff(numpy["pointer"]))
        targets, shifts = numpy["targets"], numpy["shifts"]
        lexicographic = np.where(
            shifts[:, 0] != 0,
            shifts[:, 0],
            np.where(shifts[:, 1] != 0, shifts[:, 1], shifts[:, 2]),
        )
        keep = (sources < targets) | ((sources == targets) & (lexicographic > 0))
        order = np.argsort(sources[keep], kind="stable")
        half_sources = sources[keep][order]
        pointer = np.zeros(numpy["n_atoms"] + 1, dtype=np.int32)
        np.add.at(pointer, half_sources + 1, 1)
        with pytest.raises(ValueError, match="both directions of every pair"):
            _evaluate(
                system,
                neighbor_list=jnp.asarray(
                    np.stack([half_sources, targets[keep][order]]), dtype=jnp.int32
                ),
                neighbor_ptr=jnp.asarray(np.cumsum(pointer), dtype=jnp.int32),
                unit_shifts=jnp.asarray(shifts[keep][order], dtype=jnp.int32),
            )

    def test_a_mesh_shorter_than_the_stencil_is_refused(self, device, system):
        """The order-4 stencil would wrap onto a shorter axis and revisit a node."""
        with pytest.raises(ValueError, match="at least"):
            _evaluate(system, mesh_dimensions=(2, 2, 2), spline_order=4)

    def test_a_spacing_too_coarse_for_the_stencil_is_refused(self, device, system):
        """The spacing route is held to the same minimum as explicit dimensions."""
        with pytest.raises(ValueError, match="mesh_spacing"):
            _evaluate(system, mesh_dimensions=None, mesh_spacing=100.0, spline_order=4)

    def test_rejects_both_formats(self, device, system):
        """Supplying both neighbour formats is an error."""
        with pytest.raises(ValueError, match="Cannot provide both"):
            _evaluate(system, neighbor_matrix=system["neighbor_matrix"])

    def test_rejects_neither_format(self, device, system):
        """Supplying no neighbour format is an error."""
        with pytest.raises(ValueError, match="Must provide either"):
            _evaluate(system, neighbor_list=None, neighbor_ptr=None, unit_shifts=None)


@pytest.mark.gpu
class TestPrecision:
    """Both floating precisions are dispatched."""

    def test_float32_tracks_float64(self, device, system):
        """The single-precision path reproduces the double-precision one to its own accuracy.

        The Warp kernels are dtype-overloaded, so a missing or mismatched overload shows up
        as a dispatch failure or a silently different answer rather than as a type error.
        """
        numpy = system["numpy"]
        outputs = {}
        for dtype in (jnp.float64, jnp.float32):
            parameters = FourierD3Parameters.from_tables(
                numpy["rcov"],
                numpy["r4r2"],
                numpy["c6ab"],
                numpy["cn_ref"],
                numpy["species"],
                dtype=dtype,
            )
            outputs[dtype] = _evaluate(
                system,
                positions=jnp.asarray(numpy["positions"], dtype=dtype),
                cell=jnp.asarray(numpy["cell"], dtype=dtype),
                fd3_params=parameters,
            )
        double, single = outputs[jnp.float64], outputs[jnp.float32]
        assert single[0].dtype == jnp.float32
        assert single[1].dtype == jnp.float32
        # Single precision, so forces are compared against the magnitude of the result
        # rather than component by component: the small components carry the cancellation.
        np.testing.assert_allclose(float(single[0][0]), float(double[0][0]), rtol=1e-4)
        np.testing.assert_allclose(
            np.asarray(single[1]),
            np.asarray(double[1]),
            rtol=0.0,
            atol=1e-4 * float(jnp.abs(double[1]).max()),
        )


# A genuinely skewed lattice. A cubic cell's inverse is diagonal and equal to its own
# transpose, so a cubic test cannot detect a confusion between the two.
TRICLINIC = np.array([[9.0, 0.0, 0.0], [2.6, 8.4, 0.0], [1.8, -2.1, 9.3]])


@pytest.mark.gpu
class TestSkewedCell:
    """The JAX binding's own coordinate transforms, in a cell that can detect them."""

    @staticmethod
    def _case():
        """One skewed system with its neighbour list, as plain NumPy plus JAX arrays."""
        rng = np.random.default_rng(0)
        c6ab, cn_ref, species = _reference_tables()
        max_z = c6ab.shape[0]
        rcov = np.zeros(max_z)
        rcov[[1, 6, 8]] = [0.6, 1.2, 1.1]
        r4r2 = np.zeros(max_z)
        r4r2[[1, 6, 8]] = [1.0, 1.4, 1.2]
        n_atoms = 8
        positions = rng.uniform(0.0, 1.0, (n_atoms, 3)) @ TRICLINIC
        numbers = rng.choice(species, n_atoms)
        targets, pointer, shifts, _ = _neighbour_list(positions, TRICLINIC, R_CUT)
        sources = np.repeat(np.arange(n_atoms), np.diff(pointer))
        return dict(
            positions=positions,
            numbers=jnp.asarray(numbers, dtype=jnp.int32),
            params=FourierD3Parameters.from_tables(rcov, r4r2, c6ab, cn_ref, species),
            neighbor_list=jnp.asarray(np.stack([sources, targets]), dtype=jnp.int32),
            neighbor_ptr=jnp.asarray(pointer, dtype=jnp.int32),
            unit_shifts=jnp.asarray(shifts, dtype=jnp.int32),
            n_atoms=n_atoms,
        )

    @staticmethod
    def _call(case, positions, cell, compute_virial=False):
        return fourier_dftd3(
            jnp.asarray(positions),
            case["numbers"],
            **DAMPING,
            fd3_params=case["params"],
            cell=jnp.asarray(cell),
            r_cut=R_CUT,
            mesh_dimensions=MESH,
            compute_virial=compute_virial,
            neighbor_list=case["neighbor_list"],
            neighbor_ptr=case["neighbor_ptr"],
            unit_shifts=case["unit_shifts"],
        )

    def test_forces_match_finite_differences(self, device):
        """Cartesian forces are the gradient of the energy in a skewed cell."""
        case = self._case()
        analytic = np.asarray(self._call(case, case["positions"], TRICLINIC)[1])
        step = 1e-6
        numerical = np.zeros_like(analytic)
        for atom in range(case["n_atoms"]):
            for axis in range(3):
                shifted = []
                for sign in (1.0, -1.0):
                    moved = case["positions"].copy()
                    moved[atom, axis] += sign * step
                    shifted.append(float(self._call(case, moved, TRICLINIC)[0][0]))
                numerical[atom, axis] = -(shifted[0] - shifted[1]) / (2.0 * step)
        scale = np.abs(numerical).max()
        assert scale > 0.0
        np.testing.assert_allclose(analytic, numerical, atol=1e-6 * scale)

    def test_virial_matches_finite_strain(self, device):
        r"""All six independent strain derivatives, in the repository convention.

        :math:`W = -\partial E/\partial u` with :math:`R' = R(I+u)`, :math:`C' = C(I+u)`.
        """
        case = self._case()
        analytic = np.asarray(
            self._call(case, case["positions"], TRICLINIC, compute_virial=True)[2][0]
        )
        step = 1e-6
        for row, column in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
            shifted = []
            for sign in (1.0, -1.0):
                displacement = np.zeros((3, 3))
                displacement[row, column] += sign * step / 2.0
                displacement[column, row] += sign * step / 2.0
                if row == column:
                    displacement[row, column] = sign * step
                deformation = np.eye(3) + displacement
                shifted.append(
                    float(
                        self._call(
                            case,
                            case["positions"] @ deformation,
                            TRICLINIC @ deformation,
                        )[0][0]
                    )
                )
            numerical = -(shifted[0] - shifted[1]) / (2.0 * step)
            assert abs(analytic[row, column] - numerical) < 1e-6 * max(
                abs(numerical), np.abs(analytic).max()
            ), f"component ({row}, {column}): {analytic[row, column]} vs {numerical}"


@pytest.mark.gpu
class TestEmptySystem:
    """Zero atoms, which a padded or filtered batch can produce."""

    @staticmethod
    def _call(system, num_systems, **kwargs):
        return fourier_dftd3(
            jnp.zeros((0, 3)),
            jnp.zeros(0, dtype=jnp.int32),
            **DAMPING,
            fd3_params=system["params"],
            cell=jnp.broadcast_to(system["cell"], (num_systems, 3, 3)),
            r_cut=R_CUT,
            mesh_dimensions=MESH,
            num_systems=num_systems,
            neighbor_list=jnp.zeros((2, 0), dtype=jnp.int32),
            neighbor_ptr=jnp.zeros(1, dtype=jnp.int32),
            unit_shifts=jnp.zeros((0, 3), dtype=jnp.int32),
            **kwargs,
        )

    @pytest.mark.parametrize("num_systems", [1, 3])
    def test_returns_zeros_of_the_right_shape(self, device, system, num_systems):
        """No atoms means no dispersion, and the outputs still have to be well formed."""
        energy, forces, virial = self._call(system, num_systems, compute_virial=True)
        assert energy.shape == (num_systems,)
        assert forces.shape == (0, 3)
        assert virial.shape == (num_systems, 3, 3)
        assert float(jnp.abs(energy).max()) == 0.0
        assert float(jnp.abs(virial).max()) == 0.0

    def test_survives_tracing(self, device, system):
        """``n_atoms`` is a shape, so the early return is decided when the graph is built."""
        traced = jax.jit(lambda: self._call(system, 1))
        energy, forces = traced()
        assert energy.shape == (1,)
        assert forces.shape == (0, 3)
        assert float(jnp.abs(energy).max()) == 0.0


@pytest.mark.gpu
class TestModulusConvention:
    """Both B-spline attenuation conventions, matching the Torch binding's surface."""

    def test_the_two_conventions_differ(self, device, system):
        """The switch has to do something, or agreeing with Torch proves nothing."""
        exact = _evaluate(system, exact_moduli=True)
        continuous = _evaluate(system, exact_moduli=False)
        assert abs(float(exact[0][0]) - float(continuous[0][0])) > 1e-10 * abs(
            float(exact[0][0])
        )

    @pytest.mark.parametrize("exact_moduli", [True, False])
    def test_matches_the_torch_binding(self, device, system, exact_moduli):
        """The two bindings agree on both conventions, not only on the default."""
        torch = pytest.importorskip("torch", reason="PyTorch not installed.")
        from nvalchemiops.torch.interactions.dispersion import (
            FourierD3Parameters as TorchParameters,
        )
        from nvalchemiops.torch.interactions.dispersion import (
            fourier_dftd3 as torch_fourier_dftd3,
        )

        numpy = system["numpy"]
        ours = _evaluate(system, exact_moduli=exact_moduli)

        def tensor(array, dtype=torch.float64):
            return torch.as_tensor(
                np.ascontiguousarray(array), dtype=dtype, device="cuda:0"
            )

        parameters = TorchParameters.from_tables(
            tensor(numpy["rcov"]),
            tensor(numpy["r4r2"]),
            tensor(numpy["c6ab"]),
            tensor(numpy["cn_ref"]),
            numpy["species"],
            device="cuda:0",
            dtype=torch.float64,
        )
        sources = np.repeat(np.arange(numpy["n_atoms"]), np.diff(numpy["pointer"]))
        theirs = torch_fourier_dftd3(
            tensor(numpy["positions"]),
            tensor(numpy["numbers"], torch.int32),
            **DAMPING,
            fd3_params=parameters,
            cell=tensor(numpy["cell"]),
            r_cut=R_CUT,
            mesh_dimensions=MESH,
            exact_moduli=exact_moduli,
            neighbor_list=torch.stack(
                [tensor(sources, torch.int32), tensor(numpy["targets"], torch.int32)]
            ),
            neighbor_ptr=tensor(numpy["pointer"], torch.int32),
            unit_shifts=tensor(numpy["shifts"], torch.int32),
        )
        np.testing.assert_allclose(
            float(ours[0][0]), float(theirs[0][0].cpu()), rtol=1e-11
        )
        np.testing.assert_allclose(
            np.asarray(ours[1]),
            theirs[1].cpu().numpy(),
            atol=1e-11 * float(theirs[1].abs().max()),
        )


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
