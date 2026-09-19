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

from nvalchemiops.interactions.dispersion._fourier_dftd3 import (  # noqa: E402
    _resolve_mesh,
)
from nvalchemiops.jax.interactions.dispersion import (  # noqa: E402
    FourierD3Parameters,
    fourier_dftd3,
)
from test.interactions.dispersion.test_fourier_dftd3 import (  # noqa: E402
    _neighbour_list,
    _reference_tables,
    _to_dense,
)


def _cell_lengths(cell):
    """Longest lattice-vector length per axis, the input `_resolve_mesh` expects."""
    cells = np.asarray(cell).reshape(-1, 3, 3)
    return np.linalg.norm(cells, axis=-1).max(axis=0).tolist()


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
        fourier_d3_params=system["params"],
        cell=system["cell"],
        cutoff=R_CUT,
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
        fourier_d3_params=parts["params"],
        cell=jnp.asarray(cells),
        cutoff=R_CUT,
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

    def test_a_spacing_derived_mesh_transforms_well(self, device, system):
        """An automatic mesh is rounded up to factors of 2, 3, 5 and 7, as in Torch."""
        for spacing in (0.07, 0.0709, 0.0711, 0.073, 0.11, 0.37):
            mesh = _resolve_mesh(None, spacing, _cell_lengths(system["cell"]), 4)
            for size in mesh:
                remainder = size
                for prime in (2, 3, 5, 7):
                    while remainder % prime == 0:
                        remainder //= prime
                assert remainder == 1, f"spacing {spacing} gave {mesh}"

    def test_explicit_dimensions_are_used_exactly(self, device, system):
        """A number the caller chose is passed through untouched."""
        assert _resolve_mesh((127, 127, 127), None, None, 4) == (127, 127, 127)

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
                fourier_d3_params=parameters,
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
            fourier_d3_params=case["params"],
            cell=jnp.asarray(cell),
            cutoff=R_CUT,
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
            fourier_d3_params=system["params"],
            cell=jnp.broadcast_to(system["cell"], (num_systems, 3, 3)),
            cutoff=R_CUT,
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

    @pytest.mark.parametrize(
        "bad_mesh",
        [
            pytest.param({}, id="neither"),
            pytest.param({"mesh_dimensions": MESH, "mesh_spacing": 0.3}, id="both"),
            pytest.param({"mesh_dimensions": (2, 2, 2)}, id="below_stencil"),
        ],
    )
    def test_invalid_arguments_are_still_rejected(self, device, system, bad_mesh):
        """An empty batch is validated exactly as a populated one."""
        with pytest.raises(ValueError):
            fourier_dftd3(
                jnp.zeros((0, 3)),
                jnp.zeros(0, dtype=jnp.int32),
                **DAMPING,
                fourier_d3_params=system["params"],
                cell=system["cell"],
                cutoff=R_CUT,
                neighbor_list=jnp.zeros((2, 0), dtype=jnp.int32),
                neighbor_ptr=jnp.zeros(1, dtype=jnp.int32),
                unit_shifts=jnp.zeros((0, 3), dtype=jnp.int32),
                **bad_mesh,
            )

    def test_survives_tracing(self, device, system):
        """``n_atoms`` is a shape, so the early return is decided when the graph is built."""
        traced = jax.jit(lambda: self._call(system, 1))
        energy, forces = traced()
        assert energy.shape == (1,)
        assert forces.shape == (0, 3)
        assert float(jnp.abs(energy).max()) == 0.0


@pytest.mark.gpu
class TestCrossBinding:
    """The JAX and Torch bindings evaluate the same thing."""

    def test_matches_the_torch_binding(self, device, system):
        """The two bindings agree on energy and forces."""
        torch = pytest.importorskip("torch", reason="PyTorch not installed.")
        from nvalchemiops.torch.interactions.dispersion import (
            FourierD3Parameters as TorchParameters,
        )
        from nvalchemiops.torch.interactions.dispersion import (
            fourier_dftd3 as torch_fourier_dftd3,
        )

        numpy = system["numpy"]
        ours = _evaluate(system)

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
            fourier_d3_params=parameters,
            cell=tensor(numpy["cell"]),
            cutoff=R_CUT,
            mesh_dimensions=MESH,
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
                fourier_d3_params=system["params"],
                cell=system["cell"],
                cutoff=R_CUT,
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
                fourier_d3_params=system["params"],
                cell=cell,
                cutoff=R_CUT,
                mesh_dimensions=None,
                mesh_spacing=0.3,
                neighbor_list=system["neighbor_list"],
                neighbor_ptr=system["neighbor_ptr"],
                unit_shifts=system["unit_shifts"],
                **DAMPING,
            )

        with pytest.raises(ValueError, match="not possible inside jax.jit"):
            jax.jit(evaluate)(system["cell"])


@pytest.mark.gpu
class TestBatchArgumentValidation:
    """Batch arguments must be checked before they can corrupt the result.

    An out-of-range index sends an atom to a mesh slab owned by no system and its whole
    contribution disappears: a uniform ``batch_idx`` of 5 against one system returned exactly
    zero energy with no error.
    """

    @staticmethod
    def _call(parts, **extra):
        return fourier_dftd3(
            jnp.asarray(parts["positions"]),
            jnp.asarray(parts["numbers"], dtype=jnp.int32),
            **DAMPING,
            fourier_d3_params=parts["params"],
            cell=jnp.asarray(parts["cell"]),
            cutoff=R_CUT,
            mesh_dimensions=MESH,
            neighbor_matrix=jnp.asarray(parts["matrix"], dtype=jnp.int32),
            neighbor_matrix_shifts=jnp.asarray(parts["matrix_shifts"], dtype=jnp.int32),
            **extra,
        )

    def test_num_systems_must_match_the_cells(self):
        """Static, so this is caught while tracing too."""
        parts = _single(5.0, 0)
        with pytest.raises(ValueError, match="but cell holds"):
            self._call(parts, num_systems=3)

    def test_batch_idx_must_cover_every_atom(self):
        """A short batch_idx used to fail inside a broadcast."""
        parts = _single(5.0, 0)
        with pytest.raises(ValueError, match="entries but there are"):
            self._call(parts, batch_idx=jnp.zeros(parts["n_atoms"] // 2, jnp.int32))

    @pytest.mark.parametrize("value", [-1, 5])
    def test_batch_idx_must_be_in_range(self, value):
        """Both ends matter: negative and past the last system."""
        parts = _single(5.0, 0)
        with pytest.raises(ValueError, match="outside"):
            self._call(parts, batch_idx=jnp.full((parts["n_atoms"],), value, jnp.int32))

    def test_the_static_check_survives_tracing(self):
        """``num_systems`` is a shape, so ``jax.jit`` does not hide the mismatch."""
        parts = _single(5.0, 0)
        with pytest.raises(ValueError, match="but cell holds"):
            jax.jit(lambda p: self._call(parts, num_systems=3))(
                jnp.asarray(parts["positions"])
            )

    def test_the_traced_path_is_guarded_too(self):
        """``jax.jit`` is documented as mandatory, so the check must survive tracing.

        The values cannot be read while tracing, so the indices are clamped to keep the
        kernels in bounds and the result is poisoned to keep the clamp honest.
        """
        parts = _single(5.0, 0)
        traced = jax.jit(lambda b: self._call(parts, batch_idx=b)[0])

        valid = jnp.zeros(parts["n_atoms"], jnp.int32)
        np.testing.assert_allclose(
            np.asarray(traced(valid)),
            np.asarray(self._call(parts, batch_idx=valid)[0]),
            rtol=1e-12,
        )

        out_of_range = jnp.full((parts["n_atoms"],), 5, jnp.int32)
        assert np.isnan(np.asarray(traced(out_of_range))).all()

    def test_a_valid_single_system_batch_is_accepted(self):
        """The guard must not reject the ordinary case it is wrapped around."""
        parts = _single(5.0, 0)
        explicit = self._call(
            parts, batch_idx=jnp.zeros(parts["n_atoms"], jnp.int32), num_systems=1
        )[0]
        default = self._call(parts)[0]
        np.testing.assert_allclose(
            np.asarray(explicit), np.asarray(default), rtol=1e-12
        )


@pytest.mark.gpu
class TestNeighbourArgumentValidation:
    """Each format needs its companion arrays, and says so at the API boundary.

    Without these the missing argument reaches ``jnp.asarray`` and surfaces as
    ``None is not a valid value for jnp.array``, which names neither the argument nor the
    format the caller chose.
    """

    @staticmethod
    def _csr(parts):
        """The dense list re-expressed as a directed CSR list."""
        matrix = np.asarray(parts["matrix"])
        shifts = np.asarray(parts["matrix_shifts"])
        n_atoms = parts["n_atoms"]
        sources, targets, images = [], [], []
        for i in range(n_atoms):
            for slot in range(matrix.shape[1]):
                j = int(matrix[i, slot])
                if j < n_atoms:
                    sources.append(i)
                    targets.append(j)
                    images.append(shifts[i, slot])
        pointer = np.zeros(n_atoms + 1, dtype=np.int32)
        for source in sources:
            pointer[source + 1] += 1
        return (
            jnp.asarray(np.stack([sources, targets]), dtype=jnp.int32),
            jnp.asarray(np.cumsum(pointer), dtype=jnp.int32),
            jnp.asarray(np.stack(images), dtype=jnp.int32),
        )

    @staticmethod
    def _call(parts, **neighbours):
        return fourier_dftd3(
            jnp.asarray(parts["positions"]),
            jnp.asarray(parts["numbers"], dtype=jnp.int32),
            **DAMPING,
            fourier_d3_params=parts["params"],
            cell=jnp.asarray(parts["cell"]),
            cutoff=R_CUT,
            mesh_dimensions=MESH,
            **neighbours,
        )

    def test_dense_format_requires_its_shifts(self):
        """FourierD3 is periodic, so every neighbour needs a lattice image."""
        parts = _single(5.0, 0)
        with pytest.raises(ValueError, match="neighbor_matrix_shifts is required"):
            self._call(
                parts, neighbor_matrix=jnp.asarray(parts["matrix"], dtype=jnp.int32)
            )

    def test_csr_format_requires_its_pointer(self):
        """The row offsets are how each atom's slice is found."""
        parts = _single(5.0, 0)
        targets, _pointer, images = self._csr(parts)
        with pytest.raises(ValueError, match="neighbor_ptr is required"):
            self._call(parts, neighbor_list=targets, unit_shifts=images)

    def test_csr_format_requires_its_shifts(self):
        """Same periodicity requirement as the dense path."""
        parts = _single(5.0, 0)
        targets, pointer, _images = self._csr(parts)
        with pytest.raises(ValueError, match="unit_shifts is required"):
            self._call(parts, neighbor_list=targets, neighbor_ptr=pointer)

    def test_a_complete_dense_call_still_works(self):
        """The checks must not reject a valid call."""
        parts = _single(5.0, 0)
        energy = self._call(
            parts,
            neighbor_matrix=jnp.asarray(parts["matrix"], dtype=jnp.int32),
            neighbor_matrix_shifts=jnp.asarray(parts["matrix_shifts"], dtype=jnp.int32),
        )[0]
        assert np.isfinite(np.asarray(energy)).all()


@pytest.mark.gpu
class TestUncoveredSpecies:
    """An element missing from ``fourier_d3_params`` must not be dropped in silence.

    ``species_map`` marks padding and uncovered elements alike with ``-1``, and the mesh
    grouping treats every ``-1`` as padding, so without a check a real atom simply vanishes
    from the sum and the energy comes back plausible but wrong.
    """

    @staticmethod
    def _call(parts, params, numbers):
        return fourier_dftd3(
            jnp.asarray(parts["positions"]),
            numbers,
            **DAMPING,
            fourier_d3_params=params,
            cell=jnp.asarray(parts["cell"]),
            cutoff=R_CUT,
            mesh_dimensions=MESH,
            neighbor_matrix=jnp.asarray(parts["matrix"], dtype=jnp.int32),
            neighbor_matrix_shifts=jnp.asarray(parts["matrix_shifts"], dtype=jnp.int32),
        )

    @staticmethod
    def _partial_params():
        """Parameters built without the last species the system actually contains."""
        c6ab, cn_ref, species = _reference_tables()
        max_z = c6ab.shape[0]
        rcov = np.zeros(max_z)
        rcov[[1, 6, 8]] = [0.6, 1.2, 1.1]
        r4r2 = np.zeros(max_z)
        r4r2[[1, 6, 8]] = [1.0, 1.4, 1.2]
        return FourierD3Parameters.from_tables(
            rcov, r4r2, c6ab, cn_ref, list(species[:-1])
        ), int(species[-1])

    def test_eager_rejects_an_uncovered_element(self):
        """The message has to name the element so the fix is obvious."""
        parts = _single(5.0, 0)
        params, missing = self._partial_params()
        numbers = np.asarray(parts["numbers"]).copy()
        numbers[0] = missing
        with pytest.raises(ValueError, match="not covered by fourier_d3_params"):
            self._call(parts, params, jnp.asarray(numbers, dtype=jnp.int32))

    def test_padding_is_not_mistaken_for_an_uncovered_element(self):
        """Atomic number zero is padding by design and must still be accepted."""
        parts = _single(5.0, 0)
        numbers = np.asarray(parts["numbers"]).copy()
        numbers[:2] = 0
        energy = self._call(
            parts, parts["params"], jnp.asarray(numbers, dtype=jnp.int32)
        )[0]
        assert np.isfinite(np.asarray(energy)).all()

    def test_under_jit_the_result_is_poisoned_rather_than_wrong(self):
        """``numbers`` is a tracer, so the mask cannot be read back to raise.

        Returning a finite energy here would be the damaging outcome, because ``jax.jit`` is
        the documented execution path: the caller would get a plausible number quietly
        missing an atom. NaN is unmissable and needs no host synchronisation.
        """
        parts = _single(5.0, 0)
        params, missing = self._partial_params()
        numbers = np.asarray(parts["numbers"]).copy()
        numbers[0] = missing

        traced = jax.jit(lambda n: self._call(parts, params, n))
        energy, forces = traced(jnp.asarray(numbers, dtype=jnp.int32))[:2]
        assert np.isnan(np.asarray(energy)).all()
        assert np.isnan(np.asarray(forces)).all()

    def test_a_covered_system_is_untouched_under_jit(self):
        """The guard must cost nothing when every element is present."""
        parts = _single(5.0, 0)
        numbers = jnp.asarray(parts["numbers"], dtype=jnp.int32)
        traced = jax.jit(lambda n: self._call(parts, parts["params"], n))
        energy = traced(numbers)[0]
        eager = self._call(parts, parts["params"], numbers)[0]
        np.testing.assert_allclose(np.asarray(energy), np.asarray(eager), rtol=1e-12)


@pytest.mark.gpu
class TestParametersUnderJit:
    """``FourierD3Parameters`` has to survive being a traced argument.

    A plain dataclass is one opaque leaf, so ``jax.jit`` rejects it as an argument and the
    caller is forced to close over it -- which retraces whenever the parameters change.
    """

    def test_it_is_a_registered_pytree(self):
        """Six array fields flatten out; the tolerance stays static metadata."""
        parts = _single(5.0, 0)
        params = parts["params"]
        leaves, treedef = jax.tree_util.tree_flatten(params)
        assert len(leaves) == 6

        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored.rank == params.rank
        assert restored.n_species == params.n_species
        assert restored.max_relative_error == params.max_relative_error

    def test_it_can_be_passed_as_a_runtime_argument(self):
        """Passing the container into a jitted call must match closing over it."""
        parts = _single(5.0, 0)
        positions = jnp.asarray(parts["positions"])

        def evaluate(positions, params):
            return fourier_dftd3(
                positions,
                jnp.asarray(parts["numbers"], dtype=jnp.int32),
                **DAMPING,
                fourier_d3_params=params,
                cell=jnp.asarray(parts["cell"]),
                cutoff=R_CUT,
                mesh_dimensions=MESH,
                neighbor_matrix=jnp.asarray(parts["matrix"], dtype=jnp.int32),
                neighbor_matrix_shifts=jnp.asarray(
                    parts["matrix_shifts"], dtype=jnp.int32
                ),
                compute_virial=True,
            )

        runtime = jax.jit(evaluate)(positions, parts["params"])
        closure = jax.jit(lambda x: evaluate(x, parts["params"]))(positions)
        for from_argument, from_closure in zip(runtime, closure):
            np.testing.assert_allclose(
                np.asarray(from_argument), np.asarray(from_closure), rtol=1e-12
            )


@pytest.mark.gpu
class TestEnergyIsNotDifferentiable:
    """The documented contract: forces are an output, not an autodiff result.

    The kernels run with ``enable_backward=False`` and no VJP or JVP rule is registered on
    top of them, which is what keeps the call traceable under ``jax.jit``. This pins the
    consequence so the guide cannot drift away from the behaviour.
    """

    def test_grad_raises_rather_than_returning_a_wrong_gradient(self):
        """A silent zero or a wrong gradient would be far worse than an error."""
        parts = _single(5.0, 0)

        def total(positions):
            return fourier_dftd3(
                positions,
                jnp.asarray(parts["numbers"], dtype=jnp.int32),
                **DAMPING,
                fourier_d3_params=parts["params"],
                cell=jnp.asarray(parts["cell"]),
                cutoff=R_CUT,
                mesh_dimensions=MESH,
                neighbor_matrix=jnp.asarray(parts["matrix"], dtype=jnp.int32),
                neighbor_matrix_shifts=jnp.asarray(
                    parts["matrix_shifts"], dtype=jnp.int32
                ),
            )[0].sum()

        positions = jnp.asarray(parts["positions"])
        # The forward pass works; only the transpose is missing.
        assert float(total(positions)) != 0.0
        with pytest.raises(ValueError, match="cannot be differentiated"):
            jax.grad(total)(positions)


@pytest.mark.gpu
class TestRankChunking:
    """Splitting the rank across passes trades memory for extra launches.

    Every stage after the coordination number is a sum over rank slots with no coupling
    between them, so the split must not change the answer, traced or not.
    """

    @pytest.mark.parametrize("chunk", [1, 2, 8])
    @pytest.mark.parametrize("jit", [False, True])
    def test_chunked_matches_a_single_pass(self, chunk, jit):
        """The chunk count is host-static, so the loop unrolls at trace time."""
        parts = _single(5.0, 0)

        def evaluate(positions, rank_chunk_size):
            return fourier_dftd3(
                positions,
                jnp.asarray(parts["numbers"], dtype=jnp.int32),
                **DAMPING,
                fourier_d3_params=parts["params"],
                cell=jnp.asarray(parts["cell"]),
                cutoff=R_CUT,
                mesh_dimensions=MESH,
                neighbor_matrix=jnp.asarray(parts["matrix"], dtype=jnp.int32),
                neighbor_matrix_shifts=jnp.asarray(
                    parts["matrix_shifts"], dtype=jnp.int32
                ),
                rank_chunk_size=rank_chunk_size,
                compute_virial=True,
            )

        run = jax.jit(evaluate, static_argnums=1) if jit else evaluate
        positions = jnp.asarray(parts["positions"])
        whole = run(positions, None)
        split = run(positions, chunk)
        for reference, chunked in zip(whole, split):
            np.testing.assert_allclose(
                np.asarray(chunked), np.asarray(reference), rtol=1e-11, atol=1e-13
            )

    def test_a_non_positive_chunk_is_refused(self):
        """Zero would make no progress and loop forever."""
        parts = _single(5.0, 0)
        with pytest.raises(ValueError, match="at least 1"):
            fourier_dftd3(
                jnp.asarray(parts["positions"]),
                jnp.asarray(parts["numbers"], dtype=jnp.int32),
                **DAMPING,
                fourier_d3_params=parts["params"],
                cell=jnp.asarray(parts["cell"]),
                cutoff=R_CUT,
                mesh_dimensions=MESH,
                neighbor_matrix=jnp.asarray(parts["matrix"], dtype=jnp.int32),
                neighbor_matrix_shifts=jnp.asarray(
                    parts["matrix_shifts"], dtype=jnp.int32
                ),
                rank_chunk_size=0,
            )


class TestParameterValidation:
    """Same contract as the Torch container, and it must survive being a pytree.

    ``__post_init__`` runs again when the dataclass is rebuilt inside ``jax.jit``, where the
    fields are tracers, so it may only read attributes a tracer carries.
    """

    @staticmethod
    def _fields():
        return dict(
            rcov=jnp.zeros(5),
            sqrt_q=jnp.zeros(2),
            cn_ref=jnp.zeros((2, 3)),
            v_q=jnp.zeros((2, 3, 4)),
            eigs=jnp.zeros(4),
            species_map=jnp.zeros(5, dtype=jnp.int32),
            max_relative_error=0.0,
        )

    def test_a_well_formed_bundle_is_accepted(self):
        """The guard must not reject the shape it is built around."""
        params = FourierD3Parameters(**self._fields())
        assert params.rank == 4
        assert params.n_species == 2

    def test_it_still_round_trips_through_jit(self):
        """Validation on a pytree node must tolerate tracers as leaves."""
        params = FourierD3Parameters(**self._fields())
        total = jax.jit(lambda bundle: bundle.eigs.sum())(params)
        assert float(total) == 0.0

    @pytest.mark.parametrize("name", ["rcov", "sqrt_q", "cn_ref", "v_q", "eigs"])
    def test_numeric_fields_must_be_floating(self, name):
        """An integral eigs silently truncates the decomposition."""
        fields = self._fields()
        fields[name] = fields[name].astype(jnp.int32)
        with pytest.raises(TypeError, match="must be float32 or float64"):
            FourierD3Parameters(**fields)

    def test_species_map_must_be_integral(self):
        """It indexes the reference tables, so a float is not merely imprecise."""
        fields = self._fields()
        fields["species_map"] = fields["species_map"].astype(jnp.float32)
        with pytest.raises(TypeError, match="must be int32 or int64"):
            FourierD3Parameters(**fields)

    @pytest.mark.parametrize(
        ("name", "shape"),
        [("rcov", (1, 5)), ("sqrt_q", (1, 2)), ("eigs", (1, 4)), ("cn_ref", (2, 3, 1))],
    )
    def test_ranks_are_enforced(self, name, shape):
        """A stray axis passes the pairwise shape checks but not the kernels."""
        fields = self._fields()
        fields[name] = fields[name].reshape(shape)
        with pytest.raises(ValueError, match="must be .D"):
            FourierD3Parameters(**fields)


@pytest.mark.gpu
class TestBackendGuard:
    """The block-per-atom passes must give the same answer on either backend.

    They reduce within a block, which Warp's CPU backend cannot do; the launches narrow to
    one thread per item there. Getting that wrong does not fail, it returns a plausible
    energy several percent out, so the backends are compared directly.
    """

    @staticmethod
    def _call(parts, arrays):
        positions, numbers, cell, matrix, shifts = arrays
        return fourier_dftd3(
            positions,
            numbers,
            **DAMPING,
            fourier_d3_params=parts["params"],
            cell=cell,
            cutoff=R_CUT,
            mesh_dimensions=MESH,
            neighbor_matrix=matrix,
            neighbor_matrix_shifts=shifts,
        )[0]

    @staticmethod
    def _arrays(parts, device=None):
        arrays = (
            jnp.asarray(parts["positions"]),
            jnp.asarray(parts["numbers"], dtype=jnp.int32),
            jnp.asarray(parts["cell"]),
            jnp.asarray(parts["matrix"], dtype=jnp.int32),
            jnp.asarray(parts["matrix_shifts"], dtype=jnp.int32),
        )
        if device is None:
            return arrays
        return tuple(jax.device_put(array, device) for array in arrays)

    def test_cpu_and_gpu_agree(self):
        """Eager placement is concrete, so the width follows the arrays' own device."""
        parts = _single(5.0, 0)
        cpu = jax.devices("cpu")[0]
        on_gpu = self._call(parts, self._arrays(parts))
        on_cpu = self._call(parts, self._arrays(parts, cpu))
        np.testing.assert_allclose(np.asarray(on_cpu), np.asarray(on_gpu), rtol=1e-12)

    def test_cpu_and_gpu_agree_under_jit(self):
        """A tracer carries no device, so lowering picks the width instead.

        This is the case a concrete check cannot reach: arrays placed on CPU and then
        jitted while the process default is GPU.
        """
        parts = _single(5.0, 0)
        cpu = jax.devices("cpu")[0]
        traced = jax.jit(lambda *a: self._call(parts, a))
        on_gpu = traced(*self._arrays(parts))
        on_cpu = traced(*self._arrays(parts, cpu))
        np.testing.assert_allclose(np.asarray(on_cpu), np.asarray(on_gpu), rtol=1e-12)

    def test_gpu_placement_is_accepted(self):
        """The guard must not reject the backend the binding is built for."""
        parts = _single(5.0, 0)
        energy = self._call(parts, self._arrays(parts))
        assert np.isfinite(np.asarray(energy)).all()

    def test_tracing_does_not_guess_the_backend(self):
        """A tracer carries no placement, so the check steps aside rather than guess.

        Guessing from ``jax.default_backend()`` was wrong in both directions: it let a
        CPU-placed jit call through, and would have rejected a GPU jit call whenever the
        process default happened to be CPU. This pins the ordinary jitted GPU path working.
        """
        parts = _single(5.0, 0)
        arrays = self._arrays(parts)
        traced = jax.jit(lambda *a: self._call(parts, a))(*arrays)
        assert np.isfinite(np.asarray(traced)).all()
        np.testing.assert_allclose(
            np.asarray(traced), np.asarray(self._call(parts, arrays)), rtol=1e-12
        )
