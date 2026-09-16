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

"""PyTorch binding tests for FourierD3.

The Warp layer is already covered against a NumPy reference and a direct lattice sum in
``test/interactions/dispersion/test_fourier_dftd3.py``. These tests check what the binding
itself adds: the two Fourier transforms, tensor plumbing, neighbour-format handling, unit and
mesh validation, and the parameter object.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from nvalchemiops.torch.interactions.dispersion import (
        FourierD3Parameters,
        FourierD3Setup,
        fourier_dftd3,
    )
    from test.interactions.dispersion.test_fourier_dftd3 import (
        _neighbour_list,
        _reference_tables,
        _to_dense,
    )

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed - these tests require torch to be available",
)

DAMPING = dict(a1=0.4289, a2=4.4407, s8=0.7875, s6=1.0)
R_CUT = 4.0
MESH = (32, 32, 32)


def _system(device, dtype=torch.float64, n_atoms=8, box=9.0, seed=0):
    """A small periodic cell with its neighbour list in both formats."""
    rng = np.random.default_rng(seed)
    c6ab, cn_ref, species = _reference_tables()
    max_z = c6ab.shape[0]
    rcov = np.zeros(max_z)
    rcov[[1, 6, 8]] = [0.6, 1.2, 1.1]
    r4r2 = np.zeros(max_z)
    r4r2[[1, 6, 8]] = [1.0, 1.4, 1.2]

    positions = rng.uniform(0.0, box, (n_atoms, 3))
    numbers = rng.choice(species, n_atoms)
    cell = np.eye(3) * box
    targets, pointer, shifts, _ = _neighbour_list(positions, cell, R_CUT)
    sources = np.repeat(np.arange(n_atoms), np.diff(pointer))

    def tensor(array, torch_dtype=dtype):
        return torch.as_tensor(
            np.ascontiguousarray(array), dtype=torch_dtype, device=device
        )

    parameters = FourierD3Parameters.from_tables(
        tensor(rcov),
        tensor(r4r2),
        tensor(c6ab),
        tensor(cn_ref),
        species,
        device=device,
        dtype=dtype,
    )
    matrix, matrix_shifts = _to_dense(targets, pointer, shifts, n_atoms)
    return {
        "positions": tensor(positions),
        "numbers": tensor(numbers, torch.int32),
        "cell": tensor(cell),
        "params": parameters,
        "neighbor_list": torch.stack(
            [tensor(sources, torch.int32), tensor(targets, torch.int32)]
        ),
        "neighbor_ptr": tensor(pointer, torch.int32),
        "unit_shifts": tensor(shifts, torch.int32),
        "neighbor_matrix": tensor(matrix, torch.int32),
        "neighbor_matrix_shifts": tensor(matrix_shifts, torch.int32),
        "n_atoms": n_atoms,
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

    def test_matches_the_warp_pipeline(self):
        """Energy and forces agree with the NumPy-driven harness.

        The harness runs the same launchers with NumPy transforms, so this isolates the
        binding's own plumbing and its use of ``torch.fft``.
        """
        from test.interactions.dispersion._fourier_harness import fourier_d3_energy

        device = "cuda:0"
        system = _system(device)
        energy, forces = _evaluate(system)

        parameters = system["params"]
        reference = fourier_d3_energy(
            system["positions"].cpu().numpy(),
            system["numbers"].cpu().numpy(),
            parameters.species_map.cpu().numpy()[system["numbers"].cpu().numpy()],
            np.zeros(system["n_atoms"], dtype=np.int32),
            system["cell"].cpu().numpy()[None],
            parameters.rcov.cpu().numpy(),
            _decomposition_view(parameters),
            parameters.sqrt_q.cpu().numpy(),
            system["neighbor_list"][1].cpu().numpy(),
            system["neighbor_ptr"].cpu().numpy(),
            system["unit_shifts"].cpu().numpy() @ system["cell"].cpu().numpy(),
            R_CUT,
            MESH,
            (DAMPING["s6"], DAMPING["s8"], DAMPING["a1"], DAMPING["a2"]),
            device=device,
        )
        np.testing.assert_allclose(
            energy.cpu().numpy(), reference["energy"], rtol=1e-12
        )
        np.testing.assert_allclose(
            forces.cpu().numpy(),
            reference["forces"],
            atol=1e-11 * np.abs(reference["forces"]).max(),
        )

    def test_forces_match_finite_differences(self):
        """The returned forces are the gradient of the returned energy."""
        device = "cuda:0"
        system = _system(device, n_atoms=6, seed=3)
        analytic = _evaluate(system)[1].cpu().numpy()

        step = 1e-5
        base = system["positions"].clone()
        numerical = np.zeros_like(analytic)
        for atom in range(system["n_atoms"]):
            for axis in range(3):
                for sign in (1.0, -1.0):
                    system["positions"] = base.clone()
                    system["positions"][atom, axis] += sign * step
                    energy = _evaluate(system)[0]
                    numerical[atom, axis] -= sign * float(energy) / (2.0 * step)
        system["positions"] = base
        np.testing.assert_allclose(
            analytic, numerical, atol=1e-6 * np.abs(numerical).max()
        )

    def test_virial_matches_finite_strain(self):
        """The returned virial is the strain derivative of the returned energy."""
        device = "cuda:0"
        system = _system(device, n_atoms=6, seed=3)
        analytic = _evaluate(system, compute_virial=True)[2][0].cpu().numpy()

        step = 1e-6
        base_positions = system["positions"].clone()
        base_cell = system["cell"].clone()
        numerical = np.zeros((3, 3))
        for row in range(3):
            for column in range(3):
                energies = []
                for sign in (1.0, -1.0):
                    strain = torch.zeros(3, 3, dtype=base_cell.dtype, device=device)
                    strain[row, column] = sign * step
                    deformation = (
                        torch.eye(3, dtype=base_cell.dtype, device=device) + strain
                    )
                    system["positions"] = base_positions @ deformation.T
                    system["cell"] = base_cell @ deformation.T
                    energies.append(float(_evaluate(system)[0]))
                numerical[row, column] = (energies[0] - energies[1]) / (2.0 * step)
        system["positions"], system["cell"] = base_positions, base_cell
        np.testing.assert_allclose(
            analytic, numerical, atol=1e-6 * np.abs(numerical).max()
        )


def _decomposition_view(parameters):
    """Adapt a parameter object back to what the NumPy harness expects."""

    class _View:
        species = None
        eigs = parameters.eigs.cpu().numpy()
        v_q = parameters.v_q.cpu().numpy()
        cnref = parameters.cnref.cpu().numpy()
        species_map = parameters.species_map.cpu().numpy()
        n_species = parameters.n_species
        rank = parameters.rank

    return _View()


@pytest.mark.gpu
class TestNeighbourFormats:
    """Both neighbour representations, and the validation around them."""

    def test_dense_and_csr_agree(self):
        """The two formats describe the same neighbourhood and give the same answer."""
        system = _system("cuda:0")
        csr = _evaluate(system)
        dense = _evaluate(
            system,
            neighbor_list=None,
            neighbor_ptr=None,
            unit_shifts=None,
            neighbor_matrix=system["neighbor_matrix"],
            neighbor_matrix_shifts=system["neighbor_matrix_shifts"],
        )
        np.testing.assert_allclose(
            csr[0].cpu().numpy(), dense[0].cpu().numpy(), rtol=1e-12
        )
        np.testing.assert_allclose(
            csr[1].cpu().numpy(),
            dense[1].cpu().numpy(),
            atol=1e-11 * float(csr[1].abs().max()),
        )

    def test_rejects_both_formats(self):
        """Supplying both neighbour formats is an error."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="Cannot provide both"):
            _evaluate(system, neighbor_matrix=system["neighbor_matrix"])

    def test_rejects_neither_format(self):
        """Supplying no neighbour format is an error."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="Must provide either"):
            _evaluate(system, neighbor_list=None, neighbor_ptr=None, unit_shifts=None)

    def test_rejects_mismatched_shifts(self):
        """Each format needs its own shift representation."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="unit_shifts is for neighbor_list"):
            _evaluate(
                system,
                neighbor_list=None,
                neighbor_ptr=None,
                neighbor_matrix=system["neighbor_matrix"],
                neighbor_matrix_shifts=system["neighbor_matrix_shifts"],
                unit_shifts=system["unit_shifts"],
            )

    def test_requires_shifts(self):
        """Periodic images are mandatory, since the method is periodic."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="unit_shifts is required"):
            _evaluate(system, unit_shifts=None)


@pytest.mark.gpu
class TestMeshAndUnits:
    """Mesh sizing and the unit contract, both of which fail silently if left implicit."""

    def test_every_spline_order_reaches_the_same_energy(self):
        """The lattice sum does not depend on the interpolation order used to reach it.

        The B-spline attenuation factors are order-dependent and an odd order places its
        stencil differently from an even one. A wrong modulus for a single order would still
        give a finite, plausible, self-consistent answer -- it would just converge somewhere
        else -- so the orders have to be checked against each other rather than themselves.

        Accuracy at a fixed mesh must also improve with order, which is the property that
        makes a low order a cost/accuracy trade rather than a mistake.
        """
        system = _system("cuda:0")
        fine = (96, 96, 96)
        reference = _evaluate(system, mesh_dimensions=fine, spline_order=6)[0].item()
        errors = {
            order: abs(
                _evaluate(system, mesh_dimensions=fine, spline_order=order)[0].item()
                - reference
            )
            / abs(reference)
            for order in (2, 3, 4, 5)
        }
        # Every order lands on the same number; order 2 is linear interpolation and gets
        # there far more slowly, and order 3 is noticeably noisier than the even orders.
        assert errors[2] < 1e-3, errors
        assert errors[3] < 1e-5, errors
        assert errors[4] < 1e-7, errors
        assert errors[5] < 1e-8, errors
        ordered = [errors[o] for o in (2, 3, 4, 5)]
        assert ordered == sorted(ordered, reverse=True), (
            f"accuracy should improve with spline order, got {errors}"
        )

    def test_refining_the_mesh_improves_every_spline_order(self):
        """A stencil offset would leave a residual the mesh cannot reduce.

        Comparing one order against another at a single mesh cannot tell a constant offset
        from ordinary discretisation error; only refining can.
        """
        system = _system("cuda:0")
        reference = _evaluate(system, mesh_dimensions=(128, 128, 128), spline_order=6)[
            0
        ].item()
        for order in (2, 4, 5):
            coarse, fine = (
                abs(
                    _evaluate(system, mesh_dimensions=(m, m, m), spline_order=order)[
                        0
                    ].item()
                    - reference
                )
                / abs(reference)
                for m in (24, 96)
            )
            assert fine < coarse, f"order {order} not converging: {coarse} -> {fine}"

    def test_requires_exactly_one_mesh_option(self):
        """Neither or both of the two ways to size the mesh is an error.

        There is no accuracy-based estimator to fall back on, so guessing would be worse
        than refusing.
        """
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="exactly one of mesh_dimensions"):
            _evaluate(system, mesh_dimensions=None)
        with pytest.raises(ValueError, match="exactly one of mesh_dimensions"):
            _evaluate(system, mesh_spacing=0.3)

    def test_mesh_spacing_sizes_from_the_cell(self):
        """A spacing gives the same answer as the dimensions it implies."""
        system = _system("cuda:0")
        spacing = 9.0 / 32.0
        by_spacing = _evaluate(system, mesh_dimensions=None, mesh_spacing=spacing)
        by_dimensions = _evaluate(system)
        np.testing.assert_allclose(
            by_spacing[0].cpu().numpy(), by_dimensions[0].cpu().numpy(), rtol=1e-12
        )

    def test_rejects_bad_mesh_arguments(self):
        """Degenerate mesh requests are rejected rather than clamped."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="three positive integers"):
            _evaluate(system, mesh_dimensions=(0, 8, 8))
        with pytest.raises(ValueError, match="mesh_spacing must be positive"):
            _evaluate(system, mesh_dimensions=None, mesh_spacing=-1.0)

    @pytest.mark.parametrize("scale", [1.5, 3.0])
    def test_energy_is_invariant_under_a_consistent_unit_change(self, scale):
        """Restating the same physical system in another length unit changes nothing.

        This is the check that a unit mistake would fail. Every dimensioned quantity has to
        move together: with ``[C6] = energy * length**6`` and
        ``R0 = a1 * sqrt(3 * sqrt_q_A * sqrt_q_B) + a2``, the length-carrying quantities are
        ``positions``, ``cell``, ``rcov``, ``r_cut``, ``sqrt_q`` and ``a2``, while ``eigs``
        carries ``length**6`` and ``s6``, ``s8`` and ``a1`` are dimensionless.

        Agreement is close but not exact because the counting function carries one absolute
        regulariser, which is the single scale-dependent constant in the method.
        """
        device = "cuda:0"
        base = _system(device)
        expected = float(_evaluate(base)[0])

        parameters = base["params"]
        rescaled = _system(device)
        rescaled["positions"] = base["positions"] * scale
        rescaled["cell"] = base["cell"] * scale
        rescaled["params"] = FourierD3Parameters(
            rcov=parameters.rcov * scale,
            sqrt_q=parameters.sqrt_q * scale,
            cnref=parameters.cnref,
            v_q=parameters.v_q,
            eigs=parameters.eigs * scale**6,
            species_map=parameters.species_map,
            max_relative_error=parameters.max_relative_error,
        )
        actual = float(
            _evaluate(
                rescaled,
                r_cut=R_CUT * scale,
                a1=DAMPING["a1"],
                a2=DAMPING["a2"] * scale,
                s8=DAMPING["s8"],
                s6=DAMPING["s6"],
            )[0]
        )
        assert abs(actual - expected) < 1e-9 * abs(expected)

    def test_rejects_uncovered_species(self):
        """An atom the decomposition does not cover is reported, not silently zeroed."""
        system = _system("cuda:0")
        numbers = system["numbers"].clone()
        numbers[0] = 7
        system["numbers"] = numbers
        with pytest.raises(ValueError, match="not covered by fd3_params"):
            _evaluate(system)


@pytest.mark.gpu
class TestParameters:
    """The parameter object."""

    def test_carries_no_damping_parameters(self):
        """Damping is supplied per call, so a stored copy cannot go stale.

        Keeping a derived self-energy term alongside call-time damping would let a caller mix
        one functional's reciprocal sum with another's self-energy and get a plausible but
        wrong number.
        """
        fields = set(FourierD3Parameters.__dataclass_fields__)
        assert not (fields & {"s6", "s8", "a1", "a2", "selfcont", "phi_zero"})

    def test_changing_damping_changes_the_energy(self):
        """The same parameter object under two functionals gives two answers."""
        system = _system("cuda:0")
        first = float(_evaluate(system)[0])
        second = float(_evaluate(system, a1=0.35, a2=5.0, s8=1.2)[0])
        assert abs(second - first) > 1e-6 * abs(first)

    def test_reports_its_truncation_error(self):
        """The achieved reconstruction error is available to the caller."""
        system = _system("cuda:0")
        assert 0.0 <= system["params"].max_relative_error < 1e-3

    def test_to_moves_device_and_dtype(self):
        """``to`` converts the floating fields and leaves the channel map integral."""
        system = _system("cuda:0")
        moved = system["params"].to(device="cpu", dtype=torch.float32)
        assert moved.rcov.device.type == "cpu"
        assert moved.eigs.dtype == torch.float32
        assert moved.species_map.dtype == torch.int32

    def test_rejects_inconsistent_shapes(self):
        """Mismatched factor shapes are caught at construction."""
        system = _system("cuda:0")
        parameters = system["params"]
        with pytest.raises(ValueError, match="eigs has rank"):
            FourierD3Parameters(
                rcov=parameters.rcov,
                sqrt_q=parameters.sqrt_q,
                cnref=parameters.cnref,
                v_q=parameters.v_q,
                eigs=parameters.eigs[:-1],
                species_map=parameters.species_map,
                max_relative_error=0.0,
            )

    def test_rejects_mixed_devices(self):
        """All parameter tensors must live together."""
        system = _system("cuda:0")
        parameters = system["params"]
        with pytest.raises(ValueError, match="must share one device"):
            FourierD3Parameters(
                rcov=parameters.rcov.cpu(),
                sqrt_q=parameters.sqrt_q,
                cnref=parameters.cnref,
                v_q=parameters.v_q,
                eigs=parameters.eigs,
                species_map=parameters.species_map,
                max_relative_error=0.0,
            )


@pytest.mark.gpu
class TestPrecision:
    """Both floating precisions are dispatched."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_outputs_follow_the_input_dtype(self, dtype):
        """Energy, forces and virial come back in the precision they went in with."""
        system = _system("cuda:0", dtype=dtype)
        energy, forces, virial = _evaluate(system, compute_virial=True)
        assert energy.dtype == dtype
        assert forces.dtype == dtype
        assert virial.dtype == dtype
        assert torch.isfinite(energy).all()

    def test_single_and_double_agree_to_single_precision(self):
        """The float32 path tracks the float64 one to its own accuracy."""
        double = _system("cuda:0", dtype=torch.float64)
        single = _system("cuda:0", dtype=torch.float32)
        reference = float(_evaluate(double)[0])
        assert abs(float(_evaluate(single)[0]) - reference) < 1e-4 * abs(reference)


@pytest.mark.gpu
class TestTorchCompile:
    """The op has to survive tracing, and do so without falling out of the graph."""

    @staticmethod
    def _callable(system):
        """Close over everything but the positions, as an MD step would."""

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

        return evaluate

    def test_compiled_matches_eager(self):
        """Compilation does not change the result."""
        system = _system("cuda:0")
        evaluate = self._callable(system)
        eager_energy, eager_forces = evaluate(system["positions"])
        energy, forces = torch.compile(evaluate)(system["positions"])
        np.testing.assert_allclose(
            energy.cpu().numpy(), eager_energy.cpu().numpy(), rtol=1e-12
        )
        np.testing.assert_allclose(
            forces.cpu().numpy(),
            eager_forces.cpu().numpy(),
            atol=1e-12 * float(eager_forces.abs().max()),
        )

    def test_traces_without_graph_breaks(self):
        """No graph breaks.

        A break here would mean a device synchronisation inside the molecular-dynamics step
        this op exists to make cheap, which is the reason the species-coverage check is
        skipped while tracing.
        """
        import torch._dynamo as dynamo

        system = _system("cuda:0")
        explanation = dynamo.explain(self._callable(system))(system["positions"])
        assert explanation.graph_break_count == 0

    def test_repeated_calls_are_stable(self):
        """Calling the compiled function repeatedly keeps giving the same answer.

        Output buffers are freshly allocated and zeroed each call; a stale-buffer bug would
        show up as drift here.
        """
        system = _system("cuda:0")
        compiled = torch.compile(self._callable(system))
        first = float(compiled(system["positions"])[0])
        for _ in range(3):
            assert abs(float(compiled(system["positions"])[0]) - first) < 1e-12 * abs(
                first
            )


@pytest.mark.gpu
class TestPrecomputedSetup:
    """Cell- and mesh-derived quantities reused across steps."""

    def test_matches_computing_them_inline(self):
        """Supplying the setup gives the same answer as letting the call derive it."""
        system = _system("cuda:0")
        setup = FourierD3Setup.build(system["cell"], system["params"].n_species, MESH)
        inline = _evaluate(system)
        reused = _evaluate(system, setup=setup)
        np.testing.assert_allclose(
            reused[0].cpu().numpy(), inline[0].cpu().numpy(), rtol=1e-12
        )
        np.testing.assert_allclose(
            reused[1].cpu().numpy(),
            inline[1].cpu().numpy(),
            atol=1e-12 * float(inline[1].abs().max()),
        )

    def test_enables_cuda_graph_capture(self):
        """A CUDA graph can be captured only when the setup is precomputed.

        ``torch.linalg.inv`` cannot be recorded into a graph, so deriving the cell inverse
        inside the call makes ``torch.compile(mode="reduce-overhead")`` fail. This is the
        test that pins that down; without it the failure would only appear to a user trying
        to speed up an MD loop.
        """
        system = _system("cuda:0")
        setup = FourierD3Setup.build(system["cell"], system["params"].n_species, MESH)

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
                setup=setup,
                **DAMPING,
            )

        expected = evaluate(system["positions"])
        compiled = torch.compile(evaluate, mode="reduce-overhead")
        for _ in range(3):
            actual = compiled(system["positions"])
        np.testing.assert_allclose(
            actual[0].cpu().numpy(), expected[0].cpu().numpy(), rtol=1e-10
        )

    def test_records_what_it_was_built_for(self):
        """The setup carries its mesh and spline order, so the call cannot disagree."""
        system = _system("cuda:0")
        setup = FourierD3Setup.build(
            system["cell"], system["params"].n_species, (16, 16, 16), spline_order=5
        )
        assert setup.mesh_dimensions == (16, 16, 16)
        assert setup.spline_order == 5
        # The call follows the setup rather than its own arguments.
        result = _evaluate(system, setup=setup, mesh_dimensions=MESH)
        coarse = _evaluate(system, mesh_dimensions=(16, 16, 16), spline_order=5)
        np.testing.assert_allclose(
            result[0].cpu().numpy(), coarse[0].cpu().numpy(), rtol=1e-12
        )
