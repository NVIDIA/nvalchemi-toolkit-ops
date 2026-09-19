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

# Imported rather than guarded by a flag, so that the module body cannot name ``torch``
# while it is undefined. Default arguments, class bodies and decorator arguments all run at
# import time, which is before a module-level skipif can fire, so a flag would turn a missing
# optional dependency into a NameError during collection.
torch = pytest.importorskip("torch", reason="PyTorch not installed.")

from nvalchemiops.torch.interactions.dispersion import (  # noqa: E402
    FourierD3Parameters,
    FourierD3Setup,
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


def _system(device, dtype=None, n_atoms=8, box=9.0, seed=0, cell=None):
    """A small periodic cell with its neighbour list in both formats.

    ``dtype`` defaults to ``torch.float64``, resolved on the call rather than written into
    the signature. Default arguments are evaluated when the module body runs, which is
    before pytest can apply the module-level skip, so naming ``torch`` there would raise a
    NameError during collection wherever Torch is not installed instead of skipping.
    """
    dtype = torch.float64 if dtype is None else dtype
    rng = np.random.default_rng(seed)
    c6ab, cn_ref, species = _reference_tables()
    max_z = c6ab.shape[0]
    rcov = np.zeros(max_z)
    rcov[[1, 6, 8]] = [0.6, 1.2, 1.1]
    r4r2 = np.zeros(max_z)
    r4r2[[1, 6, 8]] = [1.0, 1.4, 1.2]

    # A cell may be supplied to exercise a skewed lattice; positions are then drawn in
    # fractional coordinates so the atoms sit inside it.
    if cell is None:
        cell = np.eye(3) * box
        positions = rng.uniform(0.0, box, (n_atoms, 3))
    else:
        cell = np.asarray(cell, dtype=np.float64)
        positions = rng.uniform(0.0, 1.0, (n_atoms, 3)) @ cell
    numbers = rng.choice(species, n_atoms)
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


def _batched(systems):
    """Concatenate single-system dictionaries into one batch, in both neighbour formats.

    The systems are expected to have different cells. A batch whose cells are identical
    cannot detect a shift conversion that uses the wrong one, which is the whole point of
    exercising the bindings here rather than only at the Warp layer.
    """
    device = systems[0]["positions"].device
    counts = [system["n_atoms"] for system in systems]
    offsets = np.cumsum([0] + counts[:-1]).astype(np.int64)
    total = int(sum(counts))

    batch_idx = torch.cat(
        [
            torch.full((count,), index, dtype=torch.int32, device=device)
            for index, count in enumerate(counts)
        ]
    )

    sources, targets, unit_shifts = [], [], []
    pointer = [torch.zeros(1, dtype=torch.int32, device=device)]
    edges_so_far = 0
    for system, offset in zip(systems, offsets):
        sources.append(system["neighbor_list"][0] + int(offset))
        targets.append(system["neighbor_list"][1] + int(offset))
        unit_shifts.append(system["unit_shifts"])
        pointer.append(system["neighbor_ptr"][1:] + edges_so_far)
        edges_so_far += int(system["neighbor_ptr"][-1])

    # Dense rows are padded to a common width, and padding must point past every atom.
    width = max(int(system["neighbor_matrix"].shape[1]) for system in systems)
    matrices, matrix_shifts = [], []
    for system, offset in zip(systems, offsets):
        own = system["neighbor_matrix"]
        matrix = torch.full(
            (system["n_atoms"], width), total, dtype=torch.int32, device=device
        )
        shifts = torch.zeros(
            (system["n_atoms"], width, 3), dtype=torch.int32, device=device
        )
        padded = own >= system["n_atoms"]
        matrix[:, : own.shape[1]] = torch.where(
            padded, torch.full_like(own, total), own + int(offset)
        )
        shifts[:, : own.shape[1]] = system["neighbor_matrix_shifts"]
        matrices.append(matrix)
        matrix_shifts.append(shifts)

    return {
        "positions": torch.cat([system["positions"] for system in systems]),
        "numbers": torch.cat([system["numbers"] for system in systems]),
        "cell": torch.stack([system["cell"] for system in systems]),
        "params": systems[0]["params"],
        "batch_idx": batch_idx,
        "num_systems": len(systems),
        "neighbor_list": torch.stack([torch.cat(sources), torch.cat(targets)]),
        "neighbor_ptr": torch.cat(pointer),
        "unit_shifts": torch.cat(unit_shifts),
        "neighbor_matrix": torch.cat(matrices),
        "neighbor_matrix_shifts": torch.cat(matrix_shifts),
        "fill_value": total,
        "n_atoms": total,
    }


def _evaluate_at(system, positions=None, cell=None, **kwargs):
    """Evaluate with the positions or cell replaced, leaving the neighbour list alone.

    The list is built for the undeformed configuration and reused: a finite difference has to
    hold the neighbour topology fixed, or it measures the list changing rather than the
    energy.
    """
    if cell is not None:
        kwargs["cell"] = cell
    replaced = dict(system)
    if positions is not None:
        replaced["positions"] = positions
    return _evaluate(replaced, **kwargs)


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


def _decomposition_view(parameters):
    """Adapt a parameter object back to what the NumPy harness expects."""

    class _View:
        species = None
        eigs = parameters.eigs.cpu().numpy()
        v_q = parameters.v_q.cpu().numpy()
        cn_ref = parameters.cn_ref.cpu().numpy()
        species_map = parameters.species_map.cpu().numpy()
        n_species = parameters.n_species
        rank = parameters.rank

    return _View()


# A genuinely skewed lattice. The inverse of a cubic cell is diagonal and so equal to its own
# transpose, which makes a cubic test blind to a confusion between the two.
TRICLINIC = np.array([[9.0, 0.0, 0.0], [2.6, 8.4, 0.0], [1.8, -2.1, 9.3]])


@pytest.mark.gpu
class TestSkewedCell:
    """The binding's own coordinate transforms, in a cell that can detect them.

    The Warp layer is covered separately. This exercises what the binding adds: the inverse
    cell it builds for the mesh, and the Cartesian image shifts it derives from the cell.
    """

    @staticmethod
    def _energy(system, positions=None, cell=None):
        arguments = {}
        if positions is not None:
            arguments["positions"] = positions
        if cell is not None:
            arguments["cell"] = cell
        return float(_evaluate_at(system, **arguments)[0])

    def test_the_virial_is_symmetric(self):
        """Both contributions are symmetric by construction, so the total must be."""
        system = _system("cuda:0", cell=TRICLINIC)
        virial = _evaluate(system, compute_virial=True)[2][0].cpu().numpy()
        np.testing.assert_allclose(virial, virial.T, atol=1e-12 * np.abs(virial).max())


@pytest.mark.gpu
class TestEmptySystem:
    """Zero atoms, which a padded or filtered batch can produce."""

    @staticmethod
    def _call(system, num_systems, **kwargs):
        device = system["positions"].device
        dtype = system["positions"].dtype
        zeros = torch.zeros(0, 3, dtype=dtype, device=device)
        return fourier_dftd3(
            zeros,
            torch.zeros(0, dtype=torch.int32, device=device),
            **DAMPING,
            fourier_d3_params=system["params"],
            cell=system["cell"].expand(num_systems, 3, 3),
            cutoff=R_CUT,
            mesh_dimensions=MESH,
            num_systems=num_systems,
            neighbor_list=torch.zeros(2, 0, dtype=torch.int32, device=device),
            neighbor_ptr=torch.zeros(1, dtype=torch.int32, device=device),
            unit_shifts=torch.zeros(0, 3, dtype=torch.int32, device=device),
            **kwargs,
        )

    @pytest.mark.parametrize("num_systems", [1, 3])
    def test_returns_zeros_of_the_right_shape(self, num_systems):
        """No atoms means no dispersion, and the outputs still have to be well formed."""
        system = _system("cuda:0")
        energy, forces, virial = self._call(system, num_systems, compute_virial=True)
        assert energy.shape == (num_systems,)
        assert forces.shape == (0, 3)
        assert virial.shape == (num_systems, 3, 3)
        assert float(energy.abs().max()) == 0.0
        assert float(virial.abs().max()) == 0.0

    @pytest.mark.parametrize(
        "bad_mesh",
        [
            pytest.param({}, id="neither"),
            pytest.param({"mesh_dimensions": MESH, "mesh_spacing": 0.3}, id="both"),
            pytest.param({"mesh_dimensions": (2, 2, 2)}, id="below_stencil"),
        ],
    )
    def test_invalid_arguments_are_still_rejected(self, bad_mesh):
        """An empty batch is validated exactly as a populated one.

        Returning before the checks would let a mistake through whenever a batch happened to
        be empty, and surface it only once a later batch contained an atom.
        """
        system = _system("cuda:0")
        device = system["positions"].device
        with pytest.raises(ValueError):
            fourier_dftd3(
                torch.zeros(0, 3, dtype=system["positions"].dtype, device=device),
                torch.zeros(0, dtype=torch.int32, device=device),
                **DAMPING,
                fourier_d3_params=system["params"],
                cell=system["cell"],
                cutoff=R_CUT,
                neighbor_list=torch.zeros(2, 0, dtype=torch.int32, device=device),
                neighbor_ptr=torch.zeros(1, dtype=torch.int32, device=device),
                unit_shifts=torch.zeros(0, 3, dtype=torch.int32, device=device),
                **bad_mesh,
            )


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
class TestPaddingAtoms:
    """Atomic number zero, which the public docstring calls a padding atom.

    The kernels are built for it -- the coordination passes skip such an atom and the mesh
    passes guard on a negative channel -- so the wrapper has to let it through rather than
    rejecting it as an uncovered element.
    """

    @staticmethod
    def _pad(system, count):
        """Append ``count`` padding atoms, with no neighbours of their own."""
        device = system["positions"].device
        n_atoms = system["n_atoms"]
        padded = dict(system)
        padded["positions"] = torch.cat(
            [
                system["positions"],
                torch.zeros(count, 3, dtype=system["positions"].dtype, device=device),
            ]
        )
        padded["numbers"] = torch.cat(
            [system["numbers"], torch.zeros(count, dtype=torch.int32, device=device)]
        )
        # The CSR pointer simply stops advancing: padding atoms own no edges.
        padded["neighbor_ptr"] = torch.cat(
            [system["neighbor_ptr"], system["neighbor_ptr"][-1].repeat(count)]
        )
        padded["n_atoms"] = n_atoms + count
        return padded

    def test_padding_atoms_change_nothing(self):
        """Appending padding must leave the energy, forces and virial untouched."""
        system = _system("cuda:0")
        plain = _evaluate(system, compute_virial=True)
        padded = _evaluate(self._pad(system, 5), compute_virial=True)
        n_atoms = system["n_atoms"]
        np.testing.assert_allclose(
            padded[0].cpu().numpy(), plain[0].cpu().numpy(), rtol=1e-12
        )
        np.testing.assert_allclose(
            padded[1][:n_atoms].cpu().numpy(), plain[1].cpu().numpy(), atol=1e-12
        )
        np.testing.assert_allclose(
            padded[2].cpu().numpy(), plain[2].cpu().numpy(), atol=1e-12
        )

    def test_padding_atoms_feel_no_force(self):
        """A padding atom is not a particle, so nothing may push it."""
        system = _system("cuda:0")
        n_atoms = system["n_atoms"]
        forces = _evaluate(self._pad(system, 5))[1]
        assert float(forces[n_atoms:].abs().max()) == 0.0


@pytest.mark.gpu
class TestBatching:
    """Several systems in one call.

    The bindings build the Cartesian shifts themselves, so the Warp-layer batching tests do
    not cover this; the cells must differ for the coverage to mean anything.
    """

    @staticmethod
    def _systems():
        # Both boxes must be small enough relative to R_CUT to have periodic neighbours
        # well inside the cutoff. At box 13 there are none at all, and a system whose image
        # shifts are all zero cannot detect which cell they were multiplied by.
        return [
            _system("cuda:0", box=5.0, seed=0),
            _system("cuda:0", box=7.0, seed=1),
        ]

    @staticmethod
    def _dense(arguments, system):
        """Swap the CSR arguments for the dense matrix ones."""
        arguments.update(
            neighbor_list=None,
            neighbor_ptr=None,
            unit_shifts=None,
            neighbor_matrix=system["neighbor_matrix"],
            neighbor_matrix_shifts=system["neighbor_matrix_shifts"],
            fill_value=system.get("fill_value"),
        )
        return arguments

    @pytest.mark.parametrize("dense", [False, True])
    def test_each_system_keeps_its_own_cell(self, dense):
        """A system's periodic images must be built from its own lattice.

        Converting every image shift with the first system's cell leaves systems after the
        first with neighbours in the wrong places, which corrupts their coordination numbers
        and so their energies, forces and virial. It is invisible in a batch of identical
        cells.
        """
        systems = self._systems()
        batch = _batched(systems)
        extra = {"compute_virial": True}
        together = _evaluate(
            batch,
            batch_idx=batch["batch_idx"],
            num_systems=batch["num_systems"],
            **(self._dense(dict(extra), batch) if dense else extra),
        )
        start = 0
        for index, system in enumerate(systems):
            alone = _evaluate(
                system, **(self._dense(dict(extra), system) if dense else extra)
            )
            stop = start + system["n_atoms"]
            np.testing.assert_allclose(
                together[0][index].item(), alone[0][0].item(), rtol=1e-11
            )
            np.testing.assert_allclose(
                together[1][start:stop].cpu().numpy(),
                alone[1].cpu().numpy(),
                atol=1e-11 * float(alone[1].abs().max()),
            )
            np.testing.assert_allclose(
                together[2][index].cpu().numpy(),
                alone[2][0].cpu().numpy(),
                atol=1e-11 * float(alone[2].abs().max()),
            )
            start = stop


@pytest.mark.gpu
class TestMeshAndUnits:
    """Mesh sizing and the unit contract, both of which fail silently if left implicit."""

    def test_the_setup_applies_the_same_minimum(self):
        """A setup is the third way to arrive at a mesh, and is held to the same rule."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="at least"):
            FourierD3Setup.build(
                system["cell"], system["params"].n_species, (2, 2, 2), spline_order=4
            )

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
        ``positions``, ``cell``, ``rcov``, ``cutoff``, ``sqrt_q`` and ``a2``, while ``eigs``
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
            cn_ref=parameters.cn_ref,
            v_q=parameters.v_q,
            eigs=parameters.eigs * scale**6,
            species_map=parameters.species_map,
            max_relative_error=parameters.max_relative_error,
        )
        actual = float(
            _evaluate(
                rescaled,
                cutoff=R_CUT * scale,
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
        with pytest.raises(ValueError, match="not covered by fourier_d3_params"):
            _evaluate(system)


@pytest.mark.gpu
class TestParameters:
    """The parameter object."""

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
                cn_ref=parameters.cn_ref,
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
                cn_ref=parameters.cn_ref,
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
class TestDeviceAgreement:
    """The same pipeline on CPU and on CUDA.

    The Warp layer's end-to-end classes are GPU-only, so nothing else runs the mesh passes
    on CPU. That matters because the two devices take different paths through the block
    reductions: a block is a warp on CUDA and a single thread on CPU, and a reduction
    written against the wrong width silences most of the mesh while leaving the forces --
    which come from the cotangent field rather than a reduction -- untouched.
    """

    @pytest.mark.parametrize("mesh", [(16, 16, 16), (32, 32, 32)])
    def test_energy_forces_and_virial_match_between_devices(self, mesh):
        """Every output, not just the forces, has to agree across devices."""
        results = {}
        for device in ("cpu", "cuda:0"):
            system = _system(device)
            results[device] = _evaluate(
                system, mesh_dimensions=mesh, compute_virial=True
            )
        cpu, gpu = results["cpu"], results["cuda:0"]
        np.testing.assert_allclose(
            cpu[0].cpu().numpy(), gpu[0].cpu().numpy(), rtol=1e-11
        )
        np.testing.assert_allclose(
            cpu[1].cpu().numpy(),
            gpu[1].cpu().numpy(),
            atol=1e-11 * float(gpu[1].abs().max()),
        )
        np.testing.assert_allclose(
            cpu[2].cpu().numpy(),
            gpu[2].cpu().numpy(),
            atol=1e-11 * float(gpu[2].abs().max()),
        )


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
                fourier_d3_params=system["params"],
                cell=system["cell"],
                cutoff=R_CUT,
                mesh_dimensions=MESH,
                neighbor_list=system["neighbor_list"],
                neighbor_ptr=system["neighbor_ptr"],
                unit_shifts=system["unit_shifts"],
                **DAMPING,
            )

        return evaluate

    def test_compiled_matches_eager(self):
        """Compilation does not change the result.

        ``fullgraph=True`` because a graph break means a device synchronisation inside the
        molecular-dynamics step this op exists to make cheap; it is also why the
        species-coverage check is skipped while tracing.
        """
        system = _system("cuda:0")
        evaluate = self._callable(system)
        eager_energy, eager_forces = evaluate(system["positions"])
        energy, forces = torch.compile(evaluate, fullgraph=True)(system["positions"])
        np.testing.assert_allclose(
            energy.cpu().numpy(), eager_energy.cpu().numpy(), rtol=1e-12
        )
        np.testing.assert_allclose(
            forces.cpu().numpy(),
            eager_forces.cpu().numpy(),
            atol=1e-12 * float(eager_forces.abs().max()),
        )

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
                fourier_d3_params=system["params"],
                cell=system["cell"],
                cutoff=R_CUT,
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
        """The setup carries its mesh and spline order, and is used when the call omits them."""
        system = _system("cuda:0")
        setup = FourierD3Setup.build(
            system["cell"], system["params"].n_species, (16, 16, 16), spline_order=5
        )
        assert setup.mesh_dimensions == (16, 16, 16)
        assert setup.spline_order == 5
        result = _evaluate(system, setup=setup, mesh_dimensions=None)
        coarse = _evaluate(system, mesh_dimensions=(16, 16, 16), spline_order=5)
        np.testing.assert_allclose(
            result[0].cpu().numpy(), coarse[0].cpu().numpy(), rtol=1e-12
        )

    @pytest.mark.parametrize("order", [-2, 0, 1, 7, 9])
    def test_an_unsupported_spline_order_is_refused(self, order):
        """A setup overrides the call's order, so it has to apply the same range check.

        Without it the bad order reaches the kernels: order 1 returned an energy around
        1e19 and order 7 returned the wrong sign, both without complaint.
        """
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="between 2 and 6"):
            FourierD3Setup.build(
                system["cell"], system["params"].n_species, MESH, spline_order=order
            )

    @pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
    def test_supported_orders_match_the_direct_call(self, order):
        """The guard must not narrow the range that already worked."""
        system = _system("cuda:0")
        setup = FourierD3Setup.build(
            system["cell"], system["params"].n_species, MESH, spline_order=order
        )
        direct = _evaluate(system, spline_order=order)[0]
        reused = _evaluate(system, setup=setup, mesh_dimensions=None)[0]
        np.testing.assert_allclose(
            reused.cpu().numpy(), direct.cpu().numpy(), rtol=1e-12
        )

    def test_it_holds_no_autograd_graph(self):
        """A setup outlives the step that built it, so it must not pin the cell's graph.

        The derived tensors are never differentiated -- the kernels run with
        ``enable_backward=False`` -- so retaining the graph only keeps memory alive.
        """
        system = _system("cuda:0")
        cell = system["cell"].clone().requires_grad_(True)
        setup = FourierD3Setup.build(cell, system["params"].n_species, MESH)
        for field in ("cell_inv_grouped", "volumes", "k_matrix", "cell"):
            tensor = getattr(setup, field)
            assert not tensor.requires_grad, f"{field} retains the graph"
            assert tensor.grad_fn is None, f"{field} retains the graph"

    def test_a_conflicting_mesh_is_an_error(self):
        """Asking for one mesh while handing over a setup built for another is ambiguous.

        Silently following the setup would discard an argument the caller wrote down.
        """
        system = _system("cuda:0")
        setup = FourierD3Setup.build(
            system["cell"], system["params"].n_species, (16, 16, 16)
        )
        with pytest.raises(ValueError, match="built for mesh"):
            _evaluate(system, setup=setup, mesh_dimensions=(32, 32, 32))

    def test_a_spacing_alongside_a_setup_is_refused(self):
        """The setup already fixes the mesh, so a spacing cannot be honoured.

        Resolving it would have to read the cell lengths off the device, which is the cost
        the setup exists to avoid, so it is refused rather than quietly ignored.
        """
        system = _system("cuda:0")
        setup = FourierD3Setup.build(
            system["cell"], system["params"].n_species, (16, 16, 16)
        )
        with pytest.raises(ValueError, match="already fixes the mesh") as raised:
            _evaluate(system, setup=setup, mesh_dimensions=None, mesh_spacing=0.5)
        # The remedy has to be one the caller can actually follow: ``build`` takes
        # ``mesh_dimensions`` only, so pointing them at a ``mesh_spacing`` argument on it
        # would just move the failure one call along.
        assert "mesh_dimensions only" in str(raised.value)

    def test_a_setup_alone_needs_no_mesh_selector(self):
        """The "exactly one of them" rule does not apply once a setup carries the mesh."""
        system = _system("cuda:0")
        setup = FourierD3Setup.build(
            system["cell"], system["params"].n_species, (16, 16, 16)
        )
        with_setup = _evaluate(system, setup=setup, mesh_dimensions=None)
        explicit = _evaluate(system, mesh_dimensions=(16, 16, 16))
        np.testing.assert_allclose(
            with_setup[0].cpu().numpy(), explicit[0].cpu().numpy(), rtol=1e-12
        )

    def test_a_setup_from_another_cell_is_refused(self):
        """The mesh transforms would come from one cell and the image shifts from another.

        That is not a stale answer but an incoherent one, so it cannot be allowed to pass
        quietly.
        """
        system = _system("cuda:0")
        other = FourierD3Setup.build(
            system["cell"] * 1.05, system["params"].n_species, MESH
        )
        with pytest.raises(ValueError, match="different cell"):
            _evaluate(system, setup=other, mesh_dimensions=None)

    def test_an_in_place_cell_change_is_caught(self):
        """The recorded cell must be a snapshot, not a view of the caller's tensor.

        Variable-cell dynamics updates the cell in place. If the setup merely referenced it,
        the record would move with the mutation while the inverse cell, volumes, wave vectors
        and moduli stayed behind, and the comparison would be a tensor against itself. The
        stale setup then passes and the evaluation is badly wrong, not subtly so.
        """
        system = _system("cuda:0")
        cell = system["cell"]
        setup = FourierD3Setup.build(cell, system["params"].n_species, MESH)
        assert setup.cell.data_ptr() != cell.data_ptr()

        cell.mul_(1.10)
        with pytest.raises(ValueError, match="different cell"):
            _evaluate(system, setup=setup, mesh_dimensions=None)

    def test_a_setup_for_a_different_batch_is_refused(self):
        """Shape mismatches are caught without reading any device memory."""
        system = _system("cuda:0")
        batched = FourierD3Setup.build(
            system["cell"].expand(3, 3, 3), system["params"].n_species, MESH
        )
        with pytest.raises(ValueError, match="built for 3 system"):
            _evaluate(system, setup=batched, mesh_dimensions=None)

    def test_a_setup_of_the_wrong_precision_is_refused(self):
        """Reusing a float32 setup for a float64 call would silently mix precisions."""
        system = _system("cuda:0")
        single = FourierD3Setup.build(
            system["cell"].float(), system["params"].n_species, MESH
        )
        with pytest.raises(ValueError, match="Rebuild it for this precision"):
            _evaluate(system, setup=single, mesh_dimensions=None)


@pytest.mark.gpu
class TestEnergyIsNotDifferentiable:
    """The documented contract: forces are an output, not an autograd result.

    The kernels run with ``enable_backward=False`` and no autograd rule is registered on
    top of them, which is what keeps the pipeline capturable into a CUDA graph. These tests
    pin the consequence so the guide cannot drift away from the behaviour.
    """

    def test_energy_is_detached_from_positions(self):
        """No grad_fn, so the energy carries no path back to the inputs."""
        system = _system("cuda:0")
        system["positions"].requires_grad_(True)
        energy = _evaluate(system)[0]
        assert not energy.requires_grad
        assert energy.grad_fn is None

    def test_autograd_raises_rather_than_returning_a_wrong_gradient(self):
        """A silent zero or a wrong gradient would be far worse than an error."""
        system = _system("cuda:0")
        system["positions"].requires_grad_(True)
        energy = _evaluate(system)[0]
        with pytest.raises(RuntimeError, match="does not require grad"):
            torch.autograd.grad(energy.sum(), system["positions"])


@pytest.mark.gpu
class TestRankChunking:
    """Splitting the rank across passes trades memory for extra launches.

    Every stage after the coordination number is a sum over rank slots with no coupling
    between them, so the split must not change the answer.
    """

    @pytest.mark.parametrize("chunk", [1, 2, 3, 8])
    def test_chunked_matches_a_single_pass(self, chunk):
        """Energy, forces and virial all have to survive the split."""
        system = _system("cuda:0")
        whole = _evaluate(system, compute_virial=True)
        split = _evaluate(system, compute_virial=True, rank_chunk_size=chunk)
        for reference, chunked in zip(whole, split):
            np.testing.assert_allclose(
                chunked.cpu().numpy(), reference.cpu().numpy(), rtol=1e-11, atol=1e-13
            )

    def test_a_chunk_larger_than_the_rank_is_one_pass(self):
        """Clamping keeps an oversized request from producing empty trailing chunks."""
        system = _system("cuda:0")
        rank = system["params"].rank
        whole = _evaluate(system)[0]
        split = _evaluate(system, rank_chunk_size=rank * 4)[0]
        np.testing.assert_allclose(split.cpu().numpy(), whole.cpu().numpy(), rtol=1e-12)

    def test_a_non_integer_chunk_is_refused(self):
        """The size decides how many kernels launch, so it cannot be a tensor."""
        system = _system("cuda:0")
        with pytest.raises(TypeError, match="must be an int or None"):
            _evaluate(system, rank_chunk_size=torch.tensor(2))

    def test_a_non_positive_chunk_is_refused(self):
        """Zero would make no progress and loop forever."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="at least 1"):
            _evaluate(system, rank_chunk_size=0)


@pytest.mark.gpu
class TestBatchArgumentValidation:
    """Batch arguments must be checked before they can corrupt the result.

    An out-of-range index sends an atom to a mesh slab owned by no system, and its whole
    contribution disappears: a uniform ``batch_idx`` of 5 against one system returned
    exactly zero energy with no error at all.
    """

    def test_num_systems_must_match_the_cells(self):
        """Otherwise the outputs are padded with zeros for systems that do not exist."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="but cell holds"):
            _evaluate(system, num_systems=3)

    def test_batch_idx_must_cover_every_atom(self):
        """A short batch_idx used to fail deep inside a broadcast."""
        system = _system("cuda:0")
        short = torch.zeros(
            system["positions"].shape[0] // 2, dtype=torch.int32, device="cuda:0"
        )
        with pytest.raises(ValueError, match="entries but there are"):
            _evaluate(system, batch_idx=short)

    @pytest.mark.parametrize("value", [-1, 5])
    def test_batch_idx_must_be_in_range(self, value):
        """Both ends matter: negative and past the last system."""
        system = _system("cuda:0")
        out_of_range = torch.full(
            (system["positions"].shape[0],), value, dtype=torch.int32, device="cuda:0"
        )
        with pytest.raises(ValueError, match="outside"):
            _evaluate(system, batch_idx=out_of_range)

    def test_the_compiled_path_is_guarded_too(self):
        """``reduce-overhead`` is documented, so the check has to survive capture.

        A host read is illegal under graph capture, so the bounds cannot be raised on.
        Instead the indices are clamped, which keeps the kernels from reading the grouped
        cell and the mesh out of range, and the result is poisoned so the clamp cannot pass
        bad input off as a plausible energy.
        """
        system = _system("cuda:0")
        setup = FourierD3Setup.build(system["cell"], system["params"].n_species, MESH)
        n_atoms = system["positions"].shape[0]

        def run(batch):
            return _evaluate(
                system, setup=setup, mesh_dimensions=None, batch_idx=batch
            )[0]

        compiled = torch.compile(run, mode="reduce-overhead")
        valid = torch.zeros(n_atoms, dtype=torch.int32, device="cuda:0")
        for _ in range(3):
            warmed = compiled(valid)
        torch.cuda.synchronize()
        np.testing.assert_allclose(
            warmed.cpu().numpy(), run(valid).cpu().numpy(), rtol=1e-12
        )

        out_of_range = torch.full((n_atoms,), 5, dtype=torch.int32, device="cuda:0")
        poisoned = compiled(out_of_range)
        torch.cuda.synchronize()
        assert np.isnan(poisoned.cpu().numpy()).all()

    def test_a_valid_single_system_batch_is_accepted(self):
        """The guard must not reject the ordinary case it is wrapped around."""
        system = _system("cuda:0")
        explicit = torch.zeros(
            system["positions"].shape[0], dtype=torch.int32, device="cuda:0"
        )
        with_batch = _evaluate(system, batch_idx=explicit, num_systems=1)[0]
        default = _evaluate(system)[0]
        np.testing.assert_allclose(
            with_batch.cpu().numpy(), default.cpu().numpy(), rtol=1e-12
        )


class TestParameterValidation:
    """The container is public, so a malformed bundle must be refused at construction.

    Before these checks an integral ``eigs`` truncated the decomposition to whole numbers, a
    floating ``species_map`` indexed as something else, and a stray leading axis on ``rcov``
    all returned a plausible energy. A two-dimensional ``sqrt_q`` was worse: it reached the
    device and triggered a CUDA assert, which takes the process with it.
    """

    @staticmethod
    def _fields():
        return dict(
            rcov=torch.zeros(5),
            sqrt_q=torch.zeros(2),
            cn_ref=torch.zeros(2, 3),
            v_q=torch.zeros(2, 3, 4),
            eigs=torch.zeros(4),
            species_map=torch.zeros(5, dtype=torch.int32),
            max_relative_error=0.0,
        )

    def test_a_well_formed_bundle_is_accepted(self):
        """The guard must not reject the shape it is built around."""
        params = FourierD3Parameters(**self._fields())
        assert params.rank == 4
        assert params.n_species == 2

    @pytest.mark.parametrize("name", ["rcov", "sqrt_q", "cn_ref", "v_q", "eigs"])
    def test_numeric_fields_must_be_floating(self, name):
        """An integral eigs silently truncates the decomposition."""
        fields = self._fields()
        fields[name] = fields[name].to(torch.int64)
        with pytest.raises(TypeError, match="must be float32 or float64"):
            FourierD3Parameters(**fields)

    def test_species_map_must_be_integral(self):
        """It indexes the reference tables, so a float is not merely imprecise."""
        fields = self._fields()
        fields["species_map"] = fields["species_map"].to(torch.float32)
        with pytest.raises(TypeError, match="must be int32 or int64"):
            FourierD3Parameters(**fields)

    @pytest.mark.parametrize("name", ["rcov", "sqrt_q", "eigs", "species_map"])
    def test_flat_tables_must_be_one_dimensional(self, name):
        """A stray leading axis passes the pairwise shape checks but not the kernels."""
        fields = self._fields()
        fields[name] = fields[name].reshape(1, -1)
        with pytest.raises(ValueError, match="must be 1D"):
            FourierD3Parameters(**fields)
