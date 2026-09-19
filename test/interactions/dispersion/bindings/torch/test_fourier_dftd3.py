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
from nvalchemiops.torch.interactions.dispersion import (  # noqa: E402
    _fourier_dftd3 as _fd3,
)
from nvalchemiops.torch.interactions.dispersion._fourier_dftd3 import (  # noqa: E402
    _resolve_mesh,
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


def _halve(system):
    """Keep one direction of each pair, the shape a ``half_fill=True`` builder produces."""
    sources = system["neighbor_list"][0].cpu().numpy()
    targets = system["neighbor_list"][1].cpu().numpy()
    shifts = system["unit_shifts"].cpu().numpy()
    # Between distinct atoms keep the ascending direction; an atom paired with its own
    # periodic image appears as (i, i, s) and (i, i, -s), so break that tie on the shift.
    lexicographic = np.where(
        shifts[:, 0] != 0,
        shifts[:, 0],
        np.where(shifts[:, 1] != 0, shifts[:, 1], shifts[:, 2]),
    )
    keep = (sources < targets) | ((sources == targets) & (lexicographic > 0))
    sources, targets, shifts = sources[keep], targets[keep], shifts[keep]
    order = np.argsort(sources, kind="stable")
    sources, targets, shifts = sources[order], targets[order], shifts[order]
    n_atoms = system["n_atoms"]
    pointer = np.zeros(n_atoms + 1, dtype=np.int32)
    np.add.at(pointer, sources + 1, 1)
    pointer = np.cumsum(pointer).astype(np.int32)

    def tensor(array, dtype):
        return torch.as_tensor(
            np.ascontiguousarray(array), dtype=dtype, device=system["positions"].device
        )

    matrix, matrix_shifts = _to_dense(targets, pointer, shifts, n_atoms)
    return {
        "neighbor_list": torch.stack(
            [tensor(sources, torch.int32), tensor(targets, torch.int32)]
        ),
        "neighbor_ptr": tensor(pointer, torch.int32),
        "unit_shifts": tensor(shifts, torch.int32),
        "neighbor_matrix": tensor(matrix, torch.int32),
        "neighbor_matrix_shifts": tensor(matrix_shifts, torch.int32),
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
        """The returned virial is minus the strain derivative of the returned energy."""
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
                # Negated: conventions.md defines the virial as -dE/du, so the
                # finite difference has to carry the same sign to compare against.
                numerical[row, column] = -(energies[0] - energies[1]) / (2.0 * step)
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

    def test_forces_match_finite_differences(self):
        """Cartesian forces are the gradient of the energy in a skewed cell."""
        system = _system("cuda:0", cell=TRICLINIC)
        analytic = _evaluate(system)[1].cpu().numpy()
        base = system["positions"].clone()
        step = 1e-6
        numerical = np.zeros_like(analytic)
        for atom in range(len(base)):
            for axis in range(3):
                shifted = []
                for sign in (1.0, -1.0):
                    moved = base.clone()
                    moved[atom, axis] += sign * step
                    shifted.append(self._energy(system, positions=moved))
                numerical[atom, axis] = -(shifted[0] - shifted[1]) / (2.0 * step)
        scale = np.abs(numerical).max()
        assert scale > 0.0
        np.testing.assert_allclose(analytic, numerical, atol=1e-6 * scale)

    def test_virial_matches_finite_strain(self):
        r"""All six independent strain derivatives, in the repository convention.

        ``conventions.md`` defines :math:`W = -\partial E/\partial u` with
        :math:`R' = R(I+u)` and :math:`C' = C(I+u)`, which is the recipe applied here.
        """
        system = _system("cuda:0", cell=TRICLINIC)
        analytic = _evaluate(system, compute_virial=True)[2][0].cpu().numpy()
        base_positions = system["positions"].clone()
        base_cell = system["cell"].clone()
        identity = torch.eye(3, dtype=base_cell.dtype, device=base_cell.device)
        step = 1e-6
        for row, column in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
            shifted = []
            for sign in (1.0, -1.0):
                displacement = torch.zeros_like(identity)
                # Symmetric off-diagonal perturbation: the energy depends on the symmetric
                # part of u to first order, so this is the derivative that is defined.
                displacement[row, column] += sign * step / 2.0
                displacement[column, row] += sign * step / 2.0
                if row == column:
                    displacement[row, column] = sign * step
                deformation = identity + displacement
                shifted.append(
                    self._energy(
                        system,
                        positions=base_positions @ deformation,
                        cell=base_cell @ deformation,
                    )
                )
            numerical = -(shifted[0] - shifted[1]) / (2.0 * step)
            assert abs(analytic[row, column] - numerical) < 1e-6 * max(
                abs(numerical), np.abs(analytic).max()
            ), f"component ({row}, {column}): {analytic[row, column]} vs {numerical}"

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
            fd3_params=system["params"],
            cell=system["cell"].expand(num_systems, 3, 3),
            r_cut=R_CUT,
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
                fd3_params=system["params"],
                cell=system["cell"],
                r_cut=R_CUT,
                neighbor_list=torch.zeros(2, 0, dtype=torch.int32, device=device),
                neighbor_ptr=torch.zeros(1, dtype=torch.int32, device=device),
                unit_shifts=torch.zeros(0, 3, dtype=torch.int32, device=device),
                **bad_mesh,
            )

    def test_does_not_allocate_the_mesh(self):
        """The mesh is the largest allocation in the call and nothing would be spread onto it.

        Guards against the early return being removed: without it this allocates
        ``num_systems * n_species * rank`` slabs of the full mesh volume.
        """
        system = _system("cuda:0")
        self._call(system, 1)  # warm any lazy allocator state first
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        self._call(system, 1)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
        mesh_bytes = (
            MESH[0]
            * MESH[1]
            * MESH[2]
            * system["params"].n_species
            * system["params"].rank
            * system["positions"].element_size()
        )
        assert peak < mesh_bytes / 8, f"allocated {peak} bytes; a mesh is {mesh_bytes}"


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

    def test_rejects_a_half_filled_list(self):
        """A half-filled list silently loses coordination, so it must not be accepted.

        Each atom's coordination number is accumulated from its own row alone, with the
        reverse edge walked by the other atom. Half of the contributions simply go missing,
        which shifts the energy and leaves the forces non-conservative rather than raising.
        """
        system = _system("cuda:0", box=5.0)
        half = _halve(system)
        with pytest.raises(ValueError, match="both directions of every pair"):
            _evaluate(
                system,
                neighbor_list=half["neighbor_list"],
                neighbor_ptr=half["neighbor_ptr"],
                unit_shifts=half["unit_shifts"],
            )

    def test_rejects_a_half_filled_matrix(self):
        """The dense format carries the same requirement."""
        system = _system("cuda:0", box=5.0)
        half = _halve(system)
        with pytest.raises(ValueError, match="both directions of every pair"):
            _evaluate(
                system,
                neighbor_list=None,
                neighbor_ptr=None,
                unit_shifts=None,
                neighbor_matrix=half["neighbor_matrix"],
                neighbor_matrix_shifts=half["neighbor_matrix_shifts"],
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

    def test_an_uncovered_real_element_is_still_rejected(self):
        """Relaxing the check for padding must not relax it for a missing species."""
        system = _system("cuda:0")
        numbers = system["numbers"].clone()
        # Nitrogen: inside the table's extent, but not one of the decomposed species. A
        # number beyond the table would fail on the lookup itself rather than on the check.
        numbers[0] = 7
        rejected = dict(system)
        rejected["numbers"] = numbers
        with pytest.raises(ValueError, match="not covered by fd3_params"):
            _evaluate(rejected)


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

    @pytest.mark.parametrize("mesh_size", [1, 2, 3])
    def test_a_mesh_shorter_than_the_stencil_is_refused(self, mesh_size):
        """The order-4 stencil wraps onto a shorter axis and visits a node twice.

        Positive is not sufficient: the interpolation stops being the B-spline that the
        gather differentiates. Without the check this surfaced as a tensor-size error from
        the spline moduli, which says nothing about the cause.
        """
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="at least"):
            _evaluate(system, mesh_dimensions=(mesh_size,) * 3, spline_order=4)

    def test_a_mesh_equal_to_the_stencil_is_allowed(self):
        """At equality every stencil point still lands on its own node."""
        system = _system("cuda:0")
        energy = _evaluate(system, mesh_dimensions=(4, 4, 4), spline_order=4)[0]
        assert torch.isfinite(energy).all()

    def test_a_spacing_too_coarse_for_the_stencil_is_refused(self):
        """The same minimum applies however the mesh was arrived at."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="mesh_spacing"):
            _evaluate(system, mesh_dimensions=None, mesh_spacing=100.0, spline_order=4)

    def test_the_setup_applies_the_same_minimum(self):
        """A setup is the third way to arrive at a mesh, and is held to the same rule."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="at least"):
            FourierD3Setup.build(
                system["cell"], system["params"].n_species, (2, 2, 2), spline_order=4
            )

    def test_explicit_dimensions_are_used_exactly(self):
        """A number the caller chose is not second-guessed, even when it transforms badly."""
        system = _system("cuda:0")
        setup = FourierD3Setup.build(
            system["cell"], system["params"].n_species, (127, 127, 127)
        )
        assert setup.mesh_dimensions == (127, 127, 127)

    def test_a_spacing_derived_mesh_transforms_well(self):
        """An automatic mesh is rounded up to factors of 2, 3, 5 and 7.

        cuFFT falls back to Bluestein's algorithm otherwise; a prime edge measured 6.7x
        slower than a nearby smooth one on the same transform.
        """
        system = _system("cuda:0")
        for spacing in (0.07, 0.0709, 0.0711, 0.073, 0.11, 0.37):
            mesh = _resolve_mesh(None, spacing, system["cell"].reshape(-1, 3, 3), 4)
            for size in mesh:
                remainder = size
                for prime in (2, 3, 5, 7):
                    while remainder % prime == 0:
                        remainder //= prime
                assert remainder == 1, f"spacing {spacing} gave {mesh}"

    def test_rounding_never_coarsens(self):
        """Rounding goes up, so the mesh is never sparser than the spacing asked for."""
        cell = torch.eye(3, dtype=torch.float64, device="cuda:0") * 9.0
        for spacing in (0.05, 0.0707, 0.09, 0.13, 0.5):
            mesh = _resolve_mesh(None, spacing, cell.reshape(-1, 3, 3), 4)
            for size in mesh:
                assert size >= int(np.ceil(9.0 / spacing))

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

    def test_it_lowers_the_peak_mesh_allocation(self):
        """The point of the option: a smaller resident mesh, not just the same answer."""
        system = _system("cuda:0")
        rank = system["params"].rank
        if rank < 2:
            pytest.skip("needs a rank of at least 2 to split")

        def peak(chunk):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _evaluate(system, mesh_dimensions=(48, 48, 48), rank_chunk_size=chunk)
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated()

        assert peak(1) < peak(None)

    def test_only_the_reciprocal_stages_repeat(self):
        """The chain rule walks the whole neighbour list, so it must run once, not per chunk.

        Counting op invocations is what distinguishes "correct" from "correct but doing the
        real-space work ``rank`` times over", which no numerical comparison can catch.
        """
        system = _system("cuda:0")
        counts = {}
        originals = {}
        for name in ("_fd3_gather_op", "_fd3_finalise_op"):
            originals[name] = getattr(_fd3, name)

            def counted(*args, _name=name, **kwargs):
                counts[_name] = counts.get(_name, 0) + 1
                return originals[_name](*args, **kwargs)

            setattr(_fd3, name, counted)
        try:
            _evaluate(system, rank_chunk_size=1)
        finally:
            for name, original in originals.items():
                setattr(_fd3, name, original)

        assert counts["_fd3_gather_op"] == system["params"].rank
        assert counts["_fd3_finalise_op"] == 1

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
