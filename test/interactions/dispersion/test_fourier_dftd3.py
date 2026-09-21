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

"""Warp-layer tests for FourierD3.

Every kernel is checked against the NumPy reference in
``test/interactions/dispersion/_fourier_reference.py``. The scalar device functions are
exercised through small probe kernels, since Warp functions cannot be called from Python
directly.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from nvalchemiops.interactions.dispersion import _fourier_dftd3 as fd3
from nvalchemiops.interactions.dispersion._c6_decomposition import (
    decompose_c6_reference,
)
from test.interactions.dispersion import _fourier_reference as reference

R_CUT = 4.0


def _warp_types(np_dtype):
    """Scalar and vector Warp dtypes matching a NumPy floating dtype."""
    if np_dtype == np.float64:
        return wp.float64, wp.vec3d
    return wp.float32, wp.vec3f


@wp.kernel(enable_backward=False)
def _probe_counting_kernel(
    distance: wp.array(dtype=wp.float64),
    covalent_distance: wp.float64,
    cutoff: wp.float64,
    value: wp.array(dtype=wp.float64),
    derivative: wp.array(dtype=wp.float64),
):
    """Expose the counting function to Python.

    Thread launch
    -------------
    One thread per sample distance.

    Modifies
    --------
    ``value`` and ``derivative`` at each sample.
    """
    i = wp.tid()
    v, d = fd3._cn_counting(distance[i], covalent_distance, cutoff, True)
    value[i] = v
    derivative[i] = d


@wp.kernel(enable_backward=False)
def _probe_kernel_transform(
    k_norm: wp.array(dtype=wp.float64),
    r0: wp.float64,
    sqrt_q_product: wp.float64,
    s6: wp.float64,
    s8: wp.float64,
    value: wp.array(dtype=wp.float64),
    derivative: wp.array(dtype=wp.float64),
):
    """Expose the reciprocal-space kernel and its radial derivative to Python.

    Thread launch
    -------------
    One thread per sample wave vector.

    Modifies
    --------
    ``value`` and ``derivative`` at each sample.
    """
    i = wp.tid()
    v, d = fd3._reciprocal_kernel(k_norm[i], r0, sqrt_q_product, s6, s8)
    value[i] = v
    derivative[i] = d


def _reference_tables(seed=0):
    """Synthetic D3 reference tables with a realistic ragged reference count."""
    rng = np.random.default_rng(seed)
    max_z, n_ref = 9, 5
    c6ab = np.zeros((max_z + 1, max_z + 1, n_ref, n_ref))
    cn_ref = np.zeros_like(c6ab)
    used = {1: 2, 6: 5, 8: 3}
    coordination = {z: np.linspace(0.0, 3.5, n) for z, n in used.items()}
    factors = {z: rng.normal(size=(n, 3)) for z, n in used.items()}
    for z_i in used:
        for z_j in used:
            c6ab[z_i, z_j, : used[z_i], : used[z_j]] = (
                factors[z_i] @ factors[z_j].T + 20.0
            )
            for p in range(used[z_i]):
                cn_ref[z_i, z_j, p, : used[z_j]] = coordination[z_i][p]
    return c6ab, cn_ref, list(used)


def _neighbour_list(positions, cell, cutoff):
    """Directed neighbour list in CSR form, with both orientations of every pair."""
    n_atoms = len(positions)
    reach = int(np.ceil(cutoff / cell.diagonal().min())) + 1
    offsets = np.arange(-reach, reach + 1)
    lattice = np.stack(
        np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1
    ).reshape(-1, 3)

    sources, targets, shifts = [], [], []
    for translation in lattice:
        delta = positions[None, :, :] + translation @ cell - positions[:, None, :]
        distance = np.linalg.norm(delta, axis=-1)
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j and not translation.any():
                    continue
                if distance[i, j] >= cutoff:
                    continue
                sources.append(i)
                targets.append(j)
                shifts.append(translation)

    order = np.argsort(sources, kind="stable")
    sources = np.asarray(sources)[order]
    targets = np.asarray(targets, dtype=np.int32)[order]
    shifts = np.asarray(shifts)[order]
    pointer = np.zeros(n_atoms + 1, dtype=np.int32)
    for source in sources:
        pointer[source + 1] += 1
    pointer = np.cumsum(pointer).astype(np.int32)
    return targets, pointer, shifts, np.stack([sources, targets], axis=1)


@pytest.fixture(scope="module")
def system():
    """A small periodic cell with three species and its neighbour list."""
    rng = np.random.default_rng(0)
    c6ab, cn_ref, species = _reference_tables()
    decomposition = decompose_c6_reference(c6ab, cn_ref, species)

    n_atoms = 24
    cell = np.eye(3) * 9.0
    positions = rng.uniform(0.0, 9.0, (n_atoms, 3))
    numbers = rng.choice(species, n_atoms)
    rcov = np.zeros(c6ab.shape[0])
    rcov[[1, 6, 8]] = [0.6, 1.2, 1.1]

    targets, pointer, shifts, edges = _neighbour_list(positions, cell, R_CUT)
    return {
        "decomposition": decomposition,
        "positions": positions,
        "numbers": numbers,
        "rcov": rcov,
        "cell": cell,
        "targets": targets,
        "pointer": pointer,
        "shifts": shifts,
        "edges": edges,
        "channels": decomposition.species_map[numbers],
    }


class TestCountingFunction:
    """The modified coordination-number counting function, on device."""

    def test_matches_reference(self, device):
        """Device evaluation agrees with the NumPy reference."""
        distance = np.linspace(0.6, R_CUT - 1e-6, 500)
        covalent, cutoff = 1.8, R_CUT
        value = wp.zeros(len(distance), dtype=wp.float64, device=device)
        derivative = wp.zeros_like(value)
        wp.launch(
            _probe_counting_kernel,
            dim=len(distance),
            inputs=[
                wp.array(distance, dtype=wp.float64, device=device),
                wp.float64(covalent),
                wp.float64(cutoff),
            ],
            outputs=[value, derivative],
            device=device,
        )
        expected = reference.coordination_pair_term(distance, covalent, cutoff)
        np.testing.assert_allclose(value.numpy(), expected, rtol=1e-12, atol=1e-14)

    def test_derivative_matches_finite_differences(self, device):
        """The analytic radial derivative is the derivative of the value it ships with."""
        distance = np.linspace(0.8, R_CUT - 0.05, 300)
        covalent, cutoff = 1.8, R_CUT
        value = wp.zeros(len(distance), dtype=wp.float64, device=device)
        derivative = wp.zeros_like(value)
        wp.launch(
            _probe_counting_kernel,
            dim=len(distance),
            inputs=[
                wp.array(distance, dtype=wp.float64, device=device),
                wp.float64(covalent),
                wp.float64(cutoff),
            ],
            outputs=[value, derivative],
            device=device,
        )
        step = 1e-7
        numerical = (
            reference.coordination_pair_term(distance + step, covalent, cutoff)
            - reference.coordination_pair_term(distance - step, covalent, cutoff)
        ) / (2.0 * step)
        np.testing.assert_allclose(derivative.numpy(), numerical, rtol=1e-5, atol=1e-7)


class TestReciprocalKernel:
    """The closed-form transform and its radial derivative, on device."""

    def test_matches_reference(self, device):
        """Device evaluation agrees with the NumPy reference across both branches."""
        k_norm = np.concatenate(
            [np.logspace(-8.0, -1.1, 50), np.linspace(0.03, 4.0, 450)]
        )
        r0, q_product, s6, s8 = 3.0, 1.4, 1.0, 0.7875
        value = wp.zeros(len(k_norm), dtype=wp.float64, device=device)
        derivative = wp.zeros_like(value)
        wp.launch(
            _probe_kernel_transform,
            dim=len(k_norm),
            inputs=[
                wp.array(k_norm, dtype=wp.float64, device=device),
                wp.float64(r0),
                wp.float64(q_product),
                wp.float64(s6),
                wp.float64(s8),
            ],
            outputs=[value, derivative],
            device=device,
        )
        expected = reference.reciprocal_kernel(k_norm, r0, q_product, s6, s8)
        np.testing.assert_allclose(value.numpy(), expected, rtol=1e-12, atol=1e-14)

    def test_derivative_matches_finite_differences(self, device):
        """``dK/d|k|`` is consistent with ``K``.

        This derivative is the only source of the reciprocal-space virial, and it cannot be
        recovered downstream once the wave vector has left scope.
        """
        k_norm = np.linspace(0.2, 4.0, 400)
        r0, q_product, s6, s8 = 3.0, 1.4, 1.0, 0.7875
        value = wp.zeros(len(k_norm), dtype=wp.float64, device=device)
        derivative = wp.zeros_like(value)
        wp.launch(
            _probe_kernel_transform,
            dim=len(k_norm),
            inputs=[
                wp.array(k_norm, dtype=wp.float64, device=device),
                wp.float64(r0),
                wp.float64(q_product),
                wp.float64(s6),
                wp.float64(s8),
            ],
            outputs=[value, derivative],
            device=device,
        )
        step = 1e-6
        numerical = (
            reference.reciprocal_kernel(k_norm + step, r0, q_product, s6, s8)
            - reference.reciprocal_kernel(k_norm - step, r0, q_product, s6, s8)
        ) / (2.0 * step)
        scale = np.abs(numerical).max()
        np.testing.assert_allclose(
            derivative.numpy(), numerical, rtol=1e-6, atol=1e-8 * scale
        )


class TestCoordinationNumbers:
    """Pass 1: coordination numbers from the neighbour list."""

    def test_counting_function_crosses_half_at_the_covalent_distance(self, device):
        """The counting function is on the same radius scale as ``dftd3``.

        The shipped covalent-radius table already folds in Grimme's 4/3 factor, so a pair
        separated by exactly the sum of its two radii must count as one half. Applying that
        factor a second time would move the half-point to three quarters of this separation
        and inflate every coordination number, which changes the reference weighting and so
        every C6 coefficient. This pins the convention against that.
        """
        rcov = np.zeros(10)
        rcov[[1, 6]] = [0.8, 1.9]
        covalent_distance = rcov[1] + rcov[6]
        positions = np.array([[0.0, 0.0, 0.0], [covalent_distance, 0.0, 0.0]])
        numbers = np.array([1, 6], dtype=np.int32)
        # A directed list holding both orientations of the single pair.
        targets = np.array([1, 0], dtype=np.int32)
        pointer = np.array([0, 1, 2], dtype=np.int32)
        shifts = np.zeros((2, 3))

        coordination = wp.zeros(2, dtype=wp.float64, device=device)
        fd3.fd3_coordination_numbers(
            wp.array(positions, dtype=wp.vec3d, device=device),
            wp.array(numbers, dtype=wp.int32, device=device),
            wp.array(targets, dtype=wp.int32, device=device),
            wp.array(pointer, dtype=wp.int32, device=device),
            wp.array(shifts, dtype=wp.vec3d, device=device),
            wp.array(rcov, dtype=wp.float64, device=device),
            8.0,
            coordination,
            wp.float64,
            device,
        )
        np.testing.assert_allclose(coordination.numpy(), [0.5, 0.5], atol=1e-12)

    @pytest.mark.parametrize(
        "np_dtype, tolerance", [(np.float64, 1e-13), (np.float32, 1e-5)]
    )
    def test_matches_reference(self, device, system, np_dtype, tolerance):
        """Both precisions reproduce the reference coordination numbers."""
        wp_dtype, vec_dtype = _warp_types(np_dtype)
        n_atoms = len(system["positions"])
        coordination = wp.zeros(n_atoms, dtype=wp_dtype, device=device)
        fd3.fd3_coordination_numbers(
            wp.array(
                system["positions"].astype(np_dtype), dtype=vec_dtype, device=device
            ),
            wp.array(system["numbers"].astype(np.int32), dtype=wp.int32, device=device),
            wp.array(system["targets"], dtype=wp.int32, device=device),
            wp.array(system["pointer"], dtype=wp.int32, device=device),
            wp.array(
                (system["shifts"] @ system["cell"]).astype(np_dtype),
                dtype=vec_dtype,
                device=device,
            ),
            wp.array(system["rcov"].astype(np_dtype), dtype=wp_dtype, device=device),
            R_CUT,
            coordination,
            wp_dtype,
            device,
        )
        expected = reference.modified_coordination_number(
            system["positions"],
            system["numbers"],
            system["rcov"],
            system["edges"],
            system["shifts"],
            system["cell"],
            R_CUT,
        )
        assert expected.mean() > 0.5
        np.testing.assert_allclose(coordination.numpy(), expected, atol=tolerance)

    def test_padding_atoms_are_skipped(self, device, system):
        """Atoms with atomic number zero contribute nothing and receive nothing."""
        numbers = system["numbers"].copy()
        numbers[:4] = 0
        n_atoms = len(system["positions"])
        coordination = wp.zeros(n_atoms, dtype=wp.float64, device=device)
        fd3.fd3_coordination_numbers(
            wp.array(system["positions"], dtype=wp.vec3d, device=device),
            wp.array(numbers.astype(np.int32), dtype=wp.int32, device=device),
            wp.array(system["targets"], dtype=wp.int32, device=device),
            wp.array(system["pointer"], dtype=wp.int32, device=device),
            wp.array(system["shifts"] @ system["cell"], dtype=wp.vec3d, device=device),
            wp.array(system["rcov"], dtype=wp.float64, device=device),
            R_CUT,
            coordination,
            wp.float64,
            device,
        )
        np.testing.assert_array_equal(coordination.numpy()[:4], 0.0)

    def test_empty_system_is_a_no_op(self, device):
        """A zero-atom launch returns without touching the device."""
        empty = wp.zeros(0, dtype=wp.float64, device=device)
        fd3.fd3_coordination_numbers(
            wp.zeros(0, dtype=wp.vec3d, device=device),
            wp.zeros(0, dtype=wp.int32, device=device),
            wp.zeros(0, dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.zeros(0, dtype=wp.vec3d, device=device),
            wp.zeros(1, dtype=wp.float64, device=device),
            R_CUT,
            empty,
            wp.float64,
            device,
        )
        assert empty.numpy().size == 0


class TestCoefficients:
    """Pass 2: separable coefficients and their coordination derivative."""

    @pytest.mark.parametrize(
        "np_dtype, tolerance", [(np.float64, 1e-13), (np.float32, 1e-5)]
    )
    def test_matches_reference(self, device, system, np_dtype, tolerance):
        """Coefficients and their derivative agree with the reference in both precisions."""
        wp_dtype, _ = _warp_types(np_dtype)
        decomposition = system["decomposition"]
        coordination = reference.modified_coordination_number(
            system["positions"],
            system["numbers"],
            system["rcov"],
            system["edges"],
            system["shifts"],
            system["cell"],
            R_CUT,
        )
        n_atoms = len(coordination)
        coefficients = wp.zeros(
            (n_atoms, decomposition.rank), dtype=wp_dtype, device=device
        )
        derivative = wp.zeros_like(coefficients)
        fd3.fd3_coefficients(
            wp.array(coordination.astype(np_dtype), dtype=wp_dtype, device=device),
            wp.array(
                system["channels"].astype(np.int32), dtype=wp.int32, device=device
            ),
            wp.array(
                decomposition.cn_ref.astype(np_dtype), dtype=wp_dtype, device=device
            ),
            wp.array(decomposition.v_q.astype(np_dtype), dtype=wp_dtype, device=device),
            coefficients,
            derivative,
            wp_dtype,
            device,
        )
        expected, expected_derivative = reference.low_rank_coefficients(
            coordination, system["channels"], decomposition
        )
        np.testing.assert_allclose(
            coefficients.numpy(),
            expected,
            rtol=tolerance,
            atol=tolerance * np.abs(expected).max(),
        )
        np.testing.assert_allclose(
            derivative.numpy(),
            expected_derivative,
            rtol=tolerance,
            atol=tolerance * np.abs(expected_derivative).max(),
        )

    def test_uncovered_species_yields_zero(self, device, system):
        """An atom whose species is outside the decomposition gets zero coefficients."""
        decomposition = system["decomposition"]
        channels = system["channels"].copy()
        channels[:3] = -1
        n_atoms = len(channels)
        coefficients = wp.zeros(
            (n_atoms, decomposition.rank), dtype=wp.float64, device=device
        )
        derivative = wp.zeros_like(coefficients)
        fd3.fd3_coefficients(
            wp.array(np.ones(n_atoms), dtype=wp.float64, device=device),
            wp.array(channels.astype(np.int32), dtype=wp.int32, device=device),
            wp.array(decomposition.cn_ref, dtype=wp.float64, device=device),
            wp.array(decomposition.v_q, dtype=wp.float64, device=device),
            coefficients,
            derivative,
            wp.float64,
            device,
        )
        np.testing.assert_array_equal(coefficients.numpy()[:3], 0.0)
        np.testing.assert_array_equal(derivative.numpy()[:3], 0.0)


def _mesh_evaluate(system, mesh_dimensions, device, **kwargs):
    """Run the full pipeline through the test FFT harness."""
    from test.interactions.dispersion._fourier_harness import fourier_d3_energy

    n_atoms = len(system["positions"])
    return fourier_d3_energy(
        system["positions"],
        system["numbers"],
        system["channels"],
        np.zeros(n_atoms, dtype=np.int32),
        system["cell"][None],
        system["rcov"],
        system["decomposition"],
        system["sqrt_q"],
        system["targets"],
        system["pointer"],
        system["shifts"] @ system["cell"],
        R_CUT,
        mesh_dimensions,
        DAMPING,
        device=device,
        **kwargs,
    )


DAMPING = (1.0, 0.7875, 0.4289, 4.4407)


@pytest.fixture(scope="module")
def small_system():
    """A cell small enough for the direct lattice sum to be affordable."""
    rng = np.random.default_rng(0)
    c6ab, cn_ref, species = _reference_tables()
    decomposition = decompose_c6_reference(c6ab, cn_ref, species)
    box = 9.0
    positions = rng.uniform(0.0, box, (8, 3))
    numbers = rng.choice(species, 8)
    rcov = np.zeros(c6ab.shape[0])
    rcov[[1, 6, 8]] = [0.6, 1.2, 1.1]
    cell = np.eye(3) * box
    targets, pointer, shifts, edges = _neighbour_list(positions, cell, R_CUT)
    return {
        "decomposition": decomposition,
        "positions": positions,
        "numbers": numbers,
        "rcov": rcov,
        "cell": cell,
        "targets": targets,
        "pointer": pointer,
        "shifts": shifts,
        "edges": edges,
        "channels": decomposition.species_map[numbers],
        "sqrt_q": np.full(decomposition.n_species, 1.2),
    }


@pytest.fixture(scope="module")
def triclinic_system():
    """The same thing in a genuinely skewed cell.

    A cubic cell cannot exercise the fractional-to-Cartesian transforms properly: the inverse
    cell is diagonal, so it equals its own transpose and a confusion between the two is
    invisible. Every other cell in this file is cubic.
    """
    rng = np.random.default_rng(0)
    c6ab, cn_ref, species = _reference_tables()
    decomposition = decompose_c6_reference(c6ab, cn_ref, species)
    cell = np.array([[9.0, 0.0, 0.0], [2.6, 8.4, 0.0], [1.8, -2.1, 9.3]])
    positions = rng.uniform(0.0, 1.0, (8, 3)) @ cell
    numbers = rng.choice(species, 8)
    rcov = np.zeros(c6ab.shape[0])
    rcov[[1, 6, 8]] = [0.6, 1.2, 1.1]
    targets, pointer, shifts, edges = _neighbour_list(positions, cell, R_CUT)
    return {
        "decomposition": decomposition,
        "positions": positions,
        "numbers": numbers,
        "rcov": rcov,
        "cell": cell,
        "targets": targets,
        "pointer": pointer,
        "shifts": shifts,
        "edges": edges,
        "channels": decomposition.species_map[numbers],
        "sqrt_q": np.full(decomposition.n_species, 1.2),
    }


@pytest.mark.gpu
class TestMeshEnergy:
    """Passes 3 to 8 end to end, against the direct lattice sum."""

    def test_matches_direct_lattice_sum(self, small_system):
        """The mesh evaluation reproduces the same energy summed in real space.

        Both consume the same coordination numbers and coefficients, so a discrepancy is
        summation error rather than a difference of model. The oracle itself is only good to
        about 1e-7: its tail estimate assumes an isotropic distribution beyond the cutoff,
        which a small cell only approximately satisfies.
        """
        coordination = reference.modified_coordination_number(
            small_system["positions"],
            small_system["numbers"],
            small_system["rcov"],
            small_system["edges"],
            small_system["shifts"],
            small_system["cell"],
            R_CUT,
        )
        coefficients, _ = reference.low_rank_coefficients(
            coordination, small_system["channels"], small_system["decomposition"]
        )
        expected = reference.direct_lattice_energy(
            small_system["positions"],
            small_system["channels"],
            coefficients,
            small_system["decomposition"],
            small_system["sqrt_q"],
            small_system["cell"],
            *DAMPING,
            cutoff=80.0,
        )
        result = _mesh_evaluate(
            small_system, (48, 48, 48), "cuda:0", compute_forces=False
        )
        assert abs(result["energy"][0] - expected) / abs(expected) < 1e-6

    def test_refining_the_mesh_reduces_the_error(self, small_system):
        """Coarse meshes are worse than fine ones, over the range where the mesh dominates.

        Beyond about 32 cubed the comparison is limited by the oracle rather than the mesh,
        so only the coarse end is asserted.
        """
        coordination = reference.modified_coordination_number(
            small_system["positions"],
            small_system["numbers"],
            small_system["rcov"],
            small_system["edges"],
            small_system["shifts"],
            small_system["cell"],
            R_CUT,
        )
        coefficients, _ = reference.low_rank_coefficients(
            coordination, small_system["channels"], small_system["decomposition"]
        )
        expected = reference.direct_lattice_energy(
            small_system["positions"],
            small_system["channels"],
            coefficients,
            small_system["decomposition"],
            small_system["sqrt_q"],
            small_system["cell"],
            *DAMPING,
            cutoff=80.0,
        )
        errors = []
        for size in (16, 24, 32):
            result = _mesh_evaluate(
                small_system, (size, size, size), "cuda:0", compute_forces=False
            )
            errors.append(abs(result["energy"][0] - expected) / abs(expected))
        assert errors[0] > errors[1] > errors[2]
        assert errors[2] < 1e-6

    def test_forces_match_the_mesh_free_lattice_sum(self, small_system):
        """Mesh forces against a reference that never touches a mesh.

        The energy tests above cannot see the deconvolution: whether the attenuation divided
        out in reciprocal space really is the one the B-spline spread applies shows up in the
        gradient first. Refining the mesh cannot show it either, since a coarse and a fine
        mesh share the same modulus. So the reference here is central differences of
        ``direct_lattice_energy`` -- a brute-force real-space sum with no spline, no
        transform and no deconvolution anywhere in it.
        """
        step = 1e-5

        def lattice_energy(positions):
            targets, pointer, shifts, edges = _neighbour_list(
                positions, small_system["cell"], R_CUT
            )
            coordination = reference.modified_coordination_number(
                positions,
                small_system["numbers"],
                small_system["rcov"],
                edges,
                shifts,
                small_system["cell"],
                R_CUT,
            )
            coefficients, _ = reference.low_rank_coefficients(
                coordination, small_system["channels"], small_system["decomposition"]
            )
            return reference.direct_lattice_energy(
                positions,
                small_system["channels"],
                coefficients,
                small_system["decomposition"],
                small_system["sqrt_q"],
                small_system["cell"],
                *DAMPING,
                cutoff=80.0,
            )

        expected = np.zeros_like(small_system["positions"])
        for atom in range(len(expected)):
            for axis in range(3):
                shifted = []
                for sign in (1.0, -1.0):
                    moved = small_system["positions"].copy()
                    moved[atom, axis] += sign * step
                    shifted.append(lattice_energy(moved))
                expected[atom, axis] = -(shifted[0] - shifted[1]) / (2.0 * step)

        scale = np.abs(expected).max()
        errors = [
            np.abs(
                _mesh_evaluate(small_system, (size, size, size), "cuda:0")["forces"]
                - expected
            ).max()
            / scale
            for size in (32, 64)
        ]
        # Measured 2.9e-05 and 7.5e-06; refining has to help, and the fine mesh has to be
        # close, or the deconvolution does not match the spread.
        assert errors[1] < errors[0], errors
        assert errors[0] < 1e-4 and errors[1] < 2e-5, errors

    def test_energy_is_attractive(self, small_system):
        """Dispersion lowers the energy."""
        result = _mesh_evaluate(
            small_system, (32, 32, 32), "cuda:0", compute_forces=False
        )
        assert result["energy"][0] < 0.0


@pytest.mark.gpu
class TestForces:
    """Pass 7 and pass 9 against finite differences of the energy."""

    @staticmethod
    def _finite_difference_forces(system, mesh_dimensions, step=1e-5, **kwargs):
        """Central differences of the total energy with respect to every coordinate."""
        base = system["positions"].copy()
        forces = np.zeros_like(base)
        for atom in range(len(base)):
            for axis in range(3):
                for sign in (1.0, -1.0):
                    system["positions"] = base.copy()
                    system["positions"][atom, axis] += sign * step
                    energy = _mesh_evaluate(
                        system,
                        mesh_dimensions,
                        "cuda:0",
                        compute_forces=False,
                        **kwargs,
                    )["energy"][0]
                    forces[atom, axis] -= sign * energy / (2.0 * step)
        system["positions"] = base
        return forces

    def test_total_force_matches_finite_differences(self, small_system):
        """The analytic force is the gradient of the energy, mesh and chain rule together."""
        mesh = (32, 32, 32)
        analytic = _mesh_evaluate(small_system, mesh, "cuda:0")["forces"]
        numerical = self._finite_difference_forces(small_system, mesh)
        scale = np.abs(numerical).max()
        assert scale > 0.0
        np.testing.assert_allclose(analytic, numerical, atol=1e-6 * scale)

    def test_forces_match_finite_differences_in_a_skewed_cell(self, triclinic_system):
        """The adjoint of the fractional map must transpose it back.

        A position becomes fractional through the transpose of the inverse cell, so the
        gradient coming back has to be turned by the inverse cell itself. The two agree only
        for a diagonal cell, so a cubic test passes either way; in a skewed cell the wrong
        one leaves the forces off by parts in ten thousand while the energy and the virial
        stay right.
        """
        mesh = (32, 32, 32)
        analytic = _mesh_evaluate(triclinic_system, mesh, "cuda:0")["forces"]
        numerical = self._finite_difference_forces(triclinic_system, mesh)
        scale = np.abs(numerical).max()
        assert scale > 0.0
        np.testing.assert_allclose(analytic, numerical, atol=1e-6 * scale)

    def test_mesh_force_matches_finite_differences(self, small_system):
        """The mesh contribution alone is the gradient of the mesh energy alone.

        Holding the coordination numbers fixed removes the chain-rule term, so a failure here
        is in the spread, the transforms or the gather rather than in the neighbour list.
        """
        mesh = (32, 32, 32)
        coordination = _mesh_evaluate(
            small_system, mesh, "cuda:0", compute_forces=False
        )["coordination"]
        frozen = {"coordination_override": coordination, "apply_cn_chain": False}
        analytic = _mesh_evaluate(small_system, mesh, "cuda:0", **frozen)["forces"]
        numerical = self._finite_difference_forces(small_system, mesh, **frozen)
        scale = np.abs(numerical).max()
        np.testing.assert_allclose(analytic, numerical, atol=1e-6 * scale)

    def test_net_force_is_small(self, small_system):
        """The net force is an interpolation artifact, not exactly zero.

        Interpolating onto a mesh makes the energy depend very slightly on where the atoms
        sit relative to the grid, so it is not exactly translation invariant and the forces
        do not cancel exactly. The residual is a few parts in ten thousand of the largest
        force here, and shrinks as the mesh is refined. Demanding an exact zero would be
        asserting a property the method does not have.
        """
        forces = _mesh_evaluate(small_system, (32, 32, 32), "cuda:0")["forces"]
        net = np.abs(forces.sum(axis=0)).max()
        assert net < 1e-2 * np.abs(forces).max()

    def test_net_force_shrinks_with_mesh_refinement(self, small_system):
        """Refining the mesh reduces the translation-invariance artifact."""
        coarse = _mesh_evaluate(small_system, (16, 16, 16), "cuda:0")["forces"]
        fine = _mesh_evaluate(small_system, (48, 48, 48), "cuda:0")["forces"]
        coarse_net = np.abs(coarse.sum(axis=0)).max() / np.abs(coarse).max()
        fine_net = np.abs(fine.sum(axis=0)).max() / np.abs(fine).max()
        assert fine_net < coarse_net


@pytest.mark.gpu
class TestVirial:
    """The strain derivative, checked one source at a time.

    A total-only check can pass with one contribution missing whenever the other dominates,
    which is exactly what happened while this was being written: a sign error in the
    coordination-number term showed up as a 6e-3 discrepancy in the total and nothing at all
    in the reciprocal part.
    """

    @staticmethod
    def _finite_strain_virial(system, mesh_dimensions, step=1e-6, **kwargs):
        """``dE/d(strain)`` by central differences, straining positions and cell together."""
        base_positions = system["positions"].copy()
        base_cell = system["cell"].copy()
        result = np.zeros((3, 3))
        for row in range(3):
            for column in range(3):
                energies = []
                for sign in (1.0, -1.0):
                    strain = np.zeros((3, 3))
                    strain[row, column] = sign * step
                    deformation = np.eye(3) + strain
                    system["positions"] = base_positions @ deformation.T
                    system["cell"] = base_cell @ deformation.T
                    energies.append(
                        _mesh_evaluate(
                            system,
                            mesh_dimensions,
                            "cuda:0",
                            compute_forces=False,
                            **kwargs,
                        )["energy"][0]
                    )
                # Negated: conventions.md defines the virial as -dE/du, so the
                # finite difference has to carry the same sign to compare against.
                result[row, column] = -(energies[0] - energies[1]) / (2.0 * step)
        system["positions"] = base_positions
        system["cell"] = base_cell
        return result

    def test_reciprocal_virial_matches_finite_strain(self, small_system):
        """The reciprocal contribution alone is correct.

        Holding the coordination numbers fixed leaves only the volume and reciprocal-lattice
        terms, which is the part no later pass could recover.
        """
        mesh = (32, 32, 32)
        coordination = _mesh_evaluate(
            small_system, mesh, "cuda:0", compute_forces=False
        )["coordination"]
        frozen = {"coordination_override": coordination, "apply_cn_chain": False}
        analytic = _mesh_evaluate(
            small_system, mesh, "cuda:0", compute_virial=True, **frozen
        )["virial"][0]
        numerical = self._finite_strain_virial(small_system, mesh, **frozen)
        np.testing.assert_allclose(
            analytic, numerical, atol=1e-6 * np.abs(numerical).max()
        )

    def test_total_virial_matches_finite_strain(self, small_system):
        """Reciprocal and coordination contributions together."""
        mesh = (32, 32, 32)
        analytic = _mesh_evaluate(small_system, mesh, "cuda:0", compute_virial=True)[
            "virial"
        ][0]
        numerical = self._finite_strain_virial(small_system, mesh)
        np.testing.assert_allclose(
            analytic, numerical, atol=1e-6 * np.abs(numerical).max()
        )

    def test_coordination_contribution_is_present(self, small_system):
        """The coordination term contributes, and contributes correctly.

        Comparing the two configurations isolates the term a total-only test would let a sign
        error hide in.
        """
        mesh = (32, 32, 32)
        coordination = _mesh_evaluate(
            small_system, mesh, "cuda:0", compute_forces=False
        )["coordination"]
        frozen = {"coordination_override": coordination, "apply_cn_chain": False}
        reciprocal = _mesh_evaluate(
            small_system, mesh, "cuda:0", compute_virial=True, **frozen
        )["virial"][0]
        total = _mesh_evaluate(small_system, mesh, "cuda:0", compute_virial=True)[
            "virial"
        ][0]
        analytic = total - reciprocal
        numerical = self._finite_strain_virial(
            small_system, mesh
        ) - self._finite_strain_virial(small_system, mesh, **frozen)
        assert np.abs(analytic).max() > 0.0
        np.testing.assert_allclose(
            analytic, numerical, atol=1e-4 * np.abs(numerical).max()
        )

    def test_virial_is_symmetric(self, small_system):
        """The strain derivative of a rotationally invariant energy is symmetric."""
        virial = _mesh_evaluate(
            small_system, (32, 32, 32), "cuda:0", compute_virial=True
        )["virial"][0]
        np.testing.assert_allclose(virial, virial.T, atol=1e-12 * np.abs(virial).max())

    def test_not_requested_leaves_virial_zero(self, small_system):
        """The virial stays untouched when it is not asked for."""
        virial = _mesh_evaluate(
            small_system, (32, 32, 32), "cuda:0", compute_virial=False
        )["virial"]
        np.testing.assert_array_equal(virial, 0.0)


def _batched_system(cells_and_atoms, decomposition, rcov):
    """Concatenate several independent cells into one batched problem.

    Neighbour indices are made global and the CSR pointer is continued across systems, which
    is how the batched launchers expect a multi-system list to arrive.
    """
    positions, numbers, batch_idx, cells = [], [], [], []
    targets, pointers, shifts = [], [0], []
    atom_offset, edge_offset = 0, 0
    for index, (atoms, species, cell) in enumerate(cells_and_atoms):
        local_targets, local_pointer, local_shifts, _ = _neighbour_list(
            atoms, cell, R_CUT
        )
        positions.append(atoms)
        numbers.append(species)
        cells.append(cell)
        batch_idx.append(np.full(len(atoms), index, dtype=np.int32))
        targets.append(local_targets + atom_offset)
        shifts.append(local_shifts @ cell)
        pointers.extend((local_pointer[1:] + edge_offset).tolist())
        atom_offset += len(atoms)
        edge_offset += int(local_pointer[-1])
    numbers = np.concatenate(numbers)
    return {
        "decomposition": decomposition,
        "positions": np.concatenate(positions),
        "numbers": numbers,
        "rcov": rcov,
        "cells": np.stack(cells),
        "batch_idx": np.concatenate(batch_idx),
        "targets": np.concatenate(targets).astype(np.int32),
        "pointer": np.asarray(pointers, dtype=np.int32),
        "shifts_cartesian": np.concatenate(shifts),
        "channels": decomposition.species_map[numbers],
        "sqrt_q": np.full(decomposition.n_species, 1.2),
    }


def _evaluate_batched(system, mesh_dimensions, device, **kwargs):
    """Run the harness on a batched problem."""
    from test.interactions.dispersion._fourier_harness import fourier_d3_energy

    return fourier_d3_energy(
        system["positions"],
        system["numbers"],
        system["channels"],
        system["batch_idx"],
        system["cells"],
        system["rcov"],
        system["decomposition"],
        system["sqrt_q"],
        system["targets"],
        system["pointer"],
        system["shifts_cartesian"],
        R_CUT,
        mesh_dimensions,
        DAMPING,
        device=device,
        **kwargs,
    )


@pytest.mark.gpu
class TestBatching:
    """Several independent cells evaluated in one launch."""

    @staticmethod
    def _two_cells(seed=1):
        """Two chemically different cells and the decomposition covering both."""
        rng = np.random.default_rng(seed)
        c6ab, cn_ref, species = _reference_tables()
        decomposition = decompose_c6_reference(c6ab, cn_ref, species)
        rcov = np.zeros(c6ab.shape[0])
        rcov[[1, 6, 8]] = [0.6, 1.2, 1.1]
        first = (rng.uniform(0.0, 9.0, (6, 3)), rng.choice(species, 6), np.eye(3) * 9.0)
        second = (
            rng.uniform(0.0, 8.0, (5, 3)),
            rng.choice(species, 5),
            np.eye(3) * 8.0,
        )
        return decomposition, rcov, first, second

    def test_matches_each_cell_evaluated_alone(self):
        """A batch reproduces the single-system results exactly.

        Cells of different size and composition, so a layout error that happened to work for
        identical systems would show up here.
        """
        decomposition, rcov, first, second = self._two_cells()
        mesh = (24, 24, 24)
        together = _evaluate_batched(
            _batched_system([first, second], decomposition, rcov), mesh, "cuda:0"
        )
        boundary = len(first[0])
        for index, (cell, atoms) in enumerate(
            ((first, slice(0, boundary)), (second, slice(boundary, None)))
        ):
            alone = _evaluate_batched(
                _batched_system([cell], decomposition, rcov), mesh, "cuda:0"
            )
            np.testing.assert_allclose(
                together["energy"][index], alone["energy"][0], rtol=1e-12
            )
            np.testing.assert_allclose(
                together["forces"][atoms], alone["forces"], atol=1e-12
            )

    def test_identical_cells_give_identical_results(self):
        """Repeating a cell in a batch repeats its energy."""
        decomposition, rcov, first, _ = self._two_cells()
        result = _evaluate_batched(
            _batched_system([first, first], decomposition, rcov), (24, 24, 24), "cuda:0"
        )
        np.testing.assert_allclose(result["energy"][0], result["energy"][1], rtol=1e-12)

    def test_systems_do_not_leak_into_each_other(self):
        """Moving one cell's atoms leaves every other cell untouched.

        All systems share one mesh allocation, distinguished only by an index. This is the
        check that catches an error in that indexing; equal-energy tests on identical systems
        would not.

        The untouched system is compared to round-off rather than bit-for-bit: the energy is
        reduced with atomic adds, whose summation order varies between launches, so repeated
        identical evaluations differ in the last bit. Real leakage would be many orders larger
        than the tolerance used here.
        """
        decomposition, rcov, first, second = self._two_cells()
        mesh = (24, 24, 24)
        batch = _batched_system([first, second], decomposition, rcov)
        before = _evaluate_batched(batch, mesh, "cuda:0")

        boundary = len(first[0])
        moved = dict(batch)
        positions = batch["positions"].copy()
        # A non-rigid rearrangement. Translating the whole cell would leave the energy
        # essentially unchanged, since the method is translation invariant, and the check
        # below would then be comparing two identical numbers.
        positions[:boundary] += np.random.default_rng(11).uniform(
            -0.6, 0.6, (boundary, 3)
        )
        moved["positions"] = positions
        after = _evaluate_batched(moved, mesh, "cuda:0")

        assert abs(after["energy"][0] - before["energy"][0]) > 1e-6 * abs(
            before["energy"][0]
        )
        np.testing.assert_allclose(after["energy"][1], before["energy"][1], rtol=1e-12)
        np.testing.assert_allclose(
            after["forces"][boundary:],
            before["forces"][boundary:],
            atol=1e-12 * np.abs(before["forces"][boundary:]).max(),
        )

    def test_batched_virial_matches_single_systems(self):
        """Each system's virial is computed from its own cell."""
        decomposition, rcov, first, second = self._two_cells()
        mesh = (24, 24, 24)
        together = _evaluate_batched(
            _batched_system([first, second], decomposition, rcov),
            mesh,
            "cuda:0",
            compute_virial=True,
        )
        for index, cell in enumerate((first, second)):
            alone = _evaluate_batched(
                _batched_system([cell], decomposition, rcov),
                mesh,
                "cuda:0",
                compute_virial=True,
            )
            np.testing.assert_allclose(
                together["virial"][index], alone["virial"][0], atol=1e-12
            )

    @pytest.mark.parametrize("mesh", [(10, 10, 10), (16, 16, 16), (14, 14, 12)])
    def test_batch_isolation_holds_for_any_mesh(self, mesh):
        """No system picks up another's contribution, whatever the mesh size.

        The reciprocal pass reduces within a block and pads each system's bin count up to a
        multiple of the block size so that a block never straddles two systems. Meshes whose
        bin count is already a multiple of the block size skip that padding, so both cases
        are covered here.
        """
        decomposition, rcov, first, second = self._two_cells()
        together = _evaluate_batched(
            _batched_system([first, second], decomposition, rcov),
            mesh,
            "cuda:0",
            compute_virial=True,
        )
        boundary = len(first[0])
        for index, (cell, atoms) in enumerate(
            ((first, slice(0, boundary)), (second, slice(boundary, None)))
        ):
            alone = _evaluate_batched(
                _batched_system([cell], decomposition, rcov),
                mesh,
                "cuda:0",
                compute_virial=True,
            )
            np.testing.assert_allclose(
                together["energy"][index], alone["energy"][0], rtol=1e-11
            )
            np.testing.assert_allclose(
                together["forces"][atoms], alone["forces"], atol=1e-11
            )
            np.testing.assert_allclose(
                together["virial"][index], alone["virial"][0], atol=1e-11
            )


def _to_dense(targets, pointer, shifts_cartesian, n_atoms):
    """Convert a CSR neighbour list to the padded dense matrix form."""
    counts = np.diff(pointer)
    max_neighbors = int(counts.max()) if len(counts) else 0
    matrix = np.full((n_atoms, max_neighbors), n_atoms, dtype=np.int32)
    shifts = np.zeros((n_atoms, max_neighbors, 3))
    for atom in range(n_atoms):
        start, stop = pointer[atom], pointer[atom + 1]
        width = stop - start
        matrix[atom, :width] = targets[start:stop]
        shifts[atom, :width] = shifts_cartesian[start:stop]
    return matrix, shifts


@pytest.mark.gpu
class TestNeighbourMatrixFormat:
    """The dense neighbour matrix must agree with the CSR list."""

    @pytest.mark.parametrize("np_dtype", [np.float64, np.float32])
    def test_coordination_numbers_agree(self, device, system, np_dtype):
        """Both neighbour formats produce the same coordination numbers."""
        wp_dtype, vec_dtype = _warp_types(np_dtype)
        n_atoms = len(system["positions"])
        shifts_cartesian = system["shifts"] @ system["cell"]
        matrix, dense_shifts = _to_dense(
            system["targets"], system["pointer"], shifts_cartesian, n_atoms
        )

        common = dict(device=device)
        positions = wp.array(
            system["positions"].astype(np_dtype), dtype=vec_dtype, **common
        )
        numbers = wp.array(system["numbers"].astype(np.int32), dtype=wp.int32, **common)
        rcov = wp.array(system["rcov"].astype(np_dtype), dtype=wp_dtype, **common)

        csr = wp.zeros(n_atoms, dtype=wp_dtype, device=device)
        fd3.fd3_coordination_numbers(
            positions,
            numbers,
            wp.array(system["targets"], dtype=wp.int32, **common),
            wp.array(system["pointer"], dtype=wp.int32, **common),
            wp.array(shifts_cartesian.astype(np_dtype), dtype=vec_dtype, **common),
            rcov,
            R_CUT,
            csr,
            wp_dtype,
            device,
        )

        dense = wp.zeros(n_atoms, dtype=wp_dtype, device=device)
        fd3.fd3_coordination_numbers_matrix(
            positions,
            numbers,
            wp.array(matrix, dtype=wp.int32, **common),
            wp.array(dense_shifts.astype(np_dtype), dtype=vec_dtype, **common),
            rcov,
            R_CUT,
            dense,
            wp_dtype,
            device,
        )
        # The two formats sum an atom's neighbours in different orders, so agreement is
        # bounded by the accumulator's precision rather than by the neighbour format.
        tolerance = 1e-12 if np_dtype is np.float64 else 1e-6
        np.testing.assert_allclose(
            dense.numpy(), csr.numpy(), rtol=tolerance, atol=tolerance
        )

    def test_chain_rule_forces_agree(self, device, system):
        """Both neighbour formats produce the same chain-rule forces and virial."""
        n_atoms = len(system["positions"])
        rank = system["decomposition"].rank
        shifts_cartesian = system["shifts"] @ system["cell"]
        matrix, dense_shifts = _to_dense(
            system["targets"], system["pointer"], shifts_cartesian, n_atoms
        )
        rng = np.random.default_rng(5)

        common = dict(device=device)
        shared = dict(
            positions=wp.array(system["positions"], dtype=wp.vec3d, **common),
            numbers=wp.array(
                system["numbers"].astype(np.int32), dtype=wp.int32, **common
            ),
            rcov=wp.array(system["rcov"], dtype=wp.float64, **common),
            batch_idx=wp.zeros(n_atoms, dtype=wp.int32, **common),
        )
        d_energy_d_c6 = wp.array(
            rng.normal(size=(n_atoms, rank)), dtype=wp.float64, **common
        )
        dc6_dcn = wp.array(rng.normal(size=(n_atoms, rank)), dtype=wp.float64, **common)

        def evaluate(dense):
            forces = wp.zeros(n_atoms, dtype=wp.vec3d, **common)
            virial = wp.zeros(1, dtype=wp.mat33d, **common)
            sensitivity = wp.zeros(n_atoms, dtype=wp.float64, **common)
            if dense:
                fd3.fd3_cn_chain_matrix(
                    d_energy_d_c6,
                    dc6_dcn,
                    shared["positions"],
                    shared["numbers"],
                    wp.array(matrix, dtype=wp.int32, **common),
                    wp.array(dense_shifts, dtype=wp.vec3d, **common),
                    shared["rcov"],
                    R_CUT,
                    shared["batch_idx"],
                    sensitivity,
                    forces,
                    virial,
                    wp.float64,
                    device,
                    True,
                )
            else:
                fd3.fd3_cn_chain(
                    d_energy_d_c6,
                    dc6_dcn,
                    shared["positions"],
                    shared["numbers"],
                    wp.array(system["targets"], dtype=wp.int32, **common),
                    wp.array(system["pointer"], dtype=wp.int32, **common),
                    wp.array(shifts_cartesian, dtype=wp.vec3d, **common),
                    shared["rcov"],
                    R_CUT,
                    shared["batch_idx"],
                    sensitivity,
                    forces,
                    virial,
                    wp.float64,
                    device,
                    True,
                )
            return forces.numpy(), virial.numpy()

        dense_forces, dense_virial = evaluate(True)
        csr_forces, csr_virial = evaluate(False)
        assert np.abs(csr_forces).max() > 0.0
        np.testing.assert_allclose(dense_forces, csr_forces, atol=1e-12)
        np.testing.assert_allclose(dense_virial, csr_virial, atol=1e-12)

    def test_padding_slots_are_ignored(self, device, system):
        """Widening the matrix with padding changes nothing."""
        n_atoms = len(system["positions"])
        shifts_cartesian = system["shifts"] @ system["cell"]
        matrix, dense_shifts = _to_dense(
            system["targets"], system["pointer"], shifts_cartesian, n_atoms
        )
        padded = np.full((n_atoms, matrix.shape[1] + 7), n_atoms, dtype=np.int32)
        padded[:, : matrix.shape[1]] = matrix
        padded_shifts = np.zeros((n_atoms, padded.shape[1], 3))
        padded_shifts[:, : matrix.shape[1]] = dense_shifts

        common = dict(device=device)
        positions = wp.array(system["positions"], dtype=wp.vec3d, **common)
        numbers = wp.array(system["numbers"].astype(np.int32), dtype=wp.int32, **common)
        rcov = wp.array(system["rcov"], dtype=wp.float64, **common)

        results = []
        for indices, offsets in ((matrix, dense_shifts), (padded, padded_shifts)):
            out = wp.zeros(n_atoms, dtype=wp.float64, **common)
            fd3.fd3_coordination_numbers_matrix(
                positions,
                numbers,
                wp.array(indices, dtype=wp.int32, **common),
                wp.array(offsets, dtype=wp.vec3d, **common),
                rcov,
                R_CUT,
                out,
                wp.float64,
                device,
            )
            results.append(out.numpy())
        np.testing.assert_array_equal(results[0], results[1])


class TestMeshSelection:
    """The shared mesh helpers, which both bindings call rather than reimplement."""

    def test_explicit_dimensions_are_used_exactly(self):
        """A number the caller chose is not second-guessed, even when it transforms badly."""
        assert fd3._resolve_mesh((127, 127, 127), None, None, 4) == (127, 127, 127)

    @pytest.mark.parametrize(
        "spacing", [0.07, 0.0709, 0.0711, 0.073, 0.11, 0.37, 0.5, 1.3]
    )
    def test_a_spacing_derived_mesh_transforms_well(self, spacing):
        """cuFFT falls back to Bluestein otherwise.

        A prime edge measured 6.7x slower than a nearby smooth one on the same transform.
        """
        mesh = fd3._resolve_mesh(None, spacing, [9.0, 9.0, 9.0], 4)
        for size in mesh:
            remainder = size
            for prime in (2, 3, 5, 7):
                while remainder % prime == 0:
                    remainder //= prime
            assert remainder == 1, f"spacing {spacing} gave {mesh}"

    @pytest.mark.parametrize("spacing", [0.05, 0.0707, 0.09, 0.13, 0.5])
    def test_rounding_never_coarsens(self, spacing):
        """Rounding goes up, so the mesh is never sparser than the spacing asked for."""
        for size in fd3._resolve_mesh(None, spacing, [9.0, 9.0, 9.0], 4):
            assert size >= int(np.ceil(9.0 / spacing))

    @pytest.mark.parametrize("mesh_size", [1, 2, 3])
    def test_a_mesh_shorter_than_the_stencil_is_refused(self, mesh_size):
        """The order-4 stencil wraps onto a shorter axis and visits a node twice.

        Positive is not sufficient: the interpolation stops being the B-spline that the
        gather differentiates.
        """
        with pytest.raises(ValueError, match="at least"):
            fd3._check_mesh_supports_stencil((mesh_size,) * 3, 4, "test")

    def test_a_mesh_equal_to_the_stencil_is_allowed(self):
        """At equality every stencil point still lands on its own node."""
        assert fd3._check_mesh_supports_stencil((4, 4, 4), 4, "test") == (4, 4, 4)

    def test_a_spacing_too_coarse_for_the_stencil_is_refused(self):
        """The spacing route is held to the same minimum as explicit dimensions."""
        with pytest.raises(ValueError, match="mesh_spacing"):
            fd3._resolve_mesh(None, 100.0, [9.0, 9.0, 9.0], 4)
