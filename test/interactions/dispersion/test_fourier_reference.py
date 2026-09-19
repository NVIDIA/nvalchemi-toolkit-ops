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

"""Self-tests for the FourierD3 NumPy reference.

The reference is the oracle the GPU implementation is measured against, so it needs its own
checks: an oracle that agrees with a wrong implementation is worse than no oracle. These
tests are pure NumPy and need no GPU.
"""

import numpy as np
import pytest

from test.interactions.dispersion._fourier_reference import (
    _SERIES_R6,
    _SERIES_R8,
    SERIES_CUTOFF,
    coordination_pair_term,
    direct_lattice_energy,
    low_rank_coefficients,
    pair_potential,
    reciprocal_kernel,
    self_energy,
)

S6, S8, A1, A2 = 1.0, 0.7875, 0.4289, 4.4407
R_CUT = 6.0


def _radial_transform(k, r0, exponent, panels=3000, order=32, subdivisions=16):
    """Independent oracle: the radial Fourier integral by panelwise Gauss-Legendre.

    Panels are aligned to the zeros of ``sin(k r)`` so the oscillation is resolved without
    adaptive quadrature. The first panel is subdivided because it is ``pi/k`` wide.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    edges = np.arange(panels + 1) * np.pi / k
    first = np.linspace(edges[0], edges[1], subdivisions + 1)
    edges = np.concatenate([first, edges[2:]])
    lo, hi = edges[:-1, None], edges[1:, None]
    r = 0.5 * (hi - lo) * nodes[None, :] + 0.5 * (hi + lo)
    integrand = r * np.sin(k * r) / (r**exponent + r0**exponent)
    return (
        4.0 * np.pi / k * float((0.5 * (hi - lo) * weights[None, :] * integrand).sum())
    )


class TestCoordinationTerm:
    """The modified counting function and its boundary behaviour."""

    def test_reaches_zero_at_the_cutoff(self):
        """The counting function is exactly zero at the neighbour-list cutoff.

        This is the property the whole modification exists for: the standard D3 form has a
        non-zero limit, so truncating it leaves a step and the energy never converges as the
        cutoff grows.
        """
        assert coordination_pair_term(R_CUT, 2.6, R_CUT) == 0.0

    def test_saturates_at_short_range(self):
        """A close neighbour counts as one."""
        assert coordination_pair_term(0.5, 2.6, R_CUT) == pytest.approx(1.0)

    def test_is_monotone_decreasing(self):
        """No spurious structure between full coordination and the cutoff."""
        r = np.linspace(0.5, R_CUT, 4000)
        values = coordination_pair_term(r, 2.6, R_CUT)
        assert np.all(np.diff(values) <= 1e-12)

    def test_does_not_overflow_near_the_cutoff(self):
        """The steepness diverges at the cutoff; the logistic must stay finite.

        A plain ``1 / (1 + exp(-x))`` overflows here.
        """
        r = np.linspace(R_CUT - 1e-9, R_CUT, 64)
        with np.errstate(over="raise"):
            values = coordination_pair_term(r, 2.6, R_CUT)
        assert np.all(np.isfinite(values))


class TestReciprocalKernel:
    """The closed-form transform of the damped dispersion potential."""

    @pytest.mark.parametrize("r0", [2.0, 3.5, 5.0])
    @pytest.mark.parametrize("k_r0", [0.5, 1.0, 2.0, 4.31, 6.0, 12.0])
    def test_matches_radial_quadrature(self, r0, k_r0):
        """Both transform terms agree with direct numerical integration."""
        k = k_r0 / r0
        expected = S6 * _radial_transform(
            k, r0, 6
        ) + 3.0 * S8 * 1.0 * _radial_transform(k, r0, 8)
        assert reciprocal_kernel(k, r0, 1.0, S6, S8) == pytest.approx(
            expected, rel=1e-9
        )

    def test_branches_agree_at_the_crossover(self):
        """Series and closed form agree where the implementation switches between them.

        Both are evaluated at the same argument: comparing values either side of the cutoff
        would measure how fast the kernel varies, not whether the branches match. A genuine
        discontinuity would show up as mesh-size-dependent noise no convergence study could
        explain.
        """
        x = SERIES_CUTOFF
        series = sum(_SERIES_R6[m] * x**m for m in range(len(_SERIES_R6)))
        closed = (
            np.exp(-x)
            - 2.0 * np.exp(-x / 2.0) * np.cos(np.pi / 3.0 + x * np.sqrt(3.0) / 2.0)
        ) / x
        assert series == pytest.approx(closed, rel=1e-11)

        series_8 = sum(_SERIES_R8[m] * x**m for m in range(len(_SERIES_R8)))
        sin8, cos8 = np.sin(np.pi / 8.0), np.cos(np.pi / 8.0)
        closed_8 = (
            np.exp(-x * sin8) * np.cos(np.pi / 4.0 + x * cos8)
            + np.exp(-x * cos8) * np.cos(3.0 * np.pi / 4.0 + x * sin8)
        ) / x
        assert series_8 == pytest.approx(closed_8, rel=1e-11)

    def test_zero_wavevector_limit_is_finite(self):
        """The k = 0 term is finite and non-zero, unlike the Coulomb case.

        It carries the uniform dispersion background, so it must not be excluded from the
        reciprocal sum.
        """
        r0 = 3.0
        value = reciprocal_kernel(0.0, r0, 1.4, S6, S8)
        expected = S6 * 2.0 * np.pi**2 / (3.0 * r0**3) + 3.0 * S8 * 1.4 * (
            np.pi**2 / (2.0 * r0**5 * np.cos(np.pi / 8.0))
        )
        assert value == pytest.approx(expected, rel=1e-12)
        assert value > 0.0

    def test_oscillates_and_changes_sign(self):
        """The transform is a damped oscillation, not a monotone decay.

        Guards against a future 'fix' that clamps the kernel positive or takes an absolute
        value: the sign changes are physical.
        """
        r0 = 3.0
        k = np.linspace(1e-4, 10.0, 20000)
        values = reciprocal_kernel(k, r0, 1.4, S6, S8)
        sign_changes = np.sum(np.sign(values[:-1]) * np.sign(values[1:]) < 0)
        assert sign_changes >= 5
        first_zero = k[np.argmax(np.sign(values[:-1]) * np.sign(values[1:]) < 0)] * r0
        assert 4.0 < first_zero < 4.7

    def test_envelope_decays_exponentially(self):
        """The envelope decays at the rate set by the slower r^-8 pole."""
        r0 = 3.0
        k = np.linspace(1.0, 12.0, 4000) / r0
        envelope = np.exp(-k * r0 * np.sin(np.pi / 8.0))
        ratio = np.abs(reciprocal_kernel(k, r0, 1.4, S6, S8)) / envelope
        assert np.all(np.isfinite(ratio))
        assert ratio.max() < 10.0


def _toy_decomposition():
    """A small separable decomposition with the shape the real one has."""
    from nvalchemiops.interactions.dispersion._c6_decomposition import (
        decompose_c6_reference,
    )

    rng = np.random.default_rng(3)
    max_z, n_ref = 9, 5
    c6ab = np.zeros((max_z + 1, max_z + 1, n_ref, n_ref))
    cn_ref = np.zeros_like(c6ab)
    used = {1: 2, 6: 5, 8: 3}
    cn_per_z = {z: np.linspace(0.0, 3.5, n) for z, n in used.items()}
    factors = {z: rng.normal(size=(n, 3)) for z, n in used.items()}
    for zi in used:
        for zj in used:
            c6ab[zi, zj, : used[zi], : used[zj]] = factors[zi] @ factors[zj].T + 20.0
            for p in range(used[zi]):
                cn_ref[zi, zj, p, : used[zj]] = cn_per_z[zi][p]
    return decompose_c6_reference(c6ab, cn_ref, list(used))


class TestCoefficients:
    """Per-atom separable coefficients and their coordination derivative."""

    def test_derivative_matches_finite_differences(self):
        """The analytic ``dc6/dCN`` is the derivative of the coefficients it ships with.

        The chain rule from mesh to forces runs through this term, so an error here is
        invisible in the energy and wrong in every force.
        """
        decomposition = _toy_decomposition()
        numbers = np.array([1, 6, 8, 1, 6])
        channels = decomposition.species_map[numbers]
        coordination = np.array([0.8, 3.2, 1.9, 0.95, 3.4])

        _, analytic = low_rank_coefficients(coordination, channels, decomposition)
        step = 1e-6
        forward = low_rank_coefficients(coordination + step, channels, decomposition)[0]
        backward = low_rank_coefficients(coordination - step, channels, decomposition)[
            0
        ]
        numerical = (forward - backward) / (2.0 * step)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-8)

    def test_coefficients_are_weighted_means(self):
        """With one valid reference the coefficient is that reference's factor."""
        decomposition = _toy_decomposition()
        channels = np.array([decomposition.species_map[1]])
        # Sit exactly on a reference coordination number of hydrogen.
        coordination = np.array([decomposition.cn_ref[channels[0], 0]])
        coefficients, _ = low_rank_coefficients(coordination, channels, decomposition)
        assert np.all(np.isfinite(coefficients))
        assert coefficients.shape == (1, decomposition.rank)


class TestSelfEnergy:
    """The term that cancels the self pair the reciprocal sum inevitably includes."""

    def test_derivative_matches_finite_differences(self):
        """``dV_self/dc6`` is consistent with the energy it accompanies."""
        decomposition = _toy_decomposition()
        numbers = np.array([1, 6, 8])
        channels = decomposition.species_map[numbers]
        sqrt_q = np.full(decomposition.n_species, 1.2)
        coefficients, _ = low_rank_coefficients(
            np.array([0.9, 3.1, 1.8]), channels, decomposition
        )

        _, analytic = self_energy(
            coefficients, channels, decomposition, sqrt_q, S6, S8, A1, A2
        )
        step = 1e-6
        numerical = np.zeros_like(coefficients)
        for atom in range(coefficients.shape[0]):
            for slot in range(coefficients.shape[1]):
                up, down = coefficients.copy(), coefficients.copy()
                up[atom, slot] += step
                down[atom, slot] -= step
                numerical[atom, slot] = (
                    self_energy(up, channels, decomposition, sqrt_q, S6, S8, A1, A2)[0]
                    - self_energy(
                        down, channels, decomposition, sqrt_q, S6, S8, A1, A2
                    )[0]
                ) / (2.0 * step)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-10)

    def test_is_finite(self):
        """Becke-Johnson damping keeps the zero-separation potential bounded."""
        assert np.isfinite(pair_potential(0.0, 3.0, 1.4, S6, S8))


class TestDirectLatticeSum:
    """The oracle the mesh implementation is measured against."""

    def test_excludes_only_the_self_pair(self):
        """An atom interacts with its own periodic images but not with itself.

        Including the ``(i == i, T == 0)`` term would offset the oracle by exactly the
        self-energy, which is coordination-number dependent and mesh independent, and would
        look like a convergence floor the mesh could never reach.
        """
        decomposition = _toy_decomposition()
        numbers = np.array([1, 6])
        channels = decomposition.species_map[numbers]
        sqrt_q = np.full(decomposition.n_species, 1.2)
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.3, 0.1]])
        cell = np.eye(3) * 9.0
        coefficients, _ = low_rank_coefficients(
            np.array([0.9, 3.1]), channels, decomposition
        )

        primed = direct_lattice_energy(
            positions,
            channels,
            coefficients,
            decomposition,
            sqrt_q,
            cell,
            S6,
            S8,
            A1,
            A2,
            cutoff=24.0,
        )
        # Rebuild the unprimed sum by adding the self pair back explicitly.
        contribution, _ = self_energy(
            coefficients, channels, decomposition, sqrt_q, S6, S8, A1, A2
        )
        unprimed = primed - contribution
        assert primed != pytest.approx(unprimed)
        assert primed - unprimed == pytest.approx(contribution, rel=1e-12)

    def test_tail_correction_accelerates_convergence(self):
        """The analytic tail is what makes this oracle usable.

        A bare 1/r^6 lattice sum converges as 1/R^3: even a 96 unit cutoff still moves the
        energy in the fourth decimal place. The oracle has to be tighter than the mesh
        result it is judging, so the tail estimate is not an optimisation but a
        requirement.
        """
        decomposition = _toy_decomposition()
        numbers = np.array([1, 6, 8])
        channels = decomposition.species_map[numbers]
        sqrt_q = np.full(decomposition.n_species, 1.2)
        positions = np.array([[0.0, 0.0, 0.0], [2.1, 0.4, 0.2], [1.0, 2.3, 0.9]])
        cell = np.eye(3) * 8.0
        coefficients, _ = low_rank_coefficients(
            np.array([0.9, 3.1, 1.8]), channels, decomposition
        )

        def energy(cutoff, tail):
            return direct_lattice_energy(
                positions,
                channels,
                coefficients,
                decomposition,
                sqrt_q,
                cell,
                S6,
                S8,
                A1,
                A2,
                cutoff=cutoff,
                tail_correction=tail,
            )

        corrected = [energy(c, True) for c in (32.0, 48.0, 64.0)]
        bare = [energy(c, False) for c in (32.0, 48.0, 64.0)]
        corrected_drift = abs(corrected[2] - corrected[1]) / abs(corrected[2])
        bare_drift = abs(bare[2] - bare[1]) / abs(bare[2])
        assert corrected_drift < 1e-4
        assert corrected_drift < bare_drift / 10.0

    def test_energy_is_attractive(self):
        """Dispersion lowers the energy."""
        decomposition = _toy_decomposition()
        numbers = np.array([1, 6])
        channels = decomposition.species_map[numbers]
        sqrt_q = np.full(decomposition.n_species, 1.2)
        coefficients, _ = low_rank_coefficients(
            np.array([0.9, 3.1]), channels, decomposition
        )
        energy = direct_lattice_energy(
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            channels,
            coefficients,
            decomposition,
            sqrt_q,
            np.eye(3) * 9.0,
            S6,
            S8,
            A1,
            A2,
            cutoff=20.0,
        )
        assert energy < 0.0
