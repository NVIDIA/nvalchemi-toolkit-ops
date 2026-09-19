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

"""Tests for the host-side low-rank decomposition of the DFT-D3 reference tensor.

These tests are pure NumPy and need neither a GPU nor a deep-learning framework.
"""

import numpy as np
import pytest

from nvalchemiops.interactions.dispersion._c6_decomposition import (
    CNREF_INVALID,
    K3_WEIGHT,
    _select_rank,
    _truncation_error,
    clear_decomposition_cache,
    decompose_c6_reference,
    extract_species_reference_cn,
)

N_REF = 5
MAX_Z = 10


def _synthetic_tables(seed: int = 0, n_used=(2, 3, 5)):
    """Build reference tables with the structure of the Grimme parametrisation.

    The generated tensor is symmetric under ``(Zi, p) <-> (Zj, q)`` and gives each element a
    different number of populated reference slots, mirroring the real tables where hydrogen
    has two references and carbon has five.
    """
    rng = np.random.default_rng(seed)
    c6ab = np.zeros((MAX_Z + 1, MAX_Z + 1, N_REF, N_REF))
    cn_ref = np.zeros((MAX_Z + 1, MAX_Z + 1, N_REF, N_REF))

    used = {z: n_used[z % len(n_used)] for z in range(1, MAX_Z + 1)}
    cn_per_z = {z: np.sort(rng.uniform(0.0, 4.0, used[z])) for z in range(1, MAX_Z + 1)}
    # A low-rank generator keeps the tensor compressible, as the real tensor is.
    factors = {z: rng.normal(size=(used[z], 3)) for z in range(1, MAX_Z + 1)}

    for zi in range(1, MAX_Z + 1):
        for zj in range(1, MAX_Z + 1):
            block = factors[zi] @ factors[zj].T + 10.0
            c6ab[zi, zj, : used[zi], : used[zj]] = block
            for p in range(used[zi]):
                cn_ref[zi, zj, p, : used[zj]] = cn_per_z[zi][p]
    return c6ab, cn_ref, cn_per_z


def _reference_weights(decomposition, channel, coordination_number):
    """Gaussian reference weights for one atom, zero on unused slots."""
    delta = coordination_number - decomposition.cnref[channel]
    return np.where(decomposition.valid[channel], np.exp(-K3_WEIGHT * delta**2), 0.0)


def _low_rank_c6(decomposition, z_i, z_j, cn_i, cn_j):
    """Pair coefficient rebuilt from the separable per-atom factors."""

    def coefficients(z, cn):
        channel = decomposition.species_map[z]
        weights = _reference_weights(decomposition, channel, cn)
        return (decomposition.v_q[channel].T @ weights) / weights.sum()

    return float(
        decomposition.eigs @ (coefficients(z_i, cn_i) * coefficients(z_j, cn_j))
    )


def _direct_c6(c6ab, cn_ref, z_i, z_j, cn_i, cn_j):
    """Grimme's Gaussian-weighted pair interpolation, evaluated directly."""
    numerator = denominator = 0.0
    for p in range(N_REF):
        for q in range(N_REF):
            if c6ab[z_i, z_j, p, q] == 0.0:
                continue
            d_i = cn_i - cn_ref[z_i, z_j, p, q]
            d_j = cn_j - cn_ref[z_j, z_i, q, p]
            weight = np.exp(-K3_WEIGHT * (d_i * d_i + d_j * d_j))
            numerator += c6ab[z_i, z_j, p, q] * weight
            denominator += weight
    return numerator / denominator


@pytest.fixture(autouse=True)
def _isolate_cache():
    """Keep cached decompositions from leaking between tests."""
    clear_decomposition_cache()
    yield
    clear_decomposition_cache()


@pytest.fixture(scope="module")
def tables():
    """Synthetic reference tables shaped like the DFT-D3 parametrisation."""
    return _synthetic_tables()


class TestExtractSpeciesReferenceCN:
    """Collapsing the pair-indexed reference tables to per-species arrays."""

    def test_recovers_per_species_values(self, tables):
        """Extracted reference CNs match the values the tables were built from."""
        c6ab, cn_ref, cn_per_z = tables
        cnref, valid = extract_species_reference_cn(c6ab, cn_ref, [1, 3, 5])
        for row, z in enumerate([1, 3, 5]):
            expected = cn_per_z[z]
            np.testing.assert_allclose(cnref[row, valid[row]], expected)

    def test_unused_slots_are_padded(self, tables):
        """Slots a species does not use are marked invalid and padded."""
        c6ab, cn_ref, cn_per_z = tables
        cnref, valid = extract_species_reference_cn(c6ab, cn_ref, [1])
        assert valid[0].sum() == len(cn_per_z[1])
        assert np.all(cnref[0, ~valid[0]] == CNREF_INVALID)

    def test_rejects_inconsistent_reference_cn(self, tables):
        """A species whose reference CN varies by partner element is rejected.

        The per-species form is only well defined because the tables repeat the same value
        across partners. Silently taking the first partner would give wrong coefficients.
        """
        c6ab, cn_ref, _ = tables
        corrupted = cn_ref.copy()
        corrupted[1, 3, 0, 0] += 0.5
        with pytest.raises(ValueError, match="not consistent across partner elements"):
            extract_species_reference_cn(c6ab, corrupted, [1, 3])

    def test_rejects_unparametrised_element(self, tables):
        """An element with no reference data raises rather than producing empty factors."""
        c6ab, cn_ref, _ = tables
        empty = c6ab.copy()
        empty[2] = 0.0
        empty[:, 2] = 0.0
        with pytest.raises(ValueError, match="No reference C6 data"):
            extract_species_reference_cn(empty, cn_ref, [2])


class TestDecomposition:
    """Rank selection and reconstruction accuracy."""

    def test_reconstructs_pair_coefficients(self, tables):
        """The separable form reproduces the direct pair interpolation within tolerance.

        This is the property the whole method rests on: FourierD3 replaces a non-separable
        pair coefficient with a sum of products of atom-centred factors.
        """
        c6ab, cn_ref, _ = tables
        species = [1, 2, 3, 4]
        decomposition = decompose_c6_reference(c6ab, cn_ref, species, tol=1e-6)
        rng = np.random.default_rng(1)
        for _ in range(200):
            z_i, z_j = rng.choice(species, 2)
            cn_i, cn_j = rng.uniform(0.0, 4.0, 2)
            direct = _direct_c6(c6ab, cn_ref, z_i, z_j, cn_i, cn_j)
            separable = _low_rank_c6(decomposition, z_i, z_j, cn_i, cn_j)
            assert abs(separable - direct) / abs(direct) < 1e-5

    def test_tighter_tolerance_needs_more_rank(self, tables):
        """Rank is monotonic in the accuracy requested."""
        c6ab, cn_ref, _ = tables
        loose = decompose_c6_reference(c6ab, cn_ref, [1, 2, 3], tol=1e-2)
        tight = decompose_c6_reference(c6ab, cn_ref, [1, 2, 3], tol=1e-10)
        assert tight.rank >= loose.rank
        assert tight.max_relative_error <= loose.max_relative_error

    def test_achieved_error_meets_tolerance(self, tables):
        """The reported error is the true error of the returned rank."""
        c6ab, cn_ref, _ = tables
        decomposition = decompose_c6_reference(c6ab, cn_ref, [1, 2, 3, 4, 5], tol=1e-6)
        assert decomposition.max_relative_error <= 1e-6

    def test_max_rank_reports_rather_than_raises(self, tables):
        """A rank ceiling that cannot meet the tolerance still returns, with the error.

        Callers trading accuracy for mesh channels need the achieved error, not an
        exception.
        """
        c6ab, cn_ref, _ = tables
        decomposition = decompose_c6_reference(
            c6ab, cn_ref, [1, 2, 3, 4, 5], tol=1e-12, max_rank=1
        )
        assert decomposition.rank == 1
        assert decomposition.max_relative_error > 1e-12

    def test_shapes_and_channel_map(self, tables):
        """Factor shapes and the atomic-number to channel map are consistent."""
        c6ab, cn_ref, _ = tables
        species = [3, 1, 5]
        decomposition = decompose_c6_reference(c6ab, cn_ref, species)
        assert decomposition.species.tolist() == [1, 3, 5]
        assert decomposition.v_q.shape == (3, N_REF, decomposition.rank)
        assert decomposition.eigs.shape == (decomposition.rank,)
        assert decomposition.num_channels == 3 * decomposition.rank
        assert decomposition.species_map[[1, 3, 5]].tolist() == [0, 1, 2]
        assert decomposition.species_map[2] == -1

    def test_invalid_slots_carry_no_weight(self, tables):
        """Eigenvector rows for unused reference slots are zero."""
        c6ab, cn_ref, _ = tables
        decomposition = decompose_c6_reference(c6ab, cn_ref, [1, 2, 3])
        for channel in range(decomposition.n_species):
            unused = ~decomposition.valid[channel]
            np.testing.assert_allclose(decomposition.v_q[channel][unused], 0.0)


class TestRankSelection:
    """Choosing the smallest rank that meets the tolerance.

    The spectrum is signed and ordered by magnitude, so adding a term can cancel against
    earlier ones and make the reconstruction worse. Any search that assumes the error falls
    monotonically with rank can step over a qualifying smaller rank.
    """

    @staticmethod
    def _non_monotonic_block():
        """A block whose truncation error rises from rank 2 to rank 3.

        Built from a fixed rotation and a signed spectrum, so the curve is deterministic.
        """
        rng = np.random.default_rng(3)
        rotation, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        spectrum = np.array([5.0, -3.0, 2.0, -1.0, 0.5, -0.2])
        block = (rotation * spectrum) @ rotation.T
        block = (block + block.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(block)
        order = np.argsort(np.abs(eigvals))[::-1]
        return block, eigvals[order], eigvecs[:, order]

    def test_the_error_is_not_monotonic_in_rank(self):
        """Guards the premise: without this the test below proves nothing."""
        block, eigvals, eigvecs = self._non_monotonic_block()
        nonzero = block != 0.0
        errors = [
            _truncation_error(block, eigvals, eigvecs, rank, nonzero)
            for rank in range(1, 7)
        ]
        assert errors[2] > errors[1]

    def test_it_returns_the_smallest_qualifying_rank(self):
        """A bisection over this curve would return rank 4 instead of rank 2."""
        block, eigvals, eigvecs = self._non_monotonic_block()
        nonzero = block != 0.0
        tol = _truncation_error(block, eigvals, eigvecs, 2, nonzero)

        rank, error = _select_rank(block, eigvals, eigvecs, tol, None)

        assert rank == 2
        assert error <= tol

    def test_an_unreachable_tolerance_reports_the_best_effort(self):
        """No rank meets the tolerance, so the cap and its true error come back."""
        block, eigvals, eigvecs = self._non_monotonic_block()
        rank, error = _select_rank(block, eigvals, eigvecs, 1e-30, 4)
        assert rank == 4
        assert error > 1e-30


class TestCaching:
    """The cache must never return factors computed from different inputs."""

    def test_species_order_is_canonicalised(self, tables):
        """Permuting the species argument hits the same cache entry."""
        c6ab, cn_ref, _ = tables
        first = decompose_c6_reference(c6ab, cn_ref, [1, 2, 3])
        assert decompose_c6_reference(c6ab, cn_ref, [3, 1, 2]) is first
        assert decompose_c6_reference(c6ab, cn_ref, [1, 1, 2, 3, 3]) is first

    def test_mutated_c6ab_is_not_a_cache_hit(self, tables):
        """Changing the C6 table changes the factors.

        A cache keyed only on species and tolerance would return the previous tensor's
        eigenvectors here, producing wrong energies with no error.
        """
        c6ab, cn_ref, _ = tables
        base = decompose_c6_reference(c6ab, cn_ref, [1, 2])
        mutated = c6ab.copy()
        mutated[1, 2, 0, 0] *= 1.05
        assert decompose_c6_reference(mutated, cn_ref, [1, 2]) is not base

    def test_mutated_cn_ref_is_not_a_cache_hit(self, tables):
        """Changing the reference coordination numbers changes the result."""
        c6ab, cn_ref, _ = tables
        base = decompose_c6_reference(c6ab, cn_ref, [1, 2])
        mutated = cn_ref.copy()
        mutated[1, :, 0, :] += 0.1
        result = decompose_c6_reference(c6ab, mutated, [1, 2])
        assert not np.allclose(result.cnref, base.cnref)

    def test_options_participate_in_the_key(self, tables):
        """Every option that can change the rank is part of the cache key."""
        c6ab, cn_ref, _ = tables
        base = decompose_c6_reference(c6ab, cn_ref, [1, 2, 3], tol=1e-4)
        assert decompose_c6_reference(c6ab, cn_ref, [1, 2, 3], tol=1e-10) is not base
        assert decompose_c6_reference(c6ab, cn_ref, [1, 2, 3], max_rank=2) is not base

    def test_cache_is_bounded(self, tables):
        """Sweeping many compositions does not grow the cache without limit."""
        from nvalchemiops.interactions.dispersion import _c6_decomposition

        c6ab, cn_ref, _ = tables
        for tol in np.logspace(-2, -10, 40):
            decompose_c6_reference(c6ab, cn_ref, [1, 2], tol=float(tol))
        assert len(_c6_decomposition._CACHE) <= _c6_decomposition._CACHE_SIZE


class TestValidation:
    """Input validation at the public entry point."""

    def test_rejects_empty_species(self, tables):
        """An empty species list raises."""
        c6ab, cn_ref, _ = tables
        with pytest.raises(ValueError, match="at least one atomic number"):
            decompose_c6_reference(c6ab, cn_ref, [])

    @pytest.mark.parametrize("tol", [0.0, -1e-4])
    def test_rejects_non_positive_tolerance(self, tables, tol):
        """Tolerance must be positive."""
        c6ab, cn_ref, _ = tables
        with pytest.raises(ValueError, match="tol must be positive"):
            decompose_c6_reference(c6ab, cn_ref, [1], tol=tol)

    def test_rejects_non_positive_max_rank(self, tables):
        """A rank ceiling below one leaves nothing to return."""
        c6ab, cn_ref, _ = tables
        with pytest.raises(ValueError, match="max_rank must be at least 1"):
            decompose_c6_reference(c6ab, cn_ref, [1], max_rank=0)

    def test_rejects_mismatched_table_shapes(self, tables):
        """The two reference tables must be indexed identically."""
        c6ab, cn_ref, _ = tables
        with pytest.raises(ValueError, match="same shape"):
            decompose_c6_reference(c6ab, cn_ref[:-1], [1])

    @pytest.mark.parametrize("z", [0, MAX_Z + 1])
    def test_rejects_out_of_range_species(self, tables, z):
        """Atomic numbers outside the table are reported, not clipped."""
        c6ab, cn_ref, _ = tables
        with pytest.raises(ValueError, match="outside the reference table"):
            decompose_c6_reference(c6ab, cn_ref, [z])
