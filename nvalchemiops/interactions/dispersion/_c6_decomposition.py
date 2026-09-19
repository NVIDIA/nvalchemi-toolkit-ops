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

r"""
Low-Rank Decomposition of the DFT-D3 Reference Tensor
=====================================================

FourierD3 evaluates the D3 dispersion sum on a particle mesh. Mesh methods require the
pairwise coefficient to separate into atom-centred factors, which D3's
environment-dependent :math:`C_6` coefficients do not do: they couple the coordination
numbers of both atoms.

This module restores separability by decomposing Grimme's :math:`C_6^{\text{ref}}` tensor
once, on the host, into

.. math::

    C_6^{\text{ref}}(Z_i, \theta^p; Z_j, \theta^q)
        \approx \sum_{\ell} \lambda_{\ell}\,
                v_{\ell, Z_i}(\theta^p)\, v_{\ell, Z_j}(\theta^q)

so that the per-atom coefficients

.. math::

    c_6[i, \ell] = \frac{\sum_p v_{\ell, Z_i}(\theta^p) L_i^p}{\sum_p L_i^p},
    \qquad L_i^p = \exp\!\big(-4 (\theta_i - \theta_{Z_i}^p)^2\big)

reproduce the pair coefficient as
:math:`C_6^{ij} = \sum_{\ell} \lambda_{\ell}\, c_6[i,\ell]\, c_6[j,\ell]`. Each rank slot
then spreads onto its own mesh channel and is summed independently.

Rank is the accuracy/cost knob: it sets the number of mesh channels, and therefore both the
FFT cost and the mesh memory. The reference tensor is strongly linearly dependent, so the
rank needed for a given tolerance grows sublinearly with the number of species.

The decomposition depends only on the reference tables, the species present, and the
tolerance. It is **independent of the damping parameters**, so one result is valid for every
functional parametrisation.

Units
-----
This module is unit-agnostic. ``eigs`` and ``v_q`` carry whatever units ``c6ab`` uses, and
``cnref`` whatever units ``cn_ref`` uses (coordination numbers are dimensionless).

References
----------
Valeeva et al., *A fast summation method for the DFT-D3 dispersion correction*,
arXiv:2607.15103, Section II B and Eq. (20)-(22).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "C6Decomposition",
    "decompose_c6_reference",
    "extract_species_reference_cn",
    "clear_decomposition_cache",
]

# Gaussian width of the D3 reference weighting, L = exp(-K3_WEIGHT * (cn - cn_ref)^2).
# Grimme's k3 = -4 with the opposite sign convention used in the kernel.
K3_WEIGHT = 4.0

# Bounded so that a long-running process sweeping many compositions cannot grow without limit.
_CACHE_SIZE = 32
_CACHE: OrderedDict[tuple, C6Decomposition] = OrderedDict()


@dataclass(frozen=True)
class C6Decomposition:
    """Separable low-rank factors of the DFT-D3 reference tensor.

    Attributes
    ----------
    species : np.ndarray, shape (n_species,), dtype=int32
        Atomic numbers covered, ascending. Position in this array is the channel index.
    eigs : np.ndarray, shape (rank,), dtype=float64
        Eigenvalues :math:`\\lambda_{\\ell}`, ordered by descending magnitude. May be
        negative; the energy expression is a signed sum, not a norm.
    v_q : np.ndarray, shape (n_species, n_ref, rank), dtype=float64
        Eigenvectors :math:`v_{\\ell, Z}(\\theta^p)`. Rows for invalid reference slots are
        zero.
    cnref : np.ndarray, shape (n_species, n_ref), dtype=float64
        Reference coordination numbers per species, padded with ``CNREF_INVALID``.
    valid : np.ndarray, shape (n_species, n_ref), dtype=bool
        True where a reference slot carries data.
    species_map : np.ndarray, shape (max_z + 1,), dtype=int32
        Atomic number to channel index; ``-1`` for elements not covered.
    max_relative_error : float
        Largest relative error of the reconstructed tensor over all valid entries.
    """

    species: np.ndarray
    eigs: np.ndarray
    v_q: np.ndarray
    cnref: np.ndarray
    valid: np.ndarray
    species_map: np.ndarray
    max_relative_error: float

    @property
    def rank(self) -> int:
        """Number of retained rank slots."""
        return int(self.eigs.shape[0])

    @property
    def n_species(self) -> int:
        """Number of species covered."""
        return int(self.species.shape[0])

    @property
    def n_ref(self) -> int:
        """Reference-coordination-number slots per species, including padding."""
        return int(self.cnref.shape[1])

    @property
    def num_channels(self) -> int:
        """Mesh channels required, ``n_species * rank``."""
        return self.n_species * self.rank


# Padding value for reference slots a species does not use. Chosen far from any physical
# coordination number so that its Gaussian weight underflows rather than contributing.
CNREF_INVALID = -1.0


def _table_fingerprint(*arrays: np.ndarray) -> str:
    """Content hash of the reference tables.

    Parameters
    ----------
    *arrays : np.ndarray
        Tables to fingerprint. Shape and dtype are folded in alongside the bytes, so two
        tables that differ only in layout hash differently.

    Returns
    -------
    str
        128-bit BLAKE2b digest, hexadecimal.

    Notes
    -----
    Hashing content rather than object identity is deliberate. CPython reuses ``id()``
    values after garbage collection, so an identity-keyed cache can return factors computed
    from a table that no longer exists.
    """
    digest = hashlib.blake2b(digest_size=16)
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def extract_species_reference_cn(
    c6ab: np.ndarray,
    cn_ref: np.ndarray,
    species: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the pair-indexed reference tables to per-species arrays.

    Grimme's tables store reference coordination numbers per element *pair*, as
    ``cn_ref[Zi, Zj, p, q]``, because they are consumed by a pairwise interpolation. The
    value depends only on ``Zi`` and ``p``; the trailing indices repeat it. FourierD3 needs
    the per-species form, since each atom is weighted independently before it reaches the
    mesh.

    Parameters
    ----------
    c6ab : np.ndarray, shape (max_z + 1, max_z + 1, n_ref, n_ref)
        Reference :math:`C_6` coefficients. Entries equal to zero mark unused reference
        slots.
    cn_ref : np.ndarray, shape (max_z + 1, max_z + 1, n_ref, n_ref)
        Reference coordination numbers, pair-indexed.
    species : Sequence[int]
        Atomic numbers to extract, in the caller's order.

    Returns
    -------
    cnref : np.ndarray, shape (n_species, n_ref), dtype=float64
        Reference coordination numbers, padded with ``CNREF_INVALID``.
    valid : np.ndarray, shape (n_species, n_ref), dtype=bool
        True where a species uses a reference slot.

    Raises
    ------
    ValueError
        If a species has no valid reference data, or if the tables disagree about a
        species' reference coordination numbers across partner elements.

    Notes
    -----
    The consistency check is not defensive padding: it is the assumption that makes the
    per-species form well defined. If a future parameter set breaks it, silently taking the
    first partner's values would produce wrong coefficients with no other symptom.
    """
    n_ref = c6ab.shape[2]
    cnref = np.full((len(species), n_ref), CNREF_INVALID, dtype=np.float64)
    valid = np.zeros((len(species), n_ref), dtype=bool)

    for row, z in enumerate(species):
        # A slot is used by this species if any partner element populates it.
        used = (c6ab[z] != 0.0).any(axis=(0, 2))
        if not used.any():
            raise ValueError(
                f"No reference C6 data for atomic number {z}. The element is either outside "
                f"the table (covers 1..{c6ab.shape[0] - 1}) or unparametrised."
            )
        for p in np.flatnonzero(used):
            partners = c6ab[z, :, p, :] != 0.0
            values = cn_ref[z, :, p, :][partners]
            if not np.allclose(values, values[0], rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"Reference coordination numbers for Z={z}, slot {p} are not consistent "
                    f"across partner elements (spread {values.min()} to {values.max()}). "
                    f"The per-species decomposition is not valid for this parameter set."
                )
            cnref[row, p] = values[0]
            valid[row, p] = True

    return cnref, valid


def _build_block_matrix(
    c6ab: np.ndarray,
    species: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Assemble the symmetric ``(n_species * n_ref)`` block matrix of Eq. (20).

    Row ``a * n_ref + p`` corresponds to species ``species[a]`` in reference slot ``p``.
    Invalid slots contribute zero rows and columns, which map to zero eigenvalues and drop
    out of the truncation.
    """
    n_species, n_ref = valid.shape
    size = n_species * n_ref
    block = np.zeros((size, size), dtype=np.float64)

    for a, z_a in enumerate(species):
        for b, z_b in enumerate(species):
            sub = c6ab[z_a, z_b].astype(np.float64)
            mask = valid[a][:, None] & valid[b][None, :]
            block[
                a * n_ref : (a + 1) * n_ref,
                b * n_ref : (b + 1) * n_ref,
            ] = np.where(mask, sub, 0.0)

    # The tables are symmetric under (Zi, p) <-> (Zj, q); averaging removes the last bits of
    # float32 storage asymmetry so that eigh sees an exactly symmetric operand.
    return 0.5 * (block + block.T)


def _truncation_error(
    block: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    rank: int,
    nonzero: np.ndarray,
) -> float:
    """Largest relative error of the rank-``rank`` reconstruction over populated entries."""
    approx = (eigvecs[:, :rank] * eigvals[:rank]) @ eigvecs[:, :rank].T
    return float(
        np.max(np.abs(approx[nonzero] - block[nonzero]) / np.abs(block[nonzero]))
    )


def _select_rank(
    block: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    tol: float,
    max_rank: int | None,
) -> tuple[int, float]:
    """Smallest rank whose reconstruction meets ``tol``, and the error it achieves.

    Scans upward rather than bisecting. The spectrum is signed and ordered by magnitude, so
    adding a term can cancel against earlier ones and make the error worse: on the shipped
    tables the error rises at 44 of 180 rank steps for ``Z <= 36``. Bisection assumes
    monotonicity and so can step over a smaller rank that already meets the tolerance --
    measured at up to six ranks too many on the full table, which is mesh channels and
    reciprocal-space work spent for nothing.

    Returns ``upper`` and its error when no rank meets ``tol``, so the caller can choose
    between loosening ``tol`` and raising ``max_rank``.
    """
    nonzero = block != 0.0
    if not nonzero.any():
        raise ValueError(
            "Reference block matrix is entirely zero; no species carry data."
        )

    upper = eigvals.shape[0] if max_rank is None else min(max_rank, eigvals.shape[0])
    error = _truncation_error(block, eigvals, eigvecs, upper, nonzero)
    for rank in range(1, upper + 1):
        candidate = _truncation_error(block, eigvals, eigvecs, rank, nonzero)
        if candidate <= tol:
            return rank, candidate
    return upper, error


def clear_decomposition_cache() -> None:
    """Drop all cached decompositions.

    Useful in tests and in long-running processes that have finished with one chemistry.
    """
    _CACHE.clear()


def decompose_c6_reference(
    c6ab: np.ndarray,
    cn_ref: np.ndarray,
    species: Sequence[int],
    tol: float = 1e-4,
    max_rank: int | None = None,
) -> C6Decomposition:
    r"""Decompose the DFT-D3 reference tensor into separable low-rank factors.

    Builds the symmetric block matrix over (species, reference-slot) pairs, eigendecomposes
    it, and truncates at the smallest rank meeting ``tol``.

    Parameters
    ----------
    c6ab : np.ndarray, shape (max_z + 1, max_z + 1, n_ref, n_ref)
        Reference :math:`C_6` coefficients from the DFT-D3 parametrisation. Zero entries
        mark unused reference slots.
    cn_ref : np.ndarray, shape (max_z + 1, max_z + 1, n_ref, n_ref)
        Reference coordination numbers, pair-indexed.
    species : Sequence[int]
        Atomic numbers present in the system. Duplicates and ordering are ignored: the
        result is canonicalised to sorted unique values, so the channel layout is
        deterministic.
    tol : float, default=1e-4
        Target maximum relative error of the reconstructed tensor. The default follows the
        FourierD3 paper, which reports it as a good balance for systems with fewer than ten
        elements.
    max_rank : int, optional
        Hard ceiling on the retained rank. When the ceiling prevents ``tol`` from being met,
        the decomposition is returned anyway and
        :attr:`C6Decomposition.max_relative_error` reports what was achieved.

    Returns
    -------
    C6Decomposition
        Factors, reference coordination numbers, channel map, and achieved error.

    Raises
    ------
    ValueError
        If ``species`` is empty, ``tol`` is not positive, ``max_rank`` is not positive, the
        tables have inconsistent shapes, or a species carries no reference data.

    Notes
    -----
    Results are cached on the content of the tables together with every option that can
    change the factors, so switching parameter sets or ceilings cannot return stale
    factors. The cache holds at most 32 entries.

    The decomposition does not depend on the damping parameters, so a single result may be
    reused across functional parametrisations.

    Examples
    --------
    >>> decomposition = decompose_c6_reference(c6ab, cn_ref, species=[1, 8])
    >>> decomposition.rank, decomposition.num_channels
    (4, 8)
    """
    if len(species) == 0:
        raise ValueError("species must contain at least one atomic number.")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}.")
    if max_rank is not None and max_rank < 1:
        raise ValueError(f"max_rank must be at least 1, got {max_rank}.")
    if c6ab.shape != cn_ref.shape:
        raise ValueError(
            f"c6ab and cn_ref must have the same shape, got {c6ab.shape} and {cn_ref.shape}."
        )
    if c6ab.ndim != 4 or c6ab.shape[0] != c6ab.shape[1]:
        raise ValueError(
            f"Expected reference tables of shape (max_z + 1, max_z + 1, n_ref, n_ref), "
            f"got {c6ab.shape}."
        )

    unique_species = np.unique(np.asarray(species, dtype=np.int64))
    max_z = c6ab.shape[0] - 1
    out_of_range = unique_species[(unique_species < 1) | (unique_species > max_z)]
    if out_of_range.size:
        raise ValueError(
            f"Atomic numbers {out_of_range.tolist()} are outside the reference table, "
            f"which covers 1..{max_z}."
        )

    key = (
        _table_fingerprint(c6ab, cn_ref),
        tuple(int(z) for z in unique_species),
        float(tol),
        max_rank,
    )
    cached = _CACHE.get(key)
    if cached is not None:
        _CACHE.move_to_end(key)
        return cached

    cnref, valid = extract_species_reference_cn(c6ab, cn_ref, unique_species)
    block = _build_block_matrix(c6ab, unique_species, valid)

    eigvals, eigvecs = np.linalg.eigh(block)
    # eigh returns ascending eigenvalues; the energy sum is signed, so rank order follows
    # magnitude rather than value.
    order = np.argsort(np.abs(eigvals))[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    rank, achieved = _select_rank(block, eigvals, eigvecs, tol, max_rank)

    n_species, n_ref = valid.shape
    factors = eigvecs[:, :rank].reshape(n_species, n_ref, rank).copy()
    # Unused reference slots span the null space of the block matrix, so their eigenvector
    # entries are zero up to round-off. Clearing them exactly means a consumer that forgets
    # the validity mask still gets the right coefficient rather than a tiny spurious one.
    factors[~valid] = 0.0
    result = C6Decomposition(
        species=unique_species.astype(np.int32),
        eigs=np.ascontiguousarray(eigvals[:rank]),
        v_q=np.ascontiguousarray(factors),
        cnref=cnref,
        valid=valid,
        species_map=_build_species_map(unique_species, max_z),
        max_relative_error=achieved,
    )

    _CACHE[key] = result
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return result


def _build_species_map(species: np.ndarray, max_z: int) -> np.ndarray:
    """Atomic number to channel index, ``-1`` for elements not covered."""
    species_map = np.full(max_z + 1, -1, dtype=np.int32)
    species_map[species] = np.arange(species.shape[0], dtype=np.int32)
    return species_map
