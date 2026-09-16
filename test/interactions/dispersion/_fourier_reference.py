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

"""NumPy reference implementation of FourierD3, used as the test oracle.

Two independent evaluations of the same energy live here:

``direct_lattice_energy``
    A brute-force real-space lattice sum over a supercell, with no mesh and no Fourier
    transform. This is the oracle the mesh implementation is measured against: because both
    consume the *same* coordination numbers and low-rank coefficients, any discrepancy is
    summation error alone rather than a difference in the underlying model.

``reciprocal_kernel`` and friends
    Scalar building blocks shared with the production kernels, kept here in a form that is
    easy to differentiate numerically.

Everything is float64 and unoptimised; correctness is the only goal.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "COORDINATION_STEEPNESS",
    "SERIES_CUTOFF",
    "modified_coordination_number",
    "coordination_pair_term",
    "low_rank_coefficients",
    "reciprocal_kernel",
    "direct_lattice_energy",
    "self_energy",
    "pair_potential",
]

# Base steepness of the D3 counting function. The modified form used by FourierD3 keeps this
# value out to the transition radius and then increases it without bound.
COORDINATION_STEEPNESS = 16.0

# The covalent-radius table folds in Grimme's 4/3 factor; the transition radius is defined on
# the bare radius, so it divides that factor back out.
COORDINATION_UNSCALE = 3.0 / 4.0

# Regulariser preventing division by zero as the separation reaches the cutoff.
COORDINATION_EPSILON = 1e-6

# Below this dimensionless argument the closed-form transforms lose precision to
# cancellation and the series form is used instead. Both agree to ~1e-14 at the crossover.
SERIES_CUTOFF = 0.1

# Taylor coefficients of N(x)/x for the r^-6 and r^-8 transforms. N(x) is entire, so these
# series are exact rather than asymptotic; seven terms hold the crossover error near 1e-14.
_SERIES_R6 = (
    1.0,
    0.0,
    -1.0 / 3.0,
    0.125,
    -1.0 / 60.0,
    0.0,
    1.0 / 5040.0,
)
_SERIES_R8 = (
    -0.54119610014619668,
    0.0,
    0.090199350024366076,
    0.0,
    -0.010888024707303151,
    1.0 / 360.0,
    -0.00025923868350721731,
)


def coordination_pair_term(distance, covalent_distance, r_cut):
    """Contribution of one neighbour to a coordination number.

    FourierD3 replaces D3's fixed-steepness counting function with one whose steepness grows
    without bound as the separation approaches the neighbour-list cutoff. The standard form
    has a non-zero limit at large separation, so truncating it leaves a discontinuity and the
    total energy never converges as the cutoff grows. This form reaches exactly zero at
    ``r_cut``, which is what makes the coordination numbers well defined independently of the
    list.

    Parameters
    ----------
    distance : float or np.ndarray
        Interatomic separation.
    covalent_distance : float or np.ndarray
        Sum of the two covalent radii.
    r_cut : float
        Neighbour-list cutoff. Must match the radius the list was actually built with:
        the counting function reaches zero here, not at some radius of its own.

    Returns
    -------
    float or np.ndarray
        Counting-function value in ``[0, 1]``.
    """
    transition = 0.5 * (COORDINATION_UNSCALE * covalent_distance + r_cut)
    stabilised = np.maximum(distance, transition)
    gap = r_cut - stabilised
    steepness = COORDINATION_STEEPNESS + (stabilised - transition) ** 2 / (
        gap * gap + COORDINATION_EPSILON
    )
    argument = steepness * (covalent_distance / distance - 1.0)
    # Branchwise logistic: the steepness diverges at the cutoff, so a plain
    # 1/(1+exp(-x)) overflows there. Each branch keeps the exponent non-positive.
    positive = np.exp(-np.abs(argument))
    return np.where(
        argument >= 0.0, 1.0 / (1.0 + positive), positive / (1.0 + positive)
    )


def modified_coordination_number(positions, numbers, rcov, edges, shifts, cell, r_cut):
    """Coordination numbers from a directed edge list.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Atomic positions.
    numbers : np.ndarray, shape (N,)
        Atomic numbers, used to look up covalent radii.
    rcov : np.ndarray, shape (max_z + 1,)
        Covalent radii indexed by atomic number.
    edges : np.ndarray, shape (E, 2)
        Directed ``(source, target)`` pairs. Both orientations must be present for the
        coordination numbers to come out symmetric.
    shifts : np.ndarray, shape (E, 3)
        Periodic image of the target, in lattice-vector units.
    cell : np.ndarray, shape (3, 3)
        Lattice vectors as rows.
    r_cut : float
        Neighbour-list cutoff.

    Returns
    -------
    np.ndarray, shape (N,)
        Coordination number per atom.
    """
    source, target = edges[:, 0], edges[:, 1]
    delta = positions[target] - positions[source] + shifts @ cell
    distance = np.linalg.norm(delta, axis=1)
    covalent = rcov[numbers[source]] + rcov[numbers[target]]
    contribution = coordination_pair_term(distance, covalent, r_cut)
    coordination = np.zeros(positions.shape[0])
    np.add.at(coordination, source, contribution)
    return coordination


def low_rank_coefficients(coordination, channels, decomposition):
    """Per-atom separable coefficients and their coordination-number derivative.

    Parameters
    ----------
    coordination : np.ndarray, shape (N,)
        Coordination number per atom.
    channels : np.ndarray, shape (N,)
        Species channel per atom, as given by ``decomposition.species_map``.
    decomposition : C6Decomposition
        Factors from :func:`decompose_c6_reference`.

    Returns
    -------
    coefficients : np.ndarray, shape (N, rank)
        Per-atom coefficients :math:`c_6[i, \\ell]`.
    derivative : np.ndarray, shape (N, rank)
        :math:`\\partial c_6[i, \\ell] / \\partial \\theta_i`.
    """
    from nvalchemiops.interactions.dispersion._c6_decomposition import K3_WEIGHT

    reference = decomposition.cnref[channels]
    valid = decomposition.valid[channels]
    delta = coordination[:, None] - reference
    weights = np.where(valid, np.exp(-K3_WEIGHT * delta**2), 0.0)
    total = weights.sum(axis=1, keepdims=True)

    factors = decomposition.v_q[channels]
    numerator = np.einsum("np,npr->nr", weights, factors)
    coefficients = numerator / total

    # d/dtheta of a softmax-weighted mean: the weight derivative is -2*k3*delta*w, and the
    # normalisation contributes the mean of that same factor.
    weight_derivative = -2.0 * K3_WEIGHT * delta * weights
    numerator_derivative = np.einsum("np,npr->nr", weight_derivative, factors)
    total_derivative = weight_derivative.sum(axis=1, keepdims=True)
    derivative = (numerator_derivative - coefficients * total_derivative) / total
    return coefficients, derivative


def _series(coefficients, x):
    """Evaluate a truncated power series by Horner's rule."""
    total = np.zeros_like(x)
    for coefficient in reversed(coefficients):
        total = total * x + coefficient
    return total


def _shape_functions(x):
    """``N(x)/x`` for the two transforms, on the branch that is numerically stable at ``x``.

    ``N(x)`` vanishes at the origin, so the closed forms divide two quantities that both go
    to zero and lose all precision for small ``x``. The series is the same entire function,
    just evaluated in a form that does not cancel.
    """
    x = np.asarray(x, dtype=np.float64)
    small = x < SERIES_CUTOFF
    safe = np.where(small, 1.0, x)

    root3 = np.sqrt(3.0)
    n6 = np.exp(-safe) - 2.0 * np.exp(-safe / 2.0) * np.cos(
        np.pi / 3.0 + safe * root3 / 2.0
    )
    sin8, cos8 = np.sin(np.pi / 8.0), np.cos(np.pi / 8.0)
    n8 = np.exp(-safe * sin8) * np.cos(np.pi / 4.0 + safe * cos8) + np.exp(
        -safe * cos8
    ) * np.cos(3.0 * np.pi / 4.0 + safe * sin8)

    g6 = np.where(small, _series(_SERIES_R6, x), n6 / safe)
    g8 = np.where(small, _series(_SERIES_R8, x), n8 / safe)
    return g6, g8


def reciprocal_kernel(k_norm, r0, sqrt_q_product, s6, s8):
    r"""Fourier transform of the damped dispersion potential.

    Becke-Johnson damping makes the real-space potential
    :math:`s_6/(r^6 + R_0^6) + 3 s_8 \sqrt{Q_A Q_B}/(r^8 + R_0^8)` bounded and absolutely
    integrable, so its three-dimensional transform exists in closed form and decays
    exponentially. That is what lets the whole dispersion sum be evaluated on a mesh with no
    real-space cutoff.

    The transform is an exponentially damped *oscillation*: it changes sign, the first zero
    falling near :math:`k R_0 \approx 4.3`. Only the envelope decays monotonically.

    Parameters
    ----------
    k_norm : float or np.ndarray
        Wave-vector magnitude.
    r0 : float or np.ndarray
        Damping radius :math:`R_0 = a_1 \sqrt{3 \sqrt{Q_A Q_B}} + a_2`.
    sqrt_q_product : float or np.ndarray
        :math:`\sqrt{Q_A Q_B}`.
    s6, s8 : float
        Functional-dependent scaling factors.

    Returns
    -------
    float or np.ndarray
        Kernel value :math:`K_{AB}(|k|)`.
    """
    x = np.asarray(k_norm, dtype=np.float64) * r0
    g6, g8 = _shape_functions(x)
    term6 = 2.0 * np.pi**2 / (3.0 * r0**3) * g6
    term8 = -(np.pi**2) / r0**5 * g8
    return s6 * term6 + 3.0 * s8 * sqrt_q_product * term8


def pair_potential(distance, r0, sqrt_q_product, s6, s8):
    """Real-space damped dispersion potential, the transform's counterpart.

    Finite at zero separation, which is why the self pair has to be handled explicitly
    rather than diverging out of the sum.
    """
    return s6 / (distance**6 + r0**6) + 3.0 * s8 * sqrt_q_product / (
        distance**8 + r0**8
    )


def _damping_radius(sqrt_q_a, sqrt_q_b, a1, a2):
    """Becke-Johnson damping radius for a species pair."""
    return a1 * np.sqrt(3.0 * sqrt_q_a * sqrt_q_b) + a2


def self_energy(coefficients, channels, decomposition, sqrt_q, s6, s8, a1, a2):
    r"""Self-interaction term.

    The reciprocal-space sum runs over every ordered pair including :math:`i = j` at zero
    separation, which no physical pair sum contains. Under Becke-Johnson damping that term is
    finite, so it is added back here with the opposite sign.

    Because it is quadratic in the coefficients, and the coefficients depend on coordination
    number, this term contributes to the coordination-number derivative as well as to the
    energy. It must be folded in before the chain rule runs, not applied afterwards as a
    scalar correction.

    Returns
    -------
    energy : float
        Self-energy contribution.
    d_coefficients : np.ndarray, shape (N, rank)
        Its derivative with respect to the per-atom coefficients.
    """
    q = sqrt_q[channels]
    r0 = _damping_radius(q, q, a1, a2)
    phi_zero = pair_potential(0.0, r0, q * q, s6, s8)
    energy = 0.5 * float(
        np.einsum("r,nr,n->", decomposition.eigs, coefficients**2, phi_zero)
    )
    d_coefficients = decomposition.eigs[None, :] * coefficients * phi_zero[:, None]
    return energy, d_coefficients


def direct_lattice_energy(
    positions,
    channels,
    coefficients,
    decomposition,
    sqrt_q,
    cell,
    s6,
    s8,
    a1,
    a2,
    cutoff,
    tail_correction=True,
):
    r"""Brute-force real-space lattice sum of the low-rank dispersion energy.

    Evaluates

    .. math::

        E = -\tfrac{1}{2} \sum_{\ell} \lambda_{\ell}
            \sideset{}{'}\sum_{i, j, \mathbf{T}}
            c_6[i,\ell]\, c_6[j,\ell]\,
            \varphi_{Z_i Z_j}(|\mathbf{r}_{ij} + \mathbf{T}|)

    over every pair separation within ``cutoff``, where the primed sum omits the
    ``(i == j, T == 0)`` term. Omitting it is what makes this the physical energy: the mesh
    evaluation includes that term and cancels it with the self-energy, so an unprimed sum
    would sit a coordination-number-dependent offset away and no mesh refinement would close
    the gap.

    Cost is ``O(N^2 \times \text{images})``, so this is for small cells only.

    Parameters
    ----------
    cutoff : float
        Spherical cutoff on the pair separation. A spherical cutoff, rather than a block of
        lattice images, is what makes the tail estimate below well defined: a block includes
        its corners, which would then be counted twice.
    tail_correction : bool, default=True
        Add an analytic estimate of everything beyond ``cutoff``. Truncating a ``1/r^6``
        lattice sum leaves an error decaying only as ``1/R^3``, so reaching the accuracy a
        convergence study needs would otherwise take a prohibitive cutoff. Treating the pair
        distribution as uniform beyond the cutoff, the remainder integrates to

        .. math::

            -\frac{1}{2 \Omega} \sum_{ij} w_{ij}
             \left[ \frac{4 \pi s_6}{3 R_c^3}
                   + \frac{12 \pi s_8 \sqrt{Q_i Q_j}}{5 R_c^5} \right].

    Returns
    -------
    float
        Dispersion energy.
    """
    q = sqrt_q[channels]
    r0 = _damping_radius(q[:, None], q[None, :], a1, a2)
    q_product = q[:, None] * q[None, :]
    pair_weight = np.einsum(
        "r,ir,jr->ij", decomposition.eigs, coefficients, coefficients
    )

    volume = abs(float(np.linalg.det(cell)))
    widths = volume / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
    # Enough images that every separation within the cutoff is represented, allowing for the
    # largest intra-cell displacement.
    span = float(np.abs(positions).max()) * 2.0
    reach = [int(np.ceil((cutoff + span) / width)) for width in widths]
    grids = np.meshgrid(*[np.arange(-n, n + 1) for n in reach], indexing="ij")
    translations = np.stack(grids, axis=-1).reshape(-1, 3) @ cell

    total = 0.0
    for translation in translations:
        delta = positions[None, :, :] + translation - positions[:, None, :]
        distance = np.linalg.norm(delta, axis=-1)
        inside = distance <= cutoff
        if not np.any(translation):
            np.fill_diagonal(inside, False)
        if not inside.any():
            continue
        potential = np.where(
            inside,
            pair_potential(np.where(inside, distance, 1.0), r0, q_product, s6, s8),
            0.0,
        )
        total += float(np.sum(pair_weight * potential))

    energy = -0.5 * total
    if tail_correction:
        energy -= (
            0.5
            / volume
            * (
                4.0 * np.pi * s6 * float(np.sum(pair_weight)) / (3.0 * cutoff**3)
                + 12.0
                * np.pi
                * s8
                * float(np.sum(pair_weight * q_product))
                / (5.0 * cutoff**5)
            )
        )
    return energy
