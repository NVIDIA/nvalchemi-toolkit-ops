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
Multipolar Ewald Real-Space Kernels
===================================

Warp kernels for the real-space half of the Ewald summation with multipole
source terms. The :math:`l_{\max} = 0` (charges-only) branch reproduces the
monopole Ewald real-space kernels in
:mod:`nvalchemiops.interactions.electrostatics.ewald_kernels` bit-for-bit
(float64).

The pair interaction between two Gaussian-smeared multipole distributions
decomposes into a sum over interaction tensors :math:`T^{(n)}` that depend
only on the separation vector :math:`\mathbf{r}` and the Ewald splitting
parameter :math:`\alpha`:

.. math::

    E_{ij} = \tfrac{1}{2} \sum_{l_1, m_1, l_2, m_2}
             Q_{l_1, m_1}^{\,i} \,
             T_{l_1 m_1, l_2 m_2}(\mathbf{r}_{ij}, \alpha) \,
             Q_{l_2, m_2}^{\,j}.

Each :math:`T^{(n)}(r, \alpha)` is obtained by taking :math:`n` spatial
gradients of the rank-0 scalar tensor

.. math::

    T^{(0)}(r, \alpha) = \frac{\mathrm{erfc}(\alpha r)}{r}.
"""

from __future__ import annotations

import math
from typing import Any

import warp as wp

from nvalchemiops.math import wp_erfc
from nvalchemiops.warp_dispatch import register_overloads

# 2 / sqrt(pi), precomputed for the A(r) coefficient in the tensor helpers.
_TWO_OVER_SQRT_PI = wp.constant(wp.float64(2.0 / math.sqrt(math.pi)))

# =============================================================================
# Interaction-tensor building blocks (@wp.func helpers)
# =============================================================================


@wp.func
def damped_coulomb_T0(distance: wp.float64, alpha: wp.float64) -> wp.float64:
    r"""Rank-0 damped-Coulomb interaction tensor.

    .. math::

        T^{(0)}(r, \alpha) = \frac{\mathrm{erfc}(\alpha r)}{r}.

    This is the short-range half of the Ewald split :math:`1/r =
    \mathrm{erfc}(\alpha r)/r + \mathrm{erf}(\alpha r)/r`. Callers are
    responsible for guarding against ``distance == 0``; the function itself
    does not special-case the singularity.

    Parameters
    ----------
    distance
        Pair separation :math:`r = |\mathbf{r}_j - \mathbf{r}_i|`, ``float64``.
    alpha
        Ewald splitting parameter :math:`\alpha`, ``float64``.

    Returns
    -------
    wp.float64
        :math:`\mathrm{erfc}(\alpha r)/r`.
    """
    return wp_erfc(alpha * distance) / distance


@wp.func
def damped_coulomb_T1(
    r_vec: wp.vec3d, distance: wp.float64, alpha: wp.float64
) -> wp.vec3d:
    r"""Rank-1 damped-Coulomb interaction tensor :math:`T^{(1)}(\mathbf{r}) = \nabla K(|\mathbf{r}|)`.

    Factored form:

    .. math::

        T^{(1)}(\mathbf{r}) \;=\; -\frac{A(r)}{r} \, \mathbf{r},
        \qquad A(r) \;=\; \frac{\mathrm{erfc}(\alpha r)}{r^2}
                           + \frac{2\alpha}{\sqrt{\pi}}\,
                           \frac{e^{-\alpha^2 r^2}}{r}.

    The pair-energy contribution from a charge ``q_i`` interacting with a dipole
    ``mu_j`` across a separation vector :math:`\mathbf{r}_{ij}` is
    :math:`q_i \, \mathbf{\mu}_j \cdot T^{(1)}(\mathbf{r}_{ij})`.

    Parameters
    ----------
    r_vec
        Separation vector :math:`\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i`
        in ``float64`` Cartesian components.
    distance
        Its magnitude :math:`|\mathbf{r}_{ij}|`. Callers are responsible for
        guarding against ``distance == 0``.
    alpha
        Ewald splitting parameter.

    Returns
    -------
    wp.vec3d
        The three components of :math:`T^{(1)}(\mathbf{r})`.
    """
    alpha_r = alpha * distance
    erfc_ar = wp_erfc(alpha_r)
    exp_term = wp.exp(-alpha_r * alpha_r)

    inv_r = wp.float64(1.0) / distance
    inv_r2 = inv_r * inv_r

    # A(r) = erfc(αr)/r^2 + (2α/√π) · exp(-α²r²) / r
    a_scalar = erfc_ar * inv_r2 + _TWO_OVER_SQRT_PI * alpha * exp_term * inv_r

    # T^(1) = -(A/r) · r_vec
    neg_a_over_r = -a_scalar * inv_r
    return wp.vec3d(
        neg_a_over_r * r_vec[0],
        neg_a_over_r * r_vec[1],
        neg_a_over_r * r_vec[2],
    )


# =============================================================================
# GTO-Ewald radial helpers
# =============================================================================
# The Ewald split for Gaussian-smeared (GTO) sources uses
# ``T^(0)(r; σ, α) = [erf(ar) − erf(br)] / r`` with a = 1/(2σ),
# b = 1/(2σ_c), σ_c = √(σ² + 1/(4α²)). Every tensor derivative decomposes
# linearly as ``[x-term at a] − [x-term at b]``, so the kernel needs the
# single-parameter radial helpers evaluated twice and subtracted.


@wp.func
def _gto_ewald_A_single(r: wp.float64, x: wp.float64) -> wp.vec4d:
    r"""Radial helpers :math:`(A, A', A'', A''')` at single smearing :math:`x`.

    These are the coefficients used to assemble the multipole interaction
    tensors from the scalar kernel :math:`f(r; x) = \mathrm{erf}(xr) / r`:

    .. math::

        A(r; x)    &= \frac{\mathrm{erf}(xr)}{r^2}
                       - \frac{2x\, e^{-x^2 r^2}}{\sqrt{\pi}\, r} \\
        A'(r; x)   &= \frac{4x^3 e^{-x^2 r^2}}{\sqrt{\pi}}
                       + \frac{4x\, e^{-x^2 r^2}}{\sqrt{\pi}\, r^2}
                       - \frac{2\, \mathrm{erf}(xr)}{r^3} \\
        A''(r; x)  &= \frac{6\, \mathrm{erf}(xr)}{r^4}
                       - \frac{12x\, e}{\sqrt{\pi}\, r^3}
                       - \frac{8x^3 e}{\sqrt{\pi}\, r}
                       - \frac{8x^5 r\, e}{\sqrt{\pi}} \\
        A'''(r; x) &= -\frac{24\, \mathrm{erf}(xr)}{r^5}
                       + \frac{48x\, e}{\sqrt{\pi}\, r^4}
                       + \frac{32x^3 e}{\sqrt{\pi}\, r^2}
                       + \frac{8x^5 e}{\sqrt{\pi}}
                       + \frac{16x^7 r^2 e}{\sqrt{\pi}}

    Uses :math:`\mathrm{erf}(xr) = 1 - \mathrm{erfc}(xr)`. Catastrophic
    cancellation is bounded because the kernel guards against ``r < 1e-8``
    and typical pair distances keep :math:`xr` above ~0.01.

    Parameters
    ----------
    r, x : wp.float64
        Pair separation and smearing parameter.

    Returns
    -------
    wp.vec4d
        :math:`(A, A', A'', A''')` at this single :math:`x`.
    """
    xr = x * r
    erf_xr = wp.float64(1.0) - wp_erfc(xr)
    exp_term = wp.exp(-xr * xr)

    inv_r = wp.float64(1.0) / r
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r * inv_r2
    inv_r4 = inv_r2 * inv_r2
    inv_r5 = inv_r2 * inv_r3

    x2 = x * x
    x3 = x * x2
    x5 = x2 * x3
    x7 = x2 * x5

    # two_isp = 2/√π; 4x³/√π = 2·two_isp·x³, etc.
    two_isp = _TWO_OVER_SQRT_PI

    a0 = erf_xr * inv_r2 - two_isp * x * exp_term * inv_r
    a1 = (
        wp.float64(2.0) * two_isp * x3 * exp_term
        + wp.float64(2.0) * two_isp * x * exp_term * inv_r2
        - wp.float64(2.0) * erf_xr * inv_r3
    )
    a2 = (
        wp.float64(6.0) * erf_xr * inv_r4
        - wp.float64(6.0) * two_isp * x * exp_term * inv_r3
        - wp.float64(4.0) * two_isp * x3 * exp_term * inv_r
        - wp.float64(4.0) * two_isp * x5 * r * exp_term
    )
    a3 = (
        -wp.float64(24.0) * erf_xr * inv_r5
        + wp.float64(24.0) * two_isp * x * exp_term * inv_r4
        + wp.float64(16.0) * two_isp * x3 * exp_term * inv_r2
        + wp.float64(4.0) * two_isp * x5 * exp_term
        + wp.float64(8.0) * two_isp * x7 * r * r * exp_term
    )
    return wp.vec4d(a0, a1, a2, a3)


@wp.func
def _gto_ewald_t0(r: wp.float64, a: wp.float64, b: wp.float64) -> wp.float64:
    r"""Scalar GTO-Ewald kernel ``T^(0)(r; σ, α) = [erf(ar) − erf(br)] / r``.

    Equivalent to ``[erfc(br) − erfc(ar)] / r`` — avoids the 1 − erfc
    cancellation since ``a > b`` means ``erfc(ar) < erfc(br)`` and the
    subtraction is numerically well-conditioned.
    """
    return (wp_erfc(b * r) - wp_erfc(a * r)) / r


@wp.func
def _gto_ewald_ab(sigma: wp.float64, alpha: wp.float64) -> wp.vec2d:
    r"""Precompute ``(a, b) = (1/(2σ), 1/(2σ_c))`` from ``(σ, α)``.

    ``σ_c = √(σ² + 1/(4α²))`` is the combined GTO + Ewald smearing.
    """
    sigma_c = wp.sqrt(sigma * sigma + wp.float64(0.25) / (alpha * alpha))
    return wp.vec2d(
        wp.float64(0.5) / sigma,
        wp.float64(0.5) / sigma_c,
    )


# -----------------------------------------------------------------------------
# Fused per-pair physics: shared energy + radial-derivative factors.
# -----------------------------------------------------------------------------


@wp.struct
class _MonopolePairTerms:
    """Fused lmax=0 per-pair physics terms.

    `t0`       — energy factor `(erfc(br) − erfc(ar)) / r`.
    `a_scalar` — radial derivative `-dt0/dr` = `A(r,a)[0] − A(r,b)[0]`,
                 where `A` is the lmax=0 component of `_gto_ewald_A_single`.
                 Used to assemble position-gradient contributions.
    """

    t0: wp.float64
    a_scalar: wp.float64


@wp.func
def _gto_ewald_monopole_pair_terms_fused(
    r: wp.float64, a: wp.float64, b: wp.float64
) -> _MonopolePairTerms:
    """Fused per-pair `(t0, a_scalar)` sharing erfc evaluations.

    Computes both quantities from two `wp_erfc` calls (vs four for the
    separate energy + analytical-backward path), sharing `inv_r`,
    `inv_r2`, and the `erf = 1 - erfc` identity. Only the `[0]` component
    of `_gto_ewald_A_single` is needed for lmax=0 forces.
    """
    erfc_ar = wp_erfc(a * r)
    erfc_br = wp_erfc(b * r)
    inv_r = wp.float64(1.0) / r
    inv_r2 = inv_r * inv_r

    t0 = (erfc_br - erfc_ar) * inv_r

    # erf(xr) = 1 - erfc(xr); reuse the erfcs above instead of calling
    # wp.erf separately.
    erf_ar = wp.float64(1.0) - erfc_ar
    erf_br = wp.float64(1.0) - erfc_br
    exp_a = wp.exp(-a * a * r * r)
    exp_b = wp.exp(-b * b * r * r)
    a_a_0 = erf_ar * inv_r2 - _TWO_OVER_SQRT_PI * a * exp_a * inv_r
    a_b_0 = erf_br * inv_r2 - _TWO_OVER_SQRT_PI * b * exp_b * inv_r
    a_scalar = a_a_0 - a_b_0

    out = _MonopolePairTerms()
    out.t0 = t0
    out.a_scalar = a_scalar
    return out


@wp.struct
class _DipoleRadialTerms:
    """Fused lmax=1 radial helpers — sharing erfc + exp evaluations.

    `t0`             — energy factor `(erfc(br) − erfc(ar)) / r`.
    `a_scalar`       — `A(r,a)[0] − A(r,b)[0]` (forward energy + gradients).
    `a_prime`        — `A(r,a)[1] − A(r,b)[1]` (forward energy + gradients).
    `a_double_prime` — `A(r,a)[2] − A(r,b)[2]` (used only by position-gradient
                        c3 coefficient; computed unconditionally because the
                        marginal cost relative to the shared erfc/exp
                        evaluations is small).
    """

    t0: wp.float64
    a_scalar: wp.float64
    a_prime: wp.float64
    a_double_prime: wp.float64


@wp.func
def _gto_ewald_dipole_pair_terms_fused(
    r: wp.float64, a: wp.float64, b: wp.float64
) -> _DipoleRadialTerms:
    """Fused lmax=1 per-pair radial helpers `(t0, a_scalar, a_prime, a_double_prime)`.

    Computes all four radial helpers from a single set of erfc + exp
    evaluations, saving 2 erfcs and 2 exps per pair vs the separate
    energy + analytical-backward path.
    """
    erfc_ar = wp_erfc(a * r)
    erfc_br = wp_erfc(b * r)
    erf_ar = wp.float64(1.0) - erfc_ar
    erf_br = wp.float64(1.0) - erfc_br
    exp_ar = wp.exp(-a * a * r * r)
    exp_br = wp.exp(-b * b * r * r)

    inv_r = wp.float64(1.0) / r
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r * inv_r2
    inv_r4 = inv_r2 * inv_r2

    a_squared = a * a
    a_cubed = a_squared * a
    a_5 = a_squared * a_cubed
    b_squared = b * b
    b_cubed = b_squared * b
    b_5 = b_squared * b_cubed

    two_isp = _TWO_OVER_SQRT_PI

    t0 = (erfc_br - erfc_ar) * inv_r

    # A_single component [0]: erf(xr)/r² − 2x exp(-x²r²)/(√π·r)
    a_a_0 = erf_ar * inv_r2 - two_isp * a * exp_ar * inv_r
    a_b_0 = erf_br * inv_r2 - two_isp * b * exp_br * inv_r
    a_scalar = a_a_0 - a_b_0

    # A_single component [1]: 4x³ exp/√π + 4x exp/(√π·r²) − 2 erf(xr)/r³
    a_a_1 = (
        wp.float64(2.0) * two_isp * a_cubed * exp_ar
        + wp.float64(2.0) * two_isp * a * exp_ar * inv_r2
        - wp.float64(2.0) * erf_ar * inv_r3
    )
    a_b_1 = (
        wp.float64(2.0) * two_isp * b_cubed * exp_br
        + wp.float64(2.0) * two_isp * b * exp_br * inv_r2
        - wp.float64(2.0) * erf_br * inv_r3
    )
    a_prime = a_a_1 - a_b_1

    # A_single component [2]: 6 erf(xr)/r⁴ − 12x exp/(√π·r³)
    #                         − 8x³ exp/(√π·r) − 8x⁵·r·exp/√π
    a_a_2 = (
        wp.float64(6.0) * erf_ar * inv_r4
        - wp.float64(6.0) * two_isp * a * exp_ar * inv_r3
        - wp.float64(4.0) * two_isp * a_cubed * exp_ar * inv_r
        - wp.float64(4.0) * two_isp * a_5 * r * exp_ar
    )
    a_b_2 = (
        wp.float64(6.0) * erf_br * inv_r4
        - wp.float64(6.0) * two_isp * b * exp_br * inv_r3
        - wp.float64(4.0) * two_isp * b_cubed * exp_br * inv_r
        - wp.float64(4.0) * two_isp * b_5 * r * exp_br
    )
    a_double_prime = a_a_2 - a_b_2

    out = _DipoleRadialTerms()
    out.t0 = t0
    out.a_scalar = a_scalar
    out.a_prime = a_prime
    out.a_double_prime = a_double_prime
    return out


@wp.struct
class _DipolePairContrib:
    """Fused lmax=1 per-pair contribution: energy + all gradient slots.

    The lmax=1 fused kernels call `_dipole_pair_contribution_fused` once
    per surviving pair (`distance > 1e-8`) and gate the atomic_add of
    each gradient slot on their closure flags. All fields are computed
    unconditionally since they share intermediates.
    """

    energy: (
        wp.float64
    )  # full pair energy (charge-charge + charge-dipole + dipole-dipole)
    dPE_dq_i: wp.float64  # ∂(pair_energy)/∂q_i
    dPE_dq_j: wp.float64  # ∂(pair_energy)/∂q_j
    dPE_dmu_i: wp.vec3d  # ∂(pair_energy)/∂μ_i
    dPE_dmu_j: wp.vec3d  # ∂(pair_energy)/∂μ_j
    dPE_dr_j: (
        wp.vec3d
    )  # ∂(pair_energy)/∂r_j; ∂/∂r_i = -∂/∂r_j by translation invariance


@wp.func
def _dipole_pair_contribution_fused(
    r_vec: wp.vec3d,
    distance: wp.float64,
    qi: wp.float64,
    mu_i: wp.vec3d,
    qj: wp.float64,
    mu_j: wp.vec3d,
    a_coef: wp.float64,
    b_coef: wp.float64,
) -> _DipolePairContrib:
    """Per-pair lmax=1 GTO-Ewald: energy + all gradient pieces in one call.

    Uses `_gto_ewald_dipole_pair_terms_fused` for the radial helpers
    (sharing erfc+exp evaluations across t0/a/a'/a''). Returns a
    struct that the calling kernel destructures into atomic_add ops
    for whichever gradient slots are requested.
    """
    radial = _gto_ewald_dipole_pair_terms_fused(distance, a_coef, b_coef)
    t0 = radial.t0
    a_scalar = radial.a_scalar
    a_prime = radial.a_prime
    a_double_prime = radial.a_double_prime

    inv_r = wp.float64(1.0) / distance
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r * inv_r2
    inv_r4 = inv_r2 * inv_r2
    inv_r5 = inv_r3 * inv_r2

    neg_a_over_r = -a_scalar * inv_r
    t1x = neg_a_over_r * r_vec[0]
    t1y = neg_a_over_r * r_vec[1]
    t1z = neg_a_over_r * r_vec[2]

    mu_i_dot_r = mu_i[0] * r_vec[0] + mu_i[1] * r_vec[1] + mu_i[2] * r_vec[2]
    mu_j_dot_r = mu_j[0] * r_vec[0] + mu_j[1] * r_vec[1] + mu_j[2] * r_vec[2]
    mu_i_dot_mu_j = mu_i[0] * mu_j[0] + mu_i[1] * mu_j[1] + mu_i[2] * mu_j[2]
    mu_i_dot_T1 = t1x * mu_i[0] + t1y * mu_i[1] + t1z * mu_i[2]
    mu_j_dot_T1 = t1x * mu_j[0] + t1y * mu_j[1] + t1z * mu_j[2]

    c_diag = a_scalar * inv_r
    c_quad = a_prime * inv_r2 - a_scalar * inv_r3
    c3 = (
        a_double_prime * inv_r3
        - wp.float64(3.0) * a_prime * inv_r4
        + wp.float64(3.0) * a_scalar * inv_r5
    )

    mu_T2_mu = c_diag * mu_i_dot_mu_j + c_quad * mu_i_dot_r * mu_j_dot_r
    pair_energy = qi * qj * t0 + (qi * mu_j_dot_T1 - qj * mu_i_dot_T1) + mu_T2_mu

    dPE_dq_i = qj * t0 + mu_j_dot_T1
    dPE_dq_j = qi * t0 - mu_i_dot_T1

    cq_muj_r = c_quad * mu_j_dot_r
    cq_mui_r = c_quad * mu_i_dot_r
    dPE_dmu_i = wp.vec3d(
        -qj * t1x + c_diag * mu_j[0] + cq_muj_r * r_vec[0],
        -qj * t1y + c_diag * mu_j[1] + cq_muj_r * r_vec[1],
        -qj * t1z + c_diag * mu_j[2] + cq_muj_r * r_vec[2],
    )
    dPE_dmu_j = wp.vec3d(
        qi * t1x + c_diag * mu_i[0] + cq_mui_r * r_vec[0],
        qi * t1y + c_diag * mu_i[1] + cq_mui_r * r_vec[1],
        qi * t1z + c_diag * mu_i[2] + cq_mui_r * r_vec[2],
    )

    rad_coeff = (
        -qi * qj * c_diag
        - c_quad * (qi * mu_j_dot_r - qj * mu_i_dot_r)
        + c_quad * mu_i_dot_mu_j
        + c3 * mu_i_dot_r * mu_j_dot_r
    )
    dir_x = (
        -c_diag * (qi * mu_j[0] - qj * mu_i[0])
        + c_quad * mu_j_dot_r * mu_i[0]
        + c_quad * mu_i_dot_r * mu_j[0]
    )
    dir_y = (
        -c_diag * (qi * mu_j[1] - qj * mu_i[1])
        + c_quad * mu_j_dot_r * mu_i[1]
        + c_quad * mu_i_dot_r * mu_j[1]
    )
    dir_z = (
        -c_diag * (qi * mu_j[2] - qj * mu_i[2])
        + c_quad * mu_j_dot_r * mu_i[2]
        + c_quad * mu_i_dot_r * mu_j[2]
    )
    dPE_dr_j = wp.vec3d(
        rad_coeff * r_vec[0] + dir_x,
        rad_coeff * r_vec[1] + dir_y,
        rad_coeff * r_vec[2] + dir_z,
    )

    out = _DipolePairContrib()
    out.energy = pair_energy
    out.dPE_dq_i = dPE_dq_i
    out.dPE_dq_j = dPE_dq_j
    out.dPE_dmu_i = dPE_dmu_i
    out.dPE_dmu_j = dPE_dmu_j
    out.dPE_dr_j = dPE_dr_j
    return out


@wp.func
def _dipole_pair_energy_only(
    r_vec: wp.vec3d,
    distance: wp.float64,
    qi: wp.float64,
    mu_i: wp.vec3d,
    qj: wp.float64,
    mu_j: wp.vec3d,
    a_coef: wp.float64,
    b_coef: wp.float64,
) -> wp.float64:
    """Energy-only per-pair lmax=1 GTO-Ewald (no gradient arithmetic).

    Used by the all-flags-False fused kernel variant to match the
    existing forward-only kernel's per-pair work (avoids computing
    `a_double_prime` and the `c3` coefficient, which only feed
    position gradients).
    """
    inv_r = wp.float64(1.0) / distance
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r * inv_r2

    t0 = _gto_ewald_t0(distance, a_coef, b_coef)
    ra = _gto_ewald_A_single(distance, a_coef)
    rb = _gto_ewald_A_single(distance, b_coef)
    a_scalar = ra[0] - rb[0]
    a_prime = ra[1] - rb[1]

    neg_a_over_r = -a_scalar * inv_r
    mu_i_dot_r = mu_i[0] * r_vec[0] + mu_i[1] * r_vec[1] + mu_i[2] * r_vec[2]
    mu_j_dot_r = mu_j[0] * r_vec[0] + mu_j[1] * r_vec[1] + mu_j[2] * r_vec[2]
    mu_i_dot_T1 = neg_a_over_r * mu_i_dot_r
    mu_j_dot_T1 = neg_a_over_r * mu_j_dot_r

    mu_i_dot_mu_j = mu_i[0] * mu_j[0] + mu_i[1] * mu_j[1] + mu_i[2] * mu_j[2]
    c_diag = a_scalar * inv_r
    c_quad = a_prime * inv_r2 - a_scalar * inv_r3
    mu_T2_mu = c_diag * mu_i_dot_mu_j + c_quad * mu_i_dot_r * mu_j_dot_r

    return qi * qj * t0 + (qi * mu_j_dot_T1 - qj * mu_i_dot_T1) + mu_T2_mu


# =============================================================================
# l_max = 2 fused radial helpers
# =============================================================================
# Extends the lmax=1 helper to 6 components (t0, A, A', A'', A''', A'''') needed
# for the rank-3 and rank-4 multipole tensor contractions in the QQ, μQ
# channels.


@wp.struct
class _QuadrupoleRadialTerms:
    """Fused lmax=2 radial helpers — sharing erfc + exp evaluations.

    Same shape as ``_DipoleRadialTerms`` plus two extra entries:

    - ``a_triple_prime``   — ``A(r,a)[3] − A(r,b)[3]``: 4th radial
                              derivative coefficient (used by μQ channel
                              and by QQ K2 / K3 coefficients).
    - ``a_quadruple_prime`` — analogous 5th radial derivative; used by
                              QQ channel K3 coefficient.

    The energy-only path uses ``a, a', a'', a''', a''''`` to compute the
    Python-convention radial helpers ``T1..T4`` (with sign inversion:
    ``T_n_python = -a^(n-1)_kernel``) and assemble the QQ closed forms

        K1 = T2/r²  −  T1/r³                          (sym3(δδ))
        K2 = T3/r  −  3·T2/r²  +  3·T1/r³             (sym6(δr̂r̂))
        K3 = T4  −  6·T3/r  +  15·T2/r²  −  15·T1/r³  (r̂r̂r̂r̂)
    """

    t0: wp.float64
    a_scalar: wp.float64  # -T1_python (single-radial component)
    a_prime: wp.float64  # -T2_python
    a_double_prime: wp.float64  # -T3_python
    a_triple_prime: wp.float64  # -T4_python
    a_quadruple_prime: wp.float64  # -T5_python (= A'''' at single x)


@wp.func
def _gto_ewald_A_single_v5(r: wp.float64, x: wp.float64) -> _QuadrupoleRadialTerms:
    r"""Single-parameter A(r; x) helpers, 5-component version.

    Returns a struct with ``a0`` through ``a4`` corresponding to

    .. math::

        A(r; x)     &= \frac{\mathrm{erf}(xr)}{r^2} - \frac{2x\, e}{\sqrt{\pi}\, r} \\
        A'(r; x)    &= \frac{4x^3 e}{\sqrt{\pi}} + \frac{4x\, e}{\sqrt{\pi}\, r^2}
                        - \frac{2\, \mathrm{erf}(xr)}{r^3} \\
        A''(r; x)   &= \frac{6\, \mathrm{erf}(xr)}{r^4} - \frac{12x\, e}{\sqrt{\pi}\, r^3}
                        - \frac{8x^3 e}{\sqrt{\pi}\, r} - \frac{8x^5 r\, e}{\sqrt{\pi}} \\
        A'''(r; x)  &= -\frac{24\, \mathrm{erf}(xr)}{r^5} + \frac{48x\, e}{\sqrt{\pi}\, r^4}
                        + \frac{32x^3 e}{\sqrt{\pi}\, r^2} + \frac{8x^5 e}{\sqrt{\pi}}
                        + \frac{16x^7 r^2 e}{\sqrt{\pi}} \\
        A''''(r; x) &= \frac{120\, \mathrm{erf}(xr)}{r^6}
                        - \frac{2}{\sqrt{\pi}}\Big(\frac{120 x\, e}{r^5}
                        + \frac{80 x^3 e}{r^3} + \frac{32 x^5 e}{r}
                        - 8 x^7 r\, e + 16 x^9 r^3 e\Big)

    where :math:`e = e^{-x^2 r^2}`. Used by
    :func:`_gto_ewald_quadrupole_pair_terms_fused` to assemble the
    6-component LMAX=2 helper struct sharing erfc + exp evaluations. The
    ``t0`` field is uninitialized here (set by the combined helper that
    subtracts at-a vs at-b); only the 5 ``a*`` fields are populated.
    """
    xr = x * r
    erf_xr = wp.float64(1.0) - wp_erfc(xr)
    exp_term = wp.exp(-xr * xr)
    inv_r = wp.float64(1.0) / r
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r * inv_r2
    inv_r4 = inv_r2 * inv_r2
    inv_r5 = inv_r2 * inv_r3
    inv_r6 = inv_r3 * inv_r3
    two_isp = _TWO_OVER_SQRT_PI
    x2 = x * x
    x3 = x * x2
    x5 = x2 * x3
    x7 = x2 * x5
    x9 = x2 * x7

    a0 = erf_xr * inv_r2 - two_isp * x * exp_term * inv_r
    a1 = (
        wp.float64(2.0) * two_isp * x3 * exp_term
        + wp.float64(2.0) * two_isp * x * exp_term * inv_r2
        - wp.float64(2.0) * erf_xr * inv_r3
    )
    a2 = (
        wp.float64(6.0) * erf_xr * inv_r4
        - wp.float64(6.0) * two_isp * x * exp_term * inv_r3
        - wp.float64(4.0) * two_isp * x3 * exp_term * inv_r
        - wp.float64(4.0) * two_isp * x5 * r * exp_term
    )
    a3 = (
        -wp.float64(24.0) * erf_xr * inv_r5
        + wp.float64(24.0) * two_isp * x * exp_term * inv_r4
        + wp.float64(16.0) * two_isp * x3 * exp_term * inv_r2
        + wp.float64(4.0) * two_isp * x5 * exp_term
        + wp.float64(8.0) * two_isp * x7 * r * r * exp_term
    )
    a4 = (
        wp.float64(120.0) * erf_xr * inv_r6
        - wp.float64(120.0) * two_isp * x * exp_term * inv_r5
        - wp.float64(80.0) * two_isp * x3 * exp_term * inv_r3
        - wp.float64(32.0) * two_isp * x5 * exp_term * inv_r
        + wp.float64(8.0) * two_isp * x7 * r * exp_term
        - wp.float64(16.0) * two_isp * x9 * r * r * r * exp_term
    )

    out = _QuadrupoleRadialTerms()
    out.t0 = wp.float64(0.0)  # not used in single-x form; set by caller
    out.a_scalar = a0
    out.a_prime = a1
    out.a_double_prime = a2
    out.a_triple_prime = a3
    out.a_quadruple_prime = a4
    return out


@wp.func
def _gto_ewald_quadrupole_pair_terms_fused(
    r: wp.float64, a: wp.float64, b: wp.float64
) -> _QuadrupoleRadialTerms:
    """Fused per-pair radial helpers for LMAX=2 — shared erfc/exp pairs."""
    erfc_ar = wp_erfc(a * r)
    erfc_br = wp_erfc(b * r)
    erf_ar = wp.float64(1.0) - erfc_ar
    erf_br = wp.float64(1.0) - erfc_br
    exp_ar = wp.exp(-a * a * r * r)
    exp_br = wp.exp(-b * b * r * r)

    inv_r = wp.float64(1.0) / r
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r * inv_r2
    inv_r4 = inv_r2 * inv_r2
    inv_r5 = inv_r2 * inv_r3
    inv_r6 = inv_r3 * inv_r3
    two_isp = _TWO_OVER_SQRT_PI

    a2 = a * a
    a3 = a * a2
    a5 = a2 * a3
    a7 = a2 * a5
    a9 = a2 * a7
    b2 = b * b
    b3 = b * b2
    b5 = b2 * b3
    b7 = b2 * b5
    b9 = b2 * b7

    t0 = (erfc_br - erfc_ar) * inv_r

    a_a_0 = erf_ar * inv_r2 - two_isp * a * exp_ar * inv_r
    a_b_0 = erf_br * inv_r2 - two_isp * b * exp_br * inv_r
    a_scalar = a_a_0 - a_b_0

    a_a_1 = (
        wp.float64(2.0) * two_isp * a3 * exp_ar
        + wp.float64(2.0) * two_isp * a * exp_ar * inv_r2
        - wp.float64(2.0) * erf_ar * inv_r3
    )
    a_b_1 = (
        wp.float64(2.0) * two_isp * b3 * exp_br
        + wp.float64(2.0) * two_isp * b * exp_br * inv_r2
        - wp.float64(2.0) * erf_br * inv_r3
    )
    a_prime = a_a_1 - a_b_1

    a_a_2 = (
        wp.float64(6.0) * erf_ar * inv_r4
        - wp.float64(6.0) * two_isp * a * exp_ar * inv_r3
        - wp.float64(4.0) * two_isp * a3 * exp_ar * inv_r
        - wp.float64(4.0) * two_isp * a5 * r * exp_ar
    )
    a_b_2 = (
        wp.float64(6.0) * erf_br * inv_r4
        - wp.float64(6.0) * two_isp * b * exp_br * inv_r3
        - wp.float64(4.0) * two_isp * b3 * exp_br * inv_r
        - wp.float64(4.0) * two_isp * b5 * r * exp_br
    )
    a_double_prime = a_a_2 - a_b_2

    a_a_3 = (
        -wp.float64(24.0) * erf_ar * inv_r5
        + wp.float64(24.0) * two_isp * a * exp_ar * inv_r4
        + wp.float64(16.0) * two_isp * a3 * exp_ar * inv_r2
        + wp.float64(4.0) * two_isp * a5 * exp_ar
        + wp.float64(8.0) * two_isp * a7 * r * r * exp_ar
    )
    a_b_3 = (
        -wp.float64(24.0) * erf_br * inv_r5
        + wp.float64(24.0) * two_isp * b * exp_br * inv_r4
        + wp.float64(16.0) * two_isp * b3 * exp_br * inv_r2
        + wp.float64(4.0) * two_isp * b5 * exp_br
        + wp.float64(8.0) * two_isp * b7 * r * r * exp_br
    )
    a_triple_prime = a_a_3 - a_b_3

    a_a_4 = (
        wp.float64(120.0) * erf_ar * inv_r6
        - wp.float64(120.0) * two_isp * a * exp_ar * inv_r5
        - wp.float64(80.0) * two_isp * a3 * exp_ar * inv_r3
        - wp.float64(32.0) * two_isp * a5 * exp_ar * inv_r
        + wp.float64(8.0) * two_isp * a7 * r * exp_ar
        - wp.float64(16.0) * two_isp * a9 * r * r * r * exp_ar
    )
    a_b_4 = (
        wp.float64(120.0) * erf_br * inv_r6
        - wp.float64(120.0) * two_isp * b * exp_br * inv_r5
        - wp.float64(80.0) * two_isp * b3 * exp_br * inv_r3
        - wp.float64(32.0) * two_isp * b5 * exp_br * inv_r
        + wp.float64(8.0) * two_isp * b7 * r * exp_br
        - wp.float64(16.0) * two_isp * b9 * r * r * r * exp_br
    )
    a_quadruple_prime = a_a_4 - a_b_4

    out = _QuadrupoleRadialTerms()
    out.t0 = t0
    out.a_scalar = a_scalar
    out.a_prime = a_prime
    out.a_double_prime = a_double_prime
    out.a_triple_prime = a_triple_prime
    out.a_quadruple_prime = a_quadruple_prime
    return out


@wp.func
def _quadrupole_pair_energy_only(
    r_vec: wp.vec3d,
    distance: wp.float64,
    qi: wp.float64,
    mu_i: wp.vec3d,
    Q_i: wp.mat33d,
    qj: wp.float64,
    mu_j: wp.vec3d,
    Q_j: wp.mat33d,
    a_coef: wp.float64,
    b_coef: wp.float64,
) -> wp.float64:
    """Per-pair LMAX=2 GTO-Ewald energy in the kernel's r_vec convention.

    Computes all 6 channels (qq, qμ, qQ, μμ, μQ, QQ) using factored
    contractions:

    - rank-≤2 channels share the LMAX=1 closed forms with the added qQ
      pieces (``Qi:T2 = (a/r)·tr(Qi) + (a'/r² - a/r³)·r·Qi·r``).
    - μQ uses the factored ``μi · (Qj:T_αβγ)`` form.
    - QQ uses the K1/K2/K3 closed form.
    """
    radial = _gto_ewald_quadrupole_pair_terms_fused(distance, a_coef, b_coef)
    t0 = radial.t0
    a_scalar = radial.a_scalar
    a_prime = radial.a_prime
    a_2prime = radial.a_double_prime
    a_3prime = radial.a_triple_prime

    # Python radial-helper convention: T_n = -a^(n-1)_kernel.
    # T1 = -a, T2 = -a', T3 = -a'', T4 = -a''', T5 = -a''''
    T1 = -a_scalar
    T2 = -a_prime
    T3 = -a_2prime
    T4 = -a_3prime

    inv_r = wp.float64(1.0) / distance
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r * inv_r2

    A_val = T1 * inv_r  # = -a/r
    B_val = T2 - T1 * inv_r  # = -a' + a/r
    A_prime = T2 * inv_r - T1 * inv_r2
    B_prime = T3 - T2 * inv_r + T1 * inv_r2

    # ---- precomputed geometric scalars --------------------------------
    mu_i_dot_r = mu_i[0] * r_vec[0] + mu_i[1] * r_vec[1] + mu_i[2] * r_vec[2]
    mu_j_dot_r = mu_j[0] * r_vec[0] + mu_j[1] * r_vec[1] + mu_j[2] * r_vec[2]
    mu_i_dot_mu_j = mu_i[0] * mu_j[0] + mu_i[1] * mu_j[1] + mu_i[2] * mu_j[2]

    tr_Qi = Q_i[0, 0] + Q_i[1, 1] + Q_i[2, 2]
    tr_Qj = Q_j[0, 0] + Q_j[1, 1] + Q_j[2, 2]

    # Qi · r_vec, Qj · r_vec (3-vectors)
    Qi_r_0 = Q_i[0, 0] * r_vec[0] + Q_i[0, 1] * r_vec[1] + Q_i[0, 2] * r_vec[2]
    Qi_r_1 = Q_i[1, 0] * r_vec[0] + Q_i[1, 1] * r_vec[1] + Q_i[1, 2] * r_vec[2]
    Qi_r_2 = Q_i[2, 0] * r_vec[0] + Q_i[2, 1] * r_vec[1] + Q_i[2, 2] * r_vec[2]
    Qj_r_0 = Q_j[0, 0] * r_vec[0] + Q_j[0, 1] * r_vec[1] + Q_j[0, 2] * r_vec[2]
    Qj_r_1 = Q_j[1, 0] * r_vec[0] + Q_j[1, 1] * r_vec[1] + Q_j[1, 2] * r_vec[2]
    Qj_r_2 = Q_j[2, 0] * r_vec[0] + Q_j[2, 1] * r_vec[1] + Q_j[2, 2] * r_vec[2]

    r_Qi_r = r_vec[0] * Qi_r_0 + r_vec[1] * Qi_r_1 + r_vec[2] * Qi_r_2
    r_Qj_r = r_vec[0] * Qj_r_0 + r_vec[1] * Qj_r_1 + r_vec[2] * Qj_r_2

    # μi · Qj · r = Σ_α μi_α (Qj·r)_α
    mu_i_Qj_r = mu_i[0] * Qj_r_0 + mu_i[1] * Qj_r_1 + mu_i[2] * Qj_r_2
    mu_j_Qi_r = mu_j[0] * Qi_r_0 + mu_j[1] * Qi_r_1 + mu_j[2] * Qi_r_2

    # Qi:Qj double contraction
    Qi_Qj = (
        Q_i[0, 0] * Q_j[0, 0]
        + Q_i[0, 1] * Q_j[0, 1]
        + Q_i[0, 2] * Q_j[0, 2]
        + Q_i[1, 0] * Q_j[1, 0]
        + Q_i[1, 1] * Q_j[1, 1]
        + Q_i[1, 2] * Q_j[1, 2]
        + Q_i[2, 0] * Q_j[2, 0]
        + Q_i[2, 1] * Q_j[2, 1]
        + Q_i[2, 2] * Q_j[2, 2]
    )

    # r · (Qi·Qj) · r — used in QQ channel
    # (Qi·Qj)_αγ = Σ_β Qi_αβ Qj_βγ. Then r · this · r = Σ_αγ r_α (Qi·Qj)_αγ r_γ
    QiQj_00 = Q_i[0, 0] * Q_j[0, 0] + Q_i[0, 1] * Q_j[1, 0] + Q_i[0, 2] * Q_j[2, 0]
    QiQj_01 = Q_i[0, 0] * Q_j[0, 1] + Q_i[0, 1] * Q_j[1, 1] + Q_i[0, 2] * Q_j[2, 1]
    QiQj_02 = Q_i[0, 0] * Q_j[0, 2] + Q_i[0, 1] * Q_j[1, 2] + Q_i[0, 2] * Q_j[2, 2]
    QiQj_10 = Q_i[1, 0] * Q_j[0, 0] + Q_i[1, 1] * Q_j[1, 0] + Q_i[1, 2] * Q_j[2, 0]
    QiQj_11 = Q_i[1, 0] * Q_j[0, 1] + Q_i[1, 1] * Q_j[1, 1] + Q_i[1, 2] * Q_j[2, 1]
    QiQj_12 = Q_i[1, 0] * Q_j[0, 2] + Q_i[1, 1] * Q_j[1, 2] + Q_i[1, 2] * Q_j[2, 2]
    QiQj_20 = Q_i[2, 0] * Q_j[0, 0] + Q_i[2, 1] * Q_j[1, 0] + Q_i[2, 2] * Q_j[2, 0]
    QiQj_21 = Q_i[2, 0] * Q_j[0, 1] + Q_i[2, 1] * Q_j[1, 1] + Q_i[2, 2] * Q_j[2, 1]
    QiQj_22 = Q_i[2, 0] * Q_j[0, 2] + Q_i[2, 1] * Q_j[1, 2] + Q_i[2, 2] * Q_j[2, 2]
    r_QiQj_r = (
        r_vec[0] * QiQj_00 * r_vec[0]
        + r_vec[0] * QiQj_01 * r_vec[1]
        + r_vec[0] * QiQj_02 * r_vec[2]
        + r_vec[1] * QiQj_10 * r_vec[0]
        + r_vec[1] * QiQj_11 * r_vec[1]
        + r_vec[1] * QiQj_12 * r_vec[2]
        + r_vec[2] * QiQj_20 * r_vec[0]
        + r_vec[2] * QiQj_21 * r_vec[1]
        + r_vec[2] * QiQj_22 * r_vec[2]
    )

    # ----- channel qq -----
    E = qi * qj * t0

    # ----- channel qμ (kernel convention: i↔j swap from Python ref) -----
    # Python's Ta_α = (r_α/r) * T1. In our kernel:
    # E_qμ = qi·(μj·Ta) - qj·(μi·Ta) = (qi*mu_j_r - qj*mu_i_r) * T1/r
    E += (qi * mu_j_dot_r - qj * mu_i_dot_r) * T1 * inv_r

    # ----- channel μμ -----
    # E_μμ = -μi · Tab · μj where Tab_αβ = (T1/r)δ + (T2 - T1/r) r̂_α r̂_β
    # μi·Tab·μj = (T1/r) μi·μj + (T2 - T1/r) (mu_i_r mu_j_r)/r²
    mu_Tab_mu = A_val * mu_i_dot_mu_j + B_val * inv_r2 * mu_i_dot_r * mu_j_dot_r
    E += -mu_Tab_mu

    # ----- channel qQ -----
    # E_qQ = 0.5·qj·(Qi:Tab) + 0.5·qi·(Qj:Tab)
    # Qi:Tab = A_val * tr(Qi) + B_val * inv_r² * (r·Qi·r)
    Qi_Tab = A_val * tr_Qi + B_val * inv_r2 * r_Qi_r
    Qj_Tab = A_val * tr_Qj + B_val * inv_r2 * r_Qj_r
    E += wp.float64(0.5) * qj * Qi_Tab + wp.float64(0.5) * qi * Qj_Tab

    # ----- channel μQ (kernel convention: i↔j swap from Python ref) -----
    # E_μQ_kernel = 0.5 * [μi · (Qj:T_αβγ) - μj · (Qi:T_αβγ)]
    # μi · (Qj:T_αβγ) = (A'+B/r) (mu_i_Qj_r/r)
    #                  + (mu_i_r/r) * [B'*(r_Qj_r/r²) + (B/r)*(tr_Qj - 2*(r_Qj_r/r²))]
    C1 = A_prime + B_val * inv_r
    rhat_Qj_rhat = r_Qj_r * inv_r2
    rhat_Qi_rhat = r_Qi_r * inv_r2
    coef_rhat_j = B_prime * rhat_Qj_rhat + B_val * inv_r * (
        tr_Qj - wp.float64(2.0) * rhat_Qj_rhat
    )
    coef_rhat_i = B_prime * rhat_Qi_rhat + B_val * inv_r * (
        tr_Qi - wp.float64(2.0) * rhat_Qi_rhat
    )
    contrib_mu_Q_i = C1 * (mu_i_Qj_r * inv_r) + (mu_i_dot_r * inv_r) * coef_rhat_j
    contrib_mu_Q_j = C1 * (mu_j_Qi_r * inv_r) + (mu_j_dot_r * inv_r) * coef_rhat_i
    # The μ-Q (T³) energy is −0.5·(μi:∇³φ:Qj − μj:∇³φ:Qi); the sign matches
    # the density-integral physics and keeps the composite l=2 total
    # α-independent.
    E -= wp.float64(0.5) * (contrib_mu_Q_i - contrib_mu_Q_j)

    # ----- channel QQ -----
    # E_QQ = 0.25 * Qi:T_αβγδ:Qj where T_αβγδ has the K1/K2/K3 closed
    # form built from the T_n radials:
    #   K1 = T2/r²  − T1/r³
    #   K2 = T3/r  − 3·T2/r²  + 3·T1/r³
    #   K3 = T4   − 6·T3/r   + 15·T2/r²  − 15·T1/r³
    # Contraction: Qi:T:Qj = K1·(tr·tr + 2·Qi:Qj) + K2·(tr_Qi·r̂Qjr̂ + tr_Qj·r̂Qir̂ + 4·r̂QiQjr̂) + K3·(r̂Qir̂)·(r̂Qjr̂)
    K1 = T2 * inv_r2 - T1 * inv_r3
    K2 = T3 * inv_r - wp.float64(3.0) * T2 * inv_r2 + wp.float64(3.0) * T1 * inv_r3
    K3 = (
        T4
        - wp.float64(6.0) * T3 * inv_r
        + wp.float64(15.0) * T2 * inv_r2
        - wp.float64(15.0) * T1 * inv_r3
    )

    rhat_QiQj_rhat = r_QiQj_r * inv_r2
    QQ_contraction = (
        K1 * (tr_Qi * tr_Qj + wp.float64(2.0) * Qi_Qj)
        + K2
        * (
            tr_Qi * rhat_Qj_rhat
            + tr_Qj * rhat_Qi_rhat
            + wp.float64(4.0) * rhat_QiQj_rhat
        )
        + K3 * rhat_Qi_rhat * rhat_Qj_rhat
    )
    E += wp.float64(0.25) * QQ_contraction

    return E


# =============================================================================
# l_max = 2 fused per-pair contribution — energy + all 7 gradient slots
# =============================================================================
# Builds energy + (∂E/∂q_i, ∂E/∂q_j, ∂E/∂μ_i, ∂E/∂μ_j, ∂E/∂Q_i, ∂E/∂Q_j,
# ∂E/∂r_j).
#
# The quadrupole gradients use the "free-index" partial convention
# (∂/∂Q_i[α,β] treats each matrix entry as an independent variable);
# the resulting matrix is symmetric for the qQ and QQ channels but
# NOT for the μQ channel.


@wp.struct
class _QuadrupolePairContrib:
    """Fused lmax=2 per-pair contribution: energy + all gradient slots.

    The lmax=2 fused kernels call ``_quadrupole_pair_contribution_fused``
    once per surviving pair and gate each ``atomic_add`` on its closure
    flag.

    The quadrupole grad fields ``dPE_dQ_i`` / ``dPE_dQ_j`` hold the
    "free-index" partials (treating each Q matrix entry as independent).
    The qQ + QQ contributions are symmetric in (α,β); the μQ contribution
    is not. Upstream torch.autograd wrappers symmetrize when needed.
    """

    energy: wp.float64
    dPE_dq_i: wp.float64
    dPE_dq_j: wp.float64
    dPE_dmu_i: wp.vec3d
    dPE_dmu_j: wp.vec3d
    dPE_dQ_i: wp.mat33d
    dPE_dQ_j: wp.mat33d
    dPE_dr_j: wp.vec3d


@wp.func
def _quadrupole_pair_contribution_fused(
    r_vec: wp.vec3d,
    distance: wp.float64,
    qi: wp.float64,
    mu_i: wp.vec3d,
    Q_i: wp.mat33d,
    qj: wp.float64,
    mu_j: wp.vec3d,
    Q_j: wp.mat33d,
    a_coef: wp.float64,
    b_coef: wp.float64,
) -> _QuadrupolePairContrib:
    """Per-pair LMAX=2 GTO-Ewald: energy + all gradient pieces in one call."""
    radial = _gto_ewald_quadrupole_pair_terms_fused(distance, a_coef, b_coef)
    t0 = radial.t0
    a_scalar = radial.a_scalar  # = -T1 (Python convention)
    a_prime = radial.a_prime  # = -T2
    a_2prime = radial.a_double_prime  # = -T3
    a_3prime = radial.a_triple_prime  # = -T4
    a_4prime = radial.a_quadruple_prime  # = -T5

    T1 = -a_scalar
    T2 = -a_prime
    T3 = -a_2prime
    T4 = -a_3prime
    T5 = -a_4prime

    inv_r = wp.float64(1.0) / distance
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r * inv_r2
    inv_r4 = inv_r2 * inv_r2

    # Radial combinations (rank ≤ 3 helpers)
    A_val = T1 * inv_r  # T1/r
    B_val = T2 - T1 * inv_r  # T2 - T1/r
    A_prime = T2 * inv_r - T1 * inv_r2  # T2/r - T1/r^2
    B_prime = T3 - T2 * inv_r + T1 * inv_r2  # T3 - T2/r + T1/r^2

    # K coefficients for T_αβγδ
    K1 = T2 * inv_r2 - T1 * inv_r3
    K2 = T3 * inv_r - wp.float64(3.0) * T2 * inv_r2 + wp.float64(3.0) * T1 * inv_r3
    K3 = (
        T4
        - wp.float64(6.0) * T3 * inv_r
        + wp.float64(15.0) * T2 * inv_r2
        - wp.float64(15.0) * T1 * inv_r3
    )
    # K'_n = ∂K_n/∂r (used in dr_j QQ channel)
    K1p = T3 * inv_r2 - wp.float64(3.0) * T2 * inv_r3 + wp.float64(3.0) * T1 * inv_r4
    K2p = (
        T4 * inv_r
        - wp.float64(4.0) * T3 * inv_r2
        + wp.float64(9.0) * T2 * inv_r3
        - wp.float64(9.0) * T1 * inv_r4
    )
    K3p = (
        T5
        - wp.float64(6.0) * T4 * inv_r
        + wp.float64(21.0) * T3 * inv_r2
        - wp.float64(45.0) * T2 * inv_r3
        + wp.float64(45.0) * T1 * inv_r4
    )

    # r̂ unit vector
    rhat_x = r_vec[0] * inv_r
    rhat_y = r_vec[1] * inv_r
    rhat_z = r_vec[2] * inv_r

    # Dot products
    mu_i_r = mu_i[0] * r_vec[0] + mu_i[1] * r_vec[1] + mu_i[2] * r_vec[2]
    mu_j_r = mu_j[0] * r_vec[0] + mu_j[1] * r_vec[1] + mu_j[2] * r_vec[2]
    mu_i_mu_j = mu_i[0] * mu_j[0] + mu_i[1] * mu_j[1] + mu_i[2] * mu_j[2]
    mui_rh = mu_i_r * inv_r
    muj_rh = mu_j_r * inv_r

    tr_Qi = Q_i[0, 0] + Q_i[1, 1] + Q_i[2, 2]
    tr_Qj = Q_j[0, 0] + Q_j[1, 1] + Q_j[2, 2]

    # Q · r_vec (3-vectors)
    Qi_r_0 = Q_i[0, 0] * r_vec[0] + Q_i[0, 1] * r_vec[1] + Q_i[0, 2] * r_vec[2]
    Qi_r_1 = Q_i[1, 0] * r_vec[0] + Q_i[1, 1] * r_vec[1] + Q_i[1, 2] * r_vec[2]
    Qi_r_2 = Q_i[2, 0] * r_vec[0] + Q_i[2, 1] * r_vec[1] + Q_i[2, 2] * r_vec[2]
    Qj_r_0 = Q_j[0, 0] * r_vec[0] + Q_j[0, 1] * r_vec[1] + Q_j[0, 2] * r_vec[2]
    Qj_r_1 = Q_j[1, 0] * r_vec[0] + Q_j[1, 1] * r_vec[1] + Q_j[1, 2] * r_vec[2]
    Qj_r_2 = Q_j[2, 0] * r_vec[0] + Q_j[2, 1] * r_vec[1] + Q_j[2, 2] * r_vec[2]

    r_Qi_r = r_vec[0] * Qi_r_0 + r_vec[1] * Qi_r_1 + r_vec[2] * Qi_r_2
    r_Qj_r = r_vec[0] * Qj_r_0 + r_vec[1] * Qj_r_1 + r_vec[2] * Qj_r_2

    # (Q·r̂)_α = Q·r_vec / r
    Qi_rh_0 = Qi_r_0 * inv_r
    Qi_rh_1 = Qi_r_1 * inv_r
    Qi_rh_2 = Qi_r_2 * inv_r
    Qj_rh_0 = Qj_r_0 * inv_r
    Qj_rh_1 = Qj_r_1 * inv_r
    Qj_rh_2 = Qj_r_2 * inv_r

    rhat_Qi_rhat = r_Qi_r * inv_r2
    rhat_Qj_rhat = r_Qj_r * inv_r2

    # μi · Qj (vec3 = row · matrix), and the symmetric (μj·Qi).
    mu_i_Qj_0 = mu_i[0] * Q_j[0, 0] + mu_i[1] * Q_j[1, 0] + mu_i[2] * Q_j[2, 0]
    mu_i_Qj_1 = mu_i[0] * Q_j[0, 1] + mu_i[1] * Q_j[1, 1] + mu_i[2] * Q_j[2, 1]
    mu_i_Qj_2 = mu_i[0] * Q_j[0, 2] + mu_i[1] * Q_j[1, 2] + mu_i[2] * Q_j[2, 2]
    mu_j_Qi_0 = mu_j[0] * Q_i[0, 0] + mu_j[1] * Q_i[1, 0] + mu_j[2] * Q_i[2, 0]
    mu_j_Qi_1 = mu_j[0] * Q_i[0, 1] + mu_j[1] * Q_i[1, 1] + mu_j[2] * Q_i[2, 1]
    mu_j_Qi_2 = mu_j[0] * Q_i[0, 2] + mu_j[1] * Q_i[1, 2] + mu_j[2] * Q_i[2, 2]

    # Scalars built from the above
    mu_i_Qj_r = mu_i_Qj_0 * r_vec[0] + mu_i_Qj_1 * r_vec[1] + mu_i_Qj_2 * r_vec[2]
    mu_j_Qi_r = mu_j_Qi_0 * r_vec[0] + mu_j_Qi_1 * r_vec[1] + mu_j_Qi_2 * r_vec[2]
    mu_i_Qj_rh = mu_i_Qj_r * inv_r
    mu_j_Qi_rh = mu_j_Qi_r * inv_r

    Qi_Qj_dd = (
        Q_i[0, 0] * Q_j[0, 0]
        + Q_i[0, 1] * Q_j[0, 1]
        + Q_i[0, 2] * Q_j[0, 2]
        + Q_i[1, 0] * Q_j[1, 0]
        + Q_i[1, 1] * Q_j[1, 1]
        + Q_i[1, 2] * Q_j[1, 2]
        + Q_i[2, 0] * Q_j[2, 0]
        + Q_i[2, 1] * Q_j[2, 1]
        + Q_i[2, 2] * Q_j[2, 2]
    )

    # QiQj = Qi @ Qj; need its r-contraction for QQ dr_j channel.
    QiQj_00 = Q_i[0, 0] * Q_j[0, 0] + Q_i[0, 1] * Q_j[1, 0] + Q_i[0, 2] * Q_j[2, 0]
    QiQj_01 = Q_i[0, 0] * Q_j[0, 1] + Q_i[0, 1] * Q_j[1, 1] + Q_i[0, 2] * Q_j[2, 1]
    QiQj_02 = Q_i[0, 0] * Q_j[0, 2] + Q_i[0, 1] * Q_j[1, 2] + Q_i[0, 2] * Q_j[2, 2]
    QiQj_10 = Q_i[1, 0] * Q_j[0, 0] + Q_i[1, 1] * Q_j[1, 0] + Q_i[1, 2] * Q_j[2, 0]
    QiQj_11 = Q_i[1, 0] * Q_j[0, 1] + Q_i[1, 1] * Q_j[1, 1] + Q_i[1, 2] * Q_j[2, 1]
    QiQj_12 = Q_i[1, 0] * Q_j[0, 2] + Q_i[1, 1] * Q_j[1, 2] + Q_i[1, 2] * Q_j[2, 2]
    QiQj_20 = Q_i[2, 0] * Q_j[0, 0] + Q_i[2, 1] * Q_j[1, 0] + Q_i[2, 2] * Q_j[2, 0]
    QiQj_21 = Q_i[2, 0] * Q_j[0, 1] + Q_i[2, 1] * Q_j[1, 1] + Q_i[2, 2] * Q_j[2, 1]
    QiQj_22 = Q_i[2, 0] * Q_j[0, 2] + Q_i[2, 1] * Q_j[1, 2] + Q_i[2, 2] * Q_j[2, 2]
    QiQj_r_0 = QiQj_00 * r_vec[0] + QiQj_01 * r_vec[1] + QiQj_02 * r_vec[2]
    QiQj_r_1 = QiQj_10 * r_vec[0] + QiQj_11 * r_vec[1] + QiQj_12 * r_vec[2]
    QiQj_r_2 = QiQj_20 * r_vec[0] + QiQj_21 * r_vec[1] + QiQj_22 * r_vec[2]
    QiQj_rh_0 = QiQj_r_0 * inv_r
    QiQj_rh_1 = QiQj_r_1 * inv_r
    QiQj_rh_2 = QiQj_r_2 * inv_r
    r_QiQj_r = r_vec[0] * QiQj_r_0 + r_vec[1] * QiQj_r_1 + r_vec[2] * QiQj_r_2
    rhat_QiQj_rhat = r_QiQj_r * inv_r2
    # QjQi · r needed too (QjQi = (QiQj)^T for symmetric Qi, Qj).
    QjQi_r_0 = QiQj_00 * r_vec[0] + QiQj_10 * r_vec[1] + QiQj_20 * r_vec[2]
    QjQi_r_1 = QiQj_01 * r_vec[0] + QiQj_11 * r_vec[1] + QiQj_21 * r_vec[2]
    QjQi_r_2 = QiQj_02 * r_vec[0] + QiQj_12 * r_vec[1] + QiQj_22 * r_vec[2]
    QjQi_rh_0 = QjQi_r_0 * inv_r
    QjQi_rh_1 = QjQi_r_1 * inv_r
    QjQi_rh_2 = QjQi_r_2 * inv_r

    # =================================================================
    # Energy (matches `_quadrupole_pair_energy_only`).
    # =================================================================
    E = qi * qj * t0

    # qμ channel (kernel convention)
    E += (qi * mu_j_r - qj * mu_i_r) * T1 * inv_r

    # μμ channel
    mu_Tab_mu = A_val * mu_i_mu_j + B_val * inv_r2 * mu_i_r * mu_j_r
    E += -mu_Tab_mu

    # qQ channel
    Qi_Tab_scalar = A_val * tr_Qi + B_val * inv_r2 * r_Qi_r
    Qj_Tab_scalar = A_val * tr_Qj + B_val * inv_r2 * r_Qj_r
    E += wp.float64(0.5) * qj * Qi_Tab_scalar + wp.float64(0.5) * qi * Qj_Tab_scalar

    # μQ channel
    C1 = A_prime + B_val * inv_r
    coef_j_rhat = B_prime * rhat_Qj_rhat + B_val * inv_r * (
        tr_Qj - wp.float64(2.0) * rhat_Qj_rhat
    )
    coef_i_rhat = B_prime * rhat_Qi_rhat + B_val * inv_r * (
        tr_Qi - wp.float64(2.0) * rhat_Qi_rhat
    )
    contrib_mu_Q_i = C1 * mu_i_Qj_rh + mui_rh * coef_j_rhat
    contrib_mu_Q_j = C1 * mu_j_Qi_rh + muj_rh * coef_i_rhat
    # The μ-Q (T³) energy is −0.5·(μi:∇³φ:Qj − μj:∇³φ:Qi); this sign keeps
    # the composite l=2 total α-independent.
    E -= wp.float64(0.5) * (contrib_mu_Q_i - contrib_mu_Q_j)

    # QQ channel
    QQ_contraction = (
        K1 * (tr_Qi * tr_Qj + wp.float64(2.0) * Qi_Qj_dd)
        + K2
        * (
            tr_Qi * rhat_Qj_rhat
            + tr_Qj * rhat_Qi_rhat
            + wp.float64(4.0) * rhat_QiQj_rhat
        )
        + K3 * rhat_Qi_rhat * rhat_Qj_rhat
    )
    E += wp.float64(0.25) * QQ_contraction

    # =================================================================
    # Gradients.
    # =================================================================
    # ---- dE/dq_i, dE/dq_j ----
    dPE_dq_i = qj * t0 + T1 * muj_rh + wp.float64(0.5) * Qj_Tab_scalar
    dPE_dq_j = qi * t0 - T1 * mui_rh + wp.float64(0.5) * Qi_Tab_scalar

    # ---- dE/dμ_i, dE/dμ_j ----
    # qμ contribution:  -qj * T1 * r̂  (for dμ_i)
    #                   +qi * T1 * r̂  (for dμ_j)
    qmu_coef_i = -qj * T1
    qmu_coef_j = qi * T1
    # μμ contribution: -Tab·μ_other
    Tab_mu_j_0 = A_val * mu_j[0] + B_val * inv_r2 * mu_j_r * r_vec[0]
    Tab_mu_j_1 = A_val * mu_j[1] + B_val * inv_r2 * mu_j_r * r_vec[1]
    Tab_mu_j_2 = A_val * mu_j[2] + B_val * inv_r2 * mu_j_r * r_vec[2]
    Tab_mu_i_0 = A_val * mu_i[0] + B_val * inv_r2 * mu_i_r * r_vec[0]
    Tab_mu_i_1 = A_val * mu_i[1] + B_val * inv_r2 * mu_i_r * r_vec[1]
    Tab_mu_i_2 = A_val * mu_i[2] + B_val * inv_r2 * mu_i_r * r_vec[2]
    # μQ contribution: +0.5 * (Qj : T_αβγ)_α  for dμ_i (the kernel's free index)
    #                  -0.5 * (Qi : T_αβγ)_α  for dμ_j
    # (Qj:T)_α = C1 · (Qj·r̂)_α + r̂_α · coef_j_rhat
    QjT_vec_0 = C1 * Qj_rh_0 + rhat_x * coef_j_rhat
    QjT_vec_1 = C1 * Qj_rh_1 + rhat_y * coef_j_rhat
    QjT_vec_2 = C1 * Qj_rh_2 + rhat_z * coef_j_rhat
    QiT_vec_0 = C1 * Qi_rh_0 + rhat_x * coef_i_rhat
    QiT_vec_1 = C1 * Qi_rh_1 + rhat_y * coef_i_rhat
    QiT_vec_2 = C1 * Qi_rh_2 + rhat_z * coef_i_rhat

    # μQ contributions to dE/dμ carry the channel sign of the −0.5·… μQ
    # energy: dμ_i gets −0.5·QjT and dμ_j gets +0.5·QiT.
    dPE_dmu_i = wp.vec3d(
        qmu_coef_i * rhat_x - Tab_mu_j_0 - wp.float64(0.5) * QjT_vec_0,
        qmu_coef_i * rhat_y - Tab_mu_j_1 - wp.float64(0.5) * QjT_vec_1,
        qmu_coef_i * rhat_z - Tab_mu_j_2 - wp.float64(0.5) * QjT_vec_2,
    )
    dPE_dmu_j = wp.vec3d(
        qmu_coef_j * rhat_x - Tab_mu_i_0 + wp.float64(0.5) * QiT_vec_0,
        qmu_coef_j * rhat_y - Tab_mu_i_1 + wp.float64(0.5) * QiT_vec_1,
        qmu_coef_j * rhat_z - Tab_mu_i_2 + wp.float64(0.5) * QiT_vec_2,
    )

    # ---- dE/dQ_i, dE/dQ_j (mat33, free-index partial) ----
    # qQ contribution: 0.5 * q_other * Tab_mat
    #   Tab_mat[α,β] = A·δ_αβ + (T2 - A)·r̂_α·r̂_β
    T2_minus_A = T2 - A_val
    Tab_00 = A_val + T2_minus_A * rhat_x * rhat_x
    Tab_01 = T2_minus_A * rhat_x * rhat_y
    Tab_02 = T2_minus_A * rhat_x * rhat_z
    Tab_10 = T2_minus_A * rhat_y * rhat_x
    Tab_11 = A_val + T2_minus_A * rhat_y * rhat_y
    Tab_12 = T2_minus_A * rhat_y * rhat_z
    Tab_20 = T2_minus_A * rhat_z * rhat_x
    Tab_21 = T2_minus_A * rhat_z * rhat_y
    Tab_22 = A_val + T2_minus_A * rhat_z * rhat_z

    # μQ contribution: -0.5 * (μ_other · T)_αβ for dQ_self
    #   (μj·T)_αβ = A'·μj_α·r̂_β + B'·muj_rh·r̂_α·r̂_β
    #              + (B/r)·(r̂_α·μj_β + muj_rh·δ_αβ - 2·muj_rh·r̂_α·r̂_β)
    B_over_r = B_val * inv_r
    muj_T_00 = (
        A_prime * mu_j[0] * rhat_x
        + B_prime * muj_rh * rhat_x * rhat_x
        + B_over_r
        * (rhat_x * mu_j[0] + muj_rh - wp.float64(2.0) * muj_rh * rhat_x * rhat_x)
    )
    muj_T_01 = (
        A_prime * mu_j[0] * rhat_y
        + B_prime * muj_rh * rhat_x * rhat_y
        + B_over_r * (rhat_x * mu_j[1] - wp.float64(2.0) * muj_rh * rhat_x * rhat_y)
    )
    muj_T_02 = (
        A_prime * mu_j[0] * rhat_z
        + B_prime * muj_rh * rhat_x * rhat_z
        + B_over_r * (rhat_x * mu_j[2] - wp.float64(2.0) * muj_rh * rhat_x * rhat_z)
    )
    muj_T_10 = (
        A_prime * mu_j[1] * rhat_x
        + B_prime * muj_rh * rhat_y * rhat_x
        + B_over_r * (rhat_y * mu_j[0] - wp.float64(2.0) * muj_rh * rhat_y * rhat_x)
    )
    muj_T_11 = (
        A_prime * mu_j[1] * rhat_y
        + B_prime * muj_rh * rhat_y * rhat_y
        + B_over_r
        * (rhat_y * mu_j[1] + muj_rh - wp.float64(2.0) * muj_rh * rhat_y * rhat_y)
    )
    muj_T_12 = (
        A_prime * mu_j[1] * rhat_z
        + B_prime * muj_rh * rhat_y * rhat_z
        + B_over_r * (rhat_y * mu_j[2] - wp.float64(2.0) * muj_rh * rhat_y * rhat_z)
    )
    muj_T_20 = (
        A_prime * mu_j[2] * rhat_x
        + B_prime * muj_rh * rhat_z * rhat_x
        + B_over_r * (rhat_z * mu_j[0] - wp.float64(2.0) * muj_rh * rhat_z * rhat_x)
    )
    muj_T_21 = (
        A_prime * mu_j[2] * rhat_y
        + B_prime * muj_rh * rhat_z * rhat_y
        + B_over_r * (rhat_z * mu_j[1] - wp.float64(2.0) * muj_rh * rhat_z * rhat_y)
    )
    muj_T_22 = (
        A_prime * mu_j[2] * rhat_z
        + B_prime * muj_rh * rhat_z * rhat_z
        + B_over_r
        * (rhat_z * mu_j[2] + muj_rh - wp.float64(2.0) * muj_rh * rhat_z * rhat_z)
    )

    mui_T_00 = (
        A_prime * mu_i[0] * rhat_x
        + B_prime * mui_rh * rhat_x * rhat_x
        + B_over_r
        * (rhat_x * mu_i[0] + mui_rh - wp.float64(2.0) * mui_rh * rhat_x * rhat_x)
    )
    mui_T_01 = (
        A_prime * mu_i[0] * rhat_y
        + B_prime * mui_rh * rhat_x * rhat_y
        + B_over_r * (rhat_x * mu_i[1] - wp.float64(2.0) * mui_rh * rhat_x * rhat_y)
    )
    mui_T_02 = (
        A_prime * mu_i[0] * rhat_z
        + B_prime * mui_rh * rhat_x * rhat_z
        + B_over_r * (rhat_x * mu_i[2] - wp.float64(2.0) * mui_rh * rhat_x * rhat_z)
    )
    mui_T_10 = (
        A_prime * mu_i[1] * rhat_x
        + B_prime * mui_rh * rhat_y * rhat_x
        + B_over_r * (rhat_y * mu_i[0] - wp.float64(2.0) * mui_rh * rhat_y * rhat_x)
    )
    mui_T_11 = (
        A_prime * mu_i[1] * rhat_y
        + B_prime * mui_rh * rhat_y * rhat_y
        + B_over_r
        * (rhat_y * mu_i[1] + mui_rh - wp.float64(2.0) * mui_rh * rhat_y * rhat_y)
    )
    mui_T_12 = (
        A_prime * mu_i[1] * rhat_z
        + B_prime * mui_rh * rhat_y * rhat_z
        + B_over_r * (rhat_y * mu_i[2] - wp.float64(2.0) * mui_rh * rhat_y * rhat_z)
    )
    mui_T_20 = (
        A_prime * mu_i[2] * rhat_x
        + B_prime * mui_rh * rhat_z * rhat_x
        + B_over_r * (rhat_z * mu_i[0] - wp.float64(2.0) * mui_rh * rhat_z * rhat_x)
    )
    mui_T_21 = (
        A_prime * mu_i[2] * rhat_y
        + B_prime * mui_rh * rhat_z * rhat_y
        + B_over_r * (rhat_z * mu_i[1] - wp.float64(2.0) * mui_rh * rhat_z * rhat_y)
    )
    mui_T_22 = (
        A_prime * mu_i[2] * rhat_z
        + B_prime * mui_rh * rhat_z * rhat_z
        + B_over_r
        * (rhat_z * mu_i[2] + mui_rh - wp.float64(2.0) * mui_rh * rhat_z * rhat_z)
    )

    # QQ contribution: 0.25 * (T:Q_other)_αβ for dQ_self
    #   (T:Qj)_αβ = K1·(δ_αβ·tr_Qj + 2·Qj_αβ)
    #              + K2·(δ_αβ·rhat_Qj_rhat + 2·r̂_β·(Qj·r̂)_α
    #                    + 2·r̂_α·(Qj·r̂)_β + r̂_α·r̂_β·tr_Qj)
    #              + K3·r̂_α·r̂_β·rhat_Qj_rhat
    T_Qj_00 = (
        K1 * (tr_Qj + wp.float64(2.0) * Q_j[0, 0])
        + K2
        * (rhat_Qj_rhat + wp.float64(4.0) * rhat_x * Qj_rh_0 + rhat_x * rhat_x * tr_Qj)
        + K3 * rhat_x * rhat_x * rhat_Qj_rhat
    )
    T_Qj_01 = (
        K1 * (wp.float64(2.0) * Q_j[0, 1])
        + K2
        * (
            wp.float64(2.0) * rhat_y * Qj_rh_0
            + wp.float64(2.0) * rhat_x * Qj_rh_1
            + rhat_x * rhat_y * tr_Qj
        )
        + K3 * rhat_x * rhat_y * rhat_Qj_rhat
    )
    T_Qj_02 = (
        K1 * (wp.float64(2.0) * Q_j[0, 2])
        + K2
        * (
            wp.float64(2.0) * rhat_z * Qj_rh_0
            + wp.float64(2.0) * rhat_x * Qj_rh_2
            + rhat_x * rhat_z * tr_Qj
        )
        + K3 * rhat_x * rhat_z * rhat_Qj_rhat
    )
    T_Qj_10 = (
        K1 * (wp.float64(2.0) * Q_j[1, 0])
        + K2
        * (
            wp.float64(2.0) * rhat_x * Qj_rh_1
            + wp.float64(2.0) * rhat_y * Qj_rh_0
            + rhat_y * rhat_x * tr_Qj
        )
        + K3 * rhat_y * rhat_x * rhat_Qj_rhat
    )
    T_Qj_11 = (
        K1 * (tr_Qj + wp.float64(2.0) * Q_j[1, 1])
        + K2
        * (rhat_Qj_rhat + wp.float64(4.0) * rhat_y * Qj_rh_1 + rhat_y * rhat_y * tr_Qj)
        + K3 * rhat_y * rhat_y * rhat_Qj_rhat
    )
    T_Qj_12 = (
        K1 * (wp.float64(2.0) * Q_j[1, 2])
        + K2
        * (
            wp.float64(2.0) * rhat_z * Qj_rh_1
            + wp.float64(2.0) * rhat_y * Qj_rh_2
            + rhat_y * rhat_z * tr_Qj
        )
        + K3 * rhat_y * rhat_z * rhat_Qj_rhat
    )
    T_Qj_20 = (
        K1 * (wp.float64(2.0) * Q_j[2, 0])
        + K2
        * (
            wp.float64(2.0) * rhat_x * Qj_rh_2
            + wp.float64(2.0) * rhat_z * Qj_rh_0
            + rhat_z * rhat_x * tr_Qj
        )
        + K3 * rhat_z * rhat_x * rhat_Qj_rhat
    )
    T_Qj_21 = (
        K1 * (wp.float64(2.0) * Q_j[2, 1])
        + K2
        * (
            wp.float64(2.0) * rhat_y * Qj_rh_2
            + wp.float64(2.0) * rhat_z * Qj_rh_1
            + rhat_z * rhat_y * tr_Qj
        )
        + K3 * rhat_z * rhat_y * rhat_Qj_rhat
    )
    T_Qj_22 = (
        K1 * (tr_Qj + wp.float64(2.0) * Q_j[2, 2])
        + K2
        * (rhat_Qj_rhat + wp.float64(4.0) * rhat_z * Qj_rh_2 + rhat_z * rhat_z * tr_Qj)
        + K3 * rhat_z * rhat_z * rhat_Qj_rhat
    )

    T_Qi_00 = (
        K1 * (tr_Qi + wp.float64(2.0) * Q_i[0, 0])
        + K2
        * (rhat_Qi_rhat + wp.float64(4.0) * rhat_x * Qi_rh_0 + rhat_x * rhat_x * tr_Qi)
        + K3 * rhat_x * rhat_x * rhat_Qi_rhat
    )
    T_Qi_01 = (
        K1 * (wp.float64(2.0) * Q_i[0, 1])
        + K2
        * (
            wp.float64(2.0) * rhat_y * Qi_rh_0
            + wp.float64(2.0) * rhat_x * Qi_rh_1
            + rhat_x * rhat_y * tr_Qi
        )
        + K3 * rhat_x * rhat_y * rhat_Qi_rhat
    )
    T_Qi_02 = (
        K1 * (wp.float64(2.0) * Q_i[0, 2])
        + K2
        * (
            wp.float64(2.0) * rhat_z * Qi_rh_0
            + wp.float64(2.0) * rhat_x * Qi_rh_2
            + rhat_x * rhat_z * tr_Qi
        )
        + K3 * rhat_x * rhat_z * rhat_Qi_rhat
    )
    T_Qi_10 = (
        K1 * (wp.float64(2.0) * Q_i[1, 0])
        + K2
        * (
            wp.float64(2.0) * rhat_x * Qi_rh_1
            + wp.float64(2.0) * rhat_y * Qi_rh_0
            + rhat_y * rhat_x * tr_Qi
        )
        + K3 * rhat_y * rhat_x * rhat_Qi_rhat
    )
    T_Qi_11 = (
        K1 * (tr_Qi + wp.float64(2.0) * Q_i[1, 1])
        + K2
        * (rhat_Qi_rhat + wp.float64(4.0) * rhat_y * Qi_rh_1 + rhat_y * rhat_y * tr_Qi)
        + K3 * rhat_y * rhat_y * rhat_Qi_rhat
    )
    T_Qi_12 = (
        K1 * (wp.float64(2.0) * Q_i[1, 2])
        + K2
        * (
            wp.float64(2.0) * rhat_z * Qi_rh_1
            + wp.float64(2.0) * rhat_y * Qi_rh_2
            + rhat_y * rhat_z * tr_Qi
        )
        + K3 * rhat_y * rhat_z * rhat_Qi_rhat
    )
    T_Qi_20 = (
        K1 * (wp.float64(2.0) * Q_i[2, 0])
        + K2
        * (
            wp.float64(2.0) * rhat_x * Qi_rh_2
            + wp.float64(2.0) * rhat_z * Qi_rh_0
            + rhat_z * rhat_x * tr_Qi
        )
        + K3 * rhat_z * rhat_x * rhat_Qi_rhat
    )
    T_Qi_21 = (
        K1 * (wp.float64(2.0) * Q_i[2, 1])
        + K2
        * (
            wp.float64(2.0) * rhat_y * Qi_rh_2
            + wp.float64(2.0) * rhat_z * Qi_rh_1
            + rhat_z * rhat_y * tr_Qi
        )
        + K3 * rhat_z * rhat_y * rhat_Qi_rhat
    )
    T_Qi_22 = (
        K1 * (tr_Qi + wp.float64(2.0) * Q_i[2, 2])
        + K2
        * (rhat_Qi_rhat + wp.float64(4.0) * rhat_z * Qi_rh_2 + rhat_z * rhat_z * tr_Qi)
        + K3 * rhat_z * rhat_z * rhat_Qi_rhat
    )

    half = wp.float64(0.5)
    quarter = wp.float64(0.25)
    qj_half = wp.float64(0.5) * qj
    qi_half = wp.float64(0.5) * qi

    # μQ contributions to dE/dQ: dQ_i gets +half·muj_T, dQ_j gets −half·mui_T.
    dPE_dQ_i = wp.mat33d(
        qj_half * Tab_00 + half * muj_T_00 + quarter * T_Qj_00,
        qj_half * Tab_01 + half * muj_T_01 + quarter * T_Qj_01,
        qj_half * Tab_02 + half * muj_T_02 + quarter * T_Qj_02,
        qj_half * Tab_10 + half * muj_T_10 + quarter * T_Qj_10,
        qj_half * Tab_11 + half * muj_T_11 + quarter * T_Qj_11,
        qj_half * Tab_12 + half * muj_T_12 + quarter * T_Qj_12,
        qj_half * Tab_20 + half * muj_T_20 + quarter * T_Qj_20,
        qj_half * Tab_21 + half * muj_T_21 + quarter * T_Qj_21,
        qj_half * Tab_22 + half * muj_T_22 + quarter * T_Qj_22,
    )
    dPE_dQ_j = wp.mat33d(
        qi_half * Tab_00 - half * mui_T_00 + quarter * T_Qi_00,
        qi_half * Tab_01 - half * mui_T_01 + quarter * T_Qi_01,
        qi_half * Tab_02 - half * mui_T_02 + quarter * T_Qi_02,
        qi_half * Tab_10 - half * mui_T_10 + quarter * T_Qi_10,
        qi_half * Tab_11 - half * mui_T_11 + quarter * T_Qi_11,
        qi_half * Tab_12 - half * mui_T_12 + quarter * T_Qi_12,
        qi_half * Tab_20 - half * mui_T_20 + quarter * T_Qi_20,
        qi_half * Tab_21 - half * mui_T_21 + quarter * T_Qi_21,
        qi_half * Tab_22 - half * mui_T_22 + quarter * T_Qi_22,
    )

    # ---- dE/dr_j (6 channels) ----
    # qq channel
    qq_coef = qi * qj * T1
    dr_x = qq_coef * rhat_x
    dr_y = qq_coef * rhat_y
    dr_z = qq_coef * rhat_z

    # qμ channel
    qmu_x = qi * mu_j[0] - qj * mu_i[0]
    qmu_y = qi * mu_j[1] - qj * mu_i[1]
    qmu_z = qi * mu_j[2] - qj * mu_i[2]
    qmu_r = qmu_x * r_vec[0] + qmu_y * r_vec[1] + qmu_z * r_vec[2]
    Tab_qmu_x = A_val * qmu_x + B_val * inv_r2 * qmu_r * r_vec[0]
    Tab_qmu_y = A_val * qmu_y + B_val * inv_r2 * qmu_r * r_vec[1]
    Tab_qmu_z = A_val * qmu_z + B_val * inv_r2 * qmu_r * r_vec[2]
    dr_x += Tab_qmu_x
    dr_y += Tab_qmu_y
    dr_z += Tab_qmu_z

    # μμ channel (subtract)
    mumu_x = (
        A_prime * mu_i_mu_j * rhat_x
        + B_prime * mui_rh * muj_rh * rhat_x
        + B_over_r
        * (
            mu_i[0] * muj_rh
            + mu_j[0] * mui_rh
            - wp.float64(2.0) * mui_rh * muj_rh * rhat_x
        )
    )
    mumu_y = (
        A_prime * mu_i_mu_j * rhat_y
        + B_prime * mui_rh * muj_rh * rhat_y
        + B_over_r
        * (
            mu_i[1] * muj_rh
            + mu_j[1] * mui_rh
            - wp.float64(2.0) * mui_rh * muj_rh * rhat_y
        )
    )
    mumu_z = (
        A_prime * mu_i_mu_j * rhat_z
        + B_prime * mui_rh * muj_rh * rhat_z
        + B_over_r
        * (
            mu_i[2] * muj_rh
            + mu_j[2] * mui_rh
            - wp.float64(2.0) * mui_rh * muj_rh * rhat_z
        )
    )
    dr_x -= mumu_x
    dr_y -= mumu_y
    dr_z -= mumu_z

    # qQ channel
    # (Qi:T)_γ = A'·tr_Qi·r̂_γ + B'·rhat_Qi_rhat·r̂_γ
    #            + (B/r)·[2·(Qi·r̂)_γ - 2·rhat_Qi_rhat·r̂_γ]
    QiT_g_x = (
        A_prime * tr_Qi * rhat_x
        + B_prime * rhat_Qi_rhat * rhat_x
        + B_over_r
        * (wp.float64(2.0) * Qi_rh_0 - wp.float64(2.0) * rhat_Qi_rhat * rhat_x)
    )
    QiT_g_y = (
        A_prime * tr_Qi * rhat_y
        + B_prime * rhat_Qi_rhat * rhat_y
        + B_over_r
        * (wp.float64(2.0) * Qi_rh_1 - wp.float64(2.0) * rhat_Qi_rhat * rhat_y)
    )
    QiT_g_z = (
        A_prime * tr_Qi * rhat_z
        + B_prime * rhat_Qi_rhat * rhat_z
        + B_over_r
        * (wp.float64(2.0) * Qi_rh_2 - wp.float64(2.0) * rhat_Qi_rhat * rhat_z)
    )
    QjT_g_x = (
        A_prime * tr_Qj * rhat_x
        + B_prime * rhat_Qj_rhat * rhat_x
        + B_over_r
        * (wp.float64(2.0) * Qj_rh_0 - wp.float64(2.0) * rhat_Qj_rhat * rhat_x)
    )
    QjT_g_y = (
        A_prime * tr_Qj * rhat_y
        + B_prime * rhat_Qj_rhat * rhat_y
        + B_over_r
        * (wp.float64(2.0) * Qj_rh_1 - wp.float64(2.0) * rhat_Qj_rhat * rhat_y)
    )
    QjT_g_z = (
        A_prime * tr_Qj * rhat_z
        + B_prime * rhat_Qj_rhat * rhat_z
        + B_over_r
        * (wp.float64(2.0) * Qj_rh_2 - wp.float64(2.0) * rhat_Qj_rhat * rhat_z)
    )
    dr_x += half * qj * QiT_g_x + half * qi * QjT_g_x
    dr_y += half * qj * QiT_g_y + half * qi * QjT_g_y
    dr_z += half * qj * QiT_g_z + half * qi * QjT_g_z

    # μQ channel
    # K1 piece: K1·(2·((μi·Qj)_δ - (μj·Qi)_δ) + μi_δ·tr_Qj - μj_δ·tr_Qi)
    muQ_K1_x = K1 * (
        wp.float64(2.0) * (mu_i_Qj_0 - mu_j_Qi_0) + mu_i[0] * tr_Qj - mu_j[0] * tr_Qi
    )
    muQ_K1_y = K1 * (
        wp.float64(2.0) * (mu_i_Qj_1 - mu_j_Qi_1) + mu_i[1] * tr_Qj - mu_j[1] * tr_Qi
    )
    muQ_K1_z = K1 * (
        wp.float64(2.0) * (mu_i_Qj_2 - mu_j_Qi_2) + mu_i[2] * tr_Qj - mu_j[2] * tr_Qi
    )
    # K2 piece:
    #   2·(mu_i_Qj_rh - mu_j_Qi_rh)·r̂_δ
    #   + μi_δ·rhat_Qj_rhat - μj_δ·rhat_Qi_rhat
    #   + (mui_rh·tr_Qj - muj_rh·tr_Qi)·r̂_δ
    #   + 2·(mui_rh·(Qj·r̂)_δ - muj_rh·(Qi·r̂)_δ)
    K2_a_scal = (
        wp.float64(2.0) * (mu_i_Qj_rh - mu_j_Qi_rh) + mui_rh * tr_Qj - muj_rh * tr_Qi
    )
    muQ_K2_x = K2 * (
        K2_a_scal * rhat_x
        + mu_i[0] * rhat_Qj_rhat
        - mu_j[0] * rhat_Qi_rhat
        + wp.float64(2.0) * (mui_rh * Qj_rh_0 - muj_rh * Qi_rh_0)
    )
    muQ_K2_y = K2 * (
        K2_a_scal * rhat_y
        + mu_i[1] * rhat_Qj_rhat
        - mu_j[1] * rhat_Qi_rhat
        + wp.float64(2.0) * (mui_rh * Qj_rh_1 - muj_rh * Qi_rh_1)
    )
    muQ_K2_z = K2 * (
        K2_a_scal * rhat_z
        + mu_i[2] * rhat_Qj_rhat
        - mu_j[2] * rhat_Qi_rhat
        + wp.float64(2.0) * (mui_rh * Qj_rh_2 - muj_rh * Qi_rh_2)
    )
    # K3 piece: K3·(mui_rh·rhat_Qj_rhat - muj_rh·rhat_Qi_rhat)·r̂_δ
    K3_scal = K3 * (mui_rh * rhat_Qj_rhat - muj_rh * rhat_Qi_rhat)
    # μQ force contribution carries the −0.5 channel sign.
    dr_x -= half * (muQ_K1_x + muQ_K2_x + K3_scal * rhat_x)
    dr_y -= half * (muQ_K1_y + muQ_K2_y + K3_scal * rhat_y)
    dr_z -= half * (muQ_K1_z + muQ_K2_z + K3_scal * rhat_z)

    # QQ channel
    S_0 = tr_Qi * tr_Qj + wp.float64(2.0) * Qi_Qj_dd
    S_2A = (
        tr_Qi * rhat_Qj_rhat + tr_Qj * rhat_Qi_rhat + wp.float64(4.0) * rhat_QiQj_rhat
    )
    S_4 = rhat_Qi_rhat * rhat_Qj_rhat
    two_over_r = wp.float64(2.0) * inv_r
    dS_2A_x = two_over_r * (
        tr_Qi * (Qj_rh_0 - rhat_Qj_rhat * rhat_x)
        + tr_Qj * (Qi_rh_0 - rhat_Qi_rhat * rhat_x)
        + wp.float64(2.0)
        * (QiQj_rh_0 + QjQi_rh_0 - wp.float64(2.0) * rhat_QiQj_rhat * rhat_x)
    )
    dS_2A_y = two_over_r * (
        tr_Qi * (Qj_rh_1 - rhat_Qj_rhat * rhat_y)
        + tr_Qj * (Qi_rh_1 - rhat_Qi_rhat * rhat_y)
        + wp.float64(2.0)
        * (QiQj_rh_1 + QjQi_rh_1 - wp.float64(2.0) * rhat_QiQj_rhat * rhat_y)
    )
    dS_2A_z = two_over_r * (
        tr_Qi * (Qj_rh_2 - rhat_Qj_rhat * rhat_z)
        + tr_Qj * (Qi_rh_2 - rhat_Qi_rhat * rhat_z)
        + wp.float64(2.0)
        * (QiQj_rh_2 + QjQi_rh_2 - wp.float64(2.0) * rhat_QiQj_rhat * rhat_z)
    )
    dS_4_x = two_over_r * (
        Qi_rh_0 * rhat_Qj_rhat
        + rhat_Qi_rhat * Qj_rh_0
        - wp.float64(2.0) * rhat_x * rhat_Qi_rhat * rhat_Qj_rhat
    )
    dS_4_y = two_over_r * (
        Qi_rh_1 * rhat_Qj_rhat
        + rhat_Qi_rhat * Qj_rh_1
        - wp.float64(2.0) * rhat_y * rhat_Qi_rhat * rhat_Qj_rhat
    )
    dS_4_z = two_over_r * (
        Qi_rh_2 * rhat_Qj_rhat
        + rhat_Qi_rhat * Qj_rh_2
        - wp.float64(2.0) * rhat_z * rhat_Qi_rhat * rhat_Qj_rhat
    )
    QQ_x = (
        K1p * rhat_x * S_0
        + K2p * rhat_x * S_2A
        + K2 * dS_2A_x
        + K3p * rhat_x * S_4
        + K3 * dS_4_x
    )
    QQ_y = (
        K1p * rhat_y * S_0
        + K2p * rhat_y * S_2A
        + K2 * dS_2A_y
        + K3p * rhat_y * S_4
        + K3 * dS_4_y
    )
    QQ_z = (
        K1p * rhat_z * S_0
        + K2p * rhat_z * S_2A
        + K2 * dS_2A_z
        + K3p * rhat_z * S_4
        + K3 * dS_4_z
    )
    dr_x += quarter * QQ_x
    dr_y += quarter * QQ_y
    dr_z += quarter * QQ_z

    out = _QuadrupolePairContrib()
    out.energy = E
    out.dPE_dq_i = dPE_dq_i
    out.dPE_dq_j = dPE_dq_j
    out.dPE_dmu_i = dPE_dmu_i
    out.dPE_dmu_j = dPE_dmu_j
    out.dPE_dQ_i = dPE_dQ_i
    out.dPE_dQ_j = dPE_dQ_j
    out.dPE_dr_j = wp.vec3d(dr_x, dr_y, dr_z)
    return out


# =============================================================================
# l_max = 0 (charges only) — energy, CSR, single-system
# =============================================================================


def multipole_real_space_monopole_csr_energy(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigma: wp.array,
    alpha: wp.array,
    pair_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Energy-only launcher for the LMAX=0 CSR single-system kernel.

    One thread per atom ``i``; the thread walks its CSR neighbor slice
    ``idx_j[neighbor_ptr[i]:neighbor_ptr[i+1]]`` and accumulates per-pair
    GTO-Ewald real-space contributions

    .. math::

        E_{ij} = \tfrac{1}{2} \, q_i q_j \, T^{(0)}(r_{ij}; \sigma, \alpha).

    The ``1/2`` factor accounts for the half-neighbor-list convention
    (each pair appears in exactly one atom's neighbor list); for full
    neighbor lists the same factor still yields the correct total energy.

    Framework-agnostic: operates directly on Warp arrays. Internally
    allocates two 1-element scratch arrays for the unused
    ``grad_positions`` / ``grad_charges`` slots.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (1,), dtype wp.mat33f or wp.mat33d
    idx_j : wp.array, shape (M,), dtype wp.int32
    neighbor_ptr : wp.array, shape (N+1,), dtype wp.int32
    unit_shifts : wp.array, shape (M,), dtype wp.vec3i
    sigma, alpha : wp.array, shape (1,)
    pair_energies : wp.array, shape (N,), dtype wp.float64
        OUTPUT (pre-zeroed). Per-atom accumulated energy.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64``.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    # Scratch buffers for the unused gradient slots.
    grad_pos_scratch = wp.zeros(1, dtype=vec_dtype, device=device)
    grad_q_scratch = wp.zeros(1, dtype=wp_dtype, device=device)

    _overload = _get_real_space_pair_overload(
        LMAX=0,
        storage="csr",
        is_batch=False,
        with_pos_grad=False,
        with_charge_grad=False,
        with_dipole_grad=False,
        with_quad_grad=False,
        with_cell_grad=False,
        vec_dtype=vec_dtype,
        scalar_dtype=wp_dtype,
    )

    wp.launch(
        _overload,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            pair_energies,
            grad_pos_scratch,
            grad_q_scratch,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Unified real-space pair-kernel factory.
# -----------------------------------------------------------------------------


def _make_real_space_pair_kernel(
    *,
    LMAX: int,
    storage: str,
    is_batch: bool,
    with_pos_grad: bool,
    with_charge_grad: bool,
    with_dipole_grad: bool,
    with_quad_grad: bool,
    with_cell_grad: bool,
):
    """Build one ``@wp.kernel`` for the requested capability combination.

    Parameters
    ----------
    LMAX : int
        Multipole order. Supports ``0``, ``1``, ``2``.
    storage : str
        Neighbor-pair storage backend. Only ``"csr"`` is supported.
    is_batch : bool
        Single-system (``False``) vs batched (``True``).
    with_pos_grad, with_charge_grad : bool
        Per-slot gradient emission flags.
    with_dipole_grad : bool
        Ignored when ``LMAX < 1``; included for signature stability.
    with_quad_grad : bool
        Ignored when ``LMAX < 2``; included for signature stability.
    with_cell_grad : bool
        Cell-gradient emission flag.

    Returns
    -------
    kernel : ``@wp.kernel``
        A Warp kernel whose body contains only the work for the
        requested capability set. Warp's source-hash cache means
        identical capability requests don't recompile.
    """

    if LMAX not in (0, 1, 2):
        raise NotImplementedError(
            f"_make_real_space_pair_kernel: LMAX={LMAX} not implemented."
        )
    if storage not in ("tile", "csr"):
        raise NotImplementedError(
            f"_make_real_space_pair_kernel: storage={storage!r} not implemented."
        )
    if with_dipole_grad and LMAX < 1:
        raise NotImplementedError(
            "_make_real_space_pair_kernel: with_dipole_grad requires LMAX>=1."
        )
    if with_quad_grad and LMAX < 2:
        raise NotImplementedError(
            "_make_real_space_pair_kernel: with_quad_grad requires LMAX>=2."
        )
    if with_cell_grad:
        raise NotImplementedError(
            "_make_real_space_pair_kernel: cell gradient emission is not supported."
        )

    # Only "csr" storage is supported; both CUDA and CPU paths route
    # through the CSR launcher.
    if storage != "csr":
        raise NotImplementedError(
            f"_make_real_space_pair_kernel: storage={storage!r} not "
            'supported (only "csr" is implemented).'
        )
    if is_batch:
        return _make_real_space_pair_kernel_csr_batched(
            LMAX=LMAX,
            with_pos_grad=with_pos_grad,
            with_charge_grad=with_charge_grad,
            with_dipole_grad=with_dipole_grad,
            with_quad_grad=with_quad_grad,
        )
    return _make_real_space_pair_kernel_csr_single(
        LMAX=LMAX,
        with_pos_grad=with_pos_grad,
        with_charge_grad=with_charge_grad,
        with_dipole_grad=with_dipole_grad,
        with_quad_grad=with_quad_grad,
    )


def _make_real_space_pair_kernel_csr_single(
    *,
    LMAX: int,
    with_pos_grad: bool,
    with_charge_grad: bool,
    with_dipole_grad: bool = False,
    with_quad_grad: bool = False,
):
    """CSR neighbor-list + single-system builder.

    One thread per atom ``i``; the thread walks its CSR neighbor slice
    ``idx_j[neighbor_ptr[i]:neighbor_ptr[i+1]]`` and accumulates per-pair
    contributions. Atomic-adds to ``grad_positions[j]`` and
    ``grad_charges[j]`` (the neighbor index) are contention-prone across
    threads but unavoidable in the CSR layout.
    """
    if LMAX == 1:
        return _make_real_space_pair_kernel_csr_single_dipole(
            with_pos_grad=with_pos_grad,
            with_charge_grad=with_charge_grad,
            with_dipole_grad=with_dipole_grad,
        )
    if LMAX == 2:
        return _make_real_space_pair_kernel_csr_single_quadrupole(
            with_pos_grad=with_pos_grad,
            with_charge_grad=with_charge_grad,
            with_dipole_grad=with_dipole_grad,
            with_quad_grad=with_quad_grad,
        )

    @wp.kernel(enable_backward=False)
    def _kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        cell: wp.array(dtype=Any),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        sigma: wp.array(dtype=Any),
        alpha: wp.array(dtype=Any),
        pair_energies: wp.array(dtype=wp.float64),
        # Gradient output slots — uniform 11-input signature across all
        # ``(wpg, wcg)`` combinations. Unused slots receive a 1-element
        # scratch array from the energy-only launcher.
        grad_positions: wp.array(dtype=Any),
        grad_charges: wp.array(dtype=Any),
    ):
        r"""Unified real-space pair kernel (LMAX=0 CSR single).

        Body branches at codegen time on the closure flags
        ``with_pos_grad`` and ``with_charge_grad``:

        - Both False: energy-only body using ``_gto_ewald_t0``. Grad
          slots untouched.
        - Any True: uses the fused ``_gto_ewald_monopole_pair_terms_fused``
          helper and emits gradients.

        Launch Grid
        -----------
        dim = [num_atoms] — one thread per atom ``i``; the thread walks its
        CSR neighbor slice ``idx_j[neighbor_ptr[i]:neighbor_ptr[i+1]]``.

        Parameters
        ----------
        positions : wp.array, shape (N,), dtype wp.vec3f or wp.vec3d
            Atomic positions.
        charges : wp.array, shape (N,), dtype wp.float32 or wp.float64
            Atomic charges.
        cell : wp.array, shape (1,), dtype wp.mat33f or wp.mat33d
            Lattice matrix.
        idx_j : wp.array, shape (M,), dtype wp.int32
            CSR neighbor target indices.
        neighbor_ptr : wp.array, shape (N+1,), dtype wp.int32
            CSR row pointers into ``idx_j`` / ``unit_shifts``.
        unit_shifts : wp.array, shape (M,), dtype wp.vec3i
            Per-edge periodic image shifts.
        sigma : wp.array, shape (1,), dtype matching ``charges``
            GTO smearing width.
        alpha : wp.array, shape (1,), dtype matching ``charges``
            Ewald splitting parameter.
        pair_energies : wp.array, shape (N,), dtype wp.float64
            OUTPUT (pre-zeroed). Per-atom accumulated energy.
        grad_positions : wp.array, shape (N,), dtype matching ``positions``
            OUTPUT. Written only when ``with_pos_grad``; otherwise a
            1-element scratch placeholder.
        grad_charges : wp.array, shape (N,), dtype matching ``charges``
            OUTPUT. Written only when ``with_charge_grad``; otherwise a
            1-element scratch placeholder.
        """
        atom_i = wp.tid()

        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        sigma_ = wp.float64(sigma[0])
        alpha_ = wp.float64(alpha[0])
        cell_t = wp.transpose(cell[0])

        ab = _gto_ewald_ab(sigma_, alpha_)
        a_coef = ab[0]
        b_coef = ab[1]

        energy_acc = wp.float64(0.0)

        j_range_start = neighbor_ptr[atom_i]
        j_range_end = neighbor_ptr[atom_i + 1]

        for edge_idx in range(j_range_start, j_range_end):
            j = idx_j[edge_idx]
            qj = wp.float64(charges[j])
            pos_j = positions[j]

            shift_vec = unit_shifts[edge_idx]
            periodic_shift = cell_t * type(pos_i)(
                type(pos_i[0])(shift_vec[0]),
                type(pos_i[0])(shift_vec[1]),
                type(pos_i[0])(shift_vec[2]),
            )

            sep = pos_j - pos_i + periodic_shift
            distance = wp.float64(wp.length(sep))

            if distance > wp.float64(1e-8):
                if with_pos_grad or with_charge_grad:
                    terms = _gto_ewald_monopole_pair_terms_fused(
                        distance, a_coef, b_coef
                    )
                    t0 = terms.t0
                    a_scalar = terms.a_scalar
                else:
                    t0 = _gto_ewald_t0(distance, a_coef, b_coef)
                    a_scalar = wp.float64(0.0)  # unused

                energy_acc += wp.float64(0.5) * qi * qj * t0

                if with_charge_grad:
                    half_t0 = wp.float64(0.5) * t0
                    wp.atomic_add(
                        grad_charges,
                        atom_i,
                        type(charges[atom_i])(half_t0 * qj),
                    )
                    wp.atomic_add(
                        grad_charges,
                        j,
                        type(charges[atom_i])(half_t0 * qi),
                    )

                if with_pos_grad:
                    inv_r = wp.float64(1.0) / distance
                    pos_coeff = wp.float64(0.5) * qi * qj * a_scalar * inv_r
                    dx = pos_coeff * wp.float64(sep[0])
                    dy = pos_coeff * wp.float64(sep[1])
                    dz = pos_coeff * wp.float64(sep[2])
                    wp.atomic_add(
                        grad_positions,
                        atom_i,
                        type(pos_i)(
                            type(pos_i[0])(dx),
                            type(pos_i[0])(dy),
                            type(pos_i[0])(dz),
                        ),
                    )
                    wp.atomic_add(
                        grad_positions,
                        j,
                        type(pos_i)(
                            type(pos_i[0])(-dx),
                            type(pos_i[0])(-dy),
                            type(pos_i[0])(-dz),
                        ),
                    )

        wp.atomic_add(pair_energies, atom_i, energy_acc)

    return _kernel


def _make_real_space_pair_kernel_csr_batched(
    *,
    LMAX: int,
    with_pos_grad: bool,
    with_charge_grad: bool,
    with_dipole_grad: bool = False,
    with_quad_grad: bool = False,
):
    """CSR + batched builder. Adds per-atom ``batch_idx`` lookup over the
    CSR single-system body (per-system ``cells[b]`` / ``sigmas[b]`` /
    ``alphas[b]``)."""
    if LMAX == 1:
        return _make_real_space_pair_kernel_csr_batched_dipole(
            with_pos_grad=with_pos_grad,
            with_charge_grad=with_charge_grad,
            with_dipole_grad=with_dipole_grad,
        )
    if LMAX == 2:
        return _make_real_space_pair_kernel_csr_batched_quadrupole(
            with_pos_grad=with_pos_grad,
            with_charge_grad=with_charge_grad,
            with_dipole_grad=with_dipole_grad,
            with_quad_grad=with_quad_grad,
        )

    @wp.kernel(enable_backward=False)
    def _kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        cells: wp.array(dtype=Any),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        sigmas: wp.array(dtype=Any),
        alphas: wp.array(dtype=Any),
        batch_idx: wp.array(dtype=wp.int32),
        pair_energies: wp.array(dtype=wp.float64),
        grad_positions: wp.array(dtype=Any),
        grad_charges: wp.array(dtype=Any),
    ):
        r"""Unified real-space pair kernel (LMAX=0 CSR batched).

        Per-system ``cells[b]`` / ``sigmas[b]`` / ``alphas[b]`` lookup via
        ``b = batch_idx[i]``. Body branches at codegen time on the
        ``with_pos_grad`` / ``with_charge_grad`` closure flags.

        Launch Grid
        -----------
        dim = [num_atoms_total] — one thread per atom across all batched
        systems; inner loop over the atom's CSR neighbor slice.

        Parameters
        ----------
        positions : wp.array, shape (N_total,), dtype wp.vec3f or wp.vec3d
            Concatenated atomic positions across all systems.
        charges : wp.array, shape (N_total,), dtype wp.float32 or wp.float64
            Concatenated atomic charges.
        cells : wp.array, shape (B,), dtype wp.mat33f or wp.mat33d
            Per-system lattice matrices.
        idx_j : wp.array, shape (M,), dtype wp.int32
            Flattened CSR neighbor target indices.
        neighbor_ptr : wp.array, shape (N_total+1,), dtype wp.int32
            CSR row pointers into ``idx_j`` / ``unit_shifts``.
        unit_shifts : wp.array, shape (M,), dtype wp.vec3i
            Per-edge periodic image shifts.
        sigmas : wp.array, shape (B,), dtype matching ``charges``
            Per-system GTO smearing widths.
        alphas : wp.array, shape (B,), dtype matching ``charges``
            Per-system Ewald splitting parameters.
        batch_idx : wp.array, shape (N_total,), dtype wp.int32
            System index ``b`` for each atom.
        pair_energies : wp.array, shape (N_total,), dtype wp.float64
            OUTPUT (pre-zeroed). Per-atom accumulated energy.
        grad_positions : wp.array, shape (N_total,), dtype matching ``positions``
            OUTPUT. Written only when ``with_pos_grad``.
        grad_charges : wp.array, shape (N_total,), dtype matching ``charges``
            OUTPUT. Written only when ``with_charge_grad``.
        """
        atom_i = wp.tid()
        b = batch_idx[atom_i]

        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        sigma_ = wp.float64(sigmas[b])
        alpha_ = wp.float64(alphas[b])
        cell_t = wp.transpose(cells[b])

        ab = _gto_ewald_ab(sigma_, alpha_)
        a_coef = ab[0]
        b_coef = ab[1]

        energy_acc = wp.float64(0.0)
        j_range_start = neighbor_ptr[atom_i]
        j_range_end = neighbor_ptr[atom_i + 1]

        for edge_idx in range(j_range_start, j_range_end):
            j = idx_j[edge_idx]
            qj = wp.float64(charges[j])
            pos_j = positions[j]

            shift_vec = unit_shifts[edge_idx]
            periodic_shift = cell_t * type(pos_i)(
                type(pos_i[0])(shift_vec[0]),
                type(pos_i[0])(shift_vec[1]),
                type(pos_i[0])(shift_vec[2]),
            )
            sep = pos_j - pos_i + periodic_shift
            distance = wp.float64(wp.length(sep))

            if distance > wp.float64(1e-8):
                if with_pos_grad or with_charge_grad:
                    terms = _gto_ewald_monopole_pair_terms_fused(
                        distance, a_coef, b_coef
                    )
                    t0 = terms.t0
                    a_scalar = terms.a_scalar
                else:
                    t0 = _gto_ewald_t0(distance, a_coef, b_coef)
                    a_scalar = wp.float64(0.0)

                energy_acc += wp.float64(0.5) * qi * qj * t0

                if with_charge_grad:
                    half_t0 = wp.float64(0.5) * t0
                    wp.atomic_add(
                        grad_charges,
                        atom_i,
                        type(charges[atom_i])(half_t0 * qj),
                    )
                    wp.atomic_add(
                        grad_charges,
                        j,
                        type(charges[atom_i])(half_t0 * qi),
                    )

                if with_pos_grad:
                    inv_r = wp.float64(1.0) / distance
                    pos_coeff = wp.float64(0.5) * qi * qj * a_scalar * inv_r
                    dx = pos_coeff * wp.float64(sep[0])
                    dy = pos_coeff * wp.float64(sep[1])
                    dz = pos_coeff * wp.float64(sep[2])
                    wp.atomic_add(
                        grad_positions,
                        atom_i,
                        type(pos_i)(
                            type(pos_i[0])(dx),
                            type(pos_i[0])(dy),
                            type(pos_i[0])(dz),
                        ),
                    )
                    wp.atomic_add(
                        grad_positions,
                        j,
                        type(pos_i)(
                            type(pos_i[0])(-dx),
                            type(pos_i[0])(-dy),
                            type(pos_i[0])(-dz),
                        ),
                    )

        wp.atomic_add(pair_energies, atom_i, energy_acc)

    return _kernel


def _make_real_space_pair_kernel_csr_single_dipole(
    *, with_pos_grad: bool, with_charge_grad: bool, with_dipole_grad: bool
):
    """LMAX=1 CSR neighbor-list + single-system builder.

    Uses ``_dipole_pair_contribution_fused`` when any gradient flag is
    True, and ``_dipole_pair_energy_only`` for the all-flags-False path.
    """
    any_grad = with_pos_grad or with_charge_grad or with_dipole_grad

    @wp.kernel(enable_backward=False)
    def _kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        cell: wp.array(dtype=Any),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        sigma: wp.array(dtype=Any),
        alpha: wp.array(dtype=Any),
        pair_energies: wp.array(dtype=wp.float64),
        grad_positions: wp.array(dtype=Any),
        grad_charges: wp.array(dtype=Any),
        grad_dipoles: wp.array(dtype=Any),
    ):
        r"""Unified real-space pair kernel (LMAX=1 CSR single).

        Body branches at codegen time on the ``with_pos_grad`` /
        ``with_charge_grad`` / ``with_dipole_grad`` closure flags; uses the
        fused ``_dipole_pair_*`` helpers (charge-charge + charge-dipole +
        dipole-dipole channels).

        Launch Grid
        -----------
        dim = [num_atoms] — one thread per atom ``i``; inner loop over the
        atom's CSR neighbor slice.

        Parameters
        ----------
        positions : wp.array, shape (N,), dtype wp.vec3f or wp.vec3d
            Atomic positions.
        charges : wp.array, shape (N,), dtype wp.float32 or wp.float64
            Atomic charges.
        dipoles : wp.array, shape (N,), dtype matching ``positions``
            Cartesian dipole moments ``(x, y, z)``.
        cell : wp.array, shape (1,), dtype wp.mat33f or wp.mat33d
            Lattice matrix.
        idx_j : wp.array, shape (M,), dtype wp.int32
            CSR neighbor target indices.
        neighbor_ptr : wp.array, shape (N+1,), dtype wp.int32
            CSR row pointers into ``idx_j`` / ``unit_shifts``.
        unit_shifts : wp.array, shape (M,), dtype wp.vec3i
            Per-edge periodic image shifts.
        sigma : wp.array, shape (1,), dtype matching ``charges``
            GTO smearing width.
        alpha : wp.array, shape (1,), dtype matching ``charges``
            Ewald splitting parameter.
        pair_energies : wp.array, shape (N,), dtype wp.float64
            OUTPUT (pre-zeroed). Per-atom accumulated energy.
        grad_positions : wp.array, shape (N,), dtype matching ``positions``
            OUTPUT. Written only when ``with_pos_grad``.
        grad_charges : wp.array, shape (N,), dtype matching ``charges``
            OUTPUT. Written only when ``with_charge_grad``.
        grad_dipoles : wp.array, shape (N,), dtype matching ``dipoles``
            OUTPUT. Written only when ``with_dipole_grad``.
        """
        atom_i = wp.tid()

        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        mu_i_native = dipoles[atom_i]
        mu_i = wp.vec3d(
            wp.float64(mu_i_native[0]),
            wp.float64(mu_i_native[1]),
            wp.float64(mu_i_native[2]),
        )

        sigma_ = wp.float64(sigma[0])
        alpha_ = wp.float64(alpha[0])
        ab = _gto_ewald_ab(sigma_, alpha_)
        a_coef = ab[0]
        b_coef = ab[1]

        cell_t = wp.transpose(cell[0])

        energy_acc = wp.float64(0.0)
        j_range_start = neighbor_ptr[atom_i]
        j_range_end = neighbor_ptr[atom_i + 1]

        for edge_idx in range(j_range_start, j_range_end):
            j = idx_j[edge_idx]
            qj = wp.float64(charges[j])
            pos_j = positions[j]
            mu_j_native = dipoles[j]
            mu_j = wp.vec3d(
                wp.float64(mu_j_native[0]),
                wp.float64(mu_j_native[1]),
                wp.float64(mu_j_native[2]),
            )

            shift_vec = unit_shifts[edge_idx]
            periodic_shift = cell_t * type(pos_i)(
                type(pos_i[0])(shift_vec[0]),
                type(pos_i[0])(shift_vec[1]),
                type(pos_i[0])(shift_vec[2]),
            )

            sep = pos_j - pos_i + periodic_shift
            distance = wp.float64(wp.length(sep))

            if distance > wp.float64(1e-8):
                r_vec = wp.vec3d(
                    wp.float64(sep[0]),
                    wp.float64(sep[1]),
                    wp.float64(sep[2]),
                )

                if any_grad:
                    contrib = _dipole_pair_contribution_fused(
                        r_vec, distance, qi, mu_i, qj, mu_j, a_coef, b_coef
                    )
                    energy_acc += wp.float64(0.5) * contrib.energy

                    if with_charge_grad:
                        half = wp.float64(0.5)
                        wp.atomic_add(
                            grad_charges,
                            atom_i,
                            type(charges[atom_i])(half * contrib.dPE_dq_i),
                        )
                        wp.atomic_add(
                            grad_charges,
                            j,
                            type(charges[atom_i])(half * contrib.dPE_dq_j),
                        )

                    if with_dipole_grad:
                        half = wp.float64(0.5)
                        wp.atomic_add(
                            grad_dipoles,
                            atom_i,
                            type(mu_i_native)(
                                type(mu_i_native[0])(half * contrib.dPE_dmu_i[0]),
                                type(mu_i_native[0])(half * contrib.dPE_dmu_i[1]),
                                type(mu_i_native[0])(half * contrib.dPE_dmu_i[2]),
                            ),
                        )
                        wp.atomic_add(
                            grad_dipoles,
                            j,
                            type(mu_i_native)(
                                type(mu_i_native[0])(half * contrib.dPE_dmu_j[0]),
                                type(mu_i_native[0])(half * contrib.dPE_dmu_j[1]),
                                type(mu_i_native[0])(half * contrib.dPE_dmu_j[2]),
                            ),
                        )

                    if with_pos_grad:
                        half = wp.float64(0.5)
                        wp.atomic_add(
                            grad_positions,
                            j,
                            type(pos_i)(
                                type(pos_i[0])(half * contrib.dPE_dr_j[0]),
                                type(pos_i[0])(half * contrib.dPE_dr_j[1]),
                                type(pos_i[0])(half * contrib.dPE_dr_j[2]),
                            ),
                        )
                        wp.atomic_add(
                            grad_positions,
                            atom_i,
                            type(pos_i)(
                                type(pos_i[0])(-half * contrib.dPE_dr_j[0]),
                                type(pos_i[0])(-half * contrib.dPE_dr_j[1]),
                                type(pos_i[0])(-half * contrib.dPE_dr_j[2]),
                            ),
                        )
                else:
                    pe = _dipole_pair_energy_only(
                        r_vec, distance, qi, mu_i, qj, mu_j, a_coef, b_coef
                    )
                    energy_acc += wp.float64(0.5) * pe

        wp.atomic_add(pair_energies, atom_i, energy_acc)

    return _kernel


def _make_real_space_pair_kernel_csr_batched_dipole(
    *, with_pos_grad: bool, with_charge_grad: bool, with_dipole_grad: bool
):
    """LMAX=1 CSR + batched builder.

    Adds per-atom ``batch_idx`` lookup over the LMAX=1 CSR single-system
    body.
    """
    any_grad = with_pos_grad or with_charge_grad or with_dipole_grad

    @wp.kernel(enable_backward=False)
    def _kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        cells: wp.array(dtype=Any),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        sigmas: wp.array(dtype=Any),
        alphas: wp.array(dtype=Any),
        batch_idx: wp.array(dtype=wp.int32),
        pair_energies: wp.array(dtype=wp.float64),
        grad_positions: wp.array(dtype=Any),
        grad_charges: wp.array(dtype=Any),
        grad_dipoles: wp.array(dtype=Any),
    ):
        r"""Unified real-space pair kernel (LMAX=1 CSR batched).

        Per-system ``cells[b]`` / ``sigmas[b]`` / ``alphas[b]`` lookup via
        ``b = batch_idx[i]``. Body branches at codegen time on the
        ``with_pos_grad`` / ``with_charge_grad`` / ``with_dipole_grad``
        closure flags.

        Launch Grid
        -----------
        dim = [num_atoms_total] — one thread per atom across all batched
        systems; inner loop over the atom's CSR neighbor slice.

        Parameters
        ----------
        positions : wp.array, shape (N_total,), dtype wp.vec3f or wp.vec3d
            Concatenated atomic positions across all systems.
        charges : wp.array, shape (N_total,), dtype wp.float32 or wp.float64
            Concatenated atomic charges.
        dipoles : wp.array, shape (N_total,), dtype matching ``positions``
            Concatenated Cartesian dipole moments ``(x, y, z)``.
        cells : wp.array, shape (B,), dtype wp.mat33f or wp.mat33d
            Per-system lattice matrices.
        idx_j : wp.array, shape (M,), dtype wp.int32
            Flattened CSR neighbor target indices.
        neighbor_ptr : wp.array, shape (N_total+1,), dtype wp.int32
            CSR row pointers into ``idx_j`` / ``unit_shifts``.
        unit_shifts : wp.array, shape (M,), dtype wp.vec3i
            Per-edge periodic image shifts.
        sigmas : wp.array, shape (B,), dtype matching ``charges``
            Per-system GTO smearing widths.
        alphas : wp.array, shape (B,), dtype matching ``charges``
            Per-system Ewald splitting parameters.
        batch_idx : wp.array, shape (N_total,), dtype wp.int32
            System index ``b`` for each atom.
        pair_energies : wp.array, shape (N_total,), dtype wp.float64
            OUTPUT (pre-zeroed). Per-atom accumulated energy.
        grad_positions : wp.array, shape (N_total,), dtype matching ``positions``
            OUTPUT. Written only when ``with_pos_grad``.
        grad_charges : wp.array, shape (N_total,), dtype matching ``charges``
            OUTPUT. Written only when ``with_charge_grad``.
        grad_dipoles : wp.array, shape (N_total,), dtype matching ``dipoles``
            OUTPUT. Written only when ``with_dipole_grad``.
        """
        atom_i = wp.tid()
        b = batch_idx[atom_i]

        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        mu_i_native = dipoles[atom_i]
        mu_i = wp.vec3d(
            wp.float64(mu_i_native[0]),
            wp.float64(mu_i_native[1]),
            wp.float64(mu_i_native[2]),
        )

        sigma_ = wp.float64(sigmas[b])
        alpha_ = wp.float64(alphas[b])
        ab = _gto_ewald_ab(sigma_, alpha_)
        a_coef = ab[0]
        b_coef = ab[1]

        cell_t = wp.transpose(cells[b])

        energy_acc = wp.float64(0.0)
        j_range_start = neighbor_ptr[atom_i]
        j_range_end = neighbor_ptr[atom_i + 1]

        for edge_idx in range(j_range_start, j_range_end):
            j = idx_j[edge_idx]
            qj = wp.float64(charges[j])
            pos_j = positions[j]
            mu_j_native = dipoles[j]
            mu_j = wp.vec3d(
                wp.float64(mu_j_native[0]),
                wp.float64(mu_j_native[1]),
                wp.float64(mu_j_native[2]),
            )

            shift_vec = unit_shifts[edge_idx]
            periodic_shift = cell_t * type(pos_i)(
                type(pos_i[0])(shift_vec[0]),
                type(pos_i[0])(shift_vec[1]),
                type(pos_i[0])(shift_vec[2]),
            )

            sep = pos_j - pos_i + periodic_shift
            distance = wp.float64(wp.length(sep))

            if distance > wp.float64(1e-8):
                r_vec = wp.vec3d(
                    wp.float64(sep[0]),
                    wp.float64(sep[1]),
                    wp.float64(sep[2]),
                )

                if any_grad:
                    contrib = _dipole_pair_contribution_fused(
                        r_vec, distance, qi, mu_i, qj, mu_j, a_coef, b_coef
                    )
                    energy_acc += wp.float64(0.5) * contrib.energy

                    if with_charge_grad:
                        half = wp.float64(0.5)
                        wp.atomic_add(
                            grad_charges,
                            atom_i,
                            type(charges[atom_i])(half * contrib.dPE_dq_i),
                        )
                        wp.atomic_add(
                            grad_charges,
                            j,
                            type(charges[atom_i])(half * contrib.dPE_dq_j),
                        )

                    if with_dipole_grad:
                        half = wp.float64(0.5)
                        wp.atomic_add(
                            grad_dipoles,
                            atom_i,
                            type(mu_i_native)(
                                type(mu_i_native[0])(half * contrib.dPE_dmu_i[0]),
                                type(mu_i_native[0])(half * contrib.dPE_dmu_i[1]),
                                type(mu_i_native[0])(half * contrib.dPE_dmu_i[2]),
                            ),
                        )
                        wp.atomic_add(
                            grad_dipoles,
                            j,
                            type(mu_i_native)(
                                type(mu_i_native[0])(half * contrib.dPE_dmu_j[0]),
                                type(mu_i_native[0])(half * contrib.dPE_dmu_j[1]),
                                type(mu_i_native[0])(half * contrib.dPE_dmu_j[2]),
                            ),
                        )

                    if with_pos_grad:
                        half = wp.float64(0.5)
                        wp.atomic_add(
                            grad_positions,
                            j,
                            type(pos_i)(
                                type(pos_i[0])(half * contrib.dPE_dr_j[0]),
                                type(pos_i[0])(half * contrib.dPE_dr_j[1]),
                                type(pos_i[0])(half * contrib.dPE_dr_j[2]),
                            ),
                        )
                        wp.atomic_add(
                            grad_positions,
                            atom_i,
                            type(pos_i)(
                                type(pos_i[0])(-half * contrib.dPE_dr_j[0]),
                                type(pos_i[0])(-half * contrib.dPE_dr_j[1]),
                                type(pos_i[0])(-half * contrib.dPE_dr_j[2]),
                            ),
                        )
                else:
                    pe = _dipole_pair_energy_only(
                        r_vec, distance, qi, mu_i, qj, mu_j, a_coef, b_coef
                    )
                    energy_acc += wp.float64(0.5) * pe

        wp.atomic_add(pair_energies, atom_i, energy_acc)

    return _kernel


# =============================================================================
# LMAX=2 internal builders.
# =============================================================================


def _make_real_space_pair_kernel_csr_single_quadrupole(
    *,
    with_pos_grad: bool = False,
    with_charge_grad: bool = False,
    with_dipole_grad: bool = False,
    with_quad_grad: bool = False,
):
    """LMAX=2 CSR + single-system builder.

    Per-pair gradient scatters are weighted by the half-pair convention
    composed with the per-atom upstream cotangent ``grad_energies``:

      w = 0.25 * (grad_energies[atom_i] + grad_energies[atom_j])

    Full neighbor lists visit each pair twice (i→j and j→i); the 0.25
    factor is 0.5 (half-pair) × 0.5 (symmetric (ge_i + ge_j)/2 weighting).
    With ``grad_energies = ones(N)`` w = 0.5.

    The energy-only branch (``any_grad=False``) does not read
    ``grad_energies`` — callers may pass a scratch array.
    """
    any_grad = with_pos_grad or with_charge_grad or with_dipole_grad or with_quad_grad

    @wp.kernel(enable_backward=False)
    def _kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        quadrupoles: wp.array(dtype=Any),
        cell: wp.array(dtype=Any),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        sigma: wp.array(dtype=Any),
        alpha: wp.array(dtype=Any),
        grad_energies: wp.array(dtype=wp.float64),
        pair_energies: wp.array(dtype=wp.float64),
        grad_positions: wp.array(dtype=Any),
        grad_charges: wp.array(dtype=Any),
        grad_dipoles: wp.array(dtype=Any),
        grad_quadrupoles: wp.array(dtype=Any),
    ):
        r"""Unified real-space pair kernel (LMAX=2 CSR single).

        Computes all 6 channels (qq, qμ, qQ, μμ, μQ, QQ); body branches at
        codegen time on the four ``with_*_grad`` closure flags. Per-pair
        gradient scatters are weighted by ``0.25 * (grad_energies[i] +
        grad_energies[j])`` (pass ``grad_energies = 1`` for uniform 0.5
        half-pair weighting).

        Launch Grid
        -----------
        dim = [num_atoms] — one thread per atom ``i``; inner loop over the
        atom's CSR neighbor slice.

        Parameters
        ----------
        positions : wp.array, shape (N,), dtype wp.vec3f or wp.vec3d
            Atomic positions.
        charges : wp.array, shape (N,), dtype wp.float32 or wp.float64
            Atomic charges.
        dipoles : wp.array, shape (N,), dtype matching ``positions``
            Cartesian dipole moments ``(x, y, z)``.
        quadrupoles : wp.array, shape (N,), dtype wp.mat33f or wp.mat33d
            Cartesian (traceless) quadrupole moment matrices.
        cell : wp.array, shape (1,), dtype wp.mat33f or wp.mat33d
            Lattice matrix.
        idx_j : wp.array, shape (M,), dtype wp.int32
            CSR neighbor target indices.
        neighbor_ptr : wp.array, shape (N+1,), dtype wp.int32
            CSR row pointers into ``idx_j`` / ``unit_shifts``.
        unit_shifts : wp.array, shape (M,), dtype wp.vec3i
            Per-edge periodic image shifts.
        sigma : wp.array, shape (1,), dtype matching ``charges``
            GTO smearing width.
        alpha : wp.array, shape (1,), dtype matching ``charges``
            Ewald splitting parameter.
        grad_energies : wp.array, shape (N,), dtype wp.float64
            Upstream cotangent ``∂L/∂pair_energies`` for the gradient
            weights; unused (1-element scratch) on the energy-only path.
        pair_energies : wp.array, shape (N,), dtype wp.float64
            OUTPUT (pre-zeroed). Per-atom accumulated energy.
        grad_positions : wp.array, shape (N,), dtype matching ``positions``
            OUTPUT. Written only when ``with_pos_grad``.
        grad_charges : wp.array, shape (N,), dtype matching ``charges``
            OUTPUT. Written only when ``with_charge_grad``.
        grad_dipoles : wp.array, shape (N,), dtype matching ``dipoles``
            OUTPUT. Written only when ``with_dipole_grad``.
        grad_quadrupoles : wp.array, shape (N,), dtype matching ``quadrupoles``
            OUTPUT. Written only when ``with_quad_grad``.
        """
        atom_i = wp.tid()
        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        mu_i_native = dipoles[atom_i]
        mu_i = wp.vec3d(
            wp.float64(mu_i_native[0]),
            wp.float64(mu_i_native[1]),
            wp.float64(mu_i_native[2]),
        )
        Q_i_native = quadrupoles[atom_i]
        Q_i = wp.mat33d(
            wp.float64(Q_i_native[0, 0]),
            wp.float64(Q_i_native[0, 1]),
            wp.float64(Q_i_native[0, 2]),
            wp.float64(Q_i_native[1, 0]),
            wp.float64(Q_i_native[1, 1]),
            wp.float64(Q_i_native[1, 2]),
            wp.float64(Q_i_native[2, 0]),
            wp.float64(Q_i_native[2, 1]),
            wp.float64(Q_i_native[2, 2]),
        )

        sigma_ = wp.float64(sigma[0])
        alpha_ = wp.float64(alpha[0])
        ab = _gto_ewald_ab(sigma_, alpha_)
        a_coef = ab[0]
        b_coef = ab[1]
        cell_t = wp.transpose(cell[0])

        energy_acc = wp.float64(0.0)
        j_range_start = neighbor_ptr[atom_i]
        j_range_end = neighbor_ptr[atom_i + 1]

        for edge_idx in range(j_range_start, j_range_end):
            j = idx_j[edge_idx]
            qj = wp.float64(charges[j])
            pos_j = positions[j]
            mu_j_native = dipoles[j]
            mu_j = wp.vec3d(
                wp.float64(mu_j_native[0]),
                wp.float64(mu_j_native[1]),
                wp.float64(mu_j_native[2]),
            )
            Q_j_native = quadrupoles[j]
            Q_j = wp.mat33d(
                wp.float64(Q_j_native[0, 0]),
                wp.float64(Q_j_native[0, 1]),
                wp.float64(Q_j_native[0, 2]),
                wp.float64(Q_j_native[1, 0]),
                wp.float64(Q_j_native[1, 1]),
                wp.float64(Q_j_native[1, 2]),
                wp.float64(Q_j_native[2, 0]),
                wp.float64(Q_j_native[2, 1]),
                wp.float64(Q_j_native[2, 2]),
            )

            shift_vec = unit_shifts[edge_idx]
            periodic_shift = cell_t * type(pos_i)(
                type(pos_i[0])(shift_vec[0]),
                type(pos_i[0])(shift_vec[1]),
                type(pos_i[0])(shift_vec[2]),
            )
            sep = pos_j - pos_i + periodic_shift
            distance = wp.float64(wp.length(sep))

            if distance > wp.float64(1e-8):
                r_vec = wp.vec3d(
                    wp.float64(sep[0]),
                    wp.float64(sep[1]),
                    wp.float64(sep[2]),
                )
                if any_grad:
                    contrib = _quadrupole_pair_contribution_fused(
                        r_vec,
                        distance,
                        qi,
                        mu_i,
                        Q_i,
                        qj,
                        mu_j,
                        Q_j,
                        a_coef,
                        b_coef,
                    )
                    energy_acc += wp.float64(0.5) * contrib.energy
                    w = wp.float64(0.25) * (grad_energies[atom_i] + grad_energies[j])
                    if with_charge_grad:
                        wp.atomic_add(
                            grad_charges,
                            atom_i,
                            type(charges[atom_i])(w * contrib.dPE_dq_i),
                        )
                        wp.atomic_add(
                            grad_charges,
                            j,
                            type(charges[atom_i])(w * contrib.dPE_dq_j),
                        )
                    if with_dipole_grad:
                        wp.atomic_add(
                            grad_dipoles,
                            atom_i,
                            type(mu_i_native)(
                                type(mu_i_native[0])(w * contrib.dPE_dmu_i[0]),
                                type(mu_i_native[0])(w * contrib.dPE_dmu_i[1]),
                                type(mu_i_native[0])(w * contrib.dPE_dmu_i[2]),
                            ),
                        )
                        wp.atomic_add(
                            grad_dipoles,
                            j,
                            type(mu_i_native)(
                                type(mu_i_native[0])(w * contrib.dPE_dmu_j[0]),
                                type(mu_i_native[0])(w * contrib.dPE_dmu_j[1]),
                                type(mu_i_native[0])(w * contrib.dPE_dmu_j[2]),
                            ),
                        )
                    if with_quad_grad:
                        wp.atomic_add(
                            grad_quadrupoles,
                            atom_i,
                            type(Q_i_native)(
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[0, 0]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[0, 1]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[0, 2]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[1, 0]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[1, 1]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[1, 2]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[2, 0]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[2, 1]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[2, 2]),
                            ),
                        )
                        wp.atomic_add(
                            grad_quadrupoles,
                            j,
                            type(Q_i_native)(
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[0, 0]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[0, 1]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[0, 2]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[1, 0]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[1, 1]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[1, 2]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[2, 0]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[2, 1]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[2, 2]),
                            ),
                        )
                    if with_pos_grad:
                        wp.atomic_add(
                            grad_positions,
                            j,
                            type(pos_i)(
                                type(pos_i[0])(w * contrib.dPE_dr_j[0]),
                                type(pos_i[0])(w * contrib.dPE_dr_j[1]),
                                type(pos_i[0])(w * contrib.dPE_dr_j[2]),
                            ),
                        )
                        wp.atomic_add(
                            grad_positions,
                            atom_i,
                            type(pos_i)(
                                type(pos_i[0])(-w * contrib.dPE_dr_j[0]),
                                type(pos_i[0])(-w * contrib.dPE_dr_j[1]),
                                type(pos_i[0])(-w * contrib.dPE_dr_j[2]),
                            ),
                        )
                else:
                    pe = _quadrupole_pair_energy_only(
                        r_vec, distance, qi, mu_i, Q_i, qj, mu_j, Q_j, a_coef, b_coef
                    )
                    energy_acc += wp.float64(0.5) * pe

        wp.atomic_add(pair_energies, atom_i, energy_acc)

    return _kernel


def _make_real_space_pair_kernel_csr_batched_quadrupole(
    *,
    with_pos_grad: bool = False,
    with_charge_grad: bool = False,
    with_dipole_grad: bool = False,
    with_quad_grad: bool = False,
):
    """LMAX=2 CSR + batched builder.

    Per-atom upstream-cotangent weighting matches the single-system
    builder: ``w = 0.25 * (grad_energies[atom_i] + grad_energies[atom_j])``.
    See :func:`_make_real_space_pair_kernel_csr_single_quadrupole` for the
    derivation.
    """
    any_grad = with_pos_grad or with_charge_grad or with_dipole_grad or with_quad_grad

    @wp.kernel(enable_backward=False)
    def _kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        quadrupoles: wp.array(dtype=Any),
        cells: wp.array(dtype=Any),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        sigmas: wp.array(dtype=Any),
        alphas: wp.array(dtype=Any),
        batch_idx: wp.array(dtype=wp.int32),
        grad_energies: wp.array(dtype=wp.float64),
        pair_energies: wp.array(dtype=wp.float64),
        grad_positions: wp.array(dtype=Any),
        grad_charges: wp.array(dtype=Any),
        grad_dipoles: wp.array(dtype=Any),
        grad_quadrupoles: wp.array(dtype=Any),
    ):
        r"""Unified real-space pair kernel (LMAX=2 CSR batched).

        Per-system ``cells[b]`` / ``sigmas[b]`` / ``alphas[b]`` lookup via
        ``b = batch_idx[i]``. Computes all 6 channels (qq, qμ, qQ, μμ, μQ,
        QQ); body branches at codegen time on the four ``with_*_grad``
        closure flags. Per-pair gradient scatters are weighted by
        ``0.25 * (grad_energies[i] + grad_energies[j])``.

        Launch Grid
        -----------
        dim = [num_atoms_total] — one thread per atom across all batched
        systems; inner loop over the atom's CSR neighbor slice.

        Parameters
        ----------
        positions : wp.array, shape (N_total,), dtype wp.vec3f or wp.vec3d
            Concatenated atomic positions across all systems.
        charges : wp.array, shape (N_total,), dtype wp.float32 or wp.float64
            Concatenated atomic charges.
        dipoles : wp.array, shape (N_total,), dtype matching ``positions``
            Concatenated Cartesian dipole moments ``(x, y, z)``.
        quadrupoles : wp.array, shape (N_total,), dtype wp.mat33f or wp.mat33d
            Concatenated Cartesian (traceless) quadrupole moment matrices.
        cells : wp.array, shape (B,), dtype wp.mat33f or wp.mat33d
            Per-system lattice matrices.
        idx_j : wp.array, shape (M,), dtype wp.int32
            Flattened CSR neighbor target indices.
        neighbor_ptr : wp.array, shape (N_total+1,), dtype wp.int32
            CSR row pointers into ``idx_j`` / ``unit_shifts``.
        unit_shifts : wp.array, shape (M,), dtype wp.vec3i
            Per-edge periodic image shifts.
        sigmas : wp.array, shape (B,), dtype matching ``charges``
            Per-system GTO smearing widths.
        alphas : wp.array, shape (B,), dtype matching ``charges``
            Per-system Ewald splitting parameters.
        batch_idx : wp.array, shape (N_total,), dtype wp.int32
            System index ``b`` for each atom.
        grad_energies : wp.array, shape (N_total,), dtype wp.float64
            Upstream cotangent ``∂L/∂pair_energies`` for the gradient
            weights; unused (1-element scratch) on the energy-only path.
        pair_energies : wp.array, shape (N_total,), dtype wp.float64
            OUTPUT (pre-zeroed). Per-atom accumulated energy.
        grad_positions : wp.array, shape (N_total,), dtype matching ``positions``
            OUTPUT. Written only when ``with_pos_grad``.
        grad_charges : wp.array, shape (N_total,), dtype matching ``charges``
            OUTPUT. Written only when ``with_charge_grad``.
        grad_dipoles : wp.array, shape (N_total,), dtype matching ``dipoles``
            OUTPUT. Written only when ``with_dipole_grad``.
        grad_quadrupoles : wp.array, shape (N_total,), dtype matching ``quadrupoles``
            OUTPUT. Written only when ``with_quad_grad``.
        """
        atom_i = wp.tid()
        b = batch_idx[atom_i]

        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        mu_i_native = dipoles[atom_i]
        mu_i = wp.vec3d(
            wp.float64(mu_i_native[0]),
            wp.float64(mu_i_native[1]),
            wp.float64(mu_i_native[2]),
        )
        Q_i_native = quadrupoles[atom_i]
        Q_i = wp.mat33d(
            wp.float64(Q_i_native[0, 0]),
            wp.float64(Q_i_native[0, 1]),
            wp.float64(Q_i_native[0, 2]),
            wp.float64(Q_i_native[1, 0]),
            wp.float64(Q_i_native[1, 1]),
            wp.float64(Q_i_native[1, 2]),
            wp.float64(Q_i_native[2, 0]),
            wp.float64(Q_i_native[2, 1]),
            wp.float64(Q_i_native[2, 2]),
        )

        sigma_ = wp.float64(sigmas[b])
        alpha_ = wp.float64(alphas[b])
        ab = _gto_ewald_ab(sigma_, alpha_)
        a_coef = ab[0]
        b_coef = ab[1]
        cell_t = wp.transpose(cells[b])

        energy_acc = wp.float64(0.0)
        j_range_start = neighbor_ptr[atom_i]
        j_range_end = neighbor_ptr[atom_i + 1]
        for edge_idx in range(j_range_start, j_range_end):
            j = idx_j[edge_idx]
            qj = wp.float64(charges[j])
            pos_j = positions[j]
            mu_j_native = dipoles[j]
            mu_j = wp.vec3d(
                wp.float64(mu_j_native[0]),
                wp.float64(mu_j_native[1]),
                wp.float64(mu_j_native[2]),
            )
            Q_j_native = quadrupoles[j]
            Q_j = wp.mat33d(
                wp.float64(Q_j_native[0, 0]),
                wp.float64(Q_j_native[0, 1]),
                wp.float64(Q_j_native[0, 2]),
                wp.float64(Q_j_native[1, 0]),
                wp.float64(Q_j_native[1, 1]),
                wp.float64(Q_j_native[1, 2]),
                wp.float64(Q_j_native[2, 0]),
                wp.float64(Q_j_native[2, 1]),
                wp.float64(Q_j_native[2, 2]),
            )

            shift_vec = unit_shifts[edge_idx]
            periodic_shift = cell_t * type(pos_i)(
                type(pos_i[0])(shift_vec[0]),
                type(pos_i[0])(shift_vec[1]),
                type(pos_i[0])(shift_vec[2]),
            )
            sep = pos_j - pos_i + periodic_shift
            distance = wp.float64(wp.length(sep))
            if distance > wp.float64(1e-8):
                r_vec = wp.vec3d(
                    wp.float64(sep[0]),
                    wp.float64(sep[1]),
                    wp.float64(sep[2]),
                )
                if any_grad:
                    contrib = _quadrupole_pair_contribution_fused(
                        r_vec,
                        distance,
                        qi,
                        mu_i,
                        Q_i,
                        qj,
                        mu_j,
                        Q_j,
                        a_coef,
                        b_coef,
                    )
                    energy_acc += wp.float64(0.5) * contrib.energy
                    w = wp.float64(0.25) * (grad_energies[atom_i] + grad_energies[j])
                    if with_charge_grad:
                        wp.atomic_add(
                            grad_charges,
                            atom_i,
                            type(charges[atom_i])(w * contrib.dPE_dq_i),
                        )
                        wp.atomic_add(
                            grad_charges,
                            j,
                            type(charges[atom_i])(w * contrib.dPE_dq_j),
                        )
                    if with_dipole_grad:
                        wp.atomic_add(
                            grad_dipoles,
                            atom_i,
                            type(mu_i_native)(
                                type(mu_i_native[0])(w * contrib.dPE_dmu_i[0]),
                                type(mu_i_native[0])(w * contrib.dPE_dmu_i[1]),
                                type(mu_i_native[0])(w * contrib.dPE_dmu_i[2]),
                            ),
                        )
                        wp.atomic_add(
                            grad_dipoles,
                            j,
                            type(mu_i_native)(
                                type(mu_i_native[0])(w * contrib.dPE_dmu_j[0]),
                                type(mu_i_native[0])(w * contrib.dPE_dmu_j[1]),
                                type(mu_i_native[0])(w * contrib.dPE_dmu_j[2]),
                            ),
                        )
                    if with_quad_grad:
                        wp.atomic_add(
                            grad_quadrupoles,
                            atom_i,
                            type(Q_i_native)(
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[0, 0]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[0, 1]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[0, 2]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[1, 0]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[1, 1]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[1, 2]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[2, 0]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[2, 1]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_i[2, 2]),
                            ),
                        )
                        wp.atomic_add(
                            grad_quadrupoles,
                            j,
                            type(Q_i_native)(
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[0, 0]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[0, 1]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[0, 2]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[1, 0]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[1, 1]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[1, 2]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[2, 0]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[2, 1]),
                                type(Q_i_native[0, 0])(w * contrib.dPE_dQ_j[2, 2]),
                            ),
                        )
                    if with_pos_grad:
                        wp.atomic_add(
                            grad_positions,
                            j,
                            type(pos_i)(
                                type(pos_i[0])(w * contrib.dPE_dr_j[0]),
                                type(pos_i[0])(w * contrib.dPE_dr_j[1]),
                                type(pos_i[0])(w * contrib.dPE_dr_j[2]),
                            ),
                        )
                        wp.atomic_add(
                            grad_positions,
                            atom_i,
                            type(pos_i)(
                                type(pos_i[0])(-w * contrib.dPE_dr_j[0]),
                                type(pos_i[0])(-w * contrib.dPE_dr_j[1]),
                                type(pos_i[0])(-w * contrib.dPE_dr_j[2]),
                            ),
                        )
                else:
                    pe = _quadrupole_pair_energy_only(
                        r_vec, distance, qi, mu_i, Q_i, qj, mu_j, Q_j, a_coef, b_coef
                    )
                    energy_acc += wp.float64(0.5) * pe

        wp.atomic_add(pair_energies, atom_i, energy_acc)

    return _kernel


def _real_space_pair_sig_monopole_csr_batched(v, t):
    """Signature builder for LMAX=0 + storage='csr' + is_batch=True."""
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=m),  # cells (B, 3, 3)
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigmas (B,)
        wp.array(dtype=t),  # alphas (B,)
        wp.array(dtype=wp.int32),  # batch_idx (N,)
        wp.array(dtype=wp.float64),  # pair_energies
        wp.array(dtype=v),  # grad_positions
        wp.array(dtype=t),  # grad_charges
    ]


def _real_space_pair_sig_dipole_csr(v, t):
    """Signature builder for the LMAX=1 CSR single-system kernel.

    Adds ``dipoles`` after ``charges`` and ``grad_dipoles`` after
    ``grad_charges``; the rest of the layout matches
    :func:`_real_space_pair_sig_monopole_csr`.
    """
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=m),  # cell
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # alpha
        wp.array(dtype=wp.float64),  # pair_energies
        wp.array(dtype=v),  # grad_positions
        wp.array(dtype=t),  # grad_charges
        wp.array(dtype=v),  # grad_dipoles
    ]


def _real_space_pair_sig_quadrupole_csr(v, t):
    """LMAX=2 CSR single-system signature.

    Includes the per-atom ``grad_energies`` upstream cotangent between
    ``alpha`` and ``pair_energies``.
    """
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=m),  # quadrupoles
        wp.array(dtype=m),  # cell
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # alpha
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=wp.float64),  # pair_energies
        wp.array(dtype=v),  # grad_positions
        wp.array(dtype=t),  # grad_charges
        wp.array(dtype=v),  # grad_dipoles
        wp.array(dtype=m),  # grad_quadrupoles
    ]


def _real_space_pair_sig_quadrupole_csr_batched(v, t):
    """LMAX=2 CSR batched signature.

    Includes the per-atom ``grad_energies`` upstream cotangent between
    ``batch_idx`` and ``pair_energies``.
    """
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=m),  # quadrupoles
        wp.array(dtype=m),  # cells (B, 3, 3)
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigmas (B,)
        wp.array(dtype=t),  # alphas (B,)
        wp.array(dtype=wp.int32),  # batch_idx
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=wp.float64),  # pair_energies
        wp.array(dtype=v),  # grad_positions
        wp.array(dtype=t),  # grad_charges
        wp.array(dtype=v),  # grad_dipoles
        wp.array(dtype=m),  # grad_quadrupoles
    ]


def _real_space_pair_sig_dipole_csr_batched(v, t):
    """Signature builder for the LMAX=1 CSR batched kernel.

    Extends :func:`_real_space_pair_sig_monopole_csr_batched` with one
    ``dipoles`` input after ``charges`` and one ``grad_dipoles`` output
    after ``grad_charges``.
    """
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=m),  # cells (B, 3, 3)
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigmas (B,)
        wp.array(dtype=t),  # alphas (B,)
        wp.array(dtype=wp.int32),  # batch_idx (N,)
        wp.array(dtype=wp.float64),  # pair_energies
        wp.array(dtype=v),  # grad_positions
        wp.array(dtype=t),  # grad_charges
        wp.array(dtype=v),  # grad_dipoles
    ]


def _real_space_pair_sig_monopole_csr(v, t):
    """Signature builder for the LMAX=0 CSR single-system kernel."""
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=m),  # cell
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # alpha
        wp.array(dtype=wp.float64),  # pair_energies
        wp.array(dtype=v),  # grad_positions
        wp.array(dtype=t),  # grad_charges
    ]


_REAL_SPACE_PAIR_SIG_BUILDERS: dict[tuple[int, str, bool], callable] = {
    (0, "csr", False): _real_space_pair_sig_monopole_csr,
    (0, "csr", True): _real_space_pair_sig_monopole_csr_batched,
    (1, "csr", False): _real_space_pair_sig_dipole_csr,
    (1, "csr", True): _real_space_pair_sig_dipole_csr_batched,
    (2, "csr", False): _real_space_pair_sig_quadrupole_csr,
    (2, "csr", True): _real_space_pair_sig_quadrupole_csr_batched,
}

_REAL_SPACE_PAIR_KERNEL_CACHE: dict = {}
_REAL_SPACE_PAIR_OVERLOAD_CACHE: dict = {}


def _get_real_space_pair_overload(
    *,
    LMAX: int,
    storage: str,
    is_batch: bool,
    with_pos_grad: bool,
    with_charge_grad: bool,
    with_dipole_grad: bool,
    with_quad_grad: bool,
    with_cell_grad: bool,
    vec_dtype,
    scalar_dtype,
):
    """Get-or-build the typed overload for a (LMAX, storage, batch, flags, dtype) combo.

    Implements the flag-matrix collapse described above: any non-empty
    grad request maps to the all-grads kernel for that LMAX.
    """
    any_grad = with_pos_grad or with_charge_grad or with_dipole_grad or with_quad_grad
    kernel_key = (
        LMAX,
        storage,
        is_batch,
        any_grad,  # collapsed wpg
        any_grad,  # collapsed wcg
        any_grad and LMAX >= 1,  # wdg gated on LMAX
        any_grad and LMAX >= 2,  # wqg gated on LMAX
        bool(with_cell_grad),  # wcell — passed through verbatim
    )
    if kernel_key not in _REAL_SPACE_PAIR_KERNEL_CACHE:
        _REAL_SPACE_PAIR_KERNEL_CACHE[kernel_key] = _make_real_space_pair_kernel(
            LMAX=LMAX,
            storage=storage,
            is_batch=is_batch,
            with_pos_grad=kernel_key[3],
            with_charge_grad=kernel_key[4],
            with_dipole_grad=kernel_key[5],
            with_quad_grad=kernel_key[6],
            with_cell_grad=kernel_key[7],
        )
    kernel = _REAL_SPACE_PAIR_KERNEL_CACHE[kernel_key]
    overload_key = (*kernel_key, vec_dtype)
    if overload_key not in _REAL_SPACE_PAIR_OVERLOAD_CACHE:
        sig_builder = _REAL_SPACE_PAIR_SIG_BUILDERS[(LMAX, storage, is_batch)]
        sig = sig_builder(vec_dtype, scalar_dtype)
        _REAL_SPACE_PAIR_OVERLOAD_CACHE[overload_key] = wp.overload(kernel, sig)
    return _REAL_SPACE_PAIR_OVERLOAD_CACHE[overload_key]


# =============================================================================
# l_max = 1 (charges + dipoles) — energy, CSR, single-system
# =============================================================================


def multipole_real_space_dipole_csr_energy(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigma: wp.array,
    alpha: wp.array,
    pair_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Energy-only launcher for the LMAX=1 CSR single-system kernel.

    Argument ordering mirrors
    :func:`multipole_real_space_monopole_csr_energy` with one ``dipoles``
    array inserted between ``charges`` and ``cell``. One thread per atom
    ``i`` walks its CSR neighbor slice and accumulates the half-weighted
    charge-charge + charge-dipole + dipole-dipole GTO-Ewald pair energies.
    Internally allocates 1-element scratch arrays for the unused gradient
    slots.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype wp.float32 or wp.float64
        Atomic charges.
    dipoles : wp.array, shape (N,), dtype matching ``positions``
        Cartesian dipole moments ``(x, y, z)``.
    cell : wp.array, shape (1,), dtype wp.mat33f or wp.mat33d
        Lattice matrix.
    idx_j : wp.array, shape (M,), dtype wp.int32
        CSR neighbor target indices.
    neighbor_ptr : wp.array, shape (N+1,), dtype wp.int32
        CSR row pointers into ``idx_j`` / ``unit_shifts``.
    unit_shifts : wp.array, shape (M,), dtype wp.vec3i
        Per-edge periodic image shifts.
    sigma : wp.array, shape (1,), dtype matching ``charges``
        GTO smearing width.
    alpha : wp.array, shape (1,), dtype matching ``charges``
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype wp.float64
        OUTPUT (pre-zeroed). Per-atom accumulated energy.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64``.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f

    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    _overload = _get_real_space_pair_overload(
        LMAX=1,
        storage="csr",
        is_batch=False,
        with_pos_grad=False,
        with_charge_grad=False,
        with_dipole_grad=False,
        with_quad_grad=False,
        with_cell_grad=False,
        vec_dtype=vec_dtype,
        scalar_dtype=wp_dtype,
    )

    # 1-element scratch arrays for the unused gradient slots.
    scratch_grad_pos = wp.zeros(1, dtype=vec_dtype, device=device)
    scratch_grad_q = wp.zeros(
        1,
        dtype=wp.float64 if wp_dtype == wp.float64 else wp.float32,
        device=device,
    )
    scratch_grad_mu = wp.zeros(1, dtype=vec_dtype, device=device)

    wp.launch(
        _overload,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            pair_energies,
            scratch_grad_pos,
            scratch_grad_q,
            scratch_grad_mu,
        ],
        device=device,
    )


# =============================================================================
# l_max = 0 analytical backward kernel — ∂/∂(positions, charges)
# =============================================================================
#
# Given upstream cotangent ``grad_energies[i] = ∂L/∂pair_energies[i]``,
# produces per-atom ``(grad_positions, grad_charges)``. One thread per atom
# walks its own CSR neighbor slice; both self-side and target-side pair-energy
# contributions are emitted via ``atomic_add`` so the same kernel works for
# both half- and full-list conventions.
#
# Per-edge math (forward ``E_ij = 0.5 · q_i · q_j · T^(0)(r_ij, α)``):
#
#   ∂E_ij/∂q_i = 0.5 · q_j · T^(0)
#   ∂E_ij/∂q_j = 0.5 · q_i · T^(0)
#   ∂E_ij/∂r_j = +0.5 · q_i · q_j · T^(1)(r_ij_vec)  = -0.5 · q_i · q_j · (A/r) · r_vec
#
# The position gradient on (i, j) is equal-and-opposite.


@wp.kernel(enable_backward=False)
def _multipole_real_space_monopole_csr_energy_backward_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    sigma: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    grad_energies: wp.array(dtype=wp.float64),
    grad_positions: wp.array(dtype=Any),
    grad_charges: wp.array(dtype=Any),
):
    r"""Analytical backward of :func:`_multipole_real_space_monopole_csr_energy_kernel`.

    Launch Grid
    -----------
    dim = [num_atoms] — one thread per atom; inner loop over CSR neighbors.

    Parameters
    ----------
    positions, charges, cell, idx_j, neighbor_ptr, unit_shifts, sigma, alpha :
        Identical to the forward kernel (the pair geometry + GTO/Ewald
        smearing parameters ``sigma`` and ``alpha``).
    grad_energies : wp.array, shape (N,), dtype wp.float64
        Upstream cotangent ``∂L/∂pair_energies``.
    grad_positions : wp.array, shape (N,), dtype ``vec3f`` / ``vec3d``
        OUTPUT (pre-zeroed). Gradient w.r.t. atomic positions.
    grad_charges : wp.array, shape (N,), dtype matching ``charges``
        OUTPUT (pre-zeroed). Gradient w.r.t. atomic charges.

    Notes
    -----
    All intermediate arithmetic runs in ``float64``; final writes cast back
    to the input dtype via ``atomic_add``. The pair math is symmetric
    across threads (thread ``i`` on edge ``(i, j)`` writes to both
    ``grad_*[i]`` and ``grad_*[j]``), which is why every output slot
    goes through ``atomic_add`` — no thread is the exclusive owner of its
    own slot under arbitrary neighbor-list conventions.
    """
    atom_i = wp.tid()

    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    sigma_ = wp.float64(sigma[0])
    alpha_ = wp.float64(alpha[0])
    cell_t = wp.transpose(cell[0])
    ge_i = grad_energies[atom_i]

    ab = _gto_ewald_ab(sigma_, alpha_)
    a_coef = ab[0]
    b_coef = ab[1]

    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]
        qj = wp.float64(charges[j])
        pos_j = positions[j]

        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )

        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            r_vec = wp.vec3d(
                wp.float64(separation_vector[0]),
                wp.float64(separation_vector[1]),
                wp.float64(separation_vector[2]),
            )

            inv_r = wp.float64(1.0) / distance

            # T^(0) = [erf(ar) − erf(br)] / r.
            t0 = _gto_ewald_t0(distance, a_coef, b_coef)

            # Combined radial helpers: A = A_single(a) − A_single(b).
            ra = _gto_ewald_A_single(distance, a_coef)
            rb = _gto_ewald_A_single(distance, b_coef)
            a_scalar = ra[0] - rb[0]

            # Charge gradients: ∂E_ij/∂q_i = 0.5 · q_j · T^(0),
            #                   ∂E_ij/∂q_j = 0.5 · q_i · T^(0).
            half_t0 = wp.float64(0.5) * t0
            wp.atomic_add(
                grad_charges, atom_i, type(charges[atom_i])(ge_i * half_t0 * qj)
            )
            wp.atomic_add(grad_charges, j, type(charges[atom_i])(ge_i * half_t0 * qi))

            # Position gradients: ∂E_ij/∂r_i = +0.5 · q_i·q_j · (A/r) · r_vec,
            #                     ∂E_ij/∂r_j = -0.5 · q_i·q_j · (A/r) · r_vec.
            pos_coeff = ge_i * wp.float64(0.5) * qi * qj * a_scalar * inv_r
            dx = pos_coeff * r_vec[0]
            dy = pos_coeff * r_vec[1]
            dz = pos_coeff * r_vec[2]
            grad_i_contrib = type(pos_i)(
                type(pos_i[0])(dx),
                type(pos_i[0])(dy),
                type(pos_i[0])(dz),
            )
            wp.atomic_add(grad_positions, atom_i, grad_i_contrib)
            grad_j_contrib = type(pos_i)(
                type(pos_i[0])(-dx),
                type(pos_i[0])(-dy),
                type(pos_i[0])(-dz),
            )
            wp.atomic_add(grad_positions, j, grad_j_contrib)


def _monopole_csr_energy_backward_sig(v, t):
    """Signature builder for :func:`_multipole_real_space_monopole_csr_energy_backward_kernel`."""
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=m),  # cell
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # alpha
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=v),  # grad_positions
        wp.array(dtype=t),  # grad_charges
    ]


_multipole_real_space_monopole_csr_energy_backward_overloads = register_overloads(
    _multipole_real_space_monopole_csr_energy_backward_kernel,
    _monopole_csr_energy_backward_sig,
)


def multipole_real_space_monopole_csr_energy_backward(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigma: wp.array,
    alpha: wp.array,
    grad_energies: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_multipole_real_space_monopole_csr_energy_backward_kernel`.

    Framework-agnostic: operates directly on Warp arrays. Produces
    ``(grad_positions, grad_charges)`` from the upstream cotangent
    ``grad_energies``; the caller is responsible for pre-zeroing the
    output arrays.

    Parameters
    ----------
    positions, charges, cell, idx_j, neighbor_ptr, unit_shifts, sigma, alpha :
        Same semantics as :func:`multipole_real_space_monopole_csr_energy`.
    grad_energies : wp.array, shape (N,), dtype wp.float64
        Upstream cotangent ``∂L/∂pair_energies``.
    grad_positions : wp.array, shape (N,), dtype matching ``positions``
        OUTPUT (pre-zeroed).
    grad_charges : wp.array, shape (N,), dtype matching ``charges``
        OUTPUT (pre-zeroed).
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the overloaded variant.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f

    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _multipole_real_space_monopole_csr_energy_backward_overloads[vec_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            grad_energies,
            grad_positions,
            grad_charges,
        ],
        device=device,
    )


# =============================================================================
# l_max = 0 — fused energy + gradient, CSR neighbor-list, single-system
# =============================================================================
#
# Gradient outputs assume uniform upstream `grad_energies = 1`; the torch
# wrapper scalar-broadcasts the actual upstream gradient.


def multipole_real_space_monopole_csr_energy_fused(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigma: wp.array,
    alpha: wp.array,
    pair_energies: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    *,
    with_pos_grad: bool,
    with_charge_grad: bool,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Fused launcher for the lmax=0 CSR kernels.

    Routes on ``(with_pos_grad, with_charge_grad)``. Output slots are
    written iff the corresponding flag is ``True``; the kernel body's
    Python-time ``if`` guards prevent writes when False. Gradient outputs
    assume uniform upstream ``grad_energies = 1`` — the torch wrapper
    scalar-broadcasts the actual upstream gradient.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (1,), dtype wp.mat33f or wp.mat33d
        Lattice matrix.
    idx_j : wp.array, shape (M,), dtype wp.int32
        CSR neighbor target indices.
    neighbor_ptr : wp.array, shape (N+1,), dtype wp.int32
        CSR row pointers into ``idx_j`` / ``unit_shifts``.
    unit_shifts : wp.array, shape (M,), dtype wp.vec3i
        Per-edge periodic image shifts.
    sigma : wp.array, shape (1,), dtype matching ``charges``
        GTO smearing width.
    alpha : wp.array, shape (1,), dtype matching ``charges``
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype wp.float64
        OUTPUT (pre-zeroed). Per-atom accumulated energy.
    grad_positions : wp.array, shape (N,), dtype matching ``positions``
        OUTPUT. Position gradient; written only when ``with_pos_grad``.
    grad_charges : wp.array, shape (N,), dtype matching ``charges``
        OUTPUT. Charge gradient; written only when ``with_charge_grad``.
    with_pos_grad, with_charge_grad : bool
        Per-slot gradient emission flags (keyword-only).
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` (keyword-only).
    device : str, optional
        Warp device string. Defaults to ``positions.device`` (keyword-only).
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    num_atoms = positions.shape[0]

    if device is None:
        device = str(positions.device)

    _overload = _get_real_space_pair_overload(
        LMAX=0,
        storage="csr",
        is_batch=False,
        with_pos_grad=with_pos_grad,
        with_charge_grad=with_charge_grad,
        with_dipole_grad=False,
        with_quad_grad=False,
        with_cell_grad=False,
        vec_dtype=vec_dtype,
        scalar_dtype=wp_dtype,
    )

    # The all-grads kernel writes to all slots unconditionally; swap any
    # 1-element placeholders for num_atoms-sized scratch so the atomic_add
    # calls stay in bounds. The caller never reads the scratch.
    any_grad = with_pos_grad or with_charge_grad
    if any_grad:
        if not with_pos_grad:
            grad_positions = wp.zeros(num_atoms, dtype=vec_dtype, device=device)
        if not with_charge_grad:
            grad_charges = wp.zeros(num_atoms, dtype=wp_dtype, device=device)

    wp.launch(
        _overload,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            pair_energies,
            grad_positions,
            grad_charges,
        ],
        device=device,
    )


# =============================================================================
# l_max = 0 second-order (double-backward) kernel
# =============================================================================
#
# Differentiates the first-order backward's scalar functional
#
#   L'(grad_pos, grad_q) = Σ_i gg_pos[i] · grad_pos[i] + Σ_i gg_chg[i] · grad_q[i]
#
# w.r.t. ``(grad_energies, positions, charges)`` — enabling a
# differentiable forces tensor via ``torch.autograd.grad(...,
# create_graph=True)``.
#
# For the lmax=0 pair energy ``E_ij = 0.5 q_i q_j T^(0)(r_ij)``, thread k's
# edge (k, b) with ``r_vec = pos_b - pos_k + shift``,
# ``A = erfc/r² + (2α/√π) exp/r``, ``c_quad = A'/r² - A/r³``:
#
#   formula_k(b)  ≡ gg_chg[k]·q_b·T^(0) + q_k·q_b·(A/r)·(gg_pos[k]·r_vec)
#   G_k(b)        ≡ -(A/r)·r_vec·gg_chg[k]·q_b
#                   + q_k·q_b·[c_quad·(gg_pos[k]·r_vec)·r_vec + (A/r)·gg_pos[k]]
#
#   ∂L'/∂ge   (both k- and b-slot get 0.5·formula_k(b))
#   ∂L'/∂pos  (k-slot −, b-slot +, scaled by 0.5·(ge[k]+ge[b])·G_k(b))
#   ∂L'/∂q    (asymmetric — k-slot and b-slot differ)
#
# Under full list, thread b's edge (b, k) makes the symmetric
# contributions; under half list, thread k's writes alone reflect the
# half-list forward.


@wp.kernel(enable_backward=False)
def _multipole_real_space_monopole_csr_energy_2nd_backward_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    sigma: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    grad_energies: wp.array(dtype=wp.float64),
    gg_positions: wp.array(dtype=Any),
    gg_charges: wp.array(dtype=Any),
    gg_grad_energies_2nd: wp.array(dtype=wp.float64),
    gg_positions_2nd: wp.array(dtype=Any),
    gg_charges_2nd: wp.array(dtype=Any),
):
    r"""Second-order backward of :func:`_multipole_real_space_monopole_csr_energy_kernel`.

    Launch Grid
    -----------
    dim = [num_atoms].

    Parameters
    ----------
    positions, charges, cell, idx_j, neighbor_ptr, unit_shifts, sigma, alpha :
        Identical to the forward kernel (the pair geometry + GTO/Ewald
        smearing parameters ``sigma`` and ``alpha``).
    grad_energies : wp.array, shape (N,), dtype wp.float64
        Original first-order upstream cotangent ``∂L/∂pair_energies``.
        Saved from the first-order backward's forward pass.
    gg_positions : wp.array, shape (N,), dtype matching ``positions``
        Upstream cotangent on ``grad_positions`` (the first-order
        backward's position output).
    gg_charges : wp.array, shape (N,), dtype matching ``charges``
        Upstream cotangent on ``grad_charges``.
    gg_grad_energies_2nd : wp.array, shape (N,), dtype wp.float64
        OUTPUT (pre-zeroed). ``∂L'/∂grad_energies``.
    gg_positions_2nd : wp.array, shape (N,), dtype matching ``positions``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂positions``.
    gg_charges_2nd : wp.array, shape (N,), dtype matching ``charges``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂charges``.
    """
    atom_i = wp.tid()

    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    sigma_ = wp.float64(sigma[0])
    alpha_ = wp.float64(alpha[0])
    cell_t = wp.transpose(cell[0])
    ge_i = grad_energies[atom_i]
    gp_i_native = gg_positions[atom_i]
    gp_i = wp.vec3d(
        wp.float64(gp_i_native[0]),
        wp.float64(gp_i_native[1]),
        wp.float64(gp_i_native[2]),
    )
    gc_i = wp.float64(gg_charges[atom_i])

    ab = _gto_ewald_ab(sigma_, alpha_)
    a_coef = ab[0]
    b_coef = ab[1]

    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]
        qj = wp.float64(charges[j])
        pos_j = positions[j]
        ge_j = grad_energies[j]

        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            r_vec = wp.vec3d(
                wp.float64(separation_vector[0]),
                wp.float64(separation_vector[1]),
                wp.float64(separation_vector[2]),
            )

            inv_r = wp.float64(1.0) / distance
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r * inv_r2

            t0 = _gto_ewald_t0(distance, a_coef, b_coef)

            # Combined radial helpers at a and b; subtract.
            ra = _gto_ewald_A_single(distance, a_coef)
            rb = _gto_ewald_A_single(distance, b_coef)
            a_scalar = ra[0] - rb[0]
            a_prime = ra[1] - rb[1]
            a_over_r = a_scalar * inv_r
            c_quad = a_prime * inv_r2 - a_scalar * inv_r3

            gp_i_dot_r = gp_i[0] * r_vec[0] + gp_i[1] * r_vec[1] + gp_i[2] * r_vec[2]

            # --- gg_ge contribution (symmetric in k / b) --------------------
            # formula_k(b) = gc_k · q_b · T^(0) + q_k·q_b·(A/r)·(gp_k · r_vec)
            formula_k_b = gc_i * qj * t0 + qi * qj * a_over_r * gp_i_dot_r
            half_formula = wp.float64(0.5) * formula_k_b
            wp.atomic_add(gg_grad_energies_2nd, atom_i, half_formula)
            wp.atomic_add(gg_grad_energies_2nd, j, half_formula)

            # --- gg_pos contribution (antisymmetric — ± G on k / b slots) ---
            ge_sum = ge_i + ge_j
            # G = -(A/r) · r_vec · gc_k · q_b
            #     + q_k·q_b · [c_quad·(gp_k·r_vec)·r_vec + (A/r)·gp_k]
            gx = -a_over_r * r_vec[0] * gc_i * qj + qi * qj * (
                c_quad * gp_i_dot_r * r_vec[0] + a_over_r * gp_i[0]
            )
            gy = -a_over_r * r_vec[1] * gc_i * qj + qi * qj * (
                c_quad * gp_i_dot_r * r_vec[1] + a_over_r * gp_i[1]
            )
            gz = -a_over_r * r_vec[2] * gc_i * qj + qi * qj * (
                c_quad * gp_i_dot_r * r_vec[2] + a_over_r * gp_i[2]
            )
            scale_pos = wp.float64(0.5) * ge_sum
            px = scale_pos * gx
            py = scale_pos * gy
            pz = scale_pos * gz
            # k-slot gets -G; b-slot gets +G.
            wp.atomic_add(
                gg_positions_2nd,
                atom_i,
                type(pos_i)(
                    type(pos_i[0])(-px),
                    type(pos_i[0])(-py),
                    type(pos_i[0])(-pz),
                ),
            )
            wp.atomic_add(
                gg_positions_2nd,
                j,
                type(pos_i)(
                    type(pos_i[0])(px),
                    type(pos_i[0])(py),
                    type(pos_i[0])(pz),
                ),
            )

            # --- gg_chg contribution (asymmetric) ---------------------------
            # k-slot: 0.5·(ge_k+ge_b)·(gp_k·r_vec)·q_b·(A/r)
            # b-slot: 0.5·(ge_k+ge_b)·[gc_k·T^(0) + (gp_k·r_vec)·q_k·(A/r)]
            gc_k_contrib = scale_pos * gp_i_dot_r * qj * a_over_r
            gc_b_contrib = scale_pos * (gc_i * t0 + gp_i_dot_r * qi * a_over_r)
            wp.atomic_add(gg_charges_2nd, atom_i, type(charges[atom_i])(gc_k_contrib))
            wp.atomic_add(gg_charges_2nd, j, type(charges[atom_i])(gc_b_contrib))


def _monopole_csr_2nd_backward_sig(v, t):
    """Signature builder for the lmax=0 second-order backward kernel."""
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=m),  # cell
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # alpha
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=v),  # gg_positions
        wp.array(dtype=t),  # gg_charges
        wp.array(dtype=wp.float64),  # gg_grad_energies_2nd
        wp.array(dtype=v),  # gg_positions_2nd
        wp.array(dtype=t),  # gg_charges_2nd
    ]


_multipole_real_space_monopole_csr_energy_2nd_backward_overloads = register_overloads(
    _multipole_real_space_monopole_csr_energy_2nd_backward_kernel,
    _monopole_csr_2nd_backward_sig,
)


def multipole_real_space_monopole_csr_energy_2nd_backward(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigma: wp.array,
    alpha: wp.array,
    grad_energies: wp.array,
    gg_positions: wp.array,
    gg_charges: wp.array,
    gg_grad_energies_2nd: wp.array,
    gg_positions_2nd: wp.array,
    gg_charges_2nd: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_multipole_real_space_monopole_csr_energy_2nd_backward_kernel`.

    Framework-agnostic; the caller pre-zeroes the three output arrays.

    Parameters
    ----------
    positions, charges, cell, idx_j, neighbor_ptr, unit_shifts, sigma, alpha :
        Same semantics as :func:`multipole_real_space_monopole_csr_energy`.
    grad_energies : wp.array, shape (N,), dtype wp.float64
        Original first-order upstream cotangent ``∂L/∂pair_energies``.
    gg_positions : wp.array, shape (N,), dtype matching ``positions``
        Upstream cotangent on the first-order ``grad_positions``.
    gg_charges : wp.array, shape (N,), dtype matching ``charges``
        Upstream cotangent on ``grad_charges``.
    gg_grad_energies_2nd : wp.array, shape (N,), dtype wp.float64
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂grad_energies``.
    gg_positions_2nd : wp.array, shape (N,), dtype matching ``positions``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂positions``.
    gg_charges_2nd : wp.array, shape (N,), dtype matching ``charges``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂charges``.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the overloaded variant.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f

    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _multipole_real_space_monopole_csr_energy_2nd_backward_overloads[vec_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            grad_energies,
            gg_positions,
            gg_charges,
            gg_grad_energies_2nd,
            gg_positions_2nd,
            gg_charges_2nd,
        ],
        device=device,
    )


@wp.kernel
def _multipole_real_space_monopole_csr_cell_grad_backward_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    sigma: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    per_direction_scale: wp.array(dtype=wp.float64),
    g_cell: wp.array(dtype=Any),  # (1,) mat33 — cotangent on grad_cell
    grad_positions: wp.array(dtype=Any),  # (N,) OUT (pre-zeroed)
    grad_charges: wp.array(dtype=Any),  # (N,) OUT (pre-zeroed)
    grad_cell_out: wp.array(dtype=Any),  # (1,) mat33 OUT (pre-zeroed, atomic)
):
    r"""Double-backward of the l=0 real-space cell-grad (stress-loss).

    The forward cell-grad is ``grad_cell[a,b] = Σ_pairs scale·n[a]·f[b]`` with
    ``f = ∂E_pair/∂r_ij = -q_i q_j (A/r) r_vec`` and ``n = unit_shifts``. For a
    cotangent ``g_cell`` on ``grad_cell``, ``S = ⟨g_cell, grad_cell⟩ =
    Σ_pairs scale·(w·f)`` with the per-pair direction ``w = g_cellᵀ·n``. The
    grads (same radial algebra as the force-loss 2nd-order kernel, with the
    position-direction replaced by ``w``):

    - ``G = q_i q_j (c_quad·(w·r_vec)·r_vec + (A/r)·w)``,
      ``∂S/∂r_i = +scale·G``, ``∂S/∂r_j = -scale·G``;
    - ``∂S/∂q_i = -scale·q_j·(A/r)·(w·r_vec)``, sym for ``q_j``;
    - ``∂S/∂cell[a,b] = -scale·n[a]·G[b]`` (cell-cell / stress-stress block).
    """
    atom_i = wp.tid()
    sigma_ = wp.float64(sigma[0])
    alpha_ = wp.float64(alpha[0])
    ab = _gto_ewald_ab(sigma_, alpha_)
    a_coef = ab[0]
    b_coef = ab[1]
    scale = per_direction_scale[0]
    cell_t = wp.transpose(cell[0])
    gcell = g_cell[0]
    g00 = wp.float64(gcell[0, 0])
    g01 = wp.float64(gcell[0, 1])
    g02 = wp.float64(gcell[0, 2])
    g10 = wp.float64(gcell[1, 0])
    g11 = wp.float64(gcell[1, 1])
    g12 = wp.float64(gcell[1, 2])
    g20 = wp.float64(gcell[2, 0])
    g21 = wp.float64(gcell[2, 1])
    g22 = wp.float64(gcell[2, 2])

    pos_i = positions[atom_i]
    qi = wp.float64(charges[atom_i])
    k_start = neighbor_ptr[atom_i]
    k_end = neighbor_ptr[atom_i + 1]
    for k in range(k_start, k_end):
        j = idx_j[k]
        shift_vec = unit_shifts[k]
        qj = wp.float64(charges[j])
        pos_j = positions[j]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        sep = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(sep))
        if distance > wp.float64(1e-8):
            r_vec = wp.vec3d(wp.float64(sep[0]), wp.float64(sep[1]), wp.float64(sep[2]))
            inv_r = wp.float64(1.0) / distance
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r * inv_r2
            ra = _gto_ewald_A_single(distance, a_coef)
            rb = _gto_ewald_A_single(distance, b_coef)
            a_scalar = ra[0] - rb[0]
            a_prime = ra[1] - rb[1]
            a_over_r = a_scalar * inv_r
            c_quad = a_prime * inv_r2 - a_scalar * inv_r3

            sh0 = wp.float64(shift_vec[0])
            sh1 = wp.float64(shift_vec[1])
            sh2 = wp.float64(shift_vec[2])
            # w = g_cellᵀ · n  (per-pair "position direction").
            w_x = g00 * sh0 + g10 * sh1 + g20 * sh2
            w_y = g01 * sh0 + g11 * sh1 + g21 * sh2
            w_z = g02 * sh0 + g12 * sh1 + g22 * sh2
            w_dot_r = w_x * r_vec[0] + w_y * r_vec[1] + w_z * r_vec[2]

            qq = qi * qj
            gx = qq * (c_quad * w_dot_r * r_vec[0] + a_over_r * w_x)
            gy = qq * (c_quad * w_dot_r * r_vec[1] + a_over_r * w_y)
            gz = qq * (c_quad * w_dot_r * r_vec[2] + a_over_r * w_z)
            sgx = scale * gx
            sgy = scale * gy
            sgz = scale * gz
            # ∂S/∂r_i = +scale·G ; ∂S/∂r_j = -scale·G.
            wp.atomic_add(
                grad_positions,
                atom_i,
                type(pos_i)(
                    type(pos_i[0])(sgx),
                    type(pos_i[0])(sgy),
                    type(pos_i[0])(sgz),
                ),
            )
            wp.atomic_add(
                grad_positions,
                j,
                type(pos_i)(
                    type(pos_i[0])(-sgx),
                    type(pos_i[0])(-sgy),
                    type(pos_i[0])(-sgz),
                ),
            )
            # ∂S/∂q_i = -scale·q_j·(A/r)·(w·r) ; sym for q_j.
            base_c = -scale * a_over_r * w_dot_r
            wp.atomic_add(grad_charges, atom_i, type(charges[atom_i])(base_c * qj))
            wp.atomic_add(grad_charges, j, type(charges[atom_i])(base_c * qi))
            # ∂S/∂cell[a,b] = -scale·n[a]·G[b]  (cell-cell block).
            cgx = -sgx
            cgy = -sgy
            cgz = -sgz
            wp.atomic_add(
                grad_cell_out,
                0,
                type(gcell)(
                    type(gcell[0, 0])(sh0 * cgx),
                    type(gcell[0, 0])(sh0 * cgy),
                    type(gcell[0, 0])(sh0 * cgz),
                    type(gcell[0, 0])(sh1 * cgx),
                    type(gcell[0, 0])(sh1 * cgy),
                    type(gcell[0, 0])(sh1 * cgz),
                    type(gcell[0, 0])(sh2 * cgx),
                    type(gcell[0, 0])(sh2 * cgy),
                    type(gcell[0, 0])(sh2 * cgz),
                ),
            )


def _monopole_csr_cell_grad_backward_sig(v, t):
    """Signature builder for the l=0 cell-grad double-backward kernel."""
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=m),  # cell
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # alpha
        wp.array(dtype=wp.float64),  # per_direction_scale
        wp.array(dtype=m),  # g_cell
        wp.array(dtype=v),  # grad_positions (out)
        wp.array(dtype=t),  # grad_charges (out)
        wp.array(dtype=m),  # grad_cell_out (out)
    ]


_multipole_real_space_monopole_csr_cell_grad_backward_overloads = register_overloads(
    _multipole_real_space_monopole_csr_cell_grad_backward_kernel,
    _monopole_csr_cell_grad_backward_sig,
)


def multipole_real_space_monopole_csr_cell_grad_backward(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigma: wp.array,
    alpha: wp.array,
    per_direction_scale: wp.array,
    g_cell: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_cell_out: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for the l=0 cell-grad double-backward (stress-loss).

    Caller pre-zeroes ``grad_positions``, ``grad_charges``, ``grad_cell_out``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(positions.device)
    wp.launch(
        _multipole_real_space_monopole_csr_cell_grad_backward_overloads[vec_dtype],
        dim=positions.shape[0],
        inputs=[
            positions,
            charges,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            per_direction_scale,
            g_cell,
        ],
        outputs=[grad_positions, grad_charges, grad_cell_out],
        device=device,
    )


@wp.kernel
def _batch_multipole_real_space_monopole_csr_cell_grad_backward_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cells: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    atom_batch_idx: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    sigma: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    per_direction_scale: wp.array(dtype=wp.float64),
    g_cell: wp.array(dtype=Any),  # (1,) mat33 — cotangent on grad_cell
    grad_positions: wp.array(dtype=Any),  # (N,) OUT (pre-zeroed)
    grad_charges: wp.array(dtype=Any),  # (N,) OUT (pre-zeroed)
    grad_cell_out: wp.array(dtype=Any),  # (1,) mat33 OUT (pre-zeroed, atomic)
):
    r"""Double-backward of the l=0 real-space cell-grad (stress-loss).

    The forward cell-grad is ``grad_cell[a,b] = Σ_pairs scale·n[a]·f[b]`` with
    ``f = ∂E_pair/∂r_ij = -q_i q_j (A/r) r_vec`` and ``n = unit_shifts``. For a
    cotangent ``g_cell`` on ``grad_cell``, ``S = ⟨g_cell, grad_cell⟩ =
    Σ_pairs scale·(w·f)`` with the per-pair direction ``w = g_cellᵀ·n``. The
    grads (same radial algebra as the force-loss 2nd-order kernel, with the
    position-direction replaced by ``w``):

    - ``G = q_i q_j (c_quad·(w·r_vec)·r_vec + (A/r)·w)``,
      ``∂S/∂r_i = +scale·G``, ``∂S/∂r_j = -scale·G``;
    - ``∂S/∂q_i = -scale·q_j·(A/r)·(w·r_vec)``, sym for ``q_j``;
    - ``∂S/∂cell[a,b] = -scale·n[a]·G[b]`` (cell-cell / stress-stress block).
    """
    atom_i = wp.tid()
    b = atom_batch_idx[atom_i]
    sigma_ = wp.float64(sigma[0])
    alpha_ = wp.float64(alpha[0])
    ab = _gto_ewald_ab(sigma_, alpha_)
    a_coef = ab[0]
    b_coef = ab[1]
    scale = per_direction_scale[0]
    cell_t = wp.transpose(cells[b])
    gcell = g_cell[b]
    g00 = wp.float64(gcell[0, 0])
    g01 = wp.float64(gcell[0, 1])
    g02 = wp.float64(gcell[0, 2])
    g10 = wp.float64(gcell[1, 0])
    g11 = wp.float64(gcell[1, 1])
    g12 = wp.float64(gcell[1, 2])
    g20 = wp.float64(gcell[2, 0])
    g21 = wp.float64(gcell[2, 1])
    g22 = wp.float64(gcell[2, 2])

    pos_i = positions[atom_i]
    qi = wp.float64(charges[atom_i])
    k_start = neighbor_ptr[atom_i]
    k_end = neighbor_ptr[atom_i + 1]
    for k in range(k_start, k_end):
        j = idx_j[k]
        shift_vec = unit_shifts[k]
        qj = wp.float64(charges[j])
        pos_j = positions[j]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        sep = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(sep))
        if distance > wp.float64(1e-8):
            r_vec = wp.vec3d(wp.float64(sep[0]), wp.float64(sep[1]), wp.float64(sep[2]))
            inv_r = wp.float64(1.0) / distance
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r * inv_r2
            ra = _gto_ewald_A_single(distance, a_coef)
            rb = _gto_ewald_A_single(distance, b_coef)
            a_scalar = ra[0] - rb[0]
            a_prime = ra[1] - rb[1]
            a_over_r = a_scalar * inv_r
            c_quad = a_prime * inv_r2 - a_scalar * inv_r3

            sh0 = wp.float64(shift_vec[0])
            sh1 = wp.float64(shift_vec[1])
            sh2 = wp.float64(shift_vec[2])
            # w = g_cellᵀ · n  (per-pair "position direction").
            w_x = g00 * sh0 + g10 * sh1 + g20 * sh2
            w_y = g01 * sh0 + g11 * sh1 + g21 * sh2
            w_z = g02 * sh0 + g12 * sh1 + g22 * sh2
            w_dot_r = w_x * r_vec[0] + w_y * r_vec[1] + w_z * r_vec[2]

            qq = qi * qj
            gx = qq * (c_quad * w_dot_r * r_vec[0] + a_over_r * w_x)
            gy = qq * (c_quad * w_dot_r * r_vec[1] + a_over_r * w_y)
            gz = qq * (c_quad * w_dot_r * r_vec[2] + a_over_r * w_z)
            sgx = scale * gx
            sgy = scale * gy
            sgz = scale * gz
            # ∂S/∂r_i = +scale·G ; ∂S/∂r_j = -scale·G.
            wp.atomic_add(
                grad_positions,
                atom_i,
                type(pos_i)(
                    type(pos_i[0])(sgx),
                    type(pos_i[0])(sgy),
                    type(pos_i[0])(sgz),
                ),
            )
            wp.atomic_add(
                grad_positions,
                j,
                type(pos_i)(
                    type(pos_i[0])(-sgx),
                    type(pos_i[0])(-sgy),
                    type(pos_i[0])(-sgz),
                ),
            )
            # ∂S/∂q_i = -scale·q_j·(A/r)·(w·r) ; sym for q_j.
            base_c = -scale * a_over_r * w_dot_r
            wp.atomic_add(grad_charges, atom_i, type(charges[atom_i])(base_c * qj))
            wp.atomic_add(grad_charges, j, type(charges[atom_i])(base_c * qi))
            # ∂S/∂cell[a,b] = -scale·n[a]·G[b]  (cell-cell block).
            cgx = -sgx
            cgy = -sgy
            cgz = -sgz
            wp.atomic_add(
                grad_cell_out,
                0,
                type(gcell)(
                    type(gcell[0, 0])(sh0 * cgx),
                    type(gcell[0, 0])(sh0 * cgy),
                    type(gcell[0, 0])(sh0 * cgz),
                    type(gcell[0, 0])(sh1 * cgx),
                    type(gcell[0, 0])(sh1 * cgy),
                    type(gcell[0, 0])(sh1 * cgz),
                    type(gcell[0, 0])(sh2 * cgx),
                    type(gcell[0, 0])(sh2 * cgy),
                    type(gcell[0, 0])(sh2 * cgz),
                ),
            )


def _batch_monopole_csr_cell_grad_backward_sig(v, t):
    """Signature builder for the l=0 cell-grad double-backward kernel."""
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=m),  # cells
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.int32),  # atom_batch_idx
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # alpha
        wp.array(dtype=wp.float64),  # per_direction_scale
        wp.array(dtype=m),  # g_cell
        wp.array(dtype=v),  # grad_positions (out)
        wp.array(dtype=t),  # grad_charges (out)
        wp.array(dtype=m),  # grad_cell_out (out)
    ]


_batch_multipole_real_space_monopole_csr_cell_grad_backward_overloads = (
    register_overloads(
        _batch_multipole_real_space_monopole_csr_cell_grad_backward_kernel,
        _batch_monopole_csr_cell_grad_backward_sig,
    )
)


def batch_multipole_real_space_monopole_csr_cell_grad_backward(
    positions: wp.array,
    charges: wp.array,
    cells: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    atom_batch_idx: wp.array,
    unit_shifts: wp.array,
    sigma: wp.array,
    alpha: wp.array,
    per_direction_scale: wp.array,
    g_cell: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_cell_out: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for the l=0 cell-grad double-backward (stress-loss).

    Caller pre-zeroes ``grad_positions``, ``grad_charges``, ``grad_cell_out``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(positions.device)
    wp.launch(
        _batch_multipole_real_space_monopole_csr_cell_grad_backward_overloads[
            vec_dtype
        ],
        dim=positions.shape[0],
        inputs=[
            positions,
            charges,
            cells,
            idx_j,
            neighbor_ptr,
            atom_batch_idx,
            unit_shifts,
            sigma,
            alpha,
            per_direction_scale,
            g_cell,
        ],
        outputs=[grad_positions, grad_charges, grad_cell_out],
        device=device,
    )


# =============================================================================
# l_max = 1 analytical backward kernel — ∂/∂(positions, charges, dipoles)
# =============================================================================
#
# Direct differentiation of the factored ``lmax=1`` pair energy:
#
#   PE = q_i q_j T^(0)
#      + (q_i μ_j - q_j μ_i) · T^(1)(r)
#      + c_diag (μ_i · μ_j) + c_quad (μ_i · r)(μ_j · r)
#
# with kernel conventions T^(0) = erfc(αr)/r, T^(1) = -(A/r) r_vec,
# c_diag = A/r, c_quad = A'/r² - A/r³. The factored coefficients:
#
#   A(r)   =  erfc/r² + (2α/√π) exp/r
#   A'(r)  = -2 erfc/r³ - (4α/√π) exp/r² - (4α³/√π) exp
#   A''(r) =  6 erfc/r⁴ + (12α/√π) exp/r³ + (8α³/√π) exp/r + (8α⁵/√π) r exp
#   c3     =  A''/r³ - 3 A'/r⁴ + 3 A/r⁵     (used for ∂c_quad/∂r = c3 · r_α)
#
# Partial derivatives (per edge; moment and position outputs still
# require the pair_energies[i]'s ``0.5`` prefactor):
#
#   ∂PE/∂q_i    = q_j T^(0) + μ_j · T^(1)
#   ∂PE/∂q_j    = q_i T^(0) - μ_i · T^(1)
#   ∂PE/∂μ_i   = -q_j T^(1) + c_diag μ_j + c_quad (μ_j · r) r_vec
#   ∂PE/∂μ_j   =  q_i T^(1) + c_diag μ_i + c_quad (μ_i · r) r_vec
#   ∂PE/∂r     =  r_vec · [-q_i q_j c_diag
#                          - c_quad (q_i (μ_j·r) - q_j (μ_i·r))
#                          + c_quad (μ_i·μ_j)
#                          + c3 (μ_i·r)(μ_j·r)]
#                + [-c_diag (q_i μ_j - q_j μ_i)
#                   + c_quad (μ_j · r) μ_i
#                   + c_quad (μ_i · r) μ_j]
#
# With r = r_j - r_i + shift, ∂PE/∂r_j = +∂PE/∂r and ∂PE/∂r_i = -∂PE/∂r.


@wp.kernel(enable_backward=False)
def _multipole_real_space_dipole_csr_energy_backward_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    dipoles: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    sigma: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    grad_energies: wp.array(dtype=wp.float64),
    grad_positions: wp.array(dtype=Any),
    grad_charges: wp.array(dtype=Any),
    grad_dipoles: wp.array(dtype=Any),
):
    r"""Analytical backward of :func:`_multipole_real_space_dipole_csr_energy_kernel`.

    Launch Grid
    -----------
    dim = [num_atoms] — one thread per atom; inner loop over CSR neighbors.
    Every output slot is written via ``atomic_add`` so the same kernel
    works for both half- and full-list conventions; the per-edge math
    above is applied symmetrically (with the ``0.5`` ``pair_energies``
    prefactor) to the ``i`` and ``j`` sides.

    Parameters
    ----------
    positions, charges, dipoles, cell, idx_j, neighbor_ptr, unit_shifts,
    sigma, alpha :
        Identical to the forward kernel (the pair geometry, source moments,
        and GTO/Ewald smearing parameters ``sigma`` and ``alpha``).
    grad_energies : wp.array, shape (N,), dtype wp.float64
        Upstream cotangent ``∂L/∂pair_energies``.
    grad_positions : wp.array, shape (N,), dtype matching ``positions``
        OUTPUT (pre-zeroed).
    grad_charges : wp.array, shape (N,), dtype matching ``charges``
        OUTPUT (pre-zeroed).
    grad_dipoles : wp.array, shape (N,), dtype matching ``dipoles``
        OUTPUT (pre-zeroed). Cartesian ``(x, y, z)`` layout.
    """
    atom_i = wp.tid()

    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    mu_i_native = dipoles[atom_i]
    mu_i = wp.vec3d(
        wp.float64(mu_i_native[0]),
        wp.float64(mu_i_native[1]),
        wp.float64(mu_i_native[2]),
    )
    sigma_ = wp.float64(sigma[0])
    alpha_ = wp.float64(alpha[0])
    cell_t = wp.transpose(cell[0])
    ge_i = grad_energies[atom_i]

    ab = _gto_ewald_ab(sigma_, alpha_)
    a_coef = ab[0]
    b_coef = ab[1]

    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]
        qj = wp.float64(charges[j])
        pos_j = positions[j]
        mu_j_native = dipoles[j]
        mu_j = wp.vec3d(
            wp.float64(mu_j_native[0]),
            wp.float64(mu_j_native[1]),
            wp.float64(mu_j_native[2]),
        )

        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )

        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            r_vec = wp.vec3d(
                wp.float64(separation_vector[0]),
                wp.float64(separation_vector[1]),
                wp.float64(separation_vector[2]),
            )

            inv_r = wp.float64(1.0) / distance
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r * inv_r2
            inv_r4 = inv_r2 * inv_r2
            inv_r5 = inv_r3 * inv_r2

            # GTO-Ewald radial helpers: A, A', A'' combined from a and b.
            t0 = _gto_ewald_t0(distance, a_coef, b_coef)
            ra = _gto_ewald_A_single(distance, a_coef)
            rb = _gto_ewald_A_single(distance, b_coef)
            a_scalar = ra[0] - rb[0]
            a_prime = ra[1] - rb[1]
            a_double_prime = ra[2] - rb[2]

            # T^(1) (kernel convention).
            neg_a_over_r = -a_scalar * inv_r
            t1x = neg_a_over_r * r_vec[0]
            t1y = neg_a_over_r * r_vec[1]
            t1z = neg_a_over_r * r_vec[2]

            # c_diag, c_quad, c3.
            c_diag = a_scalar * inv_r
            c_quad = a_prime * inv_r2 - a_scalar * inv_r3
            c3 = (
                a_double_prime * inv_r3
                - wp.float64(3.0) * a_prime * inv_r4
                + wp.float64(3.0) * a_scalar * inv_r5
            )

            # Dot products.
            mu_i_dot_r = mu_i[0] * r_vec[0] + mu_i[1] * r_vec[1] + mu_i[2] * r_vec[2]
            mu_j_dot_r = mu_j[0] * r_vec[0] + mu_j[1] * r_vec[1] + mu_j[2] * r_vec[2]
            mu_i_dot_mu_j = mu_i[0] * mu_j[0] + mu_i[1] * mu_j[1] + mu_i[2] * mu_j[2]
            mu_j_dot_T1 = t1x * mu_j[0] + t1y * mu_j[1] + t1z * mu_j[2]
            mu_i_dot_T1 = t1x * mu_i[0] + t1y * mu_i[1] + t1z * mu_i[2]

            # Pair energy contribution to pair_energies[i] is 0.5 · PE;
            # thread i's contribution to grad_anything[*] is therefore
            # grad_energies[i] · 0.5 · ∂PE/∂anything. The "cross"
            # contribution from pair_energies[j] is handled by thread j
            # (under full list) or intentionally omitted (under half
            # list; the forward also returns half-energies there).
            half_ge_i = wp.float64(0.5) * ge_i

            # ---- charge gradients ----
            # ∂E_ij/∂q_i = q_j T^(0) + μ_j · T^(1)
            # ∂E_ij/∂q_j = q_i T^(0) - μ_i · T^(1)
            dPE_dq_i = qj * t0 + mu_j_dot_T1
            dPE_dq_j = qi * t0 - mu_i_dot_T1
            wp.atomic_add(
                grad_charges,
                atom_i,
                type(charges[atom_i])(half_ge_i * dPE_dq_i),
            )
            wp.atomic_add(
                grad_charges,
                j,
                type(charges[atom_i])(half_ge_i * dPE_dq_j),
            )

            # ---- dipole gradients ----
            # ∂E_ij/∂μ_i = -q_j T^(1) + c_diag μ_j + c_quad (μ_j·r) r_vec
            # ∂E_ij/∂μ_j = +q_i T^(1) + c_diag μ_i + c_quad (μ_i·r) r_vec
            cq_muj_r = c_quad * mu_j_dot_r
            cq_mui_r = c_quad * mu_i_dot_r
            dmu_i_x = -qj * t1x + c_diag * mu_j[0] + cq_muj_r * r_vec[0]
            dmu_i_y = -qj * t1y + c_diag * mu_j[1] + cq_muj_r * r_vec[1]
            dmu_i_z = -qj * t1z + c_diag * mu_j[2] + cq_muj_r * r_vec[2]
            dmu_j_x = qi * t1x + c_diag * mu_i[0] + cq_mui_r * r_vec[0]
            dmu_j_y = qi * t1y + c_diag * mu_i[1] + cq_mui_r * r_vec[1]
            dmu_j_z = qi * t1z + c_diag * mu_i[2] + cq_mui_r * r_vec[2]

            mu_i_contrib = type(mu_i_native)(
                type(mu_i_native[0])(half_ge_i * dmu_i_x),
                type(mu_i_native[0])(half_ge_i * dmu_i_y),
                type(mu_i_native[0])(half_ge_i * dmu_i_z),
            )
            mu_j_contrib = type(mu_i_native)(
                type(mu_i_native[0])(half_ge_i * dmu_j_x),
                type(mu_i_native[0])(half_ge_i * dmu_j_y),
                type(mu_i_native[0])(half_ge_i * dmu_j_z),
            )
            wp.atomic_add(grad_dipoles, atom_i, mu_i_contrib)
            wp.atomic_add(grad_dipoles, j, mu_j_contrib)

            # ---- position gradients ----
            # ∂PE/∂r has a radial part (along r_vec) + a direct part (in μ, etc.).
            rad_coeff = (
                -qi * qj * c_diag
                - c_quad * (qi * mu_j_dot_r - qj * mu_i_dot_r)
                + c_quad * mu_i_dot_mu_j
                + c3 * mu_i_dot_r * mu_j_dot_r
            )
            dir_x = (
                -c_diag * (qi * mu_j[0] - qj * mu_i[0])
                + c_quad * mu_j_dot_r * mu_i[0]
                + c_quad * mu_i_dot_r * mu_j[0]
            )
            dir_y = (
                -c_diag * (qi * mu_j[1] - qj * mu_i[1])
                + c_quad * mu_j_dot_r * mu_i[1]
                + c_quad * mu_i_dot_r * mu_j[1]
            )
            dir_z = (
                -c_diag * (qi * mu_j[2] - qj * mu_i[2])
                + c_quad * mu_j_dot_r * mu_i[2]
                + c_quad * mu_i_dot_r * mu_j[2]
            )

            # ∂PE/∂r_j = rad · r_vec + dir_vec ; ∂PE/∂r_i = -(rad · r_vec + dir_vec)
            dPE_dr_x = rad_coeff * r_vec[0] + dir_x
            dPE_dr_y = rad_coeff * r_vec[1] + dir_y
            dPE_dr_z = rad_coeff * r_vec[2] + dir_z

            px = half_ge_i * dPE_dr_x
            py = half_ge_i * dPE_dr_y
            pz = half_ge_i * dPE_dr_z

            # r_j side gets +(px, py, pz); r_i side gets -(px, py, pz).
            wp.atomic_add(
                grad_positions,
                j,
                type(pos_i)(
                    type(pos_i[0])(px),
                    type(pos_i[0])(py),
                    type(pos_i[0])(pz),
                ),
            )
            wp.atomic_add(
                grad_positions,
                atom_i,
                type(pos_i)(
                    type(pos_i[0])(-px),
                    type(pos_i[0])(-py),
                    type(pos_i[0])(-pz),
                ),
            )


def _dipole_csr_energy_backward_sig(v, t):
    """Signature builder for :func:`_multipole_real_space_dipole_csr_energy_backward_kernel`."""
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=m),  # cell
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # alpha
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=v),  # grad_positions
        wp.array(dtype=t),  # grad_charges
        wp.array(dtype=v),  # grad_dipoles
    ]


_multipole_real_space_dipole_csr_energy_backward_overloads = register_overloads(
    _multipole_real_space_dipole_csr_energy_backward_kernel,
    _dipole_csr_energy_backward_sig,
)


def multipole_real_space_dipole_csr_energy_backward(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigma: wp.array,
    alpha: wp.array,
    grad_energies: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_dipoles: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_multipole_real_space_dipole_csr_energy_backward_kernel`.

    Framework-agnostic. Produces ``(grad_positions, grad_charges,
    grad_dipoles)`` from ``grad_energies``; caller pre-zeros all
    three output arrays.

    Parameters
    ----------
    positions, charges, dipoles, cell, idx_j, neighbor_ptr, unit_shifts,
    sigma, alpha :
        Same semantics as :func:`multipole_real_space_dipole_csr_energy`.
    grad_energies, grad_positions, grad_charges, grad_dipoles :
        Same semantics as for the l_max=0 backward, with the additional
        ``grad_dipoles`` output for the per-atom dipole-moment gradient
        in Cartesian layout.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the overloaded variant.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f

    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _multipole_real_space_dipole_csr_energy_backward_overloads[vec_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            grad_energies,
            grad_positions,
            grad_charges,
            grad_dipoles,
        ],
        device=device,
    )


# =============================================================================
# l_max = 1 second-order (double-backward) kernel
# =============================================================================
#
# Differentiates the first-order backward's scalar functional
#
#   L'(grad_pos, grad_chg, grad_dip) = Σ_i [ gg_pos[i] · grad_pos[i]
#                                          + gg_chg[i] · grad_chg[i]
#                                          + gg_dip[i] · grad_dip[i] ]
#
# w.r.t. ``(grad_energies, positions, charges, dipoles)``. Same "write
# half to both k/b slots via atomic_add" pattern as the l_max=0
# second-order kernel; under full neighbor list, thread k's
# contribution + thread b's contribution (from edge (b, k)) sum to the
# correct ∂L'/∂(...) for each pair.
#
# Per thread-k-edge quantities computed from Ω_i (the i-indexed
# contribution for thread k's frame, i=k, j=b):
#
#   Ω_i = gc_i·(q_j t0 + μ_j·T^(1))
#       + gd_i·(-q_j T^(1) + c_diag μ_j + c_quad (μ_j·r) r_vec)
#       - gp_i·(rad·r_vec + dir_vec)
#
# Writes (each via atomic_add; half-ge = 0.5·(ge_i+ge_j)):
#   gg_ge_2nd[k] += 0.5·Ω_i ;        gg_ge_2nd[b] += 0.5·Ω_i
#   gg_pos_2nd[k] += -half_ge·G_pos ; gg_pos_2nd[b] += +half_ge·G_pos
#   gg_chg_2nd[k] += half_ge·∂Ω_i/∂q_i
#   gg_chg_2nd[b] += half_ge·∂Ω_i/∂q_j
#   gg_dip_2nd[k] += half_ge·∂Ω_i/∂μ_i
#   gg_dip_2nd[b] += half_ge·∂Ω_i/∂μ_j
#
# Requires A''' (new relative to first-order backward):
#
#   A'''(r) = -24 erfc/r⁵ - 48α/(√π r⁴) exp - 32α³/(√π r²) exp
#             - 8α⁵/√π exp - 16α⁷/√π r² exp
#
# and c4 = A'''/r⁴ - 6A''/r⁵ + 15A'/r⁶ - 15A/r⁷ (appears in the
# position-derivative radial coefficient via the (μ_i·r)(μ_j·r) piece
# of rad).


@wp.kernel(enable_backward=False)
def _multipole_real_space_dipole_csr_energy_2nd_backward_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    dipoles: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    sigma: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    grad_energies: wp.array(dtype=wp.float64),
    gg_positions: wp.array(dtype=Any),
    gg_charges: wp.array(dtype=Any),
    gg_dipoles: wp.array(dtype=Any),
    gg_grad_energies_2nd: wp.array(dtype=wp.float64),
    gg_positions_2nd: wp.array(dtype=Any),
    gg_charges_2nd: wp.array(dtype=Any),
    gg_dipoles_2nd: wp.array(dtype=Any),
):
    r"""Second-order backward of the GTO-Ewald l_max=1 real-space path.

    Differentiates the first-order backward's scalar functional w.r.t.
    ``(grad_energies, positions, charges, dipoles)``, enabling a
    differentiable forces tensor via ``torch.autograd.grad(...,
    create_graph=True)``.

    Launch Grid
    -----------
    dim = [num_atoms] — one thread per atom; inner loop over the CSR
    neighbor slice ``idx_j[neighbor_ptr[i]:neighbor_ptr[i+1]]``. Every
    output slot is written via ``atomic_add`` so the kernel works for both
    half- and full-list conventions.

    Parameters
    ----------
    positions, charges, dipoles, cell, idx_j, neighbor_ptr, unit_shifts,
    sigma, alpha :
        Identical to the forward kernel (the pair geometry, source moments,
        and GTO/Ewald smearing parameters ``sigma`` and ``alpha``).
    grad_energies : wp.array, shape (N,), dtype wp.float64
        Original first-order upstream cotangent ``∂L/∂pair_energies``,
        saved from the first-order backward's forward pass.
    gg_positions : wp.array, shape (N,), dtype matching ``positions``
        Upstream cotangent on the first-order backward's ``grad_positions``
        output.
    gg_charges : wp.array, shape (N,), dtype matching ``charges``
        Upstream cotangent on ``grad_charges``.
    gg_dipoles : wp.array, shape (N,), dtype matching ``dipoles``
        Upstream cotangent on ``grad_dipoles``.
    gg_grad_energies_2nd : wp.array, shape (N,), dtype wp.float64
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂grad_energies``.
    gg_positions_2nd : wp.array, shape (N,), dtype matching ``positions``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂positions``.
    gg_charges_2nd : wp.array, shape (N,), dtype matching ``charges``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂charges``.
    gg_dipoles_2nd : wp.array, shape (N,), dtype matching ``dipoles``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂dipoles``.
    """
    atom_i = wp.tid()

    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    mu_i_native = dipoles[atom_i]
    mu_i = wp.vec3d(
        wp.float64(mu_i_native[0]),
        wp.float64(mu_i_native[1]),
        wp.float64(mu_i_native[2]),
    )
    gp_i_native = gg_positions[atom_i]
    gp_i = wp.vec3d(
        wp.float64(gp_i_native[0]),
        wp.float64(gp_i_native[1]),
        wp.float64(gp_i_native[2]),
    )
    gd_i_native = gg_dipoles[atom_i]
    gd_i = wp.vec3d(
        wp.float64(gd_i_native[0]),
        wp.float64(gd_i_native[1]),
        wp.float64(gd_i_native[2]),
    )
    gc_i = wp.float64(gg_charges[atom_i])
    sigma_ = wp.float64(sigma[0])
    alpha_ = wp.float64(alpha[0])
    cell_t = wp.transpose(cell[0])
    ge_i = grad_energies[atom_i]

    ab = _gto_ewald_ab(sigma_, alpha_)
    a_coef = ab[0]
    b_coef = ab[1]

    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]
        qj = wp.float64(charges[j])
        pos_j = positions[j]
        mu_j_native = dipoles[j]
        mu_j = wp.vec3d(
            wp.float64(mu_j_native[0]),
            wp.float64(mu_j_native[1]),
            wp.float64(mu_j_native[2]),
        )
        ge_j = grad_energies[j]

        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            r_vec = wp.vec3d(
                wp.float64(separation_vector[0]),
                wp.float64(separation_vector[1]),
                wp.float64(separation_vector[2]),
            )

            inv_r = wp.float64(1.0) / distance
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r * inv_r2
            inv_r4 = inv_r2 * inv_r2
            inv_r5 = inv_r2 * inv_r3
            inv_r6 = inv_r3 * inv_r3
            inv_r7 = inv_r3 * inv_r4

            # GTO-Ewald radial helpers: combined from single-x at a and b.
            t0 = _gto_ewald_t0(distance, a_coef, b_coef)
            ra = _gto_ewald_A_single(distance, a_coef)
            rb = _gto_ewald_A_single(distance, b_coef)
            a_scalar = ra[0] - rb[0]
            a_prime = ra[1] - rb[1]
            a_double_prime = ra[2] - rb[2]
            a_triple_prime = ra[3] - rb[3]

            c_diag = a_scalar * inv_r
            c_quad = a_prime * inv_r2 - a_scalar * inv_r3
            c3 = (
                a_double_prime * inv_r3
                - wp.float64(3.0) * a_prime * inv_r4
                + wp.float64(3.0) * a_scalar * inv_r5
            )
            c4 = (
                a_triple_prime * inv_r4
                - wp.float64(6.0) * a_double_prime * inv_r5
                + wp.float64(15.0) * a_prime * inv_r6
                - wp.float64(15.0) * a_scalar * inv_r7
            )

            # --- Dot products ---
            mu_i_dot_r = mu_i[0] * r_vec[0] + mu_i[1] * r_vec[1] + mu_i[2] * r_vec[2]
            mu_j_dot_r = mu_j[0] * r_vec[0] + mu_j[1] * r_vec[1] + mu_j[2] * r_vec[2]
            mu_dot = mu_i[0] * mu_j[0] + mu_i[1] * mu_j[1] + mu_i[2] * mu_j[2]
            gp_i_dot_r = gp_i[0] * r_vec[0] + gp_i[1] * r_vec[1] + gp_i[2] * r_vec[2]
            gd_i_dot_r = gd_i[0] * r_vec[0] + gd_i[1] * r_vec[1] + gd_i[2] * r_vec[2]
            gp_i_dot_mu_i = gp_i[0] * mu_i[0] + gp_i[1] * mu_i[1] + gp_i[2] * mu_i[2]
            gp_i_dot_mu_j = gp_i[0] * mu_j[0] + gp_i[1] * mu_j[1] + gp_i[2] * mu_j[2]
            gd_i_dot_mu_j = gd_i[0] * mu_j[0] + gd_i[1] * mu_j[1] + gd_i[2] * mu_j[2]
            dqmu_dot_r = qi * mu_j_dot_r - qj * mu_i_dot_r
            gp_i_dot_dqmu = qi * gp_i_dot_mu_j - qj * gp_i_dot_mu_i

            # --- rad scalar (from first-order kernel's pair-energy position grad) ---
            rad = (
                -qi * qj * c_diag
                - c_quad * dqmu_dot_r
                + c_quad * mu_dot
                + c3 * mu_i_dot_r * mu_j_dot_r
            )

            # --- Ω_i — written to both k and b slots of gg_ge_2nd ---
            omega_i = (
                gc_i * qj * t0
                - gc_i * c_diag * mu_j_dot_r
                + qj * c_diag * gd_i_dot_r
                + c_diag * gd_i_dot_mu_j
                + c_quad * mu_j_dot_r * gd_i_dot_r
                - rad * gp_i_dot_r
                + c_diag * gp_i_dot_dqmu
                - c_quad * mu_j_dot_r * gp_i_dot_mu_i
                - c_quad * mu_i_dot_r * gp_i_dot_mu_j
            )
            half_omega = wp.float64(0.5) * omega_i
            wp.atomic_add(gg_grad_energies_2nd, atom_i, half_omega)
            wp.atomic_add(gg_grad_energies_2nd, j, half_omega)

            half_ge = wp.float64(0.5) * (ge_i + ge_j)

            # --- ∂Ω_i/∂q_i  →  gg_chg_2nd[k] ---
            dOmega_dqi = (
                qj * c_diag + c_quad * mu_j_dot_r
            ) * gp_i_dot_r + c_diag * gp_i_dot_mu_j
            wp.atomic_add(
                gg_charges_2nd,
                atom_i,
                type(charges[atom_i])(half_ge * dOmega_dqi),
            )

            # --- ∂Ω_i/∂q_j  →  gg_chg_2nd[b] ---
            dOmega_dqj = (
                (qi * c_diag - c_quad * mu_i_dot_r) * gp_i_dot_r
                - c_diag * gp_i_dot_mu_i
                + gc_i * t0
                + c_diag * gd_i_dot_r
            )
            wp.atomic_add(
                gg_charges_2nd,
                j,
                type(charges[atom_i])(half_ge * dOmega_dqj),
            )

            # --- ∂Ω_i/∂μ_i (3-vec) →  gg_dip_2nd[k] ---
            coeff_r_dmui = (
                -c_quad * qj * gp_i_dot_r
                - c3 * mu_j_dot_r * gp_i_dot_r
                - c_quad * gp_i_dot_mu_j
            )
            coeff_muj_dmui = -c_quad * gp_i_dot_r
            coeff_gpi_dmui = -c_diag * qj - c_quad * mu_j_dot_r
            dmui_x = (
                coeff_r_dmui * r_vec[0]
                + coeff_muj_dmui * mu_j[0]
                + coeff_gpi_dmui * gp_i[0]
            )
            dmui_y = (
                coeff_r_dmui * r_vec[1]
                + coeff_muj_dmui * mu_j[1]
                + coeff_gpi_dmui * gp_i[1]
            )
            dmui_z = (
                coeff_r_dmui * r_vec[2]
                + coeff_muj_dmui * mu_j[2]
                + coeff_gpi_dmui * gp_i[2]
            )
            wp.atomic_add(
                gg_dipoles_2nd,
                atom_i,
                type(mu_i_native)(
                    type(mu_i_native[0])(half_ge * dmui_x),
                    type(mu_i_native[0])(half_ge * dmui_y),
                    type(mu_i_native[0])(half_ge * dmui_z),
                ),
            )

            # --- ∂Ω_i/∂μ_j (3-vec) →  gg_dip_2nd[b] ---
            coeff_r_dmuj = (
                c_quad * qi * gp_i_dot_r
                - c3 * mu_i_dot_r * gp_i_dot_r
                - c_quad * gp_i_dot_mu_i
                - gc_i * c_diag
                + c_quad * gd_i_dot_r
            )
            coeff_mui_dmuj = -c_quad * gp_i_dot_r
            coeff_gpi_dmuj = c_diag * qi - c_quad * mu_i_dot_r
            coeff_gdi_dmuj = c_diag
            dmuj_x = (
                coeff_r_dmuj * r_vec[0]
                + coeff_mui_dmuj * mu_i[0]
                + coeff_gpi_dmuj * gp_i[0]
                + coeff_gdi_dmuj * gd_i[0]
            )
            dmuj_y = (
                coeff_r_dmuj * r_vec[1]
                + coeff_mui_dmuj * mu_i[1]
                + coeff_gpi_dmuj * gp_i[1]
                + coeff_gdi_dmuj * gd_i[1]
            )
            dmuj_z = (
                coeff_r_dmuj * r_vec[2]
                + coeff_mui_dmuj * mu_i[2]
                + coeff_gpi_dmuj * gp_i[2]
                + coeff_gdi_dmuj * gd_i[2]
            )
            wp.atomic_add(
                gg_dipoles_2nd,
                j,
                type(mu_i_native)(
                    type(mu_i_native[0])(half_ge * dmuj_x),
                    type(mu_i_native[0])(half_ge * dmuj_y),
                    type(mu_i_native[0])(half_ge * dmuj_z),
                ),
            )

            # --- ∂Ω_i/∂r_vec (3-vec) →  ± half_ge into gg_pos_2nd ---
            # (r_vec depends on both pos_i and pos_j; k-slot gets -G, b-slot +G.)
            S_rad = (
                -gc_i * qj * c_diag
                - gc_i * c_quad * mu_j_dot_r
                + qj * c_quad * gd_i_dot_r
                + c_quad * gd_i_dot_mu_j
                + c3 * gd_i_dot_r * mu_j_dot_r
                + gp_i_dot_r
                * (
                    qi * qj * c_quad
                    + c3 * dqmu_dot_r
                    - c3 * mu_dot
                    - c4 * mu_i_dot_r * mu_j_dot_r
                )
                + c_quad * gp_i_dot_dqmu
                - c3 * gp_i_dot_mu_i * mu_j_dot_r
                - c3 * gp_i_dot_mu_j * mu_i_dot_r
            )
            coeff_muj_dr = (
                -gc_i * c_diag
                + c_quad * gd_i_dot_r
                - gp_i_dot_r * c3 * mu_i_dot_r
                - c_quad * gp_i_dot_mu_i
            )
            coeff_mui_dr = -gp_i_dot_r * c3 * mu_j_dot_r - c_quad * gp_i_dot_mu_j
            coeff_gdi_dr = qj * c_diag + c_quad * mu_j_dot_r
            coeff_gpi_dr = -rad
            coeff_dqmu_dr = gp_i_dot_r * c_quad
            dqmu_x = qi * mu_j[0] - qj * mu_i[0]
            dqmu_y = qi * mu_j[1] - qj * mu_i[1]
            dqmu_z = qi * mu_j[2] - qj * mu_i[2]
            G_pos_x = (
                S_rad * r_vec[0]
                + coeff_muj_dr * mu_j[0]
                + coeff_mui_dr * mu_i[0]
                + coeff_gdi_dr * gd_i[0]
                + coeff_gpi_dr * gp_i[0]
                + coeff_dqmu_dr * dqmu_x
            )
            G_pos_y = (
                S_rad * r_vec[1]
                + coeff_muj_dr * mu_j[1]
                + coeff_mui_dr * mu_i[1]
                + coeff_gdi_dr * gd_i[1]
                + coeff_gpi_dr * gp_i[1]
                + coeff_dqmu_dr * dqmu_y
            )
            G_pos_z = (
                S_rad * r_vec[2]
                + coeff_muj_dr * mu_j[2]
                + coeff_mui_dr * mu_i[2]
                + coeff_gdi_dr * gd_i[2]
                + coeff_gpi_dr * gp_i[2]
                + coeff_dqmu_dr * dqmu_z
            )
            px = half_ge * G_pos_x
            py = half_ge * G_pos_y
            pz = half_ge * G_pos_z
            wp.atomic_add(
                gg_positions_2nd,
                atom_i,
                type(pos_i)(
                    type(pos_i[0])(-px),
                    type(pos_i[0])(-py),
                    type(pos_i[0])(-pz),
                ),
            )
            wp.atomic_add(
                gg_positions_2nd,
                j,
                type(pos_i)(
                    type(pos_i[0])(px),
                    type(pos_i[0])(py),
                    type(pos_i[0])(pz),
                ),
            )


def _dipole_csr_2nd_backward_sig(v, t):
    """Signature builder for the lmax=1 second-order backward kernel."""
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=m),  # cell
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # alpha
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=v),  # gg_positions
        wp.array(dtype=t),  # gg_charges
        wp.array(dtype=v),  # gg_dipoles
        wp.array(dtype=wp.float64),  # gg_grad_energies_2nd
        wp.array(dtype=v),  # gg_positions_2nd
        wp.array(dtype=t),  # gg_charges_2nd
        wp.array(dtype=v),  # gg_dipoles_2nd
    ]


_multipole_real_space_dipole_csr_energy_2nd_backward_overloads = register_overloads(
    _multipole_real_space_dipole_csr_energy_2nd_backward_kernel,
    _dipole_csr_2nd_backward_sig,
)


def multipole_real_space_dipole_csr_energy_2nd_backward(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigma: wp.array,
    alpha: wp.array,
    grad_energies: wp.array,
    gg_positions: wp.array,
    gg_charges: wp.array,
    gg_dipoles: wp.array,
    gg_grad_energies_2nd: wp.array,
    gg_positions_2nd: wp.array,
    gg_charges_2nd: wp.array,
    gg_dipoles_2nd: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_multipole_real_space_dipole_csr_energy_2nd_backward_kernel`.

    Produces the four second-order gradients
    ``(gg_grad_energies_2nd, gg_positions_2nd, gg_charges_2nd, gg_dipoles_2nd)``
    from the upstream cotangents on the first-order backward's outputs
    ``(gg_positions, gg_charges, gg_dipoles)`` plus the original
    ``grad_energies``. Caller pre-zeros all four outputs.

    Parameters
    ----------
    positions, charges, dipoles, cell, idx_j, neighbor_ptr, unit_shifts,
    sigma, alpha :
        Same semantics as :func:`multipole_real_space_dipole_csr_energy`.
    grad_energies : wp.array, shape (N,), dtype wp.float64
        Original first-order upstream cotangent ``∂L/∂pair_energies``.
    gg_positions : wp.array, shape (N,), dtype matching ``positions``
        Upstream cotangent on the first-order ``grad_positions``.
    gg_charges : wp.array, shape (N,), dtype matching ``charges``
        Upstream cotangent on ``grad_charges``.
    gg_dipoles : wp.array, shape (N,), dtype matching ``dipoles``
        Upstream cotangent on ``grad_dipoles``.
    gg_grad_energies_2nd : wp.array, shape (N,), dtype wp.float64
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂grad_energies``.
    gg_positions_2nd : wp.array, shape (N,), dtype matching ``positions``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂positions``.
    gg_charges_2nd : wp.array, shape (N,), dtype matching ``charges``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂charges``.
    gg_dipoles_2nd : wp.array, shape (N,), dtype matching ``dipoles``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂dipoles``.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the overloaded variant.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f

    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _multipole_real_space_dipole_csr_energy_2nd_backward_overloads[vec_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            grad_energies,
            gg_positions,
            gg_charges,
            gg_dipoles,
            gg_grad_energies_2nd,
            gg_positions_2nd,
            gg_charges_2nd,
            gg_dipoles_2nd,
        ],
        device=device,
    )


@wp.kernel
def _multipole_real_space_dipole_csr_cell_grad_backward_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    dipoles: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    sigma: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    per_direction_scale: wp.array(dtype=wp.float64),
    g_cell: wp.array(dtype=Any),  # (1,) mat33 cotangent on grad_cell
    grad_positions: wp.array(dtype=Any),  # (N,) OUT
    grad_charges: wp.array(dtype=Any),  # (N,) OUT
    grad_dipoles: wp.array(dtype=Any),  # (N,) OUT
    grad_cell_out: wp.array(dtype=Any),  # (1,) mat33 OUT (atomic)
):
    r"""Double-backward of the l=1 real-space cell-grad (stress-loss).

    Reuses the force-loss dipole Hessian (``∂Ω/∂θ``) with the incoming
    charge/dipole directions set to zero and the position direction replaced by
    the per-pair ``w = g_cellᵀ·n`` (n = unit_shifts), so ``Ω_gp = -w·f`` with
    ``f = ∂E_pair/∂r``. With ``S = ⟨g_cell, grad_cell⟩ = Σ scale·(w·f) =
    -scale·Σ Ω_gp``: ``grad_r_i = +scale·G_pos``, ``grad_r_j = -scale·G_pos``,
    ``grad_θ = -scale·∂Ω/∂θ`` (θ = q_i,q_j,μ_i,μ_j),
    ``grad_cell[a,b] = -scale·n[a]·G_pos[b]``. Verified to reduce to the
    monopole kernel when dipoles vanish.
    """
    atom_i = wp.tid()
    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    mu_i_native = dipoles[atom_i]
    mu_i = wp.vec3d(
        wp.float64(mu_i_native[0]),
        wp.float64(mu_i_native[1]),
        wp.float64(mu_i_native[2]),
    )
    sigma_ = wp.float64(sigma[0])
    alpha_ = wp.float64(alpha[0])
    scale = per_direction_scale[0]
    cell_t = wp.transpose(cell[0])
    gcell = g_cell[0]
    g00 = wp.float64(gcell[0, 0])
    g01 = wp.float64(gcell[0, 1])
    g02 = wp.float64(gcell[0, 2])
    g10 = wp.float64(gcell[1, 0])
    g11 = wp.float64(gcell[1, 1])
    g12 = wp.float64(gcell[1, 2])
    g20 = wp.float64(gcell[2, 0])
    g21 = wp.float64(gcell[2, 1])
    g22 = wp.float64(gcell[2, 2])

    ab = _gto_ewald_ab(sigma_, alpha_)
    a_coef = ab[0]
    b_coef = ab[1]

    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]
    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]
        qj = wp.float64(charges[j])
        pos_j = positions[j]
        mu_j_native = dipoles[j]
        mu_j = wp.vec3d(
            wp.float64(mu_j_native[0]),
            wp.float64(mu_j_native[1]),
            wp.float64(mu_j_native[2]),
        )
        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))
        if distance > wp.float64(1e-8):
            r_vec = wp.vec3d(
                wp.float64(separation_vector[0]),
                wp.float64(separation_vector[1]),
                wp.float64(separation_vector[2]),
            )
            inv_r = wp.float64(1.0) / distance
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r * inv_r2
            inv_r4 = inv_r2 * inv_r2
            inv_r5 = inv_r2 * inv_r3
            inv_r6 = inv_r3 * inv_r3
            inv_r7 = inv_r3 * inv_r4
            ra = _gto_ewald_A_single(distance, a_coef)
            rb = _gto_ewald_A_single(distance, b_coef)
            a_scalar = ra[0] - rb[0]
            a_prime = ra[1] - rb[1]
            a_double_prime = ra[2] - rb[2]
            a_triple_prime = ra[3] - rb[3]
            c_diag = a_scalar * inv_r
            c_quad = a_prime * inv_r2 - a_scalar * inv_r3
            c3 = (
                a_double_prime * inv_r3
                - wp.float64(3.0) * a_prime * inv_r4
                + wp.float64(3.0) * a_scalar * inv_r5
            )
            c4 = (
                a_triple_prime * inv_r4
                - wp.float64(6.0) * a_double_prime * inv_r5
                + wp.float64(15.0) * a_prime * inv_r6
                - wp.float64(15.0) * a_scalar * inv_r7
            )

            # Per-pair direction w = g_cellᵀ·n (replaces the gp_i of force-loss).
            sh0 = wp.float64(shift_vec[0])
            sh1 = wp.float64(shift_vec[1])
            sh2 = wp.float64(shift_vec[2])
            w = wp.vec3d(
                g00 * sh0 + g10 * sh1 + g20 * sh2,
                g01 * sh0 + g11 * sh1 + g21 * sh2,
                g02 * sh0 + g12 * sh1 + g22 * sh2,
            )

            mu_i_dot_r = mu_i[0] * r_vec[0] + mu_i[1] * r_vec[1] + mu_i[2] * r_vec[2]
            mu_j_dot_r = mu_j[0] * r_vec[0] + mu_j[1] * r_vec[1] + mu_j[2] * r_vec[2]
            mu_dot = mu_i[0] * mu_j[0] + mu_i[1] * mu_j[1] + mu_i[2] * mu_j[2]
            w_dot_r = w[0] * r_vec[0] + w[1] * r_vec[1] + w[2] * r_vec[2]
            w_dot_mu_i = w[0] * mu_i[0] + w[1] * mu_i[1] + w[2] * mu_i[2]
            w_dot_mu_j = w[0] * mu_j[0] + w[1] * mu_j[1] + w[2] * mu_j[2]
            dqmu_dot_r = qi * mu_j_dot_r - qj * mu_i_dot_r
            w_dot_dqmu = qi * w_dot_mu_j - qj * w_dot_mu_i

            rad = (
                -qi * qj * c_diag
                - c_quad * dqmu_dot_r
                + c_quad * mu_dot
                + c3 * mu_i_dot_r * mu_j_dot_r
            )

            # ∂Ω/∂q_i and ∂Ω/∂q_j  (gc_i = gd_i = 0).
            d_omega_dqi = (
                qj * c_diag + c_quad * mu_j_dot_r
            ) * w_dot_r + c_diag * w_dot_mu_j
            d_omega_dqj = (
                qi * c_diag - c_quad * mu_i_dot_r
            ) * w_dot_r - c_diag * w_dot_mu_i

            # ∂Ω/∂μ_i.
            coeff_r_dmui = (
                -c_quad * qj * w_dot_r - c3 * mu_j_dot_r * w_dot_r - c_quad * w_dot_mu_j
            )
            coeff_muj_dmui = -c_quad * w_dot_r
            coeff_w_dmui = -c_diag * qj - c_quad * mu_j_dot_r
            # ∂Ω/∂μ_j.
            coeff_r_dmuj = (
                c_quad * qi * w_dot_r - c3 * mu_i_dot_r * w_dot_r - c_quad * w_dot_mu_i
            )
            coeff_mui_dmuj = -c_quad * w_dot_r
            coeff_w_dmuj = c_diag * qi - c_quad * mu_i_dot_r

            # ∂Ω/∂r_vec = G_pos.
            s_rad = (
                w_dot_r
                * (
                    qi * qj * c_quad
                    + c3 * dqmu_dot_r
                    - c3 * mu_dot
                    - c4 * mu_i_dot_r * mu_j_dot_r
                )
                + c_quad * w_dot_dqmu
                - c3 * w_dot_mu_i * mu_j_dot_r
                - c3 * w_dot_mu_j * mu_i_dot_r
            )
            coeff_muj_dr = -w_dot_r * c3 * mu_i_dot_r - c_quad * w_dot_mu_i
            coeff_mui_dr = -w_dot_r * c3 * mu_j_dot_r - c_quad * w_dot_mu_j
            coeff_w_dr = -rad
            coeff_dqmu_dr = w_dot_r * c_quad

            # Scatter (signs verified against the monopole reduction):
            # grad_q = -scale·∂Ω/∂q ; grad_μ = -scale·∂Ω/∂μ ;
            # grad_r_i = +scale·G_pos ; grad_r_j = -scale·G_pos ;
            # grad_cell[a,b] = -scale·n[a]·G_pos[b].
            wp.atomic_add(
                grad_charges, atom_i, type(charges[atom_i])(-scale * d_omega_dqi)
            )
            wp.atomic_add(grad_charges, j, type(charges[atom_i])(-scale * d_omega_dqj))
            wp.atomic_add(
                grad_dipoles,
                atom_i,
                type(mu_i_native)(
                    type(mu_i_native[0])(
                        -scale
                        * (
                            coeff_r_dmui * r_vec[0]
                            + coeff_muj_dmui * mu_j[0]
                            + coeff_w_dmui * w[0]
                        )
                    ),
                    type(mu_i_native[0])(
                        -scale
                        * (
                            coeff_r_dmui * r_vec[1]
                            + coeff_muj_dmui * mu_j[1]
                            + coeff_w_dmui * w[1]
                        )
                    ),
                    type(mu_i_native[0])(
                        -scale
                        * (
                            coeff_r_dmui * r_vec[2]
                            + coeff_muj_dmui * mu_j[2]
                            + coeff_w_dmui * w[2]
                        )
                    ),
                ),
            )
            wp.atomic_add(
                grad_dipoles,
                j,
                type(mu_i_native)(
                    type(mu_i_native[0])(
                        -scale
                        * (
                            coeff_r_dmuj * r_vec[0]
                            + coeff_mui_dmuj * mu_i[0]
                            + coeff_w_dmuj * w[0]
                        )
                    ),
                    type(mu_i_native[0])(
                        -scale
                        * (
                            coeff_r_dmuj * r_vec[1]
                            + coeff_mui_dmuj * mu_i[1]
                            + coeff_w_dmuj * w[1]
                        )
                    ),
                    type(mu_i_native[0])(
                        -scale
                        * (
                            coeff_r_dmuj * r_vec[2]
                            + coeff_mui_dmuj * mu_i[2]
                            + coeff_w_dmuj * w[2]
                        )
                    ),
                ),
            )
            dqmu_x = qi * mu_j[0] - qj * mu_i[0]
            dqmu_y = qi * mu_j[1] - qj * mu_i[1]
            dqmu_z = qi * mu_j[2] - qj * mu_i[2]
            gpx = (
                s_rad * r_vec[0]
                + coeff_muj_dr * mu_j[0]
                + coeff_mui_dr * mu_i[0]
                + coeff_w_dr * w[0]
                + coeff_dqmu_dr * dqmu_x
            )
            gpy = (
                s_rad * r_vec[1]
                + coeff_muj_dr * mu_j[1]
                + coeff_mui_dr * mu_i[1]
                + coeff_w_dr * w[1]
                + coeff_dqmu_dr * dqmu_y
            )
            gpz = (
                s_rad * r_vec[2]
                + coeff_muj_dr * mu_j[2]
                + coeff_mui_dr * mu_i[2]
                + coeff_w_dr * w[2]
                + coeff_dqmu_dr * dqmu_z
            )
            sgx = scale * gpx
            sgy = scale * gpy
            sgz = scale * gpz
            wp.atomic_add(
                grad_positions,
                atom_i,
                type(pos_i)(
                    type(pos_i[0])(sgx),
                    type(pos_i[0])(sgy),
                    type(pos_i[0])(sgz),
                ),
            )
            wp.atomic_add(
                grad_positions,
                j,
                type(pos_i)(
                    type(pos_i[0])(-sgx),
                    type(pos_i[0])(-sgy),
                    type(pos_i[0])(-sgz),
                ),
            )
            cgx = -sgx
            cgy = -sgy
            cgz = -sgz
            wp.atomic_add(
                grad_cell_out,
                0,
                type(gcell)(
                    type(gcell[0, 0])(sh0 * cgx),
                    type(gcell[0, 0])(sh0 * cgy),
                    type(gcell[0, 0])(sh0 * cgz),
                    type(gcell[0, 0])(sh1 * cgx),
                    type(gcell[0, 0])(sh1 * cgy),
                    type(gcell[0, 0])(sh1 * cgz),
                    type(gcell[0, 0])(sh2 * cgx),
                    type(gcell[0, 0])(sh2 * cgy),
                    type(gcell[0, 0])(sh2 * cgz),
                ),
            )


def _dipole_csr_cell_grad_backward_sig(v, t):
    """Signature builder for the l=1 cell-grad double-backward kernel."""
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=m),  # cell
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # alpha
        wp.array(dtype=wp.float64),  # per_direction_scale
        wp.array(dtype=m),  # g_cell
        wp.array(dtype=v),  # grad_positions (out)
        wp.array(dtype=t),  # grad_charges (out)
        wp.array(dtype=v),  # grad_dipoles (out)
        wp.array(dtype=m),  # grad_cell_out (out)
    ]


_multipole_real_space_dipole_csr_cell_grad_backward_overloads = register_overloads(
    _multipole_real_space_dipole_csr_cell_grad_backward_kernel,
    _dipole_csr_cell_grad_backward_sig,
)


def multipole_real_space_dipole_csr_cell_grad_backward(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigma: wp.array,
    alpha: wp.array,
    per_direction_scale: wp.array,
    g_cell: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_dipoles: wp.array,
    grad_cell_out: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for the l=1 cell-grad double-backward (stress-loss).

    Caller pre-zeroes the four output arrays.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(positions.device)
    wp.launch(
        _multipole_real_space_dipole_csr_cell_grad_backward_overloads[vec_dtype],
        dim=positions.shape[0],
        inputs=[
            positions,
            charges,
            dipoles,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            per_direction_scale,
            g_cell,
        ],
        outputs=[grad_positions, grad_charges, grad_dipoles, grad_cell_out],
        device=device,
    )


@wp.kernel
def _batch_multipole_real_space_dipole_csr_cell_grad_backward_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    dipoles: wp.array(dtype=Any),
    cells: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    atom_batch_idx: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    sigma: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    per_direction_scale: wp.array(dtype=wp.float64),
    g_cell: wp.array(dtype=Any),  # (1,) mat33 cotangent on grad_cell
    grad_positions: wp.array(dtype=Any),  # (N,) OUT
    grad_charges: wp.array(dtype=Any),  # (N,) OUT
    grad_dipoles: wp.array(dtype=Any),  # (N,) OUT
    grad_cell_out: wp.array(dtype=Any),  # (1,) mat33 OUT (atomic)
):
    r"""Double-backward of the l=1 real-space cell-grad (stress-loss).

    Reuses the force-loss dipole Hessian (``∂Ω/∂θ``) with the incoming
    charge/dipole directions set to zero and the position direction replaced by
    the per-pair ``w = g_cellᵀ·n`` (n = unit_shifts), so ``Ω_gp = -w·f`` with
    ``f = ∂E_pair/∂r``. With ``S = ⟨g_cell, grad_cell⟩ = Σ scale·(w·f) =
    -scale·Σ Ω_gp``: ``grad_r_i = +scale·G_pos``, ``grad_r_j = -scale·G_pos``,
    ``grad_θ = -scale·∂Ω/∂θ`` (θ = q_i,q_j,μ_i,μ_j),
    ``grad_cell[a,b] = -scale·n[a]·G_pos[b]``. Verified to reduce to the
    monopole kernel when dipoles vanish.
    """
    atom_i = wp.tid()
    b = atom_batch_idx[atom_i]
    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    mu_i_native = dipoles[atom_i]
    mu_i = wp.vec3d(
        wp.float64(mu_i_native[0]),
        wp.float64(mu_i_native[1]),
        wp.float64(mu_i_native[2]),
    )
    sigma_ = wp.float64(sigma[0])
    alpha_ = wp.float64(alpha[0])
    scale = per_direction_scale[0]
    cell_t = wp.transpose(cells[b])
    gcell = g_cell[b]
    g00 = wp.float64(gcell[0, 0])
    g01 = wp.float64(gcell[0, 1])
    g02 = wp.float64(gcell[0, 2])
    g10 = wp.float64(gcell[1, 0])
    g11 = wp.float64(gcell[1, 1])
    g12 = wp.float64(gcell[1, 2])
    g20 = wp.float64(gcell[2, 0])
    g21 = wp.float64(gcell[2, 1])
    g22 = wp.float64(gcell[2, 2])

    ab = _gto_ewald_ab(sigma_, alpha_)
    a_coef = ab[0]
    b_coef = ab[1]

    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]
    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]
        qj = wp.float64(charges[j])
        pos_j = positions[j]
        mu_j_native = dipoles[j]
        mu_j = wp.vec3d(
            wp.float64(mu_j_native[0]),
            wp.float64(mu_j_native[1]),
            wp.float64(mu_j_native[2]),
        )
        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))
        if distance > wp.float64(1e-8):
            r_vec = wp.vec3d(
                wp.float64(separation_vector[0]),
                wp.float64(separation_vector[1]),
                wp.float64(separation_vector[2]),
            )
            inv_r = wp.float64(1.0) / distance
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r * inv_r2
            inv_r4 = inv_r2 * inv_r2
            inv_r5 = inv_r2 * inv_r3
            inv_r6 = inv_r3 * inv_r3
            inv_r7 = inv_r3 * inv_r4
            ra = _gto_ewald_A_single(distance, a_coef)
            rb = _gto_ewald_A_single(distance, b_coef)
            a_scalar = ra[0] - rb[0]
            a_prime = ra[1] - rb[1]
            a_double_prime = ra[2] - rb[2]
            a_triple_prime = ra[3] - rb[3]
            c_diag = a_scalar * inv_r
            c_quad = a_prime * inv_r2 - a_scalar * inv_r3
            c3 = (
                a_double_prime * inv_r3
                - wp.float64(3.0) * a_prime * inv_r4
                + wp.float64(3.0) * a_scalar * inv_r5
            )
            c4 = (
                a_triple_prime * inv_r4
                - wp.float64(6.0) * a_double_prime * inv_r5
                + wp.float64(15.0) * a_prime * inv_r6
                - wp.float64(15.0) * a_scalar * inv_r7
            )

            # Per-pair direction w = g_cellᵀ·n (replaces the gp_i of force-loss).
            sh0 = wp.float64(shift_vec[0])
            sh1 = wp.float64(shift_vec[1])
            sh2 = wp.float64(shift_vec[2])
            w = wp.vec3d(
                g00 * sh0 + g10 * sh1 + g20 * sh2,
                g01 * sh0 + g11 * sh1 + g21 * sh2,
                g02 * sh0 + g12 * sh1 + g22 * sh2,
            )

            mu_i_dot_r = mu_i[0] * r_vec[0] + mu_i[1] * r_vec[1] + mu_i[2] * r_vec[2]
            mu_j_dot_r = mu_j[0] * r_vec[0] + mu_j[1] * r_vec[1] + mu_j[2] * r_vec[2]
            mu_dot = mu_i[0] * mu_j[0] + mu_i[1] * mu_j[1] + mu_i[2] * mu_j[2]
            w_dot_r = w[0] * r_vec[0] + w[1] * r_vec[1] + w[2] * r_vec[2]
            w_dot_mu_i = w[0] * mu_i[0] + w[1] * mu_i[1] + w[2] * mu_i[2]
            w_dot_mu_j = w[0] * mu_j[0] + w[1] * mu_j[1] + w[2] * mu_j[2]
            dqmu_dot_r = qi * mu_j_dot_r - qj * mu_i_dot_r
            w_dot_dqmu = qi * w_dot_mu_j - qj * w_dot_mu_i

            rad = (
                -qi * qj * c_diag
                - c_quad * dqmu_dot_r
                + c_quad * mu_dot
                + c3 * mu_i_dot_r * mu_j_dot_r
            )

            # ∂Ω/∂q_i and ∂Ω/∂q_j  (gc_i = gd_i = 0).
            d_omega_dqi = (
                qj * c_diag + c_quad * mu_j_dot_r
            ) * w_dot_r + c_diag * w_dot_mu_j
            d_omega_dqj = (
                qi * c_diag - c_quad * mu_i_dot_r
            ) * w_dot_r - c_diag * w_dot_mu_i

            # ∂Ω/∂μ_i.
            coeff_r_dmui = (
                -c_quad * qj * w_dot_r - c3 * mu_j_dot_r * w_dot_r - c_quad * w_dot_mu_j
            )
            coeff_muj_dmui = -c_quad * w_dot_r
            coeff_w_dmui = -c_diag * qj - c_quad * mu_j_dot_r
            # ∂Ω/∂μ_j.
            coeff_r_dmuj = (
                c_quad * qi * w_dot_r - c3 * mu_i_dot_r * w_dot_r - c_quad * w_dot_mu_i
            )
            coeff_mui_dmuj = -c_quad * w_dot_r
            coeff_w_dmuj = c_diag * qi - c_quad * mu_i_dot_r

            # ∂Ω/∂r_vec = G_pos.
            s_rad = (
                w_dot_r
                * (
                    qi * qj * c_quad
                    + c3 * dqmu_dot_r
                    - c3 * mu_dot
                    - c4 * mu_i_dot_r * mu_j_dot_r
                )
                + c_quad * w_dot_dqmu
                - c3 * w_dot_mu_i * mu_j_dot_r
                - c3 * w_dot_mu_j * mu_i_dot_r
            )
            coeff_muj_dr = -w_dot_r * c3 * mu_i_dot_r - c_quad * w_dot_mu_i
            coeff_mui_dr = -w_dot_r * c3 * mu_j_dot_r - c_quad * w_dot_mu_j
            coeff_w_dr = -rad
            coeff_dqmu_dr = w_dot_r * c_quad

            # Scatter (signs verified against the monopole reduction):
            # grad_q = -scale·∂Ω/∂q ; grad_μ = -scale·∂Ω/∂μ ;
            # grad_r_i = +scale·G_pos ; grad_r_j = -scale·G_pos ;
            # grad_cell[a,b] = -scale·n[a]·G_pos[b].
            wp.atomic_add(
                grad_charges, atom_i, type(charges[atom_i])(-scale * d_omega_dqi)
            )
            wp.atomic_add(grad_charges, j, type(charges[atom_i])(-scale * d_omega_dqj))
            wp.atomic_add(
                grad_dipoles,
                atom_i,
                type(mu_i_native)(
                    type(mu_i_native[0])(
                        -scale
                        * (
                            coeff_r_dmui * r_vec[0]
                            + coeff_muj_dmui * mu_j[0]
                            + coeff_w_dmui * w[0]
                        )
                    ),
                    type(mu_i_native[0])(
                        -scale
                        * (
                            coeff_r_dmui * r_vec[1]
                            + coeff_muj_dmui * mu_j[1]
                            + coeff_w_dmui * w[1]
                        )
                    ),
                    type(mu_i_native[0])(
                        -scale
                        * (
                            coeff_r_dmui * r_vec[2]
                            + coeff_muj_dmui * mu_j[2]
                            + coeff_w_dmui * w[2]
                        )
                    ),
                ),
            )
            wp.atomic_add(
                grad_dipoles,
                j,
                type(mu_i_native)(
                    type(mu_i_native[0])(
                        -scale
                        * (
                            coeff_r_dmuj * r_vec[0]
                            + coeff_mui_dmuj * mu_i[0]
                            + coeff_w_dmuj * w[0]
                        )
                    ),
                    type(mu_i_native[0])(
                        -scale
                        * (
                            coeff_r_dmuj * r_vec[1]
                            + coeff_mui_dmuj * mu_i[1]
                            + coeff_w_dmuj * w[1]
                        )
                    ),
                    type(mu_i_native[0])(
                        -scale
                        * (
                            coeff_r_dmuj * r_vec[2]
                            + coeff_mui_dmuj * mu_i[2]
                            + coeff_w_dmuj * w[2]
                        )
                    ),
                ),
            )
            dqmu_x = qi * mu_j[0] - qj * mu_i[0]
            dqmu_y = qi * mu_j[1] - qj * mu_i[1]
            dqmu_z = qi * mu_j[2] - qj * mu_i[2]
            gpx = (
                s_rad * r_vec[0]
                + coeff_muj_dr * mu_j[0]
                + coeff_mui_dr * mu_i[0]
                + coeff_w_dr * w[0]
                + coeff_dqmu_dr * dqmu_x
            )
            gpy = (
                s_rad * r_vec[1]
                + coeff_muj_dr * mu_j[1]
                + coeff_mui_dr * mu_i[1]
                + coeff_w_dr * w[1]
                + coeff_dqmu_dr * dqmu_y
            )
            gpz = (
                s_rad * r_vec[2]
                + coeff_muj_dr * mu_j[2]
                + coeff_mui_dr * mu_i[2]
                + coeff_w_dr * w[2]
                + coeff_dqmu_dr * dqmu_z
            )
            sgx = scale * gpx
            sgy = scale * gpy
            sgz = scale * gpz
            wp.atomic_add(
                grad_positions,
                atom_i,
                type(pos_i)(
                    type(pos_i[0])(sgx),
                    type(pos_i[0])(sgy),
                    type(pos_i[0])(sgz),
                ),
            )
            wp.atomic_add(
                grad_positions,
                j,
                type(pos_i)(
                    type(pos_i[0])(-sgx),
                    type(pos_i[0])(-sgy),
                    type(pos_i[0])(-sgz),
                ),
            )
            cgx = -sgx
            cgy = -sgy
            cgz = -sgz
            wp.atomic_add(
                grad_cell_out,
                b,
                type(gcell)(
                    type(gcell[0, 0])(sh0 * cgx),
                    type(gcell[0, 0])(sh0 * cgy),
                    type(gcell[0, 0])(sh0 * cgz),
                    type(gcell[0, 0])(sh1 * cgx),
                    type(gcell[0, 0])(sh1 * cgy),
                    type(gcell[0, 0])(sh1 * cgz),
                    type(gcell[0, 0])(sh2 * cgx),
                    type(gcell[0, 0])(sh2 * cgy),
                    type(gcell[0, 0])(sh2 * cgz),
                ),
            )


def _batch_dipole_csr_cell_grad_backward_sig(v, t):
    """Signature builder for the l=1 cell-grad double-backward kernel."""
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=m),  # cells
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.int32),  # atom_batch_idx
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # alpha
        wp.array(dtype=wp.float64),  # per_direction_scale
        wp.array(dtype=m),  # g_cell
        wp.array(dtype=v),  # grad_positions (out)
        wp.array(dtype=t),  # grad_charges (out)
        wp.array(dtype=v),  # grad_dipoles (out)
        wp.array(dtype=m),  # grad_cell_out (out)
    ]


_batch_multipole_real_space_dipole_csr_cell_grad_backward_overloads = (
    register_overloads(
        _batch_multipole_real_space_dipole_csr_cell_grad_backward_kernel,
        _batch_dipole_csr_cell_grad_backward_sig,
    )
)


def batch_multipole_real_space_dipole_csr_cell_grad_backward(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    cells: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    atom_batch_idx: wp.array,
    unit_shifts: wp.array,
    sigma: wp.array,
    alpha: wp.array,
    per_direction_scale: wp.array,
    g_cell: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_dipoles: wp.array,
    grad_cell_out: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for the l=1 cell-grad double-backward (stress-loss).

    Caller pre-zeroes the four output arrays.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(positions.device)
    wp.launch(
        _batch_multipole_real_space_dipole_csr_cell_grad_backward_overloads[vec_dtype],
        dim=positions.shape[0],
        inputs=[
            positions,
            charges,
            dipoles,
            cells,
            idx_j,
            neighbor_ptr,
            atom_batch_idx,
            unit_shifts,
            sigma,
            alpha,
            per_direction_scale,
            g_cell,
        ],
        outputs=[grad_positions, grad_charges, grad_dipoles, grad_cell_out],
        device=device,
    )


# =============================================================================
# l_max = 1 — fused energy + gradient, CSR neighbor-list, single-system
# =============================================================================


def multipole_real_space_dipole_csr_energy_fused(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigma: wp.array,
    alpha: wp.array,
    pair_energies: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_dipoles: wp.array,
    *,
    with_pos_grad: bool,
    with_charge_grad: bool,
    with_dipole_grad: bool,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Fused lmax=1 CSR launcher.

    Routes on ``(with_pos_grad, with_charge_grad, with_dipole_grad)``.
    Output slots are written iff the corresponding flag is ``True``.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype wp.float32 or wp.float64
        Atomic charges.
    dipoles : wp.array, shape (N,), dtype matching ``positions``
        Cartesian dipole moments ``(x, y, z)``.
    cell : wp.array, shape (1,), dtype wp.mat33f or wp.mat33d
        Lattice matrix.
    idx_j : wp.array, shape (M,), dtype wp.int32
        CSR neighbor target indices.
    neighbor_ptr : wp.array, shape (N+1,), dtype wp.int32
        CSR row pointers into ``idx_j`` / ``unit_shifts``.
    unit_shifts : wp.array, shape (M,), dtype wp.vec3i
        Per-edge periodic image shifts.
    sigma : wp.array, shape (1,), dtype matching ``charges``
        GTO smearing width.
    alpha : wp.array, shape (1,), dtype matching ``charges``
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype wp.float64
        OUTPUT (pre-zeroed). Per-atom accumulated energy.
    grad_positions : wp.array, shape (N,), dtype matching ``positions``
        OUTPUT. Written only when ``with_pos_grad``.
    grad_charges : wp.array, shape (N,), dtype matching ``charges``
        OUTPUT. Written only when ``with_charge_grad``.
    grad_dipoles : wp.array, shape (N,), dtype matching ``dipoles``
        OUTPUT. Written only when ``with_dipole_grad``.
    with_pos_grad, with_charge_grad, with_dipole_grad : bool
        Per-slot gradient emission flags (keyword-only).
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` (keyword-only).
    device : str, optional
        Warp device string. Defaults to ``positions.device`` (keyword-only).
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    num_atoms = positions.shape[0]

    if device is None:
        device = str(positions.device)

    _overload = _get_real_space_pair_overload(
        LMAX=1,
        storage="csr",
        is_batch=False,
        with_pos_grad=with_pos_grad,
        with_charge_grad=with_charge_grad,
        with_dipole_grad=with_dipole_grad,
        with_quad_grad=False,
        with_cell_grad=False,
        vec_dtype=vec_dtype,
        scalar_dtype=wp_dtype,
    )

    # Swap 1-element placeholders for N-sized scratch when the all-grads
    # kernel is selected but only a subset of grads was requested.
    any_grad = with_pos_grad or with_charge_grad or with_dipole_grad
    if any_grad:
        if not with_pos_grad:
            grad_positions = wp.zeros(num_atoms, dtype=vec_dtype, device=device)
        if not with_charge_grad:
            grad_charges = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        if not with_dipole_grad:
            grad_dipoles = wp.zeros(num_atoms, dtype=vec_dtype, device=device)

    wp.launch(
        _overload,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            pair_energies,
            grad_positions,
            grad_charges,
            grad_dipoles,
        ],
        device=device,
    )


# =============================================================================
# Batched variants — l_max = 0
# =============================================================================


def batch_multipole_real_space_monopole_csr_energy(
    positions: wp.array,
    charges: wp.array,
    cells: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigmas: wp.array,
    alphas: wp.array,
    batch_idx: wp.array,
    pair_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Energy-only launcher for the LMAX=0 CSR batched kernel.

    Per-system ``cells[b]`` / ``sigmas[b]`` / ``alphas[b]`` lookup via
    ``batch_idx[atom_i]``. Internally allocates 1-element scratch grad
    arrays for the unused slots.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype wp.vec3f or wp.vec3d
        Concatenated atomic positions across all systems.
    charges : wp.array, shape (N_total,), dtype wp.float32 or wp.float64
        Concatenated atomic charges.
    cells : wp.array, shape (B,), dtype wp.mat33f or wp.mat33d
        Per-system lattice matrices.
    idx_j : wp.array, shape (M,), dtype wp.int32
        Flattened CSR neighbor target indices (global atom indices).
    neighbor_ptr : wp.array, shape (N_total+1,), dtype wp.int32
        CSR row pointers into ``idx_j`` / ``unit_shifts``.
    unit_shifts : wp.array, shape (M,), dtype wp.vec3i
        Per-edge periodic image shifts.
    sigmas : wp.array, shape (B,), dtype matching ``charges``
        Per-system GTO smearing widths.
    alphas : wp.array, shape (B,), dtype matching ``charges``
        Per-system Ewald splitting parameters.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        System index ``b`` for each atom.
    pair_energies : wp.array, shape (N_total,), dtype wp.float64
        OUTPUT (pre-zeroed). Per-atom accumulated energy.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64``.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    grad_pos_scratch = wp.zeros(1, dtype=vec_dtype, device=device)
    grad_q_scratch = wp.zeros(1, dtype=wp_dtype, device=device)

    _overload = _get_real_space_pair_overload(
        LMAX=0,
        storage="csr",
        is_batch=True,
        with_pos_grad=False,
        with_charge_grad=False,
        with_dipole_grad=False,
        with_quad_grad=False,
        with_cell_grad=False,
        vec_dtype=vec_dtype,
        scalar_dtype=wp_dtype,
    )

    wp.launch(
        _overload,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cells,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigmas,
            alphas,
            batch_idx,
            pair_energies,
            grad_pos_scratch,
            grad_q_scratch,
        ],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _batch_multipole_real_space_monopole_csr_energy_backward_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cells: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    sigmas: wp.array(dtype=Any),
    alphas: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    grad_energies: wp.array(dtype=wp.float64),
    grad_positions: wp.array(dtype=Any),
    grad_charges: wp.array(dtype=Any),
):
    r"""Batched analytical backward of the GTO-Ewald lmax=0 energy kernel.

    Mirror of the single-system first-order backward with per-system
    ``cells[b]`` / ``sigmas[b]`` / ``alphas[b]`` lookup, where the system
    index ``b = batch_idx[i]``.

    Launch Grid
    -----------
    dim = [num_atoms_total] — one thread per atom across all batched
    systems; inner loop over the atom's CSR neighbor slice. Every output
    slot is written via ``atomic_add`` (half/full-list agnostic).

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype wp.vec3f or wp.vec3d
        Concatenated atomic positions across all systems.
    charges : wp.array, shape (N_total,), dtype wp.float32 or wp.float64
        Concatenated atomic charges.
    cells : wp.array, shape (B,), dtype wp.mat33f or wp.mat33d
        Per-system lattice matrices.
    idx_j : wp.array, shape (M,), dtype wp.int32
        Flattened neighbor target indices (global atom indices).
    neighbor_ptr : wp.array, shape (N_total+1,), dtype wp.int32
        CSR row pointers into ``idx_j`` / ``unit_shifts``.
    unit_shifts : wp.array, shape (M,), dtype wp.vec3i
        Per-edge periodic image shifts.
    sigmas : wp.array, shape (B,), dtype matching ``charges``
        Per-system GTO smearing widths.
    alphas : wp.array, shape (B,), dtype matching ``charges``
        Per-system Ewald splitting parameters.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        System index ``b`` for each atom.
    grad_energies : wp.array, shape (N_total,), dtype wp.float64
        Upstream cotangent ``∂L/∂pair_energies``.
    grad_positions : wp.array, shape (N_total,), dtype matching ``positions``
        OUTPUT (pre-zeroed). Gradient w.r.t. atomic positions.
    grad_charges : wp.array, shape (N_total,), dtype matching ``charges``
        OUTPUT (pre-zeroed). Gradient w.r.t. atomic charges.
    """
    atom_i = wp.tid()
    b = batch_idx[atom_i]

    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    sigma_ = wp.float64(sigmas[b])
    alpha_ = wp.float64(alphas[b])
    cell_t = wp.transpose(cells[b])
    ge_i = grad_energies[atom_i]

    ab = _gto_ewald_ab(sigma_, alpha_)
    a_coef = ab[0]
    b_coef = ab[1]

    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]
        qj = wp.float64(charges[j])
        pos_j = positions[j]

        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            r_vec = wp.vec3d(
                wp.float64(separation_vector[0]),
                wp.float64(separation_vector[1]),
                wp.float64(separation_vector[2]),
            )

            inv_r = wp.float64(1.0) / distance

            t0 = _gto_ewald_t0(distance, a_coef, b_coef)
            ra = _gto_ewald_A_single(distance, a_coef)
            rb = _gto_ewald_A_single(distance, b_coef)
            a_scalar = ra[0] - rb[0]

            half_t0 = wp.float64(0.5) * t0
            wp.atomic_add(
                grad_charges,
                atom_i,
                type(charges[atom_i])(ge_i * half_t0 * qj),
            )
            wp.atomic_add(
                grad_charges,
                j,
                type(charges[atom_i])(ge_i * half_t0 * qi),
            )

            pos_coeff = ge_i * wp.float64(0.5) * qi * qj * a_scalar * inv_r
            dx = pos_coeff * r_vec[0]
            dy = pos_coeff * r_vec[1]
            dz = pos_coeff * r_vec[2]
            wp.atomic_add(
                grad_positions,
                atom_i,
                type(pos_i)(
                    type(pos_i[0])(dx),
                    type(pos_i[0])(dy),
                    type(pos_i[0])(dz),
                ),
            )
            wp.atomic_add(
                grad_positions,
                j,
                type(pos_i)(
                    type(pos_i[0])(-dx),
                    type(pos_i[0])(-dy),
                    type(pos_i[0])(-dz),
                ),
            )


def _batch_monopole_csr_backward_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=m),  # cells
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigmas
        wp.array(dtype=t),  # alphas
        wp.array(dtype=wp.int32),  # batch_idx
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=v),  # grad_positions
        wp.array(dtype=t),  # grad_charges
    ]


_batch_multipole_real_space_monopole_csr_energy_backward_overloads = register_overloads(
    _batch_multipole_real_space_monopole_csr_energy_backward_kernel,
    _batch_monopole_csr_backward_sig,
)


def batch_multipole_real_space_monopole_csr_energy_backward(
    positions: wp.array,
    charges: wp.array,
    cells: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigmas: wp.array,
    alphas: wp.array,
    batch_idx: wp.array,
    grad_energies: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for the batched l_max=0 first-order backward kernel.

    Parameters
    ----------
    positions, charges, cells, idx_j, neighbor_ptr, unit_shifts, sigmas,
    alphas, batch_idx :
        Same semantics as
        :func:`batch_multipole_real_space_monopole_csr_energy`.
    grad_energies : wp.array, shape (N_total,), dtype wp.float64
        Upstream cotangent ``∂L/∂pair_energies``.
    grad_positions : wp.array, shape (N_total,), dtype matching ``positions``
        OUTPUT (pre-zeroed). Gradient w.r.t. atomic positions.
    grad_charges : wp.array, shape (N_total,), dtype matching ``charges``
        OUTPUT (pre-zeroed). Gradient w.r.t. atomic charges.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the overloaded variant.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)
    wp.launch(
        _batch_multipole_real_space_monopole_csr_energy_backward_overloads[vec_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cells,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigmas,
            alphas,
            batch_idx,
            grad_energies,
            grad_positions,
            grad_charges,
        ],
        device=device,
    )


# =============================================================================
# Batched l_max = 0 — fused energy + gradient
# =============================================================================


def batch_multipole_real_space_monopole_csr_energy_fused(
    positions: wp.array,
    charges: wp.array,
    cells: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigmas: wp.array,
    alphas: wp.array,
    batch_idx: wp.array,
    pair_energies: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    *,
    with_pos_grad: bool,
    with_charge_grad: bool,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Fused launcher for the batched lmax=0 CSR kernels.

    Routes on ``(with_pos_grad, with_charge_grad)``. Output slots are
    written iff the corresponding flag is ``True``.

    Parameters
    ----------
    positions, charges, cells, idx_j, neighbor_ptr, unit_shifts, sigmas,
    alphas, batch_idx :
        Same semantics as
        :func:`batch_multipole_real_space_monopole_csr_energy`.
    pair_energies : wp.array, shape (N_total,), dtype wp.float64
        OUTPUT (pre-zeroed). Per-atom accumulated energy.
    grad_positions : wp.array, shape (N_total,), dtype matching ``positions``
        OUTPUT. Written only when ``with_pos_grad``.
    grad_charges : wp.array, shape (N_total,), dtype matching ``charges``
        OUTPUT. Written only when ``with_charge_grad``.
    with_pos_grad, with_charge_grad : bool
        Per-slot gradient emission flags (keyword-only).
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` (keyword-only).
    device : str, optional
        Warp device string. Defaults to ``positions.device`` (keyword-only).
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    num_atoms = positions.shape[0]

    if device is None:
        device = str(positions.device)

    _overload = _get_real_space_pair_overload(
        LMAX=0,
        storage="csr",
        is_batch=True,
        with_pos_grad=with_pos_grad,
        with_charge_grad=with_charge_grad,
        with_dipole_grad=False,
        with_quad_grad=False,
        with_cell_grad=False,
        vec_dtype=vec_dtype,
        scalar_dtype=wp_dtype,
    )

    # Swap 1-element placeholders for N-sized scratch.
    any_grad = with_pos_grad or with_charge_grad
    if any_grad:
        if not with_pos_grad:
            grad_positions = wp.zeros(num_atoms, dtype=vec_dtype, device=device)
        if not with_charge_grad:
            grad_charges = wp.zeros(num_atoms, dtype=wp_dtype, device=device)

    wp.launch(
        _overload,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cells,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigmas,
            alphas,
            batch_idx,
            pair_energies,
            grad_positions,
            grad_charges,
        ],
        device=device,
    )


# =============================================================================
# Batched variants — l_max = 1 (charges + dipoles)
# =============================================================================


def batch_multipole_real_space_dipole_csr_energy(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    cells: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigmas: wp.array,
    alphas: wp.array,
    batch_idx: wp.array,
    pair_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Energy-only launcher for the LMAX=1 CSR batched kernel.

    Per-system ``cells[b]`` / ``sigmas[b]`` / ``alphas[b]`` lookup via
    ``batch_idx[atom_i]``. Internally allocates 1-element scratch grad
    arrays for the unused slots.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype wp.vec3f or wp.vec3d
        Concatenated atomic positions across all systems.
    charges : wp.array, shape (N_total,), dtype wp.float32 or wp.float64
        Concatenated atomic charges.
    dipoles : wp.array, shape (N_total,), dtype matching ``positions``
        Concatenated Cartesian dipole moments ``(x, y, z)``.
    cells : wp.array, shape (B,), dtype wp.mat33f or wp.mat33d
        Per-system lattice matrices.
    idx_j : wp.array, shape (M,), dtype wp.int32
        Flattened CSR neighbor target indices (global atom indices).
    neighbor_ptr : wp.array, shape (N_total+1,), dtype wp.int32
        CSR row pointers into ``idx_j`` / ``unit_shifts``.
    unit_shifts : wp.array, shape (M,), dtype wp.vec3i
        Per-edge periodic image shifts.
    sigmas : wp.array, shape (B,), dtype matching ``charges``
        Per-system GTO smearing widths.
    alphas : wp.array, shape (B,), dtype matching ``charges``
        Per-system Ewald splitting parameters.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        System index ``b`` for each atom.
    pair_energies : wp.array, shape (N_total,), dtype wp.float64
        OUTPUT (pre-zeroed). Per-atom accumulated energy.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64``.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    _overload = _get_real_space_pair_overload(
        LMAX=1,
        storage="csr",
        is_batch=True,
        with_pos_grad=False,
        with_charge_grad=False,
        with_dipole_grad=False,
        with_quad_grad=False,
        with_cell_grad=False,
        vec_dtype=vec_dtype,
        scalar_dtype=wp_dtype,
    )

    scratch_grad_pos = wp.zeros(1, dtype=vec_dtype, device=device)
    scratch_grad_q = wp.zeros(
        1,
        dtype=wp.float64 if wp_dtype == wp.float64 else wp.float32,
        device=device,
    )
    scratch_grad_mu = wp.zeros(1, dtype=vec_dtype, device=device)

    wp.launch(
        _overload,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            cells,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigmas,
            alphas,
            batch_idx,
            pair_energies,
            scratch_grad_pos,
            scratch_grad_q,
            scratch_grad_mu,
        ],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _batch_multipole_real_space_dipole_csr_energy_backward_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    dipoles: wp.array(dtype=Any),
    cells: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    sigmas: wp.array(dtype=Any),
    alphas: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    grad_energies: wp.array(dtype=wp.float64),
    grad_positions: wp.array(dtype=Any),
    grad_charges: wp.array(dtype=Any),
    grad_dipoles: wp.array(dtype=Any),
):
    r"""Batched GTO-Ewald l_max=1 first-order backward.

    Mirror of the single-system kernel with per-system ``cells[b]`` /
    ``sigmas[b]`` / ``alphas[b]`` lookup, where ``b = batch_idx[i]``.

    Launch Grid
    -----------
    dim = [num_atoms_total] — one thread per atom across all batched
    systems; inner loop over the atom's CSR neighbor slice. Every output
    slot is written via ``atomic_add`` (half/full-list agnostic).

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype wp.vec3f or wp.vec3d
        Concatenated atomic positions across all systems.
    charges : wp.array, shape (N_total,), dtype wp.float32 or wp.float64
        Concatenated atomic charges.
    dipoles : wp.array, shape (N_total,), dtype matching ``positions``
        Concatenated Cartesian dipole moments ``(x, y, z)``.
    cells : wp.array, shape (B,), dtype wp.mat33f or wp.mat33d
        Per-system lattice matrices.
    idx_j : wp.array, shape (M,), dtype wp.int32
        Flattened neighbor target indices (global atom indices).
    neighbor_ptr : wp.array, shape (N_total+1,), dtype wp.int32
        CSR row pointers into ``idx_j`` / ``unit_shifts``.
    unit_shifts : wp.array, shape (M,), dtype wp.vec3i
        Per-edge periodic image shifts.
    sigmas : wp.array, shape (B,), dtype matching ``charges``
        Per-system GTO smearing widths.
    alphas : wp.array, shape (B,), dtype matching ``charges``
        Per-system Ewald splitting parameters.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        System index ``b`` for each atom.
    grad_energies : wp.array, shape (N_total,), dtype wp.float64
        Upstream cotangent ``∂L/∂pair_energies``.
    grad_positions : wp.array, shape (N_total,), dtype matching ``positions``
        OUTPUT (pre-zeroed). Gradient w.r.t. atomic positions.
    grad_charges : wp.array, shape (N_total,), dtype matching ``charges``
        OUTPUT (pre-zeroed). Gradient w.r.t. atomic charges.
    grad_dipoles : wp.array, shape (N_total,), dtype matching ``dipoles``
        OUTPUT (pre-zeroed). Gradient w.r.t. dipole moments.
    """
    atom_i = wp.tid()
    b = batch_idx[atom_i]

    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    mu_i_native = dipoles[atom_i]
    mu_i = wp.vec3d(
        wp.float64(mu_i_native[0]),
        wp.float64(mu_i_native[1]),
        wp.float64(mu_i_native[2]),
    )
    sigma_ = wp.float64(sigmas[b])
    alpha_ = wp.float64(alphas[b])
    cell_t = wp.transpose(cells[b])
    ge_i = grad_energies[atom_i]

    ab = _gto_ewald_ab(sigma_, alpha_)
    a_coef = ab[0]
    b_coef = ab[1]

    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]
        qj = wp.float64(charges[j])
        pos_j = positions[j]
        mu_j_native = dipoles[j]
        mu_j = wp.vec3d(
            wp.float64(mu_j_native[0]),
            wp.float64(mu_j_native[1]),
            wp.float64(mu_j_native[2]),
        )

        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            r_vec = wp.vec3d(
                wp.float64(separation_vector[0]),
                wp.float64(separation_vector[1]),
                wp.float64(separation_vector[2]),
            )

            inv_r = wp.float64(1.0) / distance
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r * inv_r2
            inv_r4 = inv_r2 * inv_r2
            inv_r5 = inv_r2 * inv_r3

            # GTO-Ewald radial helpers at a and b.
            t0 = _gto_ewald_t0(distance, a_coef, b_coef)
            ra = _gto_ewald_A_single(distance, a_coef)
            rb = _gto_ewald_A_single(distance, b_coef)
            a_scalar = ra[0] - rb[0]
            a_prime = ra[1] - rb[1]
            a_double_prime = ra[2] - rb[2]

            c_diag = a_scalar * inv_r
            c_quad = a_prime * inv_r2 - a_scalar * inv_r3
            c3 = (
                a_double_prime * inv_r3
                - wp.float64(3.0) * a_prime * inv_r4
                + wp.float64(3.0) * a_scalar * inv_r5
            )

            neg_a_over_r = -c_diag
            t1x = neg_a_over_r * r_vec[0]
            t1y = neg_a_over_r * r_vec[1]
            t1z = neg_a_over_r * r_vec[2]

            mu_i_dot_r = mu_i[0] * r_vec[0] + mu_i[1] * r_vec[1] + mu_i[2] * r_vec[2]
            mu_j_dot_r = mu_j[0] * r_vec[0] + mu_j[1] * r_vec[1] + mu_j[2] * r_vec[2]
            mu_i_dot_mu_j = mu_i[0] * mu_j[0] + mu_i[1] * mu_j[1] + mu_i[2] * mu_j[2]
            mu_j_dot_T1 = t1x * mu_j[0] + t1y * mu_j[1] + t1z * mu_j[2]
            mu_i_dot_T1 = t1x * mu_i[0] + t1y * mu_i[1] + t1z * mu_i[2]

            half_ge_i = wp.float64(0.5) * ge_i

            # Charge gradients.
            dPE_dq_i = qj * t0 + mu_j_dot_T1
            dPE_dq_j = qi * t0 - mu_i_dot_T1
            wp.atomic_add(
                grad_charges,
                atom_i,
                type(charges[atom_i])(half_ge_i * dPE_dq_i),
            )
            wp.atomic_add(
                grad_charges,
                j,
                type(charges[atom_i])(half_ge_i * dPE_dq_j),
            )

            # Dipole gradients.
            cq_muj_r = c_quad * mu_j_dot_r
            cq_mui_r = c_quad * mu_i_dot_r
            dmu_i_x = -qj * t1x + c_diag * mu_j[0] + cq_muj_r * r_vec[0]
            dmu_i_y = -qj * t1y + c_diag * mu_j[1] + cq_muj_r * r_vec[1]
            dmu_i_z = -qj * t1z + c_diag * mu_j[2] + cq_muj_r * r_vec[2]
            dmu_j_x = qi * t1x + c_diag * mu_i[0] + cq_mui_r * r_vec[0]
            dmu_j_y = qi * t1y + c_diag * mu_i[1] + cq_mui_r * r_vec[1]
            dmu_j_z = qi * t1z + c_diag * mu_i[2] + cq_mui_r * r_vec[2]

            mu_i_contrib = type(mu_i_native)(
                type(mu_i_native[0])(half_ge_i * dmu_i_x),
                type(mu_i_native[0])(half_ge_i * dmu_i_y),
                type(mu_i_native[0])(half_ge_i * dmu_i_z),
            )
            mu_j_contrib = type(mu_i_native)(
                type(mu_i_native[0])(half_ge_i * dmu_j_x),
                type(mu_i_native[0])(half_ge_i * dmu_j_y),
                type(mu_i_native[0])(half_ge_i * dmu_j_z),
            )
            wp.atomic_add(grad_dipoles, atom_i, mu_i_contrib)
            wp.atomic_add(grad_dipoles, j, mu_j_contrib)

            # Position gradients.
            rad_coeff = (
                -qi * qj * c_diag
                - c_quad * (qi * mu_j_dot_r - qj * mu_i_dot_r)
                + c_quad * mu_i_dot_mu_j
                + c3 * mu_i_dot_r * mu_j_dot_r
            )
            dir_x = (
                -c_diag * (qi * mu_j[0] - qj * mu_i[0])
                + c_quad * mu_j_dot_r * mu_i[0]
                + c_quad * mu_i_dot_r * mu_j[0]
            )
            dir_y = (
                -c_diag * (qi * mu_j[1] - qj * mu_i[1])
                + c_quad * mu_j_dot_r * mu_i[1]
                + c_quad * mu_i_dot_r * mu_j[1]
            )
            dir_z = (
                -c_diag * (qi * mu_j[2] - qj * mu_i[2])
                + c_quad * mu_j_dot_r * mu_i[2]
                + c_quad * mu_i_dot_r * mu_j[2]
            )
            dPE_dr_x = rad_coeff * r_vec[0] + dir_x
            dPE_dr_y = rad_coeff * r_vec[1] + dir_y
            dPE_dr_z = rad_coeff * r_vec[2] + dir_z

            px = half_ge_i * dPE_dr_x
            py = half_ge_i * dPE_dr_y
            pz = half_ge_i * dPE_dr_z
            wp.atomic_add(
                grad_positions,
                j,
                type(pos_i)(
                    type(pos_i[0])(px),
                    type(pos_i[0])(py),
                    type(pos_i[0])(pz),
                ),
            )
            wp.atomic_add(
                grad_positions,
                atom_i,
                type(pos_i)(
                    type(pos_i[0])(-px),
                    type(pos_i[0])(-py),
                    type(pos_i[0])(-pz),
                ),
            )


def _batch_dipole_csr_backward_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=m),  # cells
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigmas
        wp.array(dtype=t),  # alphas
        wp.array(dtype=wp.int32),  # batch_idx
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=v),  # grad_positions
        wp.array(dtype=t),  # grad_charges
        wp.array(dtype=v),  # grad_dipoles
    ]


_batch_multipole_real_space_dipole_csr_energy_backward_overloads = register_overloads(
    _batch_multipole_real_space_dipole_csr_energy_backward_kernel,
    _batch_dipole_csr_backward_sig,
)


def batch_multipole_real_space_dipole_csr_energy_backward(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    cells: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigmas: wp.array,
    alphas: wp.array,
    batch_idx: wp.array,
    grad_energies: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_dipoles: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for the batched l_max=1 GTO-Ewald first-order backward kernel.

    Parameters
    ----------
    positions, charges, dipoles, cells, idx_j, neighbor_ptr, unit_shifts,
    sigmas, alphas, batch_idx :
        Same semantics as
        :func:`batch_multipole_real_space_dipole_csr_energy`.
    grad_energies : wp.array, shape (N_total,), dtype wp.float64
        Upstream cotangent ``∂L/∂pair_energies``.
    grad_positions : wp.array, shape (N_total,), dtype matching ``positions``
        OUTPUT (pre-zeroed). Gradient w.r.t. atomic positions.
    grad_charges : wp.array, shape (N_total,), dtype matching ``charges``
        OUTPUT (pre-zeroed). Gradient w.r.t. atomic charges.
    grad_dipoles : wp.array, shape (N_total,), dtype matching ``dipoles``
        OUTPUT (pre-zeroed). Gradient w.r.t. dipole moments.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the overloaded variant.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)
    wp.launch(
        _batch_multipole_real_space_dipole_csr_energy_backward_overloads[vec_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            cells,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigmas,
            alphas,
            batch_idx,
            grad_energies,
            grad_positions,
            grad_charges,
            grad_dipoles,
        ],
        device=device,
    )


# =============================================================================
# Batched l_max = 1 — fused energy + gradient
# =============================================================================


def batch_multipole_real_space_dipole_csr_energy_fused(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    cells: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigmas: wp.array,
    alphas: wp.array,
    batch_idx: wp.array,
    pair_energies: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_dipoles: wp.array,
    *,
    with_pos_grad: bool,
    with_charge_grad: bool,
    with_dipole_grad: bool,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Fused launcher for batched lmax=1 CSR.

    Routes on ``(with_pos_grad, with_charge_grad, with_dipole_grad)``.
    Output slots are written iff the corresponding flag is ``True``.

    Parameters
    ----------
    positions, charges, dipoles, cells, idx_j, neighbor_ptr, unit_shifts,
    sigmas, alphas, batch_idx :
        Same semantics as
        :func:`batch_multipole_real_space_dipole_csr_energy`.
    pair_energies : wp.array, shape (N_total,), dtype wp.float64
        OUTPUT (pre-zeroed). Per-atom accumulated energy.
    grad_positions : wp.array, shape (N_total,), dtype matching ``positions``
        OUTPUT. Written only when ``with_pos_grad``.
    grad_charges : wp.array, shape (N_total,), dtype matching ``charges``
        OUTPUT. Written only when ``with_charge_grad``.
    grad_dipoles : wp.array, shape (N_total,), dtype matching ``dipoles``
        OUTPUT. Written only when ``with_dipole_grad``.
    with_pos_grad, with_charge_grad, with_dipole_grad : bool
        Per-slot gradient emission flags (keyword-only).
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` (keyword-only).
    device : str, optional
        Warp device string. Defaults to ``positions.device`` (keyword-only).
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    num_atoms = positions.shape[0]

    if device is None:
        device = str(positions.device)

    _overload = _get_real_space_pair_overload(
        LMAX=1,
        storage="csr",
        is_batch=True,
        with_pos_grad=with_pos_grad,
        with_charge_grad=with_charge_grad,
        with_dipole_grad=with_dipole_grad,
        with_quad_grad=False,
        with_cell_grad=False,
        vec_dtype=vec_dtype,
        scalar_dtype=wp_dtype,
    )

    # Swap 1-element placeholders for N-sized scratch.
    any_grad = with_pos_grad or with_charge_grad or with_dipole_grad
    if any_grad:
        if not with_pos_grad:
            grad_positions = wp.zeros(num_atoms, dtype=vec_dtype, device=device)
        if not with_charge_grad:
            grad_charges = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        if not with_dipole_grad:
            grad_dipoles = wp.zeros(num_atoms, dtype=vec_dtype, device=device)

    wp.launch(
        _overload,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            cells,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigmas,
            alphas,
            batch_idx,
            pair_energies,
            grad_positions,
            grad_charges,
            grad_dipoles,
        ],
        device=device,
    )


# =============================================================================
# Batched variants — second-order backward kernels
# =============================================================================


@wp.kernel(enable_backward=False)
def _batch_multipole_real_space_monopole_csr_energy_2nd_backward_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cells: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    sigmas: wp.array(dtype=Any),
    alphas: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    grad_energies: wp.array(dtype=wp.float64),
    gg_positions: wp.array(dtype=Any),
    gg_charges: wp.array(dtype=Any),
    gg_grad_energies_2nd: wp.array(dtype=wp.float64),
    gg_positions_2nd: wp.array(dtype=Any),
    gg_charges_2nd: wp.array(dtype=Any),
):
    r"""Batched l_max=0 GTO-Ewald second-order backward.

    Mirror of the single-system second-order backward with per-system
    ``cells[b]`` / ``sigmas[b]`` / ``alphas[b]`` lookup, where
    ``b = batch_idx[i]``. Differentiates the first-order backward's scalar
    functional w.r.t. ``(grad_energies, positions, charges)``.

    Launch Grid
    -----------
    dim = [num_atoms_total] — one thread per atom across all batched
    systems; inner loop over the atom's CSR neighbor slice. Every output
    slot is written via ``atomic_add``.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype wp.vec3f or wp.vec3d
        Concatenated atomic positions across all systems.
    charges : wp.array, shape (N_total,), dtype wp.float32 or wp.float64
        Concatenated atomic charges.
    cells : wp.array, shape (B,), dtype wp.mat33f or wp.mat33d
        Per-system lattice matrices.
    idx_j : wp.array, shape (M,), dtype wp.int32
        Flattened neighbor target indices (global atom indices).
    neighbor_ptr : wp.array, shape (N_total+1,), dtype wp.int32
        CSR row pointers into ``idx_j`` / ``unit_shifts``.
    unit_shifts : wp.array, shape (M,), dtype wp.vec3i
        Per-edge periodic image shifts.
    sigmas : wp.array, shape (B,), dtype matching ``charges``
        Per-system GTO smearing widths.
    alphas : wp.array, shape (B,), dtype matching ``charges``
        Per-system Ewald splitting parameters.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        System index ``b`` for each atom.
    grad_energies : wp.array, shape (N_total,), dtype wp.float64
        Original first-order upstream cotangent ``∂L/∂pair_energies``.
    gg_positions : wp.array, shape (N_total,), dtype matching ``positions``
        Upstream cotangent on the first-order backward's ``grad_positions``.
    gg_charges : wp.array, shape (N_total,), dtype matching ``charges``
        Upstream cotangent on ``grad_charges``.
    gg_grad_energies_2nd : wp.array, shape (N_total,), dtype wp.float64
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂grad_energies``.
    gg_positions_2nd : wp.array, shape (N_total,), dtype matching ``positions``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂positions``.
    gg_charges_2nd : wp.array, shape (N_total,), dtype matching ``charges``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂charges``.
    """
    atom_i = wp.tid()
    b = batch_idx[atom_i]

    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    sigma_ = wp.float64(sigmas[b])
    alpha_ = wp.float64(alphas[b])
    cell_t = wp.transpose(cells[b])
    ge_i = grad_energies[atom_i]
    gp_i_native = gg_positions[atom_i]
    gp_i = wp.vec3d(
        wp.float64(gp_i_native[0]),
        wp.float64(gp_i_native[1]),
        wp.float64(gp_i_native[2]),
    )
    gc_i = wp.float64(gg_charges[atom_i])

    ab = _gto_ewald_ab(sigma_, alpha_)
    a_coef = ab[0]
    b_coef = ab[1]

    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]
        qj = wp.float64(charges[j])
        pos_j = positions[j]
        ge_j = grad_energies[j]

        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            r_vec = wp.vec3d(
                wp.float64(separation_vector[0]),
                wp.float64(separation_vector[1]),
                wp.float64(separation_vector[2]),
            )

            inv_r = wp.float64(1.0) / distance
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r * inv_r2

            t0 = _gto_ewald_t0(distance, a_coef, b_coef)
            ra = _gto_ewald_A_single(distance, a_coef)
            rb = _gto_ewald_A_single(distance, b_coef)
            a_scalar = ra[0] - rb[0]
            a_prime = ra[1] - rb[1]
            a_over_r = a_scalar * inv_r
            c_quad = a_prime * inv_r2 - a_scalar * inv_r3

            gp_i_dot_r = gp_i[0] * r_vec[0] + gp_i[1] * r_vec[1] + gp_i[2] * r_vec[2]

            formula_k_b = gc_i * qj * t0 + qi * qj * a_over_r * gp_i_dot_r
            half_formula = wp.float64(0.5) * formula_k_b
            wp.atomic_add(gg_grad_energies_2nd, atom_i, half_formula)
            wp.atomic_add(gg_grad_energies_2nd, j, half_formula)

            ge_sum = ge_i + ge_j
            gx = -a_over_r * r_vec[0] * gc_i * qj + qi * qj * (
                c_quad * gp_i_dot_r * r_vec[0] + a_over_r * gp_i[0]
            )
            gy = -a_over_r * r_vec[1] * gc_i * qj + qi * qj * (
                c_quad * gp_i_dot_r * r_vec[1] + a_over_r * gp_i[1]
            )
            gz = -a_over_r * r_vec[2] * gc_i * qj + qi * qj * (
                c_quad * gp_i_dot_r * r_vec[2] + a_over_r * gp_i[2]
            )
            scale_pos = wp.float64(0.5) * ge_sum
            px = scale_pos * gx
            py = scale_pos * gy
            pz = scale_pos * gz
            wp.atomic_add(
                gg_positions_2nd,
                atom_i,
                type(pos_i)(
                    type(pos_i[0])(-px),
                    type(pos_i[0])(-py),
                    type(pos_i[0])(-pz),
                ),
            )
            wp.atomic_add(
                gg_positions_2nd,
                j,
                type(pos_i)(
                    type(pos_i[0])(px),
                    type(pos_i[0])(py),
                    type(pos_i[0])(pz),
                ),
            )

            gc_k_contrib = scale_pos * gp_i_dot_r * qj * a_over_r
            gc_b_contrib = scale_pos * (gc_i * t0 + gp_i_dot_r * qi * a_over_r)
            wp.atomic_add(gg_charges_2nd, atom_i, type(charges[atom_i])(gc_k_contrib))
            wp.atomic_add(gg_charges_2nd, j, type(charges[atom_i])(gc_b_contrib))


def _batch_monopole_csr_2nd_backward_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=m),  # cells
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigmas
        wp.array(dtype=t),  # alphas
        wp.array(dtype=wp.int32),  # batch_idx
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=v),  # gg_positions
        wp.array(dtype=t),  # gg_charges
        wp.array(dtype=wp.float64),  # gg_grad_energies_2nd
        wp.array(dtype=v),  # gg_positions_2nd
        wp.array(dtype=t),  # gg_charges_2nd
    ]


_batch_multipole_real_space_monopole_csr_energy_2nd_backward_overloads = (
    register_overloads(
        _batch_multipole_real_space_monopole_csr_energy_2nd_backward_kernel,
        _batch_monopole_csr_2nd_backward_sig,
    )
)


def batch_multipole_real_space_monopole_csr_energy_2nd_backward(
    positions,
    charges,
    cells,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigmas,
    alphas,
    batch_idx,
    grad_energies,
    gg_positions,
    gg_charges,
    gg_grad_energies_2nd,
    gg_positions_2nd,
    gg_charges_2nd,
    wp_dtype,
    device=None,
):
    r"""Launcher for the batched l_max=0 GTO-Ewald second-order backward kernel.

    Parameters
    ----------
    positions, charges, cells, idx_j, neighbor_ptr, unit_shifts, sigmas,
    alphas, batch_idx :
        Same semantics as
        :func:`batch_multipole_real_space_monopole_csr_energy`.
    grad_energies : wp.array, shape (N_total,), dtype wp.float64
        Original first-order upstream cotangent ``∂L/∂pair_energies``.
    gg_positions : wp.array, shape (N_total,), dtype matching ``positions``
        Upstream cotangent on the first-order ``grad_positions``.
    gg_charges : wp.array, shape (N_total,), dtype matching ``charges``
        Upstream cotangent on ``grad_charges``.
    gg_grad_energies_2nd : wp.array, shape (N_total,), dtype wp.float64
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂grad_energies``.
    gg_positions_2nd : wp.array, shape (N_total,), dtype matching ``positions``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂positions``.
    gg_charges_2nd : wp.array, shape (N_total,), dtype matching ``charges``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂charges``.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the overloaded variant.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)
    wp.launch(
        _batch_multipole_real_space_monopole_csr_energy_2nd_backward_overloads[
            vec_dtype
        ],
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cells,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigmas,
            alphas,
            batch_idx,
            grad_energies,
            gg_positions,
            gg_charges,
            gg_grad_energies_2nd,
            gg_positions_2nd,
            gg_charges_2nd,
        ],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _batch_multipole_real_space_dipole_csr_energy_2nd_backward_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    dipoles: wp.array(dtype=Any),
    cells: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    sigmas: wp.array(dtype=Any),
    alphas: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    grad_energies: wp.array(dtype=wp.float64),
    gg_positions: wp.array(dtype=Any),
    gg_charges: wp.array(dtype=Any),
    gg_dipoles: wp.array(dtype=Any),
    gg_grad_energies_2nd: wp.array(dtype=wp.float64),
    gg_positions_2nd: wp.array(dtype=Any),
    gg_charges_2nd: wp.array(dtype=Any),
    gg_dipoles_2nd: wp.array(dtype=Any),
):
    r"""Batched GTO-Ewald l_max=1 second-order backward.

    Mirror of the single-system second-order backward with per-system
    ``cells[b]`` / ``sigmas[b]`` / ``alphas[b]`` lookup, where
    ``b = batch_idx[i]``. Differentiates the first-order backward's scalar
    functional w.r.t. ``(grad_energies, positions, charges, dipoles)``.

    Launch Grid
    -----------
    dim = [num_atoms_total] — one thread per atom across all batched
    systems; inner loop over the atom's CSR neighbor slice. Every output
    slot is written via ``atomic_add``.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype wp.vec3f or wp.vec3d
        Concatenated atomic positions across all systems.
    charges : wp.array, shape (N_total,), dtype wp.float32 or wp.float64
        Concatenated atomic charges.
    dipoles : wp.array, shape (N_total,), dtype matching ``positions``
        Concatenated Cartesian dipole moments ``(x, y, z)``.
    cells : wp.array, shape (B,), dtype wp.mat33f or wp.mat33d
        Per-system lattice matrices.
    idx_j : wp.array, shape (M,), dtype wp.int32
        Flattened neighbor target indices (global atom indices).
    neighbor_ptr : wp.array, shape (N_total+1,), dtype wp.int32
        CSR row pointers into ``idx_j`` / ``unit_shifts``.
    unit_shifts : wp.array, shape (M,), dtype wp.vec3i
        Per-edge periodic image shifts.
    sigmas : wp.array, shape (B,), dtype matching ``charges``
        Per-system GTO smearing widths.
    alphas : wp.array, shape (B,), dtype matching ``charges``
        Per-system Ewald splitting parameters.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        System index ``b`` for each atom.
    grad_energies : wp.array, shape (N_total,), dtype wp.float64
        Original first-order upstream cotangent ``∂L/∂pair_energies``.
    gg_positions : wp.array, shape (N_total,), dtype matching ``positions``
        Upstream cotangent on the first-order backward's ``grad_positions``.
    gg_charges : wp.array, shape (N_total,), dtype matching ``charges``
        Upstream cotangent on ``grad_charges``.
    gg_dipoles : wp.array, shape (N_total,), dtype matching ``dipoles``
        Upstream cotangent on ``grad_dipoles``.
    gg_grad_energies_2nd : wp.array, shape (N_total,), dtype wp.float64
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂grad_energies``.
    gg_positions_2nd : wp.array, shape (N_total,), dtype matching ``positions``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂positions``.
    gg_charges_2nd : wp.array, shape (N_total,), dtype matching ``charges``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂charges``.
    gg_dipoles_2nd : wp.array, shape (N_total,), dtype matching ``dipoles``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂dipoles``.
    """
    atom_i = wp.tid()
    b = batch_idx[atom_i]

    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    mu_i_native = dipoles[atom_i]
    mu_i = wp.vec3d(
        wp.float64(mu_i_native[0]),
        wp.float64(mu_i_native[1]),
        wp.float64(mu_i_native[2]),
    )
    gp_i_native = gg_positions[atom_i]
    gp_i = wp.vec3d(
        wp.float64(gp_i_native[0]),
        wp.float64(gp_i_native[1]),
        wp.float64(gp_i_native[2]),
    )
    gd_i_native = gg_dipoles[atom_i]
    gd_i = wp.vec3d(
        wp.float64(gd_i_native[0]),
        wp.float64(gd_i_native[1]),
        wp.float64(gd_i_native[2]),
    )
    gc_i = wp.float64(gg_charges[atom_i])
    sigma_ = wp.float64(sigmas[b])
    alpha_ = wp.float64(alphas[b])
    cell_t = wp.transpose(cells[b])
    ge_i = grad_energies[atom_i]

    ab = _gto_ewald_ab(sigma_, alpha_)
    a_coef = ab[0]
    b_coef = ab[1]

    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]
        qj = wp.float64(charges[j])
        pos_j = positions[j]
        mu_j_native = dipoles[j]
        mu_j = wp.vec3d(
            wp.float64(mu_j_native[0]),
            wp.float64(mu_j_native[1]),
            wp.float64(mu_j_native[2]),
        )
        ge_j = grad_energies[j]

        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            r_vec = wp.vec3d(
                wp.float64(separation_vector[0]),
                wp.float64(separation_vector[1]),
                wp.float64(separation_vector[2]),
            )

            inv_r = wp.float64(1.0) / distance
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r * inv_r2
            inv_r4 = inv_r2 * inv_r2
            inv_r5 = inv_r2 * inv_r3
            inv_r6 = inv_r3 * inv_r3
            inv_r7 = inv_r3 * inv_r4

            # GTO-Ewald radial helpers at a and b; subtract.
            t0 = _gto_ewald_t0(distance, a_coef, b_coef)
            ra = _gto_ewald_A_single(distance, a_coef)
            rb = _gto_ewald_A_single(distance, b_coef)
            a_scalar = ra[0] - rb[0]
            a_prime = ra[1] - rb[1]
            a_double_prime = ra[2] - rb[2]
            a_triple_prime = ra[3] - rb[3]

            c_diag = a_scalar * inv_r
            c_quad = a_prime * inv_r2 - a_scalar * inv_r3
            c3 = (
                a_double_prime * inv_r3
                - wp.float64(3.0) * a_prime * inv_r4
                + wp.float64(3.0) * a_scalar * inv_r5
            )
            c4 = (
                a_triple_prime * inv_r4
                - wp.float64(6.0) * a_double_prime * inv_r5
                + wp.float64(15.0) * a_prime * inv_r6
                - wp.float64(15.0) * a_scalar * inv_r7
            )

            mu_i_dot_r = mu_i[0] * r_vec[0] + mu_i[1] * r_vec[1] + mu_i[2] * r_vec[2]
            mu_j_dot_r = mu_j[0] * r_vec[0] + mu_j[1] * r_vec[1] + mu_j[2] * r_vec[2]
            mu_dot = mu_i[0] * mu_j[0] + mu_i[1] * mu_j[1] + mu_i[2] * mu_j[2]
            gp_i_dot_r = gp_i[0] * r_vec[0] + gp_i[1] * r_vec[1] + gp_i[2] * r_vec[2]
            gd_i_dot_r = gd_i[0] * r_vec[0] + gd_i[1] * r_vec[1] + gd_i[2] * r_vec[2]
            gp_i_dot_mu_i = gp_i[0] * mu_i[0] + gp_i[1] * mu_i[1] + gp_i[2] * mu_i[2]
            gp_i_dot_mu_j = gp_i[0] * mu_j[0] + gp_i[1] * mu_j[1] + gp_i[2] * mu_j[2]
            gd_i_dot_mu_j = gd_i[0] * mu_j[0] + gd_i[1] * mu_j[1] + gd_i[2] * mu_j[2]
            dqmu_dot_r = qi * mu_j_dot_r - qj * mu_i_dot_r
            gp_i_dot_dqmu = qi * gp_i_dot_mu_j - qj * gp_i_dot_mu_i

            rad = (
                -qi * qj * c_diag
                - c_quad * dqmu_dot_r
                + c_quad * mu_dot
                + c3 * mu_i_dot_r * mu_j_dot_r
            )

            omega_i = (
                gc_i * qj * t0
                - gc_i * c_diag * mu_j_dot_r
                + qj * c_diag * gd_i_dot_r
                + c_diag * gd_i_dot_mu_j
                + c_quad * mu_j_dot_r * gd_i_dot_r
                - rad * gp_i_dot_r
                + c_diag * gp_i_dot_dqmu
                - c_quad * mu_j_dot_r * gp_i_dot_mu_i
                - c_quad * mu_i_dot_r * gp_i_dot_mu_j
            )
            half_omega = wp.float64(0.5) * omega_i
            wp.atomic_add(gg_grad_energies_2nd, atom_i, half_omega)
            wp.atomic_add(gg_grad_energies_2nd, j, half_omega)

            half_ge = wp.float64(0.5) * (ge_i + ge_j)

            dOmega_dqi = (
                qj * c_diag + c_quad * mu_j_dot_r
            ) * gp_i_dot_r + c_diag * gp_i_dot_mu_j
            wp.atomic_add(
                gg_charges_2nd,
                atom_i,
                type(charges[atom_i])(half_ge * dOmega_dqi),
            )

            dOmega_dqj = (
                (qi * c_diag - c_quad * mu_i_dot_r) * gp_i_dot_r
                - c_diag * gp_i_dot_mu_i
                + gc_i * t0
                + c_diag * gd_i_dot_r
            )
            wp.atomic_add(
                gg_charges_2nd, j, type(charges[atom_i])(half_ge * dOmega_dqj)
            )

            coeff_r_dmui = (
                -c_quad * qj * gp_i_dot_r
                - c3 * mu_j_dot_r * gp_i_dot_r
                - c_quad * gp_i_dot_mu_j
            )
            coeff_muj_dmui = -c_quad * gp_i_dot_r
            coeff_gpi_dmui = -c_diag * qj - c_quad * mu_j_dot_r
            dmui_x = (
                coeff_r_dmui * r_vec[0]
                + coeff_muj_dmui * mu_j[0]
                + coeff_gpi_dmui * gp_i[0]
            )
            dmui_y = (
                coeff_r_dmui * r_vec[1]
                + coeff_muj_dmui * mu_j[1]
                + coeff_gpi_dmui * gp_i[1]
            )
            dmui_z = (
                coeff_r_dmui * r_vec[2]
                + coeff_muj_dmui * mu_j[2]
                + coeff_gpi_dmui * gp_i[2]
            )
            wp.atomic_add(
                gg_dipoles_2nd,
                atom_i,
                type(mu_i_native)(
                    type(mu_i_native[0])(half_ge * dmui_x),
                    type(mu_i_native[0])(half_ge * dmui_y),
                    type(mu_i_native[0])(half_ge * dmui_z),
                ),
            )

            coeff_r_dmuj = (
                c_quad * qi * gp_i_dot_r
                - c3 * mu_i_dot_r * gp_i_dot_r
                - c_quad * gp_i_dot_mu_i
                - gc_i * c_diag
                + c_quad * gd_i_dot_r
            )
            coeff_mui_dmuj = -c_quad * gp_i_dot_r
            coeff_gpi_dmuj = c_diag * qi - c_quad * mu_i_dot_r
            coeff_gdi_dmuj = c_diag
            dmuj_x = (
                coeff_r_dmuj * r_vec[0]
                + coeff_mui_dmuj * mu_i[0]
                + coeff_gpi_dmuj * gp_i[0]
                + coeff_gdi_dmuj * gd_i[0]
            )
            dmuj_y = (
                coeff_r_dmuj * r_vec[1]
                + coeff_mui_dmuj * mu_i[1]
                + coeff_gpi_dmuj * gp_i[1]
                + coeff_gdi_dmuj * gd_i[1]
            )
            dmuj_z = (
                coeff_r_dmuj * r_vec[2]
                + coeff_mui_dmuj * mu_i[2]
                + coeff_gpi_dmuj * gp_i[2]
                + coeff_gdi_dmuj * gd_i[2]
            )
            wp.atomic_add(
                gg_dipoles_2nd,
                j,
                type(mu_i_native)(
                    type(mu_i_native[0])(half_ge * dmuj_x),
                    type(mu_i_native[0])(half_ge * dmuj_y),
                    type(mu_i_native[0])(half_ge * dmuj_z),
                ),
            )

            S_rad = (
                -gc_i * qj * c_diag
                - gc_i * c_quad * mu_j_dot_r
                + qj * c_quad * gd_i_dot_r
                + c_quad * gd_i_dot_mu_j
                + c3 * gd_i_dot_r * mu_j_dot_r
                + gp_i_dot_r
                * (
                    qi * qj * c_quad
                    + c3 * dqmu_dot_r
                    - c3 * mu_dot
                    - c4 * mu_i_dot_r * mu_j_dot_r
                )
                + c_quad * gp_i_dot_dqmu
                - c3 * gp_i_dot_mu_i * mu_j_dot_r
                - c3 * gp_i_dot_mu_j * mu_i_dot_r
            )
            coeff_muj_dr = (
                -gc_i * c_diag
                + c_quad * gd_i_dot_r
                - gp_i_dot_r * c3 * mu_i_dot_r
                - c_quad * gp_i_dot_mu_i
            )
            coeff_mui_dr = -gp_i_dot_r * c3 * mu_j_dot_r - c_quad * gp_i_dot_mu_j
            coeff_gdi_dr = qj * c_diag + c_quad * mu_j_dot_r
            coeff_gpi_dr = -rad
            coeff_dqmu_dr = gp_i_dot_r * c_quad
            dqmu_x = qi * mu_j[0] - qj * mu_i[0]
            dqmu_y = qi * mu_j[1] - qj * mu_i[1]
            dqmu_z = qi * mu_j[2] - qj * mu_i[2]
            G_pos_x = (
                S_rad * r_vec[0]
                + coeff_muj_dr * mu_j[0]
                + coeff_mui_dr * mu_i[0]
                + coeff_gdi_dr * gd_i[0]
                + coeff_gpi_dr * gp_i[0]
                + coeff_dqmu_dr * dqmu_x
            )
            G_pos_y = (
                S_rad * r_vec[1]
                + coeff_muj_dr * mu_j[1]
                + coeff_mui_dr * mu_i[1]
                + coeff_gdi_dr * gd_i[1]
                + coeff_gpi_dr * gp_i[1]
                + coeff_dqmu_dr * dqmu_y
            )
            G_pos_z = (
                S_rad * r_vec[2]
                + coeff_muj_dr * mu_j[2]
                + coeff_mui_dr * mu_i[2]
                + coeff_gdi_dr * gd_i[2]
                + coeff_gpi_dr * gp_i[2]
                + coeff_dqmu_dr * dqmu_z
            )
            px = half_ge * G_pos_x
            py = half_ge * G_pos_y
            pz = half_ge * G_pos_z
            wp.atomic_add(
                gg_positions_2nd,
                atom_i,
                type(pos_i)(
                    type(pos_i[0])(-px),
                    type(pos_i[0])(-py),
                    type(pos_i[0])(-pz),
                ),
            )
            wp.atomic_add(
                gg_positions_2nd,
                j,
                type(pos_i)(
                    type(pos_i[0])(px),
                    type(pos_i[0])(py),
                    type(pos_i[0])(pz),
                ),
            )


def _batch_dipole_csr_2nd_backward_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=m),  # cells
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigmas
        wp.array(dtype=t),  # alphas
        wp.array(dtype=wp.int32),  # batch_idx
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=v),  # gg_positions
        wp.array(dtype=t),  # gg_charges
        wp.array(dtype=v),  # gg_dipoles
        wp.array(dtype=wp.float64),  # gg_grad_energies_2nd
        wp.array(dtype=v),  # gg_positions_2nd
        wp.array(dtype=t),  # gg_charges_2nd
        wp.array(dtype=v),  # gg_dipoles_2nd
    ]


_batch_multipole_real_space_dipole_csr_energy_2nd_backward_overloads = (
    register_overloads(
        _batch_multipole_real_space_dipole_csr_energy_2nd_backward_kernel,
        _batch_dipole_csr_2nd_backward_sig,
    )
)


def batch_multipole_real_space_dipole_csr_energy_2nd_backward(
    positions,
    charges,
    dipoles,
    cells,
    idx_j,
    neighbor_ptr,
    unit_shifts,
    sigmas,
    alphas,
    batch_idx,
    grad_energies,
    gg_positions,
    gg_charges,
    gg_dipoles,
    gg_grad_energies_2nd,
    gg_positions_2nd,
    gg_charges_2nd,
    gg_dipoles_2nd,
    wp_dtype,
    device=None,
):
    r"""Launcher for the batched GTO-Ewald l_max=1 second-order backward kernel.

    Parameters
    ----------
    positions, charges, dipoles, cells, idx_j, neighbor_ptr, unit_shifts,
    sigmas, alphas, batch_idx :
        Same semantics as
        :func:`batch_multipole_real_space_dipole_csr_energy`.
    grad_energies : wp.array, shape (N_total,), dtype wp.float64
        Original first-order upstream cotangent ``∂L/∂pair_energies``.
    gg_positions : wp.array, shape (N_total,), dtype matching ``positions``
        Upstream cotangent on the first-order ``grad_positions``.
    gg_charges : wp.array, shape (N_total,), dtype matching ``charges``
        Upstream cotangent on ``grad_charges``.
    gg_dipoles : wp.array, shape (N_total,), dtype matching ``dipoles``
        Upstream cotangent on ``grad_dipoles``.
    gg_grad_energies_2nd : wp.array, shape (N_total,), dtype wp.float64
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂grad_energies``.
    gg_positions_2nd : wp.array, shape (N_total,), dtype matching ``positions``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂positions``.
    gg_charges_2nd : wp.array, shape (N_total,), dtype matching ``charges``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂charges``.
    gg_dipoles_2nd : wp.array, shape (N_total,), dtype matching ``dipoles``
        OUTPUT (pre-zeroed). Second-order ``∂L'/∂dipoles``.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the overloaded variant.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)
    wp.launch(
        _batch_multipole_real_space_dipole_csr_energy_2nd_backward_overloads[vec_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            cells,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigmas,
            alphas,
            batch_idx,
            grad_energies,
            gg_positions,
            gg_charges,
            gg_dipoles,
            gg_grad_energies_2nd,
            gg_positions_2nd,
            gg_charges_2nd,
            gg_dipoles_2nd,
        ],
        device=device,
    )


# =============================================================================
# l_max = 2 — public launchers
# =============================================================================
# The symmetric quadrupoles array is stored as wp.mat33d (3x3, both triangles
# populated by caller).


def _quadrupole_grad_scratch(vec_dtype, wp_dtype, mat_dtype, device):
    """Allocate 1-element scratch buffers for the 4 LMAX=2 gradient slots.

    The energy-only launchers pass these to satisfy the unified 15-input
    signature; the kernel body's Python-time guards prevent writes.
    """
    return (
        wp.zeros(1, dtype=vec_dtype, device=device),  # grad_positions
        wp.zeros(1, dtype=wp_dtype, device=device),  # grad_charges
        wp.zeros(1, dtype=vec_dtype, device=device),  # grad_dipoles
        wp.zeros(1, dtype=mat_dtype, device=device),  # grad_quadrupoles
    )


def _quadrupole_grad_or_scratch(
    grad_positions,
    grad_charges,
    grad_dipoles,
    grad_quadrupoles,
    wpg,
    wcg,
    wdg,
    wqg,
    vec_dtype,
    wp_dtype,
    mat_dtype,
    device,
    num_atoms,
):
    """For each LMAX=2 grad slot return the user array (if its flag is True)
    or an N-element pre-zeroed scratch buffer (if False).

    LMAX=2 collapses the grad-flag matrix to a single (T,T,T,T) fused
    kernel that always writes to all 4 grad arrays via ``atomic_add``;
    un-flagged slots must therefore receive scratch arrays large enough
    to absorb the atomic writes.
    """
    if wpg:
        gp = grad_positions
    else:
        gp = wp.zeros(num_atoms, dtype=vec_dtype, device=device)
    if wcg:
        gq = grad_charges
    else:
        gq = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
    if wdg:
        gmu = grad_dipoles
    else:
        gmu = wp.zeros(num_atoms, dtype=vec_dtype, device=device)
    if wqg:
        gQ = grad_quadrupoles
    else:
        gQ = wp.zeros(num_atoms, dtype=mat_dtype, device=device)
    return gp, gq, gmu, gQ


def multipole_real_space_quadrupole_csr_energy(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    quadrupoles: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigma: wp.array,
    alpha: wp.array,
    pair_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """LMAX=2 CSR single-system energy launcher.

    Energy-only path: the kernel's ``any_grad`` branch is bypassed, so
    ``grad_energies`` is not read. The launcher allocates a 1-element
    scratch buffer to satisfy the kernel signature.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype wp.float32 or wp.float64
        Atomic charges.
    dipoles : wp.array, shape (N,), dtype matching ``positions``
        Cartesian dipole moments ``(x, y, z)``.
    quadrupoles : wp.array, shape (N,), dtype wp.mat33f or wp.mat33d
        Cartesian (traceless) quadrupole moment matrices.
    cell : wp.array, shape (1,), dtype wp.mat33f or wp.mat33d
        Lattice matrix.
    idx_j : wp.array, shape (M,), dtype wp.int32
        CSR neighbor target indices.
    neighbor_ptr : wp.array, shape (N+1,), dtype wp.int32
        CSR row pointers into ``idx_j`` / ``unit_shifts``.
    unit_shifts : wp.array, shape (M,), dtype wp.vec3i
        Per-edge periodic image shifts.
    sigma : wp.array, shape (1,), dtype matching ``charges``
        GTO smearing width.
    alpha : wp.array, shape (1,), dtype matching ``charges``
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype wp.float64
        OUTPUT (pre-zeroed). Per-atom accumulated energy.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64``.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    mat_dtype = wp.mat33d if wp_dtype == wp.float64 else wp.mat33f
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    _overload = _get_real_space_pair_overload(
        LMAX=2,
        storage="csr",
        is_batch=False,
        with_pos_grad=False,
        with_charge_grad=False,
        with_dipole_grad=False,
        with_quad_grad=False,
        with_cell_grad=False,
        vec_dtype=vec_dtype,
        scalar_dtype=wp_dtype,
    )
    gp, gq, gmu, gQ = _quadrupole_grad_scratch(vec_dtype, wp_dtype, mat_dtype, device)
    ge_scratch = wp.zeros(1, dtype=wp.float64, device=device)

    wp.launch(
        _overload,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            quadrupoles,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            ge_scratch,
            pair_energies,
            gp,
            gq,
            gmu,
            gQ,
        ],
        device=device,
    )


def multipole_real_space_quadrupole_csr_energy_fused(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    quadrupoles: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigma: wp.array,
    alpha: wp.array,
    grad_energies: wp.array,
    pair_energies: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_dipoles: wp.array,
    grad_quadrupoles: wp.array,
    *,
    with_pos_grad: bool,
    with_charge_grad: bool,
    with_dipole_grad: bool,
    with_quad_grad: bool,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Fused LMAX=2 CSR single-system launcher.

    Per-pair gradient scatters are weighted by
    ``w = 0.25 * (grad_energies[atom_i] + grad_energies[atom_j])``.
    Pass ``grad_energies = ones(N)`` to recover the uniform half-pair
    weighting (``w = 0.5``). Routes on the four ``with_*_grad`` flags;
    output slots are written iff their flag is ``True``.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype wp.float32 or wp.float64
        Atomic charges.
    dipoles : wp.array, shape (N,), dtype matching ``positions``
        Cartesian dipole moments ``(x, y, z)``.
    quadrupoles : wp.array, shape (N,), dtype wp.mat33f or wp.mat33d
        Cartesian (traceless) quadrupole moment matrices.
    cell : wp.array, shape (1,), dtype wp.mat33f or wp.mat33d
        Lattice matrix.
    idx_j : wp.array, shape (M,), dtype wp.int32
        CSR neighbor target indices.
    neighbor_ptr : wp.array, shape (N+1,), dtype wp.int32
        CSR row pointers into ``idx_j`` / ``unit_shifts``.
    unit_shifts : wp.array, shape (M,), dtype wp.vec3i
        Per-edge periodic image shifts.
    sigma : wp.array, shape (1,), dtype matching ``charges``
        GTO smearing width.
    alpha : wp.array, shape (1,), dtype matching ``charges``
        Ewald splitting parameter.
    grad_energies : wp.array, shape (N,), dtype wp.float64
        Upstream cotangent ``∂L/∂pair_energies`` used for the per-pair
        gradient weights (see above).
    pair_energies : wp.array, shape (N,), dtype wp.float64
        OUTPUT (pre-zeroed). Per-atom accumulated energy.
    grad_positions : wp.array, shape (N,), dtype matching ``positions``
        OUTPUT. Written only when ``with_pos_grad``.
    grad_charges : wp.array, shape (N,), dtype matching ``charges``
        OUTPUT. Written only when ``with_charge_grad``.
    grad_dipoles : wp.array, shape (N,), dtype matching ``dipoles``
        OUTPUT. Written only when ``with_dipole_grad``.
    grad_quadrupoles : wp.array, shape (N,), dtype matching ``quadrupoles``
        OUTPUT. Written only when ``with_quad_grad``.
    with_pos_grad, with_charge_grad, with_dipole_grad, with_quad_grad : bool
        Per-slot gradient emission flags (keyword-only).
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` (keyword-only).
    device : str, optional
        Warp device string. Defaults to ``positions.device`` (keyword-only).
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    mat_dtype = wp.mat33d if wp_dtype == wp.float64 else wp.mat33f
    gp, gq, gmu, gQ = _quadrupole_grad_or_scratch(
        grad_positions,
        grad_charges,
        grad_dipoles,
        grad_quadrupoles,
        with_pos_grad,
        with_charge_grad,
        with_dipole_grad,
        with_quad_grad,
        vec_dtype,
        wp_dtype,
        mat_dtype,
        device,
        num_atoms,
    )
    _overload = _get_real_space_pair_overload(
        LMAX=2,
        storage="csr",
        is_batch=False,
        with_pos_grad=True,
        with_charge_grad=True,
        with_dipole_grad=True,
        with_quad_grad=True,
        with_cell_grad=False,
        vec_dtype=vec_dtype,
        scalar_dtype=wp_dtype,
    )

    wp.launch(
        _overload,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            quadrupoles,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma,
            alpha,
            grad_energies,
            pair_energies,
            gp,
            gq,
            gmu,
            gQ,
        ],
        device=device,
    )


def batch_multipole_real_space_quadrupole_csr_energy(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    quadrupoles: wp.array,
    cells: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigmas: wp.array,
    alphas: wp.array,
    batch_idx: wp.array,
    pair_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """LMAX=2 CSR batched energy launcher.

    Energy-only path: ``grad_energies`` is not read by the kernel. The
    launcher allocates a 1-element scratch buffer to satisfy the kernel
    signature. Per-system ``cells[b]`` / ``sigmas[b]`` / ``alphas[b]``
    lookup via ``batch_idx[atom_i]``.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype wp.vec3f or wp.vec3d
        Concatenated atomic positions across all systems.
    charges : wp.array, shape (N_total,), dtype wp.float32 or wp.float64
        Concatenated atomic charges.
    dipoles : wp.array, shape (N_total,), dtype matching ``positions``
        Concatenated Cartesian dipole moments ``(x, y, z)``.
    quadrupoles : wp.array, shape (N_total,), dtype wp.mat33f or wp.mat33d
        Concatenated Cartesian (traceless) quadrupole moment matrices.
    cells : wp.array, shape (B,), dtype wp.mat33f or wp.mat33d
        Per-system lattice matrices.
    idx_j : wp.array, shape (M,), dtype wp.int32
        Flattened CSR neighbor target indices (global atom indices).
    neighbor_ptr : wp.array, shape (N_total+1,), dtype wp.int32
        CSR row pointers into ``idx_j`` / ``unit_shifts``.
    unit_shifts : wp.array, shape (M,), dtype wp.vec3i
        Per-edge periodic image shifts.
    sigmas : wp.array, shape (B,), dtype matching ``charges``
        Per-system GTO smearing widths.
    alphas : wp.array, shape (B,), dtype matching ``charges``
        Per-system Ewald splitting parameters.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        System index ``b`` for each atom.
    pair_energies : wp.array, shape (N_total,), dtype wp.float64
        OUTPUT (pre-zeroed). Per-atom accumulated energy.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64``.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    mat_dtype = wp.mat33d if wp_dtype == wp.float64 else wp.mat33f
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    _overload = _get_real_space_pair_overload(
        LMAX=2,
        storage="csr",
        is_batch=True,
        with_pos_grad=False,
        with_charge_grad=False,
        with_dipole_grad=False,
        with_quad_grad=False,
        with_cell_grad=False,
        vec_dtype=vec_dtype,
        scalar_dtype=wp_dtype,
    )
    gp, gq, gmu, gQ = _quadrupole_grad_scratch(vec_dtype, wp_dtype, mat_dtype, device)
    ge_scratch = wp.zeros(1, dtype=wp.float64, device=device)

    wp.launch(
        _overload,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            quadrupoles,
            cells,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigmas,
            alphas,
            batch_idx,
            ge_scratch,
            pair_energies,
            gp,
            gq,
            gmu,
            gQ,
        ],
        device=device,
    )


def batch_multipole_real_space_quadrupole_csr_energy_fused(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    quadrupoles: wp.array,
    cells: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    sigmas: wp.array,
    alphas: wp.array,
    batch_idx: wp.array,
    grad_energies: wp.array,
    pair_energies: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_dipoles: wp.array,
    grad_quadrupoles: wp.array,
    *,
    with_pos_grad: bool,
    with_charge_grad: bool,
    with_dipole_grad: bool,
    with_quad_grad: bool,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Fused LMAX=2 CSR batched launcher.

    Per-pair scatters weighted by
    ``w = 0.25 * (grad_energies[atom_i] + grad_energies[atom_j])``.
    Per-system ``cells[b]`` / ``sigmas[b]`` / ``alphas[b]`` lookup via
    ``batch_idx[atom_i]``. Routes on the four ``with_*_grad`` flags; output
    slots are written iff their flag is ``True``.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype wp.vec3f or wp.vec3d
        Concatenated atomic positions across all systems.
    charges : wp.array, shape (N_total,), dtype wp.float32 or wp.float64
        Concatenated atomic charges.
    dipoles : wp.array, shape (N_total,), dtype matching ``positions``
        Concatenated Cartesian dipole moments ``(x, y, z)``.
    quadrupoles : wp.array, shape (N_total,), dtype wp.mat33f or wp.mat33d
        Concatenated Cartesian (traceless) quadrupole moment matrices.
    cells : wp.array, shape (B,), dtype wp.mat33f or wp.mat33d
        Per-system lattice matrices.
    idx_j : wp.array, shape (M,), dtype wp.int32
        Flattened CSR neighbor target indices (global atom indices).
    neighbor_ptr : wp.array, shape (N_total+1,), dtype wp.int32
        CSR row pointers into ``idx_j`` / ``unit_shifts``.
    unit_shifts : wp.array, shape (M,), dtype wp.vec3i
        Per-edge periodic image shifts.
    sigmas : wp.array, shape (B,), dtype matching ``charges``
        Per-system GTO smearing widths.
    alphas : wp.array, shape (B,), dtype matching ``charges``
        Per-system Ewald splitting parameters.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        System index ``b`` for each atom.
    grad_energies : wp.array, shape (N_total,), dtype wp.float64
        Upstream cotangent ``∂L/∂pair_energies`` used for the per-pair
        gradient weights (see above).
    pair_energies : wp.array, shape (N_total,), dtype wp.float64
        OUTPUT (pre-zeroed). Per-atom accumulated energy.
    grad_positions : wp.array, shape (N_total,), dtype matching ``positions``
        OUTPUT. Written only when ``with_pos_grad``.
    grad_charges : wp.array, shape (N_total,), dtype matching ``charges``
        OUTPUT. Written only when ``with_charge_grad``.
    grad_dipoles : wp.array, shape (N_total,), dtype matching ``dipoles``
        OUTPUT. Written only when ``with_dipole_grad``.
    grad_quadrupoles : wp.array, shape (N_total,), dtype matching ``quadrupoles``
        OUTPUT. Written only when ``with_quad_grad``.
    with_pos_grad, with_charge_grad, with_dipole_grad, with_quad_grad : bool
        Per-slot gradient emission flags (keyword-only).
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` (keyword-only).
    device : str, optional
        Warp device string. Defaults to ``positions.device`` (keyword-only).
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    mat_dtype = wp.mat33d if wp_dtype == wp.float64 else wp.mat33f
    gp, gq, gmu, gQ = _quadrupole_grad_or_scratch(
        grad_positions,
        grad_charges,
        grad_dipoles,
        grad_quadrupoles,
        with_pos_grad,
        with_charge_grad,
        with_dipole_grad,
        with_quad_grad,
        vec_dtype,
        wp_dtype,
        mat_dtype,
        device,
        num_atoms,
    )
    _overload = _get_real_space_pair_overload(
        LMAX=2,
        storage="csr",
        is_batch=True,
        with_pos_grad=True,
        with_charge_grad=True,
        with_dipole_grad=True,
        with_quad_grad=True,
        with_cell_grad=False,
        vec_dtype=vec_dtype,
        scalar_dtype=wp_dtype,
    )

    wp.launch(
        _overload,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            quadrupoles,
            cells,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigmas,
            alphas,
            batch_idx,
            grad_energies,
            pair_energies,
            gp,
            gq,
            gmu,
            gQ,
        ],
        device=device,
    )
