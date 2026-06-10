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

from __future__ import annotations

from typing import Any

import warp as wp

from nvalchemiops.interactions.electrostatics.multipole_ewald_kernels import (
    _gto_ewald_ab,
)
from nvalchemiops.warp_dispatch import register_overloads

_TWO_OVER_SQRT_PI = wp.float64(1.1283791670955126)


@wp.struct
class _QuadrupoleSecondOrderContrib:
    """Per-pair :math:`l_{max}=2` second-order contribution: :math:`\\Omega` + 8 partial slots."""

    omega: wp.float64
    dw_dq_i: wp.float64
    dw_dq_j: wp.float64
    dw_dmu_i: wp.vec3d
    dw_dmu_j: wp.vec3d
    dw_dQ_i: wp.mat33d
    dw_dQ_j: wp.mat33d
    dw_dr_vec: wp.vec3d


@wp.func
def _quadrupole_compute_T_radials(
    distance: wp.float64,
    a_coef: wp.float64,
    b_coef: wp.float64,
):
    """Compute T0..T6 (Python-convention radials) at one pair."""
    r = distance
    a = a_coef
    b = b_coef
    inv_r = wp.float64(1.0) / r
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r * inv_r2
    inv_r4 = inv_r2 * inv_r2
    inv_r5 = inv_r2 * inv_r3
    inv_r6 = inv_r3 * inv_r3
    inv_r7 = inv_r3 * inv_r4
    two_isp = _TWO_OVER_SQRT_PI

    a2 = a * a
    a3 = a * a2
    a5 = a2 * a3
    a7 = a2 * a5
    a9 = a2 * a7
    a11 = a2 * a9
    b2 = b * b
    b3 = b * b2
    b5 = b2 * b3
    b7 = b2 * b5
    b9 = b2 * b7
    b11 = b2 * b9

    ar = a * r
    br = b * r
    erf_ar = wp.float64(1.0) - wp.erfc(ar)
    erf_br = wp.float64(1.0) - wp.erfc(br)
    exp_ar = wp.exp(-ar * ar)
    exp_br = wp.exp(-br * br)
    erfc_ar = wp.erfc(ar)
    erfc_br = wp.erfc(br)

    r2 = r * r

    T0_v = (erfc_br - erfc_ar) * inv_r

    a_a_0 = erf_ar * inv_r2 - two_isp * a * exp_ar * inv_r
    a_b_0 = erf_br * inv_r2 - two_isp * b * exp_br * inv_r
    T1_v = -(a_a_0 - a_b_0)

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
    T2_v = -(a_a_1 - a_b_1)

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
    T3_v = -(a_a_2 - a_b_2)

    a_a_3 = (
        -wp.float64(24.0) * erf_ar * inv_r5
        + wp.float64(24.0) * two_isp * a * exp_ar * inv_r4
        + wp.float64(16.0) * two_isp * a3 * exp_ar * inv_r2
        + wp.float64(4.0) * two_isp * a5 * exp_ar
        + wp.float64(8.0) * two_isp * a7 * r2 * exp_ar
    )
    a_b_3 = (
        -wp.float64(24.0) * erf_br * inv_r5
        + wp.float64(24.0) * two_isp * b * exp_br * inv_r4
        + wp.float64(16.0) * two_isp * b3 * exp_br * inv_r2
        + wp.float64(4.0) * two_isp * b5 * exp_br
        + wp.float64(8.0) * two_isp * b7 * r2 * exp_br
    )
    T4_v = -(a_a_3 - a_b_3)

    a_a_4 = (
        wp.float64(120.0) * erf_ar * inv_r6
        - wp.float64(120.0) * two_isp * a * exp_ar * inv_r5
        - wp.float64(80.0) * two_isp * a3 * exp_ar * inv_r3
        - wp.float64(32.0) * two_isp * a5 * exp_ar * inv_r
        + wp.float64(8.0) * two_isp * a7 * r * exp_ar
        - wp.float64(16.0) * two_isp * a9 * r * r2 * exp_ar
    )
    a_b_4 = (
        wp.float64(120.0) * erf_br * inv_r6
        - wp.float64(120.0) * two_isp * b * exp_br * inv_r5
        - wp.float64(80.0) * two_isp * b3 * exp_br * inv_r3
        - wp.float64(32.0) * two_isp * b5 * exp_br * inv_r
        + wp.float64(8.0) * two_isp * b7 * r * exp_br
        - wp.float64(16.0) * two_isp * b9 * r * r2 * exp_br
    )
    T5_v = -(a_a_4 - a_b_4)

    # T6: 5th radial derivative.
    r4 = r2 * r2
    a_a_5 = (
        -wp.float64(720.0) * erf_ar * inv_r7
        + (
            wp.float64(720.0) * two_isp * a * inv_r6
            + wp.float64(480.0) * two_isp * a3 * inv_r4
            + wp.float64(192.0) * two_isp * a5 * inv_r2
            + wp.float64(72.0) * two_isp * a7
            - wp.float64(64.0) * two_isp * a9 * r2
            + wp.float64(32.0) * two_isp * a11 * r4
        )
        * exp_ar
    )
    a_b_5 = (
        -wp.float64(720.0) * erf_br * inv_r7
        + (
            wp.float64(720.0) * two_isp * b * inv_r6
            + wp.float64(480.0) * two_isp * b3 * inv_r4
            + wp.float64(192.0) * two_isp * b5 * inv_r2
            + wp.float64(72.0) * two_isp * b7
            - wp.float64(64.0) * two_isp * b9 * r2
            + wp.float64(32.0) * two_isp * b11 * r4
        )
        * exp_br
    )
    T6_v = -(a_a_5 - a_b_5)

    return T0_v, T1_v, T2_v, T3_v, T4_v, T5_v, T6_v


@wp.func
def _sym_outer(u: wp.vec3d, v: wp.vec3d) -> wp.mat33d:
    """½·(u⊗v + v⊗u) — symmetric outer (3×3)."""
    half = wp.float64(0.5)
    s01 = half * (u[0] * v[1] + u[1] * v[0])
    s02 = half * (u[0] * v[2] + u[2] * v[0])
    s12 = half * (u[1] * v[2] + u[2] * v[1])
    return wp.mat33d(
        u[0] * v[0],
        s01,
        s02,
        s01,
        u[1] * v[1],
        s12,
        s02,
        s12,
        u[2] * v[2],
    )


@wp.func
def _quadrupole_2nd_order_pair_contribution(
    r_vec: wp.vec3d,
    distance: wp.float64,
    qi: wp.float64,
    mu_i: wp.vec3d,
    Q_i: wp.mat33d,
    qj: wp.float64,
    mu_j: wp.vec3d,
    Q_j: wp.mat33d,
    gp_i: wp.vec3d,
    gd_i: wp.vec3d,
    gQ_i: wp.mat33d,
    gc_i: wp.float64,
    gp_j: wp.vec3d,
    gd_j: wp.vec3d,
    gQ_j: wp.mat33d,
    gc_j: wp.float64,
    a_coef: wp.float64,
    b_coef: wp.float64,
) -> _QuadrupoleSecondOrderContrib:
    r"""Channel-decomposed :math:`l_{max}=2` second-order pair contribution.

    Decomposed into the 6 physics channels (qq, qmu, mumu, qQ, muQ, QQ); returns
    the per-pair :math:`\Omega` scalar plus the 8 partial-derivative slots.

    Parameters
    ----------
    r_vec : wp.vec3d
        Minimum-image separation vector :math:`r_j - r_i + \text{shift}`.
    distance : wp.float64
        :math:`|r_{vec}|`.
    qi, qj : wp.float64
        Charges of atoms ``i`` and ``j``.
    mu_i, mu_j : wp.vec3d
        Dipole moments of atoms ``i`` and ``j``.
    Q_i, Q_j : wp.mat33d
        Symmetric Cartesian quadrupole tensors of atoms ``i`` and ``j``.
    gp_i, gp_j : wp.vec3d
        Upstream cotangents w.r.t. the position gradient (atoms ``i``, ``j``).
    gd_i, gd_j : wp.vec3d
        Upstream cotangents w.r.t. the dipole gradient.
    gQ_i, gQ_j : wp.mat33d
        Upstream cotangents w.r.t. the quadrupole gradient (assumed symmetric).
    gc_i, gc_j : wp.float64
        Upstream cotangents w.r.t. the charge gradient.
    a_coef, b_coef : wp.float64
        GTO-Ewald radial split coefficients from :func:`_gto_ewald_ab`.

    Returns
    -------
    _QuadrupoleSecondOrderContrib
        Per-pair :math:`\Omega` scalar and the 8 partial-derivative slots
        (``dw_dq_{i,j}``, ``dw_dmu_{i,j}``, ``dw_dQ_{i,j}``, ``dw_dr_vec``).
    """
    # Radial preamble: T0..T6 + A/B/A'/B'/A''/B''/A'''/B''' + K1..K3 (+ 1st, 2nd r-derivs).
    T0, T1, T2, T3, T4, T5, T6 = _quadrupole_compute_T_radials(distance, a_coef, b_coef)

    ONE = wp.float64(1.0)
    TWO = wp.float64(2.0)
    THREE = wp.float64(3.0)
    FOUR = wp.float64(4.0)
    FIVE = wp.float64(5.0)
    SIX = wp.float64(6.0)
    NINE = wp.float64(9.0)
    TWELVE = wp.float64(12.0)
    FIFTEEN = wp.float64(15.0)
    SEVENTEEN = wp.float64(17.0)
    TWENTY_ONE = wp.float64(21.0)
    TWENTY_SEVEN = wp.float64(27.0)
    THIRTY_SIX = wp.float64(36.0)
    FORTY_FIVE = wp.float64(45.0)
    EIGHTY_SEVEN = wp.float64(87.0)
    ONE_EIGHTY = wp.float64(180.0)
    HALF = wp.float64(0.5)
    QUARTER = wp.float64(0.25)
    ZERO = wp.float64(0.0)

    inv_r = ONE / distance
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r * inv_r2
    inv_r4 = inv_r2 * inv_r2
    inv_r5 = inv_r * inv_r4

    A = T1 * inv_r
    B = T2 - T1 * inv_r
    A_prime = T2 * inv_r - T1 * inv_r2
    B_prime = T3 - T2 * inv_r + T1 * inv_r2
    A_2prime = T3 * inv_r - TWO * T2 * inv_r2 + TWO * T1 * inv_r3
    B_2prime = T4 - T3 * inv_r + TWO * T2 * inv_r2 - TWO * T1 * inv_r3
    A_3prime = T4 * inv_r - THREE * T3 * inv_r2 + SIX * T2 * inv_r3 - SIX * T1 * inv_r4
    B_3prime = (
        T5 - T4 * inv_r + THREE * T3 * inv_r2 - SIX * T2 * inv_r3 + SIX * T1 * inv_r4
    )

    K1 = T2 * inv_r2 - T1 * inv_r3
    K2 = T3 * inv_r - THREE * T2 * inv_r2 + THREE * T1 * inv_r3
    K3 = T4 - SIX * T3 * inv_r + FIFTEEN * T2 * inv_r2 - FIFTEEN * T1 * inv_r3
    K1_prime = T3 * inv_r2 - THREE * T2 * inv_r3 + THREE * T1 * inv_r4
    K2_prime = T4 * inv_r - FOUR * T3 * inv_r2 + NINE * T2 * inv_r3 - NINE * T1 * inv_r4
    K3_prime = (
        T5
        - SIX * T4 * inv_r
        + TWENTY_ONE * T3 * inv_r2
        - FORTY_FIVE * T2 * inv_r3
        + FORTY_FIVE * T1 * inv_r4
    )
    K1_2prime = (
        T4 * inv_r2 - FIVE * T3 * inv_r3 + TWELVE * T2 * inv_r4 - TWELVE * T1 * inv_r5
    )
    K2_2prime = (
        T5 * inv_r
        - FIVE * T4 * inv_r2
        + SEVENTEEN * T3 * inv_r3
        - THIRTY_SIX * T2 * inv_r4
        + THIRTY_SIX * T1 * inv_r5
    )
    K3_2prime = (
        T6
        - SIX * T5 * inv_r
        + TWENTY_SEVEN * T4 * inv_r2
        - EIGHTY_SEVEN * T3 * inv_r3
        + ONE_EIGHTY * T2 * inv_r4
        - ONE_EIGHTY * T1 * inv_r5
    )

    BoverR = B * inv_r  # ≡ A_prime
    Bp_m_2BoR = B_prime - TWO * BoverR
    C1 = A_prime + BoverR  # = 2·A_prime
    C1_prime = TWO * A_2prime
    C1_2prime = TWO * A_3prime

    # Geometry preamble.
    rhat = r_vec * inv_r

    mu_i_r = wp.dot(mu_i, r_vec)
    mu_j_r = wp.dot(mu_j, r_vec)
    mu_i_rh = mu_i_r * inv_r
    mu_j_rh = mu_j_r * inv_r
    mu_i_mu_j = wp.dot(mu_i, mu_j)

    Qi_r = Q_i @ r_vec
    Qj_r = Q_j @ r_vec
    Qi_rh = Qi_r * inv_r
    Qj_rh = Qj_r * inv_r
    tr_Qi = Q_i[0, 0] + Q_i[1, 1] + Q_i[2, 2]
    tr_Qj = Q_j[0, 0] + Q_j[1, 1] + Q_j[2, 2]
    rhat_Qi_rhat = wp.dot(r_vec, Qi_r) * inv_r2
    rhat_Qj_rhat = wp.dot(r_vec, Qj_r) * inv_r2

    # μi · Qj   (Qj symmetric → μi @ Qj equals Qj @ μi as vec3)
    mu_i_Qj = Q_j @ mu_i
    mu_j_Qi = Q_i @ mu_j
    mu_i_Qj_r = wp.dot(mu_i_Qj, r_vec)
    mu_j_Qi_r = wp.dot(mu_j_Qi, r_vec)
    mu_i_Qj_rh = mu_i_Qj_r * inv_r
    mu_j_Qi_rh = mu_j_Qi_r * inv_r

    # Q·Q contractions.
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
    QiQj_mat = Q_i @ Q_j
    QjQi_mat = Q_j @ Q_i
    QiQj_r = QiQj_mat @ r_vec
    QjQi_r = QjQi_mat @ r_vec
    QiQj_rh = QiQj_r * inv_r
    QjQi_rh = QjQi_r * inv_r
    rhat_QiQj_rhat = wp.dot(r_vec, QiQj_r) * inv_r2

    # Upstream-gradient intermediates.
    gpd = gp_j - gp_i
    gpd_rh = wp.dot(gpd, rhat)
    gd_i_rh = wp.dot(gd_i, rhat)
    gd_j_rh = wp.dot(gd_j, rhat)

    gpd_mui = wp.dot(gpd, mu_i)
    gpd_muj = wp.dot(gpd, mu_j)
    gpd_Qi_rh = wp.dot(gpd, Qi_rh)
    gpd_Qj_rh = wp.dot(gpd, Qj_rh)

    Qi_gpd = Q_i @ gpd
    Qj_gpd = Q_j @ gpd

    mu_i_Qj_gpd = wp.dot(mu_i, Qj_gpd)
    mu_j_Qi_gpd = wp.dot(mu_j, Qi_gpd)

    gd_i_Qj = Q_j @ gd_i
    gd_j_Qi = Q_i @ gd_j
    gd_i_Qj_rh = wp.dot(gd_i_Qj, r_vec) * inv_r
    gd_j_Qi_rh = wp.dot(gd_j_Qi, r_vec) * inv_r

    gpd_QiQj_rh = wp.dot(gpd, QiQj_rh)
    gpd_QjQi_rh = wp.dot(gpd, QjQi_rh)

    # gQ-derived (kernel contract: gQ symmetric).
    tr_gQi = gQ_i[0, 0] + gQ_i[1, 1] + gQ_i[2, 2]
    tr_gQj = gQ_j[0, 0] + gQ_j[1, 1] + gQ_j[2, 2]
    gQi_r = gQ_i @ r_vec
    gQj_r = gQ_j @ r_vec
    gQi_rh = gQi_r * inv_r
    gQj_rh = gQj_r * inv_r
    rhat_gQi_rhat = wp.dot(r_vec, gQi_r) * inv_r2
    rhat_gQj_rhat = wp.dot(r_vec, gQj_r) * inv_r2

    # μ × gQ (used in μQ channel).
    mu_j_gQi = gQ_i @ mu_j
    mu_i_gQj = gQ_j @ mu_i
    mu_j_gQi_rh = wp.dot(mu_j_gQi, r_vec) * inv_r
    mu_i_gQj_rh = wp.dot(mu_i_gQj, r_vec) * inv_r

    # gQ·Q contractions (used in QQ channel).
    gQi_Qj_dd = (
        gQ_i[0, 0] * Q_j[0, 0]
        + gQ_i[0, 1] * Q_j[0, 1]
        + gQ_i[0, 2] * Q_j[0, 2]
        + gQ_i[1, 0] * Q_j[1, 0]
        + gQ_i[1, 1] * Q_j[1, 1]
        + gQ_i[1, 2] * Q_j[1, 2]
        + gQ_i[2, 0] * Q_j[2, 0]
        + gQ_i[2, 1] * Q_j[2, 1]
        + gQ_i[2, 2] * Q_j[2, 2]
    )
    gQj_Qi_dd = (
        gQ_j[0, 0] * Q_i[0, 0]
        + gQ_j[0, 1] * Q_i[0, 1]
        + gQ_j[0, 2] * Q_i[0, 2]
        + gQ_j[1, 0] * Q_i[1, 0]
        + gQ_j[1, 1] * Q_i[1, 1]
        + gQ_j[1, 2] * Q_i[1, 2]
        + gQ_j[2, 0] * Q_i[2, 0]
        + gQ_j[2, 1] * Q_i[2, 1]
        + gQ_j[2, 2] * Q_i[2, 2]
    )
    gQi_rh_Qj_rh = wp.dot(gQi_rh, Qj_rh)
    gQj_rh_Qi_rh = wp.dot(gQj_rh, Qi_rh)

    # Helpers used by ∂U_X/∂r_β (gQ_X·Q_Y·r̂ and Q_Y·gQ_X·r̂ vec3s).
    gQi_Qj_rh_vec = gQ_i @ Qj_rh
    Qj_gQi_rh_vec = Q_j @ gQi_rh
    gQj_Qi_rh_vec = gQ_j @ Qi_rh
    Qi_gQj_rh_vec = Q_i @ gQj_rh

    # Perpendicular projections: ∂(v·r̂)/∂r_β = (v_β - (v·r̂)·r̂_β)/r.
    p_gpd = (gpd - gpd_rh * rhat) * inv_r
    p_mui = (mu_i - mu_i_rh * rhat) * inv_r
    p_muj = (mu_j - mu_j_rh * rhat) * inv_r
    p_gdi = (gd_i - gd_i_rh * rhat) * inv_r
    p_gdj = (gd_j - gd_j_rh * rhat) * inv_r
    p_mu_i_Qj = (mu_i_Qj - mu_i_Qj_rh * rhat) * inv_r
    p_mu_j_Qi = (mu_j_Qi - mu_j_Qi_rh * rhat) * inv_r
    p_mu_j_gQi = (mu_j_gQi - mu_j_gQi_rh * rhat) * inv_r
    p_mu_i_gQj = (mu_i_gQj - mu_i_gQj_rh * rhat) * inv_r
    p_gd_i_Qj = (gd_i_Qj - gd_i_Qj_rh * rhat) * inv_r
    p_gd_j_Qi = (gd_j_Qi - gd_j_Qi_rh * rhat) * inv_r

    # ∂(rhat_QX_rhat)/∂r_β = (2/r)·(QX_rh - rhat_QX_rhat·r̂).
    d_rhQir = (Qi_rh - rhat_Qi_rhat * rhat) * (TWO * inv_r)
    d_rhQjr = (Qj_rh - rhat_Qj_rhat * rhat) * (TWO * inv_r)
    d_rhgQir = (gQi_rh - rhat_gQi_rhat * rhat) * (TWO * inv_r)
    d_rhgQjr = (gQj_rh - rhat_gQj_rhat * rhat) * (TWO * inv_r)

    # ∂(rhat_QiQj_rhat)/∂r_β = inv_r·(QiQj_rh + QjQi_rh - 2·r̂·rhat_QiQj_rhat).
    d_rhQiQjr = (QiQj_rh + QjQi_rh - TWO * rhat_QiQj_rhat * rhat) * inv_r

    I3 = wp.mat33d(ONE, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ONE)
    rhrh = wp.outer(rhat, rhat)

    # Channel qq: E_qq = qi·qj·T0.
    omega_qq = qi * qj * T1 * gpd_rh + T0 * (gc_i * qj + gc_j * qi)
    dwdqi_qq = qj * T1 * gpd_rh + T0 * gc_j
    dwdqj_qq = qi * T1 * gpd_rh + T0 * gc_i
    dwdr_qq = (gc_i * qj + gc_j * qi) * T1 * rhat + qi * qj * (
        B * gpd_rh * rhat + A * gpd
    )

    # Channel qmu: E_qmu = T1·(qi·mu_j_rh - qj·mu_i_rh), with S = qi·mu_j - qj·mu_i.
    S_qmu = qi * mu_j - qj * mu_i
    S_qmu_rh = qi * mu_j_rh - qj * mu_i_rh

    omega_qmu = (
        T1 * (gc_i * mu_j_rh - gc_j * mu_i_rh)
        + T1 * (qi * gd_j_rh - qj * gd_i_rh)
        + A * wp.dot(gpd, S_qmu)
        + B * S_qmu_rh * gpd_rh
    )

    dwdqi_qmu = T1 * gd_j_rh + A * wp.dot(gpd, mu_j) + B * mu_j_rh * gpd_rh
    dwdqj_qmu = -T1 * gd_i_rh - A * wp.dot(gpd, mu_i) - B * mu_i_rh * gpd_rh

    dwdmu_i_qmu = -gc_j * T1 * rhat - qj * (A * gpd + B * gpd_rh * rhat)
    dwdmu_j_qmu = gc_i * T1 * rhat + qi * (A * gpd + B * gpd_rh * rhat)

    c1_qmu = gc_i * mu_j_rh - gc_j * mu_i_rh
    c2_qmu = qi * gd_j_rh - qj * gd_i_rh
    c3_qmu = wp.dot(gpd, S_qmu)
    c4_qmu = S_qmu_rh * gpd_rh

    dc1_qmu = inv_r * (gc_i * (mu_j - mu_j_rh * rhat) - gc_j * (mu_i - mu_i_rh * rhat))
    dc2_qmu = inv_r * (qi * (gd_j - gd_j_rh * rhat) - qj * (gd_i - gd_i_rh * rhat))
    dc4_qmu = inv_r * (S_qmu * gpd_rh + gpd * S_qmu_rh - TWO * rhat * S_qmu_rh * gpd_rh)
    dwdr_qmu = (
        T2 * rhat * (c1_qmu + c2_qmu)
        + T1 * (dc1_qmu + dc2_qmu)
        + A_prime * rhat * c3_qmu
        + B_prime * rhat * c4_qmu
        + B * dc4_qmu
    )

    # Channel mumu: E_mumu = -A·(mu_i·mu_j) - B·mu_i_rh·mu_j_rh.
    d1_mumu = wp.dot(gd_i, mu_j)
    d2_mumu = wp.dot(gd_j, mu_i)
    E1_mumu = mu_j_rh * gd_i_rh + mu_i_rh * gd_j_rh
    E2_mumu = mu_i_rh * mu_j_rh
    E3_mumu = gpd_mui * mu_j_rh + gpd_muj * mu_i_rh - TWO * gpd_rh * E2_mumu

    omega_mumu = (
        -A * (d1_mumu + d2_mumu)
        - B * E1_mumu
        - A_prime * gpd_rh * mu_i_mu_j
        - B_prime * gpd_rh * E2_mumu
        - BoverR * E3_mumu
    )

    dwdmu_i_mumu = (
        -A * gd_j
        - B * rhat * gd_j_rh
        - A_prime * gpd_rh * mu_j
        - B_prime * gpd_rh * rhat * mu_j_rh
        - BoverR * (gpd * mu_j_rh + gpd_muj * rhat - TWO * gpd_rh * rhat * mu_j_rh)
    )
    dwdmu_j_mumu = (
        -A * gd_i
        - B * rhat * gd_i_rh
        - A_prime * gpd_rh * mu_i
        - B_prime * gpd_rh * rhat * mu_i_rh
        - BoverR * (gpd * mu_i_rh + gpd_mui * rhat - TWO * gpd_rh * rhat * mu_i_rh)
    )

    dE1_mumu = p_muj * gd_i_rh + mu_j_rh * p_gdi + p_mui * gd_j_rh + mu_i_rh * p_gdj
    dE2_mumu = p_mui * mu_j_rh + mu_i_rh * p_muj
    dE3_mumu = (
        gpd_mui * p_muj
        + gpd_muj * p_mui
        - TWO * (p_gpd * E2_mumu + gpd_rh * p_mui * mu_j_rh + gpd_rh * mu_i_rh * p_muj)
    )
    dwdr_mumu = (
        -A_prime * rhat * (d1_mumu + d2_mumu)
        - B_prime * rhat * E1_mumu
        - B * dE1_mumu
        - A_2prime * rhat * gpd_rh * mu_i_mu_j
        - A_prime * p_gpd * mu_i_mu_j
        - B_2prime * rhat * gpd_rh * E2_mumu
        - B_prime * (p_gpd * E2_mumu + gpd_rh * dE2_mumu)
        - A_2prime * rhat * E3_mumu  # d(B/r)/dr_b = A_2prime·rhat_b
        - BoverR * dE3_mumu
    )

    # Channel qQ: E_qQ = ½·qj·F_i + ½·qi·F_j, with F_X = (Q_X : T_ab).
    coef_gQ_i = B_prime * rhat_gQi_rhat + BoverR * (tr_gQi - TWO * rhat_gQi_rhat)
    coef_gQ_j = B_prime * rhat_gQj_rhat + BoverR * (tr_gQj - TWO * rhat_gQj_rhat)
    F_i_qQ = A * tr_Qi + B * rhat_Qi_rhat
    F_j_qQ = A * tr_Qj + B * rhat_Qj_rhat
    G_i_qQ = A * tr_gQi + B * rhat_gQi_rhat
    G_j_qQ = A * tr_gQj + B * rhat_gQj_rhat
    H_i_qQ = (
        A_prime * gpd_rh * tr_Qi
        + B_prime * gpd_rh * rhat_Qi_rhat
        + TWO * BoverR * (gpd_Qi_rh - gpd_rh * rhat_Qi_rhat)
    )
    H_j_qQ = (
        A_prime * gpd_rh * tr_Qj
        + B_prime * gpd_rh * rhat_Qj_rhat
        + TWO * BoverR * (gpd_Qj_rh - gpd_rh * rhat_Qj_rhat)
    )

    omega_qQ = HALF * (
        gc_i * F_j_qQ
        + gc_j * F_i_qQ
        + qj * G_i_qQ
        + qi * G_j_qQ
        + qj * H_i_qQ
        + qi * H_j_qQ
    )

    dwdqi_qQ = HALF * (G_j_qQ + H_j_qQ)
    dwdqj_qQ = HALF * (G_i_qQ + H_i_qQ)

    T_mat = A * I3 + B * rhrh
    gpd_rh_outer = wp.outer(gpd, rhat) + wp.outer(rhat, gpd)
    M_dQ_qQ = (
        A_prime * gpd_rh * I3
        + (B_prime - TWO * BoverR) * gpd_rh * rhrh
        + BoverR * gpd_rh_outer
    )
    dwdQ_i_qQ = HALF * gc_j * T_mat + HALF * qj * M_dQ_qQ
    dwdQ_j_qQ = HALF * gc_i * T_mat + HALF * qi * M_dQ_qQ

    # dF/dr_b = A'·rhat·tr_X + B'·rhat·rhat_X_rhat + (2B/r)·(X_rh - rhat_X_rhat·rhat)
    dF_i_qQ = (
        A_prime * rhat * tr_Qi
        + B_prime * rhat * rhat_Qi_rhat
        + TWO * BoverR * (Qi_rh - rhat_Qi_rhat * rhat)
    )
    dF_j_qQ = (
        A_prime * rhat * tr_Qj
        + B_prime * rhat * rhat_Qj_rhat
        + TWO * BoverR * (Qj_rh - rhat_Qj_rhat * rhat)
    )
    dG_i_qQ = (
        A_prime * rhat * tr_gQi
        + B_prime * rhat * rhat_gQi_rhat
        + TWO * BoverR * (gQi_rh - rhat_gQi_rhat * rhat)
    )
    dG_j_qQ = (
        A_prime * rhat * tr_gQj
        + B_prime * rhat * rhat_gQj_rhat
        + TWO * BoverR * (gQj_rh - rhat_gQj_rhat * rhat)
    )
    d_gpdQi = inv_r * (Qi_gpd - gpd_Qi_rh * rhat)
    d_gpdQj = inv_r * (Qj_gpd - gpd_Qj_rh * rhat)
    dH_i_qQ = (
        A_2prime * rhat * gpd_rh * tr_Qi
        + A_prime * p_gpd * tr_Qi
        + B_2prime * rhat * gpd_rh * rhat_Qi_rhat
        + B_prime * (p_gpd * rhat_Qi_rhat + gpd_rh * d_rhQir)
        + TWO * A_2prime * rhat * (gpd_Qi_rh - gpd_rh * rhat_Qi_rhat)
        + TWO * BoverR * (d_gpdQi - p_gpd * rhat_Qi_rhat - gpd_rh * d_rhQir)
    )
    dH_j_qQ = (
        A_2prime * rhat * gpd_rh * tr_Qj
        + A_prime * p_gpd * tr_Qj
        + B_2prime * rhat * gpd_rh * rhat_Qj_rhat
        + B_prime * (p_gpd * rhat_Qj_rhat + gpd_rh * d_rhQjr)
        + TWO * A_2prime * rhat * (gpd_Qj_rh - gpd_rh * rhat_Qj_rhat)
        + TWO * BoverR * (d_gpdQj - p_gpd * rhat_Qj_rhat - gpd_rh * d_rhQjr)
    )
    dwdr_qQ = HALF * (
        gc_i * dF_j_qQ
        + gc_j * dF_i_qQ
        + qj * dG_i_qQ
        + qi * dG_j_qQ
        + qj * dH_i_qQ
        + qi * dH_j_qQ
    )

    # Channel muQ: E_muQ = ½·(C1·(mu_i_Qj_rh - mu_j_Qi_rh) + mu_i_rh·coef_j - mu_j_rh·coef_i).
    chi_i = tr_Qi - TWO * rhat_Qi_rhat
    chi_j = tr_Qj - TWO * rhat_Qj_rhat
    coef_i = B_prime * rhat_Qi_rhat + BoverR * chi_i
    coef_j = B_prime * rhat_Qj_rhat + BoverR * chi_j

    gd_i_term_muQ = HALF * (C1 * gd_i_Qj_rh + gd_i_rh * coef_j)
    gd_j_term_muQ = -HALF * (C1 * gd_j_Qi_rh + gd_j_rh * coef_i)
    gQ_i_term_muQ = -HALF * (C1 * mu_j_gQi_rh + mu_j_rh * coef_gQ_i)
    gQ_j_term_muQ = HALF * (C1 * mu_i_gQj_rh + mu_i_rh * coef_gQ_j)

    g1_muQ = C1_prime * gpd_rh * (mu_i_Qj_rh - mu_j_Qi_rh)
    g2_muQ = (
        C1 * inv_r * (mu_i_Qj_gpd - mu_j_Qi_gpd - gpd_rh * (mu_i_Qj_rh - mu_j_Qi_rh))
    )
    g3_muQ = inv_r * (gpd_mui - gpd_rh * mu_i_rh) * coef_j
    gpd_dot_dcoef_j = (
        B_2prime * gpd_rh * rhat_Qj_rhat
        + A_2prime * gpd_rh * chi_j
        + Bp_m_2BoR * (TWO * inv_r) * (gpd_Qj_rh - gpd_rh * rhat_Qj_rhat)
    )
    gpd_dot_dcoef_i = (
        B_2prime * gpd_rh * rhat_Qi_rhat
        + A_2prime * gpd_rh * chi_i
        + Bp_m_2BoR * (TWO * inv_r) * (gpd_Qi_rh - gpd_rh * rhat_Qi_rhat)
    )
    g4_muQ = mu_i_rh * gpd_dot_dcoef_j
    g5_muQ = -inv_r * (gpd_muj - gpd_rh * mu_j_rh) * coef_i
    g6_muQ = -mu_j_rh * gpd_dot_dcoef_i
    gpd_term_muQ = HALF * (g1_muQ + g2_muQ + g3_muQ + g4_muQ + g5_muQ + g6_muQ)

    omega_muQ = (
        gd_i_term_muQ + gd_j_term_muQ + gQ_i_term_muQ + gQ_j_term_muQ + gpd_term_muQ
    )

    Qj_gQj_rh_vec = gQ_j @ rhat
    Qi_gQi_rh_vec = gQ_i @ rhat
    dwdmu_i_muQ = (
        HALF * (C1 * Qj_gQj_rh_vec + coef_gQ_j * rhat)
        + HALF * C1_prime * gpd_rh * Qj_rh
        + HALF * C1 * inv_r * (Qj_gpd - gpd_rh * Qj_rh)
        + HALF * inv_r * (gpd - gpd_rh * rhat) * coef_j
        + HALF * rhat * gpd_dot_dcoef_j
    )
    dwdmu_j_muQ = (
        -HALF * (C1 * Qi_gQi_rh_vec + coef_gQ_i * rhat)
        - HALF * C1_prime * gpd_rh * Qi_rh
        - HALF * C1 * inv_r * (Qi_gpd - gpd_rh * Qi_rh)
        - HALF * inv_r * (gpd - gpd_rh * rhat) * coef_i
        - HALF * rhat * gpd_dot_dcoef_i
    )

    M_coef_muQ = B_prime * rhrh + BoverR * (I3 - TWO * rhrh)
    M_dgpdcoef = (
        B_2prime * gpd_rh * rhrh
        + A_2prime * gpd_rh * (I3 - TWO * rhrh)
        + Bp_m_2BoR * (TWO * inv_r) * (_sym_outer(gpd, rhat) - gpd_rh * rhrh)
    )
    dwdQ_i_muQ = (
        -HALF * C1 * _sym_outer(gd_j, rhat)
        - HALF * gd_j_rh * M_coef_muQ
        - HALF * C1_prime * gpd_rh * _sym_outer(mu_j, rhat)
        - HALF * C1 * _sym_outer(mu_j, p_gpd)
        - HALF * inv_r * (gpd_muj - gpd_rh * mu_j_rh) * M_coef_muQ
        - HALF * mu_j_rh * M_dgpdcoef
    )
    dwdQ_j_muQ = (
        HALF * C1 * _sym_outer(gd_i, rhat)
        + HALF * gd_i_rh * M_coef_muQ
        + HALF * C1_prime * gpd_rh * _sym_outer(mu_i, rhat)
        + HALF * C1 * _sym_outer(mu_i, p_gpd)
        + HALF * inv_r * (gpd_mui - gpd_rh * mu_i_rh) * M_coef_muQ
        + HALF * mu_i_rh * M_dgpdcoef
    )

    def_dcoef_dr_i = (
        rhat * (B_2prime * rhat_Qi_rhat + A_2prime * chi_i) + Bp_m_2BoR * d_rhQir
    )
    def_dcoef_dr_j = (
        rhat * (B_2prime * rhat_Qj_rhat + A_2prime * chi_j) + Bp_m_2BoR * d_rhQjr
    )
    def_dcoef_gQ_dr_i = (
        rhat * (B_2prime * rhat_gQi_rhat + A_2prime * (tr_gQi - TWO * rhat_gQi_rhat))
        + Bp_m_2BoR * d_rhgQir
    )
    def_dcoef_gQ_dr_j = (
        rhat * (B_2prime * rhat_gQj_rhat + A_2prime * (tr_gQj - TWO * rhat_gQj_rhat))
        + Bp_m_2BoR * d_rhgQjr
    )

    d_gdi_term_muQ = HALF * (
        C1_prime * rhat * gd_i_Qj_rh
        + C1 * p_gd_i_Qj
        + p_gdi * coef_j
        + gd_i_rh * def_dcoef_dr_j
    )
    d_gdj_term_muQ = -HALF * (
        C1_prime * rhat * gd_j_Qi_rh
        + C1 * p_gd_j_Qi
        + p_gdj * coef_i
        + gd_j_rh * def_dcoef_dr_i
    )
    d_gQ_i_term_muQ = -HALF * (
        C1_prime * rhat * mu_j_gQi_rh
        + C1 * p_mu_j_gQi
        + p_muj * coef_gQ_i
        + mu_j_rh * def_dcoef_gQ_dr_i
    )
    d_gQ_j_term_muQ = HALF * (
        C1_prime * rhat * mu_i_gQj_rh
        + C1 * p_mu_i_gQj
        + p_mui * coef_gQ_j
        + mu_i_rh * def_dcoef_gQ_dr_j
    )

    dg1_muQ = (
        C1_2prime * rhat * gpd_rh * (mu_i_Qj_rh - mu_j_Qi_rh)
        + C1_prime * p_gpd * (mu_i_Qj_rh - mu_j_Qi_rh)
        + C1_prime * gpd_rh * (p_mu_i_Qj - p_mu_j_Qi)
    )
    inner_g2_muQ = mu_i_Qj_gpd - mu_j_Qi_gpd - gpd_rh * (mu_i_Qj_rh - mu_j_Qi_rh)
    d_inner_g2 = -p_gpd * (mu_i_Qj_rh - mu_j_Qi_rh) - gpd_rh * (p_mu_i_Qj - p_mu_j_Qi)
    dg2_muQ = (
        C1_prime * rhat * inv_r * inner_g2_muQ
        - C1 * inv_r * inv_r * rhat * inner_g2_muQ
        + C1 * inv_r * d_inner_g2
    )
    inner_g3_muQ = gpd_mui - gpd_rh * mu_i_rh
    d_inner_g3 = -p_gpd * mu_i_rh - gpd_rh * p_mui
    dg3_muQ = (
        -inv_r * inv_r * rhat * inner_g3_muQ * coef_j
        + inv_r * d_inner_g3 * coef_j
        + inv_r * inner_g3_muQ * def_dcoef_dr_j
    )
    p_gpdQjr = inv_r * (Qj_gpd - gpd_Qj_rh * rhat)
    p_gpdQir = inv_r * (Qi_gpd - gpd_Qi_rh * rhat)
    P1_j = (
        B_3prime * rhat * gpd_rh * rhat_Qj_rhat
        + B_2prime * p_gpd * rhat_Qj_rhat
        + B_2prime * gpd_rh * d_rhQjr
    )
    d_chi_j = -TWO * d_rhQjr
    P2_j = (
        A_3prime * rhat * gpd_rh * chi_j
        + A_2prime * p_gpd * chi_j
        + A_2prime * gpd_rh * d_chi_j
    )
    factor3_muQ = Bp_m_2BoR * (TWO * inv_r)
    d_factor3 = (B_2prime - TWO * A_2prime) * rhat * (TWO * inv_r) + Bp_m_2BoR * (
        -TWO * inv_r * inv_r
    ) * rhat
    inner3_j = gpd_Qj_rh - gpd_rh * rhat_Qj_rhat
    d_inner3_j = p_gpdQjr - p_gpd * rhat_Qj_rhat - gpd_rh * d_rhQjr
    P3_j = d_factor3 * inner3_j + factor3_muQ * d_inner3_j
    d_gpd_dot_dcoef_j = P1_j + P2_j + P3_j
    dg4_muQ = p_mui * gpd_dot_dcoef_j + mu_i_rh * d_gpd_dot_dcoef_j

    inner_g5_muQ = gpd_muj - gpd_rh * mu_j_rh
    d_inner_g5 = -p_gpd * mu_j_rh - gpd_rh * p_muj
    dg5_muQ = (
        inv_r * inv_r * rhat * inner_g5_muQ * coef_i
        - inv_r * d_inner_g5 * coef_i
        - inv_r * inner_g5_muQ * def_dcoef_dr_i
    )
    P1_i = (
        B_3prime * rhat * gpd_rh * rhat_Qi_rhat
        + B_2prime * p_gpd * rhat_Qi_rhat
        + B_2prime * gpd_rh * d_rhQir
    )
    d_chi_i = -TWO * d_rhQir
    P2_i = (
        A_3prime * rhat * gpd_rh * chi_i
        + A_2prime * p_gpd * chi_i
        + A_2prime * gpd_rh * d_chi_i
    )
    inner3_i = gpd_Qi_rh - gpd_rh * rhat_Qi_rhat
    d_inner3_i = p_gpdQir - p_gpd * rhat_Qi_rhat - gpd_rh * d_rhQir
    P3_i = d_factor3 * inner3_i + factor3_muQ * d_inner3_i
    d_gpd_dot_dcoef_i = P1_i + P2_i + P3_i
    dg6_muQ = -p_muj * gpd_dot_dcoef_i - mu_j_rh * d_gpd_dot_dcoef_i

    d_gpd_term_muQ = HALF * (dg1_muQ + dg2_muQ + dg3_muQ + dg4_muQ + dg5_muQ + dg6_muQ)
    dwdr_muQ = (
        d_gdi_term_muQ
        + d_gdj_term_muQ
        + d_gQ_i_term_muQ
        + d_gQ_j_term_muQ
        + d_gpd_term_muQ
    )

    # Channel QQ: E_QQ = ¼·[K1·S_0 + K2·S_2A + K3·S_4].
    U_ij_QQ = (
        K1 * (tr_gQi * tr_Qj + TWO * gQi_Qj_dd)
        + K2 * (tr_gQi * rhat_Qj_rhat + tr_Qj * rhat_gQi_rhat + FOUR * gQi_rh_Qj_rh)
        + K3 * rhat_gQi_rhat * rhat_Qj_rhat
    )
    U_ji_QQ = (
        K1 * (tr_gQj * tr_Qi + TWO * gQj_Qi_dd)
        + K2 * (tr_gQj * rhat_Qi_rhat + tr_Qi * rhat_gQj_rhat + FOUR * gQj_rh_Qi_rh)
        + K3 * rhat_gQj_rhat * rhat_Qi_rhat
    )

    S_0_QQ = tr_Qi * tr_Qj + TWO * Qi_Qj_dd
    S_2A_QQ = tr_Qi * rhat_Qj_rhat + tr_Qj * rhat_Qi_rhat + FOUR * rhat_QiQj_rhat
    S_4_QQ = rhat_Qi_rhat * rhat_Qj_rhat
    X_i_QQ = gpd_Qi_rh - gpd_rh * rhat_Qi_rhat
    X_j_QQ = gpd_Qj_rh - gpd_rh * rhat_Qj_rhat
    Y_QQ = gpd_QiQj_rh + gpd_QjQi_rh

    gpd_dS_2A = (
        TWO
        * inv_r
        * (
            tr_Qi * X_j_QQ
            + tr_Qj * X_i_QQ
            + TWO * Y_QQ
            - FOUR * gpd_rh * rhat_QiQj_rhat
        )
    )
    gpd_dS_4 = (
        TWO
        * inv_r
        * (
            gpd_Qi_rh * rhat_Qj_rhat
            + rhat_Qi_rhat * gpd_Qj_rh
            - TWO * gpd_rh * rhat_Qi_rhat * rhat_Qj_rhat
        )
    )

    V_inner_QQ = (
        K1_prime * gpd_rh * S_0_QQ
        + K2_prime * gpd_rh * S_2A_QQ
        + K2 * gpd_dS_2A
        + K3_prime * gpd_rh * S_4_QQ
        + K3 * gpd_dS_4
    )
    omega_QQ = QUARTER * (U_ij_QQ + U_ji_QQ + V_inner_QQ)

    dU_ji_dQi = (
        K1 * (tr_gQj * I3 + TWO * gQ_j)
        + K2 * (rhat_gQj_rhat * I3 + tr_gQj * rhrh + FOUR * _sym_outer(gQj_rh, rhat))
        + K3 * rhat_gQj_rhat * rhrh
    )
    dU_ij_dQj = (
        K1 * (tr_gQi * I3 + TWO * gQ_i)
        + K2 * (rhat_gQi_rhat * I3 + tr_gQi * rhrh + FOUR * _sym_outer(gQi_rh, rhat))
        + K3 * rhat_gQi_rhat * rhrh
    )

    dS0_dQi = tr_Qj * I3 + TWO * Q_j
    dS2A_dQi = rhat_Qj_rhat * I3 + tr_Qj * rhrh + FOUR * _sym_outer(rhat, Qj_rh)
    dS4_dQi = rhat_Qj_rhat * rhrh
    d_gpd_dS_2A_dQi = (TWO * inv_r) * (
        X_j_QQ * I3
        + tr_Qj * (_sym_outer(gpd, rhat) - gpd_rh * rhrh)
        + TWO * _sym_outer(gpd, Qj_rh)
        + TWO * _sym_outer(Qj_gpd, rhat)
        - FOUR * gpd_rh * _sym_outer(rhat, Qj_rh)
    )
    d_gpd_dS_4_dQi = (TWO * inv_r) * (
        rhat_Qj_rhat * _sym_outer(gpd, rhat)
        + gpd_Qj_rh * rhrh
        - TWO * gpd_rh * rhat_Qj_rhat * rhrh
    )
    dV_inner_dQi = (
        K1_prime * gpd_rh * dS0_dQi
        + K2_prime * gpd_rh * dS2A_dQi
        + K2 * d_gpd_dS_2A_dQi
        + K3_prime * gpd_rh * dS4_dQi
        + K3 * d_gpd_dS_4_dQi
    )
    dwdQ_i_QQ = QUARTER * (dU_ji_dQi + dV_inner_dQi)

    dS0_dQj = tr_Qi * I3 + TWO * Q_i
    dS2A_dQj = rhat_Qi_rhat * I3 + tr_Qi * rhrh + FOUR * _sym_outer(rhat, Qi_rh)
    dS4_dQj = rhat_Qi_rhat * rhrh
    d_gpd_dS_2A_dQj = (TWO * inv_r) * (
        X_i_QQ * I3
        + tr_Qi * (_sym_outer(gpd, rhat) - gpd_rh * rhrh)
        + TWO * _sym_outer(gpd, Qi_rh)
        + TWO * _sym_outer(Qi_gpd, rhat)
        - FOUR * gpd_rh * _sym_outer(rhat, Qi_rh)
    )
    d_gpd_dS_4_dQj = (TWO * inv_r) * (
        rhat_Qi_rhat * _sym_outer(gpd, rhat)
        + gpd_Qi_rh * rhrh
        - TWO * gpd_rh * rhat_Qi_rhat * rhrh
    )
    dV_inner_dQj = (
        K1_prime * gpd_rh * dS0_dQj
        + K2_prime * gpd_rh * dS2A_dQj
        + K2 * d_gpd_dS_2A_dQj
        + K3_prime * gpd_rh * dS4_dQj
        + K3 * d_gpd_dS_4_dQj
    )
    dwdQ_j_QQ = QUARTER * (dU_ij_dQj + dV_inner_dQj)

    d_gQi_rh_Qj_rh = inv_r * (gQi_Qj_rh_vec + Qj_gQi_rh_vec - TWO * gQi_rh_Qj_rh * rhat)
    d_gQj_rh_Qi_rh = inv_r * (gQj_Qi_rh_vec + Qi_gQj_rh_vec - TWO * gQj_rh_Qi_rh * rhat)

    d_U_ij_QQ = (
        K1_prime * rhat * (tr_gQi * tr_Qj + TWO * gQi_Qj_dd)
        + K2_prime
        * rhat
        * (tr_gQi * rhat_Qj_rhat + tr_Qj * rhat_gQi_rhat + FOUR * gQi_rh_Qj_rh)
        + K2 * (tr_gQi * d_rhQjr + tr_Qj * d_rhgQir + FOUR * d_gQi_rh_Qj_rh)
        + K3_prime * rhat * rhat_gQi_rhat * rhat_Qj_rhat
        + K3 * (d_rhgQir * rhat_Qj_rhat + rhat_gQi_rhat * d_rhQjr)
    )
    d_U_ji_QQ = (
        K1_prime * rhat * (tr_gQj * tr_Qi + TWO * gQj_Qi_dd)
        + K2_prime
        * rhat
        * (tr_gQj * rhat_Qi_rhat + tr_Qi * rhat_gQj_rhat + FOUR * gQj_rh_Qi_rh)
        + K2 * (tr_gQj * d_rhQir + tr_Qi * d_rhgQjr + FOUR * d_gQj_rh_Qi_rh)
        + K3_prime * rhat * rhat_gQj_rhat * rhat_Qi_rhat
        + K3 * (d_rhgQjr * rhat_Qi_rhat + rhat_gQj_rhat * d_rhQir)
    )

    d_S_2A = tr_Qi * d_rhQjr + tr_Qj * d_rhQir + FOUR * d_rhQiQjr
    d_S_4 = d_rhQir * rhat_Qj_rhat + rhat_Qi_rhat * d_rhQjr

    W_2A = tr_Qi * X_j_QQ + tr_Qj * X_i_QQ + TWO * Y_QQ - FOUR * gpd_rh * rhat_QiQj_rhat
    d_X_j_QQ = (
        inv_r * (Qj_gpd - gpd_Qj_rh * rhat) - p_gpd * rhat_Qj_rhat - gpd_rh * d_rhQjr
    )
    d_X_i_QQ = (
        inv_r * (Qi_gpd - gpd_Qi_rh * rhat) - p_gpd * rhat_Qi_rhat - gpd_rh * d_rhQir
    )
    gpd_QiQj_vec_warp = Q_j @ Qi_gpd
    gpd_QjQi_vec_warp = Q_i @ Qj_gpd
    d_gpd_QiQj_rh = inv_r * (gpd_QiQj_vec_warp - gpd_QiQj_rh * rhat)
    d_gpd_QjQi_rh = inv_r * (gpd_QjQi_vec_warp - gpd_QjQi_rh * rhat)
    d_W_2A = (
        tr_Qi * d_X_j_QQ
        + tr_Qj * d_X_i_QQ
        + TWO * (d_gpd_QiQj_rh + d_gpd_QjQi_rh)
        - FOUR * (p_gpd * rhat_QiQj_rhat + gpd_rh * d_rhQiQjr)
    )
    d_gpd_dS_2A = TWO * inv_r * d_W_2A - (TWO * inv_r * inv_r) * rhat * W_2A

    W_4_QQ = (
        gpd_Qi_rh * rhat_Qj_rhat
        + rhat_Qi_rhat * gpd_Qj_rh
        - TWO * gpd_rh * rhat_Qi_rhat * rhat_Qj_rhat
    )
    d_gpd_Qi_rh = inv_r * (Qi_gpd - gpd_Qi_rh * rhat)
    d_gpd_Qj_rh = inv_r * (Qj_gpd - gpd_Qj_rh * rhat)
    d_W_4 = (
        d_gpd_Qi_rh * rhat_Qj_rhat
        + gpd_Qi_rh * d_rhQjr
        + d_rhQir * gpd_Qj_rh
        + rhat_Qi_rhat * d_gpd_Qj_rh
        - TWO
        * (
            p_gpd * rhat_Qi_rhat * rhat_Qj_rhat
            + gpd_rh * d_rhQir * rhat_Qj_rhat
            + gpd_rh * rhat_Qi_rhat * d_rhQjr
        )
    )
    d_gpd_dS_4 = TWO * inv_r * d_W_4 - (TWO * inv_r * inv_r) * rhat * W_4_QQ

    d_V_inner_QQ = (
        K1_2prime * rhat * gpd_rh * S_0_QQ
        + K1_prime * (p_gpd * S_0_QQ)
        + K2_2prime * rhat * gpd_rh * S_2A_QQ
        + K2_prime * (p_gpd * S_2A_QQ + gpd_rh * d_S_2A)
        + K2_prime * rhat * gpd_dS_2A
        + K2 * d_gpd_dS_2A
        + K3_2prime * rhat * gpd_rh * S_4_QQ
        + K3_prime * (p_gpd * S_4_QQ + gpd_rh * d_S_4)
        + K3_prime * rhat * gpd_dS_4
        + K3 * d_gpd_dS_4
    )
    dwdr_QQ = QUARTER * (d_U_ij_QQ + d_U_ji_QQ + d_V_inner_QQ)

    # Final accumulation. The muQ channel enters with a minus sign: its Omega and
    # partials are derived from the +0.5 energy, but the physical energy is -0.5.
    out = _QuadrupoleSecondOrderContrib()
    out.omega = omega_qq + omega_qmu + omega_mumu + omega_qQ - omega_muQ + omega_QQ
    out.dw_dq_i = dwdqi_qq + dwdqi_qmu + dwdqi_qQ
    out.dw_dq_j = dwdqj_qq + dwdqj_qmu + dwdqj_qQ
    out.dw_dmu_i = dwdmu_i_qmu + dwdmu_i_mumu - dwdmu_i_muQ
    out.dw_dmu_j = dwdmu_j_qmu + dwdmu_j_mumu - dwdmu_j_muQ
    out.dw_dQ_i = dwdQ_i_qQ - dwdQ_i_muQ + dwdQ_i_QQ
    out.dw_dQ_j = dwdQ_j_qQ - dwdQ_j_muQ + dwdQ_j_QQ
    out.dw_dr_vec = dwdr_qq + dwdr_qmu + dwdr_mumu + dwdr_qQ - dwdr_muQ + dwdr_QQ
    return out


# Lazy kernel factory: builds (storage, is_batch) kernels on first launch.

_QUADRUPOLE_2ND_BACKWARD_KERNEL_CACHE: dict = {}
_QUADRUPOLE_2ND_BACKWARD_OVERLOAD_CACHE: dict = {}


def _make_quadrupole_2nd_backward_kernel(storage: str, is_batch: bool):
    """Build the uninstantiated ``@wp.kernel`` for a (storage, is_batch) combo.

    Parameters
    ----------
    storage : str
        Neighbor-list storage layout; only ``"csr"`` is supported.
    is_batch : bool
        ``True`` returns the batched kernel (per-system ``cells`` indexed by
        ``atom_batch_idx``); ``False`` the single-system kernel.

    Returns
    -------
    warp.Kernel
        The uninstantiated (``dtype=Any``) second-order backward kernel.

    Raises
    ------
    ValueError
        For any unsupported ``(storage, is_batch)`` combination.
    """
    if storage == "csr" and not is_batch:
        return _build_kernel_csr_single()
    if storage == "csr" and is_batch:
        return _build_kernel_csr_batched()
    raise ValueError(f"unsupported (storage={storage!r}, is_batch={is_batch})")


def _build_kernel_csr_single():
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
        per_direction_scale: wp.array(dtype=wp.float64),
        grad_energies: wp.array(dtype=wp.float64),
        gg_positions: wp.array(dtype=Any),
        gg_charges: wp.array(dtype=Any),
        gg_dipoles: wp.array(dtype=Any),
        gg_quadrupoles: wp.array(dtype=Any),
        gg_grad_energies_2nd: wp.array(dtype=wp.float64),
        gg_positions_2nd: wp.array(dtype=Any),
        gg_charges_2nd: wp.array(dtype=Any),
        gg_dipoles_2nd: wp.array(dtype=Any),
        gg_quadrupoles_2nd: wp.array(dtype=Any),
    ):
        r"""CSR single-system :math:`l_{max}=2` second-order backward kernel.

        Realises the double-backward (``create_graph=True``) of the real-space
        GTO-Ewald multipole energy. One thread per atom ``i`` walks its CSR
        neighbor slice, calls :func:`_quadrupole_2nd_order_pair_contribution`,
        and scatters the per-pair :math:`\Omega` scalar and its 8 partial
        slots (w.r.t. ``grad_energies``, positions, charges, dipoles,
        quadrupoles) into the second-order gradient outputs. The first 4
        ``gg_*`` arrays are the *incoming* upstream gradients (the cotangents
        of the first backward); the ``*_2nd`` arrays are the outputs.

        Launch Grid
        -----------
        ``dim = n_atoms``. One thread per atom ``i``; the thread loops over
        ``neighbor_ptr[i] .. neighbor_ptr[i+1]`` and scatters into both
        endpoints of each pair.

        Parameters
        ----------
        positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Cartesian atom positions.
        charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
            Per-atom monopole charges.
        dipoles : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Per-atom dipole moments.
        quadrupoles : wp.array, shape (N,), dtype=wp.mat33f or wp.mat33d
            Per-atom (symmetric) Cartesian quadrupole tensors.
        cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
            Single-system unit cell (transposed to map ``unit_shifts``).
        idx_j : wp.array, shape (nnz,), dtype=wp.int32
            CSR neighbor column indices.
        neighbor_ptr : wp.array, shape (N + 1,), dtype=wp.int32
            CSR row pointers.
        unit_shifts : wp.array, shape (nnz,), dtype=wp.vec3i
            Integer lattice shifts :math:`\in \mathbb{Z}^3` per pair.
        sigma : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Gaussian charge width.
        alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Ewald splitting parameter.
        per_direction_scale : wp.array, shape (1,), dtype=wp.float64
            Per-direction scale (0.5 half list, 0.25 full list).
        grad_energies : wp.array, shape (N,), dtype=wp.float64
            Upstream per-atom energy weights from the first backward.
        gg_positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Incoming cotangent w.r.t. the position gradient.
        gg_charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
            Incoming cotangent w.r.t. the charge gradient.
        gg_dipoles : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Incoming cotangent w.r.t. the dipole gradient.
        gg_quadrupoles : wp.array, shape (N,), dtype=wp.mat33f or wp.mat33d
            Incoming cotangent w.r.t. the quadrupole gradient.
        gg_grad_energies_2nd : wp.array, shape (N,), dtype=wp.float64
            OUTPUT: second-order gradient w.r.t. ``grad_energies`` (atomic).
        gg_positions_2nd : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            OUTPUT: second-order gradient w.r.t. positions (atomic).
        gg_charges_2nd : wp.array, shape (N,), dtype=wp.float32 or wp.float64
            OUTPUT: second-order gradient w.r.t. charges (atomic).
        gg_dipoles_2nd : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            OUTPUT: second-order gradient w.r.t. dipoles (atomic).
        gg_quadrupoles_2nd : wp.array, shape (N,), dtype=wp.mat33f or wp.mat33d
            OUTPUT: second-order gradient w.r.t. quadrupoles (atomic).
        """
        atom_i = wp.tid()
        sigma_ = wp.float64(sigma[0])
        alpha_ = wp.float64(alpha[0])
        ab = _gto_ewald_ab(sigma_, alpha_)
        a_coef = ab[0]
        b_coef = ab[1]
        scale = per_direction_scale[0]
        cell_t = wp.transpose(cell[0])

        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        mu_i_n = dipoles[atom_i]
        mu_i = wp.vec3d(
            wp.float64(mu_i_n[0]), wp.float64(mu_i_n[1]), wp.float64(mu_i_n[2])
        )
        Q_i_n = quadrupoles[atom_i]
        Q_i = wp.mat33d(
            wp.float64(Q_i_n[0, 0]),
            wp.float64(Q_i_n[0, 1]),
            wp.float64(Q_i_n[0, 2]),
            wp.float64(Q_i_n[1, 0]),
            wp.float64(Q_i_n[1, 1]),
            wp.float64(Q_i_n[1, 2]),
            wp.float64(Q_i_n[2, 0]),
            wp.float64(Q_i_n[2, 1]),
            wp.float64(Q_i_n[2, 2]),
        )
        gp_i_n = gg_positions[atom_i]
        gp_i = wp.vec3d(
            wp.float64(gp_i_n[0]), wp.float64(gp_i_n[1]), wp.float64(gp_i_n[2])
        )
        gd_i_n = gg_dipoles[atom_i]
        gd_i = wp.vec3d(
            wp.float64(gd_i_n[0]), wp.float64(gd_i_n[1]), wp.float64(gd_i_n[2])
        )
        gQ_i_n = gg_quadrupoles[atom_i]
        gQ_i = wp.mat33d(
            wp.float64(gQ_i_n[0, 0]),
            wp.float64(gQ_i_n[0, 1]),
            wp.float64(gQ_i_n[0, 2]),
            wp.float64(gQ_i_n[1, 0]),
            wp.float64(gQ_i_n[1, 1]),
            wp.float64(gQ_i_n[1, 2]),
            wp.float64(gQ_i_n[2, 0]),
            wp.float64(gQ_i_n[2, 1]),
            wp.float64(gQ_i_n[2, 2]),
        )
        gc_i = wp.float64(gg_charges[atom_i])
        ge_i = grad_energies[atom_i]

        k_start = neighbor_ptr[atom_i]
        k_end = neighbor_ptr[atom_i + 1]
        for k in range(k_start, k_end):
            j = idx_j[k]
            shift_vec = unit_shifts[k]
            qj = wp.float64(charges[j])
            pos_j = positions[j]
            mu_j_n = dipoles[j]
            mu_j = wp.vec3d(
                wp.float64(mu_j_n[0]), wp.float64(mu_j_n[1]), wp.float64(mu_j_n[2])
            )
            Q_j_n = quadrupoles[j]
            Q_j = wp.mat33d(
                wp.float64(Q_j_n[0, 0]),
                wp.float64(Q_j_n[0, 1]),
                wp.float64(Q_j_n[0, 2]),
                wp.float64(Q_j_n[1, 0]),
                wp.float64(Q_j_n[1, 1]),
                wp.float64(Q_j_n[1, 2]),
                wp.float64(Q_j_n[2, 0]),
                wp.float64(Q_j_n[2, 1]),
                wp.float64(Q_j_n[2, 2]),
            )
            gp_j_n = gg_positions[j]
            gp_j = wp.vec3d(
                wp.float64(gp_j_n[0]), wp.float64(gp_j_n[1]), wp.float64(gp_j_n[2])
            )
            gd_j_n = gg_dipoles[j]
            gd_j = wp.vec3d(
                wp.float64(gd_j_n[0]), wp.float64(gd_j_n[1]), wp.float64(gd_j_n[2])
            )
            gQ_j_n = gg_quadrupoles[j]
            gQ_j = wp.mat33d(
                wp.float64(gQ_j_n[0, 0]),
                wp.float64(gQ_j_n[0, 1]),
                wp.float64(gQ_j_n[0, 2]),
                wp.float64(gQ_j_n[1, 0]),
                wp.float64(gQ_j_n[1, 1]),
                wp.float64(gQ_j_n[1, 2]),
                wp.float64(gQ_j_n[2, 0]),
                wp.float64(gQ_j_n[2, 1]),
                wp.float64(gQ_j_n[2, 2]),
            )
            gc_j = wp.float64(gg_charges[j])
            ge_j = grad_energies[j]

            periodic_shift = cell_t * type(pos_i)(
                type(pos_i[0])(shift_vec[0]),
                type(pos_i[0])(shift_vec[1]),
                type(pos_i[0])(shift_vec[2]),
            )
            sep = pos_j - pos_i + periodic_shift
            distance = wp.float64(wp.length(sep))

            if distance > wp.float64(1e-8):
                r_vec = wp.vec3d(
                    wp.float64(sep[0]), wp.float64(sep[1]), wp.float64(sep[2])
                )
                contrib = _quadrupole_2nd_order_pair_contribution(
                    r_vec,
                    distance,
                    qi,
                    mu_i,
                    Q_i,
                    qj,
                    mu_j,
                    Q_j,
                    gp_i,
                    gd_i,
                    gQ_i,
                    gc_i,
                    gp_j,
                    gd_j,
                    gQ_j,
                    gc_j,
                    a_coef,
                    b_coef,
                )
                # scale: 0.5 for a half list, 0.25 for a full list (each pair counted twice).
                half_omega = scale * contrib.omega
                wp.atomic_add(gg_grad_energies_2nd, atom_i, half_omega)
                wp.atomic_add(gg_grad_energies_2nd, j, half_omega)

                half_ge = scale * (ge_i + ge_j)

                wp.atomic_add(
                    gg_charges_2nd,
                    atom_i,
                    type(charges[atom_i])(half_ge * contrib.dw_dq_i),
                )
                wp.atomic_add(
                    gg_charges_2nd, j, type(charges[atom_i])(half_ge * contrib.dw_dq_j)
                )

                dmu_i = contrib.dw_dmu_i
                dmu_j = contrib.dw_dmu_j
                wp.atomic_add(
                    gg_dipoles_2nd,
                    atom_i,
                    type(mu_i_n)(
                        type(mu_i_n[0])(half_ge * dmu_i[0]),
                        type(mu_i_n[0])(half_ge * dmu_i[1]),
                        type(mu_i_n[0])(half_ge * dmu_i[2]),
                    ),
                )
                wp.atomic_add(
                    gg_dipoles_2nd,
                    j,
                    type(mu_i_n)(
                        type(mu_i_n[0])(half_ge * dmu_j[0]),
                        type(mu_i_n[0])(half_ge * dmu_j[1]),
                        type(mu_i_n[0])(half_ge * dmu_j[2]),
                    ),
                )

                dQi = contrib.dw_dQ_i
                dQj = contrib.dw_dQ_j
                wp.atomic_add(
                    gg_quadrupoles_2nd,
                    atom_i,
                    type(Q_i_n)(
                        type(Q_i_n[0, 0])(half_ge * dQi[0, 0]),
                        type(Q_i_n[0, 0])(half_ge * dQi[0, 1]),
                        type(Q_i_n[0, 0])(half_ge * dQi[0, 2]),
                        type(Q_i_n[0, 0])(half_ge * dQi[1, 0]),
                        type(Q_i_n[0, 0])(half_ge * dQi[1, 1]),
                        type(Q_i_n[0, 0])(half_ge * dQi[1, 2]),
                        type(Q_i_n[0, 0])(half_ge * dQi[2, 0]),
                        type(Q_i_n[0, 0])(half_ge * dQi[2, 1]),
                        type(Q_i_n[0, 0])(half_ge * dQi[2, 2]),
                    ),
                )
                wp.atomic_add(
                    gg_quadrupoles_2nd,
                    j,
                    type(Q_i_n)(
                        type(Q_i_n[0, 0])(half_ge * dQj[0, 0]),
                        type(Q_i_n[0, 0])(half_ge * dQj[0, 1]),
                        type(Q_i_n[0, 0])(half_ge * dQj[0, 2]),
                        type(Q_i_n[0, 0])(half_ge * dQj[1, 0]),
                        type(Q_i_n[0, 0])(half_ge * dQj[1, 1]),
                        type(Q_i_n[0, 0])(half_ge * dQj[1, 2]),
                        type(Q_i_n[0, 0])(half_ge * dQj[2, 0]),
                        type(Q_i_n[0, 0])(half_ge * dQj[2, 1]),
                        type(Q_i_n[0, 0])(half_ge * dQj[2, 2]),
                    ),
                )

                dr = contrib.dw_dr_vec
                px = half_ge * dr[0]
                py = half_ge * dr[1]
                pz = half_ge * dr[2]
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

    _kernel.__name__ = "_multipole_real_space_quadrupole_csr_2nd_backward_kernel"
    return wp.kernel(_kernel, enable_backward=False)


def _build_kernel_csr_batched():
    def _kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        quadrupoles: wp.array(dtype=Any),
        cells: wp.array(dtype=Any),
        idx_j: wp.array(dtype=wp.int32),
        neighbor_ptr: wp.array(dtype=wp.int32),
        atom_batch_idx: wp.array(dtype=wp.int32),
        unit_shifts: wp.array(dtype=wp.vec3i),
        sigma: wp.array(dtype=Any),
        alpha: wp.array(dtype=Any),
        per_direction_scale: wp.array(dtype=wp.float64),
        grad_energies: wp.array(dtype=wp.float64),
        gg_positions: wp.array(dtype=Any),
        gg_charges: wp.array(dtype=Any),
        gg_dipoles: wp.array(dtype=Any),
        gg_quadrupoles: wp.array(dtype=Any),
        gg_grad_energies_2nd: wp.array(dtype=wp.float64),
        gg_positions_2nd: wp.array(dtype=Any),
        gg_charges_2nd: wp.array(dtype=Any),
        gg_dipoles_2nd: wp.array(dtype=Any),
        gg_quadrupoles_2nd: wp.array(dtype=Any),
    ):
        r"""Batched CSR :math:`l_{max}=2` second-order backward kernel.

        Batched analog of the single-system second-order backward: atom ``i``
        maps to system ``b = atom_batch_idx[i]`` with cell ``cells[b]``. The
        per-pair :math:`\Omega` scalar and its 8 partial slots are scattered
        into the global ``*_2nd`` outputs.

        Launch Grid
        -----------
        ``dim = n_atoms`` (all systems concatenated). One thread per atom ``i``.

        Parameters
        ----------
        positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Cartesian atom positions (all systems).
        charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
            Per-atom monopole charges.
        dipoles : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Per-atom dipole moments.
        quadrupoles : wp.array, shape (N,), dtype=wp.mat33f or wp.mat33d
            Per-atom (symmetric) Cartesian quadrupole tensors.
        cells : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
            Per-system unit cells, indexed by ``atom_batch_idx``.
        idx_j : wp.array, shape (nnz,), dtype=wp.int32
            CSR neighbor column indices.
        neighbor_ptr : wp.array, shape (N + 1,), dtype=wp.int32
            CSR row pointers.
        atom_batch_idx : wp.array, shape (N,), dtype=wp.int32
            System index ``b`` for each atom.
        unit_shifts : wp.array, shape (nnz,), dtype=wp.vec3i
            Integer lattice shifts :math:`\in \mathbb{Z}^3` per pair.
        sigma : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Gaussian charge width.
        alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
            Ewald splitting parameter.
        per_direction_scale : wp.array, shape (1,), dtype=wp.float64
            Per-direction scale (0.5 half list, 0.25 full list).
        grad_energies : wp.array, shape (N,), dtype=wp.float64
            Upstream per-atom energy weights from the first backward.
        gg_positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Incoming cotangent w.r.t. the position gradient.
        gg_charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
            Incoming cotangent w.r.t. the charge gradient.
        gg_dipoles : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            Incoming cotangent w.r.t. the dipole gradient.
        gg_quadrupoles : wp.array, shape (N,), dtype=wp.mat33f or wp.mat33d
            Incoming cotangent w.r.t. the quadrupole gradient.
        gg_grad_energies_2nd : wp.array, shape (N,), dtype=wp.float64
            OUTPUT: second-order gradient w.r.t. ``grad_energies`` (atomic).
        gg_positions_2nd : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            OUTPUT: second-order gradient w.r.t. positions (atomic).
        gg_charges_2nd : wp.array, shape (N,), dtype=wp.float32 or wp.float64
            OUTPUT: second-order gradient w.r.t. charges (atomic).
        gg_dipoles_2nd : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
            OUTPUT: second-order gradient w.r.t. dipoles (atomic).
        gg_quadrupoles_2nd : wp.array, shape (N,), dtype=wp.mat33f or wp.mat33d
            OUTPUT: second-order gradient w.r.t. quadrupoles (atomic).
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

        qi = wp.float64(charges[atom_i])
        pos_i = positions[atom_i]
        mu_i_n = dipoles[atom_i]
        mu_i = wp.vec3d(
            wp.float64(mu_i_n[0]), wp.float64(mu_i_n[1]), wp.float64(mu_i_n[2])
        )
        Q_i_n = quadrupoles[atom_i]
        Q_i = wp.mat33d(
            wp.float64(Q_i_n[0, 0]),
            wp.float64(Q_i_n[0, 1]),
            wp.float64(Q_i_n[0, 2]),
            wp.float64(Q_i_n[1, 0]),
            wp.float64(Q_i_n[1, 1]),
            wp.float64(Q_i_n[1, 2]),
            wp.float64(Q_i_n[2, 0]),
            wp.float64(Q_i_n[2, 1]),
            wp.float64(Q_i_n[2, 2]),
        )
        gp_i_n = gg_positions[atom_i]
        gp_i = wp.vec3d(
            wp.float64(gp_i_n[0]), wp.float64(gp_i_n[1]), wp.float64(gp_i_n[2])
        )
        gd_i_n = gg_dipoles[atom_i]
        gd_i = wp.vec3d(
            wp.float64(gd_i_n[0]), wp.float64(gd_i_n[1]), wp.float64(gd_i_n[2])
        )
        gQ_i_n = gg_quadrupoles[atom_i]
        gQ_i = wp.mat33d(
            wp.float64(gQ_i_n[0, 0]),
            wp.float64(gQ_i_n[0, 1]),
            wp.float64(gQ_i_n[0, 2]),
            wp.float64(gQ_i_n[1, 0]),
            wp.float64(gQ_i_n[1, 1]),
            wp.float64(gQ_i_n[1, 2]),
            wp.float64(gQ_i_n[2, 0]),
            wp.float64(gQ_i_n[2, 1]),
            wp.float64(gQ_i_n[2, 2]),
        )
        gc_i = wp.float64(gg_charges[atom_i])
        ge_i = grad_energies[atom_i]

        k_start = neighbor_ptr[atom_i]
        k_end = neighbor_ptr[atom_i + 1]
        for k in range(k_start, k_end):
            j = idx_j[k]
            shift_vec = unit_shifts[k]
            qj = wp.float64(charges[j])
            pos_j = positions[j]
            mu_j_n = dipoles[j]
            mu_j = wp.vec3d(
                wp.float64(mu_j_n[0]), wp.float64(mu_j_n[1]), wp.float64(mu_j_n[2])
            )
            Q_j_n = quadrupoles[j]
            Q_j = wp.mat33d(
                wp.float64(Q_j_n[0, 0]),
                wp.float64(Q_j_n[0, 1]),
                wp.float64(Q_j_n[0, 2]),
                wp.float64(Q_j_n[1, 0]),
                wp.float64(Q_j_n[1, 1]),
                wp.float64(Q_j_n[1, 2]),
                wp.float64(Q_j_n[2, 0]),
                wp.float64(Q_j_n[2, 1]),
                wp.float64(Q_j_n[2, 2]),
            )
            gp_j_n = gg_positions[j]
            gp_j = wp.vec3d(
                wp.float64(gp_j_n[0]), wp.float64(gp_j_n[1]), wp.float64(gp_j_n[2])
            )
            gd_j_n = gg_dipoles[j]
            gd_j = wp.vec3d(
                wp.float64(gd_j_n[0]), wp.float64(gd_j_n[1]), wp.float64(gd_j_n[2])
            )
            gQ_j_n = gg_quadrupoles[j]
            gQ_j = wp.mat33d(
                wp.float64(gQ_j_n[0, 0]),
                wp.float64(gQ_j_n[0, 1]),
                wp.float64(gQ_j_n[0, 2]),
                wp.float64(gQ_j_n[1, 0]),
                wp.float64(gQ_j_n[1, 1]),
                wp.float64(gQ_j_n[1, 2]),
                wp.float64(gQ_j_n[2, 0]),
                wp.float64(gQ_j_n[2, 1]),
                wp.float64(gQ_j_n[2, 2]),
            )
            gc_j = wp.float64(gg_charges[j])
            ge_j = grad_energies[j]

            periodic_shift = cell_t * type(pos_i)(
                type(pos_i[0])(shift_vec[0]),
                type(pos_i[0])(shift_vec[1]),
                type(pos_i[0])(shift_vec[2]),
            )
            sep = pos_j - pos_i + periodic_shift
            distance = wp.float64(wp.length(sep))

            if distance > wp.float64(1e-8):
                r_vec = wp.vec3d(
                    wp.float64(sep[0]), wp.float64(sep[1]), wp.float64(sep[2])
                )
                contrib = _quadrupole_2nd_order_pair_contribution(
                    r_vec,
                    distance,
                    qi,
                    mu_i,
                    Q_i,
                    qj,
                    mu_j,
                    Q_j,
                    gp_i,
                    gd_i,
                    gQ_i,
                    gc_i,
                    gp_j,
                    gd_j,
                    gQ_j,
                    gc_j,
                    a_coef,
                    b_coef,
                )
                # scale: 0.5 for a half list, 0.25 for a full list (each pair counted twice).
                half_omega = scale * contrib.omega
                wp.atomic_add(gg_grad_energies_2nd, atom_i, half_omega)
                wp.atomic_add(gg_grad_energies_2nd, j, half_omega)

                half_ge = scale * (ge_i + ge_j)

                wp.atomic_add(
                    gg_charges_2nd,
                    atom_i,
                    type(charges[atom_i])(half_ge * contrib.dw_dq_i),
                )
                wp.atomic_add(
                    gg_charges_2nd, j, type(charges[atom_i])(half_ge * contrib.dw_dq_j)
                )

                dmu_i = contrib.dw_dmu_i
                dmu_j = contrib.dw_dmu_j
                wp.atomic_add(
                    gg_dipoles_2nd,
                    atom_i,
                    type(mu_i_n)(
                        type(mu_i_n[0])(half_ge * dmu_i[0]),
                        type(mu_i_n[0])(half_ge * dmu_i[1]),
                        type(mu_i_n[0])(half_ge * dmu_i[2]),
                    ),
                )
                wp.atomic_add(
                    gg_dipoles_2nd,
                    j,
                    type(mu_i_n)(
                        type(mu_i_n[0])(half_ge * dmu_j[0]),
                        type(mu_i_n[0])(half_ge * dmu_j[1]),
                        type(mu_i_n[0])(half_ge * dmu_j[2]),
                    ),
                )

                dQi = contrib.dw_dQ_i
                dQj = contrib.dw_dQ_j
                wp.atomic_add(
                    gg_quadrupoles_2nd,
                    atom_i,
                    type(Q_i_n)(
                        type(Q_i_n[0, 0])(half_ge * dQi[0, 0]),
                        type(Q_i_n[0, 0])(half_ge * dQi[0, 1]),
                        type(Q_i_n[0, 0])(half_ge * dQi[0, 2]),
                        type(Q_i_n[0, 0])(half_ge * dQi[1, 0]),
                        type(Q_i_n[0, 0])(half_ge * dQi[1, 1]),
                        type(Q_i_n[0, 0])(half_ge * dQi[1, 2]),
                        type(Q_i_n[0, 0])(half_ge * dQi[2, 0]),
                        type(Q_i_n[0, 0])(half_ge * dQi[2, 1]),
                        type(Q_i_n[0, 0])(half_ge * dQi[2, 2]),
                    ),
                )
                wp.atomic_add(
                    gg_quadrupoles_2nd,
                    j,
                    type(Q_i_n)(
                        type(Q_i_n[0, 0])(half_ge * dQj[0, 0]),
                        type(Q_i_n[0, 0])(half_ge * dQj[0, 1]),
                        type(Q_i_n[0, 0])(half_ge * dQj[0, 2]),
                        type(Q_i_n[0, 0])(half_ge * dQj[1, 0]),
                        type(Q_i_n[0, 0])(half_ge * dQj[1, 1]),
                        type(Q_i_n[0, 0])(half_ge * dQj[1, 2]),
                        type(Q_i_n[0, 0])(half_ge * dQj[2, 0]),
                        type(Q_i_n[0, 0])(half_ge * dQj[2, 1]),
                        type(Q_i_n[0, 0])(half_ge * dQj[2, 2]),
                    ),
                )

                dr = contrib.dw_dr_vec
                px = half_ge * dr[0]
                py = half_ge * dr[1]
                pz = half_ge * dr[2]
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

    _kernel.__name__ = "_batch_multipole_real_space_quadrupole_csr_2nd_backward_kernel"
    return wp.kernel(_kernel, enable_backward=False)


def _build_sig_csr_single(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=m),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.vec3i),
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.array(dtype=wp.float64),  # per_direction_scale
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=wp.float64),
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=m),
    ]


def _build_sig_csr_batched(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=m),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.vec3i),
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.array(dtype=wp.float64),  # per_direction_scale
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=wp.float64),
        wp.array(dtype=v),
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=m),
    ]


_SIG_BUILDERS = {
    ("csr", False): _build_sig_csr_single,
    ("csr", True): _build_sig_csr_batched,
}


def _get_quadrupole_2nd_backward_overload(
    storage: str, is_batch: bool, vec_dtype, scalar_dtype
):
    """Get-or-build the typed overload for ``(storage, is_batch, vec_dtype)``.

    The kernel function is built on first request for a ``(storage, is_batch)``;
    the typed overload is registered on first request for a
    ``(storage, is_batch, vec_dtype)``.
    """
    kernel_key = (storage, is_batch)
    if kernel_key not in _QUADRUPOLE_2ND_BACKWARD_KERNEL_CACHE:
        _QUADRUPOLE_2ND_BACKWARD_KERNEL_CACHE[kernel_key] = (
            _make_quadrupole_2nd_backward_kernel(storage, is_batch)
        )
    kernel = _QUADRUPOLE_2ND_BACKWARD_KERNEL_CACHE[kernel_key]

    overload_key = (storage, is_batch, vec_dtype)
    if overload_key not in _QUADRUPOLE_2ND_BACKWARD_OVERLOAD_CACHE:
        sig_builder = _SIG_BUILDERS[(storage, is_batch)]
        sig = sig_builder(vec_dtype, scalar_dtype)
        _QUADRUPOLE_2ND_BACKWARD_OVERLOAD_CACHE[overload_key] = wp.overload(kernel, sig)
    return _QUADRUPOLE_2ND_BACKWARD_OVERLOAD_CACHE[overload_key]


# Per-direction scattering scale for half/full neighbor list conventions.
#
# The closed forms assume each pair is processed once and contributes 0.5*omega
# to each of atom_i and atom_j. The kernels iterate the user-supplied neighbor
# list directly, so the per-direction scale must absorb the convention:
#   - full list: each pair appears twice (i->j and j->i); omega is invariant
#     under (i<->j, r_vec -> -r_vec), so scale must be 0.25 to land at 0.5*omega.
#   - half list: each pair appears once, so scale must be 0.5.
# The kernel reads ``scale = per_direction_scale[0]`` for both the omega scatter
# and the half_ge multiplier.

# Cache of (device, scale_value) -> wp.array to avoid per-launch allocation.
_QUADRUPOLE_2ND_BACKWARD_SCALE_CACHE: dict = {}


def _make_scale_array(half_neighbor_list: bool, device: str):
    """Get-or-build the 1-element scale array for a given (device, neighbor list) combo."""
    key = (str(device), bool(half_neighbor_list))
    if key not in _QUADRUPOLE_2ND_BACKWARD_SCALE_CACHE:
        value = 0.5 if half_neighbor_list else 0.25
        _QUADRUPOLE_2ND_BACKWARD_SCALE_CACHE[key] = wp.array(
            [value],
            dtype=wp.float64,
            device=device,
        )
    return _QUADRUPOLE_2ND_BACKWARD_SCALE_CACHE[key]


@wp.kernel
def _multipole_real_space_quadrupole_csr_cell_grad_backward_kernel(
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
    per_direction_scale: wp.array(dtype=wp.float64),
    grad_energies: wp.array(dtype=wp.float64),  # (N,) per-atom energy weight
    g_cell: wp.array(dtype=Any),  # (1,) mat33 cotangent on grad_cell
    grad_grad_energies: wp.array(dtype=wp.float64),  # (N,) OUT (atomic)
    grad_positions: wp.array(dtype=Any),  # (N,) OUT
    grad_charges: wp.array(dtype=Any),  # (N,) OUT
    grad_dipoles: wp.array(dtype=Any),  # (N,) OUT
    grad_quadrupoles: wp.array(dtype=Any),  # (N,) OUT
    grad_cell_out: wp.array(dtype=Any),  # (1,) mat33 OUT (atomic)
):
    r"""Double-backward of the l=2 real-space cell-grad (stress-loss).

    Reuses :func:`_quadrupole_2nd_order_pair_contribution` with the per-pair
    position direction ``gpd = gp_j - gp_i = w = g_cellᵀ·n`` (set ``gp_j=w``,
    ``gp_i=0``) and all charge/dipole/quadrupole directions zero, so
    ``Ω = -w·f`` (f = ∂E_pair/∂r). The forward cell-grad weights each pair by
    ``weight = scale·(ge_i+ge_j)/2`` (``scale`` = 1.0 half / 0.5 full;
    ``ge`` = grad_energies, ones for plain ``dE/dcell``). With
    ``S = ⟨g_cell, grad_cell⟩``: ``grad_r_i = +weight·dw_dr_vec``,
    ``grad_r_j = -weight·dw_dr_vec``, ``grad_θ = -weight·dw_dθ`` (θ = q,μ,Q),
    ``grad_cell[a,b] = -weight·n[a]·dw_dr_vec[b]``, and
    ``grad_ge_{i,j} += -scale·½·Ω`` (= ``+scale·½·(w·f)``).
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

    zero_v = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    zero_m = wp.mat33d(
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
    )

    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    mu_i_n = dipoles[atom_i]
    mu_i = wp.vec3d(wp.float64(mu_i_n[0]), wp.float64(mu_i_n[1]), wp.float64(mu_i_n[2]))
    Q_i_n = quadrupoles[atom_i]
    Q_i = wp.mat33d(
        wp.float64(Q_i_n[0, 0]),
        wp.float64(Q_i_n[0, 1]),
        wp.float64(Q_i_n[0, 2]),
        wp.float64(Q_i_n[1, 0]),
        wp.float64(Q_i_n[1, 1]),
        wp.float64(Q_i_n[1, 2]),
        wp.float64(Q_i_n[2, 0]),
        wp.float64(Q_i_n[2, 1]),
        wp.float64(Q_i_n[2, 2]),
    )
    ge_i = grad_energies[atom_i]

    k_start = neighbor_ptr[atom_i]
    k_end = neighbor_ptr[atom_i + 1]
    for k in range(k_start, k_end):
        j = idx_j[k]
        shift_vec = unit_shifts[k]
        qj = wp.float64(charges[j])
        pos_j = positions[j]
        ge_j = grad_energies[j]
        mu_j_n = dipoles[j]
        mu_j = wp.vec3d(
            wp.float64(mu_j_n[0]), wp.float64(mu_j_n[1]), wp.float64(mu_j_n[2])
        )
        Q_j_n = quadrupoles[j]
        Q_j = wp.mat33d(
            wp.float64(Q_j_n[0, 0]),
            wp.float64(Q_j_n[0, 1]),
            wp.float64(Q_j_n[0, 2]),
            wp.float64(Q_j_n[1, 0]),
            wp.float64(Q_j_n[1, 1]),
            wp.float64(Q_j_n[1, 2]),
            wp.float64(Q_j_n[2, 0]),
            wp.float64(Q_j_n[2, 1]),
            wp.float64(Q_j_n[2, 2]),
        )
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        sep = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(sep))
        if distance > wp.float64(1e-8):
            r_vec = wp.vec3d(wp.float64(sep[0]), wp.float64(sep[1]), wp.float64(sep[2]))
            sh0 = wp.float64(shift_vec[0])
            sh1 = wp.float64(shift_vec[1])
            sh2 = wp.float64(shift_vec[2])
            # Per-pair direction w = g_cellᵀ·n. The func's Ω uses
            # gpd = gp_j - gp_i with the opposite sign convention to the
            # hand-derived l≤1 kernels, so feed gp_j = -w (gpd = -w); the
            # resulting slot/Ω flip cancels against the scatter signs below
            # (FD-validated all channels).
            w = wp.vec3d(
                -(g00 * sh0 + g10 * sh1 + g20 * sh2),
                -(g01 * sh0 + g11 * sh1 + g21 * sh2),
                -(g02 * sh0 + g12 * sh1 + g22 * sh2),
            )
            contrib = _quadrupole_2nd_order_pair_contribution(
                r_vec,
                distance,
                qi,
                mu_i,
                Q_i,
                qj,
                mu_j,
                Q_j,
                zero_v,  # gp_i
                zero_v,  # gd_i
                zero_m,  # gQ_i
                wp.float64(0.0),  # gc_i
                w,  # gp_j = w
                zero_v,  # gd_j
                zero_m,  # gQ_j
                wp.float64(0.0),  # gc_j
                a_coef,
                b_coef,
            )
            # Forward weight per pair: scale·(ge_i+ge_j)/2 (ge=ones for plain
            # stress). grad_ge_{i,j} += scale·½·(w·f) = -scale·½·Ω.
            weight = scale * (ge_i + ge_j) * wp.float64(0.5)
            ge_contrib = -scale * wp.float64(0.5) * contrib.omega
            wp.atomic_add(grad_grad_energies, atom_i, ge_contrib)
            wp.atomic_add(grad_grad_energies, j, ge_contrib)
            wp.atomic_add(
                grad_charges,
                atom_i,
                type(charges[atom_i])(-weight * contrib.dw_dq_i),
            )
            wp.atomic_add(
                grad_charges, j, type(charges[atom_i])(-weight * contrib.dw_dq_j)
            )
            dmu_i = contrib.dw_dmu_i
            dmu_j = contrib.dw_dmu_j
            wp.atomic_add(
                grad_dipoles,
                atom_i,
                type(mu_i_n)(
                    type(mu_i_n[0])(-weight * dmu_i[0]),
                    type(mu_i_n[0])(-weight * dmu_i[1]),
                    type(mu_i_n[0])(-weight * dmu_i[2]),
                ),
            )
            wp.atomic_add(
                grad_dipoles,
                j,
                type(mu_i_n)(
                    type(mu_i_n[0])(-weight * dmu_j[0]),
                    type(mu_i_n[0])(-weight * dmu_j[1]),
                    type(mu_i_n[0])(-weight * dmu_j[2]),
                ),
            )
            dQi = contrib.dw_dQ_i
            dQj = contrib.dw_dQ_j
            wp.atomic_add(
                grad_quadrupoles,
                atom_i,
                type(Q_i_n)(
                    type(Q_i_n[0, 0])(-weight * dQi[0, 0]),
                    type(Q_i_n[0, 0])(-weight * dQi[0, 1]),
                    type(Q_i_n[0, 0])(-weight * dQi[0, 2]),
                    type(Q_i_n[0, 0])(-weight * dQi[1, 0]),
                    type(Q_i_n[0, 0])(-weight * dQi[1, 1]),
                    type(Q_i_n[0, 0])(-weight * dQi[1, 2]),
                    type(Q_i_n[0, 0])(-weight * dQi[2, 0]),
                    type(Q_i_n[0, 0])(-weight * dQi[2, 1]),
                    type(Q_i_n[0, 0])(-weight * dQi[2, 2]),
                ),
            )
            wp.atomic_add(
                grad_quadrupoles,
                j,
                type(Q_i_n)(
                    type(Q_i_n[0, 0])(-weight * dQj[0, 0]),
                    type(Q_i_n[0, 0])(-weight * dQj[0, 1]),
                    type(Q_i_n[0, 0])(-weight * dQj[0, 2]),
                    type(Q_i_n[0, 0])(-weight * dQj[1, 0]),
                    type(Q_i_n[0, 0])(-weight * dQj[1, 1]),
                    type(Q_i_n[0, 0])(-weight * dQj[1, 2]),
                    type(Q_i_n[0, 0])(-weight * dQj[2, 0]),
                    type(Q_i_n[0, 0])(-weight * dQj[2, 1]),
                    type(Q_i_n[0, 0])(-weight * dQj[2, 2]),
                ),
            )
            dr = contrib.dw_dr_vec
            sgx = weight * dr[0]
            sgy = weight * dr[1]
            sgz = weight * dr[2]
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


def _quadrupole_csr_cell_grad_backward_sig(v, t):
    """Signature builder for the l=2 cell-grad double-backward kernel."""
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
        wp.array(dtype=wp.float64),  # per_direction_scale
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=m),  # g_cell
        wp.array(dtype=wp.float64),  # grad_grad_energies (out)
        wp.array(dtype=v),  # grad_positions (out)
        wp.array(dtype=t),  # grad_charges (out)
        wp.array(dtype=v),  # grad_dipoles (out)
        wp.array(dtype=m),  # grad_quadrupoles (out)
        wp.array(dtype=m),  # grad_cell_out (out)
    ]


_quadrupole_csr_cell_grad_backward_overloads = register_overloads(
    _multipole_real_space_quadrupole_csr_cell_grad_backward_kernel,
    _quadrupole_csr_cell_grad_backward_sig,
)


def multipole_real_space_quadrupole_csr_cell_grad_backward(
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
    per_direction_scale,
    grad_energies,
    g_cell,
    grad_grad_energies,
    grad_positions,
    grad_charges,
    grad_dipoles,
    grad_quadrupoles,
    grad_cell_out,
    wp_dtype,
    device=None,
):
    r"""Launcher for the l=2 cell-grad double-backward (stress-loss).

    Caller pre-zeroes the six output arrays.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(positions.device)
    wp.launch(
        _quadrupole_csr_cell_grad_backward_overloads[vec_dtype],
        dim=positions.shape[0],
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
            per_direction_scale,
            grad_energies,
            g_cell,
        ],
        outputs=[
            grad_grad_energies,
            grad_positions,
            grad_charges,
            grad_dipoles,
            grad_quadrupoles,
            grad_cell_out,
        ],
        device=device,
    )


@wp.kernel
def _batch_multipole_real_space_quadrupole_csr_cell_grad_backward_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    dipoles: wp.array(dtype=Any),
    quadrupoles: wp.array(dtype=Any),
    cells: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    atom_batch_idx: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    sigma: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    per_direction_scale: wp.array(dtype=wp.float64),
    grad_energies: wp.array(dtype=wp.float64),  # (N,) per-atom energy weight
    g_cell: wp.array(dtype=Any),  # (1,) mat33 cotangent on grad_cell
    grad_grad_energies: wp.array(dtype=wp.float64),  # (N,) OUT (atomic)
    grad_positions: wp.array(dtype=Any),  # (N,) OUT
    grad_charges: wp.array(dtype=Any),  # (N,) OUT
    grad_dipoles: wp.array(dtype=Any),  # (N,) OUT
    grad_quadrupoles: wp.array(dtype=Any),  # (N,) OUT
    grad_cell_out: wp.array(dtype=Any),  # (1,) mat33 OUT (atomic)
):
    r"""Double-backward of the l=2 real-space cell-grad (stress-loss).

    Reuses :func:`_quadrupole_2nd_order_pair_contribution` with the per-pair
    position direction ``gpd = gp_j - gp_i = w = g_cellᵀ·n`` (set ``gp_j=w``,
    ``gp_i=0``) and all charge/dipole/quadrupole directions zero, so
    ``Ω = -w·f`` (f = ∂E_pair/∂r). The forward cell-grad weights each pair by
    ``weight = scale·(ge_i+ge_j)/2`` (``scale`` = 1.0 half / 0.5 full;
    ``ge`` = grad_energies, ones for plain ``dE/dcell``). With
    ``S = ⟨g_cell, grad_cell⟩``: ``grad_r_i = +weight·dw_dr_vec``,
    ``grad_r_j = -weight·dw_dr_vec``, ``grad_θ = -weight·dw_dθ`` (θ = q,μ,Q),
    ``grad_cell[a,b] = -weight·n[a]·dw_dr_vec[b]``, and
    ``grad_ge_{i,j} += -scale·½·Ω`` (= ``+scale·½·(w·f)``).
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

    zero_v = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    zero_m = wp.mat33d(
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
    )

    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    mu_i_n = dipoles[atom_i]
    mu_i = wp.vec3d(wp.float64(mu_i_n[0]), wp.float64(mu_i_n[1]), wp.float64(mu_i_n[2]))
    Q_i_n = quadrupoles[atom_i]
    Q_i = wp.mat33d(
        wp.float64(Q_i_n[0, 0]),
        wp.float64(Q_i_n[0, 1]),
        wp.float64(Q_i_n[0, 2]),
        wp.float64(Q_i_n[1, 0]),
        wp.float64(Q_i_n[1, 1]),
        wp.float64(Q_i_n[1, 2]),
        wp.float64(Q_i_n[2, 0]),
        wp.float64(Q_i_n[2, 1]),
        wp.float64(Q_i_n[2, 2]),
    )
    ge_i = grad_energies[atom_i]

    k_start = neighbor_ptr[atom_i]
    k_end = neighbor_ptr[atom_i + 1]
    for k in range(k_start, k_end):
        j = idx_j[k]
        shift_vec = unit_shifts[k]
        qj = wp.float64(charges[j])
        pos_j = positions[j]
        ge_j = grad_energies[j]
        mu_j_n = dipoles[j]
        mu_j = wp.vec3d(
            wp.float64(mu_j_n[0]), wp.float64(mu_j_n[1]), wp.float64(mu_j_n[2])
        )
        Q_j_n = quadrupoles[j]
        Q_j = wp.mat33d(
            wp.float64(Q_j_n[0, 0]),
            wp.float64(Q_j_n[0, 1]),
            wp.float64(Q_j_n[0, 2]),
            wp.float64(Q_j_n[1, 0]),
            wp.float64(Q_j_n[1, 1]),
            wp.float64(Q_j_n[1, 2]),
            wp.float64(Q_j_n[2, 0]),
            wp.float64(Q_j_n[2, 1]),
            wp.float64(Q_j_n[2, 2]),
        )
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        sep = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(sep))
        if distance > wp.float64(1e-8):
            r_vec = wp.vec3d(wp.float64(sep[0]), wp.float64(sep[1]), wp.float64(sep[2]))
            sh0 = wp.float64(shift_vec[0])
            sh1 = wp.float64(shift_vec[1])
            sh2 = wp.float64(shift_vec[2])
            # Per-pair direction w = g_cellᵀ·n. The func's Ω uses
            # gpd = gp_j - gp_i with the opposite sign convention to the
            # hand-derived l≤1 kernels, so feed gp_j = -w (gpd = -w); the
            # resulting slot/Ω flip cancels against the scatter signs below
            # (FD-validated all channels).
            w = wp.vec3d(
                -(g00 * sh0 + g10 * sh1 + g20 * sh2),
                -(g01 * sh0 + g11 * sh1 + g21 * sh2),
                -(g02 * sh0 + g12 * sh1 + g22 * sh2),
            )
            contrib = _quadrupole_2nd_order_pair_contribution(
                r_vec,
                distance,
                qi,
                mu_i,
                Q_i,
                qj,
                mu_j,
                Q_j,
                zero_v,  # gp_i
                zero_v,  # gd_i
                zero_m,  # gQ_i
                wp.float64(0.0),  # gc_i
                w,  # gp_j = w
                zero_v,  # gd_j
                zero_m,  # gQ_j
                wp.float64(0.0),  # gc_j
                a_coef,
                b_coef,
            )
            # Forward weight per pair: scale·(ge_i+ge_j)/2 (ge=ones for plain
            # stress). grad_ge_{i,j} += scale·½·(w·f) = -scale·½·Ω.
            weight = scale * (ge_i + ge_j) * wp.float64(0.5)
            ge_contrib = -scale * wp.float64(0.5) * contrib.omega
            wp.atomic_add(grad_grad_energies, atom_i, ge_contrib)
            wp.atomic_add(grad_grad_energies, j, ge_contrib)
            wp.atomic_add(
                grad_charges,
                atom_i,
                type(charges[atom_i])(-weight * contrib.dw_dq_i),
            )
            wp.atomic_add(
                grad_charges, j, type(charges[atom_i])(-weight * contrib.dw_dq_j)
            )
            dmu_i = contrib.dw_dmu_i
            dmu_j = contrib.dw_dmu_j
            wp.atomic_add(
                grad_dipoles,
                atom_i,
                type(mu_i_n)(
                    type(mu_i_n[0])(-weight * dmu_i[0]),
                    type(mu_i_n[0])(-weight * dmu_i[1]),
                    type(mu_i_n[0])(-weight * dmu_i[2]),
                ),
            )
            wp.atomic_add(
                grad_dipoles,
                j,
                type(mu_i_n)(
                    type(mu_i_n[0])(-weight * dmu_j[0]),
                    type(mu_i_n[0])(-weight * dmu_j[1]),
                    type(mu_i_n[0])(-weight * dmu_j[2]),
                ),
            )
            dQi = contrib.dw_dQ_i
            dQj = contrib.dw_dQ_j
            wp.atomic_add(
                grad_quadrupoles,
                atom_i,
                type(Q_i_n)(
                    type(Q_i_n[0, 0])(-weight * dQi[0, 0]),
                    type(Q_i_n[0, 0])(-weight * dQi[0, 1]),
                    type(Q_i_n[0, 0])(-weight * dQi[0, 2]),
                    type(Q_i_n[0, 0])(-weight * dQi[1, 0]),
                    type(Q_i_n[0, 0])(-weight * dQi[1, 1]),
                    type(Q_i_n[0, 0])(-weight * dQi[1, 2]),
                    type(Q_i_n[0, 0])(-weight * dQi[2, 0]),
                    type(Q_i_n[0, 0])(-weight * dQi[2, 1]),
                    type(Q_i_n[0, 0])(-weight * dQi[2, 2]),
                ),
            )
            wp.atomic_add(
                grad_quadrupoles,
                j,
                type(Q_i_n)(
                    type(Q_i_n[0, 0])(-weight * dQj[0, 0]),
                    type(Q_i_n[0, 0])(-weight * dQj[0, 1]),
                    type(Q_i_n[0, 0])(-weight * dQj[0, 2]),
                    type(Q_i_n[0, 0])(-weight * dQj[1, 0]),
                    type(Q_i_n[0, 0])(-weight * dQj[1, 1]),
                    type(Q_i_n[0, 0])(-weight * dQj[1, 2]),
                    type(Q_i_n[0, 0])(-weight * dQj[2, 0]),
                    type(Q_i_n[0, 0])(-weight * dQj[2, 1]),
                    type(Q_i_n[0, 0])(-weight * dQj[2, 2]),
                ),
            )
            dr = contrib.dw_dr_vec
            sgx = weight * dr[0]
            sgy = weight * dr[1]
            sgz = weight * dr[2]
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


def _batch_quadrupole_csr_cell_grad_backward_sig(v, t):
    """Signature builder for the l=2 cell-grad double-backward kernel."""
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=m),  # quadrupoles
        wp.array(dtype=m),  # cells
        wp.array(dtype=wp.int32),  # idx_j
        wp.array(dtype=wp.int32),  # neighbor_ptr
        wp.array(dtype=wp.int32),  # atom_batch_idx
        wp.array(dtype=wp.vec3i),  # unit_shifts
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # alpha
        wp.array(dtype=wp.float64),  # per_direction_scale
        wp.array(dtype=wp.float64),  # grad_energies
        wp.array(dtype=m),  # g_cell
        wp.array(dtype=wp.float64),  # grad_grad_energies (out)
        wp.array(dtype=v),  # grad_positions (out)
        wp.array(dtype=t),  # grad_charges (out)
        wp.array(dtype=v),  # grad_dipoles (out)
        wp.array(dtype=m),  # grad_quadrupoles (out)
        wp.array(dtype=m),  # grad_cell_out (out)
    ]


_batch_quadrupole_csr_cell_grad_backward_overloads = register_overloads(
    _batch_multipole_real_space_quadrupole_csr_cell_grad_backward_kernel,
    _batch_quadrupole_csr_cell_grad_backward_sig,
)


def batch_multipole_real_space_quadrupole_csr_cell_grad_backward(
    positions,
    charges,
    dipoles,
    quadrupoles,
    cells,
    idx_j,
    neighbor_ptr,
    atom_batch_idx,
    unit_shifts,
    sigma,
    alpha,
    per_direction_scale,
    grad_energies,
    g_cell,
    grad_grad_energies,
    grad_positions,
    grad_charges,
    grad_dipoles,
    grad_quadrupoles,
    grad_cell_out,
    wp_dtype,
    device=None,
):
    r"""Launcher for the l=2 cell-grad double-backward (stress-loss).

    Caller pre-zeroes the six output arrays.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(positions.device)
    wp.launch(
        _batch_quadrupole_csr_cell_grad_backward_overloads[vec_dtype],
        dim=positions.shape[0],
        inputs=[
            positions,
            charges,
            dipoles,
            quadrupoles,
            cells,
            idx_j,
            neighbor_ptr,
            atom_batch_idx,
            unit_shifts,
            sigma,
            alpha,
            per_direction_scale,
            grad_energies,
            g_cell,
        ],
        outputs=[
            grad_grad_energies,
            grad_positions,
            grad_charges,
            grad_dipoles,
            grad_quadrupoles,
            grad_cell_out,
        ],
        device=device,
    )


def multipole_real_space_quadrupole_csr_energy_2nd_backward(
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
    gg_positions,
    gg_charges,
    gg_dipoles,
    gg_quadrupoles,
    gg_grad_energies_2nd,
    gg_positions_2nd,
    gg_charges_2nd,
    gg_dipoles_2nd,
    gg_quadrupoles_2nd,
    *,
    device: str,
    half_neighbor_list: bool = False,
):
    r"""CSR single-system :math:`l_{max}=2` second-order backward launcher.

    ``half_neighbor_list`` selects the per-pair scale (0.5 for a half list,
    0.25 for a full list).

    Parameters
    ----------
    positions : shape (N,), vec3 dtype
        Cartesian atom positions.
    charges : shape (N,), scalar dtype
        Per-atom monopole charges.
    dipoles : shape (N,), vec3 dtype
        Per-atom dipole moments.
    quadrupoles : shape (N,), mat33 dtype
        Per-atom symmetric Cartesian quadrupole tensors.
    cell : shape (1,), mat33 dtype
        Single-system unit cell.
    idx_j : shape (nnz,), int32
        CSR neighbor column indices.
    neighbor_ptr : shape (N + 1,), int32
        CSR row pointers.
    unit_shifts : shape (nnz,), vec3i
        Integer lattice shifts per pair.
    sigma : shape (1,), scalar dtype
        Gaussian charge width.
    alpha : shape (1,), scalar dtype
        Ewald splitting parameter.
    grad_energies : shape (N,), float64
        Upstream per-atom energy weights from the first backward.
    gg_positions, gg_charges, gg_dipoles, gg_quadrupoles
        Incoming cotangents (first-backward outputs) w.r.t. the position,
        charge, dipole and quadrupole gradients; shapes/dtypes mirror
        ``positions``/``charges``/``dipoles``/``quadrupoles``.
    gg_grad_energies_2nd : shape (N,), float64
        OUTPUT: second-order gradient w.r.t. ``grad_energies`` (atomic).
    gg_positions_2nd, gg_charges_2nd, gg_dipoles_2nd, gg_quadrupoles_2nd
        OUTPUT: second-order gradients w.r.t. positions, charges, dipoles and
        quadrupoles (atomically accumulated; not zeroed here).
    device : str, keyword-only
        Warp device for the launch.
    half_neighbor_list : bool, keyword-only, default False
        ``True`` half list (scale 0.5); ``False`` full list (scale 0.25).
    """
    overload = _get_quadrupole_2nd_backward_overload(
        "csr", False, positions.dtype, charges.dtype
    )
    n_atoms = positions.shape[0]
    scale_arr = _make_scale_array(half_neighbor_list, device)
    wp.launch(
        overload,
        dim=n_atoms,
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
            scale_arr,
            grad_energies,
            gg_positions,
            gg_charges,
            gg_dipoles,
            gg_quadrupoles,
            gg_grad_energies_2nd,
            gg_positions_2nd,
            gg_charges_2nd,
            gg_dipoles_2nd,
            gg_quadrupoles_2nd,
        ],
        device=device,
    )


def batch_multipole_real_space_quadrupole_csr_energy_2nd_backward(
    positions,
    charges,
    dipoles,
    quadrupoles,
    cells,
    idx_j,
    neighbor_ptr,
    atom_batch_idx,
    unit_shifts,
    sigma,
    alpha,
    grad_energies,
    gg_positions,
    gg_charges,
    gg_dipoles,
    gg_quadrupoles,
    gg_grad_energies_2nd,
    gg_positions_2nd,
    gg_charges_2nd,
    gg_dipoles_2nd,
    gg_quadrupoles_2nd,
    *,
    device: str,
    half_neighbor_list: bool = False,
):
    r"""Batched CSR :math:`l_{max}=2` second-order backward launcher.

    ``half_neighbor_list`` selects the per-pair scale (0.5 for a half list,
    0.25 for a full list).

    Parameters
    ----------
    positions : shape (N,), vec3 dtype
        Cartesian atom positions (all systems).
    charges : shape (N,), scalar dtype
        Per-atom monopole charges.
    dipoles : shape (N,), vec3 dtype
        Per-atom dipole moments.
    quadrupoles : shape (N,), mat33 dtype
        Per-atom symmetric Cartesian quadrupole tensors.
    cells : shape (B,), mat33 dtype
        Per-system unit cells, indexed by ``atom_batch_idx``.
    idx_j : shape (nnz,), int32
        CSR neighbor column indices.
    neighbor_ptr : shape (N + 1,), int32
        CSR row pointers.
    atom_batch_idx : shape (N,), int32
        System index for each atom.
    unit_shifts : shape (nnz,), vec3i
        Integer lattice shifts per pair.
    sigma : shape (1,), scalar dtype
        Gaussian charge width.
    alpha : shape (1,), scalar dtype
        Ewald splitting parameter.
    grad_energies : shape (N,), float64
        Upstream per-atom energy weights from the first backward.
    gg_positions, gg_charges, gg_dipoles, gg_quadrupoles
        Incoming cotangents (first-backward outputs) w.r.t. the position,
        charge, dipole and quadrupole gradients.
    gg_grad_energies_2nd : shape (N,), float64
        OUTPUT: second-order gradient w.r.t. ``grad_energies`` (atomic).
    gg_positions_2nd, gg_charges_2nd, gg_dipoles_2nd, gg_quadrupoles_2nd
        OUTPUT: second-order gradients w.r.t. positions, charges, dipoles and
        quadrupoles (atomic).
    device : str, keyword-only
        Warp device for the launch.
    half_neighbor_list : bool, keyword-only, default False
        ``True`` half list (scale 0.5); ``False`` full list (scale 0.25).
    """
    overload = _get_quadrupole_2nd_backward_overload(
        "csr", True, positions.dtype, charges.dtype
    )
    n_atoms = positions.shape[0]
    scale_arr = _make_scale_array(half_neighbor_list, device)
    wp.launch(
        overload,
        dim=n_atoms,
        inputs=[
            positions,
            charges,
            dipoles,
            quadrupoles,
            cells,
            idx_j,
            neighbor_ptr,
            atom_batch_idx,
            unit_shifts,
            sigma,
            alpha,
            scale_arr,
            grad_energies,
            gg_positions,
            gg_charges,
            gg_dipoles,
            gg_quadrupoles,
            gg_grad_energies_2nd,
            gg_positions_2nd,
            gg_charges_2nd,
            gg_dipoles_2nd,
            gg_quadrupoles_2nd,
        ],
        device=device,
    )
