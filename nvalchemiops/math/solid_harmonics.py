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
Regular and Irregular Solid Harmonics
======================================

This module provides Warp functions for the **regular** and **irregular** real
solid harmonics for angular momentum :math:`L \leq 1`. Solid harmonics bundle
the radial factor with the spherical harmonic and are the natural basis for
multipole expansions — they replace explicit :math:`r^l` / :math:`r^{-(l+1)}`
factors scattered through kernel code.

Definitions
-----------

This module uses the "bare" convention (without the :math:`\sqrt{4\pi/(2l+1)}`
Racah prefactor):

.. math::

    R_l^m(\mathbf{r}) &= r^{l} \cdot Y_l^m(\hat{\mathbf{r}})
    \quad\text{(regular solid harmonic, finite everywhere)} \\
    I_l^m(\mathbf{r}) &= \frac{Y_l^m(\hat{\mathbf{r}})}{r^{l+1}}
    \quad\text{(irregular solid harmonic, singular at } r = 0 \text{)}

With :math:`Y_l^m` in the physics ordering
:math:`(Y_1^{-1}, Y_1^{0}, Y_1^{+1}) \propto (y, z, x)`, the L=1 solid
harmonics become

.. math::

    R_1^{m}(\mathbf{r}) &= \sqrt{\tfrac{3}{4\pi}} \cdot (y, z, x) \\
    I_1^{m}(\mathbf{r}) &= \sqrt{\tfrac{3}{4\pi}} \cdot (y, z, x) / r^{3}.

Use cases
---------

The two canonical identities that justify this module:

.. math::

    Q_{l,m} &= \sum_j q_j \cdot R_l^m(\mathbf{r}_j)
    \quad\text{(multipole moment of a discrete charge distribution)} \\
    V(\mathbf{r}) &= \sum_{l,m} Q_{l,m} \cdot I_l^m(\mathbf{r})
    \quad\text{(multipole potential at } \mathbf{r} \text{ due to moments at origin)}

Conventions & caveats
---------------------

- **Racah prefactor:** this module uses the bare ``r^l · Y_l^m`` form. To recover
  the Racah-normalized variant ``R_l^m^{(R)} = \sqrt{4\pi/(2l+1)} · r^l · Y_l^m``
  — as used, for example, by the customer reference's real-space ``R_l^m``
  evaluation —
  multiply the output of :func:`regular_solid_harmonic_l1` by
  :math:`\sqrt{4\pi/3}`.
- **Irregular harmonics are singular at the origin.** ``irregular_solid_harmonic_*``
  functions apply a tiny ``r² + _EPSILON`` safety floor (matching the pattern
  elsewhere in ``nvalchemiops.math``) so they cannot produce raw NaN/inf during
  autograd. Values very close to :math:`r = 0` are still unphysically large —
  callers are responsible for avoiding the origin when the physics demands it.
- **Scope:** only L=0 and L=1 are implemented here. L=2 and L=3 extensions
  slot in alongside the spherical harmonics / GTO extensions for those
  orders.
- **Gradients** of the solid harmonics are not yet provided. Downstream
  multipole kernels compute forces via interaction-tensor derivatives of the
  damped Coulomb kernel (``erfc(αr)/r``) rather than direct gradients of these
  basis functions, so the gradient routines are left for later phases.
"""

from __future__ import annotations

import warp as wp

from nvalchemiops.math.spherical_harmonics import Y00_COEFF, Y1_COEFF

# Tiny additive floor keeping 1/r, 1/r^3, etc. finite at r == 0. Matches the
# ``EPSILON`` convention used elsewhere in ``nvalchemiops.math``.
_EPSILON = wp.constant(wp.float64(1e-30))


# =============================================================================
# Regular Solid Harmonics R_l^m(r) = r^l · Y_l^m(r̂)
# =============================================================================


@wp.func
def regular_solid_harmonic_l0(r: wp.vec3d) -> wp.float64:
    r"""Regular solid harmonic :math:`R_0^0(\mathbf{r}) = Y_0^0 = 1 / \sqrt{4\pi}`.

    Constant across all positions, including the origin.
    """
    # Warp-decorated funcs must use `r` in their body so the compiler does not
    # strip the argument during kernel lowering.
    _ = r
    return Y00_COEFF


@wp.func
def regular_solid_harmonic_l1(r: wp.vec3d) -> wp.vec3d:
    r"""Regular solid harmonics :math:`R_1^{m}(\mathbf{r}) = r \cdot Y_1^{m}(\hat{\mathbf{r}})`.

    Returns the three components ``(R_1^{-1}, R_1^{0}, R_1^{+1})`` which evaluate
    to :math:`\sqrt{3/(4\pi)} \cdot (y, z, x)`. Vanishes smoothly at the origin.
    """
    return wp.vec3d(
        Y1_COEFF * r[1],  # m = -1: y
        Y1_COEFF * r[2],  # m =  0: z
        Y1_COEFF * r[0],  # m = +1: x
    )


# =============================================================================
# Irregular Solid Harmonics I_l^m(r) = Y_l^m(r̂) / r^{l+1}
# =============================================================================


@wp.func
def irregular_solid_harmonic_l0(r: wp.vec3d) -> wp.float64:
    r"""Irregular solid harmonic :math:`I_0^0(\mathbf{r}) = Y_0^0 / r = 1 / (\sqrt{4\pi} \cdot r)`.

    Singular at the origin; a small ``_EPSILON`` floor prevents NaN/inf, but
    values near the origin are not physically meaningful.
    """
    r_norm = wp.sqrt(wp.dot(r, r) + _EPSILON)
    return Y00_COEFF / r_norm


@wp.func
def irregular_solid_harmonic_l1(r: wp.vec3d) -> wp.vec3d:
    r"""Irregular solid harmonics :math:`I_1^{m}(\mathbf{r}) = Y_1^{m}(\hat{\mathbf{r}}) / r^{2}`.

    Equivalent to :math:`\sqrt{3/(4\pi)} \cdot (y, z, x) / r^{3}`. Singular at
    the origin; a small ``_EPSILON`` floor prevents NaN/inf, but values near the
    origin are not physically meaningful.
    """
    r2 = wp.dot(r, r) + _EPSILON
    r_norm = wp.sqrt(r2)
    inv_r3 = wp.float64(1.0) / (r2 * r_norm)
    return wp.vec3d(
        Y1_COEFF * r[1] * inv_r3,  # m = -1
        Y1_COEFF * r[2] * inv_r3,  # m =  0
        Y1_COEFF * r[0] * inv_r3,  # m = +1
    )


# =============================================================================
# Warp launch kernels
# =============================================================================


@wp.kernel
def _eval_regular_solid_harmonics_kernel(
    positions: wp.array(dtype=wp.vec3d),
    max_L: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Evaluate regular solid harmonics :math:`R_l^m` up to ``max_L``.

    Launch Grid
    -----------
    dim = [N] — one thread per position.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3d), shape (N,)
        Input positions.
    max_L : int
        Maximum angular momentum. Currently 0 or 1.
    output : wp.array2d(dtype=wp.float64), shape (N, num_components)
        Output values laid out as ``[R_0^0, R_1^{-1}, R_1^{0}, R_1^{+1}]``,
        truncated to the first ``(max_L + 1)**2`` columns.
    """
    i = wp.tid()
    r = positions[i]
    output[i, 0] = regular_solid_harmonic_l0(r)
    if max_L >= 1:
        r1 = regular_solid_harmonic_l1(r)
        output[i, 1] = r1[0]
        output[i, 2] = r1[1]
        output[i, 3] = r1[2]


@wp.kernel
def _eval_irregular_solid_harmonics_kernel(
    positions: wp.array(dtype=wp.vec3d),
    max_L: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Evaluate irregular solid harmonics :math:`I_l^m` up to ``max_L``.

    Launch Grid
    -----------
    dim = [N] — one thread per position.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3d), shape (N,)
        Input positions. Must have ``|r| > 0``; the irregular harmonics are
        singular at the origin and produce inf/NaN there.
    max_L : int
        Maximum angular momentum. Currently 0 or 1.
    output : wp.array2d(dtype=wp.float64), shape (N, num_components)
        Output values laid out as ``[I_0^0, I_1^{-1}, I_1^{0}, I_1^{+1}]``,
        truncated to the first ``(max_L + 1)**2`` columns.
    """
    i = wp.tid()
    r = positions[i]
    output[i, 0] = irregular_solid_harmonic_l0(r)
    if max_L >= 1:
        i1 = irregular_solid_harmonic_l1(r)
        output[i, 1] = i1[0]
        output[i, 2] = i1[1]
        output[i, 3] = i1[2]
