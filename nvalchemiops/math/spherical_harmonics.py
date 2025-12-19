# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

r"""
Real Spherical Harmonics Implementation
=======================================

This module provides Warp functions for evaluating real spherical harmonics
:math:`Y_l^m(r)` for angular momentum :math:`L \leq 2` (monopole, dipole, quadrupole).

Mathematical Background
-----------------------

Real spherical harmonics are defined as linear combinations of complex
spherical harmonics to produce real-valued basis functions. For a unit vector
:math:`\hat{r} = (x, y, z)/|r| = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)`:

**L = 0 (Monopole):**

.. math::

    Y_0^0 = \frac{1}{\sqrt{4\pi}}

**L = 1 (Dipole):**

.. math::

    \begin{aligned}
    Y_1^{-1} &= \sqrt{\frac{3}{4\pi}} \cdot \frac{y}{r} \\
    Y_1^0 &= \sqrt{\frac{3}{4\pi}} \cdot \frac{z}{r} \\
    Y_1^{+1} &= \sqrt{\frac{3}{4\pi}} \cdot \frac{x}{r}
    \end{aligned}

**L = 2 (Quadrupole):**

.. math::

    \begin{aligned}
    Y_2^{-2} &= \sqrt{\frac{15}{4\pi}} \cdot \frac{xy}{r^2} \\
    Y_2^{-1} &= \sqrt{\frac{15}{4\pi}} \cdot \frac{yz}{r^2} \\
    Y_2^0 &= \sqrt{\frac{5}{16\pi}} \cdot \frac{3z^2 - r^2}{r^2} = \sqrt{\frac{5}{16\pi}} \cdot \frac{2z^2 - x^2 - y^2}{r^2} \\
    Y_2^{+1} &= \sqrt{\frac{15}{4\pi}} \cdot \frac{xz}{r^2} \\
    Y_2^{+2} &= \sqrt{\frac{15}{16\pi}} \cdot \frac{x^2 - y^2}{r^2}
    \end{aligned}

Conventions
-----------

1. **Normalization**: We use orthonormal real spherical harmonics (integral
   normalization), meaning :math:`\int Y_l^m Y_{l'}^{m'} d\Omega = \delta_{ll'} \delta_{mm'}`.

2. **Ordering**: For each L, m ranges from -L to +L:
   - L=0: [:math:`Y_0^0`]                           (1 component)
   - L=1: [:math:`Y_1^{-1}, Y_1^0, Y_1^{+1}`]       (3 components)
   - L=2: [:math:`Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^{+1}, Y_2^{+2}`]  (5 components)

3. **Input**: Functions accept a 3D vector r (not necessarily normalized).
   The harmonics are evaluated for the direction :math:`\hat{r} = r/|r|`.

4. **Singularity at origin**: When r = 0, we return 0 for all harmonics
   (except :math:`Y_0^0` which is constant, but we still return 0 for consistency
   with gradient calculations).

Usage in Warp Kernels
---------------------

These are Warp functions (decorated with @wp.func) designed to be called
from within Warp kernels::

    @wp.kernel
    def my_kernel(positions: wp.array(dtype=wp.vec3d), output: wp.array(dtype=wp.float64)):
        i = wp.tid()
        r = positions[i]

        # Evaluate single harmonic
        y10 = spherical_harmonic_10(r)

        # Evaluate all L=1 harmonics
        y1_all = eval_spherical_harmonics_l1(r)  # Returns (3,) tuple

        # Evaluate all harmonics up to L=2
        y_all = eval_all_spherical_harmonics(r)  # Returns (9,) tuple

References
----------
- Blanco, M. A., Flórez, M., & Bermejo, M. (1997). "Evaluation of the rotation
  matrices in the basis of real spherical harmonics." Journal of Molecular
  Structure: THEOCHEM, 419(1-3), 19-27.
- https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
"""

from __future__ import annotations

import warp as wp

# =============================================================================
# Mathematical Constants
# =============================================================================

# Normalization constants for real spherical harmonics
# These are computed as sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!) with appropriate
# factors for the real combination

# L=0: sqrt(1/(4pi))
Y00_COEFF = wp.constant(wp.float64(0.28209479177387814))  # 1/sqrt(4pi)

# L=1: sqrt(3/(4pi))
Y1_COEFF = wp.constant(wp.float64(0.4886025119029199))  # sqrt(3/(4pi))

# L=2 coefficients
Y2_M2_COEFF = wp.constant(wp.float64(1.0925484305920792))  # sqrt(15/(4pi))
Y2_M1_COEFF = wp.constant(wp.float64(1.0925484305920792))  # sqrt(15/(4pi))
Y2_0_COEFF = wp.constant(wp.float64(0.31539156525252005))  # sqrt(5/(16pi))
Y2_P1_COEFF = wp.constant(wp.float64(1.0925484305920792))  # sqrt(15/(4pi))
Y2_P2_COEFF = wp.constant(wp.float64(0.5462742152960396))  # sqrt(15/(16pi))

# Small value for safe division
EPSILON = wp.constant(wp.float64(1e-30))


# =============================================================================
# L = 0 (Monopole) - 1 component
# =============================================================================
@wp.func
def rsqrt(x: wp.float64) -> wp.float64:
    return wp.float64(1.0) / wp.sqrt(x)


@wp.func
def spherical_harmonic_00(r: wp.vec3d) -> wp.float64:
    r"""Compute Y_0^0 (monopole).

    .. math::

        Y_0^0 = \frac{1}{\sqrt{4\pi}}

    This is a constant function (no angular dependence).

    Parameters
    ----------
    r : wp.vec3d
        Position vector (direction is extracted, magnitude ignored).

    Returns
    -------
    wp.float64
        Value of Y_0^0.
    """
    return Y00_COEFF


@wp.func
def spherical_harmonic_00_gradient(r: wp.vec3d) -> wp.vec3d:
    """Gradient of Y_0^0 with respect to r.

    Since :math:`Y_0^0` is constant, its gradient is zero.
    """
    return wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))


# =============================================================================
# L = 1 (Dipole) - 3 components
# =============================================================================


@wp.func
def spherical_harmonic_1m1(r: wp.vec3d) -> wp.float64:
    r"""Compute Y_1^{-1} (dipole, m=-1).

    .. math::

        Y_1^{-1} = \sqrt{\frac{3}{4\pi}} \cdot \frac{y}{r}

    Parameters
    ----------
    r : wp.vec3d
        Position vector.

    Returns
    -------
    wp.float64
        Value of Y_1^{-1}.
    """
    r2 = wp.dot(r, r)
    r_inv = rsqrt(r2 + EPSILON)
    return Y1_COEFF * r[1] * r_inv


@wp.func
def spherical_harmonic_1m1_gradient(r: wp.vec3d) -> wp.vec3d:
    r"""Gradient of Y_1^{-1} with respect to r.

    .. math::

        \frac{\partial}{\partial r} \left[\frac{y}{r}\right] = \frac{r^2 \hat{e}_y - y \cdot r}{r^3}

    where :math:`\hat{e}_y` is the unit vector in y direction.
    """
    r2 = wp.dot(r, r)
    r2_safe = r2 + EPSILON
    r_inv = rsqrt(r2_safe)
    r3_inv = r_inv * r_inv * r_inv

    # Gradient of y/r
    # d(y/r)/dx = -xy/r^3
    # d(y/r)/dy = (r^2 - y^2)/r^3 = 1/r - y^2/r^3
    # d(y/r)/dz = -yz/r^3
    x, y, z = r[0], r[1], r[2]

    grad_x = -x * y * r3_inv
    grad_y = (r2_safe - y * y) * r3_inv
    grad_z = -z * y * r3_inv

    return wp.vec3d(Y1_COEFF * grad_x, Y1_COEFF * grad_y, Y1_COEFF * grad_z)


@wp.func
def spherical_harmonic_10(r: wp.vec3d) -> wp.float64:
    r"""Compute Y_1^0 (dipole, m=0).

    .. math::

        Y_1^0 = \sqrt{\frac{3}{4\pi}} \cdot \frac{z}{r}

    Parameters
    ----------
    r : wp.vec3d
        Position vector.

    Returns
    -------
    wp.float64
        Value of Y_1^0.
    """
    r2 = wp.dot(r, r)
    r_inv = rsqrt(r2 + EPSILON)
    return Y1_COEFF * r[2] * r_inv


@wp.func
def spherical_harmonic_10_gradient(r: wp.vec3d) -> wp.vec3d:
    r"""Gradient of Y_1^0 with respect to r.

    .. math::

        \frac{\partial}{\partial r} \left[\frac{z}{r}\right] = \frac{r^2 \hat{e}_z - z \cdot r}{r^3}
    """
    r2 = wp.dot(r, r)
    r2_safe = r2 + EPSILON
    r_inv = rsqrt(r2_safe)
    r3_inv = r_inv * r_inv * r_inv

    x, y, z = r[0], r[1], r[2]

    grad_x = -x * z * r3_inv
    grad_y = -y * z * r3_inv
    grad_z = (r2_safe - z * z) * r3_inv

    return wp.vec3d(Y1_COEFF * grad_x, Y1_COEFF * grad_y, Y1_COEFF * grad_z)


@wp.func
def spherical_harmonic_1p1(r: wp.vec3d) -> wp.float64:
    r"""Compute Y_1^{+1} (dipole, m=+1).

    .. math::

        Y_1^{+1} = \sqrt{\frac{3}{4\pi}} \cdot \frac{x}{r}

    Parameters
    ----------
    r : wp.vec3d
        Position vector.

    Returns
    -------
    wp.float64
        Value of Y_1^{+1}.
    """
    r2 = wp.dot(r, r)
    r_inv = rsqrt(r2 + EPSILON)
    return Y1_COEFF * r[0] * r_inv


@wp.func
def spherical_harmonic_1p1_gradient(r: wp.vec3d) -> wp.vec3d:
    r"""Gradient of Y_1^{+1} with respect to r.

    .. math::

        \frac{\partial}{\partial r} \left[\frac{x}{r}\right] = \frac{r^2 \hat{e}_x - x \cdot r}{r^3}
    """
    r2 = wp.dot(r, r)
    r2_safe = r2 + EPSILON
    r_inv = rsqrt(r2_safe)
    r3_inv = r_inv * r_inv * r_inv

    x, y, z = r[0], r[1], r[2]

    grad_x = (r2_safe - x * x) * r3_inv
    grad_y = -y * x * r3_inv
    grad_z = -z * x * r3_inv

    return wp.vec3d(Y1_COEFF * grad_x, Y1_COEFF * grad_y, Y1_COEFF * grad_z)


# =============================================================================
# L = 2 (Quadrupole) - 5 components
# =============================================================================


@wp.func
def spherical_harmonic_2m2(r: wp.vec3d) -> wp.float64:
    r"""Compute Y_2^{-2} (quadrupole, m=-2).

    .. math::

        Y_2^{-2} = \sqrt{\frac{15}{4\pi}} \cdot \frac{xy}{r^2}

    Parameters
    ----------
    r : wp.vec3d
        Position vector.

    Returns
    -------
    wp.float64
        Value of Y_2^{-2}.
    """
    r2 = wp.dot(r, r)
    r2_inv = wp.float64(1.0) / (r2 + EPSILON)
    return Y2_M2_COEFF * r[0] * r[1] * r2_inv


@wp.func
def spherical_harmonic_2m2_gradient(r: wp.vec3d) -> wp.vec3d:
    r"""Gradient of Y_2^{-2} with respect to r.

    .. math::

        \frac{\partial}{\partial r} \left[\frac{xy}{r^2}\right] = \frac{r^2 (y \hat{e}_x + x \hat{e}_y) - 2xy \cdot r}{r^4}
    """
    r2 = wp.dot(r, r)
    r2_safe = r2 + EPSILON
    r2_inv = wp.float64(1.0) / r2_safe
    r4_inv = r2_inv * r2_inv

    x, y, z = r[0], r[1], r[2]
    xy = x * y

    # d(xy/r^2)/dx = y/r^2 - 2x*xy/r^4 = (yr^2 - 2x^2*y)/r^4 = y(r^2 - 2x^2)/r^4
    # d(xy/r^2)/dy = x/r^2 - 2y*xy/r^4 = (xr^2 - 2y^2*x)/r^4 = x(r^2 - 2y^2)/r^4
    # d(xy/r^2)/dz = -2z*xy/r^4

    grad_x = y * (r2_safe - wp.float64(2.0) * x * x) * r4_inv
    grad_y = x * (r2_safe - wp.float64(2.0) * y * y) * r4_inv
    grad_z = -wp.float64(2.0) * z * xy * r4_inv

    return wp.vec3d(Y2_M2_COEFF * grad_x, Y2_M2_COEFF * grad_y, Y2_M2_COEFF * grad_z)


@wp.func
def spherical_harmonic_2m1(r: wp.vec3d) -> wp.float64:
    r"""Compute Y_2^{-1} (quadrupole, m=-1).

    .. math::

        Y_2^{-1} = \sqrt{\frac{15}{4\pi}} \cdot \frac{yz}{r^2}

    Parameters
    ----------
    r : wp.vec3d
        Position vector.

    Returns
    -------
    wp.float64
        Value of Y_2^{-1}.
    """
    r2 = wp.dot(r, r)
    r2_inv = wp.float64(1.0) / (r2 + EPSILON)
    return Y2_M1_COEFF * r[1] * r[2] * r2_inv


@wp.func
def spherical_harmonic_2m1_gradient(r: wp.vec3d) -> wp.vec3d:
    r"""Gradient of Y_2^{-1} with respect to r.

    .. math::

        \frac{\partial}{\partial r} \left[\frac{yz}{r^2}\right]
    """
    r2 = wp.dot(r, r)
    r2_safe = r2 + EPSILON
    r2_inv = wp.float64(1.0) / r2_safe
    r4_inv = r2_inv * r2_inv

    x, y, z = r[0], r[1], r[2]
    yz = y * z

    grad_x = -wp.float64(2.0) * x * yz * r4_inv
    grad_y = z * (r2_safe - wp.float64(2.0) * y * y) * r4_inv
    grad_z = y * (r2_safe - wp.float64(2.0) * z * z) * r4_inv

    return wp.vec3d(Y2_M1_COEFF * grad_x, Y2_M1_COEFF * grad_y, Y2_M1_COEFF * grad_z)


@wp.func
def spherical_harmonic_20(r: wp.vec3d) -> wp.float64:
    r"""Compute Y_2^0 (quadrupole, m=0).

    .. math::

        Y_2^0 = \sqrt{\frac{5}{16\pi}} \cdot \frac{3z^2 - r^2}{r^2} = \sqrt{\frac{5}{16\pi}} \cdot \frac{2z^2 - x^2 - y^2}{r^2}

    Parameters
    ----------
    r : wp.vec3d
        Position vector.

    Returns
    -------
    wp.float64
        Value of Y_2^0.
    """
    r2 = wp.dot(r, r)
    r2_inv = wp.float64(1.0) / (r2 + EPSILON)
    z2 = r[2] * r[2]
    # 3z^2 - r^2 = 3z^2 - (x^2 + y^2 + z^2) = 2z^2 - x^2 - y^2
    return Y2_0_COEFF * (wp.float64(3.0) * z2 - r2) * r2_inv


@wp.func
def spherical_harmonic_20_gradient(r: wp.vec3d) -> wp.vec3d:
    r"""Gradient of Y_2^0 with respect to r.

    .. math::

        Y_2^0 = C \cdot \frac{3z^2 - r^2}{r^2}

    Let :math:`f = \frac{3z^2 - r^2}{r^2} = \frac{3z^2}{r^2} - 1`

    .. math::

        \begin{aligned}
        \frac{\partial f}{\partial x} &= -\frac{6xz^2}{r^4} \\
        \frac{\partial f}{\partial y} &= -\frac{6yz^2}{r^4} \\
        \frac{\partial f}{\partial z} &= \frac{6z}{r^2} - \frac{6z^3}{r^4} = \frac{6z(r^2 - z^2)}{r^4}
        \end{aligned}
    """
    r2 = wp.dot(r, r)
    r2_safe = r2 + EPSILON
    r2_inv = wp.float64(1.0) / r2_safe
    r4_inv = r2_inv * r2_inv

    x, y, z = r[0], r[1], r[2]
    z2 = z * z

    grad_x = -wp.float64(6.0) * x * z2 * r4_inv
    grad_y = -wp.float64(6.0) * y * z2 * r4_inv
    grad_z = wp.float64(6.0) * z * (r2_safe - z2) * r4_inv

    return wp.vec3d(Y2_0_COEFF * grad_x, Y2_0_COEFF * grad_y, Y2_0_COEFF * grad_z)


@wp.func
def spherical_harmonic_2p1(r: wp.vec3d) -> wp.float64:
    r"""Compute Y_2^{+1} (quadrupole, m=+1).

    .. math::

        Y_2^{+1} = \sqrt{\frac{15}{4\pi}} \cdot \frac{xz}{r^2}

    Parameters
    ----------
    r : wp.vec3d
        Position vector.

    Returns
    -------
    wp.float64
        Value of Y_2^{+1}.
    """
    r2 = wp.dot(r, r)
    r2_inv = wp.float64(1.0) / (r2 + EPSILON)
    return Y2_P1_COEFF * r[0] * r[2] * r2_inv


@wp.func
def spherical_harmonic_2p1_gradient(r: wp.vec3d) -> wp.vec3d:
    r"""Gradient of Y_2^{+1} with respect to r.

    .. math::

        \frac{\partial}{\partial r} \left[\frac{xz}{r^2}\right]
    """
    r2 = wp.dot(r, r)
    r2_safe = r2 + EPSILON
    r2_inv = wp.float64(1.0) / r2_safe
    r4_inv = r2_inv * r2_inv

    x, y, z = r[0], r[1], r[2]
    xz = x * z

    grad_x = z * (r2_safe - wp.float64(2.0) * x * x) * r4_inv
    grad_y = -wp.float64(2.0) * y * xz * r4_inv
    grad_z = x * (r2_safe - wp.float64(2.0) * z * z) * r4_inv

    return wp.vec3d(Y2_P1_COEFF * grad_x, Y2_P1_COEFF * grad_y, Y2_P1_COEFF * grad_z)


@wp.func
def spherical_harmonic_2p2(r: wp.vec3d) -> wp.float64:
    r"""Compute Y_2^{+2} (quadrupole, m=+2).

    .. math::

        Y_2^{+2} = \sqrt{\frac{15}{16\pi}} \cdot \frac{x^2 - y^2}{r^2}

    Parameters
    ----------
    r : wp.vec3d
        Position vector.

    Returns
    -------
    wp.float64
        Value of Y_2^{+2}.
    """
    r2 = wp.dot(r, r)
    r2_inv = wp.float64(1.0) / (r2 + EPSILON)
    return Y2_P2_COEFF * (r[0] * r[0] - r[1] * r[1]) * r2_inv


@wp.func
def spherical_harmonic_2p2_gradient(r: wp.vec3d) -> wp.vec3d:
    r"""Gradient of Y_2^{+2} with respect to r.

    .. math::

        \frac{\partial}{\partial r} \left[\frac{x^2 - y^2}{r^2}\right]

    Let :math:`f = \frac{x^2 - y^2}{r^2}`

    .. math::

        \begin{aligned}
        \frac{\partial f}{\partial x} &= \frac{2x}{r^2} - \frac{2x(x^2 - y^2)}{r^4} = \frac{2x(r^2 - x^2 + y^2)}{r^4} = \frac{2x(2y^2 + z^2)}{r^4} \\
        \frac{\partial f}{\partial y} &= -\frac{2y}{r^2} - \frac{2y(x^2 - y^2)}{r^4} = -\frac{2y(r^2 + x^2 - y^2)}{r^4} = -\frac{2y(2x^2 + z^2)}{r^4} \\
        \frac{\partial f}{\partial z} &= -\frac{2z(x^2 - y^2)}{r^4}
        \end{aligned}
    """
    r2 = wp.dot(r, r)
    r2_safe = r2 + EPSILON
    r2_inv = wp.float64(1.0) / r2_safe
    r4_inv = r2_inv * r2_inv

    x, y, z = r[0], r[1], r[2]
    x2, y2, z2 = x * x, y * y, z * z

    grad_x = wp.float64(2.0) * x * (wp.float64(2.0) * y2 + z2) * r4_inv
    grad_y = -wp.float64(2.0) * y * (wp.float64(2.0) * x2 + z2) * r4_inv
    grad_z = -wp.float64(2.0) * z * (x2 - y2) * r4_inv

    return wp.vec3d(Y2_P2_COEFF * grad_x, Y2_P2_COEFF * grad_y, Y2_P2_COEFF * grad_z)


# =============================================================================
# Vectorized Evaluators
# =============================================================================

# Define a type for returning all 9 harmonics (L=0,1,2)
# Using individual returns since Warp doesn't support variable-length arrays in functions


@wp.func
def eval_spherical_harmonics_l0(r: wp.vec3d) -> wp.float64:
    """Evaluate all L=0 harmonics (just Y_0^0).

    Returns
    -------
    wp.float64
        Y_0^0 value.
    """
    return spherical_harmonic_00(r)


@wp.func
def eval_spherical_harmonics_l1(r: wp.vec3d) -> wp.vec3d:
    """Evaluate all L=1 harmonics.

    Returns
    -------
    wp.vec3d
        (Y_1^{-1}, Y_1^0, Y_1^{+1}) values.
    """
    r2 = wp.dot(r, r)
    r_inv = rsqrt(r2 + EPSILON)

    return wp.vec3d(
        Y1_COEFF * r[1] * r_inv,  # Y_1^{-1}
        Y1_COEFF * r[2] * r_inv,  # Y_1^0
        Y1_COEFF * r[0] * r_inv,  # Y_1^{+1}
    )


@wp.func
def eval_spherical_harmonics_l2(r: wp.vec3d) -> wp.vec(length=5, dtype=wp.float64):
    """Evaluate all L=2 harmonics.

    Returns
    -------
    wp.vec(5)
        (Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^{+1}, Y_2^{+2}) values.
    """
    r2 = wp.dot(r, r)
    r2_inv = wp.float64(1.0) / (r2 + EPSILON)

    x, y, z = r[0], r[1], r[2]
    x2, y2, z2 = x * x, y * y, z * z

    return wp.vec(
        Y2_M2_COEFF * x * y * r2_inv,  # Y_2^{-2}
        Y2_M1_COEFF * y * z * r2_inv,  # Y_2^{-1}
        Y2_0_COEFF * (wp.float64(3.0) * z2 - r2) * r2_inv,  # Y_2^0
        Y2_P1_COEFF * x * z * r2_inv,  # Y_2^{+1}
        Y2_P2_COEFF * (x2 - y2) * r2_inv,  # Y_2^{+2}
        dtype=wp.float64,
    )


@wp.func
def eval_all_spherical_harmonics(r: wp.vec3d) -> wp.vec(length=9, dtype=wp.float64):
    """Evaluate all spherical harmonics up to L=2.

    Returns
    -------
    wp.vec(9)
        All harmonics in order:
        [Y_0^0, Y_1^{-1}, Y_1^0, Y_1^{+1}, Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^{+1}, Y_2^{+2}]
    """
    r2 = wp.dot(r, r)
    r2_safe = r2 + EPSILON
    r_inv = rsqrt(r2_safe)
    r2_inv = wp.float64(1.0) / r2_safe

    x, y, z = r[0], r[1], r[2]
    x2, y2, z2 = x * x, y * y, z * z

    return wp.vec(
        # L=0
        Y00_COEFF,  # Y_0^0
        # L=1
        Y1_COEFF * y * r_inv,  # Y_1^{-1}
        Y1_COEFF * z * r_inv,  # Y_1^0
        Y1_COEFF * x * r_inv,  # Y_1^{+1}
        # L=2
        Y2_M2_COEFF * x * y * r2_inv,  # Y_2^{-2}
        Y2_M1_COEFF * y * z * r2_inv,  # Y_2^{-1}
        Y2_0_COEFF * (wp.float64(3.0) * z2 - r2_safe) * r2_inv,  # Y_2^0
        Y2_P1_COEFF * x * z * r2_inv,  # Y_2^{+1}
        Y2_P2_COEFF * (x2 - y2) * r2_inv,  # Y_2^{+2}
        dtype=wp.float64,
    )


# =============================================================================
# PyTorch Wrappers for Testing
# =============================================================================

# These wrappers allow the spherical harmonics to be tested from PyTorch


@wp.kernel
def _eval_spherical_harmonics_kernel(
    positions: wp.array(dtype=wp.vec3d),
    L_max: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Evaluate real spherical harmonics Y_l^m at multiple positions.

    Computes orthonormalized real spherical harmonics up to angular momentum L_max.
    For each position r, evaluates Y_l^m(r̂) where r̂ = r/|r| is the unit direction.

    Launch Grid
    -----------
    dim = [N]

    One thread per position.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3d), shape (N,)
        Input position vectors (direction extracted, magnitude ignored for L=0).
    L_max : int
        Maximum angular momentum quantum number (0, 1, or 2).
    output : wp.array2d(dtype=wp.float64), shape (N, num_components)
        Output array for spherical harmonic values where num_components is:
        - L_max=0: 1 component  [Y_0^0]
        - L_max=1: 4 components [Y_0^0, Y_1^{-1}, Y_1^0, Y_1^{+1}]
        - L_max=2: 9 components [Y_0^0, Y_1^{-1}, Y_1^0, Y_1^{+1},
                                 Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^{+1}, Y_2^{+2}]

    Notes
    -----
    - Uses orthonormal real spherical harmonics with integral normalization.
    - At the origin (r=0), returns 0 for L>0 harmonics (singularity handled via EPSILON).
    - L=0 (monopole) is constant: Y_0^0 = 1/√(4π).
    - L=1 (dipole) scales as r̂ components.
    - L=2 (quadrupole) scales as products of r̂ components.
    """
    i = wp.tid()
    r = positions[i]

    # Always compute L=0
    output[i, 0] = spherical_harmonic_00(r)

    if L_max >= 1:
        y1 = eval_spherical_harmonics_l1(r)
        output[i, 1] = y1[0]
        output[i, 2] = y1[1]
        output[i, 3] = y1[2]

    if L_max >= 2:
        y2 = eval_spherical_harmonics_l2(r)
        output[i, 4] = y2[0]
        output[i, 5] = y2[1]
        output[i, 6] = y2[2]
        output[i, 7] = y2[3]
        output[i, 8] = y2[4]


@wp.kernel
def _eval_spherical_harmonics_gradient_kernel(
    positions: wp.array(dtype=wp.vec3d),
    L_max: int,
    output: wp.array(dtype=wp.vec3d, ndim=2),
):
    """Evaluate gradients of real spherical harmonics with respect to position.

    Computes ∇_r Y_l^m(r) for all harmonics up to L_max at each position.
    These gradients are useful for force calculations and sensitivity analysis.

    Launch Grid
    -----------
    dim = [N]

    One thread per position.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3d), shape (N,)
        Input position vectors.
    L_max : int
        Maximum angular momentum quantum number (0, 1, or 2).
    output : wp.array(dtype=wp.vec3d, ndim=2), shape (N, num_components)
        Output gradient vectors where each entry is a 3D gradient ∇Y_l^m.
        Number of components follows same convention as _eval_spherical_harmonics_kernel.

    Notes
    -----
    - Gradient of Y_0^0 is always zero (constant function).
    - Gradients computed analytically using chain rule on r/|r| terms.
    - Near origin (r→0), gradients may have numerical issues but are regularized.
    - Output shape: (N, num_components, 3) when viewed as torch tensor.
    """
    i = wp.tid()
    r = positions[i]

    # Always compute L=0
    output[i, 0] = spherical_harmonic_00_gradient(r)

    if L_max >= 1:
        output[i, 1] = spherical_harmonic_1m1_gradient(r)
        output[i, 2] = spherical_harmonic_10_gradient(r)
        output[i, 3] = spherical_harmonic_1p1_gradient(r)

    if L_max >= 2:
        output[i, 4] = spherical_harmonic_2m2_gradient(r)
        output[i, 5] = spherical_harmonic_2m1_gradient(r)
        output[i, 6] = spherical_harmonic_20_gradient(r)
        output[i, 7] = spherical_harmonic_2p1_gradient(r)
        output[i, 8] = spherical_harmonic_2p2_gradient(r)


def eval_spherical_harmonics_pytorch(
    positions,
    L_max: int = 2,
    device=None,
):
    """Evaluate spherical harmonics from PyTorch tensors.

    This is a convenience wrapper for testing the Warp implementation.

    Parameters
    ----------
    positions : torch.Tensor
        Input positions [N, 3] as float64.
    L_max : int
        Maximum angular momentum (0, 1, or 2). Default: 2.
    device : torch.device, optional
        Device for computation.

    Returns
    -------
    torch.Tensor
        Spherical harmonic values [N, num_components] where num_components is
        1 (L_max=0), 4 (L_max=1), or 9 (L_max=2).
    """
    import torch

    if device is None:
        device = positions.device

    N = positions.shape[0]

    # Number of components for each L_max
    num_components = {0: 1, 1: 4, 2: 9}[L_max]

    # Allocate output
    output = torch.zeros((N, num_components), dtype=torch.float64, device=device)

    # Convert to Warp arrays
    wp_device = wp.device_from_torch(device)
    wp_positions = wp.from_torch(positions.contiguous(), dtype=wp.vec3d)
    wp_output = wp.from_torch(output, dtype=wp.float64)

    # Launch kernel
    wp.launch(
        kernel=_eval_spherical_harmonics_kernel,
        dim=N,
        inputs=[wp_positions, L_max],
        outputs=[wp_output],
        device=wp_device,
    )

    return output


def eval_spherical_harmonics_gradient_pytorch(
    positions,
    L_max: int = 2,
    device=None,
):
    """Evaluate spherical harmonic gradients from PyTorch tensors.

    Parameters
    ----------
    positions : torch.Tensor
        Input positions [N, 3] as float64.
    L_max : int
        Maximum angular momentum (0, 1, or 2). Default: 2.
    device : torch.device, optional
        Device for computation.

    Returns
    -------
    torch.Tensor
        Gradient vectors [N, num_components, 3].
    """
    import torch

    if device is None:
        device = positions.device

    N = positions.shape[0]

    # Number of components for each L_max
    num_components = {0: 1, 1: 4, 2: 9}[L_max]

    # Allocate output
    output = torch.zeros((N, num_components, 3), dtype=torch.float64, device=device)

    # Convert to Warp arrays
    wp_device = wp.device_from_torch(device)
    wp_positions = wp.from_torch(positions.contiguous(), dtype=wp.vec3d)
    wp_output = wp.from_torch(output, dtype=wp.vec3d)

    # Launch kernel
    wp.launch(
        kernel=_eval_spherical_harmonics_gradient_kernel,
        dim=N,
        inputs=[wp_positions, L_max],
        outputs=[wp_output],
        device=wp_device,
    )

    return output
