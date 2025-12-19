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
Gaussian Type Orbital (GTO) Basis Functions
============================================

This module provides Warp functions for evaluating Gaussian Type Orbital (GTO)
basis functions, which are used for representing multipole charge distributions
in electrostatics calculations.

Mathematical Background
-----------------------

A Gaussian Type Orbital (GTO) density function combines a spherical harmonic
with a Gaussian radial factor:

.. math::
    \phi_{l,m}(\mathbf{r}, \sigma) = N_l \cdot Y_l^m(\hat{\mathbf{r}}) \cdot \exp\left(-\frac{r^2}{2\sigma^2}\right)

where:
- :math:`Y_l^m` is the real spherical harmonic of degree l and order m
- :math:`\sigma` is the Gaussian width parameter
- :math:`N_l` is a normalization constant
- :math:`r = |\mathbf{r}|` is the distance from the origin

Normalization Convention
------------------------

We use **integral normalization** such that:

- For L=0 (monopole): :math:`\int \phi_{0,0}(\mathbf{r}) d\mathbf{r} = 1`
- For L>0: :math:`\int \phi_{l,m}(\mathbf{r}) d\mathbf{r} = 0` (by symmetry)

The normalization constant is:

.. math::
    N_l = \frac{\sqrt{4\pi}}{(2\pi\sigma^2)^{3/2}}

This ensures that a charge distribution :math:`\rho(\mathbf{r}) = q \cdot \phi_{0,0}(\mathbf{r})`
integrates to total charge :math:`q`.

Fourier Transform
-----------------

The GTO has an analytical Fourier transform:

.. math::
    \hat{\phi}_{l,m}(\mathbf{k}, \sigma) = \left(\frac{i}{2}\right)^l \cdot \sqrt{4\pi} \cdot Y_l^m(\hat{\mathbf{k}}) \cdot \exp\left(-\frac{k^2 \sigma^2}{2}\right)

This is the key property that makes GTOs useful for reciprocal space calculations:
- The Gaussian factor damps high-frequency components (smooth in real space)
- The spherical harmonic factor encodes the angular dependence
- The :math:`(i/2)^l` factor arises from the Fourier transform of the radial part

Relationship to Ewald Parameter
-------------------------------

The GTO width :math:`\sigma` relates to the Ewald splitting parameter :math:`\alpha` by:

.. math::
    \sigma = \frac{1}{2\alpha}

This means smaller :math:`\sigma` (more localized GTOs) corresponds to larger :math:`\alpha`
(more emphasis on real-space interactions).

Usage in Warp Kernels
---------------------

These are Warp functions designed to be called from within Warp kernels::

    @wp.kernel
    def my_kernel(
        positions: wp.array(dtype=wp.vec3d),
        sigma: wp.float64,
        output: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        r = positions[i]

        # Evaluate L=0 GTO density
        density = gto_density_l0(r, sigma)

        # Evaluate L=1 GTO densities (3 components)
        density_l1 = gto_density_l1(r, sigma)

References
----------
- Helgaker, T., Jørgensen, P., & Olsen, J. (2000). Molecular electronic-structure
  theory. John Wiley & Sons. (Chapter on Gaussian basis functions)
- The e3nn library: https://e3nn.org/
"""

from __future__ import annotations

import math

import warp as wp

from .spherical_harmonics import (
    Y00_COEFF,
    Y1_COEFF,
    Y2_0_COEFF,
    Y2_M1_COEFF,
    Y2_M2_COEFF,
    Y2_P1_COEFF,
    Y2_P2_COEFF,
    eval_spherical_harmonics_l1,
    eval_spherical_harmonics_l2,
)

# =============================================================================
# Mathematical Constants
# =============================================================================

# pi and related constants
PI = wp.constant(wp.float64(math.pi))
TWOPI = wp.constant(wp.float64(2.0 * math.pi))
SQRT_4PI = wp.constant(wp.float64(math.sqrt(4.0 * math.pi)))  # sqrt(4*pi)

# Small value for numerical stability
EPSILON = wp.constant(wp.float64(1e-30))


@wp.func
def rsqrt(x: wp.float64) -> wp.float64:
    return wp.float64(1.0) / wp.sqrt(x)


# =============================================================================
# GTO Density Functions (Real Space)
# =============================================================================


@wp.func
def gto_normalization(sigma: wp.float64) -> wp.float64:
    r"""Compute the GTO normalization constant.

    .. math::

        N = \frac{\sqrt{4\pi}}{(2\pi \sigma^2)^{3/2}} = \frac{\sqrt{4\pi}}{(2\pi)^{3/2} \sigma^3}

    This ensures that :math:`\int \phi_{0,0}(r) dr = 1`.

    Parameters
    ----------
    sigma : wp.float64
        Gaussian width parameter.

    Returns
    -------
    wp.float64
        Normalization constant N.
    """
    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    # (2*pi)^(3/2) = (2*pi) * sqrt(2*pi) ~= 15.7496...
    twopi_3_2 = TWOPI * wp.sqrt(TWOPI)
    return SQRT_4PI / (twopi_3_2 * sigma3)


@wp.func
def gto_gaussian_factor(r2: wp.float64, sigma: wp.float64) -> wp.float64:
    r"""Compute the Gaussian radial factor exp(-r^2/(2sigma^2)).

    .. math::

        \exp\left(-\frac{r^2}{2\sigma^2}\right)

    Parameters
    ----------
    r2 : wp.float64
        Squared distance :math:`r^2 = |r|^2`.
    sigma : wp.float64
        Gaussian width parameter.

    Returns
    -------
    wp.float64
        :math:`\exp(-r^2/(2\sigma^2))`
    """
    sigma2 = sigma * sigma
    return wp.exp(-r2 / (wp.float64(2.0) * sigma2))


@wp.func
def gto_density_l0(r: wp.vec3d, sigma: wp.float64) -> wp.float64:
    r"""Compute GTO density for L=0 (s-orbital).

    .. math::

        \phi_{0,0}(r, \sigma) = N \cdot Y_0^0 \cdot \exp\left(-\frac{r^2}{2\sigma^2}\right)
                              = N \cdot \frac{1}{\sqrt{4\pi}} \cdot \exp\left(-\frac{r^2}{2\sigma^2}\right)

    Parameters
    ----------
    r : wp.vec3d
        Position vector relative to GTO center.
    sigma : wp.float64
        Gaussian width parameter.

    Returns
    -------
    wp.float64
        GTO density value at position r.
    """
    r2 = wp.dot(r, r)
    norm = gto_normalization(sigma)
    gauss = gto_gaussian_factor(r2, sigma)
    return norm * Y00_COEFF * gauss


@wp.func
def gto_density_l1(r: wp.vec3d, sigma: wp.float64) -> wp.vec3d:
    r"""Compute GTO density for L=1 (p-orbital).

    Returns the three L=1 components:

    .. math::

        \begin{aligned}
        \phi_{1,-1}(r, \sigma) &= N \cdot Y_1^{-1}(\hat{r}) \cdot \exp\left(-\frac{r^2}{2\sigma^2}\right) \quad (\propto y/r) \\
        \phi_{1,0}(r, \sigma) &= N \cdot Y_1^0(\hat{r}) \cdot \exp\left(-\frac{r^2}{2\sigma^2}\right) \quad (\propto z/r) \\
        \phi_{1,+1}(r, \sigma) &= N \cdot Y_1^{+1}(\hat{r}) \cdot \exp\left(-\frac{r^2}{2\sigma^2}\right) \quad (\propto x/r)
        \end{aligned}

    Parameters
    ----------
    r : wp.vec3d
        Position vector relative to GTO center.
    sigma : wp.float64
        Gaussian width parameter.

    Returns
    -------
    wp.vec3d
        :math:`(\phi_{1,-1}, \phi_{1,0}, \phi_{1,+1})` values.
    """
    r2 = wp.dot(r, r)
    norm = gto_normalization(sigma)
    gauss = gto_gaussian_factor(r2, sigma)

    # Get spherical harmonic values
    y_l1 = eval_spherical_harmonics_l1(r)  # (Y_1^{-1}, Y_1^0, Y_1^{+1})

    prefactor = norm * gauss
    return wp.vec3d(
        prefactor * y_l1[0],
        prefactor * y_l1[1],
        prefactor * y_l1[2],
    )


@wp.func
def gto_density_l2(r: wp.vec3d, sigma: wp.float64) -> wp.vec(
    length=5, dtype=wp.float64
):
    r"""Compute GTO density for L=2 (quadrupole/d-orbital).

    Returns the five L=2 components:
    :math:`\phi_{2,-2}, \phi_{2,-1}, \phi_{2,0}, \phi_{2,+1}, \phi_{2,+2}`

    Parameters
    ----------
    r : wp.vec3d
        Position vector relative to GTO center.
    sigma : wp.float64
        Gaussian width parameter.

    Returns
    -------
    wp.vec(5)
        The five L=2 GTO density values.
    """
    r2 = wp.dot(r, r)
    norm = gto_normalization(sigma)
    gauss = gto_gaussian_factor(r2, sigma)

    # Get spherical harmonic values
    y_l2 = eval_spherical_harmonics_l2(r)

    prefactor = norm * gauss
    return wp.vec(
        prefactor * y_l2[0],
        prefactor * y_l2[1],
        prefactor * y_l2[2],
        prefactor * y_l2[3],
        prefactor * y_l2[4],
        dtype=wp.float64,
    )


# =============================================================================
# GTO Fourier Transform Functions (k-Space)
# =============================================================================


@wp.func
def gto_fourier_l0(k: wp.vec3d, sigma: wp.float64) -> wp.float64:
    r"""Compute Fourier transform of L=0 GTO.

    .. math::

        \begin{aligned}
        \hat{\phi}_{0,0}(k, \sigma) &= (i/2)^0 \cdot \sqrt{4\pi} \cdot Y_0^0(\hat{k}) \cdot \exp(-k^2\sigma^2/2) \\
        &= \sqrt{4\pi} \cdot \frac{1}{\sqrt{4\pi}} \cdot \exp(-k^2\sigma^2/2) \\
        &= \exp(-k^2\sigma^2/2)
        \end{aligned}

    Note: For L=0, the result is purely real.

    Parameters
    ----------
    k : wp.vec3d
        Wave vector.
    sigma : wp.float64
        Gaussian width parameter.

    Returns
    -------
    wp.float64
        Real part of the Fourier transform (imaginary part is 0 for L=0).
    """
    k2 = wp.dot(k, k)
    sigma2 = sigma * sigma
    return wp.exp(-k2 * sigma2 / wp.float64(2.0))


@wp.func
def gto_fourier_l1_real(k: wp.vec3d, sigma: wp.float64) -> wp.vec3d:
    r"""Compute real part of Fourier transform of L=1 GTO.

    .. math::

        \begin{aligned}
        \hat{\phi}_{1,m}(k, \sigma) &= (i/2)^1 \cdot \sqrt{4\pi} \cdot Y_1^m(\hat{k}) \cdot \exp(-k^2\sigma^2/2) \\
        &= \frac{i}{2} \cdot \sqrt{4\pi} \cdot Y_1^m(\hat{k}) \cdot \exp(-k^2\sigma^2/2)
        \end{aligned}

    For L=1, the :math:`(i/2)^1 = i/2` factor makes the result purely imaginary.
    This function returns the coefficient of the imaginary part divided by i,
    i.e., the "real coefficient" such that :math:`\hat{\phi} = i \cdot \text{result}`.

    Parameters
    ----------
    k : wp.vec3d
        Wave vector.
    sigma : wp.float64
        Gaussian width parameter.

    Returns
    -------
    wp.vec3d
        Coefficients for :math:`(\hat{\phi}_{1,-1}, \hat{\phi}_{1,0}, \hat{\phi}_{1,+1})`.
        The actual Fourier transform is i times these values.
    """
    k2 = wp.dot(k, k)
    sigma2 = sigma * sigma
    gauss = wp.exp(-k2 * sigma2 / wp.float64(2.0))

    # Get spherical harmonic values for k direction
    y_l1 = eval_spherical_harmonics_l1(k)

    # Coefficient: (1/2) * sqrt(4*pi) * Y * exp(...)
    # The i factor is handled by returning the imaginary coefficient
    prefactor = wp.float64(0.5) * SQRT_4PI * gauss

    return wp.vec3d(
        prefactor * y_l1[0],
        prefactor * y_l1[1],
        prefactor * y_l1[2],
    )


@wp.func
def gto_fourier_l1_imag(k: wp.vec3d, sigma: wp.float64) -> wp.vec3d:
    r"""Compute imaginary part of Fourier transform of L=1 GTO.

    This is the same as gto_fourier_l1_real since the result is purely imaginary
    and we return the coefficient of i.

    Parameters
    ----------
    k : wp.vec3d
        Wave vector.
    sigma : wp.float64
        Gaussian width parameter.

    Returns
    -------
    wp.vec3d
        Imaginary coefficients for :math:`(\hat{\phi}_{1,-1}, \hat{\phi}_{1,0}, \hat{\phi}_{1,+1})`.
    """
    return gto_fourier_l1_real(k, sigma)


@wp.func
def gto_fourier_l2_real(k: wp.vec3d, sigma: wp.float64) -> wp.vec(
    length=5, dtype=wp.float64
):
    r"""Compute real part of Fourier transform of L=2 GTO.

    .. math::

        \begin{aligned}
        \hat{\phi}_{2,m}(k, \sigma) &= (i/2)^2 \cdot \sqrt{4\pi} \cdot Y_2^m(\hat{k}) \cdot \exp(-k^2\sigma^2/2) \\
        &= -\frac{1}{4} \cdot \sqrt{4\pi} \cdot Y_2^m(\hat{k}) \cdot \exp(-k^2\sigma^2/2)
        \end{aligned}

    For L=2, the :math:`(i/2)^2 = -1/4` factor makes the result purely real (negative).

    Parameters
    ----------
    k : wp.vec3d
        Wave vector.
    sigma : wp.float64
        Gaussian width parameter.

    Returns
    -------
    wp.vec(5)
        Real parts of the five L=2 Fourier coefficients.
    """
    k2 = wp.dot(k, k)
    sigma2 = sigma * sigma
    gauss = wp.exp(-k2 * sigma2 / wp.float64(2.0))

    # Get spherical harmonic values for k direction
    y_l2 = eval_spherical_harmonics_l2(k)

    # Coefficient: (-1/4) * sqrt(4*pi) * Y * exp(...)
    prefactor = wp.float64(-0.25) * SQRT_4PI * gauss

    return wp.vec(
        prefactor * y_l2[0],
        prefactor * y_l2[1],
        prefactor * y_l2[2],
        prefactor * y_l2[3],
        prefactor * y_l2[4],
        dtype=wp.float64,
    )


# =============================================================================
# GTO Integral Functions
# =============================================================================


@wp.func
def gto_integral_l0(sigma: wp.float64) -> wp.float64:
    r"""Compute the integral of L=0 GTO over all space.

    .. math::

        \int \phi_{0,0}(r, \sigma) d^3r = 1

    By construction with our normalization.

    Parameters
    ----------
    sigma : wp.float64
        Gaussian width parameter (not used, integral is always 1).

    Returns
    -------
    wp.float64
        Always returns 1.0.
    """
    return wp.float64(1.0)


@wp.func
def gto_self_overlap(L: int, sigma: wp.float64) -> wp.float64:
    r"""Compute the self-overlap integral of a GTO.

    .. math::

        \langle \phi_{l,m} | \phi_{l,m} \rangle = \int |\phi_{l,m}(r, \sigma)|^2 d^3r

    For our normalization, this gives a specific value depending on L.

    Parameters
    ----------
    L : int
        Angular momentum quantum number.
    sigma : wp.float64
        Gaussian width parameter.

    Returns
    -------
    wp.float64
        Self-overlap integral value.
    """
    # The self-overlap depends on sigma and L
    # For normalized GTOs: <phi|phi> = N^2 * integral |Y|^2 exp(-r^2/sigma^2) d^3r
    # This evaluates to specific values based on the Gaussian integral
    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma

    # For L=0: The integral is (sigma*sqrt(pi))^3 / (2*pi*sigma^2)^3 * 4*pi/(4*pi) = 1/(pi*sigma^3)^(1/2) approximately
    # Actually, let's compute it properly:
    # <phi|phi> = N^2 * |Y|^2 integrated over sphere * radial integral
    # For spherical harmonics: integral |Y_l^m|^2 dOmega = 1 (orthonormal)
    # Radial: integral exp(-r^2/sigma^2) 4*pi*r^2 dr = (pi*sigma^2)^(3/2)
    # So: <phi|phi> = N^2 * (pi*sigma^2)^(3/2)

    # With N = sqrt(4*pi) / (2*pi*sigma^2)^(3/2):
    # N^2 = 4*pi / (2*pi)^3 * sigma^6
    # <phi|phi> = [4*pi / (8*pi^3 * sigma^6)] * (pi*sigma^2)^(3/2)
    #           = [4*pi / (8*pi^3 * sigma^6)] * pi^(3/2) * sigma^3
    #           = 4*pi * pi^(3/2) / (8*pi^3 * sigma^3)
    #           = pi^(5/2) / (2*pi^3 * sigma^3)
    #           = 1 / (2*pi^(1/2) * sigma^3)
    #           = 1 / (2*sqrt(pi) * sigma^3)

    sqrt_pi = wp.sqrt(PI)
    return wp.float64(1.0) / (wp.float64(2.0) * sqrt_pi * sigma3)


# =============================================================================
# Combined Evaluation Functions
# =============================================================================


@wp.func
def gto_density_all(r: wp.vec3d, sigma: wp.float64) -> wp.vec(
    length=9, dtype=wp.float64
):
    r"""Compute GTO density for all L=0,1,2 components.

    Returns all 9 components (1 + 3 + 5) in order:
    :math:`[\phi_{0,0}, \phi_{1,-1}, \phi_{1,0}, \phi_{1,+1}, \phi_{2,-2}, \phi_{2,-1}, \phi_{2,0}, \phi_{2,+1}, \phi_{2,+2}]`

    Parameters
    ----------
    r : wp.vec3d
        Position vector relative to GTO center.
    sigma : wp.float64
        Gaussian width parameter.

    Returns
    -------
    wp.vec(9)
        All 9 GTO density values.
    """
    r2 = wp.dot(r, r)
    norm = gto_normalization(sigma)
    gauss = gto_gaussian_factor(r2, sigma)
    prefactor = norm * gauss

    # Compute spherical harmonics
    r2_safe = r2 + EPSILON
    r_inv = rsqrt(r2_safe)
    r2_inv = wp.float64(1.0) / r2_safe

    x, y, z = r[0], r[1], r[2]
    x2, y2, z2 = x * x, y * y, z * z

    return wp.vec(
        # L=0
        prefactor * Y00_COEFF,
        # L=1
        prefactor * Y1_COEFF * y * r_inv,
        prefactor * Y1_COEFF * z * r_inv,
        prefactor * Y1_COEFF * x * r_inv,
        # L=2
        prefactor * Y2_M2_COEFF * x * y * r2_inv,
        prefactor * Y2_M1_COEFF * y * z * r2_inv,
        prefactor * Y2_0_COEFF * (wp.float64(3.0) * z2 - r2_safe) * r2_inv,
        prefactor * Y2_P1_COEFF * x * z * r2_inv,
        prefactor * Y2_P2_COEFF * (x2 - y2) * r2_inv,
        dtype=wp.float64,
    )


# =============================================================================
# Gradient Functions
# =============================================================================


@wp.func
def gto_density_l0_gradient(r: wp.vec3d, sigma: wp.float64) -> wp.vec3d:
    r"""Compute gradient of L=0 GTO density with respect to r.

    .. math::

        \begin{aligned}
        \nabla \phi_{0,0} &= \nabla \left[N \cdot Y_0^0 \cdot \exp\left(-\frac{r^2}{2\sigma^2}\right)\right] \\
        &= N \cdot Y_0^0 \cdot \left(-\frac{r}{\sigma^2}\right) \cdot \exp\left(-\frac{r^2}{2\sigma^2}\right) \\
        &= \phi_{0,0} \cdot \left(-\frac{r}{\sigma^2}\right)
        \end{aligned}

    Parameters
    ----------
    r : wp.vec3d
        Position vector.
    sigma : wp.float64
        Gaussian width parameter.

    Returns
    -------
    wp.vec3d
        Gradient vector :math:`\nabla \phi_{0,0}`.
    """
    sigma2 = sigma * sigma
    density = gto_density_l0(r, sigma)

    # grad(phi) = phi * (-r/sigma^2)
    factor = -density / sigma2
    return wp.vec3d(factor * r[0], factor * r[1], factor * r[2])


# =============================================================================
# PyTorch Wrappers for Testing
# =============================================================================


@wp.kernel
def _eval_gto_density_kernel(
    positions: wp.array(dtype=wp.vec3d),
    sigma: wp.float64,
    L_max: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Evaluate Gaussian Type Orbital (GTO) density functions at multiple positions.

    Computes φ_{l,m}(r, σ) = N_l · Y_l^m(r̂) · exp(-r²/(2σ²)) for all harmonics
    up to L_max. The normalization N_l ensures ∫φ_{0,0} dr = 1.

    Launch Grid
    -----------
    dim = [N]

    One thread per position.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3d), shape (N,)
        Position vectors relative to GTO center.
    sigma : wp.float64
        Gaussian width parameter (σ). Relates to Ewald parameter via σ = 1/(2α).
    L_max : int
        Maximum angular momentum quantum number (0, 1, or 2).
    output : wp.array2d(dtype=wp.float64), shape (N, num_components)
        Output GTO density values where num_components is:
        - L_max=0: 1 component  (s-orbital/monopole)
        - L_max=1: 4 components (s + p-orbitals/monopole + dipole)
        - L_max=2: 9 components (s + p + d-orbitals/monopole + dipole + quadrupole)

    Notes
    -----
    - Uses integral normalization: N = √(4π) / (2πσ²)^(3/2).
    - GTOs decay exponentially with distance from center.
    - L=0 integrates to 1; L>0 integrates to 0 by symmetry.
    - Useful for multipole charge distributions in electrostatics.
    """
    i = wp.tid()
    r = positions[i]

    r2 = wp.dot(r, r)
    norm = gto_normalization(sigma)
    gauss = gto_gaussian_factor(r2, sigma)
    prefactor = norm * gauss

    # L=0
    output[i, 0] = prefactor * Y00_COEFF

    if L_max >= 1:
        y1 = eval_spherical_harmonics_l1(r)
        output[i, 1] = prefactor * y1[0]
        output[i, 2] = prefactor * y1[1]
        output[i, 3] = prefactor * y1[2]

    if L_max >= 2:
        y2 = eval_spherical_harmonics_l2(r)
        output[i, 4] = prefactor * y2[0]
        output[i, 5] = prefactor * y2[1]
        output[i, 6] = prefactor * y2[2]
        output[i, 7] = prefactor * y2[3]
        output[i, 8] = prefactor * y2[4]


@wp.kernel
def _eval_gto_fourier_kernel(
    k_vectors: wp.array(dtype=wp.vec3d),
    sigma: wp.float64,
    L_max: int,
    output_real: wp.array2d(dtype=wp.float64),
    output_imag: wp.array2d(dtype=wp.float64),
):
    """Evaluate Fourier transforms of GTO density functions at multiple k-vectors.

    Computes φ̂_{l,m}(k, σ) = (i/2)^l · √(4π) · Y_l^m(k̂) · exp(-k²σ²/2).
    The Gaussian factor damps high-frequency components, making GTOs smooth in real space.

    Launch Grid
    -----------
    dim = [K]

    One thread per k-vector.

    Parameters
    ----------
    k_vectors : wp.array(dtype=wp.vec3d), shape (K,)
        Reciprocal space wave vectors.
    sigma : wp.float64
        Gaussian width parameter (σ). Same σ as real-space GTOs.
    L_max : int
        Maximum angular momentum quantum number (0, 1, or 2).
    output_real : wp.array2d(dtype=wp.float64), shape (K, num_components)
        Real part of Fourier coefficients.
    output_imag : wp.array2d(dtype=wp.float64), shape (K, num_components)
        Imaginary part of Fourier coefficients.

    Notes
    -----
    - L=0: Purely real (imag=0). Result is exp(-k²σ²/2).
    - L=1: Purely imaginary (real=0). Factor (i/2)^1 = i/2.
    - L=2: Purely real (imag=0). Factor (i/2)^2 = -1/4.
    - Analytical Fourier transforms enable efficient reciprocal-space calculations.
    - Used for multipole electrostatics in reciprocal space.
    """
    i = wp.tid()
    k = k_vectors[i]

    # L=0: purely real
    output_real[i, 0] = gto_fourier_l0(k, sigma)
    output_imag[i, 0] = wp.float64(0.0)

    if L_max >= 1:
        # L=1: purely imaginary (real part is 0)
        f1 = gto_fourier_l1_imag(k, sigma)
        output_real[i, 1] = wp.float64(0.0)
        output_real[i, 2] = wp.float64(0.0)
        output_real[i, 3] = wp.float64(0.0)
        output_imag[i, 1] = f1[0]
        output_imag[i, 2] = f1[1]
        output_imag[i, 3] = f1[2]

    if L_max >= 2:
        # L=2: purely real (imaginary part is 0)
        f2 = gto_fourier_l2_real(k, sigma)
        output_real[i, 4] = f2[0]
        output_real[i, 5] = f2[1]
        output_real[i, 6] = f2[2]
        output_real[i, 7] = f2[3]
        output_real[i, 8] = f2[4]
        output_imag[i, 4] = wp.float64(0.0)
        output_imag[i, 5] = wp.float64(0.0)
        output_imag[i, 6] = wp.float64(0.0)
        output_imag[i, 7] = wp.float64(0.0)
        output_imag[i, 8] = wp.float64(0.0)


def eval_gto_density_pytorch(
    positions,
    sigma: float,
    L_max: int = 2,
    device=None,
):
    """Evaluate GTO densities from PyTorch tensors.

    Parameters
    ----------
    positions : torch.Tensor
        Input positions [N, 3] as float64.
    sigma : float
        Gaussian width parameter.
    L_max : int
        Maximum angular momentum (0, 1, or 2). Default: 2.
    device : torch.device, optional
        Device for computation.

    Returns
    -------
    torch.Tensor
        GTO density values [N, num_components].
    """
    import torch

    if device is None:
        device = positions.device

    N = positions.shape[0]
    num_components = {0: 1, 1: 4, 2: 9}[L_max]

    output = torch.zeros((N, num_components), dtype=torch.float64, device=device)

    wp_device = wp.device_from_torch(device)
    wp_positions = wp.from_torch(positions.contiguous(), dtype=wp.vec3d)
    wp_output = wp.from_torch(output, dtype=wp.float64)

    wp.launch(
        kernel=_eval_gto_density_kernel,
        dim=N,
        inputs=[wp_positions, wp.float64(sigma), L_max],
        outputs=[wp_output],
        device=wp_device,
    )

    return output


def eval_gto_fourier_pytorch(
    k_vectors,
    sigma: float,
    L_max: int = 2,
    device=None,
):
    """Evaluate GTO Fourier transforms from PyTorch tensors.

    Parameters
    ----------
    k_vectors : torch.Tensor
        Input wave vectors [K, 3] as float64.
    sigma : float
        Gaussian width parameter.
    L_max : int
        Maximum angular momentum (0, 1, or 2). Default: 2.
    device : torch.device, optional
        Device for computation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (real_part, imag_part) each of shape [K, num_components].
    """
    import torch

    if device is None:
        device = k_vectors.device

    K = k_vectors.shape[0]
    num_components = {0: 1, 1: 4, 2: 9}[L_max]

    output_real = torch.zeros((K, num_components), dtype=torch.float64, device=device)
    output_imag = torch.zeros((K, num_components), dtype=torch.float64, device=device)

    wp_device = wp.device_from_torch(device)
    wp_k = wp.from_torch(k_vectors.contiguous(), dtype=wp.vec3d)
    wp_real = wp.from_torch(output_real, dtype=wp.float64)
    wp_imag = wp.from_torch(output_imag, dtype=wp.float64)

    wp.launch(
        kernel=_eval_gto_fourier_kernel,
        dim=K,
        inputs=[wp_k, wp.float64(sigma), L_max],
        outputs=[wp_real, wp_imag],
        device=wp_device,
    )

    return output_real, output_imag
