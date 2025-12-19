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
Mathematical Utilities
======================

This module provides low-level mathematical functions implemented in Warp
for use in GPU-accelerated scientific computing.

Available Submodules
--------------------

spherical_harmonics
    Real spherical harmonics :math:`Y_l^m` for angular momentum :math:`L \leq 2`.
    Used in multipole expansions for electrostatics.

gto
    Gaussian Type Orbital (GTO) basis functions for multipole charge distributions.
    Includes real-space densities and Fourier transforms for :math:`L \leq 2`.
"""

from .math import (
    wp_erfc,
    wp_exp_kernel,
    wp_safe_divide,
    wpdivmod,
)
from .spherical_harmonics import (
    eval_all_spherical_harmonics,
    # Vectorized evaluators
    eval_spherical_harmonics_l0,
    eval_spherical_harmonics_l1,
    eval_spherical_harmonics_l2,
    # L=0 (monopole)
    spherical_harmonic_00,
    # Gradient functions
    spherical_harmonic_00_gradient,
    # L=1 (dipole)
    spherical_harmonic_1m1,
    spherical_harmonic_1m1_gradient,
    spherical_harmonic_1p1,
    spherical_harmonic_1p1_gradient,
    spherical_harmonic_2m1,
    spherical_harmonic_2m1_gradient,
    # L=2 (quadrupole)
    spherical_harmonic_2m2,
    spherical_harmonic_2m2_gradient,
    spherical_harmonic_2p1,
    spherical_harmonic_2p1_gradient,
    spherical_harmonic_2p2,
    spherical_harmonic_2p2_gradient,
    spherical_harmonic_10,
    spherical_harmonic_10_gradient,
    spherical_harmonic_20,
    spherical_harmonic_20_gradient,
)

__all__ = [
    # Math functions
    "wp_safe_divide",
    "wp_exp_kernel",
    "wpdivmod",
    "wp_erfc",
    # Individual harmonics
    "spherical_harmonic_00",
    "spherical_harmonic_1m1",
    "spherical_harmonic_10",
    "spherical_harmonic_1p1",
    "spherical_harmonic_2m2",
    "spherical_harmonic_2m1",
    "spherical_harmonic_20",
    "spherical_harmonic_2p1",
    "spherical_harmonic_2p2",
    # Vectorized evaluators
    "eval_spherical_harmonics_l0",
    "eval_spherical_harmonics_l1",
    "eval_spherical_harmonics_l2",
    "eval_all_spherical_harmonics",
    # Gradients
    "spherical_harmonic_00_gradient",
    "spherical_harmonic_1m1_gradient",
    "spherical_harmonic_10_gradient",
    "spherical_harmonic_1p1_gradient",
    "spherical_harmonic_2m2_gradient",
    "spherical_harmonic_2m1_gradient",
    "spherical_harmonic_20_gradient",
    "spherical_harmonic_2p1_gradient",
    "spherical_harmonic_2p2_gradient",
    # GTO basis functions
    "gto_normalization",
    "gto_gaussian_factor",
    "gto_density_l0",
    "gto_density_l1",
    "gto_density_l2",
    "gto_density_all",
    "gto_density_l0_gradient",
    "gto_fourier_l0",
    "gto_fourier_l1_real",
    "gto_fourier_l1_imag",
    "gto_fourier_l2_real",
    "gto_integral_l0",
    "gto_self_overlap",
    "eval_gto_density_pytorch",
    "eval_gto_fourier_pytorch",
]

from .gto import (
    # PyTorch wrappers
    eval_gto_density_pytorch,
    eval_gto_fourier_pytorch,
    gto_density_all,
    # Real-space densities
    gto_density_l0,
    gto_density_l0_gradient,
    gto_density_l1,
    gto_density_l2,
    # Fourier transforms
    gto_fourier_l0,
    gto_fourier_l1_imag,
    gto_fourier_l1_real,
    gto_fourier_l2_real,
    gto_gaussian_factor,
    # Integrals
    gto_integral_l0,
    # Normalization and Gaussian factor
    gto_normalization,
    gto_self_overlap,
)
