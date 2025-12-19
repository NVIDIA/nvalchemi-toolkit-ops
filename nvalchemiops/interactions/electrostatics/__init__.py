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
Electrostatics Interactions Module
==================================

This module provides GPU-accelerated implementations of various methods for
computing long-range electrostatic interactions in molecular simulations.

Available Methods
-----------------

1. **Ewald Summation** (`ewald`)
   - Classical method splitting interactions into real-space and reciprocal-space
   - :math:`O(N^2)` scaling for explicit k-vectors, good for small systems
   - Full autograd support

2. **Particle Mesh Ewald (PME)** (`pme`)
   - FFT-based method for :math:`O(N \log N)` scaling
   - Uses B-spline interpolation for charge assignment
   - Full autograd support

"""

# Ewald summation
from .ewald import (
    ewald_real_space,
    ewald_reciprocal_space,
    ewald_summation,
)

# K-vectors
from .k_vectors import (
    generate_k_vectors_ewald_summation,
    generate_k_vectors_pme,
)

# Parameter estimation
from .parameters import (
    EwaldParameters,
    PMEParameters,
    estimate_ewald_parameters,
    estimate_pme_mesh_dimensions,
    estimate_pme_parameters,
    mesh_spacing_to_dimensions,
)

# Particle Mesh Ewald
from .pme import (
    particle_mesh_ewald,
    pme_reciprocal_space,
)

__all__ = [
    # Ewald
    "ewald_real_space",
    "ewald_reciprocal_space",
    "ewald_summation",
    # K-vectors
    "generate_k_vectors_ewald_summation",
    "generate_k_vectors_pme",
    # PME
    "particle_mesh_ewald",
    "pme_reciprocal_space",
    # Parameter estimation
    "estimate_ewald_parameters",
    "estimate_pme_parameters",
    "estimate_pme_mesh_dimensions",
    "mesh_spacing_to_dimensions",
    "EwaldParameters",
    "PMEParameters",
]
