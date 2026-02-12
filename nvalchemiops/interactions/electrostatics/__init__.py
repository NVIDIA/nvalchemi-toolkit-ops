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
Electrostatics Interactions Module
==================================

This module provides GPU-accelerated implementations of various methods for
computing long-range electrostatic interactions in molecular simulations.

Architecture
------------
This module provides framework-agnostic Warp kernel launchers.
For PyTorch bindings, see ``nvalchemiops.torch.interactions.electrostatics``.

Available Methods
-----------------

1. **Coulomb** (`coulomb`)
   - Direct Coulomb energy and forces
   - Damped (erfc) Coulomb for Ewald/PME real-space contribution
   - Warp launchers: ``coulomb_energy()``, ``coulomb_energy_forces()``, etc.
   - PyTorch API: ``nvalchemiops.torch.interactions.electrostatics.coulomb``

2. **Ewald Summation** (`ewald`)
   - Classical method splitting interactions into real-space and reciprocal-space
   - :math:`O(N^2)` scaling for explicit k-vectors, good for small systems
   - Full autograd support

3. **Particle Mesh Ewald (PME)** (`pme`)
   - FFT-based method for :math:`O(N \log N)` scaling
   - Uses B-spline interpolation for charge assignment
   - Full autograd support

"""

# Coulomb - Warp launchers (framework-agnostic)
from .coulomb import (
    batch_coulomb_energy,
    batch_coulomb_energy_forces,
    batch_coulomb_energy_forces_matrix,
    batch_coulomb_energy_matrix,
    coulomb_energy,
    coulomb_energy_forces,
    coulomb_energy_forces_matrix,
    coulomb_energy_matrix,
)

# Ewald summation - PyTorch bindings are deprecated at this location
# Use nvalchemiops.torch.interactions.electrostatics.ewald instead
# Handled via __getattr__ below for lazy import with deprecation warning
# Ewald - Warp launchers (framework-agnostic)
from .ewald_kernels import (
    # Real-space batch
    batch_ewald_real_space_energy,
    batch_ewald_real_space_energy_forces,
    batch_ewald_real_space_energy_forces_charge_grad,
    batch_ewald_real_space_energy_forces_charge_grad_matrix,
    batch_ewald_real_space_energy_forces_matrix,
    batch_ewald_real_space_energy_matrix,
    batch_ewald_reciprocal_space_compute_energy,
    batch_ewald_reciprocal_space_energy_forces,
    batch_ewald_reciprocal_space_energy_forces_charge_grad,
    # Reciprocal-space batch
    batch_ewald_reciprocal_space_fill_structure_factors,
    batch_ewald_subtract_self_energy,
    # Real-space single-system
    ewald_real_space_energy,
    ewald_real_space_energy_forces,
    ewald_real_space_energy_forces_charge_grad,
    ewald_real_space_energy_forces_charge_grad_matrix,
    ewald_real_space_energy_forces_matrix,
    ewald_real_space_energy_matrix,
    ewald_reciprocal_space_compute_energy,
    ewald_reciprocal_space_energy_forces,
    ewald_reciprocal_space_energy_forces_charge_grad,
    # Reciprocal-space single-system
    ewald_reciprocal_space_fill_structure_factors,
    ewald_subtract_self_energy,
)

# PME - Warp launchers (framework-agnostic)
from .pme_kernels import (
    batch_pme_energy_corrections,
    batch_pme_energy_corrections_with_charge_grad,
    batch_pme_green_structure_factor,
    pme_energy_corrections,
    pme_energy_corrections_with_charge_grad,
    pme_green_structure_factor,
)

__all__ = [
    # Coulomb - Warp launchers
    "coulomb_energy",
    "coulomb_energy_forces",
    "coulomb_energy_matrix",
    "coulomb_energy_forces_matrix",
    "batch_coulomb_energy",
    "batch_coulomb_energy_forces",
    "batch_coulomb_energy_matrix",
    "batch_coulomb_energy_forces_matrix",
    # Ewald - Warp launchers (real-space)
    "ewald_real_space_energy",
    "ewald_real_space_energy_forces",
    "ewald_real_space_energy_matrix",
    "ewald_real_space_energy_forces_matrix",
    "ewald_real_space_energy_forces_charge_grad",
    "ewald_real_space_energy_forces_charge_grad_matrix",
    "batch_ewald_real_space_energy",
    "batch_ewald_real_space_energy_forces",
    "batch_ewald_real_space_energy_matrix",
    "batch_ewald_real_space_energy_forces_matrix",
    "batch_ewald_real_space_energy_forces_charge_grad",
    "batch_ewald_real_space_energy_forces_charge_grad_matrix",
    # Ewald - Warp launchers (reciprocal-space)
    "ewald_reciprocal_space_fill_structure_factors",
    "ewald_reciprocal_space_compute_energy",
    "ewald_subtract_self_energy",
    "ewald_reciprocal_space_energy_forces",
    "ewald_reciprocal_space_energy_forces_charge_grad",
    "batch_ewald_reciprocal_space_fill_structure_factors",
    "batch_ewald_reciprocal_space_compute_energy",
    "batch_ewald_subtract_self_energy",
    "batch_ewald_reciprocal_space_energy_forces",
    "batch_ewald_reciprocal_space_energy_forces_charge_grad",
    # PME - Warp launchers
    "pme_green_structure_factor",
    "batch_pme_green_structure_factor",
    "pme_energy_corrections",
    "batch_pme_energy_corrections",
    "pme_energy_corrections_with_charge_grad",
    "batch_pme_energy_corrections_with_charge_grad",
    # Ewald - PyTorch bindings (deprecated, use nvalchemiops.torch.interactions.electrostatics)
    "ewald_real_space",
    "ewald_reciprocal_space",
    "ewald_summation",
    # PME - PyTorch bindings (deprecated, use nvalchemiops.torch.interactions.electrostatics)
    "particle_mesh_ewald",
    "pme_reciprocal_space",
    "pme_green_structure_factor",
    "pme_energy_corrections",
    "pme_energy_corrections_with_charge_grad",
]

# Deprecated PyTorch functions - lazy import with warning
# These functions have moved to nvalchemiops.torch.interactions.electrostatics
_DEPRECATED_TORCH_EXPORTS = {
    # Ewald
    "ewald_real_space": "nvalchemiops.torch.interactions.electrostatics.ewald",
    "ewald_reciprocal_space": "nvalchemiops.torch.interactions.electrostatics.ewald",
    "ewald_summation": "nvalchemiops.torch.interactions.electrostatics.ewald",
    # PME
    "particle_mesh_ewald": "nvalchemiops.torch.interactions.electrostatics.pme",
    "pme_reciprocal_space": "nvalchemiops.torch.interactions.electrostatics.pme",
    "pme_green_structure_factor": "nvalchemiops.torch.interactions.electrostatics.pme",
    "pme_energy_corrections": "nvalchemiops.torch.interactions.electrostatics.pme",
    "pme_energy_corrections_with_charge_grad": "nvalchemiops.torch.interactions.electrostatics.pme",
    # K-vectors
    "generate_k_vectors_ewald_summation": "nvalchemiops.torch.interactions.electrostatics.k_vectors",
    "generate_k_vectors_pme": "nvalchemiops.torch.interactions.electrostatics.k_vectors",
    # Parameters
    "estimate_ewald_parameters": "nvalchemiops.torch.interactions.electrostatics.parameters",
    "estimate_pme_parameters": "nvalchemiops.torch.interactions.electrostatics.parameters",
    "estimate_pme_mesh_dimensions": "nvalchemiops.torch.interactions.electrostatics.parameters",
    "mesh_spacing_to_dimensions": "nvalchemiops.torch.interactions.electrostatics.parameters",
    "EwaldParameters": "nvalchemiops.torch.interactions.electrostatics.parameters",
    "PMEParameters": "nvalchemiops.torch.interactions.electrostatics.parameters",
}


def __getattr__(name: str):
    """Lazy import with deprecation warning for PyTorch functions."""
    if name in _DEPRECATED_TORCH_EXPORTS:
        import importlib
        import warnings

        module_path = _DEPRECATED_TORCH_EXPORTS[name]
        warnings.warn(
            f"Importing '{name}' from 'nvalchemiops.interactions.electrostatics' is deprecated. "
            f"Please import from '{module_path}' instead. "
            "This will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
