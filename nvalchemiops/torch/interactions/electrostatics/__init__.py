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

"""PyTorch bindings for electrostatics interactions."""

from nvalchemiops.torch.interactions.electrostatics.coulomb import (
    coulomb_energy,
    coulomb_energy_forces,
    coulomb_forces,
)
from nvalchemiops.torch.interactions.electrostatics.ewald import (
    ewald_real_space,
    ewald_reciprocal_space,
    ewald_summation,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
    generate_k_vectors_pme,
)
from nvalchemiops.torch.interactions.electrostatics.parameters import (
    EwaldParameters,
    PMEParameters,
    estimate_ewald_parameters,
    estimate_pme_mesh_dimensions,
    estimate_pme_parameters,
    mesh_spacing_to_dimensions,
)
from nvalchemiops.torch.interactions.electrostatics.pme import (
    particle_mesh_ewald,
    pme_energy_corrections,
    pme_energy_corrections_with_charge_grad,
    pme_green_structure_factor,
    pme_reciprocal_space,
)

__all__ = [
    # Coulomb
    "coulomb_energy",
    "coulomb_forces",
    "coulomb_energy_forces",
    # Ewald
    "ewald_real_space",
    "ewald_reciprocal_space",
    "ewald_summation",
    # PME
    "particle_mesh_ewald",
    "pme_reciprocal_space",
    "pme_green_structure_factor",
    "pme_energy_corrections",
    "pme_energy_corrections_with_charge_grad",
    # K-vectors
    "generate_k_vectors_ewald_summation",
    "generate_k_vectors_pme",
    # Parameters
    "EwaldParameters",
    "PMEParameters",
    "estimate_ewald_parameters",
    "estimate_pme_parameters",
    "estimate_pme_mesh_dimensions",
    "mesh_spacing_to_dimensions",
]
