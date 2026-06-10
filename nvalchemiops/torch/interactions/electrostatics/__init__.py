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

"""PyTorch bindings for electrostatics interactions.

Includes Coulomb, DSF, Ewald, PME, parameter helpers, k-vector generation, and
the standalone Yeh-Berkowitz / Ballenegger slab correction API.
"""

from __future__ import annotations

import inspect
import warnings

from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    infer_l_max,
    pack_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics.coulomb import (
    coulomb_energy,
    coulomb_energy_forces,
    coulomb_forces,
)
from nvalchemiops.torch.interactions.electrostatics.dsf import (
    dsf_coulomb,
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
from nvalchemiops.torch.interactions.electrostatics.multipole_electrostatics import (
    multipole_electrostatic_energy,
    multipole_reciprocal_space_energy,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
    multipole_ewald_summation,
    multipole_real_space_energy,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
    multipole_real_space_quadrupole_energy,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_features import (
    multipole_electrostatic_features,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
    MultipoleSCFCache,
    prepare_multipole_scf_cache,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
    multipole_ewald_scf_step_energy,
    multipole_scf_step_energy,
    multipole_scf_step_features,
)
from nvalchemiops.torch.interactions.electrostatics.parameters import (
    EwaldParameters,
    MultipoleEwaldParameters,
    MultipolePMEParameters,
    PMEParameters,
    estimate_ewald_parameters,
    estimate_multipole_ewald_parameters,
    estimate_multipole_pme_parameters,
    estimate_pme_mesh_dimensions,
    estimate_pme_parameters,
    mesh_spacing_to_dimensions,
)
from nvalchemiops.torch.interactions.electrostatics.pme import (
    compute_bspline_moduli_1d,
    particle_mesh_ewald,
    pme_reciprocal_space,
)
from nvalchemiops.torch.interactions.electrostatics.pme import (
    pme_energy_corrections as _pme_energy_corrections,
)
from nvalchemiops.torch.interactions.electrostatics.pme import (
    pme_energy_corrections_with_charge_grad as _pme_energy_corrections_with_charge_grad,
)
from nvalchemiops.torch.interactions.electrostatics.pme import (
    pme_green_structure_factor as _pme_green_structure_factor,
)
from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
    multipole_particle_mesh_ewald,
)
from nvalchemiops.torch.interactions.electrostatics.slab import (
    compute_slab_correction,
)


def _warn_low_level_pme_helper(name: str) -> None:
    """Warn when deprecated top-level PME helper aliases are called."""
    warnings.warn(
        f"nvalchemiops.torch.interactions.electrostatics.{name} is a low-level "
        "PME helper alias and is deprecated at the top-level namespace. Import "
        "from nvalchemiops.torch.interactions.electrostatics.pme if you need "
        "the internal helper, or use pme_reciprocal_space / particle_mesh_ewald.",
        DeprecationWarning,
        stacklevel=3,
    )


def pme_green_structure_factor(*args, **kwargs):
    """Deprecated top-level alias for the low-level PME Green helper."""
    _warn_low_level_pme_helper("pme_green_structure_factor")
    return _pme_green_structure_factor(*args, **kwargs)


def pme_energy_corrections(*args, **kwargs):
    """Deprecated top-level alias for the low-level PME correction helper."""
    _warn_low_level_pme_helper("pme_energy_corrections")
    return _pme_energy_corrections(*args, **kwargs)


def pme_energy_corrections_with_charge_grad(*args, **kwargs):
    """Deprecated top-level alias for the low-level PME correction helper."""
    _warn_low_level_pme_helper("pme_energy_corrections_with_charge_grad")
    return _pme_energy_corrections_with_charge_grad(*args, **kwargs)


def _preserve_deprecated_alias_metadata(alias, target, summary: str) -> None:
    """Expose the wrapped helper signature while keeping the deprecation note."""
    alias.__signature__ = inspect.signature(target)
    alias.__wrapped__ = target
    target_doc = inspect.getdoc(target)
    alias.__doc__ = summary if target_doc is None else f"{summary}\n\n{target_doc}"


_preserve_deprecated_alias_metadata(
    pme_green_structure_factor,
    _pme_green_structure_factor,
    "Deprecated top-level alias for the low-level PME Green helper.",
)
_preserve_deprecated_alias_metadata(
    pme_energy_corrections,
    _pme_energy_corrections,
    "Deprecated top-level alias for the low-level PME correction helper.",
)
_preserve_deprecated_alias_metadata(
    pme_energy_corrections_with_charge_grad,
    _pme_energy_corrections_with_charge_grad,
    "Deprecated top-level alias for the low-level PME correction helper.",
)


__all__ = [
    # Coulomb
    "coulomb_energy",
    "coulomb_forces",
    "coulomb_energy_forces",
    # DSF
    "dsf_coulomb",
    # Ewald
    "ewald_real_space",
    "ewald_reciprocal_space",
    "ewald_summation",
    # Slab correction (Yeh-Berkowitz / Ballenegger Eq. 29)
    "compute_slab_correction",
    # PME
    "particle_mesh_ewald",
    "pme_reciprocal_space",
    "pme_energy_corrections",
    "pme_energy_corrections_with_charge_grad",
    "pme_green_structure_factor",
    "compute_bspline_moduli_1d",
    # K-vectors
    "generate_k_vectors_ewald_summation",
    "generate_k_vectors_pme",
    # Multipole moments packing (e3nn <-> Cartesian)
    "pack_multipole_moments",
    "infer_l_max",
    # Multipole (direct k-space)
    "multipole_electrostatic_energy",
    "multipole_electrostatic_features",
    "multipole_reciprocal_space_energy",
    "MultipoleSCFCache",
    "prepare_multipole_scf_cache",
    "multipole_scf_step_energy",
    "multipole_scf_step_features",
    # Multipole Ewald (real-space l_max=0/1)
    "multipole_real_space_energy",
    # Multipole Ewald (real-space l_max=2 per-atom)
    "multipole_real_space_quadrupole_energy",
    # Composite Ewald summation (real + reciprocal - self)
    "multipole_ewald_summation",
    # Composite Particle-Mesh Ewald (l_max=0/1/2)
    "multipole_particle_mesh_ewald",
    # Cache-aware Ewald SCF step
    "multipole_ewald_scf_step_energy",
    # Parameters
    "EwaldParameters",
    "PMEParameters",
    "MultipoleEwaldParameters",
    "MultipolePMEParameters",
    "estimate_ewald_parameters",
    "estimate_pme_parameters",
    "estimate_pme_mesh_dimensions",
    "estimate_multipole_ewald_parameters",
    "estimate_multipole_pme_parameters",
    "mesh_spacing_to_dimensions",
]
