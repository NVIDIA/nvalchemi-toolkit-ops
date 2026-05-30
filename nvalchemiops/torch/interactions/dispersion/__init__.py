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
"""PyTorch bindings for dispersion corrections."""

from nvalchemiops.torch.interactions.dispersion._dftd3 import D3Parameters, dftd3
from nvalchemiops.torch.interactions.dispersion.lj_dispersion import (
    lj_dispersion_energy,
    lj_dispersion_energy_forces,
    lj_dispersion_forces,
    sigma_epsilon_to_dispersion_charge,
)
from nvalchemiops.torch.interactions.dispersion.lj_pme import (
    dispersion_pme,
    dispersion_reciprocal_space,
)
from nvalchemiops.torch.interactions.dispersion.parameters import (
    DispersionPMEParameters,
    estimate_dispersion_pme_parameters,
)

__all__ = [
    "dftd3",
    "D3Parameters",
    "lj_dispersion_energy",
    "lj_dispersion_forces",
    "lj_dispersion_energy_forces",
    "sigma_epsilon_to_dispersion_charge",
    "dispersion_pme",
    "dispersion_reciprocal_space",
    "DispersionPMEParameters",
    "estimate_dispersion_pme_parameters",
]
