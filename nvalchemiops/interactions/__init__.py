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

"""
Interaction Potentials
======================

This module provides GPU-accelerated implementations of common interaction
potentials used in molecular dynamics simulations.

Available Potentials:
- Lennard-Jones (lj): Short-range van der Waals interactions
- Electrostatics: Coulomb and Ewald summation methods
- Dispersion: Long-range dispersion corrections
"""

from nvalchemiops.interactions.lj import (
    lj_energy,
    lj_energy_forces,
    lj_energy_forces_virial,
    lj_forces,
)
from nvalchemiops.interactions.switching import (
    switch_c2,
)

__all__ = [
    "lj_energy",
    "lj_forces",
    "lj_energy_forces",
    "lj_energy_forces_virial",
    "switch_c2",
]
