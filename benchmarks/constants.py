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

"""Non-configurable constants for benchmark code.

**Principle**: User-facing benchmark parameters (system sizes, cutoffs, timing
iterations, physical parameters for D3 damping, etc.) live in YAML configs,
not Python. This module holds only values that are NOT user-configurable:

- Physical constants (unit conversions).
- Internal kernel tuning defaults that are part of the code contract, not
  user policy.
"""

from __future__ import annotations

__all__ = [
    "ANGSTROM_TO_BOHR",
    "DEFAULT_ATOMIC_DENSITY",
    "DEFAULT_NL_SAFETY_FACTOR",
]

# Physical constant — exact value, not configurable.
ANGSTROM_TO_BOHR = 1.8897259886

# Neighbor-list kernel tuning for estimate_max_neighbors(). Used by all
# runners when constructing neighbor lists. These are safety margins, not
# user policy: expose via YAML only if benchmark methodology requires it.
DEFAULT_ATOMIC_DENSITY = 0.2
DEFAULT_NL_SAFETY_FACTOR = 1.0
