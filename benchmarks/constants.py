# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
