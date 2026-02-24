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
Dynamics Module
===============

GPU-accelerated Warp kernels for molecular dynamics integrators and
geometry optimization algorithms.

All kernels use direct array inputs (no structs) and provide both:
- Mutating (in-place) versions for efficiency
- Non-mutating versions for gradient tracking

Submodules
----------
integrators
    MD integrators: velocity Verlet (NVE), Langevin (NVT), Nosé-Hoover (NVT)
optimizers
    Geometry optimizers: FIRE, FIRE2
utils
    Utility functions: temperature computation, velocity initialization

Example
-------
>>> import warp as wp
>>> from nvalchemiops.dynamics.integrators import velocity_verlet_position_update
>>>
>>> # Mutating (in-place) API
>>> velocity_verlet_position_update(
...     positions, velocities, forces, masses, dt
... )
>>>
>>> # Non-mutating API (for gradient tracking)
>>> from nvalchemiops.dynamics.integrators import velocity_verlet_position_update_out
>>> positions_out = wp.zeros_like(positions)
>>> velocities_out = wp.zeros_like(velocities)
>>> positions_out, velocities_out = velocity_verlet_position_update_out(
...     positions, velocities, forces, masses, dt,
...     positions_out, velocities_out,
... )
"""

from nvalchemiops.dynamics import integrators, optimizers, utils

__all__ = [
    # Submodules
    "integrators",
    "optimizers",
    "utils",
]
