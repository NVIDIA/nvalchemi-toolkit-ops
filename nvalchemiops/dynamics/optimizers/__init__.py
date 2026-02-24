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
Geometry Optimizers
===================

GPU-accelerated geometry optimization algorithms.

Available Optimizers
--------------------
FIRE (Fast Inertial Relaxation Engine)
    MD-based optimization with adaptive timestep and velocity mixing.

FIRE2 (Fast Inertial Relaxation Engine v2)
    Improved FIRE with adaptive damping and velocity mixing.

Main API Functions
------------------
fire_step
    Full FIRE step with MD integration. Supports single system,
    batch_idx, and atom_ptr batching modes, with optional downhill check.

fire_update
    FIRE velocity mixing and parameter update WITHOUT MD integration.
    Use for variable-cell optimization with packed extended arrays.

fire2_step
    Complete FIRE2 optimization step.
    Uses batch_idx batching only.

Kernel Selection
----------------
- Neither batch_idx nor atom_ptr: single system kernel
- batch_idx provided: batch_idx kernel (one thread per atom)
- atom_ptr provided: ptr/CSR kernel (one thread per system)
- Downhill arrays provided: downhill variant with energy check
"""

from nvalchemiops.dynamics.optimizers.fire import (
    # Low-level kernels
    _fire_step_downhill_ptr_kernel,
    _fire_step_no_downhill_ptr_kernel,
    _fire_update_params_downhill_ptr_kernel,
    _fire_update_params_no_downhill_ptr_kernel,
    # Unified API
    fire_step,
    fire_update,
)
from nvalchemiops.dynamics.optimizers.fire2 import fire2_step

__all__ = [
    # Unified API
    "fire_step",
    "fire_update",
    "fire2_step",
    # Low-level kernels
    "_fire_step_no_downhill_ptr_kernel",
    "_fire_step_downhill_ptr_kernel",
    "_fire_update_params_no_downhill_ptr_kernel",
    "_fire_update_params_downhill_ptr_kernel",
]
