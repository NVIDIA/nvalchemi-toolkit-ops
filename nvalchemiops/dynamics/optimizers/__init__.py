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

L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
    Quasi-Newton optimization with a strong Wolfe line search. Usually reaches
    a given force tolerance in far fewer energy/force evaluations than the
    FIRE optimizers, which is the cost that dominates relaxation with a
    machine-learned potential.

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

fire2_update
    FIRE2 reduction, adaptive parameter update, and velocity mixing WITHOUT
    position/cell application. Use for custom final apply phases such as
    coupled variable-cell optimization.

lbfgs_step
    Complete L-BFGS step. Consumes exactly one energy/force evaluation per
    call and reports progress through a per-system ``status`` array.
    Uses batch_idx batching only.

lbfgs_update
    L-BFGS reductions, line-search decision, history update and two-loop
    recursion WITHOUT the position update. Use with ``lbfgs_prepare_step``
    and ``lbfgs_apply_step`` for custom apply phases.

lbfgs_reduce_energy
    Sum per-atom energies into the per-system totals ``lbfgs_step`` expects,
    accumulating in float64.

    All L-BFGS buffers are caller-owned: the package allocates and initializes
    nothing. See :mod:`nvalchemiops.dynamics.optimizers.lbfgs` for the required
    shapes and initial contents.

lbfgs_set_reference_cell, lbfgs_cell_kappa, lbfgs_pack_cell,
lbfgs_unpack_cell, lbfgs_cell_trust_region
    Variable-cell relaxation. Positions and cell are mapped into a single
    packed coordinate vector so the two-loop recursion couples them
    automatically, then mapped back after the step.

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
    fire_compute_vf_vv_ff,
    fire_step,
    fire_update,
)
from nvalchemiops.dynamics.optimizers.fire2 import (
    fire2_apply_step,
    fire2_reduce,
    fire2_step,
    fire2_update,
)
from nvalchemiops.dynamics.optimizers.lbfgs import (
    LBFGS_CONVERGED,
    LBFGS_LS_FAILED,
    LBFGS_NEED_EVAL,
    lbfgs_apply_step,
    lbfgs_cell_kappa,
    lbfgs_cell_trust_region,
    lbfgs_pack_cell,
    lbfgs_prepare_step,
    lbfgs_reduce,
    lbfgs_reduce_energy,
    lbfgs_set_reference_cell,
    lbfgs_step,
    lbfgs_unpack_cell,
    lbfgs_update,
)

__all__ = [
    # Unified API
    "fire_step",
    "fire_update",
    "fire_compute_vf_vv_ff",
    "fire2_step",
    "fire2_update",
    "fire2_apply_step",
    "fire2_reduce",
    # L-BFGS
    "lbfgs_step",
    "lbfgs_update",
    "lbfgs_prepare_step",
    "lbfgs_apply_step",
    "lbfgs_reduce",
    "lbfgs_reduce_energy",
    # L-BFGS variable cell
    "lbfgs_set_reference_cell",
    "lbfgs_cell_kappa",
    "lbfgs_pack_cell",
    "lbfgs_unpack_cell",
    "lbfgs_cell_trust_region",
    "LBFGS_NEED_EVAL",
    "LBFGS_CONVERGED",
    "LBFGS_LS_FAILED",
    # Low-level kernels
    "_fire_step_no_downhill_ptr_kernel",
    "_fire_step_downhill_ptr_kernel",
    "_fire_update_params_no_downhill_ptr_kernel",
    "_fire_update_params_downhill_ptr_kernel",
]
