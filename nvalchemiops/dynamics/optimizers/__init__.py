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
    Quasi-Newton optimization with a ``maxstep`` trust region. Usually reaches
    a given force tolerance in far fewer force evaluations than the FIRE
    optimizers, which is the cost that dominates relaxation with a
    machine-learned potential. It reads no energy, so a model whose forces are
    not the gradient of its reported energy relaxes just as well.

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

lbfgs_prepare_state, lbfgs_prepare_cell_state
    Allocate, initialize and validate an ``LBFGSState`` -- and, for
    variable-cell relaxation, an ``LBFGSCellState``. Call once; calling again
    is how you reset.

lbfgs_step
    One batched L-BFGS step on coordinates. Consumes exactly one force
    evaluation: it updates the curvature history, restarts the direction if it
    stops descending, and takes one ``maxstep``-bounded step. No energy is
    read, and no tolerance is owned -- deciding when to stop is yours, as it is
    for FIRE2.

lbfgs_step_coord_cell
    The same, relaxing coordinates and cell together. Positions and cell are
    mapped into a single packed coordinate vector so the two-loop recursion
    couples them automatically, then mapped back after the step.

lbfgs_set_reference_cell, lbfgs_cell_kappa, check_cell_is_aligned
    Variable-cell setup, for callers who fill the chart themselves rather than
    passing ``cell`` and ``n_particles`` to ``lbfgs_prepare_cell_state``.

    The phases each step is built from -- the reductions, the history update,
    the two-loop recursion, the trust region, the packing -- are internal
    decomposition points rather than separate operations, so they are not
    exported here. They remain importable from
    :mod:`nvalchemiops.dynamics.optimizers.lbfgs` for anyone who genuinely
    needs to interpose logic between them.

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
    LBFGSCellState,
    LBFGSState,
    check_cell_is_aligned,
    lbfgs_cell_kappa,
    lbfgs_prepare_cell_state,
    lbfgs_prepare_state,
    lbfgs_set_reference_cell,
    lbfgs_step,
    lbfgs_step_coord_cell,
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
    # L-BFGS: state, preparation, and one step per call. The phase functions
    # the step is built from are decomposition points, not operations, so they
    # stay in the module rather than on the public surface.
    "LBFGSState",
    "LBFGSCellState",
    "lbfgs_prepare_state",
    "lbfgs_prepare_cell_state",
    "lbfgs_step",
    "lbfgs_step_coord_cell",
    # L-BFGS variable-cell setup
    "lbfgs_set_reference_cell",
    "lbfgs_cell_kappa",
    "check_cell_is_aligned",
    # Low-level kernels
    "_fire_step_no_downhill_ptr_kernel",
    "_fire_step_downhill_ptr_kernel",
    "_fire_update_params_no_downhill_ptr_kernel",
    "_fire_update_params_downhill_ptr_kernel",
]
