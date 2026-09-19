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

"""PyTorch bindings for the batched L-BFGS geometry optimizer.

L-BFGS reaches a given force tolerance in far fewer force evaluations than the
FIRE optimizers, which is the cost that dominates relaxation with a
machine-learned potential.

State
-----
:func:`lbfgs_prepare_state` allocates, initializes and validates the whole
state in one call; calling it again is how you reset. Nothing is allocated per
step, which keeps it capturable in a CUDA graph.

The tensors stay yours.
:class:`~nvalchemiops.dynamics.optimizers.lbfgs.LBFGSState` is a plain
dataclass, so every field is reachable by name and you can build one from
tensors you already own -- see
:mod:`nvalchemiops.dynamics.optimizers.lbfgs` for the shapes and the required
initial contents.

Per-system scalars are float64 whatever the coordinate precision: ``ys / yy``
scales the initial inverse Hessian, and near convergence ``y = force_base - F``
is a difference of nearly equal vectors.

Usage
-----
You own the loop. Each call consumes exactly one force evaluation and mutates
``positions`` and the state in place::

    from nvalchemiops.torch.lbfgs import (
        LBFGS_NEED_EVAL, lbfgs_prepare_state, lbfgs_step_coord,
    )

    state = lbfgs_prepare_state(num_atoms, num_systems, device=positions.device)
    while True:
        forces = model(positions)
        lbfgs_step_coord(
            positions, forces, state, batch_idx, n_particles,
            force_tol=0.05, maxstep=0.2,
        )
        if not (state.status == LBFGS_NEED_EVAL).any():
            break

``status`` is the only value to inspect: ``LBFGS_NEED_EVAL`` means keep going,
``LBFGS_CONVERGED`` means ``positions`` hold the answer. There is no failure
status and no energy input -- the step length comes from a ``maxstep`` trust
region, so a model whose forces are not the gradient of its energy relaxes just
as well.

These operations mutate their inputs and are not differentiable; they are
registered as PyTorch custom operators so they trace under ``torch.compile``.

CUDA graphs
-----------
A step captures in a CUDA graph, worth doing for a loop that runs thousands of
times. Warp launches must be bound to the capture stream by the caller::

    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        with torch.cuda.graph(graph):
            lbfgs_step_coord(...)

Without that scope the capture records nothing and replay silently does no
work.

See Also
--------
nvalchemiops.dynamics.optimizers.lbfgs : the Warp implementation, which
    documents the algorithm, sign convention and precision policy.
"""

from __future__ import annotations

import dataclasses
import inspect

import torch
import warp as wp

from nvalchemiops.dynamics.optimizers.lbfgs import (
    _CELL_BUFFERS,
    _CELL_SCRATCH,
    _OPTIMIZER_BUFFERS,
    LBFGS_CONVERGED,
    LBFGS_NEED_EVAL,
    LBFGSCellState,
    LBFGSState,
)
from nvalchemiops.dynamics.optimizers.lbfgs import (
    _lbfgs_step_coord_cell_impl as _wp_step_cell,
)
from nvalchemiops.dynamics.optimizers.lbfgs import _lbfgs_step_impl as _wp_step
from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_cell_kappa as _wp_cell_kappa
from nvalchemiops.dynamics.optimizers.lbfgs import (
    lbfgs_set_reference_cell as _wp_set_reference_cell,
)
from nvalchemiops.torch._warp_op_helpers import (
    register_noop_fake,
    scoped_warp_stream,
    torch_custom_op,
)

__all__ = [
    "LBFGSCellState",
    "LBFGSState",
    "LBFGS_CONVERGED",
    "LBFGS_NEED_EVAL",
    "lbfgs_cell_kappa",
    "lbfgs_prepare_cell_state",
    "lbfgs_prepare_state",
    "lbfgs_set_reference_cell",
    "lbfgs_step_coord",
    "lbfgs_step_coord_cell",
    "lbfgs_step_extended",
]

_TORCH_TO_WP_VEC = {torch.float32: wp.vec3f, torch.float64: wp.vec3d}
_TORCH_TO_WP_MAT = {torch.float32: wp.mat33f, torch.float64: wp.mat33d}
_TORCH_TO_WP_SCALAR = {torch.float32: wp.float32, torch.float64: wp.float64}

#: Every tensor the coordinate step writes to. Derived from the shared buffer
#: order so the two cannot drift apart.
_MUTATED = ("positions",) + _OPTIMIZER_BUFFERS


def _wp(tensor: torch.Tensor, dtype):
    """View a Torch tensor as a Warp array, without copying.

    Deliberately refuses non-contiguous input rather than calling
    ``.contiguous()`` for you. These operations write through the view, so a
    silent copy would discard every update and leave the caller watching an
    optimizer that never moves. Call ``.contiguous()`` yourself if you need to.
    """
    if not tensor.is_contiguous():
        raise ValueError(
            "L-BFGS tensors must be contiguous, because the optimizer writes "
            "through them in place; a non-contiguous tensor would be copied and "
            "the updates lost. Call .contiguous() on the argument first."
        )
    return wp.from_torch(tensor.detach(), dtype=dtype)


@torch_custom_op("nvalchemiops::lbfgs_step", mutates_args=_MUTATED)
def _lbfgs_step_op(
    positions: torch.Tensor,
    forces: torch.Tensor,
    batch_idx: torch.Tensor,
    n_particles: torch.Tensor,
    x_base: torch.Tensor,
    force_base: torch.Tensor,
    direction: torch.Tensor,
    s_history: torch.Tensor,
    y_history: torch.Tensor,
    ys: torch.Tensor,
    yy: torch.Tensor,
    alpha_hist: torch.Tensor,
    beta_hist: torch.Tensor,
    ss: torch.Tensor,
    gg: torch.Tensor,
    fmax: torch.Tensor,
    frms_sq: torch.Tensor,
    smax: torch.Tensor,
    d0: torch.Tensor,
    dmax: torch.Tensor,
    dquad: torch.Tensor,
    alpha_step: torch.Tensor,
    status: torch.Tensor,
    iteration: torch.Tensor,
    end: torch.Tensor,
    n_loop: torch.Tensor,
    history_count: torch.Tensor,
    force_tol: float,
    rms_tol: float,
    stress_tol: float,
    maxstep: float,
    curvature_eps: float,
    compute_reductions: bool,
) -> None:
    """Run one registered L-BFGS step. All tensors are passed positionally.

    Launches are bound to PyTorch's current stream so that the step can be
    captured in a CUDA graph. Without this Warp would use its own stream and a
    capture would record nothing.
    """
    vec = _TORCH_TO_WP_VEC[positions.dtype]
    with scoped_warp_stream(positions.device):
        _wp_step(
            positions=_wp(positions, vec),
            forces=_wp(forces, vec),
            batch_idx=_wp(batch_idx, wp.int32),
            n_particles=_wp(n_particles, wp.int32),
            x_base=_wp(x_base, vec),
            force_base=_wp(force_base, vec),
            direction=_wp(direction, vec),
            s_history=_wp(s_history, vec),
            y_history=_wp(y_history, vec),
            ys=_wp(ys, wp.float64),
            yy=_wp(yy, wp.float64),
            alpha_hist=_wp(alpha_hist, wp.float64),
            beta_hist=_wp(beta_hist, wp.float64),
            ss=_wp(ss, wp.float64),
            gg=_wp(gg, wp.float64),
            fmax=_wp(fmax, wp.float64),
            frms_sq=_wp(frms_sq, wp.float64),
            smax=_wp(smax, wp.float64),
            d0=_wp(d0, wp.float64),
            dmax=_wp(dmax, wp.float64),
            dquad=_wp(dquad, wp.float64),
            alpha_step=_wp(alpha_step, wp.float64),
            status=_wp(status, wp.int32),
            iteration=_wp(iteration, wp.int32),
            end=_wp(end, wp.int32),
            n_loop=_wp(n_loop, wp.int32),
            history_count=_wp(history_count, wp.int32),
            force_tol=force_tol,
            rms_tol=rms_tol,
            stress_tol=stress_tol,
            maxstep=maxstep,
            curvature_eps=curvature_eps,
            compute_reductions=compute_reductions,
        )


register_noop_fake(_lbfgs_step_op)

# The operator signature is written out by hand, so pin its parameter names to
# the shared buffer order. Registration already rejects a name in
# ``mutates_args`` that does not exist, but only this catches a *reordering*,
# which would silently swap two tensors.
# forces, batch_idx, n_particles, positions
_STATE_SLICE = slice(4, 4 + len(_OPTIMIZER_BUFFERS))
_op_buffer_params = tuple(inspect.signature(_lbfgs_step_op).parameters)[_STATE_SLICE]
if _op_buffer_params != _OPTIMIZER_BUFFERS:
    # Raised rather than asserted so the guard survives `python -O`.
    raise RuntimeError(
        "_OPTIMIZER_BUFFERS and _lbfgs_step_op parameters have diverged:\n"
        f"  expected: {_OPTIMIZER_BUFFERS}\n"
        f"  operator: {_op_buffer_params}"
    )


def lbfgs_prepare_state(
    num_dofs: int,
    num_systems: int,
    *,
    dtype: torch.dtype = torch.float64,
    device=None,
    history_size: int = 6,
) -> LBFGSState:
    """Allocate, initialize and validate a complete optimizer state.

    The three fields that do not start at zero -- ``alpha_step``,
    ``iteration``, ``status`` -- are set for you. Calling this again is how you
    reset. :class:`LBFGSState` is a plain dataclass, so you can equally build
    one from tensors you already own and call
    :meth:`~nvalchemiops.dynamics.optimizers.lbfgs.LBFGSState.validate`.

    Parameters
    ----------
    num_dofs : int
        Degrees of freedom. On the variable-cell path this is
        ``num_atoms + 2 * num_systems``.
    num_systems : int
        Independent systems in the batch.
    dtype : torch.dtype, optional
        Coordinate precision. Per-system scalars are float64 either way.
    device : optional
        Torch device.
    history_size : int, optional
        Stored curvature pairs.

    Returns
    -------
    LBFGSState
    """
    if dtype not in _TORCH_TO_WP_VEC:
        raise ValueError(f"dtype must be float32 or float64; got {dtype}")
    if history_size < 1:
        raise ValueError(f"history_size must be >= 1; got {history_size}")
    f64 = {"dtype": torch.float64, "device": device}
    i32 = {"dtype": torch.int32, "device": device}
    kw = {"dtype": dtype, "device": device}
    m, p_, n = history_size, num_dofs, num_systems
    state = LBFGSState(
        x_base=torch.zeros(p_, 3, **kw),
        force_base=torch.zeros(p_, 3, **kw),
        direction=torch.zeros(p_, 3, **kw),
        s_history=torch.zeros(m, p_, 3, **kw),
        y_history=torch.zeros(m, p_, 3, **kw),
        ys=torch.zeros(m, n, **f64),
        yy=torch.zeros(m, n, **f64),
        alpha_hist=torch.zeros(m, n, **f64),
        beta_hist=torch.zeros(m, n, **f64),
        ss=torch.zeros(n, **f64),
        gg=torch.zeros(n, **f64),
        fmax=torch.zeros(n, **f64),
        frms_sq=torch.zeros(n, **f64),
        smax=torch.zeros(n, **f64),
        d0=torch.zeros(n, **f64),
        dmax=torch.zeros(n, **f64),
        dquad=torch.zeros(n, **f64),
        alpha_step=torch.ones(n, **f64),
        status=torch.zeros(n, **i32),
        iteration=torch.full((n,), -1, **i32),
        end=torch.zeros(n, **i32),
        n_loop=torch.zeros(n, **i32),
        history_count=torch.zeros(n, **i32),
    )
    state.status.fill_(LBFGS_NEED_EVAL)
    state.validate()
    return state


def lbfgs_prepare_cell_state(
    num_atoms: int,
    num_systems: int,
    ext_batch_idx: torch.Tensor,
    ext_atom_ptr: torch.Tensor,
    *,
    cell: torch.Tensor | None = None,
    n_particles: torch.Tensor | None = None,
    cell_force_scale: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> LBFGSCellState:
    """Allocate and validate the variable-cell chart and its scratch space.

    The packed topology is yours: build ``ext_batch_idx`` and ``ext_atom_ptr``
    with the generic batch utilities so ragged batches are expressible.
    Pass ``cell`` and ``n_particles`` to get a state that is ready to step:
    the chart fields ``ref_cell``, ``ref_cell_inv`` and ``kappa`` are filled
    for you. Omit them and those three are left zeroed, which a step cannot
    use, so fill them with :func:`lbfgs_set_reference_cell` and
    :func:`lbfgs_cell_kappa` first.

    Returns
    -------
    LBFGSCellState
    """
    if dtype not in _TORCH_TO_WP_VEC:
        raise ValueError(f"dtype must be float32 or float64; got {dtype}")
    kw = {"dtype": dtype, "device": device}
    n, packed = num_systems, num_atoms + 2 * num_systems
    state = LBFGSCellState(
        ref_cell=torch.zeros(n, 3, 3, **kw),
        ref_cell_inv=torch.zeros(n, 3, 3, **kw),
        kappa=torch.zeros(n, **kw),
        ext_batch_idx=ext_batch_idx,
        ext_atom_ptr=ext_atom_ptr,
        phi=torch.zeros(n, 3, 3, **kw),
        phi_inv=torch.zeros(n, 3, 3, **kw),
        d_phi=torch.zeros(n, 3, 3, **kw),
        cell_dof_a=torch.zeros(n, 3, **kw),
        cell_dof_b=torch.zeros(n, 3, **kw),
        cell_force_a=torch.zeros(n, 3, **kw),
        cell_force_b=torch.zeros(n, 3, **kw),
        ext_positions=torch.zeros(packed, 3, **kw),
        ext_forces=torch.zeros(packed, 3, **kw),
    )
    state.validate(num_atoms=num_atoms)
    if (cell is None) != (n_particles is None):
        raise ValueError("cell and n_particles must be given together")
    if cell is not None:
        lbfgs_set_reference_cell(cell, state.ref_cell, state.ref_cell_inv)
        lbfgs_cell_kappa(n_particles, state.kappa, cell_force_scale=cell_force_scale)
    return state


def _state_args(state) -> dict:
    """The dataclass's tensors keyed by field name."""
    return {f.name: getattr(state, f.name) for f in dataclasses.fields(state)}


def lbfgs_step_coord(
    positions: torch.Tensor,
    forces: torch.Tensor,
    state: LBFGSState,
    batch_idx: torch.Tensor,
    n_particles: torch.Tensor,
    *,
    force_tol: float = 0.05,
    rms_tol: float = 0.0,
    stress_tol: float = 0.0,
    maxstep: float = 0.2,
    curvature_eps: float = 1e-10,
    compute_reductions: bool = True,
) -> None:
    """Advance one batched L-BFGS step, consuming one force evaluation.

    Mutates ``positions`` and every field of ``state`` in place.

    Parameters
    ----------
    positions, forces : torch.Tensor, shape (num_atoms, 3)
        Current geometry and the forces there. Forces, not gradients.
    state : LBFGSState
        From :func:`lbfgs_prepare_state`, or built from your own tensors.
    batch_idx : torch.Tensor, shape (num_atoms,), dtype int32
        Sorted system index per atom.
    n_particles : torch.Tensor, shape (num_systems,), dtype int32
        Atom count per system, for the optional RMS criterion.
    force_tol : float, optional
        Threshold on the largest per-atom force magnitude. Zero disables it.
    rms_tol, stress_tol : float, optional
        Additional criteria, disabled by default.
    maxstep : float, optional
        Largest distance any atom may move in one step.
    curvature_eps : float, optional
        Threshold below which a curvature pair is discarded.
    compute_reductions : bool, optional
        Set ``False`` only if you have already filled the reduction fields of
        ``state`` yourself for this geometry.

    Raises
    ------
    ValueError
        If this call's inputs are incompatible with the prepared state.
    """
    _validate(positions, forces, batch_idx, n_particles, state)
    _lbfgs_step_op(
        positions, forces, batch_idx, n_particles, **_state_args(state),
        force_tol=force_tol, rms_tol=rms_tol, stress_tol=stress_tol,
        maxstep=maxstep, curvature_eps=curvature_eps,
        compute_reductions=compute_reductions,
    )  # fmt: skip


def lbfgs_step_extended(
    ext_positions: torch.Tensor,
    ext_forces: torch.Tensor,
    state: LBFGSState,
    ext_batch_idx: torch.Tensor,
    n_particles: torch.Tensor,
    **kwargs,
) -> None:
    """Advance one step on caller-packed degrees of freedom.

    Identical to :func:`lbfgs_step_coord` and backed by the same operator; only
    the meaning of the arrays differs. Convergence is evaluated on whatever
    ``ext_forces`` holds, so if those are not Cartesian atomic forces,
    ``force_tol`` will not mean a force per atom.
    """
    lbfgs_step_coord(
        ext_positions, ext_forces, state, ext_batch_idx, n_particles, **kwargs
    )


def lbfgs_step_coord_cell(
    positions: torch.Tensor,
    cell: torch.Tensor,
    forces: torch.Tensor,
    stress: torch.Tensor,
    state: LBFGSState,
    cell_state: LBFGSCellState,
    batch_idx: torch.Tensor,
    n_particles: torch.Tensor,
    *,
    force_tol: float = 0.05,
    rms_tol: float = 0.0,
    stress_tol: float = 0.0,
    maxstep: float = 0.2,
    curvature_eps: float = 1e-10,
) -> None:
    """Advance one variable-cell step, relaxing coordinates and cell together.

    Mutates ``positions``, ``cell`` and both states in place. ``state`` must be
    sized for ``num_atoms + 2 * num_systems`` degrees of freedom.

    Parameters
    ----------
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Lattice vectors as columns, kept lower-triangular.
    stress : torch.Tensor, shape (num_systems, 3, 3)
        Cauchy stress. Drives the cell degrees of freedom and is what
        ``stress_tol`` is compared against.
    state, cell_state : LBFGSState, LBFGSCellState
        From the preparation functions, or built from your own tensors.

    See Also
    --------
    lbfgs_set_reference_cell : must be called first.
    lbfgs_cell_kappa : must be called first.
    """
    _validate_cell(positions, forces, cell, stress, batch_idx, state, cell_state)
    _lbfgs_step_coord_cell_op(
        forces, stress, batch_idx, n_particles, positions, cell,
        **_state_args(state), **_state_args(cell_state),
        force_tol=force_tol, rms_tol=rms_tol, stress_tol=stress_tol,
        maxstep=maxstep, curvature_eps=curvature_eps,
    )  # fmt: skip


def _check_scalar_precision(state) -> None:
    """Confirm the per-system scalars are float64.

    ``LBFGSState.validate`` checks only that they agree with each other, since
    it is shared with the Warp and JAX layers; the float64 requirement is
    spelled in this layer's own dtype vocabulary.
    """
    if state.ys.dtype != torch.float64:
        raise ValueError(
            f"per-system scalars must be float64, got {state.ys.dtype}; "
            "ys / yy scales the initial inverse Hessian and cancels in fp32"
        )


def _validate(positions, forces, batch_idx, n_particles, state):
    """Confirm this call's inputs match the prepared state.

    Shapes only -- no device reads. The state re-checks itself too, since its
    fields can be reassigned between steps.
    """
    state.validate()
    _check_scalar_precision(state)
    if positions.dtype not in _TORCH_TO_WP_VEC:
        raise ValueError(f"positions must be float32 or float64; got {positions.dtype}")
    if forces.shape != positions.shape:
        raise ValueError(
            f"forces shape {tuple(forces.shape)} != positions shape "
            f"{tuple(positions.shape)}"
        )
    if positions.shape[0] != state.num_dofs:
        raise ValueError(
            f"positions has {positions.shape[0]} degrees of freedom but the "
            f"state was prepared for {state.num_dofs}"
        )
    if batch_idx.shape[0] != positions.shape[0]:
        raise ValueError(
            f"batch_idx length {batch_idx.shape[0]} != positions length "
            f"{positions.shape[0]}"
        )
    if n_particles.shape[0] != state.num_systems:
        raise ValueError(
            f"n_particles length {n_particles.shape[0]} != number of systems "
            f"{state.num_systems}"
        )


_CELL_MUTATED = ("positions", "cell") + _OPTIMIZER_BUFFERS + _CELL_SCRATCH


@torch_custom_op("nvalchemiops::lbfgs_step_coord_cell", mutates_args=_CELL_MUTATED)
def _lbfgs_step_coord_cell_op(
    forces: torch.Tensor,
    stress: torch.Tensor,
    batch_idx: torch.Tensor,
    n_particles: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    x_base: torch.Tensor,
    force_base: torch.Tensor,
    direction: torch.Tensor,
    s_history: torch.Tensor,
    y_history: torch.Tensor,
    ys: torch.Tensor,
    yy: torch.Tensor,
    alpha_hist: torch.Tensor,
    beta_hist: torch.Tensor,
    ss: torch.Tensor,
    gg: torch.Tensor,
    fmax: torch.Tensor,
    frms_sq: torch.Tensor,
    smax: torch.Tensor,
    d0: torch.Tensor,
    dmax: torch.Tensor,
    dquad: torch.Tensor,
    alpha_step: torch.Tensor,
    status: torch.Tensor,
    iteration: torch.Tensor,
    end: torch.Tensor,
    n_loop: torch.Tensor,
    history_count: torch.Tensor,
    ref_cell: torch.Tensor,
    ref_cell_inv: torch.Tensor,
    kappa: torch.Tensor,
    ext_batch_idx: torch.Tensor,
    ext_atom_ptr: torch.Tensor,
    phi: torch.Tensor,
    phi_inv: torch.Tensor,
    d_phi: torch.Tensor,
    cell_dof_a: torch.Tensor,
    cell_dof_b: torch.Tensor,
    cell_force_a: torch.Tensor,
    cell_force_b: torch.Tensor,
    ext_positions: torch.Tensor,
    ext_forces: torch.Tensor,
    force_tol: float,
    rms_tol: float,
    stress_tol: float,
    maxstep: float,
    curvature_eps: float,
) -> None:
    """Run one registered variable-cell L-BFGS step."""
    vec = _TORCH_TO_WP_VEC[positions.dtype]
    mat = _TORCH_TO_WP_MAT[positions.dtype]
    scalar = _TORCH_TO_WP_SCALAR[positions.dtype]
    with scoped_warp_stream(positions.device):
        _wp_step_cell(
            positions=_wp(positions, vec),
            forces=_wp(forces, vec),
            cell=_wp(cell, mat),
            stress=_wp(stress, mat),
            batch_idx=_wp(batch_idx, wp.int32),
            n_particles=_wp(n_particles, wp.int32),
            x_base=_wp(x_base, vec),
            force_base=_wp(force_base, vec),
            direction=_wp(direction, vec),
            s_history=_wp(s_history, vec),
            y_history=_wp(y_history, vec),
            ys=_wp(ys, wp.float64),
            yy=_wp(yy, wp.float64),
            alpha_hist=_wp(alpha_hist, wp.float64),
            beta_hist=_wp(beta_hist, wp.float64),
            ss=_wp(ss, wp.float64),
            gg=_wp(gg, wp.float64),
            fmax=_wp(fmax, wp.float64),
            frms_sq=_wp(frms_sq, wp.float64),
            smax=_wp(smax, wp.float64),
            d0=_wp(d0, wp.float64),
            dmax=_wp(dmax, wp.float64),
            dquad=_wp(dquad, wp.float64),
            alpha_step=_wp(alpha_step, wp.float64),
            status=_wp(status, wp.int32),
            iteration=_wp(iteration, wp.int32),
            end=_wp(end, wp.int32),
            n_loop=_wp(n_loop, wp.int32),
            history_count=_wp(history_count, wp.int32),
            ref_cell=_wp(ref_cell, mat),
            ref_cell_inv=_wp(ref_cell_inv, mat),
            kappa=_wp(kappa, scalar),
            ext_batch_idx=_wp(ext_batch_idx, wp.int32),
            ext_atom_ptr=_wp(ext_atom_ptr, wp.int32),
            phi=_wp(phi, mat),
            phi_inv=_wp(phi_inv, mat),
            d_phi=_wp(d_phi, mat),
            cell_dof_a=_wp(cell_dof_a, vec),
            cell_dof_b=_wp(cell_dof_b, vec),
            cell_force_a=_wp(cell_force_a, vec),
            cell_force_b=_wp(cell_force_b, vec),
            ext_positions=_wp(ext_positions, vec),
            ext_forces=_wp(ext_forces, vec),
            force_tol=force_tol,
            rms_tol=rms_tol,
            stress_tol=stress_tol,
            maxstep=maxstep,
            curvature_eps=curvature_eps,
        )


register_noop_fake(_lbfgs_step_coord_cell_op)

_n_opt = len(_OPTIMIZER_BUFFERS)
_cell_params = tuple(inspect.signature(_lbfgs_step_coord_cell_op).parameters)
if _cell_params[6 : 6 + _n_opt] != _OPTIMIZER_BUFFERS:
    raise RuntimeError("_OPTIMIZER_BUFFERS and the cell operator have diverged")
if _cell_params[6 + _n_opt : 6 + _n_opt + len(_CELL_BUFFERS)] != _CELL_BUFFERS:
    raise RuntimeError("_CELL_BUFFERS and the cell operator have diverged")


def lbfgs_set_reference_cell(
    cell: torch.Tensor, ref_cell: torch.Tensor, ref_cell_inv: torch.Tensor
) -> None:
    """Capture the reference cell that defines the variable-cell chart.

    Writes the caller's ``ref_cell`` and ``ref_cell_inv`` buffers in place.

    Coordinates are measured relative to a cell held fixed for the whole
    relaxation, which is what makes history pairs from different iterations
    comparable. Call this once, before the first step. Calling it again
    re-references the chart and invalidates every stored curvature pair, so
    restore the optimizer buffers to their initial contents if you do.

    Parameters
    ----------
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Current cell, lattice vectors as columns.
    ref_cell, ref_cell_inv : torch.Tensor, shape (num_systems, 3, 3)
        OUTPUT. Caller-owned buffers for the reference cell and its inverse.
    """
    mat = _TORCH_TO_WP_MAT[cell.dtype]
    # Bind Warp to PyTorch's current stream, as the step operators do. Without
    # it the launch sits on Warp's own stream and, on a non-default Torch
    # stream, can race the producer of `cell` or the next consumer of the
    # outputs.
    with scoped_warp_stream(cell.device):
        _wp_set_reference_cell(
            _wp(cell, mat), _wp(ref_cell, mat), _wp(ref_cell_inv, mat)
        )


def lbfgs_cell_kappa(
    n_particles: torch.Tensor,
    kappa: torch.Tensor,
    *,
    cell_force_scale: float = 1.0,
) -> None:
    """Fill the per-system cell coordinate scaling, in place.

    The cell coordinate is ``kappa * Phi`` and its conjugate force is divided
    by the same ``kappa``, which is what keeps ``g . dx`` independent of the
    scaling. Depends only on topology, so compute it once and reuse it.

    Parameters
    ----------
    n_particles : torch.Tensor, shape (num_systems,), dtype int32
        Atom count per system.
    kappa : torch.Tensor, shape (num_systems,)
        OUTPUT. Must match the **coordinate** precision, not float64, because
        it scales matrices.
    cell_force_scale : float, optional
        Multiplier on the atom count. Larger values make the cell move less per
        step relative to the atoms; ``1 / atoms_per_system`` puts the two on a
        comparable footing.
    """
    if kappa.dtype not in _TORCH_TO_WP_SCALAR:
        raise ValueError(f"kappa must be float32 or float64; got {kappa.dtype}")
    with scoped_warp_stream(kappa.device):
        _wp_cell_kappa(
            _wp(n_particles, wp.int32),
            _wp(kappa, _TORCH_TO_WP_SCALAR[kappa.dtype]),
            cell_force_scale=cell_force_scale,
        )


def _validate_cell(positions, forces, cell, stress, batch_idx, state, cell_state):
    """Confirm this call's inputs match both prepared states."""
    state.validate()
    cell_state.validate(num_atoms=positions.shape[0])
    _check_scalar_precision(state)
    num_systems = state.num_systems
    if positions.dtype not in _TORCH_TO_WP_VEC:
        raise ValueError(f"positions must be float32 or float64; got {positions.dtype}")
    if forces.shape != positions.shape:
        raise ValueError(
            f"forces shape {tuple(forces.shape)} != positions shape "
            f"{tuple(positions.shape)}"
        )
    if cell.shape != (num_systems, 3, 3):
        raise ValueError(
            f"cell must have shape ({num_systems}, 3, 3); got {tuple(cell.shape)}"
        )
    if stress.shape != cell.shape:
        raise ValueError(
            f"stress shape {tuple(stress.shape)} != cell shape {tuple(cell.shape)}"
        )
    if cell.dtype != positions.dtype or stress.dtype != positions.dtype:
        raise ValueError("cell and stress must share the dtype of positions")
    if batch_idx.shape[0] != positions.shape[0]:
        raise ValueError(
            f"batch_idx length {batch_idx.shape[0]} != positions length "
            f"{positions.shape[0]}"
        )
    expected = positions.shape[0] + 2 * num_systems
    if state.num_dofs != expected:
        raise ValueError(
            f"state is sized for {state.num_dofs} degrees of freedom, but the "
            f"variable-cell path needs {expected} (num_atoms + 2 * num_systems)"
        )
    if cell_state.num_packed_dofs != expected:
        raise ValueError(
            f"cell state is packed for {cell_state.num_packed_dofs} degrees of "
            f"freedom, expected {expected}"
        )
