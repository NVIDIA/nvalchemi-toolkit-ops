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
FIRE optimizers -- the cost that dominates relaxation with a machine-learned
potential.

Usage
-----
You own the loop and the stopping rule. Each call consumes exactly one force
evaluation and mutates ``positions`` and the state in place::

    from nvalchemiops.torch.lbfgs import lbfgs_prepare_state, lbfgs_step_coord

    state = lbfgs_prepare_state(num_atoms, num_systems, device=positions.device)
    for _ in range(max_steps):
        forces = model(positions)
        if forces.norm(dim=1).max() < force_tol:
            break
        lbfgs_step_coord(positions, forces, state, batch_idx, maxstep=0.2)

Test before stepping, as above: after the step the forces you passed describe
the previous point. The optimizer owns no tolerance and has no terminal
status, matching FIRE2, and reads no energy -- so a model whose forces are not
the gradient of its energy relaxes just as well.

These operations mutate their inputs and are not differentiable; they are
registered as PyTorch custom operators, so they trace under ``torch.compile``.

State
-----
:func:`lbfgs_prepare_state` allocates, initializes and validates the whole
state in one call; calling it again is how you reset. Nothing is allocated per
step. The tensors stay yours --
:class:`~nvalchemiops.dynamics.optimizers.lbfgs.LBFGSState` is a plain
dataclass, so you can build one from tensors you already own; see
:mod:`nvalchemiops.dynamics.optimizers.lbfgs` for shapes and required initial
contents.

Every array follows the coordinate dtype, so ``torch.float32`` gives an
end-to-end fp32 state and ``torch.float64`` an fp64 one.

CUDA graphs
-----------
A step captures with nothing special required of the caller::

    with torch.cuda.graph(graph):
        lbfgs_step_coord(positions, forces, state, batch_idx, maxstep=0.2)

The registered operators bind Warp to PyTorch's current stream themselves, and
during capture that *is* the capture stream, so an outer
``wp.ScopedStream(wp.stream_from_torch(...))`` is redundant -- harmless, but
not what makes capture work. A raw Warp launch of your own in the same
captured region still needs the caller to bind the stream.

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

from nvalchemiops.batch_utils import atom_ptr_to_batch_idx as _wp_atom_ptr_to_batch_idx
from nvalchemiops.dynamics.optimizers.lbfgs import (
    _CELL_BUFFERS,
    _CELL_SCRATCH,
    _OPTIMIZER_BUFFERS,
    LBFGSCellState,
    LBFGSState,
    _resolve_curvature_eps,
    check_against_state,
)
from nvalchemiops.dynamics.optimizers.lbfgs import (
    _lbfgs_step_coord_cell_impl as _wp_step_cell,
)
from nvalchemiops.dynamics.optimizers.lbfgs import _lbfgs_step_impl as _wp_step
from nvalchemiops.dynamics.optimizers.lbfgs import (
    check_packed_topology as _wp_check_packed_topology,
)
from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_cell_kappa as _wp_cell_kappa
from nvalchemiops.dynamics.optimizers.lbfgs import (
    lbfgs_set_reference_cell as _wp_set_reference_cell,
)
from nvalchemiops.dynamics.utils.cell_filter import (
    extend_atom_ptr as _wp_extend_atom_ptr,
)
from nvalchemiops.torch._warp_op_helpers import (
    register_noop_fake,
    scoped_warp_stream,
    torch_custom_op,
)

__all__ = [
    "LBFGSCellState",
    "LBFGSState",
    "lbfgs_prepare_cell_state",
    "lbfgs_prepare_state",
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
    d0: torch.Tensor,
    dmax: torch.Tensor,
    dquad: torch.Tensor,
    alpha_step: torch.Tensor,
    iteration: torch.Tensor,
    end: torch.Tensor,
    n_loop: torch.Tensor,
    history_count: torch.Tensor,
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
    sc = _TORCH_TO_WP_SCALAR[positions.dtype]
    with scoped_warp_stream(positions.device):
        _wp_step(
            positions=_wp(positions, vec),
            forces=_wp(forces, vec),
            batch_idx=_wp(batch_idx, wp.int32),
            x_base=_wp(x_base, vec),
            force_base=_wp(force_base, vec),
            direction=_wp(direction, vec),
            s_history=_wp(s_history, vec),
            y_history=_wp(y_history, vec),
            ys=_wp(ys, sc),
            yy=_wp(yy, sc),
            alpha_hist=_wp(alpha_hist, sc),
            beta_hist=_wp(beta_hist, sc),
            ss=_wp(ss, sc),
            gg=_wp(gg, sc),
            d0=_wp(d0, sc),
            dmax=_wp(dmax, sc),
            dquad=_wp(dquad, sc),
            alpha_step=_wp(alpha_step, sc),
            iteration=_wp(iteration, wp.int32),
            end=_wp(end, wp.int32),
            n_loop=_wp(n_loop, wp.int32),
            history_count=_wp(history_count, wp.int32),
            maxstep=maxstep,
            curvature_eps=curvature_eps,
            compute_reductions=compute_reductions,
        )


register_noop_fake(_lbfgs_step_op)

# The operator signature is written out by hand, so pin its parameter names to
# the shared buffer order. Registration already rejects a name in
# ``mutates_args`` that does not exist, but only this catches a *reordering*,
# which would silently swap two tensors.
# positions, forces, batch_idx
_STATE_SLICE = slice(3, 3 + len(_OPTIMIZER_BUFFERS))
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
    ``iteration`` -- are set for you. Calling this again is how you
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
        Coordinate precision. Every per-system scalar follows it, so this
        selects an end-to-end fp32 or fp64 state.
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
    sc = {"dtype": dtype, "device": device}
    i32 = {"dtype": torch.int32, "device": device}
    kw = {"dtype": dtype, "device": device}
    m, p_, n = history_size, num_dofs, num_systems
    state = LBFGSState(
        x_base=torch.zeros(p_, 3, **kw),
        force_base=torch.zeros(p_, 3, **kw),
        direction=torch.zeros(p_, 3, **kw),
        s_history=torch.zeros(p_, m, 3, **kw),
        y_history=torch.zeros(p_, m, 3, **kw),
        ys=torch.zeros(n, m, **sc),
        yy=torch.zeros(n, m, **sc),
        alpha_hist=torch.zeros(n, m, **sc),
        beta_hist=torch.zeros(n, m, **sc),
        ss=torch.zeros(n, **sc),
        gg=torch.zeros(n, **sc),
        d0=torch.zeros(n, **sc),
        dmax=torch.zeros(n, **sc),
        dquad=torch.zeros(n, **sc),
        alpha_step=torch.ones(n, **sc),
        iteration=torch.full((n,), -1, **i32),
        end=torch.zeros(n, **i32),
        n_loop=torch.zeros(n, **i32),
        history_count=torch.zeros(n, **i32),
    )
    state.validate()
    return state


def lbfgs_prepare_cell_state(
    atom_ptr: torch.Tensor,
    cell: torch.Tensor,
    *,
    cell_force_scale: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> LBFGSCellState:
    """Build a complete, ready-to-step variable-cell chart from atom topology.

    One call: give it the ordinary ``atom_ptr`` and the aligned cells, and it
    derives the packed topology, captures the reference chart and computes
    ``kappa``. Nothing needs repairing before the first step.

    Ragged batches are unaffected -- ``atom_ptr`` already carries each
    system's atom count, so nothing here assumes an even split.

    Parameters
    ----------
    atom_ptr : torch.Tensor, shape (num_systems + 1,), int32
        CSR-style atom pointer for the batch.
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Reference lattice per system, aligned.
    cell_force_scale : float, optional
        Multiplier on the atom count in ``kappa``.
    dtype : torch.dtype, optional
        Coordinate precision.
    device : optional
        Defaults to ``atom_ptr.device``.

    Returns
    -------
    LBFGSCellState

    See Also
    --------
    nvalchemiops.dynamics.optimizers.lbfgs : the variable-cell contract,
        including the requirement that the cells be aligned first.
    """
    if dtype not in _TORCH_TO_WP_VEC:
        raise ValueError(f"dtype must be float32 or float64; got {dtype}")
    if device is None:
        device = atom_ptr.device
    # A host tensor handed to a device kernel is a segmentation fault, not an
    # error. ``torch.empty(0, ...)`` resolves an index-less ``"cuda"`` to the
    # current device so that spelling is not wrongly refused.
    resolved = torch.empty(0, device=device).device
    for name, tensor in (("atom_ptr", atom_ptr), ("cell", cell)):
        if tensor.device != resolved:
            raise ValueError(
                f"{name} is on {tensor.device}, but the state is being built "
                f"on {resolved}; preparation does not move inputs between "
                "devices"
            )

    ptr = atom_ptr.detach().cpu().tolist()
    if len(ptr) < 2:
        raise ValueError(
            f"atom_ptr must have num_systems + 1 >= 2 entries; got {len(ptr)}"
        )
    num_systems = len(ptr) - 1
    num_atoms = int(ptr[-1])
    counts = [int(ptr[s + 1]) - int(ptr[s]) for s in range(num_systems)]
    if any(c < 0 for c in counts):
        raise ValueError(f"atom_ptr must be non-decreasing; got {ptr}")

    kw = {"dtype": dtype, "device": device}
    i32 = {"dtype": torch.int32, "device": device}
    n, packed = num_systems, num_atoms + 2 * num_systems

    # Derived, not carried: a function of the topology alone.
    ext_atom_ptr = torch.zeros(n + 1, **i32)
    ext_batch_idx = torch.zeros(packed, **i32)
    with scoped_warp_stream(device):
        _wp_extend_atom_ptr(
            wp.from_torch(atom_ptr.to(torch.int32).contiguous(), dtype=wp.int32),
            wp.from_torch(ext_atom_ptr, dtype=wp.int32),
        )
        _wp_atom_ptr_to_batch_idx(
            wp.from_torch(ext_atom_ptr, dtype=wp.int32),
            wp.from_torch(ext_batch_idx, dtype=wp.int32),
        )

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
    _wp_check_packed_topology(
        wp.from_torch(ext_atom_ptr, dtype=wp.int32),
        wp.from_torch(ext_batch_idx, dtype=wp.int32),
        n,
        packed,
    )
    n_particles = torch.tensor(counts, **i32)
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
    *,
    maxstep: float = 0.2,
    curvature_eps: float | None = None,
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
    maxstep : float, optional
        Largest distance any atom may move in one step.
    curvature_eps : float, optional
        Relative threshold below which a curvature pair is discarded. Defaults
        to ``1e-6`` for float32 coordinates and ``1e-10`` for float64: ``ys``
        is accumulated at the coordinate precision, so one value cannot serve
        both.
    compute_reductions : bool, optional
        Set ``False`` only if you have already filled the reduction fields of
        ``state`` yourself for this geometry.

    Raises
    ------
    ValueError
        If this call's inputs are incompatible with the prepared state.
    """
    _validate(positions, forces, batch_idx, state)
    _lbfgs_step_op(
        positions, forces, batch_idx, **_state_args(state),
        maxstep=maxstep,
        curvature_eps=_resolve_curvature_eps(curvature_eps, _TORCH_TO_WP_SCALAR[positions.dtype]),
        compute_reductions=compute_reductions,
    )  # fmt: skip


def lbfgs_step_extended(
    ext_positions: torch.Tensor,
    ext_forces: torch.Tensor,
    state: LBFGSState,
    ext_batch_idx: torch.Tensor,
    **kwargs,
) -> None:
    """Advance one step on caller-packed degrees of freedom.

    Identical to :func:`lbfgs_step_coord` and backed by the same operator;
    only the meaning of the arrays differs. ``maxstep`` bounds the packed
    displacement, so if these are not Cartesian atom positions it is not a
    distance in angstroms.
    """
    lbfgs_step_coord(ext_positions, ext_forces, state, ext_batch_idx, **kwargs)


def lbfgs_step_coord_cell(
    positions: torch.Tensor,
    cell: torch.Tensor,
    forces: torch.Tensor,
    stress: torch.Tensor,
    state: LBFGSState,
    cell_state: LBFGSCellState,
    batch_idx: torch.Tensor,
    *,
    maxstep: float = 0.2,
    curvature_eps: float | None = None,
) -> None:
    """Advance one variable-cell step, relaxing coordinates and cell together.

    Mutates ``positions``, ``cell`` and both states in place. ``state`` must be
    sized for ``num_atoms + 2 * num_systems`` degrees of freedom.

    Parameters
    ----------
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Lattice vectors as columns, aligned by ``align_cell`` before the
        first step. See :ref:`the variable-cell contract <lbfgs-cell-contract>`
        in :mod:`nvalchemiops.dynamics.optimizers.lbfgs`, which is the single
        authoritative statement of the cell rules.
    stress : torch.Tensor, shape (num_systems, 3, 3)
        Cauchy stress, which drives the cell degrees of freedom.
    state, cell_state : LBFGSState, LBFGSCellState
        From the preparation functions, or built from your own tensors.

    See Also
    --------
    lbfgs_set_reference_cell : must be called first.
    lbfgs_cell_kappa : must be called first.
    """
    _validate_cell(positions, forces, cell, stress, batch_idx, state, cell_state)
    _lbfgs_step_coord_cell_op(
        forces, stress, batch_idx, positions, cell,
        **_state_args(state), **_state_args(cell_state),
        maxstep=maxstep,
        curvature_eps=_resolve_curvature_eps(curvature_eps, _TORCH_TO_WP_SCALAR[positions.dtype]),
    )  # fmt: skip


def _validate(positions, forces, batch_idx, state):
    """Confirm this call's inputs match the prepared state.

    Shapes only -- no device reads. The state re-checks itself too, since its
    fields can be reassigned between steps.
    """
    state.validate()
    check_against_state(
        state,
        coordinates=(("positions", positions), ("forces", forces)),
        indices=(("batch_idx", batch_idx),),
    )
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


_CELL_MUTATED = ("positions", "cell") + _OPTIMIZER_BUFFERS + _CELL_SCRATCH


@torch_custom_op("nvalchemiops::lbfgs_step_coord_cell", mutates_args=_CELL_MUTATED)
def _lbfgs_step_coord_cell_op(
    forces: torch.Tensor,
    stress: torch.Tensor,
    batch_idx: torch.Tensor,
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
    d0: torch.Tensor,
    dmax: torch.Tensor,
    dquad: torch.Tensor,
    alpha_step: torch.Tensor,
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
    maxstep: float,
    curvature_eps: float,
) -> None:
    """Run one registered variable-cell L-BFGS step."""
    vec = _TORCH_TO_WP_VEC[positions.dtype]
    mat = _TORCH_TO_WP_MAT[positions.dtype]
    sc = _TORCH_TO_WP_SCALAR[positions.dtype]
    with scoped_warp_stream(positions.device):
        _wp_step_cell(
            positions=_wp(positions, vec),
            forces=_wp(forces, vec),
            cell=_wp(cell, mat),
            stress=_wp(stress, mat),
            batch_idx=_wp(batch_idx, wp.int32),
            x_base=_wp(x_base, vec),
            force_base=_wp(force_base, vec),
            direction=_wp(direction, vec),
            s_history=_wp(s_history, vec),
            y_history=_wp(y_history, vec),
            ys=_wp(ys, sc),
            yy=_wp(yy, sc),
            alpha_hist=_wp(alpha_hist, sc),
            beta_hist=_wp(beta_hist, sc),
            ss=_wp(ss, sc),
            gg=_wp(gg, sc),
            d0=_wp(d0, sc),
            dmax=_wp(dmax, sc),
            dquad=_wp(dquad, sc),
            alpha_step=_wp(alpha_step, sc),
            iteration=_wp(iteration, wp.int32),
            end=_wp(end, wp.int32),
            n_loop=_wp(n_loop, wp.int32),
            history_count=_wp(history_count, wp.int32),
            ref_cell=_wp(ref_cell, mat),
            ref_cell_inv=_wp(ref_cell_inv, mat),
            kappa=_wp(kappa, sc),
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
            maxstep=maxstep,
            curvature_eps=curvature_eps,
        )


register_noop_fake(_lbfgs_step_coord_cell_op)

_n_opt = len(_OPTIMIZER_BUFFERS)
_cell_params = tuple(inspect.signature(_lbfgs_step_coord_cell_op).parameters)
if _cell_params[5 : 5 + _n_opt] != _OPTIMIZER_BUFFERS:
    raise RuntimeError("_OPTIMIZER_BUFFERS and the cell operator have diverged")
if _cell_params[5 + _n_opt : 5 + _n_opt + len(_CELL_BUFFERS)] != _CELL_BUFFERS:
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
    check_against_state(
        state,
        coordinates=(
            ("positions", positions),
            ("forces", forces),
            ("cell", cell),
            ("stress", stress),
        ),  # fmt: skip
        indices=(("batch_idx", batch_idx),),
        extra_states=(("cell_state", cell_state),),
    )
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
