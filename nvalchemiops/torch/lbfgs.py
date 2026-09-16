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

L-BFGS reaches a given force tolerance in far fewer energy/force evaluations
than the FIRE optimizers, which is the cost that dominates relaxation with a
machine-learned potential.

Caller-owned buffers
--------------------
As with the FIRE optimizers, **you allocate, initialize and retain every
buffer**; nothing is allocated here. That keeps allocation out of the step,
which is what makes it capturable in a CUDA graph. See
:mod:`nvalchemiops.dynamics.optimizers.lbfgs` for the full table of shapes and
required initial contents. In short, for ``P`` degrees of freedom, ``M``
systems and history depth ``m``::

    kw = dict(dtype=positions.dtype, device=positions.device)
    f64 = dict(dtype=torch.float64, device=positions.device)
    i32 = dict(dtype=torch.int32, device=positions.device)

    buffers = dict(
        x_base=torch.zeros(P, 3, **kw),
        force_base=torch.zeros(P, 3, **kw),
        direction=torch.zeros(P, 3, **kw),
        s_history=torch.zeros(m, P, 3, **kw),
        y_history=torch.zeros(m, P, 3, **kw),
        ys=torch.zeros(m, M, **f64),
        yy=torch.zeros(m, M, **f64),
        alpha_hist=torch.zeros(m, M, **f64),
        beta_hist=torch.zeros(m, M, **f64),
        ss=torch.zeros(M, **f64),
        f_base=torch.zeros(M, **f64),
        gg=torch.zeros(M, **f64),
        gd=torch.zeros(M, **f64),
        fmax=torch.zeros(M, **f64),
        frms_sq=torch.zeros(M, **f64),
        smax=torch.zeros(M, **f64),
        d0=torch.zeros(M, **f64),
        dmax=torch.zeros(M, **f64),
        dquad=torch.zeros(M, **f64),
        alpha_step=torch.ones(M, **f64),       # one, not zero
        status=torch.zeros(M, **i32),          # LBFGS_NEED_EVAL is zero
        iteration=torch.full((M,), -1, **i32),  # minus one
        end=torch.zeros(M, **i32),
        n_loop=torch.zeros(M, **i32),
        ls_trials=torch.zeros(M, **i32),
        history_count=torch.zeros(M, **i32),
    )

Per-system scalars are float64 whatever the coordinate precision: the line
search compares a difference of *total* energies, and a single-precision
accumulator would be coarser than the differences being tested.

Restoring those same values is what resets the optimizer; there is no reset
helper, because there is no state object to reset.

Usage
-----
You own the loop. Each call consumes exactly one energy/force evaluation and
mutates its buffers in place::

    from nvalchemiops.torch.lbfgs import LBFGS_NEED_EVAL, lbfgs_step_coord

    while True:
        per_atom_energy, forces = model(positions)
        lbfgs_reduce_energy(per_atom_energy, batch_idx, energy)
        lbfgs_step_coord(
            positions, forces, energy, batch_idx, n_particles,
            **buffers, force_tol=0.05, maxstep=0.2,
        )
        if not (buffers["status"] == LBFGS_NEED_EVAL).any():
            break

``status`` is the only value you need to inspect: ``LBFGS_NEED_EVAL`` means
keep going, ``LBFGS_CONVERGED`` means ``positions`` hold the answer, and
``LBFGS_LS_FAILED`` means the line search stalled even from a steepest-descent
direction and ``positions`` were restored to the last accepted point.

These operations mutate their inputs and are not differentiable; they are
registered as PyTorch custom operators so they trace correctly under
``torch.compile``.

CUDA graphs
-----------
A step captures in a CUDA graph, which is worth doing for a loop that runs
thousands of times. Warp launches have to be bound to the capture stream by the
caller, so wrap the capture::

    import warp as wp

    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        with torch.cuda.graph(graph):
            lbfgs_step_coord(...)

Without that scope the capture records nothing and replay silently does no
work. Every buffer must be pre-allocated and reused; the step itself allocates
nothing and does all of its zeroing on the device.

See Also
--------
nvalchemiops.dynamics.optimizers.lbfgs : the underlying Warp implementation,
    which documents the algorithm, the sign convention and the precision policy
    in detail.
"""

from __future__ import annotations

import inspect

import torch
import warp as wp

from nvalchemiops.dynamics.optimizers.lbfgs import (
    _CELL_BUFFERS,
    _CELL_SCRATCH,
    _OPTIMIZER_BUFFERS,
    LBFGS_CONVERGED,
    LBFGS_LS_FAILED,
    LBFGS_NEED_EVAL,
)
from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_cell_kappa as _wp_cell_kappa
from nvalchemiops.dynamics.optimizers.lbfgs import (
    lbfgs_reduce_energy as _wp_reduce_energy,
)
from nvalchemiops.dynamics.optimizers.lbfgs import (
    lbfgs_set_reference_cell as _wp_set_reference_cell,
)
from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_step as _wp_step
from nvalchemiops.dynamics.optimizers.lbfgs import (
    lbfgs_step_coord_cell as _wp_step_cell,
)
from nvalchemiops.torch._warp_op_helpers import (
    register_noop_fake,
    scoped_warp_stream,
    torch_custom_op,
)

__all__ = [
    "LBFGS_CONVERGED",
    "LBFGS_LS_FAILED",
    "LBFGS_NEED_EVAL",
    "lbfgs_cell_kappa",
    "lbfgs_reduce_energy",
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
    energy: torch.Tensor,
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
    f_base: torch.Tensor,
    gg: torch.Tensor,
    gd: torch.Tensor,
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
    ls_trials: torch.Tensor,
    history_count: torch.Tensor,
    force_tol: float,
    rms_tol: float,
    stress_tol: float,
    ftol: float,
    wolfe: float,
    step_scale_down: float,
    step_scale_up: float,
    min_step: float,
    max_step: float,
    max_ls_iter: int,
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
            energy=_wp(energy, wp.float64),
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
            f_base=_wp(f_base, wp.float64),
            gg=_wp(gg, wp.float64),
            gd=_wp(gd, wp.float64),
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
            ls_trials=_wp(ls_trials, wp.int32),
            history_count=_wp(history_count, wp.int32),
            force_tol=force_tol,
            rms_tol=rms_tol,
            stress_tol=stress_tol,
            ftol=ftol,
            wolfe=wolfe,
            step_scale_down=step_scale_down,
            step_scale_up=step_scale_up,
            min_step=min_step,
            max_step=max_step,
            max_ls_iter=max_ls_iter,
            maxstep=maxstep,
            curvature_eps=curvature_eps,
            compute_reductions=compute_reductions,
        )


register_noop_fake(_lbfgs_step_op)

# The operator signature is written out by hand, so pin its parameter names to
# the shared buffer order. Registration already rejects a name in
# ``mutates_args`` that does not exist, but only this catches a *reordering*,
# which would silently swap two tensors.
_STATE_SLICE = slice(5, 5 + len(_OPTIMIZER_BUFFERS))
_op_buffer_params = tuple(inspect.signature(_lbfgs_step_op).parameters)[_STATE_SLICE]
if _op_buffer_params != _OPTIMIZER_BUFFERS:
    # Raised rather than asserted so the guard survives `python -O`.
    raise RuntimeError(
        "_OPTIMIZER_BUFFERS and _lbfgs_step_op parameters have diverged:\n"
        f"  expected: {_OPTIMIZER_BUFFERS}\n"
        f"  operator: {_op_buffer_params}"
    )


def lbfgs_reduce_energy(
    per_atom_energy: torch.Tensor,
    batch_idx: torch.Tensor,
    energy: torch.Tensor,
) -> None:
    """Sum per-atom energies into per-system totals, accumulating in float64.

    Use this when your model returns per-atom energies, which is the common
    case and the recommended one. The Armijo test compares a difference of
    *total* energies, and at ``E ~ -1e4 eV`` a float32 total is already rounded
    to about ``1e-3 eV`` before it reaches the optimizer -- enough to make the
    line search unreliable near convergence. Summing in float64 avoids that;
    the per-atom values may be single precision.

    Parameters
    ----------
    per_atom_energy : torch.Tensor, shape (num_atoms,)
    batch_idx : torch.Tensor, shape (num_atoms,), dtype int32
        Sorted system index per atom.
    energy : torch.Tensor, shape (num_systems,), dtype float64
        OUTPUT. Zeroed internally.
    """
    if energy.dtype != torch.float64:
        raise ValueError(f"energy must be float64; got {energy.dtype}")
    wp_dtype = wp.float32 if per_atom_energy.dtype == torch.float32 else wp.float64
    _wp_reduce_energy(
        _wp(per_atom_energy, wp_dtype),
        _wp(batch_idx, wp.int32),
        _wp(energy, wp.float64),
    )


def lbfgs_step_coord(
    positions: torch.Tensor,
    forces: torch.Tensor,
    energy: torch.Tensor,
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
    f_base: torch.Tensor,
    gg: torch.Tensor,
    gd: torch.Tensor,
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
    ls_trials: torch.Tensor,
    history_count: torch.Tensor,
    *,
    force_tol: float = 0.05,
    rms_tol: float = 0.0,
    stress_tol: float = 0.0,
    ftol: float = 0.0001,
    wolfe: float = 0.9,
    step_scale_down: float = 0.5,
    step_scale_up: float = 2.1,
    min_step: float = 1e-20,
    max_step: float = 1e20,
    max_ls_iter: int = 40,
    maxstep: float = 0.2,
    curvature_eps: float = 1e-10,
    compute_reductions: bool = True,
) -> None:
    """Advance one batched L-BFGS step, consuming one force evaluation.

    Mutates ``positions`` and every optimizer buffer in place. Progress is
    reported through ``status``; see the module docstring.

    Parameters
    ----------
    positions : torch.Tensor, shape (num_atoms, 3)
        Current geometry. Advanced to the next trial point.
    forces : torch.Tensor, shape (num_atoms, 3)
        Forces at ``positions``. Forces, not gradients.
    energy : torch.Tensor, shape (num_systems,), dtype float64
        Per-system total energy. Use :func:`lbfgs_reduce_energy` if your model
        returns per-atom energies.
    batch_idx : torch.Tensor, shape (num_atoms,), dtype int32
        Sorted system index per atom.
    n_particles : torch.Tensor, shape (num_systems,), dtype int32
        Atom count per system, for the optional RMS convergence criterion.
    x_base, ..., history_count : torch.Tensor
        The caller-owned optimizer buffers, in this fixed order. See the module
        docstring for shapes and required initial contents.
    force_tol : float, optional
        Convergence threshold on the largest per-atom force magnitude, in the
        force units you supplied. Zero disables it.
    rms_tol, stress_tol : float, optional
        Additional convergence thresholds, disabled by default. All enabled
        criteria must hold.
    maxstep : float, optional
        Largest distance any atom may move in one step. Zero disables the
        trust region.
    max_ls_iter : int, optional
        Trials allowed in one line search before the history is discarded and
        the search restarts from steepest descent.
    curvature_eps : float, optional
        Relative threshold below which a curvature pair is judged unusable and
        discarded.

    Raises
    ------
    ValueError
        If dtypes or shapes are inconsistent.

    See Also
    --------
    lbfgs_step_extended : the same operator on caller-packed degrees of freedom.
    """
    _validate(positions, forces, energy, batch_idx, n_particles, s_history, status)
    _lbfgs_step_op(
        positions,
        forces,
        energy,
        batch_idx,
        n_particles,
        x_base,
        force_base,
        direction,
        s_history,
        y_history,
        ys,
        yy,
        alpha_hist,
        beta_hist,
        ss,
        f_base,
        gg,
        gd,
        fmax,
        frms_sq,
        smax,
        d0,
        dmax,
        dquad,
        alpha_step,
        status,
        iteration,
        end,
        n_loop,
        ls_trials,
        history_count,
        force_tol,
        rms_tol,
        stress_tol,
        ftol,
        wolfe,
        step_scale_down,
        step_scale_up,
        min_step,
        max_step,
        max_ls_iter,
        maxstep,
        curvature_eps,
        compute_reductions,
    )


def lbfgs_step_extended(
    ext_positions: torch.Tensor,
    ext_forces: torch.Tensor,
    energy: torch.Tensor,
    ext_batch_idx: torch.Tensor,
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
    f_base: torch.Tensor,
    gg: torch.Tensor,
    gd: torch.Tensor,
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
    ls_trials: torch.Tensor,
    history_count: torch.Tensor,
    **kwargs,
) -> None:
    """Advance one step on caller-packed degrees of freedom.

    Identical to :func:`lbfgs_step_coord` and backed by the same registered
    operator; only the meaning of the arrays differs. Use it when you have
    packed extra degrees of freedom alongside the atoms, as variable-cell
    relaxation does.

    Note that convergence is evaluated on whatever ``ext_forces`` contains. If
    those are not Cartesian atomic forces, ``force_tol`` will not mean a force
    per atom, and you should drive convergence yourself from ``status`` and
    your own reductions.
    """
    lbfgs_step_coord(
        ext_positions,
        ext_forces,
        energy,
        ext_batch_idx,
        n_particles,
        x_base,
        force_base,
        direction,
        s_history,
        y_history,
        ys,
        yy,
        alpha_hist,
        beta_hist,
        ss,
        f_base,
        gg,
        gd,
        fmax,
        frms_sq,
        smax,
        d0,
        dmax,
        dquad,
        alpha_step,
        status,
        iteration,
        end,
        n_loop,
        ls_trials,
        history_count,
        **kwargs,
    )


def _validate(positions, forces, energy, batch_idx, n_particles, s_history, status):
    """Check the shapes and dtypes that would otherwise fail deep in a kernel."""
    num_dofs = positions.shape[0]
    if positions.dtype not in _TORCH_TO_WP_VEC:
        raise ValueError(f"positions must be float32 or float64; got {positions.dtype}")
    if forces.shape != positions.shape:
        raise ValueError(
            f"forces shape {tuple(forces.shape)} != positions shape "
            f"{tuple(positions.shape)}"
        )
    if forces.dtype != positions.dtype:
        raise ValueError(
            f"forces dtype {forces.dtype} != positions dtype {positions.dtype}"
        )
    if batch_idx.shape[0] != num_dofs:
        raise ValueError(
            f"batch_idx length {batch_idx.shape[0]} != positions length {num_dofs}"
        )
    if energy.dtype != torch.float64:
        raise ValueError(f"energy must be float64; got {energy.dtype}")
    num_systems = status.shape[0]
    if energy.shape[0] != num_systems:
        raise ValueError(
            f"energy length {energy.shape[0]} != number of systems {num_systems}"
        )
    if n_particles.shape[0] != num_systems:
        raise ValueError(
            f"n_particles length {n_particles.shape[0]} != number of systems "
            f"{num_systems}"
        )
    if s_history.shape[1] != num_dofs:
        raise ValueError(
            f"history buffers hold {s_history.shape[1]} degrees of freedom, "
            f"but positions has {num_dofs}"
        )


# =============================================================================
# Variable-cell relaxation
# =============================================================================

#: Tensors the variable-cell step writes to. ``ref_cell``, ``ref_cell_inv``,
#: ``kappa``, ``ext_batch_idx`` and ``ext_atom_ptr`` are read-only
#: configuration and topology, so they are inputs rather than mutated buffers.
_CELL_MUTATED = ("positions", "cell") + _OPTIMIZER_BUFFERS + _CELL_SCRATCH


@torch_custom_op("nvalchemiops::lbfgs_step_coord_cell", mutates_args=_CELL_MUTATED)
def _lbfgs_step_coord_cell_op(
    forces: torch.Tensor,
    stress: torch.Tensor,
    energy: torch.Tensor,
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
    f_base: torch.Tensor,
    gg: torch.Tensor,
    gd: torch.Tensor,
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
    ls_trials: torch.Tensor,
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
    ftol: float,
    wolfe: float,
    step_scale_down: float,
    step_scale_up: float,
    min_step: float,
    max_step: float,
    max_ls_iter: int,
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
            energy=_wp(energy, wp.float64),
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
            f_base=_wp(f_base, wp.float64),
            gg=_wp(gg, wp.float64),
            gd=_wp(gd, wp.float64),
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
            ls_trials=_wp(ls_trials, wp.int32),
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
            ftol=ftol,
            wolfe=wolfe,
            step_scale_down=step_scale_down,
            step_scale_up=step_scale_up,
            min_step=min_step,
            max_step=max_step,
            max_ls_iter=max_ls_iter,
            maxstep=maxstep,
            curvature_eps=curvature_eps,
        )


register_noop_fake(_lbfgs_step_coord_cell_op)

_n_opt = len(_OPTIMIZER_BUFFERS)
_cell_params = tuple(inspect.signature(_lbfgs_step_coord_cell_op).parameters)
if _cell_params[7 : 7 + _n_opt] != _OPTIMIZER_BUFFERS:
    raise RuntimeError("_OPTIMIZER_BUFFERS and the cell operator have diverged")
if _cell_params[7 + _n_opt : 7 + _n_opt + len(_CELL_BUFFERS)] != _CELL_BUFFERS:
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
    _wp_set_reference_cell(_wp(cell, mat), _wp(ref_cell, mat), _wp(ref_cell_inv, mat))


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
    _wp_cell_kappa(
        _wp(n_particles, wp.int32),
        _wp(kappa, _TORCH_TO_WP_SCALAR[kappa.dtype]),
        cell_force_scale=cell_force_scale,
    )


def lbfgs_step_coord_cell(
    positions: torch.Tensor,
    cell: torch.Tensor,
    forces: torch.Tensor,
    stress: torch.Tensor,
    energy: torch.Tensor,
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
    f_base: torch.Tensor,
    gg: torch.Tensor,
    gd: torch.Tensor,
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
    ls_trials: torch.Tensor,
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
    *,
    force_tol: float = 0.05,
    rms_tol: float = 0.0,
    stress_tol: float = 0.0,
    ftol: float = 0.0001,
    wolfe: float = 0.9,
    step_scale_down: float = 0.5,
    step_scale_up: float = 2.1,
    min_step: float = 1e-20,
    max_step: float = 1e20,
    max_ls_iter: int = 40,
    maxstep: float = 0.2,
    curvature_eps: float = 1e-10,
) -> None:
    """Advance one variable-cell step, relaxing coordinates and cell together.

    Mutates ``positions``, ``cell``, every optimizer buffer and the cell
    scratch buffers in place. Consumes exactly one energy/force/stress
    evaluation, and reports progress through ``status`` exactly like the
    coordinate-only path.

    The optimizer buffers must be sized for ``num_atoms + 2 * num_systems``
    degrees of freedom, since the cell contributes two entries per system.

    Call :func:`lbfgs_set_reference_cell` and :func:`lbfgs_cell_kappa` once
    before the first step, and build ``ext_batch_idx`` / ``ext_atom_ptr``
    yourself so ragged batches are expressible.

    Parameters
    ----------
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Lattice vectors as columns. Kept lower-triangular, so the cell cannot
        drift into a rotation.
    stress : torch.Tensor, shape (num_systems, 3, 3)
        Cauchy stress. Drives the cell degrees of freedom and is what
        ``stress_tol`` is compared against; the packed cell force cannot be
        used for that, since it carries units of energy rather than stress.
    x_base, ..., history_count : torch.Tensor
        The caller-owned optimizer buffers, in the same order as
        :func:`lbfgs_step_coord`.
    ref_cell, ..., ext_forces : torch.Tensor
        The caller-owned cell buffers. The first five are read-only
        configuration and topology; the rest is scratch.
    force_tol, rms_tol, stress_tol : float, optional
        Convergence thresholds, always evaluated on the Cartesian forces and
        the stress, so ``force_tol`` keeps its meaning as a force per atom
        however far the cell deforms.
    maxstep : float, optional
        Largest Cartesian distance an atom may move in one step.

    See Also
    --------
    lbfgs_set_reference_cell : must be called first.
    lbfgs_cell_kappa : must be called first.
    lbfgs_step_coord : the coordinate-only equivalent.
    """
    _validate_cell(
        positions, forces, cell, stress, energy, batch_idx, s_history, status
    )
    _lbfgs_step_coord_cell_op(
        forces,
        stress,
        energy,
        batch_idx,
        n_particles,
        positions,
        cell,
        x_base,
        force_base,
        direction,
        s_history,
        y_history,
        ys,
        yy,
        alpha_hist,
        beta_hist,
        ss,
        f_base,
        gg,
        gd,
        fmax,
        frms_sq,
        smax,
        d0,
        dmax,
        dquad,
        alpha_step,
        status,
        iteration,
        end,
        n_loop,
        ls_trials,
        history_count,
        ref_cell,
        ref_cell_inv,
        kappa,
        ext_batch_idx,
        ext_atom_ptr,
        phi,
        phi_inv,
        d_phi,
        cell_dof_a,
        cell_dof_b,
        cell_force_a,
        cell_force_b,
        ext_positions,
        ext_forces,
        force_tol,
        rms_tol,
        stress_tol,
        ftol,
        wolfe,
        step_scale_down,
        step_scale_up,
        min_step,
        max_step,
        max_ls_iter,
        maxstep,
        curvature_eps,
    )


def _validate_cell(
    positions, forces, cell, stress, energy, batch_idx, s_history, status
):
    """Check the variable-cell shapes that would otherwise fail in a kernel."""
    num_atoms = positions.shape[0]
    num_systems = status.shape[0]
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
    if energy.dtype != torch.float64:
        raise ValueError(f"energy must be float64; got {energy.dtype}")
    if batch_idx.shape[0] != num_atoms:
        raise ValueError(
            f"batch_idx length {batch_idx.shape[0]} != number of atoms {num_atoms}"
        )
    expected = num_atoms + 2 * num_systems
    if s_history.shape[1] != expected:
        raise ValueError(
            f"optimizer buffers are sized for {s_history.shape[1]} degrees of "
            f"freedom, but the variable-cell path needs {expected} "
            f"(num_atoms + 2 * num_systems)"
        )
