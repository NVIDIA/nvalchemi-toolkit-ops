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

r"""
L-BFGS Optimizer Kernels
========================

GPU-accelerated Warp kernels for batched L-BFGS geometry optimization.

L-BFGS is a quasi-Newton method. It builds an implicit approximation to the
inverse Hessian from the last ``m`` position/gradient differences and uses it
to choose a search direction, then picks a step length along that direction
with a line search. Compared with the FIRE optimizers it typically reaches a
given force tolerance in far fewer energy/force evaluations, which is the cost
that dominates relaxation with a machine-learned potential.

Calling convention
------------------
You own the loop. Each call to :func:`lbfgs_step` consumes **exactly one**
energy/force evaluation::

    while True:
        energy, forces = my_model(positions)
        lbfgs_step(positions, forces, energy, ..., batch_idx=batch_idx)
        if not (status.numpy() == LBFGS_NEED_EVAL).any():
            break

Internally each system runs a small state machine that decides whether the
evaluation it was just handed is a line-search trial to accept or reject, or an
accepted point that starts a fresh L-BFGS iteration. Systems in a batch
therefore stay in lock step in *evaluations* while diverging in *iterations*,
which is what lets a whole batch relax in one stream of kernel launches with no
per-system host control flow.

Reading ``status``
------------------
``status`` is the only value you need to inspect:

``LBFGS_NEED_EVAL``
    Keep going: ``positions`` hold a new trial point that needs energy/forces.
``LBFGS_CONVERGED``
    Done: ``positions`` hold the converged geometry.
``LBFGS_LS_FAILED``
    The line search could not make further progress *even from a steepest-descent
    direction with no history*. ``positions`` have been restored to the last
    accepted point. This is *not* a convergence claim; if you consider a stalled
    search with acceptably small forces to be a success, apply that policy
    yourself from ``status`` and the returned forces.

A line search that stalls while history is present is not reported as a
failure. The optimizer rolls back to the last accepted point, discards the
history and continues from steepest descent, which is usually enough to get
moving again. ``iteration`` therefore counts iterations since the most recent
such restart rather than since the beginning of the run.

Forces, not gradients
---------------------
The public API is expressed in **forces**. Because ``F = -grad E``, the
optimizer's gradient is ``g = -F``. No gradient array is ever materialized:
``force_base`` stores forces, and each kernel folds the sign into its own
expression. Two consequences are worth knowing when reading the state:

- ``y_history`` holds *gradient* differences, computed as ``force_base - F``.
- A valid descent direction satisfies ``force_base . d > 0`` (the direction
  points along the force), which is the sign-folded form of ``d0 < 0``.

Precision
---------
Coordinates may be single or double precision, but **all per-system scalars are
float64 regardless**. The Armijo test compares a difference of total energies;
at ``E ~ -1e4 eV`` a float32 ULP is ``~1e-3 eV``, so a single-precision
accumulator would make the line search a coin flip near convergence. The
per-system arrays are ``O(num_systems)`` and the cost is negligible.

Caller-owned buffers
--------------------
The package allocates and initializes **nothing**: every persistent and scratch
buffer is yours to create, initialize and keep alive across steps, exactly as
with the FIRE optimizers. That keeps allocation out of the step, which is what
makes the step capturable in a CUDA graph and free of per-call allocation.

For ``P`` degrees of freedom, ``M`` systems and history depth ``m``:

============================================  ==========================  =========
Buffer                                        Shape                       dtype
============================================  ==========================  =========
``x_base``, ``force_base``, ``direction``     ``(P,)``                    vec3f/vec3d
``s_history``, ``y_history``                  ``(m, P)``                  vec3f/vec3d
``ys``, ``yy``, ``alpha_hist``, ``beta_hist`` ``(m, M)``                  float64
``ss``, ``f_base``, ``gg``, ``gd``            ``(M,)``                    float64
``fmax``, ``frms_sq``, ``smax``               ``(M,)``                    float64
``d0``, ``dmax``, ``dquad``, ``alpha_step``   ``(M,)``                    float64
``status``, ``iteration``, ``end``            ``(M,)``                    int32
``n_loop``, ``ls_trials``, ``history_count``  ``(M,)``                    int32
============================================  ==========================  =========

Per-system scalars are float64 whatever the coordinate precision; see
`Precision`_ below.

Initial contents, before the first step:

- **zero** for every buffer except the two below;
- ``alpha_step`` to **one**;
- ``iteration`` to **minus one**, the "never evaluated" marker.

``status`` starts at ``LBFGS_NEED_EVAL``, which is numerically zero, so zeroing
it is correct. Reset the optimizer -- to discard the history after changing the
potential, say -- by restoring those same values.

Variable-cell relaxation adds, for the packed path with
``P = num_atoms + 2 * M``:

=====================================================  ====================  ==========
Buffer                                                 Shape                 dtype
=====================================================  ====================  ==========
``ref_cell``, ``ref_cell_inv``, ``phi``, ``phi_inv``   ``(M,)``              mat33f/mat33d
``d_phi``                                              ``(M,)``              mat33f/mat33d
``kappa``                                              ``(M,)``              float32/float64
``cell_dof_a/b``, ``cell_force_a/b``                   ``(M,)``              vec3f/vec3d
``ext_positions``, ``ext_forces``                      ``(P,)``              vec3f/vec3d
``ext_batch_idx``                                      ``(P,)``              int32
``ext_atom_ptr``                                       ``(M + 1,)``          int32
=====================================================  ====================  ==========

``kappa`` matches the *coordinate* precision, not float64, because it scales
matrices. ``ref_cell``, ``ref_cell_inv`` and ``kappa`` are filled once by
:func:`lbfgs_set_reference_cell` and :func:`lbfgs_cell_kappa`; the rest is
scratch and may start as anything.

Build the packed topology yourself, so ragged batches are expressible::

    from nvalchemiops.batch_utils import atom_ptr_to_batch_idx
    from nvalchemiops.dynamics.utils.cell_filter import extend_atom_ptr

    extend_atom_ptr(atom_ptr, ext_atom_ptr)        # ext_atom_ptr[s] = atom_ptr[s] + 2s
    atom_ptr_to_batch_idx(ext_atom_ptr, ext_batch_idx)

Memory
------
The optimizer state costs, for ``P`` degrees of freedom, ``M`` systems and a
history depth ``m``::

    (2m + 3) * 3 * sizeof(dof) * P     per-DOF vectors and the s/y history
  + (4m + 11) * 8 * M                  per-slot and per-system float64 scalars
  +        6  * 4 * M                  per-system int32

``positions`` is not included: it belongs to the caller. The history dominates,
so ``m`` is the knob to turn if memory is tight; 3 to 7 is the usual range. At
``m = 6`` with single-precision coordinates this is 180 bytes per degree of
freedom, or 180 MB at a million.

References
----------
Nocedal, J. "Updating Quasi-Newton Matrices with Limited Storage."
*Math. Comp.* 35 (1980) 773-782.

Liu, D. C. and Nocedal, J. "On the limited memory BFGS method for large scale
optimization." *Math. Program.* 45 (1989) 503-528.

Nocedal, J. and Wright, S. J. *Numerical Optimization*, 2nd ed., chapters 3
and 7.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from nvalchemiops.dynamics.utils.cell_utils import compute_cell_inverse
from nvalchemiops.segment_ops import compute_ept

__all__ = [
    "LBFGS_CONVERGED",
    "LBFGS_LS_FAILED",
    "LBFGS_NEED_EVAL",
    "lbfgs_apply_step",
    "lbfgs_cell_kappa",
    "lbfgs_cell_trust_region",
    "lbfgs_pack_cell",
    "lbfgs_prepare_step",
    "lbfgs_reduce",
    "lbfgs_reduce_energy",
    "lbfgs_set_reference_cell",
    "lbfgs_step",
    "lbfgs_step_coord_cell",
    "lbfgs_unpack_cell",
    "lbfgs_update",
]

# =============================================================================
# State container
# =============================================================================


# =============================================================================
# Buffer ordering
#
# Every buffer is owned, allocated and retained by the caller, as with the FIRE
# optimizers. These tuples are the single source of truth for the order the
# public entry points take them in, and exist only so the bindings can validate
# their own signatures against it. They deliberately do not hold arrays.
# =============================================================================

#: Optimizer buffers, in the order every coordinate-path entry point takes them.
_OPTIMIZER_BUFFERS: tuple[str, ...] = (
    "x_base",
    "force_base",
    "direction",
    "s_history",
    "y_history",
    "ys",
    "yy",
    "alpha_hist",
    "beta_hist",
    "ss",
    "f_base",
    "gg",
    "gd",
    "fmax",
    "frms_sq",
    "smax",
    "d0",
    "dmax",
    "dquad",
    "alpha_step",
    "status",
    "iteration",
    "end",
    "n_loop",
    "ls_trials",
    "history_count",
)

#: Variable-cell buffers, appended after the optimizer buffers on that path.
#: The first five are read-only configuration and topology; the rest is scratch
#: the step writes to.
_CELL_BUFFERS: tuple[str, ...] = (
    "ref_cell",
    "ref_cell_inv",
    "kappa",
    "ext_batch_idx",
    "ext_atom_ptr",
    "phi",
    "phi_inv",
    "d_phi",
    "cell_dof_a",
    "cell_dof_b",
    "cell_force_a",
    "cell_force_b",
    "ext_positions",
    "ext_forces",
)

#: The subset of ``_CELL_BUFFERS`` the step mutates.
_CELL_SCRATCH: tuple[str, ...] = _CELL_BUFFERS[5:]


# =============================================================================
# Public status codes
# =============================================================================

#: The system needs another energy/force evaluation at the current positions.
LBFGS_NEED_EVAL = 0
#: The system has converged; ``positions`` hold the relaxed geometry.
LBFGS_CONVERGED = 1
#: The line search stalled; ``positions`` were restored to the last good point.
LBFGS_LS_FAILED = 2

# -----------------------------------------------------------------------------
# Internal ``n_loop`` sentinels.
#
# ``n_loop`` is the single per-system value every downstream kernel reads to
# decide what work it owes this call. Non-negative values carry a count:
# ``n_loop == 0`` is a line-search retry (keep the direction, only alpha
# changed) and ``n_loop == history_count + 1`` drives the two-loop recursion.
# -----------------------------------------------------------------------------
_NLOOP_PENDING = -4  # accepted; the (s, y) pair is written but not yet committed
_NLOOP_ROLLBACK = -3  # line search exhausted; restore positions from x_base
_NLOOP_SEED = -2  # converged on arrival; seed the base buffers, do not move
_NLOOP_RESTART = -1  # seed or restart: take a steepest-descent direction
_NLOOP_RETRY = 0  # line-search retry: same direction, new alpha

_BIG = 1.0e300  # stands in for "no trust-region limit"


# =============================================================================
# Device helpers
# =============================================================================


@wp.func
def _converged(
    fmax: wp.float64,
    frms_sq: wp.float64,
    smax: wp.float64,
    n_particles: wp.int32,
    force_tol: wp.float64,
    rms_tol: wp.float64,
    stress_tol: wp.float64,
) -> wp.bool:
    """Evaluate the convergence criteria for one system.

    All enabled criteria must hold. A tolerance of zero disables its criterion,
    so adding one can only make convergence stricter.

    Parameters
    ----------
    fmax
        Largest per-atom force magnitude, in Cartesian space.
    frms_sq
        Sum of squared per-atom force magnitudes, in Cartesian space.
    smax
        Spectral norm of the Cauchy stress. Ignored unless ``stress_tol > 0``.
    n_particles
        Atom count for this system (not the degree-of-freedom count).
    """
    zero = wp.float64(0.0)
    ok = True
    if force_tol > zero:
        ok = ok and (fmax <= force_tol)
    if rms_tol > zero:
        ok = ok and (frms_sq <= rms_tol * rms_tol * wp.float64(n_particles))
    if stress_tol > zero:
        ok = ok and (smax <= stress_tol)
    return ok


@wp.func
def _alpha_cap(
    a_lin: wp.float64, b_quad: wp.float64, maxstep: wp.float64
) -> wp.float64:
    """Largest step length whose Cartesian displacement stays within ``maxstep``.

    The displacement of an atom is ``alpha * a_lin + alpha**2 * b_quad`` in the
    worst case, so bounding it by ``maxstep`` is a quadratic in ``alpha``. On
    the coordinate-only path ``b_quad`` is zero and this reduces to
    ``maxstep / a_lin``.

    The positive root is taken in the form

        alpha = 2 * maxstep / (a_lin + sqrt(a_lin**2 + 4 * b_quad * maxstep))

    rather than the textbook ``(-a_lin + sqrt(...)) / (2 * b_quad)``. The two
    are equivalent in exact arithmetic, but the textbook form subtracts two
    nearly equal numbers whenever ``b_quad`` is small next to ``a_lin**2``, and
    that is the *usual* case here: the quadratic term is second order in the
    step. At ``b_quad ~ 1e-17`` the subtraction cancels completely and returns
    zero, which would freeze the optimizer with a step length of exactly zero.
    The form used here has no subtraction, so it stays accurate all the way
    down to ``b_quad = 0``.

    A non-positive ``maxstep`` disables the trust region.
    """
    zero = wp.float64(0.0)
    if maxstep <= zero or a_lin <= zero:
        return wp.float64(_BIG)
    disc = a_lin * a_lin + wp.float64(4.0) * b_quad * maxstep
    return (wp.float64(2.0) * maxstep) / (a_lin + wp.sqrt(disc))


@wp.func
def _slot(end: wp.int32, back: wp.int32, m: wp.int32) -> wp.int32:
    """Ring-buffer slot holding the ``back``-th newest pair (0 = newest).

    ``end`` is the next slot that will be written, so the newest committed pair
    sits one position behind it.
    """
    return ((end - 1 - back) % m + m) % m


# =============================================================================
# Kernel 1: per-system reductions
# =============================================================================


@wp.kernel(enable_backward=False)
def _lbfgs_reduce_kernel(
    forces: wp.array(dtype=Any),
    direction: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    gg: wp.array(dtype=wp.float64),
    gd: wp.array(dtype=wp.float64),
    n_dofs: wp.int32,
    elems_per_thread: wp.int32,
):
    """Reduce the packed force/direction inner products for one L-BFGS call.

    Computes, per system:

    ``gg``
        ``f . f`` over the packed degrees of freedom, used to normalize a
        steepest-descent direction. This lives in the space the direction lives
        in, which on a variable-cell path is *not* Cartesian space.
    ``gd``
        ``-(f . d)``, the directional derivative in gradient space. Negative
        for a valid descent direction.

    Convergence quantities are deliberately not computed here; see
    :func:`_lbfgs_convergence_kernel`.

    Thread launch
    -------------
    One thread per ``elems_per_thread`` consecutive degrees of freedom;
    ``dim = ceil(n_dofs / elems_per_thread)``. Requires ``batch_idx`` sorted in
    non-decreasing order.

    Modifies
    --------
    gg, gd
        OUTPUT. Accumulated atomically; the launcher zeroes them first.
        Systems whose ``status`` is not ``LBFGS_NEED_EVAL`` are left untouched.
    """
    tid = wp.tid()
    start = tid * elems_per_thread
    if start >= n_dofs:
        return
    stop = wp.min(start + elems_per_thread, n_dofs)

    zero = wp.float64(0.0)
    s_cur = batch_idx[start]
    acc_gg = zero
    acc_gd = zero

    for i in range(start, stop):
        s = batch_idx[i]
        if s != s_cur:
            if status[s_cur] == LBFGS_NEED_EVAL:
                wp.atomic_add(gg, s_cur, acc_gg)
                wp.atomic_add(gd, s_cur, acc_gd)
            s_cur = s
            acc_gg = zero
            acc_gd = zero
        fi = forces[i]
        acc_gg += wp.float64(wp.dot(fi, fi))
        acc_gd -= wp.float64(wp.dot(fi, direction[i]))

    if status[s_cur] == LBFGS_NEED_EVAL:
        wp.atomic_add(gg, s_cur, acc_gg)
        wp.atomic_add(gd, s_cur, acc_gd)


@wp.kernel(enable_backward=False)
def _lbfgs_convergence_kernel(
    cart_forces: wp.array(dtype=Any),
    atom_batch_idx: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    fmax: wp.array(dtype=wp.float64),
    frms_sq: wp.array(dtype=wp.float64),
    n_atoms: wp.int32,
    elems_per_thread: wp.int32,
):
    """Reduce the Cartesian force norms that convergence is tested against.

    Kept separate from the packed reduction on purpose. On a variable-cell path
    the packed atomic entries hold ``Phi^T F`` rather than ``F``, and their
    norms drift away from eV/A as the cell deforms, so a tolerance applied to
    them would not mean what it says. These reductions therefore run over the
    caller's original Cartesian forces, indexed by atom.

    Thread launch
    -------------
    One thread per ``elems_per_thread`` consecutive atoms. Requires
    ``atom_batch_idx`` sorted in non-decreasing order.

    Modifies
    --------
    fmax, frms_sq
        OUTPUT. Accumulated atomically; the launcher zeroes them first.
    """
    tid = wp.tid()
    start = tid * elems_per_thread
    if start >= n_atoms:
        return
    stop = wp.min(start + elems_per_thread, n_atoms)

    zero = wp.float64(0.0)
    s_cur = atom_batch_idx[start]
    acc = zero
    loc_max = zero

    for i in range(start, stop):
        s = atom_batch_idx[i]
        if s != s_cur:
            if status[s_cur] == LBFGS_NEED_EVAL:
                wp.atomic_add(frms_sq, s_cur, acc)
                wp.atomic_max(fmax, s_cur, loc_max)
            s_cur = s
            acc = zero
            loc_max = zero
        fi = cart_forces[i]
        ff = wp.float64(wp.dot(fi, fi))
        acc += ff
        loc_max = wp.max(loc_max, wp.sqrt(ff))

    if status[s_cur] == LBFGS_NEED_EVAL:
        wp.atomic_add(frms_sq, s_cur, acc)
        wp.atomic_max(fmax, s_cur, loc_max)


@wp.kernel(enable_backward=False)
def _lbfgs_stress_norm_kernel(
    stress: wp.array(dtype=wp.mat33d),
    smax: wp.array(dtype=wp.float64),
):
    """Spectral norm of each system's Cauchy stress, for the cell criterion.

    Compared against ``stress_tol`` in the units of the supplied stress. The
    packed cell force cannot be used for this: it carries units of energy, not
    stress, so comparing it to a stress tolerance would be dimensionally wrong.

    Thread launch
    -------------
    One thread per system; ``dim = num_systems``.

    Modifies
    --------
    smax
        OUTPUT. Largest singular value of the stress tensor.
    """
    s = wp.tid()
    u = wp.mat33d()
    sv = wp.vec3d()
    v = wp.mat33d()
    wp.svd3(stress[s], u, sv, v)
    smax[s] = wp.max(wp.max(wp.abs(sv[0]), wp.abs(sv[1])), wp.abs(sv[2]))


@wp.kernel(enable_backward=False)
def _lbfgs_energy_sum_kernel(
    per_atom_energy: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    energy_sum: wp.array(dtype=wp.float64),
    n_atoms: wp.int32,
    elems_per_thread: wp.int32,
):
    """Sum per-atom energies into per-system totals, accumulating in float64.

    Passing per-atom energies is the recommended way to supply energy. Summing
    single-precision per-atom values in float64 keeps the total accurate to
    roughly ``sqrt(N) * 1e-7 eV``, whereas summing them in single precision
    loses about ``1e-3 eV`` at ``E ~ -1e4 eV`` -- enough to make the Armijo
    test meaningless near convergence.

    Thread launch
    -------------
    One thread per ``elems_per_thread`` consecutive atoms. Requires
    ``batch_idx`` sorted in non-decreasing order.

    Modifies
    --------
    energy_sum
        OUTPUT. Zeroed by the launcher, then accumulated atomically.
    """
    tid = wp.tid()
    start = tid * elems_per_thread
    if start >= n_atoms:
        return
    stop = wp.min(start + elems_per_thread, n_atoms)

    s_cur = batch_idx[start]
    acc = wp.float64(0.0)
    for i in range(start, stop):
        s = batch_idx[i]
        if s != s_cur:
            wp.atomic_add(energy_sum, s_cur, acc)
            s_cur = s
            acc = wp.float64(0.0)
        acc += wp.float64(per_atom_energy[i])
    wp.atomic_add(energy_sum, s_cur, acc)


# =============================================================================
# Kernel 2: per-system line-search state machine
# =============================================================================


@wp.kernel(enable_backward=False)
def _lbfgs_line_search_kernel(
    energy: wp.array(dtype=wp.float64),
    fmax: wp.array(dtype=wp.float64),
    frms_sq: wp.array(dtype=wp.float64),
    smax: wp.array(dtype=wp.float64),
    n_particles: wp.array(dtype=wp.int32),
    gd: wp.array(dtype=wp.float64),
    d0: wp.array(dtype=wp.float64),
    dmax: wp.array(dtype=wp.float64),
    dquad: wp.array(dtype=wp.float64),
    f_base: wp.array(dtype=wp.float64),
    alpha_step: wp.array(dtype=wp.float64),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    ss: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    iteration: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    ls_trials: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    m: wp.int32,
    force_tol: wp.float64,
    rms_tol: wp.float64,
    stress_tol: wp.float64,
    ftol: wp.float64,
    wolfe: wp.float64,
    step_down: wp.float64,
    step_up: wp.float64,
    min_step: wp.float64,
    max_step: wp.float64,
    max_ls_iter: wp.int32,
    maxstep: wp.float64,
):
    """Advance one system's line search by exactly one evaluation.

    Interprets the energy and forces just supplied and decides what happens
    next: accept the point, shrink or grow the step and retry, declare
    convergence, or give up. This is the only kernel with per-system control
    flow, which is why it runs one thread per system rather than per atom.

    The Wolfe tests use ``d0`` (slope at the base point) and ``dt`` (slope at
    the trial point), both in gradient space and both negative for a valid
    descent direction.

    Thread launch
    -------------
    One thread per system; ``dim = num_systems``.

    Modifies
    --------
    status, iteration, end, n_loop, ls_trials
        Per-system control state.
    f_base, alpha_step
        Base-point energy and the step length for the next trial.
    ys, yy, ss
        The candidate history slot is zeroed here, ready for kernel 3.
    """
    s = wp.tid()

    # Systems that already finished stay frozen: no work is owed downstream.
    if status[s] != LBFGS_NEED_EVAL:
        n_loop[s] = _NLOOP_RETRY
        return

    e_now = energy[s]
    conv = _converged(
        fmax[s],
        frms_sq[s],
        smax[s],
        n_particles[s],
        force_tol,
        rms_tol,
        stress_tol,
    )

    # ---- first evaluation -------------------------------------------------
    if iteration[s] < 0:
        f_base[s] = e_now
        iteration[s] = 0
        if conv:
            # Already relaxed on arrival. Do not move, but the base buffers
            # still have to describe this geometry.
            status[s] = LBFGS_CONVERGED
            n_loop[s] = _NLOOP_SEED
        else:
            n_loop[s] = _NLOOP_RESTART
        return

    # ---- interpret the trial ----------------------------------------------
    d0_s = d0[s]
    dt_s = gd[s]
    alpha = alpha_step[s]
    trials = ls_trials[s] + 1

    armijo_ok = (e_now - f_base[s]) <= alpha * ftol * d0_s
    curvature_ok = dt_s >= wolfe * d0_s
    strong_ok = dt_s <= -wolfe * d0_s

    accept = False
    new_alpha = alpha

    if not armijo_ok:
        new_alpha = alpha * step_down
    elif not curvature_ok:
        # The step is too short. Growing is pointless once the trust region
        # binds, so take the point instead of spinning on the same trial.
        cap = _alpha_cap(dmax[s], dquad[s], maxstep)
        if alpha >= cap:
            accept = True
        else:
            new_alpha = alpha * step_up
    elif not strong_ok:
        new_alpha = alpha * step_down
    else:
        accept = True

    if not accept:
        # Step bounds are tested against the freshly scaled alpha, so a trial
        # that would fall outside the range is never proposed.
        if trials >= max_ls_iter or new_alpha < min_step or new_alpha > max_step:
            if history_count[s] > 0:
                # A stalled line search usually means the curvature model has
                # gone stale, not that there is no descent left. Roll back to
                # the last good point, throw the history away and start again
                # from steepest descent, which cannot inherit a bad model.
                # Only a search that fails with no history left is terminal,
                # so this can happen at most once per failure and cannot loop.
                iteration[s] = -1
                history_count[s] = 0
                ls_trials[s] = 0
                alpha_step[s] = wp.float64(1.0)
                n_loop[s] = _NLOOP_ROLLBACK
            else:
                status[s] = LBFGS_LS_FAILED
                n_loop[s] = _NLOOP_ROLLBACK
        else:
            alpha_step[s] = new_alpha
            ls_trials[s] = trials
            n_loop[s] = _NLOOP_RETRY
        return

    # ---- accepted ---------------------------------------------------------
    # Whether or not this point is converged, it becomes the new base point,
    # so the history kernel must run. The commit kernel decides which.
    f_base[s] = e_now
    iteration[s] = iteration[s] + 1
    n_loop[s] = _NLOOP_PENDING

    slot = end[s]
    ys[slot, s] = wp.float64(0.0)
    yy[slot, s] = wp.float64(0.0)
    ss[s] = wp.float64(0.0)


# =============================================================================
# Kernel 3: history update, and kernel 3b: commit
# =============================================================================


@wp.kernel(enable_backward=False)
def _lbfgs_history_update_kernel(
    positions: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    x_base: wp.array(dtype=Any),
    force_base: wp.array(dtype=Any),
    s_history: wp.array(dtype=Any, ndim=2),
    y_history: wp.array(dtype=Any, ndim=2),
    batch_idx: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    ss: wp.array(dtype=wp.float64),
    n_dofs: wp.int32,
    elems_per_thread: wp.int32,
):
    """Record the candidate ``(s, y)`` pair and move the base point forward.

    Runs for systems that just accepted a step. It writes the candidate pair
    into the slot ``end`` points at and accumulates the three inner products
    the curvature test needs. It also refreshes ``x_base`` and ``force_base``
    unconditionally, because an accepted point becomes the new base point
    whether or not it turns out to be converged.

    The pair is only *provisional* at this stage: whether it is kept is decided
    by :func:`_lbfgs_history_commit_kernel`, which cannot run earlier because
    it needs the inner products this kernel produces.

    Thread launch
    -------------
    One thread per ``elems_per_thread`` consecutive degrees of freedom.

    Modifies
    --------
    s_history, y_history
        The slot at ``end`` is overwritten with the candidate pair.
    x_base, force_base
        Advanced to the accepted point.
    ys, yy, ss
        OUTPUT. Accumulated atomically into the candidate slot.
    """
    tid = wp.tid()
    start = tid * elems_per_thread
    if start >= n_dofs:
        return
    stop = wp.min(start + elems_per_thread, n_dofs)

    zero = wp.float64(0.0)
    s_cur = batch_idx[start]
    slot = end[s_cur]
    active = n_loop[s_cur] == _NLOOP_PENDING
    acc_sy = zero
    acc_ss = zero
    acc_yy = zero

    for i in range(start, stop):
        s = batch_idx[i]
        if s != s_cur:
            if active:
                wp.atomic_add(ys, slot, s_cur, acc_sy)
                wp.atomic_add(ss, s_cur, acc_ss)
                wp.atomic_add(yy, slot, s_cur, acc_yy)
            s_cur = s
            slot = end[s_cur]
            active = n_loop[s_cur] == _NLOOP_PENDING
            acc_sy = zero
            acc_ss = zero
            acc_yy = zero
        if active:
            pi = positions[i]
            fi = forces[i]
            # s = x - x_base;  y = g - g_base = force_base - F  (g = -F)
            svec = pi - x_base[i]
            yvec = force_base[i] - fi
            s_history[slot, i] = svec
            y_history[slot, i] = yvec
            x_base[i] = pi
            force_base[i] = fi
            acc_sy += wp.float64(wp.dot(svec, yvec))
            acc_ss += wp.float64(wp.dot(svec, svec))
            acc_yy += wp.float64(wp.dot(yvec, yvec))

    if active:
        wp.atomic_add(ys, slot, s_cur, acc_sy)
        wp.atomic_add(ss, s_cur, acc_ss)
        wp.atomic_add(yy, slot, s_cur, acc_yy)


@wp.kernel(enable_backward=False)
def _lbfgs_history_commit_kernel(
    fmax: wp.array(dtype=wp.float64),
    frms_sq: wp.array(dtype=wp.float64),
    smax: wp.array(dtype=wp.float64),
    n_particles: wp.array(dtype=wp.int32),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    ss: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    m: wp.int32,
    curvature_eps: wp.float64,
    force_tol: wp.float64,
    rms_tol: wp.float64,
    stress_tol: wp.float64,
):
    """Keep or discard the candidate pair, and finalize the loop bound.

    A pair is kept only if its curvature ``s . y`` is comfortably positive.
    Exact arithmetic guarantees this under the strong Wolfe conditions, but in
    single precision ``y = g - g_prev`` cancels badly near convergence and the
    scaling ``gamma = ys / yy`` can blow up.

    Discarding is not free: kernel 3 has already written the candidate into the
    slot ``end`` points at, and when the ring is full that slot held the oldest
    valid pair. Clamping ``history_count`` to ``m - 1`` removes exactly that
    now-destroyed entry from the range the recursion walks. When the ring is
    not yet full the slot had never been written, so the clamp does nothing.

    Convergence is also settled here rather than in the state machine, so that
    kernel 3 has already refreshed the base buffers by the time a system is
    marked converged.

    Thread launch
    -------------
    One thread per system; ``dim = num_systems``.

    Modifies
    --------
    status, end, n_loop, history_count
        Per-system control state.
    """
    s = wp.tid()
    if n_loop[s] != _NLOOP_PENDING:
        return

    slot = end[s]

    if _converged(
        fmax[s],
        frms_sq[s],
        smax[s],
        n_particles[s],
        force_tol,
        rms_tol,
        stress_tol,
    ):
        status[s] = LBFGS_CONVERGED
        history_count[s] = wp.min(history_count[s], m - 1)
        n_loop[s] = _NLOOP_RETRY
        return

    sy = ys[slot, s]
    threshold = curvature_eps * wp.sqrt(ss[s] * yy[slot, s])
    if sy > threshold:
        history_count[s] = wp.min(history_count[s] + 1, m)
        end[s] = (slot + 1) % m
    else:
        history_count[s] = wp.min(history_count[s], m - 1)

    if history_count[s] > 0:
        n_loop[s] = history_count[s] + 1
    else:
        n_loop[s] = _NLOOP_RESTART


# =============================================================================
# Kernels 4 and 5: the two-loop recursion
# =============================================================================


@wp.kernel(enable_backward=False)
def _lbfgs_loop1_kernel(
    forces: wp.array(dtype=Any),
    s_history: wp.array(dtype=Any, ndim=2),
    y_history: wp.array(dtype=Any, ndim=2),
    direction: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    alpha_hist: wp.array(dtype=wp.float64, ndim=2),
    beta_hist: wp.array(dtype=wp.float64, ndim=2),
    step: wp.int32,
    m: wp.int32,
    n_dofs: wp.int32,
    elems_per_thread: wp.int32,
):
    """First loop of the two-loop recursion, one history vector per launch.

    Walks the history newest-first, subtracting each ``alpha_j * y_j`` from the
    working vector and applying the initial scaling ``gamma = ys / yy`` on the
    final step. Each launch fuses the update for one history vector with the
    dot product for the next, so a whole recursion costs one launch per vector
    rather than two.

    The launch count is fixed at ``m + 1`` regardless of how much history each
    system actually has; systems with less simply return early. That keeps the
    launch sequence independent of device state, which is what allows the whole
    step to be captured in a CUDA graph.

    Thread launch
    -------------
    One thread per ``elems_per_thread`` consecutive degrees of freedom, called
    with ``step = 0 .. m``.

    Modifies
    --------
    direction
        The working vector ``q``.
    alpha_hist, beta_hist
        OUTPUT. One coefficient accumulated per launch.
    """
    tid = wp.tid()
    start = tid * elems_per_thread
    if start >= n_dofs:
        return
    stop = wp.min(start + elems_per_thread, n_dofs)

    zero = wp.float64(0.0)
    s_cur = batch_idx[start]
    acc = zero

    # Per-system constants, refreshed whenever the segment changes.
    nl = n_loop[s_cur]
    bound = nl - 1
    e = end[s_cur]
    active = step < nl and nl > 0
    is_last = step == bound
    coeff = zero
    j_cur = wp.int32(0)
    j_prev = wp.int32(0)
    gamma = wp.float64(1.0)
    if active:
        if step >= 1:
            j_prev = _slot(e, step - 1, m)
            coeff = alpha_hist[j_prev, s_cur] / ys[j_prev, s_cur]
        if is_last:
            j_new = _slot(e, 0, m)
            gamma = ys[j_new, s_cur] / yy[j_new, s_cur]
            j_cur = _slot(e, bound - 1, m)
        else:
            j_cur = _slot(e, step, m)

    for i in range(start, stop):
        s = batch_idx[i]
        if s != s_cur:
            if active:
                if is_last:
                    wp.atomic_add(beta_hist, j_cur, s_cur, acc)
                else:
                    wp.atomic_add(alpha_hist, j_cur, s_cur, acc)
            s_cur = s
            acc = zero
            nl = n_loop[s_cur]
            bound = nl - 1
            e = end[s_cur]
            active = step < nl and nl > 0
            is_last = step == bound
            coeff = zero
            gamma = wp.float64(1.0)
            if active:
                if step >= 1:
                    j_prev = _slot(e, step - 1, m)
                    coeff = alpha_hist[j_prev, s_cur] / ys[j_prev, s_cur]
                if is_last:
                    j_new = _slot(e, 0, m)
                    gamma = ys[j_new, s_cur] / yy[j_new, s_cur]
                    j_cur = _slot(e, bound - 1, m)
                else:
                    j_cur = _slot(e, step, m)
        if active:
            if step == 0:
                # q starts at +F, which is -g.
                qi = forces[i]
            else:
                qi = direction[i] - type(direction[i][0])(coeff) * y_history[j_prev, i]
            if is_last:
                qi = type(qi[0])(gamma) * qi
                acc += wp.float64(wp.dot(y_history[j_cur, i], qi))
            else:
                acc += wp.float64(wp.dot(s_history[j_cur, i], qi))
            direction[i] = qi

    if active:
        if is_last:
            wp.atomic_add(beta_hist, j_cur, s_cur, acc)
        else:
            wp.atomic_add(alpha_hist, j_cur, s_cur, acc)


@wp.kernel(enable_backward=False)
def _lbfgs_loop2_kernel(
    s_history: wp.array(dtype=Any, ndim=2),
    y_history: wp.array(dtype=Any, ndim=2),
    direction: wp.array(dtype=Any),
    force_base: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    ys: wp.array(dtype=wp.float64, ndim=2),
    alpha_hist: wp.array(dtype=wp.float64, ndim=2),
    beta_hist: wp.array(dtype=wp.float64, ndim=2),
    d0: wp.array(dtype=wp.float64),
    step: wp.int32,
    m: wp.int32,
    n_dofs: wp.int32,
    elems_per_thread: wp.int32,
):
    """Second loop of the two-loop recursion, one history vector per launch.

    Walks the history oldest-first, adding ``(alpha_j - beta_j) * s_j``. The
    final launch also accumulates the slope ``d0`` at the base point, so that
    quantity needs no extra pass. The trust-region measure is taken separately,
    because on a variable-cell path it is not a property of the direction
    alone.

    Thread launch
    -------------
    One thread per ``elems_per_thread`` consecutive degrees of freedom, called
    with ``step = 1 .. m``.

    Modifies
    --------
    direction
        The search direction, complete after the final launch.
    beta_hist
        OUTPUT. One coefficient accumulated per launch.
    d0
        OUTPUT. Accumulated on the final launch only.
    """
    tid = wp.tid()
    start = tid * elems_per_thread
    if start >= n_dofs:
        return
    stop = wp.min(start + elems_per_thread, n_dofs)

    zero = wp.float64(0.0)
    s_cur = batch_idx[start]
    acc = zero

    nl = n_loop[s_cur]
    bound = nl - 1
    e = end[s_cur]
    active = step < nl and nl > 0
    is_last = step == bound
    j_apply = wp.int32(0)
    j_next = wp.int32(0)
    coeff = zero
    if active:
        j_apply = _slot(e, bound - step, m)
        coeff = (alpha_hist[j_apply, s_cur] - beta_hist[j_apply, s_cur]) / ys[
            j_apply, s_cur
        ]
        if not is_last:
            j_next = _slot(e, bound - step - 1, m)

    for i in range(start, stop):
        s = batch_idx[i]
        if s != s_cur:
            if active:
                if is_last:
                    wp.atomic_add(d0, s_cur, acc)
                else:
                    wp.atomic_add(beta_hist, j_next, s_cur, acc)
            s_cur = s
            acc = zero
            nl = n_loop[s_cur]
            bound = nl - 1
            e = end[s_cur]
            active = step < nl and nl > 0
            is_last = step == bound
            if active:
                j_apply = _slot(e, bound - step, m)
                coeff = (alpha_hist[j_apply, s_cur] - beta_hist[j_apply, s_cur]) / ys[
                    j_apply, s_cur
                ]
                if not is_last:
                    j_next = _slot(e, bound - step - 1, m)
        if active:
            ri = direction[i] + type(direction[i][0])(coeff) * s_history[j_apply, i]
            direction[i] = ri
            if is_last:
                # d0 = g_base . d = -(force_base . d)
                acc -= wp.float64(wp.dot(force_base[i], ri))
            else:
                acc += wp.float64(wp.dot(y_history[j_next, i], ri))

    if active:
        if is_last:
            wp.atomic_add(d0, s_cur, acc)
        else:
            wp.atomic_add(beta_hist, j_next, s_cur, acc)


@wp.kernel(enable_backward=False)
def _lbfgs_seed_direction_kernel(
    forces: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    x_base: wp.array(dtype=Any),
    force_base: wp.array(dtype=Any),
    direction: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    gg: wp.array(dtype=wp.float64),
):
    """Set a steepest-descent direction and seed the base point.

    Runs for systems that are starting fresh, either on the first evaluation or
    after a restart. Doing this before the trust region is measured means the
    direction is fully determined by the time the step length is chosen, which
    matters on a variable-cell path where the displacement a direction produces
    cannot be inferred from the force norm alone.

    Thread launch
    -------------
    One thread per degree of freedom; ``dim = n_dofs``.

    Modifies
    --------
    direction
        Set to ``F / ||F||`` in the packed space.
    x_base, force_base
        Seeded to the current point.
    """
    i = wp.tid()
    s = batch_idx[i]
    if status[s] != LBFGS_NEED_EVAL or n_loop[s] != _NLOOP_RESTART:
        return
    gn = wp.sqrt(gg[s])
    if gn > wp.float64(0.0):
        scale = type(forces[i][0])(wp.float64(1.0) / gn)
        direction[i] = scale * forces[i]
    else:
        direction[i] = type(forces[i])()
    x_base[i] = positions[i]
    force_base[i] = forces[i]


@wp.kernel(enable_backward=False)
def _lbfgs_trust_region_kernel(
    direction: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    dmax: wp.array(dtype=wp.float64),
    dquad: wp.array(dtype=wp.float64),
    n_dofs: wp.int32,
    elems_per_thread: wp.int32,
):
    """Measure how far the largest atom moves per unit step length.

    On a coordinate-only path the displacement is simply ``alpha * d``, so the
    measure is the largest direction magnitude and the quadratic term is zero.
    The variable-cell path has its own kernel, because there both the cell and
    the positions move and the displacement gains a term in ``alpha**2``.

    Thread launch
    -------------
    One thread per ``elems_per_thread`` consecutive degrees of freedom.

    Modifies
    --------
    dmax, dquad
        OUTPUT. Zeroed by the launcher, then accumulated with atomic maxima.
    """
    tid = wp.tid()
    start = tid * elems_per_thread
    if start >= n_dofs:
        return
    stop = wp.min(start + elems_per_thread, n_dofs)

    zero = wp.float64(0.0)
    s_cur = batch_idx[start]
    active = status[s_cur] == LBFGS_NEED_EVAL and n_loop[s_cur] != _NLOOP_RETRY
    loc_max = zero

    for i in range(start, stop):
        s = batch_idx[i]
        if s != s_cur:
            if active:
                wp.atomic_max(dmax, s_cur, loc_max)
            s_cur = s
            active = status[s_cur] == LBFGS_NEED_EVAL and n_loop[s_cur] != _NLOOP_RETRY
            loc_max = zero
        if active:
            loc_max = wp.max(loc_max, wp.float64(wp.length(direction[i])))

    if active:
        wp.atomic_max(dmax, s_cur, loc_max)


# =============================================================================
# Kernels 6 and 7: prepare and apply
# =============================================================================


@wp.kernel(enable_backward=False)
def _lbfgs_prepare_step_kernel(
    gg: wp.array(dtype=wp.float64),
    d0: wp.array(dtype=wp.float64),
    dmax: wp.array(dtype=wp.float64),
    dquad: wp.array(dtype=wp.float64),
    alpha_step: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    ls_trials: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    maxstep: wp.float64,
):
    """Reset the line search, repair a bad direction, and apply the trust region.

    Three things happen here, in order.

    First, any system starting a *new* direction restarts its line search at
    ``alpha = 1`` with a zero trial count. This matters more than it looks: the
    two-loop already scales the direction, so a unit step is the Newton step
    and should usually be accepted outright. Inheriting a shrunken step from
    the previous line search would quietly degrade the method to a scaled
    gradient descent. Resetting the trial count also stops it accumulating
    across iterations and tripping ``max_ls_iter`` on a healthy run.

    Second, a direction that fails the descent test is replaced by steepest
    descent rather than reported as a failure. Catching it here, where the
    direction was produced, guarantees the state machine never sees a
    non-descent slope.

    Third, the step length is capped so the resulting Cartesian displacement
    stays within ``maxstep``. The cap is applied on *every* call, not only when
    a direction is new, because a line search that keeps growing the step would
    otherwise walk straight past the limit.

    Thread launch
    -------------
    One thread per system; ``dim = num_systems``.

    Modifies
    --------
    alpha_step, ls_trials, d0, dmax, dquad, end, history_count, n_loop
        Per-system control state.
    """
    s = wp.tid()
    if status[s] != LBFGS_NEED_EVAL:
        return

    # A new direction always starts a fresh line search.
    if n_loop[s] != _NLOOP_RETRY:
        alpha_step[s] = wp.float64(1.0)
        ls_trials[s] = 0

    # An ascent direction means the history has gone bad; drop it and restart.
    if n_loop[s] > 0 and d0[s] >= wp.float64(0.0):
        n_loop[s] = _NLOOP_RESTART

    if n_loop[s] == _NLOOP_RESTART:
        # For a normalized steepest-descent direction the slope is exact.
        d0[s] = -wp.sqrt(gg[s])
        history_count[s] = 0
        end[s] = 0

    cap = _alpha_cap(dmax[s], dquad[s], maxstep)
    alpha_step[s] = wp.min(alpha_step[s], cap)


@wp.kernel(enable_backward=False)
def _lbfgs_apply_step_kernel(
    positions: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    x_base: wp.array(dtype=Any),
    force_base: wp.array(dtype=Any),
    direction: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    gg: wp.array(dtype=wp.float64),
    alpha_step: wp.array(dtype=wp.float64),
):
    """Move the positions to the next trial point, or finish the system off.

    Handles the two terminal cases as well as the ordinary step, so that
    whatever the caller reads back is always a geometry the optimizer stands
    behind, paired with matching forces in ``force_base``:

    - a line search that ran out of budget has its positions restored to the
      last accepted point;
    - a geometry that arrived already converged is left where it is, but its
      base buffers are seeded so they describe it.

    Thread launch
    -------------
    One thread per degree of freedom; ``dim = n_dofs``.

    Modifies
    --------
    positions
        Advanced to ``x_base + alpha * d``, or restored on failure.
    direction, x_base, force_base
        Seeded when a steepest-descent direction is taken.
    """
    i = wp.tid()
    s = batch_idx[i]
    nl = n_loop[s]

    if nl == _NLOOP_ROLLBACK:
        positions[i] = x_base[i]
        return

    if nl == _NLOOP_SEED:
        x_base[i] = positions[i]
        force_base[i] = forces[i]
        return

    if status[s] != LBFGS_NEED_EVAL:
        return

    a = type(direction[i][0])(alpha_step[s])
    positions[i] = x_base[i] + a * direction[i]


# =============================================================================
# Kernel overloads
#
# Coordinates may be single or double precision; every per-system scalar is
# float64 either way, so the overload key is just the vector dtype.
# =============================================================================

_VEC_TYPES = [wp.vec3f, wp.vec3d]

_reduce_overloads = {}
_convergence_overloads = {}
_seed_direction_overloads = {}
_trust_region_overloads = {}
_energy_sum_overloads = {}
_history_update_overloads = {}
_loop1_overloads = {}
_loop2_overloads = {}
_apply_step_overloads = {}

_F64 = wp.float64
_I32 = wp.int32

for _v in _VEC_TYPES:
    _reduce_overloads[_v] = wp.overload(
        _lbfgs_reduce_kernel,
        [
            wp.array(dtype=_v),  # forces
            wp.array(dtype=_v),  # direction
            wp.array(dtype=_I32),  # batch_idx
            wp.array(dtype=_I32),  # status
            wp.array(dtype=_F64),  # gg
            wp.array(dtype=_F64),  # gd
            _I32,  # n_dofs
            _I32,  # elems_per_thread
        ],
    )

    _convergence_overloads[_v] = wp.overload(
        _lbfgs_convergence_kernel,
        [
            wp.array(dtype=_v),  # cart_forces
            wp.array(dtype=_I32),  # atom_batch_idx
            wp.array(dtype=_I32),  # status
            wp.array(dtype=_F64),  # fmax
            wp.array(dtype=_F64),  # frms_sq
            _I32,  # n_atoms
            _I32,  # elems_per_thread
        ],
    )

    _seed_direction_overloads[_v] = wp.overload(
        _lbfgs_seed_direction_kernel,
        [
            wp.array(dtype=_v),  # forces
            wp.array(dtype=_v),  # positions
            wp.array(dtype=_v),  # x_base
            wp.array(dtype=_v),  # force_base
            wp.array(dtype=_v),  # direction
            wp.array(dtype=_I32),  # batch_idx
            wp.array(dtype=_I32),  # status
            wp.array(dtype=_I32),  # n_loop
            wp.array(dtype=_F64),  # gg
        ],
    )

    _trust_region_overloads[_v] = wp.overload(
        _lbfgs_trust_region_kernel,
        [
            wp.array(dtype=_v),  # direction
            wp.array(dtype=_I32),  # batch_idx
            wp.array(dtype=_I32),  # status
            wp.array(dtype=_I32),  # n_loop
            wp.array(dtype=_F64),  # dmax
            wp.array(dtype=_F64),  # dquad
            _I32,  # n_dofs
            _I32,  # elems_per_thread
        ],
    )

    _history_update_overloads[_v] = wp.overload(
        _lbfgs_history_update_kernel,
        [
            wp.array(dtype=_v),  # positions
            wp.array(dtype=_v),  # forces
            wp.array(dtype=_v),  # x_base
            wp.array(dtype=_v),  # force_base
            wp.array(dtype=_v, ndim=2),  # s_history
            wp.array(dtype=_v, ndim=2),  # y_history
            wp.array(dtype=_I32),  # batch_idx
            wp.array(dtype=_I32),  # end
            wp.array(dtype=_I32),  # n_loop
            wp.array(dtype=_F64, ndim=2),  # ys
            wp.array(dtype=_F64, ndim=2),  # yy
            wp.array(dtype=_F64),  # ss
            _I32,  # n_dofs
            _I32,  # elems_per_thread
        ],
    )

    _loop1_overloads[_v] = wp.overload(
        _lbfgs_loop1_kernel,
        [
            wp.array(dtype=_v),  # forces
            wp.array(dtype=_v, ndim=2),  # s_history
            wp.array(dtype=_v, ndim=2),  # y_history
            wp.array(dtype=_v),  # direction
            wp.array(dtype=_I32),  # batch_idx
            wp.array(dtype=_I32),  # end
            wp.array(dtype=_I32),  # n_loop
            wp.array(dtype=_F64, ndim=2),  # ys
            wp.array(dtype=_F64, ndim=2),  # yy
            wp.array(dtype=_F64, ndim=2),  # alpha_hist
            wp.array(dtype=_F64, ndim=2),  # beta_hist
            _I32,  # step
            _I32,  # m
            _I32,  # n_dofs
            _I32,  # elems_per_thread
        ],
    )

    _loop2_overloads[_v] = wp.overload(
        _lbfgs_loop2_kernel,
        [
            wp.array(dtype=_v, ndim=2),  # s_history
            wp.array(dtype=_v, ndim=2),  # y_history
            wp.array(dtype=_v),  # direction
            wp.array(dtype=_v),  # force_base
            wp.array(dtype=_I32),  # batch_idx
            wp.array(dtype=_I32),  # end
            wp.array(dtype=_I32),  # n_loop
            wp.array(dtype=_F64, ndim=2),  # ys
            wp.array(dtype=_F64, ndim=2),  # alpha_hist
            wp.array(dtype=_F64, ndim=2),  # beta_hist
            wp.array(dtype=_F64),  # d0
            _I32,  # step
            _I32,  # m
            _I32,  # n_dofs
            _I32,  # elems_per_thread
        ],
    )

    _apply_step_overloads[_v] = wp.overload(
        _lbfgs_apply_step_kernel,
        [
            wp.array(dtype=_v),  # positions
            wp.array(dtype=_v),  # forces
            wp.array(dtype=_v),  # x_base
            wp.array(dtype=_v),  # force_base
            wp.array(dtype=_v),  # direction
            wp.array(dtype=_I32),  # batch_idx
            wp.array(dtype=_I32),  # status
            wp.array(dtype=_I32),  # n_loop
            wp.array(dtype=_F64),  # gg
            wp.array(dtype=_F64),  # alpha_step
        ],
    )

for _t in (wp.float32, wp.float64):
    _energy_sum_overloads[_t] = wp.overload(
        _lbfgs_energy_sum_kernel,
        [
            wp.array(dtype=_t),  # per_atom_energy
            wp.array(dtype=_I32),  # batch_idx
            wp.array(dtype=_F64),  # energy_sum
            _I32,  # n_atoms
            _I32,  # elems_per_thread
        ],
    )


# =============================================================================
# Public API
# =============================================================================


def lbfgs_reduce_energy(
    per_atom_energy: wp.array,
    batch_idx: wp.array,
    energy_sum: wp.array,
) -> None:
    """Sum per-atom energies into per-system totals, accumulating in float64.

    Use this when your model returns per-atom energies, which is the common
    case and the recommended one: the float64 accumulation keeps the Armijo
    test meaningful even when the per-atom values are single precision.

    Parameters
    ----------
    per_atom_energy : wp.array, shape (num_atoms,)
        Per-atom energies, float32 or float64.
    batch_idx : wp.array(dtype=int32), shape (num_atoms,)
        Sorted system index for each atom.
    energy_sum : wp.array(dtype=float64), shape (num_systems,)
        OUTPUT. Per-system totals. Zeroed internally.

    See Also
    --------
    lbfgs_step : consumes the resulting per-system energies.
    """
    n_atoms = per_atom_energy.shape[0]
    energy_sum.zero_()
    if n_atoms == 0:
        return
    if batch_idx.shape[0] != n_atoms:
        raise ValueError(
            f"batch_idx length {batch_idx.shape[0]} != per_atom_energy length {n_atoms}"
        )
    device = per_atom_energy.device
    ept = compute_ept(n_atoms, max(device.sm_count, 1), False)
    wp.launch(
        _energy_sum_overloads[per_atom_energy.dtype],
        dim=(n_atoms + ept - 1) // ept,
        inputs=[per_atom_energy, batch_idx, energy_sum, n_atoms, ept],
        device=device,
    )


def lbfgs_reduce(
    forces: wp.array,
    direction: wp.array,
    batch_idx: wp.array,
    status: wp.array,
    gg: wp.array,
    gd: wp.array,
    fmax: wp.array,
    frms_sq: wp.array,
    *,
    cart_forces: wp.array | None = None,
    atom_batch_idx: wp.array | None = None,
    stress: wp.array | None = None,
    smax: wp.array | None = None,
) -> None:
    """Compute the per-system reductions for one L-BFGS call.

    Two disjoint sets of quantities, deliberately not merged:

    - ``gg`` and ``gd`` come from the **packed** arrays and drive the
      algorithm, because that is the space the search direction lives in;
    - ``fmax``, ``frms_sq`` and ``smax`` come from **Cartesian** forces and
      stress and drive convergence, so that the tolerances keep their physical
      meaning as the cell deforms.

    On a coordinate-only path the two coincide, and ``cart_forces`` /
    ``atom_batch_idx`` may be omitted.

    Normally called for you by :func:`lbfgs_update`. Call it directly only if
    you want to supply the reductions yourself, via
    ``compute_reductions=False``.

    Parameters
    ----------
    forces : wp.array, shape (num_dofs,)
        Forces on the packed degrees of freedom.
    direction : wp.array, shape (num_dofs,)
        Current search direction.
    batch_idx : wp.array(dtype=int32), shape (num_dofs,)
        Sorted system index for each degree of freedom.
    status : wp.array(dtype=int32), shape (num_systems,)
        Per-system status; finished systems are skipped.
    gg, gd, fmax, frms_sq : wp.array(dtype=float64), shape (num_systems,)
        OUTPUT. Zeroed internally before accumulation.
    cart_forces : wp.array, shape (num_atoms,), optional
        Cartesian forces for the convergence reductions. Defaults to
        ``forces``, which is correct only when the packed degrees of freedom
        are Cartesian atom positions.
    atom_batch_idx : wp.array(dtype=int32), shape (num_atoms,), optional
        Sorted system index per atom. Defaults to ``batch_idx``.
    stress : wp.array(dtype=mat33d), shape (num_systems,), optional
        Cauchy stress per system. Required for the stress criterion.
    smax : wp.array(dtype=float64), shape (num_systems,), optional
        OUTPUT. Spectral norm of ``stress``. Left untouched if ``stress`` is
        not supplied.
    """
    n_dofs = forces.shape[0]
    gg.zero_()
    gd.zero_()
    fmax.zero_()
    frms_sq.zero_()
    if n_dofs == 0:
        return

    device = forces.device
    ept = compute_ept(n_dofs, max(device.sm_count, 1), True)
    wp.launch(
        _reduce_overloads[forces.dtype],
        dim=(n_dofs + ept - 1) // ept,
        inputs=[forces, direction, batch_idx, status, gg, gd, n_dofs, ept],
        device=device,
    )

    cart = forces if cart_forces is None else cart_forces
    cart_idx = batch_idx if atom_batch_idx is None else atom_batch_idx
    n_atoms = cart.shape[0]
    if cart_idx.shape[0] != n_atoms:
        raise ValueError(
            f"atom_batch_idx length {cart_idx.shape[0]} != cart_forces length {n_atoms}"
        )
    if n_atoms:
        ept_atoms = compute_ept(n_atoms, max(device.sm_count, 1), True)
        wp.launch(
            _convergence_overloads[cart.dtype],
            dim=(n_atoms + ept_atoms - 1) // ept_atoms,
            inputs=[cart, cart_idx, status, fmax, frms_sq, n_atoms, ept_atoms],
            device=device,
        )

    if stress is not None:
        if smax is None:
            raise ValueError("smax must be provided when stress is given")
        wp.launch(
            _lbfgs_stress_norm_kernel,
            dim=smax.shape[0],
            inputs=[stress, smax],
            device=device,
        )


def lbfgs_update(
    positions: wp.array,
    forces: wp.array,
    x_base: wp.array,
    force_base: wp.array,
    direction: wp.array,
    batch_idx: wp.array,
    s_history: wp.array,
    y_history: wp.array,
    ys: wp.array,
    yy: wp.array,
    alpha_hist: wp.array,
    beta_hist: wp.array,
    ss: wp.array,
    energy: wp.array,
    f_base: wp.array,
    gg: wp.array,
    gd: wp.array,
    fmax: wp.array,
    frms_sq: wp.array,
    smax: wp.array,
    d0: wp.array,
    dmax: wp.array,
    dquad: wp.array,
    alpha_step: wp.array,
    status: wp.array,
    iteration: wp.array,
    end: wp.array,
    n_loop: wp.array,
    ls_trials: wp.array,
    history_count: wp.array,
    n_particles: wp.array,
    *,
    cart_forces: wp.array | None = None,
    atom_batch_idx: wp.array | None = None,
    stress: wp.array | None = None,
    force_tol: float = 0.05,
    rms_tol: float = 0.0,
    stress_tol: float = 0.0,
    ftol: float = 1e-4,
    wolfe: float = 0.9,
    step_scale_down: float = 0.5,
    step_scale_up: float = 2.1,
    min_step: float = 1e-20,
    max_step: float = 1e20,
    max_ls_iter: int = 40,
    maxstep: float = 0.2,
    curvature_eps: float = 1e-10,
    compute_reductions: bool = True,
    measure_trust_region: bool = True,
) -> None:
    """Advance the state machine and produce a search direction.

    Runs everything except the position update: the reductions, the
    line-search decision, the history update, and the two-loop recursion. Use
    it together with :func:`lbfgs_prepare_step` and :func:`lbfgs_apply_step`
    when you need to interpose your own logic before the atoms move;
    :func:`lbfgs_step` is the convenience wrapper that chains all three.

    Parameters
    ----------
    positions, forces : wp.array, shape (num_dofs,)
        Current geometry and the forces there. ``positions`` are read, not
        moved; ``lbfgs_apply_step`` does the moving.
    x_base, force_base, direction : wp.array, shape (num_dofs,)
        Last accepted point, its forces, and the current search direction.
    batch_idx : wp.array(dtype=int32), shape (num_dofs,)
        Sorted system index for each degree of freedom. Required.
    s_history, y_history : wp.array, shape (history_size, num_dofs,)
        Ring buffers of position and gradient differences.
    energy : wp.array(dtype=float64), shape (num_systems,)
        Per-system total energy at ``positions``. Use
        :func:`lbfgs_reduce_energy` if your model returns per-atom energies.
    n_particles : wp.array(dtype=int32), shape (num_systems,)
        Atom count per system, for the RMS convergence test. This is the atom
        count, which on a variable-cell path is not the same as the
        degree-of-freedom count.
    force_tol : float, optional
        Convergence threshold on the largest per-atom force magnitude, in the
        force units you supplied. Set to zero to disable.
    rms_tol, stress_tol : float, optional
        Additional convergence thresholds, disabled by default. All enabled
        criteria must hold.
    ftol, wolfe : float, optional
        Armijo and curvature parameters of the strong Wolfe conditions.
    step_scale_down, step_scale_up : float, optional
        Multipliers applied when the line search shrinks or grows the step.
    min_step, max_step : float, optional
        Step length bounds; falling outside them ends the line search.
    max_ls_iter : int, optional
        Trials allowed per line search before giving up.
    maxstep : float, optional
        Largest Cartesian displacement any atom may take in one step. Set to
        zero to disable the trust region.
    curvature_eps : float, optional
        Relative threshold below which a history pair is judged to carry no
        usable curvature and is discarded.
    compute_reductions : bool, optional
        When ``False``, ``gg``/``gd``/``fmax``/``frms_sq`` are taken as given
        rather than recomputed.
    measure_trust_region : bool, optional
        When ``False``, ``dmax`` and ``dquad`` are taken as given. Set this if
        the displacement a direction produces is not simply its magnitude, as
        on a variable-cell path, and supply your own measure before calling
        :func:`lbfgs_prepare_step`.

    Raises
    ------
    ValueError
        If array lengths disagree or ``batch_idx`` is missing.

    See Also
    --------
    lbfgs_prepare_step : runs next.
    lbfgs_step : chains all three phases.
    """
    n_dofs = positions.shape[0]
    if forces.shape[0] != n_dofs:
        raise ValueError(
            f"forces length {forces.shape[0]} != positions length {n_dofs}"
        )
    if x_base.shape[0] != n_dofs:
        raise ValueError(
            f"x_base length {x_base.shape[0]} != positions length {n_dofs}"
        )
    if force_base.shape[0] != n_dofs:
        raise ValueError(
            f"force_base length {force_base.shape[0]} != positions length {n_dofs}"
        )
    if direction.shape[0] != n_dofs:
        raise ValueError(
            f"direction length {direction.shape[0]} != positions length {n_dofs}"
        )
    if batch_idx is None:
        raise ValueError("batch_idx is required for lbfgs_update")
    if batch_idx.shape[0] != n_dofs:
        raise ValueError(
            f"batch_idx length {batch_idx.shape[0]} != positions length {n_dofs}"
        )
    if s_history.shape[1] != n_dofs or y_history.shape[1] != n_dofs:
        raise ValueError(
            f"history buffers must have shape (m, {n_dofs}); got "
            f"{tuple(s_history.shape)} and {tuple(y_history.shape)}"
        )
    if s_history.shape[0] != y_history.shape[0]:
        raise ValueError("s_history and y_history must share a history size")

    if n_dofs == 0:
        gg.zero_()
        gd.zero_()
        fmax.zero_()
        frms_sq.zero_()
        return

    m = s_history.shape[0]
    if m < 1:
        raise ValueError(f"history size must be >= 1; got {m}")

    vec_dtype = positions.dtype
    device = positions.device
    num_systems = status.shape[0]
    ept = compute_ept(n_dofs, max(device.sm_count, 1), True)
    grid = (n_dofs + ept - 1) // ept

    if compute_reductions:
        lbfgs_reduce(
            forces,
            direction,
            batch_idx,
            status,
            gg,
            gd,
            fmax,
            frms_sq,
            cart_forces=cart_forces,
            atom_batch_idx=atom_batch_idx,
            stress=stress,
            smax=smax if stress is not None else None,
        )

    wp.launch(
        _lbfgs_line_search_kernel,
        dim=num_systems,
        inputs=[
            energy,
            fmax,
            frms_sq,
            smax,
            n_particles,
            gd,
            d0,
            dmax,
            dquad,
            f_base,
            alpha_step,
            ys,
            yy,
            ss,
            status,
            iteration,
            end,
            n_loop,
            ls_trials,
            history_count,
            m,
            float(force_tol),
            float(rms_tol),
            float(stress_tol),
            float(ftol),
            float(wolfe),
            float(step_scale_down),
            float(step_scale_up),
            float(min_step),
            float(max_step),
            int(max_ls_iter),
            float(maxstep),
        ],
        device=device,
    )

    wp.launch(
        _history_update_overloads[vec_dtype],
        dim=grid,
        inputs=[
            positions,
            forces,
            x_base,
            force_base,
            s_history,
            y_history,
            batch_idx,
            end,
            n_loop,
            ys,
            yy,
            ss,
            n_dofs,
            ept,
        ],
        device=device,
    )

    wp.launch(
        _lbfgs_history_commit_kernel,
        dim=num_systems,
        inputs=[
            fmax,
            frms_sq,
            smax,
            n_particles,
            ys,
            yy,
            ss,
            status,
            end,
            n_loop,
            history_count,
            m,
            float(curvature_eps),
            float(force_tol),
            float(rms_tol),
            float(stress_tol),
        ],
        device=device,
    )

    # A restarting system needs its direction before the trust region can be
    # measured, so seed it here rather than in the apply kernel.
    wp.launch(
        _seed_direction_overloads[vec_dtype],
        dim=n_dofs,
        inputs=[
            forces,
            positions,
            x_base,
            force_base,
            direction,
            batch_idx,
            status,
            n_loop,
            gg,
        ],
        device=device,
    )

    # The two-loop coefficients are accumulated, so they start from zero.
    alpha_hist.zero_()
    beta_hist.zero_()
    d0_pending = d0

    # Fixed launch counts keep the sequence independent of device state, which
    # is what lets the whole step be captured in a CUDA graph. Systems with
    # less history than `m` return immediately.
    for step in range(m + 1):
        wp.launch(
            _loop1_overloads[vec_dtype],
            dim=grid,
            inputs=[
                forces,
                s_history,
                y_history,
                direction,
                batch_idx,
                end,
                n_loop,
                ys,
                yy,
                alpha_hist,
                beta_hist,
                step,
                m,
                n_dofs,
                ept,
            ],
            device=device,
        )

    _zero_pending_d0(d0_pending, n_loop, num_systems, device)

    for step in range(1, m + 1):
        wp.launch(
            _loop2_overloads[vec_dtype],
            dim=grid,
            inputs=[
                s_history,
                y_history,
                direction,
                force_base,
                batch_idx,
                end,
                n_loop,
                ys,
                alpha_hist,
                beta_hist,
                d0,
                step,
                m,
                n_dofs,
                ept,
            ],
            device=device,
        )

    if measure_trust_region:
        dmax.zero_()
        dquad.zero_()
        wp.launch(
            _trust_region_overloads[vec_dtype],
            dim=grid,
            inputs=[direction, batch_idx, status, n_loop, dmax, dquad, n_dofs, ept],
            device=device,
        )


@wp.kernel(enable_backward=False)
def _lbfgs_zero_d0_kernel(
    d0: wp.array(dtype=wp.float64),
    n_loop: wp.array(dtype=wp.int32),
):
    """Clear the accumulated slope for systems about to run the second loop.

    Systems that are only retrying a step keep the slope from the direction
    they are still searching along.

    Thread launch
    -------------
    One thread per system.

    Modifies
    --------
    d0
        Zeroed for systems with a freshly built direction.
    """
    s = wp.tid()
    if n_loop[s] > 0:
        d0[s] = wp.float64(0.0)


def _zero_pending_d0(d0, n_loop, num_systems, device) -> None:
    """Zero ``d0`` only where the second loop is about to accumulate into it."""
    wp.launch(
        _lbfgs_zero_d0_kernel,
        dim=num_systems,
        inputs=[d0, n_loop],
        device=device,
    )


def lbfgs_prepare_step(
    gg: wp.array,
    d0: wp.array,
    dmax: wp.array,
    dquad: wp.array,
    alpha_step: wp.array,
    status: wp.array,
    end: wp.array,
    n_loop: wp.array,
    ls_trials: wp.array,
    history_count: wp.array,
    *,
    maxstep: float = 0.2,
) -> None:
    """Finalize the step length before the atoms move.

    Restarts the line search for any system that has a new direction, replaces
    a non-descent direction with steepest descent, and caps the step so no atom
    moves further than ``maxstep``.

    Parameters
    ----------
    maxstep : float, optional
        Largest Cartesian displacement allowed in one step. Zero disables the
        trust region.

    See Also
    --------
    lbfgs_update : runs before this.
    lbfgs_apply_step : runs after this.
    """
    wp.launch(
        _lbfgs_prepare_step_kernel,
        dim=status.shape[0],
        inputs=[
            gg,
            d0,
            dmax,
            dquad,
            alpha_step,
            status,
            end,
            n_loop,
            ls_trials,
            history_count,
            float(maxstep),
        ],
        device=gg.device,
    )


def lbfgs_apply_step(
    positions: wp.array,
    forces: wp.array,
    x_base: wp.array,
    force_base: wp.array,
    direction: wp.array,
    batch_idx: wp.array,
    status: wp.array,
    n_loop: wp.array,
    gg: wp.array,
    alpha_step: wp.array,
) -> None:
    """Move the positions to the next trial point.

    Also handles the two terminal cases, so that after any call the positions
    and ``force_base`` describe the same geometry: a failed line search is
    rolled back to the last accepted point, and a geometry that arrived already
    converged is left alone with its base buffers seeded.

    See Also
    --------
    lbfgs_prepare_step : runs before this.
    """
    n_dofs = positions.shape[0]
    if n_dofs == 0:
        return
    wp.launch(
        _apply_step_overloads[positions.dtype],
        dim=n_dofs,
        inputs=[
            positions,
            forces,
            x_base,
            force_base,
            direction,
            batch_idx,
            status,
            n_loop,
            gg,
            alpha_step,
        ],
        device=positions.device,
    )


def lbfgs_step(
    positions: wp.array,
    forces: wp.array,
    x_base: wp.array,
    force_base: wp.array,
    direction: wp.array,
    batch_idx: wp.array,
    s_history: wp.array,
    y_history: wp.array,
    ys: wp.array,
    yy: wp.array,
    alpha_hist: wp.array,
    beta_hist: wp.array,
    ss: wp.array,
    energy: wp.array,
    f_base: wp.array,
    gg: wp.array,
    gd: wp.array,
    fmax: wp.array,
    frms_sq: wp.array,
    smax: wp.array,
    d0: wp.array,
    dmax: wp.array,
    dquad: wp.array,
    alpha_step: wp.array,
    status: wp.array,
    iteration: wp.array,
    end: wp.array,
    n_loop: wp.array,
    ls_trials: wp.array,
    history_count: wp.array,
    n_particles: wp.array,
    *,
    cart_forces: wp.array | None = None,
    atom_batch_idx: wp.array | None = None,
    stress: wp.array | None = None,
    force_tol: float = 0.05,
    rms_tol: float = 0.0,
    stress_tol: float = 0.0,
    ftol: float = 1e-4,
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
    """Consume one energy/force evaluation and produce the next trial geometry.

    This is the entry point for the common case. Call it once per model
    evaluation; when ``status`` no longer contains ``LBFGS_NEED_EVAL`` for a
    system, that system is finished and its ``positions`` hold the answer.

    Equivalent to :func:`lbfgs_update`, :func:`lbfgs_prepare_step` and
    :func:`lbfgs_apply_step` in sequence.

    Parameters
    ----------
    See :func:`lbfgs_update`; the arguments are identical.

    Examples
    --------
    >>> while True:
    ...     energy_per_atom, forces = model(positions)
    ...     lbfgs_reduce_energy(energy_per_atom, batch_idx, energy)
    ...     lbfgs_step(positions, forces, ..., batch_idx=batch_idx)
    ...     if not (status.numpy() == LBFGS_NEED_EVAL).any():
    ...         break

    See Also
    --------
    lbfgs_reduce_energy : build the per-system energies this consumes.
    """
    lbfgs_update(
        positions=positions,
        forces=forces,
        x_base=x_base,
        force_base=force_base,
        direction=direction,
        batch_idx=batch_idx,
        s_history=s_history,
        y_history=y_history,
        ys=ys,
        yy=yy,
        alpha_hist=alpha_hist,
        beta_hist=beta_hist,
        ss=ss,
        energy=energy,
        f_base=f_base,
        gg=gg,
        gd=gd,
        fmax=fmax,
        frms_sq=frms_sq,
        smax=smax,
        d0=d0,
        dmax=dmax,
        dquad=dquad,
        alpha_step=alpha_step,
        status=status,
        iteration=iteration,
        end=end,
        n_loop=n_loop,
        ls_trials=ls_trials,
        history_count=history_count,
        n_particles=n_particles,
        cart_forces=cart_forces,
        atom_batch_idx=atom_batch_idx,
        stress=stress,
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
    if positions.shape[0] == 0:
        return
    lbfgs_prepare_step(
        gg=gg,
        d0=d0,
        dmax=dmax,
        dquad=dquad,
        alpha_step=alpha_step,
        status=status,
        end=end,
        n_loop=n_loop,
        ls_trials=ls_trials,
        history_count=history_count,
        maxstep=maxstep,
    )
    lbfgs_apply_step(
        positions=positions,
        forces=forces,
        x_base=x_base,
        force_base=force_base,
        direction=direction,
        batch_idx=batch_idx,
        status=status,
        n_loop=n_loop,
        gg=gg,
        alpha_step=alpha_step,
    )


# =============================================================================
# Variable-cell support
#
# Relaxing the cell alongside the coordinates needs an explicit choice of
# generalized coordinates, because the obvious one is wrong. The stress-derived
# cell force is the gradient with respect to an affine deformation in which the
# atoms ride along with the cell. Concatenating raw Cartesian positions with raw
# cell rows gives a coordinate the atoms do *not* follow, so the stored (s, y)
# pairs would pair a displacement in one space with a gradient in another and
# the quasi-Newton model would be built from mismatched quantities.
#
# The chart used here is the one ASE's UnitCellFilter uses. With a reference
# cell H0 captured once at the start, and lattice vectors held as columns so
# that r = H s:
#
#     Phi = H H0^-1          deformation gradient, the identity at the start
#     u   = Phi^-1 r         atom coordinates, in the reference frame
#     c   = kappa * Phi      cell coordinates, six lower-triangular components
#
# with conjugate forces obtained by the chain rule:
#
#     f_u = Phi^T F                    since r = Phi u
#     f_c = -(V sigma) Phi^-T / kappa  since H = Phi H0
#
# Scaling the cell coordinate by kappa and dividing its force by the same
# factor is what keeps g . dx independent of the chart, which is what makes the
# packed pairs genuine secant pairs. The two-loop recursion itself needs no
# changes: it simply runs on the packed array.
# =============================================================================


@wp.kernel(enable_backward=False)
def _lbfgs_cell_kappa_kernel(
    n_atoms_per_system: wp.array(dtype=wp.int32),
    cell_force_scale: wp.float64,
    kappa: wp.array(dtype=Any),
):
    """Precompute the per-system cell coordinate scaling.

    ``kappa`` is stored at the coordinate precision so the packing kernels can
    scale matrices with it directly.

    Thread launch
    -------------
    One thread per system.

    Modifies
    --------
    kappa
        OUTPUT. ``cell_force_scale * num_atoms``.
    """
    s = wp.tid()
    slot = kappa[s]
    kappa[s] = type(slot)(cell_force_scale) * type(slot)(n_atoms_per_system[s])


@wp.kernel(enable_backward=False)
def _lbfgs_cell_chart_kernel(
    cell: wp.array(dtype=Any),
    ref_cell_inv: wp.array(dtype=Any),
    stress: wp.array(dtype=Any),
    kappa: wp.array(dtype=Any),
    phi: wp.array(dtype=Any),
    phi_inv: wp.array(dtype=Any),
    cell_dof_a: wp.array(dtype=Any),
    cell_dof_b: wp.array(dtype=Any),
    cell_force_a: wp.array(dtype=Any),
    cell_force_b: wp.array(dtype=Any),
    have_stress: wp.bool,
):
    """Build the deformation gradient and the packed cell degrees of freedom.

    Thread launch
    -------------
    One thread per system; ``dim = num_systems``.

    Modifies
    --------
    phi, phi_inv
        The deformation gradient ``Phi = H H0^-1`` and its inverse, reused by
        the packing, unpacking and trust-region kernels.
    cell_dof_a, cell_dof_b, cell_force_a, cell_force_b
        The six cell coordinates and their six conjugate force components,
        split into two three-vectors each to match the packed layout.
    """
    s = wp.tid()
    h = cell[s]
    k = kappa[s]
    p = h * ref_cell_inv[s]
    phi[s] = p
    p_inv = wp.inverse(p)
    phi_inv[s] = p_inv

    dof_a = cell_dof_a[s]
    dof_b = cell_dof_b[s]
    dof_a[0] = k * p[0, 0]
    dof_a[1] = k * p[1, 0]
    dof_a[2] = k * p[2, 0]
    dof_b[0] = k * p[1, 1]
    dof_b[1] = k * p[2, 1]
    dof_b[2] = k * p[2, 2]
    cell_dof_a[s] = dof_a
    cell_dof_b[s] = dof_b

    f_a = cell_force_a[s] - cell_force_a[s]
    f_b = cell_force_b[s] - cell_force_b[s]
    if have_stress:
        # f_c = -(V sigma) Phi^-T / kappa
        volume = wp.abs(wp.determinant(h))
        force = (-volume / k) * (stress[s] * wp.transpose(p_inv))
        f_a[0] = force[0, 0]
        f_a[1] = force[1, 0]
        f_a[2] = force[2, 0]
        f_b[0] = force[1, 1]
        f_b[1] = force[2, 1]
        f_b[2] = force[2, 2]
    cell_force_a[s] = f_a
    cell_force_b[s] = f_b


@wp.kernel(enable_backward=False)
def _lbfgs_pack_kernel(
    positions: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    phi: wp.array(dtype=Any),
    phi_inv: wp.array(dtype=Any),
    cell_dof_a: wp.array(dtype=Any),
    cell_dof_b: wp.array(dtype=Any),
    cell_force_a: wp.array(dtype=Any),
    cell_force_b: wp.array(dtype=Any),
    ext_batch_idx: wp.array(dtype=wp.int32),
    ext_atom_ptr: wp.array(dtype=wp.int32),
    ext_positions: wp.array(dtype=Any),
    ext_forces: wp.array(dtype=Any),
):
    """Map Cartesian positions and forces into the packed chart.

    The extended array interleaves each system's atoms with its two cell
    entries, so a single sorted index array covers both and every per-system
    reduction picks up the coupled atom-and-cell inner product for free.

    Thread launch
    -------------
    One thread per extended degree of freedom; ``dim = num_atoms + 2 * num_systems``.

    Modifies
    --------
    ext_positions, ext_forces
        OUTPUT. The packed coordinates and their conjugate forces.
    """
    e = wp.tid()
    s = ext_batch_idx[e]
    cell_start = ext_atom_ptr[s + 1] - wp.int32(2)
    if e >= cell_start:
        if e == cell_start:
            ext_positions[e] = cell_dof_a[s]
            ext_forces[e] = cell_force_a[s]
        else:
            ext_positions[e] = cell_dof_b[s]
            ext_forces[e] = cell_force_b[s]
        return
    a = e - wp.int32(2) * s
    ext_positions[e] = phi_inv[s] * positions[a]
    ext_forces[e] = wp.transpose(phi[s]) * forces[a]


@wp.kernel(enable_backward=False)
def _lbfgs_unpack_cell_kernel(
    ext_positions: wp.array(dtype=Any),
    ref_cell: wp.array(dtype=Any),
    ext_atom_ptr: wp.array(dtype=wp.int32),
    kappa: wp.array(dtype=Any),
    phi: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
):
    """Rebuild the cell from its packed coordinates.

    Only the six lower-triangular components are stored, so the cell stays
    lower-triangular by construction and cannot drift into a rotation.

    Thread launch
    -------------
    One thread per system; ``dim = num_systems``.

    Modifies
    --------
    phi
        The updated deformation gradient, reused by the atom unpacking.
    cell
        OUTPUT. ``H = Phi H0``.
    """
    s = wp.tid()
    h0 = ref_cell[s]
    k = kappa[s]
    cell_start = ext_atom_ptr[s + 1] - wp.int32(2)
    va = ext_positions[cell_start] / k
    vb = ext_positions[cell_start + 1] / k
    p = h0 - h0  # a zero matrix at the right precision
    p[0, 0] = va[0]
    p[1, 0] = va[1]
    p[2, 0] = va[2]
    p[1, 1] = vb[0]
    p[2, 1] = vb[1]
    p[2, 2] = vb[2]
    phi[s] = p
    cell[s] = p * h0


@wp.kernel(enable_backward=False)
def _lbfgs_unpack_atoms_kernel(
    ext_positions: wp.array(dtype=Any),
    phi: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=Any),
):
    """Map packed atom coordinates back to Cartesian positions, ``r = Phi u``.

    Thread launch
    -------------
    One thread per atom; ``dim = num_atoms``.

    Modifies
    --------
    positions
        OUTPUT. Cartesian positions.
    """
    a = wp.tid()
    s = batch_idx[a]
    positions[a] = phi[s] * ext_positions[a + 2 * s]


@wp.kernel(enable_backward=False)
def _lbfgs_cell_direction_kernel(
    direction: wp.array(dtype=Any),
    ext_atom_ptr: wp.array(dtype=wp.int32),
    kappa: wp.array(dtype=Any),
    d_phi: wp.array(dtype=Any),
):
    """Rebuild the cell part of the search direction as a matrix.

    Thread launch
    -------------
    One thread per system; ``dim = num_systems``.

    Modifies
    --------
    d_phi
        The direction's cell block, undone by ``kappa`` so it is a change in
        the deformation gradient rather than in the scaled coordinate.
    """
    s = wp.tid()
    k = kappa[s]
    cell_start = ext_atom_ptr[s + 1] - wp.int32(2)
    va = direction[cell_start] / k
    vb = direction[cell_start + 1] / k
    d = d_phi[s] - d_phi[s]  # a zero matrix at the right precision
    d[0, 0] = va[0]
    d[1, 0] = va[1]
    d[2, 0] = va[2]
    d[1, 1] = vb[0]
    d[2, 1] = vb[1]
    d[2, 2] = vb[2]
    d_phi[s] = d


@wp.kernel(enable_backward=False)
def _lbfgs_cell_trust_region_kernel(
    ext_positions: wp.array(dtype=Any),
    direction: wp.array(dtype=Any),
    phi: wp.array(dtype=Any),
    d_phi: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    dmax: wp.array(dtype=wp.float64),
    dquad: wp.array(dtype=wp.float64),
):
    """Measure the Cartesian displacement a variable-cell step produces.

    Both the cell and the coordinates move, so the displacement of an atom is

        dr = alpha * (Phi d_u + d_Phi u) + alpha**2 * (d_Phi d_u)

    which is quadratic in the step length, not linear. Reducing the largest
    magnitude of each term separately lets the step cap be solved in closed
    form; see :func:`_alpha_cap`. Only atoms are measured, because ``maxstep``
    is a limit on how far an atom may move.

    Thread launch
    -------------
    One thread per atom; ``dim = num_atoms``.

    Modifies
    --------
    dmax, dquad
        OUTPUT. Zeroed by the launcher, then accumulated with atomic maxima.
    """
    a = wp.tid()
    s = batch_idx[a]
    if status[s] != LBFGS_NEED_EVAL or n_loop[s] == _NLOOP_RETRY:
        return
    e = a + 2 * s
    d_u = direction[e]
    u = ext_positions[e]
    linear = phi[s] * d_u + d_phi[s] * u
    quadratic = d_phi[s] * d_u
    wp.atomic_max(dmax, s, wp.float64(wp.length(linear)))
    wp.atomic_max(dquad, s, wp.float64(wp.length(quadratic)))


_MAT_TYPES = {wp.vec3f: wp.mat33f, wp.vec3d: wp.mat33d}

_cell_kappa_overloads = {}
_cell_chart_overloads = {}
_pack_overloads = {}
_unpack_cell_overloads = {}
_unpack_atoms_overloads = {}
_cell_direction_overloads = {}
_cell_trust_region_overloads = {}

_SCALAR_OF = {wp.vec3f: wp.float32, wp.vec3d: wp.float64}

for _v, _mt in _MAT_TYPES.items():
    _sc = _SCALAR_OF[_v]
    _cell_kappa_overloads[_v] = wp.overload(
        _lbfgs_cell_kappa_kernel,
        [
            wp.array(dtype=_I32),  # n_atoms_per_system
            _F64,  # cell_force_scale
            wp.array(dtype=_sc),  # kappa
        ],
    )
    _cell_chart_overloads[_v] = wp.overload(
        _lbfgs_cell_chart_kernel,
        [
            wp.array(dtype=_mt),  # cell
            wp.array(dtype=_mt),  # ref_cell_inv
            wp.array(dtype=_mt),  # stress
            wp.array(dtype=_sc),  # kappa
            wp.array(dtype=_mt),  # phi
            wp.array(dtype=_mt),  # phi_inv
            wp.array(dtype=_v),  # cell_dof_a
            wp.array(dtype=_v),  # cell_dof_b
            wp.array(dtype=_v),  # cell_force_a
            wp.array(dtype=_v),  # cell_force_b
            wp.bool,  # have_stress
        ],
    )
    _pack_overloads[_v] = wp.overload(
        _lbfgs_pack_kernel,
        [
            wp.array(dtype=_v),  # positions
            wp.array(dtype=_v),  # forces
            wp.array(dtype=_mt),  # phi
            wp.array(dtype=_mt),  # phi_inv
            wp.array(dtype=_v),  # cell_dof_a
            wp.array(dtype=_v),  # cell_dof_b
            wp.array(dtype=_v),  # cell_force_a
            wp.array(dtype=_v),  # cell_force_b
            wp.array(dtype=_I32),  # ext_batch_idx
            wp.array(dtype=_I32),  # ext_atom_ptr
            wp.array(dtype=_v),  # ext_positions
            wp.array(dtype=_v),  # ext_forces
        ],
    )
    _unpack_cell_overloads[_v] = wp.overload(
        _lbfgs_unpack_cell_kernel,
        [
            wp.array(dtype=_v),  # ext_positions
            wp.array(dtype=_mt),  # ref_cell
            wp.array(dtype=_I32),  # ext_atom_ptr
            wp.array(dtype=_sc),  # kappa
            wp.array(dtype=_mt),  # phi
            wp.array(dtype=_mt),  # cell
        ],
    )
    _unpack_atoms_overloads[_v] = wp.overload(
        _lbfgs_unpack_atoms_kernel,
        [
            wp.array(dtype=_v),  # ext_positions
            wp.array(dtype=_mt),  # phi
            wp.array(dtype=_I32),  # batch_idx
            wp.array(dtype=_v),  # positions
        ],
    )
    _cell_direction_overloads[_v] = wp.overload(
        _lbfgs_cell_direction_kernel,
        [
            wp.array(dtype=_v),  # direction
            wp.array(dtype=_I32),  # ext_atom_ptr
            wp.array(dtype=_sc),  # kappa
            wp.array(dtype=_mt),  # d_phi
        ],
    )
    _cell_trust_region_overloads[_v] = wp.overload(
        _lbfgs_cell_trust_region_kernel,
        [
            wp.array(dtype=_v),  # ext_positions
            wp.array(dtype=_v),  # direction
            wp.array(dtype=_mt),  # phi
            wp.array(dtype=_mt),  # d_phi
            wp.array(dtype=_I32),  # batch_idx
            wp.array(dtype=_I32),  # status
            wp.array(dtype=_I32),  # n_loop
            wp.array(dtype=_F64),  # dmax
            wp.array(dtype=_F64),  # dquad
        ],
    )


def lbfgs_set_reference_cell(
    cell: wp.array,
    ref_cell: wp.array,
    ref_cell_inv: wp.array,
) -> None:
    """Capture the reference cell that defines the variable-cell chart.

    The generalized coordinates are measured relative to a cell ``H0`` that is
    fixed for the whole relaxation, which is what makes history pairs recorded
    at different iterations comparable. Call this once before the first step.

    Calling it again re-references the chart and invalidates every stored
    ``(s, y)`` pair, so restore the optimizer buffers to their required initial
    contents if you do.

    The reference cell is deliberately not part of that reset: zeroing it, as
    the other buffers are zeroed, would leave a singular matrix.

    Parameters
    ----------
    cell : wp.array(dtype=mat33), shape (num_systems,)
        Current cell, lattice vectors as columns. Should already be in the
        lower-triangular form the optimizer preserves.
    ref_cell, ref_cell_inv : wp.array(dtype=mat33), shape (num_systems,)
        OUTPUT. ``H0`` and its inverse.
    """
    wp.copy(ref_cell, cell)
    compute_cell_inverse(cell, ref_cell_inv, device=str(cell.device))


def lbfgs_cell_kappa(
    n_atoms_per_system: wp.array,
    kappa: wp.array,
    *,
    cell_force_scale: float = 1.0,
) -> None:
    """Fill the per-system cell coordinate scaling.

    The cell coordinate is ``kappa * Phi`` and its conjugate force is divided
    by the same ``kappa``, which is what keeps ``g . dx`` independent of the
    scaling. Larger values make the cell move less per step relative to the
    atoms; the usual choice, and the default here, is the atom count.

    Depends only on topology, so compute it once and reuse it.

    Parameters
    ----------
    n_atoms_per_system : wp.array(dtype=int32), shape (num_systems,)
        Atom count per system.
    kappa : wp.array, shape (num_systems,)
        OUTPUT. Must match the coordinate precision (float32 or float64).
    cell_force_scale : float, optional
        Multiplier on the atom count.
    """
    if cell_force_scale <= 0.0:
        raise ValueError(f"cell_force_scale must be positive; got {cell_force_scale}")
    vec = wp.vec3f if kappa.dtype == wp.float32 else wp.vec3d
    wp.launch(
        _cell_kappa_overloads[vec],
        dim=kappa.shape[0],
        inputs=[n_atoms_per_system, float(cell_force_scale), kappa],
        device=kappa.device,
    )


def lbfgs_pack_cell(
    positions: wp.array,
    forces: wp.array,
    cell: wp.array,
    stress: wp.array | None,
    ref_cell_inv: wp.array,
    kappa: wp.array,
    ext_batch_idx: wp.array,
    ext_atom_ptr: wp.array,
    phi: wp.array,
    phi_inv: wp.array,
    cell_dof_a: wp.array,
    cell_dof_b: wp.array,
    cell_force_a: wp.array,
    cell_force_b: wp.array,
    ext_positions: wp.array,
    ext_forces: wp.array,
) -> None:
    """Map Cartesian positions, forces, cell and stress into the packed chart.

    Parameters
    ----------
    positions, forces : wp.array, shape (num_atoms,)
        Cartesian geometry and forces.
    cell : wp.array(dtype=mat33), shape (num_systems,)
        Current cell, lattice vectors as columns.
    stress : wp.array(dtype=mat33) or None
        Cauchy stress per system. Without it the cell degrees of freedom get
        zero force and the cell will not move.
    ref_cell_inv : wp.array(dtype=mat33), shape (num_systems,)
        Inverse of the reference cell, from :func:`lbfgs_set_reference_cell`.
    ext_positions, ext_forces : wp.array, shape (num_atoms + 2 * num_systems,)
        OUTPUT. The packed arrays the optimizer works on.
    kappa : wp.array, shape (num_systems,)
        Cell coordinate scaling, from :func:`lbfgs_cell_kappa`.

    See Also
    --------
    lbfgs_unpack_cell : the inverse mapping.
    """
    device = positions.device
    vec_dtype = positions.dtype
    num_systems = cell.shape[0]
    have_stress = stress is not None
    stress_arg = stress if have_stress else cell  # unread when have_stress is False

    wp.launch(
        _cell_chart_overloads[vec_dtype],
        dim=num_systems,
        inputs=[
            cell,
            ref_cell_inv,
            stress_arg,
            kappa,
            phi,
            phi_inv,
            cell_dof_a,
            cell_dof_b,
            cell_force_a,
            cell_force_b,
            have_stress,
        ],
        device=device,
    )
    wp.launch(
        _pack_overloads[vec_dtype],
        dim=ext_positions.shape[0],
        inputs=[
            positions,
            forces,
            phi,
            phi_inv,
            cell_dof_a,
            cell_dof_b,
            cell_force_a,
            cell_force_b,
            ext_batch_idx,
            ext_atom_ptr,
            ext_positions,
            ext_forces,
        ],
        device=device,
    )


def lbfgs_unpack_cell(
    ext_positions: wp.array,
    ref_cell: wp.array,
    kappa: wp.array,
    batch_idx: wp.array,
    ext_atom_ptr: wp.array,
    phi: wp.array,
    positions: wp.array,
    cell: wp.array,
) -> None:
    """Map packed coordinates back to Cartesian positions and a cell.

    Parameters
    ----------
    ext_positions : wp.array, shape (num_atoms + 2 * num_systems,)
        Packed coordinates, as advanced by :func:`lbfgs_step`.
    ref_cell : wp.array(dtype=mat33), shape (num_systems,)
        The reference cell, from :func:`lbfgs_set_reference_cell`.
    positions, cell : wp.array
        OUTPUT. Cartesian positions and the updated cell.

    See Also
    --------
    lbfgs_pack_cell : the forward mapping.
    """
    device = ext_positions.device
    vec_dtype = ext_positions.dtype
    wp.launch(
        _unpack_cell_overloads[vec_dtype],
        dim=cell.shape[0],
        inputs=[
            ext_positions,
            ref_cell,
            ext_atom_ptr,
            kappa,
            phi,
            cell,
        ],
        device=device,
    )
    wp.launch(
        _unpack_atoms_overloads[vec_dtype],
        dim=positions.shape[0],
        inputs=[ext_positions, phi, batch_idx, positions],
        device=device,
    )


def lbfgs_cell_trust_region(
    ext_positions: wp.array,
    direction: wp.array,
    phi: wp.array,
    d_phi: wp.array,
    batch_idx: wp.array,
    ext_atom_ptr: wp.array,
    kappa: wp.array,
    status: wp.array,
    n_loop: wp.array,
    dmax: wp.array,
    dquad: wp.array,
) -> None:
    """Measure the Cartesian displacement a variable-cell direction produces.

    Call this between :func:`lbfgs_update` (with ``measure_trust_region=False``)
    and :func:`lbfgs_prepare_step`. The displacement is quadratic in the step
    length because the cell and the coordinates both move, so both the linear
    and quadratic terms are measured and the cap is solved in closed form.

    Modifies
    --------
    d_phi
        Scratch holding the direction's cell block as a matrix.
    dmax, dquad
        OUTPUT. The linear and quadratic displacement bounds.
    """
    device = ext_positions.device
    vec_dtype = ext_positions.dtype
    dmax.zero_()
    dquad.zero_()
    wp.launch(
        _cell_direction_overloads[vec_dtype],
        dim=d_phi.shape[0],
        inputs=[
            direction,
            ext_atom_ptr,
            kappa,
            d_phi,
        ],
        device=device,
    )
    wp.launch(
        _cell_trust_region_overloads[vec_dtype],
        dim=batch_idx.shape[0],
        inputs=[
            ext_positions,
            direction,
            phi,
            d_phi,
            batch_idx,
            status,
            n_loop,
            dmax,
            dquad,
        ],
        device=device,
    )


def lbfgs_step_coord_cell(
    positions: wp.array,
    forces: wp.array,
    cell: wp.array,
    stress: wp.array,
    energy: wp.array,
    batch_idx: wp.array,
    n_particles: wp.array,
    x_base: wp.array,
    force_base: wp.array,
    direction: wp.array,
    s_history: wp.array,
    y_history: wp.array,
    ys: wp.array,
    yy: wp.array,
    alpha_hist: wp.array,
    beta_hist: wp.array,
    ss: wp.array,
    f_base: wp.array,
    gg: wp.array,
    gd: wp.array,
    fmax: wp.array,
    frms_sq: wp.array,
    smax: wp.array,
    d0: wp.array,
    dmax: wp.array,
    dquad: wp.array,
    alpha_step: wp.array,
    status: wp.array,
    iteration: wp.array,
    end: wp.array,
    n_loop: wp.array,
    ls_trials: wp.array,
    history_count: wp.array,
    ref_cell: wp.array,
    ref_cell_inv: wp.array,
    kappa: wp.array,
    ext_batch_idx: wp.array,
    ext_atom_ptr: wp.array,
    phi: wp.array,
    phi_inv: wp.array,
    d_phi: wp.array,
    cell_dof_a: wp.array,
    cell_dof_b: wp.array,
    cell_force_a: wp.array,
    cell_force_b: wp.array,
    ext_positions: wp.array,
    ext_forces: wp.array,
    *,
    force_tol: float = 0.05,
    rms_tol: float = 0.0,
    stress_tol: float = 0.0,
    ftol: float = 1e-4,
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

    Maps the geometry into the packed chart, takes one L-BFGS step there, and
    maps the result back. Because both blocks live in one coordinate vector,
    the two-loop recursion couples them without any special handling.

    Consumes exactly one energy/force/stress evaluation, like the
    coordinate-only :func:`lbfgs_step`. Progress is reported the same way,
    through ``status``.

    Every buffer is caller-owned; nothing is allocated here. Call
    :func:`lbfgs_set_reference_cell` and :func:`lbfgs_cell_kappa` once before
    the first step, and build ``ext_batch_idx`` / ``ext_atom_ptr`` yourself so
    ragged batches are expressible.

    Parameters
    ----------
    positions : wp.array, shape (num_atoms,)
        Cartesian geometry, advanced in place.
    forces : wp.array, shape (num_atoms,)
        Cartesian forces at ``positions``. Forces, not gradients.
    cell : wp.array(dtype=mat33), shape (num_systems,)
        Cell with lattice vectors as columns, advanced in place. Kept
        lower-triangular, so it cannot drift into a rotation.
    stress : wp.array(dtype=mat33), shape (num_systems,)
        Cauchy stress per system. This drives the cell degrees of freedom, and
        is also what ``stress_tol`` is compared against.
    energy : wp.array(dtype=float64), shape (num_systems,)
        Per-system total energy at ``positions``.
    batch_idx : wp.array(dtype=int32), shape (num_atoms,)
        Sorted system index per atom.
    n_particles : wp.array(dtype=int32), shape (num_systems,)
        Atom count per system.
    x_base, ..., history_count
        The optimizer buffers, sized for ``num_atoms + 2 * num_systems``
        degrees of freedom rather than ``num_atoms``. See :func:`lbfgs_update`
        for shapes and required initial contents.
    ref_cell, ref_cell_inv : wp.array(dtype=mat33), shape (num_systems,)
        The reference cell and its inverse, from
        :func:`lbfgs_set_reference_cell`. Read only.
    kappa : wp.array, shape (num_systems,)
        Cell coordinate scaling, from :func:`lbfgs_cell_kappa`. Read only, and
        at the coordinate precision rather than float64.
    ext_batch_idx : wp.array(dtype=int32), shape (num_atoms + 2 * num_systems,)
        Sorted system index for each packed degree of freedom. Read only.
    ext_atom_ptr : wp.array(dtype=int32), shape (num_systems + 1,)
        Start offset of each system in the packed array, where system ``s``
        begins at ``atom_ptr[s] + 2 * s``. Read only.
    phi, phi_inv, d_phi : wp.array(dtype=mat33), shape (num_systems,)
        Scratch for the deformation gradient and the direction's cell block.
    cell_dof_a, cell_dof_b, cell_force_a, cell_force_b : wp.array, shape (num_systems,)
        Scratch for the six packed cell coordinates and their conjugate forces.
    ext_positions, ext_forces : wp.array, shape (num_atoms + 2 * num_systems,)
        Scratch for the packed coordinates and forces.
    force_tol, rms_tol, stress_tol : float, optional
        Convergence thresholds. These are always evaluated on the **Cartesian**
        forces and the stress, never on packed norms, so ``force_tol`` keeps its
        meaning as a force per atom however far the cell deforms.
    maxstep : float, optional
        Largest Cartesian distance an atom may move in one step. On this path
        the displacement is quadratic in the step length, because the cell and
        the coordinates both move, so the cap is solved rather than estimated.

    See Also
    --------
    lbfgs_set_reference_cell : must be called first.
    lbfgs_cell_kappa : must be called first.
    lbfgs_step : the coordinate-only equivalent.
    """
    lbfgs_pack_cell(
        positions,
        forces,
        cell,
        stress,
        ref_cell_inv,
        kappa,
        ext_batch_idx,
        ext_atom_ptr,
        phi,
        phi_inv,
        cell_dof_a,
        cell_dof_b,
        cell_force_a,
        cell_force_b,
        ext_positions,
        ext_forces,
    )
    lbfgs_update(
        positions=ext_positions,
        forces=ext_forces,
        batch_idx=ext_batch_idx,
        energy=energy,
        n_particles=n_particles,
        cart_forces=forces,
        atom_batch_idx=batch_idx,
        stress=stress,
        measure_trust_region=False,
        x_base=x_base,
        force_base=force_base,
        direction=direction,
        s_history=s_history,
        y_history=y_history,
        ys=ys,
        yy=yy,
        alpha_hist=alpha_hist,
        beta_hist=beta_hist,
        ss=ss,
        f_base=f_base,
        gg=gg,
        gd=gd,
        fmax=fmax,
        frms_sq=frms_sq,
        smax=smax,
        d0=d0,
        dmax=dmax,
        dquad=dquad,
        alpha_step=alpha_step,
        status=status,
        iteration=iteration,
        end=end,
        n_loop=n_loop,
        ls_trials=ls_trials,
        history_count=history_count,
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
    # The displacement a direction produces is not its magnitude here, so the
    # trust region gets its own measure before the step length is finalized.
    lbfgs_cell_trust_region(
        ext_positions,
        direction,
        phi,
        d_phi,
        batch_idx,
        ext_atom_ptr,
        kappa,
        status,
        n_loop,
        dmax,
        dquad,
    )
    lbfgs_prepare_step(
        gg=gg,
        d0=d0,
        dmax=dmax,
        dquad=dquad,
        alpha_step=alpha_step,
        status=status,
        end=end,
        n_loop=n_loop,
        ls_trials=ls_trials,
        history_count=history_count,
        maxstep=maxstep,
    )
    lbfgs_apply_step(
        positions=ext_positions,
        forces=ext_forces,
        x_base=x_base,
        force_base=force_base,
        direction=direction,
        batch_idx=ext_batch_idx,
        status=status,
        n_loop=n_loop,
        gg=gg,
        alpha_step=alpha_step,
    )
    lbfgs_unpack_cell(
        ext_positions,
        ref_cell,
        kappa,
        batch_idx,
        ext_atom_ptr,
        phi,
        positions,
        cell,
    )
