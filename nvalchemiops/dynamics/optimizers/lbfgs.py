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

r"""L-BFGS Optimizer Kernels
========================

GPU-accelerated Warp kernels for batched L-BFGS geometry optimization.

L-BFGS approximates the inverse Hessian from the last ``m`` position/force
differences to pick a direction, then steps along it as far as a ``maxstep``
trust region allows. It reaches a given force tolerance in far fewer force
evaluations than FIRE -- the cost that dominates relaxation with a
machine-learned potential.

Forces only: there is no line search
------------------------------------
**Nothing here reads an energy.** An Armijo test compares *total energies*
while the direction comes from *forces* -- the same surface only for a
conservative model, so a direct force head makes it reject good steps, and no
tuning repairs that. Bounding by ``maxstep`` instead makes the cadence uniform:
**every call is an accepted step** forming one curvature pair. The cost is that
nothing forces the energy down, so it may rise on a step; the force still
converges.

Convergence is the caller's
---------------------------
The optimizer owns no tolerance and has no terminal state. Each call updates
the history, restarts if the direction stops descending, and takes one bounded
step -- nothing more. This mirrors FIRE2, and it means the stopping rule stays
where the physics is: you may want a force threshold, a stress threshold, an
evaluation budget, or all three, and none of that belongs in a kernel.

Calling convention
------------------
You own the loop. Each :func:`lbfgs_step` call consumes **exactly one** force
evaluation::

    for _ in range(max_steps):
        forces = my_model(positions)
        if np.linalg.norm(forces.numpy(), axis=1).max() < force_tol:
            break
        lbfgs_step(positions, forces, state, batch_idx, maxstep=0.2)

Test *before* stepping, as above: the forces you were handed describe the
positions you have, and once you step they describe the previous point.

Systems stay in lock step in *evaluations* while diverging in *iterations*, so
a batch relaxes in one stream of launches with no per-system host control
flow. A batch converges when its slowest system does; to retire finished
systems earlier, compact the batch on the host.

Relation to FIRE2
-----------------
The decomposition mirrors FIRE2's, so both are driven the same way and a caller
can interpose logic at the same points:

========================  ==========================  ========================
Phase                     FIRE2                       L-BFGS
========================  ==========================  ========================
per-system reductions     ``fire2_reduce``            :func:`lbfgs_reduce`
state update, no motion   ``fire2_update``            :func:`lbfgs_update`
step-length finalization  --                          :func:`lbfgs_prepare_step`
move the atoms            ``fire2_apply_step``        :func:`lbfgs_apply_step`
all of the above          ``fire2_step``              :func:`lbfgs_step`
========================  ==========================  ========================

Cadence matches too: one state update and one step per call, and neither owns
a tolerance or a terminal status. The extra phase exists because the step
length must be settled *before* the atoms move. ``maxstep`` shrinks ``alpha``
rather than clamping each displacement, since the stored pair ``s = alpha * d``
is defined through that relation.

That table is the internal structure, not the public surface. Like FIRE2, what
is exported is the state, the preparation helpers and one step per call; the
phases are importable from this module by name for anyone who needs to
interpose logic between them, but they are decomposition points rather than
operations in their own right.

Restarts, which *are* the optimizer's
-------------------------------------
Two things are algorithmic rather than terminal, so they stay inside. A
curvature pair with ``s . y`` too small is discarded, and a two-loop direction
with ``d0 >= 0`` is replaced by steepest descent. Neither stops anything: they
keep the model well posed, so a run cannot stall.

Positions only move *forward* from the evaluated point, so the ``forces`` you
passed in still describe the ``positions`` you get back.

Forces, not gradients
---------------------
The API is in **forces**, the algorithm in gradients, with ``g = -F`` folded
into each kernel. So ``y_history`` holds *gradient* differences
(``force_base - F``), and a descent direction satisfies ``force_base . d > 0``.

Precision
---------
**Every array follows the coordinate dtype.** ``wp.vec3f`` gives an fp32 state
end to end, ``wp.vec3d`` an fp64 one. There is no mixed configuration: the
overloads are keyed so a mismatched state is rejected rather than silently
half-converted.

An earlier version pinned the per-system scalars to float64 whatever the
coordinates, on the grounds that ``y = force_base - F`` cancels near
convergence. That reasoning does not survive contact with the code. ``y`` is
formed *at the coordinate precision*, and so is each ``wp.dot``; the float64
only ever saw an already-rounded product::

    yvec = force_base[i] - fi                  # fp32 subtraction: the loss is here
    acc += wp.float64(wp.dot(svec, yvec))      # widened only after the fact

A wider accumulator cannot recover what that subtraction discarded. What it
did change is the summation across degrees of freedom -- a different and much
smaller effect. Measured on fp32 inputs: the cancellation in ``y`` costs about
``3e-2`` relative error once ``|dF|/|F|`` reaches ``1e-6``, while fp32
accumulation of the dot product costs about ``1e-5`` at a million degrees of
freedom. The accumulator was refining a quantity already swamped by three
orders of magnitude.

End to end the difference is not measurable. Over twelve Lennard-Jones
relaxations (13 to 55 atoms, four seeds each) both policies converged 12/12,
with a geometric-mean evaluation ratio of 0.96 and a best-reachable-force
ratio of 1.02 -- both inside the scatter you get from any one-ULP perturbation,
since the trajectory is a discontinuous function of reduction order.

So the float64 bought nothing and cost something: fp32 users paid for float64
arithmetic, and in JAX they were forced to enable ``JAX_ENABLE_X64`` for an
fp32 run. If some future reduction genuinely needs a wider accumulator, widen
*that* reduction and bring the measurement with it.

One threshold does follow from this and is not a free parameter.
``curvature_eps`` gates the history on ``ys > eps * sqrt(ss * yy)`` -- a floor
on the cosine between ``s`` and ``y`` -- and ``ys`` is accumulated at the
coordinate precision. The floor therefore has to sit above the level at which
that sum is still signal, which is about ``1e-8`` in float32 and ``1e-17`` in
float64. It defaults to ``1e-6`` and ``1e-10`` respectively; see
:data:`_CURVATURE_EPS`. Pass a value to override, per system precision as you
see fit.

State
-----
:func:`lbfgs_prepare_state` allocates, initializes and validates everything in
one call and hands it back as an :class:`LBFGSState`; calling it again is how
you reset. Nothing is allocated per step, so the step stays capturable in a
CUDA graph.

The arrays stay yours. :class:`LBFGSState` is a plain dataclass whose fields
are reachable by name -- ``state.iteration``, ``state.history_count`` -- so
you can also build one from buffers you already own and check it with
:meth:`LBFGSState.validate`, which compares shapes, dtypes and device without
touching the GPU. For ``P`` degrees of freedom, ``M`` systems and history
depth ``m``:

=============================================  ==========  ===========
Buffer                                         Shape       dtype
=============================================  ==========  ===========
``x_base``, ``force_base``, ``direction``      ``(P,)``    vec3f/vec3d
``s_history``, ``y_history``                   ``(m, P)``  vec3f/vec3d
``ys``, ``yy``, ``alpha_hist``, ``beta_hist``  ``(m, M)``  float32/float64
``ss``, ``gg``                                 ``(M,)``    float32/float64
``d0``, ``dmax``, ``dquad``, ``alpha_step``    ``(M,)``    float32/float64
``iteration``, ``end``                         ``(M,)``    int32
``n_loop``, ``history_count``                  ``(M,)``    int32
=============================================  ==========  ===========

Building one by hand means matching the initial contents
:func:`lbfgs_prepare_state` produces: **zero everything**, then ``alpha_step``
to **one** and ``iteration`` to **minus one** (the "never evaluated" marker).

Variable-cell relaxation adds an :class:`LBFGSCellState` from
:func:`lbfgs_prepare_cell_state`, for the packed path with
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
matrices. ``ref_cell``, ``ref_cell_inv`` and ``kappa`` are the chart; the rest
is scratch.

.. _lbfgs-cell-contract:

The variable-cell contract
--------------------------
**This is the one authoritative statement of these rules.** The PyTorch and
JAX bindings and the user guide all defer here rather than restating them.

*1. Align the cell first, exactly as FIRE2 requires.*
   Call :func:`~nvalchemiops.dynamics.utils.cell_filter.align_cell` once,
   before the first step, and pass the aligned cell and rotated positions in.
   This is the same requirement and the same helper that
   ``fire2_step_coord_cell`` documents -- not a second convention.

   The package calls the result **upper-triangular**, in the lattice-vector
   reading: ``a`` along x, ``b`` in the xy-plane, ``c`` general. As a *matrix*
   with lattice vectors in columns that is zeros strictly **above** the
   diagonal, which is why the same object gets described both ways in the
   wild. Only one wording is used here, the package's.

   :func:`lbfgs_prepare_cell_state` checks this for you when you hand it a
   ``cell``, because it is a one-time setup cost rather than a per-step one.

*2. Six components, the same six FIRE2 packs.*
   ``(0,0), (1,0), (2,0) | (1,1), (2,1), (2,2)`` -- two ``vec3`` entries per
   system, in that order, matching
   :func:`~nvalchemiops.dynamics.utils.cell_filter.pack_positions_with_cell`.
   The three strictly-upper entries are not represented at all, so the cell
   cannot drift into a rotation. This is why step 1 is required rather than
   advisory: in an unaligned frame those three entries are *not* the redundant
   ones, and constraining them constrains the wrong thing.

*3. What L-BFGS does differently, and why.*
   FIRE2 packs the cell itself. L-BFGS packs the deformation gradient
   ``Phi = H H_ref^-1`` against a **fixed** reference cell, scaled by
   ``kappa`` (ASE's ``UnitCellFilter`` chart). The stored ``(s, y)`` pairs
   compare cell coordinates *across steps*, so those coordinates must mean the
   same thing at every step -- which they do only if the reference is held
   fixed for the whole relaxation.

   **Re-referencing invalidates every stored pair.** Calling
   :func:`lbfgs_set_reference_cell` again, or rebuilding the chart mid-run,
   silently makes the history describe a frame that no longer exists. If you
   must re-reference, reset the state with :func:`lbfgs_prepare_state` in the
   same breath. FIRE2 carries no such history and so has no such rule; this is
   the one place the two genuinely differ.

*4. Topology is yours, which is what makes ragged batches work.*
   ``ext_batch_idx`` and ``ext_atom_ptr`` are required rather than derived,
   because deriving them would mean assuming an even split::

       from nvalchemiops.batch_utils import atom_ptr_to_batch_idx
       from nvalchemiops.dynamics.utils.cell_filter import extend_atom_ptr

       extend_atom_ptr(atom_ptr, ext_atom_ptr)   # ext_atom_ptr[s] = atom_ptr[s] + 2s
       atom_ptr_to_batch_idx(ext_atom_ptr, ext_batch_idx)

   Systems may have different atom counts. Each contributes exactly two packed
   entries regardless, so ``P = num_atoms + 2 * M`` holds for any split, and
   :meth:`LBFGSCellState.validate` checks that relationship.

*5. Empty systems.*
   A system with no atoms still owns its two cell degrees of freedom. Its
   ``kappa`` would be zero on the atom count alone, and ``kappa`` divides the
   cell force, so :func:`lbfgs_cell_kappa` clamps the count to one. The scale
   is arbitrary there anyway -- there is nothing to balance the cell against.
   The JAX helper follows the same contract.

Memory
------
For ``P`` degrees of freedom, ``M`` systems and history ``m``::

    bytes = (2m + 3) * 3 * sizeof(dof) * P + (4m + 6) * sizeof(dof) * M
            + 4 * 4 * M

``sizeof(dof)`` appears in both array terms because every scalar follows the
coordinate dtype. At ``m = 6`` that is 180 bytes per degree of freedom with
float32 coordinates, 360 with float64. The two histories dominate; 3 to 7 is the usual range for
``m``.

References
----------
Nocedal, J. "Updating Quasi-Newton Matrices with Limited Storage."
*Math. Comp.* 35 (1980) 773-782.

Liu, D. C. and Nocedal, J. "On the limited memory BFGS method for large scale
optimization." *Math. Program.* 45 (1989) 503-528.

Nocedal, J. and Wright, S. J. *Numerical Optimization*, 2nd ed., chapters 3
and 7."""

from __future__ import annotations

import dataclasses
from typing import Any

import warp as wp

from nvalchemiops.dynamics.utils.cell_utils import compute_cell_inverse
from nvalchemiops.segment_ops import compute_ept

#: The public contract: the two states, the preparation helpers, one step per
#: call, and the variable-cell setup. ``lbfgs_reduce``, ``lbfgs_update``,
#: ``lbfgs_prepare_step``, ``lbfgs_apply_step``, ``lbfgs_pack_cell``,
#: ``lbfgs_unpack_cell`` and ``lbfgs_cell_trust_region`` are the decomposition
#: a step is built from rather than operations in their own right, so they are
#: deliberately absent -- as ``fire2_apply_step`` and ``fire2_reduce`` are from
#: ``fire2.__all__``. They remain importable by name for anyone who needs to
#: interpose logic between phases.
__all__ = [
    "LBFGSCellState",
    "LBFGSState",
    "check_cell_is_aligned",
    "lbfgs_cell_kappa",
    "lbfgs_prepare_cell_state",
    "lbfgs_prepare_state",
    "lbfgs_set_reference_cell",
    "lbfgs_step",
    "lbfgs_step_coord_cell",
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


@dataclasses.dataclass
class LBFGSState:
    """The optimizer's persistent and scratch arrays, one field per array.

    Deliberately transparent: every array a step touches is a named field you
    can read, plot or checkpoint. Build one with
    :func:`lbfgs_prepare_state`, which allocates and initializes it, or
    construct it directly from arrays you already own -- the package never
    requires you to use its allocation policy.

    Field order is the wire ABI. It is what the PyTorch operator's
    ``mutates_args`` and the JAX callable's ``in_out_argnames`` are generated
    from, so reordering these fields reorders those.

    Attributes
    ----------
    x_base, force_base : array, shape (P,)
        Last accepted point and the FORCES there. The gradient is
        ``-force_base``; see the module docstring on the sign convention.
    direction : array, shape (P,)
        Search direction; a descent direction satisfies ``force_base . d > 0``.
    s_history, y_history : array, shape (m, P)
        Ring buffers of position and *gradient* differences.
    ys, yy, alpha_hist, beta_hist : array, shape (m, M), float64
        Per-slot curvature products and two-loop coefficients.
    ss, gg : array, shape (M,), float64
        Squared norms of the newest ``s`` and of the packed force.
    d0, dmax, dquad : array, shape (M,), float64
        Directional derivative and the trust region's linear and quadratic
        displacement coefficients.
    alpha_step : array, shape (M,), float64
        Step length, recomputed from the trust region every call.
    iteration, end, n_loop, history_count : array, shape (M,), int32
        Per-system control state. There is no ``status``: the optimizer has no
        terminal state, and deciding when to stop is the caller's, as it is
        for FIRE2.
    """

    x_base: Any
    force_base: Any
    direction: Any
    s_history: Any
    y_history: Any
    ys: Any
    yy: Any
    alpha_hist: Any
    beta_hist: Any
    ss: Any
    gg: Any
    d0: Any
    dmax: Any
    dquad: Any
    alpha_step: Any
    iteration: Any
    end: Any
    n_loop: Any
    history_count: Any

    @property
    def num_dofs(self) -> int:
        """Degrees of freedom the optimizer moves."""
        return self.x_base.shape[0]

    @property
    def num_systems(self) -> int:
        """Independent systems in the batch."""
        return self.iteration.shape[0]

    @property
    def history_size(self) -> int:
        """Stored curvature pairs, ``m``."""
        return self.s_history.shape[0]

    def validate(self) -> None:
        """Check the fields against each other: shapes, dtypes and device.

        Structural only -- no device reads -- so this is sync-free and safe to
        call anywhere. Run for you by :func:`lbfgs_prepare_state`; call it
        yourself if you built the state from your own arrays.

        Three checks. Only *leading* dimensions are compared, because the
        component axis is a framework detail: Warp stores a ``vec3`` array as
        ``(P,)`` while PyTorch and JAX store it as ``(P, 3)``. Fields that
        hold the same kind of thing must share a dtype, which is what catches
        one array left at the wrong precision. And every field must live on
        one device.

        Sharing a dtype is checked rather than a named one, since the same
        state is expressed in three frameworks' type vocabularies; that the
        per-system scalars are specifically float64 is checked by each
        binding.

        Raises
        ------
        ValueError
            If any field is inconsistent with the rest.
        """
        p, m, n = self.num_dofs, self.history_size, self.num_systems
        if m < 1:
            raise ValueError(f"history size must be >= 1; got {m}")

        def check(name, want):
            got = tuple(getattr(self, name).shape)[: len(want)]
            if got != want:
                raise ValueError(
                    f"LBFGSState.{name} starts with dimensions {got}, expected "
                    f"{want} for {p} degrees of freedom, {n} systems and "
                    f"history {m}"
                )

        for name in ("x_base", "force_base", "direction"):
            check(name, (p,))
        for name in ("s_history", "y_history"):
            check(name, (m, p))
        for name in ("ys", "yy", "alpha_hist", "beta_hist"):
            check(name, (m, n))
        for name in _OPTIMIZER_BUFFERS[9:]:
            check(name, (n,))
        _check_kinds(self, _STATE_KINDS, "LBFGSState")


@dataclasses.dataclass
class LBFGSCellState:
    """The variable-cell chart and its scratch space, one field per array.

    The first five fields are configuration and topology, written once before
    the first step and read-only thereafter; the rest is scratch the step
    rewrites every call. Build one with :func:`lbfgs_prepare_cell_state` or
    construct it from your own arrays.

    Attributes
    ----------
    ref_cell, ref_cell_inv : array, shape (M,), mat33
        Reference cell ``H0`` and its inverse, from
        :func:`lbfgs_set_reference_cell`.
    kappa : array, shape (M,)
        Cell coordinate scaling, from :func:`lbfgs_cell_kappa`. Matches the
        *coordinate* precision, not float64, because it scales matrices.
    ext_batch_idx : array, shape (P,), int32
    ext_atom_ptr : array, shape (M + 1,), int32
        Packed topology. Build these with the generic batch utilities so ragged
        batches are expressible.
    phi, phi_inv, d_phi : array, shape (M,), mat33
        Deformation gradient, its inverse, and its search direction.
    cell_dof_a, cell_dof_b, cell_force_a, cell_force_b : array, shape (M,)
        The cell's two packed degrees of freedom and their conjugate forces.
    ext_positions, ext_forces : array, shape (P,)
        The packed coordinate vector and its forces.
    """

    ref_cell: Any
    ref_cell_inv: Any
    kappa: Any
    ext_batch_idx: Any
    ext_atom_ptr: Any
    phi: Any
    phi_inv: Any
    d_phi: Any
    cell_dof_a: Any
    cell_dof_b: Any
    cell_force_a: Any
    cell_force_b: Any
    ext_positions: Any
    ext_forces: Any

    @property
    def num_systems(self) -> int:
        """Independent systems in the batch."""
        return self.ref_cell.shape[0]

    @property
    def num_packed_dofs(self) -> int:
        """Packed degrees of freedom, ``num_atoms + 2 * num_systems``."""
        return self.ext_positions.shape[0]

    def validate(self, num_atoms: int | None = None) -> None:
        """Check shapes, dtypes, device and the packed relationship.

        Structural only, so no device reads: this confirms that
        ``P = num_atoms + 2 * num_systems`` and that every array is sized for
        it, but not that ``ext_atom_ptr`` holds sensible values -- checking
        that would need a device-to-host sync. As with
        :meth:`LBFGSState.validate`, fields of the same kind must share a
        dtype and the whole state one device. ``kappa`` is exempt from the
        dtype grouping: it scales matrices, so it follows the *coordinate*
        precision rather than the float64 of the other scalars.

        Parameters
        ----------
        num_atoms : int, optional
            Atom count. When given, the packed size is checked against it.

        Raises
        ------
        ValueError
            If any field is inconsistent with the rest.
        """
        n, p = self.num_systems, self.num_packed_dofs

        def check(name, want):
            got = tuple(getattr(self, name).shape)[: len(want)]
            if got != want:
                raise ValueError(
                    f"LBFGSCellState.{name} starts with dimensions {got}, "
                    f"expected {want}"
                )

        for name in _CELL_BUFFERS:
            if name in ("ext_positions", "ext_forces", "ext_batch_idx"):
                check(name, (p,))
            elif name == "ext_atom_ptr":
                check(name, (n + 1,))
            else:
                check(name, (n,))
        _check_kinds(self, _CELL_KINDS, "LBFGSCellState")
        if num_atoms is not None and p != num_atoms + 2 * n:
            raise ValueError(
                f"packed arrays hold {p} degrees of freedom, but {num_atoms} "
                f"atoms in {n} systems needs {num_atoms + 2 * n} "
                "(num_atoms + 2 * num_systems)"
            )


#: Field order of :class:`LBFGSState`. This *is* the ABI: the PyTorch
#: operator's ``mutates_args`` and the JAX callable's ``in_out_argnames`` are
#: both generated from it, and import-time checks pin the hand-written
#: signatures to it.
_OPTIMIZER_BUFFERS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(LBFGSState)
)

#: Field order of :class:`LBFGSCellState`.
_CELL_BUFFERS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(LBFGSCellState)
)

#: The cell fields the step writes. The first five are read-only
#: configuration and topology, so they are inputs rather than in-out aliases.
_CELL_SCRATCH: tuple[str, ...] = _CELL_BUFFERS[5:]

#: Warp spells one precision as a vector, a matrix and a scalar; these map
#: onto the scalar so the three can be compared. PyTorch and JAX dtypes pass
#: through :func:`_precision_of` unchanged, because theirs already are scalars.
_PRECISION_OF = {
    wp.vec3f: wp.float32, wp.mat33f: wp.float32, wp.float32: wp.float32,
    wp.vec3d: wp.float64, wp.mat33d: wp.float64, wp.float64: wp.float64,
}  # fmt: skip

#: Groups holding integers, which carry no floating-point precision to agree
#: on. Keyed by label rather than by dtype, because ``wp.int32``,
#: ``torch.int32`` and JAX's ``int32`` are three unrelated objects.
_INTEGER_KINDS = frozenset({"per-system integer control fields", "topology indices"})


#: Fields grouped by what they hold, for validation. Every field in a group
#: must share one dtype -- that is the check, rather than a named dtype,
#: because the same state is expressed in Warp, PyTorch and JAX types.
_STATE_KINDS: dict[str, tuple[str, ...]] = {
    "per-degree-of-freedom vectors": _OPTIMIZER_BUFFERS[:5],
    "per-system scalars": _OPTIMIZER_BUFFERS[5:15],
    "per-system integer control fields": _OPTIMIZER_BUFFERS[15:],
}
_CELL_KINDS: dict[str, tuple[str, ...]] = {
    "cell matrices": ("ref_cell", "ref_cell_inv", "phi", "phi_inv", "d_phi"),
    "packed vectors": (
        "cell_dof_a",
        "cell_dof_b",
        "cell_force_a",
        "cell_force_b",
        "ext_positions",
        "ext_forces",
    ),  # fmt: skip
    "topology indices": ("ext_batch_idx", "ext_atom_ptr"),
    # Its own group: kappa scales matrices, so it is the scalar counterpart of
    # the coordinate precision rather than a companion of any other field. The
    # cross-group check below is what ties it to the rest.
    "cell scaling": ("kappa",),
}

# The groups are written out by slice and by hand, so pin them to the field
# order. Inserting a field would otherwise shift a slice silently, and the
# dtype check would then compare the wrong arrays.
for _names, _all, _label in (
    (_STATE_KINDS, _OPTIMIZER_BUFFERS, "_STATE_KINDS"),
    (_CELL_KINDS, _CELL_BUFFERS, "_CELL_KINDS"),
):
    _grouped = [n for g in _names.values() for n in g]
    _ungrouped = set(_all) - set(_grouped)
    if len(_grouped) != len(set(_grouped)) or not set(_grouped) <= set(_all):
        # Raised rather than asserted so the guard survives `python -O`.
        raise RuntimeError(f"{_label} does not partition its fields: {_grouped}")
    if _ungrouped:
        raise RuntimeError(f"{_label} leaves fields unchecked: {sorted(_ungrouped)}")
# Pin the flags to real group labels. A renamed label would otherwise drop out
# of this set silently, and its integer dtype would then be compared against
# the floating-point ones as if it were a precision.
_ALL_KIND_LABELS = set(_STATE_KINDS) | set(_CELL_KINDS)
if not _INTEGER_KINDS <= _ALL_KIND_LABELS:
    # Raised rather than asserted so the guard survives `python -O`.
    raise RuntimeError(
        f"_INTEGER_KINDS names groups that do not exist: "
        f"{sorted(_INTEGER_KINDS - _ALL_KIND_LABELS)}"
    )


def _precision_of(dtype):
    """The scalar precision a coordinate, matrix or scalar dtype implies.

    Warp spells the same precision three ways -- ``vec3d``, ``mat33d``,
    ``float64`` -- while PyTorch and JAX store coordinates in an array whose
    dtype already *is* the scalar. Mapping the Warp composites onto their
    scalar and passing everything else through unchanged lets one comparison
    serve all three frameworks.
    """
    return _PRECISION_OF.get(dtype, dtype)


def _check_kinds(state, kinds: dict[str, tuple[str, ...]], cls_name: str) -> None:
    """Check dtypes within and across groups, and the device across the state.

    Three checks. Every field in a group shares a dtype. Every group that
    carries a floating-point precision agrees on *which* precision, so a state
    cannot pair fp64 coordinates with an fp32 scalar group -- that combination
    is not registered, and without this it reaches the kernel launcher and
    fails there with a dtype error rather than here with a documented one.
    And the whole state lives on one device.

    All read off the arrays themselves, so this works unchanged for Warp,
    PyTorch and JAX. A JAX tracer has no device, so the device check is skipped
    for anything that does not expose one rather than failing under ``jit``.
    """
    precisions = {}
    for label, names in kinds.items():
        seen = {getattr(state, n).dtype for n in names}
        if len(seen) > 1:
            offenders = {n: str(getattr(state, n).dtype) for n in names}
            raise ValueError(
                f"{cls_name} {label} must share one dtype, got {len(seen)}: {offenders}"
            )
        if label not in _INTEGER_KINDS:
            precisions.setdefault(_precision_of(seen.pop()), []).append(label)
    if len(precisions) > 1:
        raise ValueError(
            f"{cls_name} mixes floating-point precisions: "
            + "; ".join(
                f"{p} in {sorted(labels)}"
                for p, labels in sorted(precisions.items(), key=lambda kv: str(kv[0]))
            )
            + ". Every array follows the coordinate dtype; see the precision "
            "section of the module documentation."
        )
    devices = {}
    for f in dataclasses.fields(state):
        device = getattr(getattr(state, f.name), "device", None)
        if device is not None:
            devices.setdefault(str(device), []).append(f.name)
    if len(devices) > 1:
        raise ValueError(
            f"{cls_name} fields are spread across devices {sorted(devices)}: "
            + "; ".join(f"{d}: {sorted(n)}" for d, n in sorted(devices.items()))
        )


# -----------------------------------------------------------------------------
# Internal ``n_loop`` sentinels.
#
# ``n_loop`` is the single per-system value every downstream kernel reads to
# decide what work it owes this call. ``n_loop == history_count + 1`` drives
# the two-loop recursion; the negative values below are sentinels.
#
# Every system does a full step on every call. There is no "finished" value,
# because termination is the caller's, exactly as it is for FIRE2.
# -----------------------------------------------------------------------------
_NLOOP_PENDING = -4  # accepted; the (s, y) pair is written but not yet committed
_NLOOP_RESTART = -1  # first step or restart: take a steepest-descent direction

# Stands in for "no trust-region limit". Must be representable in float32 as
# well as float64, since the per-system scalars follow the coordinate dtype.
_BIG = 1.0e30

#: Default relative curvature threshold, chosen per coordinate precision.
#:
#: The guard is ``ys > eps * sqrt(ss * yy)``, i.e. a floor on the cosine
#: between ``s`` and ``y``. ``ys`` is accumulated at the coordinate precision,
#: so the floor has to sit above the level at which that sum is still signal.
#: Measured on deliberately orthogonal pairs (true ``ys`` exactly zero), the
#: computed cosine lands around ``1e-9`` to ``3.5e-8`` in float32 and ``1e-17``
#: in float64. A single default of ``1e-10`` is therefore comfortably above the
#: float64 floor but *below* the float32 one, where it cannot reject anything
#: on the strength of the sign it is testing.
#:
#: The float32 value sits one to three orders of magnitude above its floor and
#: still far below any usable curvature: a pair with ``cos(s, y) < 1e-6``
#: contributes a ``gamma = ys / yy`` so small that it carries no scale. Swept
#: over twelve Lennard-Jones relaxations, every value from ``1e-10`` to
#: ``1e-3`` gave identical evaluation counts, so this costs nothing on a
#: well-conditioned system; it matters where ``ys`` crosses zero.
_CURVATURE_EPS = {wp.float32: 1.0e-6, wp.float64: 1.0e-10}


# =============================================================================
# Device helpers
# =============================================================================


@wp.func
def _alpha_cap(a_lin: Any, b_quad: Any, maxstep: Any):
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

    A purely quadratic displacement is capped too. With ``a_lin == 0`` the
    expression above reduces to ``sqrt(maxstep / b_quad)``, which is the right
    bound, so the only case that may go uncapped is a step that does not move
    anything: **both** contributions zero. Returning early on ``a_lin == 0``
    alone would let a cell whose first-order displacement happens to cancel
    take an unbounded step.

    A non-positive ``maxstep`` disables the trust region. Generic in the
    scalar type, which follows the coordinate precision.
    """
    zero = type(a_lin)(0.0)
    if maxstep <= zero or (a_lin <= zero and b_quad <= zero):
        return type(a_lin)(_BIG)
    disc = a_lin * a_lin + type(a_lin)(4.0) * b_quad * maxstep
    return (type(a_lin)(2.0) * maxstep) / (a_lin + wp.sqrt(disc))


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
    batch_idx: wp.array(dtype=wp.int32),
    gg: wp.array(dtype=Any),
    n_dofs: wp.int32,
    elems_per_thread: wp.int32,
):
    """Reduce the packed force norm for one L-BFGS call.

    Computes, per system:

    ``gg``
        ``f . f`` over the packed degrees of freedom, used to normalize a
        steepest-descent direction. This lives in the space the direction lives
        in, which on a variable-cell path is *not* Cartesian space.

    Thread launch
    -------------
    One thread per ``elems_per_thread`` consecutive degrees of freedom;
    ``dim = ceil(n_dofs / elems_per_thread)``. Requires ``batch_idx`` sorted in
    non-decreasing order.

    Modifies
    --------
    gg
        OUTPUT. Accumulated atomically; the launcher zeroes it first.
    """
    tid = wp.tid()
    start = tid * elems_per_thread
    if start >= n_dofs:
        return
    stop = wp.min(start + elems_per_thread, n_dofs)

    zero = type(gg[0])(0.0)
    s_cur = batch_idx[start]
    acc_gg = zero

    for i in range(start, stop):
        s = batch_idx[i]
        if s != s_cur:
            wp.atomic_add(gg, s_cur, acc_gg)
            s_cur = s
            acc_gg = zero
        fi = forces[i]
        acc_gg += type(gg[0])(wp.dot(fi, fi))

    wp.atomic_add(gg, s_cur, acc_gg)


@wp.kernel(enable_backward=False)
def _lbfgs_step_decision_kernel(
    ys: wp.array(dtype=Any, ndim=2),
    yy: wp.array(dtype=Any, ndim=2),
    ss: wp.array(dtype=Any),
    iteration: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
):
    """Decide what the evaluation just supplied means.

    There is no line search, so there is nothing to reject: every evaluation is
    an accepted point. Each call therefore forms one secant pair and hands a
    direction downstream for one trust-region step. The step length is bounded
    by ``maxstep`` rather than chosen by comparing energies, so nothing here
    reads an energy at all.

    That matters for machine-learned potentials. A model with a direct force
    head does not return forces that are the gradient of its energy, and even a
    conservative model can have a rough energy surface. An Armijo test compares
    total energies while the search direction comes from forces; when the two
    disagree the test rejects good steps, and no tuning repairs it because the
    predicate is measuring the wrong surface.

    Nothing here tests convergence: the optimizer has no terminal state and
    every system steps on every call. The caller decides when to stop, as it
    does for FIRE2.

    This is the only kernel with per-system control flow, which is why it runs
    one thread per system rather than per atom.

    Thread launch
    -------------
    One thread per system; ``dim = num_systems``.

    Modifies
    --------
    iteration, n_loop
        Per-system control state.
    ys, yy, ss
        The candidate history slot is zeroed here, ready for the history
        kernels.
    """
    tid = wp.tid()

    # ---- first evaluation -------------------------------------------------
    if iteration[tid] < 0:
        # No previous accepted point, so there is no secant pair to form yet.
        iteration[tid] = 0
        n_loop[tid] = _NLOOP_RESTART
        return

    # ---- every later evaluation is an accepted step -----------------------
    iteration[tid] = iteration[tid] + 1
    n_loop[tid] = _NLOOP_PENDING

    slot = end[tid]
    ys[slot, tid] = type(ss[0])(0.0)
    yy[slot, tid] = type(ss[0])(0.0)
    ss[tid] = type(ss[0])(0.0)


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
    ys: wp.array(dtype=Any, ndim=2),
    yy: wp.array(dtype=Any, ndim=2),
    ss: wp.array(dtype=Any),
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

    zero = type(ss[0])(0.0)
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
            acc_sy += type(ss[0])(wp.dot(svec, yvec))
            acc_ss += type(ss[0])(wp.dot(svec, svec))
            acc_yy += type(ss[0])(wp.dot(yvec, yvec))

    if active:
        wp.atomic_add(ys, slot, s_cur, acc_sy)
        wp.atomic_add(ss, s_cur, acc_ss)
        wp.atomic_add(yy, slot, s_cur, acc_yy)


@wp.kernel(enable_backward=False)
def _lbfgs_history_commit_kernel(
    ys: wp.array(dtype=Any, ndim=2),
    yy: wp.array(dtype=Any, ndim=2),
    ss: wp.array(dtype=Any),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    m: wp.int32,
    curvature_eps: Any,
):
    """Keep or discard the candidate pair, and finalize the loop bound.

    A pair is kept only if its curvature ``s . y`` is comfortably positive.
    A line search would enforce this through the Wolfe conditions, but in
    single precision ``y = g - g_prev`` cancels badly near convergence and the
    scaling ``gamma = ys / yy`` can blow up.

    Discarding is not free: kernel 3 has already written the candidate into the
    slot ``end`` points at, and when the ring is full that slot held the oldest
    valid pair. Clamping ``history_count`` to ``m - 1`` removes exactly that
    now-destroyed entry from the range the recursion walks. When the ring is
    not yet full the slot had never been written, so the clamp does nothing.

    Thread launch
    -------------
    One thread per system; ``dim = num_systems``.

    Modifies
    --------
    end, n_loop, history_count
        Per-system control state.
    """
    tid = wp.tid()
    if n_loop[tid] != _NLOOP_PENDING:
        return

    slot = end[tid]

    sy = ys[slot, tid]
    threshold = curvature_eps * wp.sqrt(ss[tid] * yy[slot, tid])
    if sy > threshold:
        history_count[tid] = wp.min(history_count[tid] + 1, m)
        end[tid] = (slot + 1) % m
    else:
        history_count[tid] = wp.min(history_count[tid], m - 1)

    if history_count[tid] > 0:
        n_loop[tid] = history_count[tid] + 1
    else:
        n_loop[tid] = _NLOOP_RESTART


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
    ys: wp.array(dtype=Any, ndim=2),
    yy: wp.array(dtype=Any, ndim=2),
    alpha_hist: wp.array(dtype=Any, ndim=2),
    beta_hist: wp.array(dtype=Any, ndim=2),
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

    zero = type(ys[0, 0])(0.0)
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
    gamma = type(ys[0, 0])(1.0)
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
            gamma = type(ys[0, 0])(1.0)
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
                acc += type(ys[0, 0])(wp.dot(y_history[j_cur, i], qi))
            else:
                acc += type(ys[0, 0])(wp.dot(s_history[j_cur, i], qi))
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
    ys: wp.array(dtype=Any, ndim=2),
    alpha_hist: wp.array(dtype=Any, ndim=2),
    beta_hist: wp.array(dtype=Any, ndim=2),
    d0: wp.array(dtype=Any),
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

    zero = type(ys[0, 0])(0.0)
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
                acc -= type(ys[0, 0])(wp.dot(force_base[i], ri))
            else:
                acc += type(ys[0, 0])(wp.dot(y_history[j_next, i], ri))

    if active:
        if is_last:
            wp.atomic_add(d0, s_cur, acc)
        else:
            wp.atomic_add(beta_hist, j_next, s_cur, acc)


@wp.kernel(enable_backward=False)
def _lbfgs_restart_check_kernel(
    gg: wp.array(dtype=Any),
    d0: wp.array(dtype=Any),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
):
    """Demote a two-loop direction that does not descend, before it is used.

    ``d0 >= 0`` means the curvature model has gone bad and the recursion
    returned an ascent direction. This must run *after* the two-loop that
    produces ``d0`` and *before* the seed kernel that rebuilds ``direction``:
    demoting later would leave the rejected direction in place and the atoms
    would be moved along it, uphill, which is the opposite of the intent.

    Thread launch
    -------------
    One thread per system; ``dim = num_systems``.

    Modifies
    --------
    d0, end, n_loop, history_count
        Per-system control state.
    """
    tid = wp.tid()
    if n_loop[tid] > 0 and d0[tid] >= type(d0[0])(0.0):
        n_loop[tid] = _NLOOP_RESTART
        history_count[tid] = 0
        end[tid] = 0

    if n_loop[tid] == _NLOOP_RESTART:
        # For a normalized steepest-descent direction the slope is exact.
        d0[tid] = -wp.sqrt(gg[tid])


@wp.kernel(enable_backward=False)
def _lbfgs_seed_direction_kernel(
    forces: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    x_base: wp.array(dtype=Any),
    force_base: wp.array(dtype=Any),
    direction: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    gg: wp.array(dtype=Any),
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
    tid = wp.tid()
    s = batch_idx[tid]
    if n_loop[s] != _NLOOP_RESTART:
        return
    gn = wp.sqrt(gg[s])
    if gn > type(gg[0])(0.0):
        scale = type(forces[tid][0])(type(gg[0])(1.0) / gn)
        direction[tid] = scale * forces[tid]
    else:
        direction[tid] = type(forces[tid])()
    x_base[tid] = positions[tid]
    force_base[tid] = forces[tid]


@wp.kernel(enable_backward=False)
def _lbfgs_trust_region_kernel(
    direction: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dmax: wp.array(dtype=Any),
    dquad: wp.array(dtype=Any),
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

    zero = type(dmax[0])(0.0)
    s_cur = batch_idx[start]
    loc_max = zero

    for i in range(start, stop):
        s = batch_idx[i]
        if s != s_cur:
            wp.atomic_max(dmax, s_cur, loc_max)
            s_cur = s
            loc_max = zero
        loc_max = wp.max(loc_max, type(dmax[0])(wp.length(direction[i])))

    wp.atomic_max(dmax, s_cur, loc_max)


# =============================================================================
# Kernels 6 and 7: prepare and apply
# =============================================================================


@wp.kernel(enable_backward=False)
def _lbfgs_prepare_step_kernel(
    gg: wp.array(dtype=Any),
    d0: wp.array(dtype=Any),
    dmax: wp.array(dtype=Any),
    dquad: wp.array(dtype=Any),
    alpha_step: wp.array(dtype=Any),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    maxstep: Any,
):
    """Repair a bad direction and apply the trust region.

    Three things happen here, in order.

    First, the step length is reset to ``alpha = 1``. Nothing is carried
    between calls: the two-loop already scales the direction, so a unit step is
    the Newton step, and the trust region below is the only thing that shortens
    it. Starting from anything else would quietly degrade the method to a
    scaled gradient descent.

    Second, a direction that fails the descent test is replaced by steepest
    descent. Catching it here, where the direction was produced, guarantees the
    state machine never sees a non-descent slope.

    Third, the step is shortened so the resulting Cartesian displacement stays
    within ``maxstep``. This is the trust region, and it is the sole determinant
    of how far the atoms move.

    Thread launch
    -------------
    One thread per system; ``dim = num_systems``.

    Modifies
    --------
    alpha_step, d0, dmax, dquad, end, history_count, n_loop
        Per-system control state.
    """
    tid = wp.tid()
    # The step length is never carried between calls: the trust region below
    # determines it outright, starting from the full quasi-Newton step.
    alpha_step[tid] = type(alpha_step[0])(1.0)

    cap = _alpha_cap(dmax[tid], dquad[tid], maxstep)
    alpha_step[tid] = wp.min(alpha_step[tid], cap)


@wp.kernel(enable_backward=False)
def _lbfgs_apply_step_kernel(
    positions: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    x_base: wp.array(dtype=Any),
    force_base: wp.array(dtype=Any),
    direction: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    gg: wp.array(dtype=Any),
    alpha_step: wp.array(dtype=Any),
):
    """Move the positions to the next point, or finish the system off.

    Handles the terminal case as well as the ordinary step, so that whatever
    the caller reads back is always a geometry the optimizer stands behind,
    paired with matching forces in ``force_base``: a geometry that arrived
    already converged is left where it is, but its base buffers are seeded so
    they describe it.

    Thread launch
    -------------
    One thread per degree of freedom; ``dim = n_dofs``.

    Modifies
    --------
    positions
        Advanced to ``x_base + alpha * d``.
    direction, x_base, force_base
        Seeded when a steepest-descent direction is taken.
    """
    tid = wp.tid()
    s = batch_idx[tid]

    a = type(direction[tid][0])(alpha_step[s])
    positions[tid] = x_base[tid] + a * direction[tid]


# =============================================================================
# Kernel overloads
#
# Coordinates may be single or double precision; every per-system scalar is
# scalars follow them, so the overload key is just the vector dtype.
# =============================================================================

_VEC_TYPES = [wp.vec3f, wp.vec3d]

_reduce_overloads = {}
_seed_direction_overloads = {}
_trust_region_overloads = {}
_history_update_overloads = {}
_loop1_overloads = {}
_loop2_overloads = {}
_apply_step_overloads = {}
_step_decision_overloads = {}
_history_commit_overloads = {}
_restart_check_overloads = {}
_prepare_step_overloads = {}
_zero_d0_overloads = {}

_I32 = wp.int32

#: The per-system scalar that goes with each coordinate precision. Every
#: optimizer scalar follows the coordinates, so the overload key stays the
#: vector type alone and the instantiation count does not grow with it.
_SCALAR_FOR = {wp.vec3f: wp.float32, wp.vec3d: wp.float64}

for _v in _VEC_TYPES:
    _F64 = _SCALAR_FOR[_v]  # named for history; it is the coordinate scalar
    _reduce_overloads[_v] = wp.overload(
        _lbfgs_reduce_kernel,
        [
            wp.array(dtype=_v),  # forces
            wp.array(dtype=_I32),  # batch_idx
            wp.array(dtype=_F64),  # gg
            _I32,  # n_dofs
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
            wp.array(dtype=_I32),  # n_loop
            wp.array(dtype=_F64),  # gg
        ],
    )

    _trust_region_overloads[_v] = wp.overload(
        _lbfgs_trust_region_kernel,
        [
            wp.array(dtype=_v),  # direction
            wp.array(dtype=_I32),  # batch_idx
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
            wp.array(dtype=_I32),  # n_loop
            wp.array(dtype=_F64),  # gg
            wp.array(dtype=_F64),  # alpha_step
        ],
    )

    _step_decision_overloads[_F64] = wp.overload(
        _lbfgs_step_decision_kernel,
        [
            wp.array(dtype=_F64, ndim=2),  # ys
            wp.array(dtype=_F64, ndim=2),  # yy
            wp.array(dtype=_F64),  # ss
            wp.array(dtype=_I32),  # iteration
            wp.array(dtype=_I32),  # end
            wp.array(dtype=_I32),  # n_loop
        ],
    )

    _history_commit_overloads[_F64] = wp.overload(
        _lbfgs_history_commit_kernel,
        [
            wp.array(dtype=_F64, ndim=2),  # ys
            wp.array(dtype=_F64, ndim=2),  # yy
            wp.array(dtype=_F64),  # ss
            wp.array(dtype=_I32),  # end
            wp.array(dtype=_I32),  # n_loop
            wp.array(dtype=_I32),  # history_count
            _I32,  # m
            _F64,  # curvature_eps
        ],
    )

    _restart_check_overloads[_F64] = wp.overload(
        _lbfgs_restart_check_kernel,
        [
            wp.array(dtype=_F64),  # gg
            wp.array(dtype=_F64),  # d0
            wp.array(dtype=_I32),  # end
            wp.array(dtype=_I32),  # n_loop
            wp.array(dtype=_I32),  # history_count
        ],
    )

    _prepare_step_overloads[_F64] = wp.overload(
        _lbfgs_prepare_step_kernel,
        [
            wp.array(dtype=_F64),  # gg
            wp.array(dtype=_F64),  # d0
            wp.array(dtype=_F64),  # dmax
            wp.array(dtype=_F64),  # dquad
            wp.array(dtype=_F64),  # alpha_step
            wp.array(dtype=_I32),  # end
            wp.array(dtype=_I32),  # n_loop
            wp.array(dtype=_I32),  # history_count
            _F64,  # maxstep
        ],
    )


# =============================================================================
# Public API
# =============================================================================


def lbfgs_prepare_state(
    num_dofs: int,
    num_systems: int,
    *,
    dtype=wp.vec3d,
    history_size: int = 6,
    device=None,
) -> LBFGSState:
    """Allocate, initialize and validate a complete optimizer state.

    Call this once, before the first step. The two fields that do not start
    at zero are set for you -- ``alpha_step`` to one and ``iteration`` to minus
    one -- which is the part that is easy to get wrong by hand. Calling it
    again is how you reset.

    You are not obliged to use it: :class:`LBFGSState` is a plain dataclass, so
    you can build one from arrays you already own and call
    :meth:`LBFGSState.validate` yourself.

    Parameters
    ----------
    num_dofs : int
        Degrees of freedom the optimizer moves. On the variable-cell path this
        is ``num_atoms + 2 * num_systems``, not the atom count.
    num_systems : int
        Independent systems in the batch.
    dtype : optional
        Coordinate precision, ``wp.vec3f`` or ``wp.vec3d``. Every per-system
        scalar follows it, so ``wp.vec3f`` gives an end-to-end fp32 state and
        ``wp.vec3d`` an end-to-end fp64 one.
    history_size : int, optional
        Stored curvature pairs ``m``; 3 to 7 is the usual range. Memory is
        dominated by the two ``(m, num_dofs)`` history buffers.
    device : optional
        Warp device.

    Returns
    -------
    LBFGSState
    """
    if dtype not in _VEC_TYPES:
        raise ValueError(f"dtype must be wp.vec3f or wp.vec3d; got {dtype}")
    if history_size < 1:
        raise ValueError(f"history_size must be >= 1; got {history_size}")

    def vec(*shape):
        return wp.zeros(shape if len(shape) > 1 else shape[0], dtype=dtype,
                        device=device)  # fmt: skip

    scalar = _SCALAR_FOR[dtype]

    def sc(*shape):
        return wp.zeros(shape if len(shape) > 1 else shape[0], dtype=scalar,
                        device=device)  # fmt: skip

    def i32(n):
        return wp.zeros(n, dtype=wp.int32, device=device)

    state = LBFGSState(
        x_base=vec(num_dofs),
        force_base=vec(num_dofs),
        direction=vec(num_dofs),
        s_history=vec(history_size, num_dofs),
        y_history=vec(history_size, num_dofs),
        ys=sc(history_size, num_systems),
        yy=sc(history_size, num_systems),
        alpha_hist=sc(history_size, num_systems),
        beta_hist=sc(history_size, num_systems),
        ss=sc(num_systems),
        gg=sc(num_systems),
        d0=sc(num_systems),
        dmax=sc(num_systems),
        dquad=sc(num_systems),
        alpha_step=sc(num_systems),
        iteration=i32(num_systems),
        end=i32(num_systems),
        n_loop=i32(num_systems),
        history_count=i32(num_systems),
    )
    # The only two fields whose initial value is not zero.
    state.alpha_step.fill_(1.0)
    state.iteration.fill_(-1)
    state.validate()
    return state


def lbfgs_prepare_cell_state(
    num_atoms: int,
    num_systems: int,
    ext_batch_idx,
    ext_atom_ptr,
    *,
    cell=None,
    n_particles=None,
    cell_force_scale: float = 1.0,
    dtype=wp.vec3d,
    device=None,
) -> LBFGSCellState:
    """Allocate and validate the variable-cell chart and its scratch space.

    The packed topology is *yours*: pass ``ext_batch_idx`` and ``ext_atom_ptr``
    built with :func:`~nvalchemiops.dynamics.utils.cell_filter.extend_atom_ptr`
    and :func:`~nvalchemiops.batch_utils.atom_ptr_to_batch_idx`. Requiring them
    rather than deriving them is deliberate: deriving would mean assuming an
    even split, which is exactly what makes ragged batches inexpressible.

    Pass ``cell`` and ``n_particles`` to get a state that is ready to step:
    the chart fields ``ref_cell``, ``ref_cell_inv`` and ``kappa`` are filled
    for you. Omit them and those three are left zeroed, which a step cannot
    use -- a zeroed ``ref_cell`` is singular and a zeroed ``kappa`` is divided
    into the cell force -- so fill them with :func:`lbfgs_set_reference_cell`
    and :func:`lbfgs_cell_kappa` before the first step.

    Parameters
    ----------
    num_atoms : int
        Total atoms across the batch.
    num_systems : int
        Independent systems.
    ext_batch_idx, ext_atom_ptr : wp.array(dtype=int32)
        Packed topology, shapes ``(num_atoms + 2 * num_systems,)`` and
        ``(num_systems + 1,)``.
    cell : wp.array(dtype=mat33), optional
        Reference lattice per system. Given together with ``n_particles``, the
        chart is captured here instead of in a separate call.
    n_particles : wp.array(dtype=int32), optional
        Atom count per system, used for ``kappa``.
    cell_force_scale : float, optional
        Multiplier on the atom count in ``kappa``; only read when
        ``n_particles`` is given.
    dtype : optional
        Coordinate precision, ``wp.vec3f`` or ``wp.vec3d``.
    device : optional
        Warp device.

    Returns
    -------
    LBFGSCellState
    """
    if dtype not in _VEC_TYPES:
        raise ValueError(f"dtype must be wp.vec3f or wp.vec3d; got {dtype}")
    mat = _MAT_TYPES[dtype]
    scalar = wp.float32 if dtype == wp.vec3f else wp.float64
    num_packed = num_atoms + 2 * num_systems

    def mats(n):
        return wp.zeros(n, dtype=mat, device=device)

    def vec(n):
        return wp.zeros(n, dtype=dtype, device=device)

    state = LBFGSCellState(
        ref_cell=mats(num_systems),
        ref_cell_inv=mats(num_systems),
        kappa=wp.zeros(num_systems, dtype=scalar, device=device),
        ext_batch_idx=ext_batch_idx,
        ext_atom_ptr=ext_atom_ptr,
        phi=mats(num_systems),
        phi_inv=mats(num_systems),
        d_phi=mats(num_systems),
        cell_dof_a=vec(num_systems),
        cell_dof_b=vec(num_systems),
        cell_force_a=vec(num_systems),
        cell_force_b=vec(num_systems),
        ext_positions=vec(num_packed),
        ext_forces=vec(num_packed),
    )
    state.validate(num_atoms=num_atoms)
    if (cell is None) != (n_particles is None):
        raise ValueError("cell and n_particles must be given together")
    if cell is not None:
        lbfgs_set_reference_cell(cell, state.ref_cell, state.ref_cell_inv)
        lbfgs_cell_kappa(n_particles, state.kappa, cell_force_scale=cell_force_scale)
    return state


def _lbfgs_reduce_impl(
    forces: wp.array,
    direction: wp.array,
    batch_idx: wp.array,
    gg: wp.array,
) -> None:
    """Compute the per-system reduction for one L-BFGS call.

    Only ``gg``, the squared packed force norm, which normalizes a
    steepest-descent direction. It is taken over the **packed** arrays because
    that is the space the search direction lives in -- on a variable-cell path
    that is not Cartesian space.

    Nothing Cartesian is reduced here. Convergence is the caller's, so the
    quantities a tolerance would be applied to are the caller's to compute.

    Normally called for you by :func:`lbfgs_update`. Call it directly only if
    you want to supply the reduction yourself, via ``compute_reductions=False``.

    Parameters
    ----------
    forces : wp.array, shape (num_dofs,)
        Forces on the packed degrees of freedom.
    direction : wp.array, shape (num_dofs,)
        Current search direction.
    batch_idx : wp.array(dtype=int32), shape (num_dofs,)
        Sorted system index for each degree of freedom.
    gg : wp.array(dtype=float64), shape (num_systems,)
        OUTPUT. Zeroed internally before accumulation.
    """
    n_dofs = forces.shape[0]
    gg.zero_()
    if n_dofs == 0:
        return

    device = forces.device
    ept = compute_ept(n_dofs, max(device.sm_count, 1), True)
    wp.launch(
        _reduce_overloads[forces.dtype],
        dim=(n_dofs + ept - 1) // ept,
        inputs=[forces, batch_idx, gg, n_dofs, ept],
        device=device,
    )


def _resolve_curvature_eps(value, scalar_dtype) -> float:
    """Pick the curvature threshold, defaulting by coordinate precision.

    ``None`` means "choose for me"; see :data:`_CURVATURE_EPS` for why the
    default cannot be a single number across both precisions.
    """
    if value is not None:
        return float(value)
    try:
        return _CURVATURE_EPS[scalar_dtype]
    except KeyError:  # pragma: no cover - guarded by the overload registry
        raise ValueError(f"no curvature default for dtype {scalar_dtype}") from None


def _lbfgs_update_impl(
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
    gg: wp.array,
    d0: wp.array,
    dmax: wp.array,
    dquad: wp.array,
    alpha_step: wp.array,
    iteration: wp.array,
    end: wp.array,
    n_loop: wp.array,
    history_count: wp.array,
    *,
    maxstep: float = 0.2,
    curvature_eps: float | None = None,
    compute_reductions: bool = True,
    measure_trust_region: bool = True,
) -> None:
    """Advance the state machine and produce a search direction.

    Runs everything except the position update: the reduction, the state
    machine, the history update and its curvature test, and the two-loop
    recursion. There is no line search and nothing decides you have finished --
    convergence is the caller's. Use it together with
    :func:`lbfgs_prepare_step` and :func:`lbfgs_apply_step` when you need to
    interpose your own logic before the atoms move; :func:`lbfgs_step` chains
    all three.

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
    maxstep : float, optional
        Largest Cartesian displacement any atom may take in one step. Set to
        zero to disable the trust region.
    curvature_eps : float, optional
        Relative threshold below which a history pair is judged to carry no
        usable curvature and is discarded. Defaults to ``1e-6`` for float32
        coordinates and ``1e-10`` for float64: ``ys`` is accumulated at the
        coordinate precision, so one value cannot serve both.
    compute_reductions : bool, optional
        When ``False``, ``gg`` is taken as given rather than recomputed.
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
        return

    m = s_history.shape[0]
    if m < 1:
        raise ValueError(f"history size must be >= 1; got {m}")

    vec_dtype = positions.dtype
    device = positions.device
    num_systems = iteration.shape[0]
    ept = compute_ept(n_dofs, max(device.sm_count, 1), True)
    grid = (n_dofs + ept - 1) // ept

    if compute_reductions:
        _lbfgs_reduce_impl(forces, direction, batch_idx, gg)

    wp.launch(
        _step_decision_overloads[ss.dtype],
        dim=num_systems,
        inputs=[ys, yy, ss, iteration, end, n_loop],
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
        _history_commit_overloads[ss.dtype],
        dim=num_systems,
        inputs=[
            ys,
            yy,
            ss,
            end,
            n_loop,
            history_count,
            m,
            ss.dtype(_resolve_curvature_eps(curvature_eps, ss.dtype)),
        ],  # fmt: skip
        device=device,
    )

    # A restarting system needs its direction before the trust region can be
    # measured, so seed it here rather than in the apply kernel.

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

    # An ascent direction is demoted here -- after the two-loop that produces
    # `d0`, and before the seed kernel rebuilds `direction`. Demoting any later
    # would leave the rejected direction in place for the apply kernel.
    wp.launch(
        _restart_check_overloads[gg.dtype],
        dim=num_systems,
        inputs=[gg, d0, end, n_loop, history_count],
        device=device,
    )

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
            n_loop,
            gg,
        ],
        device=device,
    )

    if measure_trust_region:
        dmax.zero_()
        dquad.zero_()
        wp.launch(
            _trust_region_overloads[vec_dtype],
            dim=grid,
            inputs=[direction, batch_idx, dmax, dquad, n_dofs, ept],
            device=device,
        )


@wp.kernel(enable_backward=False)
def _lbfgs_zero_d0_kernel(
    d0: wp.array(dtype=Any),
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
    tid = wp.tid()
    if n_loop[tid] > 0:
        d0[tid] = type(d0[0])(0.0)


for _v in _VEC_TYPES:
    _zero_d0_overloads[_SCALAR_FOR[_v]] = wp.overload(
        _lbfgs_zero_d0_kernel,
        [
            wp.array(dtype=_SCALAR_FOR[_v]),  # d0
            wp.array(dtype=_I32),  # n_loop
        ],
    )


def _zero_pending_d0(d0, n_loop, num_systems, device) -> None:
    """Zero ``d0`` only where the second loop is about to accumulate into it."""
    wp.launch(
        _zero_d0_overloads[d0.dtype],
        dim=num_systems,
        inputs=[d0, n_loop],
        device=device,
    )


def _lbfgs_prepare_step_impl(
    gg: wp.array,
    d0: wp.array,
    dmax: wp.array,
    dquad: wp.array,
    alpha_step: wp.array,
    end: wp.array,
    n_loop: wp.array,
    history_count: wp.array,
    *,
    maxstep: float = 0.2,
) -> None:
    """Finalize the step length before the atoms move.

    Resets the step length, replaces
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
        _prepare_step_overloads[gg.dtype],
        dim=gg.shape[0],
        inputs=[
            gg,
            d0,
            dmax,
            dquad,
            alpha_step,
            end,
            n_loop,
            history_count,
            gg.dtype(maxstep),
        ],  # fmt: skip
        device=gg.device,
    )


def _lbfgs_apply_step_impl(
    positions: wp.array,
    forces: wp.array,
    x_base: wp.array,
    force_base: wp.array,
    direction: wp.array,
    batch_idx: wp.array,
    n_loop: wp.array,
    gg: wp.array,
    alpha_step: wp.array,
) -> None:
    """Move the positions to the next trial point.

    Every system moves: there is no terminal case, because the optimizer has
    no terminal state.

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
            n_loop,
            gg,
            alpha_step,
        ],
        device=positions.device,
    )


def _lbfgs_step_impl(
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
    gg: wp.array,
    d0: wp.array,
    dmax: wp.array,
    dquad: wp.array,
    alpha_step: wp.array,
    iteration: wp.array,
    end: wp.array,
    n_loop: wp.array,
    history_count: wp.array,
    *,
    maxstep: float = 0.2,
    curvature_eps: float | None = None,
    compute_reductions: bool = True,
) -> None:
    """Consume one energy/force evaluation and produce the next trial geometry.

    This is the entry point for the common case. Call it once per model
    evaluation. It never decides you are finished -- there is no terminal
    state and every system takes a step on every call, so testing convergence
    and stopping the loop are yours, exactly as they are for FIRE2.

    Equivalent to :func:`lbfgs_update`, :func:`lbfgs_prepare_step` and
    :func:`lbfgs_apply_step` in sequence.

    Parameters
    ----------
    See :func:`lbfgs_update`; the arguments are identical.

    Examples
    --------
    >>> for _ in range(max_steps):
    ...     forces = model(positions)
    ...     if np.linalg.norm(forces.numpy(), axis=1).max() < force_tol:
    ...         break
    ...     lbfgs_step(positions, forces, state, batch_idx, maxstep=0.2)

    """
    _lbfgs_update_impl(
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
        gg=gg,
        d0=d0,
        dmax=dmax,
        dquad=dquad,
        alpha_step=alpha_step,
        iteration=iteration,
        end=end,
        n_loop=n_loop,
        history_count=history_count,
        maxstep=maxstep,
        curvature_eps=curvature_eps,
        compute_reductions=compute_reductions,
    )
    if positions.shape[0] == 0:
        return
    _lbfgs_prepare_step_impl(
        gg=gg,
        d0=d0,
        dmax=dmax,
        dquad=dquad,
        alpha_step=alpha_step,
        end=end,
        n_loop=n_loop,
        history_count=history_count,
        maxstep=maxstep,
    )
    _lbfgs_apply_step_impl(
        positions=positions,
        forces=forces,
        x_base=x_base,
        force_base=force_base,
        direction=direction,
        batch_idx=batch_idx,
        n_loop=n_loop,
        gg=gg,
        alpha_step=alpha_step,
    )


# =============================================================================
# Variable-cell support
#
# The chart is ASE's UnitCellFilter. Concatenating raw positions with raw cell
# rows would be wrong: the stress-derived cell force is conjugate to an affine
# deformation the atoms ride along with, so the (s, y) pairs would mix spaces.
# With H0 captured once and lattice vectors as columns (r = H s):
#
#     Phi = H H0^-1          deformation gradient, identity at the start
#     u   = Phi^-1 r         atom coordinates in the reference frame
#     c   = kappa * Phi      cell coordinates, the six aligned components
#     f_u = Phi^T F
#     f_c = -(V sigma) Phi^-T / kappa
#
# Scaling the cell coordinate by kappa and dividing its force by the same
# factor keeps g . dx chart-independent, which is what makes the packed pairs
# genuine secant pairs. The two-loop recursion needs no changes.
# =============================================================================


@wp.kernel(enable_backward=False)
def _lbfgs_cell_kappa_kernel(
    n_atoms_per_system: wp.array(dtype=wp.int32),
    cell_force_scale: Any,
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
    tid = wp.tid()
    slot = kappa[tid]
    # A system with no atoms still owns two cell degrees of freedom, and kappa
    # is divided into the cell force and the unpacked cell, so it must stay
    # strictly positive or those become infinite. With no atoms there is
    # nothing to balance the cell against, which makes the scale free; clamping
    # the count to one keeps it continuous with a single-atom system.
    count = wp.max(n_atoms_per_system[tid], wp.int32(1))
    kappa[tid] = type(slot)(cell_force_scale) * type(slot)(count)


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
    tid = wp.tid()
    h = cell[tid]
    k = kappa[tid]
    p = h * ref_cell_inv[tid]
    phi[tid] = p
    p_inv = wp.inverse(p)
    phi_inv[tid] = p_inv

    dof_a = cell_dof_a[tid]
    dof_b = cell_dof_b[tid]
    dof_a[0] = k * p[0, 0]
    dof_a[1] = k * p[1, 0]
    dof_a[2] = k * p[2, 0]
    dof_b[0] = k * p[1, 1]
    dof_b[1] = k * p[2, 1]
    dof_b[2] = k * p[2, 2]
    cell_dof_a[tid] = dof_a
    cell_dof_b[tid] = dof_b

    f_a = cell_force_a[tid] - cell_force_a[tid]
    f_b = cell_force_b[tid] - cell_force_b[tid]
    if have_stress:
        # f_c = -(V sigma) Phi^-T / kappa
        volume = wp.abs(wp.determinant(h))
        force = (-volume / k) * (stress[tid] * wp.transpose(p_inv))
        f_a[0] = force[0, 0]
        f_a[1] = force[1, 0]
        f_a[2] = force[2, 0]
        f_b[0] = force[1, 1]
        f_b[1] = force[2, 1]
        f_b[2] = force[2, 2]
    cell_force_a[tid] = f_a
    cell_force_b[tid] = f_b


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
    tid = wp.tid()
    s = ext_batch_idx[tid]
    cell_start = ext_atom_ptr[s + 1] - wp.int32(2)
    if tid >= cell_start:
        if tid == cell_start:
            ext_positions[tid] = cell_dof_a[s]
            ext_forces[tid] = cell_force_a[s]
        else:
            ext_positions[tid] = cell_dof_b[s]
            ext_forces[tid] = cell_force_b[s]
        return
    atom_i = tid - wp.int32(2) * s
    ext_positions[tid] = phi_inv[s] * positions[atom_i]
    ext_forces[tid] = wp.transpose(phi[s]) * forces[atom_i]


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

    Only six components are stored -- the same six FIRE2 packs -- so the cell
    keeps its aligned form by construction and cannot drift into a rotation.

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
    tid = wp.tid()
    h0 = ref_cell[tid]
    k = kappa[tid]
    cell_start = ext_atom_ptr[tid + 1] - wp.int32(2)
    va = ext_positions[cell_start] / k
    vb = ext_positions[cell_start + 1] / k
    p = h0 - h0  # a zero matrix at the right precision
    p[0, 0] = va[0]
    p[1, 0] = va[1]
    p[2, 0] = va[2]
    p[1, 1] = vb[0]
    p[2, 1] = vb[1]
    p[2, 2] = vb[2]
    phi[tid] = p
    cell[tid] = p * h0


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
    atom_i = wp.tid()
    s = batch_idx[atom_i]
    positions[atom_i] = phi[s] * ext_positions[atom_i + 2 * s]


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
    tid = wp.tid()
    k = kappa[tid]
    cell_start = ext_atom_ptr[tid + 1] - wp.int32(2)
    va = direction[cell_start] / k
    vb = direction[cell_start + 1] / k
    d = d_phi[tid] - d_phi[tid]  # a zero matrix at the right precision
    d[0, 0] = va[0]
    d[1, 0] = va[1]
    d[2, 0] = va[2]
    d[1, 1] = vb[0]
    d[2, 1] = vb[1]
    d[2, 2] = vb[2]
    d_phi[tid] = d


@wp.kernel(enable_backward=False)
def _lbfgs_cell_trust_region_kernel(
    ext_positions: wp.array(dtype=Any),
    direction: wp.array(dtype=Any),
    phi: wp.array(dtype=Any),
    d_phi: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dmax: wp.array(dtype=Any),
    dquad: wp.array(dtype=Any),
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
    atom_i = wp.tid()
    s = batch_idx[atom_i]
    e = atom_i + 2 * s
    d_u = direction[e]
    u = ext_positions[e]
    linear = phi[s] * d_u + d_phi[s] * u
    quadratic = d_phi[s] * d_u
    wp.atomic_max(dmax, s, type(dmax[0])(wp.length(linear)))
    wp.atomic_max(dquad, s, type(dmax[0])(wp.length(quadratic)))


_MAT_TYPES = {wp.vec3f: wp.mat33f, wp.vec3d: wp.mat33d}

_cell_kappa_overloads = {}
_cell_chart_overloads = {}
_pack_overloads = {}
_unpack_cell_overloads = {}
_unpack_atoms_overloads = {}
_cell_direction_overloads = {}
_cell_trust_region_overloads = {}

_SCALAR_OF = _SCALAR_FOR

for _v, _mt in _MAT_TYPES.items():
    _sc = _SCALAR_OF[_v]
    _cell_kappa_overloads[_v] = wp.overload(
        _lbfgs_cell_kappa_kernel,
        [
            wp.array(dtype=_I32),  # n_atoms_per_system
            _sc,  # cell_force_scale
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
            wp.array(dtype=_sc),  # dmax
            wp.array(dtype=_sc),  # dquad
        ],
    )


def check_cell_is_aligned(cell, atol: float = 1e-10) -> None:
    """Confirm the cell is in the aligned form the packing assumes.

    The six-component cell parameterization represents only the entries that
    are non-zero once
    :func:`~nvalchemiops.dynamics.utils.cell_filter.align_cell` has run; in an
    unaligned frame the omitted entries are not the redundant ones, and the
    cell keeps a rotation the optimizer cannot remove. See
    :ref:`the variable-cell contract <lbfgs-cell-contract>`.

    This reads the cell back to the host, so it is a setup-time check. It is
    called for you by :func:`lbfgs_prepare_cell_state` and by
    :func:`lbfgs_set_reference_cell`, both of which run once.

    Parameters
    ----------
    cell : array(dtype=mat33), shape (num_systems,)
        Cell matrices, lattice vectors in columns.
    atol : float, optional
        Absolute tolerance on the entries that must be zero.

    Raises
    ------
    ValueError
        If any system's cell is not aligned.
    """
    import numpy as _np

    values = _np.asarray(cell.numpy())
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError(
            f"cell must have shape (num_systems, 3, 3); got {values.shape}"
        )
    # Strictly above the diagonal: zero once the cell is aligned.
    offenders = _np.abs(_np.triu(values, 1)).max(axis=(1, 2))
    bad = _np.flatnonzero(offenders > atol)
    if bad.size:
        raise ValueError(
            f"cell for system(s) {bad.tolist()} is not aligned: the entries "
            f"above the diagonal reach {offenders[bad].max():.3e}, not zero. "
            "Call nvalchemiops.dynamics.utils.cell_filter.align_cell(positions, "
            "cell, transform) once before the first step, as fire2_step_coord_cell "
            "also requires; see the variable-cell contract in this module."
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
        Current cell, lattice vectors as columns, already aligned by
        :func:`~nvalchemiops.dynamics.utils.cell_filter.align_cell`. Checked
        here, since this runs once.
    ref_cell, ref_cell_inv : wp.array(dtype=mat33), shape (num_systems,)
        OUTPUT. ``H0`` and its inverse.

    See Also
    --------
    :ref:`The variable-cell contract <lbfgs-cell-contract>` : why the reference
        is fixed, and what calling this a second time costs.
    """
    check_cell_is_aligned(cell)
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

    Notes
    -----
    ``kappa`` is divided into the cell force, so it must never be zero. A
    system with no atoms is treated as having one, since with nothing to
    balance the cell against the scale is arbitrary anyway. If you fill
    ``kappa`` yourself rather than calling this, keep every entry strictly
    positive.
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
            dmax,
            dquad,
        ],
        device=device,
    )


def _lbfgs_step_coord_cell_impl(
    positions: wp.array,
    forces: wp.array,
    cell: wp.array,
    stress: wp.array,
    batch_idx: wp.array,
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
    gg: wp.array,
    d0: wp.array,
    dmax: wp.array,
    dquad: wp.array,
    alpha_step: wp.array,
    iteration: wp.array,
    end: wp.array,
    n_loop: wp.array,
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
    maxstep: float = 0.2,
    curvature_eps: float | None = None,
) -> None:
    """Advance one variable-cell step, relaxing coordinates and cell together.

    Maps the geometry into the packed chart, takes one L-BFGS step there, and
    maps the result back. Because both blocks live in one coordinate vector,
    the two-loop recursion couples them without any special handling.

    Consumes exactly one force/stress evaluation, like the coordinate-only
    :func:`lbfgs_step`, and like it has no terminal state: deciding when the
    forces and stress are small enough is yours.

    Align the cell before the first step and keep the reference fixed for the
    whole relaxation; build ``ext_batch_idx`` / ``ext_atom_ptr`` yourself so
    ragged batches stay expressible. All of that is stated once in
    :ref:`the variable-cell contract <lbfgs-cell-contract>`, which this
    function follows rather than restates.

    Parameters
    ----------
    positions : wp.array, shape (num_atoms,)
        Cartesian geometry, advanced in place.
    forces : wp.array, shape (num_atoms,)
        Cartesian forces at ``positions``. Forces, not gradients.
    cell : wp.array(dtype=mat33), shape (num_systems,)
        Cell with lattice vectors as columns, advanced in place. Must be
        aligned first; see :ref:`the variable-cell contract <lbfgs-cell-contract>`.
    stress : wp.array(dtype=mat33), shape (num_systems,)
        Cauchy stress per system, which drives the cell degrees of freedom.
    batch_idx : wp.array(dtype=int32), shape (num_atoms,)
        Sorted system index per atom.
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
    _lbfgs_update_impl(
        positions=ext_positions,
        forces=ext_forces,
        batch_idx=ext_batch_idx,
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
        gg=gg,
        d0=d0,
        dmax=dmax,
        dquad=dquad,
        alpha_step=alpha_step,
        iteration=iteration,
        end=end,
        n_loop=n_loop,
        history_count=history_count,
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
        dmax,
        dquad,
    )
    _lbfgs_prepare_step_impl(
        gg=gg,
        d0=d0,
        dmax=dmax,
        dquad=dquad,
        alpha_step=alpha_step,
        end=end,
        n_loop=n_loop,
        history_count=history_count,
        maxstep=maxstep,
    )
    _lbfgs_apply_step_impl(
        positions=ext_positions,
        forces=ext_forces,
        x_base=x_base,
        force_base=force_base,
        direction=direction,
        batch_idx=ext_batch_idx,
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


# =============================================================================
# Public API: the phases, driven by the state objects
# =============================================================================


def _arrays(state) -> dict:
    """The dataclass's arrays keyed by field name, without copying anything.

    Passed through by keyword rather than position: the internal launchers
    interleave the buffers with the topology arrays, so a positional splat
    would silently misalign them.
    """
    return {f.name: getattr(state, f.name) for f in dataclasses.fields(state)}


def _check_inputs(positions, forces, batch_idx, state: LBFGSState) -> None:
    """Confirm this call's inputs match the state that was prepared.

    Shapes only -- no device reads, so this costs nothing next to the launches
    that follow. The state re-checks itself too: it is a plain dataclass whose
    fields can be reassigned between steps, and a history buffer that no longer
    matches would read out of bounds inside a kernel rather than raise.
    """
    state.validate()
    if positions.shape[0] != state.num_dofs:
        raise ValueError(
            f"positions has {positions.shape[0]} degrees of freedom but the "
            f"state was prepared for {state.num_dofs}"
        )
    if forces.shape[0] != positions.shape[0]:
        raise ValueError(
            f"forces has {forces.shape[0]} entries, positions has {positions.shape[0]}"
        )
    if batch_idx.shape[0] != positions.shape[0]:
        raise ValueError(
            f"batch_idx has {batch_idx.shape[0]} entries, positions has "
            f"{positions.shape[0]}"
        )


def _check_cell_inputs(positions, forces, cell, stress, batch_idx, state, cell_state):
    """Confirm this call's inputs match both prepared states.

    The variable-cell counterpart of :func:`_check_inputs`, and as thorough:
    the entry point is public, both states are plain dataclasses whose fields
    can be reassigned between steps, and the kernels here index a *packed*
    array built from three separate topologies. Shapes and dtypes only -- no
    device reads -- so it costs nothing next to the launches that follow.

    This does not reuse :func:`_check_inputs`: there ``positions`` spans the
    whole degree-of-freedom vector, whereas here it holds atoms only and the
    state is sized for ``num_atoms + 2 * num_systems``.
    """
    state.validate()
    cell_state.validate(num_atoms=positions.shape[0])
    num_systems = state.num_systems

    if state.num_dofs != cell_state.num_packed_dofs:
        raise ValueError(
            f"state is sized for {state.num_dofs} degrees of freedom but the "
            f"cell state is packed for {cell_state.num_packed_dofs}"
        )
    expected = positions.shape[0] + 2 * num_systems
    if expected != state.num_dofs:
        raise ValueError(
            f"{positions.shape[0]} atoms in {num_systems} systems needs "
            f"{expected} degrees of freedom, but the state has {state.num_dofs}"
        )
    if forces.shape[0] != positions.shape[0]:
        raise ValueError(
            f"forces has {forces.shape[0]} entries, positions has {positions.shape[0]}"
        )
    if batch_idx.shape[0] != positions.shape[0]:
        raise ValueError(
            f"batch_idx has {batch_idx.shape[0]} entries, positions has "
            f"{positions.shape[0]}"
        )
    for name, array in (("cell", cell), ("stress", stress)):
        if array.shape[0] != num_systems:
            raise ValueError(
                f"{name} has {array.shape[0]} entries, but there are "
                f"{num_systems} systems"
            )
    # The packed arrays are written from the Cartesian ones, so a precision
    # mismatch would surface as a kernel launch failure rather than as this.
    if forces.dtype != positions.dtype:
        raise ValueError(
            f"forces dtype {forces.dtype} != positions dtype {positions.dtype}"
        )
    if cell.dtype != stress.dtype:
        raise ValueError(f"cell dtype {cell.dtype} != stress dtype {stress.dtype}")
    devices = {str(a.device) for a in (positions, forces, cell, stress, batch_idx)}
    if len(devices) > 1:
        raise ValueError(f"inputs are spread across devices {sorted(devices)}")


def lbfgs_reduce(forces, state: LBFGSState, batch_idx) -> None:
    """Compute the per-system reduction for one call. See :func:`lbfgs_step`."""
    _lbfgs_reduce_impl(forces, state.direction, batch_idx, state.gg)


def lbfgs_update(positions, forces, state: LBFGSState, batch_idx,
                 **kwargs) -> None:  # fmt: skip
    """Advance the state machine and produce a search direction.

    Everything except the position update. Use with :func:`lbfgs_prepare_step`
    and :func:`lbfgs_apply_step` when you need to interpose your own logic
    before the atoms move; :func:`lbfgs_step` chains all three.
    """
    _check_inputs(positions, forces, batch_idx, state)
    _lbfgs_update_impl(
        positions=positions, forces=forces, batch_idx=batch_idx,
        **_arrays(state), **kwargs,
    )  # fmt: skip


def lbfgs_prepare_step(state: LBFGSState, **kwargs) -> None:
    """Repair a bad direction and apply the trust region."""
    _lbfgs_prepare_step_impl(
        state.gg, state.d0, state.dmax, state.dquad, state.alpha_step,
        state.end, state.n_loop, state.history_count, **kwargs,
    )  # fmt: skip


def lbfgs_apply_step(positions, forces, state: LBFGSState, batch_idx) -> None:
    """Move the positions to the next point."""
    _lbfgs_apply_step_impl(
        positions, forces, state.x_base, state.force_base, state.direction,
        batch_idx, state.n_loop, state.gg, state.alpha_step,
    )  # fmt: skip


def lbfgs_step(positions, forces, state: LBFGSState, batch_idx,
               **kwargs) -> None:  # fmt: skip
    """Consume one force evaluation and produce the next geometry.

    One history update, one algorithmic restart if the direction stops
    descending, and one ``maxstep``-bounded step. It never stops: testing
    convergence and ending the loop are yours, as they are for FIRE2.

    Parameters
    ----------
    positions, forces : array, shape (num_dofs,)
        Current geometry and the forces there. Mutated in place.
    state : LBFGSState
        From :func:`lbfgs_prepare_state`, or built from your own arrays.
    batch_idx : array, shape (num_dofs,), dtype int32
        Sorted system index per degree of freedom.
    **kwargs
        ``maxstep``, ``curvature_eps`` and ``compute_reductions``.
    """
    _check_inputs(positions, forces, batch_idx, state)
    _lbfgs_step_impl(
        positions=positions, forces=forces, batch_idx=batch_idx,
        **_arrays(state), **kwargs,
    )  # fmt: skip


def lbfgs_step_coord_cell(positions, forces, cell, stress, state: LBFGSState,
                          cell_state: LBFGSCellState, batch_idx,
                          **kwargs) -> None:  # fmt: skip
    """Advance one variable-cell step, relaxing coordinates and cell together.

    ``state`` must be sized for ``num_atoms + 2 * num_systems`` degrees of
    freedom; ``cell_state`` carries the chart. Call
    :func:`lbfgs_set_reference_cell` and :func:`lbfgs_cell_kappa` once before
    the first step.
    """
    _check_cell_inputs(positions, forces, cell, stress, batch_idx, state, cell_state)
    _lbfgs_step_coord_cell_impl(
        positions=positions, forces=forces, cell=cell, stress=stress,
        batch_idx=batch_idx,
        **_arrays(state), **_arrays(cell_state), **kwargs,
    )  # fmt: skip
