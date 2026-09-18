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

"""JAX bindings for the batched L-BFGS geometry optimizer.

L-BFGS reaches a given force tolerance in far fewer force evaluations
than the FIRE optimizers, which is the cost that dominates relaxation with a
machine-learned potential.

Every optimizer buffer is yours: this module allocates nothing and hides
nothing. You pass the same 26 arrays to every step and get 26 back, in the
same order. JAX arrays are immutable, so unlike the PyTorch binding these
entry points **return** new arrays rather than writing them in place::

    import functools, jax, jax.numpy as jnp
    from nvalchemiops.jax.lbfgs import LBFGS_NEED_EVAL, lbfgs_step_coord

    m, P, M = 6, num_atoms, num_systems
    dt = jnp.float64
    buffers = [
        jnp.zeros((P, 3), dt), jnp.zeros((P, 3), dt), jnp.zeros((P, 3), dt),
        jnp.zeros((m, P, 3), dt), jnp.zeros((m, P, 3), dt),
        *[jnp.zeros((m, M), jnp.float64) for _ in range(4)],
        *[jnp.zeros((M,), jnp.float64) for _ in range(10)],
        jnp.ones((M,), jnp.float64),                  # alpha_step starts at 1
        jnp.zeros((M,), jnp.int32),                   # status = LBFGS_NEED_EVAL
        jnp.full((M,), -1, jnp.int32),                # iteration = -1
        *[jnp.zeros((M,), jnp.int32) for _ in range(4)],
    ]

    @functools.partial(jax.jit, donate_argnums=tuple(range(len(buffers) + 1)))
    def relax_step(positions, *buffers, forces=None):
        return lbfgs_step_coord(
            positions, forces, batch_idx, n_particles, *buffers,
        )

    while True:
        forces = model(positions)
        positions, *buffers = relax_step(positions, *buffers, forces=forces)
        if bool(lbfgs_converged(buffers[20])):
            break

Required buffer contents
------------------------
The optimizer never initializes anything, so the first step reads whatever you
hand it. Zero every buffer, then set exactly three things:

=================  ========================================================
``alpha_step``     ``1.0`` -- the line-search step length for a new direction
``iteration``      ``-1`` -- the "never evaluated" marker
``status``         ``LBFGS_NEED_EVAL``, which is numerically zero
=================  ========================================================

Shapes, with ``P`` degrees of freedom, ``M`` systems and history size ``m``:
``x_base``, ``force_base`` and ``direction`` are ``(P, 3)``; ``s_history`` and
``y_history`` are ``(m, P, 3)``; ``ys``, ``yy``, ``alpha_hist`` and
``beta_hist`` are ``(m, M)`` float64; ``ss`` through ``alpha_step`` are
``(M,)`` float64; ``status`` through ``history_count`` are ``(M,)`` int32.
The float64 scalars stay float64 whatever precision the coordinates use,
because the ``ys / yy`` ratio that scales the initial inverse Hessian is a
ratio of differences of nearly equal vectors near convergence.

To restart a relaxation, rebuild the buffers in that initial state. There is
no separate reset: in a functional setting resetting and allocating are the
same operation.

Positions and forces stay consistent
------------------------------------
``forces`` is an input: it is read, never aliased or returned. Positions only
move *forward* from the point the forces were evaluated at -- nothing is ever
rolled back -- so the ``forces`` you passed in still describe the ``positions``
you get back. ``force_base`` holds the same forces alongside the matching
``x_base`` if you would rather read them from the returned buffers.

Donation and pointer stability
------------------------------
Every mutable array is declared as an input-output alias, so XLA may reuse each
input buffer for the matching output. Two things follow, and both matter for
performance rather than correctness:

- **Donate the buffers.** ``jax.jit(donate_argnums=...)`` over ``positions``
  and all 26 buffers lets them round-trip at stable addresses. Without donation
  JAX copies, which doubles peak memory and moves the pointers.
- **Keep the topology out of the donated set.** ``batch_idx`` and
  ``n_particles`` never change, so close over them rather than donating them.

Under ``JaxCallableGraphMode.WARP`` the step is captured and replayed as a CUDA graph. The
capture is keyed on the input buffer addresses, so a fresh ``forces`` array each
step produces a small working set of graphs rather than one; measurements on a
six-atom system settle at four or five captures and stay there, well inside the
default cache. If you see the count grow without bound, pass
``graph_mode="warp_staged"``, which keys the capture on the call instead and
patches the changing buffers in, at the cost of one copy per staged array.

Scalars are baked into the compiled call, so changing ``force_tol`` or
``maxstep`` between steps triggers a recompilation. Hold them fixed for the
duration of a relaxation.

These operations are **not differentiable**. ``jax.grad`` through a step will
fail rather than return a silently wrong answer. Callers that relax and then
differentiate the relaxed energy should stop the gradient at the relaxed
positions themselves.

See Also
--------
nvalchemiops.dynamics.optimizers.lbfgs : the underlying Warp implementation,
    which documents the algorithm, the sign convention and the precision policy.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import warp as wp
from warp import JaxCallableGraphMode, jax_callable

from nvalchemiops.dynamics.optimizers.lbfgs import (
    _CELL_BUFFERS,
    _CELL_SCRATCH,
    _OPTIMIZER_BUFFERS,
    LBFGS_CONVERGED,
    LBFGS_NEED_EVAL,
)
from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_step as _warp_step
from nvalchemiops.dynamics.optimizers.lbfgs import (
    lbfgs_step_coord_cell as _warp_step_cell,
)

__all__ = [
    "LBFGS_CONVERGED",
    "LBFGS_NEED_EVAL",
    "lbfgs_cell_kappa",
    "lbfgs_converged",
    "lbfgs_set_reference_cell",
    "lbfgs_step_coord",
    "lbfgs_step_coord_cell",
]

#: The mutable arrays, in the order the callable takes them, which is also the
#: order :func:`lbfgs_step_coord` returns them in.
_LBFGS_IN_OUT_ARGS: tuple[str, ...] = ("positions",) + _OPTIMIZER_BUFFERS

_GRAPH_MODES = {
    "none": JaxCallableGraphMode.NONE,
    "warp": JaxCallableGraphMode.WARP,
    "warp_staged": JaxCallableGraphMode.WARP_STAGED,
}


def _lbfgs_body_f32(
    forces: wp.array(dtype=wp.vec3f),
    batch_idx: wp.array(dtype=wp.int32),
    n_particles: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    x_base: wp.array(dtype=wp.vec3f),
    force_base: wp.array(dtype=wp.vec3f),
    direction: wp.array(dtype=wp.vec3f),
    s_history: wp.array(dtype=wp.vec3f, ndim=2),
    y_history: wp.array(dtype=wp.vec3f, ndim=2),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    alpha_hist: wp.array(dtype=wp.float64, ndim=2),
    beta_hist: wp.array(dtype=wp.float64, ndim=2),
    ss: wp.array(dtype=wp.float64),
    gg: wp.array(dtype=wp.float64),
    fmax: wp.array(dtype=wp.float64),
    frms_sq: wp.array(dtype=wp.float64),
    smax: wp.array(dtype=wp.float64),
    d0: wp.array(dtype=wp.float64),
    dmax: wp.array(dtype=wp.float64),
    dquad: wp.array(dtype=wp.float64),
    alpha_step: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    iteration: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    force_tol: wp.float64,
    rms_tol: wp.float64,
    stress_tol: wp.float64,
    maxstep: wp.float64,
    curvature_eps: wp.float64,
) -> None:
    """Advance one L-BFGS step on f32 coordinates.

    Every mutable array is listed before the static scalars and named in
    ``_LBFGS_IN_OUT_ARGS``, so XLA aliases each one to the matching output and
    the step can be written as if it mutated in place.
    """
    _warp_step(
        positions=positions,
        forces=forces,
        batch_idx=batch_idx,
        n_particles=n_particles,
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
        history_count=history_count,
        force_tol=force_tol,
        rms_tol=rms_tol,
        stress_tol=stress_tol,
        maxstep=maxstep,
        curvature_eps=curvature_eps,
    )


def _lbfgs_body_f64(
    forces: wp.array(dtype=wp.vec3d),
    batch_idx: wp.array(dtype=wp.int32),
    n_particles: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3d),
    x_base: wp.array(dtype=wp.vec3d),
    force_base: wp.array(dtype=wp.vec3d),
    direction: wp.array(dtype=wp.vec3d),
    s_history: wp.array(dtype=wp.vec3d, ndim=2),
    y_history: wp.array(dtype=wp.vec3d, ndim=2),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    alpha_hist: wp.array(dtype=wp.float64, ndim=2),
    beta_hist: wp.array(dtype=wp.float64, ndim=2),
    ss: wp.array(dtype=wp.float64),
    gg: wp.array(dtype=wp.float64),
    fmax: wp.array(dtype=wp.float64),
    frms_sq: wp.array(dtype=wp.float64),
    smax: wp.array(dtype=wp.float64),
    d0: wp.array(dtype=wp.float64),
    dmax: wp.array(dtype=wp.float64),
    dquad: wp.array(dtype=wp.float64),
    alpha_step: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    iteration: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    force_tol: wp.float64,
    rms_tol: wp.float64,
    stress_tol: wp.float64,
    maxstep: wp.float64,
    curvature_eps: wp.float64,
) -> None:
    """Advance one L-BFGS step on f64 coordinates.

    Every mutable array is listed before the static scalars and named in
    ``_LBFGS_IN_OUT_ARGS``, so XLA aliases each one to the matching output and
    the step can be written as if it mutated in place.
    """
    _warp_step(
        positions=positions,
        forces=forces,
        batch_idx=batch_idx,
        n_particles=n_particles,
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
        history_count=history_count,
        force_tol=force_tol,
        rms_tol=rms_tol,
        stress_tol=stress_tol,
        maxstep=maxstep,
        curvature_eps=curvature_eps,
    )


_BODIES = {jnp.float32: _lbfgs_body_f32, jnp.float64: _lbfgs_body_f64}

# The bodies are written out by hand, so pin their parameter names to the
# canonical buffer order. A reordering would silently swap two arrays, and
# neither warp nor XLA would notice.
# forces, batch_idx, n_particles, positions
_STATE_SLICE = slice(4, 4 + len(_OPTIMIZER_BUFFERS))
for _name, _body in (("f32", _lbfgs_body_f32), ("f64", _lbfgs_body_f64)):
    _params = tuple(inspect.signature(_body).parameters)[_STATE_SLICE]
    if _params != _OPTIMIZER_BUFFERS:
        raise RuntimeError(
            f"_OPTIMIZER_BUFFERS and _lbfgs_body_{_name} parameters have diverged:\n"
            f"  expected: {_OPTIMIZER_BUFFERS}\n"
            f"  body:     {_params}"
        )

_CALLABLES: dict[tuple, object] = {}


def _get_callable(dtype, graph_mode: str):
    """Return the registered callable for a dtype, building it on first use.

    Registration is deferred so that importing this module does not compile
    anything or touch the GPU.
    """
    key = (jnp.dtype(dtype).type, graph_mode)
    if key not in _CALLABLES:
        body = _BODIES.get(key[0])
        if body is None:
            raise ValueError(
                f"positions must be float32 or float64; got {jnp.dtype(dtype)}"
            )
        if graph_mode not in _GRAPH_MODES:
            raise ValueError(
                f"graph_mode must be one of {sorted(_GRAPH_MODES)}; got {graph_mode!r}"
            )
        kwargs = {}
        if graph_mode == "warp_staged":
            # Stage only the arrays that change identity every step; staging the
            # history buffers would copy megabytes per call for no benefit.
            kwargs["stage_in_argnames"] = ["forces"]
        _CALLABLES[key] = jax_callable(
            body,
            num_outputs=len(_LBFGS_IN_OUT_ARGS),
            in_out_argnames=list(_LBFGS_IN_OUT_ARGS),
            graph_mode=_GRAPH_MODES[graph_mode],
            **kwargs,
        )
    return _CALLABLES[key]


def lbfgs_step_coord(
    positions: jax.Array,
    forces: jax.Array,
    batch_idx: jax.Array,
    n_particles: jax.Array,
    x_base: jax.Array,
    force_base: jax.Array,
    direction: jax.Array,
    s_history: jax.Array,
    y_history: jax.Array,
    ys: jax.Array,
    yy: jax.Array,
    alpha_hist: jax.Array,
    beta_hist: jax.Array,
    ss: jax.Array,
    gg: jax.Array,
    fmax: jax.Array,
    frms_sq: jax.Array,
    smax: jax.Array,
    d0: jax.Array,
    dmax: jax.Array,
    dquad: jax.Array,
    alpha_step: jax.Array,
    status: jax.Array,
    iteration: jax.Array,
    end: jax.Array,
    n_loop: jax.Array,
    history_count: jax.Array,
    *,
    force_tol: float = 0.05,
    rms_tol: float = 0.0,
    stress_tol: float = 0.0,
    maxstep: float = 0.2,
    curvature_eps: float = 1e-10,
    graph_mode: str = "warp",
) -> tuple[jax.Array, ...]:
    """Advance one batched L-BFGS step, consuming one force evaluation.

    Returns new arrays; nothing is mutated in place. Donate ``positions`` and
    every buffer so XLA can reuse their memory.

    Parameters
    ----------
    positions : jax.Array, shape (num_atoms, 3)
        Current geometry.
    forces : jax.Array, shape (num_atoms, 3)
        Forces at ``positions``. Forces, not gradients.
    batch_idx : jax.Array, shape (num_atoms,), dtype int32
        Sorted system index per atom. Static topology, so close over it.
    n_particles : jax.Array, shape (num_systems,), dtype int32
        Atom count per system, for the optional RMS criterion.
    x_base, ..., history_count : jax.Array
        The 26 caller-owned optimizer buffers, in the fixed order documented in
        the module docstring. They must be passed positionally.
    force_tol : float, optional
        Threshold on the largest per-atom force magnitude. Zero disables it.
    rms_tol, stress_tol : float, optional
        Additional criteria, disabled by default; all enabled ones must hold.
    maxstep : float, optional
        Largest distance an atom may move in one step. Zero disables the trust
        region.
    graph_mode : {"warp", "warp_staged", "none"}, optional
        How the step is captured. ``"warp"`` replays a CUDA graph and is the
        default; ``"warp_staged"`` keys the capture on the call rather than on
        buffer addresses, which helps if the capture count grows without bound;
        ``"none"`` disables capture.

    Returns
    -------
    tuple of jax.Array
        27 arrays: the advanced ``positions`` followed by the 26 buffers, in
        exactly the order they were passed in.

    Raises
    ------
    ValueError
        If dtypes or shapes are inconsistent.
    """
    _validate(positions, forces, batch_idx, n_particles, s_history, status)
    call = _get_callable(positions.dtype, graph_mode)
    return tuple(
        call(
            forces,
            batch_idx,
            n_particles,
            positions,
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
            gg,
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
            history_count,
            float(force_tol),
            float(rms_tol),
            float(stress_tol),
            float(maxstep),
            float(curvature_eps),
        )
    )


def lbfgs_converged(status: jax.Array) -> jax.Array:
    """Whether every system has finished, as a device-side boolean.

    Reading this back costs a host synchronization, so a caller that wants to
    amortize it can check every few steps instead of every step.

    Parameters
    ----------
    status : jax.Array, shape (num_systems,), dtype int32
        The ``status`` buffer returned by the last step.
    """
    return jnp.all(status != LBFGS_NEED_EVAL)


def _validate(positions, forces, batch_idx, n_particles, s_history, status) -> None:
    """Check the shapes and dtypes that would otherwise fail inside the FFI."""
    num_dofs = positions.shape[0]
    if jnp.dtype(positions.dtype).type not in _BODIES:
        raise ValueError(f"positions must be float32 or float64; got {positions.dtype}")
    if forces.shape != positions.shape:
        raise ValueError(
            f"forces shape {forces.shape} != positions shape {positions.shape}"
        )
    if forces.dtype != positions.dtype:
        raise ValueError(
            f"forces dtype {forces.dtype} != positions dtype {positions.dtype}"
        )
    if batch_idx.shape[0] != num_dofs:
        raise ValueError(
            f"batch_idx length {batch_idx.shape[0]} != positions length {num_dofs}"
        )
    num_systems = status.shape[0]
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


def _lbfgs_cell_body_f32(
    forces: wp.array(dtype=wp.vec3f),
    stress: wp.array(dtype=wp.mat33f),
    batch_idx: wp.array(dtype=wp.int32),
    n_particles: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    x_base: wp.array(dtype=wp.vec3f),
    force_base: wp.array(dtype=wp.vec3f),
    direction: wp.array(dtype=wp.vec3f),
    s_history: wp.array(dtype=wp.vec3f, ndim=2),
    y_history: wp.array(dtype=wp.vec3f, ndim=2),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    alpha_hist: wp.array(dtype=wp.float64, ndim=2),
    beta_hist: wp.array(dtype=wp.float64, ndim=2),
    ss: wp.array(dtype=wp.float64),
    gg: wp.array(dtype=wp.float64),
    fmax: wp.array(dtype=wp.float64),
    frms_sq: wp.array(dtype=wp.float64),
    smax: wp.array(dtype=wp.float64),
    d0: wp.array(dtype=wp.float64),
    dmax: wp.array(dtype=wp.float64),
    dquad: wp.array(dtype=wp.float64),
    alpha_step: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    iteration: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    ref_cell: wp.array(dtype=wp.mat33f),
    ref_cell_inv: wp.array(dtype=wp.mat33f),
    kappa: wp.array(dtype=wp.float32),
    ext_batch_idx: wp.array(dtype=wp.int32),
    ext_atom_ptr: wp.array(dtype=wp.int32),
    phi: wp.array(dtype=wp.mat33f),
    phi_inv: wp.array(dtype=wp.mat33f),
    d_phi: wp.array(dtype=wp.mat33f),
    cell_dof_a: wp.array(dtype=wp.vec3f),
    cell_dof_b: wp.array(dtype=wp.vec3f),
    cell_force_a: wp.array(dtype=wp.vec3f),
    cell_force_b: wp.array(dtype=wp.vec3f),
    ext_positions: wp.array(dtype=wp.vec3f),
    ext_forces: wp.array(dtype=wp.vec3f),
    force_tol: wp.float64,
    rms_tol: wp.float64,
    stress_tol: wp.float64,
    maxstep: wp.float64,
    curvature_eps: wp.float64,
) -> None:
    """Advance one variable-cell L-BFGS step on f32 coordinates.

    The first five cell arrays are read-only configuration and topology, so
    they are plain inputs; everything else is aliased through
    ``_CELL_IN_OUT_ARGS``.
    """
    _warp_step_cell(
        positions=positions,
        forces=forces,
        cell=cell,
        stress=stress,
        batch_idx=batch_idx,
        n_particles=n_particles,
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
        history_count=history_count,
        ref_cell=ref_cell,
        ref_cell_inv=ref_cell_inv,
        kappa=kappa,
        ext_batch_idx=ext_batch_idx,
        ext_atom_ptr=ext_atom_ptr,
        phi=phi,
        phi_inv=phi_inv,
        d_phi=d_phi,
        cell_dof_a=cell_dof_a,
        cell_dof_b=cell_dof_b,
        cell_force_a=cell_force_a,
        cell_force_b=cell_force_b,
        ext_positions=ext_positions,
        ext_forces=ext_forces,
        force_tol=force_tol,
        rms_tol=rms_tol,
        stress_tol=stress_tol,
        maxstep=maxstep,
        curvature_eps=curvature_eps,
    )


def _lbfgs_cell_body_f64(
    forces: wp.array(dtype=wp.vec3d),
    stress: wp.array(dtype=wp.mat33d),
    batch_idx: wp.array(dtype=wp.int32),
    n_particles: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    x_base: wp.array(dtype=wp.vec3d),
    force_base: wp.array(dtype=wp.vec3d),
    direction: wp.array(dtype=wp.vec3d),
    s_history: wp.array(dtype=wp.vec3d, ndim=2),
    y_history: wp.array(dtype=wp.vec3d, ndim=2),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    alpha_hist: wp.array(dtype=wp.float64, ndim=2),
    beta_hist: wp.array(dtype=wp.float64, ndim=2),
    ss: wp.array(dtype=wp.float64),
    gg: wp.array(dtype=wp.float64),
    fmax: wp.array(dtype=wp.float64),
    frms_sq: wp.array(dtype=wp.float64),
    smax: wp.array(dtype=wp.float64),
    d0: wp.array(dtype=wp.float64),
    dmax: wp.array(dtype=wp.float64),
    dquad: wp.array(dtype=wp.float64),
    alpha_step: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    iteration: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    ref_cell: wp.array(dtype=wp.mat33d),
    ref_cell_inv: wp.array(dtype=wp.mat33d),
    kappa: wp.array(dtype=wp.float64),
    ext_batch_idx: wp.array(dtype=wp.int32),
    ext_atom_ptr: wp.array(dtype=wp.int32),
    phi: wp.array(dtype=wp.mat33d),
    phi_inv: wp.array(dtype=wp.mat33d),
    d_phi: wp.array(dtype=wp.mat33d),
    cell_dof_a: wp.array(dtype=wp.vec3d),
    cell_dof_b: wp.array(dtype=wp.vec3d),
    cell_force_a: wp.array(dtype=wp.vec3d),
    cell_force_b: wp.array(dtype=wp.vec3d),
    ext_positions: wp.array(dtype=wp.vec3d),
    ext_forces: wp.array(dtype=wp.vec3d),
    force_tol: wp.float64,
    rms_tol: wp.float64,
    stress_tol: wp.float64,
    maxstep: wp.float64,
    curvature_eps: wp.float64,
) -> None:
    """Advance one variable-cell L-BFGS step on f64 coordinates.

    The first five cell arrays are read-only configuration and topology, so
    they are plain inputs; everything else is aliased through
    ``_CELL_IN_OUT_ARGS``.
    """
    _warp_step_cell(
        positions=positions,
        forces=forces,
        cell=cell,
        stress=stress,
        batch_idx=batch_idx,
        n_particles=n_particles,
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
        history_count=history_count,
        ref_cell=ref_cell,
        ref_cell_inv=ref_cell_inv,
        kappa=kappa,
        ext_batch_idx=ext_batch_idx,
        ext_atom_ptr=ext_atom_ptr,
        phi=phi,
        phi_inv=phi_inv,
        d_phi=d_phi,
        cell_dof_a=cell_dof_a,
        cell_dof_b=cell_dof_b,
        cell_force_a=cell_force_a,
        cell_force_b=cell_force_b,
        ext_positions=ext_positions,
        ext_forces=ext_forces,
        force_tol=force_tol,
        rms_tol=rms_tol,
        stress_tol=stress_tol,
        maxstep=maxstep,
        curvature_eps=curvature_eps,
    )


#: Everything the variable-cell step writes, in return order. ``ref_cell``,
#: ``ref_cell_inv``, ``kappa``, ``ext_batch_idx`` and ``ext_atom_ptr`` are
#: read-only configuration and topology, so they are not aliased and not
#: returned.
_CELL_IN_OUT_ARGS: tuple[str, ...] = (
    ("positions", "cell") + _OPTIMIZER_BUFFERS + _CELL_SCRATCH
)

_CELL_BODIES = {jnp.float32: _lbfgs_cell_body_f32, jnp.float64: _lbfgs_cell_body_f64}

for _name, _body in (("f32", _lbfgs_cell_body_f32), ("f64", _lbfgs_cell_body_f64)):
    _p = tuple(inspect.signature(_body).parameters)
    _n = len(_OPTIMIZER_BUFFERS)
    if _p[6 : 6 + _n] != _OPTIMIZER_BUFFERS:
        raise RuntimeError(
            f"_OPTIMIZER_BUFFERS and _lbfgs_cell_body_{_name} have diverged"
        )
    if _p[6 + _n : 6 + _n + len(_CELL_BUFFERS)] != _CELL_BUFFERS:
        raise RuntimeError(f"_CELL_BUFFERS and _lbfgs_cell_body_{_name} have diverged")

_CELL_CALLABLES: dict[tuple, object] = {}


def _get_cell_callable(dtype, graph_mode: str):
    """Return the registered variable-cell callable, building it on first use."""
    key = (jnp.dtype(dtype).type, graph_mode)
    if key not in _CELL_CALLABLES:
        body = _CELL_BODIES.get(key[0])
        if body is None:
            raise ValueError(
                f"positions must be float32 or float64; got {jnp.dtype(dtype)}"
            )
        if graph_mode not in _GRAPH_MODES:
            raise ValueError(
                f"graph_mode must be one of {sorted(_GRAPH_MODES)}; got {graph_mode!r}"
            )
        kwargs = {}
        if graph_mode == "warp_staged":
            kwargs["stage_in_argnames"] = ["forces", "stress"]
        _CELL_CALLABLES[key] = jax_callable(
            body,
            num_outputs=len(_CELL_IN_OUT_ARGS),
            in_out_argnames=list(_CELL_IN_OUT_ARGS),
            graph_mode=_GRAPH_MODES[graph_mode],
            **kwargs,
        )
    return _CELL_CALLABLES[key]


def lbfgs_set_reference_cell(cell: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Capture the reference cell that defines the variable-cell chart.

    Coordinates are measured relative to a cell held fixed for the whole
    relaxation, which is what makes history pairs from different iterations
    comparable. Call this once before the first step. Calling it again
    re-references the chart and invalidates every stored curvature pair, so
    rebuild the optimizer buffers in their initial state if you do.

    The reference is returned as an independent copy. Sharing a buffer with the
    caller's live ``cell`` would make it impossible to donate both to the same
    jitted step, because XLA rejects the same buffer being donated twice.

    Parameters
    ----------
    cell : jax.Array, shape (num_systems, 3, 3)
        Current cell, lattice vectors as columns.

    Returns
    -------
    ref_cell, ref_cell_inv : jax.Array, shape (num_systems, 3, 3)
        Pass these to every :func:`lbfgs_step_coord_cell` call. They are
        read-only, so do not donate them.
    """
    return jnp.array(cell, copy=True), jnp.linalg.inv(cell)


def lbfgs_cell_kappa(
    n_particles: jax.Array,
    *,
    cell_force_scale: float = 1.0,
    dtype=None,
) -> jax.Array:
    """Return the per-system cell coordinate scaling.

    The cell coordinate is ``kappa * Phi`` and its conjugate force is divided
    by the same ``kappa``, which is what keeps ``g . dx`` independent of the
    scaling. Depends only on topology, so compute it once and reuse it.

    Parameters
    ----------
    n_particles : jax.Array, shape (num_systems,), dtype int32
        Atom count per system.
    cell_force_scale : float, optional
        Multiplier on the atom count. Larger values make the cell move less per
        step relative to the atoms; ``1 / atoms_per_system`` puts the two on a
        comparable footing.
    dtype : optional
        Result precision, defaulting to the coordinate precision in use.
        ``kappa`` scales matrices, so it must match the **coordinates**, not
        the float64 per-system scalars.

    Returns
    -------
    jax.Array, shape (num_systems,)
    """
    if cell_force_scale <= 0.0:
        raise ValueError(f"cell_force_scale must be positive; got {cell_force_scale}")
    dtype = jnp.float64 if dtype is None else jnp.dtype(dtype).type
    if dtype not in _CELL_BODIES:
        raise ValueError(f"dtype must be float32 or float64; got {dtype}")
    return jnp.asarray(n_particles, dtype) * dtype(cell_force_scale)


def lbfgs_step_coord_cell(
    positions: jax.Array,
    cell: jax.Array,
    forces: jax.Array,
    stress: jax.Array,
    batch_idx: jax.Array,
    n_particles: jax.Array,
    x_base: jax.Array,
    force_base: jax.Array,
    direction: jax.Array,
    s_history: jax.Array,
    y_history: jax.Array,
    ys: jax.Array,
    yy: jax.Array,
    alpha_hist: jax.Array,
    beta_hist: jax.Array,
    ss: jax.Array,
    gg: jax.Array,
    fmax: jax.Array,
    frms_sq: jax.Array,
    smax: jax.Array,
    d0: jax.Array,
    dmax: jax.Array,
    dquad: jax.Array,
    alpha_step: jax.Array,
    status: jax.Array,
    iteration: jax.Array,
    end: jax.Array,
    n_loop: jax.Array,
    history_count: jax.Array,
    ref_cell: jax.Array,
    ref_cell_inv: jax.Array,
    kappa: jax.Array,
    ext_batch_idx: jax.Array,
    ext_atom_ptr: jax.Array,
    phi: jax.Array,
    phi_inv: jax.Array,
    d_phi: jax.Array,
    cell_dof_a: jax.Array,
    cell_dof_b: jax.Array,
    cell_force_a: jax.Array,
    cell_force_b: jax.Array,
    ext_positions: jax.Array,
    ext_forces: jax.Array,
    *,
    force_tol: float = 0.05,
    rms_tol: float = 0.0,
    stress_tol: float = 0.0,
    maxstep: float = 0.2,
    curvature_eps: float = 1e-10,
    graph_mode: str = "warp",
) -> tuple[jax.Array, ...]:
    """Advance one variable-cell step, relaxing coordinates and cell together.

    Returns new arrays; nothing is mutated. Donate everything that comes back
    so XLA can reuse the memory.

    The optimizer buffers must be sized for ``num_atoms + 2 * num_systems``
    degrees of freedom, since the cell contributes two entries per system.

    Call :func:`lbfgs_set_reference_cell` and :func:`lbfgs_cell_kappa` once
    before the first step, and build ``ext_batch_idx`` / ``ext_atom_ptr``
    yourself -- with :func:`nvalchemiops.dynamics.utils.cell_filter.extend_atom_ptr`
    and :func:`nvalchemiops.dynamics.utils.batch_utils.atom_ptr_to_batch_idx` --
    so ragged batches are expressible.

    Parameters
    ----------
    cell : jax.Array, shape (num_systems, 3, 3)
        Lattice vectors as columns. Kept lower-triangular, so the cell cannot
        drift into a rotation.
    stress : jax.Array, shape (num_systems, 3, 3)
        Cauchy stress. Drives the cell degrees of freedom, and is what
        ``stress_tol`` is compared against.
    x_base, ..., history_count : jax.Array
        The 26 caller-owned optimizer buffers, in the same order as
        :func:`lbfgs_step_coord`.
    ref_cell, ..., ext_forces : jax.Array
        The 14 caller-owned cell buffers. The first five are read-only, so do
        not donate them and do not expect them back.
    force_tol, rms_tol, stress_tol : float, optional
        Convergence thresholds, always evaluated on the Cartesian forces and
        the stress rather than on packed norms.

    Returns
    -------
    tuple of jax.Array
        37 arrays: ``positions``, ``cell``, the 26 optimizer buffers, then
        ``phi``, ``phi_inv``, ``d_phi``, ``cell_dof_a``, ``cell_dof_b``,
        ``cell_force_a``, ``cell_force_b``, ``ext_positions``, ``ext_forces``.

    See Also
    --------
    lbfgs_set_reference_cell : must be called first.
    lbfgs_cell_kappa : must be called first.
    lbfgs_step_coord : the coordinate-only equivalent.
    """
    _validate_cell(positions, forces, cell, stress, batch_idx, s_history, status)
    call = _get_cell_callable(positions.dtype, graph_mode)
    return tuple(
        call(
            forces,
            stress,
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
            gg,
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
            float(force_tol),
            float(rms_tol),
            float(stress_tol),
            float(maxstep),
            float(curvature_eps),
        )
    )


def _validate_cell(
    positions, forces, cell, stress, batch_idx, s_history, status
) -> None:
    """Check the variable-cell shapes that would otherwise fail inside the FFI.

    The degree-of-freedom check differs from the coordinate path, so this does
    not reuse :func:`_validate`: the buffers are sized for the packed array,
    not for the atoms.
    """
    num_atoms = positions.shape[0]
    num_systems = status.shape[0]
    if jnp.dtype(positions.dtype).type not in _CELL_BODIES:
        raise ValueError(f"positions must be float32 or float64; got {positions.dtype}")
    if forces.shape != positions.shape:
        raise ValueError(
            f"forces shape {forces.shape} != positions shape {positions.shape}"
        )
    if forces.dtype != positions.dtype:
        raise ValueError(
            f"forces dtype {forces.dtype} != positions dtype {positions.dtype}"
        )
    if batch_idx.shape[0] != num_atoms:
        raise ValueError(
            f"batch_idx length {batch_idx.shape[0]} != positions length {num_atoms}"
        )
    if cell.shape != (num_systems, 3, 3):
        raise ValueError(
            f"cell must have shape ({num_systems}, 3, 3); got {cell.shape}"
        )
    if stress.shape != cell.shape:
        raise ValueError(f"stress shape {stress.shape} != cell shape {cell.shape}")
    if cell.dtype != positions.dtype or stress.dtype != positions.dtype:
        raise ValueError("cell and stress must share the dtype of positions")
    expected = positions.shape[0] + 2 * num_systems
    if s_history.shape[1] != expected:
        raise ValueError(
            f"optimizer buffers are sized for {s_history.shape[1]} degrees of "
            f"freedom, but the variable-cell path needs {expected} "
            f"(num_atoms + 2 * num_systems)"
        )
