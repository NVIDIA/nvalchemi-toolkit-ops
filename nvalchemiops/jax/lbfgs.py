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

L-BFGS reaches a given force tolerance in far fewer force evaluations than the
FIRE optimizers -- the cost that dominates relaxation with a machine-learned
potential.

JAX arrays are immutable, so unlike the PyTorch binding these entry points
**return** a new state. Both
:class:`~nvalchemiops.dynamics.optimizers.lbfgs.LBFGSState` and
:class:`~nvalchemiops.dynamics.optimizers.lbfgs.LBFGSCellState` are registered
pytrees, so a state crosses ``jax.jit`` as one argument and one
``donate_argnums`` entry donates every field::

    state = lbfgs_prepare_state(num_atoms, num_systems)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def relax_step(positions, state, forces):
        return lbfgs_step_coord(positions, forces, state, batch_idx)

    for _ in range(max_steps):
        forces = model(positions)
        if float(jnp.linalg.norm(forces, axis=1).max()) < force_tol:
            break
        positions, state = relax_step(positions, state, forces)

Test before stepping, as above: the returned ``positions`` are a new,
unevaluated point, and the forces you passed describe the point before it
(kept in ``x_base``/``force_base``). The optimizer owns no tolerance and has
no terminal status, matching FIRE2. Reading a norm back costs a host
synchronization, so test every few steps if that matters.

These operations are **not differentiable**: ``jax.grad`` through a step fails
rather than returning a silently wrong answer.

State
-----
:func:`lbfgs_prepare_state` allocates, initializes and validates the whole
state; in a functional setting resetting and allocating are the same thing, so
calling it again is how you restart. The arrays stay yours -- the state is a
plain dataclass with a ``validate()`` method; see
:mod:`nvalchemiops.dynamics.optimizers.lbfgs` for shapes and required initial
contents.

Every array follows the coordinate dtype, so an fp32 state is fp32 end to end
and **needs no** ``JAX_ENABLE_X64``.

Donation and pointer stability
------------------------------
Every mutable array is an input-output alias, so XLA may reuse each input
buffer for the matching output. Both consequences are about performance:

- **Donate the state.** Without donation JAX copies, doubling peak memory and
  moving the pointers. On the variable-cell path donate ``state`` but not
  ``cell_state``: its five chart fields are read-only and come back unchanged,
  so XLA cannot reuse their buffers.
- **Keep topology out of the donated set.** ``batch_idx`` never changes, so
  close over it.

Under ``JaxCallableGraphMode.WARP`` the step replays as a CUDA graph. The
capture is keyed on input addresses, so a fresh ``forces`` array each step
gives a small working set rather than one graph per call; measurements settle
at four or five. If the count grows without bound, pass
``graph_mode="warp_staged"``, which keys on the call instead at the cost of
one copy per staged array.

Scalars are baked into the compiled call, so changing ``maxstep`` between
steps triggers a recompilation.

See Also
--------
nvalchemiops.dynamics.optimizers.lbfgs : the Warp implementation, which
    documents the algorithm, sign convention and precision policy.
"""

from __future__ import annotations

import dataclasses
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from warp import JaxCallableGraphMode, jax_callable

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
    _lbfgs_step_coord_cell_impl as _warp_step_cell,
)
from nvalchemiops.dynamics.optimizers.lbfgs import _lbfgs_step_impl as _warp_step

__all__ = [
    "LBFGSCellState",
    "LBFGSState",
    "lbfgs_prepare_cell_state",
    "lbfgs_prepare_state",
    "lbfgs_step_coord",
    "lbfgs_step_coord_cell",
]

# The states carry nothing but arrays, so every field is a pytree leaf and
# there is no static metadata. Registering them here rather than in the Core
# keeps the Core free of a JAX import.
for _cls in (LBFGSState, LBFGSCellState):
    jax.tree_util.register_dataclass(
        _cls, data_fields=[f.name for f in dataclasses.fields(_cls)], meta_fields=[]
    )

#: The mutable arrays, in the order the callable takes them, which is also the
#: order :func:`lbfgs_step_coord` returns them in.
_LBFGS_IN_OUT_ARGS: tuple[str, ...] = ("positions",) + _OPTIMIZER_BUFFERS

#: JAX coordinate dtype -> the Warp scalar the curvature default is keyed on.
_WP_SCALAR = {jnp.float32: wp.float32, jnp.float64: wp.float64}

_GRAPH_MODES = {
    "none": JaxCallableGraphMode.NONE,
    "warp": JaxCallableGraphMode.WARP,
    "warp_staged": JaxCallableGraphMode.WARP_STAGED,
}


def _lbfgs_body_f32(
    forces: wp.array(dtype=wp.vec3f),
    batch_idx: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    x_base: wp.array(dtype=wp.vec3f),
    force_base: wp.array(dtype=wp.vec3f),
    direction: wp.array(dtype=wp.vec3f),
    s_history: wp.array(dtype=wp.vec3f, ndim=2),
    y_history: wp.array(dtype=wp.vec3f, ndim=2),
    ys: wp.array(dtype=wp.float32, ndim=2),
    yy: wp.array(dtype=wp.float32, ndim=2),
    alpha_hist: wp.array(dtype=wp.float32, ndim=2),
    beta_hist: wp.array(dtype=wp.float32, ndim=2),
    ss: wp.array(dtype=wp.float32),
    gg: wp.array(dtype=wp.float32),
    d0: wp.array(dtype=wp.float32),
    dmax: wp.array(dtype=wp.float32),
    dquad: wp.array(dtype=wp.float32),
    alpha_step: wp.array(dtype=wp.float32),
    iteration: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    maxstep: wp.float32,
    curvature_eps: wp.float32,
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


def _lbfgs_body_f64(
    forces: wp.array(dtype=wp.vec3d),
    batch_idx: wp.array(dtype=wp.int32),
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
    d0: wp.array(dtype=wp.float64),
    dmax: wp.array(dtype=wp.float64),
    dquad: wp.array(dtype=wp.float64),
    alpha_step: wp.array(dtype=wp.float64),
    iteration: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
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


_BODIES = {jnp.float32: _lbfgs_body_f32, jnp.float64: _lbfgs_body_f64}

# The bodies are written out by hand, so pin their parameter names to the
# canonical buffer order. A reordering would silently swap two arrays, and
# neither warp nor XLA would notice.
# forces, batch_idx, positions
_STATE_SLICE = slice(3, 3 + len(_OPTIMIZER_BUFFERS))
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


def lbfgs_prepare_state(
    num_dofs: int,
    num_systems: int,
    *,
    dtype=jnp.float64,
    history_size: int = 6,
) -> LBFGSState:
    """Allocate, initialize and validate a complete optimizer state.

    Call this once, before the first step. The three fields that do not start
    at zero are set for you -- ``alpha_step`` to one, ``iteration`` to minus
    one. Calling it again is how you reset.

    :class:`LBFGSState` is a registered pytree, so the whole thing can be
    passed through ``jax.jit``, donated, and carried by ``lax.while_loop``.
    You are not obliged to use this function: build one from arrays you own
    and call :meth:`LBFGSState.validate` yourself.

    Parameters
    ----------
    num_dofs : int
        Degrees of freedom the optimizer moves. On the variable-cell path this
        is ``num_atoms + 2 * num_systems``, not the atom count.
    num_systems : int
        Independent systems in the batch.
    dtype : optional
        Coordinate precision, float32 or float64. Every per-system scalar
        follows it, so this selects an end-to-end fp32 or fp64 state -- and an
        fp32 state needs no ``JAX_ENABLE_X64``.
    history_size : int, optional
        Stored curvature pairs ``m``; 3 to 7 is the usual range.

    Returns
    -------
    LBFGSState
    """
    if jnp.dtype(dtype).type not in _BODIES:
        raise ValueError(f"dtype must be float32 or float64; got {jnp.dtype(dtype)}")
    if history_size < 1:
        raise ValueError(f"history_size must be >= 1; got {history_size}")
    m, n = history_size, num_systems
    sc = jnp.dtype(dtype)
    z64 = lambda *s: jnp.zeros(s, sc)  # noqa: E731
    state = LBFGSState(
        x_base=jnp.zeros((num_dofs, 3), dtype),
        force_base=jnp.zeros((num_dofs, 3), dtype),
        direction=jnp.zeros((num_dofs, 3), dtype),
        s_history=jnp.zeros((num_dofs, m, 3), dtype),
        y_history=jnp.zeros((num_dofs, m, 3), dtype),
        ys=z64(n, m), yy=z64(n, m), alpha_hist=z64(n, m), beta_hist=z64(n, m),
        ss=z64(n), gg=z64(n),
        d0=z64(n), dmax=z64(n), dquad=z64(n),
        # The only three fields whose initial value is not zero.
        alpha_step=jnp.ones(n, sc),
        iteration=jnp.full(n, -1, jnp.int32),
        end=jnp.zeros(n, jnp.int32),
        n_loop=jnp.zeros(n, jnp.int32),
        history_count=jnp.zeros(n, jnp.int32),
    )  # fmt: skip
    state.validate()
    return state


def lbfgs_step_coord(
    positions: jax.Array,
    forces: jax.Array,
    state: LBFGSState,
    batch_idx: jax.Array,
    *,
    maxstep: float = 0.2,
    curvature_eps: float | None = None,
    graph_mode: str = "warp",
) -> tuple[jax.Array, LBFGSState]:
    """Advance one batched L-BFGS step, consuming one force evaluation.

    Returns new arrays; nothing is mutated in place. Donate ``positions`` and
    ``state`` so XLA can reuse their memory.

    Parameters
    ----------
    positions : jax.Array, shape (num_atoms, 3)
        Current geometry.
    forces : jax.Array, shape (num_atoms, 3)
        Forces at ``positions``. Forces, not gradients.
    state : LBFGSState
        From :func:`lbfgs_prepare_state`, or built from your own arrays.
    batch_idx : jax.Array, shape (num_atoms,), dtype int32
        Sorted system index per atom. Static topology, so close over it.
    maxstep : float, optional
        Largest distance an atom may move in one step. Zero disables the trust
        region.
    curvature_eps : float, optional
        Relative threshold below which a curvature pair is discarded. Defaults
        to ``1e-6`` for float32 coordinates and ``1e-10`` for float64: ``ys``
        is accumulated at the coordinate precision, so one value cannot serve
        both.
    graph_mode : {"warp", "warp_staged", "none"}, optional
        How the step is captured. ``"warp"`` replays a CUDA graph and is the
        default; ``"warp_staged"`` keys the capture on the call rather than on
        buffer addresses, which helps if the capture count grows without bound;
        ``"none"`` disables capture.

    Returns
    -------
    positions : jax.Array
        The advanced geometry.
    state : LBFGSState
        The advanced state. The input state is unchanged.

    Raises
    ------
    ValueError
        If this call's inputs are incompatible with the state.
    """
    _validate(positions, forces, batch_idx, state)
    call = _get_callable(positions.dtype, graph_mode)
    out = call(
        forces, batch_idx, positions,
        *(getattr(state, name) for name in _OPTIMIZER_BUFFERS),
        float(maxstep),
        float(_resolve_curvature_eps(curvature_eps, _WP_SCALAR[jnp.dtype(positions.dtype).type])),
    )  # fmt: skip
    return out[0], LBFGSState(**dict(zip(_OPTIMIZER_BUFFERS, out[1:], strict=True)))


def _validate(positions, forces, batch_idx, state) -> None:
    """Confirm this call's inputs match the state.

    The state checked its own internal consistency when it was built, so only
    the arrays that arrive fresh each call are re-checked here.
    """
    state.validate()
    check_against_state(
        state,
        coordinates=(("positions", positions), ("forces", forces)),
        indices=(("batch_idx", batch_idx),),
    )
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


def _lbfgs_cell_body_f32(
    forces: wp.array(dtype=wp.vec3f),
    stress: wp.array(dtype=wp.mat33f),
    batch_idx: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    x_base: wp.array(dtype=wp.vec3f),
    force_base: wp.array(dtype=wp.vec3f),
    direction: wp.array(dtype=wp.vec3f),
    s_history: wp.array(dtype=wp.vec3f, ndim=2),
    y_history: wp.array(dtype=wp.vec3f, ndim=2),
    ys: wp.array(dtype=wp.float32, ndim=2),
    yy: wp.array(dtype=wp.float32, ndim=2),
    alpha_hist: wp.array(dtype=wp.float32, ndim=2),
    beta_hist: wp.array(dtype=wp.float32, ndim=2),
    ss: wp.array(dtype=wp.float32),
    gg: wp.array(dtype=wp.float32),
    d0: wp.array(dtype=wp.float32),
    dmax: wp.array(dtype=wp.float32),
    dquad: wp.array(dtype=wp.float32),
    alpha_step: wp.array(dtype=wp.float32),
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
    maxstep: wp.float32,
    curvature_eps: wp.float32,
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
        maxstep=maxstep,
        curvature_eps=curvature_eps,
    )


def _lbfgs_cell_body_f64(
    forces: wp.array(dtype=wp.vec3d),
    stress: wp.array(dtype=wp.mat33d),
    batch_idx: wp.array(dtype=wp.int32),
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
    d0: wp.array(dtype=wp.float64),
    dmax: wp.array(dtype=wp.float64),
    dquad: wp.array(dtype=wp.float64),
    alpha_step: wp.array(dtype=wp.float64),
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
    if _p[5 : 5 + _n] != _OPTIMIZER_BUFFERS:
        raise RuntimeError(
            f"_OPTIMIZER_BUFFERS and _lbfgs_cell_body_{_name} have diverged"
        )
    if _p[5 + _n : 5 + _n + len(_CELL_BUFFERS)] != _CELL_BUFFERS:
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
    dtype,
    cell_force_scale: float = 1.0,
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
    dtype : jnp.float32 or jnp.float64
        Result precision. **Required**, and must match the *coordinates*:
        ``kappa`` scales matrices, so unlike the other per-system scalars it is
        not float64. There is no coordinate array here to infer it from, and
        guessing float64 would silently hand an fp32 caller a buffer the cell
        step rejects. The Warp and PyTorch helpers take no ``dtype`` because
        they write into an array the caller already allocated.

    Returns
    -------
    jax.Array, shape (num_systems,)

    Notes
    -----
    ``kappa`` is divided into the cell force, so it must never be zero. A
    batch containing a system with no atoms is rejected rather than given an
    invented scale, matching the Warp and PyTorch layers; see the variable-cell
    contract in :mod:`nvalchemiops.dynamics.optimizers.lbfgs`.
    """
    if cell_force_scale <= 0.0:
        raise ValueError(f"cell_force_scale must be positive; got {cell_force_scale}")
    dtype = jnp.dtype(dtype).type
    if dtype not in _CELL_BODIES:
        raise ValueError(f"dtype must be float32 or float64; got {dtype}")
    counts = jnp.asarray(n_particles)
    # A setup-time check, so concretizing is fine. Under ``jit`` the counts are
    # a tracer with no value to inspect; skip rather than fail, as the device
    # checks in ``validate`` do.
    if not isinstance(counts, jax.core.Tracer):
        empty = np.flatnonzero(np.asarray(counts) <= 0)
        if empty.size:
            raise ValueError(
                f"system(s) {empty.tolist()} have no atoms, which the "
                "variable-cell path does not support: kappa scales the cell "
                "against the atoms, so there is no scale to give them. Drop "
                "the empty systems from the batch. (Empty input on the "
                "coordinate-only path is fine and is a no-op.)"
            )
    return counts.astype(dtype) * dtype(cell_force_scale)


def lbfgs_prepare_cell_state(
    atom_ptr: jax.Array,
    cell: jax.Array,
    *,
    cell_force_scale: float = 1.0,
    dtype=jnp.float64,
) -> LBFGSCellState:
    """Build a complete, ready-to-step variable-cell chart from atom topology.

    One call: give it the ordinary ``atom_ptr`` and the aligned cells, and it
    derives the packed topology, captures the reference chart and computes
    ``kappa``. Nothing needs repairing before the first step.

    Ragged batches are unaffected -- ``atom_ptr`` already carries each
    system's atom count, so nothing here assumes an even split.

    Parameters
    ----------
    atom_ptr : jax.Array, shape (num_systems + 1,), dtype int32
        CSR-style atom pointer for the batch.
    cell : jax.Array, shape (num_systems, 3, 3)
        Reference lattice per system, aligned.
    cell_force_scale : float, optional
        Multiplier on the atom count in ``kappa``.
    dtype : optional
        Coordinate precision; must match the coordinates and the state.

    Returns
    -------
    LBFGSCellState

    See Also
    --------
    nvalchemiops.dynamics.optimizers.lbfgs : the variable-cell contract,
        including the requirement that the cells be aligned first.
    """
    if jnp.dtype(dtype).type not in _CELL_BODIES:
        raise ValueError(f"dtype must be float32 or float64; got {jnp.dtype(dtype)}")

    ptr = [int(v) for v in np.asarray(atom_ptr)]
    if len(ptr) < 2:
        raise ValueError(
            f"atom_ptr must have num_systems + 1 >= 2 entries; got {len(ptr)}"
        )
    n = len(ptr) - 1
    num_atoms = ptr[-1]
    counts = [ptr[s + 1] - ptr[s] for s in range(n)]
    if any(c < 0 for c in counts):
        raise ValueError(f"atom_ptr must be non-decreasing; got {ptr}")
    packed = num_atoms + 2 * n

    # Derived, not carried: a function of the topology alone. Built with jnp
    # rather than through Warp, as the rest of this binding's helpers are.
    ext_atom_ptr = jnp.asarray([ptr[s] + 2 * s for s in range(n + 1)], jnp.int32)
    ext_batch_idx = jnp.repeat(
        jnp.arange(n, dtype=jnp.int32),
        jnp.asarray(counts, jnp.int32) + 2,
        total_repeat_length=packed,
    )

    mat = lambda: jnp.zeros((n, 3, 3), dtype)  # noqa: E731
    ref_cell, ref_cell_inv = lbfgs_set_reference_cell(cell)
    state = LBFGSCellState(
        ref_cell=ref_cell, ref_cell_inv=ref_cell_inv,
        kappa=lbfgs_cell_kappa(
            jnp.asarray(counts, jnp.int32), dtype=dtype,
            cell_force_scale=cell_force_scale,
        ),
        ext_batch_idx=ext_batch_idx, ext_atom_ptr=ext_atom_ptr,
        phi=mat(), phi_inv=mat(), d_phi=mat(),
        cell_dof_a=jnp.zeros((n, 3), dtype), cell_dof_b=jnp.zeros((n, 3), dtype),
        cell_force_a=jnp.zeros((n, 3), dtype),
        cell_force_b=jnp.zeros((n, 3), dtype),
        ext_positions=jnp.zeros((packed, 3), dtype),
        ext_forces=jnp.zeros((packed, 3), dtype),
    )  # fmt: skip
    state.validate(num_atoms=num_atoms)
    return state


def lbfgs_step_coord_cell(
    positions: jax.Array,
    cell: jax.Array,
    forces: jax.Array,
    stress: jax.Array,
    state: LBFGSState,
    cell_state: LBFGSCellState,
    batch_idx: jax.Array,
    *,
    maxstep: float = 0.2,
    curvature_eps: float | None = None,
    graph_mode: str = "warp",
) -> tuple[jax.Array, jax.Array, LBFGSState, LBFGSCellState]:
    """Advance one variable-cell step, relaxing coordinates and cell together.

    Returns new arrays; nothing is mutated. Donate everything that comes back
    so XLA can reuse the memory.

    ``state`` must be sized for ``num_atoms + 2 * num_systems`` degrees of
    freedom, since the cell contributes two entries per system. Fill
    ``cell_state`` from :func:`lbfgs_set_reference_cell` and
    :func:`lbfgs_cell_kappa` once before the first step.

    Parameters
    ----------
    cell : jax.Array, shape (num_systems, 3, 3)
        Lattice vectors as columns, aligned by ``align_cell`` before the
        first step. See :ref:`the variable-cell contract <lbfgs-cell-contract>`
        in :mod:`nvalchemiops.dynamics.optimizers.lbfgs`, which is the single
        authoritative statement of the cell rules.
    stress : jax.Array, shape (num_systems, 3, 3)
        Cauchy stress, which drives the cell degrees of freedom.
    state, cell_state : LBFGSState, LBFGSCellState
        From the preparation functions, or built from your own arrays.

    Returns
    -------
    positions, cell : jax.Array
        The advanced geometry and lattice.
    state, cell_state : LBFGSState, LBFGSCellState
        The advanced states. The chart fields of ``cell_state`` come back
        unchanged; only its scratch fields are rewritten.

    See Also
    --------
    lbfgs_set_reference_cell : must be called first.
    lbfgs_cell_kappa : must be called first.
    lbfgs_step_coord : the coordinate-only equivalent.
    """
    _validate_cell(positions, forces, cell, stress, batch_idx, state, cell_state)
    call = _get_cell_callable(positions.dtype, graph_mode)
    out = call(
        forces, stress, batch_idx, positions, cell,
        *(getattr(state, name) for name in _OPTIMIZER_BUFFERS),
        *(getattr(cell_state, name) for name in _CELL_BUFFERS),
        float(maxstep),
        float(_resolve_curvature_eps(curvature_eps, _WP_SCALAR[jnp.dtype(positions.dtype).type])),
    )  # fmt: skip
    n_opt = len(_OPTIMIZER_BUFFERS)
    new_state = LBFGSState(
        **dict(zip(_OPTIMIZER_BUFFERS, out[2 : 2 + n_opt], strict=True))
    )
    # Only the scratch fields come back; the chart and topology are read-only.
    new_cell = dataclasses.replace(
        cell_state, **dict(zip(_CELL_SCRATCH, out[2 + n_opt :], strict=True))
    )
    return out[0], out[1], new_state, new_cell


def _validate_cell(
    positions, forces, cell, stress, batch_idx, state, cell_state
) -> None:
    """Confirm this call's inputs match both states.

    The degree-of-freedom check differs from the coordinate path, so this does
    not reuse :func:`_validate`: the state is sized for the packed array, not
    for the atoms.
    """
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
    if batch_idx.shape[0] != positions.shape[0]:
        raise ValueError(
            f"batch_idx length {batch_idx.shape[0]} != positions length "
            f"{positions.shape[0]}"
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
    if state.num_dofs != expected:
        raise ValueError(
            f"state is sized for {state.num_dofs} degrees of freedom, but the "
            f"variable-cell path needs {expected} (num_atoms + 2 * num_systems)"
        )
