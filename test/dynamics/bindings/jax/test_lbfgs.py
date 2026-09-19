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

"""Tests for the JAX L-BFGS binding.

Tests cover:

- The registration contract: input-output aliasing, argument order and graph
  mode. These need no GPU and catch ABI drift cheaply.
- Relaxation results, checked against the Warp layer rather than restating the
  algorithm.
- Donation and CUDA-graph replay, including that the capture count stays
  bounded rather than growing with the step count.
- That the step is not differentiable, which is a contract rather than an
  oversight.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import warnings

import numpy as np
import pytest

from nvalchemiops.dynamics.optimizers.lbfgs import (
    _CELL_BUFFERS,
    _CELL_SCRATCH,
    _OPTIMIZER_BUFFERS,
)

from .conftest import requires_gpu

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from nvalchemiops.jax.lbfgs import (  # noqa: E402
    LBFGS_CONVERGED,
    LBFGS_NEED_EVAL,
    lbfgs_converged,
    lbfgs_prepare_cell_state,
    lbfgs_prepare_state,
    lbfgs_step_coord,
)

STIFFNESS = np.array([1.0, 4.0, 9.0])


def make_jax_state(num_dofs, num_systems, dtype=None, history_size=6):
    """Return an :class:`LBFGSState` ready for the first step."""
    return lbfgs_prepare_state(
        num_dofs,
        num_systems,
        dtype=jnp.float64 if dtype is None else dtype,
        history_size=history_size,
    )


def make_jax_cell_state(num_atoms, num_systems, dtype=jnp.float64, counts=None):
    """Return an :class:`LBFGSCellState` with its topology and ``kappa`` filled.

    The extended topology comes from the generic batch utilities, which is what
    makes a ragged batch expressible; ``counts`` defaults to an even split.
    """
    import warp as wp

    from nvalchemiops.batch_utils import atom_ptr_to_batch_idx
    from nvalchemiops.dynamics.utils.cell_filter import extend_atom_ptr
    from nvalchemiops.jax.lbfgs import lbfgs_cell_kappa

    num_ext = num_atoms + 2 * num_systems
    if counts is None:
        counts = [num_atoms // num_systems] * num_systems
    counts_np = np.asarray(counts, np.int32)
    assert int(counts_np.sum()) == num_atoms

    atom_ptr = wp.array(
        np.concatenate([[0], np.cumsum(counts_np)]).astype(np.int32),
        dtype=wp.int32,
        device="cuda:0",
    )
    ext_atom_ptr = wp.zeros(num_systems + 1, dtype=wp.int32, device="cuda:0")
    extend_atom_ptr(atom_ptr, ext_atom_ptr, device="cuda:0")
    ext_batch_idx = wp.zeros(num_ext, dtype=wp.int32, device="cuda:0")
    atom_ptr_to_batch_idx(ext_atom_ptr, ext_batch_idx)

    state = lbfgs_prepare_cell_state(
        num_atoms,
        num_systems,
        jnp.asarray(ext_batch_idx.numpy()),
        jnp.asarray(ext_atom_ptr.numpy()),
        dtype=dtype,
    )
    return dataclasses.replace(
        state,
        kappa=lbfgs_cell_kappa(
            jnp.asarray(counts_np),
            cell_force_scale=1.0 / float(counts_np.max()),
            dtype=dtype,
        ),
    )


def _cluster(num_atoms, seed=42, scale=2.0):
    return np.random.default_rng(seed).normal(size=(num_atoms, 3)) * scale


class JaxDriver:
    """Relaxes an anisotropic quadratic through the JAX binding."""

    def __init__(self, positions, num_systems=1, dtype=jnp.float64, history_size=6):
        self.num_dofs = positions.shape[0]
        self.num_systems = num_systems
        per_system = self.num_dofs // num_systems
        self.dtype = dtype
        self.positions = jnp.asarray(positions, dtype)
        self.batch_idx = jnp.repeat(
            jnp.arange(num_systems, dtype=jnp.int32), per_system
        )
        self.n_particles = jnp.full((num_systems,), per_system, jnp.int32)
        self.state = make_jax_state(self.num_dofs, num_systems, dtype, history_size)
        self.stiffness = jnp.asarray(STIFFNESS)
        self.n_evals = 0

    def model(self, positions):
        """Forces only: the optimizer never asks for an energy."""
        x = positions.astype(jnp.float64)
        return (-(self.stiffness * x)).astype(self.dtype)

    def step(self, **kwargs):
        forces = self.model(self.positions)
        self.n_evals += 1
        self.positions, self.state = lbfgs_step_coord(
            self.positions,
            forces,
            self.state,
            self.batch_idx,
            self.n_particles,
            **kwargs,
        )

    @property
    def status(self):
        return self.state.status

    def run(self, max_evals=200, **kwargs):
        for _ in range(max_evals):
            self.step(**kwargs)
            if bool(lbfgs_converged(self.status)):
                break
        return self


class TestLBFGSJaxRegistration:
    """The registration contract. These run without a GPU."""

    def test_in_out_argnames_match_the_state_field_order(self):
        """The aliased arrays are exactly ``positions`` plus the 26 state.

        This is also the order results come back in, so callers unpack by
        position and a reordering here would silently rebind every buffer.
        """
        from nvalchemiops.jax.lbfgs import _LBFGS_IN_OUT_ARGS

        assert _LBFGS_IN_OUT_ARGS == ("positions",) + _OPTIMIZER_BUFFERS

    def test_removed_state_api_is_absent(self):
        """The superseded entry points are gone, with no shims.

        The state dataclasses and the preparation functions replaced them
        outright, so anything that owned state on the caller's behalf under
        the old semantics must not quietly survive as an alias.
        """
        import nvalchemiops.jax.lbfgs as module

        for name in (
            "lbfgs_allocate_state",
            "lbfgs_allocate_cell_state",
            "lbfgs_reset",
            "lbfgs_reduce_energy",
            "LBFGS_LS_FAILED",
        ):
            assert not hasattr(module, name), f"{name} still exists"
            assert name not in module.__all__

    def test_no_line_search_parameters_survive(self):
        """The step length is a trust region, and nothing may reintroduce a search.

        The line search was removed for a measured reason: it compares *total
        energies* while the direction comes from *forces*, which are different
        surfaces for a model with a direct force head, so it converged poorly
        on OMat24. A parameter creeping back in would reintroduce that failure
        silently.

        Covers the JAX layer only. Torch and JAX are independent extras, so
        each suite checks its own binding rather than importing the other's.
        """
        import nvalchemiops.jax.lbfgs as module

        banned = {
            "energy", "f_base", "gd", "ls_trials",
            "ftol", "wolfe", "step_scale_down", "step_scale_up",
            "min_step", "max_step", "max_ls_iter",
        }  # fmt: skip
        for fn in (module.lbfgs_step_coord, module.lbfgs_step_coord_cell):
            params = set(inspect.signature(fn).parameters)
            leaked = params & banned
            assert not leaked, f"{fn.__name__} takes {sorted(leaked)}"
            assert "maxstep" in params, f"{fn.__name__} lost its trust region"

        # The callable bodies are hand-written, so they can drift separately.
        for name in ("_lbfgs_body_f32", "_lbfgs_body_f64",
                     "_lbfgs_cell_body_f32", "_lbfgs_cell_body_f64"):  # fmt: skip
            params = set(inspect.signature(getattr(module, name)).parameters)
            assert not (params & banned), f"{name} takes {sorted(params & banned)}"
            assert "maxstep" in params, f"{name} lost its trust region"

        # And no status can report a line-search failure.
        assert {c for c in dir(module) if c.startswith("LBFGS_")} == {
            "LBFGS_NEED_EVAL",
            "LBFGS_CONVERGED",
        }

    def test_public_entry_points_take_the_states(self):
        """Both wrappers take the states as objects, in a fixed position.

        They forward the fields positionally into the callable, so what has to
        hold is that the dataclass field order and the callable's parameter
        order agree -- checked by
        ``test_body_parameter_order_matches_the_state_fields`` -- and that callers
        still find ``state`` where they expect it.
        """
        from nvalchemiops.jax.lbfgs import lbfgs_step_coord_cell

        coord = tuple(inspect.signature(lbfgs_step_coord).parameters)
        assert coord[:5] == (
            "positions",
            "forces",
            "state",
            "batch_idx",
            "n_particles",
        )
        cell = tuple(inspect.signature(lbfgs_step_coord_cell).parameters)
        assert cell[:8] == (
            "positions",
            "cell",
            "forces",
            "stress",
            "state",
            "cell_state",
            "batch_idx",
            "n_particles",
        )

    @pytest.mark.parametrize("suffix", ["f32", "f64"])
    def test_body_parameter_order_matches_the_state_fields(self, suffix):
        """A reordering would silently swap two arrays; nothing else catches it."""
        import nvalchemiops.jax.lbfgs as module

        body = getattr(module, f"_lbfgs_body_{suffix}")
        params = tuple(inspect.signature(body).parameters)
        offset = 4  # forces, batch_idx, n_particles, positions
        assert params[offset : offset + len(_OPTIMIZER_BUFFERS)] == _OPTIMIZER_BUFFERS

    @pytest.mark.parametrize("graph_mode", ["none", "warp", "warp_staged"])
    def test_callable_has_no_pure_outputs(self, graph_mode):
        """Every output is an alias of an input.

        With no pure outputs, warp's "in-out before output" ordering rule
        cannot bind. Any future diagnostic array must therefore be added
        *after* every aliased one.
        """
        from nvalchemiops.jax.lbfgs import _LBFGS_IN_OUT_ARGS, _get_callable

        call = _get_callable(jnp.float64, graph_mode)
        assert call.num_in_out == len(_LBFGS_IN_OUT_ARGS)
        assert call.num_outputs == len(_LBFGS_IN_OUT_ARGS)
        assert len(call.output_args) == 0

    def test_unknown_graph_mode_is_rejected(self):
        from nvalchemiops.jax.lbfgs import _get_callable

        with pytest.raises(ValueError, match="graph_mode must be one of"):
            _get_callable(jnp.float64, "cuda-graphs-please")

    def test_unsupported_dtype_is_rejected(self):
        """Preparation refuses the dtype, so it cannot reach a kernel."""
        with pytest.raises(ValueError, match="float32 or float64"):
            _ = make_jax_state(4, 1, dtype=jnp.float16)

    def test_documented_initial_contents(self):
        """The documented starting state is what preparation must produce.

        The three non-zero fields and the float64 scalar policy are part of
        the published contract, not the allocator's private business: a caller
        who builds the state from their own arrays has to match them.
        """
        state = make_jax_state(7, 3, history_size=4)
        assert state.s_history.shape == (4, 7, 3)
        assert state.ys.shape == (4, 3)
        # Per-system scalars are float64 whatever the coordinate precision.
        assert state.gg.dtype == jnp.float64
        np.testing.assert_array_equal(state.iteration, np.full(3, -1))
        np.testing.assert_array_equal(state.alpha_step, np.ones(3))
        np.testing.assert_array_equal(state.status, np.zeros(3, np.int32))
        # Everything else starts at zero.
        for name in set(_OPTIMIZER_BUFFERS) - {"iteration", "alpha_step"}:
            assert not np.asarray(getattr(state, name)).any(), (
                f"{name} should start zeroed"
            )


@pytest.mark.parametrize("_gpu", [pytest.param(None, marks=requires_gpu)])
class TestLBFGSJax:
    """Behaviour on device."""

    def test_reaches_the_minimum(self, _gpu):
        d = JaxDriver(_cluster(8)).run(force_tol=1e-8, maxstep=0.5)
        assert int(d.status[0]) == LBFGS_CONVERGED
        assert float(jnp.abs(d.positions).max()) < 1e-7

    def test_batched_systems_converge_independently(self, _gpu):
        rng = np.random.default_rng(17)
        blocks = [rng.normal(size=(4, 3)) * s for s in (0.001, 1.0, 5.0)]
        d = JaxDriver(np.vstack(blocks), num_systems=3).run(force_tol=1e-8, maxstep=0.5)
        np.testing.assert_array_equal(np.asarray(d.status), np.full(3, LBFGS_CONVERGED))

    def test_matches_the_warp_layer(self, _gpu):
        """A thin adapter must agree with the Warp layer step for step."""
        import warp as wp

        from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_step as warp_step

        from ...conftest import make_lbfgs_state

        start = _cluster(5, seed=17)
        d = JaxDriver(start)

        n = start.shape[0]
        device = "cuda:0"
        wp_pos = wp.array(start.copy(), dtype=wp.vec3d, device=device)
        wp_forces = wp.zeros(n, dtype=wp.vec3d, device=device)
        wp_batch = wp.zeros(n, dtype=wp.int32, device=device)
        wp_nparts = wp.array(np.array([n], np.int32), dtype=wp.int32, device=device)
        wp_state = make_lbfgs_state(n, 1, 6, wp.vec3d, device)

        for _ in range(30):
            d.step(force_tol=1e-8, maxstep=0.5)
            x = wp_pos.numpy()
            wp_forces.assign(-(STIFFNESS * x))
            warp_step(
                positions=wp_pos,
                forces=wp_forces,
                state=wp_state,
                batch_idx=wp_batch,
                n_particles=wp_nparts,
                force_tol=1e-8,
                maxstep=0.5,
            )
            wp.synchronize()
            np.testing.assert_allclose(
                np.asarray(d.positions), wp_pos.numpy(), rtol=1e-12, atol=1e-14
            )
            if bool(lbfgs_converged(d.status)):
                break
        np.testing.assert_array_equal(np.asarray(d.status), wp_state.status.numpy())

    @pytest.mark.parametrize("graph_mode", ["none", "warp", "warp_staged"])
    def test_graph_modes_agree(self, _gpu, graph_mode):
        """Capture must not change the answer.

        Compared bit-for-bit against the ungraphed baseline from an identical
        start: replay that silently drops work would show up here.
        """
        start = _cluster(6, seed=23)
        reference = JaxDriver(start).run(
            force_tol=1e-10, maxstep=0.5, graph_mode="none"
        )
        candidate = JaxDriver(start).run(
            force_tol=1e-10, maxstep=0.5, graph_mode=graph_mode
        )
        np.testing.assert_array_equal(
            np.asarray(candidate.positions), np.asarray(reference.positions)
        )
        np.testing.assert_array_equal(
            np.asarray(candidate.status), np.asarray(reference.status)
        )

    def test_donated_replay_keeps_the_capture_count_bounded(self, _gpu):
        """The graph working set must plateau, not grow with the step count.

        ``forces`` arrives from the model with a fresh buffer every step, and
        the capture is keyed on input addresses, so a small set of graphs is
        expected rather than exactly one. What must not happen is a capture per
        call, which would make graph mode slower than no graph at all.
        """
        from nvalchemiops.jax.lbfgs import _get_callable

        d = JaxDriver(_cluster(6, seed=31))
        batch_idx, n_particles = d.batch_idx, d.n_particles

        # Donate positions and all 26 state, which is what lets XLA reuse
        # them in place instead of copying a fresh set every step.
        @functools.partial(jax.jit, donate_argnums=(0, 1))
        def relax_step(positions, state, forces):
            return lbfgs_step_coord(
                positions,
                forces,
                state,
                batch_idx,
                n_particles,
                force_tol=1e-12,
                maxstep=0.5,
            )

        call = _get_callable(jnp.float64, "warp")
        # The callable is cached module-wide, so earlier tests may already have
        # populated it. Measure growth from here rather than absolute counts.
        baseline = len(getattr(call, "captures", {}))
        counts = []
        with warnings.catch_warnings():
            # A refused donation is only a warning; make it fail the test.
            warnings.simplefilter("error", UserWarning)
            for i in range(40):
                forces = d.model(d.positions)
                d.positions, d.state = relax_step(d.positions, d.state, forces)
                if i in (9, 19, 39):
                    jax.block_until_ready(d.positions)
                    counts.append(len(getattr(call, "captures", {})) - baseline)

        # A little growth is expected as JAX cycles through a pool of state;
        # what would mean replay is not happening is growth proportional to the
        # step count. Measured working set on this system is three to five.
        steps = 40
        assert counts[-1] <= steps // 5, (
            f"capture count {counts} is growing with the step count; "
            "replay is not happening, so graph_mode='warp_staged' is needed"
        )
        assert counts[-1] - counts[0] <= 2, (
            f"working set still growing between steps 10 and 40: {counts}"
        )

    def test_step_is_not_differentiable(self, _gpu):
        """Differentiating must fail loudly rather than return zeros."""
        d = JaxDriver(_cluster(4))
        forces = d.model(d.positions)

        def loss(positions):
            moved, _ = lbfgs_step_coord(
                positions,
                forces,
                d.state,
                d.batch_idx,
                d.n_particles,
                force_tol=1e-8,
            )
            return moved.sum()

        with pytest.raises(Exception):
            jax.grad(loss)(d.positions)


@pytest.mark.parametrize("_gpu", [pytest.param(None, marks=requires_gpu)])
class TestLBFGSJaxErrors:
    """Input validation."""

    def test_force_shape_mismatch(self, _gpu):
        d = JaxDriver(_cluster(3))
        with pytest.raises(ValueError, match="forces shape"):
            lbfgs_step_coord(
                d.positions,
                jnp.zeros((2, 3)),
                d.state,
                d.batch_idx,
                d.n_particles,
            )

    def test_scalars_must_be_float64(self, _gpu):
        """Without ``JAX_ENABLE_X64`` every float64 silently becomes float32.

        JAX only warns, so preparation would hand back a state the algorithm
        cannot use. The group is uniformly float32 and therefore internally
        consistent, which is why this is checked separately from ``validate``.
        """
        d = JaxDriver(_cluster(3))
        st = dataclasses.replace(
            d.state,
            **{
                name: getattr(d.state, name).astype(jnp.float32)
                for name in _OPTIMIZER_BUFFERS[5:18]
            },
        )
        st.validate()  # internally consistent
        with pytest.raises(ValueError, match="must be float64"):
            lbfgs_step_coord(
                d.positions, d.model(d.positions), st, d.batch_idx, d.n_particles
            )

    def test_history_size_mismatch(self, _gpu):
        d = JaxDriver(_cluster(3))
        forces = d.model(d.positions)
        other = make_jax_state(9, 1)
        with pytest.raises(ValueError, match="prepared for 9"):
            lbfgs_step_coord(
                d.positions,
                forces,
                other,
                d.batch_idx,
                d.n_particles,
            )


@pytest.mark.parametrize("_gpu", [pytest.param(None, marks=requires_gpu)])
class TestLBFGSJaxCoordCell:
    """The variable-cell binding."""

    #: Where ``positions`` and ``cell`` sit in the flat result tuple.
    N_OPT = len(_OPTIMIZER_BUFFERS)

    @staticmethod
    def _setup(num_atoms=6, seed=11):
        from nvalchemiops.jax.lbfgs import lbfgs_set_reference_cell

        from ...conftest import CellPotential

        rng = np.random.default_rng(seed)
        s0 = np.array([0.1, -0.2, 0.05])
        potential = CellPotential(s0)
        cell_np = np.diag([6.0, 6.5, 7.0])
        frac = s0 + rng.normal(size=(num_atoms, 3)) * 0.05
        positions = jnp.asarray(np.ascontiguousarray((cell_np @ frac.T).T))
        cell = jnp.asarray(cell_np[None])

        cell_state = make_jax_cell_state(num_atoms, 1)
        ref_cell, ref_cell_inv = lbfgs_set_reference_cell(cell)
        cell_state = dataclasses.replace(
            cell_state, ref_cell=ref_cell, ref_cell_inv=ref_cell_inv
        )
        # The cell contributes two packed entries per system.
        state = make_jax_state(num_atoms + 2, 1)
        return positions, cell, state, cell_state, potential

    def test_relaxes_cell_and_coordinates(self, _gpu):
        """A compressed cell expands to the target volume while atoms relax."""
        from nvalchemiops.jax.lbfgs import lbfgs_step_coord_cell

        n = 6
        positions, cell, state, cell_state, potential = self._setup(n)
        batch_idx = jnp.zeros(n, jnp.int32)
        n_particles = jnp.full((1,), n, jnp.int32)

        for _ in range(400):
            _, f, s = potential.energy_forces_stress(
                np.asarray(positions), np.asarray(cell)[0]
            )
            positions, cell, state, cell_state = lbfgs_step_coord_cell(
                positions,
                cell,
                jnp.asarray(f),
                jnp.asarray(s[None]),
                state,
                cell_state,
                batch_idx,
                n_particles,
                force_tol=1e-6,
                stress_tol=1e-6,
                maxstep=0.2,
            )
            if int(state.status[0]) != LBFGS_NEED_EVAL:
                break

        assert int(state.status[0]) == LBFGS_CONVERGED, int(state.status[0])
        volume = abs(np.linalg.det(np.asarray(cell)[0]))
        np.testing.assert_allclose(volume, potential.target_volume, rtol=1e-4)

    def test_matches_the_warp_layer(self, _gpu):
        """A thin adapter must agree with the Warp layer step for step."""
        import warp as wp

        from nvalchemiops.dynamics.optimizers.lbfgs import (
            lbfgs_set_reference_cell as warp_set_ref,
        )
        from nvalchemiops.dynamics.optimizers.lbfgs import (
            lbfgs_step_coord_cell as warp_step_cell,
        )
        from nvalchemiops.jax.lbfgs import lbfgs_step_coord_cell

        from ...conftest import make_lbfgs_cell_state, make_lbfgs_state

        n, device = 6, "cuda:0"
        positions, cell, state, cell_state, potential = self._setup(n)
        batch_idx = jnp.zeros(n, jnp.int32)
        n_particles = jnp.full((1,), n, jnp.int32)

        start_pos = np.asarray(positions).copy()
        start_cell = np.asarray(cell)[0].copy()
        wp_pos = wp.array(start_pos, dtype=wp.vec3d, device=device)
        wp_cell = wp.array(start_cell[None], dtype=wp.mat33d, device=device)
        wp_forces = wp.zeros(n, dtype=wp.vec3d, device=device)
        wp_stress = wp.zeros(1, dtype=wp.mat33d, device=device)
        wp_batch = wp.zeros(n, dtype=wp.int32, device=device)
        wp_nparts = wp.array(np.array([n], np.int32), dtype=wp.int32, device=device)
        wp_cell_state = make_lbfgs_cell_state(n, 1, wp.vec3d, device)
        warp_set_ref(wp_cell, wp_cell_state.ref_cell, wp_cell_state.ref_cell_inv)
        wp_state = make_lbfgs_state(n + 2, 1, 6, wp.vec3d, device)

        for _ in range(25):
            _, f, s = potential.energy_forces_stress(
                np.asarray(positions), np.asarray(cell)[0]
            )
            positions, cell, state, cell_state = lbfgs_step_coord_cell(
                positions,
                cell,
                jnp.asarray(f),
                jnp.asarray(s[None]),
                state,
                cell_state,
                batch_idx,
                n_particles,
                force_tol=1e-8,
                stress_tol=1e-8,
                maxstep=0.2,
            )

            _, f2, s2 = potential.energy_forces_stress(
                wp_pos.numpy(), wp_cell.numpy()[0]
            )
            wp_forces.assign(f2)
            wp_stress.assign(s2[None])
            warp_step_cell(
                wp_pos,
                wp_forces,
                wp_cell,
                wp_stress,
                wp_state,
                wp_cell_state,
                wp_batch,
                wp_nparts,
                force_tol=1e-8,
                stress_tol=1e-8,
                maxstep=0.2,
            )
            wp.synchronize()
            np.testing.assert_allclose(
                np.asarray(positions), wp_pos.numpy(), rtol=1e-12, atol=1e-14
            )
            np.testing.assert_allclose(
                np.asarray(cell), wp_cell.numpy(), rtol=1e-12, atol=1e-14
            )
            if int(state.status[0]) != LBFGS_NEED_EVAL:
                break
        np.testing.assert_array_equal(np.asarray(state.status), wp_state.status.numpy())

    def test_cell_callable_aliases_every_mutable_array(self, _gpu):
        """All 37 outputs are aliases; a future pure output must come last.

        The five read-only chart inputs must stay out of the aliased set, or
        XLA would demand them back and callers would have to donate topology
        they never change.
        """
        from nvalchemiops.jax.lbfgs import _CELL_IN_OUT_ARGS, _get_cell_callable

        call = _get_cell_callable(jnp.float64, "warp")
        assert call.num_in_out == len(_CELL_IN_OUT_ARGS)
        assert call.num_outputs == len(_CELL_IN_OUT_ARGS)
        assert len(call.output_args) == 0
        assert _CELL_IN_OUT_ARGS == (
            ("positions", "cell") + _OPTIMIZER_BUFFERS + _CELL_SCRATCH
        )
        assert len(_CELL_IN_OUT_ARGS) == 2 + len(_OPTIMIZER_BUFFERS) + len(
            _CELL_SCRATCH
        )
        assert set(_CELL_IN_OUT_ARGS).isdisjoint(_CELL_BUFFERS[:5])

    def test_buffers_sized_for_the_wrong_dof_count_are_rejected(self, _gpu):
        from nvalchemiops.jax.lbfgs import lbfgs_step_coord_cell

        n = 6
        positions, cell, _, cell_state, potential = self._setup(n)
        wrong = make_jax_state(n, 1)
        _, f, s = potential.energy_forces_stress(
            np.asarray(positions), np.asarray(cell)[0]
        )
        with pytest.raises(ValueError, match="num_atoms \\+ 2 \\* num_systems"):
            lbfgs_step_coord_cell(
                positions,
                cell,
                jnp.asarray(f),
                jnp.asarray(s[None]),
                wrong,
                cell_state,
                jnp.zeros(n, jnp.int32),
                jnp.full((1,), n, jnp.int32),
            )

    def test_jit_with_donation_matches_uncompiled(self, _gpu):
        """The variable-cell step runs under jit with everything donated.

        Donating ``cell`` alongside the state only works because the
        reference cell is an independent copy; sharing the caller's buffer
        would make XLA reject the same buffer being donated twice.
        """
        from nvalchemiops.jax.lbfgs import lbfgs_step_coord_cell

        n = 6
        batch_idx = jnp.zeros(n, jnp.int32)
        n_particles = jnp.full((1,), n, jnp.int32)
        opts = dict(force_tol=1e-8, stress_tol=1e-8, maxstep=0.2)

        def one_run(jit):
            positions, cell, state, cell_state, potential = self._setup(n)
            _, f, s = potential.energy_forces_stress(
                np.asarray(positions), np.asarray(cell)[0]
            )

            def body(p, c, f_, s_, st, cs):
                return lbfgs_step_coord_cell(
                    p, c, f_, s_, st, cs, batch_idx, n_particles, **opts
                )

            # Donate positions, cell and the optimizer state. The cell state
            # is left undonated: its five chart fields are read-only and come
            # back unchanged, so XLA cannot reuse their state.
            fn = jax.jit(body, donate_argnums=(0, 1, 4)) if jit else body
            out_p, out_c, out_st, _ = fn(
                positions,
                cell,
                jnp.asarray(f),
                jnp.asarray(s[None]),
                state,
                cell_state,
            )
            jax.block_until_ready(out_p)
            return np.asarray(out_p), np.asarray(out_c), np.asarray(out_st.status)

        eager = one_run(False)
        jitted = one_run(True)
        np.testing.assert_array_equal(jitted[0], eager[0])
        np.testing.assert_array_equal(jitted[1], eager[1])
        np.testing.assert_array_equal(jitted[2], eager[2])

    def test_cell_kappa_requires_an_explicit_dtype(self, _gpu):
        """``kappa`` must match the coordinates, and cannot be guessed here.

        Unlike the Warp and PyTorch helpers, which write into an array the
        caller already allocated, this one *returns* the array and so has to
        choose a precision -- with nothing passed in to infer it from. A
        float64 default silently handed fp32 callers a buffer the cell step
        rejects, so the argument is required.
        """
        from nvalchemiops.jax.lbfgs import lbfgs_cell_kappa

        counts = jnp.asarray([4, 3], jnp.int32)
        assert lbfgs_cell_kappa(counts, dtype=jnp.float32).dtype == jnp.float32
        assert lbfgs_cell_kappa(counts, dtype=jnp.float64).dtype == jnp.float64
        with pytest.raises(TypeError, match="dtype"):
            lbfgs_cell_kappa(counts)

    def test_fp32_variable_cell_step_runs(self, _gpu):
        """The fp32 cell path works end to end, kappa included.

        Every other variable-cell test here is float64, which is how an fp32
        kappa mismatch went unnoticed.
        """
        from nvalchemiops.jax.lbfgs import (
            lbfgs_cell_kappa,
            lbfgs_set_reference_cell,
            lbfgs_step_coord_cell,
        )

        n, m_sys, f32 = 4, 1, jnp.float32
        cell = jnp.asarray(np.diag([6.0, 6.5, 7.0])[None], f32)
        state = make_jax_state(n + 2 * m_sys, m_sys, dtype=f32)
        cell_state = make_jax_cell_state(n, m_sys, dtype=f32)
        ref_cell, ref_cell_inv = lbfgs_set_reference_cell(cell)
        cell_state = dataclasses.replace(
            cell_state,
            ref_cell=ref_cell,
            ref_cell_inv=ref_cell_inv,
            kappa=lbfgs_cell_kappa(jnp.asarray([n], jnp.int32), dtype=f32),
        )

        out_p, _, _, _ = lbfgs_step_coord_cell(
            jnp.zeros((n, 3), f32),
            cell,
            jnp.full((n, 3), 0.01, f32),
            jnp.zeros((m_sys, 3, 3), f32),
            state,
            cell_state,
            jnp.zeros(n, jnp.int32),
            jnp.full((m_sys,), n, jnp.int32),
            force_tol=1e-4,
            maxstep=0.2,
        )
        assert out_p.dtype == f32
        assert bool(jnp.isfinite(out_p).all())

    def test_zero_atom_system_gets_a_positive_kappa(self, _gpu):
        """An empty system must not produce ``kappa = 0``.

        ``kappa`` is divided into the cell force and the unpacked cell, so a
        zero turns both infinite. All three layers share one contract: a system
        with no atoms counts as one, since with nothing to balance the cell
        against the scale is arbitrary. This is asserted per layer rather than
        cross-layer because Torch and JAX are independent extras.
        """
        from nvalchemiops.jax.lbfgs import lbfgs_cell_kappa

        kappa = np.asarray(
            lbfgs_cell_kappa(
                jnp.asarray([4, 0, 3], jnp.int32),
                dtype=jnp.float64,
                cell_force_scale=0.25,
            )
        )
        assert (kappa > 0.0).all(), f"non-positive kappa: {kappa}"
        assert np.isfinite(1.0 / kappa).all()
        np.testing.assert_allclose(kappa, [1.0, 0.25, 0.75])

    def test_reference_cell_is_an_independent_copy(self, _gpu):
        """``ref_cell`` must not share a buffer with the caller's ``cell``.

        If it did, donating both to one jitted step would fail outright.
        """
        from nvalchemiops.jax.lbfgs import lbfgs_set_reference_cell

        cell = jnp.asarray(np.diag([6.0, 6.5, 7.0])[None])
        ref_cell, ref_cell_inv = lbfgs_set_reference_cell(cell)
        np.testing.assert_array_equal(np.asarray(ref_cell), np.asarray(cell))
        np.testing.assert_allclose(
            np.asarray(ref_cell_inv), np.linalg.inv(np.asarray(cell)), rtol=1e-14
        )
        assert ref_cell is not cell, (
            "ref_cell aliases the caller's cell; donating both would be rejected"
        )
