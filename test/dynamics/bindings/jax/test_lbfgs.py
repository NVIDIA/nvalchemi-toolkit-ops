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
    lbfgs_step_coord,
)

STIFFNESS = np.array([1.0, 4.0, 9.0])
STATUS = _OPTIMIZER_BUFFERS.index("status")


def make_jax_buffers(num_dofs, num_systems, dtype=None, history_size=6):
    """Allocate the caller-owned optimizer buffers, ready for the first step.

    The package allocates and initializes nothing, so this is a test-only
    factory. Returns them as a list in the canonical order, which is how every
    entry point takes them and returns them.
    """
    dtype = jnp.float64 if dtype is None else jnp.dtype(dtype).type
    if dtype not in (jnp.float32, jnp.float64):
        raise ValueError(f"dtype must be float32 or float64; got {dtype}")
    f64 = jnp.float64
    buffers = {
        "x_base": jnp.zeros((num_dofs, 3), dtype),
        "force_base": jnp.zeros((num_dofs, 3), dtype),
        "direction": jnp.zeros((num_dofs, 3), dtype),
        "s_history": jnp.zeros((history_size, num_dofs, 3), dtype),
        "y_history": jnp.zeros((history_size, num_dofs, 3), dtype),
        "ys": jnp.zeros((history_size, num_systems), f64),
        "yy": jnp.zeros((history_size, num_systems), f64),
        "alpha_hist": jnp.zeros((history_size, num_systems), f64),
        "beta_hist": jnp.zeros((history_size, num_systems), f64),
        **{n: jnp.zeros((num_systems,), f64) for n in _OPTIMIZER_BUFFERS[9:19]},
        # Three buffers do not start at zero.
        "alpha_step": jnp.ones((num_systems,), f64),
        "status": jnp.full((num_systems,), LBFGS_NEED_EVAL, jnp.int32),
        "iteration": jnp.full((num_systems,), -1, jnp.int32),
        **{n: jnp.zeros((num_systems,), jnp.int32) for n in _OPTIMIZER_BUFFERS[22:]},
    }
    assert tuple(buffers) == _OPTIMIZER_BUFFERS, "factory drifted from buffer order"
    return list(buffers.values())


def make_jax_cell_buffers(num_atoms, num_systems, dtype=jnp.float64, counts=None):
    """Allocate the caller-owned variable-cell buffers, in canonical order.

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

    buffers = {
        "ref_cell": jnp.zeros((num_systems, 3, 3), dtype),
        "ref_cell_inv": jnp.zeros((num_systems, 3, 3), dtype),
        "kappa": lbfgs_cell_kappa(
            jnp.asarray(counts_np),
            cell_force_scale=1.0 / float(counts_np.max()),
            dtype=dtype,
        ),
        "ext_batch_idx": jnp.asarray(ext_batch_idx.numpy()),
        "ext_atom_ptr": jnp.asarray(ext_atom_ptr.numpy()),
        "phi": jnp.zeros((num_systems, 3, 3), dtype),
        "phi_inv": jnp.zeros((num_systems, 3, 3), dtype),
        "d_phi": jnp.zeros((num_systems, 3, 3), dtype),
        "cell_dof_a": jnp.zeros((num_systems, 3), dtype),
        "cell_dof_b": jnp.zeros((num_systems, 3), dtype),
        "cell_force_a": jnp.zeros((num_systems, 3), dtype),
        "cell_force_b": jnp.zeros((num_systems, 3), dtype),
        "ext_positions": jnp.zeros((num_ext, 3), dtype),
        "ext_forces": jnp.zeros((num_ext, 3), dtype),
    }
    assert tuple(buffers) == _CELL_BUFFERS, "factory drifted from buffer order"
    return list(buffers.values())


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
        self.buffers = make_jax_buffers(self.num_dofs, num_systems, dtype, history_size)
        self.stiffness = jnp.asarray(STIFFNESS)
        self.n_evals = 0

    def model(self, positions):
        x = positions.astype(jnp.float64)
        per_atom = 0.5 * (self.stiffness * x**2).sum(axis=1)
        energy = jax.ops.segment_sum(per_atom, self.batch_idx, self.num_systems)
        return energy, (-(self.stiffness * x)).astype(self.dtype)

    def step(self, **kwargs):
        energy, forces = self.model(self.positions)
        self.n_evals += 1
        out = lbfgs_step_coord(
            self.positions,
            forces,
            energy,
            self.batch_idx,
            self.n_particles,
            *self.buffers,
            **kwargs,
        )
        self.positions, self.buffers = out[0], list(out[1:])

    @property
    def status(self):
        return self.buffers[STATUS]

    def run(self, max_evals=200, **kwargs):
        for _ in range(max_evals):
            self.step(**kwargs)
            if bool(lbfgs_converged(self.status)):
                break
        return self


class TestLBFGSJaxRegistration:
    """The registration contract. These run without a GPU."""

    def test_in_out_argnames_match_the_buffer_order(self):
        """The aliased arrays are exactly ``positions`` plus the 26 buffers.

        This is also the order results come back in, so callers unpack by
        position and a reordering here would silently rebind every buffer.
        """
        from nvalchemiops.jax.lbfgs import _LBFGS_IN_OUT_ARGS

        assert _LBFGS_IN_OUT_ARGS == ("positions",) + _OPTIMIZER_BUFFERS

    def test_removed_state_api_is_absent(self):
        """The state containers and allocators are gone, with no shims.

        Buffers are caller-owned, so anything that allocated or owned them on
        the caller's behalf must not quietly survive as an alias.
        """
        import nvalchemiops.jax.lbfgs as module

        for name in (
            "LBFGSState",
            "LBFGSCellState",
            "lbfgs_allocate_state",
            "lbfgs_allocate_cell_state",
            "lbfgs_reset",
        ):
            assert not hasattr(module, name), f"{name} still exists"
            assert name not in module.__all__

    def test_public_entry_points_take_the_buffers_in_order(self):
        """The wrappers forward positionally, so they must agree exactly."""
        from nvalchemiops.jax.lbfgs import lbfgs_step_coord_cell

        n_opt = len(_OPTIMIZER_BUFFERS)
        coord = tuple(inspect.signature(lbfgs_step_coord).parameters)
        assert coord[5 : 5 + n_opt] == _OPTIMIZER_BUFFERS
        cell = tuple(inspect.signature(lbfgs_step_coord_cell).parameters)
        assert cell[7 : 7 + n_opt] == _OPTIMIZER_BUFFERS
        assert cell[7 + n_opt : 7 + n_opt + len(_CELL_BUFFERS)] == _CELL_BUFFERS

    @pytest.mark.parametrize("suffix", ["f32", "f64"])
    def test_body_parameter_order_matches_the_buffers(self, suffix):
        """A reordering would silently swap two arrays; nothing else catches it."""
        import nvalchemiops.jax.lbfgs as module

        body = getattr(module, f"_lbfgs_body_{suffix}")
        params = tuple(inspect.signature(body).parameters)
        offset = 5  # forces, energy, batch_idx, n_particles, positions
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
        with pytest.raises(ValueError, match="float32 or float64"):
            _ = make_jax_buffers(4, 1, dtype=jnp.float16)

    def test_documented_initial_contents(self):
        """The documented starting state is what the buffers must hold.

        Users build these themselves now, so the three non-zero buffers and
        the float64 scalar policy are part of the published contract rather
        than an allocator's private business.
        """
        buffers = dict(zip(_OPTIMIZER_BUFFERS, make_jax_buffers(7, 3, history_size=4)))
        assert buffers["s_history"].shape == (4, 7, 3)
        assert buffers["ys"].shape == (4, 3)
        # Per-system scalars are float64 whatever the coordinate precision.
        assert buffers["gg"].dtype == jnp.float64
        np.testing.assert_array_equal(buffers["iteration"], np.full(3, -1))
        np.testing.assert_array_equal(buffers["alpha_step"], np.ones(3))
        np.testing.assert_array_equal(buffers["status"], np.zeros(3, np.int32))
        # Everything else starts at zero.
        for name in set(_OPTIMIZER_BUFFERS) - {"iteration", "alpha_step"}:
            assert not np.asarray(buffers[name]).any(), f"{name} should start zeroed"


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
        wp_energy = wp.zeros(1, dtype=wp.float64, device=device)
        wp_batch = wp.zeros(n, dtype=wp.int32, device=device)
        wp_nparts = wp.array(np.array([n], np.int32), dtype=wp.int32, device=device)
        wp_state = make_lbfgs_state(n, 1, 6, wp.vec3d, device)

        for _ in range(30):
            d.step(force_tol=1e-8, maxstep=0.5)
            x = wp_pos.numpy()
            wp_forces.assign(-(STIFFNESS * x))
            wp_energy.assign(np.array([(0.5 * STIFFNESS * x**2).sum()]))
            warp_step(
                positions=wp_pos,
                forces=wp_forces,
                energy=wp_energy,
                batch_idx=wp_batch,
                n_particles=wp_nparts,
                force_tol=1e-8,
                maxstep=0.5,
                **wp_state,
            )
            wp.synchronize()
            np.testing.assert_allclose(
                np.asarray(d.positions), wp_pos.numpy(), rtol=1e-12, atol=1e-14
            )
            if bool(lbfgs_converged(d.status)):
                break
        np.testing.assert_array_equal(np.asarray(d.status), wp_state["status"].numpy())

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

        # Donate positions and all 26 buffers, which is what lets XLA reuse
        # them in place instead of copying a fresh set every step.
        @functools.partial(
            jax.jit, donate_argnums=tuple(range(1 + len(_OPTIMIZER_BUFFERS)))
        )
        def relax_step(positions, *buffers_and_inputs):
            buffers = buffers_and_inputs[: len(_OPTIMIZER_BUFFERS)]
            forces, energy = buffers_and_inputs[len(_OPTIMIZER_BUFFERS) :]
            return lbfgs_step_coord(
                positions,
                forces,
                energy,
                batch_idx,
                n_particles,
                *buffers,
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
                energy, forces = d.model(d.positions)
                out = relax_step(d.positions, *d.buffers, forces, energy)
                d.positions, d.buffers = out[0], list(out[1:])
                if i in (9, 19, 39):
                    jax.block_until_ready(d.positions)
                    counts.append(len(getattr(call, "captures", {})) - baseline)

        # A little growth is expected as JAX cycles through a pool of buffers;
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
        energy, forces = d.model(d.positions)

        def loss(positions):
            out = lbfgs_step_coord(
                positions,
                forces,
                energy,
                d.batch_idx,
                d.n_particles,
                *d.buffers,
                force_tol=1e-8,
            )
            return out[0].sum()

        with pytest.raises(Exception):
            jax.grad(loss)(d.positions)


@pytest.mark.parametrize("_gpu", [pytest.param(None, marks=requires_gpu)])
class TestLBFGSJaxErrors:
    """Input validation."""

    def test_force_shape_mismatch(self, _gpu):
        d = JaxDriver(_cluster(3))
        energy, _ = d.model(d.positions)
        with pytest.raises(ValueError, match="forces shape"):
            lbfgs_step_coord(
                d.positions,
                jnp.zeros((2, 3)),
                energy,
                d.batch_idx,
                d.n_particles,
                *d.buffers,
            )

    def test_energy_must_be_float64(self, _gpu):
        d = JaxDriver(_cluster(3))
        energy, forces = d.model(d.positions)
        with pytest.raises(ValueError, match="energy must be float64"):
            lbfgs_step_coord(
                d.positions,
                forces,
                energy.astype(jnp.float32),
                d.batch_idx,
                d.n_particles,
                *d.buffers,
            )

    def test_history_size_mismatch(self, _gpu):
        d = JaxDriver(_cluster(3))
        energy, forces = d.model(d.positions)
        other = make_jax_buffers(9, 1)
        with pytest.raises(ValueError, match="history buffers"):
            lbfgs_step_coord(
                d.positions,
                forces,
                energy,
                d.batch_idx,
                d.n_particles,
                *other,
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

        cell_buffers = make_jax_cell_buffers(num_atoms, 1)
        ref_cell, ref_cell_inv = lbfgs_set_reference_cell(cell)
        cell_buffers[0], cell_buffers[1] = ref_cell, ref_cell_inv
        # The cell contributes two packed entries per system.
        buffers = make_jax_buffers(num_atoms + 2, 1)
        return positions, cell, buffers, cell_buffers, potential

    @classmethod
    def _unpack(cls, out):
        """Split a flat variable-cell result into its four groups.

        The read-only chart inputs are not returned, so the caller carries
        them forward unchanged.
        """
        n = cls.N_OPT
        return out[0], out[1], list(out[2 : 2 + n]), list(out[2 + n :])

    def test_relaxes_cell_and_coordinates(self, _gpu):
        """A compressed cell expands to the target volume while atoms relax."""
        from nvalchemiops.jax.lbfgs import lbfgs_step_coord_cell

        n = 6
        positions, cell, buffers, cell_buffers, potential = self._setup(n)
        batch_idx = jnp.zeros(n, jnp.int32)
        n_particles = jnp.full((1,), n, jnp.int32)

        for _ in range(400):
            e, f, s = potential.energy_forces_stress(
                np.asarray(positions), np.asarray(cell)[0]
            )
            out = lbfgs_step_coord_cell(
                positions,
                cell,
                jnp.asarray(f),
                jnp.asarray(s[None]),
                jnp.asarray([e]),
                batch_idx,
                n_particles,
                *buffers,
                *cell_buffers,
                force_tol=1e-6,
                stress_tol=1e-6,
                maxstep=0.2,
            )
            positions, cell, buffers, scratch = self._unpack(out)
            cell_buffers = cell_buffers[:5] + scratch
            if int(buffers[STATUS][0]) != LBFGS_NEED_EVAL:
                break

        assert int(buffers[STATUS][0]) == LBFGS_CONVERGED, int(buffers[STATUS][0])
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
        positions, cell, buffers, cell_buffers, potential = self._setup(n)
        batch_idx = jnp.zeros(n, jnp.int32)
        n_particles = jnp.full((1,), n, jnp.int32)

        start_pos = np.asarray(positions).copy()
        start_cell = np.asarray(cell)[0].copy()
        wp_pos = wp.array(start_pos, dtype=wp.vec3d, device=device)
        wp_cell = wp.array(start_cell[None], dtype=wp.mat33d, device=device)
        wp_forces = wp.zeros(n, dtype=wp.vec3d, device=device)
        wp_stress = wp.zeros(1, dtype=wp.mat33d, device=device)
        wp_energy = wp.zeros(1, dtype=wp.float64, device=device)
        wp_batch = wp.zeros(n, dtype=wp.int32, device=device)
        wp_nparts = wp.array(np.array([n], np.int32), dtype=wp.int32, device=device)
        wp_cell_state = make_lbfgs_cell_state(n, 1, wp.vec3d, device)
        warp_set_ref(wp_cell, wp_cell_state["ref_cell"], wp_cell_state["ref_cell_inv"])
        wp_state = make_lbfgs_state(n + 2, 1, 6, wp.vec3d, device)

        for _ in range(25):
            e, f, s = potential.energy_forces_stress(
                np.asarray(positions), np.asarray(cell)[0]
            )
            out = lbfgs_step_coord_cell(
                positions,
                cell,
                jnp.asarray(f),
                jnp.asarray(s[None]),
                jnp.asarray([e]),
                batch_idx,
                n_particles,
                *buffers,
                *cell_buffers,
                force_tol=1e-8,
                stress_tol=1e-8,
                maxstep=0.2,
            )
            positions, cell, buffers, scratch = self._unpack(out)
            cell_buffers = cell_buffers[:5] + scratch

            e2, f2, s2 = potential.energy_forces_stress(
                wp_pos.numpy(), wp_cell.numpy()[0]
            )
            wp_forces.assign(f2)
            wp_stress.assign(s2[None])
            wp_energy.assign(np.array([e2]))
            warp_step_cell(
                wp_pos,
                wp_forces,
                wp_cell,
                wp_stress,
                wp_energy,
                wp_batch,
                wp_nparts,
                *wp_state.values(),
                *wp_cell_state.values(),
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
            if int(buffers[STATUS][0]) != LBFGS_NEED_EVAL:
                break
        np.testing.assert_array_equal(
            np.asarray(buffers[STATUS]), wp_state["status"].numpy()
        )

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
        assert len(_CELL_IN_OUT_ARGS) == 37
        assert set(_CELL_IN_OUT_ARGS).isdisjoint(_CELL_BUFFERS[:5])

    def test_buffers_sized_for_the_wrong_dof_count_are_rejected(self, _gpu):
        from nvalchemiops.jax.lbfgs import lbfgs_step_coord_cell

        n = 6
        positions, cell, _, cell_buffers, potential = self._setup(n)
        wrong = make_jax_buffers(n, 1)
        e, f, s = potential.energy_forces_stress(
            np.asarray(positions), np.asarray(cell)[0]
        )
        with pytest.raises(ValueError, match="num_atoms \\+ 2 \\* num_systems"):
            lbfgs_step_coord_cell(
                positions,
                cell,
                jnp.asarray(f),
                jnp.asarray(s[None]),
                jnp.asarray([e]),
                jnp.zeros(n, jnp.int32),
                jnp.full((1,), n, jnp.int32),
                *wrong,
                *cell_buffers,
            )

    def test_jit_with_donation_matches_uncompiled(self, _gpu):
        """The variable-cell step runs under jit with everything donated.

        Donating ``cell`` alongside the buffers only works because the
        reference cell is an independent copy; sharing the caller's buffer
        would make XLA reject the same buffer being donated twice.
        """
        from nvalchemiops.jax.lbfgs import lbfgs_step_coord_cell

        n = 6
        batch_idx = jnp.zeros(n, jnp.int32)
        n_particles = jnp.full((1,), n, jnp.int32)
        opts = dict(force_tol=1e-8, stress_tol=1e-8, maxstep=0.2)

        def one_run(jit):
            positions, cell, buffers, cell_buffers, potential = self._setup(n)
            e, f, s = potential.energy_forces_stress(
                np.asarray(positions), np.asarray(cell)[0]
            )

            def body(p, c, f_, s_, e_, *all_buffers):
                return lbfgs_step_coord_cell(
                    p, c, f_, s_, e_, batch_idx, n_particles, *all_buffers, **opts
                )

            # Donate positions, cell and every buffer the step writes. The five
            # read-only chart inputs are deliberately left undonated, since
            # they are not returned.
            donated = (0, 1, *range(5, 5 + len(_OPTIMIZER_BUFFERS)))
            fn = jax.jit(body, donate_argnums=donated) if jit else body
            out = fn(
                positions,
                cell,
                jnp.asarray(f),
                jnp.asarray(s[None]),
                jnp.asarray([e]),
                *buffers,
                *cell_buffers,
            )
            jax.block_until_ready(out[0])
            return (
                np.asarray(out[0]),
                np.asarray(out[1]),
                np.asarray(out[2 + STATUS]),
            )

        eager = one_run(False)
        jitted = one_run(True)
        np.testing.assert_array_equal(jitted[0], eager[0])
        np.testing.assert_array_equal(jitted[1], eager[1])
        np.testing.assert_array_equal(jitted[2], eager[2])

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
