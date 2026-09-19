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
import os
import warnings

import numpy as np
import pytest

from nvalchemiops.dynamics.optimizers.lbfgs import (
    _OPTIMIZER_BUFFERS,
)

from .conftest import requires_gpu

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from nvalchemiops.jax.lbfgs import (  # noqa: E402
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
            self.positions, forces, self.state, self.batch_idx, **kwargs
        )

    def fmax(self):
        """Largest per-atom force magnitude per system, as a caller would."""
        norms = jnp.linalg.norm(self.model(self.positions), axis=1)
        return jnp.zeros(self.num_systems, norms.dtype).at[self.batch_idx].max(norms)

    def run(self, max_evals=200, force_tol=1e-8, **kwargs):
        """Relax until every system is under ``force_tol``.

        Tested before stepping: the forces describe the current positions.
        """
        for _ in range(max_evals):
            self.converged = self.fmax() <= force_tol
            if bool(self.converged.all()):
                break
            self.step(**kwargs)
        return self


class TestLBFGSJaxRegistration:
    """The registration contract. These run without a GPU."""

    def test_unknown_graph_mode_is_rejected(self):
        from nvalchemiops.jax.lbfgs import _get_callable

        with pytest.raises(ValueError, match="graph_mode must be one of"):
            _get_callable(jnp.float64, "cuda-graphs-please")

    def test_unsupported_dtype_is_rejected(self):
        """Preparation refuses the dtype, so it cannot reach a kernel."""
        with pytest.raises(ValueError, match="float32 or float64"):
            _ = make_jax_state(4, 1, dtype=jnp.float16)


@pytest.mark.parametrize("_gpu", [pytest.param(None, marks=requires_gpu)])
class TestLBFGSJax:
    """Behaviour on device."""

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
        wp_state = make_lbfgs_state(n, 1, 6, wp.vec3d, device)
        stiffness = np.asarray(STIFFNESS)

        for _ in range(30):
            d.step(maxstep=0.5)
            wp_forces.assign(-(stiffness * wp_pos.numpy()))
            warp_step(
                positions=wp_pos,
                forces=wp_forces,
                state=wp_state,
                batch_idx=wp_batch,
                maxstep=0.5,
            )
            wp.synchronize()
            np.testing.assert_allclose(
                np.asarray(d.positions), wp_pos.numpy(), rtol=1e-12, atol=1e-14
            )
        np.testing.assert_array_equal(
            np.asarray(d.state.iteration), wp_state.iteration.numpy()
        )

    @pytest.mark.parametrize("graph_mode", ["none", "warp", "warp_staged"])
    def test_graph_modes_agree(self, _gpu, graph_mode):
        """Capture must not change the answer.

        Compared bit-for-bit against the ungraphed baseline from an identical
        start: replay that silently drops work would show up here.
        """
        start = _cluster(6, seed=23)
        reference = JaxDriver(start).run(maxstep=0.5, graph_mode="none")
        candidate = JaxDriver(start).run(maxstep=0.5, graph_mode=graph_mode)
        np.testing.assert_array_equal(
            np.asarray(candidate.positions), np.asarray(reference.positions)
        )
        np.testing.assert_array_equal(
            np.asarray(candidate.state.iteration),
            np.asarray(reference.state.iteration),
        )

    def test_graph_mode_replays_instead_of_recapturing(self, _gpu):
        """One graph for many steps, not one graph per step.

        The other graph-mode tests compare numbers, and this regression
        produces *correct* numbers -- it just captures afresh every call, so
        graph mode becomes pure overhead. Nothing else here would notice.

        Measured from a cleared cache so the separation is unambiguous:
        replaying leaves a handful of entries however many steps are taken,
        while capturing per step leaves one per step (up to warp's own cache
        cap). No machine-tuned constant, because 1 and 20 are not close.

        What this does *not* claim to bound is memory: warp caps the cache at
        ``graph_cache_max`` itself, so captures cannot grow without limit
        whatever this code does.
        """
        from nvalchemiops.jax.lbfgs import _get_callable

        call = _get_callable(jnp.float64, "warp")
        # Fail here rather than silently measuring nothing if warp renames it.
        assert hasattr(call, "graph_cache_size") and hasattr(call, "captures"), (
            "warp no longer exposes its graph cache; this test measures "
            "nothing until it is pointed at the replacement"
        )

        d = JaxDriver(_cluster(6, seed=31))
        batch_idx = d.batch_idx

        @functools.partial(jax.jit, donate_argnums=(0, 1))
        def relax_step(positions, state, forces):
            return lbfgs_step_coord(positions, forces, state, batch_idx, maxstep=0.5)

        steps = 20
        # Start from empty, so this measures these steps rather than whatever
        # earlier tests left in the module-wide callable.
        call.captures.clear()
        # A refused donation is only a warning; make it fail, since losing
        # donation is the usual route to a growing capture set.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            for _ in range(steps):
                forces = d.model(d.positions)
                d.positions, d.state = relax_step(d.positions, d.state, forces)
        jax.block_until_ready(d.positions)

        assert call.graph_cache_size < steps // 4, (
            f"{steps} steps left {call.graph_cache_size} graphs cached; the "
            "capture count is tracking the step count, so replay is not "
            "happening and graph mode is pure overhead"
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
            )

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_scalars_follow_the_coordinate_dtype(self, _gpu, dtype):
        """An fp32 state is fp32 throughout.

        That is what frees the fp32 path from ``JAX_ENABLE_X64``: nothing in it
        asks JAX for a float64 array. See the Warp module on why fp32
        coordinates do not need float64 scalars.
        """
        st = lbfgs_prepare_state(3, 1, dtype=dtype)
        for name in _OPTIMIZER_BUFFERS[5:15]:
            assert jnp.dtype(getattr(st, name).dtype) == jnp.dtype(dtype), name

    def test_fp32_runs_without_x64(self, _gpu):
        """The fp32 path must not need ``JAX_ENABLE_X64``.

        Run in a subprocess, because x64 is a global flag fixed at import and
        the rest of this suite enables it. This is the whole user-facing point
        of letting the scalars follow the coordinate dtype: under the old
        policy an fp32 caller was silently handed float32 scalars by a JAX that
        refused the float64 request, and the step then rejected its own state.
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent("""
            import numpy as np, jax, jax.numpy as jnp
            assert not jax.config.jax_enable_x64, "x64 leaked into the subprocess"
            from nvalchemiops.jax.lbfgs import (
                lbfgs_prepare_state, lbfgs_step_coord)
            st = lbfgs_prepare_state(13, 1, dtype=jnp.float32)
            assert jnp.dtype(st.ys.dtype) == jnp.float32, st.ys.dtype
            K = jnp.array([1.0, 4.0, 9.0], jnp.float32)
            pos = jnp.asarray(
                np.random.default_rng(3).normal(size=(13, 3)) * 2.0, jnp.float32)
            bidx = jnp.zeros(13, jnp.int32)
            for _ in range(500):
                f = -(K * pos)
                if float(jnp.linalg.norm(f, axis=1).max()) < 1e-4:
                    break
                pos, st = lbfgs_step_coord(pos, f, st, bidx, maxstep=0.5)
            assert float(jnp.abs(pos).max()) < 1e-4, float(jnp.abs(pos).max())
            print("ok")
        """)
        env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
        out = subprocess.run(  # noqa: S603 - fixed script, this interpreter
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        )
        assert out.returncode == 0, out.stderr[-2000:]
        assert "ok" in out.stdout

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

        start_pos = np.asarray(positions).copy()
        start_cell = np.asarray(cell)[0].copy()
        wp_pos = wp.array(start_pos, dtype=wp.vec3d, device=device)
        wp_cell = wp.array(start_cell[None], dtype=wp.mat33d, device=device)
        wp_forces = wp.zeros(n, dtype=wp.vec3d, device=device)
        wp_stress = wp.zeros(1, dtype=wp.mat33d, device=device)
        wp_batch = wp.zeros(n, dtype=wp.int32, device=device)
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
                maxstep=0.2,
            )
            wp.synchronize()
            np.testing.assert_allclose(
                np.asarray(positions), wp_pos.numpy(), rtol=1e-12, atol=1e-14
            )
            np.testing.assert_allclose(
                np.asarray(cell), wp_cell.numpy(), rtol=1e-12, atol=1e-14
            )
        np.testing.assert_array_equal(
            np.asarray(state.iteration), wp_state.iteration.numpy()
        )

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
        opts = dict(maxstep=0.2)

        def one_run(jit):
            positions, cell, state, cell_state, potential = self._setup(n)
            _, f, s = potential.energy_forces_stress(
                np.asarray(positions), np.asarray(cell)[0]
            )

            def body(p, c, f_, s_, st, cs):
                return lbfgs_step_coord_cell(p, c, f_, s_, st, cs, batch_idx, **opts)

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
            return np.asarray(out_p), np.asarray(out_c), np.asarray(out_st.iteration)

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
