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

"""Tests for the Warp-level L-BFGS optimizer.

Tests cover:

- The two-loop recursion, against a NumPy reference and against algebraic
  identities that hold for any correct implementation.
- The line-search state machine, against closed-form outcomes derived on paper
  for a one-dimensional quadratic.
- Convergence on anisotropic quadratics, single and batched.
- The force/gradient sign convention.
- Ring-buffer bookkeeping when a low-curvature pair is discarded.
- Edge cases and input validation.

Note on multi-step comparisons: the state machine is a discontinuous function
of the reduction order, so a one-ULP difference in a curvature scalar can flip
an accept/reject and send two otherwise-equivalent runs down permanently
different paths. Multi-step tests therefore assert converged endpoints and
evaluation counts, never step-by-step trajectory equality.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.optimizers.lbfgs import (
    LBFGS_CONVERGED,
    LBFGS_LS_FAILED,
    LBFGS_NEED_EVAL,
    LBFGSState,
    lbfgs_apply_step,
    lbfgs_cell_trust_region,
    lbfgs_pack_cell,
    lbfgs_prepare_step,
    lbfgs_reduce_energy,
    lbfgs_reset,
    lbfgs_set_reference_cell,
    lbfgs_step,
    lbfgs_step_coord_cell,
    lbfgs_unpack_cell,
    lbfgs_update,
)

from .conftest import (
    DEVICES,
    DTYPE_CONFIGS,
    CellPotential,
    Quadratic,
    history_slots,
    lower_triangular_cell,
    make_lbfgs_cell_state,
    make_lbfgs_state,
    numpy_two_loop,
)

HISTORY_SIZE = 6


class Driver:
    """Runs a batched relaxation and records what happened."""

    def __init__(
        self,
        positions,
        num_systems,
        vec_dtype,
        np_dtype,
        device,
        history_size=HISTORY_SIZE,
        potential=None,
    ):
        self.np_dtype = np_dtype
        self.device = device
        self.potential = potential or Quadratic()
        self.num_dofs = positions.shape[0]
        self.num_systems = num_systems
        self.history_size = history_size

        per_system = self.num_dofs // num_systems
        self.batch_np = np.repeat(np.arange(num_systems), per_system).astype(np.int32)
        self.batch_idx = wp.array(self.batch_np, dtype=wp.int32, device=device)
        self.n_particles = wp.array(
            np.full(num_systems, per_system, np.int32), dtype=wp.int32, device=device
        )
        self.positions = wp.array(
            positions.astype(np_dtype), dtype=vec_dtype, device=device
        )
        self.forces = wp.zeros(self.num_dofs, dtype=vec_dtype, device=device)
        self.per_atom_energy = wp.zeros(self.num_dofs, dtype=wp.float64, device=device)
        self.energy = wp.zeros(num_systems, dtype=wp.float64, device=device)
        self.state = make_lbfgs_state(
            self.num_dofs, num_systems, history_size, vec_dtype, device
        )
        lbfgs_reset(**self.state)
        self.n_evals = 0

    def evaluate(self):
        xs = self.positions.numpy().astype(np.float64)
        e_atom, f = self.potential.energy_forces(xs)
        self.forces.assign(f.astype(self.np_dtype))
        self.per_atom_energy.assign(e_atom)
        lbfgs_reduce_energy(self.per_atom_energy, self.batch_idx, self.energy)
        self.n_evals += 1

    def step(self, **kwargs):
        lbfgs_step(
            positions=self.positions,
            forces=self.forces,
            energy=self.energy,
            batch_idx=self.batch_idx,
            n_particles=self.n_particles,
            **kwargs,
            **self.state,
        )
        wp.synchronize()

    def run(self, max_evals=300, **kwargs):
        for _ in range(max_evals):
            self.evaluate()
            self.step(**kwargs)
            if not (self.state["status"].numpy() == LBFGS_NEED_EVAL).any():
                break
        return self

    @property
    def status(self):
        return self.state["status"].numpy()

    def system_mask(self, s):
        return self.batch_np == s


def _cluster(num_systems, atoms_per_system, seed=42, scale=2.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(num_systems * atoms_per_system, 3)) * scale


# =============================================================================
# Two-loop recursion
# =============================================================================


class TestLBFGSTwoLoop:
    """The inverse-Hessian application, which is the algorithmic core."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("vec_dtype,scalar_dtype,np_dtype", DTYPE_CONFIGS)
    def test_matches_numpy_reference_at_every_depth(
        self, device, vec_dtype, scalar_dtype, np_dtype
    ):
        """Direction matches a NumPy two-loop at every history depth.

        Runs long enough to fill the ring buffer and wrap it, so the slot
        indexing is exercised in both the partially-filled and full regimes.
        """
        d = Driver(_cluster(2, 4), 2, vec_dtype, np_dtype, device)
        worst = 0.0
        depths = set()
        for _ in range(40):
            d.evaluate()
            d.step(force_tol=1e-10, maxstep=0.5)
            st = d.state
            n_loop = st["n_loop"].numpy()
            hist_count = st["history_count"].numpy()
            end = st["end"].numpy()
            s_hist = st["s_history"].numpy()
            y_hist = st["y_history"].numpy()
            ys = st["ys"].numpy()
            yy = st["yy"].numpy()
            direction = st["direction"].numpy()
            force_base = st["force_base"].numpy()
            for s in range(d.num_systems):
                if n_loop[s] <= 0 or hist_count[s] == 0:
                    continue
                depths.add(int(hist_count[s]))
                slots = history_slots(end[s], hist_count[s], d.history_size)
                mask = d.system_mask(s)
                ref = numpy_two_loop(
                    [s_hist[j][mask] for j in slots],
                    [y_hist[j][mask] for j in slots],
                    [ys[j][s] for j in slots],
                    [yy[j][s] for j in slots],
                    force_base[mask],
                )
                scale = max(np.abs(ref).max(), 1e-30)
                worst = max(worst, np.abs(ref - direction[mask]).max() / scale)
            if not (d.status == LBFGS_NEED_EVAL).any():
                break

        # The history vectors live at the coordinate precision, so the
        # achievable agreement follows the DOF dtype rather than the float64
        # scalars.
        tol = 1e-5 if np_dtype == np.float32 else 1e-12

        assert depths, "no two-loop direction was ever built"
        assert max(depths) == HISTORY_SIZE, (
            f"ring never filled; depths seen: {sorted(depths)}"
        )
        assert worst < tol, f"worst relative direction error {worst:.3e}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_secant_condition_holds(self, device):
        """``H y_newest == s_newest`` for the history the kernels produced.

        This is an identity of the L-BFGS operator, independent of the line
        search and of the initial scaling, so it needs no reference to compare
        against.
        """
        d = Driver(_cluster(1, 5), 1, wp.vec3d, np.float64, device).run(
            max_evals=25, force_tol=1e-12, maxstep=0.5
        )
        st = d.state
        hist_count = int(st["history_count"].numpy()[0])
        assert hist_count > 0
        slots = history_slots(st["end"].numpy()[0], hist_count, d.history_size)
        s_hist, y_hist = st["s_history"].numpy(), st["y_history"].numpy()
        ys, yy = st["ys"].numpy(), st["yy"].numpy()
        mask = d.system_mask(0)
        s_vecs = [s_hist[j][mask] for j in slots]
        y_vecs = [y_hist[j][mask] for j in slots]
        ys_v = [ys[j][0] for j in slots]
        yy_v = [yy[j][0] for j in slots]

        for gamma in (None, 1.0, 3.7):
            got = numpy_two_loop(s_vecs, y_vecs, ys_v, yy_v, y_vecs[0], gamma=gamma)
            np.testing.assert_allclose(got, s_vecs[0], rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("device", DEVICES)
    def test_operator_is_symmetric_and_positive_definite(self, device):
        """``u . H v == v . H u`` and ``g . H g > 0`` on the produced history."""
        d = Driver(_cluster(1, 5), 1, wp.vec3d, np.float64, device).run(
            max_evals=25, force_tol=1e-12, maxstep=0.5
        )
        st = d.state
        hist_count = int(st["history_count"].numpy()[0])
        slots = history_slots(st["end"].numpy()[0], hist_count, d.history_size)
        mask = d.system_mask(0)
        s_vecs = [st["s_history"].numpy()[j][mask] for j in slots]
        y_vecs = [st["y_history"].numpy()[j][mask] for j in slots]
        ys_v = [st["ys"].numpy()[j][0] for j in slots]
        yy_v = [st["yy"].numpy()[j][0] for j in slots]

        rng = np.random.default_rng(7)
        u = rng.normal(size=s_vecs[0].shape)
        v = rng.normal(size=s_vecs[0].shape)
        hu = numpy_two_loop(s_vecs, y_vecs, ys_v, yy_v, u)
        hv = numpy_two_loop(s_vecs, y_vecs, ys_v, yy_v, v)
        np.testing.assert_allclose((u * hv).sum(), (v * hu).sum(), rtol=1e-10)
        assert (u * hu).sum() > 0.0

    @pytest.mark.parametrize("device", DEVICES)
    def test_matches_scipy_with_identity_initial_hessian(self, device):
        """Cross-check the recursion against scipy at ``gamma == 1``.

        ``LbfgsInvHessProduct`` uses an identity initial inverse Hessian with no
        ``ys / yy`` scaling, so the comparison is only meaningful when the two
        agree on that factor. Building the history with ``ys == yy`` forces
        ``gamma == 1`` and still exercises the whole alpha/beta structure.
        """
        scipy_opt = pytest.importorskip("scipy.optimize")
        rng = np.random.default_rng(11)
        n, b = 12, 4
        s_vecs, y_vecs = [], []
        for _ in range(b):
            sv = rng.normal(size=n)
            yv = rng.normal(size=n)
            if sv @ yv < 0:
                yv = -yv
            yv *= (sv @ yv) / (yv @ yv)  # force ys == yy, hence gamma == 1
            s_vecs.append(sv)
            y_vecs.append(yv)
        ys_v = [s @ y for s, y in zip(s_vecs, y_vecs)]
        yy_v = [y @ y for y in y_vecs]
        np.testing.assert_allclose(ys_v, yy_v, rtol=1e-12)

        q = rng.normal(size=n)
        # scipy stores pairs oldest-first; ours are newest-first.
        prod = scipy_opt.LbfgsInvHessProduct(
            np.array(s_vecs[::-1]), np.array(y_vecs[::-1])
        )
        np.testing.assert_allclose(
            numpy_two_loop(s_vecs, y_vecs, ys_v, yy_v, q), prod.matvec(q), rtol=1e-10
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_discarding_a_pair_on_a_full_ring_drops_the_bound(self, device):
        """A discard on a full ring must not leave the recursion reading it.

        The candidate pair is written into the slot ``end`` points at before
        its curvature is known, because the curvature comes from that same
        reduction. When the ring is full that slot holds the oldest valid pair,
        so a discard destroys it. ``history_count`` has to drop to ``m - 1`` or
        the recursion would read the destroyed slot, whose curvature is now
        near zero, and divide by it.
        """
        d = Driver(_cluster(1, 5, seed=31), 1, wp.vec3d, np.float64, device)

        # Fill the ring.
        for _ in range(60):
            d.evaluate()
            d.step(force_tol=1e-12, maxstep=0.5)
            if d.state["history_count"].numpy()[0] == d.history_size:
                break
        assert d.state["history_count"].numpy()[0] == d.history_size, (
            "ring never filled"
        )
        end_before = int(d.state["end"].numpy()[0])

        # A curvature threshold this large rejects every pair, so the next
        # accepted step is guaranteed to be discarded.
        for _ in range(40):
            d.evaluate()
            d.step(force_tol=1e-12, maxstep=0.5, curvature_eps=1e30)
            if int(d.state["history_count"].numpy()[0]) < d.history_size:
                break

        assert d.state["history_count"].numpy()[0] == d.history_size - 1, (
            "history_count was not clamped after a discard on a full ring"
        )
        assert int(d.state["end"].numpy()[0]) == end_before, (
            "end advanced despite the pair being discarded"
        )
        assert np.isfinite(d.state["direction"].numpy()).all()
        assert d.status[0] != LBFGS_LS_FAILED

    @pytest.mark.parametrize("device", DEVICES)
    def test_direction_uses_only_committed_slots_after_a_discard(self, device):
        """After a discard the direction matches a reference built from the
        remaining committed pairs, not from the clobbered slot."""
        d = Driver(_cluster(1, 5, seed=33), 1, wp.vec3d, np.float64, device)
        for _ in range(60):
            d.evaluate()
            d.step(force_tol=1e-12, maxstep=0.5)
            if d.state["history_count"].numpy()[0] == d.history_size:
                break

        for _ in range(40):
            d.evaluate()
            d.step(force_tol=1e-12, maxstep=0.5, curvature_eps=1e30)
            st = d.state
            count = int(st["history_count"].numpy()[0])
            if count == d.history_size:
                continue
            if int(st["n_loop"].numpy()[0]) <= 0:
                continue
            slots = history_slots(st["end"].numpy()[0], count, d.history_size)
            mask = d.system_mask(0)
            ref = numpy_two_loop(
                [st["s_history"].numpy()[j][mask] for j in slots],
                [st["y_history"].numpy()[j][mask] for j in slots],
                [st["ys"].numpy()[j][0] for j in slots],
                [st["yy"].numpy()[j][0] for j in slots],
                st["force_base"].numpy()[mask],
            )
            np.testing.assert_allclose(
                st["direction"].numpy()[mask], ref, rtol=1e-11, atol=1e-13
            )
            return
        pytest.skip("no post-discard direction was produced")


def _one_atom(x0, device, k=(1.0, 1.0, 1.0)):
    """One atom on the x axis of an isotropic quadratic.

    Reduces exactly to the one-dimensional case ``E = 0.5 x^2`` with a unit
    steepest-descent direction, so the whole line search has hand-computable
    outcomes.
    """
    return Driver(
        np.array([[x0, 0.0, 0.0]]),
        1,
        wp.vec3d,
        np.float64,
        device,
        potential=Quadratic(k),
    )


class TestLBFGSLineSearch:
    """The per-system state machine, against closed-form outcomes.

    Each expectation below is derived on paper for ``E = 0.5 x^2`` with
    ``d = -sign(x)``, so these tests do not depend on any reference
    implementation sharing the design.
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_accepts_a_good_first_step(self, device):
        """``x0 = 10, alpha = 1``: Armijo, curvature and strong Wolfe all hold."""
        d = _one_atom(10.0, device)
        d.evaluate()
        d.step(force_tol=1e-10, maxstep=100.0)  # seeds direction and base point
        np.testing.assert_allclose(d.positions.numpy()[0, 0], 9.0)

        d.evaluate()
        d.step(force_tol=1e-10, maxstep=100.0)
        assert d.state["iteration"].numpy()[0] == 1, "step was not accepted"
        assert d.state["history_count"].numpy()[0] == 1
        assert d.state["ls_trials"].numpy()[0] == 0, "trial count not reset"
        np.testing.assert_allclose(d.state["alpha_step"].numpy()[0], 1.0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_grows_a_step_that_is_too_short(self, device):
        """``x0 = 100``: ``dt = -99 < 0.9 * d0 = -90``, so the step grows by 2.1."""
        d = _one_atom(100.0, device)
        d.evaluate()
        d.step(force_tol=1e-10, maxstep=1000.0)
        d.evaluate()
        d.step(force_tol=1e-10, maxstep=1000.0)

        np.testing.assert_allclose(d.state["alpha_step"].numpy()[0], 2.1, rtol=1e-12)
        assert d.state["ls_trials"].numpy()[0] == 1
        assert d.state["iteration"].numpy()[0] == 0, "should not have accepted"
        np.testing.assert_allclose(d.positions.numpy()[0, 0], 100.0 - 2.1, rtol=1e-12)

    @pytest.mark.parametrize("device", DEVICES)
    def test_shrinks_when_armijo_is_violated(self, device):
        """``x0 = 1, alpha = 3``: energy rises by 1.5, so the step halves."""
        d = _one_atom(1.0, device)
        d.evaluate()
        d.step(force_tol=1e-10, maxstep=100.0)

        # Place the trial at alpha = 3 along the seeded direction.
        d.state["alpha_step"].assign(np.array([3.0]))
        d.positions.assign(np.array([[-2.0, 0.0, 0.0]]))
        d.evaluate()
        d.step(force_tol=1e-10, maxstep=100.0)

        np.testing.assert_allclose(d.state["alpha_step"].numpy()[0], 1.5, rtol=1e-12)
        assert d.state["iteration"].numpy()[0] == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_shrinks_when_the_step_overshoots(self, device):
        """``x0 = 1, alpha = 1.95``: Armijo holds but ``|dt| > 0.9 |d0|``."""
        d = _one_atom(1.0, device)
        d.evaluate()
        d.step(force_tol=1e-10, maxstep=100.0)

        d.state["alpha_step"].assign(np.array([1.95]))
        d.positions.assign(np.array([[-0.95, 0.0, 0.0]]))
        d.evaluate()
        d.step(force_tol=1e-10, maxstep=100.0)

        np.testing.assert_allclose(d.state["alpha_step"].numpy()[0], 0.975, rtol=1e-12)
        assert d.state["iteration"].numpy()[0] == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_reports_failure_and_restores_the_base_point(self, device):
        """A line search out of budget gives up and rolls the positions back.

        Uses the "too short" case with a single trial allowed, so the first
        retry exhausts the budget.
        """
        d = _one_atom(100.0, device)
        d.evaluate()
        d.step(force_tol=1e-10, maxstep=1000.0, max_ls_iter=1)
        d.evaluate()
        d.step(force_tol=1e-10, maxstep=1000.0, max_ls_iter=1)

        assert d.status[0] == LBFGS_LS_FAILED
        np.testing.assert_allclose(d.positions.numpy()[0, 0], 100.0)
        np.testing.assert_allclose(
            d.positions.numpy(), d.state["x_base"].numpy(), atol=0
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_stall_with_history_restarts_instead_of_failing(self, device):
        """A stalled line search discards the history and keeps going.

        A stall usually means the curvature model has gone stale, not that
        there is no descent direction left. Reporting failure immediately loses
        runs that a steepest-descent restart would have finished. Terminal
        failure is reserved for a search that stalls with no history at all.
        """
        d = Driver(_cluster(1, 6, seed=2), 1, wp.vec3d, np.float64, device)

        # Build up some history first.
        for _ in range(40):
            d.evaluate()
            d.step(force_tol=1e-10, maxstep=0.5)
            if d.state["history_count"].numpy()[0] >= 3:
                break
        assert d.state["history_count"].numpy()[0] >= 3
        x_base_before = d.state["x_base"].numpy().copy()

        # Force a genuine stall: place the trial far past the minimum so the
        # energy rises and Armijo fails, with only one trial allowed.
        d.state["alpha_step"].assign(np.full(1, 50.0))
        d.positions.assign(x_base_before + 50.0 * d.state["direction"].numpy())
        d.evaluate()
        d.step(force_tol=1e-10, maxstep=1e6, max_ls_iter=1)

        assert d.status[0] == LBFGS_NEED_EVAL, (
            "a stall with history present should restart, not terminate"
        )
        assert d.state["history_count"].numpy()[0] == 0, "history was not discarded"
        np.testing.assert_allclose(d.positions.numpy(), x_base_before, atol=0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_stall_without_history_is_terminal(self, device):
        """With no history to discard there is nothing left to try."""
        d = _one_atom(100.0, device)
        d.evaluate()
        d.step(force_tol=1e-10, maxstep=1000.0, max_ls_iter=1)
        assert d.state["history_count"].numpy()[0] == 0
        d.evaluate()
        d.step(force_tol=1e-10, maxstep=1000.0, max_ls_iter=1)
        assert d.status[0] == LBFGS_LS_FAILED

    @pytest.mark.parametrize("device", DEVICES)
    def test_trial_count_does_not_accumulate_across_iterations(self, device):
        """``ls_trials`` resets per direction, so it cannot trip the budget.

        A cumulative counter would produce a spurious failure part-way through
        an otherwise healthy relaxation.
        """
        d = Driver(_cluster(1, 6, seed=5), 1, wp.vec3d, np.float64, device)
        d.run(max_evals=200, force_tol=1e-8, maxstep=0.2, max_ls_iter=6)
        assert d.status[0] == LBFGS_CONVERGED, (
            f"status {d.status[0]} after {d.n_evals} evaluations; "
            "a cumulative trial counter would look like this"
        )
        assert d.state["iteration"].numpy()[0] > 6, (
            "too few iterations to be a real test"
        )


class TestLBFGSConvergence:
    """End-to-end relaxation behaviour."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("vec_dtype,scalar_dtype,np_dtype", DTYPE_CONFIGS)
    def test_reaches_the_minimum(self, device, vec_dtype, scalar_dtype, np_dtype):
        """An anisotropic quadratic relaxes to the origin."""
        force_tol = 1e-3 if np_dtype == np.float32 else 1e-8
        d = Driver(_cluster(2, 4), 2, vec_dtype, np_dtype, device).run(
            force_tol=force_tol, maxstep=0.5
        )
        assert (d.status == LBFGS_CONVERGED).all(), d.status
        assert np.abs(d.positions.numpy()).max() < 10 * force_tol

    @pytest.mark.parametrize("device", DEVICES)
    def test_converges_in_a_bounded_number_of_evaluations(self, device):
        """Regression guard on evaluation count, the cost that actually matters."""
        d = Driver(_cluster(1, 8, seed=3), 1, wp.vec3d, np.float64, device).run(
            force_tol=1e-6, maxstep=0.5
        )
        assert d.status[0] == LBFGS_CONVERGED
        assert d.n_evals < 60, f"took {d.n_evals} evaluations"

    @pytest.mark.parametrize("device", DEVICES)
    def test_systems_converge_independently(self, device):
        """Systems started at very different displacements each finish correctly.

        Also exercises the case where some systems in a batch are already done
        while others are still stepping.
        """
        rng = np.random.default_rng(17)
        blocks = [rng.normal(size=(4, 3)) * scale for scale in (0.001, 1.0, 5.0)]
        d = Driver(np.vstack(blocks), 3, wp.vec3d, np.float64, device).run(
            force_tol=1e-8, maxstep=0.5
        )
        assert (d.status == LBFGS_CONVERGED).all(), d.status
        for s in range(3):
            block = d.positions.numpy()[d.system_mask(s)]
            assert np.abs(block).max() < 1e-7, f"system {s} not relaxed"

    @pytest.mark.parametrize("device", DEVICES)
    def test_already_converged_input_does_not_move(self, device):
        """A relaxed geometry is reported converged and left untouched.

        The base buffers must still be seeded, or they would stay at the zeros
        left by ``lbfgs_reset`` while the positions hold the real geometry.
        """
        start = np.full((3, 3), 1e-9)
        d = Driver(start, 1, wp.vec3d, np.float64, device)
        d.evaluate()
        d.step(force_tol=1e-5, maxstep=0.5)

        assert d.status[0] == LBFGS_CONVERGED
        np.testing.assert_allclose(d.positions.numpy(), start, atol=0)
        np.testing.assert_allclose(d.state["x_base"].numpy(), start, atol=0)
        np.testing.assert_allclose(
            d.state["force_base"].numpy(), d.forces.numpy(), atol=0
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_base_buffers_track_the_returned_geometry(self, device):
        """On any terminal status the base buffers describe what is returned."""
        d = Driver(_cluster(1, 5, seed=9), 1, wp.vec3d, np.float64, device).run(
            force_tol=1e-8, maxstep=0.5
        )
        assert d.status[0] == LBFGS_CONVERGED
        np.testing.assert_allclose(
            d.state["x_base"].numpy(), d.positions.numpy(), atol=0
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_tolerances_combine_conservatively(self, device):
        """Enabling a second criterion can only make convergence stricter."""
        loose = Driver(_cluster(1, 4, seed=21), 1, wp.vec3d, np.float64, device).run(
            force_tol=1e-4, maxstep=0.5
        )
        strict = Driver(_cluster(1, 4, seed=21), 1, wp.vec3d, np.float64, device).run(
            force_tol=1e-4, rms_tol=1e-8, maxstep=0.5
        )
        assert loose.status[0] == LBFGS_CONVERGED
        assert strict.status[0] == LBFGS_CONVERGED
        assert strict.n_evals >= loose.n_evals


class TestLBFGSSignConvention:
    """The force/gradient relation, which is silent when wrong.

    A reference implementation written from the same misunderstanding would
    agree with a sign error, so these assertions are deliberately physical.
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_first_step_moves_toward_the_minimum(self, device):
        """A global sign flip would move away and diverge immediately."""
        d = _one_atom(3.0, device)
        d.evaluate()
        d.step(force_tol=1e-10, maxstep=0.5)
        assert d.positions.numpy()[0, 0] < 3.0

    @pytest.mark.parametrize("device", DEVICES)
    def test_energy_decreases_monotonically(self, device):
        """Accepted steps never raise the energy; Armijo guarantees it."""
        d = Driver(_cluster(1, 5, seed=13), 1, wp.vec3d, np.float64, device)
        accepted = []
        for _ in range(80):
            d.evaluate()
            before = int(d.state["iteration"].numpy()[0])
            energy = float(d.energy.numpy()[0])
            d.step(force_tol=1e-8, maxstep=0.5)
            if int(d.state["iteration"].numpy()[0]) > before:
                accepted.append(energy)
            if not (d.status == LBFGS_NEED_EVAL).any():
                break
        assert len(accepted) > 3
        assert all(b >= a for b, a in zip(accepted, accepted[1:])), accepted

    @pytest.mark.parametrize("device", DEVICES)
    def test_stored_curvature_is_positive(self, device):
        """Every committed pair has ``s . y > 0``; a flipped y makes it negative."""
        d = Driver(_cluster(1, 5, seed=4), 1, wp.vec3d, np.float64, device).run(
            force_tol=1e-8, maxstep=0.5
        )
        count = int(d.state["history_count"].numpy()[0])
        assert count > 0
        slots = history_slots(d.state["end"].numpy()[0], count, d.history_size)
        ys = d.state["ys"].numpy()
        for j in slots:
            assert ys[j][0] > 0.0, f"slot {j} has non-positive curvature"

    @pytest.mark.parametrize("device", DEVICES)
    def test_direction_points_along_the_force(self, device):
        """``force_base . d > 0``, the sign-folded form of ``d0 < 0``."""
        d = Driver(_cluster(1, 5, seed=6), 1, wp.vec3d, np.float64, device)
        for _ in range(20):
            d.evaluate()
            d.step(force_tol=1e-10, maxstep=0.5)
            if d.status[0] != LBFGS_NEED_EVAL:
                break
            dotted = (
                d.state["force_base"].numpy() * d.state["direction"].numpy()
            ).sum()
            assert dotted > 0.0, f"direction opposes the force: {dotted}"


class TestLBFGSEdgeCases:
    """Degenerate inputs and boundary conditions."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_single_atom(self, device):
        d = _one_atom(2.0, device).run(force_tol=1e-9, maxstep=0.5)
        assert d.status[0] == LBFGS_CONVERGED

    @pytest.mark.parametrize("device", DEVICES)
    def test_zero_force_converges_without_moving(self, device):
        """A geometry exactly at the minimum is handled without dividing by zero."""
        d = Driver(np.zeros((3, 3)), 1, wp.vec3d, np.float64, device)
        d.evaluate()
        d.step(force_tol=1e-8, maxstep=0.5)
        assert d.status[0] == LBFGS_CONVERGED
        assert np.isfinite(d.positions.numpy()).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_empty_system(self, device):
        """Zero degrees of freedom is a no-op rather than a crash."""
        state = make_lbfgs_state(0, 1, HISTORY_SIZE, wp.vec3d, device)
        lbfgs_reset(**state)
        lbfgs_step(
            positions=wp.zeros(0, dtype=wp.vec3d, device=device),
            forces=wp.zeros(0, dtype=wp.vec3d, device=device),
            energy=wp.zeros(1, dtype=wp.float64, device=device),
            batch_idx=wp.zeros(0, dtype=wp.int32, device=device),
            n_particles=wp.zeros(1, dtype=wp.int32, device=device),
            **state,
        )
        wp.synchronize()

    @pytest.mark.parametrize("device", DEVICES)
    def test_history_shorter_than_the_run(self, device):
        """A ring of one still converges; the recursion degenerates gracefully."""
        d = Driver(
            _cluster(1, 4, seed=8), 1, wp.vec3d, np.float64, device, history_size=1
        ).run(force_tol=1e-6, maxstep=0.5)
        assert d.status[0] == LBFGS_CONVERGED
        assert d.state["history_count"].numpy()[0] <= 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_direction_stays_finite_throughout(self, device):
        """No NaN or infinity ever reaches the search direction."""
        d = Driver(_cluster(2, 6, seed=99), 2, wp.vec3d, np.float64, device)
        for _ in range(120):
            d.evaluate()
            d.step(force_tol=1e-9, maxstep=0.3)
            assert np.isfinite(d.state["direction"].numpy()).all()
            assert np.isfinite(d.positions.numpy()).all()
            if not (d.status == LBFGS_NEED_EVAL).any():
                break


class TestLBFGSStepErrors:
    """Input validation at the public entry points."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_mismatched_force_length(self, device):
        d = Driver(_cluster(1, 3), 1, wp.vec3d, np.float64, device)
        d.evaluate()
        with pytest.raises(ValueError, match="forces length"):
            lbfgs_step(
                positions=d.positions,
                forces=wp.zeros(2, dtype=wp.vec3d, device=device),
                energy=d.energy,
                batch_idx=d.batch_idx,
                n_particles=d.n_particles,
                **d.state,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_mismatched_batch_idx_length(self, device):
        d = Driver(_cluster(1, 3), 1, wp.vec3d, np.float64, device)
        d.evaluate()
        with pytest.raises(ValueError, match="batch_idx length"):
            lbfgs_step(
                positions=d.positions,
                forces=d.forces,
                energy=d.energy,
                batch_idx=wp.zeros(2, dtype=wp.int32, device=device),
                n_particles=d.n_particles,
                **d.state,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_history_buffer_shape(self, device):
        d = Driver(_cluster(1, 3), 1, wp.vec3d, np.float64, device)
        d.evaluate()
        d.state["s_history"] = wp.zeros(
            (HISTORY_SIZE, 2), dtype=wp.vec3d, device=device
        )
        with pytest.raises(ValueError, match="history buffers"):
            lbfgs_step(
                positions=d.positions,
                forces=d.forces,
                energy=d.energy,
                batch_idx=d.batch_idx,
                n_particles=d.n_particles,
                **d.state,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_energy_reduction_length_mismatch(self, device):
        with pytest.raises(ValueError, match="batch_idx length"):
            lbfgs_reduce_energy(
                wp.zeros(4, dtype=wp.float64, device=device),
                wp.zeros(3, dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.float64, device=device),
            )


class CellDriver:
    """Runs a variable-cell relaxation."""

    def __init__(self, positions, cell, potential, device, history_size=HISTORY_SIZE):
        self.device = device
        self.potential = potential
        self.num_atoms = positions.shape[0]
        self.num_ext = self.num_atoms + 2

        v, m, f = wp.vec3d, wp.mat33d, wp.float64
        self.positions = wp.array(positions.copy(), dtype=v, device=device)
        self.cell = wp.array(cell[None].copy(), dtype=m, device=device)
        self.forces = wp.zeros(self.num_atoms, dtype=v, device=device)
        self.stress = wp.zeros(1, dtype=m, device=device)
        self.energy = wp.zeros(1, dtype=f, device=device)
        self.batch_idx = wp.zeros(self.num_atoms, dtype=wp.int32, device=device)
        self.n_atoms_per_system = wp.array(
            np.array([self.num_atoms], np.int32), dtype=wp.int32, device=device
        )

        self.cell_state = make_lbfgs_cell_state(self.num_atoms, 1, v, device)
        lbfgs_set_reference_cell(
            self.cell, self.cell_state.ref_cell, self.cell_state.ref_cell_inv
        )

        self.state_dict = make_lbfgs_state(self.num_ext, 1, history_size, v, device)
        lbfgs_reset(**self.state_dict)
        self.state = LBFGSState(**self.state_dict)
        self.n_evals = 0

    def evaluate(self):
        e, f, s = self.potential.energy_forces_stress(
            self.positions.numpy(), self.cell.numpy()[0]
        )
        self.forces.assign(f)
        self.stress.assign(s[None])
        self.energy.assign(np.array([e]))
        self.n_evals += 1

    def pack(self):
        cs = self.cell_state
        lbfgs_pack_cell(
            self.positions,
            self.forces,
            self.cell,
            self.stress,
            cs.ref_cell_inv,
            cs.kappa,
            cs.ext_batch_idx,
            cs.ext_atom_ptr,
            cs.phi,
            cs.phi_inv,
            cs.cell_dof_a,
            cs.cell_dof_b,
            cs.cell_force_a,
            cs.cell_force_b,
            cs.ext_positions,
            cs.ext_forces,
        )

    def unpack(self):
        cs = self.cell_state
        lbfgs_unpack_cell(
            cs.ext_positions,
            cs.ref_cell,
            cs.kappa,
            self.batch_idx,
            cs.ext_atom_ptr,
            cs.phi,
            self.positions,
            self.cell,
        )

    def step(self, **kwargs):
        """One step through the public orchestrator."""
        lbfgs_step_coord_cell(
            self.positions,
            self.forces,
            self.cell,
            self.stress,
            self.energy,
            self.batch_idx,
            self.n_atoms_per_system,
            self.cell_state,
            self.state,
            **kwargs,
        )
        wp.synchronize()

    def step_composed(self, **kwargs):
        """The same step written out by hand.

        Kept so the orchestrator can be checked against the sequence of calls
        it is meant to replace.
        """
        cs, st = self.cell_state, self.state_dict
        self.pack()
        lbfgs_update(
            positions=cs.ext_positions,
            forces=cs.ext_forces,
            energy=self.energy,
            batch_idx=cs.ext_batch_idx,
            n_particles=self.n_atoms_per_system,
            cart_forces=self.forces,
            atom_batch_idx=self.batch_idx,
            stress=self.stress,
            measure_trust_region=False,
            **kwargs,
            **st,
        )
        lbfgs_cell_trust_region(
            cs.ext_positions,
            st["direction"],
            cs.phi,
            cs.d_phi,
            self.batch_idx,
            cs.ext_atom_ptr,
            cs.kappa,
            st["status"],
            st["n_loop"],
            st["dmax"],
            st["dquad"],
        )
        lbfgs_prepare_step(
            gg=st["gg"],
            d0=st["d0"],
            dmax=st["dmax"],
            dquad=st["dquad"],
            alpha_step=st["alpha_step"],
            status=st["status"],
            end=st["end"],
            n_loop=st["n_loop"],
            ls_trials=st["ls_trials"],
            history_count=st["history_count"],
            maxstep=kwargs.get("maxstep", 0.2),
        )
        lbfgs_apply_step(
            positions=cs.ext_positions,
            forces=cs.ext_forces,
            x_base=st["x_base"],
            force_base=st["force_base"],
            direction=st["direction"],
            batch_idx=cs.ext_batch_idx,
            status=st["status"],
            n_loop=st["n_loop"],
            gg=st["gg"],
            alpha_step=st["alpha_step"],
        )
        self.unpack()
        wp.synchronize()

    def run(self, max_evals=400, **kwargs):
        for _ in range(max_evals):
            self.evaluate()
            self.step(**kwargs)
            if self.state.status.numpy()[0] != LBFGS_NEED_EVAL:
                break
        return self


class TestLBFGSVariableCell:
    """The generalized-coordinate chart used for variable-cell relaxation.

    A wrong chart is silent: a raw concatenation of Cartesian positions and
    cell rows still converges on many systems, just more slowly, so these
    assertions target the chart directly.
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_round_trip(self, device):
        """``unpack(pack(r, H))`` returns the input, for non-commuting cells."""
        rng = np.random.default_rng(0)
        ref = lower_triangular_cell(1)
        cell = lower_triangular_cell(2)
        assert not np.allclose(cell @ np.linalg.inv(ref), np.linalg.inv(ref) @ cell), (
            "cells commute, so this test could not detect a reversed convention"
        )

        positions = rng.normal(size=(5, 3)) * 2.0
        d = CellDriver(positions, cell, CellPotential(np.zeros(3)), device)
        # Re-reference so the chart is not the identity.
        lbfgs_set_reference_cell(
            wp.array(ref[None], dtype=wp.mat33d, device=device),
            d.cell_state.ref_cell,
            d.cell_state.ref_cell_inv,
        )
        d.evaluate()
        d.pack()
        d.unpack()
        wp.synchronize()
        np.testing.assert_allclose(d.positions.numpy(), positions, atol=1e-12)
        np.testing.assert_allclose(d.cell.numpy()[0], cell, atol=1e-12)

    @pytest.mark.parametrize("device", DEVICES)
    def test_packed_force_is_the_packed_gradient(self, device):
        """Finite-difference the energy along a packed direction.

        This is the assertion a raw-concatenation implementation fails, and no
        other test in the suite catches it: perturb the atom block *and* the
        cell block together and check the slope against ``-(f_packed . d)``.
        """
        rng = np.random.default_rng(3)
        ref = lower_triangular_cell(1)
        cell = lower_triangular_cell(2)
        positions = rng.normal(size=(5, 3)) * 2.0
        potential = CellPotential(np.zeros(3))

        d = CellDriver(positions, cell, potential, device)
        lbfgs_set_reference_cell(
            wp.array(ref[None], dtype=wp.mat33d, device=device),
            d.cell_state.ref_cell,
            d.cell_state.ref_cell_inv,
        )
        d.evaluate()
        d.pack()
        wp.synchronize()
        packed = d.cell_state.ext_positions.numpy().copy()
        packed_force = d.cell_state.ext_forces.numpy().copy()
        direction = rng.normal(size=packed.shape) * 0.05

        def energy_at(t):
            d.cell_state.ext_positions.assign(packed + t * direction)
            d.unpack()
            wp.synchronize()
            return potential.energy_forces_stress(
                d.positions.numpy(), d.cell.numpy()[0]
            )[0]

        h = 1e-6
        finite_difference = (energy_at(h) - energy_at(-h)) / (2 * h)
        analytic = -(packed_force * direction).sum()
        np.testing.assert_allclose(finite_difference, analytic, rtol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_identity_chart_at_reset(self, device):
        """With ``Phi = I`` the packed arrays are the Cartesian ones."""
        rng = np.random.default_rng(5)
        cell = lower_triangular_cell(7)
        positions = rng.normal(size=(4, 3)) * 2.0
        d = CellDriver(positions, cell, CellPotential(np.zeros(3)), device)
        d.evaluate()
        d.pack()
        wp.synchronize()
        n = d.num_atoms
        np.testing.assert_allclose(
            d.cell_state.ext_positions.numpy()[:n], positions, atol=1e-12
        )
        np.testing.assert_allclose(
            d.cell_state.ext_forces.numpy()[:n], d.forces.numpy(), atol=1e-12
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_orchestrator_matches_the_composed_sequence(self, device):
        """``lbfgs_step_coord_cell`` is exactly the six calls it replaces.

        Both run the same kernels in the same order on identical inputs, so
        this is an exact comparison rather than a tolerance check.
        """
        rng = np.random.default_rng(41)
        cell = lower_triangular_cell(3)
        s0 = np.array([0.05, -0.1, 0.02])
        frac = s0 + rng.normal(size=(5, 3)) * 0.05
        positions = (cell @ frac.T).T
        potential = CellPotential(s0)

        one = CellDriver(positions, cell, potential, device)
        two = CellDriver(positions, cell, potential, device)
        for _ in range(12):
            one.evaluate()
            one.step(force_tol=1e-9, stress_tol=1e-9, maxstep=0.2)
            two.evaluate()
            two.step_composed(force_tol=1e-9, stress_tol=1e-9, maxstep=0.2)
            np.testing.assert_array_equal(one.positions.numpy(), two.positions.numpy())
            np.testing.assert_array_equal(one.cell.numpy(), two.cell.numpy())
            if one.state.status.numpy()[0] != LBFGS_NEED_EVAL:
                break
        np.testing.assert_array_equal(
            one.state.status.numpy(), two.state.status.numpy()
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_tiny_quadratic_term_does_not_freeze_the_step(self, device):
        """A small-but-positive quadratic term must not collapse the step to zero.

        The trust-region cap solves ``B*a^2 + A*a = maxstep``. Written the
        textbook way, ``(-A + sqrt(A^2 + 4*B*maxstep)) / (2*B)`` cancels
        completely once ``4*B*maxstep`` falls below the rounding of ``A^2``, and
        returns exactly zero — which freezes the optimizer with a step length of
        zero and no error. That regime is the common one here, because the
        quadratic term is second order in the step.

        A near-perfect lattice reaches it immediately: the atomic forces cancel
        by symmetry, so the direction lies almost entirely in the cell block and
        ``dquad`` lands around 1e-17.
        """
        cell = np.diag([6.0, 6.5, 7.0])
        # Atoms spread through the cell but sitting exactly at their targets,
        # so the atomic forces vanish while the coordinates stay non-zero.
        frac = np.random.default_rng(3).uniform(0.1, 0.9, size=(6, 3))
        potential = CellPotential(frac)
        positions = np.ascontiguousarray((cell @ frac.T).T)

        d = CellDriver(positions, cell, potential, device)
        d.evaluate()
        d.step(force_tol=1e-8, stress_tol=1e-8, maxstep=0.2)

        dquad = float(d.state.dquad.numpy()[0])
        alpha = float(d.state.alpha_step.numpy()[0])
        dmax = float(d.state.dmax.numpy()[0])
        assert dmax > 0.0, (
            "no displacement measured; the test is not exercising the cap"
        )
        assert alpha > 0.0, (
            f"step length collapsed to zero with dmax={dmax:.3e}, dquad={dquad:.3e}; "
            "the trust-region cap is losing precision"
        )
        # With a negligible quadratic term the cap must match the linear limit.
        if dquad < 1e-12 * dmax * dmax:
            np.testing.assert_allclose(alpha, min(1.0, 0.2 / dmax), rtol=1e-9)

    @pytest.mark.parametrize("device", DEVICES)
    def test_relaxes_cell_and_coordinates(self, device):
        """A compressed cell expands to the target volume while atoms relax."""
        rng = np.random.default_rng(11)
        s0 = np.array([0.1, -0.2, 0.05])
        potential = CellPotential(s0)
        cell = np.diag([6.0, 6.5, 7.0])  # compressed relative to the target
        frac = s0 + rng.normal(size=(6, 3)) * 0.05
        positions = (cell @ frac.T).T

        d = CellDriver(positions, cell, potential, device).run(
            force_tol=1e-6, stress_tol=1e-6, maxstep=0.2
        )
        assert d.state.status.numpy()[0] == LBFGS_CONVERGED, (
            f"status {d.state.status.numpy()[0]} after {d.n_evals} evaluations"
        )
        volume = abs(np.linalg.det(d.cell.numpy()[0]))
        np.testing.assert_allclose(volume, potential.target_volume, rtol=1e-4)
        frac_final = (np.linalg.inv(d.cell.numpy()[0]) @ d.positions.numpy().T).T
        np.testing.assert_allclose(
            frac_final, np.broadcast_to(s0, frac.shape), atol=1e-5
        )
