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

import pathlib

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.optimizers.lbfgs import (
    _OPTIMIZER_BUFFERS,
    _resolve_curvature_eps,
    check_cell_is_aligned,
    lbfgs_apply_step,
    lbfgs_cell_kappa,
    lbfgs_cell_trust_region,
    lbfgs_pack_cell,
    lbfgs_prepare_cell_state,
    lbfgs_prepare_state,
    lbfgs_prepare_step,
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
        self.state = make_lbfgs_state(
            self.num_dofs, num_systems, history_size, vec_dtype, device
        )
        self.n_evals = 0

    def evaluate(self):
        xs = self.positions.numpy().astype(np.float64)
        _, f = self.potential.energy_forces(xs)
        self.forces.assign(f.astype(self.np_dtype))
        self.n_evals += 1

    def step(self, **kwargs):
        lbfgs_step(
            positions=self.positions,
            forces=self.forces,
            state=self.state,
            batch_idx=self.batch_idx,
            **kwargs,
        )
        wp.synchronize()

    def fmax(self):
        """Largest per-atom force magnitude in each system.

        The optimizer owns no tolerance, so the driver computes this itself,
        which is exactly what a real caller does.
        """
        norms = np.linalg.norm(self.forces.numpy().astype(np.float64), axis=1)
        return np.array(
            [
                norms[self.batch_np == s].max(initial=0.0)
                for s in range(self.num_systems)
            ]
        )

    def run(self, max_evals=300, force_tol=1e-6, **kwargs):
        """Relax until every system is under ``force_tol``.

        Convergence is tested *before* stepping: the forces describe the
        current positions, and after a step they describe the previous point.
        """
        for _ in range(max_evals):
            self.evaluate()
            self.converged = self.fmax() <= force_tol
            if self.converged.all():
                break
            self.step(**kwargs)
        return self

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
            d.step(maxstep=0.5)
            st = d.state
            n_loop = st.n_loop.numpy()
            hist_count = st.history_count.numpy()
            end = st.end.numpy()
            s_hist = st.s_history.numpy()
            y_hist = st.y_history.numpy()
            ys = st.ys.numpy()
            yy = st.yy.numpy()
            direction = st.direction.numpy()
            force_base = st.force_base.numpy()
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
            if (d.fmax() <= 1e-12).all():
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
        hist_count = int(st.history_count.numpy()[0])
        assert hist_count > 0
        slots = history_slots(st.end.numpy()[0], hist_count, d.history_size)
        s_hist, y_hist = st.s_history.numpy(), st.y_history.numpy()
        ys, yy = st.ys.numpy(), st.yy.numpy()
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
        hist_count = int(st.history_count.numpy()[0])
        slots = history_slots(st.end.numpy()[0], hist_count, d.history_size)
        mask = d.system_mask(0)
        s_vecs = [st.s_history.numpy()[j][mask] for j in slots]
        y_vecs = [st.y_history.numpy()[j][mask] for j in slots]
        ys_v = [st.ys.numpy()[j][0] for j in slots]
        yy_v = [st.yy.numpy()[j][0] for j in slots]

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
            d.step(maxstep=0.5)
            if d.state.history_count.numpy()[0] == d.history_size:
                break
        assert d.state.history_count.numpy()[0] == d.history_size, "ring never filled"
        end_before = int(d.state.end.numpy()[0])

        # A curvature threshold this large rejects every pair, so the next
        # accepted step is guaranteed to be discarded.
        for _ in range(40):
            d.evaluate()
            d.step(maxstep=0.5, curvature_eps=1e30)
            if int(d.state.history_count.numpy()[0]) < d.history_size:
                break

        assert d.state.history_count.numpy()[0] == d.history_size - 1, (
            "history_count was not clamped after a discard on a full ring"
        )
        assert int(d.state.end.numpy()[0]) == end_before, (
            "end advanced despite the pair being discarded"
        )
        assert np.isfinite(d.state.direction.numpy()).all()
        assert d.fmax()[0] > 1e-12, "the run converged before the discard mattered"

    @pytest.mark.parametrize("device", DEVICES)
    def test_direction_uses_only_committed_slots_after_a_discard(self, device):
        """After a discard the direction matches a reference built from the
        remaining committed pairs, not from the clobbered slot."""
        d = Driver(_cluster(1, 5, seed=33), 1, wp.vec3d, np.float64, device)
        for _ in range(60):
            d.evaluate()
            d.step(maxstep=0.5)
            if d.state.history_count.numpy()[0] == d.history_size:
                break

        for _ in range(40):
            d.evaluate()
            d.step(maxstep=0.5, curvature_eps=1e30)
            st = d.state
            count = int(st.history_count.numpy()[0])
            if count == d.history_size:
                continue
            if int(st.n_loop.numpy()[0]) <= 0:
                continue
            slots = history_slots(st.end.numpy()[0], count, d.history_size)
            mask = d.system_mask(0)
            ref = numpy_two_loop(
                [st.s_history.numpy()[j][mask] for j in slots],
                [st.y_history.numpy()[j][mask] for j in slots],
                [st.ys.numpy()[j][0] for j in slots],
                [st.yy.numpy()[j][0] for j in slots],
                st.force_base.numpy()[mask],
            )
            np.testing.assert_allclose(
                st.direction.numpy()[mask], ref, rtol=1e-11, atol=1e-13
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


class TestLBFGSTrustRegion:
    """Step-length selection, which is now a trust region rather than a search.

    The defining property is that no evaluation is ever rejected, so the
    cadence is uniform and nothing depends on an energy.
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_every_call_is_an_accepted_step(self, device):
        """One evaluation in, one history update and one step out.

        ``iteration`` advances on every call after the first, which is what
        distinguishes this from a line search: there are no trials to reject,
        so a call can never consume an evaluation without making progress.
        """
        d = _one_atom(5.0, device)
        d.evaluate()
        d.step(maxstep=0.5)
        assert int(d.state.iteration.numpy()[0]) == 0  # seeded
        for expected in range(1, 8):
            d.evaluate()
            d.step(maxstep=0.5)
            assert int(d.state.iteration.numpy()[0]) == expected

    @pytest.mark.parametrize("device", DEVICES)
    def test_no_atom_moves_further_than_maxstep(self, device):
        """The trust region is the only thing bounding the displacement."""
        maxstep = 0.05
        d = Driver(_cluster(1, 6, seed=4), 1, wp.vec3d, np.float64, device)
        for _ in range(40):
            before = d.positions.numpy().copy()
            d.evaluate()
            d.step(maxstep=maxstep)
            moved = np.abs(d.positions.numpy() - before).max()
            assert moved <= maxstep + 1e-12, moved
            if (d.fmax() <= 1e-12).all():
                break

    @pytest.mark.parametrize("device", DEVICES)
    def test_step_length_is_not_carried_between_calls(self, device):
        """A shortened step must not persist and starve later iterations.

        Nothing resets ``alpha_step`` on the caller's behalf, so if the step
        were inherited rather than recomputed the method would decay into a
        scaled gradient descent.
        """
        d = _one_atom(100.0, device)
        d.evaluate()
        d.step(maxstep=0.1)  # heavily capped
        assert float(d.state.alpha_step.numpy()[0]) < 1.0
        # With the cap lifted the full quasi-Newton step must be available.
        d.evaluate()
        d.step(maxstep=0.0)  # trust region disabled
        np.testing.assert_allclose(d.state.alpha_step.numpy()[0], 1.0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_disabled_trust_region_takes_the_full_step(self, device):
        """``maxstep = 0`` means no bound, not a zero-length step."""
        d = _one_atom(1.0, device)
        d.evaluate()
        d.step(maxstep=0.0)
        d.evaluate()
        d.step(maxstep=0.0)
        np.testing.assert_allclose(d.state.alpha_step.numpy()[0], 1.0)
        assert np.isfinite(d.positions.numpy()).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_purely_quadratic_displacement_is_still_capped(self, device):
        """A zero linear term must not leave the step uncapped.

        On the variable-cell path the displacement is
        ``alpha * a_lin + alpha**2 * b_quad``. If the first-order term happens
        to cancel (``a_lin == 0``) while the cell still contributes at second
        order, the bound is ``sqrt(maxstep / b_quad)`` -- not "no bound". Only
        a step that moves nothing, both terms zero, may go uncapped.
        """
        maxstep, b_quad = 0.2, 5.0
        st = make_lbfgs_state(4, 1, HISTORY_SIZE, wp.vec3d, device)
        st.d0.assign(np.array([-1.0]))  # a valid descent direction
        st.n_loop.assign(np.array([2], np.int32))
        st.history_count.assign(np.array([2], np.int32))
        st.dmax.assign(np.array([0.0]))  # no linear displacement
        st.dquad.assign(np.array([b_quad]))

        lbfgs_prepare_step(st, maxstep=maxstep)
        wp.synchronize()

        alpha = float(st.alpha_step.numpy()[0])
        np.testing.assert_allclose(alpha, np.sqrt(maxstep / b_quad), rtol=1e-12)
        # The whole point: the displacement it produces respects the bound.
        np.testing.assert_allclose(b_quad * alpha**2, maxstep, rtol=1e-12)

    @pytest.mark.parametrize("device", DEVICES)
    def test_motionless_step_is_left_uncapped(self, device):
        """Both terms zero is the one case that legitimately has no bound."""
        st = make_lbfgs_state(4, 1, HISTORY_SIZE, wp.vec3d, device)
        st.d0.assign(np.array([-1.0]))
        st.n_loop.assign(np.array([2], np.int32))
        st.history_count.assign(np.array([2], np.int32))
        st.dmax.assign(np.array([0.0]))
        st.dquad.assign(np.array([0.0]))

        lbfgs_prepare_step(st, maxstep=0.2)
        wp.synchronize()
        # alpha stays at the full quasi-Newton step rather than being shrunk.
        np.testing.assert_allclose(st.alpha_step.numpy()[0], 1.0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_ascent_direction_is_replaced_before_the_atoms_move(self, device):
        """A rejected direction must not still be the one that gets applied.

        ``d0 >= 0`` means the curvature model produced an ascent direction. It
        is detected only after the two-loop that computes ``d0``, so the
        demotion has to happen *before* the kernel that rebuilds ``direction``
        -- otherwise the bookkeeping says "restarted" while the apply kernel
        still walks the atoms along the rejected direction, uphill.

        Mirrors the tail of :func:`lbfgs_update` (restart check, seed, trust
        region) so the ordering itself is pinned; a regression here moves atoms
        the wrong way while every scalar still looks right.
        """
        from nvalchemiops.dynamics.optimizers.lbfgs import (
            _lbfgs_restart_check_kernel,
            _seed_direction_overloads,
        )

        n = 4
        st = make_lbfgs_state(n, 1, HISTORY_SIZE, wp.vec3d, device)
        force = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))  # points along +x
        start = np.zeros((n, 3))

        positions = wp.array(start.copy(), dtype=wp.vec3d, device=device)
        forces = wp.array(force, dtype=wp.vec3d, device=device)
        batch_idx = wp.zeros(n, dtype=wp.int32, device=device)

        st.x_base.assign(start)
        st.force_base.assign(force)
        st.direction.assign(-force)  # uphill: opposes the force
        gg = float((force * force).sum())
        st.gg.assign(np.array([gg]))
        st.d0.assign(np.array([gg]))  # d0 = -(force_base . d) > 0
        st.n_loop.assign(np.array([2], np.int32))  # a two-loop direction
        st.history_count.assign(np.array([2], np.int32))
        st.dmax.assign(np.array([1.0]))

        wp.launch(
            _lbfgs_restart_check_kernel,
            dim=1,
            inputs=[st.gg, st.d0, st.end, st.n_loop, st.history_count],
            device=device,
        )  # fmt: skip
        wp.launch(
            _seed_direction_overloads[wp.vec3d],
            dim=n,
            inputs=[forces, positions, st.x_base, st.force_base,
                    st.direction, batch_idx, st.n_loop, st.gg],
            device=device,
        )  # fmt: skip
        lbfgs_prepare_step(st, maxstep=0.2)
        lbfgs_apply_step(positions, forces, st, batch_idx)
        wp.synchronize()

        assert int(st.n_loop.numpy()[0]) == -1, "did not restart"
        assert int(st.history_count.numpy()[0]) == 0, "history not discarded"
        # The direction must have been rebuilt from the force, not left as-is.
        np.testing.assert_allclose(
            st.direction.numpy(), force / np.sqrt(gg), rtol=1e-12
        )
        # The load-bearing assertion: the atoms moved DOWNHILL.
        displacement = positions.numpy() - start
        assert float((displacement * force).sum()) > 0.0, (
            "atoms were moved along the rejected ascent direction"
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
        assert d.converged.all(), d.fmax()
        assert np.abs(d.positions.numpy()).max() < 10 * force_tol

    @pytest.mark.parametrize("device", DEVICES)
    def test_converges_in_a_bounded_number_of_evaluations(self, device):
        """Regression guard on evaluation count, the cost that actually matters."""
        d = Driver(_cluster(1, 8, seed=3), 1, wp.vec3d, np.float64, device).run(
            force_tol=1e-6, maxstep=0.5
        )
        assert d.converged[0]
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
        assert d.converged.all(), d.fmax()
        for s in range(3):
            block = d.positions.numpy()[d.system_mask(s)]
            assert np.abs(block).max() < 1e-7, f"system {s} not relaxed"

    @pytest.mark.parametrize("device", DEVICES)
    def test_a_restart_step_moves_by_maxstep_however_small_the_force(self, device):
        """The sharp edge of owning convergence: test *before* you step.

        A restart direction is the force normalized, so the trust region caps
        it at exactly ``maxstep`` no matter how small the force is. Stepping a
        geometry that is already at the minimum therefore kicks it by
        ``maxstep`` rather than leaving it alone. The optimizer cannot know it
        has arrived -- that is the caller's test -- so this is pinned here to
        keep the documented "test before stepping" advice honest.
        """
        start = np.full((3, 3), 1e-9)
        d = Driver(start, 1, wp.vec3d, np.float64, device)
        d.evaluate()
        assert d.fmax()[0] < 1e-8, "the fixture is meant to start at the minimum"
        d.step(maxstep=0.5)

        # ``maxstep`` bounds the per-atom displacement norm, not a component.
        moved = np.linalg.norm(d.positions.numpy() - start, axis=1).max()
        np.testing.assert_allclose(moved, 0.5, rtol=1e-9)

    @pytest.mark.parametrize("device", DEVICES)
    def test_a_two_loop_step_vanishes_with_the_force(self, device):
        """Once a curvature model exists the step scales with the force.

        This is the complement of the restart case above: the direction is
        ``H F`` rather than a unit vector, so approaching the minimum shrinks
        the step instead of holding it at ``maxstep``.
        """
        d = Driver(_cluster(1, 5, seed=9), 1, wp.vec3d, np.float64, device)
        displacements = []
        for _ in range(40):
            d.evaluate()
            if d.state.history_count.numpy()[0] > 0:
                before = d.positions.numpy().copy()
                d.step(maxstep=0.5)
                displacements.append(np.abs(d.positions.numpy() - before).max())
            else:
                d.step(maxstep=0.5)
            if d.fmax()[0] < 1e-10:
                break

        assert len(displacements) > 5, "no two-loop steps were taken"
        assert displacements[-1] < displacements[0] * 1e-3, (
            f"step did not shrink with the force: {displacements[0]:.3e} -> "
            f"{displacements[-1]:.3e}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_returned_positions_are_not_where_the_forces_were_taken(self, device):
        """The documented contract, stated as the negative it actually is.

        The docs used to say the forces you passed in "still describe the
        positions you get back". They do not: the step moves the positions to a
        point that has not been evaluated. The forces belong to ``x_base``.
        Getting this wrong would have a caller test convergence on the wrong
        geometry.
        """
        d = Driver(_cluster(1, 5, seed=4), 1, wp.vec3d, np.float64, device)
        d.evaluate()
        evaluated = d.positions.numpy().copy()
        d.step(maxstep=0.5)

        returned = d.positions.numpy()
        assert np.abs(returned - evaluated).max() > 0.0, "the step did not move"
        # The evaluated point is kept, and it is not the one returned.
        np.testing.assert_allclose(d.state.x_base.numpy(), evaluated, atol=0)
        assert np.abs(d.state.x_base.numpy() - returned).max() > 0.0

    @pytest.mark.parametrize("device", DEVICES)
    def test_base_buffers_describe_the_evaluated_point(self, device):
        """``x_base``/``force_base`` hold the point the forces were taken at.

        Not the point that comes back: the step moves *forward* from the
        evaluated geometry, so after a call ``positions`` is ahead of
        ``x_base`` by exactly ``alpha * direction``. This is what lets the
        caller test convergence on the forces it just supplied.
        """
        d = Driver(_cluster(1, 5, seed=9), 1, wp.vec3d, np.float64, device)
        for _ in range(6):
            d.evaluate()
            evaluated = d.positions.numpy().copy()
            forces_here = d.forces.numpy().copy()
            d.step(maxstep=0.5)
            np.testing.assert_allclose(d.state.x_base.numpy(), evaluated, atol=0)
            np.testing.assert_allclose(d.state.force_base.numpy(), forces_here, atol=0)

            alpha = float(d.state.alpha_step.numpy()[0])
            expected = evaluated + alpha * d.state.direction.numpy()
            np.testing.assert_allclose(d.positions.numpy(), expected, rtol=1e-12)


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
        d.step(maxstep=0.5)
        assert d.positions.numpy()[0, 0] < 3.0

    @pytest.mark.parametrize("device", DEVICES)
    def test_stored_curvature_is_positive(self, device):
        """Every committed pair has ``s . y > 0``; a flipped y makes it negative.

        With no Wolfe condition to enforce it, the curvature guard is the only
        thing keeping the inverse-Hessian model positive definite.
        """
        d = Driver(_cluster(1, 5, seed=4), 1, wp.vec3d, np.float64, device).run(
            force_tol=1e-8, maxstep=0.5
        )
        count = int(d.state.history_count.numpy()[0])
        assert count > 0
        slots = history_slots(d.state.end.numpy()[0], count, d.history_size)
        ys = d.state.ys.numpy()
        for j in slots:
            assert ys[j][0] > 0.0, f"slot {j} has non-positive curvature"

    @pytest.mark.parametrize("device", DEVICES)
    def test_direction_points_along_the_force(self, device):
        """``force_base . d > 0``, the sign-folded form of ``d0 < 0``."""
        d = Driver(_cluster(1, 5, seed=6), 1, wp.vec3d, np.float64, device)
        for _ in range(20):
            d.evaluate()
            d.step(maxstep=0.5)
            if d.fmax()[0] <= 1e-12:
                break
            dotted = (d.state.force_base.numpy() * d.state.direction.numpy()).sum()
            assert dotted > 0.0, f"direction opposes the force: {dotted}"


class TestLBFGSEdgeCases:
    """Degenerate inputs and boundary conditions."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_single_atom(self, device):
        d = _one_atom(2.0, device).run(force_tol=1e-9, maxstep=0.5)
        assert d.converged[0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_zero_force_does_not_divide_by_zero(self, device):
        """Exactly zero force normalizes to nothing rather than to NaN.

        The restart direction is ``F / ||F||``, so this is the one case where
        it cannot be normalized. It must produce no motion, not a NaN.
        """
        d = Driver(np.zeros((3, 3)), 1, wp.vec3d, np.float64, device)
        d.evaluate()
        np.testing.assert_array_equal(d.forces.numpy(), 0.0)
        d.step(maxstep=0.5)
        assert np.isfinite(d.positions.numpy()).all()
        np.testing.assert_array_equal(d.positions.numpy(), 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_empty_system(self, device):
        """Zero degrees of freedom is a no-op rather than a crash."""
        state = make_lbfgs_state(0, 1, HISTORY_SIZE, wp.vec3d, device)
        lbfgs_step(
            positions=wp.zeros(0, dtype=wp.vec3d, device=device),
            forces=wp.zeros(0, dtype=wp.vec3d, device=device),
            state=state,
            batch_idx=wp.zeros(0, dtype=wp.int32, device=device),
        )
        wp.synchronize()

    @pytest.mark.parametrize("device", DEVICES)
    def test_history_shorter_than_the_run(self, device):
        """A ring of one still converges; the recursion degenerates gracefully."""
        d = Driver(
            _cluster(1, 4, seed=8), 1, wp.vec3d, np.float64, device, history_size=1
        ).run(force_tol=1e-6, maxstep=0.5)
        assert d.converged[0]
        assert d.state.history_count.numpy()[0] <= 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_direction_stays_finite_throughout(self, device):
        """No NaN or infinity ever reaches the search direction."""
        d = Driver(_cluster(2, 6, seed=99), 2, wp.vec3d, np.float64, device)
        for _ in range(120):
            d.evaluate()
            d.step(maxstep=0.3)
            assert np.isfinite(d.state.direction.numpy()).all()
            assert np.isfinite(d.positions.numpy()).all()
            if (d.fmax() <= 1e-12).all():
                break


class TestLBFGSStepErrors:
    """Input validation at the public entry points."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_mismatched_force_length(self, device):
        d = Driver(_cluster(1, 3), 1, wp.vec3d, np.float64, device)
        d.evaluate()
        with pytest.raises(ValueError, match="forces has 2 entries"):
            lbfgs_step(
                positions=d.positions,
                forces=wp.zeros(2, dtype=wp.vec3d, device=device),
                batch_idx=d.batch_idx,
                state=d.state,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_mismatched_batch_idx_length(self, device):
        d = Driver(_cluster(1, 3), 1, wp.vec3d, np.float64, device)
        d.evaluate()
        with pytest.raises(ValueError, match="batch_idx has 2 entries"):
            lbfgs_step(
                positions=d.positions,
                forces=d.forces,
                batch_idx=wp.zeros(2, dtype=wp.int32, device=device),
                state=d.state,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_history_buffer_shape(self, device):
        d = Driver(_cluster(1, 3), 1, wp.vec3d, np.float64, device)
        d.evaluate()
        d.state.s_history = wp.zeros((HISTORY_SIZE, 2), dtype=wp.vec3d, device=device)
        with pytest.raises(ValueError, match="s_history starts with dimensions"):
            lbfgs_step(
                positions=d.positions,
                forces=d.forces,
                batch_idx=d.batch_idx,
                state=d.state,
            )


def _commit_one_pair(cos_target, vec, np_dtype, device, eps=None, n=64):
    """Drive exactly one curvature pair with a chosen ``cos(s, y)``.

    ``s`` and ``y`` are built directly rather than produced by a relaxation, so
    the angle between them is the single thing under test. Setting
    ``iteration`` to zero makes the state machine treat this as an accepted
    step, which is what puts the pair through the commit kernel.

    Returns ``(kept, computed_ys)``.
    """
    rng = np.random.default_rng(0)
    s = rng.normal(size=(n, 3))
    perp = rng.normal(size=(n, 3))
    perp -= s * (np.dot(s.ravel(), perp.ravel()) / np.dot(s.ravel(), s.ravel()))
    s /= np.linalg.norm(s)
    perp /= np.linalg.norm(perp)
    y = cos_target * s + np.sqrt(1.0 - cos_target**2) * perp

    st = make_lbfgs_state(n, 1, HISTORY_SIZE, vec, device)
    st.iteration.assign(np.array([0], np.int32))
    # x_base and force_base start at zero, so s = positions and y = -forces.
    lbfgs_update(
        wp.array(s.astype(np_dtype), dtype=vec, device=device),
        wp.array((-y).astype(np_dtype), dtype=vec, device=device),
        st,
        wp.zeros(n, dtype=wp.int32, device=device),
        maxstep=0.2,
        curvature_eps=eps,
    )
    wp.synchronize()
    return bool(st.history_count.numpy()[0]), float(st.ys.numpy()[0, 0])


class TestLBFGSPublicSurface:
    """What the package exports, pinned.

    The surface had drifted both ways: it advertised the phase functions a
    step is built from, and it omitted ``lbfgs_step_coord_cell`` entirely, so
    the variable-cell entry point could not be reached from the package at all.
    """

    #: State, preparation, one step per call, and the variable-cell setup.
    CONTRACT = frozenset(
        {
            "LBFGSState",
            "LBFGSCellState",
            "lbfgs_prepare_state",
            "lbfgs_prepare_cell_state",
            "lbfgs_step",
            "lbfgs_step_coord_cell",
            "lbfgs_set_reference_cell",
            "lbfgs_cell_kappa",
            "check_cell_is_aligned",
        }
    )

    #: Decomposition points a step is built from. Importable, not advertised.
    PHASES = (
        "lbfgs_reduce",
        "lbfgs_update",
        "lbfgs_prepare_step",
        "lbfgs_apply_step",
        "lbfgs_pack_cell",
        "lbfgs_unpack_cell",
        "lbfgs_cell_trust_region",
    )

    def test_package_exports_exactly_the_contract(self):
        import nvalchemiops.dynamics.optimizers as package

        exported = {
            n
            for n in package.__all__
            if "lbfgs" in n.lower() or n == "check_cell_is_aligned"
        }
        assert exported == set(self.CONTRACT)

    def test_every_exported_name_resolves(self):
        """An `__all__` entry that is not importable breaks ``import *``."""
        import nvalchemiops.dynamics.optimizers as package

        missing = [n for n in self.CONTRACT if not hasattr(package, n)]
        assert not missing, f"exported but absent: {missing}"

    def test_the_variable_cell_entry_point_is_reachable(self):
        """It was missing, so the whole cell path was package-private."""
        from nvalchemiops.dynamics.optimizers import lbfgs_step_coord_cell

        assert callable(lbfgs_step_coord_cell)

    def test_phases_are_not_advertised(self):
        """Neither the package nor the module offers them for ``import *``."""
        import nvalchemiops.dynamics.optimizers as package
        import nvalchemiops.dynamics.optimizers.lbfgs as module

        leaked = [n for n in self.PHASES if n in package.__all__ or n in module.__all__]
        assert not leaked, f"decomposition points advertised as API: {leaked}"

    def test_phases_remain_importable(self):
        """Un-advertised is not removed: the composed path still works."""
        import nvalchemiops.dynamics.optimizers.lbfgs as module

        missing = [n for n in self.PHASES if not hasattr(module, n)]
        assert not missing, f"no longer importable: {missing}"


class TestLBFGSDocumentedContract:
    """The published prose must not re-acquire the removed behaviour.

    Convergence moved to the caller in code, but the claim survived in several
    docstrings and both examples long after the API did -- a reader following
    them would look for a ``status`` array that no longer exists. Signatures
    are checked elsewhere; this checks the sentences.
    """

    #: Phrases that describe an optimizer owning convergence or a line search.
    RETIRED = (
        "line-search decision",
        "reports progress through",
        "status says",
        "the line search chooses",
    )

    #: Every public surface a user reads before writing a loop.
    SURFACES = (
        "nvalchemiops/dynamics/optimizers/lbfgs.py",
        "nvalchemiops/dynamics/optimizers/__init__.py",
        "nvalchemiops/torch/lbfgs.py",
        "nvalchemiops/jax/lbfgs.py",
        "examples/dynamics/12_lbfgs_optimization.py",
        "examples/dynamics/13_lbfgs_variable_cell.py",
        "docs/userguide/components/dynamics.md",
        "docs/modules/warp/dynamics.rst",
    )

    @staticmethod
    def _repo_root():
        here = pathlib.Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "nvalchemiops").is_dir():
                return parent
        raise AssertionError("could not locate the repository root")

    def test_no_surface_claims_the_optimizer_owns_convergence(self):
        root = self._repo_root()
        offenders = []
        for rel in self.SURFACES:
            path = root / rel
            assert path.is_file(), f"{rel} moved; this test is now blind to it"
            text = path.read_text().lower()
            offenders += [
                f"{rel}: {phrase!r}" for phrase in self.RETIRED if phrase in text
            ]
        assert not offenders, (
            "documentation still describes convergence or a line search as the "
            f"optimizer's: {offenders}"
        )

    def test_the_status_constants_are_really_gone(self):
        """Prose aside, nothing may re-export a terminal status."""
        import nvalchemiops.dynamics.optimizers as package
        import nvalchemiops.dynamics.optimizers.lbfgs as module

        for namespace in (package, module):
            leaked = [n for n in dir(namespace) if n.startswith("LBFGS_")]
            assert not leaked, f"{namespace.__name__} exposes {leaked}"


class TestLBFGSCurvatureThreshold:
    """The curvature guard, which has to be read in the coordinate precision.

    ``ys`` is accumulated at the coordinate precision, so the relative
    threshold ``ys > eps * sqrt(ss * yy)`` can only reject on a sign it can
    actually resolve. Measured on deliberately orthogonal pairs the computed
    cosine bottoms out around 1e-8 in float32 and 1e-17 in float64, so a single
    default cannot serve both.
    """

    def test_default_follows_the_coordinate_dtype(self):
        """One number cannot sit above both noise floors."""
        assert _resolve_curvature_eps(None, wp.float32) == 1e-6
        assert _resolve_curvature_eps(None, wp.float64) == 1e-10
        assert _resolve_curvature_eps(3e-9, wp.float32) == 3e-9, "override ignored"

    @pytest.mark.parametrize("cos", [1e-9, 1e-8, 1e-7])
    @pytest.mark.parametrize("device", DEVICES)
    def test_fp32_rejects_a_pair_it_cannot_resolve(self, device, cos):
        """Inside the float32 noise band the pair must be discarded.

        The magnitude is not merely imprecise there, it is wrong: at
        ``cos = 1e-9`` the computed ``ys`` comes out near ``9e-9``, and it is
        that value which would scale the initial inverse Hessian through
        ``gamma = ys / yy``.
        """
        kept, ys = _commit_one_pair(cos, wp.vec3f, np.float32, device)
        assert not kept, f"kept a pair at cos={cos:.0e} with ys={ys:.3e}"

    @pytest.mark.parametrize("cos", [1e-5, 1e-3, 0.5])
    @pytest.mark.parametrize("device", DEVICES)
    def test_fp32_keeps_a_pair_it_can_resolve(self, device, cos):
        """The guard must not become so strict that it discards real curvature."""
        kept, _ = _commit_one_pair(cos, wp.vec3f, np.float32, device)
        assert kept, f"discarded a usable pair at cos={cos:.0e}"

    @pytest.mark.parametrize("cos", [1e-9, 1e-8, 1e-7])
    @pytest.mark.parametrize("device", DEVICES)
    def test_fp64_keeps_what_fp32_cannot(self, device, cos):
        """float64 resolves this band comfortably, so it must not be rejected.

        This is what makes the default dtype-dependent rather than simply
        raised: the same pair is signal in one precision and noise in the other.
        """
        kept, _ = _commit_one_pair(cos, wp.vec3d, np.float64, device)
        assert kept, f"discarded a resolvable pair at cos={cos:.0e}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_the_old_shared_default_admitted_noise(self, device):
        """Pins what changed: 1e-10 in float32 accepts an unresolvable pair.

        Passing it explicitly still does, since the caller overrides the
        default -- what moved is only what happens when nothing is passed.
        """
        kept_old, _ = _commit_one_pair(1e-8, wp.vec3f, np.float32, device, eps=1e-10)
        kept_new, _ = _commit_one_pair(1e-8, wp.vec3f, np.float32, device)
        assert kept_old and not kept_new


class TestLBFGSStateValidation:
    """What ``validate`` catches beyond shapes.

    The state is a plain dataclass, so callers may build or patch one. These
    are the mistakes that would otherwise reach a kernel as silently wrong
    numbers rather than as an error.
    """

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("vec", [wp.vec3f, wp.vec3d])
    def test_preparation_produces_the_documented_initial_contents(self, device, vec):
        """The published starting state, which a hand-built state must match.

        Core owns this because it is the algorithm's contract, not a binding
        detail: ``iteration = -1`` is what makes the first call take a
        steepest-descent step, and ``alpha_step = 1`` is the full quasi-Newton
        step the trust region then shrinks.
        """
        st = lbfgs_prepare_state(7, 3, dtype=vec, history_size=4, device=device)
        np.testing.assert_array_equal(st.iteration.numpy(), np.full(3, -1))
        np.testing.assert_array_equal(st.alpha_step.numpy(), np.ones(3))
        for name in set(_OPTIMIZER_BUFFERS) - {"iteration", "alpha_step"}:
            assert not getattr(st, name).numpy().any(), f"{name} should start zeroed"

    @pytest.mark.parametrize("device", DEVICES)
    def test_scalar_group_must_share_a_dtype(self, device):
        """One odd array out is the realistic way to get this wrong."""
        st = make_lbfgs_state(3, 1, HISTORY_SIZE, wp.vec3d, device)
        st.ys = wp.zeros((HISTORY_SIZE, 1), dtype=wp.float32, device=device)
        with pytest.raises(ValueError, match="must share one dtype"):
            st.validate()

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "vec,scalar", [(wp.vec3f, wp.float32), (wp.vec3d, wp.float64)]
    )
    def test_scalars_follow_the_coordinate_dtype(self, device, vec, scalar):
        """fp32 coordinates give an fp32 state, fp64 an fp64 one.

        There is no mixed configuration. ``y = force_base - F`` and the dot
        products are formed at the coordinate precision, so a wider
        accumulator cannot recover what that subtraction already lost -- see
        the module docstring. Making the scalars follow instead keeps an fp32
        run free of float64 arithmetic entirely.
        """
        st = make_lbfgs_state(6, 2, HISTORY_SIZE, vec, device)
        for name in _OPTIMIZER_BUFFERS[5:15]:
            assert getattr(st, name).dtype is scalar, name
        st.validate()

    @pytest.mark.parametrize("device", DEVICES)
    def test_scalars_must_match_the_coordinate_precision(self, device):
        """A uniformly fp32 scalar group under fp64 coordinates is not a state.

        Each group agreeing with itself is not enough: that combination is not
        a registered overload, so without the cross-group check it passes
        validation and then dies inside the kernel launcher with a dtype error
        instead of here.
        """
        st = make_lbfgs_state(4, 1, HISTORY_SIZE, wp.vec3d, device)
        for name in _OPTIMIZER_BUFFERS[5:15]:
            old = getattr(st, name)
            setattr(st, name, wp.zeros(old.shape, dtype=wp.float32, device=device))
        with pytest.raises(ValueError, match="mixes floating-point precisions"):
            st.validate()

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_scaling_must_match_the_coordinate_precision(self, device):
        """``kappa`` is checked too, though it shares a group with nothing."""
        cs = make_lbfgs_cell_state(4, 1, wp.vec3d, device)
        cs.kappa = wp.zeros(1, dtype=wp.float32, device=device)
        with pytest.raises(ValueError, match="mixes floating-point precisions"):
            cs.validate()

    @pytest.mark.parametrize("device", DEVICES)
    def test_a_mixed_state_is_rejected(self, device):
        """Promoting one scalar to float64 must not silently half-work."""
        st = make_lbfgs_state(6, 1, HISTORY_SIZE, wp.vec3f, device)
        st.gg = wp.zeros(1, dtype=wp.float64, device=device)
        with pytest.raises(ValueError, match="must share one dtype"):
            st.validate()

    @pytest.mark.parametrize("device", DEVICES)
    def test_fields_must_share_one_device(self, device):
        """A single array left on the host launches a kernel on bad memory."""
        st = make_lbfgs_state(3, 1, HISTORY_SIZE, wp.vec3d, device)
        st.gg = wp.zeros(1, dtype=wp.float64, device="cpu")
        with pytest.raises(ValueError, match="spread across devices"):
            st.validate()

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_matrices_must_share_a_dtype(self, device):
        cs = make_lbfgs_cell_state(4, 1, wp.vec3d, device)
        cs.phi = wp.zeros(1, dtype=wp.mat33f, device=device)
        with pytest.raises(ValueError, match="must share one dtype"):
            cs.validate()


class CellDriver:
    """Runs a variable-cell relaxation."""

    def __init__(self, positions, cell, potential, device, history_size=HISTORY_SIZE):
        self.device = device
        self.potential = potential
        self.num_atoms = positions.shape[0]
        self.num_ext = self.num_atoms + 2

        v, m = wp.vec3d, wp.mat33d
        self.positions = wp.array(positions.copy(), dtype=v, device=device)
        self.cell = wp.array(cell[None].copy(), dtype=m, device=device)
        self.forces = wp.zeros(self.num_atoms, dtype=v, device=device)
        self.stress = wp.zeros(1, dtype=m, device=device)
        self.batch_idx = wp.zeros(self.num_atoms, dtype=wp.int32, device=device)
        self.n_atoms_per_system = wp.array(
            np.array([self.num_atoms], np.int32), dtype=wp.int32, device=device
        )

        self.cell_state = make_lbfgs_cell_state(self.num_atoms, 1, v, device)
        lbfgs_set_reference_cell(
            self.cell,
            self.cell_state.ref_cell,
            self.cell_state.ref_cell_inv,
        )

        self.state = make_lbfgs_state(self.num_ext, 1, history_size, v, device)
        self.n_evals = 0

    def evaluate(self):
        _, f, s = self.potential.energy_forces_stress(
            self.positions.numpy(), self.cell.numpy()[0]
        )
        self.forces.assign(f)
        self.stress.assign(s[None])
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
            self.state,
            self.cell_state,
            self.batch_idx,
            **kwargs,
        )
        wp.synchronize()

    def step_composed(self, **kwargs):
        """The same step written out by hand.

        Kept so the orchestrator can be checked against the sequence of calls
        it is meant to replace.
        """
        cs, st = self.cell_state, self.state
        self.pack()
        lbfgs_update(
            positions=cs.ext_positions,
            forces=cs.ext_forces,
            state=st,
            batch_idx=cs.ext_batch_idx,
            measure_trust_region=False,
            **kwargs,
        )
        lbfgs_cell_trust_region(
            cs.ext_positions,
            st.direction,
            cs.phi,
            cs.d_phi,
            self.batch_idx,
            cs.ext_atom_ptr,
            cs.kappa,
            st.dmax,
            st.dquad,
        )
        lbfgs_prepare_step(st, maxstep=kwargs.get("maxstep", 0.2))
        lbfgs_apply_step(
            positions=cs.ext_positions,
            forces=cs.ext_forces,
            state=st,
            batch_idx=cs.ext_batch_idx,
        )
        self.unpack()
        wp.synchronize()

    def fmax_smax(self):
        """Largest force magnitude and stress component, as a caller would."""
        f = np.linalg.norm(self.forces.numpy(), axis=1).max(initial=0.0)
        return f, np.abs(self.stress.numpy()).max()

    def run(self, max_evals=400, force_tol=1e-6, stress_tol=1e-6, **kwargs):
        """Relax until both the forces and the stress are small enough."""
        for _ in range(max_evals):
            self.evaluate()
            f, s = self.fmax_smax()
            self.converged = f <= force_tol and s <= stress_tol
            if self.converged:
                break
            self.step(**kwargs)
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
            one.step(maxstep=0.2)
            two.evaluate()
            two.step_composed(maxstep=0.2)
            np.testing.assert_array_equal(one.positions.numpy(), two.positions.numpy())
            np.testing.assert_array_equal(one.cell.numpy(), two.cell.numpy())
        np.testing.assert_array_equal(
            one.state.iteration.numpy(), two.state.iteration.numpy()
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
        d.step(maxstep=0.2)

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
        assert d.converged, f"fmax/smax {d.fmax_smax()} after {d.n_evals} evaluations"
        volume = abs(np.linalg.det(d.cell.numpy()[0]))
        np.testing.assert_allclose(volume, potential.target_volume, rtol=1e-4)
        frac_final = (np.linalg.inv(d.cell.numpy()[0]) @ d.positions.numpy().T).T
        np.testing.assert_allclose(
            frac_final, np.broadcast_to(s0, frac.shape), atol=1e-5
        )


class TestLBFGSCellKappa:
    """The cell coordinate scaling, which is divided into the cell force."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_an_empty_system_is_rejected(self, device):
        """The variable-cell path has no scale to give a system with no atoms.

        ``kappa`` scales the cell coordinate against the atomic ones and is
        divided into the cell force, so an empty system has no value it could
        take. Substituting one would invent a number with no physical basis and
        make an unsupported configuration look valid, so the batch is rejected
        and the offending systems are named.
        """
        counts = wp.array(np.array([4, 0, 3], np.int32), dtype=wp.int32, device=device)
        kappa = wp.zeros(3, dtype=wp.float64, device=device)
        with pytest.raises(ValueError, match=r"system\(s\) \[1\] have no atoms"):
            lbfgs_cell_kappa(counts, kappa, cell_force_scale=0.25)

    @pytest.mark.parametrize("device", DEVICES)
    def test_populated_systems_keep_the_documented_scale(self, device):
        """Rejecting the empty case must not disturb the ordinary one."""
        counts = wp.array(np.array([4, 2, 3], np.int32), dtype=wp.int32, device=device)
        kappa = wp.zeros(3, dtype=wp.float64, device=device)
        lbfgs_cell_kappa(counts, kappa, cell_force_scale=0.25)
        wp.synchronize()
        np.testing.assert_allclose(kappa.numpy(), 0.25 * np.array([4, 2, 3]))

    @pytest.mark.parametrize("device", DEVICES)
    def test_an_atomless_system_is_rejected_by_the_topology_too(self, device):
        """Rejected even when ``kappa`` is filled separately.

        A system spanning exactly two packed entries has its two cell rows and
        no atoms, so preparation rejects it from the topology alone.
        """
        n, m = 4, 2
        # System 0 gets all four atoms; system 1 gets none.
        ptr = wp.array(np.array([0, 6, 8], np.int32), dtype=wp.int32, device=device)
        idx = wp.array(
            np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int32), dtype=wp.int32, device=device
        )
        with pytest.raises(ValueError, match="contain no atoms"):
            lbfgs_prepare_cell_state(n, m, idx, ptr, device=device)

    @pytest.mark.parametrize("device", DEVICES)
    def test_coordinate_only_empty_input_is_still_supported(self, device):
        """The separate, explicitly documented case: a no-op, not an error.

        There is no cell to scale against on the coordinate path, so zero
        degrees of freedom is simply nothing to do.
        """
        state = lbfgs_prepare_state(0, 1, device=device)
        lbfgs_step(
            positions=wp.zeros(0, dtype=wp.vec3d, device=device),
            forces=wp.zeros(0, dtype=wp.vec3d, device=device),
            state=state,
            batch_idx=wp.zeros(0, dtype=wp.int32, device=device),
        )
        wp.synchronize()


class TestLBFGSCellPrecision:
    """The variable-cell path at both coordinate precisions.

    Every cell test elsewhere in this file uses float64, which is how an fp32
    dispatch failure went unnoticed: the stress reduction was hard-coded to
    ``mat33d`` while the fp32 bindings pass ``mat33f``.
    """

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        ("vec", "mat", "np_dtype"),
        [
            pytest.param(wp.vec3f, wp.mat33f, np.float32, id="cell_f32"),
            pytest.param(wp.vec3d, wp.mat33d, np.float64, id="cell_f64"),
        ],
    )
    def test_variable_cell_step_dispatches(self, device, vec, mat, np_dtype):
        """A cell step runs, and the stress criterion sees the right norm."""
        n, num_systems = 4, 1
        rng = np.random.default_rng(0)
        cell_np = np.diag([6.0, 6.5, 7.0]).astype(np_dtype)
        # Diagonal stress, so the spectral norm is the largest entry.
        stress_np = np.diag([0.3, 0.2, 0.1]).astype(np_dtype)[None]

        st = make_lbfgs_state(n + 2 * num_systems, num_systems, HISTORY_SIZE,
                              vec, device)  # fmt: skip
        cs = make_lbfgs_cell_state(n, num_systems, vec, device)
        positions = wp.array(
            rng.normal(size=(n, 3)).astype(np_dtype), dtype=vec, device=device
        )
        forces = wp.array(
            (rng.normal(size=(n, 3)) * 0.1).astype(np_dtype), dtype=vec, device=device
        )
        cell = wp.array(cell_np[None], dtype=mat, device=device)
        stress = wp.array(stress_np, dtype=mat, device=device)
        batch_idx = wp.zeros(n, dtype=wp.int32, device=device)

        lbfgs_set_reference_cell(cell, cs.ref_cell, cs.ref_cell_inv)
        lbfgs_step_coord_cell(
            positions, forces, cell, stress, st, cs, batch_idx, maxstep=0.2,
        )  # fmt: skip
        wp.synchronize()

        # A non-zero stress must drive the cell, at either precision.
        assert np.isfinite(positions.numpy()).all()
        assert np.isfinite(cell.numpy()).all()
        assert not np.allclose(cell.numpy()[0], cell_np), "the cell did not move"


class TestLBFGSInputsAgainstState:
    """Fresh inputs are checked against the *state*, not only each other.

    An array agreeing with its neighbours but not with the prepared state
    still reaches a kernel typed for the state. Measured before this existed:
    a precision mismatch surfaced as a warp launch error, and a *device*
    mismatch as a **segmentation fault**.
    """

    @staticmethod
    def _call(device, positions=None, forces=None, batch_idx=None):
        st = lbfgs_prepare_state(4, 1, dtype=wp.vec3d, device=device)
        lbfgs_step(
            positions=positions
            if positions is not None
            else wp.zeros(4, dtype=wp.vec3d, device=device),
            forces=forces
            if forces is not None
            else wp.zeros(4, dtype=wp.vec3d, device=device),
            state=st,
            batch_idx=batch_idx
            if batch_idx is not None
            else wp.zeros(4, dtype=wp.int32, device=device),
        )
        wp.synchronize()

    @pytest.mark.parametrize("device", DEVICES)
    def test_accepts_matching_inputs(self, device):
        """The fixture must be valid, or the rest proves nothing."""
        self._call(device)

    @pytest.mark.parametrize("name", ["positions", "forces"])
    @pytest.mark.parametrize("device", DEVICES)
    def test_a_coordinate_precision_mismatch_is_rejected(self, device, name):
        with pytest.raises(ValueError, match="precision"):
            self._call(device, **{name: wp.zeros(4, dtype=wp.vec3f, device=device)})

    @pytest.mark.parametrize("device", DEVICES)
    def test_a_topology_dtype_mismatch_is_rejected(self, device):
        with pytest.raises(ValueError, match="batch_idx has dtype"):
            self._call(device, batch_idx=wp.zeros(4, dtype=wp.int64, device=device))

    @pytest.mark.parametrize("device", DEVICES)
    def test_an_input_on_the_wrong_device_is_rejected(self, device):
        """This one segfaulted rather than raising."""
        with pytest.raises(ValueError, match="is on cpu"):
            self._call(device, positions=wp.zeros(4, dtype=wp.vec3d, device="cpu"))


class TestLBFGSPackedTopology:
    """``ext_atom_ptr`` and ``ext_batch_idx`` values, checked at preparation.

    ``validate`` stops at shapes because it runs every step and reading these
    means a host sync. Preparation runs once, so it can afford the values --
    and every kernel on the cell path indexes the packed array through them.
    """

    @staticmethod
    def _topology(counts, device):
        from nvalchemiops.batch_utils import atom_ptr_to_batch_idx
        from nvalchemiops.dynamics.utils.cell_filter import extend_atom_ptr

        counts = np.asarray(counts, np.int32)
        n, m = int(counts.sum()), len(counts)
        atom_ptr = wp.array(
            np.concatenate([[0], np.cumsum(counts)]).astype(np.int32),
            dtype=wp.int32, device=device,
        )  # fmt: skip
        ext_ptr = wp.zeros(m + 1, dtype=wp.int32, device=device)
        extend_atom_ptr(atom_ptr, ext_ptr, device=device)
        ext_idx = wp.zeros(n + 2 * m, dtype=wp.int32, device=device)
        atom_ptr_to_batch_idx(ext_ptr, ext_idx)
        return n, m, ext_ptr, ext_idx

    @pytest.mark.parametrize("device", DEVICES)
    def test_a_ragged_topology_from_the_utilities_is_accepted(self, device):
        """Built the documented way, including unequal atom counts."""
        n, m, ptr, idx = self._topology([4, 7], device)
        lbfgs_prepare_cell_state(n, m, idx, ptr, device=device)

    @pytest.mark.parametrize("device", DEVICES)
    def test_a_batch_index_disagreeing_with_the_pointers_is_rejected(self, device):
        n, m, ptr, idx = self._topology([4, 7], device)
        reversed_idx = wp.array(idx.numpy()[::-1].copy(), dtype=wp.int32, device=device)
        with pytest.raises(ValueError, match="ext_batch_idx disagrees"):
            lbfgs_prepare_cell_state(n, m, reversed_idx, ptr, device=device)

    @pytest.mark.parametrize("device", DEVICES)
    def test_a_system_without_room_for_its_cell_rows_is_rejected(self, device):
        """Every system owns two packed entries however few atoms it has."""
        n, m, _, idx = self._topology([4, 7], device)
        bad = wp.array(np.array([0, 1, n + 2 * m], np.int32),
                       dtype=wp.int32, device=device)  # fmt: skip
        with pytest.raises(ValueError, match="fewer than 2 packed entries"):
            lbfgs_prepare_cell_state(n, m, idx, bad, device=device)


class TestLBFGSCellStepErrors:
    """The variable-cell entry point validates as thoroughly as the others.

    It is public, both states are plain dataclasses whose fields can be
    reassigned after preparation, and its kernels index a *packed* array built
    from three separate topologies -- so anything inconsistent that gets past
    here is an out-of-bounds read rather than an error. Measured before this
    was added: a short ``forces``, a short ``batch_idx`` and a state field left
    on the host all reached the kernels with no error at all.
    """

    @staticmethod
    def _inputs(device, n=4, m=1):
        cell = wp.array((np.eye(3) * 6.0)[None], dtype=wp.mat33d, device=device)
        npart = wp.array(np.array([n], np.int32), dtype=wp.int32, device=device)
        # The *extended* pointer spans the cell rows too, so it ends at
        # n + 2 * m rather than n. Passing the atom pointer here is a real
        # mistake that went unnoticed until the values were validated.
        ep = wp.array(np.array([0, n + 2 * m], np.int32), dtype=wp.int32, device=device)
        eb = wp.zeros(n + 2 * m, dtype=wp.int32, device=device)
        st = lbfgs_prepare_state(n + 2 * m, m, device=device)
        cs = lbfgs_prepare_cell_state(
            n, m, eb, ep, cell=cell, n_particles=npart, device=device
        )
        return dict(
            positions=wp.zeros(n, dtype=wp.vec3d, device=device),
            forces=wp.zeros(n, dtype=wp.vec3d, device=device),
            cell=cell,
            stress=wp.zeros(m, dtype=wp.mat33d, device=device),
            state=st,
            cell_state=cs,
            batch_idx=wp.zeros(n, dtype=wp.int32, device=device),
        )

    @staticmethod
    def _step(kw):
        lbfgs_step_coord_cell(
            kw["positions"], kw["forces"], kw["cell"], kw["stress"],
            kw["state"], kw["cell_state"], kw["batch_idx"], maxstep=0.2,
        )  # fmt: skip
        wp.synchronize()

    @pytest.mark.parametrize("device", DEVICES)
    def test_accepts_consistent_inputs(self, device):
        """The fixture itself must be valid, or the rest proves nothing."""
        self._step(self._inputs(device))

    @pytest.mark.parametrize("device", DEVICES)
    def test_short_forces_are_rejected(self, device):
        kw = self._inputs(device)
        kw["forces"] = wp.zeros(2, dtype=wp.vec3d, device=device)
        with pytest.raises(ValueError, match="forces has 2 entries"):
            self._step(kw)

    @pytest.mark.parametrize("device", DEVICES)
    def test_short_batch_idx_is_rejected(self, device):
        kw = self._inputs(device)
        kw["batch_idx"] = wp.zeros(2, dtype=wp.int32, device=device)
        with pytest.raises(ValueError, match="batch_idx has 2 entries"):
            self._step(kw)

    @pytest.mark.parametrize("device", DEVICES)
    def test_a_reassigned_history_buffer_is_rejected(self, device):
        kw = self._inputs(device)
        kw["state"].s_history = wp.zeros((6, 2), dtype=wp.vec3d, device=device)
        with pytest.raises(ValueError, match="s_history starts with dimensions"):
            self._step(kw)

    @pytest.mark.parametrize("device", DEVICES)
    def test_a_reassigned_scalar_precision_is_rejected(self, device):
        """Without this it surfaced as a raw warp launch error, not a ValueError."""
        kw = self._inputs(device)
        kw["state"].gg = wp.zeros(1, dtype=wp.float32, device=device)
        with pytest.raises(ValueError, match="must share one dtype"):
            self._step(kw)

    @pytest.mark.parametrize("device", DEVICES)
    def test_a_field_left_on_the_host_is_rejected(self, device):
        kw = self._inputs(device)
        kw["state"].d0 = wp.zeros(1, dtype=wp.float64, device="cpu")
        with pytest.raises(ValueError, match="spread across devices"):
            self._step(kw)

    @pytest.mark.parametrize("device", DEVICES)
    def test_a_reassigned_cell_scratch_buffer_is_rejected(self, device):
        kw = self._inputs(device)
        kw["cell_state"].ext_forces = wp.zeros(2, dtype=wp.vec3d, device=device)
        with pytest.raises(ValueError, match="ext_forces starts with dimensions"):
            self._step(kw)

    @pytest.mark.parametrize("device", DEVICES)
    def test_a_stress_for_the_wrong_system_count_is_rejected(self, device):
        kw = self._inputs(device)
        kw["stress"] = wp.zeros(3, dtype=wp.mat33d, device=device)
        with pytest.raises(ValueError, match="stress has 3 entries"):
            self._step(kw)


class TestLBFGSCellConvention:
    """The cell rules L-BFGS shares with FIRE2, pinned against FIRE2 itself.

    The contract in the module docstring claims the alignment and the
    six-component packing are *the same* as FIRE2's, not merely similar. That
    claim is only worth making if something checks it.
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_packs_the_same_six_components_as_fire2(self, device):
        """Same index set, same order, read out of FIRE2's own packer."""
        from nvalchemiops.dynamics.utils.cell_filter import pack_positions_with_cell

        n, m = 4, 1
        rng = np.random.default_rng(0)
        # An aligned cell with distinct entries, so a transposition would show.
        h = np.tril(rng.uniform(1.0, 2.0, size=(3, 3))) + np.eye(3) * 5.0
        cell = wp.array(h[None], dtype=wp.mat33d, device=device)
        positions = wp.array(rng.normal(size=(n, 3)), dtype=wp.vec3d, device=device)
        atom_ptr = wp.array(np.array([0, n], np.int32), dtype=wp.int32, device=device)
        ext_atom_ptr = wp.array(
            np.array([0, n + 2], np.int32), dtype=wp.int32, device=device
        )
        packed = wp.zeros(n + 2 * m, dtype=wp.vec3d, device=device)
        pack_positions_with_cell(
            positions, cell, packed, atom_ptr, ext_atom_ptr, device=device
        )
        fire2_rows = packed.numpy()[n : n + 2]

        # The same six entries the L-BFGS chart writes, in the same order.
        expected = np.array([[h[0, 0], h[1, 0], h[2, 0]], [h[1, 1], h[2, 1], h[2, 2]]])
        np.testing.assert_allclose(fire2_rows, expected, rtol=0, atol=0)

        # And the L-BFGS chart, with the reference set to the identity so that
        # Phi == H and kappa == 1, must produce those same six numbers.
        cs = make_lbfgs_cell_state(n, m, wp.vec3d, device)
        cs.ref_cell.assign(np.eye(3)[None])
        cs.ref_cell_inv.assign(np.eye(3)[None])
        cs.kappa.assign(np.ones(m))
        lbfgs_pack_cell(
            positions,
            wp.zeros(n, dtype=wp.vec3d, device=device),
            cell,
            wp.zeros(m, dtype=wp.mat33d, device=device),
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
        wp.synchronize()
        lbfgs_rows = np.stack([cs.cell_dof_a.numpy()[0], cs.cell_dof_b.numpy()[0]])
        np.testing.assert_allclose(lbfgs_rows, expected, rtol=1e-12, atol=0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_an_unaligned_cell_is_rejected_at_setup(self, device):
        """The packing only means what it says in the aligned frame.

        Measured: relaxing an unaligned cell leaves entries above the diagonal
        at O(1) -- a rotation the six components cannot remove. FIRE2 documents
        the same requirement; this one is checked, because preparation runs
        once and can afford the read.
        """
        unaligned = np.array([[5.0, 1.0, 2.0], [0.0, 6.0, 1.0], [0.0, 0.0, 7.0]])
        with pytest.raises(ValueError, match="not aligned"):
            check_cell_is_aligned(
                wp.array(unaligned[None], dtype=wp.mat33d, device=device)
            )
        aligned = np.tril(np.array([[5.0, 0.0, 0.0], [1.0, 6.0, 0.0], [2.0, 1.0, 7.0]]))
        check_cell_is_aligned(wp.array(aligned[None], dtype=wp.mat33d, device=device))

    @pytest.mark.parametrize("device", DEVICES)
    def test_align_cell_output_is_accepted(self, device):
        """Whatever ``align_cell`` produces must satisfy the check.

        Pins the two against each other, so a change to either is caught.
        """
        from nvalchemiops.dynamics.utils.cell_filter import align_cell

        rng = np.random.default_rng(3)
        h = rng.normal(size=(3, 3)) + np.eye(3) * 6.0
        positions = wp.array(rng.normal(size=(5, 3)), dtype=wp.vec3d, device=device)
        cell = wp.array(h[None], dtype=wp.mat33d, device=device)
        align_cell(cell=cell, positions=positions,
                   transform=wp.zeros(1, dtype=wp.mat33d, device=device))  # fmt: skip
        check_cell_is_aligned(cell)


class TestLBFGSRaggedVariableCell:
    """Variable-cell relaxation of a batch whose systems differ in size."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_ragged_batch_relaxes_every_system(self, device):
        """Each system reaches its own target volume.

        Nothing in the packed layout requires a uniform atom count: the
        extended CSR pointers and per-entry system index come from the generic
        batch utilities, so a ragged batch is expressible by construction.
        This would pass trivially with equal counts, so the counts are
        deliberately different and the cells are deliberately dissimilar.
        """
        counts = [4, 7]
        num_atoms, num_systems = sum(counts), len(counts)
        num_ext = num_atoms + 2 * num_systems

        rng = np.random.default_rng(3)
        potentials = [
            CellPotential(np.array([0.1, -0.2, 0.05]), p=0.35, c=140.0),
            CellPotential(np.array([-0.05, 0.15, 0.2]), p=0.5, c=180.0),
        ]
        cells_np = np.stack([np.diag([6.0, 6.5, 7.0]), np.diag([5.0, 5.4, 6.2])])
        blocks = []
        for s, n in enumerate(counts):
            frac = potentials[s].s0 + rng.normal(size=(n, 3)) * 0.05
            blocks.append((cells_np[s] @ frac.T).T)
        positions_np = np.ascontiguousarray(np.vstack(blocks))

        v, m = wp.vec3d, wp.mat33d
        positions = wp.array(positions_np, dtype=v, device=device)
        cell = wp.array(cells_np, dtype=m, device=device)
        forces = wp.zeros(num_atoms, dtype=v, device=device)
        stress = wp.zeros(num_systems, dtype=m, device=device)
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), counts).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )
        cell_state = make_lbfgs_cell_state(
            num_atoms, num_systems, v, device, counts=counts
        )
        lbfgs_set_reference_cell(cell, cell_state.ref_cell, cell_state.ref_cell_inv)
        state = make_lbfgs_state(num_ext, num_systems, HISTORY_SIZE, v, device)

        # The packed layout interleaves each system's atoms with its two cell
        # entries, so the pointers must follow the ragged counts.
        np.testing.assert_array_equal(
            cell_state.ext_atom_ptr.numpy(), [0, counts[0] + 2, num_ext]
        )

        offsets = np.concatenate([[0], np.cumsum(counts)])
        for _ in range(400):
            xs = positions.numpy()
            cells = cell.numpy()
            all_f = np.zeros_like(xs)
            all_s = np.zeros((num_systems, 3, 3))
            for s in range(num_systems):
                lo, hi = offsets[s], offsets[s + 1]
                _, all_f[lo:hi], all_s[s] = potentials[s].energy_forces_stress(
                    xs[lo:hi], cells[s]
                )
            forces.assign(np.ascontiguousarray(all_f))
            stress.assign(np.ascontiguousarray(all_s))

            lbfgs_step_coord_cell(
                positions,
                forces,
                cell,
                stress,
                state,
                cell_state,
                batch_idx,
                maxstep=0.2,
            )
            wp.synchronize()

        # Convergence is the caller's, so it is tested here rather than read
        # off a status array.
        fmax = np.array(
            [
                np.linalg.norm(all_f[offsets[s] : offsets[s + 1]], axis=1).max()
                for s in range(num_systems)
            ]
        )
        smax = np.abs(all_s).reshape(num_systems, -1).max(axis=1)
        assert (fmax <= 1e-6).all(), fmax
        assert (smax <= 1e-6).all(), smax
        volumes = np.abs(np.linalg.det(cell.numpy()))
        np.testing.assert_allclose(
            volumes,
            [p.target_volume for p in potentials],
            rtol=1e-4,
        )
