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

"""Tests for the PyTorch L-BFGS binding.

Binding behaviour only: the numerics belong to the Core suite, and are not
restated here. What is specific to PyTorch is that the state survives eager,
traced, compiled and captured execution intact.

Tests cover:

- The registered operators' schemas: which arguments they declare written,
  and that nothing read-only carries an alias. Compared as sets -- the names
  are the contract, the order they appear in is not.
- Tracing under ``make_fx`` and compilation under ``torch.compile``.
- CUDA-graph capture and replay, and that a step allocates nothing.
- Parity with the Warp layer over a few steps, comparing the complete public
  state each time rather than a long trajectory seen through positions alone.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
import torch

from nvalchemiops.torch.lbfgs import (
    lbfgs_prepare_cell_state,
    lbfgs_prepare_state,
    lbfgs_step_coord,
)

from ...conftest import (
    CELL_FIELDS,
    CELL_FIELDS_READ_ONLY,
    CELL_FIELDS_WRITTEN,
    STATE_FIELDS,
    assert_state_matches,
    per_system_scalar_fields,
)

DEVICES = ["cuda:0"]
DTYPES = [
    pytest.param(torch.float32, id="dof_f32"),
    pytest.param(torch.float64, id="dof_f64"),
]
STIFFNESS = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)


def make_torch_state(num_dofs, num_systems, dtype, device, history_size=6):
    """Return an :class:`LBFGSState` ready for the first step."""
    return lbfgs_prepare_state(
        num_dofs, num_systems, dtype=dtype, history_size=history_size, device=device
    )


def make_torch_cell_state(num_atoms, num_systems, dtype, device, counts=None):
    """Return an :class:`LBFGSCellState` with its topology and ``kappa`` filled.

    The extended topology is built with the generic batch utilities, which is
    what lets a ragged batch work; ``counts`` defaults to an even split.
    """
    import warp as wp

    from nvalchemiops.batch_utils import atom_ptr_to_batch_idx
    from nvalchemiops.dynamics.utils.cell_filter import extend_atom_ptr
    from nvalchemiops.torch.lbfgs import lbfgs_cell_kappa

    i32 = torch.int32
    num_ext = num_atoms + 2 * num_systems
    if counts is None:
        counts = [num_atoms // num_systems] * num_systems
    counts_np = np.asarray(counts, np.int32)
    assert int(counts_np.sum()) == num_atoms

    atom_ptr = wp.array(
        np.concatenate([[0], np.cumsum(counts_np)]).astype(np.int32),
        dtype=wp.int32,
        device=device,
    )
    ext_atom_ptr = torch.zeros(num_systems + 1, dtype=i32, device=device)
    ext_batch_idx = torch.zeros(num_ext, dtype=i32, device=device)
    extend_atom_ptr(
        atom_ptr, wp.from_torch(ext_atom_ptr, dtype=wp.int32), device=device
    )
    atom_ptr_to_batch_idx(
        wp.from_torch(ext_atom_ptr, dtype=wp.int32),
        wp.from_torch(ext_batch_idx, dtype=wp.int32),
    )

    state = lbfgs_prepare_cell_state(
        num_atoms, num_systems, ext_batch_idx, ext_atom_ptr, dtype=dtype, device=device
    )
    n_per_system = torch.tensor(counts_np, dtype=i32, device=device)
    lbfgs_cell_kappa(
        n_per_system, state.kappa, cell_force_scale=1.0 / float(counts_np.max())
    )
    return state


def _schema_of(op_name):
    """The schema PyTorch actually registered, not the tuple we handed it."""
    return getattr(torch.ops.nvalchemiops, op_name).default._schema


def _schema_writable(op_name):
    """Argument names the registered schema declares as written.

    Read from ``alias_info`` rather than from the private ``mutates_args``
    tuple. Those are the input and the output of registration: checking the
    input cannot catch a name that failed to reach the schema, and a state
    argument missing its write declaration is exactly what lets
    ``torch.compile`` treat that state as unchanged.
    """
    return tuple(
        a.name
        for a in _schema_of(op_name).arguments
        if a.alias_info is not None and a.alias_info.is_write
    )


def _snapshot(*states):
    """Clone every tensor field of the given dataclasses, keyed by name."""
    out = {}
    for prefix, state in states:
        for field in dataclasses.fields(state):
            value = getattr(state, field.name)
            out[f"{prefix}.{field.name}"] = value.clone()
    return out


def _assert_same_state(compiled, eager):
    """Every mutated tensor must agree exactly, not just the ones we plot."""
    assert compiled.keys() == eager.keys()
    differing = [name for name in eager if not torch.equal(compiled[name], eager[name])]
    assert not differing, (
        "compiled and eager execution left different state in "
        f"{differing}; a missing writable declaration lets torch.compile "
        "treat that argument as unchanged"
    )


class TorchDriver:
    """Relaxes an anisotropic quadratic through the Torch binding."""

    def __init__(self, positions, num_systems, dtype, device, history_size=6):
        self.device = device
        self.dtype = dtype
        self.num_dofs = positions.shape[0]
        self.num_systems = num_systems
        per_system = self.num_dofs // num_systems

        self.positions = torch.tensor(positions, dtype=dtype, device=device)
        self.forces = torch.zeros_like(self.positions)
        self.batch_idx = torch.repeat_interleave(
            torch.arange(num_systems, dtype=torch.int32, device=device), per_system
        )
        self.n_particles = torch.full(
            (num_systems,), per_system, dtype=torch.int32, device=device
        )
        self.state = make_torch_state(
            self.num_dofs, num_systems, dtype, device, history_size
        )
        self.stiffness = STIFFNESS.to(device)
        self.n_evals = 0

    def evaluate(self):
        x = self.positions.to(torch.float64)
        self.forces.copy_((-(self.stiffness * x)).to(self.dtype))
        self.n_evals += 1

    def step(self, **kwargs):
        lbfgs_step_coord(
            self.positions, self.forces, self.state, self.batch_idx, **kwargs
        )

    def fmax(self):
        """Largest per-atom force magnitude per system, as a caller would."""
        norms = self.forces.norm(dim=1).to(torch.float64)
        out = torch.zeros(self.num_systems, dtype=torch.float64, device=self.device)
        return out.scatter_reduce(0, self.batch_idx.long(), norms, reduce="amax")

    def run(self, max_evals=200, force_tol=1e-8, **kwargs):
        """Relax until every system is under ``force_tol``.

        Tested before stepping: the forces describe the current positions.
        """
        for _ in range(max_evals):
            self.evaluate()
            self.converged = self.fmax() <= force_tol
            if bool(self.converged.all()):
                break
            self.step(**kwargs)
        return self


def _cluster(num_systems, atoms_per_system, seed=42, scale=2.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(num_systems * atoms_per_system, 3)) * scale


class TestLBFGSTorchState:
    """The public surface, and the field order the operator schema relies on."""

    @pytest.mark.parametrize(
        "op_name,expected",
        [
            ("lbfgs_step", ("positions",) + STATE_FIELDS),
            (
                "lbfgs_step_coord_cell",
                ("positions", "cell") + STATE_FIELDS + CELL_FIELDS_WRITTEN,
            ),
        ],
    )
    def test_registered_schema_declares_the_writable_state(self, op_name, expected):
        """The *schema* must declare every mutated argument.

        Only this catches a name that never reached the schema -- under which
        ``torch.compile`` is entitled to assume that argument is unchanged and
        to reuse a stale value.

        Compared as a set: which arguments are declared written is the
        contract, the order they appear in is not, and pinning the order turns
        a harmless reordering of the state class into a failure.
        """
        assert set(_schema_writable(op_name)) == set(expected)

    @pytest.mark.parametrize(
        "op_name,read_only",
        [
            ("lbfgs_step", ("forces", "batch_idx")),
            (
                "lbfgs_step_coord_cell",
                ("forces", "stress", "batch_idx") + CELL_FIELDS_READ_ONLY,
            ),
        ],
    )
    def test_registered_schema_declares_nothing_else_writable(self, op_name, read_only):
        """Inputs the step only reads must carry no alias at all.

        Over-declaring is not free: the chart and topology arrays are shared
        across systems and never written, and claiming otherwise would block
        legitimate reuse and force needless copies.
        """
        by_name = {a.name: a for a in _schema_of(op_name).arguments}
        for name in read_only:
            assert by_name[name].alias_info is None, (
                f"{op_name} declares {name} as aliased, but the step only reads it"
            )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("history_size", [4, 6, 8])
    @pytest.mark.parametrize("num_dofs", [10_000, 100_000])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_memory_matches_the_documented_formula(
        self, device, history_size, num_dofs, dtype
    ):
        """Buffer size must match what the documentation promises.

        Callers size ``history_size`` against a memory budget and now allocate
        the state themselves, so the published formula has to stay true::

            (2m + 3) * 3 * sizeof(dof) * num_dofs   per-DOF vectors + s/y history
          + (4m + 6) * sizeof(dof) * num_systems    per-slot and per-system scalars
          +        4 * 4 * num_systems              per-system int32

        Every scalar follows the coordinate dtype, so ``sizeof(dof)`` appears
        in both of the first two terms.
        """
        num_systems = 8
        element_size = 4 if dtype == torch.float32 else 8
        expected = (
            (2 * history_size + 3) * 3 * element_size * num_dofs
            + (4 * history_size + 6) * element_size * num_systems
            + 4 * 4 * num_systems
        )

        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(device)
        state = make_torch_state(num_dofs, num_systems, dtype, device, history_size)
        torch.cuda.synchronize()
        measured = sum(
            getattr(state, f.name).numel() * getattr(state, f.name).element_size()
            for f in dataclasses.fields(state)
        )
        allocated = torch.cuda.memory_allocated(device) - before

        assert measured == expected, (
            f"state are {measured} bytes but the formula says {expected}; "
            "the documented memory model has drifted"
        )
        # The caching allocator rounds up, so compare loosely against it.
        assert allocated >= measured
        assert allocated <= measured * 1.05 + 4096, (
            f"allocator overhead {allocated - measured} bytes is larger than expected"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_bad_dtype_is_rejected(self, device):
        """Preparation refuses the dtype, so it cannot reach a kernel."""
        with pytest.raises(ValueError, match="float32 or float64"):
            lbfgs_prepare_state(3, 1, dtype=torch.float16, device=device)


class TestLBFGSTorchCoord:
    """Relaxation through the binding."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_matches_the_warp_layer(self, device):
        """The binding is a thin adapter, so it must agree exactly.

        A few steps, but the *whole* public state each time, rather than a
        long trajectory checked through positions alone: a binding that fails
        to pass an array through is visible only in that array, which a narrow
        comparison is by construction not looking at. Both sides run the same
        kernels in the same order, so this is exact rather than a tolerance.
        Convergence on this potential is Core's to certify, not the binding's.
        """
        import warp as wp

        from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_step as warp_step

        from ...conftest import make_lbfgs_state

        start = _cluster(1, 5, seed=17)
        torch_driver = TorchDriver(start, 1, torch.float64, device)

        num_dofs = start.shape[0]
        wp_positions = wp.array(start.copy(), dtype=wp.vec3d, device=device)
        wp_forces = wp.zeros(num_dofs, dtype=wp.vec3d, device=device)
        wp_batch = wp.zeros(num_dofs, dtype=wp.int32, device=device)
        wp_state = make_lbfgs_state(num_dofs, 1, 6, wp.vec3d, device)
        stiffness = STIFFNESS.numpy()

        for _ in range(3):
            torch_driver.evaluate()
            torch_driver.step(maxstep=0.5)

            x = wp_positions.numpy()
            wp_forces.assign(-(stiffness * x))
            warp_step(
                positions=wp_positions,
                forces=wp_forces,
                state=wp_state,
                batch_idx=wp_batch,
                maxstep=0.5,
            )
            wp.synchronize()
            torch.cuda.synchronize()
            np.testing.assert_array_equal(
                torch_driver.positions.cpu().numpy(), wp_positions.numpy()
            )
            assert_state_matches(torch_driver.state, wp_state, STATE_FIELDS)


class TestLBFGSTorchRegistration:
    """Tracing, compilation and graph capture."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_traces_as_a_custom_op(self, device):
        """``make_fx`` must see one opaque call with the full argument list.

        This is what catches a missing or wrong fake registration, and any
        drift between the schema and the call site.
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        d = TorchDriver(_cluster(1, 3), 1, torch.float64, device)
        d.evaluate()

        def run(positions, forces, batch_idx, *fields):
            # make_fx traces tensors, not dataclasses, so the state is rebuilt
            # here from the traced ones to keep every tensor in the graph.
            state = dataclasses.replace(
                d.state, **dict(zip(STATE_FIELDS, fields, strict=True))
            )
            lbfgs_step_coord(
                positions, forces, state, batch_idx,
                maxstep=0.5,
            )  # fmt: skip
            return positions

        graph = make_fx(run, tracing_mode="fake")(
            d.positions,
            d.forces,
            d.batch_idx,
            *(getattr(d.state, name) for name in STATE_FIELDS),
        )
        target = torch.ops.nvalchemiops.lbfgs_step.default
        nodes = [n for n in graph.graph.nodes if n.target is target]
        assert len(nodes) == 1, "the step did not trace as a single custom op"
        assert len(nodes[0].args) == len(target._schema.arguments)

    @pytest.mark.parametrize("device", DEVICES)
    def test_compiles_fullgraph_without_breaks(self, device):
        """``torch.compile(fullgraph=True)`` must succeed and match eager."""
        eager = TorchDriver(_cluster(1, 4), 1, torch.float64, device)
        eager.evaluate()
        eager.step(maxstep=0.5)
        torch.cuda.synchronize()

        compiled_driver = TorchDriver(_cluster(1, 4), 1, torch.float64, device)
        compiled_driver.evaluate()

        torch._dynamo.reset()

        def run(positions, forces):
            lbfgs_step_coord(
                positions,
                forces,
                compiled_driver.state,
                compiled_driver.batch_idx,
                maxstep=0.5,
            )

        torch.compile(run, fullgraph=True)(
            compiled_driver.positions, compiled_driver.forces
        )
        torch.cuda.synchronize()

        # The whole state, not just the positions: an argument torch.compile
        # believes is unchanged is by definition one a narrow comparison does
        # not look at.
        expected = _snapshot(("state", eager.state))
        expected["positions"] = eager.positions.clone()
        actual = _snapshot(("state", compiled_driver.state))
        actual["positions"] = compiled_driver.positions.clone()
        assert len(expected) == 1 + len(STATE_FIELDS)
        _assert_same_state(actual, expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_compiled_region_sees_the_state_change(self, device):
        """Clone state inside the graph, step, and diff -- the stale-value probe.

        Comparing state *after* a compiled call is a weak test of the write
        declarations: measured against a deliberately under-declared operator,
        the mutation still lands, because nothing gave Dynamo a reason to act
        on its wrong assumption. It acts when a value read *before* the call is
        reused *after* it -- then an undeclared argument is folded to its old
        value and the difference collapses to zero.

        So this clones inside the compiled region rather than outside, which is
        the shape that actually exercises the declaration.
        """
        d = TorchDriver(_cluster(1, 4), 1, torch.float64, device)
        d.evaluate()
        torch._dynamo.reset()

        def body(positions, forces):
            # ``iteration`` moves -1 -> 0 on the first step, and ``positions``
            # moves by the trust-region step.
            iteration_before = d.state.iteration.clone()
            positions_before = positions.clone()
            lbfgs_step_coord(positions, forces, d.state, d.batch_idx, maxstep=0.5)
            return (
                d.state.iteration - iteration_before,
                (positions - positions_before).abs().max(),
            )

        d_iter, d_pos = torch.compile(body, fullgraph=True)(d.positions, d.forces)
        torch.cuda.synchronize()
        assert int(d_iter.item()) == 1, (
            "the compiled graph did not observe the iteration counter advance; "
            "the operator's write declaration is not reaching torch.compile"
        )
        assert float(d_pos.item()) > 0.0, "the compiled graph saw no motion"

    @pytest.mark.parametrize("device", DEVICES)
    def test_compiles_with_zero_graph_breaks(self, device):
        """Assert the break count directly, not just that fullgraph succeeded.

        Also compiles a region containing a toy force model, which is how a
        caller would actually use it: the whole loop body should be one graph.
        """
        d = TorchDriver(_cluster(1, 4), 1, torch.float64, device)
        d.evaluate()

        def step_only(positions, forces):
            lbfgs_step_coord(
                positions,
                forces,
                d.state,
                d.batch_idx,
                maxstep=0.5,
            )

        def step_with_model(positions, forces):
            forces.copy_(-(d.stiffness * positions))
            lbfgs_step_coord(
                positions,
                forces,
                d.state,
                d.batch_idx,
                maxstep=0.5,
            )

        for fn in (step_only, step_with_model):
            torch._dynamo.reset()
            explanation = torch._dynamo.explain(fn)(d.positions, d.forces)
            assert explanation.graph_break_count == 0, (
                f"{fn.__name__} broke the graph {explanation.graph_break_count} times"
            )
            assert explanation.graph_count == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_step_allocates_nothing(self, device):
        """A pre-allocated state must not grow memory across steps."""
        d = TorchDriver(_cluster(1, 6), 1, torch.float64, device)
        for _ in range(3):  # warm up caching allocator and kernel cache
            d.evaluate()
            d.step(maxstep=0.5)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(device)
        for _ in range(10):
            d.evaluate()
            d.step(maxstep=0.5)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated(device) == before

    @pytest.mark.parametrize("device", DEVICES)
    def test_capture_needs_no_caller_stream_scope(self, device):
        """The documented claim: no ``wp.ScopedStream`` around the capture.

        The operator binds Warp to PyTorch's current stream itself, and during
        capture that is the capture stream. The docs previously told callers to
        wrap it and warned that otherwise "the capture records nothing" --
        which is false, and would have had them managing stream ownership they
        do not own.

        Asserted by capturing *without* any scope and checking replay does real
        work, not by inspecting internals.
        """
        d = TorchDriver(_cluster(1, 6, seed=5), 1, torch.float64, device)

        def step():
            d.evaluate()
            lbfgs_step_coord(d.positions, d.forces, d.state, d.batch_idx, maxstep=0.5)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):  # deliberately no wp.ScopedStream
            step()
        torch.cuda.synchronize()

        before = d.positions.clone()
        iteration_before = int(d.state.iteration.item())
        graph.replay()
        torch.cuda.synchronize()

        assert (d.positions - before).abs().max().item() > 0.0, (
            "replay did no work, so the capture recorded nothing"
        )
        assert int(d.state.iteration.item()) == iteration_before + 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_cuda_graph_replay_matches_eager(self, device):
        """The step captures in a non-empty CUDA graph and replays correctly.

        Only holds because every buffer is pre-allocated, all zeroing happens
        inside the operator, and Warp launches are bound to PyTorch's stream.
        Without the stream binding the capture would silently record nothing.
        """
        import warp as wp

        start = _cluster(1, 5, seed=23)
        eager = TorchDriver(start, 1, torch.float64, device)
        graphed = TorchDriver(start, 1, torch.float64, device)
        opts = dict(maxstep=0.5)

        # Identical warm-up on both so they enter the comparison in step.
        for driver in (eager, graphed):
            driver.evaluate()
            driver.step(**opts)
        torch.cuda.synchronize()
        wp.synchronize()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                graphed.step(**opts)
        torch.cuda.current_stream().wait_stream(side)
        for _ in range(3):
            eager.step(**opts)
        torch.cuda.synchronize()
        torch.testing.assert_close(graphed.positions, eager.positions)

        graph = torch.cuda.CUDAGraph()
        with warnings.catch_warnings():
            # An empty capture is only a warning, so make it fail the test.
            warnings.simplefilter("error", UserWarning)
            # Warp launches must be bound to the capture stream by the caller.
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
                with torch.cuda.graph(graph):
                    graphed.step(**opts)
        torch.cuda.synchronize()

        eager.step(**opts)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graphed.positions, eager.positions)
        assert torch.isfinite(graphed.positions).all()


class TestLBFGSTorchErrors:
    """Input validation."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_force_shape_mismatch(self, device):
        d = TorchDriver(_cluster(1, 3), 1, torch.float64, device)
        d.evaluate()
        with pytest.raises(ValueError, match="forces shape"):
            lbfgs_step_coord(
                d.positions,
                torch.zeros(2, 3, dtype=torch.float64, device=device),
                d.state,
                d.batch_idx,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_non_contiguous_input_is_rejected(self, device):
        """A non-contiguous tensor must raise, not be silently copied.

        These operations write through a zero-copy view, so copying the input
        would discard every update and leave the caller watching an optimizer
        that never moves. Loud beats silent.
        """
        d = TorchDriver(_cluster(1, 4), 1, torch.float64, device)
        d.evaluate()
        # A strided view is the usual way to end up here by accident.
        strided = torch.zeros(4, 6, dtype=torch.float64, device=device)[:, ::2]
        assert not strided.is_contiguous()
        with pytest.raises(ValueError, match="must be contiguous"):
            lbfgs_step_coord(
                strided,
                d.forces,
                d.state,
                d.batch_idx,
            )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_scalars_follow_the_coordinate_dtype(self, device, dtype):
        """An fp32 state is fp32 throughout, so fp32 needs no fp64 arithmetic."""
        st = lbfgs_prepare_state(3, 1, dtype=dtype, device=device)
        for name in per_system_scalar_fields(st, 3):
            assert getattr(st, name).dtype == dtype, name

    @pytest.mark.parametrize("device", DEVICES)
    def test_history_size_mismatch(self, device):
        d = TorchDriver(_cluster(1, 3), 1, torch.float64, device)
        d.evaluate()
        other = make_torch_state(9, 1, torch.float64, device)
        with pytest.raises(ValueError, match="prepared for 9"):
            lbfgs_step_coord(
                d.positions,
                d.forces,
                other,
                d.batch_idx,
            )


class TestLBFGSTorchCoordCell:
    """The variable-cell binding."""

    @staticmethod
    def _setup(device, num_atoms=6, seed=11):
        """A compressed cell with perturbed fractional coordinates."""
        from nvalchemiops.torch.lbfgs import lbfgs_set_reference_cell

        from ...conftest import CellPotential

        rng = np.random.default_rng(seed)
        s0 = np.array([0.1, -0.2, 0.05])
        potential = CellPotential(s0)
        cell_np = np.diag([6.0, 6.5, 7.0])
        frac = s0 + rng.normal(size=(num_atoms, 3)) * 0.05
        positions_np = np.ascontiguousarray((cell_np @ frac.T).T)

        positions = torch.tensor(positions_np, dtype=torch.float64, device=device)
        cell = torch.tensor(cell_np[None], dtype=torch.float64, device=device)
        cell_state = make_torch_cell_state(num_atoms, 1, torch.float64, device)
        lbfgs_set_reference_cell(cell, cell_state.ref_cell, cell_state.ref_cell_inv)
        # The cell contributes two packed entries per system.
        state = make_torch_state(num_atoms + 2, 1, torch.float64, device)
        return positions, cell, state, cell_state, potential

    @staticmethod
    def _evaluate(positions, cell, potential, device):
        _, f, s = potential.energy_forces_stress(
            positions.cpu().numpy(), cell.cpu().numpy()[0]
        )
        return (
            torch.tensor(f, dtype=torch.float64, device=device),
            torch.tensor(s[None], dtype=torch.float64, device=device),
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_setup_helpers_bind_warp_to_the_torch_stream(self, device, monkeypatch):
        """Every Warp launch from this module must be stream-scoped.

        Unbound, a launch sits on Warp's own stream. That happens to serialize
        against PyTorch's legacy default stream, so the defect is invisible in
        ordinary use -- but on a non-default stream nothing orders it against
        the producer of its inputs or the consumer of its outputs.

        Asserted structurally rather than by racing: the race is real but
        latent, and a timing-based test passes just as happily without the fix,
        which would be worse than no test at all.
        """
        import nvalchemiops.torch.lbfgs as module

        entered = []
        original = module.scoped_warp_stream

        def spy(dev):
            entered.append(dev)
            return original(dev)

        monkeypatch.setattr(module, "scoped_warp_stream", spy)

        cell = torch.eye(3, dtype=torch.float64, device=device).expand(2, 3, 3)
        cell = cell.contiguous()
        ref_cell, ref_cell_inv = torch.zeros_like(cell), torch.zeros_like(cell)
        module.lbfgs_set_reference_cell(cell, ref_cell, ref_cell_inv)
        assert entered, "lbfgs_set_reference_cell launched Warp unscoped"

        entered.clear()
        counts = torch.full((2,), 8, dtype=torch.int32, device=device)
        kappa = torch.zeros(2, dtype=torch.float64, device=device)
        module.lbfgs_cell_kappa(counts, kappa, cell_force_scale=0.5)
        assert entered, "lbfgs_cell_kappa launched Warp unscoped"

        # ...and the results are still right when driven from a side stream.
        torch.cuda.synchronize()
        torch.testing.assert_close(ref_cell, cell)
        torch.testing.assert_close(kappa, torch.full_like(kappa, 4.0))

    @pytest.mark.parametrize("device", DEVICES)
    def test_matches_the_warp_layer(self, device):
        """The binding is a thin adapter, so it must agree exactly.

        As in the coordinate case: a few steps, both public states compared in
        full each time.
        """
        import warp as wp

        from nvalchemiops.dynamics.optimizers.lbfgs import (
            lbfgs_set_reference_cell as warp_set_ref,
        )
        from nvalchemiops.dynamics.optimizers.lbfgs import (
            lbfgs_step_coord_cell as warp_step_cell,
        )
        from nvalchemiops.torch.lbfgs import lbfgs_step_coord_cell

        from ...conftest import make_lbfgs_cell_state, make_lbfgs_state

        n = 6
        positions, cell, state, cell_state, potential = self._setup(device, n)
        batch_idx = torch.zeros(n, dtype=torch.int32, device=device)

        start_pos = positions.cpu().numpy().copy()
        start_cell = cell.cpu().numpy()[0].copy()
        wp_pos = wp.array(start_pos, dtype=wp.vec3d, device=device)
        wp_cell = wp.array(start_cell[None], dtype=wp.mat33d, device=device)
        wp_forces = wp.zeros(n, dtype=wp.vec3d, device=device)
        wp_stress = wp.zeros(1, dtype=wp.mat33d, device=device)
        wp_batch = wp.zeros(n, dtype=wp.int32, device=device)
        wp_cell_state = make_lbfgs_cell_state(n, 1, wp.vec3d, device)
        warp_set_ref(wp_cell, wp_cell_state.ref_cell, wp_cell_state.ref_cell_inv)
        wp_state = make_lbfgs_state(n + 2, 1, 6, wp.vec3d, device)

        for _ in range(3):
            forces, stress = self._evaluate(positions, cell, potential, device)
            lbfgs_step_coord_cell(
                positions,
                cell,
                forces,
                stress,
                state,
                cell_state,
                batch_idx,
                maxstep=0.2,
            )

            _, f, s = potential.energy_forces_stress(wp_pos.numpy(), wp_cell.numpy()[0])
            wp_forces.assign(f)
            wp_stress.assign(s[None])
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
            torch.cuda.synchronize()
            np.testing.assert_array_equal(positions.cpu().numpy(), wp_pos.numpy())
            np.testing.assert_array_equal(cell.cpu().numpy(), wp_cell.numpy())
            assert_state_matches(state, wp_state, STATE_FIELDS, "optimizer ")
            assert_state_matches(cell_state, wp_cell_state, CELL_FIELDS, "cell ")

    @pytest.mark.parametrize("device", DEVICES)
    def test_state_sized_for_the_wrong_dof_count_is_rejected(self, device):
        """The cell adds two degrees of freedom per system; coordinate-sized
        state would silently under-cover the packed array."""
        from nvalchemiops.torch.lbfgs import lbfgs_step_coord_cell

        n = 6
        positions, cell, _, cell_state, potential = self._setup(device, n)
        wrong = make_torch_state(n, 1, torch.float64, device)
        forces, stress = self._evaluate(positions, cell, potential, device)
        with pytest.raises(ValueError, match="num_atoms \\+ 2 \\* num_systems"):
            lbfgs_step_coord_cell(
                positions,
                cell,
                forces,
                stress,
                wrong,
                cell_state,
                torch.zeros(n, dtype=torch.int32, device=device),
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_compiled_run_leaves_identical_state(self, device):
        """Every mutated tensor must match after compilation, not just ``cell``.

        Comparing a handful of outputs cannot distinguish a correctly declared
        state from one ``torch.compile`` believes is unchanged: the arguments
        it is free to treat as read-only are exactly the ones a narrow
        comparison never looks at.
        """
        from nvalchemiops.torch.lbfgs import lbfgs_step_coord_cell

        n = 6
        batch_idx = torch.zeros(n, dtype=torch.int32, device=device)

        def one_run(compiled):
            positions, cell, state, cell_state, potential = self._setup(device, n)
            forces, stress = self._evaluate(positions, cell, potential, device)

            def body(p, c, f, s):
                lbfgs_step_coord_cell(
                    p, c, f, s, state, cell_state, batch_idx, maxstep=0.2
                )

            torch._dynamo.reset()
            fn = torch.compile(body, fullgraph=True) if compiled else body
            fn(positions, cell, forces, stress)
            torch.cuda.synchronize()
            snap = _snapshot(("state", state), ("cell_state", cell_state))
            snap["positions"] = positions.clone()
            snap["cell"] = cell.clone()
            return snap

        eager = one_run(False)
        compiled = one_run(True)
        # 2 tensors + 19 optimizer fields + 14 cell fields.
        assert len(eager) == 2 + len(STATE_FIELDS) + len(CELL_FIELDS)
        _assert_same_state(compiled, eager)

    @pytest.mark.parametrize("device", DEVICES)
    def test_compiled_cell_region_sees_the_state_change(self, device):
        """The stale-value probe for the variable-cell operator.

        Same shape as the coordinate test: clone inside the graph, step, diff.
        This is what exercises the write declarations, as distinct from
        comparing state after the call.
        """
        from nvalchemiops.torch.lbfgs import lbfgs_step_coord_cell

        n = 6
        positions, cell, state, cell_state, potential = self._setup(device, n)
        forces, stress = self._evaluate(positions, cell, potential, device)
        batch_idx = torch.zeros(n, dtype=torch.int32, device=device)
        torch._dynamo.reset()

        def body(p, c, f, s):
            iteration_before = state.iteration.clone()
            cell_before = c.clone()
            scratch_before = cell_state.ext_positions.clone()
            lbfgs_step_coord_cell(p, c, f, s, state, cell_state, batch_idx, maxstep=0.2)
            return (
                state.iteration - iteration_before,
                (c - cell_before).abs().max(),
                (cell_state.ext_positions - scratch_before).abs().max(),
            )

        d_iter, d_cell, d_scratch = torch.compile(body, fullgraph=True)(
            positions, cell, forces, stress
        )
        torch.cuda.synchronize()
        assert int(d_iter.item()) == 1, "iteration change invisible to the graph"
        assert float(d_cell.item()) > 0.0, "cell change invisible to the graph"
        # A scratch buffer, to cover the written half of the cell schema.
        assert float(d_scratch.item()) > 0.0, "scratch change invisible to the graph"

    @pytest.mark.parametrize("device", DEVICES)
    def test_compiles_fullgraph_and_matches_eager(self, device):
        """The variable-cell step compiles as one graph and agrees with eager.

        Each run gets its own freshly allocated state, since the step mutates
        it; comparing against a run whose state had already advanced would be
        meaningless.
        """
        from nvalchemiops.torch.lbfgs import lbfgs_step_coord_cell

        n = 6
        opts = dict(maxstep=0.2)
        batch_idx = torch.zeros(n, dtype=torch.int32, device=device)

        def one_run(compiled):
            positions, cell, state, cell_state, potential = self._setup(device, n)
            forces, stress = self._evaluate(positions, cell, potential, device)

            def body(p, c, f, s):
                lbfgs_step_coord_cell(
                    p,
                    c,
                    f,
                    s,
                    state,
                    cell_state,
                    batch_idx,
                    **opts,
                )

            torch._dynamo.reset()
            fn = torch.compile(body, fullgraph=True) if compiled else body
            fn(positions, cell, forces, stress)
            torch.cuda.synchronize()
            return positions.clone(), cell.clone(), state.iteration.clone()

        eager = one_run(False)
        compiled = one_run(True)
        torch.testing.assert_close(compiled[0], eager[0], rtol=0, atol=0)
        torch.testing.assert_close(compiled[1], eager[1], rtol=0, atol=0)
        torch.testing.assert_close(compiled[2], eager[2])
