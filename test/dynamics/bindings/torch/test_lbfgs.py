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

Tests cover:

- The registered operator's schema, and that its parameters stay in the
  canonical caller-owned buffer order.
- That the removed state containers and allocators really are gone.
- Tracing under ``make_fx`` and compilation under ``torch.compile``.
- CUDA-graph capture and replay, and that a step allocates nothing.
- Relaxation results, checked against the Warp layer rather than restating the
  algorithm.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
import torch

from nvalchemiops.dynamics.optimizers.lbfgs import (
    _CELL_BUFFERS,
    _OPTIMIZER_BUFFERS,
)
from nvalchemiops.torch.lbfgs import (
    LBFGS_CONVERGED,
    LBFGS_NEED_EVAL,
    lbfgs_step_coord,
)

DEVICES = ["cuda:0"]
DTYPES = [
    pytest.param(torch.float32, id="dof_f32"),
    pytest.param(torch.float64, id="dof_f64"),
]
STIFFNESS = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)


def make_torch_buffers(num_dofs, num_systems, dtype, device, history_size=6):
    """Allocate the caller-owned optimizer buffers, ready for the first step.

    The package allocates and initializes nothing, so this is a test-only
    factory. Keyed by name in the canonical order, so it can be splatted
    positionally with ``*buffers.values()``.
    """
    f64, i32 = torch.float64, torch.int32

    def z(*shape, dt=f64):
        return torch.zeros(shape, dtype=dt, device=device)

    buffers = {
        "x_base": z(num_dofs, 3, dt=dtype),
        "force_base": z(num_dofs, 3, dt=dtype),
        "direction": z(num_dofs, 3, dt=dtype),
        "s_history": z(history_size, num_dofs, 3, dt=dtype),
        "y_history": z(history_size, num_dofs, 3, dt=dtype),
        "ys": z(history_size, num_systems),
        "yy": z(history_size, num_systems),
        "alpha_hist": z(history_size, num_systems),
        "beta_hist": z(history_size, num_systems),
        "ss": z(num_systems),
        "gg": z(num_systems),
        "fmax": z(num_systems),
        "frms_sq": z(num_systems),
        "smax": z(num_systems),
        "d0": z(num_systems),
        "dmax": z(num_systems),
        "dquad": z(num_systems),
        "alpha_step": z(num_systems),
        "status": z(num_systems, dt=i32),
        "iteration": z(num_systems, dt=i32),
        "end": z(num_systems, dt=i32),
        "n_loop": z(num_systems, dt=i32),
        "history_count": z(num_systems, dt=i32),
    }
    assert tuple(buffers) == _OPTIMIZER_BUFFERS, "factory drifted from buffer order"
    # Exactly three buffers do not start at zero.
    buffers["alpha_step"].fill_(1.0)
    buffers["iteration"].fill_(-1)
    buffers["status"].fill_(LBFGS_NEED_EVAL)
    return buffers


def make_torch_cell_buffers(num_atoms, num_systems, dtype, device, counts=None):
    """Allocate the caller-owned variable-cell buffers, keyed by name.

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

    def z(*shape):
        return torch.zeros(shape, dtype=dtype, device=device)

    kappa = z(num_systems)
    n_per_system = torch.tensor(counts_np, dtype=i32, device=device)
    lbfgs_cell_kappa(n_per_system, kappa, cell_force_scale=1.0 / float(counts_np.max()))

    buffers = {
        "ref_cell": z(num_systems, 3, 3),
        "ref_cell_inv": z(num_systems, 3, 3),
        "kappa": kappa,
        "ext_batch_idx": ext_batch_idx,
        "ext_atom_ptr": ext_atom_ptr,
        "phi": z(num_systems, 3, 3),
        "phi_inv": z(num_systems, 3, 3),
        "d_phi": z(num_systems, 3, 3),
        "cell_dof_a": z(num_systems, 3),
        "cell_dof_b": z(num_systems, 3),
        "cell_force_a": z(num_systems, 3),
        "cell_force_b": z(num_systems, 3),
        "ext_positions": z(num_ext, 3),
        "ext_forces": z(num_ext, 3),
    }
    assert tuple(buffers) == _CELL_BUFFERS, "factory drifted from buffer order"
    return buffers


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
        self.buffers = make_torch_buffers(
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
            self.positions,
            self.forces,
            self.batch_idx,
            self.n_particles,
            *self.buffers.values(),
            **kwargs,
        )

    @property
    def status(self):
        return self.buffers["status"]

    def run(self, max_evals=200, **kwargs):
        for _ in range(max_evals):
            self.evaluate()
            self.step(**kwargs)
            torch.cuda.synchronize()
            if not (self.buffers["status"] == LBFGS_NEED_EVAL).any():
                break
        return self


def _cluster(num_systems, atoms_per_system, seed=42, scale=2.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(num_systems * atoms_per_system, 3)) * scale


class TestLBFGSTorchState:
    """The public surface, and the buffer order every caller relies on."""

    def test_removed_state_api_is_absent(self):
        """The state containers and allocators are gone, with no shims.

        Buffers are caller-owned, so anything that allocated or owned them on
        the caller's behalf must not quietly survive as an alias. An import
        that still resolves would let old code keep working against an API
        that no longer has the semantics it assumes.
        """
        import nvalchemiops.dynamics.optimizers as warp_optimizers
        import nvalchemiops.jax.lbfgs as jax_lbfgs
        import nvalchemiops.torch as torch_pkg
        import nvalchemiops.torch.lbfgs as torch_lbfgs
        from nvalchemiops.dynamics.optimizers import lbfgs as warp_lbfgs

        removed = (
            "LBFGSState",
            "LBFGSCellState",
            "lbfgs_allocate_state",
            "lbfgs_allocate_cell_state",
            "lbfgs_reset",
            "lbfgs_reduce_energy",
            "LBFGS_LS_FAILED",
        )
        modules = (
            warp_lbfgs,
            warp_optimizers,
            torch_lbfgs,
            torch_pkg,
            jax_lbfgs,
        )
        for module in modules:
            for name in removed:
                assert not hasattr(module, name), (
                    f"{module.__name__}.{name} still exists; caller-owned buffers "
                    "were meant to replace it outright"
                )
                assert name not in getattr(module, "__all__", ()), (
                    f"{module.__name__}.__all__ still exports {name}"
                )

    def test_operator_parameters_match_the_buffer_order(self):
        """A reordering here would silently swap two tensors at the boundary.

        Registration already rejects a name in ``mutates_args`` that does not
        exist as a parameter, but only an ordered comparison catches a swap,
        and every caller now passes these 26 tensors positionally.
        """
        from nvalchemiops.torch.lbfgs import _lbfgs_step_op

        params = tuple(inspect.signature(_lbfgs_step_op).parameters)
        offset = 4  # positions, forces, batch_idx, n_particles
        assert params[offset : offset + len(_OPTIMIZER_BUFFERS)] == _OPTIMIZER_BUFFERS

    def test_public_wrapper_takes_the_buffers_in_the_same_order(self):
        """The wrapper and the operator must not drift apart.

        The wrapper forwards positionally, so a mismatch would pass every
        shape check and then compute nonsense.
        """
        params = tuple(inspect.signature(lbfgs_step_coord).parameters)
        offset = 4  # positions, forces, batch_idx, n_particles
        assert params[offset : offset + len(_OPTIMIZER_BUFFERS)] == _OPTIMIZER_BUFFERS

    def test_mutates_args_covers_every_buffer(self):
        """Every buffer is declared mutable, and nothing else is."""
        from nvalchemiops.torch.lbfgs import _MUTATED

        assert set(_MUTATED) == {"positions"} | set(_OPTIMIZER_BUFFERS)
        assert len(_MUTATED) == len(_OPTIMIZER_BUFFERS) + 1

    def test_schema_arity_matches_the_signature(self):
        """The registered schema sees every argument the implementation takes."""
        from nvalchemiops.torch.lbfgs import _lbfgs_step_op

        schema = torch.ops.nvalchemiops.lbfgs_step.default._schema
        assert len(schema.arguments) == len(
            inspect.signature(_lbfgs_step_op).parameters
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
        the buffers themselves, so the published formula has to stay true::

            (2m + 3) * 3 * sizeof(dof) * num_dofs   per-DOF vectors + s/y history
          + (4m + 9) * 8 * num_systems              per-slot and per-system float64
          +        5 * 4 * num_systems              per-system int32
        """
        num_systems = 8
        element_size = 4 if dtype == torch.float32 else 8
        expected = (
            (2 * history_size + 3) * 3 * element_size * num_dofs
            + (4 * history_size + 9) * 8 * num_systems
            + 5 * 4 * num_systems
        )

        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(device)
        buffers = make_torch_buffers(num_dofs, num_systems, dtype, device, history_size)
        torch.cuda.synchronize()
        measured = sum(b.numel() * b.element_size() for b in buffers.values())
        allocated = torch.cuda.memory_allocated(device) - before

        assert measured == expected, (
            f"buffers are {measured} bytes but the formula says {expected}; "
            "the documented memory model has drifted"
        )
        # The caching allocator rounds up, so compare loosely against it.
        assert allocated >= measured
        assert allocated <= measured * 1.05 + 4096, (
            f"allocator overhead {allocated - measured} bytes is larger than expected"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_bad_dtype_is_rejected(self, device):
        d = TorchDriver(_cluster(1, 3), 1, torch.float64, device)
        d.evaluate()
        half = make_torch_buffers(3, 1, torch.float16, device)
        with pytest.raises(ValueError, match="float32 or float64"):
            lbfgs_step_coord(
                d.positions.to(torch.float16),
                d.forces.to(torch.float16),
                d.batch_idx,
                d.n_particles,
                *half.values(),
            )


class TestLBFGSTorchCoord:
    """Relaxation through the binding."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_reaches_the_minimum(self, device, dtype):
        force_tol = 1e-3 if dtype == torch.float32 else 1e-8
        d = TorchDriver(_cluster(2, 4), 2, dtype, device).run(
            force_tol=force_tol, maxstep=0.5
        )
        assert (d.buffers["status"] == LBFGS_CONVERGED).all(), d.buffers["status"]
        assert d.positions.abs().max().item() < 10 * force_tol

    @pytest.mark.parametrize("device", DEVICES)
    def test_matches_the_warp_layer(self, device):
        """The binding is a thin adapter, so it must agree exactly.

        Compared step by step against the Warp implementation on identical
        inputs. Both run the same kernels in the same order, so this is an
        exact comparison rather than a tolerance check.
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
        wp_nparts = wp.array(
            np.array([num_dofs], np.int32), dtype=wp.int32, device=device
        )
        wp_state = make_lbfgs_state(num_dofs, 1, 6, wp.vec3d, device)
        stiffness = STIFFNESS.numpy()

        for _ in range(30):
            torch_driver.evaluate()
            torch_driver.step(force_tol=1e-8, maxstep=0.5)

            x = wp_positions.numpy()
            wp_forces.assign(-(stiffness * x))
            warp_step(
                positions=wp_positions,
                forces=wp_forces,
                batch_idx=wp_batch,
                n_particles=wp_nparts,
                force_tol=1e-8,
                maxstep=0.5,
                **wp_state,
            )
            wp.synchronize()
            torch.cuda.synchronize()
            np.testing.assert_array_equal(
                torch_driver.positions.cpu().numpy(), wp_positions.numpy()
            )
            if not (torch_driver.buffers["status"] == LBFGS_NEED_EVAL).any():
                break
        np.testing.assert_array_equal(
            torch_driver.buffers["status"].cpu().numpy(), wp_state["status"].numpy()
        )


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

        def run(positions, forces, batch_idx, n_particles, *buffers):
            lbfgs_step_coord(
                positions,
                forces,
                batch_idx,
                n_particles,
                *buffers,
                force_tol=1e-8,
                maxstep=0.5,
            )
            return positions

        graph = make_fx(run, tracing_mode="fake")(
            d.positions,
            d.forces,
            d.batch_idx,
            d.n_particles,
            *d.buffers.values(),
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
        eager.step(force_tol=1e-8, maxstep=0.5)
        torch.cuda.synchronize()

        compiled_driver = TorchDriver(_cluster(1, 4), 1, torch.float64, device)
        compiled_driver.evaluate()

        torch._dynamo.reset()

        def run(positions, forces):
            lbfgs_step_coord(
                positions,
                forces,
                compiled_driver.batch_idx,
                compiled_driver.n_particles,
                *compiled_driver.buffers.values(),
                force_tol=1e-8,
                maxstep=0.5,
            )

        torch.compile(run, fullgraph=True)(
            compiled_driver.positions, compiled_driver.forces
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(compiled_driver.positions, eager.positions)

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
                d.batch_idx,
                d.n_particles,
                *d.buffers.values(),
                force_tol=1e-8,
                maxstep=0.5,
            )

        def step_with_model(positions, forces):
            forces.copy_(-(d.stiffness * positions))
            lbfgs_step_coord(
                positions,
                forces,
                d.batch_idx,
                d.n_particles,
                *d.buffers.values(),
                force_tol=1e-8,
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
            d.step(force_tol=1e-10, maxstep=0.5)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(device)
        for _ in range(10):
            d.evaluate()
            d.step(force_tol=1e-10, maxstep=0.5)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated(device) == before

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
        opts = dict(force_tol=1e-12, maxstep=0.5)

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
                d.batch_idx,
                d.n_particles,
                *d.buffers.values(),
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
                d.batch_idx,
                d.n_particles,
                *d.buffers.values(),
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_history_size_mismatch(self, device):
        d = TorchDriver(_cluster(1, 3), 1, torch.float64, device)
        d.evaluate()
        other = make_torch_buffers(9, 1, torch.float64, device)
        with pytest.raises(ValueError, match="history buffers"):
            lbfgs_step_coord(
                d.positions,
                d.forces,
                d.batch_idx,
                d.n_particles,
                *other.values(),
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
        cell_buffers = make_torch_cell_buffers(num_atoms, 1, torch.float64, device)
        lbfgs_set_reference_cell(
            cell, cell_buffers["ref_cell"], cell_buffers["ref_cell_inv"]
        )
        # The cell contributes two packed entries per system.
        buffers = make_torch_buffers(num_atoms + 2, 1, torch.float64, device)
        return positions, cell, buffers, cell_buffers, potential

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
    def test_relaxes_cell_and_coordinates(self, device):
        """A compressed cell expands to the target volume while atoms relax."""
        from nvalchemiops.torch.lbfgs import lbfgs_step_coord_cell

        n = 6
        positions, cell, buffers, cell_buffers, potential = self._setup(device, n)
        batch_idx = torch.zeros(n, dtype=torch.int32, device=device)
        n_particles = torch.full((1,), n, dtype=torch.int32, device=device)
        status = buffers["status"]

        for _ in range(400):
            forces, stress = self._evaluate(positions, cell, potential, device)
            lbfgs_step_coord_cell(
                positions,
                cell,
                forces,
                stress,
                batch_idx,
                n_particles,
                *buffers.values(),
                *cell_buffers.values(),
                force_tol=1e-6,
                stress_tol=1e-6,
                maxstep=0.2,
            )
            torch.cuda.synchronize()
            if status.item() != LBFGS_NEED_EVAL:
                break

        assert status.item() == LBFGS_CONVERGED, status.item()
        volume = abs(np.linalg.det(cell.cpu().numpy()[0]))
        np.testing.assert_allclose(volume, potential.target_volume, rtol=1e-4)

    @pytest.mark.parametrize("device", DEVICES)
    def test_matches_the_warp_layer(self, device):
        """The binding is a thin adapter, so it must agree exactly."""
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
        positions, cell, buffers, cell_buffers, potential = self._setup(device, n)
        batch_idx = torch.zeros(n, dtype=torch.int32, device=device)
        n_particles = torch.full((1,), n, dtype=torch.int32, device=device)
        status = buffers["status"]

        start_pos = positions.cpu().numpy().copy()
        start_cell = cell.cpu().numpy()[0].copy()
        wp_pos = wp.array(start_pos, dtype=wp.vec3d, device=device)
        wp_cell = wp.array(start_cell[None], dtype=wp.mat33d, device=device)
        wp_forces = wp.zeros(n, dtype=wp.vec3d, device=device)
        wp_stress = wp.zeros(1, dtype=wp.mat33d, device=device)
        wp_batch = wp.zeros(n, dtype=wp.int32, device=device)
        wp_nparts = wp.array(np.array([n], np.int32), dtype=wp.int32, device=device)
        wp_cell_state = make_lbfgs_cell_state(n, 1, wp.vec3d, device)
        warp_set_ref(wp_cell, wp_cell_state["ref_cell"], wp_cell_state["ref_cell_inv"])
        wp_state_dict = make_lbfgs_state(n + 2, 1, 6, wp.vec3d, device)

        for _ in range(25):
            forces, stress = self._evaluate(positions, cell, potential, device)
            lbfgs_step_coord_cell(
                positions,
                cell,
                forces,
                stress,
                batch_idx,
                n_particles,
                *buffers.values(),
                *cell_buffers.values(),
                force_tol=1e-8,
                stress_tol=1e-8,
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
                wp_batch,
                wp_nparts,
                *wp_state_dict.values(),
                *wp_cell_state.values(),
                force_tol=1e-8,
                stress_tol=1e-8,
                maxstep=0.2,
            )
            wp.synchronize()
            torch.cuda.synchronize()
            np.testing.assert_array_equal(positions.cpu().numpy(), wp_pos.numpy())
            np.testing.assert_array_equal(cell.cpu().numpy(), wp_cell.numpy())
            if status.item() != LBFGS_NEED_EVAL:
                break
        np.testing.assert_array_equal(
            status.cpu().numpy(), wp_state_dict["status"].numpy()
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_buffers_sized_for_the_wrong_dof_count_are_rejected(self, device):
        """The cell adds two degrees of freedom per system; coordinate-sized
        buffers would silently under-cover the packed array."""
        from nvalchemiops.torch.lbfgs import lbfgs_step_coord_cell

        n = 6
        positions, cell, _, cell_buffers, potential = self._setup(device, n)
        wrong = make_torch_buffers(n, 1, torch.float64, device)
        forces, stress = self._evaluate(positions, cell, potential, device)
        with pytest.raises(ValueError, match="num_atoms \\+ 2 \\* num_systems"):
            lbfgs_step_coord_cell(
                positions,
                cell,
                forces,
                stress,
                torch.zeros(n, dtype=torch.int32, device=device),
                torch.full((1,), n, dtype=torch.int32, device=device),
                *wrong.values(),
                *cell_buffers.values(),
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_operator_schema_matches_the_buffer_order(self, device):
        """Both buffer groups must line up with the operator, in order.

        Callers pass all 40 tensors positionally, so an ordering mismatch
        would type-check and then compute nonsense.
        """
        from nvalchemiops.torch.lbfgs import (
            _CELL_MUTATED,
            _lbfgs_step_coord_cell_op,
            lbfgs_step_coord_cell,
        )

        params = tuple(inspect.signature(_lbfgs_step_coord_cell_op).parameters)
        offset = 6  # forces, stress, batch_idx, n_particles, positions, cell
        n_opt = len(_OPTIMIZER_BUFFERS)
        assert params[offset : offset + n_opt] == _OPTIMIZER_BUFFERS
        assert params[offset + n_opt : offset + n_opt + len(_CELL_BUFFERS)] == (
            _CELL_BUFFERS
        )

        # The public wrapper forwards positionally, so it must agree too.
        wrapper = tuple(inspect.signature(lbfgs_step_coord_cell).parameters)
        assert wrapper[6 : 6 + n_opt] == _OPTIMIZER_BUFFERS
        assert wrapper[6 + n_opt : 6 + n_opt + len(_CELL_BUFFERS)] == _CELL_BUFFERS

        # The read-only chart inputs must not be declared as mutated.
        assert set(_CELL_MUTATED).isdisjoint(_CELL_BUFFERS[:5])
        assert set(_CELL_MUTATED) == (
            {"positions", "cell"} | set(_OPTIMIZER_BUFFERS) | set(_CELL_BUFFERS[5:])
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_compiles_fullgraph_and_matches_eager(self, device):
        """The variable-cell step compiles as one graph and agrees with eager.

        Each run gets its own freshly allocated state, since the step mutates
        it; comparing against a run whose state had already advanced would be
        meaningless.
        """
        from nvalchemiops.torch.lbfgs import lbfgs_step_coord_cell

        n = 6
        opts = dict(force_tol=1e-8, stress_tol=1e-8, maxstep=0.2)
        batch_idx = torch.zeros(n, dtype=torch.int32, device=device)
        n_particles = torch.full((1,), n, dtype=torch.int32, device=device)

        def one_run(compiled):
            positions, cell, buffers, cell_buffers, potential = self._setup(device, n)
            forces, stress = self._evaluate(positions, cell, potential, device)

            def body(p, c, f, s):
                lbfgs_step_coord_cell(
                    p,
                    c,
                    f,
                    s,
                    batch_idx,
                    n_particles,
                    *buffers.values(),
                    *cell_buffers.values(),
                    **opts,
                )

            torch._dynamo.reset()
            fn = torch.compile(body, fullgraph=True) if compiled else body
            fn(positions, cell, forces, stress)
            torch.cuda.synchronize()
            return positions.clone(), cell.clone(), buffers["status"].clone()

        eager = one_run(False)
        compiled = one_run(True)
        torch.testing.assert_close(compiled[0], eager[0], rtol=0, atol=0)
        torch.testing.assert_close(compiled[1], eager[1], rtol=0, atol=0)
        torch.testing.assert_close(compiled[2], eager[2])
