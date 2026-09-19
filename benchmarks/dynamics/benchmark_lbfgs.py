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

"""Compare L-BFGS and FIRE2 by force evaluations to convergence.

Evaluation count is the metric that matters for relaxation driven by a
machine-learned potential: the model call dominates the optimizer's own kernel
time by orders of magnitude, so the optimizer that reaches a given force
tolerance in fewer evaluations wins regardless of its per-step cost.

The test system is an argon cluster relaxed with the package's own Lennard-Jones
kernels behind a warp-native neighbor list, using the LJ and neighbor
parameters from the shared ``potential`` config block -- the same path the
other dynamics benchmarks take. FIRE2's timestep is swept and its **best**
converged configuration is reported, so the baseline is not handicapped by a
poor choice of hyperparameters.

Evaluation counts vary by roughly 20% run to run: neighbor-list rebuild
ordering perturbs the forces in the last bits, and both optimizers amplify that
into a different trajectory. Read the aggregate, not a single cell.

Passing ``--gates`` instead measures the per-step cost commitments: optimizer
time against FIRE2 at scale, what CUDA-graph replay recovers, and the model
cost above which L-BFGS wins end to end.

Defaults come from the ``lbfgs`` section of ``benchmark_config.yaml``, next to
this file, so the knobs live alongside the FIRE and FIRE2 ones. Any command-line
flag overrides the file.

Usage
-----
    python -m benchmarks.dynamics.benchmark_lbfgs [--config PATH]
                                                  [--sizes 13 32 55]
                                                  [--seeds 5]
                                                  [--force-tol 1e-4]
                                                  [--output-dir DIR]
                                                  [--device cuda:0]
    python -m benchmarks.dynamics.benchmark_lbfgs --gates [--eval-ratio 0.59]
                                                          [--device cuda:0]

``--device`` matches ``benchmark_fire2.py``, since this runner reports a ratio
against FIRE2 and the two have to be pointed at the same GPU to compare. It is
threaded through every allocation *and* made the current device at entry --
CUDA events, streams, graphs and ``current_stream()`` take the current device
rather than an argument, so without that the data would sit on the requested
GPU while the capture and timing resources came from another.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import pathlib

import numpy as np
import torch
import warp as wp

from benchmarks.dynamics.shared_utils import (
    MDSystem,
    create_random_cluster,
    load_config,
)
from nvalchemiops.dynamics.optimizers import (
    fire2_step,
    lbfgs_prepare_state,
    lbfgs_step,
)

#: Default for ``--device``; every function takes the choice explicitly,
#: matching ``benchmark_fire2.py`` so the two can be run side by side.
DEFAULT_DEVICE = "cuda:0"

#: Gate model: anisotropic harmonic with its minimum out of reach of a
#: ``maxstep``-capped walk, so the optimizer stays in steady state.
GATE_CENTRE = 1.0e6
GATE_HISTORY = 6

#: Fallbacks used when a knob is absent from the config file, so the benchmark
#: still runs standalone.
EVAL_CAP = 20000
CONFIG_PATH = pathlib.Path(__file__).with_name("benchmark_config.yaml")
#: FIRE2 grid, in the real units the shared ``potential`` block implies (fs,
#: angstrom). ``tmax`` is swept as a multiple of ``dt_start`` because FIRE2
#: clamps its timestep to ``tmax``, so a large starting step with a small cap
#: is not a distinct configuration. ``maxstep`` is *not* swept: measured on
#: this workload it never binds, so sweeping it only multiplies the runtime.
FIRE2_SWEEP = {
    "dt_start": [0.15, 0.25, 0.4, 0.6, 1.0, 1.5, 2.0, 3.0],
    "tmax_factor": [2.0],
}
FIRE2_MAXSTEP = 0.25

#: Argon, matching the ``potential`` block of the shared config and the
#: defaults the other dynamics benchmarks use.
DEFAULT_POTENTIAL = {"epsilon": 0.0104, "sigma": 3.40, "cutoff": 8.5, "skin": 1.0}

#: Coordinate precisions, spelled the way ``benchmark_fire2.py`` spells them so
#: the two runners can be compared row for row. Every L-BFGS array follows the
#: coordinate dtype, so one entry fixes the whole state; FIRE2's scalars are
#: selected the same way here, or the comparison would be measuring precision
#: rather than optimizer.
DTYPES = {
    "float32": (torch.float32, wp.vec3f, wp.float32, np.float32),
    "float64": (torch.float64, wp.vec3d, wp.float64, np.float64),
}

#: fp32 is the default because it is what a machine-learned potential emits.
#: fp64 is supported and benchmarked, but it is a deliberate choice rather
#: than the only configuration the runner can express.
DEFAULT_DTYPES = ["float32", "float64"]


def make_cluster(num_atoms, potential, device, seed=0, dtype=torch.float32):
    """An argon cluster evaluated by the package's own LJ kernels.

    Uses :class:`~benchmarks.dynamics.shared_utils.MDSystem`, as the other
    dynamics benchmarks do, so the forces come from
    :func:`nvalchemiops.interactions.lj_energy_forces` behind a warp-native
    neighbor list and the LJ and neighbor parameters are the ones in the
    shared ``potential`` config block.

    The cluster sits in a large non-periodic box, so it stays a cluster: the
    box only has to exceed the cluster diameter plus the cutoff.

    Parameters
    ----------
    num_atoms : int
        Atoms in the cluster.
    potential : dict
        The ``potential`` block: ``epsilon``, ``sigma``, ``cutoff``, ``skin``.
    device : str
        CUDA device.
    seed : int
        Starting geometry.
    dtype : torch.dtype
        Coordinate precision. Carried into the neighbor list and the LJ
        evaluation as well as the optimizer state, so an fp32 run is fp32 end
        to end rather than fp64 arrays relabelled.

    Returns
    -------
    MDSystem
    """
    sigma = potential["sigma"]
    cutoff = potential["cutoff"]
    # Spread the atoms so the cluster is loose enough to be a real relaxation
    # rather than a rattle, and keep them apart enough to avoid the r^-12 wall.
    radius = 0.75 * sigma * num_atoms ** (1.0 / 3.0) + sigma
    positions = create_random_cluster(
        num_atoms, radius=radius, min_dist=0.9 * sigma, seed=seed
    )
    box = 2.0 * (radius + cutoff) + potential["skin"]
    return MDSystem(
        positions=torch.tensor(positions, dtype=dtype, device=device),
        cell=torch.eye(3, dtype=dtype, device=device) * box,
        pbc=torch.zeros(3, dtype=torch.bool, device=device),
        epsilon=potential["epsilon"],
        sigma=sigma,
        cutoff=cutoff,
        skin=potential["skin"],
        device=device,
        dtype=dtype,
    )


def _fmax(system):
    """Largest per-atom force magnitude, read back from the device."""
    return float(
        np.linalg.norm(wp.to_torch(system.wp_forces).cpu().numpy(), axis=1).max()
    )


def _allocate_lbfgs_state(num_dofs, num_systems, history_size, device,
                          vec_dtype=wp.vec3f):  # fmt: skip
    """The Warp state, already in its required start state."""
    return lbfgs_prepare_state(
        num_dofs, num_systems, dtype=vec_dtype, history_size=history_size,
        device=device,
    )  # fmt: skip


def _allocate_lbfgs_buffers_torch(num_dofs, num_systems, history_size, device,
                                  dtype=torch.float32):  # fmt: skip
    """The same state as torch tensors."""
    from nvalchemiops.torch.lbfgs import lbfgs_prepare_state as prepare

    return prepare(
        num_dofs, num_systems, dtype=dtype, history_size=history_size,
        device=device,
    )  # fmt: skip


def run_lbfgs(system, force_tol, history_size=6, maxstep=0.2, eval_cap=EVAL_CAP,
              device=DEFAULT_DEVICE):  # fmt: skip
    """Relax ``system`` with L-BFGS; return (evaluations, converged, max force).

    ``system`` is consumed: its positions are relaxed in place. The state is
    allocated at the system's own coordinate precision, so the dtype comes
    from the caller's choice of system rather than from a default here.
    """
    num_atoms = system.num_atoms
    batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=device)
    state = _allocate_lbfgs_state(
        num_atoms, 1, history_size, device, vec_dtype=system.wp_positions.dtype
    )

    # Deliberately the same shape as ``run_fire2`` below: both optimizers
    # leave convergence to the caller, so both loops test then step.
    for n_evals in range(1, eval_cap + 1):
        system.compute_forces()
        current = _fmax(system)
        if current <= force_tol:
            return n_evals, True, current
        lbfgs_step(
            positions=system.wp_positions,
            forces=system.wp_forces,
            state=state,
            batch_idx=batch_idx,
            maxstep=maxstep,
        )
        wp.synchronize()
    system.compute_forces()
    return eval_cap, False, _fmax(system)


def run_fire2(
    system, force_tol, dt_start=0.02, maxstep=0.05, tmax=0.1, eval_cap=EVAL_CAP,
    device=DEFAULT_DEVICE,
):  # fmt: skip
    """Relax ``system`` with FIRE2; return (evaluations, converged, max force).

    ``system`` is consumed: its positions are relaxed in place.
    """
    num_atoms = system.num_atoms
    # FIRE2 runs at the system's precision too: comparing an fp32 L-BFGS
    # against an fp64 FIRE2 would be measuring precision, not optimizer.
    vec_dtype = system.wp_positions.dtype
    scalar_dtype = wp.float32 if vec_dtype == wp.vec3f else wp.float64
    np_dtype = np.float32 if vec_dtype == wp.vec3f else np.float64
    velocities = wp.zeros(num_atoms, dtype=vec_dtype, device=device)
    batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=device)
    alpha = wp.array(np.array([0.09], np_dtype), dtype=scalar_dtype, device=device)
    dt = wp.array(np.array([dt_start], np_dtype), dtype=scalar_dtype, device=device)
    nsteps_inc = wp.zeros(1, dtype=wp.int32, device=device)
    scratch = [wp.zeros(1, dtype=scalar_dtype, device=device) for _ in range(4)]

    for n_evals in range(1, eval_cap + 1):
        system.compute_forces()
        current = _fmax(system)
        if current <= force_tol:
            return n_evals, True, current
        fire2_step(
            system.wp_positions,
            velocities,
            system.wp_forces,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            *scratch,
            maxstep=maxstep,
            tmax=tmax,
            tmin=0.002,
            dtgrow=1.1,
            dtshrink=0.5,
            delaystep=5,
            alpha0=0.09,
            alphashrink=0.99,
        )
        wp.synchronize()
    return eval_cap, False, current


def best_fire2(make_system, force_tol, sweep=None, eval_cap=EVAL_CAP,
               device=DEFAULT_DEVICE):  # fmt: skip
    """FIRE2 at its best over a small hyperparameter sweep.

    Comparing against an untuned baseline would overstate the result; FIRE2 is
    sensitive to its timestep on this system. The grid comes from the
    ``lbfgs.fire2_sweep`` block of the config file.

    ``make_system`` is called once per sweep entry, because each relaxation
    consumes the system it is given.

    When nothing converges, the first capped run is reported as it stands
    rather than re-run: it already bounds the evaluation count from below, and
    repeating a configuration that has just hit the cap would spend another
    ``eval_cap`` evaluations reproducing a number already in hand.
    """
    sweep = FIRE2_SWEEP if sweep is None else sweep
    best = None
    first_capped = None
    for dt_start in sweep["dt_start"]:
        for factor in sweep.get("tmax_factor", [2.0]):
            evals, converged, final = run_fire2(
                make_system(),
                force_tol,
                dt_start=dt_start,
                maxstep=FIRE2_MAXSTEP,
                tmax=factor * dt_start,
                eval_cap=eval_cap,
                device=device,
            )
            if converged:
                if best is None or evals < best[0]:
                    best = (evals, converged, final, (dt_start, factor))
            elif first_capped is None:
                first_capped = (
                    evals, converged, final,
                    f"none converged ({dt_start}, {factor})",
                )  # fmt: skip
    return best if best is not None else first_capped


def select_device(device):
    """Make ``device`` current, and return it resolved.

    Passing a device to every allocation is not enough. CUDA events, streams,
    graphs and ``torch.cuda.current_stream()`` all come from whatever device is
    *current*, so on a multi-GPU host ``--device cuda:1`` would otherwise put
    the data on one GPU and the capture and timing resources on another --
    which fails outright, or, worse, times the wrong device.

    Set once at entry rather than scoped per block: this runner targets a
    single GPU for its whole run, and every helper below creates one of those
    implicitly-placed resources.

    Parameters
    ----------
    device : str
        CUDA device, e.g. ``"cuda:0"``.

    Returns
    -------
    torch.device
    """
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"this benchmark needs a CUDA device; got {device!r}")
    torch.cuda.set_device(resolved)
    return resolved


def _time_ms(fn, warmup=10, runs=50):
    """Median-free mean over CUDA events, warmup excluded."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    wp.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(runs):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / runs


def run_gates(sizes, eval_ratio, warmup=10, runs=50, device=DEFAULT_DEVICE,
              dtype_name="float32"):  # fmt: skip
    """Measure per-step cost, graph replay, and the break-even model cost.

    Reported per system size:

    ``eager`` / ``graph``
        Optimizer-only time for one L-BFGS step, launched from Python and
        replayed from a CUDA graph. The gap is Python launch overhead; the step
        issues far more kernels than FIRE2, so at small sizes it is entirely
        launch-bound.
    ``fire2``
        One FIRE2 step, for the same size, eager.
    ``break-even``
        Model cost per evaluation above which L-BFGS wins end to end, given the
        evaluation-count advantage measured by the default benchmark. Negative
        means it wins outright, because needing far fewer evaluations more than
        pays for a costlier step.

    ``dtype_name`` fixes the coordinate precision for both arms. Measured, the
    effect is smaller than the halved byte count suggests -- 0.98 against 1.03
    ms at 1e5 atoms -- because FIRE2 halves too, so the ratio moves from 8.4x
    to 9.0x rather than doubling. Reported per precision all the same: it is
    a measurement, not something to be inferred from one run and a factor.
    """
    torch_dtype, vec_dtype, scalar_dtype, np_dtype = DTYPES[dtype_name]
    from nvalchemiops.dynamics.optimizers import fire2_step
    from nvalchemiops.torch.lbfgs import lbfgs_step_coord

    # Before any stream, event, graph or scoped stream is created.
    select_device(device)
    print(f"coordinate precision: {dtype_name}")
    print(
        f"{'atoms':>9} {'eager ms':>9} {'graph ms':>9} {'fire2 ms':>9} "
        f"{'eager/f2':>9} {'graph gain':>11} {'break-even':>12}"
    )
    rows = []
    for num_atoms in sizes:
        rng = np.random.default_rng(0)
        start = torch.tensor(
            rng.normal(size=(num_atoms, 3)), dtype=torch_dtype, device=device
        )
        batch_idx = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        stiffness = torch.tensor([1.0, 4.0, 9.0], dtype=torch_dtype, device=device)

        positions = start.clone()
        forces = torch.empty_like(positions)
        buffers = _allocate_lbfgs_buffers_torch(
            num_atoms, 1, GATE_HISTORY, device, dtype=torch_dtype
        )

        def evaluate(pos, out):
            """The model, on device so it stays CUDA-graph capturable.

            Re-evaluated every call. Against a force array filled once,
            ``y = force_base - F`` is zero, so the curvature guard rejects
            every pair and the two-loop is masked out of every step. The
            minimum at ``GATE_CENTRE`` is out of reach of a ``maxstep``-capped
            walk, keeping the optimizer in steady state rather than converging
            partway through and timing the early-return path.
            """
            out.copy_(-(stiffness * (pos - GATE_CENTRE)))

        def reset():
            """Return to the same starting state before each timed phase.

            The tensors are copied into rather than rebound, so a captured
            graph keeps pointing at the buffers it recorded.
            """
            positions.copy_(start)
            fresh = _allocate_lbfgs_buffers_torch(
                num_atoms, 1, GATE_HISTORY, device, dtype=torch_dtype
            )
            for field in dataclasses.fields(buffers):
                getattr(buffers, field.name).copy_(getattr(fresh, field.name))

        def model_only():
            evaluate(positions, forces)

        def step():
            evaluate(positions, forces)
            lbfgs_step_coord(positions, forces, buffers, batch_idx, maxstep=0.5)

        def check(phase):
            """Fail loudly if the timed steps were not representative."""
            history = int(buffers.history_count.item())
            if history != GATE_HISTORY:
                raise RuntimeError(
                    f"{phase} timing at {num_atoms} atoms left "
                    f"history_count={history}, not {GATE_HISTORY}; the steps "
                    "measured were not full-history iterations"
                )

        # Subtract the model, which runs inside every timed call.
        reset()
        model_ms = _time_ms(model_only, warmup, runs)

        reset()
        eager = _time_ms(step, warmup, runs) - model_ms
        check("eager")

        reset()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        wp.synchronize()
        graph = torch.cuda.CUDAGraph()
        # No wp.ScopedStream here: step() is the PyTorch model plus the
        # registered L-BFGS op, and that op binds Warp to the current stream
        # itself, which during capture is the capture stream.
        with torch.cuda.graph(graph):
            step()
        torch.cuda.synchronize()
        graphed = _time_ms(graph.replay, warmup, runs) - model_ms
        check("graph")

        # FIRE2 gets the same model, so the ratio compares like with like.
        f2_positions = start.clone()
        f2_forces = torch.empty_like(f2_positions)
        # Bind Warp to the stream the events are recorded on: the model is a
        # PyTorch op and fire2_step a raw Warp launcher, so otherwise the
        # events bracket only the PyTorch half. The scope covers the Warp
        # allocations and the ``from_torch`` views as well as the launches --
        # a view built on one stream and launched on another is ordered
        # against neither its producer nor its consumer. The L-BFGS arm's
        # custom op binds the stream itself and needs none of this.
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
            wp_positions = wp.from_torch(f2_positions, dtype=vec_dtype)
            wp_forces = wp.from_torch(f2_forces, dtype=vec_dtype)
            wp_velocities = wp.zeros(num_atoms, dtype=vec_dtype, device=device)
            wp_batch = wp.zeros(num_atoms, dtype=wp.int32, device=device)
            alpha = wp.array(
                np.array([0.09], np_dtype), dtype=scalar_dtype, device=device
            )
            dt = wp.array(np.array([0.02], np_dtype), dtype=scalar_dtype, device=device)
            nsteps_inc = wp.zeros(1, dtype=wp.int32, device=device)
            scratch = [wp.zeros(1, dtype=scalar_dtype, device=device) for _ in range(4)]

            def fire2_once():
                evaluate(f2_positions, f2_forces)
                fire2_step(
                    wp_positions,
                    wp_velocities,
                    wp_forces,
                    wp_batch,
                    alpha,
                    dt,
                    nsteps_inc,
                    *scratch,
                    maxstep=0.05,
                )

            fire2 = _time_ms(fire2_once, warmup, runs) - model_ms

        # n_L (C + O_L) < n_F (C + O_F), with n_L / n_F = eval_ratio.
        n_fire2 = 1000.0
        n_lbfgs = eval_ratio * n_fire2
        break_even_ms = (n_lbfgs * eager - n_fire2 * fire2) / (n_fire2 - n_lbfgs)
        rows.append((dtype_name, num_atoms, eager, graphed, fire2, break_even_ms))
        print(
            f"{num_atoms:>9} {eager:>9.4f} {graphed:>9.4f} {fire2:>9.4f} "
            f"{eager / fire2:>8.2f}x {eager / graphed:>10.1f}x "
            f"{break_even_ms * 1000:>11.1f}us"
        )

    largest = rows[-1]
    print(
        f"\nper-step ratio at {largest[1]} atoms ({dtype_name}): "
        f"{largest[2] / largest[4]:.2f}x FIRE2"
    )
    print(
        "break-even is the model cost per evaluation above which L-BFGS wins end "
        "to end. Where it is negative L-BFGS wins outright; where positive, "
        "compare it against your model -- a machine-learned potential typically "
        "costs milliseconds per evaluation, orders of magnitude more than these "
        "figures."
    )
    return rows


def _load_lbfgs_config(path):
    """Read the ``lbfgs``, ``output`` and ``potential`` blocks of the config.

    The benchmark is useful standalone, so a missing file or section is not an
    error -- the module-level fallbacks apply instead.

    Returns
    -------
    tuple of dict
        The ``lbfgs`` section, and the shared ``output`` and ``potential``
        sections. The potential falls back to the argon parameters the other
        dynamics benchmarks default to, so this still runs standalone.
    """
    path = pathlib.Path(path)
    document = load_config(path) if path.is_file() else {}
    potential = {**DEFAULT_POTENTIAL, **(document.get("potential", {}) or {})}
    return (
        document.get("lbfgs", {}) or {},
        document.get("output", {}) or {},
        potential,
    )


def _resolve_output_dir(explicit, output_config, config_path):
    """Where results go: the flag first, then ``output.results_dir``.

    ``results_dir`` is interpreted relative to the config file that declares
    it, which is how the neighborlist and interactions configs read their
    ``base_dir``. ``save_timing: false`` turns writing off, but an explicit
    ``--output-dir`` still wins -- asking for a directory on the command line
    is unambiguous.

    Returns ``None`` when nothing should be written.
    """
    if explicit is not None:
        return pathlib.Path(explicit)
    if not output_config.get("save_timing", True):
        return None
    results_dir = output_config.get("results_dir")
    if results_dir is None:
        return None
    return (pathlib.Path(config_path).parent / results_dir).resolve()


def _write_csv(output_dir, name, fieldnames, rows):
    """Write ``rows`` to ``output_dir/name``; no-op when there is nowhere to go."""
    if output_dir is None or not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main():
    """Run the evaluation-count comparison, or the per-step cost gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=CONFIG_PATH,
        help="benchmark_config.yaml supplying the defaults below",
    )
    # Every default is None so an explicit flag can be told apart from an
    # unset one, and therefore override the config file rather than shadow it.
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--force-tol", type=float, default=None)
    parser.add_argument("--eval-cap", type=int, default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--gates",
        action="store_true",
        help="measure per-step cost, graph replay and break-even model cost",
    )
    parser.add_argument("--gate-sizes", type=int, nargs="+", default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        nargs="+",
        choices=sorted(DTYPES),
        default=None,
        help=(
            "coordinate precisions to run; every optimizer array follows this. "
            "Defaults to the config file's lbfgs.dtypes, else float32 and float64"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="CUDA device",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=None,
        help="measured L-BFGS/FIRE2 evaluation ratio, used for the break-even cost",
    )
    args = parser.parse_args()

    select_device(args.device)
    config, output_config, potential = _load_lbfgs_config(args.config)
    gates_config = config.get("gates", {}) or {}
    output_dir = _resolve_output_dir(args.output_dir, output_config, args.config)

    # `enabled` works the way it does for the other dynamics benchmarks: it is
    # how a config file turns a suite off without editing the runner.
    if not config.get("enabled", True):
        print("lbfgs benchmark disabled in the config file; nothing to do")
        return
    if args.gates and not gates_config.get("enabled", True):
        print("lbfgs gates disabled in the config file; nothing to do")
        return

    def pick(flag, key, fallback, section=config):
        """Command line first, then the config file, then the fallback."""
        return flag if flag is not None else section.get(key, fallback)

    sizes = pick(args.sizes, "system_sizes", [13, 32, 55])
    seeds = pick(args.seeds, "seeds", 5)
    force_tol = pick(args.force_tol, "force_tolerance", 1e-4)
    eval_cap = pick(args.eval_cap, "eval_cap", EVAL_CAP)
    history_size = config.get("history_size", 6)
    maxstep = config.get("maxstep", 0.2)
    sweep = config.get("fire2_sweep", FIRE2_SWEEP)

    dtypes = pick(args.dtype, "dtypes", DEFAULT_DTYPES)

    if args.gates:
        gate_rows = []
        for dtype_name in dtypes:
            gate_rows += run_gates(
                pick(
                    args.gate_sizes,
                    "system_sizes",
                    [10_000, 100_000, 1_000_000],
                    section=gates_config,
                ),
                pick(args.eval_ratio, "eval_ratio", 0.59, section=gates_config),
                gates_config.get("warmup", 10),
                gates_config.get("runs", 50),
                device=args.device,
                dtype_name=dtype_name,
            )
        # These rows are what the published per-step table is drawn from, so
        # they have to survive the run.
        _write_csv(
            output_dir,
            "lbfgs_gate_timings.csv",
            [
                "dtype",
                "atoms",
                "eager_ms",
                "graph_ms",
                "fire2_ms",
                "eager_over_fire2",
                "graph_gain",
                "break_even_us",
            ],
            [
                {
                    "dtype": dtype_name,
                    "atoms": atoms,
                    "eager_ms": f"{eager:.6f}",
                    "graph_ms": f"{graphed:.6f}",
                    "fire2_ms": f"{fire2:.6f}",
                    "eager_over_fire2": f"{eager / fire2:.4f}" if fire2 else "",
                    "graph_gain": f"{eager / graphed:.4f}" if graphed else "",
                    "break_even_us": f"{break_even * 1e3:.4f}",
                }
                for dtype_name, atoms, eager, graphed, fire2, break_even in gate_rows
            ],
        )
        return

    rows = []
    print(
        f"{'dtype':>8} {'atoms':>6} {'seed':>5} {'lbfgs':>7} {'fire2':>7} "
        f"{'ratio':>7}  {'fire2 config':>14}"
    )
    for dtype_name, num_atoms, seed in itertools.product(dtypes, sizes, range(seeds)):
        torch_dtype = DTYPES[dtype_name][0]

        def make_system(n=num_atoms, s=seed, d=torch_dtype):
            return make_cluster(n, potential, args.device, seed=s, dtype=d)

        lb_evals, lb_ok, lb_force = run_lbfgs(
            make_system(), force_tol, history_size, maxstep, eval_cap,
            args.device,
        )  # fmt: skip
        f2_evals, f2_ok, f2_force, f2_cfg = best_fire2(
            make_system, force_tol, sweep, eval_cap, args.device
        )
        ratio = lb_evals / f2_evals
        rows.append(
            {
                "dtype": dtype_name,
                "num_atoms": num_atoms,
                "seed": seed,
                "lbfgs_evals": lb_evals,
                "lbfgs_converged": lb_ok,
                "lbfgs_fmax": lb_force,
                "fire2_evals": f2_evals,
                "fire2_converged": f2_ok,
                "fire2_fmax": f2_force,
                "fire2_config": str(f2_cfg),
                "ratio": ratio,
            }
        )
        print(
            f"{dtype_name:>8} {num_atoms:>6} {seed:>5} {lb_evals:>7} "
            f"{f2_evals:>7} {ratio:>7.3f}  {str(f2_cfg):>14}"
            f"{'' if (lb_ok and f2_ok) else '  (capped)'}"
        )

    # The aggregate is over cases where *both* optimizers converged. A capped
    # FIRE2 run only bounds its evaluation count from below, so its ratio
    # bounds rather than measures; averaging bounds together with measurements
    # would report a number that is partly not a measurement -- and in the
    # direction that flatters L-BFGS, since a capped run divides by a count
    # that is too small.
    # Reported per precision, not pooled: fp32 and fp64 are two different
    # measurements of two different runs, and one geometric mean over both
    # would describe neither.
    for dtype_name in dtypes:
        of_dtype = [r for r in rows if r["dtype"] == dtype_name]
        comparable = [
            r for r in of_dtype if r["lbfgs_converged"] and r["fire2_converged"]
        ]
        print(f"\n{dtype_name}:")
        if comparable:
            ratios = [r["ratio"] for r in comparable]
            geo_mean = float(np.exp(np.mean(np.log(ratios))))
            print(
                f"  cases where both converged:                      "
                f"{len(comparable)}/{len(of_dtype)}"
            )
            print(f"  geometric mean evaluation ratio (L-BFGS / FIRE2): {geo_mean:.3f}")
            print(
                f"  worst individual ratio:                          {max(ratios):.3f}"
            )
        else:
            print("  no case had both optimizers converge; no ratio can be quoted")

    fire2_capped = [
        r for r in rows if r["lbfgs_converged"] and not r["fire2_converged"]
    ]
    lbfgs_failed = [r for r in rows if not r["lbfgs_converged"]]
    both_failed = [r for r in lbfgs_failed if not r["fire2_converged"]]

    if fire2_capped:
        bounds = ", ".join(f"{r['ratio']:.3f}" for r in fire2_capped)
        print(
            f"\nexcluded -- FIRE2 hit the {eval_cap}-evaluation cap in "
            f"{len(fire2_capped)} case(s). Its true count is at least the cap, so "
            f"each ratio ({bounds}) is an upper bound on that case's ratio, not a "
            "measurement. L-BFGS converged in all of them."
        )
    if lbfgs_failed:
        print(
            f"\nexcluded -- L-BFGS hit the cap in {len(lbfgs_failed)} case(s)"
            + (
                f", {len(both_failed)} of which FIRE2 also failed "
                "(those establish no comparison at all)"
                if both_failed
                else ""
            )
            + "."
        )

    if output_dir is not None and rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "lbfgs_vs_fire2_evaluations.csv"
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
