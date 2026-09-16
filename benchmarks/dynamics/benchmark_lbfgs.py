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

"""Compare L-BFGS and FIRE2 by energy/force evaluations to convergence.

Evaluation count is the metric that matters for relaxation driven by a
machine-learned potential: the model call dominates the optimizer's own kernel
time by orders of magnitude, so the optimizer that reaches a given force
tolerance in fewer evaluations wins regardless of its per-step cost.

The test system is a Lennard-Jones cluster in reduced units, which is cheap to
evaluate on the host and anharmonic enough to be a fair test. FIRE2's timestep
and step cap are swept and its **best** configuration is reported, so the
baseline is not handicapped by a poor choice of hyperparameters.

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
    python -m benchmarks.dynamics.benchmark_lbfgs --gates [--eval-ratio 0.142]
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np
import torch
import warp as wp

from benchmarks.dynamics.shared_utils import load_config
from nvalchemiops.dynamics.optimizers import (
    LBFGS_CONVERGED,
    LBFGS_NEED_EVAL,
    fire2_step,
    lbfgs_step,
)

DEVICE = "cuda:0"

#: Fallbacks used when a knob is absent from the config file, so the benchmark
#: still runs standalone.
EVAL_CAP = 20000
CONFIG_PATH = pathlib.Path(__file__).with_name("benchmark_config.yaml")
FIRE2_SWEEP = {"dt_start": [0.005, 0.01, 0.02, 0.05], "maxstep": [0.05, 0.1, 0.2]}


def lennard_jones(positions):
    """All-pairs Lennard-Jones in reduced units (epsilon = sigma = 1)."""
    delta = positions[:, None, :] - positions[None, :, :]
    r2 = (delta**2).sum(-1)
    np.fill_diagonal(r2, np.inf)
    inv6 = r2**-3
    inv12 = inv6**2
    per_atom_energy = 0.5 * (4.0 * (inv12 - inv6)).sum(1)
    coefficient = 24.0 * (2.0 * inv12 - inv6) / r2
    forces = (coefficient[..., None] * delta).sum(1)
    return per_atom_energy, forces


def _allocate_lbfgs_state(num_dofs, num_systems, history_size):
    """Allocate the caller-owned buffers, already in their required start state."""

    def f64(n):
        return wp.zeros(n, dtype=wp.float64, device=DEVICE)

    def f64_2d(a, b):
        return wp.zeros((a, b), dtype=wp.float64, device=DEVICE)

    def vec(n):
        return wp.zeros(n, dtype=wp.vec3d, device=DEVICE)

    def i32(n):
        return wp.zeros(n, dtype=wp.int32, device=DEVICE)

    buffers = {
        "x_base": vec(num_dofs),
        "force_base": vec(num_dofs),
        "direction": vec(num_dofs),
        "s_history": wp.zeros((history_size, num_dofs), dtype=wp.vec3d, device=DEVICE),
        "y_history": wp.zeros((history_size, num_dofs), dtype=wp.vec3d, device=DEVICE),
        "ys": f64_2d(history_size, num_systems),
        "yy": f64_2d(history_size, num_systems),
        "alpha_hist": f64_2d(history_size, num_systems),
        "beta_hist": f64_2d(history_size, num_systems),
        "ss": f64(num_systems),
        "f_base": f64(num_systems),
        "gg": f64(num_systems),
        "gd": f64(num_systems),
        "fmax": f64(num_systems),
        "frms_sq": f64(num_systems),
        "smax": f64(num_systems),
        "d0": f64(num_systems),
        "dmax": f64(num_systems),
        "dquad": f64(num_systems),
        "alpha_step": f64(num_systems),
        "status": i32(num_systems),
        "iteration": i32(num_systems),
        "end": i32(num_systems),
        "n_loop": i32(num_systems),
        "ls_trials": i32(num_systems),
        "history_count": i32(num_systems),
    }
    # Three buffers do not start at zero; the optimizer initializes nothing.
    buffers["alpha_step"].fill_(1.0)
    buffers["iteration"].fill_(-1)
    buffers["status"].fill_(LBFGS_NEED_EVAL)
    return buffers


def _allocate_lbfgs_buffers_torch(num_dofs, num_systems, history_size):
    """The same buffers as torch tensors, in the order the step takes them."""
    import torch

    f64, i32 = torch.float64, torch.int32

    def z(*shape, dt=f64):
        return torch.zeros(shape, dtype=dt, device=DEVICE)

    alpha_step = torch.ones(num_systems, dtype=f64, device=DEVICE)
    iteration = torch.full((num_systems,), -1, dtype=i32, device=DEVICE)
    return (
        z(num_dofs, 3), z(num_dofs, 3), z(num_dofs, 3),
        z(history_size, num_dofs, 3), z(history_size, num_dofs, 3),
        z(history_size, num_systems), z(history_size, num_systems),
        z(history_size, num_systems), z(history_size, num_systems),
        *[z(num_systems) for _ in range(10)],
        alpha_step,
        z(num_systems, dt=i32), iteration,
        *[z(num_systems, dt=i32) for _ in range(4)],
    )  # fmt: skip


def run_lbfgs(start, force_tol, history_size=6, maxstep=0.2, eval_cap=EVAL_CAP):
    """Relax with L-BFGS; return (evaluations, converged, final max force)."""
    num_atoms = start.shape[0]
    positions = wp.array(start.copy(), dtype=wp.vec3d, device=DEVICE)
    forces = wp.zeros(num_atoms, dtype=wp.vec3d, device=DEVICE)
    energy = wp.zeros(1, dtype=wp.float64, device=DEVICE)
    batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=DEVICE)
    n_particles = wp.array(
        np.array([num_atoms], np.int32), dtype=wp.int32, device=DEVICE
    )
    state = _allocate_lbfgs_state(num_atoms, 1, history_size)

    for n_evals in range(1, eval_cap + 1):
        per_atom_energy, f = lennard_jones(positions.numpy())
        forces.assign(f)
        energy.assign(np.array([per_atom_energy.sum()]))
        lbfgs_step(
            positions=positions,
            forces=forces,
            energy=energy,
            batch_idx=batch_idx,
            n_particles=n_particles,
            force_tol=force_tol,
            maxstep=maxstep,
            **state,
        )
        wp.synchronize()
        if state["status"].numpy()[0] != LBFGS_NEED_EVAL:
            final = np.linalg.norm(lennard_jones(positions.numpy())[1], axis=1).max()
            return n_evals, state["status"].numpy()[0] == LBFGS_CONVERGED, final
    final = np.linalg.norm(lennard_jones(positions.numpy())[1], axis=1).max()
    return eval_cap, False, final


def run_fire2(
    start, force_tol, dt_start=0.02, maxstep=0.05, tmax=0.1, eval_cap=EVAL_CAP
):
    """Relax with FIRE2; return (evaluations, converged, final max force)."""
    num_atoms = start.shape[0]
    positions = wp.array(start.copy(), dtype=wp.vec3d, device=DEVICE)
    velocities = wp.zeros(num_atoms, dtype=wp.vec3d, device=DEVICE)
    forces = wp.zeros(num_atoms, dtype=wp.vec3d, device=DEVICE)
    batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=DEVICE)
    alpha = wp.array(np.array([0.09]), dtype=wp.float64, device=DEVICE)
    dt = wp.array(np.array([dt_start]), dtype=wp.float64, device=DEVICE)
    nsteps_inc = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    scratch = [wp.zeros(1, dtype=wp.float64, device=DEVICE) for _ in range(4)]

    for n_evals in range(1, eval_cap + 1):
        f = lennard_jones(positions.numpy())[1]
        current = np.linalg.norm(f, axis=1).max()
        if current <= force_tol:
            return n_evals, True, current
        forces.assign(f)
        fire2_step(
            positions,
            velocities,
            forces,
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


def best_fire2(start, force_tol, sweep=None, eval_cap=EVAL_CAP):
    """FIRE2 at its best over a small hyperparameter sweep.

    Comparing against an untuned baseline would overstate the result; FIRE2 is
    sensitive to its timestep on this system. The grid comes from the
    ``lbfgs.fire2_sweep`` block of the config file.
    """
    sweep = FIRE2_SWEEP if sweep is None else sweep
    best = (eval_cap + 1, False, np.inf, None)
    for dt_start in sweep["dt_start"]:
        for maxstep in sweep["maxstep"]:
            evals, converged, final = run_fire2(
                start,
                force_tol,
                dt_start=dt_start,
                maxstep=maxstep,
                eval_cap=eval_cap,
            )
            if converged and evals < best[0]:
                best = (evals, converged, final, (dt_start, maxstep))
    if best[3] is None:
        evals, converged, final = run_fire2(start, force_tol, eval_cap=eval_cap)
        return evals, converged, final, "none converged"
    return best


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


def run_gates(sizes, eval_ratio, warmup=10, runs=50):
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
    """
    from nvalchemiops.dynamics.optimizers import fire2_step
    from nvalchemiops.torch.lbfgs import lbfgs_step_coord

    print(
        f"{'atoms':>9} {'eager ms':>9} {'graph ms':>9} {'fire2 ms':>9} "
        f"{'eager/f2':>9} {'graph gain':>11} {'break-even':>12}"
    )
    rows = []
    for num_atoms in sizes:
        rng = np.random.default_rng(0)
        positions = torch.tensor(
            rng.normal(size=(num_atoms, 3)), dtype=torch.float64, device=DEVICE
        )
        forces = torch.zeros_like(positions)
        energy = torch.zeros(1, dtype=torch.float64, device=DEVICE)
        batch_idx = torch.zeros(num_atoms, dtype=torch.int32, device=DEVICE)
        n_particles = torch.full((1,), num_atoms, dtype=torch.int32, device=DEVICE)
        buffers = _allocate_lbfgs_buffers_torch(num_atoms, 1, 6)
        forces.copy_(-positions)
        energy.copy_((0.5 * (positions**2).sum()).reshape(1))

        def step():
            lbfgs_step_coord(
                positions,
                forces,
                energy,
                batch_idx,
                n_particles,
                *buffers,
                force_tol=1e-12,
                maxstep=0.5,
            )

        eager = _time_ms(step, warmup, runs)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        wp.synchronize()
        graph = torch.cuda.CUDAGraph()
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
            with torch.cuda.graph(graph):
                step()
        torch.cuda.synchronize()
        graphed = _time_ms(graph.replay, warmup, runs)

        wp_positions = wp.array(
            rng.normal(size=(num_atoms, 3)), dtype=wp.vec3d, device=DEVICE
        )
        wp_velocities = wp.zeros(num_atoms, dtype=wp.vec3d, device=DEVICE)
        wp_forces = wp.zeros(num_atoms, dtype=wp.vec3d, device=DEVICE)
        wp_batch = wp.zeros(num_atoms, dtype=wp.int32, device=DEVICE)
        alpha = wp.array(np.array([0.09]), dtype=wp.float64, device=DEVICE)
        dt = wp.array(np.array([0.02]), dtype=wp.float64, device=DEVICE)
        nsteps_inc = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        scratch = [wp.zeros(1, dtype=wp.float64, device=DEVICE) for _ in range(4)]
        fire2 = _time_ms(
            lambda: fire2_step(
                wp_positions,
                wp_velocities,
                wp_forces,
                wp_batch,
                alpha,
                dt,
                nsteps_inc,
                *scratch,
                maxstep=0.05,
            ),
            warmup,
            runs,
        )

        # n_L (C + O_L) < n_F (C + O_F), with n_L / n_F = eval_ratio.
        n_fire2 = 1000.0
        n_lbfgs = eval_ratio * n_fire2
        break_even_ms = (n_lbfgs * eager - n_fire2 * fire2) / (n_fire2 - n_lbfgs)
        rows.append((num_atoms, eager, graphed, fire2, break_even_ms))
        print(
            f"{num_atoms:>9} {eager:>9.4f} {graphed:>9.4f} {fire2:>9.4f} "
            f"{eager / fire2:>8.2f}x {eager / graphed:>10.1f}x "
            f"{break_even_ms * 1000:>11.1f}us"
        )

    largest = rows[-1]
    print(
        f"\nper-step ratio at {largest[0]} atoms: {largest[1] / largest[3]:.2f}x FIRE2"
    )
    print(
        "break-even model cost is negative wherever L-BFGS wins outright; any "
        "realistic machine-learned potential costs far more than these figures."
    )
    return rows


def _load_lbfgs_config(path):
    """Read the ``lbfgs`` block of the benchmark config, if there is one.

    The benchmark is useful standalone, so a missing file or section is not an
    error -- the module-level fallbacks apply instead.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return {}
    return load_config(path).get("lbfgs", {}) or {}


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
        "--eval-ratio",
        type=float,
        default=None,
        help="measured L-BFGS/FIRE2 evaluation ratio, used for the break-even cost",
    )
    args = parser.parse_args()

    config = _load_lbfgs_config(args.config)
    gates_config = config.get("gates", {}) or {}

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

    if args.gates:
        run_gates(
            pick(
                args.gate_sizes,
                "system_sizes",
                [10_000, 100_000, 1_000_000],
                section=gates_config,
            ),
            pick(args.eval_ratio, "eval_ratio", 0.142, section=gates_config),
            gates_config.get("warmup", 10),
            gates_config.get("runs", 50),
        )
        return

    rows = []
    print(
        f"{'atoms':>6} {'seed':>5} {'lbfgs':>7} {'fire2':>7} "
        f"{'ratio':>7}  {'fire2 config':>14}"
    )
    for num_atoms in sizes:
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            start = rng.normal(size=(num_atoms, 3)) * (num_atoms ** (1 / 3)) * 0.55

            lb_evals, lb_ok, lb_force = run_lbfgs(
                start, force_tol, history_size, maxstep, eval_cap
            )
            f2_evals, f2_ok, f2_force, f2_cfg = best_fire2(
                start, force_tol, sweep, eval_cap
            )
            ratio = lb_evals / f2_evals
            rows.append(
                {
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
                f"{num_atoms:>6} {seed:>5} {lb_evals:>7} {f2_evals:>7} "
                f"{ratio:>7.3f}  {str(f2_cfg):>14}"
                f"{'' if (lb_ok and f2_ok) else '  (capped)'}"
            )

    ratios = [r["ratio"] for r in rows]
    geo_mean = float(np.exp(np.mean(np.log(ratios))))
    all_lbfgs_ok = all(r["lbfgs_converged"] for r in rows)
    print(f"\ngeometric mean evaluation ratio (L-BFGS / FIRE2): {geo_mean:.3f}")
    print(f"worst individual ratio:                          {max(ratios):.3f}")
    print(f"every L-BFGS run converged:                      {all_lbfgs_ok}")
    capped = [r for r in rows if not r["fire2_converged"]]
    if capped:
        print(
            f"note: FIRE2 hit the {eval_cap}-evaluation cap in {len(capped)} case(s), "
            "so those ratios are upper bounds on L-BFGS's advantage."
        )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "lbfgs_vs_fire2_evaluations.csv"
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
