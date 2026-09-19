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

"""L-BFGS against FIRE2 on OMat24 structures under a machine-learned potential.

The companion to ``benchmark_lbfgs.py``, which measures Lennard-Jones argon
clusters. That workload is reproducible from this repository alone but is not
the setting L-BFGS is meant for: a smooth pair potential on a small cluster
exercises neither the stiff, anisotropic curvature of a relaxing crystal nor a
force field that is not the gradient of its own energy. This runner measures
that setting directly -- periodic inorganic structures, relaxed with MACE.

Metric, as in the LJ runner, is **force evaluations to convergence**: with a
machine-learned potential the model dominates, so evaluations are the cost.
Each structure is relaxed independently, and FIRE2's timestep is swept per
structure with its best converged run used as the baseline, so the comparison
is not skewed by an untuned baseline.

Not part of the default suite. It needs ``mace-torch`` and a model checkpoint,
neither of which this package depends on, and a structure collection supplied
by the caller::

    python -m benchmarks.dynamics.benchmark_lbfgs_omat24 \\
        --structures path/to/omat-collection/ --output-dir ./benchmark_results

``mace-torch`` 0.3.x needs ``e3nn<0.5``; this package needs a newer one, so run
it from a virtual environment that pins the older ``e3nn`` rather than
downgrading in place.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pathlib
import time

import numpy as np
import torch
import warp as wp

from nvalchemiops.dynamics.optimizers import fire2_step
from nvalchemiops.torch.lbfgs import lbfgs_prepare_state, lbfgs_step_coord

#: Convergence criterion on the largest per-atom force, eV/Angstrom. 0.05 is
#: the usual target for a machine-learned relaxation; tighter than the model's
#: own accuracy buys nothing.
FORCE_TOL = 0.05

#: Give up after this many evaluations. A capped run bounds its count from
#: below, so its ratio is a bound rather than a measurement, and is reported
#: separately.
EVAL_CAP = 300

#: Largest displacement in one step, Angstrom. Shared by both optimizers, so
#: neither is handed a different trust region.
MAXSTEP = 0.2

#: FIRE2 timestep grid, femtoseconds. Swept per structure because FIRE2 is
#: sensitive to it and an untuned baseline would overstate the result.
FIRE2_DT_GRID = (0.5, 1.0, 2.0, 4.0)

DEFAULT_DEVICE = "cuda:0"
DTYPES = {"float32": (torch.float32, wp.vec3f, wp.float32, np.float32),
          "float64": (torch.float64, wp.vec3d, wp.float64, np.float64)}  # fmt: skip


def load_structures(path, limit=None):
    """Every frame of every ``.xyz`` under ``path``, in filename order."""
    import ase.io

    files = sorted(glob.glob(os.path.join(str(path), "*.xyz")))
    if not files:
        raise SystemExit(f"no .xyz files under {path}")
    frames = []
    for name in files:
        frames += ase.io.read(name, ":")
    return frames[:limit] if limit else frames


def make_calculator(model, device, dtype_name):
    """A MACE calculator, with a readable failure if mace-torch is absent."""
    try:
        from mace.calculators import mace_mp
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "this benchmark needs mace-torch, which this package does not "
            "depend on. Install it in a virtual environment that also pins "
            "e3nn<0.5, then re-run."
        ) from exc
    return mace_mp(model=model, default_dtype=dtype_name, device=device)


class Model:
    """Forces for one structure, counted.

    Counting here rather than in each optimizer loop keeps the two honest:
    whatever a loop does, the number reported is the number of times the
    potential actually ran.
    """

    def __init__(self, atoms, calculator):
        self.atoms = atoms.copy()
        self.atoms.calc = calculator
        self.n_evals = 0

    def forces(self, positions):
        """Forces at ``positions``, as a NumPy array; cell held fixed."""
        self.atoms.set_positions(positions)
        self.n_evals += 1
        return self.atoms.get_forces()


def _fmax(forces):
    return float(np.linalg.norm(forces, axis=1).max())


def run_lbfgs(atoms, calculator, device, torch_dtype, force_tol=FORCE_TOL,
              maxstep=MAXSTEP, eval_cap=EVAL_CAP, history_size=6):  # fmt: skip
    """Relax with L-BFGS; return (evaluations, converged, final fmax)."""
    model = Model(atoms, calculator)
    n = len(atoms)
    positions = torch.tensor(atoms.get_positions(), dtype=torch_dtype, device=device)
    forces = torch.empty_like(positions)
    batch_idx = torch.zeros(n, dtype=torch.int32, device=device)
    state = lbfgs_prepare_state(n, 1, dtype=torch_dtype, history_size=history_size,
                                device=device)  # fmt: skip

    for _ in range(eval_cap):
        f = model.forces(positions.detach().cpu().numpy())
        current = _fmax(f)
        if current <= force_tol:
            return model.n_evals, True, current
        forces.copy_(torch.as_tensor(f, dtype=torch_dtype, device=device))
        lbfgs_step_coord(positions, forces, state, batch_idx, maxstep=maxstep)
        torch.cuda.synchronize()
    return model.n_evals, False, current


def run_fire2(atoms, calculator, device, torch_dtype, dt_start, force_tol=FORCE_TOL,
              maxstep=MAXSTEP, eval_cap=EVAL_CAP):  # fmt: skip
    """Relax with FIRE2 at one timestep; return (evaluations, converged, fmax)."""
    model = Model(atoms, calculator)
    n = len(atoms)
    vec_dtype = wp.vec3f if torch_dtype == torch.float32 else wp.vec3d
    scalar = wp.float32 if torch_dtype == torch.float32 else wp.float64
    np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64

    positions = torch.tensor(atoms.get_positions(), dtype=torch_dtype, device=device)
    forces = torch.empty_like(positions)

    # ``fire2_step`` is a raw Warp launcher over PyTorch storage, so Warp has
    # to be bound to the Torch stream. The scope covers the Warp allocations
    # and the ``from_torch`` views as well as the launches, and is held across
    # the whole relaxation: a view built on one stream and launched on another
    # is ordered against neither its producer nor its consumer. The L-BFGS arm
    # needs none of this -- its registered operator binds the stream itself.
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        wp_positions = wp.from_torch(positions, dtype=vec_dtype)
        wp_forces = wp.from_torch(forces, dtype=vec_dtype)
        wp_velocities = wp.zeros(n, dtype=vec_dtype, device=device)
        wp_batch = wp.zeros(n, dtype=wp.int32, device=device)
        alpha = wp.array(np.array([0.09], np_dtype), dtype=scalar, device=device)
        dt = wp.array(np.array([dt_start], np_dtype), dtype=scalar, device=device)
        nsteps_inc = wp.zeros(1, dtype=wp.int32, device=device)
        scratch = [wp.zeros(1, dtype=scalar, device=device) for _ in range(4)]

        for _ in range(eval_cap):
            f = model.forces(positions.detach().cpu().numpy())
            current = _fmax(f)
            if current <= force_tol:
                return model.n_evals, True, current
            forces.copy_(torch.as_tensor(f, dtype=torch_dtype, device=device))
            fire2_step(
                wp_positions, wp_velocities, wp_forces, wp_batch,
                alpha, dt, nsteps_inc, *scratch,
                maxstep=maxstep, tmax=2.0 * dt_start, tmin=0.002,
                dtgrow=1.1, dtshrink=0.5, delaystep=5,
                alpha0=0.09, alphashrink=0.99,
            )  # fmt: skip
            torch.cuda.synchronize()
    return model.n_evals, False, current


def best_fire2(atoms, calculator, device, torch_dtype, grid=FIRE2_DT_GRID, **kwargs):
    """FIRE2 at its best over the timestep grid.

    When nothing converges, the first capped run is reported as it stands
    rather than re-run. It already bounds the evaluation count from below, and
    repeating a configuration that has just hit the cap would spend another
    ``eval_cap`` model evaluations reproducing a number already in hand --
    which with MACE is the most expensive thing this benchmark could do.
    """
    best = None
    first_capped = None
    for dt_start in grid:
        evals, converged, final = run_fire2(
            atoms, calculator, device, torch_dtype, dt_start, **kwargs
        )
        if converged:
            if best is None or evals < best[0]:
                best = (evals, converged, final, dt_start)
        elif first_capped is None:
            first_capped = (evals, converged, final, f"none converged (dt={dt_start})")
    return best if best is not None else first_capped


def main():
    """Relax every structure with both optimizers and report the ratio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=pathlib.Path, required=True,
                        help="directory of .xyz files")  # fmt: skip
    parser.add_argument("--model", type=str, default="medium-mpa-0",
                        help="MACE model name passed to mace_mp")  # fmt: skip
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N structures")  # fmt: skip
    parser.add_argument("--force-tol", type=float, default=FORCE_TOL)
    parser.add_argument("--eval-cap", type=int, default=EVAL_CAP)
    parser.add_argument("--maxstep", type=float, default=MAXSTEP)
    parser.add_argument("--dtype", type=str, choices=sorted(DTYPES),
                        default="float64",
                        help="MACE recommends float64 for geometry optimization")  # fmt: skip
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    torch.cuda.set_device(torch.device(args.device))
    torch_dtype = DTYPES[args.dtype][0]
    frames = load_structures(args.structures, args.limit)
    calculator = make_calculator(args.model, args.device, args.dtype)

    print(f"{len(frames)} structures, {args.model}, {args.dtype}, "
          f"fmax <= {args.force_tol} eV/A, cap {args.eval_cap}")  # fmt: skip
    print(f"{'#':>4} {'atoms':>6} {'lbfgs':>7} {'fire2':>7} {'ratio':>7} "
          f"{'fire2 dt':>9}")  # fmt: skip

    common = dict(force_tol=args.force_tol, maxstep=args.maxstep,
                  eval_cap=args.eval_cap)  # fmt: skip
    rows = []
    started = time.perf_counter()
    for i, atoms in enumerate(frames):
        lb_evals, lb_ok, lb_force = run_lbfgs(
            atoms, calculator, args.device, torch_dtype, **common
        )
        f2_evals, f2_ok, f2_force, f2_dt = best_fire2(
            atoms, calculator, args.device, torch_dtype, **common
        )
        ratio = lb_evals / f2_evals
        rows.append({
            "index": i, "atoms": len(atoms), "formula": atoms.get_chemical_formula(),
            "lbfgs_evals": lb_evals, "lbfgs_converged": lb_ok, "lbfgs_fmax": lb_force,
            "fire2_evals": f2_evals, "fire2_converged": f2_ok, "fire2_fmax": f2_force,
            "fire2_dt": str(f2_dt), "ratio": ratio,
        })  # fmt: skip
        print(f"{i:>4} {len(atoms):>6} {lb_evals:>7} {f2_evals:>7} {ratio:>7.3f} "
              f"{str(f2_dt):>9}"
              f"{'' if (lb_ok and f2_ok) else '  (capped)'}")  # fmt: skip

    comparable = [r for r in rows if r["lbfgs_converged"] and r["fire2_converged"]]
    fire2_capped = [
        r for r in rows if r["lbfgs_converged"] and not r["fire2_converged"]
    ]
    lbfgs_failed = [r for r in rows if not r["lbfgs_converged"]]

    print(f"\n{len(frames)} structures in {time.perf_counter() - started:.0f} s")
    if comparable:
        ratios = np.array([r["ratio"] for r in comparable])
        print(f"cases where both converged:                      "
              f"{len(comparable)}/{len(rows)}")  # fmt: skip
        print(f"geometric mean evaluation ratio (L-BFGS / FIRE2): "
              f"{float(np.exp(np.mean(np.log(ratios)))):.3f}")  # fmt: skip
        print(f"median ratio:                                    "
              f"{float(np.median(ratios)):.3f}")  # fmt: skip
        print(f"worst individual ratio:                          "
              f"{ratios.max():.3f}")  # fmt: skip
        print(f"structures where L-BFGS used fewer evaluations:  "
              f"{int((ratios < 1.0).sum())}/{len(ratios)}")  # fmt: skip
    else:
        print("no structure had both optimizers converge; no ratio can be quoted")

    if fire2_capped:
        print(f"\nexcluded -- FIRE2 hit the {args.eval_cap}-evaluation cap in "
              f"{len(fire2_capped)} case(s); each ratio there is an upper bound, "
              "not a measurement. L-BFGS converged in all of them.")  # fmt: skip
    if lbfgs_failed:
        print(f"\nexcluded -- L-BFGS hit the cap in {len(lbfgs_failed)} case(s).")

    if args.output_dir is not None and rows:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "lbfgs_vs_fire2_omat24.csv"
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
