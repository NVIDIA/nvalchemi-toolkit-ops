#!/usr/bin/env python3
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

"""
FIRE vs FIRE2 Accuracy Comparison Benchmark
=============================================

Compares FIRE1 and FIRE2 optimizers on Lennard-Jones systems, reporting both
steps-to-convergence and wall-clock time.  Supports two optimization modes:

- **Fixed cell** (coordinate-only): ``run_fire()`` vs ``run_fire2()``
- **Variable cell** (extended arrays with cell DOFs):
  ``run_fire_cell()`` vs ``run_fire2_cell()``

Configuration is loaded from ``benchmark_config.yaml`` (``fire_compare`` section).

Usage::

    python -m benchmarks.dynamics.benchmark_fire_compare [--config benchmark_config.yaml]
                                                          [--output-dir ./benchmark_results]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from benchmarks.dynamics.shared_utils import (
    NvalchemiOpsBenchmark,
    NvalchemiopsLJModel,
    create_lj_system,
    get_gpu_sku,
    load_config,
)

# LJ potential defaults (used when not specified in config)
_POTENTIAL = dict(epsilon=0.0104, sigma=3.40, cutoff=8.5)
_SKIN = 1.0
_NEIGHBOR_REBUILD = 10


_DEFAULT_SEED = 42


def _make_bench(num_atoms, perturbation=0.1, batch_size=1, seed=_DEFAULT_SEED):
    """Create a NvalchemiOpsBenchmark for a perturbed LJ system."""
    torch.manual_seed(seed)
    positions, cell, masses, velocities = create_lj_system(
        num_atoms=num_atoms,
        lattice_constant=5.26,
        temperature=300.0,
        device="cuda",
        dtype=torch.float64,
    )
    actual_atoms = positions.shape[0]

    if batch_size == 1:
        positions = positions + torch.randn_like(positions) * perturbation
        pbc = torch.tensor([True, True, True], device=positions.device)

        lj_model = NvalchemiopsLJModel(
            **_POTENTIAL,
            cell=cell,
            batch_idx=None,
            device="cuda",
            dtype=torch.float64,
        )
        return NvalchemiOpsBenchmark(
            positions=positions,
            cell=cell,
            masses=masses,
            pbc=pbc,
            model=lj_model,
            skin=_SKIN,
            neighbor_rebuild_interval=_NEIGHBOR_REBUILD,
        ), actual_atoms
    else:
        # Batched system
        pos_list, mass_list, cell_list = [], [], []
        for _ in range(batch_size):
            pos_list.append(positions + torch.randn_like(positions) * perturbation)
            mass_list.append(masses)
            cell_list.append(cell)
        batch_positions = torch.cat(pos_list, dim=0)
        batch_masses = torch.cat(mass_list, dim=0)
        batch_cells = torch.cat(cell_list, dim=0)
        batch_idx = torch.repeat_interleave(
            torch.arange(batch_size, device="cuda"),
            actual_atoms,
        ).to(torch.int32)
        atom_ptr = torch.arange(
            0,
            (batch_size + 1) * actual_atoms,
            actual_atoms,
            device="cuda",
            dtype=torch.int64,
        )
        pbc = torch.tensor([True, True, True], device="cuda")

        lj_model = NvalchemiopsLJModel(
            **_POTENTIAL,
            cell=batch_cells,
            batch_idx=batch_idx,
            device="cuda",
            dtype=torch.float64,
        )
        return NvalchemiOpsBenchmark(
            positions=batch_positions,
            cell=batch_cells,
            masses=batch_masses,
            pbc=pbc,
            model=lj_model,
            skin=_SKIN,
            neighbor_rebuild_interval=_NEIGHBOR_REBUILD,
            batch_idx=batch_idx,
            atom_ptr=atom_ptr,
        ), actual_atoms


# ---------------------------------------------------------------------------
# Fixed-cell comparison
# ---------------------------------------------------------------------------


def run_fixed_cell_comparison(
    system_sizes,
    force_tol,
    max_steps,
    check_interval,
    fire1_params,
    fire2_params,
    perturbation,
    seed=_DEFAULT_SEED,
):
    """Run fixed-cell (coordinate-only) FIRE vs FIRE2 comparison.

    Parameters
    ----------
    system_sizes : list[int]
        Number of atoms per system to benchmark.
    force_tol : float
        Convergence criterion for maximum force magnitude (eV/A).
    max_steps : int
        Maximum number of optimization steps.
    check_interval : int
        Interval to check convergence.
    fire1_params : dict
        FIRE1 hyperparameters passed to ``run_fire()``.
    fire2_params : dict
        FIRE2 hyperparameters passed to ``run_fire2()``.
    perturbation : float
        Random perturbation magnitude (A) applied to initial positions.
    seed : int
        RNG seed for reproducible initial perturbations.

    Returns
    -------
    list[dict]
        Result dicts for CSV output.
    """
    rows = []

    header = (
        f"{'N':>8} {'FIRE1 steps':>12} {'FIRE1 time':>12} "
        f"{'FIRE2 steps':>12} {'FIRE2 time':>12} {'Step ratio':>11} {'Speedup':>8}"
    )
    print(f"\n--- Fixed Cell (coordinate-only) — ftol={force_tol} eV/A ---")
    print(header)
    print("-" * len(header))

    for num_atoms in system_sizes:
        # FIRE1
        bench1, actual = _make_bench(num_atoms, perturbation=perturbation, seed=seed)
        r1 = bench1.run_fire(
            max_steps=max_steps,
            force_tolerance=force_tol,
            check_interval=check_interval,
            **fire1_params,
        )

        # FIRE2 (fresh benchmark with same system)
        bench2, _ = _make_bench(num_atoms, perturbation=perturbation, seed=seed)
        r2 = bench2.run_fire2(
            max_steps=max_steps,
            force_tolerance=force_tol,
            check_interval=check_interval,
            **fire2_params,
        )

        step_ratio = r2.num_steps / r1.num_steps if r1.num_steps > 0 else float("nan")
        speedup = r1.total_time / r2.total_time if r2.total_time > 0 else float("nan")
        f1_converged = r1.num_steps < max_steps
        f2_converged = r2.num_steps < max_steps

        print(
            f"{actual:>8} "
            f"{r1.num_steps:>12} {r1.total_time:>10.3f}s "
            f"{r2.num_steps:>12} {r2.total_time:>10.3f}s "
            f"{step_ratio:>10.2f}x {speedup:>7.2f}x"
        )

        rows.append(
            {
                "num_atoms": actual,
                "opt_type": "fixed_cell",
                "method": "fire1",
                "steps": r1.num_steps,
                "wall_time_s": f"{r1.total_time:.4f}",
                "converged": f1_converged,
            }
        )
        rows.append(
            {
                "num_atoms": actual,
                "opt_type": "fixed_cell",
                "method": "fire2",
                "steps": r2.num_steps,
                "wall_time_s": f"{r2.total_time:.4f}",
                "converged": f2_converged,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Variable-cell comparison
# ---------------------------------------------------------------------------


def run_variable_cell_comparison(
    system_sizes,
    force_tol,
    pressure_tol,
    max_steps,
    check_interval,
    fire1_params,
    fire2_params,
    perturbation,
    seed=_DEFAULT_SEED,
):
    """Run variable-cell FIRE vs FIRE2 comparison.

    Parameters
    ----------
    system_sizes : list[int]
        Number of atoms per system to benchmark.
    force_tol : float
        Convergence criterion for maximum force magnitude (eV/A).
    pressure_tol : float
        Convergence criterion for maximum stress (kBar).
    max_steps : int
        Maximum number of optimization steps.
    check_interval : int
        Interval to check convergence.
    fire1_params : dict
        FIRE1 hyperparameters passed to ``run_fire_cell()``.
    fire2_params : dict
        FIRE2 hyperparameters passed to ``run_fire2_cell()``.
    perturbation : float
        Random perturbation magnitude (A) applied to initial positions.
    seed : int
        RNG seed for reproducible initial perturbations.

    Returns
    -------
    list[dict]
        Result dicts for CSV output.
    """
    rows = []

    header = (
        f"{'N':>8} {'FIRE1 steps':>12} {'FIRE1 time':>12} "
        f"{'FIRE2 steps':>12} {'FIRE2 time':>12} {'Step ratio':>11} {'Speedup':>8}"
    )
    print(f"\n--- Variable Cell — ftol={force_tol} eV/A, ptol={pressure_tol} kBar ---")
    print(header)
    print("-" * len(header))

    for num_atoms in system_sizes:
        # FIRE1 variable-cell
        bench1, actual = _make_bench(num_atoms, perturbation=perturbation, seed=seed)
        r1 = bench1.run_fire_cell(
            max_steps=max_steps,
            force_tolerance=force_tol,
            pressure_tolerance=pressure_tol,
            check_interval=check_interval,
            **fire1_params,
        )

        # FIRE2 variable-cell (fresh benchmark with same system)
        bench2, _ = _make_bench(num_atoms, perturbation=perturbation, seed=seed)
        r2 = bench2.run_fire2_cell(
            max_steps=max_steps,
            force_tolerance=force_tol,
            pressure_tolerance=pressure_tol,
            check_interval=check_interval,
            **fire2_params,
        )

        step_ratio = r2.num_steps / r1.num_steps if r1.num_steps > 0 else float("nan")
        speedup = r1.total_time / r2.total_time if r2.total_time > 0 else float("nan")
        f1_converged = r1.num_steps < max_steps
        f2_converged = r2.num_steps < max_steps

        print(
            f"{actual:>8} "
            f"{r1.num_steps:>12} {r1.total_time:>10.3f}s "
            f"{r2.num_steps:>12} {r2.total_time:>10.3f}s "
            f"{step_ratio:>10.2f}x {speedup:>7.2f}x"
        )

        rows.append(
            {
                "num_atoms": actual,
                "opt_type": "variable_cell",
                "method": "fire1",
                "steps": r1.num_steps,
                "wall_time_s": f"{r1.total_time:.4f}",
                "converged": f1_converged,
            }
        )
        rows.append(
            {
                "num_atoms": actual,
                "opt_type": "variable_cell",
                "method": "fire2",
                "steps": r2.num_steps,
                "wall_time_s": f"{r2.total_time:.4f}",
                "converged": f2_converged,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_benchmarks(config: dict, output_dir: Path, seed: int = _DEFAULT_SEED) -> None:
    """Run FIRE1 vs FIRE2 accuracy comparison benchmarks.

    Parameters
    ----------
    config : dict
        Benchmark configuration (uses the ``fire_compare`` section).
    output_dir : Path
        Output directory for CSV files.
    seed : int
        RNG seed for reproducible initial perturbations (overrides config).
    """
    cmp_config = config.get("fire_compare", {})
    if not cmp_config.get("enabled", True):
        print("fire_compare benchmarks disabled in config")
        return

    system_sizes = cmp_config.get("system_sizes", [256, 512, 1024, 2048])
    force_tol = cmp_config.get("force_tolerance", 0.005)
    pressure_tol = cmp_config.get("pressure_tolerance", 0.3)
    max_steps = cmp_config.get("max_steps", 2000)
    check_interval = cmp_config.get("check_interval", 10)
    seed = seed if seed is not None else cmp_config.get("seed", _DEFAULT_SEED)

    # Fixed-cell config
    fixed_cfg = cmp_config.get("fixed_cell", {})
    fixed_enabled = fixed_cfg.get("enabled", True)
    fixed_perturbation = fixed_cfg.get("perturbation", 0.1)

    # Variable-cell config
    var_cfg = cmp_config.get("variable_cell", {})
    var_enabled = var_cfg.get("enabled", True)
    var_perturbation = var_cfg.get("perturbation", 0.1)

    # FIRE1 hyperparameters
    f1_cfg = cmp_config.get("fire1", {})
    fire1_params = {
        "dt_start": f1_cfg.get("dt_start", 1.0),
        "dt_max": f1_cfg.get("dt_max", 10.0),
        "dt_min": f1_cfg.get("dt_min", 0.001),
        "alpha_start": f1_cfg.get("alpha_start", 0.1),
        "n_min": f1_cfg.get("n_min", 5),
        "f_inc": f1_cfg.get("f_inc", 1.1),
        "f_dec": f1_cfg.get("f_dec", 0.5),
        "f_alpha": f1_cfg.get("f_alpha", 0.99),
        "maxstep": f1_cfg.get("maxstep", 0.2),
    }

    # FIRE2 hyperparameters
    f2_cfg = cmp_config.get("fire2", {})
    fire2_params = {
        "dt_start": f2_cfg.get("dt_start", 0.045),
        "tmax": f2_cfg.get("tmax", 0.10),
        "tmin": f2_cfg.get("tmin", 0.005),
        "delaystep": f2_cfg.get("delaystep", 50),
        "dtgrow": f2_cfg.get("dtgrow", 1.09),
        "dtshrink": f2_cfg.get("dtshrink", 0.95),
        "alpha0": f2_cfg.get("alpha0", 0.20),
        "alphashrink": f2_cfg.get("alphashrink", 0.985),
        "maxstep": f2_cfg.get("maxstep", 0.25),
    }

    gpu_sku = get_gpu_sku()
    all_rows = []

    print("FIRE vs FIRE2 Accuracy Comparison")
    print("This benchmark compares convergence speed and wall-clock time.")
    print(f"GPU: {gpu_sku}")
    print(f"Force tolerance: {force_tol} eV/A")
    print(f"Max steps: {max_steps}")
    print(f"Seed: {seed}")

    # Fixed-cell comparison
    if fixed_enabled:
        rows = run_fixed_cell_comparison(
            system_sizes,
            force_tol,
            max_steps,
            check_interval,
            fire1_params,
            fire2_params,
            fixed_perturbation,
            seed=seed,
        )
        all_rows.extend(rows)

    # Variable-cell comparison
    if var_enabled:
        rows = run_variable_cell_comparison(
            system_sizes,
            force_tol,
            pressure_tol,
            max_steps,
            check_interval,
            fire1_params,
            fire2_params,
            var_perturbation,
            seed=seed,
        )
        all_rows.extend(rows)

    # Write CSV
    if all_rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"fire_compare_{gpu_sku}.csv"
        fieldnames = [
            "num_atoms",
            "opt_type",
            "method",
            "steps",
            "wall_time_s",
            "converged",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote results to {csv_path}")

    print("\nStep ratio < 1 = FIRE2 needs fewer steps")
    print("Speedup > 1 = FIRE2 is faster in wall-clock time")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FIRE vs FIRE2 accuracy comparison benchmark"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--system-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Override system sizes from config",
    )
    parser.add_argument(
        "--force-tol",
        type=float,
        default=None,
        help="Override force convergence tolerance (eV/A)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum optimization steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible perturbations (default: from config or 42)",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.system_sizes is not None:
        config.setdefault("fire_compare", {})["system_sizes"] = args.system_sizes
    if args.force_tol is not None:
        config.setdefault("fire_compare", {})["force_tolerance"] = args.force_tol
    if args.max_steps is not None:
        config.setdefault("fire_compare", {})["max_steps"] = args.max_steps

    seed = args.seed
    if seed is None:
        seed = config.get("fire_compare", {}).get("seed", _DEFAULT_SEED)

    output_dir = Path(args.output_dir)
    run_benchmarks(config, output_dir, seed=seed)


if __name__ == "__main__":
    main()
