#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified Benchmark Suite for NVIDIA ALCHEMI Toolkit-Ops.

Loads per-module YAML configs and dispatches benchmarks in-process.
CLI flags override YAML values. Each sub-benchmark can also run standalone.
Plots are generated automatically unless --no-plot is specified.

Usage:
    python benchmark_suite.py --benchmark all
    python benchmark_suite.py --benchmark nl --system cscl --mode system_size
    python benchmark_suite.py --benchmark d3 el --system nh3
    python benchmark_suite.py --no-plot --benchmark nl
    python benchmark_suite.py --plot-only benchmarks/benchmark-results/run_2026-02-17/
"""

import argparse
import importlib
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

from benchmarks.config import (
    add_common_cli_args,
    load_yaml_config,
    merge_common_cli_overrides,
)
from benchmarks.utils import (
    create_run_directory,
    write_run_log,
)

SCRIPT_DIR = Path(__file__).parent

# Per-module config + runner module mapping. ``label`` is the pretty
# suffix used in the results summary; ``module`` is the import path to
# the runner's ``run_from_config`` entry point.
RUNNERS = {
    "nl": {
        "label": "NL",
        "config": SCRIPT_DIR / "neighborlist" / "benchmark_config.yaml",
        "module": "benchmarks.neighborlist.benchmark_neighborlist",
    },
    "d3": {
        "label": "D3",
        "config": SCRIPT_DIR / "interactions" / "dispersion" / "benchmark_config.yaml",
        "module": "benchmarks.interactions.dispersion.benchmark_dftd3",
    },
    "el": {
        "label": "EL",
        "config": SCRIPT_DIR
        / "interactions"
        / "electrostatics"
        / "benchmark_config.yaml",
        "module": "benchmarks.interactions.electrostatics.benchmark_electrostatics",
    },
}


def parse_args():
    """Parse command-line arguments for the benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Unified Benchmark Suite (NL + D3 + Electrostatics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_suite.py --benchmark all
    python benchmark_suite.py --benchmark d3 --system cscl --mode system_size
    python benchmark_suite.py --benchmark all --timing-runs 50
    python benchmark_suite.py --benchmark nl --no-plot
    python benchmark_suite.py --plot-only benchmark-results/run_2026-02-17/

Benchmark aliases:
    nl      Neighbor List
    d3      DFT-D3 Dispersion
    el      Electrostatics (Ewald + PME)
    all     All benchmarks

Each module reads its own benchmark_config.yaml. Global CLI flags override
YAML values across all modules. Run individual benchmarks standalone for
module-specific CLI options.
        """,
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        nargs="+",
        default=["all"],
        choices=["nl", "d3", "el", "all"],
        help="Benchmarks to run",
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting after benchmarks",
    )
    parser.add_argument(
        "--plot-only",
        type=Path,
        default=None,
        metavar="RESULTS_DIR",
        help="Skip benchmarks, only generate plots from existing results directory",
    )
    return parser.parse_args()


def main():
    """Run the unified benchmark suite."""
    args = parse_args()

    benchmarks = {"nl", "d3", "el"} if "all" in args.benchmark else set(args.benchmark)

    # JAX_ENABLE_X64 must be set BEFORE the first `import jax`. EL needs
    # f64 (PME/Ewald accuracy); NL/D3 are f32-safe but share the same
    # Python process in the suite, so JAX commits to whatever x64 was
    # when NL ran first. Set it unconditionally when any JAX benchmark
    # is queued so the env is consistent regardless of module order.
    if args.backend == "jax":
        os.environ.setdefault("JAX_ENABLE_X64", "1")

    print("=" * 70)
    print("NVIDIA ALCHEMI Toolkit-Ops Benchmark Suite")
    print("=" * 70)
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except (AssertionError, RuntimeError):
        gpu_name = "N/A (no CUDA)"
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Benchmarks: {', '.join(sorted(benchmarks))}")
    print(f"Systems: {args.system or 'all (from YAML)'}")
    print(f"Modes: {args.mode or 'all (from YAML)'}")
    print("=" * 70)

    start_time = datetime.now()

    base_dir = args.output_dir or (SCRIPT_DIR / "benchmark-results")
    run_dir = create_run_directory(base_dir, prefix="run")
    print(f"\nOutput: {run_dir}")

    results_summary = {}
    for key in ("nl", "d3", "el"):
        if key not in benchmarks:
            continue
        info = RUNNERS[key]
        config_path = info["config"]
        if not config_path.exists():
            print(
                f"\nWARNING: {info['label']} config not found at {config_path}, skipping"
            )
            continue
        runner = importlib.import_module(info["module"])
        config = load_yaml_config(config_path)
        config = merge_common_cli_overrides(config, args)
        results = runner.run_from_config(config, output_dir=run_dir)
        results_summary[info["label"]] = len(results)

    end_time = datetime.now()
    extra = {
        "Benchmarks run": ", ".join(sorted(benchmarks)),
        "Systems": str(args.system or "all"),
        "Modes": str(args.mode or "all"),
    }
    for name, count in results_summary.items():
        extra[f"{name} results"] = count
    write_run_log(run_dir, start_time, end_time, extra_info=extra)

    # --- Plotting ---
    if not args.no_plot:
        _generate_plots(run_dir)

    # Summary
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUITE COMPLETE")
    for name, count in results_summary.items():
        print(f"  {name}: {count} results")
    total = sum(results_summary.values())
    print(f"  Total: {total} results")
    print(f"  Output: {run_dir}")
    print(f"  Run log: {run_dir / 'RUN_LOG.md'}")
    print("=" * 70)

    return 0 if total > 0 else 1


def _generate_plots(results_dir):
    """Generate 3-panel review plots and single-panel docs plots from CSVs.

    Plotting is a secondary concern — CSVs and RUN_LOG.md are already
    on disk by the time this runs, so a missing matplotlib or plotting
    import failure must not mask a successful benchmark run.
    """
    try:
        from benchmarks.plotting.plot_benchmarks import (
            detect_and_plot,
            plot_single_panel,
        )
    except ImportError as e:
        print(f"\n  WARNING: plotting unavailable ({e}). Results saved to {results_dir}")
        return

    results_dir = Path(results_dir)
    csvs = sorted(results_dir.glob("*.csv"))
    if not csvs:
        print("\n  No CSVs found, skipping plots")
        return

    print(f"\n{'=' * 70}")
    print(f"GENERATING PLOTS ({len(csvs)} CSVs)")
    print("=" * 70)

    # 3-panel review plots
    three_panel_dir = results_dir
    for csv in csvs:
        try:
            detect_and_plot(csv, three_panel_dir)
        except Exception as e:
            print(f"  ERROR (3-panel) {csv.name}: {e}")

    # Single-panel plots for docs
    single_dir = results_dir / "single-panels"
    single_dir.mkdir(exist_ok=True)
    for csv in csvs:
        for panel in ("time", "throughput", "memory"):
            try:
                out = single_dir / f"{csv.stem}-{panel}.png"
                plot_single_panel(csv, panel, out)
            except Exception as e:
                print(f"  ERROR (single) {csv.stem}-{panel}: {e}")

    print(f"  3-panel plots: {three_panel_dir}")
    print(f"  Single-panel plots: {single_dir}")


if __name__ == "__main__":
    args_check = parse_args()
    if args_check.plot_only:
        _generate_plots(args_check.plot_only)
        sys.exit(0)
    sys.exit(main())
