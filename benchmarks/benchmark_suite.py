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
import sys
from pathlib import Path

import torch

from benchmarks.utils import (
    create_run_directory,
    get_gpu_sku,
    get_timestamp,
    write_run_readme,
)

SCRIPT_DIR = Path(__file__).parent

# Per-module config paths (relative to benchmarks/)
MODULE_CONFIGS = {
    "nl": SCRIPT_DIR / "neighborlist" / "benchmark_config.yaml",
    "d3": SCRIPT_DIR / "interactions" / "dispersion" / "benchmark_config.yaml",
    "el": SCRIPT_DIR / "interactions" / "electrostatics" / "benchmark_config.yaml",
}


def apply_global_overrides(config, args):
    """Apply global CLI overrides to a per-module config.

    This merges the suite-level CLI flags into a module's YAML config.
    Module-specific flags (cutoffs, methods, accuracies) are not applied here;
    those are handled by each module's own merge_cli_overrides.
    """
    # Timing overrides
    if args.timing_runs is not None:
        config["parameters"]["timing_runs"] = args.timing_runs
    if args.timing_mode is not None:
        config["parameters"]["timing_mode"] = args.timing_mode
    if args.warmup_runs is not None:
        config["parameters"]["warmup_runs"] = args.warmup_runs

    # System filter
    if args.system is not None and "all" not in args.system:
        for sys_name in list(config["systems"].keys()):
            config["systems"][sys_name]["enabled"] = sys_name in args.system

    # Mode filter
    if args.mode is not None and "all" not in args.mode:
        for mode_name in list(config["scaling"].keys()):
            if isinstance(config["scaling"][mode_name], dict):
                config["scaling"][mode_name]["enabled"] = mode_name in args.mode

    # Output override
    if args.output_dir is not None:
        config["output"]["base_dir"] = str(args.output_dir)

    return config


def parse_args():
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
    parser.add_argument(
        "--system",
        "-s",
        nargs="+",
        default=None,
        help="Filter systems (cscl, nh3, or all)",
    )
    parser.add_argument(
        "--mode", "-m", nargs="+", default=None, help="Filter scaling modes"
    )
    parser.add_argument(
        "--timing-runs", "-n", type=int, default=None, help="Override timing iterations"
    )
    parser.add_argument(
        "--timing-mode",
        default=None,
        choices=["batch", "per_run"],
        help="Override timing mode",
    )
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override output directory"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plotting after benchmarks",
    )
    parser.add_argument(
        "--plot-only", type=Path, default=None, metavar="RESULTS_DIR",
        help="Skip benchmarks, only generate plots from existing results directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    benchmarks = {"nl", "d3", "el"} if "all" in args.benchmark else set(args.benchmark)

    print("=" * 70)
    print("NVIDIA ALCHEMI Toolkit-Ops Benchmark Suite")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Benchmarks: {', '.join(sorted(benchmarks))}")
    print(f"Systems: {args.system or 'all (from YAML)'}")
    print(f"Modes: {args.mode or 'all (from YAML)'}")
    print("=" * 70)

    # Create single timestamped run directory for all results
    from datetime import datetime

    start_time = datetime.now()

    base_dir = args.output_dir or (SCRIPT_DIR / "benchmark-results")
    run_dir = create_run_directory(base_dir, prefix="run")
    print(f"\nOutput: {run_dir}")

    results_summary = {}

    # --- Neighbor List ---
    if "nl" in benchmarks:
        config_path = MODULE_CONFIGS["nl"]
        if not config_path.exists():
            print(f"\nWARNING: NL config not found at {config_path}, skipping")
        else:
            from benchmarks.neighborlist.benchmark_neighborlist import (
                load_config,
                run_from_config,
            )

            config = load_config(config_path)
            config = apply_global_overrides(config, args)
            nl_results = run_from_config(config, output_dir=run_dir)
            results_summary["NL"] = len(nl_results)

    # --- DFT-D3 ---
    if "d3" in benchmarks:
        config_path = MODULE_CONFIGS["d3"]
        if not config_path.exists():
            print(f"\nWARNING: D3 config not found at {config_path}, skipping")
        else:
            from benchmarks.interactions.dispersion.benchmark_dftd3 import (
                load_config as load_d3_config,
            )
            from benchmarks.interactions.dispersion.benchmark_dftd3 import (
                run_from_config as run_d3,
            )

            config = load_d3_config(config_path)
            config = apply_global_overrides(config, args)
            d3_results = run_d3(config, output_dir=run_dir)
            results_summary["D3"] = len(d3_results)

    # --- Electrostatics ---
    if "el" in benchmarks:
        config_path = MODULE_CONFIGS["el"]
        if not config_path.exists():
            print(
                f"\nWARNING: Electrostatics config not found at {config_path}, skipping"
            )
        else:
            from benchmarks.interactions.electrostatics.benchmark_electrostatics import (
                load_config as load_el_config,
            )
            from benchmarks.interactions.electrostatics.benchmark_electrostatics import (
                run_from_config as run_el,
            )

            config = load_el_config(config_path)
            config = apply_global_overrides(config, args)
            el_results = run_el(config, output_dir=run_dir)
            results_summary["EL"] = len(el_results)

    # Write README with reproducibility info
    end_time = datetime.now()
    extra = {
        "Benchmarks run": ", ".join(sorted(benchmarks)),
        "Systems": str(args.system or "all"),
        "Modes": str(args.mode or "all"),
    }
    for name, count in results_summary.items():
        extra[f"{name} results"] = count
    write_run_readme(run_dir, start_time, end_time, extra_info=extra)

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
    print(f"  README: {run_dir / 'README.md'}")
    print("=" * 70)

    return 0 if total > 0 else 1


def _generate_plots(results_dir):
    """Generate 3-panel review plots and single-panel docs plots from CSVs."""
    from benchmarks.plotting.plot_benchmarks import plot_single_panel, load_csv, _detect_and_plot

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
            _detect_and_plot(csv, three_panel_dir)
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
