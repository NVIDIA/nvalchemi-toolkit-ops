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

"""Unified Benchmark Suite for NVIDIA ALCHEMI Toolkit-Ops.

Loads per-module YAML configs and dispatches benchmarks in-process.
CLI flags override YAML values. Each sub-benchmark can also run standalone.
Plots are generated automatically unless --no-plot is specified.

Usage:
    python -m benchmarks.benchmark_suite --benchmark all
    python -m benchmarks.benchmark_suite --benchmark nl --system cscl --mode system_size
    python -m benchmarks.benchmark_suite --benchmark d3 el --system nh3
    python -m benchmarks.benchmark_suite --no-plot --benchmark nl
    python -m benchmarks.benchmark_suite --count --benchmark all
    python -m benchmarks.benchmark_suite --plot-only benchmarks/benchmark-results/run_2026-02-17/
"""

import argparse
import csv
import importlib
import io
import sys
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from benchmarks.config import (
    add_common_cli_args,
    load_yaml_config,
    normalize_method_name,
)
from benchmarks.suite_orchestration import (
    _benchmark_case_key,
    _format_case_difference,
    _prepare_run_directory,
    _print_suite_header,
    _report_plan,
    _report_run,
    _run_requested_benchmarks,
    _write_suite_run_log,
)
from benchmarks.suite_orchestration_plots import _generate_plots as _orchestrate_plots
from benchmarks.suite_utils import configure_jax_environment

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
        "module": "benchmarks.interactions.electrostatics.benchmark_electrostatics_suite",
    },
}

SUITE_CSV_PREFIXES = ("nl-", "d3-", "el-")
REPORTABLE_CSV_NAMES = {
    f"{module}-{system}-{mode}-scaling.csv"
    for module in ("nl", "d3", "el")
    for system in ("cscl", "nh3")
    for mode in ("system-size", "constant-workload", "batch")
}

SUPPORTED_BACKENDS = {
    "nl": {"torch", "jax", "warp"},
    "d3": {"torch", "jax"},
    "el": {"torch", "jax"},
}
NL_DOC_CUTOFFS = (6.0, 15.0, 25.0)
REPORTABLE_TIMING_RUNS = 10
REPORTABLE_WARMUP_RUNS = 3


def build_reportable_plan(
    backends: set[str], *, nh3_dir: Path | None = None
) -> list[dict]:
    """Build the complete shipped case matrix for selected backends."""
    rows: list[dict] = []
    for backend in sorted(backends):
        for key in ("nl", "d3", "el"):
            if backend not in SUPPORTED_BACKENDS[key]:
                continue
            info = RUNNERS[key]
            config = load_yaml_config(info["config"])
            if nh3_dir is not None:
                nh3_config = config.get("systems", {}).get("nh3")
                if isinstance(nh3_config, dict):
                    nh3_config["pdb_dir"] = str(nh3_dir)
            runner = importlib.import_module(info["module"])
            with redirect_stdout(io.StringIO()):
                rows.extend(runner.dry_run_from_config(config, backend=backend))
    return rows


def validate_reportable_case_matrix(
    csv_paths: list[Path],
    backends: set[str],
    *,
    nh3_dir: Path | None = None,
) -> dict[str, int]:
    """Validate that result CSVs contain every configured case exactly once."""
    planned_rows = build_reportable_plan(backends, nh3_dir=nh3_dir)
    emitted_rows: list[dict] = []
    for path in csv_paths:
        with path.open(newline="") as csv_file:
            emitted_rows.extend(csv.DictReader(csv_file))

    protocol_values = {
        "timing_runs": {row.get("timing_runs", "") for row in emitted_rows},
        "warmup_runs": {row.get("warmup_runs", "") for row in emitted_rows},
    }
    expected_protocol = {
        "timing_runs": {str(REPORTABLE_TIMING_RUNS)},
        "warmup_runs": {str(REPORTABLE_WARMUP_RUNS)},
    }
    mismatches = [
        f"{field}={sorted(values)} (expected {sorted(expected_protocol[field])})"
        for field, values in protocol_values.items()
        if values != expected_protocol[field]
    ]
    if mismatches:
        raise ValueError(
            "reportable timing protocol mismatch; " + "; ".join(mismatches)
        )

    planned = Counter(_benchmark_case_key(row) for row in planned_rows)
    emitted = Counter(_benchmark_case_key(row) for row in emitted_rows)
    if planned != emitted:
        missing = list((planned - emitted).elements())
        unexpected = list((emitted - planned).elements())
        details = []
        if missing:
            details.append(_format_case_difference("missing", missing))
        if unexpected:
            details.append(_format_case_difference("unexpected", unexpected))
        raise ValueError("reportable case matrix mismatch; " + "; ".join(details))

    return {
        "planned": sum(planned.values()),
        "emitted": sum(emitted.values()),
    }


def validate_backend_selection(backend: str | None, benchmarks: set[str]) -> None:
    """Validate suite-level backend compatibility for the selected benchmarks."""
    if backend is None:
        return
    unsupported = sorted(
        key for key in benchmarks if backend not in SUPPORTED_BACKENDS.get(key, set())
    )
    if unsupported:
        raise ValueError(
            f"Backend {backend!r} is not supported for requested benchmark(s): "
            f"{', '.join(unsupported)}. Supported backends: "
            + ", ".join(
                f"{key}={sorted(SUPPORTED_BACKENDS[key])}"
                for key in sorted(unsupported)
            )
        )


def validate_method_selection(methods: list[str] | None, benchmarks: set[str]) -> None:
    """Validate suite-level method compatibility for selected benchmarks."""
    if methods is None or "el" not in benchmarks:
        return
    multipoles = [
        method
        for method in methods
        if normalize_method_name(method).startswith("multipole")
    ]
    if multipoles:
        raise ValueError(
            "Multipole electrostatics benchmarks are intentionally excluded from "
            f"this suite path: {', '.join(multipoles)}"
        )


def _suite_needs_jax_env(args: argparse.Namespace, benchmarks: set[str]) -> bool:
    """Return True when CLI or selected YAML configs request the JAX backend."""
    if args.backend == "jax":
        return True
    if args.backend is not None:
        return False
    for key in benchmarks:
        config_path = RUNNERS[key]["config"]
        if not config_path.exists():
            continue
        config = load_yaml_config(config_path)
        if config.get("runtime", {}).get("backend") == "jax":
            return True
    return False


def parse_args():
    """Parse command-line arguments for the benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Unified Benchmark Suite (NL + D3 + Electrostatics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
Examples:
    python -m benchmarks.benchmark_suite --benchmark all
    python -m benchmarks.benchmark_suite --benchmark d3 --system cscl --mode system_size
    python -m benchmarks.benchmark_suite --benchmark all --timing-runs 50
    python -m benchmarks.benchmark_suite --benchmark nl --no-plot
    python -m benchmarks.benchmark_suite --plot-only benchmark-results/run_2026-02-17/

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
    add_common_cli_args(parser, include_d3_params_path=True)
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
    parser.add_argument(
        "--expected-backends",
        nargs="+",
        choices=("torch", "jax", "warp"),
        default=None,
        help=(
            "Backends required by reportable --plot-only validation "
            "(default: torch jax warp)"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        metavar="RESULTS_DIR",
        help=(
            "Write benchmark CSVs directly to this directory instead of creating "
            "a timestamped run_* subdirectory. Intended for merging separate "
            "Torch and JAX passes into one reportable result set."
        ),
    )
    parser.add_argument(
        "--run-log-name",
        default="RUN_LOG.md",
        metavar="BASENAME",
        help=(
            "Run-log basename inside --run-dir. Parallel scheduler shards must "
            "use distinct names."
        ),
    )
    parser.add_argument(
        "--cutoffs",
        "-c",
        type=float,
        nargs="+",
        default=None,
        help="Override cutoff radii for NL/D3 benchmarks",
    )
    parser.add_argument(
        "--accuracies",
        "-a",
        type=float,
        nargs="+",
        default=None,
        help="Override electrostatics target accuracies",
    )
    parser.add_argument(
        "--profile-components",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Also time electrostatics real/reciprocal components. "
            "Disabled for reportable full-workload runs."
        ),
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=["all"],
        choices=["all", "time", "throughput", "memory"],
        help="Plot panels to generate after benchmarks",
    )
    return parser.parse_args()


def main():
    """Run the unified benchmark suite."""
    args = parse_args()

    benchmarks = set(RUNNERS) if "all" in args.benchmark else set(args.benchmark)
    plan_only = (
        getattr(args, "dry_run", False)
        or getattr(args, "list_plan", False)
        or getattr(args, "count_plan", False)
    )

    # JAX_ENABLE_X64 must be set BEFORE the first `import jax`. EL needs
    # f64 (PME/Ewald accuracy); NL/D3 are f32-safe but share the same
    # Python process in the suite, so JAX commits to whatever x64 was
    # when NL ran first. Set it unconditionally when any JAX benchmark
    # is queued so the env is consistent regardless of module order.
    if not plan_only and _suite_needs_jax_env(args, benchmarks):
        configure_jax_environment(need_x64=True, context="Benchmark suite")

    if args.plot_only:
        expected_backends = set(
            getattr(args, "expected_backends", None) or ("torch", "jax", "warp")
        )
        return (
            0
            if _generate_plots(
                args.plot_only,
                plots=args.plots,
                require_complete_suite=True,
                expected_backends=expected_backends,
            )
            else 1
        )

    try:
        validate_backend_selection(args.backend, benchmarks)
        validate_method_selection(args.methods, benchmarks)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    count_plan = getattr(args, "count_plan", False)
    _print_suite_header(args, benchmarks, count_plan=count_plan)

    start_time = datetime.now()
    run_dir = _prepare_run_directory(
        args,
        plan_only=plan_only,
        default_base_dir=SCRIPT_DIR / "benchmark-results",
    )
    summary = _run_requested_benchmarks(
        args,
        benchmarks,
        RUNNERS,
        run_dir=run_dir,
        plan_only=plan_only,
        config_loader=load_yaml_config,
        runner_importer=importlib.import_module,
    )

    if plan_only:
        return _report_plan(summary, count_plan=count_plan)

    if run_dir is None:
        raise RuntimeError("Runtime benchmark execution requires a run directory")
    end_time = datetime.now()
    _write_suite_run_log(run_dir, start_time, end_time, args, benchmarks, summary)

    # --- Plotting ---
    if not args.no_plot:
        plot_ok = _generate_plots(run_dir, plots=args.plots)
    else:
        plot_ok = True
    return _report_run(summary, run_dir, plot_ok=plot_ok)


def _generate_plots(
    results_dir,
    plots=None,
    *,
    require_complete_suite=False,
    expected_backends: set[str] | None = None,
):
    """Generate 3-panel review plots and single-panel docs plots from CSVs.

    Complete-suite publication requires every expected plot to render.
    """
    publication_backends = expected_backends or {"torch", "jax", "warp"}

    def validate_complete(paths: list[Path]) -> None:
        _validate_complete_reportable_suite(paths, publication_backends)

    return _orchestrate_plots(
        results_dir,
        plots=plots,
        csv_prefixes=SUITE_CSV_PREFIXES,
        nl_doc_cutoffs=NL_DOC_CUTOFFS,
        expected_csv_names=(REPORTABLE_CSV_NAMES if require_complete_suite else None),
        complete_suite_validator=(
            validate_complete if require_complete_suite else None
        ),
        require_all_plots=require_complete_suite,
    )


def _validate_complete_reportable_suite(
    csv_paths: list[Path], expected_backends: set[str]
) -> None:
    """Validate the complete publication matrix for selected backends."""
    validate_reportable_case_matrix(csv_paths, expected_backends)


if __name__ == "__main__":
    sys.exit(main())
