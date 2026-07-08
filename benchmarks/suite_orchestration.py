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

"""Internal execution orchestration for the unified benchmark suite."""

from __future__ import annotations

import argparse
import copy
import io
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from benchmarks.config import merge_common_cli_overrides
from benchmarks.suite_utils import create_run_directory, write_run_log

__all__: list[str] = []


class _Runner(Protocol):
    """Describe the runner methods used by suite orchestration."""

    def dry_run_from_config(self, config: dict) -> list[dict]: ...

    def run_from_config(self, config: dict, *, output_dir: Path) -> list[dict]: ...


@dataclass
class _SuiteResults:
    """Track per-runner row and success counts."""

    row_counts: dict[str, int] = field(default_factory=dict)
    success_counts: dict[str, int] = field(default_factory=dict)
    coverage_errors: dict[str, str] = field(default_factory=dict)
    run_log_name: str = "RUN_LOG.md"

    @property
    def total(self) -> int:
        """Return the total number of emitted or planned rows."""
        return sum(self.row_counts.values())

    @property
    def successful_total(self) -> int:
        """Return the total number of successful runtime rows."""
        return sum(self.success_counts.values())

    def record(self, label: str, rows: list[dict], *, plan_only: bool) -> None:
        """Record row counts from one runner."""
        self.row_counts[label] = len(rows)
        if not plan_only:
            self.success_counts[label] = sum(
                row.get("success", True) is not False for row in rows
            )

    def record_missing(self, label: str, *, plan_only: bool) -> None:
        """Record a selected runner whose config is unavailable."""
        self.row_counts[label] = 0
        if not plan_only:
            self.success_counts[label] = 0

    def validate_coverage(
        self,
        label: str,
        planned_rows: list[dict],
        emitted_rows: list[dict],
    ) -> None:
        """Record a concise error when runtime rows do not match the dry plan."""
        planned = Counter(_benchmark_case_key(row) for row in planned_rows)
        emitted = Counter(_benchmark_case_key(row) for row in emitted_rows)
        if planned == emitted:
            return

        missing = list((planned - emitted).elements())
        unexpected = list((emitted - planned).elements())
        details = []
        if missing:
            details.append(_format_case_difference("missing", missing))
        if unexpected:
            details.append(_format_case_difference("unexpected", unexpected))
        self.coverage_errors[label] = "; ".join(details)


def _as_bool(value: object) -> bool:
    """Normalize bool-like plan and CSV values."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _optional_float(value: object) -> float | None:
    """Normalize an optional numeric case dimension."""
    if value in (None, ""):
        return None
    return float(value)


def _benchmark_case_key(row: Mapping[str, object]) -> tuple[object, ...]:
    """Return the dimensions that identify one planned benchmark attempt."""
    method = str(row.get("method", ""))
    benchmark = str(row.get("benchmark", ""))
    if not benchmark:
        if "accuracy" in row or "compute_charge_gradients" in row:
            benchmark = "el"
        elif method == "dftd3":
            benchmark = "d3"
        else:
            benchmark = "nl"
    if benchmark == "el":
        derivative_contract = str(row.get("derivative_contract", ""))
        workload = str(row.get("workload", ""))
        compute_forces = _as_bool(row.get("compute_forces", False))
        compute_charge_gradients = _as_bool(row.get("compute_charge_gradients", False))
    else:
        derivative_contract = None
        workload = None
        compute_forces = None
        compute_charge_gradients = None

    return (
        benchmark,
        str(row.get("backend", "")),
        str(row.get("system", "")),
        str(row.get("scaling_mode", row.get("mode", ""))),
        method,
        int(row.get("atoms_per_system", 0)),
        int(row.get("batch_size", 0)),
        int(row.get("total_atoms", 0)),
        _optional_float(row.get("cutoff")),
        _optional_float(row.get("accuracy")),
        derivative_contract,
        workload,
        compute_forces,
        compute_charge_gradients,
    )


def _format_case_difference(label: str, cases: list[tuple[object, ...]]) -> str:
    """Format a bounded sample of missing or unexpected case keys."""
    sample = ", ".join(repr(case) for case in cases[:3])
    suffix = "" if len(cases) <= 3 else f", ... ({len(cases)} total)"
    return f"{label}: {sample}{suffix}"


def _planned_rows(runner: _Runner, config: dict) -> list[dict]:
    """Expand a runner plan without adding plan text to runtime logs."""
    plan_config = copy.deepcopy(config)
    plan_config.setdefault("runtime", {})["plan_output"] = "count"
    with redirect_stdout(io.StringIO()):
        return runner.dry_run_from_config(plan_config)


def _prepare_run_directory(
    args: argparse.Namespace,
    *,
    plan_only: bool,
    default_base_dir: Path,
) -> Path | None:
    """Resolve and create the output directory for a runtime suite pass."""
    if plan_only:
        return None

    requested_run_dir = getattr(args, "run_dir", None)
    if requested_run_dir is not None:
        run_dir = Path(requested_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = args.output_dir or default_base_dir
        run_dir = create_run_directory(base_dir, prefix="run")
    print(f"\nOutput: {run_dir}")
    return run_dir


def _print_suite_header(
    args: argparse.Namespace,
    benchmarks: set[str],
    *,
    count_plan: bool,
) -> None:
    """Print environment and selection details before suite execution."""
    if count_plan:
        return

    import torch

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


def _run_requested_benchmarks(
    args: argparse.Namespace,
    benchmarks: set[str],
    runners: Mapping[str, Mapping[str, object]],
    *,
    run_dir: Path | None,
    plan_only: bool,
    config_loader: Callable[[Path], dict],
    runner_importer: Callable[[str], _Runner],
) -> _SuiteResults:
    """Load configs and dispatch the selected benchmark runners in order."""
    summary = _SuiteResults()
    for key in ("nl", "d3", "el"):
        if key not in benchmarks:
            continue
        info = runners[key]
        label = str(info["label"])
        config_path = Path(info["config"])
        if not config_path.exists():
            print(f"\nWARNING: {label} config not found at {config_path}, skipping")
            summary.record_missing(label, plan_only=plan_only)
            continue

        runner = runner_importer(str(info["module"]))
        config = config_loader(config_path)
        if hasattr(runner, "merge_cli_overrides"):
            config = runner.merge_cli_overrides(config, args)
        else:
            config = merge_common_cli_overrides(config, args)

        if plan_only:
            rows = runner.dry_run_from_config(config)
        else:
            if run_dir is None:
                raise RuntimeError(
                    "Runtime benchmark execution requires a run directory"
                )
            planned_rows = _planned_rows(runner, config)
            rows = runner.run_from_config(config, output_dir=run_dir)
            summary.validate_coverage(label, planned_rows, rows)
        summary.record(label, rows, plan_only=plan_only)
    return summary


def _report_plan(summary: _SuiteResults, *, count_plan: bool) -> int:
    """Print a planning summary and return its process exit code."""
    label = "COUNT" if count_plan else "DRY RUN"
    print(f"\n{label} COMPLETE: {summary.total} planned row(s)")
    empty = _labels_with_no_rows(summary.row_counts)
    if empty:
        print(
            "ERROR: no planned rows for requested benchmark(s): " + ", ".join(empty),
            file=sys.stderr,
        )
        return 1
    return 0 if summary.total > 0 else 1


def _write_suite_run_log(
    run_dir: Path,
    start_time: datetime,
    end_time: datetime,
    args: argparse.Namespace,
    benchmarks: set[str],
    summary: _SuiteResults,
) -> None:
    """Write suite selections and per-runner counts to the run log."""
    extra = {
        "Benchmarks run": ", ".join(sorted(benchmarks)),
        "Backend": str(args.backend or "config default"),
        "Systems": str(args.system or "all"),
        "Modes": str(args.mode or "all"),
    }
    for name, count in summary.row_counts.items():
        extra[f"{name} results"] = count
    for name, count in summary.success_counts.items():
        extra[f"{name} successful results"] = count
    for name, error in summary.coverage_errors.items():
        extra[f"{name} coverage error"] = error
    summary.run_log_name = str(getattr(args, "run_log_name", "RUN_LOG.md"))
    write_run_log(
        run_dir,
        start_time,
        end_time,
        extra_info=extra,
        filename=summary.run_log_name,
    )


def _report_run(summary: _SuiteResults, run_dir: Path, *, plot_ok: bool) -> int:
    """Print the runtime summary and return its process exit code."""
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUITE COMPLETE")
    for name, count in summary.row_counts.items():
        successes = summary.success_counts.get(name)
        if successes is None:
            print(f"  {name}: {count} results")
        else:
            print(f"  {name}: {successes}/{count} successful results")
    print(f"  Total: {summary.total} results")
    print(f"  Successful: {summary.successful_total} results")
    print(f"  Output: {run_dir}")
    print(f"  Run log: {run_dir / summary.run_log_name}")
    print("=" * 70)

    if summary.total <= 0:
        return 1
    if summary.coverage_errors:
        for name, error in summary.coverage_errors.items():
            print(
                f"ERROR: emitted rows do not match the configured plan for "
                f"{name}: {error}",
                file=sys.stderr,
            )
        return 1
    empty = _labels_with_no_rows(summary.row_counts)
    if empty:
        print(
            "ERROR: no result rows for requested benchmark(s): " + ", ".join(empty),
            file=sys.stderr,
        )
        return 1
    failed = _labels_with_no_rows(summary.success_counts)
    if failed:
        print(
            "ERROR: no successful rows for requested benchmark(s): "
            + ", ".join(failed),
            file=sys.stderr,
        )
        return 1
    if summary.successful_total <= 0:
        print("ERROR: no successful benchmark rows were produced", file=sys.stderr)
        return 1
    return 0 if plot_ok else 1


def _labels_with_no_rows(summary: Mapping[str, int]) -> list[str]:
    """Return benchmark labels that produced no rows."""
    return sorted(label for label, count in summary.items() if count <= 0)
