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

"""Internal plot orchestration for the unified benchmark suite."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.suite_utils import validate_result_files

__all__: list[str] = []


@dataclass
class _PlotResults:
    """Track attempted, successful, and failed plot renders."""

    attempted: int = 0
    succeeded: int = 0
    errors: list[str] = field(default_factory=list)

    def attempt(self, label: str, render: Callable[[], bool]) -> None:
        """Run one plot render and record its outcome."""
        self.attempted += 1
        try:
            if render():
                self.succeeded += 1
                return
            message = f"{label}: no successful data"
        except Exception as error:
            message = f"{label}: {error}"
        self.errors.append(message)
        print(f"  ERROR ({message})")

    def all_attempts_failed(self) -> bool:
        """Return whether plotting attempted work without any success."""
        return self.attempted > 0 and self.succeeded == 0


def _generate_plots(
    results_dir: str | Path,
    *,
    plots: Sequence[str] | None,
    csv_prefixes: tuple[str, ...],
    nl_doc_cutoffs: tuple[float, ...],
    expected_csv_names: set[str] | None = None,
    complete_suite_validator: Callable[[list[Path]], object] | None = None,
    require_all_plots: bool = False,
) -> bool:
    """Generate suite review and docs plots from result CSVs."""
    try:
        from benchmarks.plotting.plot_benchmarks import (
            detect_and_plot,
            plot_single_panel,
        )
    except ImportError as error:
        level = "ERROR" if require_all_plots else "WARNING"
        print(
            f"\n  {level}: plotting unavailable ({error}). Results saved to {results_dir}"
        )
        return not require_all_plots

    results_path = Path(results_dir)
    all_csvs = sorted(results_path.glob("*.csv"))
    csvs = [path for path in all_csvs if path.name.startswith(csv_prefixes)]
    if not csvs:
        if expected_csv_names is not None:
            print("\n  ERROR: refusing to plot an empty reportable suite")
            return False
        print("\n  No unified suite CSVs found, skipping plots")
        return True
    _report_skipped_csvs(all_csvs, csvs)

    run_id_path = results_path / ".benchmark-run-id"
    if expected_csv_names is not None:
        found_names = {path.name for path in csvs}
        if found_names != expected_csv_names:
            missing = sorted(expected_csv_names - found_names)
            unexpected = sorted(found_names - expected_csv_names)
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected: " + ", ".join(unexpected))
            print(
                "\n  ERROR: refusing to plot an incomplete reportable suite ("
                + "; ".join(details)
                + ")"
            )
            return False

    if expected_csv_names is not None or run_id_path.exists():
        run_id = (
            run_id_path.read_text(encoding="ascii").strip()
            if run_id_path.exists()
            else None
        )
        try:
            validate_result_files(csvs, expected_run_id=run_id)
        except ValueError as error:
            print(f"\n  ERROR: refusing to plot mixed or invalid results ({error})")
            return False

    if complete_suite_validator is not None:
        try:
            complete_suite_validator(csvs)
        except ValueError as error:
            print(f"\n  ERROR: refusing to plot an incomplete result matrix ({error})")
            return False

    requested = set(plots or ["all"])
    all_panels = "all" in requested
    panels = ("time", "throughput", "memory") if all_panels else tuple(plots or ())
    results = _PlotResults()

    print(f"\n{'=' * 70}")
    print(f"GENERATING PLOTS ({len(csvs)} CSVs)")
    print("=" * 70)

    if all_panels:
        _generate_review_plots(csvs, results_path, detect_and_plot, results)

    single_dir = results_path / "single-panels"
    single_dir.mkdir(exist_ok=True)
    _generate_single_panel_plots(
        csvs,
        panels,
        single_dir,
        plot_single_panel,
        nl_doc_cutoffs,
        results,
    )

    print(f"  3-panel plots: {results_path}")
    print(f"  Single-panel plots: {single_dir}")
    if results.errors:
        print(f"\n{len(results.errors)} plot(s) failed:")
        for error in results.errors:
            print(f"  - {error}")
    if require_all_plots:
        return results.attempted > 0 and not results.errors
    return not results.all_attempts_failed()


def _report_skipped_csvs(all_csvs: list[Path], suite_csvs: list[Path]) -> None:
    """Report CSV files excluded from unified suite plotting."""
    skipped = len(all_csvs) - len(suite_csvs)
    if skipped:
        print(f"\n  Skipping {skipped} non-suite CSV(s) during suite plot generation")


def _generate_review_plots(
    csvs: list[Path],
    output_dir: Path,
    detect_and_plot: Callable[[Path, Path], bool],
    results: _PlotResults,
) -> None:
    """Generate one three-panel review plot per suite CSV."""
    for csv_path in csvs:
        results.attempt(
            f"3-panel {csv_path.name}",
            lambda csv_path=csv_path: detect_and_plot(csv_path, output_dir),
        )


def _generate_single_panel_plots(
    csvs: list[Path],
    panels: tuple[str, ...],
    output_dir: Path,
    plot_single_panel: Callable[..., bool],
    nl_doc_cutoffs: tuple[float, ...],
    results: _PlotResults,
) -> None:
    """Generate standard and neighbor-list cutoff-specific docs plots."""
    for csv_path in csvs:
        for panel in panels:
            output_path = output_dir / f"{csv_path.stem}-{panel}.png"
            results.attempt(
                f"single {csv_path.stem}-{panel}",
                lambda csv_path=csv_path, panel=panel, output_path=output_path: (
                    plot_single_panel(csv_path, panel, output_path)
                ),
            )
            if csv_path.name.startswith("nl-"):
                _generate_cutoff_plots(
                    csv_path,
                    panel,
                    output_dir,
                    plot_single_panel,
                    nl_doc_cutoffs,
                    results,
                )


def _generate_cutoff_plots(
    csv_path: Path,
    panel: str,
    output_dir: Path,
    plot_single_panel: Callable[..., bool],
    cutoffs: tuple[float, ...],
    results: _PlotResults,
) -> None:
    """Generate neighbor-list docs plots for each reportable cutoff."""
    for cutoff in cutoffs:
        cutoff_label = f"{cutoff:g}A"
        output_path = output_dir / f"{csv_path.stem}-cutoff-{cutoff_label}-{panel}.png"
        results.attempt(
            f"single {csv_path.stem}-cutoff-{cutoff_label}-{panel}",
            lambda cutoff=cutoff, output_path=output_path: plot_single_panel(
                csv_path,
                panel,
                output_path,
                filters={"cutoff": cutoff},
                title_suffix=f"{cutoff:g}A cutoff",
            ),
        )
