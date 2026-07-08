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
Generate plots from benchmark CSV files.

This script is run during the Sphinx documentation build to create
visualization plots from benchmark results.
"""

from __future__ import annotations

import argparse
import io
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.plotting.styles import (  # noqa: E402
    BACKEND_COLORS,
    DATA_LINE_WIDTH,
    DATA_MARKER_SIZE,
    LINE_ALPHA,
    NAIVE_ORANGE,
    NVIDIA_BLUE,
    NVIDIA_GREEN,
    PNG_EXPORT_DPI,
    SECONDARY_LINESTYLE,
    setup_plot_style,
)

NL_DOC_CUTOFFS = (6.0, 15.0, 25.0)
SUITE_RESULTS_ENV = "BENCHMARK_SUITE_RESULTS_DIR"
PLOT_JOBS_ENV = "BENCHMARK_PLOT_JOBS"
_PLOT_PROCESS_CONTEXT = multiprocessing.get_context("spawn")
_SUITE_SYSTEMS = ("cscl", "nh3")
_SUITE_MODES = ("system-size", "constant-workload", "batch")
_EXPECTED_SUITE_CSV_NAMES = {
    f"{module}-{system}-{mode}-scaling.csv"
    for module in ("nl", "d3", "el")
    for system in _SUITE_SYSTEMS
    for mode in _SUITE_MODES
}


def _available_cpu_count() -> int:
    """Return CPUs available to this process, respecting scheduler affinity."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _resolve_plot_jobs(jobs: int | str, task_count: int) -> int:
    """Resolve ``auto`` or an integer plot-worker request for this workload."""
    if isinstance(jobs, str) and jobs.strip().lower() == "auto":
        requested = _available_cpu_count()
    else:
        try:
            requested = int(jobs)
        except (TypeError, ValueError) as exc:
            raise ValueError("plot jobs must be 'auto' or a positive integer") from exc
    if requested < 1:
        raise ValueError("plot jobs must be 'auto' or a positive integer")
    return min(requested, max(task_count, 1))


def _filter_successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only successful benchmark rows when the CSV has status data."""
    if "success" not in df.columns:
        return df
    return df[df["success"].astype(str).str.lower() == "true"]


def plot_series(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str | None = None,
    x_label: str = "Number of atoms",
    y_label: str = "Value",
    caption: str | None = None,
) -> None:
    """
    Plot multiple data series on a log-log scale.

    Parameters
    ----------
    series
        Dictionary mapping series labels to (x, y) tuples.
    output_path
        Path to save the plot.
    title
        Plot title.
    x_label
        X-axis label.
    y_label
        Y-axis label.
    caption
        Caption text below the plot.
    """
    setup_plot_style()
    num_series = len(series)

    # Determine figure size based on number of series (accommodate legend)
    fig_width = 10 if num_series > 3 else 8
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), constrained_layout=True)

    palette = [NVIDIA_GREEN, NVIDIA_BLUE, NAIVE_ORANGE, "#440154", "#555555"]
    markers = ["o", "s", "^", "D", "P"]

    for idx, (label, (xs, ys)) in enumerate(series.items()):
        if xs is None or ys is None:
            continue

        label_lower = str(label).lower()
        backend = next(
            (
                name
                for name in BACKEND_COLORS
                if label_lower == name or f"({name})" in label_lower
            ),
            None,
        )
        color = BACKEND_COLORS[backend] if backend else palette[idx % len(palette)]
        linestyle = (
            (0, (1, 2))
            if backend == "jax"
            else SECONDARY_LINESTYLE
            if backend == "warp"
            else "-"
        )

        # matplotlib automatically skips nan values, creating gaps in lines
        ax.plot(
            xs,
            ys,
            marker=markers[idx % len(markers)],
            linestyle=linestyle,
            linewidth=DATA_LINE_WIDTH,
            markersize=DATA_MARKER_SIZE,
            label=label,
            color=color,
            dash_capstyle="round",
            markeredgewidth=0.9,
            markeredgecolor=color,
            alpha=LINE_ALPHA,
        )

    # Axis labels and scales
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Ensure sufficient tick marks on both axes
    # Use LogLocator with numticks parameter for better control
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))

    # Add minor ticks for additional reference points
    ax.xaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20)
    )
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20)
    )

    # Enhance tick labels
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)

    # Title with proper spacing
    if title is not None:
        ax.set_title(title, fontsize=14, pad=12)

    # Refined grid
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.3)
    ax.grid(False, which="minor")

    # Legend placement: outside plot area to avoid overlap
    if num_series <= 4:
        # Few series: place inside upper left
        ax.legend(
            frameon=False,
            fontsize=12,
            loc="upper left",
            framealpha=0.95,
            edgecolor="gray",
            fancybox=False,
        )
    else:
        # Many series: place outside to the right
        ax.legend(
            frameon=False,
            fontsize=11,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            framealpha=0.95,
            edgecolor="gray",
            fancybox=False,
        )

    # Caption if provided
    if caption is not None:
        fig.text(
            0.5,
            0.02,
            caption,
            wrap=True,
            horizontalalignment="center",
            fontsize=11,
            style="italic",
        )

    plt.savefig(output_path.as_posix(), dpi=PNG_EXPORT_DPI, bbox_inches="tight")
    plt.close()


def plot_throughput(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str | None = None,
    caption: str | None = None,
) -> None:
    """
    Plot throughput (atoms/s) vs system size.

    Parameters
    ----------
    series
        Dictionary mapping series labels to (total_atoms, median_time_ms) tuples.
    output_path
        Path to save the plot.
    title
        Plot title.
    caption
        Caption text below the plot.
    """
    # Convert time series to throughput
    throughput_series = {}
    for label, (atoms, times_ms) in series.items():
        if atoms is None or times_ms is None:
            continue
        # Division with nan propagates nan, which matplotlib will skip.
        throughput = atoms / times_ms * 1000.0
        throughput_series[label] = (atoms, throughput)

    plot_series(
        throughput_series,
        output_path,
        title=title,
        x_label="Number of atoms",
        y_label="Throughput (atoms/s)",
        caption=caption,
    )


def plot_memory(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str | None = None,
    caption: str | None = None,
) -> None:
    """
    Plot memory utilization vs system size.

    Parameters
    ----------
    series
        Dictionary mapping series labels to (total_atoms, peak_memory_mb) tuples.
    output_path
        Path to save the plot.
    title
        Plot title.
    caption
        Caption text below the plot.
    """
    plot_series(
        series,
        output_path,
        title=title,
        x_label="Number of atoms",
        y_label="Peak memory (MB)",
        caption=caption,
    )


def load_dynamics_csv(filepath: Path) -> pd.DataFrame:
    """
    Load dynamics benchmark results from CSV file.

    Parameters
    ----------
    filepath
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with dynamics benchmark data.
        Detects single-system vs batched based on presence of batch_size column.
    """
    df = pd.read_csv(filepath)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return _filter_successful_rows(df)


def _parse_dynamics_filename(filename: str) -> dict[str, str]:
    """
    Parse dynamics benchmark filename.

    Expected format: dynamics_{md|opt}_{single|batch}_{backend}_{gpu_sku}.csv

    Parameters
    ----------
    filename
        CSV filename.

    Returns
    -------
    dict
        Dictionary with keys: benchmark_type, system_type, backend, gpu_sku
    """
    parts = filename.replace(".csv", "").split("_")
    if len(parts) < 5 or parts[0] != "dynamics":
        return {}

    return {
        "benchmark_type": parts[1],  # md or opt
        "system_type": parts[2],  # single or batch
        "backend": parts[3],  # nvalchemiops, ase, torchsim
        "gpu_sku": "_".join(parts[4:]),  # rest is GPU SKU
    }


def generate_dynamics_plots(results_dir: Path, output_dir: Path) -> None:
    """
    Generate plots for dynamics benchmarks.

    Creates plots for:
    - Single-system MD benchmarks
    - Single-system optimization benchmarks
    - Batched MD benchmarks
    - Batched optimization benchmarks

    Parameters
    ----------
    results_dir
        Directory containing benchmark CSV files.
    output_dir
        Directory to save plots.
    """
    print("\nGenerating dynamics benchmark plots...")

    dynamics_files = list(results_dir.glob("dynamics_*.csv"))
    if not dynamics_files:
        print("  No dynamics benchmark results found")
        return

    files_by_category = {}
    for filepath in dynamics_files:
        info = _parse_dynamics_filename(filepath.name)
        if not info:
            continue

        category = f"{info['benchmark_type']}_{info['system_type']}"
        files_by_category.setdefault(category, {})

        backend = info["backend"]
        files_by_category[category].setdefault(backend, {})
        files_by_category[category][backend] = {
            "path": filepath,
            "gpu_sku": info["gpu_sku"],
        }

    for category, backends in files_by_category.items():
        benchmark_type, system_type = category.split("_")
        print(f"\n  Processing {benchmark_type.upper()} {system_type} benchmarks...")

        is_batched = system_type == "batch"
        all_data = {}
        gpu_sku = "unknown"
        for backend, file_info in backends.items():
            df = load_dynamics_csv(file_info["path"])
            all_data[backend] = df
            gpu_sku = file_info["gpu_sku"]

        if len(all_data) > 1:
            print("    Creating comparison plots...")
            _generate_dynamics_comparison_plots(
                all_data, benchmark_type, system_type, is_batched, gpu_sku, output_dir
            )

        for backend, df in all_data.items():
            print(f"    Creating {backend} detail plots...")
            _generate_dynamics_backend_plots(
                df,
                backend,
                benchmark_type,
                system_type,
                is_batched,
                gpu_sku,
                output_dir,
            )


def _generate_dynamics_comparison_plots(
    data_by_backend: dict[str, pd.DataFrame],
    benchmark_type: str,
    system_type: str,
    is_batched: bool,
    gpu_sku: str,
    output_dir: Path,
) -> None:
    """Generate comparison plots across backends."""
    # Scaling plot: num_atoms vs avg_step_time_ms
    series = {}
    for backend, df in data_by_backend.items():
        if is_batched:
            # For batched, average across batch sizes for each num_atoms
            grouped = df.groupby("num_atoms")["avg_step_time_ms"].mean()
            series[backend] = (grouped.index.values, grouped.values)
        else:
            # For single-system, average across methods for each num_atoms
            grouped = df.groupby("num_atoms")["avg_step_time_ms"].mean()
            series[backend] = (grouped.index.values, grouped.values)

    output_path = (
        output_dir
        / f"dynamics_{benchmark_type}_{system_type}_scaling_comparison_{gpu_sku}.png"
    )
    plot_series(
        series,
        output_path,
        title=f"{benchmark_type.upper()} {system_type.title()} Scaling Comparison",
        x_label="Number of atoms",
        y_label="Avg step time (ms)",
    )
    print(f"      Generated: {output_path.name}")

    # Throughput plot: num_atoms vs throughput_atom_steps_per_s
    series = {}
    for backend, df in data_by_backend.items():
        if is_batched:
            grouped = df.groupby("num_atoms")["throughput_atom_steps_per_s"].mean()
            series[backend] = (grouped.index.values, grouped.values)
        else:
            grouped = df.groupby("num_atoms")["throughput_atom_steps_per_s"].mean()
            series[backend] = (grouped.index.values, grouped.values)

    output_path = (
        output_dir
        / f"dynamics_{benchmark_type}_{system_type}_throughput_comparison_{gpu_sku}.png"
    )
    plot_series(
        series,
        output_path,
        title=f"{benchmark_type.upper()} {system_type.title()} Throughput Comparison",
        x_label="Number of atoms",
        y_label="Atom-steps/s",
    )
    print(f"      Generated: {output_path.name}")

    # For batched: batch scaling plot
    if is_batched and "batch_throughput_system_steps_per_s" in df.columns:
        series = {}
        for backend, df in data_by_backend.items():
            # Average across num_atoms for each batch_size
            grouped = df.groupby("batch_size")[
                "batch_throughput_system_steps_per_s"
            ].mean()
            series[backend] = (grouped.index.values, grouped.values)

        output_path = (
            output_dir
            / f"dynamics_{benchmark_type}_{system_type}_batch_scaling_comparison_{gpu_sku}.png"
        )
        plot_series(
            series,
            output_path,
            title=f"{benchmark_type.upper()} Batch Scaling Comparison",
            x_label="Batch size",
            y_label="System-steps/s",
        )
        print(f"      Generated: {output_path.name}")


def _generate_dynamics_backend_plots(
    df: pd.DataFrame,
    backend: str,
    benchmark_type: str,
    system_type: str,
    is_batched: bool,
    gpu_sku: str,
    output_dir: Path,
) -> None:
    """Generate per-backend detail plots."""
    if is_batched:
        # For batched data, use total_atoms as x-axis.
        # Choose series grouping: model_type if multiple, otherwise method.
        model_types = df["model_type"].dropna().replace("", pd.NA).dropna().unique()
        if len(model_types) > 1:
            group_col = "model_type"
        else:
            group_col = "method"
        group_vals = df[group_col].unique()
        x_col = "total_atoms"
        x_label = "Total atoms (num_atoms × batch_size)"
    else:
        group_col = "method"
        group_vals = df[group_col].unique()
        x_col = "num_atoms"
        x_label = "Number of atoms"

    # Scaling plot
    series = {}
    for val in group_vals:
        df_sub = df[df[group_col] == val]
        grouped = df_sub.groupby(x_col)["avg_step_time_ms"].mean()
        series[val] = (grouped.index.values, grouped.values)

    if series:
        output_path = (
            output_dir
            / f"dynamics_{benchmark_type}_{system_type}_{backend}_scaling_{gpu_sku}.png"
        )
        plot_series(
            series,
            output_path,
            title=f"{benchmark_type.upper()} {system_type.title()} Scaling ({backend})",
            x_label=x_label,
            y_label="Avg step time (ms)",
        )
        print(f"      Generated: {output_path.name}")

    # Throughput plot
    series = {}
    for val in group_vals:
        df_sub = df[df[group_col] == val]
        grouped = df_sub.groupby(x_col)["throughput_atom_steps_per_s"].mean()
        series[val] = (grouped.index.values, grouped.values)

    if series:
        output_path = (
            output_dir
            / f"dynamics_{benchmark_type}_{system_type}_{backend}_throughput_{gpu_sku}.png"
        )
        plot_series(
            series,
            output_path,
            title=f"{benchmark_type.upper()} {system_type.title()} Throughput ({backend})",
            x_label=x_label,
            y_label="Atom-steps/s",
        )
        print(f"      Generated: {output_path.name}")


def _suite_csv_dirs(
    results_dir: Path,
    *,
    require_complete: bool = False,
) -> list[Path]:
    """Return one CSV source, validating complete build and override inputs."""
    override = os.getenv(SUITE_RESULTS_ENV)
    suite_dir = Path(override).expanduser().resolve() if override else results_dir
    if not override and not require_complete:
        return [suite_dir]
    csv_paths = sorted(
        path
        for path in suite_dir.glob("*.csv")
        if path.name.startswith(("nl-", "d3-", "el-"))
        and not path.name.startswith("nl-backend-")
    )
    found = {path.name for path in csv_paths}
    if found != _EXPECTED_SUITE_CSV_NAMES:
        missing = sorted(_EXPECTED_SUITE_CSV_NAMES - found)
        unexpected = sorted(found - _EXPECTED_SUITE_CSV_NAMES)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise RuntimeError(
            f"{suite_dir} must contain one complete 18-file suite; "
            + "; ".join(details)
        )

    from benchmarks.benchmark_suite import validate_reportable_case_matrix
    from benchmarks.suite_utils import validate_result_files

    marker = suite_dir / ".benchmark-run-id"
    expected_run_id = (
        marker.read_text(encoding="ascii").strip() if marker.is_file() else None
    )
    try:
        validate_result_files(csv_paths, expected_run_id=expected_run_id)
        validate_reportable_case_matrix(csv_paths, {"torch", "jax", "warp"})
    except ValueError as exc:
        raise RuntimeError(f"Invalid benchmark results in {suite_dir}: {exc}") from exc
    return [suite_dir]


def _write_no_data_placeholder(output_path: Path, title: str, details: str) -> None:
    """Write an explicit placeholder for selector views with no successful rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    ax.axis("off")
    ax.text(
        0.5,
        0.58,
        title,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.42,
        details,
        ha="center",
        va="center",
        fontsize=11,
        wrap=True,
    )
    fig.savefig(output_path.as_posix(), dpi=PNG_EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def _render_suite_csv(
    csv_file: Path,
    output_dir: Path,
) -> None:
    """Render every docs panel derived from one standardized suite CSV."""
    from benchmarks.plotting.plot_benchmarks import (
        plot_comparison_panel,
        plot_single_panel,
    )

    for panel in ("time", "throughput", "memory"):
        output_path = output_dir / f"{csv_file.stem}-{panel}.png"
        if plot_single_panel(csv_file, panel, output_path):
            print(f"      Generated: {output_path.name}")
        else:
            _write_no_data_placeholder(
                output_path,
                "No successful benchmark rows",
                f"{csv_file.stem}, {panel}. See the CSV error_type column.",
            )
            print(f"      Placeholder: {csv_file.name} ({panel}, no data)")
        if csv_file.name.startswith("nl-"):
            svg_output = output_dir / f"{csv_file.stem}-{panel}.svg"
            if plot_single_panel(csv_file, panel, svg_output):
                print(f"      Generated: {svg_output.name}")
            else:
                _write_no_data_placeholder(
                    svg_output,
                    "No successful benchmark rows",
                    f"{csv_file.stem}, {panel}. See the CSV error_type column.",
                )
        if panel in {"time", "throughput"}:
            jax_output = output_dir / f"{csv_file.stem}-jax-{panel}.png"
            if plot_single_panel(
                csv_file,
                panel,
                jax_output,
                filters={"backend": "jax"},
                title_suffix="JAX",
            ):
                print(f"      Generated: {jax_output.name}")
            else:
                _write_no_data_placeholder(
                    jax_output,
                    "No successful JAX benchmark rows",
                    f"{csv_file.stem}, {panel}. See the CSV error_type column.",
                )
                print(f"      Placeholder: {csv_file.name} ({panel}, jax, no data)")
            if csv_file.name.startswith("nl-"):
                jax_svg_output = output_dir / f"{csv_file.stem}-jax-{panel}.svg"
                if plot_single_panel(
                    csv_file,
                    panel,
                    jax_svg_output,
                    filters={"backend": "jax"},
                    title_suffix="JAX",
                ):
                    print(f"      Generated: {jax_svg_output.name}")
                else:
                    _write_no_data_placeholder(
                        jax_svg_output,
                        "No successful JAX benchmark rows",
                        (f"{csv_file.stem}, {panel}. See the CSV error_type column."),
                    )
    if csv_file.name.startswith(("d3-", "el-")):
        module = csv_file.name.split("-", 1)[0]
        for panel in ("time", "throughput", "memory"):
            output_path = (
                output_dir
                / f"{csv_file.stem.replace('-scaling', f'-comparison-{panel}')}.png"
            )
            output_path.unlink(missing_ok=True)
            plot_comparison_panel(csv_file, panel, output_path, module)
            if not output_path.is_file():
                _write_no_data_placeholder(
                    output_path,
                    "No matched Torch/JAX benchmark rows",
                    (
                        f"{csv_file.stem}, {panel}. Both backends need successful "
                        "rows at the same x values."
                    ),
                )


def _render_suite_csv_task(
    task: tuple[Path, Path],
) -> str:
    """Process-worker entry point that returns ordered render output."""
    output = io.StringIO()
    with redirect_stdout(output):
        _render_suite_csv(*task)
    return output.getvalue()


def generate_suite_csv_plots(
    results_dir: Path,
    output_dir: Path,
    *,
    jobs: int | str = 1,
) -> None:
    """Generate docs panels from the unified suite's standardized CSV names."""
    try:
        from benchmarks.plotting import plot_benchmarks  # noqa: F401
    except ImportError as exc:
        print(f"Skipping suite CSV plots: {exc}")
        return

    seen: set[str] = set()
    csv_files = []
    for csv_dir in _suite_csv_dirs(results_dir):
        for pattern in ("nl-*.csv", "d3-*.csv", "el-*.csv"):
            for path in sorted(csv_dir.glob(pattern)):
                if path.name in seen or path.name.startswith("nl-backend-"):
                    continue
                seen.add(path.name)
                csv_files.append(path)

    if not csv_files:
        print("No unified suite CSV files found")
        return

    for stale_path in output_dir.glob("nl-*-cutoff-*.png"):
        stale_path.unlink()
    worker_count = _resolve_plot_jobs(jobs, len(csv_files))
    print(
        f"\nGenerating unified suite plots ({len(csv_files)} CSVs, "
        f"jobs={worker_count})..."
    )
    tasks = [(csv_file, output_dir) for csv_file in csv_files]
    if worker_count == 1:
        for task in tasks:
            _render_suite_csv(*task)
        return

    # Sphinx-Gallery may have initialized JAX threads before this hook runs.
    # Spawning clean plot workers avoids the unsafe fork-after-JAX path.
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=_PLOT_PROCESS_CONTEXT,
    ) as executor:
        for output in executor.map(_render_suite_csv_task, tasks):
            print(output, end="")


def generate_nl_backend_comparison_plots(results_dir: Path, output_dir: Path) -> None:
    """Generate layered current-suite Torch/JAX NL overlays."""
    from benchmarks.plotting.plot_benchmarks import plot_comparison_panel

    seen: set[str] = set()
    csv_files: list[Path] = []
    for csv_dir in _suite_csv_dirs(results_dir):
        for path in sorted(csv_dir.glob("nl-*.csv")):
            if path.name.startswith("nl-backend-") or path.name in seen:
                continue
            seen.add(path.name)
            csv_files.append(path)

    if not csv_files:
        print("No NL backend-comparison CSV files found")
        return

    print(f"\nGenerating NL backend-comparison plots ({len(csv_files)} CSVs)...")
    for csv_file in csv_files:
        out_stem = f"nl-backend-{csv_file.stem[3:]}"
        for panel in ("time", "throughput"):
            for suffix in (".png", ".svg"):
                output_path = output_dir / f"{out_stem}-{panel}{suffix}"
                if plot_comparison_panel(
                    csv_file,
                    panel,
                    output_path,
                    "nl",
                    layer_all_params=True,
                ):
                    continue
                _write_no_data_placeholder(
                    output_path,
                    "No matched Torch/JAX benchmark rows",
                    (
                        f"{csv_file.stem}, {panel}. Both backends need successful, "
                        "comparison-eligible rows at shared x values."
                    ),
                )


def main(*, jobs: int | str = "auto") -> None:
    """Generate all plots from benchmark results."""
    print("Generating benchmark plots...")

    # Determine paths relative to this script
    results_dir = Path(__file__).parent / "benchmark_results"
    output_dir = Path(__file__).parent / "_static"

    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Fail before rendering if the bundled or overridden reportable snapshot is
    # incomplete, mixed, or missing required provenance.
    _suite_csv_dirs(results_dir, require_complete=True)

    # Generate plots for each benchmark type
    generate_suite_csv_plots(results_dir, output_dir, jobs=jobs)

    # Keep downstream plots independent of whether suite panels ran locally or
    # in worker processes.
    setup_plot_style()
    generate_nl_backend_comparison_plots(results_dir, output_dir)
    generate_dynamics_plots(results_dir, output_dir)

    print("\nPlot generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        default=os.getenv(PLOT_JOBS_ENV, "auto"),
        help="Plot worker processes: 'auto' or a positive integer.",
    )
    main(jobs=parser.parse_args().jobs)
