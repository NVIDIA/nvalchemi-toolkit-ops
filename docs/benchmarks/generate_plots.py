#!/usr/bin/env python3
"""
Generate plots from benchmark CSV files.

This script is run during the Sphinx documentation build to create
visualization plots from benchmark results.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


class BenchmarkData(NamedTuple):
    """Container for benchmark data series."""

    total_atoms: np.ndarray
    median_time_ms: np.ndarray
    peak_memory_mb: np.ndarray


def load_nl_csv(
    filepath: Path,
) -> dict[int, BenchmarkData]:
    """
    Load neighbor list benchmark results from CSV file.

    Parameters
    ----------
    filepath
        Path to the CSV file.

    Returns
    -------
    dict[int, BenchmarkData]
        Dictionary mapping batch_size to BenchmarkData containing
        total_atoms, median_time_ms, and peak_memory_mb arrays.
    """
    df = pd.read_csv(filepath)

    # Convert inf to nan so matplotlib will skip those points
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    batch_sizes = df["batch_size"].unique()
    series = {}
    for batch_size in batch_sizes:
        df_batch = df[df["batch_size"] == batch_size]
        series[batch_size] = BenchmarkData(
            total_atoms=df_batch["total_atoms"].values,
            median_time_ms=df_batch["median_time_ms"].values,
            peak_memory_mb=df_batch["peak_memory_mb"].values,
        )
    return series


def load_dftd3_csv(
    filepath: Path, batched: bool = False
) -> dict[int, BenchmarkData] | BenchmarkData:
    """
    Load DFT-D3 benchmark results from CSV file.

    Parameters
    ----------
    filepath
        Path to the CSV file.
    batched
        If True, group by batch_size and return dict of series.
        If False, return single BenchmarkData.

    Returns
    -------
    dict[int, BenchmarkData] | BenchmarkData
        If batched, dictionary mapping batch_size to BenchmarkData.
        Otherwise, single BenchmarkData tuple.
    """
    df = pd.read_csv(filepath)

    # Convert inf to nan so matplotlib will skip those points
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if batched:
        batch_sizes = df["batch_size"].unique()
        series = {}
        for batch_size in batch_sizes:
            df_batch = df[df["batch_size"] == batch_size]
            series[batch_size] = BenchmarkData(
                total_atoms=df_batch["total_atoms"].values,
                median_time_ms=df_batch["median_time_ms"].values,
                peak_memory_mb=df_batch["peak_memory_mb"].values,
            )
        return series
    else:
        return BenchmarkData(
            total_atoms=df["total_atoms"].values,
            median_time_ms=df["median_time_ms"].values,
            peak_memory_mb=df["peak_memory_mb"].values,
        )


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
    num_series = len(series)

    # Determine figure size based on number of series (accommodate legend)
    fig_width = 10 if num_series > 3 else 8
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), constrained_layout=True)

    # Use YlGn sequential colormap
    if num_series == 1:
        colors = ["#2E7D32"]  # Single dark green
    else:
        # Use YlGn colormap, avoiding very light colors
        cmap = plt.cm.YlGn
        colors = [cmap(0.3 + 0.7 * i / (num_series - 1)) for i in range(num_series)]

    for idx, (label, (xs, ys)) in enumerate(series.items()):
        if xs is None or ys is None:
            continue

        color = colors[idx]

        # matplotlib automatically skips nan values, creating gaps in lines
        ax.plot(
            xs,
            ys,
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=6.0,
            label=label,
            color=color,
            markeredgewidth=0.5,
            markeredgecolor="black",
            alpha=0.9,
        )

    # Axis labels and scales
    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
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
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    # Refined grid
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.3, color="gray")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.2, color="gray")

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

    plt.savefig(output_path.as_posix(), dpi=300, bbox_inches="tight")
    plt.close()


def plot_throughput(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str | None = None,
    caption: str | None = None,
) -> None:
    """
    Plot throughput (atoms/ms) vs system size.

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
        # Division with nan propagates nan, which matplotlib will skip
        throughput = atoms / times_ms
        throughput_series[label] = (atoms, throughput)

    plot_series(
        throughput_series,
        output_path,
        title=title,
        x_label="Number of atoms",
        y_label="Throughput (atoms/ms)",
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


def generate_nl_plots(results_dir: Path, output_dir: Path) -> None:
    """
    Generate all neighbor list benchmark plots.

    Parameters
    ----------
    results_dir
        Directory containing CSV benchmark results.
    output_dir
        Directory to write output plots.
    """
    nl_pattern = "neighbor_list_benchmark_*.csv"
    nl_csv_files = list(results_dir.glob(nl_pattern))

    if not nl_csv_files:
        print("No neighbor list CSV files found")
        return

    print(f"\nFound {len(nl_csv_files)} neighbor list CSV files")

    for csv_file in nl_csv_files:
        # Extract method name from filename
        # Format: neighbor_list_benchmark_<method>_<gpu_sku>.csv
        parts = csv_file.stem.split("_")
        benchmark_index = parts.index("benchmark")
        method = parts[benchmark_index + 1]
        # Get rest of parts as GPU SKU
        gpu_sku = "_".join(parts[benchmark_index + 2 :])

        # Load data
        data = load_nl_csv(csv_file)

        # Prepare series for plotting
        time_series = {
            f"batch={bs}": (d.total_atoms, d.median_time_ms) for bs, d in data.items()
        }
        memory_series = {
            f"batch={bs}": (d.total_atoms, d.peak_memory_mb) for bs, d in data.items()
        }

        method_title = method.replace("-", " ").title()

        # 1. Time scaling plot
        output_path = output_dir / f"neighborlist_scaling_{method}_{gpu_sku}.png"
        plot_series(
            time_series,
            output_path,
            title=f"Neighbor List Scaling ({method_title})",
            x_label="Number of atoms",
            y_label="Median time (ms)",
        )
        print(f"  Generated: {output_path.name}")

        # 2. Throughput plot
        output_path = output_dir / f"neighborlist_throughput_{method}_{gpu_sku}.png"
        plot_throughput(
            time_series,
            output_path,
            title=f"Neighbor List Throughput ({method_title})",
        )
        print(f"  Generated: {output_path.name}")

        # 3. Memory utilization plot
        output_path = output_dir / f"neighborlist_memory_{method}_{gpu_sku}.png"
        plot_memory(
            memory_series,
            output_path,
            title=f"Neighbor List Memory ({method_title})",
        )
        print(f"  Generated: {output_path.name}")


def _parse_dftd3_filename(filename: str, is_batched: bool) -> tuple[str, str] | None:
    """
    Parse DFT-D3 benchmark filename to extract backend and GPU SKU.

    Parameters
    ----------
    filename
        The filename stem (without extension).
    is_batched
        Whether this is a batched benchmark file.

    Returns
    -------
    tuple[str, str] | None
        Tuple of (backend, gpu_sku) or None if parsing fails.

    Notes
    -----
    Filenames follow these patterns:
    - Non-batched: dftd3_benchmark_<backend>_<gpu_sku>.csv
    - Batched: dftd3_benchmark_batch_<backend>_<gpu_sku>.csv

    Backend names may contain underscores (e.g., "torch_dftd"), so we use
    known backend names to parse correctly.
    """
    known_backends = ["nvalchemiops", "torch_dftd"]

    if is_batched:
        prefix = "dftd3_benchmark_batch_"
    else:
        prefix = "dftd3_benchmark_"

    if not filename.startswith(prefix):
        return None

    remainder = filename[len(prefix) :]

    # Try to match known backends
    for backend in known_backends:
        if remainder.startswith(backend + "_"):
            gpu_sku = remainder[len(backend) + 1 :]
            return backend, gpu_sku

    # Fallback: assume single-token backend name
    parts = remainder.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]

    return None


def generate_dftd3_plots(results_dir: Path, output_dir: Path) -> None:
    """
    Generate all DFT-D3 benchmark plots.

    Parameters
    ----------
    results_dir
        Directory containing CSV benchmark results.
    output_dir
        Directory to write output plots.
    """
    d3_pattern = "dftd3_benchmark_*.csv"
    d3_csv_files = list(results_dir.glob(d3_pattern))

    if not d3_csv_files:
        print("No DFT-D3 CSV files found")
        return

    print(f"\nFound {len(d3_csv_files)} DFT-D3 CSV files")

    # Separate batched and non-batched files
    non_batched_files = []
    batched_files = []

    for csv_file in d3_csv_files:
        filename = csv_file.stem
        if "batch_" in filename:
            batched_files.append(csv_file)
        else:
            non_batched_files.append(csv_file)

    # 1. Plot comparison between non-batched backends
    if non_batched_files:
        print(
            f"  Creating comparison plots from {len(non_batched_files)} non-batched files..."
        )
        comparison_time_series = {}
        comparison_memory_series = {}
        gpu_sku = None

        for csv_file in non_batched_files:
            parsed = _parse_dftd3_filename(csv_file.stem, is_batched=False)
            if parsed is None:
                print(f"  Warning: Could not parse filename {csv_file.name}")
                continue
            backend, gpu_sku = parsed

            # Load data
            data = load_dftd3_csv(csv_file, batched=False)
            comparison_time_series[backend] = (data.total_atoms, data.median_time_ms)
            comparison_memory_series[backend] = (data.total_atoms, data.peak_memory_mb)

        if gpu_sku and comparison_time_series:
            # Time scaling comparison
            output_path = output_dir / f"dftd3_scaling_comparison_{gpu_sku}.png"
            plot_series(
                comparison_time_series,
                output_path,
                title="DFT-D3 Scaling (Backend Comparison)",
                x_label="Number of atoms",
                y_label="Median time (ms)",
            )
            print(f"  Generated: {output_path.name}")

            # Throughput comparison
            output_path = output_dir / f"dftd3_throughput_comparison_{gpu_sku}.png"
            plot_throughput(
                comparison_time_series,
                output_path,
                title="DFT-D3 Throughput (Backend Comparison)",
            )
            print(f"  Generated: {output_path.name}")

            # Memory comparison
            output_path = output_dir / f"dftd3_memory_comparison_{gpu_sku}.png"
            plot_memory(
                comparison_memory_series,
                output_path,
                title="DFT-D3 Memory (Backend Comparison)",
            )
            print(f"  Generated: {output_path.name}")

    # 2. Plot scaling for all batched backends
    for csv_file in batched_files:
        parsed = _parse_dftd3_filename(csv_file.stem, is_batched=True)
        if parsed is None:
            print(f"  Warning: Could not parse filename {csv_file.name}")
            continue
        backend, gpu_sku = parsed

        print(f"  Creating batched scaling plots for {backend}...")

        # Load batched data (batch sizes as series)
        data = load_dftd3_csv(csv_file, batched=True)

        time_series = {
            f"batch={bs}": (d.total_atoms, d.median_time_ms) for bs, d in data.items()
        }
        memory_series = {
            f"batch={bs}": (d.total_atoms, d.peak_memory_mb) for bs, d in data.items()
        }

        # Time scaling
        output_path = output_dir / f"dftd3_scaling_batch_{backend}_{gpu_sku}.png"
        plot_series(
            time_series,
            output_path,
            title=f"DFT-D3 Scaling ({backend})",
            x_label="Total atoms",
            y_label="Median time (ms)",
        )
        print(f"  Generated: {output_path.name}")

        # Throughput
        output_path = output_dir / f"dftd3_throughput_batch_{backend}_{gpu_sku}.png"
        plot_throughput(
            time_series,
            output_path,
            title=f"DFT-D3 Throughput ({backend})",
        )
        print(f"  Generated: {output_path.name}")

        # Memory
        output_path = output_dir / f"dftd3_memory_batch_{backend}_{gpu_sku}.png"
        plot_memory(
            memory_series,
            output_path,
            title=f"DFT-D3 Memory ({backend})",
        )
        print(f"  Generated: {output_path.name}")

    # 3. Generate per-backend comparison plots (single vs batched)
    _generate_dftd3_per_backend_plots(non_batched_files, batched_files, output_dir)


def _generate_dftd3_per_backend_plots(
    non_batched_files: list[Path],
    batched_files: list[Path],
    output_dir: Path,
) -> None:
    """
    Generate per-backend comparison plots showing single vs batched results.

    Parameters
    ----------
    non_batched_files
        List of non-batched CSV file paths.
    batched_files
        List of batched CSV file paths.
    output_dir
        Directory to write output plots.
    """
    # Build mapping of backend -> (single_file, batched_file)
    backend_files: dict[str, dict[str, Path]] = {}

    for csv_file in non_batched_files:
        parsed = _parse_dftd3_filename(csv_file.stem, is_batched=False)
        if parsed is None:
            continue
        backend, gpu_sku = parsed

        if backend not in backend_files:
            backend_files[backend] = {}
        backend_files[backend]["single"] = csv_file
        backend_files[backend]["gpu_sku"] = gpu_sku

    for csv_file in batched_files:
        parsed = _parse_dftd3_filename(csv_file.stem, is_batched=True)
        if parsed is None:
            continue
        backend, _ = parsed

        if backend not in backend_files:
            backend_files[backend] = {}
        backend_files[backend]["batched"] = csv_file

    # Generate plots for each backend
    for backend, files in backend_files.items():
        if "single" not in files:
            continue

        gpu_sku = files.get("gpu_sku", "unknown")
        print(f"  Creating per-backend plots for {backend}...")

        # Load single system data
        single_data = load_dftd3_csv(files["single"], batched=False)

        # Build series starting with single system
        time_series = {"single": (single_data.total_atoms, single_data.median_time_ms)}
        memory_series = {
            "single": (single_data.total_atoms, single_data.peak_memory_mb)
        }

        # Add batched data if available
        if "batched" in files:
            batched_data = load_dftd3_csv(files["batched"], batched=True)
            for bs, d in batched_data.items():
                time_series[f"batch={bs}"] = (d.total_atoms, d.median_time_ms)
                memory_series[f"batch={bs}"] = (d.total_atoms, d.peak_memory_mb)

        # Time scaling
        output_path = output_dir / f"dftd3_scaling_{backend}_{gpu_sku}.png"
        plot_series(
            time_series,
            output_path,
            title=f"DFT-D3 Scaling ({backend})",
            x_label="Total atoms",
            y_label="Median time (ms)",
        )
        print(f"  Generated: {output_path.name}")

        # Throughput
        output_path = output_dir / f"dftd3_throughput_{backend}_{gpu_sku}.png"
        plot_throughput(
            time_series,
            output_path,
            title=f"DFT-D3 Throughput ({backend})",
        )
        print(f"  Generated: {output_path.name}")

        # Memory
        output_path = output_dir / f"dftd3_memory_{backend}_{gpu_sku}.png"
        plot_memory(
            memory_series,
            output_path,
            title=f"DFT-D3 Memory ({backend})",
        )
        print(f"  Generated: {output_path.name}")


def main() -> None:
    """Generate all plots from benchmark results."""
    print("Generating benchmark plots...")

    # Determine paths relative to this script
    results_dir = Path(__file__).parent / "benchmark_results"
    output_dir = Path(__file__).parent / "_static"

    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Generate plots for each benchmark type
    generate_nl_plots(results_dir, output_dir)
    generate_dftd3_plots(results_dir, output_dir)

    print("\nPlot generation complete!")


if __name__ == "__main__":
    main()
