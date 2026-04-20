#!/usr/bin/env python3
"""Generate plots from benchmark CSV files during the Sphinx build.

Two live paths:

- ``generate_dynamics_plots`` — reads ``dynamics_*.csv`` produced by the
  benchmarks under ``benchmarks/dynamics/``.
- ``main`` (formerly ``_generate_tme_plots``) — reads the
  ``{nl,d3,el}-{system}-{mode}.csv`` files shipped under
  ``docs/benchmarks/benchmark_results/`` and renders the single-panel
  and comparison PNGs used by ``neighborlist.md`` / ``dftd3.md`` /
  ``electrostatics.md``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Generic plotting helpers (used by the dynamics path below)
# --------------------------------------------------------------------------

def plot_series(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str | None = None,
    x_label: str = "Number of atoms",
    y_label: str = "Value",
    caption: str | None = None,
) -> None:
    """Plot multiple data series on a log-log scale."""
    num_series = len(series)
    fig_width = 10 if num_series > 3 else 8
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), constrained_layout=True)

    if num_series == 1:
        colors = ["#2E7D32"]
    else:
        cmap = plt.cm.YlGn
        colors = [cmap(0.3 + 0.7 * i / (num_series - 1)) for i in range(num_series)]

    for idx, (label, (xs, ys)) in enumerate(series.items()):
        if xs is None or ys is None:
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=6.0,
            label=label,
            color=colors[idx],
            markeredgewidth=0.5,
            markeredgecolor="black",
            alpha=0.9,
        )

    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20)
    )
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20)
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)

    if title is not None:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.3, color="gray")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.2, color="gray")

    if num_series <= 4:
        ax.legend(
            frameon=False,
            fontsize=12,
            loc="upper left",
            framealpha=0.95,
            edgecolor="gray",
            fancybox=False,
        )
    else:
        ax.legend(
            frameon=False,
            fontsize=11,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            framealpha=0.95,
            edgecolor="gray",
            fancybox=False,
        )

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


# --------------------------------------------------------------------------
# Dynamics path — reads dynamics_*.csv
# --------------------------------------------------------------------------

def load_dynamics_csv(filepath: Path) -> pd.DataFrame:
    """Load a dynamics benchmark CSV and convert inf → nan for plotting."""
    df = pd.read_csv(filepath)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def _parse_dynamics_filename(filename: str) -> dict[str, str]:
    """Parse ``dynamics_{md|opt}_{single|batch}_{backend}_{gpu_sku}.csv``."""
    parts = filename.replace(".csv", "").split("_")
    if len(parts) < 5 or parts[0] != "dynamics":
        return {}
    return {
        "benchmark_type": parts[1],
        "system_type": parts[2],
        "backend": parts[3],
        "gpu_sku": "_".join(parts[4:]),
    }


def generate_dynamics_plots(results_dir: Path, output_dir: Path) -> None:
    """Generate scaling, throughput, and (for batched) batch-scaling plots
    for every ``dynamics_*.csv`` in ``results_dir``.
    """
    print("\nGenerating dynamics benchmark plots...")

    dynamics_files = list(results_dir.glob("dynamics_*.csv"))
    if not dynamics_files:
        print("  No dynamics benchmark results found")
        return

    files_by_category: dict[str, dict[str, dict]] = {}
    for filepath in dynamics_files:
        info = _parse_dynamics_filename(filepath.name)
        if not info:
            continue
        category = f"{info['benchmark_type']}_{info['system_type']}"
        files_by_category.setdefault(category, {})
        files_by_category[category][info["backend"]] = {
            "path": filepath,
            "gpu_sku": info["gpu_sku"],
        }

    for category, backends in files_by_category.items():
        benchmark_type, system_type = category.split("_")
        print(f"\n  Processing {benchmark_type.upper()} {system_type} benchmarks...")
        is_batched = system_type == "batch"

        all_data: dict[str, pd.DataFrame] = {}
        gpu_sku = "unknown"
        for backend, file_info in backends.items():
            all_data[backend] = load_dynamics_csv(file_info["path"])
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
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for backend, df in data_by_backend.items():
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

    series = {}
    for backend, df in data_by_backend.items():
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

    has_batch_throughput = any(
        "batch_throughput_system_steps_per_s" in d.columns
        for d in data_by_backend.values()
    )
    if is_batched and has_batch_throughput:
        series = {}
        for backend, df in data_by_backend.items():
            if "batch_throughput_system_steps_per_s" not in df.columns:
                continue
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
        model_types = df["model_type"].dropna().replace("", pd.NA).dropna().unique()
        group_col = "model_type" if len(model_types) > 1 else "method"
        x_col = "total_atoms"
        x_label = "Total atoms (num_atoms × batch_size)"
    else:
        group_col = "method"
        x_col = "num_atoms"
        x_label = "Number of atoms"
    group_vals = df[group_col].unique()

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
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


# --------------------------------------------------------------------------
# NL / D3 / EL path — reads {nl,d3,el}-{system}-{mode}.csv
# --------------------------------------------------------------------------

def _generate_benchmark_plots(results_dir: Path, output_dir: Path) -> None:
    """Generate single-panel and comparison PNGs from the shipped
    ``{nl,d3,el}-{system}-{mode}.csv`` files.

    At Sphinx build time ``docs/benchmarks/__init__.py`` shadows the
    project-root ``benchmarks`` package because ``docs/`` sits earlier
    on ``sys.path``. We undo that shadowing locally so the real
    ``benchmarks.plotting.plot_benchmarks`` module is importable.
    """
    import shutil
    import sys

    project_root = str(Path(__file__).resolve().parent.parent.parent)
    pb_path = Path(project_root) / "benchmarks" / "plotting" / "plot_benchmarks.py"
    if not pb_path.exists():
        print("  benchmarks.plotting.plot_benchmarks not found; skipping NL/D3/EL plots")
        return

    docs_dir = Path(__file__).resolve().parent.parent
    removed_paths = []
    try:
        for p in list(sys.path):
            try:
                if Path(p).resolve() == docs_dir:
                    sys.path.remove(p)
                    removed_paths.append(p)
            except (ValueError, OSError):
                continue
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        for key in list(sys.modules):
            if key == "benchmarks" or key.startswith("benchmarks."):
                del sys.modules[key]

        from benchmarks.plotting.plot_benchmarks import (
            SINGLE_PANEL_SIZE,
            detect_module_mode,
            generate_comparison_panels,
            load_csv,
            plot_single_panel,
            render_d3_panel,
            render_el_panel,
            render_nl_panel,
            setup_plot_style,
        )
    except Exception as e:
        print(f"  NL/D3/EL plotting not available: {e}")
        return
    finally:
        for p in removed_paths:
            if p not in sys.path:
                sys.path.append(p)

    import matplotlib

    matplotlib.use("Agg")

    csvs = (
        sorted(results_dir.glob("nl-*.csv"))
        + sorted(results_dir.glob("d3-*.csv"))
        + sorted(results_dir.glob("el-*.csv"))
    )
    # Failure sidecars are loaded by the plotter per-CSV; skip them here
    # so we don't try to plot their schema as a main benchmark CSV.
    csvs = [c for c in csvs if not c.stem.endswith("-failures")]
    if not csvs:
        print("  No NL/D3/EL CSVs found, skipping")
        return

    print(f"  Generating NL/D3/EL plots from {len(csvs)} CSVs...")
    setup_plot_style()

    renderers = {
        "nl": render_nl_panel,
        "d3": render_d3_panel,
        "el": render_el_panel,
    }

    for csv_path in csvs:
        for panel in ("time", "throughput", "memory"):
            plot_single_panel(
                csv_path, panel, output_dir / f"{csv_path.stem}-{panel}.png"
            )

        data = load_csv(csv_path)
        jax_data = [r for r in data if r.get("backend") == "jax"]
        if jax_data:
            name = csv_path.stem
            module, mode = detect_module_mode(name)
            if module and module in renderers:
                system = jax_data[0].get("system", "unknown")
                for panel in ("time", "throughput", "memory"):
                    fig, ax = plt.subplots(1, 1, figsize=SINGLE_PANEL_SIZE)
                    try:
                        renderers[module](ax, jax_data, system, mode, panel)
                        plt.tight_layout()
                        plt.savefig(
                            output_dir / f"{name}-jax-{panel}.png",
                            dpi=150,
                            bbox_inches="tight",
                        )
                    except Exception as exc:
                        print(f"    SKIP JAX {name}/{panel}: {exc}")
                    plt.close()

    comp_dir = results_dir / "_comparison_tmp"
    comp_dir.mkdir(exist_ok=True)
    generate_comparison_panels(results_dir, comp_dir)
    for png in comp_dir.glob("*.png"):
        shutil.copy2(png, output_dir / png.name)
    shutil.rmtree(comp_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Entry point wired from docs/conf.py
# --------------------------------------------------------------------------

def main() -> None:
    """Generate all benchmark plots consumed by the Sphinx build."""
    print("Generating benchmark plots...")

    results_dir = Path(__file__).parent / "benchmark_results"
    output_dir = Path(__file__).parent / "_static"

    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    output_dir.mkdir(exist_ok=True)

    generate_dynamics_plots(results_dir, output_dir)
    _generate_benchmark_plots(results_dir, output_dir)

    print("\nPlot generation complete!")


if __name__ == "__main__":
    main()
