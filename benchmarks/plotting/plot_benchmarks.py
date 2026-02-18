#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark Plotting — generates 3-panel figures (time, throughput, memory).

Each figure = 1 row of 3 panels for one (module, system, scaling_mode) combination.
Reads CSV files produced by the benchmark scripts.

Usage:
    python plot_benchmarks.py /path/to/results/
    python plot_benchmarks.py /path/to/results/ --output-dir /path/to/plots/
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.plotting.styles import (
    AXIS_LABEL_SIZE,
    CUTOFF_COLORS,
    CUTOFF_STYLES,
    D3_CUTOFF_COLORS,
    DARK_GREEN,
    DARKEST_GREEN,
    GPU_VRAM_REFS,
    GRAY,
    GRID_STYLE,
    LEGEND_SIZE,
    LIGHT_GREEN,
    METHOD_COLORS,
    METHOD_STYLES,
    NVIDIA_GREEN,
    SINGLE_PANEL_SIZE,
    THREE_PANEL_SIZE,
    TICK_LABEL_SIZE,
    TITLE_SIZE,
    X_AXIS_LIMITS,
    X_AXIS_TICKS_LARGE,
    X_AXIS_TICKS_MEDIUM,
    X_AXIS_TICKS_SMALL,
    add_scaling_reference,
    create_table_legend,
    format_accuracy,
    format_legend_label,
    format_num,
    format_system_name,
    get_cutoff_style,
    get_method_style,
    memory_formatter,
    setup_log2_xaxis,
    setup_plot_style,
    throughput_formatter,
)

# Line alpha for all data lines — more transparent
LINE_ALPHA = 0.55


def add_vram_reference_lines(ax, unit="MB", gpu_vram_gb=None):
    """Add horizontal GPU VRAM reference line and cap y-axis just above it.

    Parameters
    ----------
    gpu_vram_gb : int, optional
        Override VRAM in GB (e.g. 141 for H200). If None, uses GPU_VRAM_REFS default.
    """
    if gpu_vram_gb:
        refs = [("GPU", gpu_vram_gb)]
    else:
        refs = list(GPU_VRAM_REFS.items())

    max_vram_val = 0
    for gpu_name, vram_gb in refs:
        val = vram_gb * 1024 if unit == "MB" else vram_gb
        max_vram_val = max(max_vram_val, val)
        ax.axhline(
            y=val, color="black", linestyle=":", linewidth=1.5, alpha=0.5, zorder=1
        )
    # Set top just above VRAM so the 80 GB tick + H100 label are visible
    if max_vram_val > 0:
        ax.set_ylim(top=max_vram_val * 5)
        # Add VRAM as explicit extra tick
        major_ticks = [t for t in ax.get_yticks() if t > 0 and t < max_vram_val * 0.9]
        major_ticks.append(max_vram_val)
        ax.set_yticks(major_ticks)
        # Custom labels: normal formatter for all, but VRAM tick gets "80 GB H100"
        labels = []
        for t in major_ticks:
            if t == max_vram_val:
                gpu_name = refs[0][0] if refs else ""
                vram_gb = refs[0][1] if refs else int(t / 1024)
                labels.append(f"{vram_gb} GB\n{gpu_name}")
            else:
                labels.append(memory_formatter(t, None))
        ax.set_yticklabels(labels)


# =============================================================================
# CSV Loading
# =============================================================================


def load_csv(filepath):
    """Load benchmark CSV with automatic type conversion."""
    results = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                try:
                    val = row[key]
                    if val in ("True", "False"):
                        row[key] = val == "True"
                    elif "." in val or "e" in val.lower():
                        row[key] = float(val)
                    elif val.lstrip("-").isdigit():
                        row[key] = int(val)
                except (ValueError, AttributeError):
                    pass
            results.append(row)
    return results


def find_csvs(input_dir, prefix):
    """Find all CSVs matching a prefix (e.g., 'nl_cscl_system_size')."""
    input_dir = Path(input_dir)
    matches = sorted(input_dir.glob(f"{prefix}*.csv"))
    return matches


# =============================================================================
# NL Plotting
# =============================================================================


def _plot_nl_lines(ax, grouped, y_key, y_transform=None):
    """Plot NL lines grouped by cutoff (color) with method as linestyle.

    Within each cutoff group:
    - Solid line + circle marker = cell (O(N))
    - Dashed line + square marker = naive (O(N²))

    Lines are sorted by cutoff first so the legend groups by cutoff.
    """
    # Sort by cutoff first, then method (cell before naive for legend order)
    sorted_keys = sorted(grouped.keys(), key=lambda k: (k[1], k[0]))

    for method, cutoff in sorted_keys:
        rows = grouped[(method, cutoff)]
        x = [r["total_atoms"] for r in rows]
        if y_transform:
            y = [y_transform(r) for r in rows]
        else:
            y = [r[y_key] for r in rows]

        color = CUTOFF_COLORS.get(cutoff, GRAY)
        # Naive = dashed (tight pattern for visible legend handles), Cell = solid
        is_naive = method == "naive"
        linestyle = (0, (4, 2)) if is_naive else "-"  # tight dashes for visibility
        marker = "s" if is_naive else "o"
        label = format_legend_label(method.capitalize(), cutoff, is_cutoff=True)

        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            markersize=5,
            alpha=LINE_ALPHA,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=label,
        )


def _plot_3panel(data, system_name, mode, module, output_dir, fname):
    """Generic 3-panel figure: delegates each panel to the single-panel renderer.

    This is the ONLY place where 3-panel figures are assembled. All plotting
    logic lives in _render_nl_panel / _render_d3_panel / _render_el_panel.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=THREE_PANEL_SIZE)

    renderer = {"nl": _render_nl_panel, "d3": _render_d3_panel, "el": _render_el_panel}[
        module
    ]

    for ax, panel in zip(axes, ("time", "throughput", "memory")):
        renderer(ax, data, system_name, mode, panel)
        # Single-panel renderers set full titles; for 3-panel, use short sub-titles
        panel_titles = {
            "time": "Time per Atom",
            "throughput": "Throughput",
            "memory": "Peak Memory (VRAM)",
        }
        ax.set_title(panel_titles[panel], fontsize=TITLE_SIZE)

    # Suptitle from the first panel's title (renderer sets it, we override)
    sys_label = format_system_name(system_name)
    module_names = {"nl": "Neighbor List", "d3": "DFT-D3", "el": "Electrostatics"}
    # Build descriptive mode string from data (same logic as renderers)
    if mode == "constant_workload":
        target = data[0].get("total_atoms", 131072) if data else 131072
        mode_str = f"Constant {format_num(target)} Atoms"
    elif mode == "batch_scaling" and module == "nl":
        mode_str = "Batch Scaling (15\u00c5)"
    elif mode == "batch_scaling":
        mode_str = "Batch Scaling"
    else:
        mode_str = "System Size Scaling"
    fig.suptitle(
        f"{module_names[module]} | {sys_label} | {mode_str}",
        fontsize=TITLE_SIZE + 1,
        y=1.02,
    )

    plt.tight_layout()
    output_path = Path(output_dir) / fname
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_nl_system_size(data, system_name, output_dir):
    """NL system-size scaling 3-panel plot. Delegates to _render_nl_panel."""
    return _plot_3panel(
        data,
        system_name,
        "system_size",
        "nl",
        output_dir,
        f"nl-{system_name}-system-size-scaling.png",
    )


def plot_nl_constant_total(data, system_name, output_dir):
    """NL constant-workload 3-panel plot. Delegates to _render_nl_panel."""
    return _plot_3panel(
        data,
        system_name,
        "constant_workload",
        "nl",
        output_dir,
        f"nl-{system_name}-constant-workload-scaling.png",
    )


def plot_nl_constant_atoms(data, system_name, output_dir):
    """NL batch-scaling 3-panel plot. Delegates to _render_nl_panel."""
    return _plot_3panel(
        data,
        system_name,
        "batch_scaling",
        "nl",
        output_dir,
        f"nl-{system_name}-batch-scaling.png",
    )


# =============================================================================
# D3 Plotting (same 3-panel style as NL)
# =============================================================================


def plot_d3_generic(data, system_name, mode_name, output_dir):
    """D3 3-panel plot. Delegates to _render_d3_panel."""
    mode_fnames = {
        "system_size": "system-size-scaling",
        "constant_workload": "constant-workload-scaling",
        "batch_scaling": "batch-scaling",
    }
    fname = f"d3-{system_name}-{mode_fnames.get(mode_name, mode_name)}.png"
    return _plot_3panel(data, system_name, mode_name, "d3", output_dir, fname)


# =============================================================================
# Electrostatics Plotting (same 3-panel style)
# =============================================================================

# Accuracy colors (viridis-shifted like cutoff)
ACCURACY_COLORS = {
    1e-4: NVIDIA_GREEN,
    0.0001: NVIDIA_GREEN,
    1e-6: "#31688E",
    0.000001: "#31688E",
}

# Method styles: solid for default (1e-6), dashed for lower accuracy (1e-4)
# Distinguish PME vs Ewald by marker shape
EL_METHOD_STYLES = {
    "pme": {"marker": "o"},
    "ewald": {"marker": "^"},
    "pme_cg": {"marker": "o"},
    "ewald_cg": {"marker": "^"},
}

# Accuracy determines linestyle: 1e-6 = solid (default), 1e-4 = dashed
ACCURACY_LINESTYLES = {
    1e-6: "-",
    0.000001: "-",
    1e-4: (0, (4, 2)),
    0.0001: (0, (4, 2)),
}


def plot_el_generic(data, system_name, mode_name, output_dir):
    """EL 3-panel plot. Delegates to _render_el_panel."""
    mode_fnames = {
        "system_size": "system-size-scaling",
        "constant_workload": "constant-workload-scaling",
        "batch_scaling": "batch-scaling",
    }
    fname = f"el-{system_name}-{mode_fnames.get(mode_name, mode_name)}.png"
    return _plot_3panel(data, system_name, mode_name, "el", output_dir, fname)


def _plot_el_generic_OLD(data, system_name, mode_name, output_dir):
    """DEPRECATED -- kept temporarily for reference. Will be removed.

    For batch_scaling: groups by (method, accuracy, atoms_per_system),
    colors by atom size. For other modes: groups by (method, accuracy),
    colors by accuracy.
    """
    # Strict filter: only _cg methods (forces + charge gradients)
    cg_data = [r for r in data if r.get("method", "").endswith("_cg")]
    if cg_data:
        data = cg_data
    # Warn if expected methods are missing
    methods_present = sorted({r.get("method", "") for r in data})
    for expected in ("pme_cg", "ewald_cg"):
        if expected not in methods_present:
            print(f"  WARNING: {expected} missing from data — re-run benchmark")
    fig, axes = plt.subplots(1, 3, figsize=THREE_PANEL_SIZE)
    mode_labels = {
        "system_size": "System Size Scaling",
        "constant_workload": "Constant Workload Scaling",
        "batch_scaling": "Batch Scaling",
    }
    mode_label = mode_labels.get(mode_name, mode_name.replace("_", " ").title())
    sys_label = format_system_name(system_name)
    fig.suptitle(
        f"Electrostatics | {sys_label} | {mode_label}",
        fontsize=TITLE_SIZE + 1,
        y=1.02,
    )

    is_batch = mode_name == "batch_scaling"

    # Grouping depends on mode
    grouped = defaultdict(list)
    for r in data:
        if is_batch:
            key = (r["method"], r.get("accuracy", 0), r["atoms_per_system"])
        else:
            key = (r["method"], r.get("accuracy", 0))
        grouped[key].append(r)

    # X-axis
    if mode_name in ("constant_workload",):
        x_key, x_label = "atoms_per_system", "Atoms per System"
    elif is_batch:
        x_key, x_label = "total_atoms", "Total Atoms"
    else:
        x_key, x_label = "total_atoms", "System Size (atoms)"

    for k in grouped:
        grouped[k] = sorted(grouped[k], key=lambda r: r.get(x_key, 0))

    x_ticks = X_AXIS_TICKS_MEDIUM
    x_limits = X_AXIS_LIMITS["medium"]

    # Color setup for batch mode: unique color per (method_base × aps)
    # Linestyle encodes accuracy so every line is visually distinct
    if is_batch:
        aps_values = sorted({r["atoms_per_system"] for r in data})
        aps_idx = {aps: i for i, aps in enumerate(aps_values)}
        # 4-color palette: green family for PME, blue family for Ewald
        _batch_el_palette = {
            ("pme", 0): NVIDIA_GREEN,  # PME + small N
            ("pme", 1): "#21918C",  # PME + large N (viridis teal)
            ("ewald", 0): "#31688E",  # Ewald + small N (viridis blue)
            ("ewald", 1): "#440154",  # Ewald + large N (viridis purple)
        }

    def _plot_el_panel(ax, y_key, y_transform=None):
        for key in sorted(grouped.keys()):
            rows = grouped[key]
            x = [r[x_key] for r in rows]
            y = [y_transform(r) if y_transform else r[y_key] for r in rows]

            if is_batch:
                method, accuracy, aps = key
                method_base = "ewald" if "ewald" in method.lower() else "pme"
                ai = min(aps_idx.get(aps, 0), 1)
                color = _batch_el_palette.get((method_base, ai), GRAY)
                method_clean = "Ewald" if method_base == "ewald" else "PME"
                marker = "^" if method_base == "ewald" else "o"
                # Linestyle encodes accuracy: solid=tight (1e-6), dashed=loose (1e-4)
                linestyle = "-" if accuracy <= 1e-5 else (0, (4, 2))
                # Table-aligned: "PME      [256, 10⁻⁴]"
                aps_str = format_num(aps).rjust(3)
                label = f"{method_clean:<7s}  [{aps_str}, {format_accuracy(accuracy)}]"
            else:
                method, accuracy = key
                color = ACCURACY_COLORS.get(accuracy, GRAY)
                method_clean = "Ewald" if "ewald" in method.lower() else "PME"
                is_ewald = "ewald" in method.lower()
                marker = "^" if is_ewald else "o"
                linestyle = (0, (4, 2)) if is_ewald else "-"
                label = format_legend_label(method_clean, accuracy, is_cutoff=False)

            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=5,
                alpha=LINE_ALPHA,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=label,
            )

    def _setup_el_x(ax):
        setup_log2_xaxis(
            ax, ticks=x_ticks, limits=x_limits, show_every_n=2, label=x_label
        )

    def _el_legend(ax, loc="best"):
        """Table legend for all EL modes; 'batch' header when batch scaling."""
        header = "batch" if is_batch else "accuracy"
        create_table_legend(ax, loc=loc, col2_header=header)

    # Panel A: Time per atom
    ax = axes[0]
    ax.set_title("Time per Atom", fontsize=TITLE_SIZE)
    _plot_el_panel(ax, "time_us_per_atom")
    _setup_el_x(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Time per atom [\u03bcs]", fontsize=AXIS_LABEL_SIZE)
    _el_legend(ax)
    ax.grid(True, which="both", **GRID_STYLE)

    # Panel B: Throughput
    ax = axes[1]
    ax.set_title("Throughput", fontsize=TITLE_SIZE)
    _plot_el_panel(ax, None, y_transform=lambda r: r["throughput_atoms_per_sec"] / 1e6)
    _setup_el_x(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Throughput [10\u2076 atoms/s]", fontsize=AXIS_LABEL_SIZE)
    import matplotlib.ticker as ticker_el

    ax.yaxis.set_major_formatter(ticker_el.FuncFormatter(throughput_formatter))
    _el_legend(ax)
    ax.grid(True, which="both", **GRID_STYLE)

    # Panel C: Peak Memory
    ax = axes[2]
    ax.set_title("Peak Memory (VRAM)", fontsize=TITLE_SIZE)
    _plot_el_panel(ax, None, y_transform=lambda r: r["mem_peak_gb"] * 1024)
    _setup_el_x(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Peak Memory", fontsize=AXIS_LABEL_SIZE)
    import matplotlib.ticker as ticker_elm

    ax.yaxis.set_major_formatter(ticker_elm.FuncFormatter(memory_formatter))
    add_vram_reference_lines(ax, unit="MB")
    _el_legend(ax)
    ax.grid(True, which="both", **GRID_STYLE)

    plt.tight_layout()
    from benchmarks.utils import make_plot_name

    fname = make_plot_name("el", system_name, mode_name)
    output_path = Path(output_dir) / fname
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


# =============================================================================
# Single-Panel API — for docs/generate_plots.py and Sphinx integration
# =============================================================================


def plot_single_panel(csv_path, panel, output_path):
    """Render one panel (time/throughput/memory) from a CSV to a standalone PNG.

    This is the primary entry point for docs/benchmarks/generate_plots.py.
    It auto-detects the module and scaling mode from the CSV filename, loads
    the data, and renders a single panel with the same styling as the 3-panel
    review figures.

    Parameters
    ----------
    csv_path : str or Path
        Path to a benchmark CSV file (e.g., nl-nh3-system-size-scaling.csv).
    panel : str
        One of 'time', 'throughput', 'memory'.
    output_path : str or Path
        Where to save the PNG.
    """
    import matplotlib.ticker as ticker_sp

    csv_path = Path(csv_path)
    output_path = Path(output_path)
    data = load_csv(csv_path)
    if not data:
        return

    setup_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=SINGLE_PANEL_SIZE)

    system = data[0].get("system", "unknown")
    name = csv_path.stem

    # Detect module + mode from filename
    module, mode = _detect_module_mode(name)

    # Dispatch to the right plotting logic
    if module == "nl":
        _render_nl_panel(ax, data, system, mode, panel)
    elif module == "d3":
        _render_d3_panel(ax, data, system, mode, panel)
    elif module == "el":
        _render_el_panel(ax, data, system, mode, panel)
    else:
        print(f"  Unknown module for {name}")
        plt.close()
        return

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def _detect_module_mode(name):
    """Parse CSV stem to determine (module, scaling_mode)."""
    if name.startswith(("nl_", "nl-")):
        module = "nl"
    elif name.startswith(("d3_", "d3-")):
        module = "d3"
    elif name.startswith(("el_", "el-")):
        module = "el"
    else:
        return None, None

    if "system_size" in name or "system-size" in name:
        mode = "system_size"
    elif (
        "constant_total" in name
        or "constant-workload" in name
        or "constant_workload" in name
    ):
        mode = "constant_workload"
    elif "constant_atoms" in name or "batch-scaling" in name or "batch_scaling" in name:
        mode = "batch_scaling"
    else:
        mode = "system_size"
    return module, mode


def _setup_panel_axes(ax, panel, x_ticks=None, x_limits=None, x_label=None):
    """Common y-axis and grid setup for single panels.

    NOTE: does NOT call setup_log2_xaxis -- renderers handle x-axis
    themselves (with mode-specific show_batch_size etc).
    """
    import matplotlib.ticker as ticker_sp

    ax.set_yscale("log")

    # Ensure sufficient y-axis ticks on log scale: place major ticks at
    # 1-2-5 subdivisions per decade so narrow ranges still get labels.
    ax.yaxis.set_major_locator(
        ticker_sp.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12)
    )
    ax.yaxis.set_minor_locator(
        ticker_sp.LogLocator(base=10, subs="auto", numticks=12)
    )
    ax.yaxis.set_minor_formatter(ticker_sp.NullFormatter())

    ax.grid(True, which="both", **GRID_STYLE)

    if panel == "time":
        ax.set_ylabel("Time per atom [\u03bcs]", fontsize=AXIS_LABEL_SIZE)
        ax.yaxis.set_major_formatter(ticker_sp.FuncFormatter(throughput_formatter))
    elif panel == "throughput":

        ax.set_ylabel("Throughput [10\u2076 atoms/s]", fontsize=AXIS_LABEL_SIZE)
        ax.yaxis.set_major_formatter(ticker_sp.FuncFormatter(throughput_formatter))
    elif panel == "memory":
        ax.set_ylabel("Peak Memory", fontsize=AXIS_LABEL_SIZE)
        ax.yaxis.set_major_formatter(ticker_sp.FuncFormatter(memory_formatter))
        add_vram_reference_lines(ax, unit="MB")


def _render_nl_panel(ax, data, system, mode, panel):
    """Render a single NL panel -- mirrors 3-panel logic exactly."""
    x_ticks = X_AXIS_TICKS_MEDIUM
    x_limits = X_AXIS_LIMITS["medium"]

    if mode == "batch_scaling":
        # Mirror plot_nl_constant_atoms: filter to 15Å, color by aps, linestyle by method
        data_15 = [r for r in data if abs(r.get("cutoff", 15.0) - 15.0) < 1.0]
        if not data_15:
            data_15 = data
        grouped = defaultdict(list)
        for r in data_15:
            key = (r["method"], r["atoms_per_system"])
            grouped[key].append(r)
        for key in grouped:
            grouped[key] = sorted(grouped[key], key=lambda r: r["batch_size"])

        aps_values = sorted({r["atoms_per_system"] for r in data_15})
        color_palette = [NVIDIA_GREEN, "#31688E", "#440154"]
        aps_colors = {
            aps: color_palette[i % len(color_palette)]
            for i, aps in enumerate(aps_values)
        }

        for method, aps in sorted(grouped.keys(), key=lambda k: (k[1], k[0])):
            rows = grouped[(method, aps)]
            x = [r["total_atoms"] for r in rows]
            y = _get_panel_y(rows, panel)
            color = aps_colors.get(aps, GRAY)
            is_naive = method == "naive"
            method_str = "Naive" if is_naive else "Cell"
            label = f"{method_str:<7s}  N={format_num(aps)}"
            ax.plot(
                x,
                y,
                color=color,
                linestyle=(0, (4, 2)) if is_naive else "-",
                marker="s" if is_naive else "o",
                linewidth=2,
                markersize=5,
                alpha=LINE_ALPHA,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=label,
            )

        x_label = "Total Atoms"
        mode_str = "Batch Scaling (15\u00c5)"

    elif mode == "constant_workload":
        # Mirror plot_nl_constant_total: color by cutoff, linestyle by method
        target = data[0].get("total_atoms", 131072) if data else 131072
        grouped = defaultdict(list)
        for r in data:
            key = (r["method"], r.get("cutoff", 0))
            grouped[key].append(r)
        for key in grouped:
            grouped[key] = sorted(grouped[key], key=lambda r: r["atoms_per_system"])

        for method, cutoff in sorted(grouped.keys(), key=lambda k: (k[1], k[0])):
            rows = grouped[(method, cutoff)]
            x = [r["atoms_per_system"] for r in rows]
            y = _get_panel_y(rows, panel)
            color = CUTOFF_COLORS.get(cutoff, GRAY)
            is_naive = method == "naive"
            ax.plot(
                x,
                y,
                color=color,
                linestyle=(0, (4, 2)) if is_naive else "-",
                marker="s" if is_naive else "o",
                linewidth=2,
                markersize=5,
                alpha=LINE_ALPHA,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=format_legend_label(method.capitalize(), cutoff),
            )

        x_label = "Atoms per System [\u00d7batch]"
        mode_str = f"Constant {format_num(target)} Atoms"

    else:
        # system_size: color by cutoff, linestyle by method
        grouped = defaultdict(list)
        for r in data:
            key = (r["method"], r.get("cutoff", 0))
            grouped[key].append(r)
        for key in grouped:
            grouped[key] = sorted(grouped[key], key=lambda r: r["total_atoms"])

        for method, cutoff in sorted(grouped.keys(), key=lambda k: (k[1], k[0])):
            rows = grouped[(method, cutoff)]
            x = [r["total_atoms"] for r in rows]
            y = _get_panel_y(rows, panel)
            color = CUTOFF_COLORS.get(cutoff, GRAY)
            is_naive = method == "naive"
            ax.plot(
                x,
                y,
                color=color,
                linestyle=(0, (4, 2)) if is_naive else "-",
                marker="s" if is_naive else "o",
                linewidth=2,
                markersize=5,
                alpha=LINE_ALPHA,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=format_legend_label(method.capitalize(), cutoff),
            )

        x_label = "System Size (atoms)"
        mode_str = "System Size Scaling"

    sys_label = format_system_name(system)
    ax.set_title(f"Neighbor List | {sys_label} | {mode_str}", fontsize=TITLE_SIZE)

    if mode == "constant_workload":
        setup_log2_xaxis(
            ax,
            ticks=x_ticks,
            limits=x_limits,
            show_batch_size=True,
            target_atoms=target,
            label=x_label,
        )
    else:
        setup_log2_xaxis(ax, ticks=x_ticks, limits=x_limits, label=x_label)
    _setup_panel_axes(ax, panel, x_ticks, x_limits, x_label)
    legend_header = "batch" if mode == "batch_scaling" else "cutoff"
    create_table_legend(ax, loc="best", col2_header=legend_header)


def _render_d3_panel(ax, data, system, mode, panel):
    """Render a single D3 panel -- mirrors 3-panel logic exactly."""
    is_batch = mode == "batch_scaling"
    x_ticks = X_AXIS_TICKS_MEDIUM
    x_limits = X_AXIS_LIMITS["medium"]

    grouped = defaultdict(list)
    for r in data:
        if is_batch:
            key = (r.get("cutoff", 15.0), r["atoms_per_system"])
        else:
            key = r.get("cutoff", 15.0)
        grouped[key].append(r)

    if mode == "constant_workload":
        x_key, x_label = "atoms_per_system", "Atoms per System [\u00d7batch]"
        target = data[0].get("total_atoms", 131072) if data else 131072
    elif is_batch:
        x_key, x_label = "total_atoms", "Total Atoms"
    else:
        x_key, x_label = "total_atoms", "System Size (atoms)"

    for k in grouped:
        grouped[k] = sorted(grouped[k], key=lambda r: r.get(x_key, 0))

    if is_batch:
        # Batch mode: color = cutoff (same as other D3 plots), linestyle = system size
        aps_values = sorted({r["atoms_per_system"] for r in data})

        for key in sorted(grouped.keys()):
            rows = grouped[key]
            cutoff, aps = key
            x = [r[x_key] for r in rows]
            y_raw = [r.get("time_d3_us_per_atom", r["time_us_per_atom"]) for r in rows]
            y = _get_panel_y_from_raw(rows, panel, y_raw)
            color = D3_CUTOFF_COLORS.get(cutoff, GRAY)
            is_large = aps == max(aps_values) if len(aps_values) > 1 else False
            linestyle = (0, (4, 2)) if is_large else "-"
            style = CUTOFF_STYLES.get(cutoff, {"marker": "o"})
            marker = style.get("marker", "o")
            label = f"D3 {int(cutoff)}\u00c5 N={format_num(aps)}"
            ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2,
                markersize=5,
                alpha=LINE_ALPHA,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=label,
            )
    else:
        # System size / constant workload: color by cutoff (D3 palette)
        for key in sorted(grouped.keys()):
            rows = grouped[key]
            cutoff = key
            x = [r[x_key] for r in rows]
            y_raw = [r.get("time_d3_us_per_atom", r["time_us_per_atom"]) for r in rows]
            y = _get_panel_y_from_raw(rows, panel, y_raw)
            label = format_legend_label("D3", cutoff)
            color = D3_CUTOFF_COLORS.get(cutoff, GRAY)
            style = CUTOFF_STYLES.get(cutoff, {"marker": "o", "linestyle": "-"})
            ax.plot(
                x,
                y,
                color=color,
                **style,
                linewidth=2,
                markersize=5,
                alpha=LINE_ALPHA,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=label,
            )

    sys_label = format_system_name(system)
    if mode == "constant_workload":
        mode_str = f"Constant {format_num(target)} Atoms"
    elif is_batch:
        mode_str = "Batch Scaling"
    else:
        mode_str = "System Size Scaling"
    ax.set_title(f"DFT-D3 | {sys_label} | {mode_str}", fontsize=TITLE_SIZE)

    if mode == "constant_workload":
        setup_log2_xaxis(
            ax,
            ticks=x_ticks,
            limits=x_limits,
            show_batch_size=True,
            target_atoms=target,
            label=x_label,
        )
    else:
        setup_log2_xaxis(ax, ticks=x_ticks, limits=x_limits, label=x_label)
    _setup_panel_axes(ax, panel, x_ticks, x_limits, x_label)
    create_table_legend(ax, loc="best", col2_header="cutoff")


def _render_el_panel(ax, data, system, mode, panel):
    """Render a single EL panel."""
    # Filter to _cg methods
    method_bases = sorted(
        {
            r.get("method", "").replace("_cg", "")
            for r in data
            if r.get("method", "") not in ("", "method")
        }
    )
    cg_data = [r for r in data if r.get("method", "").endswith("_cg")]
    if cg_data:
        data = cg_data
    for expected in ("pme_cg", "ewald_cg"):
        if expected not in {r.get("method", "") for r in data}:
            print(f"  WARNING: {expected} missing from data")

    is_batch = mode == "batch_scaling"
    grouped = defaultdict(list)
    for r in data:
        if is_batch:
            key = (r["method"], r.get("accuracy", 0), r["atoms_per_system"])
        else:
            key = (r["method"], r.get("accuracy", 0))
        grouped[key].append(r)

    if mode == "constant_workload":
        x_key, x_label = "atoms_per_system", "Atoms per System [\u00d7batch]"
        target = data[0].get("total_atoms", 131072) if data else 131072
    elif is_batch:
        x_key, x_label = "total_atoms", "Total Atoms"
    else:
        x_key, x_label = "total_atoms", "System Size (atoms)"

    for k in grouped:
        grouped[k] = sorted(grouped[k], key=lambda r: r.get(x_key, 0))

    x_ticks = X_AXIS_TICKS_MEDIUM
    x_limits = X_AXIS_LIMITS["medium"]

    # Color setup for batch mode
    # Color by aps for batch (green=small, orange=large), by accuracy for non-batch
    ACCURACY_COLORS = {
        1e-4: NVIDIA_GREEN,
        0.0001: NVIDIA_GREEN,
        1e-6: "#31688E",
        0.000001: "#31688E",
    }

    if is_batch:
        aps_values = sorted({r["atoms_per_system"] for r in data})
        acc_values = sorted({r.get("accuracy", 0) for r in data})
        # 4 colors: one per (aps, accuracy) combo
        _batch_4colors = [NVIDIA_GREEN, "#31688E", "#E67E22", "#440154"]
        combo_colors = {}
        ci = 0
        for aps in aps_values:
            for acc in acc_values:
                combo_colors[(aps, acc)] = _batch_4colors[ci % len(_batch_4colors)]
                ci += 1

    # Sort by (accuracy, method) so same-accuracy lines are adjacent in legend
    if is_batch:
        sorted_keys = sorted(grouped.keys(), key=lambda k: (k[1], k[2], k[0]))
    else:
        sorted_keys = sorted(grouped.keys(), key=lambda k: (k[1], k[0]))

    for key in sorted_keys:
        rows = grouped[key]
        x = [r[x_key] for r in rows]
        y = _get_panel_y(rows, panel)

        if is_batch:
            method, accuracy, aps = key
            method_base = "ewald" if "ewald" in method.lower() else "pme"
            color = combo_colors.get((aps, accuracy), GRAY)
            method_clean = "Ewald" if method_base == "ewald" else "PME"
            marker = "^" if method_base == "ewald" else "o"
            linestyle = (0, (4, 2)) if method_base == "ewald" else "-"
            aps_str = format_num(aps).rjust(3)
            label = f"{method_clean:<7s}  [{aps_str}, {format_accuracy(accuracy)}]"
        else:
            method, accuracy = key
            color = ACCURACY_COLORS.get(accuracy, GRAY)
            method_clean = "Ewald" if "ewald" in method.lower() else "PME"
            is_ewald = "ewald" in method.lower()
            marker = "^" if is_ewald else "o"
            linestyle = (0, (4, 2)) if is_ewald else "-"
            label = format_legend_label(method_clean, accuracy, is_cutoff=False)

        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=5,
            alpha=LINE_ALPHA,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=label,
        )

    sys_label = format_system_name(system)
    if mode == "constant_workload":
        target = data[0].get("total_atoms", 131072) if data else 131072
        mode_str = f"Constant {format_num(target)} Atoms"
    elif mode == "batch_scaling":
        mode_str = "Batch Scaling"
    else:
        mode_str = "System Size Scaling"
    ax.set_title(f"Electrostatics | {sys_label} | {mode_str}", fontsize=TITLE_SIZE)

    if mode == "constant_workload":
        setup_log2_xaxis(
            ax,
            ticks=x_ticks,
            limits=x_limits,
            show_batch_size=True,
            target_atoms=target,
            label=x_label,
        )
    else:
        setup_log2_xaxis(ax, ticks=x_ticks, limits=x_limits, label=x_label)
    _setup_panel_axes(ax, panel, x_ticks, x_limits, x_label)
    header = "batch" if is_batch else "accuracy"
    create_table_legend(ax, loc="best", col2_header=header)


def _get_panel_y(rows, panel):
    """Extract y-values for a given panel type."""
    if panel == "time":
        return [r["time_us_per_atom"] for r in rows]
    elif panel == "throughput":
        return [r["throughput_atoms_per_sec"] / 1e6 for r in rows]
    elif panel == "memory":
        return [r["mem_peak_gb"] * 1024 for r in rows]  # GB -> MB
    return []


def _get_panel_y_from_raw(rows, panel, time_values):
    """Extract y-values for D3 panels (uses pre-extracted D3-only time).

    time_values are per-atom times in microseconds, so throughput in
    Matoms/s = 1 / time_us_per_atom.
    """
    if panel == "time":
        return time_values
    elif panel == "throughput":
        return [1.0 / t if t > 0 else 0 for t in time_values]
    elif panel == "memory":
        return [r["mem_peak_gb"] * 1024 for r in rows]
    return []


# =============================================================================
# CLI — auto-detects all CSVs and plots them
# =============================================================================


def _detect_and_plot(csv_path, output_dir):
    """Detect CSV type from filename and plot accordingly.

    Supports both old naming (nl_cscl_system_size_gpu.csv) and
    new naming (nl-cscl-system-size-scaling.csv).
    """
    data = load_csv(csv_path)
    if not data:
        return
    system = data[0].get("system", "unknown")
    name = csv_path.stem

    # Detect module from prefix (supports both nl_ and nl-)
    if name.startswith(("nl_", "nl-")):
        if "system_size" in name or "system-size" in name:
            plot_nl_system_size(data, system, output_dir)
        elif "constant_total" in name or "constant-workload" in name:
            plot_nl_constant_total(data, system, output_dir)
        elif "constant_atoms" in name or "batch-scaling" in name:
            plot_nl_constant_atoms(data, system, output_dir)
    elif name.startswith(("d3_", "d3-")):
        if "system_size" in name or "system-size" in name:
            mode = "system_size"
        elif (
            "constant_total" in name
            or "constant-workload" in name
            or "constant_workload" in name
        ):
            mode = "constant_workload"
        else:
            mode = "batch_scaling"
        plot_d3_generic(data, system, mode, output_dir)
    elif name.startswith(("el_", "el-")):
        if "system_size" in name or "system-size" in name:
            mode = "system_size"
        elif (
            "constant_total" in name
            or "constant-workload" in name
            or "constant_workload" in name
        ):
            mode = "constant_workload"
        else:
            mode = "batch_scaling"
        plot_el_generic(data, system, mode, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("input_dir", type=Path, help="Directory with CSV files")
    parser.add_argument("--output-dir", "-o", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    setup_plot_style()

    # Auto-detect and plot all CSVs
    csv_files = sorted(args.input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files in {args.input_dir}")
    for csv_path in csv_files:
        try:
            _detect_and_plot(csv_path, output_dir)
        except Exception as e:
            print(f"  ERROR plotting {csv_path.name}: {e}")


if __name__ == "__main__":
    main()
